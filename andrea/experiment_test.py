from __future__ import annotations

import json
import torch
import numpy as np
import pandas as pd
from torch import nn
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple

from sklearn.metrics import average_precision_score

from andrea.experiment_select_clients_utils import (
    ClientData,
    choose_subsets_and_load_clients,
    parse_subset_clients,
    load_client_from_dir,
    make_bidirected_hetero,
)
from andrea.multigraph_generation import TASKS
from models.pna_reverse_mp import PNANetReverseMP
from utils.graph_helpers import max_port_cols
from utils.seed import set_seed

# CSV_PATH = "./andrea/test_generation_parameters.csv"
TRAIN_SELECTED_GRAPHS_CSV_PATH = "./andrea/training_selected_graphs.csv"

# EXPERIMENT_PATH = Path("./andrea/experiment.csv")
TRAIN_SELECTED_EXPERIMENT_LOG_PATH = Path("./andrea/experiment_test.csv")

SUBSET_PATH = "./andrea/heterogeneity.csv"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"

# OUT_ROOT = "andrea/runs/experiment_fedavg_heterogeneity"
OUT_ROOT = "andrea/experiment_test_runs"

ROUNDS = 100
LOCAL_EPOCHS = 1
CLIENT_FRACTION = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SEEDS = [0, 1, 2]
MCW = ["auto"]
NUM_LAYERS = [6]
LRS = [5e-4, 1e-3, 5e-3]
WEIGHT_DECAYS = [0, 1e-4]
DROPOUTS = [0, 0.1]
HIDDEN_DIMS = [32, 64]

MODEL_CONFIG_KEYS = [
    "mcw",
    "num_layers",
    "lr",
    "weight_decay",
    "dropout",
    "hidden_dim",
    "seed",
]


def build_model_tag(mcw, num_layers, lr, weight_decay, dropout, hidden_dim):
    return (
        f"mcw{mcw}"
        f"_layers{num_layers}"
        f"_lr{lr}"
        f"_wd{weight_decay}"
        f"_do{dropout}"
        f"_hd{hidden_dim}"
    )


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_cfg(config_path: str, key: str) -> Dict:
    with open(config_path, "r") as f:
        all_cfg = json.load(f)
    cfg_obj = all_cfg[key]
    cfg = dict(cfg_obj.get("default_hparams", {}))
    # carry top-level knobs that may matter
    cfg["model_name"] = cfg_obj.get("model_name", key)
    cfg["best_model_path"] = cfg_obj.get("best_model_path", "")
    cfg["use_ego_ids"] = cfg_obj.get("use_ego_ids", True)
    cfg["use_port_ids"] = cfg_obj.get("use_port_ids", True)
    cfg["use_mini_batch"] = cfg_obj.get("use_mini_batch", True)
    cfg["batch_size"] = cfg_obj.get("batch_size", cfg.get("batch_size", 256))
    cfg["port_emb_dim"] = cfg_obj.get("port_emb_dim", cfg.get("port_emb_dim", 8))
    cfg["num_epochs"] = cfg_obj.get("num_epochs", cfg.get("num_epochs", 1))
    return cfg


def load_experiment_runs(experiment_path):
    # load existing experiment log once
    # if experiment_path.exists():
    #     existing_exp_df = pd.read_csv(experiment_path)
    #     existing_run_dirs = set(existing_exp_df["out_dir"].astype(str).tolist())
    # else:
    existing_exp_df = pd.DataFrame()
    existing_run_dirs = set()

    return existing_exp_df, existing_run_dirs


def compute_global_port_vocab(*graph_lists: List) -> Tuple[int, int]:
    max_in, max_out = 0, 0
    for graphs in graph_lists:
        for g in graphs:
            mi, mo = max_port_cols(g)
            max_in = max(max_in, int(mi))
            max_out = max(max_out, int(mo))
    return max_in + 1, max_out + 1


def compute_global_degree_hists(
    train_graphs: List,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_in_deg = []
    all_out_deg = []

    for g in train_graphs:
        ei = g.edge_index
        n = g.num_nodes
        in_deg = torch.bincount(ei[1], minlength=n).cpu()
        out_deg = torch.bincount(ei[0], minlength=n).cpu()
        all_in_deg.append(in_deg)
        all_out_deg.append(out_deg)

    in_deg_cat = torch.cat(all_in_deg, dim=0)
    out_deg_cat = torch.cat(all_out_deg, dim=0)

    deg_fwd_hist = torch.bincount(
        in_deg_cat, minlength=int(in_deg_cat.max().item()) + 1
    ).float()
    deg_rev_hist = torch.bincount(
        out_deg_cat, minlength=int(out_deg_cat.max().item()) + 1
    ).float()
    return deg_fwd_hist, deg_rev_hist


def make_model(
    cfg: Dict,
    x_dim: int,
    out_dim: int,
    deg_fwd_hist: torch.Tensor,
    deg_rev_hist: torch.Tensor,
    ego_dim: int,
    in_vocab: int,
    out_vocab: int,
) -> PNANetReverseMP:
    model = PNANetReverseMP(
        in_dim=x_dim,
        out_dim=out_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        deg_fwd=deg_fwd_hist,
        deg_rev=deg_rev_hist,
        ego_dim=ego_dim,
        in_port_vocab_size=in_vocab,
        out_port_vocab_size=out_vocab,
        port_emb_dim=(
            cfg.get("port_emb_dim", 0) if cfg.get("use_port_ids", True) else None
        ),
    )
    return model


def fedavg_state_dict(
    state_dicts: List[Dict[str, torch.Tensor]], weights: List[int]
) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    out = {}
    for k in state_dicts[0].keys():
        acc = None
        for sd, w in zip(state_dicts, weights):
            t = sd[k].float() * (float(w) / total)
            acc = t if acc is None else acc + t
        out[k] = acc.to(state_dicts[0][k].dtype)
    return out


def append_rows_csv(csv_path: Path, rows: List[Dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def train_epoch_fullbatch(
    model,
    graph,
    optimizer,
    criterion,
    device,
):
    model.train()
    optimizer.zero_grad()

    graph = graph.to(device)

    x_in = {"n": graph["n"].x}
    y_true = graph["n"].y

    edge_in = {
        ("n", "fwd", "n"): graph[("n", "fwd", "n")].edge_index,
        ("n", "rev", "n"): graph[("n", "rev", "n")].edge_index,
    }

    n_nodes = int(getattr(graph["n"], "num_nodes", 0) or graph["n"].x.size(0))

    edge_attr_dict = {}
    for rel in [("n", "fwd", "n"), ("n", "rev", "n")]:
        if "edge_attr" in graph[rel]:
            ea = graph[rel].edge_attr
            if ea.dtype != torch.long:
                ea = ea.long()
            edge_attr_dict[rel] = ea

    if not getattr(model, "_dbg_printed", False):
        f_ea = edge_attr_dict[("n", "fwd", "n")]
        r_ea = edge_attr_dict[("n", "rev", "n")]
        print(
            f"[NODES]: {n_nodes}\n"
            f"[PORT] fwd edge_attr: {tuple(f_ea.shape)} (dtype={f_ea.dtype}) | "
            f"rev edge_attr: {tuple(r_ea.shape)} (dtype={r_ea.dtype})"
        )
        print(f"[TRAIN MODE] full-batch | ego_dim={getattr(model, 'ego_dim', 0)}")
        model._dbg_printed = True

    out = model(
        x_in,
        edge_in,
        edge_attr_dict=edge_attr_dict,
        device=device,
    )

    loss = criterion(out, y_true.float())
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_fullbatch(
    model,
    graph,
    criterion,
    device,
    thresholds: list[float] | None = None,
):
    model.eval()

    graph = graph.to(device)

    x_in = {"n": graph["n"].x}
    y_true = graph["n"].y
    edge_in = {
        ("n", "fwd", "n"): graph[("n", "fwd", "n")].edge_index,
        ("n", "rev", "n"): graph[("n", "rev", "n")].edge_index,
    }

    n_nodes = int(getattr(graph["n"], "num_nodes", 0) or graph["n"].x.size(0))

    edge_attr_dict = {}
    for rel in [("n", "fwd", "n"), ("n", "rev", "n")]:
        if "edge_attr" in graph[rel]:
            ea = graph[rel].edge_attr
            if ea.dtype != torch.long:
                ea = ea.long()
            edge_attr_dict[rel] = ea

    logits = model(
        x_in,
        edge_in,
        edge_attr_dict=edge_attr_dict,
        device=device,
    )

    probs = torch.sigmoid(logits)
    thr = torch.tensor(thresholds, device=probs.device, dtype=probs.dtype).view(1, -1)

    preds = probs > thr

    per_task_auprc = []

    for task_id in range(y_true.shape[1]):
        per_task_auprc.append(
            _safe_average_precision(y_true[:, task_id], probs[:, task_id])
        )

    loss = criterion(logits, y_true.float())

    y_bool = y_true.bool()

    # pair accuracy / element-wise accuracy
    pair_acc = (preds == y_bool).float().mean().item()
    # subset / exact-match accuracy, turn each row into one boolean
    subset_acc = (preds == y_bool).all(dim=1).float().mean().item()

    # micro metrics
    tp = (preds & y_bool).sum().item()
    fp = (preds & (~y_bool)).sum().item()
    fn = ((~preds) & y_bool).sum().item()

    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    micro_f1 = (
        0.0
        if (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )

    # per-task metrics
    C = y_true.size(1)
    tp_task, fp_task, tn_task, fn_task = [], [], [], []
    prec_task, rec_task, f1_task = [], [], []

    for c in range(C):
        p = preds[:, c]
        y = y_bool[:, c]

        tp = int((p & y).sum().item())
        fp = int((p & (~y)).sum().item())
        tn = int(((~p) & (~y)).sum().item())
        fn = int(((~p) & y).sum().item())

        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = (
            0.0
            if (precision + recall) == 0
            else 2 * precision * recall / (precision + recall)
        )

        tp_task.append(tp)
        fp_task.append(fp)
        tn_task.append(tn)
        fn_task.append(fn)
        prec_task.append(float(precision))
        rec_task.append(float(recall))
        f1_task.append(float(f1))

    macro_f1 = float(sum(f1_task) / len(f1_task))
    pos_cnt = y_true.sum(dim=0).long().tolist()
    pos_rate = y_true.mean(dim=0).tolist()

    return {
        "scalar": {
            "loss": float(loss.item()),
            "pair_acc": float(pair_acc),
            "subset_acc": float(subset_acc),
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
        },
        "per_task": {
            "tp": tp_task,
            "fp": fp_task,
            "tn": tn_task,
            "fn": fn_task,
            "precision": prec_task,
            "recall": rec_task,
            "f1": f1_task,
            "threshold": thresholds,
            "auprc": per_task_auprc,
        },
        "counts": {
            "num_nodes": int(n_nodes),
            "pos_cnt": [int(x) for x in pos_cnt],
            "pos_rate": [float(x) for x in pos_rate],
        },
    }


def get_label_stats_from_graph(graph):
    """
    graph: hetero graph with graph['n'].y of shape [num_nodes, num_tasks]
    returns:
        pos_cnt: [C]
        neg_cnt: [C]
        pos_weight: [C] for BCEWithLogitsLoss(pos_weight=...)
    """
    y = graph["n"].y.float()
    pos_cnt = y.sum(dim=0)
    total_cnt = torch.tensor(float(y.size(0)), device=y.device)
    neg_cnt = total_cnt - pos_cnt

    pos_weight = neg_cnt / torch.clamp(pos_cnt, min=1.0)

    return pos_cnt, neg_cnt, pos_weight


def evaluate_dummy_per_task(graph, task_names):
    y = graph["n"].y.bool().cpu()

    preds = torch.ones_like(y)

    rows = []
    for task_id, task_name in enumerate(task_names):
        yc = y[:, task_id]
        pc = preds[:, task_id]
        tp = int((pc & yc).sum().item())
        fp = int((pc & (~yc)).sum().item())
        fn = int(((~pc) & yc).sum().item())
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = (
            0.0
            if (precision + recall) == 0
            else 2 * precision * recall / (precision + recall)
        )
        rows.append(
            {
                "task": task_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pos_cnt": int(yc.sum().item()),
                "pos_rate": float(yc.float().mean().item()),
            }
        )
    return rows


def build_criterion_for_client(graph, cfg, device):
    """
    Supports:
      - minority_class_weight = None   -> plain BCE
      - minority_class_weight = "auto" -> per-task pos_weight from training labels
      - minority_class_weight = number -> constant scalar pos_weight for all tasks
    """
    mcw = cfg.get("minority_class_weight", None)

    if mcw is None:
        return nn.BCEWithLogitsLoss()

    if mcw == "auto":
        pos_cnt, neg_cnt, pos_weight = get_label_stats_from_graph(graph)
        pos_weight = pos_weight.to(device)
        pos_weight = torch.sqrt(pos_weight)
        print("[criterion] BCEWithLogitsLoss with AUTO pos_weight")
        print("  pos_cnt   :", pos_cnt)
        print("  neg_cnt   :", neg_cnt)
        print("  pos_weight:", pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@torch.no_grad()
def get_fullbatch_logits_and_labels(model, graph, device):
    model.eval()
    graph = graph.to(device)

    x_in = {"n": graph["n"].x}
    y_true = graph["n"].y
    edge_in = {
        ("n", "fwd", "n"): graph[("n", "fwd", "n")].edge_index,
        ("n", "rev", "n"): graph[("n", "rev", "n")].edge_index,
    }

    edge_attr_dict = {}
    for rel in [("n", "fwd", "n"), ("n", "rev", "n")]:
        if "edge_attr" in graph[rel]:
            ea = graph[rel].edge_attr
            if ea.dtype != torch.long:
                ea = ea.long()
            edge_attr_dict[rel] = ea

    logits = model(
        x_in,
        edge_in,
        edge_attr_dict=edge_attr_dict,
        device=device,
    )
    return logits.detach().cpu(), y_true.detach().cpu()


def _safe_average_precision(y_true_np: np.ndarray, y_score_np: np.ndarray) -> float:
    # AP is undefined when there are no positives in y_true.
    if y_true_np.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true_np, y_score_np))


def tune_thresholds_from_logits(
    logits_cpu: torch.Tensor, y_cpu: torch.Tensor, grid=None
):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    probs = torch.sigmoid(logits_cpu).numpy()
    y = y_cpu.numpy().astype(np.int32)

    thresholds = []

    for c in range(y.shape[1]):
        yc = y[:, c]
        pc = probs[:, c]

        best_t = 0.5
        best_f1 = -1.0

        for t in grid:
            pred = (pc >= t).astype(np.int32)
            tp = int(((pred == 1) & (yc == 1)).sum())
            fp = int(((pred == 1) & (yc == 0)).sum())
            fn = int(((pred == 0) & (yc == 1)).sum())
            precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
            recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
            f1 = (
                0.0
                if (precision + recall) == 0
                else 2 * precision * recall / (precision + recall)
            )

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        thresholds.append(best_t)

    return thresholds


def run_fedavg(
    data: ClientData,
    cfg: Dict,
    seed: int,
    out_dir: str,
) -> Dict:

    set_seed(seed)

    homo_train, homo_val, homo_test = [], [], []
    homo_train.append(data.train_g)
    homo_val.append(data.val_g)
    homo_test.append(data.test_g)

    x_dim = int(homo_train[0].x.size(-1))
    out_dim = int(homo_train[0].y.size(-1))
    print(f"x_dim={x_dim}, out_dim={out_dim}")

    # Global port vocab
    if cfg.get("use_port_ids", True):
        in_vocab, out_vocab = compute_global_port_vocab(homo_train, homo_val, homo_test)
        port_vocab = max(in_vocab, out_vocab)
        in_vocab = out_vocab = port_vocab
    else:
        in_vocab = out_vocab = 0

    print(f"Global port vocab: in={in_vocab}, out={out_vocab}")

    # Global degree hists
    deg_fwd_hist, deg_rev_hist = compute_global_degree_hists(homo_train)

    local_rows = []

    train_graph = data.train_h
    val_graph = data.val_h
    test_graph = data.test_h

    # 1) initialize local model from global state
    client_criterion = build_criterion_for_client(train_graph, cfg, DEVICE)
    # usually if we do full batch training, we set ego_dim = graph.nodes but now test ego_dim = 0
    # ego_dim = data.train_g.num_nodes
    ego_dim = 0
    local_model = make_model(
        cfg,
        x_dim,
        out_dim,
        deg_fwd_hist,
        deg_rev_hist,
        ego_dim,
        in_vocab,
        out_vocab,
    ).to(DEVICE)

    print("ego_dim:", ego_dim)

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    best_state = None
    best_thresholds = None
    best_epoch = -1
    best_score = -1.0  # mean val auprc
    best_f1 = -1.0  # tie-break
    patience = 20
    bad_epochs = 0

    local_rows.extend(
        [
            {"seed": seed, "graph_id": data.data_dir, "phase": "baseline", **row}
            for row in evaluate_dummy_per_task(val_graph, TASKS)
        ]
    )
    for local_epoch in range(1, ROUNDS * LOCAL_EPOCHS + 1):
        print(f"\n=== train client (full-batch) {data.data_dir} ===")

        train_loss = train_epoch_fullbatch(
            local_model,
            train_graph,
            optimizer,
            client_criterion,
            DEVICE,
        )

        # get predictions' logits
        val_logits_prediction, val_y_true = get_fullbatch_logits_and_labels(
            local_model, val_graph, DEVICE
        )

        # tune the threshold on validation set
        tuned_thresholds = tune_thresholds_from_logits(
            val_logits_prediction, val_y_true
        )

        # get the tuned performances
        val_metrics = evaluate_fullbatch(
            local_model,
            val_graph,
            client_criterion,
            DEVICE,
            thresholds=tuned_thresholds,
        )

        # add the scalar metrics
        local_rows.append(
            {
                "seed": seed,
                "graph_id": data.data_dir,
                "phase": "val",
                "local_epoch": local_epoch,
                "train_loss": float(train_loss),
                "val_loss": val_metrics["scalar"]["loss"],
                "pair_acc": val_metrics["scalar"]["pair_acc"],
                "subset_acc": val_metrics["scalar"]["subset_acc"],
                "micro_f1": val_metrics["scalar"]["micro_f1"],
                "macro_f1": val_metrics["scalar"]["macro_f1"],
                "num_nodes": val_metrics["counts"]["num_nodes"],
            }
        )

        print("local_epoch:", local_epoch, "train_loss:", train_loss)
        print("local_epoch:", local_epoch, "val_loss:", val_metrics["scalar"]["loss"])

        # add the per-task metrics
        for task_id, task_name in enumerate(TASKS):
            local_rows.append(
                {
                    "seed": seed,
                    "graph_id": data.data_dir,
                    "phase": "val_task",
                    "local_epoch": local_epoch,
                    "task": task_name,
                    "tp": val_metrics["per_task"]["tp"][task_id],
                    "fp": val_metrics["per_task"]["fp"][task_id],
                    "tn": val_metrics["per_task"]["tn"][task_id],
                    "fn": val_metrics["per_task"]["fn"][task_id],
                    "precision": val_metrics["per_task"]["precision"][task_id],
                    "recall": val_metrics["per_task"]["recall"][task_id],
                    "f1": val_metrics["per_task"]["f1"][task_id],
                    "threshold": val_metrics["per_task"]["threshold"][task_id],
                    "auprc": val_metrics["per_task"]["auprc"][task_id],
                    "pos_cnt": val_metrics["counts"]["pos_cnt"][task_id],
                    "pos_rate": val_metrics["counts"]["pos_rate"][task_id],
                }
            )

        # get the mean auprc and mean f1
        mean_val_auprc = np.mean(val_metrics["per_task"]["auprc"])
        mean_val_f1 = val_metrics["scalar"]["macro_f1"]

        print(mean_val_auprc)
        print(mean_val_f1)

        if (mean_val_auprc > best_score) or (
            np.isclose(mean_val_auprc, best_score) and mean_val_f1 > best_f1
        ):
            best_score = mean_val_auprc
            best_f1 = mean_val_f1
            best_epoch = local_epoch
            best_thresholds = list(tuned_thresholds)
            best_state = {
                k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(
                    f"[early-stop] stop at epoch={local_epoch} (best_epoch={best_epoch}, best_val_mean_auprc={best_score:.4f})"
                )
                break

    local_model.load_state_dict(best_state)

    for split_name, split_graph in [
        ("train", train_graph),
        ("val", val_graph),
        ("test", test_graph),
    ]:
        # get the tuned performances
        metrics = evaluate_fullbatch(
            local_model,
            split_graph,
            client_criterion,
            DEVICE,
            thresholds=best_thresholds,
        )

        # add the scalar metrics
        local_rows.append(
            {
                "seed": seed,
                "graph_id": data.data_dir,
                "phase": f"best_{split_name}",
                "best_epoch": best_epoch,
                "best_val_mean_auprc": best_score,
                "best_val_mean_f1": best_f1,
                "eval_loss": metrics["scalar"]["loss"],
                "micro_f1": metrics["scalar"]["micro_f1"],
                "macro_f1": metrics["scalar"]["macro_f1"],
                "num_nodes": val_metrics["counts"]["num_nodes"],
            }
        )

        # add the per-task metrics
        for task_id, task_name in enumerate(TASKS):
            local_rows.append(
                {
                    "seed": seed,
                    "graph_id": data.data_dir,
                    "phase": f"best_{split_name}_task",
                    "best_epoch": best_epoch,
                    "threshold": best_thresholds[task_id],
                    "task": task_name,
                    "tp": metrics["per_task"]["tp"][task_id],
                    "fp": metrics["per_task"]["fp"][task_id],
                    "tn": metrics["per_task"]["tn"][task_id],
                    "fn": metrics["per_task"]["fn"][task_id],
                    "precision": metrics["per_task"]["precision"][task_id],
                    "recall": metrics["per_task"]["recall"][task_id],
                    "f1": metrics["per_task"]["f1"][task_id],
                    "auprc": metrics["per_task"]["auprc"][task_id],
                    "pos_cnt": metrics["counts"]["pos_cnt"][task_id],
                    "pos_rate": metrics["counts"]["pos_rate"][task_id],
                }
            )

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(Path(out_dir) / "local.csv", index=False)


def load_graphs(chosen_df):

    id_to_client: Dict[int, ClientData] = {}

    for _, row in chosen_df.iterrows():
        cid = int(row["graph_id"])
        data_dir = row["data_dir"]

        tr, va, te = load_client_from_dir(data_dir)
        train_h = make_bidirected_hetero(tr)
        val_h = make_bidirected_hetero(va)
        test_h = make_bidirected_hetero(te)

        id_to_client[cid] = ClientData(
            client_id=str(cid),
            data_dir=data_dir,
            train_g=tr,
            val_g=va,
            test_g=te,
            train_h=train_h,
            val_h=val_h,
            test_h=test_h,
        )

    return id_to_client


def main():

    ensure_dir(OUT_ROOT)

    chosen_df = pd.read_csv(TRAIN_SELECTED_GRAPHS_CSV_PATH)
    print(chosen_df)

    id_to_client = load_graphs(chosen_df)

    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)
    print(base_cfg)

    # load existing experiment log so we APPEND instead of overwrite
    existing_exp_df, existing_run_dirs = load_experiment_runs(
        TRAIN_SELECTED_EXPERIMENT_LOG_PATH
    )

    if existing_exp_df.empty:
        log_experiments = []
    else:
        log_experiments = existing_exp_df.to_dict(orient="records")
    sweep_space = list(
        product(
            SEEDS,
            MCW,
            NUM_LAYERS,
            LRS,
            WEIGHT_DECAYS,
            DROPOUTS,
            HIDDEN_DIMS,
        )
    )
    total = len(sweep_space) * 3
    print(total)
    now = 0
    for seed, mcw, num_layers, lr, weight_decay, dropout, hidden_dim in sweep_space:
        cfg = dict(base_cfg)
        cfg["minority_class_weight"] = mcw
        cfg["num_layers"] = num_layers
        cfg["lr"] = lr
        cfg["weight_decay"] = weight_decay
        cfg["dropout"] = dropout
        cfg["hidden_dim"] = hidden_dim

        model_tag = build_model_tag(
            mcw,
            num_layers,
            lr,
            weight_decay,
            dropout,
            hidden_dim,
        )
        for _, row in chosen_df.iterrows():
            dataset_id = row["dataset_id"]

            run_dir = Path(OUT_ROOT) / (
                f"{dataset_id}"
                f"_rounds{ROUNDS}"
                f"_epoch{LOCAL_EPOCHS}"
                f"_{model_tag}"
                f"_seed{seed}"
            )

            # skip already-finished runs
            if str(run_dir) in existing_run_dirs:
                print(f"[skip] already logged: {run_dir}")
                continue

            ensure_dir(run_dir)

            run_fedavg(
                data=id_to_client[row["graph_id"]],
                cfg=cfg,
                seed=seed,
                out_dir=str(run_dir),
            )
            now += 1
            exp_row = row.to_dict()
            exp_row["out_dir"] = str(run_dir)
            exp_row["rounds"] = ROUNDS
            exp_row["local_epochs"] = LOCAL_EPOCHS
            for key, value in {
                "mcw": mcw,
                "num_layers": num_layers,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "seed": seed,
            }.items():
                exp_row[key] = value
            exp_row["model_tag"] = model_tag

            log_experiments.append(exp_row)
            existing_run_dirs.add(str(run_dir))

            print(f"logged performances under {run_dir}")
            print(f"=======================================================")
            print(f"===================={now} / {total}====================")
            print(f"=======================================================")

    logging_df = pd.DataFrame(log_experiments)
    logging_df.to_csv(TRAIN_SELECTED_EXPERIMENT_LOG_PATH, index=False)
    print(f"logged experiments -> {TRAIN_SELECTED_EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()
