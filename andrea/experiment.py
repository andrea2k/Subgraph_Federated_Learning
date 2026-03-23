from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product

import pandas as pd
import torch
from torch import nn

from andrea.experiment_select_clients_utils import (
    ClientData,
    choose_subsets_and_load_clients,
    parse_subset_clients,
)
from andrea.multigraph_generation import TASKS
from models.pna_reverse_mp import PNANetReverseMP
from utils.graph_helpers import max_port_cols
from utils.seed import set_seed

CSV_PATH = "./andrea/test_generation_parameters.csv"
EXPERIMENT_PATH = Path("./andrea/experiment.csv")
SUBSET_PATH = "./andrea/heterogeneity.csv"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
OUT_ROOT = "andrea/runs/experiment_fedavg_heterogeneity"

SEEDS = [0, 1, 2, 3]
ROUNDS = 100
LOCAL_EPOCHS = 1
CLIENT_FRACTION = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    if experiment_path.exists():
        existing_exp_df = pd.read_csv(experiment_path)
        existing_run_dirs = set(existing_exp_df["out_dir"].astype(str).tolist())
    else:
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
    threshold: float = 0.5,  # kept for future metrics
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

    if not getattr(model, "_eval_dbg_printed", False):
        f_ea = edge_attr_dict[("n", "fwd", "n")]
        r_ea = edge_attr_dict[("n", "rev", "n")]
        print(
            f"[EVALUATE MODE] full-batch | ego_dim={getattr(model, 'ego_dim', 0)}\n"
            f"[NODES]: {n_nodes}\n"
            f"[PORT] fwd edge_attr: {tuple(f_ea.shape)} (dtype={f_ea.dtype}) | "
            f"rev edge_attr: {tuple(r_ea.shape)} (dtype={r_ea.dtype})"
        )
        model._eval_dbg_printed = True

    logits = model(
        x_in,
        edge_in,
        edge_attr_dict=edge_attr_dict,
        device=device,
    )

    loss = criterion(logits, y_true.float())

    probs = torch.sigmoid(logits)
    preds = probs > threshold
    y_bool = y_true.bool()

    # pair accuracy / element-wise accuracy
    pair_acc = (preds == y_bool).float().mean().item()

    # subset / exact-match accuracy, turn each row into one boolean
    subset_acc = (preds == y_bool).all(dim=1).float().mean().item()

    # micro metrics
    tp = (preds & y_bool).sum().item()
    fp = (preds & (~y_bool)).sum().item()
    fn = ((~preds) & y_bool).sum().item()

    micro_prec = tp / max(tp + fp, 1)
    micro_rec = tp / max(tp + fn, 1)
    micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-12)

    # per-task metrics
    C = y_true.size(1)
    tp_task, fp_task, tn_task, fn_task = [], [], [], []
    prec_task, rec_task, f1_task = [], [], []

    for c in range(C):
        p = preds[:, c]
        y = y_bool[:, c]

        tp_c = int((p & y).sum().item())
        fp_c = int((p & (~y)).sum().item())
        tn_c = int(((~p) & (~y)).sum().item())
        fn_c = int(((~p) & y).sum().item())

        prec_c = tp_c / max(tp_c + fp_c, 1)
        rec_c = tp_c / max(tp_c + fn_c, 1)
        f1_c = 2 * prec_c * rec_c / max(prec_c + rec_c, 1e-12)

        tp_task.append(tp_c)
        fp_task.append(fp_c)
        tn_task.append(tn_c)
        fn_task.append(fn_c)
        prec_task.append(float(prec_c))
        rec_task.append(float(rec_c))
        f1_task.append(float(f1_c))

    macro_f1 = float(sum(f1_task) / max(len(f1_task), 1))
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
        print("[criterion] BCEWithLogitsLoss with AUTO pos_weight")
        print("  pos_cnt   :", pos_cnt)
        print("  neg_cnt   :", neg_cnt)
        print("  pos_weight:", pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def run_fedavg_on_subset(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    out_dir: str,
) -> Dict:

    set_seed(seed)

    homo_train, homo_val, homo_test = [], [], []
    for c in subset_clients:
        homo_train.append(c.train_g)
        homo_val.append(c.val_g)
        homo_test.append(c.test_g)

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

    # ego_dim = 0 for full-batch training/evaluation
    ego_dim = 0
    global_model = make_model(
        cfg, x_dim, out_dim, deg_fwd_hist, deg_rev_hist, ego_dim, in_vocab, out_vocab
    ).to(DEVICE)
    global_state = {
        k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()
    }

    fedavg_rows = []
    client_rows = []

    num_clients = len(subset_clients)

    py_rng = random.Random(seed)
    for rnd in range(1, ROUNDS + 1):
        print(f"\n=== FedAvg round {rnd} ===")

        m = max(1, int(CLIENT_FRACTION * num_clients))
        selected = (
            list(range(num_clients))
            if m == num_clients
            else py_rng.sample(range(num_clients), m)
        )

        client_states = []
        client_weights = []

        for idx in selected:
            c = subset_clients[idx]
            train_graph = c.train_h
            test_graph = c.test_h

            # 1) initialize local model from global state
            client_criterion = build_criterion_for_client(train_graph, cfg, DEVICE)

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

            local_model.load_state_dict(global_model.state_dict())

            # 2) evaluate current global model on this client BEFORE local training

            pre_metrics = evaluate_fullbatch(
                local_model, test_graph, client_criterion, DEVICE
            )

            client_rows.append(
                {
                    "seed": seed,
                    "round": rnd,
                    "client_id": c.client_id,
                    "phase": "pre_local_val",
                    "eval_loss": pre_metrics["scalar"]["loss"],
                    "pair_acc": pre_metrics["scalar"]["pair_acc"],
                    "subset_acc": pre_metrics["scalar"]["subset_acc"],
                    "micro_f1": pre_metrics["scalar"]["micro_f1"],
                    "macro_f1": pre_metrics["scalar"]["macro_f1"],
                    "test_num_nodes": pre_metrics["counts"]["num_nodes"],
                }
            )

            for task_id in range(out_dim):
                client_rows.append(
                    {
                        "seed": seed,
                        "round": rnd,
                        "client_id": c.client_id,
                        "phase": "pre_local_val_task",
                        "task": TASKS[task_id],
                        "tp": pre_metrics["per_task"]["tp"][task_id],
                        "fp": pre_metrics["per_task"]["fp"][task_id],
                        "tn": pre_metrics["per_task"]["tn"][task_id],
                        "fn": pre_metrics["per_task"]["fn"][task_id],
                        "precision": pre_metrics["per_task"]["precision"][task_id],
                        "recall": pre_metrics["per_task"]["recall"][task_id],
                        "f1": pre_metrics["per_task"]["f1"][task_id],
                        "pos_cnt": pre_metrics["counts"]["pos_cnt"][task_id],
                        "pos_rate": pre_metrics["counts"]["pos_rate"][task_id],
                    }
                )
            # 3) train model (full-batch training)

            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
            )

            for local_epoch in range(LOCAL_EPOCHS):
                print(f"\n=== train client (full-batch) {c.client_id} ===")
                train_loss = train_epoch_fullbatch(
                    local_model,
                    train_graph,
                    optimizer,
                    client_criterion,
                    DEVICE,
                )
                client_rows.append(
                    {
                        "seed": seed,
                        "round": rnd,
                        "client_id": c.client_id,
                        "phase": "training",
                        "local_epoch": local_epoch,
                        "train_loss": float(train_loss),
                        "train_num_nodes": int(train_graph["n"].num_nodes),
                    }
                )
                print("local_epoch:", local_epoch, "train_loss:", train_loss)

            # 4) evaluate model again (full-batch evaluation)

            post_metrics = evaluate_fullbatch(
                local_model, test_graph, client_criterion, DEVICE
            )

            client_rows.append(
                {
                    "seed": seed,
                    "round": rnd,
                    "client_id": c.client_id,
                    "phase": "post_local_val",
                    "eval_loss": post_metrics["scalar"]["loss"],
                    "pair_acc": post_metrics["scalar"]["pair_acc"],
                    "subset_acc": post_metrics["scalar"]["subset_acc"],
                    "micro_f1": post_metrics["scalar"]["micro_f1"],
                    "macro_f1": post_metrics["scalar"]["macro_f1"],
                    "test_num_nodes": post_metrics["counts"]["num_nodes"],
                }
            )
            for task_id in range(out_dim):
                client_rows.append(
                    {
                        "seed": seed,
                        "round": rnd,
                        "client_id": c.client_id,
                        "phase": "post_local_val_task",
                        "task": TASKS[task_id],
                        "tp": post_metrics["per_task"]["tp"][task_id],
                        "fp": post_metrics["per_task"]["fp"][task_id],
                        "tn": post_metrics["per_task"]["tn"][task_id],
                        "fn": post_metrics["per_task"]["fn"][task_id],
                        "precision": post_metrics["per_task"]["precision"][task_id],
                        "recall": post_metrics["per_task"]["recall"][task_id],
                        "f1": post_metrics["per_task"]["f1"][task_id],
                        "pos_cnt": post_metrics["counts"]["pos_cnt"][task_id],
                        "pos_rate": post_metrics["counts"]["pos_rate"][task_id],
                    }
                )

            n_k = int(train_graph["n"].num_nodes)
            client_weights.append(n_k)
            client_states.append(
                {
                    k: v.detach().cpu().clone()
                    for k, v in local_model.state_dict().items()
                }
            )

        global_state = fedavg_state_dict(client_states, client_weights)
        global_model.load_state_dict(global_state, strict=True)

        for c in subset_clients:
            train_graph = c.train_h
            test_graph = c.test_h

            # 1) initialize local model from global state
            client_criterion = build_criterion_for_client(train_graph, cfg, DEVICE)

            metrics = evaluate_fullbatch(
                global_model, test_graph, client_criterion, DEVICE
            )

            fedavg_rows.append(
                {
                    "seed": seed,
                    "round": rnd,
                    "client_id": c.client_id,
                    "phase": "global_val_client",
                    "eval_loss": metrics["scalar"]["loss"],
                    "pair_acc": metrics["scalar"]["pair_acc"],
                    "subset_acc": metrics["scalar"]["subset_acc"],
                    "micro_f1": metrics["scalar"]["micro_f1"],
                    "macro_f1": metrics["scalar"]["macro_f1"],
                    "test_num_nodes": metrics["counts"]["num_nodes"],
                }
            )

            for task_id in range(out_dim):
                fedavg_rows.append(
                    {
                        "seed": seed,
                        "round": rnd,
                        "client_id": c.client_id,
                        "phase": "global_val_client_task",
                        "task": TASKS[task_id],
                        "tp": metrics["per_task"]["tp"][task_id],
                        "fp": metrics["per_task"]["fp"][task_id],
                        "tn": metrics["per_task"]["tn"][task_id],
                        "fn": metrics["per_task"]["fn"][task_id],
                        "precision": metrics["per_task"]["precision"][task_id],
                        "recall": metrics["per_task"]["recall"][task_id],
                        "f1": metrics["per_task"]["f1"][task_id],
                        "pos_cnt": metrics["counts"]["pos_cnt"][task_id],
                        "pos_rate": metrics["counts"]["pos_rate"][task_id],
                    }
                )

    fedavg_df = pd.DataFrame(fedavg_rows)
    fedavg_df.to_csv(Path(out_dir) / "fedavg.csv", index=False)

    client_df = pd.DataFrame(client_rows)
    client_df.to_csv(Path(out_dir) / "client.csv", index=False)

    local_rows = []

    for c in subset_clients:
        train_graph = c.train_h
        test_graph = c.test_h

        # 1) initialize local model from global state
        client_criterion = build_criterion_for_client(train_graph, cfg, DEVICE)

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

        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

        for local_epoch in range(1, ROUNDS * LOCAL_EPOCHS + 1):
            print(f"\n=== train client (full-batch) {c.client_id} ===")
            train_loss = train_epoch_fullbatch(
                local_model,
                train_graph,
                optimizer,
                client_criterion,
                DEVICE,
            )
            local_rows.append(
                {
                    "seed": seed,
                    "client_id": c.client_id,
                    "phase": "training",
                    "local_epoch": local_epoch,
                    "train_loss": float(train_loss),
                    "num_train_nodes": int(train_graph["n"].num_nodes),
                }
            )
            print("local_epoch:", local_epoch, "train_loss:", train_loss)

            metrics = evaluate_fullbatch(
                local_model, test_graph, client_criterion, DEVICE
            )
            local_rows.append(
                {
                    "seed": seed,
                    "local_epoch": local_epoch,
                    "client_id": c.client_id,
                    "phase": "val",
                    "eval_loss": metrics["scalar"]["loss"],
                    "pair_acc": metrics["scalar"]["pair_acc"],
                    "subset_acc": metrics["scalar"]["subset_acc"],
                    "micro_f1": metrics["scalar"]["micro_f1"],
                    "macro_f1": metrics["scalar"]["macro_f1"],
                    "num_nodes": metrics["counts"]["num_nodes"],
                }
            )
            for task_id in range(out_dim):
                local_rows.append(
                    {
                        "seed": seed,
                        "local_epoch": local_epoch,
                        "client_id": c.client_id,
                        "phase": "val_task",
                        "task": TASKS[task_id],
                        "tp": metrics["per_task"]["tp"][task_id],
                        "fp": metrics["per_task"]["fp"][task_id],
                        "tn": metrics["per_task"]["tn"][task_id],
                        "fn": metrics["per_task"]["fn"][task_id],
                        "precision": metrics["per_task"]["precision"][task_id],
                        "recall": metrics["per_task"]["recall"][task_id],
                        "f1": metrics["per_task"]["f1"][task_id],
                        "pos_cnt": metrics["counts"]["pos_cnt"][task_id],
                        "pos_rate": metrics["counts"]["pos_rate"][task_id],
                    }
                )

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(Path(out_dir) / "local.csv", index=False)


# =========================
# Main pipeline
# =========================
def main():

    ensure_dir(OUT_ROOT)

    df_heterogeneity = pd.read_csv(SUBSET_PATH).copy()

    TARGET_METRICS = [
        "label_prev_jsd_mean",
        "motif_profile_jsd_mean",
        "in_degree_jsd_mean",
    ]

    chosen_df, id_to_client = choose_subsets_and_load_clients(
        df_heterogeneity=df_heterogeneity,
        target_metrics=TARGET_METRICS,
        csv_path=CSV_PATH,
        preview_cols=[
            "subset_size",
            "target_metric",
            "level",
            "subset_id",
            "subset_clients",
            "label_prev_jsd_mean",
            "motif_profile_jsd_mean",
            "in_degree_jsd_mean",
        ],
        verbose=True,
    )

    cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)
    existing_exp_df, existing_run_dirs = load_experiment_runs(EXPERIMENT_PATH)

    log_experiments = []

    for seed, mcw, num_layers in product(SEEDS, ["auto", None], [3, 6]):

        cfg["minority_class_weight"] = mcw
        cfg["num_layers"] = num_layers

        for _, row in chosen_df.iterrows():

            subset_clients_ids = parse_subset_clients(row["subset_clients"])
            subset_clients = [id_to_client[cid] for cid in subset_clients_ids]
            subset_id = str(row["subset_id"])

            run_dir = (
                Path(OUT_ROOT)
                / f"{subset_id}_rounds{ROUNDS}_epoch{LOCAL_EPOCHS}_mcw{mcw}_layers{num_layers}_seed{seed}"
            )

            if str(run_dir) in existing_run_dirs:
                print(f"skip existing run: {str(run_dir)}")
                break

            ensure_dir(run_dir)

            run_fedavg_on_subset(
                subset_clients=subset_clients,
                cfg=cfg,
                seed=seed,
                out_dir=str(run_dir),
            )

            exp_row = row.to_dict()
            exp_row["out_dir"] = str(run_dir)
            exp_row["rounds"] = ROUNDS
            exp_row["local_epochs"] = LOCAL_EPOCHS
            exp_row["mcw"] = mcw
            exp_row["num_layers"] = num_layers
            exp_row["seed"] = seed
            log_experiments.append(exp_row)
            print(f"logged performances under {run_dir}")
            break

    # only append if there are new runs
    if log_experiments:
        new_df = pd.DataFrame(log_experiments)
        if not existing_exp_df.empty:
            exp_df = pd.concat([existing_exp_df, new_df], ignore_index=True)
        else:
            exp_df = new_df
        exp_df.to_csv(EXPERIMENT_PATH, index=False)
        print(f"logged experiments -> {EXPERIMENT_PATH}")
    else:
        print("no new experiments to log")


if __name__ == "__main__":
    main()
