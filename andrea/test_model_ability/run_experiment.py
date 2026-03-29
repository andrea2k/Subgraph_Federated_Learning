from __future__ import annotations
import os
import json
import time
import torch
import pandas as pd
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple

from andrea.experiment_select_clients_utils import (
    ClientData,
    load_client_from_dir,
    make_bidirected_hetero,
)
from andrea.multigraph_generation import TASKS
from andrea.training_eval_helpers import (
    build_hetero_neighbor_loader,
    build_criterion_for_client,
    train_epoch_neighbor,
    evaluate_loader,
)
from models.pna_reverse_mp import PNANetReverseMP
from utils.graph_helpers import max_port_cols
from utils.seed import set_seed

TRAIN_SELECTED_GRAPHS_CSV_PATH = (
    "./andrea/test_model_ability/graphs_profile_selected.csv"
)
TRAIN_SELECTED_EXPERIMENT_LOG_PATH = Path(
    "./andrea/test_model_ability/experiment_small_graph_test_test.csv"
)
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
OUT_ROOT = "andrea/test_model_ability/model_logs_small_graph_test_test"
LOCAL_RUNS_ROOT = "andrea/local_runs"

ROUNDS = 1
LOCAL_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [0, 1, 2, 3, 4]
MCW = ["auto"]
NUM_LAYERS = [6]
LRS = [0.001]
WEIGHT_DECAYS = [0.0001]
DROPOUTS = [0.1]
HIDDEN_DIMS = [64]
USE_EGO_IDS = [True]
BATCH_SIZE = [64]


def build_model_tag(
    mcw, num_layers, lr, weight_decay, dropout, hidden_dim, use_ego_ids, batch_size
):
    return (
        f"mcw{mcw}"
        f"_layers{num_layers}"
        f"_lr{lr}"
        f"_wd{weight_decay}"
        f"_do{dropout}"
        f"_hd{hidden_dim}"
        f"_ego{use_ego_ids}"
        f"_bs{batch_size}"
    )


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_cfg(config_path: str, key: str) -> Dict:
    with open(config_path, "r") as f:
        all_cfg = json.load(f)
    cfg_obj = all_cfg[key]
    cfg = dict(cfg_obj.get("default_hparams", {}))
    cfg["model_name"] = cfg_obj.get("model_name", key)
    cfg["use_ego_ids"] = cfg_obj.get("use_ego_ids", True)
    cfg["use_port_ids"] = cfg_obj.get("use_port_ids", True)
    cfg["use_mini_batch"] = cfg_obj.get("use_mini_batch", False)
    cfg["batch_size"] = cfg_obj.get("batch_size", cfg.get("batch_size", 256))
    cfg["port_emb_dim"] = cfg_obj.get("port_emb_dim", cfg.get("port_emb_dim", 8))
    cfg["num_epochs"] = cfg_obj.get("num_epochs", 100)

    return cfg


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

    return PNANetReverseMP(
        in_dim=x_dim,
        out_dim=out_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        deg_fwd=deg_fwd_hist,
        deg_rev=deg_rev_hist,
        ego_dim=ego_dim,
        # combine="sum",
        in_port_vocab_size=in_vocab,
        out_port_vocab_size=out_vocab,
        port_emb_dim=(
            cfg.get("port_emb_dim", 0) if cfg.get("use_port_ids", True) else None
        ),
    )


def run_single_graph(
    data: ClientData,
    cfg: Dict,
    seed: int,
    out_dir: str,
    local_dir: str,
):
    set_seed(seed)

    homo_train = [data.train_g]
    homo_val = [data.val_g]
    homo_test = [data.test_g]

    x_dim = int(homo_train[0].x.size(-1))
    out_dim = int(homo_train[0].y.size(-1))
    print(f"x_dim={x_dim}, out_dim={out_dim}")
    if cfg.get("use_port_ids", True):
        in_vocab, out_vocab = compute_global_port_vocab(homo_train, homo_val, homo_test)
        port_vocab = max(in_vocab, out_vocab)
        in_vocab = out_vocab = port_vocab
    else:
        in_vocab = out_vocab = 0
    print(f"Global port vocab: in={in_vocab}, out={out_vocab}")

    deg_fwd_hist, deg_rev_hist = compute_global_degree_hists(homo_train)

    train_graph = data.train_h
    val_graph = data.val_h
    test_graph = data.test_h

    client_criterion = build_criterion_for_client(train_graph, cfg, DEVICE)

    use_ego_ids = bool(cfg.get("use_ego_ids", True))
    batch_size = cfg.get("batch_size", 32)
    ego_dim = 1 if use_ego_ids else 0
    neighbors_per_hop = cfg.get("neighbors_per_hop", None)
    print("ego_dim:", ego_dim)
    print("train batch_size:", batch_size)
    print("neighbors per hop", neighbors_per_hop)
    shuffle = True
    train_loader = build_hetero_neighbor_loader(
        train_graph,
        batch_size=batch_size,
        neighbors_per_hop=neighbors_per_hop,
        device=DEVICE,
        shuffle=shuffle,
    )
    val_loader = build_hetero_neighbor_loader(
        val_graph,
        batch_size=batch_size,
        neighbors_per_hop=neighbors_per_hop,
        device=DEVICE,
        shuffle=shuffle,
    )

    test_loader = build_hetero_neighbor_loader(
        test_graph,
        batch_size=batch_size,
        neighbors_per_hop=neighbors_per_hop,
        device=DEVICE,
        shuffle=shuffle,
    )
    print("shuffle:", shuffle)
    model = make_model(
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
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    local_rows = []

    for local_epoch in range(1, ROUNDS * LOCAL_EPOCHS + 1):
        print(f"\n=== train client (neighbor-loader) {data.data_dir} ===")
        print(f"\n=================epoch {local_epoch}==================")

        run_start_epoch = time.perf_counter()

        train_loss = train_epoch_neighbor(
            model,
            train_loader,
            optimizer,
            client_criterion,
            DEVICE,
            use_ego_ids=use_ego_ids,
            ego_dim=ego_dim,
        )

        val_metrics = evaluate_loader(
            model,
            val_loader,
            client_criterion,
            DEVICE,
            use_ego_ids=use_ego_ids,
            ego_dim=ego_dim,
            threshold=0.5,
        )

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
                "macro_pos_f1": val_metrics["scalar"]["macro_pos_f1"],
                "macro_minority_f1": val_metrics["scalar"]["macro_minority_f1"],
                "num_nodes": val_metrics["counts"]["num_nodes"],
            }
        )

        print("local_epoch:", local_epoch, "train_loss:", train_loss)
        print("local_epoch:", local_epoch, "val_loss:", val_metrics["scalar"]["loss"])
        print(
            "val_macro_minority_f1:",
            val_metrics["scalar"]["macro_minority_f1"],
        )
        print(
            "val_minority_f1:",
            val_metrics["per_task"]["minority_f1"],
        )
        print(
            "val_positive_f1:",
            val_metrics["per_task"]["positive_f1"],
        )
        run_epoch = time.perf_counter() - run_start_epoch
        print(f"==this epoch took: {format_seconds(run_epoch)}===")

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
                    "positive_f1": val_metrics["per_task"]["positive_f1"][task_id],
                    "minority_f1": val_metrics["per_task"]["minority_f1"][task_id],
                    "pos_cnt": val_metrics["counts"]["pos_cnt"][task_id],
                    "pos_rate": val_metrics["counts"]["pos_rate"][task_id],
                }
            )

        val_loss = val_metrics["scalar"]["loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = local_epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    for split_name, split_loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:

        metrics = evaluate_loader(
            model,
            split_loader,
            client_criterion,
            DEVICE,
            use_ego_ids=use_ego_ids,
            ego_dim=ego_dim,
            threshold=0.5,
        )

        local_rows.append(
            {
                "seed": seed,
                "graph_id": data.data_dir,
                "phase": f"best_{split_name}",
                "best_epoch": best_epoch,
                "eval_loss": metrics["scalar"]["loss"],
                "micro_f1": metrics["scalar"]["micro_f1"],
                "macro_f1": metrics["scalar"]["macro_f1"],
                "macro_pos_f1": metrics["scalar"]["macro_pos_f1"],
                "macro_minority_f1": metrics["scalar"]["macro_minority_f1"],
                "num_nodes": metrics["counts"]["num_nodes"],
            }
        )

        for task_id, task_name in enumerate(TASKS):
            local_rows.append(
                {
                    "seed": seed,
                    "graph_id": data.data_dir,
                    "phase": f"best_{split_name}_task",
                    "best_epoch": best_epoch,
                    "task": task_name,
                    "tp": metrics["per_task"]["tp"][task_id],
                    "fp": metrics["per_task"]["fp"][task_id],
                    "tn": metrics["per_task"]["tn"][task_id],
                    "fn": metrics["per_task"]["fn"][task_id],
                    "precision": metrics["per_task"]["precision"][task_id],
                    "recall": metrics["per_task"]["recall"][task_id],
                    "f1": metrics["per_task"]["f1"][task_id],
                    "positive_f1": metrics["per_task"]["positive_f1"][task_id],
                    "minority_f1": metrics["per_task"]["minority_f1"][task_id],
                    "pos_cnt": metrics["counts"]["pos_cnt"][task_id],
                    "pos_rate": metrics["counts"]["pos_rate"][task_id],
                }
            )

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(out_dir, index=False)
    local_df.to_csv(local_dir, index=False)


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
    total_start = time.perf_counter()
    ensure_dir(OUT_ROOT)

    chosen_df = pd.read_csv(TRAIN_SELECTED_GRAPHS_CSV_PATH)
    print(chosen_df)
    id_to_client = load_graphs(chosen_df)
    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)

    log_experiments = []

    sweep_space = list(
        product(
            SEEDS,
            MCW,
            NUM_LAYERS,
            LRS,
            WEIGHT_DECAYS,
            DROPOUTS,
            HIDDEN_DIMS,
            USE_EGO_IDS,
            BATCH_SIZE,
        )
    )

    total = len(sweep_space) * len(chosen_df)
    print(total)
    now = 0
    for (
        seed,
        mcw,
        num_layers,
        lr,
        weight_decay,
        dropout,
        hidden_dim,
        use_ego_ids,
        batch_size,
    ) in sweep_space:
        cfg = dict(base_cfg)
        cfg["minority_class_weight"] = mcw
        cfg["num_layers"] = num_layers
        cfg["lr"] = lr
        cfg["weight_decay"] = weight_decay
        cfg["dropout"] = dropout
        cfg["hidden_dim"] = hidden_dim
        cfg["use_ego_ids"] = use_ego_ids
        cfg["batch_size"] = batch_size

        model_tag = build_model_tag(
            mcw,
            num_layers,
            lr,
            weight_decay,
            dropout,
            hidden_dim,
            use_ego_ids,
            batch_size,
        )

        for _, row in chosen_df.iterrows():
            dataset_id = row["dataset_id"]
            run_dir = Path(OUT_ROOT) / (
                f"{dataset_id}"
                f"_rounds{ROUNDS}"
                f"_epoch{LOCAL_EPOCHS}"
                f"_{model_tag}"
                f"_seed{seed}"
                f".csv"
            )
            # add to local_runs, such that we know the run already exist during local vs fed experiment
            local_run_dir = Path(LOCAL_RUNS_ROOT) / (
                f"{dataset_id}"
                f"_rounds{ROUNDS}"
                f"_epoch{LOCAL_EPOCHS}"
                f"_{model_tag}"
                f"_seed{seed}"
                f".csv"
            )
            exp_row = row.to_dict()
            exp_row["out_dir"] = str(run_dir)
            exp_row["rounds"] = ROUNDS
            exp_row["local_epochs"] = LOCAL_EPOCHS
            exp_row["selection_metric"] = "val_macro_minority_f1"
            for key, value in {
                "mcw": mcw,
                "num_layers": num_layers,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "use_ego_ids": use_ego_ids,
                "batch_size": batch_size,
                "seed": seed,
            }.items():
                exp_row[key] = value
            exp_row["model_tag"] = model_tag

            if run_dir.exists():
                print(f"[skip] model-log csv already exists: {run_dir}")
                log_experiments.append(exp_row)
                now += 1
                print("=======================================================")
                print(f"===================={now} / {total}====================")
                print("=======================================================")
                continue

            run_start = time.perf_counter()
            print(cfg)
            run_single_graph(
                data=id_to_client[row["graph_id"]],
                cfg=cfg,
                seed=seed,
                out_dir=str(run_dir),
                local_dir=str(local_run_dir),
            )
            now += 1
            log_experiments.append(exp_row)

            run_elapsed = time.perf_counter() - run_start
            print(f"logged performances under {run_dir} and {local_run_dir}")
            print("=======================================================")
            print(f"===================={now} / {total}====================")
            print(f"==this training took: {format_seconds(run_elapsed)}===")
            print("=======================================================")

    total_elapsed = time.perf_counter() - total_start
    print(f"\nALL DONE. total time: {format_seconds(total_elapsed)}")
    logging_df = pd.DataFrame(log_experiments)
    logging_df.to_csv(TRAIN_SELECTED_EXPERIMENT_LOG_PATH, index=False)
    print(f"logged experiments -> {TRAIN_SELECTED_EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()
