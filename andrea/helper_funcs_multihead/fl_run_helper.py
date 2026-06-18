from __future__ import annotations

import os
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from andrea.helper_funcs_multihead.load_client_helper import ClientData, format_seconds
from andrea.multigraph_generation import TASKS

from andrea.helper_funcs_multihead.train_eval_helper import (
    build_criterion_for_client,
    build_hetero_neighbor_loader,
    build_model_tag,
    evaluate_loader,
    train_epoch_neighbor,
    train_epoch_neighbor_fedprox,
    augment_batch_x_with_ego,
    unpack_batch_edges,
)
from models.pna_reverse_mp_taskhead import PNANetReverseMP
from utils.graph_helpers import max_port_cols
from utils.seed import set_seed


@dataclass(frozen=True)
class RunPaths:
    csv_path: Path
    ckpt_path: Path


@dataclass
class RuntimeClient:
    client: ClientData
    train_loader: object
    val_loader: object
    test_loader: object
    criterion: object
    num_train_nodes: int


# -----------------------------------------------------------------------------
# filesystem / ids / manifest
# -----------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_root_from_runs_root(runs_root: str | Path) -> Path:
    runs_root = Path(runs_root)
    return ensure_dir(runs_root.parent / "checkpoints")


def _model_tag_from_cfg(cfg: Dict) -> str:
    base_tag = build_model_tag(
        cfg.get("minority_class_weight", None),
        cfg["num_layers"],
        cfg["lr"],
        cfg["weight_decay"],
        cfg["dropout"],
        cfg["hidden_dim"],
        cfg.get("use_ego_ids", True),
        cfg.get("batch_size", 64),
    )

    output_head = str(cfg.get("output_head", "multi"))
    if output_head == "multi" and not base_tag.endswith("_headmulti"):
        return f"{base_tag}_headmulti"

    return base_tag


def _subset_clients_str(subset_clients: Sequence[ClientData]) -> str:
    return "|".join(str(client.graph_id) for client in subset_clients)


def create_local_run_paths(
    local_root: str | Path,
    dataset_id: str,
    rounds: int,
    local_epochs: int,
    model_tag: str,
    seed: int,
) -> RunPaths:
    root = ensure_dir(local_root)
    stem = (
        f"{dataset_id}"
        f"_rounds{rounds}"
        f"_epoch{local_epochs}"
        f"_{model_tag}"
        f"_seed{seed}"
    )
    return RunPaths(
        csv_path=root / f"{stem}.csv",
        ckpt_path=checkpoint_root_from_runs_root(local_root) / f"{stem}.pt",
    )


def create_fedavg_run_paths(
    fedavg_root: str | Path,
    subset_id: str,
    rounds: int,
    local_epochs: int,
    model_tag: str,
    seed: int,
) -> RunPaths:
    root = ensure_dir(fedavg_root)
    stem = (
        f"fedavg"
        f"{subset_id}"
        f"_rounds{rounds}"
        f"_epoch{local_epochs}"
        f"_{model_tag}"
        f"_seed{seed}"
    )
    return RunPaths(
        csv_path=root / f"{stem}.csv",
        ckpt_path=checkpoint_root_from_runs_root(fedavg_root) / f"{stem}.pt",
    )


def create_fedprox_run_paths(
    fedprox_root: str | Path,
    subset_id: str,
    rounds: int,
    local_epochs: int,
    fedprox_mu: float,
    model_tag: str,
    seed: int,
) -> RunPaths:
    root = ensure_dir(fedprox_root)
    stem = (
        f"fedprox"
        f"{subset_id}"
        f"_rounds{rounds}"
        f"_epoch{local_epochs}"
        f"_mu{fedprox_mu}"
        f"_{model_tag}"
        f"_seed{seed}"
    )
    return RunPaths(
        csv_path=root / f"{stem}.csv",
        ckpt_path=checkpoint_root_from_runs_root(fedprox_root) / f"{stem}.pt",
    )


def upsert_experiment_rows(log_csv: str | Path, rows: List[Dict]) -> int:
    log_csv = Path(log_csv)
    if not rows:
        return 0

    new_df = pd.DataFrame(rows)

    if log_csv.exists():
        old_df = pd.read_csv(log_csv)
        full = pd.concat([old_df, new_df], axis=0, ignore_index=True)

    else:
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        full = new_df

    subset = ["out_csv"]
    full = full.drop_duplicates(subset=subset, keep="last")
    full.to_csv(log_csv, index=False)


# -----------------------------------------------------------------------------
# model / runtime construction
# -----------------------------------------------------------------------------
def compute_global_port_vocab(*graph_lists: List) -> tuple[int, int]:
    max_in, max_out = 0, 0
    for graphs in graph_lists:
        for g in graphs:
            mi, mo = max_port_cols(g)
            max_in = max(max_in, int(mi))
            max_out = max(max_out, int(mo))
    return max_in + 1, max_out + 1


def compute_global_degree_hists(
    train_graphs: List,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_in_deg = []
    all_out_deg = []

    for g in train_graphs:
        edge_index = g.edge_index
        num_nodes = g.num_nodes
        in_deg = torch.bincount(edge_index[1], minlength=num_nodes).cpu()
        out_deg = torch.bincount(edge_index[0], minlength=num_nodes).cpu()
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


def build_model_context(clients: List[ClientData], cfg: Dict) -> Dict:
    homo_train = [client.train_g for client in clients]
    homo_val = [client.val_g for client in clients]
    homo_test = [client.test_g for client in clients]

    x_dim = int(homo_train[0].x.size(-1))
    out_dim = int(homo_train[0].y.size(-1))

    if cfg.get("use_port_ids", True):
        in_vocab, out_vocab = compute_global_port_vocab(homo_train, homo_val, homo_test)
        port_vocab = max(in_vocab, out_vocab)
        in_vocab = port_vocab
        out_vocab = port_vocab
    else:
        in_vocab = 0
        out_vocab = 0

    deg_fwd_hist, deg_rev_hist = compute_global_degree_hists(homo_train)
    use_ego_ids = bool(cfg.get("use_ego_ids", True))
    ego_dim = 1 if use_ego_ids else 0

    return {
        "x_dim": x_dim,
        "out_dim": out_dim,
        "in_vocab": in_vocab,
        "out_vocab": out_vocab,
        "deg_fwd_hist": deg_fwd_hist,
        "deg_rev_hist": deg_rev_hist,
        "ego_dim": ego_dim,
        "use_ego_ids": use_ego_ids,
    }


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
        in_port_vocab_size=in_vocab,
        out_port_vocab_size=out_vocab,
        port_emb_dim=(
            cfg.get("port_emb_dim", 0) if cfg.get("use_port_ids", True) else 0
        ),
        output_head=cfg.get("output_head", "multi"),
    )


def _count_parameters(model) -> tuple[int, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return total, trainable


def _interesting_state_keys(model, max_keys: int = 80) -> list[str]:
    tokens = ["head", "output", "classifier", "lin", "mlp"]
    keys = [
        key
        for key in model.state_dict().keys()
        if any(token in key.lower() for token in tokens)
    ]
    return keys[:max_keys]


def print_model_debug_summary(
    *,
    model,
    cfg: Dict,
    ctx: Dict,
    client: ClientData,
    train_loader,
    device: torch.device,
) -> None:
    """
    Debug the actual model that will be trained.

    Confirms:
      - architecture is multihead
      - output_head is multi
      - model class is the task-head PNA model
      - output logits have shape [batch_size, num_tasks]
    """
    total_params, trainable_params = _count_parameters(model)

    print("\n" + "=" * 100)
    print("MODEL DEBUG SUMMARY")
    print("=" * 100)

    print("client graph_id :", client.graph_id)
    print("client dataset  :", client.dataset_id)
    print("model class     :", model.__class__.__module__, model.__class__.__name__)
    print("cfg architecture:", cfg.get("architecture"))
    print("cfg output_head :", cfg.get("output_head"))
    print("ctx x_dim       :", ctx["x_dim"])
    print("ctx out_dim     :", ctx["out_dim"])
    print("ctx ego_dim     :", ctx["ego_dim"])
    print("ctx in_vocab    :", ctx["in_vocab"])
    print("ctx out_vocab   :", ctx["out_vocab"])
    print("total params    :", total_params)
    print("trainable params:", trainable_params)

    print("\nModel object:")
    print(model)

    print("\nSelected state_dict keys / parameter shapes:")
    for key in _interesting_state_keys(model):
        value = model.state_dict()[key]
        print(f"  {key}: {tuple(value.shape)}")

    # One-batch forward shape check.
    was_training = model.training
    model.eval()

    try:
        batch = next(iter(train_loader))
        batch = batch.to(device)

        x_in, y_seed, B = augment_batch_x_with_ego(
            batch,
            use_ego_ids=ctx["use_ego_ids"],
            ego_dim=ctx["ego_dim"],
        )
        edge_in, edge_attr_dict = unpack_batch_edges(batch)

        with torch.no_grad():
            out = model(
                x_in,
                edge_in,
                edge_attr_dict=edge_attr_dict,
                device=device,
            )

        out_seed = out[:B]

        print("\nOne-batch forward check:")
        print("  batch seed nodes B:", B)
        print("  x_in['n'] shape   :", tuple(x_in["n"].shape))
        print("  y_seed shape      :", tuple(y_seed.shape))
        print("  raw out shape     :", tuple(out.shape))
        print("  out_seed shape    :", tuple(out_seed.shape))

        if out_seed.ndim != 2:
            print("  WARNING: out_seed is not 2D.")

        if int(out_seed.shape[-1]) != int(ctx["out_dim"]):
            print(
                "  WARNING: output dimension mismatch:",
                "out_seed.shape[-1]=",
                out_seed.shape[-1],
                "ctx out_dim=",
                ctx["out_dim"],
            )
        else:
            print("  PASS: logits output dimension matches number of tasks.")

        if "label_mask" in batch["n"]:
            label_mask_seed = batch["n"].label_mask[:B].float()
            visible_per_task = label_mask_seed.sum(dim=0).detach().cpu().to(torch.long)
            print(
                "  visible labels per task in first batch:", visible_per_task.tolist()
            )

    finally:
        if was_training:
            model.train()

    print("=" * 100)


def make_optimizer(model, cfg: Dict):
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )


def build_client_loaders(
    client: ClientData,
    cfg: Dict,
    device: torch.device,
) -> tuple[object, object, object]:
    batch_size = int(cfg.get("batch_size", 64))
    neighbors_per_hop = cfg.get("neighbors_per_hop", None)

    train_loader = build_hetero_neighbor_loader(
        client.train_h,
        batch_size=batch_size,
        neighbors_per_hop=neighbors_per_hop,
        device=device,
        shuffle=True,
    )
    val_loader = build_hetero_neighbor_loader(
        client.val_h,
        batch_size=batch_size,
        neighbors_per_hop=neighbors_per_hop,
        device=device,
        shuffle=True,
    )
    test_loader = build_hetero_neighbor_loader(
        client.test_h,
        batch_size=batch_size,
        neighbors_per_hop=neighbors_per_hop,
        device=device,
        shuffle=True,
    )
    return train_loader, val_loader, test_loader


def build_runtime_clients(
    subset_clients: List[ClientData],
    cfg: Dict,
    device: torch.device,
) -> List[RuntimeClient]:
    runtimes: List[RuntimeClient] = []

    for client in subset_clients:
        train_loader, val_loader, test_loader = build_client_loaders(
            client, cfg, device
        )
        criterion = build_criterion_for_client(client.train_h, cfg, device)

        runtimes.append(
            RuntimeClient(
                client=client,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                criterion=criterion,
                num_train_nodes=int(client.train_h["n"].num_nodes),
            )
        )

    return runtimes


# -----------------------------------------------------------------------------
# logging row helpers
# -----------------------------------------------------------------------------
def append_train_row(
    rows: List[Dict],
    *,
    run_type: str,
    algorithm: str,
    subset_id: Optional[str],
    subset_clients: Optional[str],
    seed: int,
    graph_id: str,
    dataset_id: str,
    phase: str,
    split: str,
    train_loss: float,
    num_nodes: int,
    round_idx: Optional[int] = None,
    local_epoch: Optional[int] = None,
) -> None:
    rows.append(
        {
            "run_type": run_type,
            "algorithm": algorithm,
            "subset_id": subset_id,
            "subset_clients": subset_clients,
            "seed": seed,
            "graph_id": graph_id,
            "dataset_id": dataset_id,
            "phase": phase,
            "split": split,
            "task": None,
            "round": round_idx,
            "local_epoch": local_epoch,
            "train_loss": float(train_loss),
            "eval_loss": None,
            "pair_acc": None,
            "subset_acc": None,
            "micro_f1": None,
            "macro_f1": None,
            "macro_pos_f1": None,
            "macro_minority_f1": None,
            "num_nodes": int(num_nodes),
            "tp": None,
            "fp": None,
            "tn": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "positive_f1": None,
            "minority_f1": None,
            "pos_cnt": None,
            "pos_rate": None,
        }
    )


def append_eval_rows(
    rows: List[Dict],
    *,
    run_type: str,
    algorithm: str,
    subset_id: Optional[str],
    subset_clients: Optional[str],
    seed: int,
    phase: str,
    split: str,
    graph_id: str,
    dataset_id: str,
    metrics: Dict,
    round_idx: Optional[int] = None,
    local_epoch: Optional[int] = None,
    eval_mask_mode: str = "full",
    selection_protocol: Optional[str] = None,
    selected_by: Optional[str] = None,
) -> None:
    rows.append(
        {
            "run_type": run_type,
            "algorithm": algorithm,
            "subset_id": subset_id,
            "subset_clients": subset_clients,
            "seed": seed,
            "graph_id": graph_id,
            "dataset_id": dataset_id,
            "phase": phase,
            "split": split,
            "task": None,
            "round": round_idx,
            "local_epoch": local_epoch,
            "train_loss": None,
            "eval_loss": metrics["scalar"]["loss"],
            "pair_acc": metrics["scalar"]["pair_acc"],
            "subset_acc": metrics["scalar"]["subset_acc"],
            "micro_f1": metrics["scalar"]["micro_f1"],
            "macro_f1": metrics["scalar"]["macro_f1"],
            "macro_pos_f1": metrics["scalar"].get("macro_pos_f1"),
            "macro_minority_f1": metrics["scalar"].get("macro_minority_f1"),
            "num_nodes": metrics["counts"]["num_nodes"],
            "tp": None,
            "fp": None,
            "tn": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "positive_f1": None,
            "minority_f1": None,
            "pos_cnt": None,
            "pos_rate": None,
            "eval_mask_mode": eval_mask_mode,
            "selection_protocol": selection_protocol,
            "selected_by": selected_by,
            "visible_pairs": metrics["counts"].get("visible_pairs"),
            "total_pairs": metrics["counts"].get("total_pairs"),
            "visible_pair_rate": metrics["counts"].get("visible_pair_rate"),
        }
    )

    for task_idx, task_name in enumerate(TASKS):
        rows.append(
            {
                "run_type": run_type,
                "algorithm": algorithm,
                "subset_id": subset_id,
                "subset_clients": subset_clients,
                "seed": seed,
                "graph_id": graph_id,
                "dataset_id": dataset_id,
                "phase": phase + "_task",
                "split": split,
                "task": task_name,
                "round": round_idx,
                "local_epoch": local_epoch,
                "train_loss": None,
                "eval_loss": None,
                "pair_acc": None,
                "subset_acc": None,
                "micro_f1": None,
                "macro_f1": None,
                "macro_pos_f1": None,
                "macro_minority_f1": None,
                "num_nodes": metrics["counts"]["num_nodes"],
                "tp": metrics["per_task"]["tp"][task_idx],
                "fp": metrics["per_task"]["fp"][task_idx],
                "tn": metrics["per_task"]["tn"][task_idx],
                "fn": metrics["per_task"]["fn"][task_idx],
                "precision": metrics["per_task"]["precision"][task_idx],
                "recall": metrics["per_task"]["recall"][task_idx],
                "f1": metrics["per_task"]["f1"][task_idx],
                "positive_f1": metrics["per_task"].get(
                    "positive_f1", metrics["per_task"]["f1"]
                )[task_idx],
                "minority_f1": metrics["per_task"].get(
                    "minority_f1", metrics["per_task"]["f1"]
                )[task_idx],
                "pos_cnt": metrics["counts"]["pos_cnt"][task_idx],
                "pos_rate": metrics["counts"]["pos_rate"][task_idx],
                "eval_mask_mode": eval_mask_mode,
                "selection_protocol": selection_protocol,
                "selected_by": selected_by,
                "visible_pairs": metrics["counts"].get("visible_pairs"),
                "total_pairs": metrics["counts"].get("total_pairs"),
                "visible_pair_rate": metrics["counts"].get("visible_pair_rate"),
            }
        )


def weighted_scalar_summary(eval_infos: List[Dict]) -> Dict[str, float]:
    keys = [
        "loss",
        "pair_acc",
        "subset_acc",
        "micro_f1",
        "macro_f1",
        "macro_pos_f1",
        "macro_minority_f1",
    ]
    total_weight = 0.0
    sums = {key: 0.0 for key in keys}

    for metrics in eval_infos:
        weight = float(metrics["counts"]["num_nodes"])
        total_weight += weight
        for key in keys:
            value = metrics["scalar"].get(key)
            if value is not None:
                sums[key] += weight * float(value)

    total_weight = max(total_weight, 1.0)
    return {key: sums[key] / total_weight for key in keys}


def append_mean_eval_row(
    rows: List[Dict],
    *,
    run_type: str,
    algorithm: str,
    subset_id: Optional[str],
    subset_clients: Optional[str],
    seed: int,
    phase: str,
    split: str,
    round_idx: Optional[int],
    eval_infos: List[Dict],
    eval_mask_mode: str = "full",
    selection_protocol: Optional[str] = None,
    selected_by: Optional[str] = None,
) -> None:
    mean_scalar = weighted_scalar_summary(eval_infos)

    num_nodes = sum(int(m["counts"]["num_nodes"]) for m in eval_infos)
    visible_pairs = sum(int(m["counts"].get("visible_pairs", 0)) for m in eval_infos)
    total_pairs = sum(int(m["counts"].get("total_pairs", 0)) for m in eval_infos)
    visible_pair_rate = visible_pairs / max(total_pairs, 1)

    rows.append(
        {
            "run_type": run_type,
            "algorithm": algorithm,
            "subset_id": subset_id,
            "subset_clients": subset_clients,
            "seed": seed,
            "graph_id": "all",
            "dataset_id": "all",
            "phase": phase,
            "split": split,
            "task": None,
            "round": round_idx,
            "local_epoch": None,
            "eval_mask_mode": eval_mask_mode,
            "selection_protocol": selection_protocol,
            "selected_by": selected_by,
            "train_loss": None,
            "eval_loss": mean_scalar["loss"],
            "pair_acc": mean_scalar["pair_acc"],
            "subset_acc": mean_scalar["subset_acc"],
            "micro_f1": mean_scalar["micro_f1"],
            "macro_f1": mean_scalar["macro_f1"],
            "macro_pos_f1": mean_scalar["macro_pos_f1"],
            "macro_minority_f1": mean_scalar["macro_minority_f1"],
            "num_nodes": num_nodes,
            "tp": None,
            "fp": None,
            "tn": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "positive_f1": None,
            "minority_f1": None,
            "pos_cnt": None,
            "pos_rate": None,
            "visible_pairs": visible_pairs,
            "total_pairs": total_pairs,
            "visible_pair_rate": visible_pair_rate,
        }
    )


# -----------------------------------------------------------------------------
# local baseline
# -----------------------------------------------------------------------------
def run_local_experiment(
    client: ClientData,
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    *,
    rounds: int,
    local_epochs: int,
    device: torch.device,
    selection_metric: str = "loss",
) -> None:
    set_seed(seed)

    ctx = build_model_context([client], cfg)
    train_graph = client.train_h

    train_loader, val_loader, test_loader = build_client_loaders(client, cfg, device)
    criterion = build_criterion_for_client(train_graph, cfg, device)
    model = make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)
    optimizer = make_optimizer(model, cfg)

    if os.environ.get("DEBUG_MODEL", "0") == "1":
        print_model_debug_summary(
            model=model,
            cfg=cfg,
            ctx=ctx,
            client=client,
            train_loader=train_loader,
            device=device,
        )

    rows: List[Dict] = []

    initial_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }

    best_by_eval_mode = {
        "full": {
            "value": float("inf"),
            "epoch": -1,
            "state": initial_state,
        },
        "visible": {
            "value": float("inf"),
            "epoch": -1,
            "state": initial_state,
        },
    }

    total_epochs = int(rounds * local_epochs)

    run_start_time = time.perf_counter()

    for local_epoch in range(1, total_epochs + 1):

        train_loss = train_epoch_neighbor(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_ego_ids=ctx["use_ego_ids"],
            ego_dim=ctx["ego_dim"],
        )
        append_train_row(
            rows,
            run_type="local",
            algorithm="local",
            subset_id=None,
            subset_clients=None,
            seed=seed,
            graph_id=str(client.graph_id),
            dataset_id=str(client.dataset_id),
            phase="train_epoch",
            split="train",
            train_loss=train_loss,
            num_nodes=int(train_graph["n"].num_nodes),
            local_epoch=local_epoch,
        )

        print("local_epoch:", local_epoch, "train_loss:", train_loss)

        for eval_mask_mode in ("full", "visible"):
            val_metrics = evaluate_loader(
                model,
                val_loader,
                criterion,
                device,
                use_ego_ids=ctx["use_ego_ids"],
                ego_dim=ctx["ego_dim"],
                threshold=0.5,
                eval_mask_mode=eval_mask_mode,
            )

            append_eval_rows(
                rows,
                run_type="local",
                algorithm="local",
                subset_id=None,
                subset_clients=None,
                seed=seed,
                phase=f"val_epoch_{eval_mask_mode}",
                split="val",
                graph_id=str(client.graph_id),
                dataset_id=str(client.dataset_id),
                metrics=val_metrics,
                local_epoch=local_epoch,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )

            current_value = val_metrics["scalar"][selection_metric]

            print(
                "local_epoch:",
                local_epoch,
                "eval_mask_mode:",
                eval_mask_mode,
                "val_loss:",
                val_metrics["scalar"]["loss"],
            )
            print("val_macro_minority_f1:", val_metrics["scalar"]["macro_minority_f1"])
            print("val_minority_f1:", val_metrics["per_task"]["minority_f1"])
            print("val_positive_f1:", val_metrics["per_task"]["positive_f1"])
            print(
                "current vs best:",
                "eval_mask_mode:",
                eval_mask_mode,
                "current:",
                current_value,
                "best:",
                best_by_eval_mode[eval_mask_mode]["value"],
            )

            if current_value < best_by_eval_mode[eval_mask_mode]["value"]:
                best_by_eval_mode[eval_mask_mode]["value"] = current_value
                best_by_eval_mode[eval_mask_mode]["epoch"] = local_epoch
                best_by_eval_mode[eval_mask_mode]["state"] = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    run_elapsed = time.perf_counter() - run_start_time
    print(f"run time: {format_seconds(run_elapsed)}")
    print("=======================================================")
    print("=======================================================")

    # Legacy-compatible default: use the full-validation selected model.
    best_state = best_by_eval_mode["full"]["state"]
    best_epoch = best_by_eval_mode["full"]["epoch"]
    best_loss = best_by_eval_mode["full"]["value"]

    model.load_state_dict(best_state, strict=True)

    checkpoint = {
        # Legacy-compatible fields.
        "state_dict": best_state,
        "cfg": cfg,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "selection_metric": selection_metric,
        # New dual-selection fields.
        "state_dict_by_eval_mode": {
            "full": best_by_eval_mode["full"]["state"],
            "visible": best_by_eval_mode["visible"]["state"],
        },
        "best_epoch_by_eval_mode": {
            "full": best_by_eval_mode["full"]["epoch"],
            "visible": best_by_eval_mode["visible"]["epoch"],
        },
        "best_value_by_eval_mode": {
            "full": best_by_eval_mode["full"]["value"],
            "visible": best_by_eval_mode["visible"]["value"],
        },
        "eval_protocols": [
            "oracle_full",
            "realistic_visible",
            "realistic_selection_oracle",
        ],
        "dataset_id": str(client.dataset_id),
        "graph_id": str(client.graph_id),
        "x_dim": ctx["x_dim"],
        "out_dim": ctx["out_dim"],
        "ego_dim": ctx["ego_dim"],
        "in_vocab": ctx["in_vocab"],
        "out_vocab": ctx["out_vocab"],
        "deg_fwd_hist": ctx["deg_fwd_hist"].cpu(),
        "deg_rev_hist": ctx["deg_rev_hist"].cpu(),
    }
    torch.save(checkpoint, run_paths.ckpt_path)
    print(f"saved best checkpoint -> {run_paths.ckpt_path}")

    final_eval_jobs = [
        {
            "selection_protocol": "oracle_full",
            "selected_by": "full",
            "eval_mask_mode": "full",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_visible",
            "selected_by": "visible",
            "eval_mask_mode": "visible",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_selection_oracle",
            "selected_by": "visible",
            "eval_mask_mode": "full",
            "splits": ["test"],
        },
    ]

    split_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    for job in final_eval_jobs:
        selection_protocol = job["selection_protocol"]
        selected_by = job["selected_by"]
        eval_mask_mode = job["eval_mask_mode"]

        selected_state = best_by_eval_mode[selected_by]["state"]
        selected_epoch = best_by_eval_mode[selected_by]["epoch"]

        model.load_state_dict(selected_state, strict=True)

        for split_name in job["splits"]:
            metrics = evaluate_loader(
                model,
                split_loaders[split_name],
                criterion,
                device,
                use_ego_ids=ctx["use_ego_ids"],
                ego_dim=ctx["ego_dim"],
                threshold=0.5,
                eval_mask_mode=eval_mask_mode,
            )

            append_eval_rows(
                rows,
                run_type="local",
                algorithm="local",
                subset_id=None,
                subset_clients=None,
                seed=seed,
                phase=f"best_{selection_protocol}_{split_name}",
                split=split_name,
                graph_id=str(client.graph_id),
                dataset_id=str(client.dataset_id),
                metrics=metrics,
                local_epoch=selected_epoch,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=selection_protocol,
                selected_by=selected_by,
            )

    out_df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("LOCAL DUAL-EVALUATION SMOKE CHECK")
    print("=" * 100)
    print("csv path:", run_paths.csv_path)
    print("checkpoint path:", run_paths.ckpt_path)
    print(
        "best epochs by eval mode:",
        {mode: best_by_eval_mode[mode]["epoch"] for mode in ["full", "visible"]},
    )
    print(
        "best values by eval mode:",
        {mode: best_by_eval_mode[mode]["value"] for mode in ["full", "visible"]},
    )

    final_rows = out_df[
        (out_df["task"].isna())
        & (out_df["split"] == "test")
        & (
            out_df["selection_protocol"].isin(
                [
                    "oracle_full",
                    "realistic_visible",
                    "realistic_selection_oracle",
                ]
            )
        )
    ]

    print("final test protocol rows:")
    print(
        final_rows[
            [
                "phase",
                "eval_mask_mode",
                "selection_protocol",
                "selected_by",
                "graph_id",
                "eval_loss",
                "micro_f1",
                "macro_pos_f1",
                "macro_minority_f1",
                "visible_pair_rate",
            ]
        ].to_string(index=False)
    )

    print("=" * 100)

    out_df.to_csv(run_paths.csv_path, index=False)
    print(f"saved csv -> {run_paths.csv_path}")


def run_local(
    client: ClientData,
    cfg: Dict,
    seed: int,
    local_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    device: torch.device,
    selection_metric: str = "loss",
) -> RunPaths:
    model_tag = _model_tag_from_cfg(cfg)
    run_paths = create_local_run_paths(
        local_root,
        str(client.dataset_id),
        rounds,
        local_epochs,
        model_tag,
        seed,
    )

    if run_paths.csv_path.exists():
        return run_paths

    run_local_experiment(
        client,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        device=device,
        selection_metric=selection_metric,
    )
    return run_paths


# -----------------------------------------------------------------------------
# FedAvg
# -----------------------------------------------------------------------------
def fedavg_state_dict(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[int],
) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("FedAvg weights sum to zero.")

    out: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        acc = None
        for state_dict, weight in zip(state_dicts, weights):
            term = state_dict[key].float() * (float(weight) / total)
            acc = term if acc is None else acc + term
        out[key] = acc.to(state_dicts[0][key].dtype)
    return out


def run_fedavg_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    device: torch.device,
    selection_metric: str = "loss",
) -> None:
    if not subset_clients:
        raise ValueError("subset_clients is empty")

    set_seed(seed)

    subset_id = _subset_clients_str(subset_clients)
    subset_clients_str = subset_id

    ctx = build_model_context(subset_clients, cfg)
    runtime_clients = build_runtime_clients(subset_clients, cfg, device)

    global_model = make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)

    rows: List[Dict] = []

    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    best_by_eval_mode = {
        "full": {
            "value": float("inf"),
            "round": -1,
            "state": initial_state,
        },
        "visible": {
            "value": float("inf"),
            "round": -1,
            "state": initial_state,
        },
    }

    num_clients = len(runtime_clients)
    generator = torch.Generator().manual_seed(seed)

    for round_idx in range(1, rounds + 1):

        run_start_time = time.perf_counter()

        num_selected = max(1, int(round(client_fraction * num_clients)))
        if num_selected >= num_clients:
            selected_indices = list(range(num_clients))
        else:
            permutation = torch.randperm(num_clients, generator=generator).tolist()
            selected_indices = permutation[:num_selected]

        local_states = []
        local_weights = []

        for idx in selected_indices:
            runtime = runtime_clients[idx]

            local_model = make_model(
                cfg,
                ctx["x_dim"],
                ctx["out_dim"],
                ctx["deg_fwd_hist"],
                ctx["deg_rev_hist"],
                ctx["ego_dim"],
                ctx["in_vocab"],
                ctx["out_vocab"],
            ).to(device)
            local_model.load_state_dict(global_model.state_dict(), strict=True)

            optimizer = make_optimizer(local_model, cfg)

            for local_epoch_idx in range(1, local_epochs + 1):
                train_loss = train_epoch_neighbor(
                    local_model,
                    runtime.train_loader,
                    optimizer,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                )
                append_train_row(
                    rows,
                    run_type="fedavg",
                    algorithm="fedavg",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    phase="train_epoch",
                    split="train",
                    train_loss=train_loss,
                    num_nodes=runtime.num_train_nodes,
                    round_idx=round_idx,
                    local_epoch=local_epoch_idx,
                )

            for eval_mask_mode in ("full", "visible"):

                post_metrics = evaluate_loader(
                    local_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                append_eval_rows(
                    rows,
                    run_type="fedavg",
                    algorithm="fedavg",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"val_epoch_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=post_metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )

            local_states.append(
                {
                    key: value.detach().cpu().clone()
                    for key, value in local_model.state_dict().items()
                }
            )
            local_weights.append(runtime.num_train_nodes)

        aggregated_state = fedavg_state_dict(local_states, local_weights)
        global_model.load_state_dict(aggregated_state, strict=True)

        for eval_mask_mode in ("full", "visible"):
            round_eval_infos = []

            for runtime in runtime_clients:
                metrics = evaluate_loader(
                    global_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                round_eval_infos.append(metrics)

                print("fed eval on client:", runtime.client.graph_id)
                print(
                    "fedavg round:",
                    round_idx,
                    "eval_mask_mode:",
                    eval_mask_mode,
                    "val_loss:",
                    metrics["scalar"]["loss"],
                )
                print("val_macro_minority_f1:", metrics["scalar"]["macro_minority_f1"])
                print("val_minority_f1:", metrics["per_task"]["minority_f1"])
                print("val_positive_f1:", metrics["per_task"]["positive_f1"])

                append_eval_rows(
                    rows,
                    run_type="fedavg",
                    algorithm="fedavg",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"global_val_client_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )

            append_mean_eval_row(
                rows,
                run_type="fedavg",
                algorithm="fedavg",
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                phase=f"global_val_mean_{eval_mask_mode}",
                split="val",
                round_idx=round_idx,
                eval_infos=round_eval_infos,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )

            mean_scalar = weighted_scalar_summary(round_eval_infos)
            current_value = mean_scalar[selection_metric]

            print(
                "current vs best:",
                "eval_mask_mode:",
                eval_mask_mode,
                "current:",
                current_value,
                "best:",
                best_by_eval_mode[eval_mask_mode]["value"],
            )

            if current_value < best_by_eval_mode[eval_mask_mode]["value"]:
                best_by_eval_mode[eval_mask_mode]["value"] = current_value
                best_by_eval_mode[eval_mask_mode]["round"] = round_idx
                best_by_eval_mode[eval_mask_mode]["state"] = {
                    key: value.detach().cpu().clone()
                    for key, value in global_model.state_dict().items()
                }

        run_elapsed = time.perf_counter() - run_start_time
        print(f"run time: {format_seconds(run_elapsed)}")
        print("=======================================================")
        print(f"==================== round {round_idx} / {rounds} ====================")
        print("=======================================================")

    # Keep legacy behavior: default checkpoint state_dict is the full-validation selection.
    best_state = best_by_eval_mode["full"]["state"]
    best_round = best_by_eval_mode["full"]["round"]
    best_loss = best_by_eval_mode["full"]["value"]

    global_model.load_state_dict(best_state, strict=True)

    checkpoint = {
        # Legacy-compatible fields.
        "state_dict": best_state,
        "cfg": cfg,
        "seed": seed,
        "best_round": best_round,
        "best_loss": best_loss,
        "selection_metric": selection_metric,
        # New dual-selection fields.
        "state_dict_by_eval_mode": {
            "full": best_by_eval_mode["full"]["state"],
            "visible": best_by_eval_mode["visible"]["state"],
        },
        "best_round_by_eval_mode": {
            "full": best_by_eval_mode["full"]["round"],
            "visible": best_by_eval_mode["visible"]["round"],
        },
        "best_value_by_eval_mode": {
            "full": best_by_eval_mode["full"]["value"],
            "visible": best_by_eval_mode["visible"]["value"],
        },
        "eval_protocols": [
            "oracle_full",
            "realistic_visible",
            "realistic_selection_oracle",
        ],
        "subset_id": subset_id,
        "subset_clients": subset_clients_str,
        "x_dim": ctx["x_dim"],
        "out_dim": ctx["out_dim"],
        "ego_dim": ctx["ego_dim"],
        "in_vocab": ctx["in_vocab"],
        "out_vocab": ctx["out_vocab"],
        "deg_fwd_hist": ctx["deg_fwd_hist"].cpu(),
        "deg_rev_hist": ctx["deg_rev_hist"].cpu(),
    }

    torch.save(checkpoint, run_paths.ckpt_path)
    print(f"saved best checkpoint -> {run_paths.ckpt_path}")

    final_eval_jobs = [
        {
            "selection_protocol": "oracle_full",
            "selected_by": "full",
            "eval_mask_mode": "full",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_visible",
            "selected_by": "visible",
            "eval_mask_mode": "visible",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_selection_oracle",
            "selected_by": "visible",
            "eval_mask_mode": "full",
            "splits": ["test"],
        },
    ]

    for job in final_eval_jobs:
        selection_protocol = job["selection_protocol"]
        selected_by = job["selected_by"]
        eval_mask_mode = job["eval_mask_mode"]

        selected_state = best_by_eval_mode[selected_by]["state"]
        selected_round = best_by_eval_mode[selected_by]["round"]

        global_model.load_state_dict(selected_state, strict=True)

        for split_name in job["splits"]:
            for runtime in runtime_clients:
                split_loader = {
                    "train": runtime.train_loader,
                    "val": runtime.val_loader,
                    "test": runtime.test_loader,
                }[split_name]

                metrics = evaluate_loader(
                    global_model,
                    split_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )

                append_eval_rows(
                    rows,
                    run_type="fedavg",
                    algorithm="fedavg",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"best_{selection_protocol}_{split_name}",
                    split=split_name,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=selected_round,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=selection_protocol,
                    selected_by=selected_by,
                )

    out_df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("FEDAVG DUAL-EVALUATION SMOKE CHECK")
    print("=" * 100)
    print("csv path:", run_paths.csv_path)
    print("checkpoint path:", run_paths.ckpt_path)
    print(
        "best rounds by eval mode:",
        {mode: best_by_eval_mode[mode]["round"] for mode in ["full", "visible"]},
    )
    print(
        "best values by eval mode:",
        {mode: best_by_eval_mode[mode]["value"] for mode in ["full", "visible"]},
    )

    final_rows = out_df[
        (out_df["task"].isna())
        & (out_df["split"] == "test")
        & (
            out_df["selection_protocol"].isin(
                [
                    "oracle_full",
                    "realistic_visible",
                    "realistic_selection_oracle",
                ]
            )
        )
    ]

    print("final test protocol rows:")
    print(
        final_rows[
            [
                "phase",
                "eval_mask_mode",
                "selection_protocol",
                "selected_by",
                "graph_id",
                "eval_loss",
                "micro_f1",
                "macro_pos_f1",
                "macro_minority_f1",
                "visible_pair_rate",
            ]
        ].to_string(index=False)
    )

    print("=" * 100)

    out_df.to_csv(run_paths.csv_path, index=False)
    print(f"saved csv -> {run_paths.csv_path}")


def run_fedavg(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    fedavg_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    device: torch.device,
    selection_metric: str = "loss",
) -> RunPaths:
    subset_id = _subset_clients_str(subset_clients)
    model_tag = _model_tag_from_cfg(cfg)

    run_paths = create_fedavg_run_paths(
        fedavg_root,
        subset_id,
        rounds,
        local_epochs,
        model_tag,
        seed,
    )

    if run_paths.csv_path.exists():
        return run_paths

    run_fedavg_experiment(
        subset_clients,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        device=device,
        selection_metric=selection_metric,
    )
    return run_paths


# -----------------------------------------------------------------------------
# fedprox
# -----------------------------------------------------------------------------


def run_fedprox_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    fedprox_mu: float,
    device: torch.device,
    selection_metric: str = "loss",
) -> None:
    if not subset_clients:
        raise ValueError("subset_clients is empty")

    set_seed(seed)

    subset_id = _subset_clients_str(subset_clients)
    subset_clients_str = subset_id

    ctx = build_model_context(subset_clients, cfg)
    runtime_clients = build_runtime_clients(subset_clients, cfg, device)

    global_model = make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)

    rows: List[Dict] = []

    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    best_by_eval_mode = {
        "full": {
            "value": float("inf"),
            "round": -1,
            "state": initial_state,
        },
        "visible": {
            "value": float("inf"),
            "round": -1,
            "state": initial_state,
        },
    }

    num_clients = len(runtime_clients)
    generator = torch.Generator().manual_seed(seed)

    aggregated = False

    for round_idx in range(1, rounds + 1):

        run_start_time = time.perf_counter()

        num_selected = max(1, int(round(client_fraction * num_clients)))
        if num_selected >= num_clients:
            selected_indices = list(range(num_clients))
        else:
            permutation = torch.randperm(num_clients, generator=generator).tolist()
            selected_indices = permutation[:num_selected]

        local_states = []
        local_weights = []

        for idx in selected_indices:
            runtime = runtime_clients[idx]

            local_model = make_model(
                cfg,
                ctx["x_dim"],
                ctx["out_dim"],
                ctx["deg_fwd_hist"],
                ctx["deg_rev_hist"],
                ctx["ego_dim"],
                ctx["in_vocab"],
                ctx["out_vocab"],
            ).to(device)
            local_model.load_state_dict(global_model.state_dict(), strict=True)

            # snapshot of the downloaded global model for the proximal term
            global_params = {
                name: param.detach().clone()
                for name, param in local_model.named_parameters()
            }

            optimizer = make_optimizer(local_model, cfg)

            for local_epoch_idx in range(1, local_epochs + 1):
                train_loss = train_epoch_neighbor_fedprox(
                    local_model,
                    global_params,
                    runtime.train_loader,
                    optimizer,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    fedprox_mu=fedprox_mu,
                    aggregated=aggregated,
                )
                append_train_row(
                    rows,
                    run_type="fedprox",
                    algorithm="fedprox",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    phase="train_epoch",
                    split="train",
                    train_loss=train_loss,
                    num_nodes=runtime.num_train_nodes,
                    round_idx=round_idx,
                    local_epoch=local_epoch_idx,
                )

            for eval_mask_mode in ("full", "visible"):
                post_metrics = evaluate_loader(
                    local_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )

                append_eval_rows(
                    rows,
                    run_type="fedprox",
                    algorithm="fedprox",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"val_epoch_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=post_metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )

            local_states.append(
                {
                    key: value.detach().cpu().clone()
                    for key, value in local_model.state_dict().items()
                }
            )
            local_weights.append(runtime.num_train_nodes)

        # FedProx still uses normal weighted averaging on the server
        aggregated_state = fedavg_state_dict(local_states, local_weights)
        global_model.load_state_dict(aggregated_state, strict=True)
        aggregated = True

        for eval_mask_mode in ("full", "visible"):
            round_eval_infos = []

            for runtime in runtime_clients:
                metrics = evaluate_loader(
                    global_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                round_eval_infos.append(metrics)

                print("fed eval on client:", runtime.client.graph_id)
                print(
                    "fedprox round:",
                    round_idx,
                    "eval_mask_mode:",
                    eval_mask_mode,
                    "val_loss:",
                    metrics["scalar"]["loss"],
                )
                print("val_macro_minority_f1:", metrics["scalar"]["macro_minority_f1"])
                print("val_minority_f1:", metrics["per_task"]["minority_f1"])
                print("val_positive_f1:", metrics["per_task"]["positive_f1"])

                append_eval_rows(
                    rows,
                    run_type="fedprox",
                    algorithm="fedprox",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"global_val_client_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )

            append_mean_eval_row(
                rows,
                run_type="fedprox",
                algorithm="fedprox",
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                phase=f"global_val_mean_{eval_mask_mode}",
                split="val",
                round_idx=round_idx,
                eval_infos=round_eval_infos,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )

            mean_scalar = weighted_scalar_summary(round_eval_infos)
            current_value = mean_scalar[selection_metric]

            print(
                "current vs best:",
                "eval_mask_mode:",
                eval_mask_mode,
                "current:",
                current_value,
                "best:",
                best_by_eval_mode[eval_mask_mode]["value"],
            )

            if current_value < best_by_eval_mode[eval_mask_mode]["value"]:
                best_by_eval_mode[eval_mask_mode]["value"] = current_value
                best_by_eval_mode[eval_mask_mode]["round"] = round_idx
                best_by_eval_mode[eval_mask_mode]["state"] = {
                    key: value.detach().cpu().clone()
                    for key, value in global_model.state_dict().items()
                }

        run_elapsed = time.perf_counter() - run_start_time
        print(f"run time: {format_seconds(run_elapsed)}")
        print("=======================================================")
        print(f"==================== round {round_idx} / {rounds} ====================")
        print("=======================================================")

    # Keep legacy behavior: default checkpoint state_dict is the full-validation selection.
    best_state = best_by_eval_mode["full"]["state"]
    best_round = best_by_eval_mode["full"]["round"]
    best_loss = best_by_eval_mode["full"]["value"]

    global_model.load_state_dict(best_state, strict=True)

    checkpoint = {
        # Legacy-compatible fields.
        "state_dict": best_state,
        "cfg": cfg,
        "seed": seed,
        "best_round": best_round,
        "best_loss": best_loss,
        "selection_metric": selection_metric,
        # New dual-selection fields.
        "state_dict_by_eval_mode": {
            "full": best_by_eval_mode["full"]["state"],
            "visible": best_by_eval_mode["visible"]["state"],
        },
        "best_round_by_eval_mode": {
            "full": best_by_eval_mode["full"]["round"],
            "visible": best_by_eval_mode["visible"]["round"],
        },
        "best_value_by_eval_mode": {
            "full": best_by_eval_mode["full"]["value"],
            "visible": best_by_eval_mode["visible"]["value"],
        },
        "eval_protocols": [
            "oracle_full",
            "realistic_visible",
            "realistic_selection_oracle",
        ],
        "subset_id": subset_id,
        "subset_clients": subset_clients_str,
        "fedprox_mu": fedprox_mu,
        "x_dim": ctx["x_dim"],
        "out_dim": ctx["out_dim"],
        "ego_dim": ctx["ego_dim"],
        "in_vocab": ctx["in_vocab"],
        "out_vocab": ctx["out_vocab"],
        "deg_fwd_hist": ctx["deg_fwd_hist"].cpu(),
        "deg_rev_hist": ctx["deg_rev_hist"].cpu(),
    }

    torch.save(checkpoint, run_paths.ckpt_path)
    print(f"saved best checkpoint -> {run_paths.ckpt_path}")

    final_eval_jobs = [
        {
            "selection_protocol": "oracle_full",
            "selected_by": "full",
            "eval_mask_mode": "full",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_visible",
            "selected_by": "visible",
            "eval_mask_mode": "visible",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_selection_oracle",
            "selected_by": "visible",
            "eval_mask_mode": "full",
            "splits": ["test"],
        },
    ]

    for job in final_eval_jobs:
        selection_protocol = job["selection_protocol"]
        selected_by = job["selected_by"]
        eval_mask_mode = job["eval_mask_mode"]

        selected_state = best_by_eval_mode[selected_by]["state"]
        selected_round = best_by_eval_mode[selected_by]["round"]

        global_model.load_state_dict(selected_state, strict=True)

        for split_name in job["splits"]:
            for runtime in runtime_clients:
                split_loader = {
                    "train": runtime.train_loader,
                    "val": runtime.val_loader,
                    "test": runtime.test_loader,
                }[split_name]

                metrics = evaluate_loader(
                    global_model,
                    split_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )

                append_eval_rows(
                    rows,
                    run_type="fedprox",
                    algorithm="fedprox",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"best_{selection_protocol}_{split_name}",
                    split=split_name,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=selected_round,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=selection_protocol,
                    selected_by=selected_by,
                )

    pd.DataFrame(rows).to_csv(run_paths.csv_path, index=False)
    print(f"saved csv -> {run_paths.csv_path}")


def run_fedprox(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    fedprox_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    fedprox_mu: float,
    device: torch.device,
    selection_metric: str = "loss",
) -> RunPaths:
    subset_id = _subset_clients_str(subset_clients)
    model_tag = _model_tag_from_cfg(cfg)

    run_paths = create_fedprox_run_paths(
        fedprox_root,
        subset_id,
        rounds,
        local_epochs,
        fedprox_mu,
        model_tag,
        seed,
    )

    if run_paths.csv_path.exists():
        return run_paths

    run_fedprox_experiment(
        subset_clients,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        fedprox_mu=fedprox_mu,
        device=device,
        selection_metric=selection_metric,
    )
    return run_paths


# -----------------------------------------------------------------------------
# experiment manifest row builders
# -----------------------------------------------------------------------------
def build_local_log_row(
    *,
    client: ClientData,
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    rounds: int,
    local_epochs: int,
    selection_metric: str,
) -> Dict:
    model_tag = _model_tag_from_cfg(cfg)
    return {
        "run_type": "local",
        "algorithm": "local",
        "subset_id": None,
        "subset_clients": None,
        "graph_id": str(client.graph_id),
        "dataset_id": str(client.dataset_id),
        "num_clients": 1,
        "client_fraction": 1.0,
        "out_csv": str(run_paths.csv_path),
        "ckpt_path": str(run_paths.ckpt_path),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "selection_metric": selection_metric,
        "mcw": cfg.get("minority_class_weight", None),
        "num_layers": cfg["num_layers"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "dropout": cfg["dropout"],
        "hidden_dim": cfg["hidden_dim"],
        "use_ego_ids": cfg.get("use_ego_ids", True),
        "batch_size": cfg.get("batch_size", 64),
        "output_head": cfg.get("output_head", "multi"),
        "architecture": cfg.get("architecture", "multihead"),
        "seed": seed,
        "model_tag": model_tag,
    }


def build_fedavg_log_row(
    *,
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    selection_metric: str,
) -> Dict:
    subset_id = _subset_clients_str(subset_clients)
    dataset_ids = "|".join(str(client.dataset_id) for client in subset_clients)
    model_tag = _model_tag_from_cfg(cfg)

    return {
        "run_type": "fedavg",
        "algorithm": "fedavg",
        "subset_id": subset_id,
        "subset_clients": subset_id,
        "graph_id": "all",
        "dataset_id": dataset_ids,
        "num_clients": len(subset_clients),
        "client_fraction": client_fraction,
        "out_csv": str(run_paths.csv_path),
        "ckpt_path": str(run_paths.ckpt_path),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "selection_metric": selection_metric,
        "mcw": cfg.get("minority_class_weight", None),
        "num_layers": cfg["num_layers"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "dropout": cfg["dropout"],
        "hidden_dim": cfg["hidden_dim"],
        "use_ego_ids": cfg.get("use_ego_ids", True),
        "batch_size": cfg.get("batch_size", 64),
        "output_head": cfg.get("output_head", "multi"),
        "architecture": cfg.get("architecture", "multihead"),
        "seed": seed,
        "model_tag": model_tag,
    }


def build_fedprox_log_row(
    *,
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    fedprox_mu: float,
    selection_metric: str,
) -> Dict:
    subset_id = _subset_clients_str(subset_clients)
    dataset_ids = "|".join(str(client.dataset_id) for client in subset_clients)
    model_tag = _model_tag_from_cfg(cfg)

    return {
        "run_type": "fedprox",
        "algorithm": "fedprox",
        "subset_id": subset_id,
        "subset_clients": subset_id,
        "graph_id": "all",
        "dataset_id": dataset_ids,
        "num_clients": len(subset_clients),
        "client_fraction": client_fraction,
        "fedprox_mu": fedprox_mu,
        "out_csv": str(run_paths.csv_path),
        "ckpt_path": str(run_paths.ckpt_path),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "selection_metric": selection_metric,
        "mcw": cfg.get("minority_class_weight", None),
        "num_layers": cfg["num_layers"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "dropout": cfg["dropout"],
        "hidden_dim": cfg["hidden_dim"],
        "use_ego_ids": cfg.get("use_ego_ids", True),
        "batch_size": cfg.get("batch_size", 64),
        "output_head": cfg.get("output_head", "multi"),
        "architecture": cfg.get("architecture", "multihead"),
        "seed": seed,
        "model_tag": model_tag,
    }


# -----------------------------------------------------------------------------
# sweep helper kept for compatibility with run_fedavg.py
# -----------------------------------------------------------------------------
def iter_sweep_cfgs(
    base_cfg: Dict,
    *,
    seeds: Iterable[int],
    mcw: Iterable,
    num_layers: Iterable[int],
    lrs: Iterable[float],
    weight_decays: Iterable[float],
    dropouts: Iterable[float],
    hidden_dims: Iterable[int],
    use_ego_ids: Iterable[bool],
    batch_sizes: Iterable[int],
) -> Iterator[Tuple[Dict, Dict]]:
    for (
        seed,
        minority_class_weight,
        n_layers,
        lr,
        weight_decay,
        dropout,
        hidden_dim,
        ego_flag,
        batch_size,
    ) in product(
        seeds,
        mcw,
        num_layers,
        lrs,
        weight_decays,
        dropouts,
        hidden_dims,
        use_ego_ids,
        batch_sizes,
    ):
        cfg = dict(base_cfg)
        cfg["minority_class_weight"] = minority_class_weight
        cfg["num_layers"] = n_layers
        cfg["lr"] = lr
        cfg["weight_decay"] = weight_decay
        cfg["dropout"] = dropout
        cfg["hidden_dim"] = hidden_dim
        cfg["use_ego_ids"] = ego_flag
        cfg["batch_size"] = batch_size

        cfg["output_head"] = "multi"
        cfg["architecture"] = "multihead"

        meta = {
            "seed": seed,
            "minority_class_weight": minority_class_weight,
            "num_layers": n_layers,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "use_ego_ids": ego_flag,
            "batch_size": batch_size,
            "output_head": cfg["output_head"],
            "architecture": cfg["architecture"],
        }
        yield cfg, meta
