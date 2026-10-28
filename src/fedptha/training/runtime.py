from __future__ import annotations
from .selection import is_better_selection_value
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import torch
from fedptha.data.clients import ClientData, format_seconds
from fedptha.data.tasks import TASKS
from fedptha.training.steps import (
    build_criterion_for_client,
    build_hetero_neighbor_loader,
    evaluate_loader,
    train_epoch_neighbor,
)
from fedptha.models.pna import PNANetReverseMP
from fedptha.data.preprocessing import max_port_cols
from fedptha.training.random import set_seed


@dataclass(frozen=True)
class RunPaths:
    csv_path: Path

    @property
    def dr_csv_path(self):
        return self.csv_path.with_name("routing.csv")

    @property
    def ala_csv_path(self):
        return self.csv_path.with_name("refinement.csv")

    @property
    def clusters_csv_path(self):
        return self.csv_path.with_name("clusters.csv")

    @property
    def representatives_csv_path(self):
        return self.csv_path.with_name("representatives.csv")


def create_run_paths(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(root / "metrics.csv")


def init_best_states(model, key_name: str):
    initial_state = {
        key: value.detach().cpu().clone() for (key, value) in model.state_dict().items()
    }
    return {
        eval_mode: {"value": float("-inf"), key_name: -1, "state": initial_state}
        for eval_mode in ("full", "visible")
    }


def update_best_state(
    *,
    best_by_eval_mode,
    eval_mask_mode: str,
    metrics: Dict,
    model,
    step_value: int,
    key_name: str,
) -> None:
    scalar = metrics.get("scalar", {})
    current_value = scalar.get("macro_pr_auc")
    best_record = best_by_eval_mode[eval_mask_mode]
    print(
        "current vs best:",
        "eval_mask_mode:",
        eval_mask_mode,
        "selector_metric:",
        "macro_pr_auc",
        "current:",
        current_value,
        "best:",
        best_record["value"],
    )
    if is_better_selection_value(current_value, best_record["value"]):
        best_record["value"] = float(current_value)
        best_record[key_name] = int(step_value)
        best_record["state"] = {
            key: value.detach().cpu().clone()
            for (key, value) in model.state_dict().items()
        }


def tag_recent_eval_rows(rows: List[Dict], **extra) -> None:
    n = 1 + len(TASKS)
    for row in rows[-n:]:
        row.update(extra)


@dataclass
class RuntimeClient:
    client: ClientData
    train_loader: object
    val_loader: object
    test_loader: object
    criterion: object
    num_train_nodes: int


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _subset_clients_str(subset_clients: Sequence[ClientData]) -> str:
    return "|".join((str(client.graph_id) for client in subset_clients))


def compute_global_port_vocab(*graph_lists: List) -> tuple[int, int]:
    max_in, max_out = (0, 0)
    for graphs in graph_lists:
        for g in graphs:
            mi, mo = max_port_cols(g)
            max_in = max(max_in, int(mi))
            max_out = max(max_out, int(mo))
    return (max_in + 1, max_out + 1)


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
    return (deg_fwd_hist, deg_rev_hist)


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
        port_emb_dim=cfg.get("port_emb_dim", 0) if cfg.get("use_port_ids", True) else 0,
        output_head=cfg.get("output_head", "multi"),
    )


def make_optimizer(model, cfg: Dict):
    return torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )


def build_client_loaders(
    client: ClientData, cfg: Dict, device: torch.device
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
    return (train_loader, val_loader, test_loader)


def build_runtime_clients(
    subset_clients: List[ClientData], cfg: Dict, device: torch.device
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
            "micro_pr_auc": None,
            "macro_pr_auc": None,
            "pr_auc": None,
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
            "micro_pr_auc": metrics["scalar"].get("micro_pr_auc"),
            "macro_pr_auc": metrics["scalar"].get("macro_pr_auc"),
            "pr_auc": None,
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
            "pr_auc_definition": "trapezoidal_precision_recall_curve_auc",
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
                "micro_pr_auc": None,
                "macro_pr_auc": None,
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
                "pr_auc": metrics["per_task"].get("pr_auc", [None] * len(TASKS))[
                    task_idx
                ],
                "pos_cnt": metrics["counts"]["pos_cnt"][task_idx],
                "pos_rate": metrics["counts"]["pos_rate"][task_idx],
                "eval_mask_mode": eval_mask_mode,
                "selection_protocol": selection_protocol,
                "selected_by": selected_by,
                "pr_auc_definition": "trapezoidal_precision_recall_curve_auc",
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
        "micro_pr_auc",
        "macro_pr_auc",
    ]
    sums = {key: 0.0 for key in keys}
    weights = {key: 0.0 for key in keys}
    for metrics in eval_infos:
        weight = float(metrics["counts"]["num_nodes"])
        for key in keys:
            value = metrics["scalar"].get(key)
            if value is None:
                continue
            value = float(value)
            if math.isnan(value):
                continue
            sums[key] += weight * value
            weights[key] += weight
    return {
        key: sums[key] / weights[key] if weights[key] > 0.0 else float("nan")
        for key in keys
    }


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
    num_nodes = sum((int(m["counts"]["num_nodes"]) for m in eval_infos))
    visible_pairs = sum((int(m["counts"].get("visible_pairs", 0)) for m in eval_infos))
    total_pairs = sum((int(m["counts"].get("total_pairs", 0)) for m in eval_infos))
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
            "micro_pr_auc": mean_scalar["micro_pr_auc"],
            "macro_pr_auc": mean_scalar["macro_pr_auc"],
            "pr_auc": None,
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


def fedavg_state_dict(
    state_dicts: List[Dict[str, torch.Tensor]], weights: List[int]
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
        cfg["auto_pos_weight_cap"] = 100.0
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
            "auto_pos_weight_cap": 100.0,
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
        yield (cfg, meta)


def clone_model_from_state(
    state_dict: Dict[str, torch.Tensor], cfg: Dict, ctx: Dict, device: torch.device
):
    model = make_task_head_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model


def make_task_head_model(
    cfg, x_dim, out_dim, deg_fwd_hist, deg_rev_hist, ego_dim, in_vocab, out_vocab
):
    """
    Build the task-head PNA model.

    This is intentionally separate from fl_run_helper.make_model(), so the
    stable old baselines and old APPLE implementation remain untouched.
    """
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
        port_emb_dim=cfg.get("port_emb_dim", 0) if cfg.get("use_port_ids", True) else 0,
        output_head=cfg.get("output_head", "multi"),
    )
