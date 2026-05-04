from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import json

from andrea.helper_funcs.fl_run_helper import (
    _model_tag_from_cfg,
    _subset_clients_str,
    append_eval_rows,
    append_train_row,
    build_model_context,
    build_runtime_clients,
    ensure_dir,
    fedavg_state_dict,
    make_model,
    make_optimizer,
    weighted_scalar_summary,
)
from andrea.helper_funcs.load_client_helper import ClientData, format_seconds
from andrea.helper_funcs.train_eval_helper import evaluate_loader, train_epoch_neighbor
from utils.seed import set_seed


@dataclass(frozen=True)
class GcflRunPaths:
    csv_path: Path
    ckpt_path: Path
    clusters_csv_path: Path


@dataclass
class GcflClientState:
    client_idx: int
    cluster_id: int
    train_size: int
    W: Optional[Dict[str, torch.Tensor]] = None
    W_old: Optional[Dict[str, torch.Tensor]] = None
    dW: Optional[Dict[str, torch.Tensor]] = None
    grad_seq: List[float] = field(default_factory=list)


@dataclass
class GcflClusterState:
    cluster_id: int
    member_indices: List[int]
    state_dict: Dict[str, torch.Tensor]
    born_round: int
    mean_norm_history: List[float] = field(default_factory=list)
    max_norm_history: List[float] = field(default_factory=list)


def create_gcfl_run_paths(
    gcfl_root: str | Path,
    subset_id: str,
    rounds: int,
    local_epochs: int,
    warmup_rounds: int,
    grad_seq_len: int,
    min_cluster_size: int,
    min_child_size: int,
    eps1_quantile: float,
    eps2_quantile: float,
    model_tag: str,
    seed: int,
) -> GcflRunPaths:
    root = ensure_dir(gcfl_root)

    stem = (
        f"gcflplus"
        f"{subset_id}"
        f"_rounds{rounds}"
        f"_epoch{local_epochs}"
        f"_warm{warmup_rounds}"
        f"_hist{grad_seq_len}"
        f"_mcs{min_cluster_size}"
        f"_mchild{min_child_size}"
        f"_eps1q{eps1_quantile}"
        f"_eps2q{eps2_quantile}"
        f"_{model_tag}"
        f"_seed{seed}"
    )
    return GcflRunPaths(
        csv_path=root / f"{stem}.csv",
        ckpt_path=root / f"{stem}.pt",
        clusters_csv_path=root / f"{stem}_clusters.csv",
    )


def build_gcfl_log_row(
    *,
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: GcflRunPaths,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    warmup_rounds: int,
    min_cluster_size: int,
    min_child_size: int,
    grad_seq_len: int,
    eps1_quantile: float,
    eps2_quantile: float,
    selection_metric: str,
) -> Dict:
    subset_id = _subset_clients_str(subset_clients)
    dataset_ids = "|".join(str(client.dataset_id) for client in subset_clients)
    model_tag = _model_tag_from_cfg(cfg)

    return {
        "run_type": "gcfl_plus",
        "algorithm": "gcfl_plus",
        "subset_id": subset_id,
        "subset_clients": subset_id,
        "graph_id": "all",
        "dataset_id": dataset_ids,
        "num_clients": len(subset_clients),
        "client_fraction": client_fraction,
        "out_csv": str(run_paths.csv_path),
        "ckpt_path": str(run_paths.ckpt_path),
        "clusters_csv_path": str(run_paths.clusters_csv_path),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "warmup_rounds": warmup_rounds,
        "min_cluster_size": min_cluster_size,
        "min_child_size": min_child_size,
        "grad_seq_len": grad_seq_len,
        "eps1_quantile": eps1_quantile,
        "eps2_quantile": eps2_quantile,
        "selection_metric": selection_metric,
        "mcw": cfg.get("minority_class_weight", None),
        "num_layers": cfg["num_layers"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "dropout": cfg["dropout"],
        "hidden_dim": cfg["hidden_dim"],
        "use_ego_ids": cfg.get("use_ego_ids", True),
        "batch_size": cfg.get("batch_size", 64),
        "seed": seed,
        "model_tag": model_tag,
    }


# -----------------------------------------------------------------------------
# state-dict helpers
# -----------------------------------------------------------------------------
def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def subtract_state_dicts(
    new_state: Dict[str, torch.Tensor],
    old_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in new_state.keys():
        out[key] = new_state[key].float() - old_state[key].float()
    return out


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    flats = []
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if not torch.is_floating_point(value):
            value = value.float()
        flats.append(value.reshape(-1).detach().cpu())
    if not flats:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(flats, dim=0).float()


def l2_norm_of_vector(vec: torch.Tensor) -> float:
    if vec.numel() == 0:
        return 0.0
    return float(torch.norm(vec, p=2).item())


def mean_update_norm(delta_vecs: List[torch.Tensor]) -> float:
    """
    GCFL-style mean update norm:
    norm(mean(delta_i))
    """
    if not delta_vecs:
        return 0.0
    stacked = torch.stack(delta_vecs, dim=0)
    mean_vec = stacked.mean(dim=0)
    return l2_norm_of_vector(mean_vec)


def max_update_norm(delta_vecs: List[torch.Tensor]) -> float:
    if not delta_vecs:
        return 0.0
    return max(l2_norm_of_vector(vec) for vec in delta_vecs)


# -----------------------------------------------------------------------------
# DTW / partitioning
# -----------------------------------------------------------------------------
def dtw_distance(seq_a: Sequence[float], seq_b: Sequence[float]) -> float:
    n = len(seq_a)
    m = len(seq_b)

    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float("inf")

    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        ai = float(seq_a[i - 1])
        for j in range(1, m + 1):
            bj = float(seq_b[j - 1])
            cost = abs(ai - bj)
            dp[i, j] = cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    return float(dp[n, m])


def tail_grad_history(history: List[float], max_len: int) -> List[float]:
    if len(history) <= max_len:
        return list(history)
    return list(history[-max_len:])


def dtw_affinity_matrix_from_histories(
    histories: List[List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a nonnegative affinity matrix for Stoer-Wagner from DTW distances.
    similarity = 1 / (1 + distance)
    """
    n = len(histories)
    dists = np.zeros((n, n), dtype=np.float64)
    sims = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        sims[i, i] = 1.0
        for j in range(i + 1, n):
            dist = dtw_distance(histories[i], histories[j])
            sim = 1.0 / (1.0 + dist)

            dists[i, j] = dist
            dists[j, i] = dist
            sims[i, j] = sim
            sims[j, i] = sim

    return sims, dists


def stoer_wagner_partition(
    member_indices: List[int],
    affinity: np.ndarray,
) -> Tuple[float, List[int], List[int]]:
    graph = nx.Graph()
    for idx in member_indices:
        graph.add_node(idx)

    n = len(member_indices)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(affinity[i, j])
            if w < 0.0:
                raise ValueError("Stoer-Wagner requires nonnegative edge weights.")
            graph.add_edge(member_indices[i], member_indices[j], weight=w)

    cut_value, partition = nx.stoer_wagner(graph, weight="weight")
    left = sorted(list(partition[0]))
    right = sorted(list(partition[1]))
    return float(cut_value), left, right


# -----------------------------------------------------------------------------
# split criteria
# -----------------------------------------------------------------------------
def should_split_cluster_gcfl(
    cluster: GcflClusterState,
    *,
    round_idx: int,
    warmup_rounds: int,
    min_cluster_size: int,
    min_child_size: int,
    full_participation: bool,
    current_mean_norm: float,
    current_max_norm: float,
    eps1_quantile: float,
    eps2_quantile: float,
) -> Dict[str, object]:
    cluster_size = len(cluster.member_indices)
    cluster_age = round_idx - cluster.born_round + 1

    prev_mean = cluster.mean_norm_history[:-1]
    prev_max = cluster.max_norm_history[:-1]

    gate_cluster_size_ok = cluster_size >= min_cluster_size
    gate_can_form_two_children = cluster_size >= 2 * min_child_size
    gate_full_participation_ok = bool(full_participation)
    gate_warmup_ok = cluster_age > warmup_rounds
    gate_history_ok = (len(prev_mean) >= warmup_rounds) and (
        len(prev_max) >= warmup_rounds
    )

    eps1 = None
    eps2 = None
    gate_mean_below_eps1 = None
    gate_max_above_eps2 = None

    if not gate_cluster_size_ok:
        split_status = "cluster_too_small"
        do_split = False

    elif not gate_can_form_two_children:
        split_status = "cluster_cannot_split_under_min_child"
        do_split = False

    elif not gate_full_participation_ok:
        split_status = "not_full_participation"
        do_split = False

    elif not gate_warmup_ok:
        split_status = "warmup"
        do_split = False

    elif not gate_history_ok:
        split_status = "insufficient_history"
        do_split = False

    else:
        eps1 = float(np.quantile(np.asarray(prev_mean), eps1_quantile))
        eps2 = float(np.quantile(np.asarray(prev_max), eps2_quantile))

        gate_mean_below_eps1 = bool(current_mean_norm < eps1)
        gate_max_above_eps2 = bool(current_max_norm > eps2)

        split_status = "eligible"
        do_split = gate_mean_below_eps1 and gate_max_above_eps2

    return {
        "do_split": bool(do_split),
        "eps1": eps1,
        "eps2": eps2,
        "split_status": split_status,
        "cluster_size": int(cluster_size),
        "cluster_age": int(cluster_age),
        "history_len_prev_mean": int(len(prev_mean)),
        "history_len_prev_max": int(len(prev_max)),
        "gate_cluster_size_ok": int(gate_cluster_size_ok),
        "gate_can_form_two_children": int(gate_can_form_two_children),
        "gate_full_participation_ok": int(gate_full_participation_ok),
        "gate_warmup_ok": int(gate_warmup_ok),
        "gate_history_ok": int(gate_history_ok),
        "gate_mean_below_eps1": gate_mean_below_eps1,
        "gate_max_above_eps2": gate_max_above_eps2,
    }


def _safe_stats(values: List[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(v)], dtype=np.float64
    )
    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_min": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
        }
    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_std": float(np.std(arr)),
    }


def _matrix_stats(mat: Optional[np.ndarray], prefix: str) -> Dict[str, float]:
    if mat is None or mat.shape[0] <= 1:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_min": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
        }

    tri = np.triu_indices(mat.shape[0], k=1)
    vals = mat[tri].tolist()
    return _safe_stats(vals, prefix)


def _submatrix_pair_stats(
    mat: Optional[np.ndarray], positions: List[int], prefix: str
) -> Dict[str, float]:
    if mat is None or len(positions) <= 1:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_min": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
        }

    sub = mat[np.ix_(positions, positions)]
    tri = np.triu_indices(sub.shape[0], k=1)
    vals = sub[tri].tolist()
    return _safe_stats(vals, prefix)


def _crossmatrix_pair_stats(
    mat: Optional[np.ndarray],
    left_positions: List[int],
    right_positions: List[int],
    prefix: str,
) -> Dict[str, float]:
    if mat is None or len(left_positions) == 0 or len(right_positions) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_min": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
        }

    cross = mat[np.ix_(left_positions, right_positions)].reshape(-1).tolist()
    return _safe_stats(cross, prefix)


def _member_mean_offdiag(mat: Optional[np.ndarray]) -> List[float]:
    if mat is None or mat.shape[0] <= 1:
        return []

    out = []
    for i in range(mat.shape[0]):
        row = np.delete(mat[i], i)
        if row.size == 0:
            out.append(np.nan)
        else:
            out.append(float(np.mean(row)))
    return out


def _member_sum_offdiag(mat: Optional[np.ndarray]) -> List[float]:
    if mat is None or mat.shape[0] <= 1:
        return []

    out = []
    for i in range(mat.shape[0]):
        row = np.delete(mat[i], i)
        if row.size == 0:
            out.append(np.nan)
        else:
            out.append(float(np.sum(row)))
    return out


def _member_max_offdiag(mat: Optional[np.ndarray]) -> List[float]:
    if mat is None or mat.shape[0] <= 1:
        return []

    out = []
    for i in range(mat.shape[0]):
        row = np.delete(mat[i], i)
        if row.size == 0:
            out.append(np.nan)
        else:
            out.append(float(np.max(row)))
    return out


def _member_argextreme_partner_text(
    runtime_clients: List,
    member_indices: List[int],
    mat: Optional[np.ndarray],
    *,
    extreme: str = "max",
) -> str:
    """
    For each member, record the partner with the extreme pairwise value.
    extreme="max" for DTW, extreme="min" for affinity.
    Format: gid:partner_gid(value)
    """
    if mat is None or mat.shape[0] <= 1:
        return ""

    parts = []
    n = mat.shape[0]

    for i in range(n):
        row = mat[i].copy()

        if extreme == "max":
            row[i] = -np.inf
            j = int(np.argmax(row))
        elif extreme == "min":
            row[i] = np.inf
            j = int(np.argmin(row))
        else:
            raise ValueError("extreme must be 'max' or 'min'")

        val = float(row[j])
        gid = str(runtime_clients[member_indices[i]].client.graph_id)
        partner_gid = str(runtime_clients[member_indices[j]].client.graph_id)
        parts.append(f"{gid}:{partner_gid}({val:.4f})")

    return "|".join(parts)


def _global_extreme_pair(
    runtime_clients: List,
    member_indices: List[int],
    mat: Optional[np.ndarray],
    *,
    extreme: str = "max",
    value_name: str = "value",
) -> Dict[str, object]:
    if mat is None or mat.shape[0] <= 1:
        return {
            value_name: np.nan,
            f"{value_name}_graph_i": None,
            f"{value_name}_graph_j": None,
            f"{value_name}_member_idx_i": None,
            f"{value_name}_member_idx_j": None,
        }

    tri = np.triu_indices(mat.shape[0], k=1)
    vals = mat[tri]

    if vals.size == 0:
        return {
            value_name: np.nan,
            f"{value_name}_graph_i": None,
            f"{value_name}_graph_j": None,
            f"{value_name}_member_idx_i": None,
            f"{value_name}_member_idx_j": None,
        }

    if extreme == "max":
        k = int(np.argmax(vals))
    elif extreme == "min":
        k = int(np.argmin(vals))
    else:
        raise ValueError("extreme must be 'max' or 'min'")

    i = int(tri[0][k])
    j = int(tri[1][k])

    idx_i = member_indices[i]
    idx_j = member_indices[j]

    return {
        value_name: float(vals[k]),
        f"{value_name}_graph_i": str(runtime_clients[idx_i].client.graph_id),
        f"{value_name}_graph_j": str(runtime_clients[idx_j].client.graph_id),
        f"{value_name}_member_idx_i": int(idx_i),
        f"{value_name}_member_idx_j": int(idx_j),
    }


def _format_member_scores(
    runtime_clients: List,
    member_indices: List[int],
    values: List[float],
) -> str:
    parts = []
    for idx, val in zip(member_indices, values):
        gid = runtime_clients[idx].client.graph_id
        if val is None or not np.isfinite(val):
            parts.append(f"{gid}:nan")
        else:
            parts.append(f"{gid}:{val:.4f}")
    return "|".join(parts)


def _pairwise_upper_triangle_records(
    runtime_clients: List,
    member_indices: List[int],
    mat: Optional[np.ndarray],
    value_name: str,
) -> List[Dict]:
    """
    Serialize the upper triangle of a symmetric pairwise matrix.
    This is enough to fully reconstruct the matrix later.
    """
    if mat is None:
        return []

    records: List[Dict] = []
    n = len(member_indices)
    for i in range(n):
        idx_i = member_indices[i]
        gid_i = str(runtime_clients[idx_i].client.graph_id)
        did_i = str(runtime_clients[idx_i].client.dataset_id)

        for j in range(i + 1, n):
            idx_j = member_indices[j]
            gid_j = str(runtime_clients[idx_j].client.graph_id)
            did_j = str(runtime_clients[idx_j].client.dataset_id)

            val = float(mat[i, j])
            records.append(
                {
                    "row_pos": i,
                    "col_pos": j,
                    "member_idx_i": int(idx_i),
                    "member_idx_j": int(idx_j),
                    "graph_id_i": gid_i,
                    "graph_id_j": gid_j,
                    "dataset_id_i": did_i,
                    "dataset_id_j": did_j,
                    value_name: val,
                }
            )
    return records


def _json_dumps_compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _member_order_records(
    runtime_clients: List, member_indices: List[int]
) -> List[Dict]:
    out = []
    for pos, idx in enumerate(member_indices):
        out.append(
            {
                "pos": pos,
                "member_idx": int(idx),
                "graph_id": str(runtime_clients[idx].client.graph_id),
                "dataset_id": str(runtime_clients[idx].client.dataset_id),
            }
        )
    return out


def _member_histories_records(
    runtime_clients: List,
    member_indices: List[int],
    histories_by_member: Dict[int, List[float]],
) -> List[Dict]:
    out = []
    for pos, idx in enumerate(member_indices):
        out.append(
            {
                "pos": pos,
                "member_idx": int(idx),
                "graph_id": str(runtime_clients[idx].client.graph_id),
                "dataset_id": str(runtime_clients[idx].client.dataset_id),
                "history": [float(v) for v in histories_by_member.get(idx, [])],
            }
        )
    return out


def snapshot_clusters(
    clusters: Dict[int, GcflClusterState],
) -> Dict[int, Dict]:
    snap: Dict[int, Dict] = {}
    for cid, cluster in clusters.items():
        snap[cid] = {
            "cluster_id": int(cluster.cluster_id),
            "member_indices": list(cluster.member_indices),
            "state_dict": clone_state_dict(cluster.state_dict),
            "born_round": int(cluster.born_round),
            "mean_norm_history": list(cluster.mean_norm_history),
            "max_norm_history": list(cluster.max_norm_history),
        }
    return snap


def restore_clusters(snapshot: Dict[int, Dict]) -> Dict[int, GcflClusterState]:
    restored: Dict[int, GcflClusterState] = {}
    for cid, item in snapshot.items():
        restored[int(cid)] = GcflClusterState(
            cluster_id=int(item["cluster_id"]),
            member_indices=list(item["member_indices"]),
            state_dict=clone_state_dict(item["state_dict"]),
            born_round=int(item["born_round"]),
            mean_norm_history=list(item["mean_norm_history"]),
            max_norm_history=list(item["max_norm_history"]),
        )
    return restored


# -----------------------------------------------------------------------------
# main experiment
# -----------------------------------------------------------------------------
def run_gcfl_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: GcflRunPaths,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    warmup_rounds: int,
    min_cluster_size: int,
    min_child_size: int,
    grad_seq_len: int,
    eps1_quantile: float,
    eps2_quantile: float,
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

    base_model = make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)
    initial_state = clone_state_dict(base_model.state_dict())

    client_states: List[GcflClientState] = []
    for idx, runtime in enumerate(runtime_clients):
        client_states.append(
            GcflClientState(
                client_idx=idx,
                cluster_id=0,
                train_size=runtime.num_train_nodes,
            )
        )

    clusters: Dict[int, GcflClusterState] = {
        0: GcflClusterState(
            cluster_id=0,
            member_indices=list(range(len(runtime_clients))),
            state_dict=initial_state,
            born_round=1,
        )
    }
    next_cluster_id = 1

    rows: List[Dict] = []
    cluster_rows: List[Dict] = []
    applied_split_event_idx = 0

    best_loss = float("inf")
    best_round = -1
    best_clusters_snapshot = snapshot_clusters(clusters)

    generator = torch.Generator().manual_seed(seed)

    for round_idx in range(1, rounds + 1):
        run_start_time = time.perf_counter()

        new_clusters: Dict[int, GcflClusterState] = {}

        for cluster_id in sorted(clusters.keys()):
            cluster = clusters[cluster_id]
            members = list(cluster.member_indices)

            num_members = len(members)
            num_selected = max(1, int(round(client_fraction * num_members)))
            if num_selected >= num_members:
                selected_members = list(members)
            else:
                perm = torch.randperm(num_members, generator=generator).tolist()
                selected_members = [members[p] for p in perm[:num_selected]]

            local_state_by_member: Dict[int, Dict[str, torch.Tensor]] = {}
            delta_vec_by_member: Dict[int, torch.Tensor] = {}
            weight_by_member: Dict[int, int] = {}

            for idx in selected_members:
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
                local_model.load_state_dict(cluster.state_dict, strict=True)

                client_states[idx].W_old = clone_state_dict(cluster.state_dict)

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
                        run_type="gcfl_plus",
                        algorithm="gcfl_plus",
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

                post_metrics = evaluate_loader(
                    local_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                )
                append_eval_rows(
                    rows,
                    run_type="gcfl_plus",
                    algorithm="gcfl_plus",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase="val_epoch",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=post_metrics,
                    round_idx=round_idx,
                )

                client_states[idx].W = clone_state_dict(local_model.state_dict())
                client_states[idx].dW = subtract_state_dicts(
                    client_states[idx].W,
                    client_states[idx].W_old,
                )

                delta_vec = flatten_state_dict(client_states[idx].dW)
                delta_norm = l2_norm_of_vector(delta_vec)
                client_states[idx].grad_seq.append(delta_norm)
                if len(client_states[idx].grad_seq) > grad_seq_len:
                    client_states[idx].grad_seq = client_states[idx].grad_seq[
                        -grad_seq_len:
                    ]

                local_state_by_member[idx] = client_states[idx].W
                delta_vec_by_member[idx] = delta_vec
                weight_by_member[idx] = runtime.num_train_nodes

            ordered_selected = list(selected_members)
            ordered_delta_vecs = [delta_vec_by_member[idx] for idx in ordered_selected]

            current_mean_norm = mean_update_norm(ordered_delta_vecs)
            current_max_norm = max_update_norm(ordered_delta_vecs)

            cluster.mean_norm_history.append(current_mean_norm)
            cluster.max_norm_history.append(current_max_norm)

            full_participation = len(selected_members) == len(members)

            member_tail_histories = {
                idx: tail_grad_history(client_states[idx].grad_seq, grad_seq_len)
                for idx in members
            }

            split_info = should_split_cluster_gcfl(
                cluster,
                round_idx=round_idx,
                warmup_rounds=warmup_rounds,
                min_cluster_size=min_cluster_size,
                min_child_size=min_child_size,
                full_participation=full_participation,
                current_mean_norm=current_mean_norm,
                current_max_norm=current_max_norm,
                eps1_quantile=eps1_quantile,
                eps2_quantile=eps2_quantile,
            )

            do_split = bool(split_info["do_split"])
            eps1 = split_info["eps1"]
            eps2 = split_info["eps2"]
            split_status = split_info["split_status"]

            split_applied = False
            child_cluster_ids: List[int] = []

            left_child_id: Optional[int] = None
            right_child_id: Optional[int] = None

            left_members: List[int] = []
            right_members: List[int] = []
            left_size: Optional[int] = None
            right_size: Optional[int] = None

            dtw_dist_mean = None
            dtw_dist_max = None
            cut_status: Optional[str] = None
            split_rejected_reason: Optional[str] = None

            cut_value = None
            affinity = None
            dtw_dists = None

            if do_split:
                histories = [member_tail_histories[idx] for idx in members]
                affinity, dtw_dists = dtw_affinity_matrix_from_histories(histories)

                if len(members) > 1:
                    tri = np.triu_indices(len(members), k=1)
                    dtw_dist_mean = float(dtw_dists[tri].mean())
                    dtw_dist_max = float(dtw_dists[tri].max())
                else:
                    dtw_dist_mean = 0.0
                    dtw_dist_max = 0.0

                cut_value, left_members, right_members = stoer_wagner_partition(
                    members, affinity
                )
                left_size = len(left_members)
                right_size = len(right_members)

                if left_size < min_child_size or right_size < min_child_size:
                    cut_status = "invalid_cut_too_small_child"
                    split_rejected_reason = (
                        f"left_size={left_size}, right_size={right_size}, "
                        f"min_child_size={min_child_size}"
                    )
                else:
                    cut_status = "valid_cut"

                    left_state = fedavg_state_dict(
                        [local_state_by_member[idx] for idx in left_members],
                        [weight_by_member[idx] for idx in left_members],
                    )
                    right_state = fedavg_state_dict(
                        [local_state_by_member[idx] for idx in right_members],
                        [weight_by_member[idx] for idx in right_members],
                    )

                    left_id = next_cluster_id
                    next_cluster_id += 1
                    right_id = next_cluster_id
                    next_cluster_id += 1

                    left_child_id = left_id
                    right_child_id = right_id

                    new_clusters[left_id] = GcflClusterState(
                        cluster_id=left_id,
                        member_indices=list(left_members),
                        state_dict=left_state,
                        born_round=round_idx + 1,
                    )
                    new_clusters[right_id] = GcflClusterState(
                        cluster_id=right_id,
                        member_indices=list(right_members),
                        state_dict=right_state,
                        born_round=round_idx + 1,
                    )

                    for idx in left_members:
                        client_states[idx].cluster_id = left_id
                        client_states[idx].grad_seq = []

                    for idx in right_members:
                        client_states[idx].cluster_id = right_id
                        client_states[idx].grad_seq = []

                    split_applied = True
                    child_cluster_ids = [left_id, right_id]

            if not split_applied:
                aggregated_state = fedavg_state_dict(
                    [local_state_by_member[idx] for idx in ordered_selected],
                    [weight_by_member[idx] for idx in ordered_selected],
                )

                new_clusters[cluster_id] = GcflClusterState(
                    cluster_id=cluster.cluster_id,
                    member_indices=list(cluster.member_indices),
                    state_dict=aggregated_state,
                    born_round=cluster.born_round,
                    mean_norm_history=list(cluster.mean_norm_history),
                    max_norm_history=list(cluster.max_norm_history),
                )

                for idx in cluster.member_indices:
                    client_states[idx].cluster_id = cluster.cluster_id

            member_pos = {idx: pos for pos, idx in enumerate(members)}

            left_positions = (
                [member_pos[idx] for idx in left_members] if left_members else []
            )
            right_positions = (
                [member_pos[idx] for idx in right_members] if right_members else []
            )

            all_dtw_stats = _matrix_stats(dtw_dists, "all_dtw")
            all_aff_stats = _matrix_stats(affinity, "all_aff")

            left_dtw_stats = _submatrix_pair_stats(
                dtw_dists, left_positions, "left_dtw"
            )
            right_dtw_stats = _submatrix_pair_stats(
                dtw_dists, right_positions, "right_dtw"
            )
            cross_dtw_stats = _crossmatrix_pair_stats(
                dtw_dists, left_positions, right_positions, "cross_dtw"
            )

            left_aff_stats = _submatrix_pair_stats(affinity, left_positions, "left_aff")
            right_aff_stats = _submatrix_pair_stats(
                affinity, right_positions, "right_aff"
            )
            cross_aff_stats = _crossmatrix_pair_stats(
                affinity, left_positions, right_positions, "cross_aff"
            )

            member_mean_dtw = _member_mean_offdiag(dtw_dists)
            member_mean_aff = _member_mean_offdiag(affinity)

            member_mean_dtw_text = _format_member_scores(
                runtime_clients, members, member_mean_dtw
            )
            member_mean_aff_text = _format_member_scores(
                runtime_clients, members, member_mean_aff
            )

            member_sum_aff = _member_sum_offdiag(affinity)
            member_max_dtw = _member_max_offdiag(dtw_dists)

            member_sum_aff_text = _format_member_scores(
                runtime_clients, members, member_sum_aff
            )
            member_max_dtw_text = _format_member_scores(
                runtime_clients, members, member_max_dtw
            )

            member_max_dtw_partner_text = _member_argextreme_partner_text(
                runtime_clients, members, dtw_dists, extreme="max"
            )
            member_min_aff_partner_text = _member_argextreme_partner_text(
                runtime_clients, members, affinity, extreme="min"
            )

            global_max_pair_dtw_info = _global_extreme_pair(
                runtime_clients,
                members,
                dtw_dists,
                extreme="max",
                value_name="global_max_pair_dtw",
            )
            global_min_pair_aff_info = _global_extreme_pair(
                runtime_clients,
                members,
                affinity,
                extreme="min",
                value_name="global_min_pair_aff",
            )

            delta_norms_selected = [
                l2_norm_of_vector(delta_vec_by_member[idx]) for idx in ordered_selected
            ]
            delta_norm_stats = _safe_stats(delta_norms_selected, "selected_delta_norm")

            selected_histories_text = "|".join(
                f"{runtime_clients[idx].client.graph_id}:{','.join(f'{v:.4f}' for v in member_tail_histories.get(idx, []))}"
                for idx in members
            )

            smaller_child = None
            larger_child = None
            balance_ratio = None
            singleton_side = None
            singleton_idx = None
            singleton_graph_id = None
            singleton_dataset_id = None

            if left_size is not None and right_size is not None:
                smaller_child = int(min(left_size, right_size))
                larger_child = int(max(left_size, right_size))
                if larger_child > 0:
                    balance_ratio = float(smaller_child / larger_child)

                if left_size == 1:
                    singleton_side = "left"
                    singleton_idx = left_members[0]
                    singleton_graph_id = str(
                        runtime_clients[singleton_idx].client.graph_id
                    )
                    singleton_dataset_id = str(
                        runtime_clients[singleton_idx].client.dataset_id
                    )
                elif right_size == 1:
                    singleton_side = "right"
                    singleton_idx = right_members[0]
                    singleton_graph_id = str(
                        runtime_clients[singleton_idx].client.graph_id
                    )
                    singleton_dataset_id = str(
                        runtime_clients[singleton_idx].client.dataset_id
                    )

            peeled_sum_aff = None
            peeled_max_dtw = None
            peeled_max_dtw_partner_graph_id = None
            peeled_in_global_max_pair = None

            if (
                singleton_idx is not None
                and singleton_idx in member_pos
                and affinity is not None
                and dtw_dists is not None
            ):
                s_pos = member_pos[singleton_idx]

                aff_row = np.delete(affinity[s_pos], s_pos)
                dtw_row = np.delete(dtw_dists[s_pos], s_pos)

                if aff_row.size > 0:
                    peeled_sum_aff = float(np.sum(aff_row))

                if dtw_row.size > 0:
                    peeled_max_dtw = float(np.max(dtw_row))

                    other_positions = [p for p in range(len(members)) if p != s_pos]
                    partner_pos = other_positions[int(np.argmax(dtw_row))]
                    partner_idx = members[partner_pos]
                    peeled_max_dtw_partner_graph_id = str(
                        runtime_clients[partner_idx].client.graph_id
                    )

                g_i = global_max_pair_dtw_info["global_max_pair_dtw_graph_i"]
                g_j = global_max_pair_dtw_info["global_max_pair_dtw_graph_j"]
                peeled_in_global_max_pair = int(
                    str(singleton_graph_id) == str(g_i)
                    or str(singleton_graph_id) == str(g_j)
                )

            pairwise_member_order_json = None
            pairwise_histories_json = None
            pairwise_dtw_json = None
            pairwise_affinity_json = None
            pairwise_logged_for_applied_split = 0
            current_applied_split_event_idx = None

            if split_applied:
                applied_split_event_idx += 1
                current_applied_split_event_idx = applied_split_event_idx

                pairwise_member_order_json = _json_dumps_compact(
                    _member_order_records(runtime_clients, members)
                )
                pairwise_histories_json = _json_dumps_compact(
                    _member_histories_records(
                        runtime_clients, members, member_tail_histories
                    )
                )
                pairwise_dtw_json = _json_dumps_compact(
                    _pairwise_upper_triangle_records(
                        runtime_clients, members, dtw_dists, "dtw"
                    )
                )
                pairwise_affinity_json = _json_dumps_compact(
                    _pairwise_upper_triangle_records(
                        runtime_clients, members, affinity, "affinity"
                    )
                )
                pairwise_logged_for_applied_split = 1
            cluster_rows.append(
                {
                    "subset_id": subset_id,
                    "subset_clients": subset_clients_str,
                    "seed": seed,
                    "round": round_idx,
                    "cluster_id": cluster.cluster_id,
                    "members": "|".join(
                        str(runtime_clients[idx].client.graph_id)
                        for idx in cluster.member_indices
                    ),
                    "member_indices": "|".join(
                        str(idx) for idx in cluster.member_indices
                    ),
                    "cluster_size": len(cluster.member_indices),
                    "selected_members": "|".join(
                        str(runtime_clients[idx].client.graph_id)
                        for idx in selected_members
                    ),
                    "selected_member_indices": "|".join(
                        str(idx) for idx in selected_members
                    ),
                    "selected_size": len(selected_members),
                    "full_participation": int(full_participation),
                    "born_round": cluster.born_round,
                    "cluster_age": round_idx - cluster.born_round + 1,
                    "mean_update_norm": current_mean_norm,
                    "max_update_norm": current_max_norm,
                    "eps1": eps1,
                    "eps2": eps2,
                    "split_status": split_status,
                    "split_triggered": int(do_split),
                    "split_applied": int(split_applied),
                    "cut_status": cut_status,
                    "split_rejected_reason": split_rejected_reason,
                    "cut_value": cut_value,
                    "left_child_id": left_child_id,
                    "right_child_id": right_child_id,
                    "child_cluster_ids": "|".join(str(x) for x in child_cluster_ids),
                    "left_size": left_size,
                    "right_size": right_size,
                    "smaller_child_size": smaller_child,
                    "larger_child_size": larger_child,
                    "child_balance_ratio": balance_ratio,
                    "left_members": "|".join(
                        str(runtime_clients[idx].client.graph_id)
                        for idx in left_members
                    ),
                    "right_members": "|".join(
                        str(runtime_clients[idx].client.graph_id)
                        for idx in right_members
                    ),
                    "singleton_side": singleton_side,
                    "singleton_graph_id": singleton_graph_id,
                    "singleton_dataset_id": singleton_dataset_id,
                    "dtw_dist_mean": dtw_dist_mean,
                    "dtw_dist_max": dtw_dist_max,
                    **all_dtw_stats,
                    **all_aff_stats,
                    **left_dtw_stats,
                    **right_dtw_stats,
                    **cross_dtw_stats,
                    **left_aff_stats,
                    **right_aff_stats,
                    **cross_aff_stats,
                    **delta_norm_stats,
                    "member_mean_dtw_text": member_mean_dtw_text,
                    "member_mean_aff_text": member_mean_aff_text,
                    "member_sum_aff_text": member_sum_aff_text,
                    "member_max_dtw_text": member_max_dtw_text,
                    "member_max_dtw_partner_text": member_max_dtw_partner_text,
                    "member_min_aff_partner_text": member_min_aff_partner_text,
                    **global_max_pair_dtw_info,
                    **global_min_pair_aff_info,
                    "peeled_sum_aff": peeled_sum_aff,
                    "peeled_max_dtw": peeled_max_dtw,
                    "peeled_max_dtw_partner_graph_id": peeled_max_dtw_partner_graph_id,
                    "peeled_in_global_max_pair": peeled_in_global_max_pair,
                    "selected_grad_tails_text": selected_histories_text,
                    "applied_split_event_idx": current_applied_split_event_idx,
                    "pairwise_n_members": len(members) if split_applied else None,
                    "pairwise_logged_for_applied_split": pairwise_logged_for_applied_split,
                    "pairwise_member_order_json": pairwise_member_order_json,
                    "pairwise_histories_json": pairwise_histories_json,
                    "pairwise_dtw_json": pairwise_dtw_json,
                    "pairwise_affinity_json": pairwise_affinity_json,
                    "gate_cluster_size_ok": split_info["gate_cluster_size_ok"],
                    "gate_can_form_two_children": split_info[
                        "gate_can_form_two_children"
                    ],
                    "gate_full_participation_ok": split_info[
                        "gate_full_participation_ok"
                    ],
                    "gate_warmup_ok": split_info["gate_warmup_ok"],
                    "gate_history_ok": split_info["gate_history_ok"],
                    "gate_mean_below_eps1": split_info["gate_mean_below_eps1"],
                    "gate_max_above_eps2": split_info["gate_max_above_eps2"],
                    "history_len_prev_mean": split_info["history_len_prev_mean"],
                    "history_len_prev_max": split_info["history_len_prev_max"],
                    "min_cluster_size": min_cluster_size,
                    "min_child_size": min_child_size,
                }
            )

        clusters = new_clusters

        round_eval_infos = []
        for cluster_id in sorted(clusters.keys()):
            cluster = clusters[cluster_id]
            cluster_model = make_model(
                cfg,
                ctx["x_dim"],
                ctx["out_dim"],
                ctx["deg_fwd_hist"],
                ctx["deg_rev_hist"],
                ctx["ego_dim"],
                ctx["in_vocab"],
                ctx["out_vocab"],
            ).to(device)
            cluster_model.load_state_dict(cluster.state_dict, strict=True)

            for idx in cluster.member_indices:
                runtime = runtime_clients[idx]
                metrics = evaluate_loader(
                    cluster_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                )
                round_eval_infos.append(metrics)

                print("gcfl+ eval on client:", runtime.client.graph_id)
                print(
                    "gcfl+ round:",
                    round_idx,
                    "cluster:",
                    cluster_id,
                    "val_loss:",
                    metrics["scalar"]["loss"],
                )
                print("val_macro_minority_f1:", metrics["scalar"]["macro_minority_f1"])
                print("val_minority_f1:", metrics["per_task"]["minority_f1"])
                print("val_positive_f1:", metrics["per_task"]["positive_f1"])

                append_eval_rows(
                    rows,
                    run_type="gcfl_plus",
                    algorithm="gcfl_plus",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase="cluster_val_client",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=round_idx,
                )

        mean_scalar = weighted_scalar_summary(round_eval_infos)
        rows.append(
            {
                "run_type": "gcfl_plus",
                "algorithm": "gcfl_plus",
                "subset_id": subset_id,
                "subset_clients": subset_clients_str,
                "seed": seed,
                "graph_id": "all",
                "dataset_id": "all",
                "phase": "gcfl_round_diag",
                "split": "val",
                "task": None,
                "round": round_idx,
                "local_epoch": None,
                "train_loss": None,
                "eval_loss": mean_scalar["loss"],
                "pair_acc": mean_scalar["pair_acc"],
                "subset_acc": mean_scalar["subset_acc"],
                "micro_f1": mean_scalar["micro_f1"],
                "macro_f1": mean_scalar["macro_f1"],
                "macro_pos_f1": mean_scalar["macro_pos_f1"],
                "macro_minority_f1": mean_scalar["macro_minority_f1"],
                "num_nodes": sum(
                    int(metrics["counts"]["num_nodes"]) for metrics in round_eval_infos
                ),
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
                "num_active_clusters": len(clusters),
                "active_cluster_ids": "|".join(
                    str(cid) for cid in sorted(clusters.keys())
                ),
                "num_split_triggered": sum(
                    1
                    for row in cluster_rows
                    if row["seed"] == seed
                    and row["round"] == round_idx
                    and row["split_triggered"] == 1
                ),
                "num_split_applied": sum(
                    1
                    for row in cluster_rows
                    if row["seed"] == seed
                    and row["round"] == round_idx
                    and row["split_applied"] == 1
                ),
                "num_singleton_cut_candidates": sum(
                    1
                    for row in cluster_rows
                    if row["seed"] == seed
                    and row["round"] == round_idx
                    and row.get("singleton_graph_id") is not None
                ),
            }
        )
        current_value = mean_scalar[selection_metric]
        print("current vs best:", current_value, best_loss)
        if current_value < best_loss:
            best_loss = current_value
            best_round = round_idx
            best_clusters_snapshot = snapshot_clusters(clusters)

        run_elapsed = time.perf_counter() - run_start_time
        print(f"run time: {format_seconds(run_elapsed)}")
        print("=======================================================")
        print(f"==================== round {round_idx} / {rounds} ====================")
        print("=======================================================")

    best_clusters = restore_clusters(best_clusters_snapshot)

    checkpoint = {
        "clusters": {
            cid: {
                "member_indices": list(cluster.member_indices),
                "state_dict": clone_state_dict(cluster.state_dict),
                "born_round": int(cluster.born_round),
                "mean_norm_history": list(cluster.mean_norm_history),
                "max_norm_history": list(cluster.max_norm_history),
            }
            for cid, cluster in best_clusters.items()
        },
        "cfg": cfg,
        "seed": seed,
        "best_round": best_round,
        "best_loss": best_loss,
        "selection_metric": selection_metric,
        "subset_id": subset_id,
        "subset_clients": subset_clients_str,
        "warmup_rounds": warmup_rounds,
        "min_cluster_size": min_cluster_size,
        "min_child_size": min_child_size,
        "grad_seq_len": grad_seq_len,
        "eps1_quantile": eps1_quantile,
        "eps2_quantile": eps2_quantile,
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

    for split_name in ["train", "val", "test"]:
        for cluster_id in sorted(best_clusters.keys()):
            cluster = best_clusters[cluster_id]

            cluster_model = make_model(
                cfg,
                ctx["x_dim"],
                ctx["out_dim"],
                ctx["deg_fwd_hist"],
                ctx["deg_rev_hist"],
                ctx["ego_dim"],
                ctx["in_vocab"],
                ctx["out_vocab"],
            ).to(device)
            cluster_model.load_state_dict(cluster.state_dict, strict=True)

            for idx in cluster.member_indices:
                runtime = runtime_clients[idx]
                split_loader = {
                    "train": runtime.train_loader,
                    "val": runtime.val_loader,
                    "test": runtime.test_loader,
                }[split_name]

                metrics = evaluate_loader(
                    cluster_model,
                    split_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                )
                append_eval_rows(
                    rows,
                    run_type="gcfl_plus",
                    algorithm="gcfl_plus",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"best_cluster_{split_name}",
                    split=split_name,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=best_round,
                )

    pd.DataFrame(rows).to_csv(run_paths.csv_path, index=False)
    pd.DataFrame(cluster_rows).to_csv(run_paths.clusters_csv_path, index=False)
    print(f"saved csv -> {run_paths.csv_path}")
    print(f"saved cluster log csv -> {run_paths.clusters_csv_path}")


def run_gcfl(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    gcfl_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    warmup_rounds: int,
    min_cluster_size: int,
    min_child_size: int,
    grad_seq_len: int,
    eps1_quantile: float,
    eps2_quantile: float,
    device: torch.device,
    selection_metric: str = "loss",
) -> GcflRunPaths:
    subset_id = _subset_clients_str(subset_clients)
    model_tag = _model_tag_from_cfg(cfg)

    run_paths = create_gcfl_run_paths(
        gcfl_root,
        subset_id,
        rounds,
        local_epochs,
        warmup_rounds,
        grad_seq_len,
        min_cluster_size,
        min_child_size,
        eps1_quantile,
        eps2_quantile,
        model_tag,
        seed,
    )

    if (
        run_paths.csv_path.exists()
        and run_paths.ckpt_path.exists()
        and run_paths.clusters_csv_path.exists()
    ):
        return run_paths

    run_gcfl_experiment(
        subset_clients,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        warmup_rounds=warmup_rounds,
        min_cluster_size=min_cluster_size,
        min_child_size=min_child_size,
        grad_seq_len=grad_seq_len,
        eps1_quantile=eps1_quantile,
        eps2_quantile=eps2_quantile,
        device=device,
        selection_metric=selection_metric,
    )
    return run_paths
