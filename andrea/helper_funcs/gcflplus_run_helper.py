from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

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
    NOT mean(norm(delta_i))
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
) -> Tuple[List[int], List[int]]:
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

    _, partition = nx.stoer_wagner(graph, weight="weight")
    left = sorted(list(partition[0]))
    right = sorted(list(partition[1]))
    return left, right


# -----------------------------------------------------------------------------
# split criteria
# -----------------------------------------------------------------------------
def should_split_cluster_gcfl(
    cluster: GcflClusterState,
    *,
    round_idx: int,
    warmup_rounds: int,
    min_cluster_size: int,
    full_participation: bool,
    current_mean_norm: float,
    current_max_norm: float,
    eps1_quantile: float,
    eps2_quantile: float,
) -> Tuple[bool, Optional[float], Optional[float], str]:
    if len(cluster.member_indices) < min_cluster_size:
        return False, None, None, "cluster_too_small"

    if not full_participation:
        return False, None, None, "not_full_participation"

    cluster_age = round_idx - cluster.born_round + 1
    if cluster_age <= warmup_rounds:
        return False, None, None, "warmup"

    prev_mean = cluster.mean_norm_history[:-1]
    prev_max = cluster.max_norm_history[:-1]

    if len(prev_mean) < warmup_rounds or len(prev_max) < warmup_rounds:
        return False, None, None, "insufficient_history"

    eps1 = float(np.quantile(np.asarray(prev_mean), eps1_quantile))
    eps2 = float(np.quantile(np.asarray(prev_max), eps2_quantile))

    do_split = (current_mean_norm < eps1) and (current_max_norm > eps2)
    return do_split, eps1, eps2, "eligible"


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

            do_split, eps1, eps2, split_status = should_split_cluster_gcfl(
                cluster,
                round_idx=round_idx,
                warmup_rounds=warmup_rounds,
                min_cluster_size=min_cluster_size,
                full_participation=full_participation,
                current_mean_norm=current_mean_norm,
                current_max_norm=current_max_norm,
                eps1_quantile=eps1_quantile,
                eps2_quantile=eps2_quantile,
            )

            split_applied = False
            child_cluster_ids: List[int] = []
            left_members: List[int] = []
            right_members: List[int] = []
            dtw_dist_mean = None
            dtw_dist_max = None

            if do_split:
                histories = [
                    tail_grad_history(client_states[idx].grad_seq, grad_seq_len)
                    for idx in members
                ]
                affinity, dtw_dists = dtw_affinity_matrix_from_histories(histories)

                if len(members) > 1:
                    tri = np.triu_indices(len(members), k=1)
                    dtw_dist_mean = float(dtw_dists[tri].mean())
                    dtw_dist_max = float(dtw_dists[tri].max())
                else:
                    dtw_dist_mean = 0.0
                    dtw_dist_max = 0.0

                left_members, right_members = stoer_wagner_partition(members, affinity)

                if (
                    len(left_members) >= min_child_size
                    and len(right_members) >= min_child_size
                ):
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

                    # IMPORTANT GCFL+ fix:
                    # reset per-client grad history after a split so the child-cluster
                    # DTW history reflects only the new cluster dynamics.
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
                    "child_cluster_ids": "|".join(str(x) for x in child_cluster_ids),
                    "left_members": "|".join(
                        str(runtime_clients[idx].client.graph_id)
                        for idx in left_members
                    ),
                    "right_members": "|".join(
                        str(runtime_clients[idx].client.graph_id)
                        for idx in right_members
                    ),
                    "dtw_dist_mean": dtw_dist_mean,
                    "dtw_dist_max": dtw_dist_max,
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
                "phase": "cluster_val_mean",
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
