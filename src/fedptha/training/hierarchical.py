from __future__ import annotations
from fedptha.training.runtime import RunPaths
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch
from fedptha.data.tasks import TASKS
from fedptha.training.runtime import (
    append_eval_rows,
    tag_recent_eval_rows,
    weighted_scalar_summary,
)
from fedptha.algorithms.fedala import FedALAClientState, learn_ala_weights_for_client
from fedptha.data.clients import load_clients, parse_subset_clients
from fedptha.training.steps import evaluate_loader
from fedptha.algorithms.fedptha import run_fedptha_with_artifacts
from fedptha.training.runtime import clone_model_from_state
from fedptha.algorithms.routing import (
    _task_idx_from_output_head_param_name,
    count_visible_positives,
)
from fedptha.training.oracle import expand_oracle_hard_clusters


def _copy_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for (k, v) in state.items()}


def _selected_client_state(
    selected_state: Dict, client_idx: int
) -> Dict[str, torch.Tensor]:
    """Return the exact materialized client state used by the existing oracle diagnostic."""
    return _copy_state(selected_state["client_states"][int(client_idx)])


def _cluster_task_head_summary(
    client_states: List[Dict[str, torch.Tensor]],
    visible_support: torch.Tensor,
    pseudo_count: float,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Support-weighted task-head summary for one physical community.

    Returns:
        summary_head_state: task-head entries only.
        within_weights: [task, client] support weights.
    """
    if len(client_states) == 0:
        raise ValueError("client_states is empty")
    if visible_support.ndim != 2:
        raise ValueError(
            f"visible_support must be [client, task], got {tuple(visible_support.shape)}"
        )
    if visible_support.size(0) != len(client_states):
        raise ValueError("client/support count mismatch")
    scores = visible_support.T.float() + float(pseudo_count)
    within_weights = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    out: Dict[str, torch.Tensor] = {}
    for name, first in client_states[0].items():
        task_idx = _task_idx_from_output_head_param_name(name)
        if task_idx is None:
            continue
        if task_idx >= within_weights.size(0):
            raise ValueError(f"Task index {task_idx} outside support matrix")
        if not torch.is_floating_point(first):
            out[name] = first.detach().cpu().clone()
            continue
        acc = None
        for client_idx, state in enumerate(client_states):
            value = state[name].detach().cpu()
            coef = within_weights[task_idx, client_idx].to(dtype=value.dtype)
            term = coef * value
            acc = term if acc is None else acc + term
        out[name] = acc.clone()
    if not out:
        raise RuntimeError(
            "No task-head parameters found while building cluster summary"
        )
    return (out, within_weights.detach().cpu())


def _leave_one_cluster_out_inter_heads(
    cluster_head_states: Dict[int, Dict[str, torch.Tensor]],
    cluster_support: Dict[int, torch.Tensor],
    receiver_cluster: int,
    pseudo_count: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[int, torch.Tensor]]:
    """Task-wise support-weighted average over all *other* physical clusters."""
    source_clusters = [
        c for c in sorted(cluster_head_states) if c != int(receiver_cluster)
    ]
    if not source_clusters:
        raise ValueError("Need at least one external physical cluster")
    num_tasks = int(next(iter(cluster_support.values())).numel())
    support_stack = torch.stack(
        [cluster_support[c].float() for c in source_clusters], dim=0
    )
    score = support_stack.T + float(pseudo_count)
    task_source_weights = score / score.sum(dim=1, keepdim=True).clamp_min(1e-12)
    per_source_weight = {
        c: task_source_weights[:, pos].detach().cpu().clone()
        for (pos, c) in enumerate(source_clusters)
    }
    template = cluster_head_states[source_clusters[0]]
    out: Dict[str, torch.Tensor] = {}
    for name, first in template.items():
        task_idx = _task_idx_from_output_head_param_name(name)
        if task_idx is None:
            continue
        if not torch.is_floating_point(first):
            out[name] = first.detach().cpu().clone()
            continue
        acc = None
        for pos, source_cluster in enumerate(source_clusters):
            value = cluster_head_states[source_cluster][name].detach().cpu()
            coef = task_source_weights[task_idx, pos].to(dtype=value.dtype)
            term = coef * value
            acc = term if acc is None else acc + term
        out[name] = acc.clone()
    return (out, per_source_weight)


def _replace_heads(
    receiver_state: Dict[str, torch.Tensor], inter_head_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    out = _copy_state(receiver_state)
    for name, value in inter_head_state.items():
        if name not in out:
            raise KeyError(
                f"Inter task-head parameter {name!r} absent from receiver state"
            )
        out[name] = value.detach().cpu().clone().to(dtype=out[name].dtype)
    return out


def _tag_recent_rows(rows: List[Dict], **extra) -> None:
    n = 1 + len(TASKS)
    for row in rows[-n:]:
        row.update(extra)


def _append_system_mean_row(
    rows: List[Dict],
    *,
    subset_id: str,
    subset_clients: str,
    seed: int,
    split: str,
    selection_protocol: str,
    selected_by: str,
    eval_mask_mode: str,
    eval_infos: List[Dict],
) -> None:
    mean = weighted_scalar_summary(eval_infos)
    visible_pairs = int(
        sum((int(m["counts"].get("visible_pairs", 0)) for m in eval_infos))
    )
    total_pairs = int(sum((int(m["counts"].get("total_pairs", 0)) for m in eval_infos)))
    rows.append(
        {
            "run_type": RESULT_RUN_TYPE,
            "algorithm": RESULT_ALGORITHM,
            "subset_id": subset_id,
            "subset_clients": subset_clients,
            "seed": seed,
            "graph_id": "all",
            "dataset_id": "all",
            "phase": f"best_{selection_protocol}_{split}_mean",
            "split": split,
            "task": None,
            "round": None,
            "local_epoch": None,
            "train_loss": None,
            "eval_loss": mean.get("loss"),
            "pair_acc": mean.get("pair_acc"),
            "subset_acc": mean.get("subset_acc"),
            "micro_f1": mean.get("micro_f1"),
            "macro_f1": mean.get("macro_f1"),
            "macro_pos_f1": mean.get("macro_pos_f1"),
            "macro_minority_f1": mean.get("macro_minority_f1"),
            "micro_pr_auc": mean.get("micro_pr_auc"),
            "macro_pr_auc": mean.get("macro_pr_auc"),
            "pr_auc": None,
            "num_nodes": int(sum((int(m["counts"]["num_nodes"]) for m in eval_infos))),
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
            "selection_protocol": None,
            "mean_selection_protocol": selection_protocol,
            "selected_by": None,
            "pr_auc_definition": "trapezoidal_precision_recall_curve_auc",
            "visible_pairs": visible_pairs,
            "total_pairs": total_pairs,
            "visible_pair_rate": float(visible_pairs / max(total_pairs, 1)),
            "selection_metrics": "macro_pr_auc",
            "selector_metric": "macro_pr_auc",
            "selector_direction": "max",
            "selected_by_eval_mode": selected_by,
            "hierarchy_stage": "posthoc_inter_cluster",
            "intra_method": "oracle_physical_community_support_simplex_postala",
            "inter_method": "leave_one_cluster_out_support_weighted_task_fedavg",
            "inter_ala_mode": "head_only",
        }
    )


def _build_hierarchical_states(
    cluster_artifacts: Dict[int, Dict], *, selected_by: str, pseudo_count: float
) -> Tuple[Dict[Tuple[int, int], Dict[str, torch.Tensor]], List[Dict], List[Dict]]:
    """Build one-shot inter-cluster task exchange + receiver-local HeadOnly ALA."""
    selector = "macro_pr_auc"
    cluster_local_states: Dict[int, List[Dict[str, torch.Tensor]]] = {}
    cluster_support_matrix: Dict[int, torch.Tensor] = {}
    cluster_total_support: Dict[int, torch.Tensor] = {}
    cluster_head_summary: Dict[int, Dict[str, torch.Tensor]] = {}
    within_weight_rows: List[Dict] = []
    for cluster_id, artifact in sorted(cluster_artifacts.items()):
        selected = artifact["selected_by_eval_mode"][selected_by]
        runtimes = artifact["runtime_clients"]
        states = [_selected_client_state(selected, i) for i in range(len(runtimes))]
        support = (
            count_visible_positives(
                runtimes, int(artifact["num_tasks"]), device=torch.device("cpu")
            )
            .detach()
            .cpu()
        )
        summary, within_weights = _cluster_task_head_summary(
            states, support, pseudo_count
        )
        cluster_local_states[cluster_id] = states
        cluster_support_matrix[cluster_id] = support
        cluster_total_support[cluster_id] = support.sum(dim=0)
        cluster_head_summary[cluster_id] = summary
        for task_idx, task_name in enumerate(TASKS):
            for client_idx, runtime in enumerate(runtimes):
                within_weight_rows.append(
                    {
                        "selected_by": selected_by,
                        "physical_cluster": int(cluster_id),
                        "task_idx": int(task_idx),
                        "task": task_name,
                        "client_idx_within_cluster": int(client_idx),
                        "graph_id": str(runtime.client.graph_id),
                        "visible_train_positive_support": float(
                            support[client_idx, task_idx]
                        ),
                        "within_cluster_task_weight": float(
                            within_weights[task_idx, client_idx]
                        ),
                        "cluster_visible_train_positive_support": float(
                            cluster_total_support[cluster_id][task_idx]
                        ),
                    }
                )
    final_states: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    inter_rows: List[Dict] = []
    ala_rows: List[Dict] = []
    for receiver_cluster, artifact in sorted(cluster_artifacts.items()):
        inter_heads, per_source_weight = _leave_one_cluster_out_inter_heads(
            cluster_head_summary, cluster_total_support, receiver_cluster, pseudo_count
        )
        selected = artifact["selected_by_eval_mode"][selected_by]
        selected_round = int(selected["round"])
        runtimes = artifact["runtime_clients"]
        cfg = artifact["cfg"]
        ctx = artifact["ctx"]
        for task_idx, task_name in enumerate(TASKS):
            for source_cluster, task_weights in sorted(per_source_weight.items()):
                inter_rows.append(
                    {
                        "selected_by": selected_by,
                        "receiver_physical_cluster": int(receiver_cluster),
                        "source_physical_cluster": int(source_cluster),
                        "task_idx": int(task_idx),
                        "task": task_name,
                        "source_cluster_visible_train_positive_support": float(
                            cluster_total_support[source_cluster][task_idx]
                        ),
                        "inter_cluster_task_weight": float(task_weights[task_idx]),
                        "inter_aggregation": "leave_one_cluster_out_support_weighted_task_fedavg",
                        "support_pseudo_count": float(pseudo_count),
                    }
                )
        for client_idx, runtime in enumerate(runtimes):
            local_state = cluster_local_states[receiver_cluster][client_idx]
            external_state = _replace_heads(local_state, inter_heads)
            ala_state = FedALAClientState()
            filtered_state, stats = learn_ala_weights_for_client(
                client_idx=client_idx,
                runtime=runtime,
                cfg=cfg,
                ctx=ctx,
                client_state=ala_state,
                old_local_state=local_state,
                global_state=external_state,
                mode="head_only",
                ala_lr=INTER_ALA_LR,
                ala_rand_percent=INTER_ALA_RAND_PERCENT,
                ala_convergence_std=INTER_ALA_CONVERGENCE_STD,
                ala_convergence_window=INTER_ALA_CONVERGENCE_WINDOW,
                ala_max_steps=INTER_ALA_MAX_STEPS,
                device=DEVICE,
                round_idx=selected_round,
                debug=INTER_ALA_DEBUG,
            )
            final_states[receiver_cluster, client_idx] = _copy_state(filtered_state)
            row = dict(stats)
            row.update(
                {
                    "selected_by": selected_by,
                    "selector_metric": selector,
                    "selected_round": selected_round,
                    "physical_cluster": int(receiver_cluster),
                    "client_idx_within_cluster": int(client_idx),
                    "graph_id": str(runtime.client.graph_id),
                    "dataset_id": str(runtime.client.dataset_id),
                    "inter_ala_mode": "head_only",
                    "inter_ala_training_data": "visible_train_only",
                }
            )
            ala_rows.append(row)
    return (final_states, within_weight_rows + inter_rows, ala_rows)


def _evaluate_hierarchy(
    cluster_artifacts: Dict[int, Dict],
    final_states_by_mode: Dict[str, Dict[Tuple[int, int], Dict[str, torch.Tensor]]],
    *,
    parent_subset_id: str,
    parent_subset_clients: str,
    seed: int,
) -> List[Dict]:
    rows: List[Dict] = []
    jobs = [
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
    for job in jobs:
        selected_by = job["selected_by"]
        selection_protocol = job["selection_protocol"]
        eval_mask_mode = job["eval_mask_mode"]
        final_states = final_states_by_mode[selected_by]
        for split_name in job["splits"]:
            split_eval_infos: List[Dict] = []
            for cluster_id, artifact in sorted(cluster_artifacts.items()):
                selected_state = artifact["selected_by_eval_mode"][selected_by]
                selected_round = int(selected_state["round"])
                cfg = artifact["cfg"]
                ctx = artifact["ctx"]
                for client_idx, runtime in enumerate(artifact["runtime_clients"]):
                    model = clone_model_from_state(
                        final_states[cluster_id, client_idx], cfg, ctx, DEVICE
                    )
                    split_loader = {
                        "train": runtime.train_loader,
                        "val": runtime.val_loader,
                        "test": runtime.test_loader,
                    }[split_name]
                    metrics = evaluate_loader(
                        model,
                        split_loader,
                        runtime.criterion,
                        DEVICE,
                        use_ego_ids=ctx["use_ego_ids"],
                        ego_dim=ctx["ego_dim"],
                        threshold=0.5,
                        eval_mask_mode=eval_mask_mode,
                    )
                    split_eval_infos.append(metrics)
                    append_eval_rows(
                        rows,
                        run_type=RESULT_RUN_TYPE,
                        algorithm=RESULT_ALGORITHM,
                        subset_id=parent_subset_id,
                        subset_clients=parent_subset_clients,
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
                    tag_recent_eval_rows(
                        rows,
                        selection_metrics="macro_pr_auc",
                        selector_metric="macro_pr_auc",
                        selector_direction="max",
                        selected_by_eval_mode=selected_by,
                        selected_round=selected_round,
                        selected_epoch=None,
                        best_val_metric_value=selected_state["value"],
                    )
                    _tag_recent_rows(
                        rows,
                        selection_metric="macro_pr_auc",
                        selection_direction="max",
                        oracle_hard_cluster_id=int(cluster_id),
                        hierarchy_stage="posthoc_inter_cluster",
                        intra_method="oracle_physical_community_support_simplex_postala",
                        inter_method="leave_one_cluster_out_support_weighted_task_fedavg",
                        inter_ala_mode="head_only",
                        inter_ala_training_data="visible_train_only",
                        inter_support_pseudo_count=float(INTER_SUPPORT_PSEUDO_COUNT),
                    )
                    del model
            _append_system_mean_row(
                rows,
                subset_id=parent_subset_id,
                subset_clients=parent_subset_clients,
                seed=seed,
                split=split_name,
                selection_protocol=selection_protocol,
                selected_by=selected_by,
                eval_mask_mode=eval_mask_mode,
                eval_infos=split_eval_infos,
            )
    return rows


def _write_parent_outputs(
    *,
    parent_row: pd.Series,
    seed: int,
    cfg: Dict,
    cluster_artifacts: Dict[int, Dict],
    hierarchy_rows: List[Dict],
    weight_rows: List[Dict],
    ala_rows: List[Dict],
) -> Tuple[Path, Path, Path]:
    root = Path(RUNS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "metrics.csv"
    weights_path = root / "hierarchy_weights.csv"
    ala_path = root / "refinement.csv"
    pd.DataFrame(hierarchy_rows).to_csv(csv_path, index=False)
    pd.DataFrame(weight_rows).to_csv(weights_path, index=False)
    pd.DataFrame(ala_rows).to_csv(ala_path, index=False)
    print("saved hierarchical result ->", csv_path)
    print("saved hierarchy weights ->", weights_path)
    print("saved inter ALA diagnostics ->", ala_path)
    return (csv_path, weights_path, ala_path)


def run_experiment(
    parent_row,
    cfg,
    seed,
    root,
    *,
    registry,
    rounds,
    local_epochs,
    parameters,
    refinement,
    device,
):
    """Run four independent physical federations, then the published inter-region diagnostic."""
    from .experiment import build_client_metadata

    globals().update(
        RUNS_ROOT=str(root),
        ROUNDS=rounds,
        LOCAL_EPOCHS=local_epochs,
        RESULT_ALGORITHM="hierarchical",
        RESULT_RUN_TYPE="hierarchical",
        DEVICE=device,
    )
    defaults = {
        "lr": 1.0,
        "rand_percent": 100.0,
        "convergence_std": 0.1,
        "convergence_window": 10,
        "max_steps": 100,
        "debug": False,
    }
    for key, default in defaults.items():
        globals()["INTER_ALA_" + key.upper()] = refinement.get(key, default)
    globals()["INTER_SUPPORT_PSEUDO_COUNT"] = parameters["support_pseudo_count"]
    expanded = expand_oracle_hard_clusters(pd.DataFrame([parent_row]))
    loaded = load_clients(
        expanded, csv_path=registry, verbose=True, mask_splits=("train", "val", "test")
    )
    artifacts_by_cluster = {}
    for _, cluster_row in expanded.sort_values("oracle_hard_cluster_id").iterrows():
        cluster_id = int(cluster_row["oracle_hard_cluster_id"])
        clients = [
            loaded[x] for x in parse_subset_clients(cluster_row["subset_clients"])
        ]
        paths, artifacts = run_fedptha_with_artifacts(
            clients,
            cfg,
            seed,
            Path(root) / "intra_cluster_runs" / f"region_{cluster_id}",
            rounds=rounds,
            local_epochs=local_epochs,
            client_fraction=1.0,
            device=device,
            client_metadata=build_client_metadata(cluster_row, clients),
            **parameters,
        )
        artifacts["oracle_hard_cluster_id"] = cluster_id
        artifacts["intra_run_paths"] = paths
        artifacts_by_cluster[cluster_id] = artifacts
    if sorted(artifacts_by_cluster) != [0, 1, 2, 3]:
        raise ValueError("Hierarchical diagnostic requires four physical regions.")
    states_by_mode = {}
    weight_rows = []
    ala_rows = []
    for mode in ("visible", "full"):
        states, weights, alas = _build_hierarchical_states(
            artifacts_by_cluster,
            selected_by=mode,
            pseudo_count=parameters["support_pseudo_count"],
        )
        states_by_mode[mode] = states
        weight_rows.extend(weights)
        ala_rows.extend(alas)
    rows = _evaluate_hierarchy(
        artifacts_by_cluster,
        states_by_mode,
        parent_subset_id=str(parent_row.get("subset_id", "parent")),
        parent_subset_clients=str(parent_row["subset_clients"]),
        seed=seed,
    )
    output, _, _ = _write_parent_outputs(
        parent_row=parent_row,
        seed=seed,
        cfg=cfg,
        cluster_artifacts=artifacts_by_cluster,
        hierarchy_rows=rows,
        weight_rows=weight_rows,
        ala_rows=ala_rows,
    )
    return RunPaths(output)
