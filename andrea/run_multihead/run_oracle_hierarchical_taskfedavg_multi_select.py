from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from andrea.multigraph_generation import TASKS
from andrea.helper_funcs_multihead.benchmark_config import resolve_benchmark_paths
from andrea.helper_funcs_multihead.fl_run_helper_multi_select import (
    append_eval_rows,
    iter_sweep_cfgs,
    selection_direction,
    tag_recent_eval_rows,
    upsert_experiment_rows,
    weighted_scalar_summary,
)
from andrea.helper_funcs_multihead.fedala_run_helper_multi_select import (
    FedALAClientState,
    learn_ala_weights_for_client,
)
from andrea.helper_funcs_multihead.load_client_helper import (
    ProgressPrinter,
    audit_loaded_q_label_masks,
    load_clients,
    parse_subset_clients,
)
from andrea.helper_funcs_multihead.train_eval_helper import evaluate_loader
from andrea.helper_funcs_multihead.oracle_hierarchical_taskfedavg_run_helper_multi_select import (
    _task_idx_from_output_head_param_name,
    clone_model_from_state,
    make_visible_positive_support_matrix,
    run_apple_with_artifacts,
)
from andrea.run_multihead.run_oracle_hard_cluster_apple_post_ala_multi_select import (
    add_subset_metadata,
    build_client_metadata,
    expand_oracle_hard_clusters,
    load_cfg,
)


BENCHMARK = resolve_benchmark_paths()
SELECT_SUBSET_PATH = BENCHMARK.select_subset_path
SELECT_SUBSET = BENCHMARK.select_subset

RUN_TAG = os.environ.get("RUN_TAG", "prauc_20260824").strip()
BASE_EXPERIMENT_LOG_FOLDER = "oracle_hierarchical_taskfedavg_ala_experiment_multi_select"
EXPERIMENT_LOG_FOLDER = (
    f"{BASE_EXPERIMENT_LOG_FOLDER}_{RUN_TAG}" if RUN_TAG else BASE_EXPERIMENT_LOG_FOLDER
)
DATA_DIR = f"{SELECT_SUBSET_PATH}/cluster_generation_parameters.csv"
SELECTED_SUBSETS_CSV_PATH = f"./andrea/{SELECT_SUBSET_PATH}/{SELECT_SUBSET}.csv"
EXPERIMENT_LOG_CSV = Path(f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv")
ALL_DATA_LOGS = f"./andrea/{DATA_DIR}"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
RUNS_ROOT = os.environ.get(
    "RUNS_ROOT",
    "andrea/final_runs_prauc_20260824/twenty_client/oracle_hierarchical_taskfedavg_ala",
)

ROUNDS = int(os.environ.get("ROUNDS", 80))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
CLIENT_FRACTION = float(os.environ.get("CLIENT_FRACTION", 1.0))
SELECTION_METRICS = ["macro_pr_auc"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

APPLE_DR_LR = float(os.environ.get("APPLE_DR_LR", 1e-3))
APPLE_SUPPORT_PSEUDO_COUNT = float(os.environ.get("APPLE_SUPPORT_PSEUDO_COUNT", 1.0))

# Existing intra-cluster PostALA configuration.
APPLE_USE_HEAD_ALA = os.environ.get("APPLE_USE_HEAD_ALA", "True") == "True"
APPLE_ALA_LR = float(os.environ.get("APPLE_ALA_LR", 1.0))
APPLE_ALA_RAND_PERCENT = float(os.environ.get("APPLE_ALA_RAND_PERCENT", 100.0))
APPLE_ALA_CONVERGENCE_STD = float(os.environ.get("APPLE_ALA_CONVERGENCE_STD", 0.1))
APPLE_ALA_CONVERGENCE_WINDOW = int(os.environ.get("APPLE_ALA_CONVERGENCE_WINDOW", 10))
APPLE_ALA_MAX_STEPS = int(os.environ.get("APPLE_ALA_MAX_STEPS", 100))
APPLE_ALA_DEBUG = os.environ.get("APPLE_ALA_DEBUG", "0") == "1"

# One-shot inter-cluster HeadOnly ALA configuration. Defaults intentionally match
# the existing PostALA configuration unless explicitly overridden.
INTER_ALA_LR = float(os.environ.get("INTER_ALA_LR", APPLE_ALA_LR))
INTER_ALA_RAND_PERCENT = float(
    os.environ.get("INTER_ALA_RAND_PERCENT", APPLE_ALA_RAND_PERCENT)
)
INTER_ALA_CONVERGENCE_STD = float(
    os.environ.get("INTER_ALA_CONVERGENCE_STD", APPLE_ALA_CONVERGENCE_STD)
)
INTER_ALA_CONVERGENCE_WINDOW = int(
    os.environ.get("INTER_ALA_CONVERGENCE_WINDOW", APPLE_ALA_CONVERGENCE_WINDOW)
)
INTER_ALA_MAX_STEPS = int(
    os.environ.get("INTER_ALA_MAX_STEPS", APPLE_ALA_MAX_STEPS)
)
INTER_ALA_DEBUG = os.environ.get("INTER_ALA_DEBUG", "0") == "1"
INTER_SUPPORT_PSEUDO_COUNT = float(
    os.environ.get("INTER_SUPPORT_PSEUDO_COUNT", APPLE_SUPPORT_PSEUDO_COUNT)
)

SEEDS = [0, 1, 2]
MCW = ["auto"]
NUM_LAYERS = [6]
LRS = [0.001]
WEIGHT_DECAYS = [0.0001]
DROPOUTS = [0.1]
HIDDEN_DIMS = [64]
USE_EGO_IDS = [True]
BATCH_SIZE = [64]
MASK_SPLITS = ("train", "val", "test")

RESULT_RUN_TYPE = "oracle_hierarchical_taskfedavg_ala_multi_select"
RESULT_ALGORITHM = "oracle_hierarchical_taskfedavg_ala_multi_select"
DISPLAY_NAME = "Oracle Hierarchical Intra-SS + Inter-TaskFedAvg + ALA"


def _safe_stem(raw: str, max_chars: int = 120) -> str:
    raw = str(raw)
    safe = raw.replace("|", "-").replace("/", "_").replace("\\", "_")
    if len(safe) <= max_chars:
        return safe
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:max_chars-13]}_{digest}"


def _copy_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _selected_client_state(selected_state: Dict, client_idx: int) -> Dict[str, torch.Tensor]:
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
        raise RuntimeError("No task-head parameters found while building cluster summary")
    return out, within_weights.detach().cpu()


def _leave_one_cluster_out_inter_heads(
    cluster_head_states: Dict[int, Dict[str, torch.Tensor]],
    cluster_support: Dict[int, torch.Tensor],
    receiver_cluster: int,
    pseudo_count: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[int, torch.Tensor]]:
    """Task-wise support-weighted average over all *other* physical clusters."""
    source_clusters = [c for c in sorted(cluster_head_states) if c != int(receiver_cluster)]
    if not source_clusters:
        raise ValueError("Need at least one external physical cluster")

    num_tasks = int(next(iter(cluster_support.values())).numel())
    support_stack = torch.stack(
        [cluster_support[c].float() for c in source_clusters], dim=0
    )  # [source_cluster, task]
    score = support_stack.T + float(pseudo_count)  # [task, source_cluster]
    task_source_weights = score / score.sum(dim=1, keepdim=True).clamp_min(1e-12)

    per_source_weight = {
        c: task_source_weights[:, pos].detach().cpu().clone()
        for pos, c in enumerate(source_clusters)
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

    return out, per_source_weight


def _replace_heads(
    receiver_state: Dict[str, torch.Tensor],
    inter_head_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out = _copy_state(receiver_state)
    for name, value in inter_head_state.items():
        if name not in out:
            raise KeyError(f"Inter task-head parameter {name!r} absent from receiver state")
        out[name] = value.detach().cpu().clone().to(dtype=out[name].dtype)
    return out


def _tag_recent_rows(rows: List[Dict], **extra) -> None:
    # append_eval_rows adds one scalar row plus one row per task.
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
    visible_pairs = int(sum(int(m["counts"].get("visible_pairs", 0)) for m in eval_infos))
    total_pairs = int(sum(int(m["counts"].get("total_pairs", 0)) for m in eval_infos))
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
            "num_nodes": int(sum(int(m["counts"]["num_nodes"]) for m in eval_infos)),
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
    cluster_artifacts: Dict[int, Dict],
    *,
    selected_by: str,
    pseudo_count: float,
) -> Tuple[
    Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    List[Dict],
    List[Dict],
]:
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
        support = make_visible_positive_support_matrix(
            runtimes,
            int(artifact["num_tasks"]),
            device=torch.device("cpu"),
        ).detach().cpu()
        summary, within_weights = _cluster_task_head_summary(
            states,
            support,
            pseudo_count,
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
                        "visible_train_positive_support": float(support[client_idx, task_idx]),
                        "within_cluster_task_weight": float(within_weights[task_idx, client_idx]),
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
            cluster_head_summary,
            cluster_total_support,
            receiver_cluster,
            pseudo_count,
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
            final_states[(receiver_cluster, client_idx)] = _copy_state(filtered_state)
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

    return final_states, within_weight_rows + inter_rows, ala_rows


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
                        final_states[(cluster_id, client_idx)],
                        cfg,
                        ctx,
                        DEVICE,
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
    parent_subset_id = str(parent_row.get("subset_id", "parent"))
    parent_clients = str(parent_row["subset_clients"])
    q = parent_row.get("q_value", parent_row.get("gamma", "na"))
    try:
        qtag = f"q{int(round(float(q) * 100)):02d}"
    except Exception:
        qtag = "qna"
    stem = _safe_stem(
        f"oracle_hierarchical_taskfedavg_ala_{qtag}_{parent_subset_id}_r{ROUNDS}_e{LOCAL_EPOCHS}_s{seed}"
    )
    csv_path = root / f"{stem}.csv"
    weights_path = root / f"{stem}_hierarchy_weights.csv"
    ala_path = root / f"{stem}_inter_ala.csv"

    pd.DataFrame(hierarchy_rows).to_csv(csv_path, index=False)
    pd.DataFrame(weight_rows).to_csv(weights_path, index=False)
    pd.DataFrame(ala_rows).to_csv(ala_path, index=False)

    print("saved hierarchical result ->", csv_path)
    print("saved hierarchy weights ->", weights_path)
    print("saved inter ALA diagnostics ->", ala_path)
    return csv_path, weights_path, ala_path


def main() -> None:
    print("BENCHMARK_SETUP:", BENCHMARK.setup)
    if BENCHMARK.setup != "twenty_client":
        raise ValueError(
            "Oracle hierarchical diagnostic is defined only for BENCHMARK_SETUP=twenty_client. "
            f"Got {BENCHMARK.setup!r}."
        )

    if os.environ.get("SAVE_CHECKPOINTS", "0") == "1":
        raise RuntimeError("SAVE_CHECKPOINTS=1 is forbidden for this CSV-only diagnostic.")

    print("RUN VARIANT:", DISPLAY_NAME)
    print("RUNS_ROOT:", RUNS_ROOT)
    print("ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)
    print("SELECTOR: macro_pr_auc / max / true trapezoidal PR-AUC")
    print("INTRA: Oracle physical community Support-Simplex + PostALA")
    print("INTER: one-shot leave-one-cluster-out support-weighted TaskFedAvg")
    print("INTER FILTER: HeadOnly ALA on visible local training data")

    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)
    only_q = os.environ.get("ONLY_Q")
    if only_q is not None:
        q = float(only_q)
        chosen_df = chosen_df[
            pd.to_numeric(chosen_df["q_value"], errors="coerce")
            .round(10)
            .eq(round(q, 10))
        ].copy()
        print("ONLY_Q filter:", only_q, "remaining parents:", len(chosen_df))

    max_subsets = os.environ.get("MAX_SUBSETS")
    if max_subsets is not None:
        chosen_df = chosen_df.head(int(max_subsets)).copy()
        print("MAX_SUBSETS filter:", max_subsets, "remaining parents:", len(chosen_df))

    if chosen_df.empty:
        raise ValueError("No parent subsets remain after filtering")

    expanded_df = expand_oracle_hard_clusters(chosen_df)
    id_to_client = load_clients(
        expanded_df,
        csv_path=ALL_DATA_LOGS,
        verbose=True,
        mask_splits=MASK_SPLITS,
    )
    if os.environ.get("SKIP_Q_MASK_AUDIT", "0") != "1":
        audit_loaded_q_label_masks(
            expanded_df,
            id_to_client,
            strict=True,
            mask_splits=MASK_SPLITS,
        )

    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)
    base_cfg["output_head"] = "multi"
    base_cfg["apple_mixing_mode"] = (
        "oracle_hierarchical_intra_fedavg_backbone_support_simplex_task_head"
    )
    base_cfg["selection_tag"] = "prauc_20260824"
    base_cfg["result_algorithm"] = RESULT_ALGORITHM
    base_cfg["apple_use_head_ala"] = APPLE_USE_HEAD_ALA
    base_cfg["apple_ala_lr"] = APPLE_ALA_LR
    base_cfg["apple_ala_rand_percent"] = APPLE_ALA_RAND_PERCENT
    base_cfg["apple_ala_convergence_std"] = APPLE_ALA_CONVERGENCE_STD
    base_cfg["apple_ala_convergence_window"] = APPLE_ALA_CONVERGENCE_WINDOW
    base_cfg["apple_ala_max_steps"] = APPLE_ALA_MAX_STEPS
    base_cfg["apple_ala_debug"] = APPLE_ALA_DEBUG
    base_cfg["apple_support_pseudo_count"] = APPLE_SUPPORT_PSEUDO_COUNT
    base_cfg["apple_routing_mode"] = "adaptive_visible_support_simplex"

    only_seed = os.environ.get("ONLY_SEED")
    if only_seed is not None:
        active_seeds = [int(only_seed)]
        experiment_log_csv = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log_{only_seed}.csv"
        )
    else:
        active_seeds = SEEDS
        experiment_log_csv = EXPERIMENT_LOG_CSV

    sweep = list(
        iter_sweep_cfgs(
            base_cfg,
            seeds=active_seeds,
            mcw=MCW,
            num_layers=NUM_LAYERS,
            lrs=LRS,
            weight_decays=WEIGHT_DECAYS,
            dropouts=DROPOUTS,
            hidden_dims=HIDDEN_DIMS,
            use_ego_ids=USE_EGO_IDS,
            batch_sizes=BATCH_SIZE,
        )
    )
    progress = ProgressPrinter(len(sweep) * len(chosen_df))

    for cfg, meta in sweep:
        seed = int(meta["seed"])
        for _, parent_row in chosen_df.iterrows():
            run_start = time.perf_counter()
            parent_subset_id = str(parent_row.get("subset_id", "parent"))
            parent_subset_clients = str(parent_row["subset_clients"])
            cluster_rows = expanded_df[
                expanded_df["oracle_parent_subset_id"].astype(str).eq(parent_subset_id)
            ].copy()
            if len(cluster_rows) != 4:
                raise ValueError(
                    f"Expected 4 oracle physical communities for {parent_subset_id}, "
                    f"found {len(cluster_rows)}"
                )

            print("\n" + "=" * 110)
            print("HIERARCHICAL PARENT:", parent_subset_id)
            print("clients:", parent_subset_clients)
            print("seed:", seed)
            print("=" * 110)

            cluster_artifacts: Dict[int, Dict] = {}
            intra_root = Path(RUNS_ROOT) / "intra_cluster_runs"
            os.environ["APPLE_EXPERIMENT_ALGORITHM"] = (
                "oracle_hierarchical_intra_support_simplex_postala"
            )

            for _, cluster_row in cluster_rows.sort_values("oracle_hard_cluster_id").iterrows():
                cluster_id = int(cluster_row["oracle_hard_cluster_id"])
                subset_ids = parse_subset_clients(cluster_row["subset_clients"])
                subset_clients = [id_to_client[gid] for gid in subset_ids]
                client_metadata = build_client_metadata(cluster_row, subset_clients)
                print(
                    f"\n--- Intra cluster {cluster_id}: {cluster_row['subset_clients']} ---"
                )
                run_paths, artifacts = run_apple_with_artifacts(
                    subset_clients,
                    cfg,
                    seed,
                    intra_root,
                    rounds=ROUNDS,
                    local_epochs=LOCAL_EPOCHS,
                    client_fraction=CLIENT_FRACTION,
                    dr_lr=APPLE_DR_LR,
                    support_pseudo_count=APPLE_SUPPORT_PSEUDO_COUNT,
                    device=DEVICE,
                    selection_metrics=SELECTION_METRICS,
                    client_metadata=client_metadata,
                )
                artifacts["oracle_hard_cluster_id"] = cluster_id
                artifacts["intra_run_paths"] = run_paths
                cluster_artifacts[cluster_id] = artifacts

            if sorted(cluster_artifacts) != [0, 1, 2, 3]:
                raise ValueError(
                    f"Expected physical clusters [0,1,2,3], found {sorted(cluster_artifacts)}"
                )

            final_states_by_mode: Dict[str, Dict[Tuple[int, int], Dict[str, torch.Tensor]]] = {}
            all_weight_rows: List[Dict] = []
            all_ala_rows: List[Dict] = []
            for selected_by in ("visible", "full"):
                final_states, weight_rows, ala_rows = _build_hierarchical_states(
                    cluster_artifacts,
                    selected_by=selected_by,
                    pseudo_count=INTER_SUPPORT_PSEUDO_COUNT,
                )
                final_states_by_mode[selected_by] = final_states
                all_weight_rows.extend(weight_rows)
                all_ala_rows.extend(ala_rows)

            hierarchy_rows = _evaluate_hierarchy(
                cluster_artifacts,
                final_states_by_mode,
                parent_subset_id=parent_subset_id,
                parent_subset_clients=parent_subset_clients,
                seed=seed,
            )

            out_csv, weights_csv, ala_csv = _write_parent_outputs(
                parent_row=parent_row,
                seed=seed,
                cfg=cfg,
                cluster_artifacts=cluster_artifacts,
                hierarchy_rows=hierarchy_rows,
                weight_rows=all_weight_rows,
                ala_rows=all_ala_rows,
            )

            log_row = {
                "run_type": RESULT_RUN_TYPE,
                "algorithm": RESULT_ALGORITHM,
                "display_name": DISPLAY_NAME,
                "subset_id": parent_subset_id,
                "subset_clients": parent_subset_clients,
                "seed": seed,
                "out_csv": str(out_csv),
                "hierarchy_weights_csv": str(weights_csv),
                "inter_ala_csv": str(ala_csv),
                "rounds": ROUNDS,
                "local_epochs": LOCAL_EPOCHS,
                "selection_metric": "macro_pr_auc",
                "selection_metrics": "macro_pr_auc",
                "selection_direction": "max",
                "pr_auc_definition": "trapezoidal_precision_recall_curve_auc",
                "mask_splits": "|".join(MASK_SPLITS),
                "selection_modes": "full|visible",
                "eval_protocols": "oracle_full|realistic_visible|realistic_selection_oracle",
                "intra_method": "oracle_physical_community_support_simplex_postala",
                "intra_rounds": ROUNDS,
                "inter_timing": "one_shot_after_intra_validation_selection",
                "inter_method": "leave_one_cluster_out_support_weighted_task_fedavg",
                "inter_scope": "task_heads_only",
                "inter_backbone_exchange": False,
                "inter_support_source": "visible_train_positive_support",
                "inter_support_pseudo_count": INTER_SUPPORT_PSEUDO_COUNT,
                "inter_ala_mode": "head_only",
                "inter_ala_training_data": "visible_train_only",
                "inter_ala_lr": INTER_ALA_LR,
                "inter_ala_rand_percent": INTER_ALA_RAND_PERCENT,
                "inter_ala_convergence_std": INTER_ALA_CONVERGENCE_STD,
                "inter_ala_convergence_window": INTER_ALA_CONVERGENCE_WINDOW,
                "inter_ala_max_steps": INTER_ALA_MAX_STEPS,
                "communication": "intra_cluster_training_then_one_shot_inter_cluster_task_head_exchange",
                "num_physical_clusters": 4,
                "clients_per_physical_cluster": 5,
            }
            add_subset_metadata(log_row, parent_row)
            upsert_experiment_rows(experiment_log_csv, [log_row])
            progress.step(
                label=f"Oracle hierarchical diagnostic done -> {out_csv}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
