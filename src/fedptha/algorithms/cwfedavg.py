from __future__ import annotations
from fedptha.training.runtime import create_run_paths
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence
import pandas as pd
import torch
from fedptha.training import runtime as common
from fedptha.data.clients import ClientData
from fedptha.algorithms.routing import count_visible_positives

RUN_TYPE = "task_cwfedavg_empirical"
ALGORITHM = "task_cwfedavg_empirical"
DISTRIBUTION_SOURCE = "visible_positive_train_support_empirical_no_pseudocount"
CLASSWISE_AGGREGATION = "task_conditioned_full_model_weighted_aggregation"
PERSONALIZATION = "receiver_task_distribution_weighted_full_model_recombination"
WDR_MODE = "disabled_empirical_distribution"


def _clone_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for (key, value) in state.items()}


def _clone_state_list(
    states: Sequence[Dict[str, torch.Tensor]],
) -> List[Dict[str, torch.Tensor]]:
    return [_clone_state_dict(state) for state in states]


def make_empirical_cwfedavg_weights(
    visible_support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cwFedAvg-style empirical weights adapted to overlapping tasks.

    Args:
        visible_support:
            Tensor [client, task] containing visible positive TRAIN support s_{i,t}.

    Returns:
        donor_weights:
            Tensor [task, client].  For task t,

                q_{t,j} = s_{j,t} / sum_k s_{k,t}.

            This is the direct analogue of cwFedAvg's class-wise donor weight
            n_{j,c} / sum_k n_{k,c}, with a task-positive event taking the role
            of a class sample.

        receiver_task_distribution:
            Tensor [client, task].  For receiver i,

                p_{i,t} = s_{i,t} / sum_u s_{i,u}.

            Because this benchmark is multi-label rather than mutually exclusive
            multi-class classification, this row normalization is an explicit
            multi-task adaptation: p_{i,:} is the receiver's relative visible
            positive support distribution across tasks.

    No pseudocount is used.  This keeps the empirical-distribution baseline
    separate from Support-Simplex's c=1 support prior and from Task-Weighted
    FedAvg's +1 smoothing.
    """
    support = visible_support.detach().float().cpu()
    if support.ndim != 2:
        raise ValueError(
            f"visible_support must be [client, task], got {tuple(support.shape)}"
        )
    if support.numel() == 0:
        raise ValueError("visible_support is empty")
    if torch.any(support < 0):
        raise ValueError("visible positive support must be non-negative")
    per_task_total = support.sum(dim=0)
    if torch.any(per_task_total <= 0):
        bad = torch.where(per_task_total <= 0)[0].tolist()
        raise ValueError(
            f"Task-cwFedAvg empirical aggregation is undefined because the federation has zero visible positive TRAIN support for task(s) {bad}."
        )
    donor_weights = (support / per_task_total.unsqueeze(0)).T.contiguous()
    per_client_total = support.sum(dim=1, keepdim=True)
    if torch.any(per_client_total <= 0):
        bad = torch.where(per_client_total.squeeze(1) <= 0)[0].tolist()
        raise ValueError(
            f"Task-cwFedAvg personalization is undefined because client(s) {bad} have zero visible positive TRAIN support across all tasks."
        )
    receiver_task_distribution = support / per_client_total
    if not torch.allclose(
        donor_weights.sum(dim=1),
        torch.ones(donor_weights.shape[0], dtype=donor_weights.dtype),
        atol=1e-06,
        rtol=1e-06,
    ):
        raise RuntimeError("Task-cwFedAvg donor weights do not sum to one per task")
    if not torch.allclose(
        receiver_task_distribution.sum(dim=1),
        torch.ones(
            receiver_task_distribution.shape[0], dtype=receiver_task_distribution.dtype
        ),
        atol=1e-06,
        rtol=1e-06,
    ):
        raise RuntimeError(
            "Task-cwFedAvg receiver task distributions do not sum to one"
        )
    return (donor_weights, receiver_task_distribution)


def weighted_state_dict(
    state_dicts: Sequence[Dict[str, torch.Tensor]], coefficients: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Weighted average of complete model states using one simplex vector.

    The implementation mirrors the repository's FedAvg state semantics: every
    state entry is accumulated in floating point and cast back to the reference
    dtype afterwards.  In contrast to Task-Weighted FedAvg, this function mixes
    the complete model, not only one output head.
    """
    if not state_dicts:
        raise ValueError("state_dicts is empty")
    coeffs = coefficients.detach().float().cpu().flatten()
    if coeffs.numel() != len(state_dicts):
        raise ValueError(
            f"Expected {len(state_dicts)} coefficients, got {coeffs.numel()}"
        )
    if not torch.isfinite(coeffs).all():
        raise ValueError("Non-finite aggregation coefficient")
    if torch.any(coeffs < -1e-08):
        raise ValueError("Aggregation coefficients must be non-negative")
    if not math.isclose(float(coeffs.sum().item()), 1.0, rel_tol=1e-06, abs_tol=1e-06):
        raise ValueError(
            f"Aggregation coefficients must sum to one, got {coeffs.sum().item()}"
        )
    ref_keys = list(state_dicts[0].keys())
    for idx, state in enumerate(state_dicts[1:], start=1):
        if list(state.keys()) != ref_keys:
            raise ValueError(f"state_dict key mismatch at index {idx}")
    out: Dict[str, torch.Tensor] = {}
    for key in ref_keys:
        first = state_dicts[0][key]
        acc = None
        for state, coef in zip(state_dicts, coeffs):
            value = state[key].detach().cpu()
            term = value.float() * float(coef.item())
            acc = term if acc is None else acc + term
        out[key] = acc.to(dtype=first.dtype)
    return out


def make_task_global_states(
    local_states: Sequence[Dict[str, torch.Tensor]], donor_weights: torch.Tensor
) -> List[Dict[str, torch.Tensor]]:
    """Eq. (6)-style aggregation: create one complete global model per task."""
    if donor_weights.ndim != 2:
        raise ValueError(
            f"donor_weights must be [task, client], got {tuple(donor_weights.shape)}"
        )
    if int(donor_weights.shape[1]) != len(local_states):
        raise ValueError(
            "donor_weights client dimension does not match local state count"
        )
    return [
        weighted_state_dict(local_states, donor_weights[task_idx])
        for task_idx in range(int(donor_weights.shape[0]))
    ]


def make_personalized_states(
    task_global_states: Sequence[Dict[str, torch.Tensor]],
    receiver_task_distribution: torch.Tensor,
) -> List[Dict[str, torch.Tensor]]:
    """Eq. (7)-style aggregation: recombine task-global models per receiver."""
    if receiver_task_distribution.ndim != 2:
        raise ValueError(
            f"receiver_task_distribution must be [client, task], got {tuple(receiver_task_distribution.shape)}"
        )
    if int(receiver_task_distribution.shape[1]) != len(task_global_states):
        raise ValueError(
            "receiver task dimension does not match task-global model count"
        )
    return [
        weighted_state_dict(task_global_states, receiver_task_distribution[i])
        for i in range(int(receiver_task_distribution.shape[0]))
    ]


def _write_distribution_sidecar(
    run_paths: common.RunPaths,
    runtime_clients,
    visible_support: torch.Tensor,
    donor_weights: torch.Tensor,
    receiver_task_distribution: torch.Tensor,
) -> Path:
    sidecar = run_paths.csv_path.with_name("task_distributions.csv")
    rows = []
    task_names = list(common.TASKS)
    num_clients, num_tasks = visible_support.shape
    for client_idx in range(int(num_clients)):
        runtime = runtime_clients[client_idx]
        for task_idx in range(int(num_tasks)):
            task_name = (
                str(task_names[task_idx])
                if task_idx < len(task_names)
                else f"task_{task_idx}"
            )
            rows.append(
                {
                    "client_idx": client_idx,
                    "graph_id": str(runtime.client.graph_id),
                    "dataset_id": str(runtime.client.dataset_id),
                    "task_idx": task_idx,
                    "task": task_name,
                    "visible_train_positive_support": float(
                        visible_support[client_idx, task_idx].item()
                    ),
                    "task_global_donor_weight_q_tj": float(
                        donor_weights[task_idx, client_idx].item()
                    ),
                    "receiver_task_proportion_p_it": float(
                        receiver_task_distribution[client_idx, task_idx].item()
                    ),
                    "distribution_source": DISTRIBUTION_SOURCE,
                    "wdr_mode": WDR_MODE,
                }
            )
    pd.DataFrame(rows).to_csv(sidecar, index=False)
    print(f"saved Task-cwFedAvg distribution sidecar -> {sidecar}")
    return sidecar


def _init_best_personalized(personalized_states: Sequence[Dict[str, torch.Tensor]]):
    return {
        eval_mode: {
            "value": float("-inf"),
            "round": -1,
            "states": _clone_state_list(personalized_states),
        }
        for eval_mode in ("full", "visible")
    }


def _update_best_personalized(
    *,
    best_by_eval_mode,
    eval_mask_mode: str,
    mean_scalar: Dict,
    personalized_states: Sequence[Dict[str, torch.Tensor]],
    round_idx: int,
) -> None:
    current_value = mean_scalar.get("macro_pr_auc")
    record = best_by_eval_mode[eval_mask_mode]
    print(
        "Task-cwFedAvg current vs best:",
        "eval_mask_mode=",
        eval_mask_mode,
        "selector_metric=",
        "macro_pr_auc",
        "current=",
        current_value,
        "best=",
        record["value"],
    )
    if common.is_better_selection_value(current_value, record["value"]):
        record["value"] = float(current_value)
        record["round"] = int(round_idx)
        record["states"] = _clone_state_list(personalized_states)


def run_task_cwfedavg_empirical_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: common.RunPaths,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    device: torch.device,
) -> None:
    """Empirical Task-cwFedAvg using full-model two-stage personalization.

    This implements a multi-task adaptation of cwFedAvg Eqs. (6)-(7) using
    visible positive TRAIN support instead of mutually-exclusive class counts.
    It deliberately does NOT implement WDR.

    Every round:
      1. client i starts from its personalized full model;
      2. client i performs ordinary local training;
      3. server forms one complete task-conditioned global model per task;
      4. server recombines those task-global models using receiver i's empirical
         task-support distribution, producing one full personalized model per
         client for the next round.
    """
    if not subset_clients:
        raise ValueError("subset_clients is empty")
    if abs(float(client_fraction) - 1.0) > 1e-12:
        raise ValueError(
            "Task-cwFedAvg final baseline requires CLIENT_FRACTION=1.0 so the empirical task distributions and aggregation population are fixed."
        )
    common.set_seed(seed)
    subset_id = common._subset_clients_str(subset_clients)
    subset_clients_str = subset_id
    ctx = common.build_model_context(subset_clients, cfg)
    runtime_clients = common.build_runtime_clients(subset_clients, cfg, device)
    initial_model = common.make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)
    initial_state = _clone_state_dict(initial_model.state_dict())
    del initial_model
    visible_support = count_visible_positives(
        runtime_clients, int(ctx["out_dim"]), device=device
    )
    donor_weights, receiver_task_distribution = make_empirical_cwfedavg_weights(
        visible_support
    )
    print("[Task-cwFedAvg] visible positive TRAIN support [client, task]:")
    print(visible_support)
    print("[Task-cwFedAvg] Eq.(6)-style task donor weights [task, client]:")
    print(donor_weights)
    print("[Task-cwFedAvg] Eq.(7)-style receiver task proportions [client, task]:")
    print(receiver_task_distribution)
    _write_distribution_sidecar(
        run_paths,
        runtime_clients,
        visible_support,
        donor_weights,
        receiver_task_distribution,
    )
    personalized_states = [_clone_state_dict(initial_state) for _ in runtime_clients]
    rows: List[Dict] = []
    best_by_eval_mode = _init_best_personalized(personalized_states)
    for round_idx in range(1, rounds + 1):
        run_start_time = time.perf_counter()
        local_states: List[Dict[str, torch.Tensor]] = []
        for client_idx, runtime in enumerate(runtime_clients):
            local_model = common.make_model(
                cfg,
                ctx["x_dim"],
                ctx["out_dim"],
                ctx["deg_fwd_hist"],
                ctx["deg_rev_hist"],
                ctx["ego_dim"],
                ctx["in_vocab"],
                ctx["out_vocab"],
            ).to(device)
            local_model.load_state_dict(personalized_states[client_idx], strict=True)
            optimizer = common.make_optimizer(local_model, cfg)
            for local_epoch_idx in range(1, local_epochs + 1):
                train_loss = common.train_epoch_neighbor(
                    local_model,
                    runtime.train_loader,
                    optimizer,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                )
                common.append_train_row(
                    rows,
                    run_type=RUN_TYPE,
                    algorithm=ALGORITHM,
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
                rows[-1]["selection_metrics"] = "macro_pr_auc"
            for eval_mask_mode in ("full", "visible"):
                metrics = common.evaluate_loader(
                    local_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                common.append_eval_rows(
                    rows,
                    run_type=RUN_TYPE,
                    algorithm=ALGORITHM,
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"local_val_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )
                common.tag_recent_eval_rows(rows, selection_metrics="macro_pr_auc")
            local_states.append(_clone_state_dict(local_model.state_dict()))
            del optimizer
            del local_model
        task_global_states = make_task_global_states(local_states, donor_weights)
        personalized_states = make_personalized_states(
            task_global_states, receiver_task_distribution
        )
        del local_states
        del task_global_states
        eval_model = common.make_model(
            cfg,
            ctx["x_dim"],
            ctx["out_dim"],
            ctx["deg_fwd_hist"],
            ctx["deg_rev_hist"],
            ctx["ego_dim"],
            ctx["in_vocab"],
            ctx["out_vocab"],
        ).to(device)
        for eval_mask_mode in ("full", "visible"):
            round_eval_infos = []
            for client_idx, runtime in enumerate(runtime_clients):
                eval_model.load_state_dict(personalized_states[client_idx], strict=True)
                metrics = common.evaluate_loader(
                    eval_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                round_eval_infos.append(metrics)
                print(
                    "task-cwfedavg eval:",
                    "client=",
                    runtime.client.graph_id,
                    "round=",
                    round_idx,
                    "eval_mask_mode=",
                    eval_mask_mode,
                    "val_macro_pr_auc=",
                    metrics["scalar"].get("macro_pr_auc"),
                )
                common.append_eval_rows(
                    rows,
                    run_type=RUN_TYPE,
                    algorithm=ALGORITHM,
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"personalized_val_client_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )
                common.tag_recent_eval_rows(rows, selection_metrics="macro_pr_auc")
            common.append_mean_eval_row(
                rows,
                run_type=RUN_TYPE,
                algorithm=ALGORITHM,
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                phase=f"personalized_val_mean_{eval_mask_mode}",
                split="val",
                round_idx=round_idx,
                eval_infos=round_eval_infos,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )
            rows[-1]["selection_metrics"] = "macro_pr_auc"
            mean_scalar = common.weighted_scalar_summary(round_eval_infos)
            _update_best_personalized(
                best_by_eval_mode=best_by_eval_mode,
                eval_mask_mode=eval_mask_mode,
                mean_scalar=mean_scalar,
                personalized_states=personalized_states,
                round_idx=round_idx,
            )
        del eval_model
        elapsed = time.perf_counter() - run_start_time
        print(
            f"[Task-cwFedAvg] round {round_idx}/{rounds} time={common.format_seconds(elapsed)}"
        )
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
    final_model = common.make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)
    for job in final_eval_jobs:
        selected_by = job["selected_by"]
        selected_record = best_by_eval_mode[selected_by]
        selected_states = selected_record["states"]
        selected_round = selected_record["round"]
        best_value = selected_record["value"]
        for split_name in job["splits"]:
            split_eval_infos = []
            for client_idx, runtime in enumerate(runtime_clients):
                final_model.load_state_dict(selected_states[client_idx], strict=True)
                loader = {
                    "train": runtime.train_loader,
                    "val": runtime.val_loader,
                    "test": runtime.test_loader,
                }[split_name]
                metrics = common.evaluate_loader(
                    final_model,
                    loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=job["eval_mask_mode"],
                )
                split_eval_infos.append(metrics)
                common.append_eval_rows(
                    rows,
                    run_type=RUN_TYPE,
                    algorithm=ALGORITHM,
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"best_{job['selection_protocol']}_{split_name}",
                    split=split_name,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=selected_round,
                    eval_mask_mode=job["eval_mask_mode"],
                    selection_protocol=job["selection_protocol"],
                    selected_by=selected_by,
                )
                common.tag_recent_eval_rows(
                    rows,
                    selection_metrics="macro_pr_auc",
                    selector_metric="macro_pr_auc",
                    selector_direction="max",
                    selected_by_eval_mode=selected_by,
                    selected_epoch=None,
                    selected_round=selected_round,
                    best_val_metric_value=best_value,
                )
            common.append_mean_eval_row(
                rows,
                run_type=RUN_TYPE,
                algorithm=ALGORITHM,
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                phase=f"best_{job['selection_protocol']}_{split_name}_mean",
                split=split_name,
                round_idx=selected_round,
                eval_infos=split_eval_infos,
                eval_mask_mode=job["eval_mask_mode"],
                selection_protocol=job["selection_protocol"],
                selected_by=selected_by,
            )
            rows[-1].update(
                {
                    "selection_metrics": "macro_pr_auc",
                    "selector_metric": "macro_pr_auc",
                    "selector_direction": "max",
                    "selected_by_eval_mode": selected_by,
                    "selected_epoch": None,
                    "selected_round": selected_round,
                    "best_val_metric_value": best_value,
                }
            )
    out_df = pd.DataFrame(rows)
    out_df["distribution_source"] = DISTRIBUTION_SOURCE
    out_df["classwise_aggregation"] = CLASSWISE_AGGREGATION
    out_df["personalization"] = PERSONALIZATION
    out_df["wdr_mode"] = WDR_MODE
    out_df["pr_auc_definition"] = "trapezoidal_precision_recall_curve_auc"
    out_df.to_csv(run_paths.csv_path, index=False)
    print(f"saved csv -> {run_paths.csv_path}")


def run_task_cwfedavg_empirical(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    device: torch.device,
) -> common.RunPaths:
    cfg = dict(cfg)
    cfg["selection_tag"] = "macro_pr_auc_task_cwfedavg_empirical"
    subset_id = common._subset_clients_str(subset_clients)
    run_paths = create_run_paths(root)
    if run_paths.csv_path.exists():
        return run_paths
    run_task_cwfedavg_empirical_experiment(
        subset_clients,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        device=device,
    )
    return run_paths
