from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch

from andrea.helper_funcs_multihead import fl_run_helper_multi_select as fl
from andrea.helper_funcs_multihead.load_client_helper import ClientData
from andrea.helper_funcs_multihead.apple_post_ala_run_helper_multi_select import (
    make_visible_positive_support_matrix,
)

RUN_TYPE = "task_cwfedavg_empirical"
ALGORITHM = "task_cwfedavg_empirical"
DISTRIBUTION_SOURCE = "visible_positive_train_support_empirical_no_pseudocount"
CLASSWISE_AGGREGATION = "task_conditioned_full_model_weighted_aggregation"
PERSONALIZATION = "receiver_task_distribution_weighted_full_model_recombination"
WDR_MODE = "disabled_empirical_distribution"


def _clone_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


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

    # Eq. (6)-style donor weights, adapted from classes to tasks.
    per_task_total = support.sum(dim=0)  # [task]
    if torch.any(per_task_total <= 0):
        bad = torch.where(per_task_total <= 0)[0].tolist()
        raise ValueError(
            "Task-cwFedAvg empirical aggregation is undefined because the "
            f"federation has zero visible positive TRAIN support for task(s) {bad}."
        )
    donor_weights = (support / per_task_total.unsqueeze(0)).T.contiguous()

    # Eq. (7)-style receiver distribution.  In the original multi-class setting,
    # empirical class proportions sum to one automatically.  Here tasks overlap,
    # so we explicitly normalize positive task-event counts across tasks.
    per_client_total = support.sum(dim=1, keepdim=True)  # [client, 1]
    if torch.any(per_client_total <= 0):
        bad = torch.where(per_client_total.squeeze(1) <= 0)[0].tolist()
        raise ValueError(
            "Task-cwFedAvg personalization is undefined because client(s) "
            f"{bad} have zero visible positive TRAIN support across all tasks."
        )
    receiver_task_distribution = support / per_client_total

    if not torch.allclose(
        donor_weights.sum(dim=1),
        torch.ones(donor_weights.shape[0], dtype=donor_weights.dtype),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("Task-cwFedAvg donor weights do not sum to one per task")

    if not torch.allclose(
        receiver_task_distribution.sum(dim=1),
        torch.ones(
            receiver_task_distribution.shape[0],
            dtype=receiver_task_distribution.dtype,
        ),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError(
            "Task-cwFedAvg receiver task distributions do not sum to one"
        )

    return donor_weights, receiver_task_distribution


def weighted_state_dict(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    coefficients: torch.Tensor,
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
    if torch.any(coeffs < -1e-8):
        raise ValueError("Aggregation coefficients must be non-negative")
    if not math.isclose(float(coeffs.sum().item()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
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
    local_states: Sequence[Dict[str, torch.Tensor]],
    donor_weights: torch.Tensor,
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
            "receiver_task_distribution must be [client, task], got "
            f"{tuple(receiver_task_distribution.shape)}"
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
    run_paths: fl.RunPaths,
    runtime_clients,
    visible_support: torch.Tensor,
    donor_weights: torch.Tensor,
    receiver_task_distribution: torch.Tensor,
) -> Path:
    sidecar = run_paths.csv_path.with_name(
        run_paths.csv_path.stem + "_cwfedavg_task_distributions.csv"
    )
    rows = []
    task_names = list(fl.TASKS)
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


def create_task_cwfedavg_run_paths(
    root: str | Path,
    subset_id: str,
    rounds: int,
    local_epochs: int,
    model_tag: str,
    seed: int,
) -> fl.RunPaths:
    root = fl.ensure_dir(root)
    stem = (
        f"task_cwfedavg_empirical{subset_id}"
        f"_rounds{rounds}"
        f"_epoch{local_epochs}"
        f"_{model_tag}"
        f"_seed{seed}"
    )
    stem = fl.safe_stem_for_suffixes(
        stem,
        (".csv", ".pt", "_cwfedavg_task_distributions.csv"),
    )
    return fl.RunPaths(
        csv_path=root / f"{stem}.csv",
        ckpt_path=fl.checkpoint_root_from_runs_root(root) / f"{stem}.pt",
    )


def _init_best_personalized(
    personalized_states: Sequence[Dict[str, torch.Tensor]],
    selection_metrics: Sequence[str],
):
    return {
        eval_mode: {
            metric: {
                "value": fl.initial_selection_value(metric),
                "round": -1,
                "states": _clone_state_list(personalized_states),
            }
            for metric in selection_metrics
        }
        for eval_mode in ("full", "visible")
    }


def _update_best_personalized(
    *,
    best_by_eval_mode,
    eval_mask_mode: str,
    mean_scalar: Dict,
    selection_metrics: Sequence[str],
    personalized_states: Sequence[Dict[str, torch.Tensor]],
    round_idx: int,
) -> None:
    for selector_metric in selection_metrics:
        current_value = mean_scalar.get(selector_metric)
        record = best_by_eval_mode[eval_mask_mode][selector_metric]

        print(
            "Task-cwFedAvg current vs best:",
            "eval_mask_mode=",
            eval_mask_mode,
            "selector_metric=",
            selector_metric,
            "current=",
            current_value,
            "best=",
            record["value"],
        )

        if fl.is_better_selection_value(
            current_value,
            record["value"],
            selector_metric,
        ):
            record["value"] = float(current_value)
            record["round"] = int(round_idx)
            record["states"] = _clone_state_list(personalized_states)


def run_task_cwfedavg_empirical_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: fl.RunPaths,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    device: torch.device,
    selection_metrics: Optional[Sequence[str]] = None,
    selection_metric: Optional[str] = None,
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
            "Task-cwFedAvg final baseline requires CLIENT_FRACTION=1.0 so the "
            "empirical task distributions and aggregation population are fixed."
        )

    fl.set_seed(seed)
    selection_metrics = fl.normalize_selection_metrics(
        selection_metrics, selection_metric
    )

    subset_id = fl._subset_clients_str(subset_clients)
    subset_clients_str = subset_id
    ctx = fl.build_model_context(subset_clients, cfg)
    runtime_clients = fl.build_runtime_clients(subset_clients, cfg, device)

    initial_model = fl.make_model(
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

    visible_support = make_visible_positive_support_matrix(
        runtime_clients,
        int(ctx["out_dim"]),
        device=device,
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

    # All clients begin from the same seeded initialization.  This is equivalent
    # to initializing every task-global model identically; their receiver-weighted
    # combination is therefore the same initial state in round 1.
    personalized_states = [_clone_state_dict(initial_state) for _ in runtime_clients]

    rows: List[Dict] = []
    best_by_eval_mode = _init_best_personalized(
        personalized_states,
        selection_metrics,
    )

    for round_idx in range(1, rounds + 1):
        run_start_time = time.perf_counter()
        local_states: List[Dict[str, torch.Tensor]] = []

        # Client update from the receiver-specific personalized state.
        for client_idx, runtime in enumerate(runtime_clients):
            local_model = fl.make_model(
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
            optimizer = fl.make_optimizer(local_model, cfg)

            for local_epoch_idx in range(1, local_epochs + 1):
                train_loss = fl.train_epoch_neighbor(
                    local_model,
                    runtime.train_loader,
                    optimizer,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                )
                fl.append_train_row(
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
                rows[-1]["selection_metrics"] = "|".join(selection_metrics)

            # Diagnostic local validation before server aggregation.
            for eval_mask_mode in ("full", "visible"):
                metrics = fl.evaluate_loader(
                    local_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                fl.append_eval_rows(
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
                fl.tag_recent_eval_rows(
                    rows,
                    selection_metrics="|".join(selection_metrics),
                )

            local_states.append(_clone_state_dict(local_model.state_dict()))
            del optimizer
            del local_model

        # cwFedAvg stage 1: one complete task-conditioned global model per task.
        task_global_states = make_task_global_states(local_states, donor_weights)

        # cwFedAvg stage 2: receiver-specific recombination of those complete models.
        personalized_states = make_personalized_states(
            task_global_states,
            receiver_task_distribution,
        )
        del local_states
        del task_global_states

        # Model selection must evaluate personalized states, not one shared model.
        eval_model = fl.make_model(
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
                metrics = fl.evaluate_loader(
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

                fl.append_eval_rows(
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
                fl.tag_recent_eval_rows(
                    rows,
                    selection_metrics="|".join(selection_metrics),
                )

            fl.append_mean_eval_row(
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
            rows[-1]["selection_metrics"] = "|".join(selection_metrics)

            mean_scalar = fl.weighted_scalar_summary(round_eval_infos)
            _update_best_personalized(
                best_by_eval_mode=best_by_eval_mode,
                eval_mask_mode=eval_mask_mode,
                mean_scalar=mean_scalar,
                selection_metrics=selection_metrics,
                personalized_states=personalized_states,
                round_idx=round_idx,
            )

        del eval_model
        elapsed = time.perf_counter() - run_start_time
        print(
            f"[Task-cwFedAvg] round {round_idx}/{rounds} "
            f"time={fl.format_seconds(elapsed)}"
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

    final_model = fl.make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)

    for selector_metric in selection_metrics:
        for job in final_eval_jobs:
            selected_by = job["selected_by"]
            selected_record = best_by_eval_mode[selected_by][selector_metric]
            selected_states = selected_record["states"]
            selected_round = selected_record["round"]
            best_value = selected_record["value"]

            for split_name in job["splits"]:
                split_eval_infos = []
                for client_idx, runtime in enumerate(runtime_clients):
                    final_model.load_state_dict(
                        selected_states[client_idx], strict=True
                    )
                    loader = {
                        "train": runtime.train_loader,
                        "val": runtime.val_loader,
                        "test": runtime.test_loader,
                    }[split_name]
                    metrics = fl.evaluate_loader(
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

                    fl.append_eval_rows(
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
                    fl.tag_recent_eval_rows(
                        rows,
                        selection_metrics="|".join(selection_metrics),
                        selector_metric=selector_metric,
                        selector_direction=fl.selection_direction(selector_metric),
                        selected_by_eval_mode=selected_by,
                        selected_epoch=None,
                        selected_round=selected_round,
                        best_val_metric_value=best_value,
                    )

                # Keep one aggregate final row for provenance.  The paper-facing
                # analysis may still recompute its own aggregation from client rows.
                fl.append_mean_eval_row(
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
                        "selection_metrics": "|".join(selection_metrics),
                        "selector_metric": selector_metric,
                        "selector_direction": fl.selection_direction(selector_metric),
                        "selected_by_eval_mode": selected_by,
                        "selected_epoch": None,
                        "selected_round": selected_round,
                        "best_val_metric_value": best_value,
                    }
                )

    checkpoint = {
        "cfg": cfg,
        "seed": seed,
        "selection_metrics": list(selection_metrics),
        "best_round_by_eval_mode_and_metric": {
            mode: {m: rec["round"] for m, rec in by_metric.items()}
            for mode, by_metric in best_by_eval_mode.items()
        },
        "best_value_by_eval_mode_and_metric": {
            mode: {m: rec["value"] for m, rec in by_metric.items()}
            for mode, by_metric in best_by_eval_mode.items()
        },
        "subset_id": subset_id,
        "subset_clients": subset_clients_str,
        "distribution_source": DISTRIBUTION_SOURCE,
        "classwise_aggregation": CLASSWISE_AGGREGATION,
        "personalization": PERSONALIZATION,
        "wdr_mode": WDR_MODE,
    }
    fl.maybe_save_checkpoint(checkpoint, run_paths.ckpt_path)

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
    selection_metrics: Optional[Sequence[str]] = None,
    selection_metric: Optional[str] = None,
) -> fl.RunPaths:
    selection_metrics = fl.normalize_selection_metrics(
        selection_metrics, selection_metric
    )
    cfg = dict(cfg)
    cfg["selection_tag"] = "macro_pr_auc_task_cwfedavg_empirical"

    subset_id = fl._subset_clients_str(subset_clients)
    subset_file_tag = fl._safe_subset_file_tag(subset_id)
    model_tag = f"{fl._model_tag_from_cfg(cfg)}_multiselect"
    run_paths = create_task_cwfedavg_run_paths(
        root,
        subset_file_tag,
        rounds,
        local_epochs,
        model_tag,
        seed,
    )

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
        selection_metrics=selection_metrics,
    )
    return run_paths
