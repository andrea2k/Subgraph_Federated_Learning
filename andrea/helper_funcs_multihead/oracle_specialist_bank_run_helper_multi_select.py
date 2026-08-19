from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from andrea.multigraph_generation import TASKS
from andrea.helper_funcs_multihead.fl_run_helper_multi_select import (
    _model_tag_from_cfg,
    _subset_clients_str,
    append_eval_rows,
    append_train_row,
    build_model_context,
    build_runtime_clients,
    ensure_dir,
    initial_selection_value,
    is_better_selection_value,
    maybe_save_checkpoint,
    normalize_selection_metrics,
    selection_direction,
    tag_recent_eval_rows,
    weighted_scalar_summary,
)
from andrea.helper_funcs_multihead.fedala_run_helper_multi_select import (
    FedALAClientState,
    learn_ala_weights_for_client,
)
from andrea.helper_funcs_multihead.load_client_helper import ClientData, format_seconds
from andrea.helper_funcs_multihead.train_eval_helper import (
    augment_batch_x_with_ego,
    evaluate_loader,
    get_seed_label_mask,
    masked_loss_from_logits,
    train_epoch_neighbor,
    unpack_batch_edges,
)
from andrea.helper_funcs_multihead.apple_post_ala_run_helper_multi_select import (
    AppleDRState,
    AppleRunPaths,
    _copy_ala_weights,
    _copy_ala_weights_by_client,
    _task_idx_from_output_head_param_name,
    clone_model_from_state,
    clone_state_dict,
    core_state_dicts,
    create_apple_run_paths,
    dr_backbone_matrix_cpu,
    dr_logits_matrix_cpu,
    dr_matrix_cpu,
    effective_task_weights,
    fedavg_backbone_state_dict,
    functional_call_compat,
    learned_task_weights,
    load_backbone_state_into_models,
    make_task_head_model,
    self_buffer_dict,
)
from utils.seed import set_seed


@dataclass(frozen=True)
class SpecialistBankRunPaths:
    csv_path: Path
    ckpt_path: Path
    dr_csv_path: Path
    representatives_csv_path: Path


def _algorithm() -> str:
    return os.environ.get(
        "APPLE_EXPERIMENT_ALGORITHM",
        "oracle_specialist_bank_apple_post_ala_support_simplex_multi_select",
    )


def create_specialist_run_paths(
    root: str,
    subset_id: str,
    rounds: int,
    local_epochs: int,
    dr_lr: float,
    support_pseudo_count: float,
    model_tag: str,
    seed: int,
) -> SpecialistBankRunPaths:
    base = create_apple_run_paths(
        root,
        subset_id,
        rounds,
        local_epochs,
        dr_lr,
        support_pseudo_count,
        model_tag,
        seed,
    )
    # create_apple_run_paths already produces a descriptive filename close
    # to the filesystem filename-length limit. Adding "_representatives.csv"
    # can therefore exceed Linux NAME_MAX (typically 255 bytes).
    # Keep the readable prefix but deterministically shorten this sidecar name.
    rep_suffix = "_representatives.csv"
    rep_stem = base.csv_path.stem
    rep_name = rep_stem + rep_suffix
    if len(rep_name.encode("utf-8")) > 240:
        digest = hashlib.sha1(rep_stem.encode("utf-8")).hexdigest()[:12]
        keep = 240 - len(rep_suffix) - len(digest) - 1
        rep_name = f"{rep_stem[:keep]}_{digest}{rep_suffix}"

    rep_path = base.csv_path.parent / rep_name
    return SpecialistBankRunPaths(
        csv_path=base.csv_path,
        ckpt_path=base.ckpt_path,
        dr_csv_path=base.dr_csv_path,
        representatives_csv_path=rep_path,
    )


def _visible_positive_support_matrix(runtime_clients, num_tasks: int, device) -> torch.Tensor:
    rows = []
    for runtime in runtime_clients:
        y = runtime.client.train_h["n"].y.float()
        if "label_mask" in runtime.client.train_h["n"]:
            mask = runtime.client.train_h["n"].label_mask.float()
        else:
            mask = torch.ones_like(y)
        rows.append((y * mask).sum(dim=0)[: int(num_tasks)])
    if not rows:
        raise ValueError("runtime_clients is empty")
    return torch.stack(rows, dim=0).to(device=device, dtype=torch.float32)


def compute_specialist_support_routing(
    receiver_visible_pos: torch.Tensor,
    donor_visible_pos: torch.Tensor,
    *,
    pseudo_count: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return support prior S [task, donor] and rho [receiver, task]."""
    if receiver_visible_pos.ndim != 2 or donor_visible_pos.ndim != 2:
        raise ValueError("support tensors must be rank-2")
    if receiver_visible_pos.size(1) != donor_visible_pos.size(1):
        raise ValueError("receiver/donor task dimensions do not match")
    c = float(pseudo_count)
    if c < 0.0:
        raise ValueError("pseudo_count must be non-negative")

    score = donor_visible_pos.T + c
    support = score / score.sum(dim=1, keepdim=True).clamp_min(1e-12)

    donor_task_max = donor_visible_pos.max(dim=0).values
    denom = (donor_task_max + c).clamp_min(1e-12)
    reliability = (receiver_visible_pos + c) / denom.unsqueeze(0)
    rho = (1.0 - reliability).clamp(0.0, 1.0)
    return support, rho


def initialize_specialist_dr_vectors(
    runtime_clients,
    representative_indices: Sequence[int],
    num_tasks: int,
    *,
    device,
    support_pseudo_count: float,
) -> Tuple[List[AppleDRState], torch.Tensor]:
    donor_runtimes = [runtime_clients[int(i)] for i in representative_indices]
    donor_sizes = torch.tensor(
        [float(runtime.num_train_nodes) for runtime in donor_runtimes],
        dtype=torch.float32,
        device=device,
    )
    p0 = donor_sizes / donor_sizes.sum().clamp_min(1.0)

    receiver_support = _visible_positive_support_matrix(runtime_clients, num_tasks, device)
    donor_support = receiver_support[
        torch.tensor(list(representative_indices), dtype=torch.long, device=device)
    ]
    support, rho_all = compute_specialist_support_routing(
        receiver_support,
        donor_support,
        pseudo_count=support_pseudo_count,
    )

    base_logits = torch.log(p0.clamp_min(1e-12))
    task_logits_base = base_logits.unsqueeze(0).repeat(int(num_tasks), 1)

    print("[OracleSpecialistBank] representative indices:", list(representative_indices))
    print("[OracleSpecialistBank] receiver visible support [20,task]:")
    print(receiver_support.detach().cpu())
    print("[OracleSpecialistBank] donor visible support [5,task]:")
    print(donor_support.detach().cpu())
    print("[OracleSpecialistBank] support prior S [task,5]:")
    print(support.detach().cpu())
    print("[OracleSpecialistBank] rho [20,task]:")
    print(rho_all.detach().cpu())

    states: List[AppleDRState] = []
    for receiver_idx in range(len(runtime_clients)):
        states.append(
            AppleDRState(
                backbone=torch.nn.Parameter(p0.detach().clone(), requires_grad=False),
                task_logits=torch.nn.Parameter(task_logits_base.clone()),
                support=support.detach().clone(),
                rho=rho_all[receiver_idx].detach().clone(),
            )
        )
    return states, p0


def _selection_eval_mode(selection_view: str) -> str:
    if selection_view == "visible":
        return "visible"
    if selection_view == "oracle":
        return "full"
    raise ValueError("SPECIALIST_SELECTION_VIEW must be 'visible' or 'oracle'")


def select_specialist_representatives(
    core_models: Sequence[torch.nn.Module],
    runtime_clients,
    client_metadata: Dict[str, Dict],
    *,
    common_backbone_state: Dict[str, torch.Tensor],
    selection_view: str,
    ctx: Dict,
    device,
) -> Tuple[List[int], List[Dict]]:
    """Pick one fixed representative per known task-specialist group.

    Selection intentionally compares *heads* rather than whole local models:
    every candidate is first placed on the same all-20 FedAvg backbone, then
    its assigned-task head is evaluated on all four same-task validation
    clients.  The representative is the candidate with the highest macro mean
    assigned-task validation PR-AUC across those four clients.
    """
    eval_mode = _selection_eval_mode(selection_view)
    selected: List[int] = []
    rows: List[Dict] = []

    # Make comparison fair: candidate heads differ, backbone is identical.
    candidate_models = [copy.deepcopy(model).to(device) for model in core_models]
    load_backbone_state_into_models(candidate_models, common_backbone_state)

    for task_idx, task_name in enumerate(TASKS):
        candidates = []
        for idx, runtime in enumerate(runtime_clients):
            gid = str(runtime.client.graph_id)
            assigned = str(client_metadata.get(gid, {}).get("assigned_task", ""))
            if assigned == str(task_name):
                candidates.append(idx)

        if len(candidates) != 4:
            raise ValueError(
                "Oracle specialist bank expects exactly four specialists for "
                f"{task_name}, found {len(candidates)}: {candidates}"
            )

        scored = []
        for idx in candidates:
            candidate_model = candidate_models[idx]
            per_target_scores = []
            target_details = []

            # Evaluate this candidate head on the SAME four same-task clients.
            for target_idx in candidates:
                target_runtime = runtime_clients[target_idx]
                metrics = evaluate_loader(
                    candidate_model,
                    target_runtime.val_loader,
                    target_runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mode,
                )
                score = float(metrics["per_task"]["pr_auc"][task_idx])
                if not math.isnan(score):
                    per_target_scores.append(score)
                target_gid = str(target_runtime.client.graph_id)
                target_details.append(
                    {
                        "target_client_idx": int(target_idx),
                        "target_graph_id": target_gid,
                        "target_physical_community": client_metadata.get(
                            target_gid, {}
                        ).get("physical_community"),
                        "task_pr_auc": score,
                    }
                )

            mean_score = (
                float(sum(per_target_scores) / len(per_target_scores))
                if per_target_scores
                else float("nan")
            )
            sort_score = mean_score if not math.isnan(mean_score) else float("-inf")
            gid_num = int(runtime_clients[idx].client.graph_id)
            scored.append(
                (
                    sort_score,
                    -gid_num,
                    idx,
                    mean_score,
                    len(per_target_scores),
                    target_details,
                )
            )

        scored.sort(reverse=True)
        best = scored[0]
        rep_idx = int(best[2])
        selected.append(rep_idx)

        for sort_score, neg_gid, idx, mean_score, finite_count, target_details in scored:
            runtime = runtime_clients[idx]
            gid = str(runtime.client.graph_id)
            meta = client_metadata.get(gid, {})
            rows.append(
                {
                    "selection_view": selection_view,
                    "eval_mask_mode": eval_mode,
                    "selection_backbone": "all20_fedavg_common_backbone",
                    "selection_scope": "macro_mean_over_4_same_task_validation_clients",
                    "task_idx": task_idx,
                    "task": task_name,
                    "candidate_client_idx": idx,
                    "candidate_graph_id": gid,
                    "candidate_dataset_id": str(runtime.client.dataset_id),
                    "candidate_physical_community": meta.get("physical_community"),
                    "candidate_assigned_task": meta.get("assigned_task"),
                    "candidate_group_mean_val_pr_auc": mean_score,
                    "candidate_group_finite_val_count": finite_count,
                    "candidate_per_target_val_pr_auc_json": json.dumps(target_details),
                    "is_selected_representative": int(idx == rep_idx),
                }
            )

        print(
            "[OracleSpecialistBank] selected",
            task_name,
            "-> client_idx",
            rep_idx,
            "graph_id",
            runtime_clients[rep_idx].client.graph_id,
            "group_mean_val_pr_auc",
            best[3],
            "view",
            selection_view,
        )

    if len(set(selected)) != len(TASKS):
        raise RuntimeError(f"Representative indices are not unique: {selected}")
    return selected, rows


def _mixed_params_from_bank(
    receiver_model: torch.nn.Module,
    donor_models: Sequence[torch.nn.Module],
    dr_state: AppleDRState,
) -> Dict[str, torch.Tensor]:
    task_weights = effective_task_weights(dr_state)
    if task_weights.shape[1] != len(donor_models):
        raise ValueError(
            f"DR donor dimension={task_weights.shape[1]} but donor_models={len(donor_models)}"
        )

    receiver_params = dict(receiver_model.named_parameters())
    donor_params = [dict(model.named_parameters()) for model in donor_models]
    mixed: Dict[str, torch.Tensor] = {}

    for name, receiver_value in receiver_params.items():
        task_idx = _task_idx_from_output_head_param_name(name)
        if task_idx is None:
            mixed[name] = receiver_value.detach()
            continue

        acc = None
        weights = task_weights[int(task_idx)]
        for donor_idx, params in enumerate(donor_params):
            value = params[name].detach()
            coef = weights[donor_idx].to(device=value.device, dtype=value.dtype)
            term = coef * value
            acc = term if acc is None else acc + term
        if acc is None:
            raise RuntimeError(f"Could not mix donor head parameter {name}")
        mixed[name] = acc

    return mixed


def specialist_bank_forward(
    receiver_model,
    donor_models,
    dr_state,
    x_in,
    edge_in,
    *,
    edge_attr_dict=None,
    device=None,
):
    params = _mixed_params_from_bank(receiver_model, donor_models, dr_state)
    buffers = self_buffer_dict(receiver_model)
    return functional_call_compat(
        receiver_model,
        params,
        buffers,
        args=(x_in, edge_in),
        kwargs={"edge_attr_dict": edge_attr_dict, "device": device},
    )


def train_dr_epoch(
    receiver_model,
    donor_models,
    dr_state: AppleDRState,
    loader,
    criterion,
    device,
    *,
    dr_lr: float,
    use_ego_ids: bool,
    ego_dim: int,
) -> float:
    """Fit only the 5-way APPLE routing logits after ordinary local training."""
    optimizer = torch.optim.Adam([dr_state.task_logits], lr=float(dr_lr))
    total = 0.0
    count = 0
    receiver_model.eval()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        x_in, y_seed, B = augment_batch_x_with_ego(batch, use_ego_ids, ego_dim)
        edge_in, edge_attr_dict = unpack_batch_edges(batch)
        out = specialist_bank_forward(
            receiver_model,
            donor_models,
            dr_state,
            x_in,
            edge_in,
            edge_attr_dict=edge_attr_dict,
            device=device,
        )
        out_seed = out[:B]
        label_mask_seed = get_seed_label_mask(batch, B)
        loss = masked_loss_from_logits(
            criterion,
            out_seed,
            y_seed.float(),
            label_mask_seed,
        )
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * B
        count += B
    return total / max(count, 1)


def specialist_personalized_state_dict(
    receiver_model,
    donor_models,
    dr_state: AppleDRState,
) -> Dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in receiver_model.state_dict().items()}
    mixed_params = _mixed_params_from_bank(receiver_model, donor_models, dr_state)
    for name, value in mixed_params.items():
        if _task_idx_from_output_head_param_name(name) is not None:
            out[name] = value.detach().cpu().clone()
    return out


def _append_bank_dr_rows(
    rows: List[Dict],
    *,
    runtime_clients,
    representative_indices: Sequence[int],
    dr_vectors: Sequence[AppleDRState],
    p0: torch.Tensor,
    subset_id: str,
    seed: int,
    round_idx: int,
    client_metadata: Dict[str, Dict],
) -> None:
    effective = dr_matrix_cpu(dr_vectors)
    learned = torch.stack([learned_task_weights(x).detach().cpu() for x in dr_vectors])
    support = torch.stack([x.support.detach().cpu() for x in dr_vectors])
    logits = torch.stack([x.task_logits.detach().cpu() for x in dr_vectors])
    rho = torch.stack([x.rho.detach().cpu() for x in dr_vectors])

    if effective.shape != (len(runtime_clients), len(TASKS), len(representative_indices)):
        raise RuntimeError(f"Unexpected DR shape: {tuple(effective.shape)}")

    for i, runtime in enumerate(runtime_clients):
        receiver_gid = str(runtime.client.graph_id)
        receiver_meta = client_metadata.get(receiver_gid, {})
        for task_idx, task_name in enumerate(TASKS):
            for donor_slot, donor_client_idx in enumerate(representative_indices):
                donor_runtime = runtime_clients[int(donor_client_idx)]
                donor_gid = str(donor_runtime.client.graph_id)
                donor_meta = client_metadata.get(donor_gid, {})
                value = float(effective[i, task_idx, donor_slot].item())
                rows.append(
                    {
                        "row_type": "task_pair",
                        "run_type": _algorithm(),
                        "algorithm": _algorithm(),
                        "subset_id": subset_id,
                        "seed": seed,
                        "round": round_idx,
                        "client_idx": i,
                        "graph_id": receiver_gid,
                        "assigned_task": receiver_meta.get("assigned_task"),
                        "task_idx": task_idx,
                        "task": task_name,
                        "donor_slot": donor_slot,
                        "donor_client_idx": int(donor_client_idx),
                        "donor_graph_id": donor_gid,
                        "donor_assigned_task": donor_meta.get("assigned_task"),
                        "donor_physical_community": donor_meta.get("physical_community"),
                        "is_fixed_representative": 1,
                        "p_task_itj": value,
                        "learned_softmax_itj": float(learned[i, task_idx, donor_slot].item()),
                        "support_prior_tj": float(support[i, task_idx, donor_slot].item()),
                        "routing_rho": float(rho[i, task_idx].item()),
                        "raw_logit_itj": float(logits[i, task_idx, donor_slot].item()),
                        "is_negative_effective": int(value < 0.0),
                        "p0_j": float(p0[donor_slot].detach().cpu().item()),
                        "routing_mode": "oracle_specialist_bank_support_simplex",
                    }
                )


def _append_mean_row(
    rows: List[Dict],
    *,
    phase: str,
    split: str,
    subset_id: str,
    seed: int,
    round_idx: int,
    eval_infos: List[Dict],
    eval_mask_mode: str,
) -> None:
    scalar = weighted_scalar_summary(eval_infos)
    row = {
        "run_type": _algorithm(),
        "algorithm": _algorithm(),
        "subset_id": subset_id,
        "seed": seed,
        "graph_id": "mean",
        "dataset_id": "mean",
        "phase": phase,
        "split": split,
        "round": round_idx,
        "task": None,
        "eval_mask_mode": eval_mask_mode,
    }
    for key, value in scalar.items():
        row[key if key.startswith("eval_") else f"eval_{key}"] = value
    rows.append(row)


def _initial_best_record(states):
    return {
        "value": None,
        "round": -1,
        "local_states": copy.deepcopy(states),
    }


def run_oracle_specialist_bank_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: SpecialistBankRunPaths,
    *,
    rounds: int,
    warmup_rounds: int,
    local_epochs: int,
    client_fraction: float,
    dr_lr: float,
    support_pseudo_count: float,
    specialist_selection_view: str,
    device,
    selection_metrics: Sequence[str],
    client_metadata: Dict[str, Dict],
) -> None:
    if len(subset_clients) != 20:
        raise ValueError(
            "OracleSpecialistBank is defined for exactly 20 clients; "
            f"got {len(subset_clients)}"
        )
    if warmup_rounds < 1 or warmup_rounds >= rounds:
        raise ValueError(
            f"Need 1 <= warmup_rounds < rounds, got {warmup_rounds} and {rounds}"
        )
    if abs(float(client_fraction) - 1.0) > 1e-12:
        raise ValueError(
            "OracleSpecialistBank currently requires CLIENT_FRACTION=1.0 so all "
            "20 backbones participate each collaborative round."
        )

    selection_metrics = normalize_selection_metrics(selection_metrics, None)
    set_seed(seed)
    subset_id = _subset_clients_str(subset_clients)
    ctx = build_model_context(subset_clients, cfg)
    runtime_clients = build_runtime_clients(subset_clients, cfg, device)
    num_tasks = int(ctx["out_dim"])
    if num_tasks != len(TASKS):
        raise ValueError(f"Expected {len(TASKS)} tasks, got {num_tasks}")

    base_model = make_task_head_model(
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
    core_models = [clone_model_from_state(initial_state, cfg, ctx, device) for _ in runtime_clients]

    rows: List[Dict] = []
    dr_rows: List[Dict] = []

    # ------------------------------------------------------------------
    # Phase 1: no communication. Every client learns its own backbone+heads.
    # ------------------------------------------------------------------
    for round_idx in range(1, warmup_rounds + 1):
        start = time.perf_counter()
        updated = []
        for idx, runtime in enumerate(runtime_clients):
            local_model = clone_model_from_state(core_models[idx].state_dict(), cfg, ctx, device)
            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
            )
            for local_epoch_idx in range(1, local_epochs + 1):
                loss = train_epoch_neighbor(
                    local_model,
                    runtime.train_loader,
                    optimizer,
                    runtime.criterion,
                    device,
                    ctx["use_ego_ids"],
                    ctx["ego_dim"],
                )
                append_train_row(
                    rows,
                    run_type=_algorithm(),
                    algorithm=_algorithm(),
                    subset_id=subset_id,
                    subset_clients=subset_id,
                    seed=seed,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    phase="warmup_local_train",
                    split="train",
                    train_loss=loss,
                    num_nodes=runtime.num_train_nodes,
                    round_idx=round_idx,
                    local_epoch=local_epoch_idx,
                )
            updated.append(local_model)
        core_models = updated
        print(
            f"[OracleSpecialistBank] warmup round {round_idx}/{warmup_rounds} "
            f"time={format_seconds(time.perf_counter() - start)}"
        )

    # First communication after warm-up: all 20 backbones -> one FedAvg backbone.
    warmup_global_backbone = fedavg_backbone_state_dict(core_models, runtime_clients)

    # Select representatives only after the common backbone exists. Candidate
    # heads are compared on that same backbone and across all four same-task
    # validation clients, so selection measures transferable task-head expertise
    # rather than whole-model performance on four different local validation sets.
    representative_indices, representative_rows = select_specialist_representatives(
        core_models,
        runtime_clients,
        client_metadata,
        common_backbone_state=warmup_global_backbone,
        selection_view=specialist_selection_view,
        ctx=ctx,
        device=device,
    )
    pd.DataFrame(representative_rows).to_csv(run_paths.representatives_csv_path, index=False)

    # All 20 clients now enter collaboration with the same FedAvg backbone while
    # preserving their own warm-up task heads.
    load_backbone_state_into_models(core_models, warmup_global_backbone)

    dr_vectors, p0 = initialize_specialist_dr_vectors(
        runtime_clients,
        representative_indices,
        num_tasks,
        device=device,
        support_pseudo_count=support_pseudo_count,
    )

    expected_shape = (len(runtime_clients), num_tasks, len(representative_indices))
    actual_shape = tuple(dr_matrix_cpu(dr_vectors).shape)
    print("[OracleSpecialistBank] DR tensor shape:", actual_shape)
    if actual_shape != expected_shape:
        raise RuntimeError(f"Expected DR shape {expected_shape}, got {actual_shape}")

    use_head_ala = bool(cfg.get("apple_use_head_ala", True))
    ala_states = [FedALAClientState() for _ in runtime_clients]
    current_ala_weights_by_client = {i: {} for i in range(len(runtime_clients))}

    initial_states = core_state_dicts(core_models)
    best_by_eval_mode = {
        mode: {metric: _initial_best_record(initial_states) for metric in selection_metrics}
        for mode in ("full", "visible")
    }

    # ------------------------------------------------------------------
    # Phase 2: all clients train locally; 20 backbones FedAvg; 5 donor heads APPLE.
    # ------------------------------------------------------------------
    for round_idx in range(warmup_rounds + 1, rounds + 1):
        start = time.perf_counter()
        round_start_states = core_state_dicts(core_models)
        donor_models = [
            clone_model_from_state(round_start_states[int(rep_idx)], cfg, ctx, device)
            for rep_idx in representative_indices
        ]

        updated_models: Dict[int, torch.nn.Module] = {}
        round_local_states = copy.deepcopy(round_start_states)
        round_eval_infos = {"full": [], "visible": []}
        round_ala_weights = _copy_ala_weights_by_client(current_ala_weights_by_client)

        for idx, runtime in enumerate(runtime_clients):
            # A. Ordinary local learning: keeps every client's own heads alive.
            local_model = clone_model_from_state(round_start_states[idx], cfg, ctx, device)
            local_optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
            )
            for local_epoch_idx in range(1, local_epochs + 1):
                local_loss = train_epoch_neighbor(
                    local_model,
                    runtime.train_loader,
                    local_optimizer,
                    runtime.criterion,
                    device,
                    ctx["use_ego_ids"],
                    ctx["ego_dim"],
                )
                append_train_row(
                    rows,
                    run_type=_algorithm(),
                    algorithm=_algorithm(),
                    subset_id=subset_id,
                    subset_clients=subset_id,
                    seed=seed,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    phase="specialist_bank_local_train",
                    split="train",
                    train_loss=local_loss,
                    num_nodes=runtime.num_train_nodes,
                    round_idx=round_idx,
                    local_epoch=local_epoch_idx,
                )

            trained_local_state = clone_state_dict(local_model.state_dict())

            # B. Fit only the 5-way DR logits on the same visible local train labels.
            dr_loss = train_dr_epoch(
                local_model,
                donor_models,
                dr_vectors[idx],
                runtime.train_loader,
                runtime.criterion,
                device,
                dr_lr=dr_lr,
                use_ego_ids=ctx["use_ego_ids"],
                ego_dim=ctx["ego_dim"],
            )
            rows.append(
                {
                    "run_type": _algorithm(),
                    "algorithm": _algorithm(),
                    "subset_id": subset_id,
                    "seed": seed,
                    "graph_id": str(runtime.client.graph_id),
                    "dataset_id": str(runtime.client.dataset_id),
                    "phase": "specialist_bank_dr_train",
                    "split": "train",
                    "round": round_idx,
                    "task": None,
                    "train_loss": dr_loss,
                    "dr_num_donors": len(representative_indices),
                }
            )

            # C. APPLE donor mixture over 5 reps, then existing head-only PostALA
            #    against the locally trained receiver head.
            incoming_state = specialist_personalized_state_dict(
                local_model,
                donor_models,
                dr_vectors[idx],
            )
            if use_head_ala:
                filtered_state, ala_stats = learn_ala_weights_for_client(
                    client_idx=idx,
                    runtime=runtime,
                    cfg=cfg,
                    ctx=ctx,
                    client_state=ala_states[idx],
                    old_local_state=trained_local_state,
                    global_state=incoming_state,
                    mode="head_only",
                    ala_lr=float(cfg.get("apple_ala_lr", 1.0)),
                    ala_rand_percent=float(cfg.get("apple_ala_rand_percent", 100.0)),
                    ala_convergence_std=float(cfg.get("apple_ala_convergence_std", 0.1)),
                    ala_convergence_window=int(cfg.get("apple_ala_convergence_window", 10)),
                    ala_max_steps=int(cfg.get("apple_ala_max_steps", 100)),
                    device=device,
                    round_idx=round_idx,
                    debug=bool(cfg.get("apple_ala_debug", False)),
                )
                local_model.load_state_dict(
                    {k: v.to(device=device) for k, v in filtered_state.items()},
                    strict=True,
                )
                current_ala_weights_by_client[idx] = _copy_ala_weights(ala_states[idx].weights)
                round_ala_weights[idx] = _copy_ala_weights(ala_states[idx].weights)
                rows.append(
                    {
                        "run_type": _algorithm(),
                        "algorithm": _algorithm(),
                        "subset_id": subset_id,
                        "seed": seed,
                        "graph_id": str(runtime.client.graph_id),
                        "dataset_id": str(runtime.client.dataset_id),
                        "phase": "specialist_bank_postala_diag",
                        "split": "train",
                        "round": round_idx,
                        "task": None,
                        **ala_stats,
                    }
                )

            updated_models[idx] = local_model
            round_local_states[idx] = clone_state_dict(local_model.state_dict())

            for eval_mode in ("full", "visible"):
                metrics = evaluate_loader(
                    local_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mode,
                )
                round_eval_infos[eval_mode].append(metrics)
                append_eval_rows(
                    rows,
                    run_type=_algorithm(),
                    algorithm=_algorithm(),
                    subset_id=subset_id,
                    subset_clients=subset_id,
                    seed=seed,
                    phase=f"specialist_bank_val_{eval_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mode,
                    selection_protocol=None,
                    selected_by=None,
                )

        for idx, model in updated_models.items():
            core_models[idx] = model

        # D. Backbone uses ALL 20 clients. Only heads are compressed to the bank.
        global_backbone = fedavg_backbone_state_dict(core_models, runtime_clients)
        load_backbone_state_into_models(core_models, global_backbone)

        for eval_mode in ("full", "visible"):
            _append_mean_row(
                rows,
                phase=f"specialist_bank_val_mean_{eval_mode}",
                split="val",
                subset_id=subset_id,
                seed=seed,
                round_idx=round_idx,
                eval_infos=round_eval_infos[eval_mode],
                eval_mask_mode=eval_mode,
            )
            local_mean_scalar = weighted_scalar_summary(round_eval_infos[eval_mode])
            for metric in selection_metrics:
                current = local_mean_scalar.get(metric)
                best = best_by_eval_mode[eval_mode][metric]
                comparison = initial_selection_value(metric) if best["value"] is None else best["value"]
                if is_better_selection_value(current, comparison, metric):
                    best["value"] = float(current)
                    best["round"] = round_idx
                    best["local_states"] = copy.deepcopy(round_local_states)
                    best["dr_matrix"] = dr_matrix_cpu(dr_vectors).clone()
                    best["backbone_dr_matrix"] = dr_backbone_matrix_cpu(dr_vectors).clone()
                    best["ala_weights_by_client"] = _copy_ala_weights_by_client(round_ala_weights)

        _append_bank_dr_rows(
            dr_rows,
            runtime_clients=runtime_clients,
            representative_indices=representative_indices,
            dr_vectors=dr_vectors,
            p0=p0,
            subset_id=subset_id,
            seed=seed,
            round_idx=round_idx,
            client_metadata=client_metadata,
        )

        print(
            f"[OracleSpecialistBank] collaboration round {round_idx}/{rounds} "
            f"time={format_seconds(time.perf_counter() - start)}"
        )

    # Final protocols depend on how the bank itself was selected.
    if specialist_selection_view == "visible":
        final_jobs = [
            ("visible_bank_VV", "visible", "visible"),
            ("visible_bank_VO", "visible", "full"),
        ]
    else:
        final_jobs = [("oracle_bank_OO", "full", "full")]

    checkpoint = {
        "algorithm": _algorithm(),
        "cfg": cfg,
        "seed": seed,
        "subset_id": subset_id,
        "rounds": rounds,
        "warmup_rounds": warmup_rounds,
        "representative_indices": list(representative_indices),
        "representative_graph_ids": [
            str(runtime_clients[int(i)].client.graph_id) for i in representative_indices
        ],
        "specialist_selection_view": specialist_selection_view,
        "dr_shape": tuple(dr_matrix_cpu(dr_vectors).shape),
        "best_round_by_eval_mode_and_metric": {
            mode: {metric: best_by_eval_mode[mode][metric]["round"] for metric in selection_metrics}
            for mode in ("full", "visible")
        },
    }
    maybe_save_checkpoint(checkpoint, run_paths.ckpt_path)

    for selector_metric in selection_metrics:
        for protocol, selected_by, eval_mode in final_jobs:
            selected_state = best_by_eval_mode[selected_by][selector_metric]
            if selected_state["round"] < 0:
                raise RuntimeError(
                    f"No selected collaborative checkpoint for {selected_by}/{selector_metric}"
                )
            eval_infos = []
            for idx, runtime in enumerate(runtime_clients):
                model = clone_model_from_state(
                    selected_state["local_states"][idx], cfg, ctx, device
                )
                metrics = evaluate_loader(
                    model,
                    runtime.test_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mode,
                )
                eval_infos.append(metrics)
                append_eval_rows(
                    rows,
                    run_type=_algorithm(),
                    algorithm=_algorithm(),
                    subset_id=subset_id,
                    subset_clients=subset_id,
                    seed=seed,
                    phase=f"best_{protocol}_{selector_metric}_test",
                    split="test",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=selected_state["round"],
                    eval_mask_mode=eval_mode,
                    selection_protocol=protocol,
                    selected_by=selected_by,
                )
                tag_recent_eval_rows(
                    rows,
                    selection_metrics="|".join(selection_metrics),
                    selector_metric=selector_metric,
                    selector_direction=selection_direction(selector_metric),
                    selected_by_eval_mode=selected_by,
                    selected_round=selected_state["round"],
                    selected_epoch=None,
                    best_val_metric_value=selected_state["value"],
                )
            _append_mean_row(
                rows,
                phase=f"best_{protocol}_{selector_metric}_test_mean",
                split="test",
                subset_id=subset_id,
                seed=seed,
                round_idx=selected_state["round"],
                eval_infos=eval_infos,
                eval_mask_mode=eval_mode,
            )
            rows[-1]["mean_selection_protocol"] = protocol

    pd.DataFrame(rows).to_csv(run_paths.csv_path, index=False)
    pd.DataFrame(dr_rows).to_csv(run_paths.dr_csv_path, index=False)
    print("saved csv ->", run_paths.csv_path)
    print("saved DR csv ->", run_paths.dr_csv_path)
    print("saved representative map ->", run_paths.representatives_csv_path)


def run_oracle_specialist_bank(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    root: str,
    *,
    rounds: int,
    warmup_rounds: int,
    local_epochs: int,
    client_fraction: float,
    dr_lr: float,
    support_pseudo_count: float,
    specialist_selection_view: str,
    device,
    selection_metrics: Sequence[str],
    client_metadata: Dict[str, Dict],
) -> SpecialistBankRunPaths:
    subset_id = _subset_clients_str(subset_clients)
    model_tag = _model_tag_from_cfg(cfg)
    paths = create_specialist_run_paths(
        root,
        subset_id,
        rounds,
        local_epochs,
        dr_lr,
        support_pseudo_count,
        model_tag,
        seed,
    )
    run_oracle_specialist_bank_experiment(
        subset_clients,
        cfg,
        seed,
        paths,
        rounds=rounds,
        warmup_rounds=warmup_rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        dr_lr=dr_lr,
        support_pseudo_count=support_pseudo_count,
        specialist_selection_view=specialist_selection_view,
        device=device,
        selection_metrics=selection_metrics,
        client_metadata=client_metadata,
    )
    return paths
