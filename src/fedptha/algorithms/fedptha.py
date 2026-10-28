from __future__ import annotations
from fedptha.training.runtime import RunPaths, create_run_paths
from fedptha.training.state import clone_state_dict
from fedptha.training.runtime import clone_model_from_state, make_task_head_model
from fedptha.algorithms.routing import (
    RoutingState,
    effective_task_weights,
    learned_task_weights,
    routing_matrix_cpu,
    learned_matrix_cpu,
    logits_matrix_cpu,
    support_matrix_cpu,
    reliance_matrix_cpu,
    backbone_matrix_cpu,
    initialize_routing,
    make_optimizer,
    fedavg_backbone_state_dict,
    load_backbone_state_into_models,
    mixed_parameter_dict,
    personalized_forward,
)
from fedptha.training.state import (
    core_state_dicts,
    make_p0_from_train_sizes,
    self_buffer_dict,
    personalized_call,
    _client_extra_metadata,
)
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd
import torch
from fedptha.training.runtime import (
    _subset_clients_str,
    append_eval_rows,
    append_train_row,
    build_model_context,
    build_runtime_clients,
    weighted_scalar_summary,
    is_better_selection_value,
    tag_recent_eval_rows,
)
from fedptha.algorithms.fedala import FedALAClientState, learn_ala_weights_for_client
from fedptha.data.clients import ClientData, format_seconds
from fedptha.training.steps import (
    augment_batch_x_with_ego,
    evaluate_loader,
    get_seed_label_mask,
    masked_loss_from_logits,
    unpack_batch_edges,
)
from fedptha.training.random import set_seed
from fedptha.data.tasks import TASKS


def _exp_algorithm() -> str:
    return os.environ.get("FEDPTHA_METHOD", "fedptha")


def _copy_ala_weights(
    weights: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not weights:
        return {}
    return {name: value.detach().cpu().clone() for (name, value) in weights.items()}


def _copy_ala_weights_by_client(
    weights_by_client: Optional[Dict[int, Dict[str, torch.Tensor]]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    if not weights_by_client:
        return {}
    return {
        int(idx): _copy_ala_weights(weights)
        for (idx, weights) in weights_by_client.items()
    }


def train_personalized_epoch(
    models_for_mix: Sequence[torch.nn.Module],
    client_idx: int,
    dr_vector: RoutingState,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    *,
    use_ego_ids: bool,
    ego_dim: int,
    ala_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    if getattr(dr_vector, "mode", "adaptive") == "learned":
        result = _learned_train_personalized_epoch(
            models_for_mix,
            client_idx,
            dr_vector,
            loader,
            optimizer,
            criterion,
            device,
            use_ego_ids=use_ego_ids,
            ego_dim=ego_dim,
            ala_weights=ala_weights,
        )
        return result
    if getattr(dr_vector, "mode", "adaptive") == "support":
        result = _support_train_personalized_epoch(
            models_for_mix,
            client_idx,
            dr_vector,
            loader,
            optimizer,
            criterion,
            device,
            use_ego_ids=use_ego_ids,
            ego_dim=ego_dim,
            ala_weights=ala_weights,
        )
        return result
    "Train local model + routing logits using ONLY the masked BCE objective.\n\n    Revised APPLE-PostALA has no DR proximal penalty, no lambda scheduler, and no\n    regularization toward p0. Visible task support enters persistently through the\n    routing definition itself.\n    "
    receiver_model = models_for_mix[client_idx]
    receiver_model.train()
    total_loss = 0.0
    total_count = 0
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx % 50 == 0:
            print("train-batch:", batch_idx)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        x_in, y_seed, B = augment_batch_x_with_ego(batch, use_ego_ids, ego_dim)
        edge_in, edge_attr_dict = unpack_batch_edges(batch)
        out = personalized_forward(
            models_for_mix,
            client_idx,
            dr_vector,
            x_in,
            edge_in,
            edge_attr_dict=edge_attr_dict,
            device=device,
            ala_weights=ala_weights,
        )
        out_seed = out[:B]
        label_mask_seed = get_seed_label_mask(batch, B)
        loss = masked_loss_from_logits(
            criterion, out_seed, y_seed.float(), label_mask_seed
        )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * B
        total_count += B
    denom = max(total_count, 1)
    mean_loss = total_loss / denom
    return {"loss": mean_loss, "base_loss": mean_loss, "prox_loss": 0.0}


@torch.no_grad()
def append_training_diagnostics(
    rows: List[Dict],
    *,
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: int,
    local_epoch: int,
    runtime,
    client_idx: int,
    train_stats: Dict[str, float],
    dr_vector: RoutingState,
    p0: torch.Tensor,
) -> None:
    if getattr(dr_vector, "mode", "adaptive") == "learned":
        result = _learned_append_training_diagnostics(
            rows,
            subset_id=subset_id,
            subset_clients=subset_clients,
            seed=seed,
            round_idx=round_idx,
            local_epoch=local_epoch,
            runtime=runtime,
            client_idx=client_idx,
            train_stats=train_stats,
            dr_vector=dr_vector,
            p0=p0,
        )
        return result
    if getattr(dr_vector, "mode", "adaptive") == "support":
        result = _support_append_training_diagnostics(
            rows,
            subset_id=subset_id,
            subset_clients=subset_clients,
            seed=seed,
            round_idx=round_idx,
            local_epoch=local_epoch,
            runtime=runtime,
            client_idx=client_idx,
            train_stats=train_stats,
            dr_vector=dr_vector,
            p0=p0,
        )
        return result
    "Training diagnostics for support-aware simplex APPLE-PostALA."
    task_cpu = effective_task_weights(dr_vector).detach().cpu()
    learned_cpu = learned_task_weights(dr_vector).detach().cpu()
    support_cpu = dr_vector.support.detach().cpu()
    rho_cpu = dr_vector.rho.detach().cpu()
    p0_cpu = p0.detach().cpu()
    task_self_weights = task_cpu[:, int(client_idx)]
    task_l2_to_p0 = torch.norm(task_cpu - p0_cpu.unsqueeze(0), p=2, dim=1)
    rows.append(
        {
            "run_type": _exp_algorithm(),
            "algorithm": _exp_algorithm(),
            "subset_id": subset_id,
            "subset_clients": subset_clients,
            "seed": seed,
            "graph_id": str(runtime.client.graph_id),
            "dataset_id": str(runtime.client.dataset_id),
            "phase": "apple_train_diag",
            "split": "train",
            "task": None,
            "round": round_idx,
            "local_epoch": local_epoch,
            "train_loss": float(train_stats["loss"]),
            "base_train_loss": float(train_stats["base_loss"]),
            "dr_prox_loss": 0.0,
            "dr_row_sum": float(task_cpu.sum(dim=1).mean().item()),
            "dr_l2_to_p0": float(task_l2_to_p0.mean().item()),
            "dr_self_weight": float(task_self_weights.mean().item()),
            "dr_min_weight": float(task_cpu.min().item()),
            "dr_max_weight": float(task_cpu.max().item()),
            "dr_rho_mean": float(rho_cpu.mean().item()),
            "dr_row_sum_by_task_json": json.dumps(
                [float(x) for x in task_cpu.sum(dim=1).tolist()]
            ),
            "dr_self_weight_by_task_json": json.dumps(
                [float(x) for x in task_self_weights.tolist()]
            ),
            "dr_l2_to_p0_by_task_json": json.dumps(
                [float(x) for x in task_l2_to_p0.tolist()]
            ),
            "dr_rho_by_task_json": json.dumps([float(x) for x in rho_cpu.tolist()]),
            "dr_effective_by_task_json": json.dumps(
                [[float(x) for x in row] for row in task_cpu.tolist()]
            ),
            "dr_learned_by_task_json": json.dumps(
                [[float(x) for x in row] for row in learned_cpu.tolist()]
            ),
            "dr_support_by_task_json": json.dumps(
                [[float(x) for x in row] for row in support_cpu.tolist()]
            ),
            "dr_backbone_row_sum": float(
                dr_vector.backbone.detach().cpu().sum().item()
            ),
            "dr_backbone_self_weight": float(
                dr_vector.backbone.detach().cpu()[int(client_idx)].item()
            ),
            "dr_backbone_l2_to_p0": float(
                torch.norm(dr_vector.backbone.detach().cpu() - p0_cpu, p=2).item()
            ),
            "num_nodes": int(runtime.num_train_nodes),
        }
    )


def append_mean_evaluation(
    rows: List[Dict],
    *,
    phase: str,
    split: str,
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: Optional[int],
    eval_infos: List[Dict],
    num_clients: Optional[int] = None,
    eval_mask_mode: str = "full",
    selection_protocol: Optional[str] = None,
    selected_by: Optional[str] = None,
) -> None:
    mean_scalar = weighted_scalar_summary(eval_infos)
    visible_pairs = int(
        sum((int(m["counts"].get("visible_pairs", 0)) for m in eval_infos))
    )
    total_pairs = int(sum((int(m["counts"].get("total_pairs", 0)) for m in eval_infos)))
    visible_pair_rate = float(visible_pairs / max(total_pairs, 1))
    rows.append(
        {
            "run_type": _exp_algorithm(),
            "algorithm": _exp_algorithm(),
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
            "train_loss": None,
            "eval_loss": mean_scalar["loss"],
            "pair_acc": mean_scalar["pair_acc"],
            "subset_acc": mean_scalar["subset_acc"],
            "micro_f1": mean_scalar["micro_f1"],
            "macro_f1": mean_scalar["macro_f1"],
            "macro_pos_f1": mean_scalar["macro_pos_f1"],
            "macro_minority_f1": mean_scalar["macro_minority_f1"],
            "num_nodes": sum((int(m["counts"]["num_nodes"]) for m in eval_infos)),
            "visible_pairs": visible_pairs,
            "total_pairs": total_pairs,
            "visible_pair_rate": visible_pair_rate,
            "eval_mask_mode": eval_mask_mode,
            "selection_protocol": selection_protocol,
            "selected_by": selected_by,
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
            "num_clients": num_clients,
        }
    )


def append_routing_diagnostics(
    dr_rows: List[Dict],
    *,
    runtime_clients,
    dr_vectors: Sequence[RoutingState],
    p0: torch.Tensor,
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: int,
    client_metadata: Optional[Dict[str, Dict]] = None,
) -> None:
    if getattr(dr_vectors[0], "mode", "adaptive") == "learned":
        result = _learned_append_routing_diagnostics(
            dr_rows,
            runtime_clients=runtime_clients,
            dr_vectors=dr_vectors,
            p0=p0,
            subset_id=subset_id,
            subset_clients=subset_clients,
            seed=seed,
            round_idx=round_idx,
            client_metadata=client_metadata,
        )
        return result
    if getattr(dr_vectors[0], "mode", "adaptive") == "support":
        result = _support_append_routing_diagnostics(
            dr_rows,
            runtime_clients=runtime_clients,
            dr_vectors=dr_vectors,
            p0=p0,
            subset_id=subset_id,
            subset_clients=subset_clients,
            seed=seed,
            round_idx=round_idx,
            client_metadata=client_metadata,
        )
        return result
    "Log effective p, learned A, support S, rho, and raw logits z."
    task_tensor = routing_matrix_cpu(dr_vectors)
    learned_tensor = learned_matrix_cpu(dr_vectors)
    logits_tensor = logits_matrix_cpu(dr_vectors)
    support_tensor = support_matrix_cpu(dr_vectors)
    rho_matrix = reliance_matrix_cpu(dr_vectors)
    backbone_matrix = backbone_matrix_cpu(dr_vectors)
    p0_cpu = p0.detach().cpu()
    num_receivers, num_tasks, num_donors = task_tensor.shape
    if num_receivers != len(runtime_clients) or num_donors != len(runtime_clients):
        raise ValueError(
            f"Task routing/client mismatch: tensor={tuple(task_tensor.shape)}, runtime_clients={len(runtime_clients)}"
        )
    for i, runtime_i in enumerate(runtime_clients):
        meta_i = _client_extra_metadata(runtime_i, client_metadata)
        brow = backbone_matrix[i]
        dr_rows.append(
            {
                "row_type": "backbone_summary",
                "run_type": _exp_algorithm(),
                "algorithm": _exp_algorithm(),
                "subset_id": subset_id,
                "subset_clients": subset_clients,
                "seed": seed,
                "round": round_idx,
                "client_idx": i,
                "graph_id": str(runtime_i.client.graph_id),
                "dataset_id": str(runtime_i.client.dataset_id),
                "family": meta_i.get("family"),
                "assigned_task": meta_i.get("assigned_task"),
                "task_idx": None,
                "task": "backbone",
                "dr_self_weight": float(brow[i].item()),
                "dr_row_sum": float(brow.sum().item()),
                "dr_row_abs_sum": float(brow.abs().sum().item()),
                "dr_l2_to_p0": float(torch.norm(brow - p0_cpu, p=2).item()),
                "dr_vector_json": json.dumps([float(x) for x in brow.tolist()]),
                "p0_json": json.dumps([float(x) for x in p0_cpu.tolist()]),
                "routing_mode": "fedavg_backbone",
            }
        )
        for j, runtime_j in enumerate(runtime_clients):
            meta_j = _client_extra_metadata(runtime_j, client_metadata)
            value = float(brow[j].item())
            dr_rows.append(
                {
                    "row_type": "backbone_pair",
                    "run_type": _exp_algorithm(),
                    "algorithm": _exp_algorithm(),
                    "subset_id": subset_id,
                    "subset_clients": subset_clients,
                    "seed": seed,
                    "round": round_idx,
                    "client_idx": i,
                    "donor_idx": j,
                    "graph_id": str(runtime_i.client.graph_id),
                    "dataset_id": str(runtime_i.client.dataset_id),
                    "family": meta_i.get("family"),
                    "assigned_task": meta_i.get("assigned_task"),
                    "task_idx": None,
                    "task": "backbone",
                    "donor_graph_id": str(runtime_j.client.graph_id),
                    "donor_dataset_id": str(runtime_j.client.dataset_id),
                    "donor_family": meta_j.get("family"),
                    "donor_assigned_task": meta_j.get("assigned_task"),
                    "p_itj": value,
                    "p_ij": value,
                    "p_backbone_ij": value,
                    "abs_p_itj": abs(value),
                    "p0_j": float(p0_cpu[j].item()),
                    "is_self": int(i == j),
                    "routing_mode": "fedavg_backbone",
                }
            )
        for task_idx in range(num_tasks):
            task_name = TASKS[task_idx] if task_idx < len(TASKS) else f"task{task_idx}"
            row = task_tensor[i, task_idx]
            learned_row = learned_tensor[i, task_idx]
            support_row = support_tensor[i, task_idx]
            logits_row = logits_tensor[i, task_idx]
            rho = float(rho_matrix[i, task_idx].item())
            if float(row.min().item()) < -1e-07:
                raise RuntimeError(
                    "Effective support-simplex routing produced a negative coefficient"
                )
            if abs(float(row.sum().item()) - 1.0) > 1e-05:
                raise RuntimeError(
                    "Effective support-simplex routing row does not sum to one"
                )
            top_order = torch.argsort(row, descending=True).tolist()
            top_parts = [
                f"{runtime_clients[j].client.graph_id}:{float(row[j].item()):.6g}"
                for j in top_order[: min(3, num_donors)]
            ]
            dr_rows.append(
                {
                    "row_type": "task_summary",
                    "run_type": _exp_algorithm(),
                    "algorithm": _exp_algorithm(),
                    "subset_id": subset_id,
                    "subset_clients": subset_clients,
                    "seed": seed,
                    "round": round_idx,
                    "client_idx": i,
                    "graph_id": str(runtime_i.client.graph_id),
                    "dataset_id": str(runtime_i.client.dataset_id),
                    "family": meta_i.get("family"),
                    "assigned_task": meta_i.get("assigned_task"),
                    "mask_task": meta_i.get("mask_task"),
                    "mask_fraction": meta_i.get("mask_fraction"),
                    "task_idx": task_idx,
                    "task": task_name,
                    "dr_self_weight": float(row[i].item()),
                    "dr_row_sum": float(row.sum().item()),
                    "dr_row_abs_sum": float(row.abs().sum().item()),
                    "dr_offdiag_pos_mass": float(row.sum().item() - row[i].item()),
                    "dr_offdiag_neg_mass": 0.0,
                    "dr_offdiag_abs_mass": float(row.sum().item() - row[i].item()),
                    "dr_l2_to_p0": float(torch.norm(row - p0_cpu, p=2).item()),
                    "dr_top_abs_donors": "|".join(top_parts),
                    "routing_rho": rho,
                    "routing_effective_min": float(row.min().item()),
                    "routing_effective_max": float(row.max().item()),
                    "routing_learned_entropy": float(
                        (
                            -(
                                learned_row.clamp_min(1e-12)
                                * learned_row.clamp_min(1e-12).log()
                            ).sum()
                        ).item()
                    ),
                    "routing_support_entropy": float(
                        (
                            -(
                                support_row.clamp_min(1e-12)
                                * support_row.clamp_min(1e-12).log()
                            ).sum()
                        ).item()
                    ),
                    "dr_vector_json": json.dumps([float(x) for x in row.tolist()]),
                    "learned_vector_json": json.dumps(
                        [float(x) for x in learned_row.tolist()]
                    ),
                    "support_vector_json": json.dumps(
                        [float(x) for x in support_row.tolist()]
                    ),
                    "logits_vector_json": json.dumps(
                        [float(x) for x in logits_row.tolist()]
                    ),
                    "p0_json": json.dumps([float(x) for x in p0_cpu.tolist()]),
                    "routing_mode": "adaptive_visible_support_simplex",
                }
            )
            for j, runtime_j in enumerate(runtime_clients):
                meta_j = _client_extra_metadata(runtime_j, client_metadata)
                value = float(row[j].item())
                dr_rows.append(
                    {
                        "row_type": "task_pair",
                        "run_type": _exp_algorithm(),
                        "algorithm": _exp_algorithm(),
                        "subset_id": subset_id,
                        "subset_clients": subset_clients,
                        "seed": seed,
                        "round": round_idx,
                        "client_idx": i,
                        "donor_idx": j,
                        "graph_id": str(runtime_i.client.graph_id),
                        "dataset_id": str(runtime_i.client.dataset_id),
                        "family": meta_i.get("family"),
                        "assigned_task": meta_i.get("assigned_task"),
                        "task_idx": task_idx,
                        "task": task_name,
                        "donor_graph_id": str(runtime_j.client.graph_id),
                        "donor_dataset_id": str(runtime_j.client.dataset_id),
                        "donor_family": meta_j.get("family"),
                        "donor_assigned_task": meta_j.get("assigned_task"),
                        "p_itj": value,
                        "p_ij": value,
                        "p_task_itj": value,
                        "abs_p_itj": abs(value),
                        "abs_p_ij": abs(value),
                        "abs_p_task_itj": abs(value),
                        "learned_softmax_itj": float(learned_row[j].item()),
                        "support_prior_tj": float(support_row[j].item()),
                        "routing_rho": rho,
                        "raw_logit_itj": float(logits_row[j].item()),
                        "is_negative_effective": int(value < 0.0),
                        "p0_j": float(p0_cpu[j].item()),
                        "is_self": int(i == j),
                        "routing_mode": "adaptive_visible_support_simplex",
                    }
                )


def personalized_state_dict(
    models_for_mix: Sequence[torch.nn.Module], client_idx: int, dr_vector: RoutingState
) -> Dict[str, torch.Tensor]:
    """Return the APPLE-personalized state used by the post-training ALA filter.

    The state is a full state_dict. Non-head entries/buffers come from the
    receiver model; task-head parameters are the current APPLE task-wise donor
    mixture without ALA gating.
    """
    receiver = models_for_mix[int(client_idx)]
    out = {
        key: value.detach().cpu().clone()
        for (key, value) in receiver.state_dict().items()
    }
    mixed_params = mixed_parameter_dict(
        models_for_mix, int(client_idx), dr_vector, ala_weights=None
    )
    for name, value in mixed_params.items():
        out[name] = value.detach().cpu().clone()
    return out


def append_refinement_diagnostics(
    rows: List[Dict],
    *,
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: int,
    runtime,
    client_idx: int,
    ala_stats: Dict[str, float],
) -> None:
    row = {
        "run_type": _exp_algorithm(),
        "algorithm": _exp_algorithm(),
        "subset_id": subset_id,
        "subset_clients": subset_clients,
        "seed": seed,
        "graph_id": str(runtime.client.graph_id),
        "dataset_id": str(runtime.client.dataset_id),
        "phase": "apple_head_ala_diag",
        "split": "train",
        "task": None,
        "round": round_idx,
        "local_epoch": None,
        "train_loss": None,
        "num_nodes": int(runtime.num_train_nodes),
        "client_idx": int(client_idx),
    }
    for key, value in ala_stats.items():
        row[key] = value
    rows.append(row)


def run_fedptha_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    dr_lr: float,
    support_pseudo_count: float,
    device: torch.device,
    client_metadata: Optional[Dict[str, Dict]] = None,
) -> None:
    if not subset_clients:
        raise ValueError("subset_clients is empty")
    set_seed(seed)
    subset_id = _subset_clients_str(subset_clients)
    subset_clients_str = subset_id
    ctx = build_model_context(subset_clients, cfg)
    runtime_clients = build_runtime_clients(subset_clients, cfg, device)
    num_clients = len(runtime_clients)
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
    core_models = [
        clone_model_from_state(initial_state, cfg, ctx, device)
        for _ in range(num_clients)
    ]
    p0 = make_p0_from_train_sizes(runtime_clients, device)
    num_tasks = int(ctx["out_dim"])
    use_head_ala = bool(cfg.get("apple_use_head_ala", False))
    ala_lr = float(cfg.get("apple_ala_lr", 1.0))
    ala_rand_percent = float(cfg.get("apple_ala_rand_percent", 20.0))
    ala_convergence_std = float(cfg.get("apple_ala_convergence_std", 0.1))
    ala_convergence_window = int(cfg.get("apple_ala_convergence_window", 10))
    ala_max_steps = int(cfg.get("apple_ala_max_steps", 100))
    ala_debug = bool(cfg.get("apple_ala_debug", False))
    dr_vectors = initialize_routing(
        num_clients,
        num_tasks,
        p0,
        runtime_clients=runtime_clients,
        device=device,
        support_pseudo_count=support_pseudo_count,
        routing_mode=cfg.get("routing_mode", "adaptive"),
    )
    ala_states = [FedALAClientState() for _ in range(num_clients)]
    current_ala_weights_by_client: Dict[int, Dict[str, torch.Tensor]] = {
        i: {} for i in range(num_clients)
    }
    print(
        "[fedavg-backbone+personalized task heads]",
        "num_clients=",
        num_clients,
        "num_tasks=",
        num_tasks,
        "output_head=",
        cfg.get("output_head"),
        "mixing_mode=",
        cfg.get("apple_mixing_mode"),
        "task_dr_tensor_shape=",
        tuple(routing_matrix_cpu(dr_vectors).shape),
        "backbone=fedavg",
        "postala_refinement=",
        use_head_ala,
        "routing=adaptive_visible_support_simplex",
        "support_pseudo_count=",
        support_pseudo_count,
    )
    rows: List[Dict] = []
    dr_rows: List[Dict] = []
    initial_local_eval_states_by_client = {
        i: core_state_dicts(core_models) for i in range(num_clients)
    }
    initial_dr_matrix = routing_matrix_cpu(dr_vectors)
    initial_backbone_dr_matrix = backbone_matrix_cpu(dr_vectors)

    def _initial_best_record():
        record = {
            "value": None,
            "round": -1,
            "local_eval_states_by_client": copy.deepcopy(
                initial_local_eval_states_by_client
            ),
            "dr_matrix": initial_dr_matrix.clone(),
            "backbone_dr_matrix": initial_backbone_dr_matrix.clone(),
            "ala_weights_by_client": {i: {} for i in range(num_clients)},
        }
        return record

    best_by_eval_mode = {
        eval_mode: _initial_best_record() for eval_mode in ("full", "visible")
    }
    generator = torch.Generator().manual_seed(seed)
    for round_idx in range(1, rounds + 1):
        run_start_time = time.perf_counter()
        num_selected = max(1, int(round(client_fraction * num_clients)))
        if num_selected >= num_clients:
            selected_indices = list(range(num_clients))
        else:
            permutation = torch.randperm(num_clients, generator=generator).tolist()
            selected_indices = permutation[:num_selected]
        round_start_states = core_state_dicts(core_models)
        downloaded_core_models = [
            clone_model_from_state(state, cfg, ctx, device)
            for state in round_start_states
        ]
        updated_models: Dict[int, torch.nn.Module] = {}
        round_local_eval_infos_by_mode: Dict[str, List[Dict]] = {
            "full": [],
            "visible": [],
        }
        round_local_eval_states_by_client: Dict[int, List[Dict[str, torch.Tensor]]] = {
            i: core_state_dicts(downloaded_core_models) for i in range(num_clients)
        }
        round_local_eval_ala_weights_by_client: Dict[int, Dict[str, torch.Tensor]] = (
            _copy_ala_weights_by_client(current_ala_weights_by_client)
        )
        for idx in selected_indices:
            runtime = runtime_clients[idx]
            local_model = clone_model_from_state(
                round_start_states[idx], cfg, ctx, device
            )
            models_for_mix = list(downloaded_core_models)
            models_for_mix[idx] = local_model
            ala_weights_for_idx = None
            optimizer = make_optimizer(local_model, dr_vectors[idx], cfg, dr_lr)
            for local_epoch_idx in range(1, local_epochs + 1):
                train_stats = train_personalized_epoch(
                    models_for_mix,
                    idx,
                    dr_vectors[idx],
                    runtime.train_loader,
                    optimizer,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    ala_weights=ala_weights_for_idx,
                )
                append_train_row(
                    rows,
                    run_type=_exp_algorithm(),
                    algorithm=_exp_algorithm(),
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    phase="train_epoch",
                    split="train",
                    train_loss=train_stats["loss"],
                    num_nodes=runtime.num_train_nodes,
                    round_idx=round_idx,
                    local_epoch=local_epoch_idx,
                )
                append_training_diagnostics(
                    rows,
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    round_idx=round_idx,
                    local_epoch=local_epoch_idx,
                    runtime=runtime,
                    client_idx=idx,
                    train_stats=train_stats,
                    dr_vector=dr_vectors[idx],
                    p0=p0,
                )
            trained_local_state = clone_state_dict(local_model.state_dict())
            incoming_state = personalized_state_dict(
                models_for_mix, idx, dr_vectors[idx]
            )
            if use_head_ala:
                ala_filtered_state, ala_stats = learn_ala_weights_for_client(
                    client_idx=idx,
                    runtime=runtime,
                    cfg=cfg,
                    ctx=ctx,
                    client_state=ala_states[idx],
                    old_local_state=trained_local_state,
                    global_state=incoming_state,
                    mode="head_only",
                    ala_lr=ala_lr,
                    ala_rand_percent=ala_rand_percent,
                    ala_convergence_std=ala_convergence_std,
                    ala_convergence_window=ala_convergence_window,
                    ala_max_steps=ala_max_steps,
                    device=device,
                    round_idx=round_idx,
                    debug=ala_debug,
                )
                local_model.load_state_dict(
                    {k: v.to(device=device) for (k, v) in ala_filtered_state.items()},
                    strict=True,
                )
                models_for_mix[idx] = local_model
                ala_weights_for_idx = _copy_ala_weights(ala_states[idx].weights)
                current_ala_weights_by_client[idx] = _copy_ala_weights(
                    ala_states[idx].weights
                )
                round_local_eval_ala_weights_by_client[idx] = _copy_ala_weights(
                    ala_states[idx].weights
                )
                append_refinement_diagnostics(
                    rows,
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    round_idx=round_idx,
                    runtime=runtime,
                    client_idx=idx,
                    ala_stats=ala_stats,
                )
            else:
                local_model.load_state_dict(
                    {k: v.to(device=device) for (k, v) in incoming_state.items()},
                    strict=True,
                )
                models_for_mix[idx] = local_model
                ala_weights_for_idx = None
            round_local_eval_states_by_client[idx] = core_state_dicts(models_for_mix)
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
                round_local_eval_infos_by_mode[eval_mask_mode].append(post_metrics)
                append_eval_rows(
                    rows,
                    run_type=_exp_algorithm(),
                    algorithm=_exp_algorithm(),
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
            updated_models[idx] = local_model
        for eval_mask_mode in ("full", "visible"):
            append_mean_evaluation(
                rows,
                phase=f"apple_local_val_mean_{eval_mask_mode}",
                split="val",
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                round_idx=round_idx,
                eval_infos=round_local_eval_infos_by_mode[eval_mask_mode],
                num_clients=num_clients,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )
        for idx, updated_model in updated_models.items():
            core_models[idx] = updated_model
        selected_models_for_backbone = [core_models[idx] for idx in selected_indices]
        selected_runtimes_for_backbone = [
            runtime_clients[idx] for idx in selected_indices
        ]
        fedavg_backbone_state = fedavg_backbone_state_dict(
            selected_models_for_backbone, selected_runtimes_for_backbone
        )
        load_backbone_state_into_models(core_models, fedavg_backbone_state)
        round_eval_infos_by_mode: Dict[str, List[Dict]] = {"full": [], "visible": []}
        for eval_mask_mode in ("full", "visible"):
            for idx, runtime in enumerate(runtime_clients):
                metrics = evaluate_loader(
                    core_models[idx],
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                round_eval_infos_by_mode[eval_mask_mode].append(metrics)
                print("apple eval on client:", runtime.client.graph_id)
                print(
                    "apple round:",
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
                    run_type=_exp_algorithm(),
                    algorithm=_exp_algorithm(),
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"apple_val_client_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )
            append_mean_evaluation(
                rows,
                phase=f"apple_val_mean_{eval_mask_mode}",
                split="val",
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                round_idx=round_idx,
                eval_infos=round_eval_infos_by_mode[eval_mask_mode],
                num_clients=num_clients,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )
        append_routing_diagnostics(
            dr_rows,
            runtime_clients=runtime_clients,
            dr_vectors=dr_vectors,
            p0=p0,
            subset_id=subset_id,
            subset_clients=subset_clients_str,
            seed=seed,
            round_idx=round_idx,
            client_metadata=client_metadata,
        )
        for eval_mask_mode in ("full", "visible"):
            local_mean_scalar = weighted_scalar_summary(
                round_local_eval_infos_by_mode[eval_mask_mode]
            )
            current_value = local_mean_scalar.get("macro_pr_auc")
            best_record = best_by_eval_mode[eval_mask_mode]
            print(
                "current local-update APPLE vs best:",
                "eval_mask_mode:",
                eval_mask_mode,
                "selector_metric:",
                "macro_pr_auc",
                "current:",
                current_value,
                "best:",
                best_record["value"],
            )
            if is_better_selection_value(
                current_value,
                float("-inf") if best_record["value"] is None else best_record["value"],
            ):
                best_record["value"] = float(current_value)
                best_record["round"] = round_idx
                best_record["local_eval_states_by_client"] = copy.deepcopy(
                    round_local_eval_states_by_client
                )
                best_record["dr_matrix"] = routing_matrix_cpu(dr_vectors)
                best_record["backbone_dr_matrix"] = backbone_matrix_cpu(dr_vectors)
                best_record["ala_weights_by_client"] = _copy_ala_weights_by_client(
                    round_local_eval_ala_weights_by_client
                )
        run_elapsed = time.perf_counter() - run_start_time
        print(f"run time: {format_seconds(run_elapsed)}")
        print("=======================================================")
        print(f"==================== round {round_idx} / {rounds} ====================")
        print("=======================================================")
        del downloaded_core_models
    diagnostic_outputs: Dict[str, str] = {}
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
        selected_state = best_by_eval_mode[selected_by]
        selected_round = selected_state["round"]
        selected_dr_matrix = selected_state["dr_matrix"]
        selected_ala_weights_by_client = selected_state.get("ala_weights_by_client", {})
        if "backbone_dr_matrix" in selected_state:
            selected_backbone_dr_matrix = selected_state["backbone_dr_matrix"]
        else:
            selected_backbone_dr_matrix = torch.stack(
                [p0.detach().cpu().clone() for _ in range(selected_dr_matrix.size(0))],
                dim=0,
            )
        for split_name in job["splits"]:
            split_eval_infos = []
            for idx, runtime in enumerate(runtime_clients):
                split_loader = {
                    "train": runtime.train_loader,
                    "val": runtime.val_loader,
                    "test": runtime.test_loader,
                }[split_name]
                best_model = clone_model_from_state(
                    selected_state["local_eval_states_by_client"][idx][idx],
                    cfg,
                    ctx,
                    device,
                )
                metrics = evaluate_loader(
                    best_model,
                    split_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                split_eval_infos.append(metrics)
                append_eval_rows(
                    rows,
                    run_type=_exp_algorithm(),
                    algorithm=_exp_algorithm(),
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"best_{selection_protocol}_macro_pr_auc_{split_name}",
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
            append_mean_evaluation(
                rows,
                phase=f"best_{selection_protocol}_macro_pr_auc_{split_name}_mean",
                split=split_name,
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                round_idx=selected_round,
                eval_infos=split_eval_infos,
                num_clients=num_clients,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=selection_protocol,
                selected_by=selected_by,
            )
            rows[-1].update(
                {
                    "mean_selection_protocol": selection_protocol,
                    "selection_protocol": None,
                    "selected_by": None,
                    "selection_metrics": "macro_pr_auc",
                    "selector_metric": "macro_pr_auc",
                    "selector_direction": "max",
                    "selected_by_eval_mode": selected_by,
                    "selected_round": selected_round,
                    "selected_epoch": None,
                    "best_val_metric_value": selected_state["value"],
                }
            )
    out_df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("APPLE EXPERIMENTAL DUAL-EVALUATION SMOKE CHECK")
    print("=" * 100)
    print("csv path:", run_paths.csv_path)
    print(
        "best rounds by eval mode and selector:",
        {mode: best_by_eval_mode[mode]["round"] for mode in ("full", "visible")},
    )
    print(
        "best values by eval mode and selector:",
        {mode: best_by_eval_mode[mode]["value"] for mode in ("full", "visible")},
    )
    final_rows = out_df[
        out_df["task"].isna()
        & (out_df["split"] == "test")
        & out_df["selection_protocol"].isin(
            ["oracle_full", "realistic_visible", "realistic_selection_oracle"]
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
    pd.DataFrame(dr_rows).to_csv(run_paths.dr_csv_path, index=False)
    print(f"saved csv -> {run_paths.csv_path}")
    print(f"saved DR csv -> {run_paths.dr_csv_path}")
    if cfg.get("return_selected_states", False):
        hierarchy_selected_by_eval_mode = {}
        for hierarchy_mode in ("visible", "full"):
            hierarchy_record = best_by_eval_mode[hierarchy_mode]
            hierarchy_selected_by_eval_mode[hierarchy_mode] = {
                "round": int(hierarchy_record["round"]),
                "value": float(hierarchy_record["value"]),
                "client_states": [
                    copy.deepcopy(hierarchy_record["local_eval_states_by_client"][i][i])
                    for i in range(num_clients)
                ],
            }
        return {
            "selected_by_eval_mode": hierarchy_selected_by_eval_mode,
            "runtime_clients": runtime_clients,
            "ctx": ctx,
            "cfg": cfg,
            "subset_id": subset_id,
            "subset_clients": subset_clients_str,
            "seed": seed,
            "num_tasks": num_tasks,
            "support_pseudo_count": support_pseudo_count,
        }


def run_fedptha(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    apple_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    dr_lr: float,
    support_pseudo_count: float = 1.0,
    device: torch.device,
    client_metadata: Optional[Dict[str, Dict]] = None,
) -> RunPaths:
    subset_id = _subset_clients_str(subset_clients)
    run_paths = create_run_paths(apple_root)
    if run_paths.csv_path.exists() and run_paths.dr_csv_path.exists():
        return run_paths
    run_fedptha_experiment(
        subset_clients,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        dr_lr=dr_lr,
        support_pseudo_count=support_pseudo_count,
        device=device,
        client_metadata=client_metadata,
    )
    return run_paths


def _learned_train_personalized_epoch(
    models_for_mix: Sequence[torch.nn.Module],
    client_idx: int,
    dr_vector: RoutingState,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    *,
    use_ego_ids: bool,
    ego_dim: int,
    ala_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Train local model + routing logits using ONLY the masked BCE objective.

    Learned-Simplex + PostALA has no DR proximal penalty, no lambda scheduler, and no
    regularization toward p0. Task routing is learned through the
    routing definition itself.
    """
    receiver_model = models_for_mix[client_idx]
    receiver_model.train()
    total_loss = 0.0
    total_count = 0
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx % 50 == 0:
            print("train-batch:", batch_idx)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        x_in, y_seed, B = augment_batch_x_with_ego(batch, use_ego_ids, ego_dim)
        edge_in, edge_attr_dict = unpack_batch_edges(batch)
        out = personalized_forward(
            models_for_mix,
            client_idx,
            dr_vector,
            x_in,
            edge_in,
            edge_attr_dict=edge_attr_dict,
            device=device,
            ala_weights=ala_weights,
        )
        out_seed = out[:B]
        label_mask_seed = get_seed_label_mask(batch, B)
        loss = masked_loss_from_logits(
            criterion, out_seed, y_seed.float(), label_mask_seed
        )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * B
        total_count += B
    denom = max(total_count, 1)
    mean_loss = total_loss / denom
    return {"loss": mean_loss, "base_loss": mean_loss, "prox_loss": 0.0}


def _support_train_personalized_epoch(
    models_for_mix: Sequence[torch.nn.Module],
    client_idx: int,
    dr_vector: RoutingState,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    *,
    use_ego_ids: bool,
    ego_dim: int,
    ala_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Train local model + routing logits using ONLY the masked BCE objective.

    Support-Only + PostALA has no learned routing, DR proximal penalty, or lambda
    scheduler. Visible task support is the fixed task-head donor distribution.
    """
    receiver_model = models_for_mix[client_idx]
    receiver_model.train()
    total_loss = 0.0
    total_count = 0
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx % 50 == 0:
            print("train-batch:", batch_idx)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        x_in, y_seed, B = augment_batch_x_with_ego(batch, use_ego_ids, ego_dim)
        edge_in, edge_attr_dict = unpack_batch_edges(batch)
        out = personalized_forward(
            models_for_mix,
            client_idx,
            dr_vector,
            x_in,
            edge_in,
            edge_attr_dict=edge_attr_dict,
            device=device,
            ala_weights=ala_weights,
        )
        out_seed = out[:B]
        label_mask_seed = get_seed_label_mask(batch, B)
        loss = masked_loss_from_logits(
            criterion, out_seed, y_seed.float(), label_mask_seed
        )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * B
        total_count += B
    denom = max(total_count, 1)
    mean_loss = total_loss / denom
    return {"loss": mean_loss, "base_loss": mean_loss, "prox_loss": 0.0}


def _learned_append_training_diagnostics(
    rows: List[Dict],
    *,
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: int,
    local_epoch: int,
    runtime,
    client_idx: int,
    train_stats: Dict[str, float],
    dr_vector: RoutingState,
    p0: torch.Tensor,
) -> None:
    """Training diagnostics for Learned-Simplex + PostALA."""
    task_cpu = effective_task_weights(dr_vector).detach().cpu()
    learned_cpu = learned_task_weights(dr_vector).detach().cpu()
    support_cpu = dr_vector.support.detach().cpu()
    rho_cpu = dr_vector.rho.detach().cpu()
    p0_cpu = p0.detach().cpu()
    task_self_weights = task_cpu[:, int(client_idx)]
    task_l2_to_p0 = torch.norm(task_cpu - p0_cpu.unsqueeze(0), p=2, dim=1)
    rows.append(
        {
            "run_type": _exp_algorithm(),
            "algorithm": _exp_algorithm(),
            "subset_id": subset_id,
            "subset_clients": subset_clients,
            "seed": seed,
            "graph_id": str(runtime.client.graph_id),
            "dataset_id": str(runtime.client.dataset_id),
            "phase": "apple_train_diag",
            "split": "train",
            "task": None,
            "round": round_idx,
            "local_epoch": local_epoch,
            "train_loss": float(train_stats["loss"]),
            "base_train_loss": float(train_stats["base_loss"]),
            "dr_prox_loss": 0.0,
            "dr_row_sum": float(task_cpu.sum(dim=1).mean().item()),
            "dr_l2_to_p0": float(task_l2_to_p0.mean().item()),
            "dr_self_weight": float(task_self_weights.mean().item()),
            "dr_min_weight": float(task_cpu.min().item()),
            "dr_max_weight": float(task_cpu.max().item()),
            "dr_rho_mean": float(rho_cpu.mean().item()),
            "dr_row_sum_by_task_json": json.dumps(
                [float(x) for x in task_cpu.sum(dim=1).tolist()]
            ),
            "dr_self_weight_by_task_json": json.dumps(
                [float(x) for x in task_self_weights.tolist()]
            ),
            "dr_l2_to_p0_by_task_json": json.dumps(
                [float(x) for x in task_l2_to_p0.tolist()]
            ),
            "dr_rho_by_task_json": json.dumps([float(x) for x in rho_cpu.tolist()]),
            "dr_effective_by_task_json": json.dumps(
                [[float(x) for x in row] for row in task_cpu.tolist()]
            ),
            "dr_learned_by_task_json": json.dumps(
                [[float(x) for x in row] for row in learned_cpu.tolist()]
            ),
            "dr_support_by_task_json": json.dumps(
                [[float(x) for x in row] for row in support_cpu.tolist()]
            ),
            "dr_backbone_row_sum": float(
                dr_vector.backbone.detach().cpu().sum().item()
            ),
            "dr_backbone_self_weight": float(
                dr_vector.backbone.detach().cpu()[int(client_idx)].item()
            ),
            "dr_backbone_l2_to_p0": float(
                torch.norm(dr_vector.backbone.detach().cpu() - p0_cpu, p=2).item()
            ),
            "num_nodes": int(runtime.num_train_nodes),
        }
    )


def _support_append_training_diagnostics(
    rows: List[Dict],
    *,
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: int,
    local_epoch: int,
    runtime,
    client_idx: int,
    train_stats: Dict[str, float],
    dr_vector: RoutingState,
    p0: torch.Tensor,
) -> None:
    """Training diagnostics for support-only simplex + PostALA."""
    task_cpu = effective_task_weights(dr_vector).detach().cpu()
    learned_cpu = learned_task_weights(dr_vector).detach().cpu()
    support_cpu = dr_vector.support.detach().cpu()
    rho_cpu = dr_vector.rho.detach().cpu()
    p0_cpu = p0.detach().cpu()
    task_self_weights = task_cpu[:, int(client_idx)]
    task_l2_to_p0 = torch.norm(task_cpu - p0_cpu.unsqueeze(0), p=2, dim=1)
    rows.append(
        {
            "run_type": _exp_algorithm(),
            "algorithm": _exp_algorithm(),
            "subset_id": subset_id,
            "subset_clients": subset_clients,
            "seed": seed,
            "graph_id": str(runtime.client.graph_id),
            "dataset_id": str(runtime.client.dataset_id),
            "phase": "apple_train_diag",
            "split": "train",
            "task": None,
            "round": round_idx,
            "local_epoch": local_epoch,
            "train_loss": float(train_stats["loss"]),
            "base_train_loss": float(train_stats["base_loss"]),
            "dr_prox_loss": 0.0,
            "dr_row_sum": float(task_cpu.sum(dim=1).mean().item()),
            "dr_l2_to_p0": float(task_l2_to_p0.mean().item()),
            "dr_self_weight": float(task_self_weights.mean().item()),
            "dr_min_weight": float(task_cpu.min().item()),
            "dr_max_weight": float(task_cpu.max().item()),
            "dr_rho_mean": float(rho_cpu.mean().item()),
            "dr_row_sum_by_task_json": json.dumps(
                [float(x) for x in task_cpu.sum(dim=1).tolist()]
            ),
            "dr_self_weight_by_task_json": json.dumps(
                [float(x) for x in task_self_weights.tolist()]
            ),
            "dr_l2_to_p0_by_task_json": json.dumps(
                [float(x) for x in task_l2_to_p0.tolist()]
            ),
            "dr_rho_by_task_json": json.dumps([float(x) for x in rho_cpu.tolist()]),
            "dr_effective_by_task_json": json.dumps(
                [[float(x) for x in row] for row in task_cpu.tolist()]
            ),
            "dr_learned_by_task_json": json.dumps(
                [[float(x) for x in row] for row in learned_cpu.tolist()]
            ),
            "dr_support_by_task_json": json.dumps(
                [[float(x) for x in row] for row in support_cpu.tolist()]
            ),
            "dr_backbone_row_sum": float(
                dr_vector.backbone.detach().cpu().sum().item()
            ),
            "dr_backbone_self_weight": float(
                dr_vector.backbone.detach().cpu()[int(client_idx)].item()
            ),
            "dr_backbone_l2_to_p0": float(
                torch.norm(dr_vector.backbone.detach().cpu() - p0_cpu, p=2).item()
            ),
            "num_nodes": int(runtime.num_train_nodes),
        }
    )


def _learned_append_routing_diagnostics(
    dr_rows: List[Dict],
    *,
    runtime_clients,
    dr_vectors: Sequence[RoutingState],
    p0: torch.Tensor,
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: int,
    client_metadata: Optional[Dict[str, Dict]] = None,
) -> None:
    """Log effective p=softmax(z), learned A, disabled-support placeholders, and raw logits z."""
    task_tensor = routing_matrix_cpu(dr_vectors)
    learned_tensor = learned_matrix_cpu(dr_vectors)
    logits_tensor = logits_matrix_cpu(dr_vectors)
    support_tensor = support_matrix_cpu(dr_vectors)
    rho_matrix = reliance_matrix_cpu(dr_vectors)
    backbone_matrix = backbone_matrix_cpu(dr_vectors)
    p0_cpu = p0.detach().cpu()
    num_receivers, num_tasks, num_donors = task_tensor.shape
    if num_receivers != len(runtime_clients) or num_donors != len(runtime_clients):
        raise ValueError(
            f"Task routing/client mismatch: tensor={tuple(task_tensor.shape)}, runtime_clients={len(runtime_clients)}"
        )
    for i, runtime_i in enumerate(runtime_clients):
        meta_i = _client_extra_metadata(runtime_i, client_metadata)
        brow = backbone_matrix[i]
        dr_rows.append(
            {
                "row_type": "backbone_summary",
                "run_type": _exp_algorithm(),
                "algorithm": _exp_algorithm(),
                "subset_id": subset_id,
                "subset_clients": subset_clients,
                "seed": seed,
                "round": round_idx,
                "client_idx": i,
                "graph_id": str(runtime_i.client.graph_id),
                "dataset_id": str(runtime_i.client.dataset_id),
                "family": meta_i.get("family"),
                "assigned_task": meta_i.get("assigned_task"),
                "task_idx": None,
                "task": "backbone",
                "dr_self_weight": float(brow[i].item()),
                "dr_row_sum": float(brow.sum().item()),
                "dr_row_abs_sum": float(brow.abs().sum().item()),
                "dr_l2_to_p0": float(torch.norm(brow - p0_cpu, p=2).item()),
                "dr_vector_json": json.dumps([float(x) for x in brow.tolist()]),
                "p0_json": json.dumps([float(x) for x in p0_cpu.tolist()]),
                "routing_mode": "fedavg_backbone",
            }
        )
        for j, runtime_j in enumerate(runtime_clients):
            meta_j = _client_extra_metadata(runtime_j, client_metadata)
            value = float(brow[j].item())
            dr_rows.append(
                {
                    "row_type": "backbone_pair",
                    "run_type": _exp_algorithm(),
                    "algorithm": _exp_algorithm(),
                    "subset_id": subset_id,
                    "subset_clients": subset_clients,
                    "seed": seed,
                    "round": round_idx,
                    "client_idx": i,
                    "donor_idx": j,
                    "graph_id": str(runtime_i.client.graph_id),
                    "dataset_id": str(runtime_i.client.dataset_id),
                    "family": meta_i.get("family"),
                    "assigned_task": meta_i.get("assigned_task"),
                    "task_idx": None,
                    "task": "backbone",
                    "donor_graph_id": str(runtime_j.client.graph_id),
                    "donor_dataset_id": str(runtime_j.client.dataset_id),
                    "donor_family": meta_j.get("family"),
                    "donor_assigned_task": meta_j.get("assigned_task"),
                    "p_itj": value,
                    "p_ij": value,
                    "p_backbone_ij": value,
                    "abs_p_itj": abs(value),
                    "p0_j": float(p0_cpu[j].item()),
                    "is_self": int(i == j),
                    "routing_mode": "fedavg_backbone",
                }
            )
        for task_idx in range(num_tasks):
            task_name = TASKS[task_idx] if task_idx < len(TASKS) else f"task{task_idx}"
            row = task_tensor[i, task_idx]
            learned_row = learned_tensor[i, task_idx]
            support_row = support_tensor[i, task_idx]
            logits_row = logits_tensor[i, task_idx]
            rho = float(rho_matrix[i, task_idx].item())
            if float(row.min().item()) < -1e-07:
                raise RuntimeError(
                    "Effective learned-simplex routing produced a negative coefficient"
                )
            if abs(float(row.sum().item()) - 1.0) > 1e-05:
                raise RuntimeError(
                    "Effective learned-simplex routing row does not sum to one"
                )
            top_order = torch.argsort(row, descending=True).tolist()
            top_parts = [
                f"{runtime_clients[j].client.graph_id}:{float(row[j].item()):.6g}"
                for j in top_order[: min(3, num_donors)]
            ]
            dr_rows.append(
                {
                    "row_type": "task_summary",
                    "run_type": _exp_algorithm(),
                    "algorithm": _exp_algorithm(),
                    "subset_id": subset_id,
                    "subset_clients": subset_clients,
                    "seed": seed,
                    "round": round_idx,
                    "client_idx": i,
                    "graph_id": str(runtime_i.client.graph_id),
                    "dataset_id": str(runtime_i.client.dataset_id),
                    "family": meta_i.get("family"),
                    "assigned_task": meta_i.get("assigned_task"),
                    "mask_task": meta_i.get("mask_task"),
                    "mask_fraction": meta_i.get("mask_fraction"),
                    "task_idx": task_idx,
                    "task": task_name,
                    "dr_self_weight": float(row[i].item()),
                    "dr_row_sum": float(row.sum().item()),
                    "dr_row_abs_sum": float(row.abs().sum().item()),
                    "dr_offdiag_pos_mass": float(row.sum().item() - row[i].item()),
                    "dr_offdiag_neg_mass": 0.0,
                    "dr_offdiag_abs_mass": float(row.sum().item() - row[i].item()),
                    "dr_l2_to_p0": float(torch.norm(row - p0_cpu, p=2).item()),
                    "dr_top_abs_donors": "|".join(top_parts),
                    "routing_rho": rho,
                    "routing_effective_min": float(row.min().item()),
                    "routing_effective_max": float(row.max().item()),
                    "routing_learned_entropy": float(
                        (
                            -(
                                learned_row.clamp_min(1e-12)
                                * learned_row.clamp_min(1e-12).log()
                            ).sum()
                        ).item()
                    ),
                    "routing_support_entropy": float(
                        (
                            -(
                                support_row.clamp_min(1e-12)
                                * support_row.clamp_min(1e-12).log()
                            ).sum()
                        ).item()
                    ),
                    "dr_vector_json": json.dumps([float(x) for x in row.tolist()]),
                    "learned_vector_json": json.dumps(
                        [float(x) for x in learned_row.tolist()]
                    ),
                    "support_vector_json": json.dumps(
                        [float(x) for x in support_row.tolist()]
                    ),
                    "logits_vector_json": json.dumps(
                        [float(x) for x in logits_row.tolist()]
                    ),
                    "p0_json": json.dumps([float(x) for x in p0_cpu.tolist()]),
                    "routing_mode": "learned_simplex_softmax",
                }
            )
            for j, runtime_j in enumerate(runtime_clients):
                meta_j = _client_extra_metadata(runtime_j, client_metadata)
                value = float(row[j].item())
                dr_rows.append(
                    {
                        "row_type": "task_pair",
                        "run_type": _exp_algorithm(),
                        "algorithm": _exp_algorithm(),
                        "subset_id": subset_id,
                        "subset_clients": subset_clients,
                        "seed": seed,
                        "round": round_idx,
                        "client_idx": i,
                        "donor_idx": j,
                        "graph_id": str(runtime_i.client.graph_id),
                        "dataset_id": str(runtime_i.client.dataset_id),
                        "family": meta_i.get("family"),
                        "assigned_task": meta_i.get("assigned_task"),
                        "task_idx": task_idx,
                        "task": task_name,
                        "donor_graph_id": str(runtime_j.client.graph_id),
                        "donor_dataset_id": str(runtime_j.client.dataset_id),
                        "donor_family": meta_j.get("family"),
                        "donor_assigned_task": meta_j.get("assigned_task"),
                        "p_itj": value,
                        "p_ij": value,
                        "p_task_itj": value,
                        "abs_p_itj": abs(value),
                        "abs_p_ij": abs(value),
                        "abs_p_task_itj": abs(value),
                        "learned_softmax_itj": float(learned_row[j].item()),
                        "support_prior_tj": float(support_row[j].item()),
                        "routing_rho": rho,
                        "raw_logit_itj": float(logits_row[j].item()),
                        "is_negative_effective": int(value < 0.0),
                        "p0_j": float(p0_cpu[j].item()),
                        "is_self": int(i == j),
                        "routing_mode": "learned_simplex_softmax",
                    }
                )


def _support_append_routing_diagnostics(
    dr_rows: List[Dict],
    *,
    runtime_clients,
    dr_vectors: Sequence[RoutingState],
    p0: torch.Tensor,
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: int,
    client_metadata: Optional[Dict[str, Dict]] = None,
) -> None:
    """Log effective p, learned A, support S, rho, and raw logits z."""
    task_tensor = routing_matrix_cpu(dr_vectors)
    learned_tensor = learned_matrix_cpu(dr_vectors)
    logits_tensor = logits_matrix_cpu(dr_vectors)
    support_tensor = support_matrix_cpu(dr_vectors)
    rho_matrix = reliance_matrix_cpu(dr_vectors)
    backbone_matrix = backbone_matrix_cpu(dr_vectors)
    p0_cpu = p0.detach().cpu()
    num_receivers, num_tasks, num_donors = task_tensor.shape
    if num_receivers != len(runtime_clients) or num_donors != len(runtime_clients):
        raise ValueError(
            f"Task routing/client mismatch: tensor={tuple(task_tensor.shape)}, runtime_clients={len(runtime_clients)}"
        )
    for i, runtime_i in enumerate(runtime_clients):
        meta_i = _client_extra_metadata(runtime_i, client_metadata)
        brow = backbone_matrix[i]
        dr_rows.append(
            {
                "row_type": "backbone_summary",
                "run_type": _exp_algorithm(),
                "algorithm": _exp_algorithm(),
                "subset_id": subset_id,
                "subset_clients": subset_clients,
                "seed": seed,
                "round": round_idx,
                "client_idx": i,
                "graph_id": str(runtime_i.client.graph_id),
                "dataset_id": str(runtime_i.client.dataset_id),
                "family": meta_i.get("family"),
                "assigned_task": meta_i.get("assigned_task"),
                "task_idx": None,
                "task": "backbone",
                "dr_self_weight": float(brow[i].item()),
                "dr_row_sum": float(brow.sum().item()),
                "dr_row_abs_sum": float(brow.abs().sum().item()),
                "dr_l2_to_p0": float(torch.norm(brow - p0_cpu, p=2).item()),
                "dr_vector_json": json.dumps([float(x) for x in brow.tolist()]),
                "p0_json": json.dumps([float(x) for x in p0_cpu.tolist()]),
                "routing_mode": "fedavg_backbone",
            }
        )
        for j, runtime_j in enumerate(runtime_clients):
            meta_j = _client_extra_metadata(runtime_j, client_metadata)
            value = float(brow[j].item())
            dr_rows.append(
                {
                    "row_type": "backbone_pair",
                    "run_type": _exp_algorithm(),
                    "algorithm": _exp_algorithm(),
                    "subset_id": subset_id,
                    "subset_clients": subset_clients,
                    "seed": seed,
                    "round": round_idx,
                    "client_idx": i,
                    "donor_idx": j,
                    "graph_id": str(runtime_i.client.graph_id),
                    "dataset_id": str(runtime_i.client.dataset_id),
                    "family": meta_i.get("family"),
                    "assigned_task": meta_i.get("assigned_task"),
                    "task_idx": None,
                    "task": "backbone",
                    "donor_graph_id": str(runtime_j.client.graph_id),
                    "donor_dataset_id": str(runtime_j.client.dataset_id),
                    "donor_family": meta_j.get("family"),
                    "donor_assigned_task": meta_j.get("assigned_task"),
                    "p_itj": value,
                    "p_ij": value,
                    "p_backbone_ij": value,
                    "abs_p_itj": abs(value),
                    "p0_j": float(p0_cpu[j].item()),
                    "is_self": int(i == j),
                    "routing_mode": "fedavg_backbone",
                }
            )
        for task_idx in range(num_tasks):
            task_name = TASKS[task_idx] if task_idx < len(TASKS) else f"task{task_idx}"
            row = task_tensor[i, task_idx]
            learned_row = learned_tensor[i, task_idx]
            support_row = support_tensor[i, task_idx]
            logits_row = logits_tensor[i, task_idx]
            rho = float(rho_matrix[i, task_idx].item())
            if float(row.min().item()) < -1e-07:
                raise RuntimeError(
                    "Effective support-only routing produced a negative coefficient"
                )
            if abs(float(row.sum().item()) - 1.0) > 1e-05:
                raise RuntimeError(
                    "Effective support-only routing row does not sum to one"
                )
            top_order = torch.argsort(row, descending=True).tolist()
            top_parts = [
                f"{runtime_clients[j].client.graph_id}:{float(row[j].item()):.6g}"
                for j in top_order[: min(3, num_donors)]
            ]
            dr_rows.append(
                {
                    "row_type": "task_summary",
                    "run_type": _exp_algorithm(),
                    "algorithm": _exp_algorithm(),
                    "subset_id": subset_id,
                    "subset_clients": subset_clients,
                    "seed": seed,
                    "round": round_idx,
                    "client_idx": i,
                    "graph_id": str(runtime_i.client.graph_id),
                    "dataset_id": str(runtime_i.client.dataset_id),
                    "family": meta_i.get("family"),
                    "assigned_task": meta_i.get("assigned_task"),
                    "mask_task": meta_i.get("mask_task"),
                    "mask_fraction": meta_i.get("mask_fraction"),
                    "task_idx": task_idx,
                    "task": task_name,
                    "dr_self_weight": float(row[i].item()),
                    "dr_row_sum": float(row.sum().item()),
                    "dr_row_abs_sum": float(row.abs().sum().item()),
                    "dr_offdiag_pos_mass": float(row.sum().item() - row[i].item()),
                    "dr_offdiag_neg_mass": 0.0,
                    "dr_offdiag_abs_mass": float(row.sum().item() - row[i].item()),
                    "dr_l2_to_p0": float(torch.norm(row - p0_cpu, p=2).item()),
                    "dr_top_abs_donors": "|".join(top_parts),
                    "routing_rho": rho,
                    "routing_effective_min": float(row.min().item()),
                    "routing_effective_max": float(row.max().item()),
                    "routing_learned_entropy": float(
                        (
                            -(
                                learned_row.clamp_min(1e-12)
                                * learned_row.clamp_min(1e-12).log()
                            ).sum()
                        ).item()
                    ),
                    "routing_support_entropy": float(
                        (
                            -(
                                support_row.clamp_min(1e-12)
                                * support_row.clamp_min(1e-12).log()
                            ).sum()
                        ).item()
                    ),
                    "dr_vector_json": json.dumps([float(x) for x in row.tolist()]),
                    "learned_vector_json": json.dumps(
                        [float(x) for x in learned_row.tolist()]
                    ),
                    "support_vector_json": json.dumps(
                        [float(x) for x in support_row.tolist()]
                    ),
                    "logits_vector_json": json.dumps(
                        [float(x) for x in logits_row.tolist()]
                    ),
                    "p0_json": json.dumps([float(x) for x in p0_cpu.tolist()]),
                    "routing_mode": "visible_support_only_simplex",
                }
            )
            for j, runtime_j in enumerate(runtime_clients):
                meta_j = _client_extra_metadata(runtime_j, client_metadata)
                value = float(row[j].item())
                dr_rows.append(
                    {
                        "row_type": "task_pair",
                        "run_type": _exp_algorithm(),
                        "algorithm": _exp_algorithm(),
                        "subset_id": subset_id,
                        "subset_clients": subset_clients,
                        "seed": seed,
                        "round": round_idx,
                        "client_idx": i,
                        "donor_idx": j,
                        "graph_id": str(runtime_i.client.graph_id),
                        "dataset_id": str(runtime_i.client.dataset_id),
                        "family": meta_i.get("family"),
                        "assigned_task": meta_i.get("assigned_task"),
                        "task_idx": task_idx,
                        "task": task_name,
                        "donor_graph_id": str(runtime_j.client.graph_id),
                        "donor_dataset_id": str(runtime_j.client.dataset_id),
                        "donor_family": meta_j.get("family"),
                        "donor_assigned_task": meta_j.get("assigned_task"),
                        "p_itj": value,
                        "p_ij": value,
                        "p_task_itj": value,
                        "abs_p_itj": abs(value),
                        "abs_p_ij": abs(value),
                        "abs_p_task_itj": abs(value),
                        "learned_softmax_itj": float(learned_row[j].item()),
                        "support_prior_tj": float(support_row[j].item()),
                        "routing_rho": rho,
                        "raw_logit_itj": float(logits_row[j].item()),
                        "is_negative_effective": int(value < 0.0),
                        "p0_j": float(p0_cpu[j].item()),
                        "is_self": int(i == j),
                        "routing_mode": "visible_support_only_simplex",
                    }
                )


def run_fedptha_with_artifacts(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    apple_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    dr_lr: float,
    support_pseudo_count: float = 1.0,
    device: torch.device,
    client_metadata: Optional[Dict[str, Dict]] = None,
) -> Tuple[RunPaths, Dict]:
    """Run one intra-cluster Support-Simplex(+PostALA) federation and
    return its validation-selected states in memory.

    This function intentionally recomputes the intra run even if matching CSVs
    already exist, because the final hierarchical exchange requires model states
    and the final PR-AUC protocol forbids persisted checkpoints.
    """
    subset_id = _subset_clients_str(subset_clients)
    cfg = dict(cfg, return_selected_states=True)
    run_paths = create_run_paths(apple_root)
    artifacts = run_fedptha_experiment(
        subset_clients,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        dr_lr=dr_lr,
        support_pseudo_count=support_pseudo_count,
        device=device,
        client_metadata=client_metadata,
    )
    if artifacts is None:
        raise RuntimeError(
            "Hierarchical intra helper did not return selected in-memory states."
        )
    return (run_paths, artifacts)
