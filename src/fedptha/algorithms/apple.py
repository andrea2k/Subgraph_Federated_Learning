from __future__ import annotations
from fedptha.training.runtime import RunPaths, create_run_paths
from fedptha.training.state import clone_state_dict
from fedptha.training.runtime import clone_model_from_state, make_task_head_model
from fedptha.training.state import (
    core_state_dicts,
    make_p0_from_train_sizes,
    _named_parameter_dict,
    self_buffer_dict,
    personalized_call,
    _client_extra_metadata,
)
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
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
from fedptha.data.clients import ClientData, format_seconds
from fedptha.training.steps import (
    augment_batch_x_with_ego,
    get_seed_label_mask,
    masked_loss_from_logits,
    unpack_batch_edges,
)
from fedptha.evaluation.metrics import pr_auc_binary, nanmean_float
from fedptha.training.random import set_seed


@dataclass
class AppleDRState:
    """Single APPLE directed-relationship vector p_i for one receiver client."""

    vector: torch.nn.Parameter


def dr_matrix_cpu(dr_vectors: Sequence[AppleDRState]) -> torch.Tensor:
    """Return the single APPLE DR matrix with shape [receiver, donor]."""
    return torch.stack(
        [state.vector.detach().cpu().clone() for state in dr_vectors], dim=0
    )


def apple_loss_scheduler(
    round_idx: int,
    rounds: int,
    *,
    scheduler_type: str = "cosine",
    scheduler_fraction: float = 0.2,
    epsilon: float = 0.001,
) -> float:
    """
    Loss scheduler lambda(r) for APPLE's proximal DR term.

    The APPLE appendix defines L as the round after which lambda(r) is zero.
    We use t = round_idx - 1 so that the first training round starts at lambda=1.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    L = max(1, int(round(float(scheduler_fraction) * float(rounds))))
    t = max(0, int(round_idx) - 1)
    if t >= L:
        return 0.0
    scheduler_type = str(scheduler_type).lower().strip()
    progress = float(t) / float(L)
    if scheduler_type in {"cos", "cosine"}:
        return float((math.cos(math.pi * progress) + 1.0) / 2.0)
    if scheduler_type in {"exp", "exponential"}:
        return float(epsilon**progress)
    raise ValueError(f"Unknown APPLE scheduler_type={scheduler_type}")


def initialize_dr_vectors(
    num_clients: int, p0: torch.Tensor, *, init: str = "sample_size"
) -> List[AppleDRState]:
    """Initialize one unconstrained donor-weight vector p_i per receiver client."""
    init = str(init).lower().strip()
    out: List[AppleDRState] = []
    if init in {"sample_size", "p0"}:
        base = p0.detach().clone()
    elif init == "uniform":
        base = torch.full_like(p0, 1.0 / max(num_clients, 1))
    else:
        raise ValueError(f"Unknown APPLE DR init={init}")
    for _ in range(num_clients):
        out.append(AppleDRState(vector=torch.nn.Parameter(base.clone())))
    return out


def make_apple_optimizer(
    core_model: torch.nn.Module, dr_vector: AppleDRState, cfg: Dict, dr_lr: float
):
    return torch.optim.Adam(
        [
            {
                "params": list(core_model.parameters()),
                "lr": float(cfg["lr"]),
                "weight_decay": float(cfg["weight_decay"]),
            },
            {"params": [dr_vector.vector], "lr": float(dr_lr), "weight_decay": 0.0},
        ]
    )


def mixed_parameter_dict(
    models_for_mix: Sequence[torch.nn.Module], client_idx: int, dr_vector: AppleDRState
) -> Dict[str, torch.Tensor]:
    """Mix every learnable parameter with the same receiver-specific APPLE vector p_i.

    For every model parameter name k:
        theta_i^p[k] = sum_j p_i[j] * theta_j^c[k]

    This includes the PNA backbone and every task head.  There is no task-specific
    DR vector in this baseline.
    """
    if len(models_for_mix) == 0:
        raise ValueError("models_for_mix is empty")
    weights = dr_vector.vector
    if weights.ndim != 1:
        raise ValueError(
            f"Expected APPLE DR shape [num_clients], got {tuple(weights.shape)}"
        )
    if weights.numel() != len(models_for_mix):
        raise ValueError(
            f"APPLE DR donor dimension mismatch: dr has {weights.numel()}, models_for_mix has {len(models_for_mix)}"
        )
    param_dicts = [_named_parameter_dict(model) for model in models_for_mix]
    names = list(param_dicts[0].keys())
    mixed: Dict[str, torch.Tensor] = {}
    for name in names:
        acc = None
        for donor_idx, params in enumerate(param_dicts):
            value = params[name]
            if donor_idx != client_idx:
                value = value.detach()
            coef = weights[donor_idx].to(device=value.device, dtype=value.dtype)
            term = coef * value
            acc = term if acc is None else acc + term
        mixed[name] = acc
    return mixed


def apple_forward(
    models_for_mix: Sequence[torch.nn.Module],
    client_idx: int,
    dr_vector: AppleDRState,
    x_in,
    edge_in,
    *,
    edge_attr_dict=None,
    device=None,
):
    receiver_model = models_for_mix[client_idx]
    params = mixed_parameter_dict(models_for_mix, client_idx, dr_vector)
    buffers = self_buffer_dict(receiver_model)
    return personalized_call(
        receiver_model,
        params,
        buffers,
        args=(x_in, edge_in),
        kwargs={"edge_attr_dict": edge_attr_dict, "device": device},
    )


def train_epoch_neighbor_apple(
    models_for_mix: Sequence[torch.nn.Module],
    client_idx: int,
    dr_vector: AppleDRState,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    *,
    use_ego_ids: bool,
    ego_dim: int,
    p0: torch.Tensor,
    apple_mu: float,
    lambda_r: float,
) -> Dict[str, float]:
    receiver_model = models_for_mix[client_idx]
    receiver_model.train()
    total_loss = 0.0
    total_base_loss = 0.0
    total_prox_loss = 0.0
    total_count = 0
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx % 50 == 0:
            print("train-batch:", batch_idx)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        x_in, y_seed, B = augment_batch_x_with_ego(batch, use_ego_ids, ego_dim)
        edge_in, edge_attr_dict = unpack_batch_edges(batch)
        out = apple_forward(
            models_for_mix,
            client_idx,
            dr_vector,
            x_in,
            edge_in,
            edge_attr_dict=edge_attr_dict,
            device=device,
        )
        out_seed = out[:B]
        label_mask_seed = get_seed_label_mask(batch, B)
        base_loss = masked_loss_from_logits(
            criterion, out_seed, y_seed.float(), label_mask_seed
        )
        p_i = dr_vector.vector
        p0_i = p0.to(device=p_i.device, dtype=p_i.dtype)
        prox_body = torch.sum((p_i - p0_i) ** 2)
        prox_loss = 0.5 * float(lambda_r) * float(apple_mu) * prox_body
        loss = base_loss + prox_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * B
        total_base_loss += float(base_loss.item()) * B
        total_prox_loss += float(prox_loss.item()) * B
        total_count += B
    denom = max(total_count, 1)
    return {
        "loss": total_loss / denom,
        "base_loss": total_base_loss / denom,
        "prox_loss": total_prox_loss / denom,
    }


@torch.no_grad()
def evaluate_loader_apple(
    core_models: Sequence[torch.nn.Module],
    client_idx: int,
    dr_vector,
    loader,
    criterion,
    device: torch.device,
    *,
    use_ego_ids: bool,
    ego_dim: int,
    threshold: float = 0.5,
    eval_mask_mode: str = "full",
) -> Dict:
    """
    APPLE evaluation with two modes.

    eval_mask_mode="full":
        Oracle/full-label evaluation. Ignores label_mask and evaluates all labels.

    eval_mask_mode="visible":
        Realistic client-visible evaluation. Uses label_mask and evaluates only
        visible label entries.
    """
    if eval_mask_mode not in {"full", "visible"}:
        raise ValueError(
            f"eval_mask_mode must be 'full' or 'visible', got {eval_mask_mode}"
        )
    receiver_model = core_models[client_idx]
    receiver_model.eval()
    total_loss = 0.0
    total_weight = 0.0
    all_logits = []
    all_labels = []
    all_masks = []
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx % 50 == 0:
            print("eval-batch:", batch_idx)
        batch = batch.to(device)
        x_in, y_seed, B = augment_batch_x_with_ego(batch, use_ego_ids, ego_dim)
        edge_in, edge_attr_dict = unpack_batch_edges(batch)
        out = apple_forward(
            core_models,
            client_idx,
            dr_vector,
            x_in,
            edge_in,
            edge_attr_dict=edge_attr_dict,
            device=device,
        )
        out_used = out[:B]
        y_used = y_seed.float()
        label_mask_seed = get_seed_label_mask(batch, B)
        if eval_mask_mode == "full":
            eval_label_mask = None
            metric_mask = torch.ones_like(y_used, dtype=torch.float32)
        elif label_mask_seed is None:
            eval_label_mask = None
            metric_mask = torch.ones_like(y_used, dtype=torch.float32)
        else:
            eval_label_mask = label_mask_seed.float()
            metric_mask = label_mask_seed.float()
        loss = masked_loss_from_logits(
            criterion, out_used, y_used, label_mask=eval_label_mask
        )
        if eval_label_mask is None:
            batch_weight = float(y_used.numel())
        else:
            batch_weight = float(eval_label_mask.sum().item())
        total_loss += float(loss.item()) * max(batch_weight, 1.0)
        total_weight += max(batch_weight, 1.0)
        all_logits.append(out_used.detach().cpu())
        all_labels.append(y_used.detach().cpu())
        all_masks.append(metric_mask.detach().cpu())
    logits = torch.cat(all_logits, dim=0) if len(all_logits) else torch.empty((0,))
    labels = torch.cat(all_labels, dim=0) if len(all_labels) else torch.empty((0,))
    masks = torch.cat(all_masks, dim=0) if len(all_masks) else torch.empty((0,))
    if labels.numel() == 0:
        return {
            "scalar": {
                "loss": float("nan"),
                "pair_acc": float("nan"),
                "subset_acc": float("nan"),
                "micro_f1": float("nan"),
                "macro_f1": float("nan"),
                "macro_pos_f1": float("nan"),
                "macro_minority_f1": float("nan"),
                "micro_pr_auc": float("nan"),
                "macro_pr_auc": float("nan"),
            },
            "per_task": {
                "tp": [],
                "fp": [],
                "tn": [],
                "fn": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "positive_f1": [],
                "minority_f1": [],
                "pr_auc": [],
            },
            "counts": {
                "num_nodes": 0,
                "pos_cnt": [],
                "pos_rate": [],
                "visible_pairs": 0,
                "total_pairs": 0,
                "visible_pair_rate": 0.0,
                "eval_mask_mode": eval_mask_mode,
            },
        }
    avg_loss = total_loss / max(total_weight, 1.0)
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    y_bool = labels.bool()
    valid = masks > 0.5
    eps = 1e-12
    C = labels.size(1)
    correct_pairs = ((preds == y_bool) & valid).sum().item()
    total_pairs = valid.sum().item()
    pair_acc = correct_pairs / max(total_pairs, 1)
    visible_per_node = valid.sum(dim=1)
    node_has_visible = visible_per_node > 0
    node_correct_visible = ((preds == y_bool) | ~valid).all(dim=1) & node_has_visible
    if node_has_visible.sum().item() > 0:
        subset_acc = (
            node_correct_visible.float().sum().item() / node_has_visible.sum().item()
        )
    else:
        subset_acc = float("nan")
    tp_task, fp_task, tn_task, fn_task = ([], [], [], [])
    prec_task, rec_task = ([], [])
    positive_f1_task = []
    minority_f1_task = []
    pr_auc_task = []
    pos_cnt = []
    pos_rate = []
    for c in range(C):
        vc = valid[:, c]
        p = preds[vc, c]
        y = y_bool[vc, c]
        if y.numel() == 0:
            tp = fp = tn = fn = 0
            precision = recall = f1_pos = f1_min = float("nan")
            pr_auc = float("nan")
            pc = 0
            pr = float("nan")
        else:
            tp = int((p & y).sum().item())
            fp = int((p & ~y).sum().item())
            tn = int((~p & ~y).sum().item())
            fn = int((~p & y).sum().item())
            precision = tp / max(tp + fp, eps)
            recall = tp / max(tp + fn, eps)
            f1_pos = 2 * precision * recall / max(precision + recall, eps)
            positives = int(y.sum().item())
            negatives = int((~y).sum().item())
            pc = positives
            pr = positives / max(positives + negatives, 1)
            pr_auc = pr_auc_binary(probs[vc, c], y.float())
            if positives <= negatives:
                min_tp = tp
                min_fp = fp
                min_fn = fn
            else:
                min_tp = tn
                min_fp = fn
                min_fn = fp
            min_precision = min_tp / max(min_tp + min_fp, eps)
            min_recall = min_tp / max(min_tp + min_fn, eps)
            f1_min = (
                2 * min_precision * min_recall / max(min_precision + min_recall, eps)
            )
        tp_task.append(tp)
        fp_task.append(fp)
        tn_task.append(tn)
        fn_task.append(fn)
        prec_task.append(float(precision))
        rec_task.append(float(recall))
        positive_f1_task.append(float(f1_pos))
        minority_f1_task.append(float(f1_min))
        pr_auc_task.append(float(pr_auc))
        pos_cnt.append(int(pc))
        pos_rate.append(float(pr))
    positive_f1_tensor = torch.tensor(positive_f1_task, dtype=torch.float32)
    minority_f1_tensor = torch.tensor(minority_f1_task, dtype=torch.float32)
    macro_pos_f1 = float(torch.nanmean(positive_f1_tensor).item())
    macro_minority_f1 = float(torch.nanmean(minority_f1_tensor).item())
    macro_pr_auc = nanmean_float(pr_auc_task)
    tp_micro = int((preds & y_bool & valid).sum().item())
    fp_micro = int((preds & ~y_bool & valid).sum().item())
    fn_micro = int((~preds & y_bool & valid).sum().item())
    micro_prec = tp_micro / max(tp_micro + fp_micro, eps)
    micro_rec = tp_micro / max(tp_micro + fn_micro, eps)
    micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, eps)
    micro_pr_auc = pr_auc_binary(probs[valid], labels[valid])
    return {
        "scalar": {
            "loss": float(avg_loss),
            "pair_acc": float(pair_acc),
            "subset_acc": float(subset_acc),
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_pos_f1),
            "macro_pos_f1": float(macro_pos_f1),
            "macro_minority_f1": float(macro_minority_f1),
            "micro_pr_auc": float(micro_pr_auc),
            "macro_pr_auc": float(macro_pr_auc),
        },
        "per_task": {
            "tp": tp_task,
            "fp": fp_task,
            "tn": tn_task,
            "fn": fn_task,
            "precision": prec_task,
            "recall": rec_task,
            "f1": positive_f1_task,
            "positive_f1": positive_f1_task,
            "minority_f1": minority_f1_task,
            "pr_auc": pr_auc_task,
        },
        "counts": {
            "num_nodes": int(labels.size(0)),
            "pos_cnt": [int(x) for x in pos_cnt],
            "pos_rate": [float(x) for x in pos_rate],
            "visible_pairs": int(valid.sum().item()),
            "total_pairs": int(valid.numel()),
            "visible_pair_rate": float(valid.float().mean().item()),
            "eval_mask_mode": eval_mask_mode,
        },
    }


def append_apple_train_diag_row(
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
    lambda_r: float,
    apple_mu: float,
    dr_vector: AppleDRState,
    p0: torch.Tensor,
) -> None:
    """Training diagnostic row for original APPLE with one p_i vector."""
    p_cpu = dr_vector.vector.detach().cpu()
    p0_cpu = p0.detach().cpu()
    if p_cpu.ndim != 1:
        raise ValueError(f"Expected APPLE DR vector, got shape={tuple(p_cpu.shape)}")
    rows.append(
        {
            "run_type": "apple_original_multi_select",
            "algorithm": "apple_original_multi_select",
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
            "dr_prox_loss": float(train_stats["prox_loss"]),
            "apple_lambda": float(lambda_r),
            "apple_mu": float(apple_mu),
            "dr_row_sum": float(p_cpu.sum().item()),
            "dr_row_abs_sum": float(p_cpu.abs().sum().item()),
            "dr_l2_to_p0": float(torch.norm(p_cpu - p0_cpu, p=2).item()),
            "dr_self_weight": float(p_cpu[int(client_idx)].item()),
            "dr_vector_json": json.dumps([float(x) for x in p_cpu.tolist()]),
            "p0_json": json.dumps([float(x) for x in p0_cpu.tolist()]),
            "num_nodes": int(runtime.num_train_nodes),
        }
    )


def append_apple_mean_row(
    rows: List[Dict],
    *,
    phase: str,
    split: str,
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: Optional[int],
    eval_infos: List[Dict],
    lambda_r: Optional[float] = None,
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
            "run_type": "apple_original_multi_select",
            "algorithm": "apple_original_multi_select",
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
            "apple_lambda": lambda_r,
            "num_clients": num_clients,
        }
    )


def append_apple_dr_rows(
    dr_rows: List[Dict],
    *,
    runtime_clients,
    dr_vectors: Sequence[AppleDRState],
    p0: torch.Tensor,
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: int,
    lambda_r: float,
    apple_mu: float,
    client_metadata: Optional[Dict[str, Dict]] = None,
) -> None:
    """Log the single APPLE DR matrix P with shape [receiver, donor]."""
    matrix = dr_matrix_cpu(dr_vectors)
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected APPLE DR matrix shape [num_clients, num_clients], got shape={tuple(matrix.shape)}"
        )
    num_receivers, num_donors = matrix.shape
    if num_receivers != len(runtime_clients) or num_donors != len(runtime_clients):
        raise ValueError(
            f"APPLE DR/client mismatch: matrix={tuple(matrix.shape)}, runtime_clients={len(runtime_clients)}"
        )
    p0_cpu = p0.detach().cpu()
    incoming_abs = matrix.abs().sum(dim=0)
    incoming_signed = matrix.sum(dim=0)
    for i, runtime_i in enumerate(runtime_clients):
        meta_i = _client_extra_metadata(runtime_i, client_metadata)
        row = matrix[i]
        off_mask = torch.ones(num_donors, dtype=torch.bool)
        off_mask[i] = False
        off = row[off_mask]
        top_order = torch.argsort(row.abs(), descending=True).tolist()
        top_parts = []
        for donor_idx in top_order[: min(3, num_donors)]:
            donor_graph = str(runtime_clients[donor_idx].client.graph_id)
            top_parts.append(f"{donor_graph}:{float(row[donor_idx].item()):.6g}")
        dr_rows.append(
            {
                "row_type": "dr_summary",
                "run_type": "apple_original_multi_select",
                "algorithm": "apple_original_multi_select",
                "subset_id": subset_id,
                "subset_clients": subset_clients,
                "seed": seed,
                "round": round_idx,
                "apple_lambda": float(lambda_r),
                "apple_mu": float(apple_mu),
                "client_idx": i,
                "graph_id": str(runtime_i.client.graph_id),
                "dataset_id": str(runtime_i.client.dataset_id),
                "family": meta_i.get("family"),
                "assigned_task": meta_i.get("assigned_task"),
                "mask_task": meta_i.get("mask_task"),
                "mask_fraction": meta_i.get("mask_fraction"),
                "task_idx": None,
                "task": None,
                "dr_self_weight": float(row[i].item()),
                "dr_row_sum": float(row.sum().item()),
                "dr_row_abs_sum": float(row.abs().sum().item()),
                "dr_offdiag_pos_mass": float(torch.clamp(off, min=0.0).sum().item()),
                "dr_offdiag_neg_mass": float(torch.clamp(off, max=0.0).sum().item()),
                "dr_offdiag_abs_mass": float(off.abs().sum().item()),
                "dr_l2_to_p0": float(torch.norm(row - p0_cpu, p=2).item()),
                "dr_top_abs_donors": "|".join(top_parts),
                "incoming_abs_mass_this_client": float(incoming_abs[i].item()),
                "incoming_signed_mass_this_client": float(incoming_signed[i].item()),
                "dr_vector_json": json.dumps([float(x) for x in row.tolist()]),
                "p0_json": json.dumps([float(x) for x in p0_cpu.tolist()]),
            }
        )
        for j, runtime_j in enumerate(runtime_clients):
            meta_j = _client_extra_metadata(runtime_j, client_metadata)
            value = float(row[j].item())
            dr_rows.append(
                {
                    "row_type": "dr_pair",
                    "run_type": "apple_original_multi_select",
                    "algorithm": "apple_original_multi_select",
                    "subset_id": subset_id,
                    "subset_clients": subset_clients,
                    "seed": seed,
                    "round": round_idx,
                    "apple_lambda": float(lambda_r),
                    "apple_mu": float(apple_mu),
                    "client_idx": i,
                    "donor_idx": j,
                    "graph_id": str(runtime_i.client.graph_id),
                    "dataset_id": str(runtime_i.client.dataset_id),
                    "family": meta_i.get("family"),
                    "assigned_task": meta_i.get("assigned_task"),
                    "donor_graph_id": str(runtime_j.client.graph_id),
                    "donor_dataset_id": str(runtime_j.client.dataset_id),
                    "donor_family": meta_j.get("family"),
                    "donor_assigned_task": meta_j.get("assigned_task"),
                    "p_ij": value,
                    "abs_p_ij": float(abs(value)),
                    "p0_j": float(p0_cpu[j].item()),
                    "is_self": int(i == j),
                }
            )


def run_apple_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    dr_lr: float,
    apple_mu: float,
    scheduler_type: str,
    scheduler_fraction: float,
    dr_init: str,
    dr_constraint: str,
    download_strategy: str,
    device: torch.device,
    client_metadata: Optional[Dict[str, Dict]] = None,
) -> None:
    if not subset_clients:
        raise ValueError("subset_clients is empty")
    if str(dr_constraint).lower().strip() != "unconstrained":
        raise ValueError(
            "This implementation is the faithful unconstrained APPLE variant."
        )
    if str(download_strategy).lower().strip() != "full":
        raise ValueError(
            "This implementation currently supports full APPLE download only."
        )
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
    dr_vectors = initialize_dr_vectors(num_clients, p0, init=dr_init)
    print(
        "[original apple single-dr]",
        "num_clients=",
        num_clients,
        "num_tasks=",
        num_tasks,
        "output_head=",
        cfg.get("output_head"),
        "mixing_mode=",
        cfg.get("apple_mixing_mode"),
        "dr_matrix_shape=",
        tuple(dr_matrix_cpu(dr_vectors).shape),
    )
    rows: List[Dict] = []
    dr_rows: List[Dict] = []
    initial_local_eval_states_by_client = {
        i: core_state_dicts(core_models) for i in range(num_clients)
    }
    initial_dr_matrix = dr_matrix_cpu(dr_vectors)

    def _initial_best_record():
        record = {
            "value": None,
            "round": -1,
            "local_eval_states_by_client": copy.deepcopy(
                initial_local_eval_states_by_client
            ),
            "dr_matrix": initial_dr_matrix.clone(),
        }
        return record

    best_by_eval_mode = {
        eval_mode: _initial_best_record() for eval_mode in ("full", "visible")
    }
    generator = torch.Generator().manual_seed(seed)
    for round_idx in range(1, rounds + 1):
        run_start_time = time.perf_counter()
        lambda_r = apple_loss_scheduler(
            round_idx,
            rounds,
            scheduler_type=scheduler_type,
            scheduler_fraction=scheduler_fraction,
        )
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
        for idx in selected_indices:
            runtime = runtime_clients[idx]
            local_model = clone_model_from_state(
                round_start_states[idx], cfg, ctx, device
            )
            models_for_mix = list(downloaded_core_models)
            models_for_mix[idx] = local_model
            optimizer = make_apple_optimizer(local_model, dr_vectors[idx], cfg, dr_lr)
            for local_epoch_idx in range(1, local_epochs + 1):
                train_stats = train_epoch_neighbor_apple(
                    models_for_mix,
                    idx,
                    dr_vectors[idx],
                    runtime.train_loader,
                    optimizer,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    p0=p0,
                    apple_mu=apple_mu,
                    lambda_r=lambda_r,
                )
                append_train_row(
                    rows,
                    run_type="apple_original_multi_select",
                    algorithm="apple_original_multi_select",
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
                append_apple_train_diag_row(
                    rows,
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    round_idx=round_idx,
                    local_epoch=local_epoch_idx,
                    runtime=runtime,
                    client_idx=idx,
                    train_stats=train_stats,
                    lambda_r=lambda_r,
                    apple_mu=apple_mu,
                    dr_vector=dr_vectors[idx],
                    p0=p0,
                )
            round_local_eval_states_by_client[idx] = core_state_dicts(models_for_mix)
            for eval_mask_mode in ("full", "visible"):
                post_metrics = evaluate_loader_apple(
                    models_for_mix,
                    idx,
                    dr_vectors[idx],
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
                    run_type="apple_original_multi_select",
                    algorithm="apple_original_multi_select",
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
            append_apple_mean_row(
                rows,
                phase=f"apple_local_val_mean_{eval_mask_mode}",
                split="val",
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                round_idx=round_idx,
                eval_infos=round_local_eval_infos_by_mode[eval_mask_mode],
                lambda_r=lambda_r,
                num_clients=num_clients,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )
        for idx, updated_model in updated_models.items():
            core_models[idx] = updated_model
        round_eval_infos_by_mode: Dict[str, List[Dict]] = {"full": [], "visible": []}
        for eval_mask_mode in ("full", "visible"):
            for idx, runtime in enumerate(runtime_clients):
                metrics = evaluate_loader_apple(
                    core_models,
                    idx,
                    dr_vectors[idx],
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
                    run_type="apple_original_multi_select",
                    algorithm="apple_original_multi_select",
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
            append_apple_mean_row(
                rows,
                phase=f"apple_val_mean_{eval_mask_mode}",
                split="val",
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                round_idx=round_idx,
                eval_infos=round_eval_infos_by_mode[eval_mask_mode],
                lambda_r=lambda_r,
                num_clients=num_clients,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )
        append_apple_dr_rows(
            dr_rows,
            runtime_clients=runtime_clients,
            dr_vectors=dr_vectors,
            p0=p0,
            subset_id=subset_id,
            subset_clients=subset_clients_str,
            seed=seed,
            round_idx=round_idx,
            lambda_r=lambda_r,
            apple_mu=apple_mu,
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
                best_record["dr_matrix"] = dr_matrix_cpu(dr_vectors)
        run_elapsed = time.perf_counter() - run_start_time
        print(f"run time: {format_seconds(run_elapsed)}")
        print("=======================================================")
        print(f"==================== round {round_idx} / {rounds} ====================")
        print("=======================================================")
        del downloaded_core_models
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
        selected_dr_vectors = [
            AppleDRState(
                vector=torch.nn.Parameter(
                    selected_dr_matrix[i].to(device).clone(), requires_grad=False
                )
            )
            for i in range(selected_dr_matrix.size(0))
        ]
        for split_name in job["splits"]:
            split_eval_infos = []
            for idx, runtime in enumerate(runtime_clients):
                split_loader = {
                    "train": runtime.train_loader,
                    "val": runtime.val_loader,
                    "test": runtime.test_loader,
                }[split_name]
                best_models_for_mix = [
                    clone_model_from_state(state, cfg, ctx, device)
                    for state in selected_state["local_eval_states_by_client"][idx]
                ]
                metrics = evaluate_loader_apple(
                    best_models_for_mix,
                    idx,
                    selected_dr_vectors[idx],
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
                    run_type="apple_original_multi_select",
                    algorithm="apple_original_multi_select",
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
            append_apple_mean_row(
                rows,
                phase=f"best_{selection_protocol}_macro_pr_auc_{split_name}_mean",
                split=split_name,
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                round_idx=selected_round,
                eval_infos=split_eval_infos,
                lambda_r=None,
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
    print("APPLE ORIGINAL SINGLE-DR DUAL-EVALUATION SMOKE CHECK")
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


def run_apple(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    apple_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    dr_lr: float,
    apple_mu: float,
    scheduler_type: str,
    scheduler_fraction: float,
    dr_init: str = "sample_size",
    dr_constraint: str = "unconstrained",
    download_strategy: str = "full",
    device: torch.device,
    client_metadata: Optional[Dict[str, Dict]] = None,
) -> RunPaths:
    subset_id = _subset_clients_str(subset_clients)
    run_paths = create_run_paths(apple_root)
    if run_paths.csv_path.exists() and run_paths.dr_csv_path.exists():
        return run_paths
    run_apple_experiment(
        subset_clients,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        dr_lr=dr_lr,
        apple_mu=apple_mu,
        scheduler_type=scheduler_type,
        scheduler_fraction=scheduler_fraction,
        dr_init=dr_init,
        dr_constraint=dr_constraint,
        download_strategy=download_strategy,
        device=device,
        client_metadata=client_metadata,
    )
    return run_paths
