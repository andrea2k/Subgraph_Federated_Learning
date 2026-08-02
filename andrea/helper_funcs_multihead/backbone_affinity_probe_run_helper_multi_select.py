from __future__ import annotations

import copy
import json
import os
import math
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from andrea.helper_funcs_multihead.fl_run_helper_multi_select import (
    _model_tag_from_cfg,
    _subset_clients_str,
    append_eval_rows,
    append_train_row,
    build_model_context,
    build_runtime_clients,
    ensure_dir,
    make_model,
    weighted_scalar_summary,
    normalize_selection_metrics,
    selection_direction,
    initial_selection_value,
    is_better_selection_value,
    maybe_save_checkpoint,
    tag_recent_eval_rows,
)
from andrea.helper_funcs_multihead.fedala_run_helper_multi_select import (
    FedALAClientState,
    learn_ala_weights_for_client,
)
from andrea.helper_funcs_multihead.load_client_helper import ClientData, format_seconds
from andrea.helper_funcs_multihead.train_eval_helper import (
    augment_batch_x_with_ego,
    evaluate_loader,
    compute_minority_f1_score_per_task,
    compute_positive_f1_score_per_task,
    get_seed_label_mask,
    masked_loss_from_logits,
    unpack_batch_edges,
    average_precision_binary,
    nanmean_float,
)
from utils.seed import set_seed

from andrea.multigraph_generation import TASKS
from models.pna_reverse_mp_taskhead import PNANetReverseMP as PNANetReverseMPTaskHead

try:
    from torch.func import functional_call as _torch_functional_call
except Exception:  # pragma: no cover - compatibility path for older PyTorch
    from torch.nn.utils.stateless import functional_call as _torch_functional_call


def _exp_algorithm() -> str:
    return os.environ.get(
        "APPLE_EXPERIMENT_ALGORITHM",
        "backbone_affinity_probe_multi_select",
    )


def _copy_ala_weights(weights: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not weights:
        return {}
    return {name: value.detach().cpu().clone() for name, value in weights.items()}


def _copy_ala_weights_by_client(weights_by_client: Optional[Dict[int, Dict[str, torch.Tensor]]]) -> Dict[int, Dict[str, torch.Tensor]]:
    if not weights_by_client:
        return {}
    return {int(idx): _copy_ala_weights(weights) for idx, weights in weights_by_client.items()}


@dataclass(frozen=True)
class AppleRunPaths:
    csv_path: Path
    ckpt_path: Path
    dr_csv_path: Path


@dataclass
class AppleDRState:
    """Learned donor weights for one receiver client.

    In this FedAvg-backbone variant:

    - backbone is kept only as a fixed p0/debug vector, not optimized and not used
      for parameter mixing.
    - task is learned and used for task-wise APPLE head mixing.
    """

    backbone: torch.nn.Parameter
    task: torch.nn.Parameter




def compact_subset_id_for_filename(subset_id: str) -> str:
    """Return a short readable subset id safe for macOS/Linux filenames."""
    raw = str(subset_id)
    parts = [p for p in raw.split("|") if p]
    try:
        ids = [int(p) for p in parts]
        if len(ids) == 5 and min(ids) >= 3000000 and max(ids) <= 3000014:
            first = min(ids)
            last = max(ids)
            if first >= 3000010:
                qtag = "q80"
            elif first >= 3000005:
                qtag = "q50"
            else:
                qtag = "q20"
            return f"{qtag}_{first}-{last}"
    except Exception:
        pass
    safe = raw.replace("|", "-").replace("/", "_").replace(" ", "_")
    if len(safe) <= 48:
        return safe
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"subset_{safe[:28]}_{digest}"


def compact_model_tag_for_filename(model_tag: str) -> str:
    """Shorten the multi-select suffix while keeping core hyperparameters readable."""
    tag = str(model_tag)
    replacements = {
        "selmultiselect_loss_micro_f1_macro_pos_f1_micro_pr_auc_macro_pr_auc_multiselect": "selms5",
        "selmultiselect_loss_micro_f1_macro_pos_f1_micro_pr_auc_macro_pr_auc": "selms5",
        "multiselect_loss_micro_f1_macro_pos_f1_micro_pr_auc_macro_pr_auc": "ms5",
    }
    for old, new in replacements.items():
        tag = tag.replace(old, new)
    if len(tag) <= 100:
        return tag
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:10]
    return f"{tag[:88]}_{digest}"


# -----------------------------------------------------------------------------
# Run paths and manifest rows
# -----------------------------------------------------------------------------
def create_apple_run_paths(
    apple_root: str | Path,
    subset_id: str,
    rounds: int,
    local_epochs: int,
    dr_lr: float,
    apple_mu: float,
    scheduler_type: str,
    scheduler_fraction: float,
    model_tag: str,
    seed: int,
) -> AppleRunPaths:
    root = ensure_dir(apple_root)
    ckpt_root = ensure_dir(apple_root)
    subset_id = compact_subset_id_for_filename(subset_id)
    model_tag = compact_model_tag_for_filename(model_tag)
    stem = (
        f"{_exp_algorithm()}"
        f"{subset_id}"
        f"_rounds{rounds}"
        f"_epoch{local_epochs}"
        f"_drlr{dr_lr}"
        f"_mu{apple_mu}"
        f"_sched{scheduler_type}"
        f"_Lfrac{scheduler_fraction}"
        f"_{model_tag}"
        f"_selectlocal"
        f"_seed{seed}"
    )

    suffixes = (
        ".csv", ".pt", "_dr.csv", "_affinity_clients.csv",
        "_affinity_layers.csv", "_affinity_pairs.csv",
        "_affinity_parameter_pairs.csv",
    )
    if max(len(f"{stem}{suffix}".encode("utf-8")) for suffix in suffixes) > 255:
        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
        algorithm = "".join(
            c if c.isalnum() else "_" for c in str(_exp_algorithm())
        ).strip("_")
        if len(algorithm) > 28:
            algorithm = (
                f"{algorithm[:19]}_"
                f"{hashlib.sha1(algorithm.encode('utf-8')).hexdigest()[:8]}"
            )
        subset_safe = str(subset_id)
        if len(subset_safe) > 36:
            subset_safe = (
                f"{subset_safe[:27]}_"
                f"{hashlib.sha1(subset_safe.encode('utf-8')).hexdigest()[:8]}"
            )
        stem = f"{algorithm}_{subset_safe}_r{rounds}_e{local_epochs}_s{seed}_{digest}"

    for suffix in suffixes:
        if len(f"{stem}{suffix}".encode("utf-8")) > 255:
            raise RuntimeError(f"Filename remains too long: {stem}{suffix}")

    return AppleRunPaths(
        csv_path=root / f"{stem}.csv",
        ckpt_path=ckpt_root / f"{stem}.pt",
        dr_csv_path=root / f"{stem}_dr.csv",
    )


def build_apple_log_row(
    *,
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: AppleRunPaths,
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
    selection_metrics: Optional[Sequence[str]] = None,
    selection_metric: Optional[str] = None,
) -> Dict:
    selection_metrics = normalize_selection_metrics(selection_metrics, selection_metric)
    subset_id = _subset_clients_str(subset_clients)
    dataset_ids = "|".join(str(client.dataset_id) for client in subset_clients)
    model_tag = _model_tag_from_cfg(cfg)

    return {
        "run_type": _exp_algorithm(),
        "algorithm": _exp_algorithm(),
        "subset_id": subset_id,
        "subset_clients": subset_id,
        "graph_id": "all",
        "dataset_id": dataset_ids,
        "num_clients": len(subset_clients),
        "client_fraction": client_fraction,
        "out_csv": str(run_paths.csv_path),
        "ckpt_path": str(run_paths.ckpt_path),
        "dr_csv_path": str(run_paths.dr_csv_path),
        "affinity_clients_csv_path": str(affinity_probe_output_paths(run_paths)["clients"]),
        "affinity_layers_csv_path": str(affinity_probe_output_paths(run_paths)["layers"]),
        "affinity_pairs_csv_path": str(affinity_probe_output_paths(run_paths)["pairs"]),
        "affinity_parameter_pairs_csv_path": str(
            affinity_probe_output_paths(run_paths)["parameter_pairs"]
        ),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "selection_metric": "multi",
        "selection_metrics": "|".join(selection_metrics),
        "selection_direction": "mixed",
        "apple_dr_lr": dr_lr,
        "apple_mu": apple_mu,
        "apple_scheduler_type": scheduler_type,
        "apple_scheduler_fraction": scheduler_fraction,
        "apple_dr_init": dr_init,
        "apple_dr_constraint": dr_constraint,
        "apple_download_strategy": download_strategy,
        "apple_use_head_ala": cfg.get("apple_use_head_ala", False),
        "apple_ala_lr": cfg.get("apple_ala_lr", None),
        "apple_ala_rand_percent": cfg.get("apple_ala_rand_percent", None),
        "apple_ala_convergence_std": cfg.get("apple_ala_convergence_std", None),
        "apple_ala_convergence_window": cfg.get("apple_ala_convergence_window", None),
        "apple_ala_max_steps": cfg.get("apple_ala_max_steps", None),
        "apple_expert_rho": cfg.get("apple_expert_rho", None),
        "apple_expert_pseudo_count": cfg.get("apple_expert_pseudo_count", None),
        "apple_mixing_mode": cfg.get("apple_mixing_mode", "fedavg_backbone_task_head"),
        "output_head": cfg.get("output_head", "multi"),
        "architecture": cfg.get("architecture", "multihead"),
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
# Generic state/model helpers
# -----------------------------------------------------------------------------
def make_task_head_model(
    cfg, x_dim, out_dim, deg_fwd_hist, deg_rev_hist, ego_dim, in_vocab, out_vocab
):
    """
    Build the task-head PNA model.

    This is intentionally separate from fl_run_helper.make_model(), so the
    stable old baselines and old APPLE implementation remain untouched.
    """
    return PNANetReverseMPTaskHead(
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
        port_emb_dim=(
            cfg.get("port_emb_dim", 0) if cfg.get("use_port_ids", True) else 0
        ),
        output_head=cfg.get("output_head", "multi"),
    )


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def clone_model_from_state(
    state_dict: Dict[str, torch.Tensor],
    cfg: Dict,
    ctx: Dict,
    device: torch.device,
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


def core_state_dicts(
    core_models: Sequence[torch.nn.Module],
) -> List[Dict[str, torch.Tensor]]:
    return [clone_state_dict(model.state_dict()) for model in core_models]


def dr_matrix_cpu(
    dr_vectors: Sequence[AppleDRState],
) -> torch.Tensor:
    """Return task-head DR tensor with shape [receiver, task, donor]."""
    return torch.stack(
        [state.task.detach().cpu().clone() for state in dr_vectors], dim=0
    )


def dr_backbone_matrix_cpu(
    dr_vectors: Sequence[AppleDRState],
) -> torch.Tensor:
    """Return backbone DR matrix with shape [receiver, donor]."""
    return torch.stack(
        [state.backbone.detach().cpu().clone() for state in dr_vectors], dim=0
    )


# -----------------------------------------------------------------------------
# APPLE-specific math
# -----------------------------------------------------------------------------
def apple_loss_scheduler(
    round_idx: int,
    rounds: int,
    *,
    scheduler_type: str = "cosine",
    scheduler_fraction: float = 0.2,
    epsilon: float = 1e-3,
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


def make_p0_from_train_sizes(runtime_clients, device: torch.device) -> torch.Tensor:
    sizes = torch.tensor(
        [float(runtime.num_train_nodes) for runtime in runtime_clients],
        dtype=torch.float32,
        device=device,
    )
    total = sizes.sum().clamp_min(1.0)
    return sizes / total


def make_visible_support_task_matrix(
    runtime_clients,
    num_tasks: int,
    p0: torch.Tensor,
    *,
    device: torch.device,
    expert_rho: float = 0.8,
    expert_pseudo_count: float = 1.0,
) -> torch.Tensor:
    """Build a non-oracle task-expert donor prior from visible train positives.

    Output shape is [num_tasks, num_clients]. Row t is a donor distribution for
    task t. It uses only labels that are visible through label_mask, not q metadata.

    task_prior[t, j] = (1-rho) * sample_size_weight[j]
                     + rho     * normalized_visible_positive_support[t, j]
    """
    if runtime_clients is None:
        raise ValueError("visible_support init requires runtime_clients")

    rows = []
    for runtime in runtime_clients:
        y = runtime.client.train_h["n"].y.float()
        if "label_mask" in runtime.client.train_h["n"]:
            mask = runtime.client.train_h["n"].label_mask.float()
        else:
            mask = torch.ones_like(y)

        visible_pos = (y * mask).sum(dim=0)
        rows.append(visible_pos[: int(num_tasks)])

    if not rows:
        raise ValueError("runtime_clients is empty")

    pos_matrix = torch.stack(rows, dim=0).to(device=device, dtype=torch.float32)
    # [num_clients, num_tasks] -> [num_tasks, num_clients]
    score = pos_matrix.T + float(expert_pseudo_count)
    expert = score / score.sum(dim=1, keepdim=True).clamp_min(1e-12)

    p0_row = p0.to(device=device, dtype=torch.float32).unsqueeze(0)
    rho = float(expert_rho)
    if rho < 0.0 or rho > 1.0:
        raise ValueError(f"expert_rho must be in [0, 1], got {rho}")

    task_prior = (1.0 - rho) * p0_row + rho * expert
    task_prior = task_prior / task_prior.sum(dim=1, keepdim=True).clamp_min(1e-12)

    print("[TaskExpert init] visible positive support matrix [donor, task]:")
    print(pos_matrix.detach().cpu())
    print("[TaskExpert init] task donor prior [task, donor]:")
    print(task_prior.detach().cpu())
    print("[TaskExpert init] rho=", rho, "pseudo_count=", expert_pseudo_count)

    return task_prior


def initialize_dr_vectors(
    num_clients: int,
    num_tasks: int,
    p0: torch.Tensor,
    *,
    init: str = "sample_size",
    runtime_clients=None,
    device: Optional[torch.device] = None,
    expert_rho: float = 0.8,
    expert_pseudo_count: float = 1.0,
) -> List[AppleDRState]:
    """
    Backbone+task-head APPLE DR initialization.

    For each receiver client i we learn:

        backbone_i[j]      = donor weight for backbone / non-head parameters
        task_i[t, j]       = donor weight for task-head t parameters

    Shapes:
        backbone_i: [num_clients]
        task_i:     [num_tasks, num_clients]
    """
    init = str(init).lower().strip()
    out: List[AppleDRState] = []

    if init in {"sample_size", "p0"}:
        backbone_base = p0.detach().clone()
        task_base = p0.detach().clone().unsqueeze(0).repeat(int(num_tasks), 1)
    elif init == "uniform":
        backbone_base = torch.full_like(p0, 1.0 / max(num_clients, 1))
        task_base = (
            backbone_base.detach().clone().unsqueeze(0).repeat(int(num_tasks), 1)
        )
    elif init in {"visible_support", "taskexpert", "task_expert", "visible_task_support"}:
        if device is None:
            device = p0.device
        backbone_base = p0.detach().clone()
        task_base = make_visible_support_task_matrix(
            runtime_clients,
            int(num_tasks),
            p0,
            device=device,
            expert_rho=expert_rho,
            expert_pseudo_count=expert_pseudo_count,
        ).detach().clone()
    else:
        raise ValueError(f"Unknown APPLE DR init={init}")

    for _ in range(num_clients):
        out.append(
            AppleDRState(
                backbone=torch.nn.Parameter(backbone_base.clone()),
                task=torch.nn.Parameter(task_base.clone()),
            )
        )
    return out


def make_apple_optimizer(
    core_model: torch.nn.Module,
    dr_vector: AppleDRState,
    cfg: Dict,
    dr_lr: float,
):
    # Backbone parameters are optimized as ordinary local model parameters.
    # The backbone is later aggregated by FedAvg after local training.
    #
    # Only task-wise APPLE head DR is learned. The backbone DR vector is kept
    # fixed for logging/debug compatibility and is not optimized.
    return torch.optim.Adam(
        [
            {
                "params": list(core_model.parameters()),
                "lr": float(cfg["lr"]),
                "weight_decay": float(cfg["weight_decay"]),
            },
            {
                "params": [dr_vector.task],
                "lr": float(dr_lr),
                "weight_decay": 0.0,
            },
        ]
    )


def _named_parameter_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return dict(model.named_parameters())


def _task_idx_from_output_head_param_name(name: str) -> Optional[int]:
    """
    Parse task index from a multi-head parameter name.

    Expected names for output_head="multi" look like:
        output_head.heads.0.0.weight
        output_head.heads.0.0.bias
        output_head.heads.0.2.weight
        output_head.heads.0.2.bias
        output_head.heads.1.0.weight
        ...

    Returns:
        task index if this parameter belongs to one task head;
        None if this is a backbone / non-head parameter.
    """
    prefix = "output_head.heads."
    if not name.startswith(prefix):
        return None

    rest = name[len(prefix) :]
    first = rest.split(".", 1)[0]

    try:
        return int(first)
    except ValueError:
        return None


def _is_task_head_state_name(name: str) -> bool:
    return _task_idx_from_output_head_param_name(name) is not None


def fedavg_backbone_state_dict(
    models: Sequence[torch.nn.Module],
    runtimes: Sequence,
) -> Dict[str, torch.Tensor]:
    """
    Weighted FedAvg over backbone / non-head state entries only.

    Task-head entries are intentionally excluded because task heads remain
    client-specific and are combined by task-wise APPLE DR during forward/eval.
    """
    if len(models) == 0:
        raise ValueError("models is empty")
    if len(models) != len(runtimes):
        raise ValueError(
            f"models/runtimes length mismatch: {len(models)} vs {len(runtimes)}"
        )

    weights = torch.tensor(
        [float(runtime.num_train_nodes) for runtime in runtimes],
        dtype=torch.float32,
    )
    weights = weights / weights.sum().clamp_min(1.0)

    state_dicts = [model.state_dict() for model in models]
    keys = list(state_dicts[0].keys())

    out: Dict[str, torch.Tensor] = {}

    for key in keys:
        if _is_task_head_state_name(key):
            continue

        first = state_dicts[0][key]

        # Average only floating tensors. For integer buffers such as
        # num_batches_tracked, keep the first selected client's value.
        if not torch.is_floating_point(first):
            out[key] = first.detach().clone()
            continue

        acc = None
        for weight, state in zip(weights, state_dicts):
            value = state[key].detach().cpu()
            coef = weight.to(dtype=value.dtype)
            term = coef * value
            acc = term if acc is None else acc + term

        out[key] = acc.clone()

    return out


def load_backbone_state_into_models(
    models: Sequence[torch.nn.Module],
    backbone_state: Dict[str, torch.Tensor],
) -> None:
    """
    Copy the averaged FedAvg backbone into every client model while preserving
    each client's own task heads.
    """
    for model in models:
        current = model.state_dict()

        for key, value in backbone_state.items():
            if key in current:
                current[key] = value.to(
                    device=current[key].device, dtype=current[key].dtype
                )

        model.load_state_dict(current, strict=True)


def mixed_parameter_dict(
    models_for_mix: Sequence[torch.nn.Module],
    client_idx: int,
    dr_vector: AppleDRState,
    *,
    ala_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build the personalized parameter dictionary for:

        FedAvg backbone + task-wise APPLE heads

    Design:

    - Backbone / non-head parameters are NOT APPLE-mixed.
      They come directly from the receiver client's model. During local training,
      this is the client's local backbone initialized from the current FedAvg
      backbone. After local training, the server FedAvg-averages these backbone
      parameters across clients.

    - Task-head parameters ARE APPLE-mixed task-wise:

          head_{i,t} = sum_j p_task[i, t, j] * head_{j,t}

    This gives a stable FedAvg representation while preserving task-specific
    donor selection for the output heads.
    """
    if len(models_for_mix) == 0:
        raise ValueError("models_for_mix is empty")

    task_weights = dr_vector.task

    if task_weights.ndim != 2:
        raise ValueError(
            f"Expected task DR shape [num_tasks, num_clients], "
            f"got shape={tuple(task_weights.shape)}"
        )

    num_tasks, num_clients_from_task = task_weights.shape

    if num_clients_from_task != len(models_for_mix):
        raise ValueError(
            f"Task DR donor dimension mismatch: "
            f"dr has {num_clients_from_task}, models_for_mix has {len(models_for_mix)}"
        )

    param_dicts = [_named_parameter_dict(model) for model in models_for_mix]
    receiver_params = param_dicts[int(client_idx)]

    names = list(param_dicts[0].keys())
    mixed: Dict[str, torch.Tensor] = {}

    for name in names:
        task_idx = _task_idx_from_output_head_param_name(name)

        if task_idx is None:
            # Backbone / non-head parameter:
            # use receiver client's own trainable parameter.
            mixed[name] = receiver_params[name]
            continue

        if task_idx < 0 or task_idx >= num_tasks:
            raise ValueError(
                f"Task index {task_idx} from parameter {name} is outside "
                f"num_tasks={num_tasks}"
            )

        # Task-head parameter:
        # mix same-task heads using task-specific APPLE donor weights.
        weights = task_weights[task_idx]

        acc = None
        for donor_idx, params in enumerate(param_dicts):
            value = params[name]

            # Donor models are frozen/downloaded constants.
            # Receiver client's own task-head parameter remains attached to autograd.
            if donor_idx != client_idx:
                value = value.detach()

            coef = weights[donor_idx].to(device=value.device, dtype=value.dtype)
            term = coef * value
            acc = term if acc is None else acc + term

        if ala_weights is not None and name in ala_weights:
            # Head-only ALA gate: alpha=1 keeps APPLE mixed head,
            # alpha=0 keeps receiver-local head.
            receiver_value = receiver_params[name]
            alpha = ala_weights[name].to(
                device=receiver_value.device,
                dtype=receiver_value.dtype,
            ).clamp(0.0, 1.0)
            acc = receiver_value + alpha * (acc - receiver_value)

        mixed[name] = acc

    return mixed


def self_buffer_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Use the receiving client's own buffers, especially BatchNorm statistics.

    APPLE's equations are over learnable model parameters. For PyTorch modules,
    state_dict also contains non-learnable buffers. Using the receiving client's
    buffers keeps BatchNorm state local instead of mixing running statistics with
    unconstrained, possibly negative DR coefficients.
    """
    return dict(model.named_buffers())


def functional_call_compat(
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    args: Tuple,
    kwargs: Optional[Dict] = None,
):
    kwargs = {} if kwargs is None else dict(kwargs)
    try:
        return _torch_functional_call(
            model, (params, buffers), args, kwargs, strict=False
        )
    except TypeError:
        merged = dict(params)
        merged.update(buffers)
        return _torch_functional_call(model, merged, args, kwargs)


def apple_forward(
    models_for_mix: Sequence[torch.nn.Module],
    client_idx: int,
    dr_vector: AppleDRState,
    x_in,
    edge_in,
    *,
    edge_attr_dict=None,
    device=None,
    ala_weights: Optional[Dict[str, torch.Tensor]] = None,
):
    receiver_model = models_for_mix[client_idx]
    params = mixed_parameter_dict(
        models_for_mix,
        client_idx,
        dr_vector,
        ala_weights=ala_weights,
    )
    buffers = self_buffer_dict(receiver_model)
    return functional_call_compat(
        receiver_model,
        params,
        buffers,
        args=(x_in, edge_in),
        kwargs={"edge_attr_dict": edge_attr_dict, "device": device},
    )


# -----------------------------------------------------------------------------
# APPLE training and evaluation
# -----------------------------------------------------------------------------
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
    ala_weights: Optional[Dict[str, torch.Tensor]] = None,
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
            ala_weights=ala_weights,
        )
        out_seed = out[:B]

        label_mask_seed = get_seed_label_mask(batch, B)
        base_loss = masked_loss_from_logits(
            criterion,
            out_seed,
            y_seed.float(),
            label_mask_seed,
        )

        # Regularize only the learned task-wise DR rows toward p0.
        # The backbone is handled by FedAvg and has no learned DR vector in this
        # method.
        task = dr_vector.task
        p0_task = p0.unsqueeze(0).to(device=task.device, dtype=task.dtype)

        task_row_l2 = torch.sum((task - p0_task) ** 2, dim=1)
        prox_body = torch.mean(task_row_l2)
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
    ala_weights: Optional[Dict[str, torch.Tensor]] = None,
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
            ala_weights=ala_weights,
        )

        out_used = out[:B]
        y_used = y_seed.float()

        label_mask_seed = get_seed_label_mask(batch, B)

        if eval_mask_mode == "full":
            eval_label_mask = None
            metric_mask = torch.ones_like(y_used, dtype=torch.float32)
        else:
            if label_mask_seed is None:
                eval_label_mask = None
                metric_mask = torch.ones_like(y_used, dtype=torch.float32)
            else:
                eval_label_mask = label_mask_seed.float()
                metric_mask = label_mask_seed.float()

        loss = masked_loss_from_logits(
            criterion,
            out_used,
            y_used,
            label_mask=eval_label_mask,
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
    node_correct_visible = (
        ((preds == y_bool) | (~valid)).all(dim=1)
    ) & node_has_visible

    if node_has_visible.sum().item() > 0:
        subset_acc = (
            node_correct_visible.float().sum().item() / node_has_visible.sum().item()
        )
    else:
        subset_acc = float("nan")

    tp_task, fp_task, tn_task, fn_task = [], [], [], []
    prec_task, rec_task = [], []
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
            ap = float("nan")
            pc = 0
            pr = float("nan")
        else:
            tp = int((p & y).sum().item())
            fp = int((p & (~y)).sum().item())
            tn = int(((~p) & (~y)).sum().item())
            fn = int(((~p) & y).sum().item())

            precision = tp / max(tp + fp, eps)
            recall = tp / max(tp + fn, eps)
            f1_pos = 2 * precision * recall / max(precision + recall, eps)

            positives = int(y.sum().item())
            negatives = int((~y).sum().item())
            pc = positives
            pr = positives / max(positives + negatives, 1)
            ap = average_precision_binary(probs[vc, c], y.float())

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
                2
                * min_precision
                * min_recall
                / max(
                    min_precision + min_recall,
                    eps,
                )
            )

        tp_task.append(tp)
        fp_task.append(fp)
        tn_task.append(tn)
        fn_task.append(fn)
        prec_task.append(float(precision))
        rec_task.append(float(recall))
        positive_f1_task.append(float(f1_pos))
        minority_f1_task.append(float(f1_min))
        pr_auc_task.append(float(ap))
        pos_cnt.append(int(pc))
        pos_rate.append(float(pr))

    positive_f1_tensor = torch.tensor(positive_f1_task, dtype=torch.float32)
    minority_f1_tensor = torch.tensor(minority_f1_task, dtype=torch.float32)

    macro_pos_f1 = float(torch.nanmean(positive_f1_tensor).item())
    macro_minority_f1 = float(torch.nanmean(minority_f1_tensor).item())
    macro_pr_auc = nanmean_float(pr_auc_task)

    tp_micro = int((preds & y_bool & valid).sum().item())
    fp_micro = int((preds & (~y_bool) & valid).sum().item())
    fn_micro = int(((~preds) & y_bool & valid).sum().item())

    micro_prec = tp_micro / max(tp_micro + fp_micro, eps)
    micro_rec = tp_micro / max(tp_micro + fn_micro, eps)
    micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, eps)
    micro_pr_auc = average_precision_binary(probs[valid], labels[valid])

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


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
def _client_extra_metadata(
    runtime,
    client_metadata: Optional[Dict[str, Dict]],
) -> Dict:
    graph_id = str(runtime.client.graph_id)
    meta = dict(client_metadata.get(graph_id, {})) if client_metadata else {}

    mask_meta = runtime.client.mask_meta or {}
    if "assigned_task" not in meta and "assigned_task" in mask_meta:
        meta["assigned_task"] = mask_meta.get("assigned_task")
    if "mask_task" not in meta and "mask_task" in mask_meta:
        meta["mask_task"] = mask_meta.get("mask_task")
    if "mask_fraction" not in meta and "mask_fraction" in mask_meta:
        meta["mask_fraction"] = mask_meta.get("mask_fraction")

    return meta


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
    """Training diagnostic row for backbone+task-head APPLE."""
    backbone_cpu = dr_vector.backbone.detach().cpu()
    task_cpu = dr_vector.task.detach().cpu()
    p0_cpu = p0.detach().cpu()

    if backbone_cpu.ndim != 1:
        raise ValueError(
            f"Expected backbone DR vector, got shape={tuple(backbone_cpu.shape)}"
        )
    if task_cpu.ndim != 2:
        raise ValueError(f"Expected task DR matrix, got shape={tuple(task_cpu.shape)}")

    task_row_sums = task_cpu.sum(dim=1)
    task_self_weights = task_cpu[:, int(client_idx)]
    task_l2_to_p0 = torch.norm(task_cpu - p0_cpu.unsqueeze(0), p=2, dim=1)

    backbone_row_sum = float(backbone_cpu.sum().item())
    backbone_self_weight = float(backbone_cpu[int(client_idx)].item())
    backbone_l2_to_p0 = float(torch.norm(backbone_cpu - p0_cpu, p=2).item())

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
            "dr_prox_loss": float(train_stats["prox_loss"]),
            "apple_lambda": float(lambda_r),
            "apple_mu": float(apple_mu),
            # Backward-compatible averaged task-head summaries.
            "dr_row_sum": float(task_row_sums.mean().item()),
            "dr_l2_to_p0": float(task_l2_to_p0.mean().item()),
            "dr_self_weight": float(task_self_weights.mean().item()),
            "dr_row_sum_by_task_json": json.dumps(
                [float(x) for x in task_row_sums.tolist()]
            ),
            "dr_self_weight_by_task_json": json.dumps(
                [float(x) for x in task_self_weights.tolist()]
            ),
            "dr_l2_to_p0_by_task_json": json.dumps(
                [float(x) for x in task_l2_to_p0.tolist()]
            ),
            # New backbone diagnostics.
            "dr_backbone_row_sum": backbone_row_sum,
            "dr_backbone_self_weight": backbone_self_weight,
            "dr_backbone_l2_to_p0": backbone_l2_to_p0,
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
        sum(int(m["counts"].get("visible_pairs", 0)) for m in eval_infos)
    )
    total_pairs = int(sum(int(m["counts"].get("total_pairs", 0)) for m in eval_infos))
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
            "num_nodes": sum(int(m["counts"]["num_nodes"]) for m in eval_infos),
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
    """
    DR logging for backbone+task-head APPLE.

    Logged objects:
        backbone matrix: [receiver_client, donor_client]
        task tensor:     [receiver_client, task, donor_client]

    Rows written:
        row_type="backbone_summary"   one row per receiver-client
        row_type="backbone_pair"      one row per receiver-client/donor-client
        row_type="task_summary"       one row per receiver-client/task
        row_type="task_pair"          one row per receiver-client/task/donor-client
    """
    task_tensor = dr_matrix_cpu(dr_vectors)
    backbone_matrix = dr_backbone_matrix_cpu(dr_vectors)

    if task_tensor.ndim != 3:
        raise ValueError(
            f"Expected task DR tensor shape [num_clients, num_tasks, num_clients], "
            f"got shape={tuple(task_tensor.shape)}"
        )
    if backbone_matrix.ndim != 2:
        raise ValueError(
            f"Expected backbone DR matrix shape [num_clients, num_clients], "
            f"got shape={tuple(backbone_matrix.shape)}"
        )

    p0_cpu = p0.detach().cpu()
    num_receivers, num_tasks, num_donors = task_tensor.shape

    if num_receivers != len(runtime_clients) or num_donors != len(runtime_clients):
        raise ValueError(
            f"Task DR tensor/client mismatch: tensor={tuple(task_tensor.shape)}, "
            f"runtime_clients={len(runtime_clients)}"
        )
    if backbone_matrix.shape != (len(runtime_clients), len(runtime_clients)):
        raise ValueError(
            f"Backbone DR/client mismatch: matrix={tuple(backbone_matrix.shape)}, "
            f"runtime_clients={len(runtime_clients)}"
        )

    incoming_abs_task = task_tensor.abs().sum(dim=0)  # [task, donor]
    incoming_signed_task = task_tensor.sum(dim=0)  # [task, donor]
    incoming_abs_backbone = backbone_matrix.abs().sum(dim=0)  # [donor]
    incoming_signed_backbone = backbone_matrix.sum(dim=0)  # [donor]

    for i, runtime_i in enumerate(runtime_clients):
        meta_i = _client_extra_metadata(runtime_i, client_metadata)

        # Backbone rows.
        brow = backbone_matrix[i]
        boff_mask = torch.ones(num_donors, dtype=torch.bool)
        boff_mask[i] = False
        boff = brow[boff_mask]
        b_abs_row = brow.abs()
        b_top_order = torch.argsort(b_abs_row, descending=True).tolist()
        b_top_parts = []
        for donor_idx in b_top_order[: min(3, num_donors)]:
            donor_graph = str(runtime_clients[donor_idx].client.graph_id)
            b_top_parts.append(f"{donor_graph}:{float(brow[donor_idx].item()):.6g}")

        dr_rows.append(
            {
                "row_type": "backbone_summary",
                "run_type": _exp_algorithm(),
                "algorithm": _exp_algorithm(),
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
                "task": "backbone",
                "dr_self_weight": float(brow[i].item()),
                "dr_row_sum": float(brow.sum().item()),
                "dr_row_abs_sum": float(brow.abs().sum().item()),
                "dr_offdiag_pos_mass": float(torch.clamp(boff, min=0.0).sum().item()),
                "dr_offdiag_neg_mass": float(torch.clamp(boff, max=0.0).sum().item()),
                "dr_offdiag_abs_mass": float(boff.abs().sum().item()),
                "dr_l2_to_p0": float(torch.norm(brow - p0_cpu, p=2).item()),
                "dr_top_abs_donors": "|".join(b_top_parts),
                "incoming_abs_mass_this_client_task": None,
                "incoming_signed_mass_this_client_task": None,
                "incoming_abs_mass_this_client_backbone": float(
                    incoming_abs_backbone[i].item()
                ),
                "incoming_signed_mass_this_client_backbone": float(
                    incoming_signed_backbone[i].item()
                ),
                "dr_vector_json": json.dumps([float(x) for x in brow.tolist()]),
                "p0_json": json.dumps([float(x) for x in p0_cpu.tolist()]),
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
                    "apple_lambda": float(lambda_r),
                    "apple_mu": float(apple_mu),
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
                    "abs_p_itj": float(abs(value)),
                    "abs_p_ij": float(abs(value)),
                    "abs_p_backbone_ij": float(abs(value)),
                    "p0_j": float(p0_cpu[j].item()),
                    "is_self": int(i == j),
                }
            )

        # Task-head rows.
        for task_idx in range(num_tasks):
            task_name = TASKS[task_idx] if task_idx < len(TASKS) else f"task{task_idx}"
            row = task_tensor[i, task_idx]

            off_mask = torch.ones(num_donors, dtype=torch.bool)
            off_mask[i] = False
            off = row[off_mask]

            pos_off = torch.clamp(off, min=0.0).sum().item()
            neg_off = torch.clamp(off, max=0.0).sum().item()
            abs_off = off.abs().sum().item()

            abs_row = row.abs()
            top_order = torch.argsort(abs_row, descending=True).tolist()
            top_parts = []
            for donor_idx in top_order[: min(3, num_donors)]:
                donor_graph = str(runtime_clients[donor_idx].client.graph_id)
                top_parts.append(f"{donor_graph}:{float(row[donor_idx].item()):.6g}")

            dr_rows.append(
                {
                    "row_type": "task_summary",
                    "run_type": _exp_algorithm(),
                    "algorithm": _exp_algorithm(),
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
                    "task_idx": task_idx,
                    "task": task_name,
                    "dr_self_weight": float(row[i].item()),
                    "dr_row_sum": float(row.sum().item()),
                    "dr_row_abs_sum": float(row.abs().sum().item()),
                    "dr_offdiag_pos_mass": float(pos_off),
                    "dr_offdiag_neg_mass": float(neg_off),
                    "dr_offdiag_abs_mass": float(abs_off),
                    "dr_l2_to_p0": float(torch.norm(row - p0_cpu, p=2).item()),
                    "dr_top_abs_donors": "|".join(top_parts),
                    "incoming_abs_mass_this_client_task": float(
                        incoming_abs_task[task_idx, i].item()
                    ),
                    "incoming_signed_mass_this_client_task": float(
                        incoming_signed_task[task_idx, i].item()
                    ),
                    "incoming_abs_mass_this_client_backbone": None,
                    "incoming_signed_mass_this_client_backbone": None,
                    "dr_vector_json": json.dumps([float(x) for x in row.tolist()]),
                    "p0_json": json.dumps([float(x) for x in p0_cpu.tolist()]),
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
                        "apple_lambda": float(lambda_r),
                        "apple_mu": float(apple_mu),
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
                        "abs_p_itj": float(abs(value)),
                        "abs_p_ij": float(abs(value)),
                        "abs_p_task_itj": float(abs(value)),
                        "p0_j": float(p0_cpu[j].item()),
                        "is_self": int(i == j),
                    }
                )


# -----------------------------------------------------------------------------
# APPLE + post-training HeadOnly-FedALA helpers
# -----------------------------------------------------------------------------
def apple_personalized_state_dict(
    models_for_mix: Sequence[torch.nn.Module],
    client_idx: int,
    dr_vector: AppleDRState,
) -> Dict[str, torch.Tensor]:
    """Return the APPLE-personalized state used by the post-training ALA filter.

    The state is a full state_dict. Non-head entries/buffers come from the
    receiver model; task-head parameters are the current APPLE task-wise donor
    mixture without ALA gating.
    """
    receiver = models_for_mix[int(client_idx)]
    out = {
        key: value.detach().cpu().clone()
        for key, value in receiver.state_dict().items()
    }
    mixed_params = mixed_parameter_dict(
        models_for_mix,
        int(client_idx),
        dr_vector,
        ala_weights=None,
    )
    for name, value in mixed_params.items():
        out[name] = value.detach().cpu().clone()
    return out


def append_apple_ala_diag_row(
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



# -----------------------------------------------------------------------------
# Passive backbone-affinity diagnostics
# -----------------------------------------------------------------------------
# BACKBONE_AFFINITY_PROBE_IMPLEMENTED_V1


def affinity_probe_output_paths(
    run_paths: AppleRunPaths,
) -> Dict[str, Path]:
    stem = run_paths.csv_path.stem
    root = run_paths.csv_path.parent

    return {
        "clients": root / f"{stem}_affinity_clients.csv",
        "layers": root / f"{stem}_affinity_layers.csv",
        "pairs": root / f"{stem}_affinity_pairs.csv",
        "parameter_pairs": root / f"{stem}_affinity_parameter_pairs.csv",
    }


def capture_backbone_parameter_updates(
    model: torch.nn.Module,
    start_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return local backbone parameter changes on CPU in float32.

    Task-head parameters and all non-trainable buffers are excluded.
    """
    updates: Dict[str, torch.Tensor] = {}

    for name, parameter in model.named_parameters():
        if _is_task_head_state_name(name):
            continue

        if name not in start_state:
            raise KeyError(
                f"Missing start-of-round parameter: {name}"
            )

        current = (
            parameter.detach()
            .cpu()
            .to(dtype=torch.float32)
        )
        start = (
            start_state[name]
            .detach()
            .cpu()
            .to(dtype=torch.float32)
        )

        if current.shape != start.shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"current={tuple(current.shape)}, "
                f"start={tuple(start.shape)}"
            )

        updates[name] = current - start

    if not updates:
        raise ValueError(
            "No backbone parameters were captured."
        )

    return updates


def _flatten_named_tensors(
    tensors: Dict[str, torch.Tensor],
    names: Sequence[str],
) -> torch.Tensor:
    return torch.cat(
        [
            tensors[name]
            .reshape(-1)
            .to(dtype=torch.float32)
            for name in names
        ],
        dim=0,
    )


def _safe_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    left = left.reshape(-1).to(dtype=torch.float32)
    right = right.reshape(-1).to(dtype=torch.float32)

    denominator = (
        float(torch.norm(left, p=2).item())
        * float(torch.norm(right, p=2).item())
    )

    if denominator <= eps:
        return float("nan")

    return float(
        torch.dot(left, right).item()
        / denominator
    )


def _finite_summary(
    values: Sequence[float],
) -> Dict[str, float]:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if np.isfinite(value)
        ],
        dtype=np.float64,
    )

    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _weighted_finite_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("values/weights length mismatch")
    pairs = [
        (float(v), float(w))
        for v, w in zip(values, weights)
        if np.isfinite(v) and np.isfinite(w) and float(w) > 0.0
    ]
    if not pairs:
        return float("nan")
    return float(
        sum(v * w for v, w in pairs)
        / max(sum(w for _, w in pairs), 1e-12)
    )


def append_backbone_affinity_round(
    *,
    client_rows: List[Dict],
    layer_rows: List[Dict],
    pair_rows: List[Dict],
    parameter_pair_rows: List[Dict],
    runtime_clients,
    selected_indices: Sequence[int],
    round_updates: Dict[
        int,
        Dict[str, torch.Tensor],
    ],
    round_start_states: Sequence[
        Dict[str, torch.Tensor]
    ],
    subset_id: str,
    subset_clients: str,
    seed: int,
    round_idx: int,
    client_metadata: Optional[
        Dict[str, Dict]
    ] = None,
) -> None:
    selected_indices = [
        int(idx)
        for idx in selected_indices
    ]

    if len(selected_indices) < 2:
        raise ValueError(
            "Affinity diagnostics require at least "
            "two selected clients."
        )

    if set(selected_indices) != set(
        round_updates
    ):
        raise ValueError(
            "Selected-client/update mismatch: "
            f"selected={selected_indices}, "
            f"updates={sorted(round_updates)}"
        )

    first_idx = selected_indices[0]
    parameter_names = list(
        round_updates[first_idx].keys()
    )

    if not parameter_names:
        raise ValueError(
            "No backbone parameter names found."
        )

    for idx in selected_indices:
        names = list(round_updates[idx].keys())

        if names != parameter_names:
            raise ValueError(
                f"Backbone parameter ordering differs "
                f"for client index {idx}."
            )

    update_vectors = torch.stack(
        [
            _flatten_named_tensors(
                round_updates[idx],
                parameter_names,
            )
            for idx in selected_indices
        ],
        dim=0,
    )

    centered_vectors = (
        update_vectors
        - update_vectors.mean(
            dim=0,
            keepdim=True,
        )
    )

    update_norms = torch.norm(
        update_vectors,
        p=2,
        dim=1,
    )
    centered_norms = torch.norm(
        centered_vectors,
        p=2,
        dim=1,
    )

    start_vectors = []

    for idx in selected_indices:
        start_named = {
            name: (
                round_start_states[idx][name]
                .detach()
                .cpu()
                .to(dtype=torch.float32)
            )
            for name in parameter_names
        }
        start_vectors.append(
            _flatten_named_tensors(
                start_named,
                parameter_names,
            )
        )

    start_vectors = torch.stack(
        start_vectors,
        dim=0,
    )

    start_reference = start_vectors[0]
    start_distances = torch.norm(
        start_vectors
        - start_reference.unsqueeze(0),
        p=2,
        dim=1,
    )
    start_max_distance = float(
        start_distances.max().item()
    )

    layer_means = {
        name: torch.stack(
            [
                round_updates[idx][name]
                for idx in selected_indices
            ],
            dim=0,
        ).mean(dim=0)
        for name in parameter_names
    }

    num_parameters = int(
        update_vectors.shape[1]
    )

    # --------------------------------------------------------------
    # Per-client and per-layer diagnostics
    # --------------------------------------------------------------
    for position, idx in enumerate(
        selected_indices
    ):
        runtime = runtime_clients[idx]
        meta = _client_extra_metadata(
            runtime,
            client_metadata,
        )

        update_norm = float(
            update_norms[position].item()
        )
        centered_norm = float(
            centered_norms[position].item()
        )
        start_norm = float(
            torch.norm(
                start_vectors[position],
                p=2,
            ).item()
        )

        client_rows.append(
            {
                "run_type": _exp_algorithm(),
                "algorithm": _exp_algorithm(),
                "subset_id": subset_id,
                "subset_clients": subset_clients,
                "seed": int(seed),
                "round": int(round_idx),
                "client_idx": int(idx),
                "graph_id": str(
                    runtime.client.graph_id
                ),
                "dataset_id": str(
                    runtime.client.dataset_id
                ),
                "num_train_nodes": int(
                    runtime.num_train_nodes
                ),
                "num_backbone_parameters": (
                    num_parameters
                ),
                "num_backbone_tensors": int(
                    len(parameter_names)
                ),
                "backbone_update_norm": (
                    update_norm
                ),
                "centered_backbone_update_norm": (
                    centered_norm
                ),
                "start_backbone_norm": start_norm,
                "relative_backbone_update_norm": (
                    update_norm
                    / max(start_norm, 1e-12)
                ),
                "start_backbone_l2_to_first": (
                    float(
                        start_distances[
                            position
                        ].item()
                    )
                ),
                # Analysis-only metadata. Never used
                # to construct affinities.
                "analysis_only_assigned_task": (
                    meta.get("assigned_task")
                ),
                "analysis_only_family": (
                    meta.get("family")
                ),
                "analysis_only_planted_community_id": (
                    meta.get(
                        "planted_community_id"
                    )
                ),
                "analysis_only_base_graph_id": (
                    meta.get("base_graph_id")
                ),
                "analysis_only_q_value": meta.get("q_value"),
            }
        )

        for name in parameter_names:
            update = round_updates[idx][name]
            start = (
                round_start_states[idx][name]
                .detach()
                .cpu()
                .to(dtype=torch.float32)
            )

            update_layer_norm = float(
                torch.norm(
                    update.reshape(-1),
                    p=2,
                ).item()
            )
            start_layer_norm = float(
                torch.norm(
                    start.reshape(-1),
                    p=2,
                ).item()
            )

            layer_rows.append(
                {
                    "run_type": (
                        _exp_algorithm()
                    ),
                    "algorithm": (
                        _exp_algorithm()
                    ),
                    "subset_id": subset_id,
                    "subset_clients": (
                        subset_clients
                    ),
                    "seed": int(seed),
                    "round": int(round_idx),
                    "client_idx": int(idx),
                    "graph_id": str(
                        runtime.client.graph_id
                    ),
                    "dataset_id": str(
                        runtime.client.dataset_id
                    ),
                    "parameter_name": name,
                    "parameter_numel": int(
                        update.numel()
                    ),
                    "update_norm": (
                        update_layer_norm
                    ),
                    "start_parameter_norm": (
                        start_layer_norm
                    ),
                    "relative_update_norm": (
                        update_layer_norm
                        / max(
                            start_layer_norm,
                            1e-12,
                        )
                    ),
                }
            )

    # --------------------------------------------------------------
    # Unordered pair diagnostics
    # --------------------------------------------------------------
    for left_position in range(
        len(selected_indices)
    ):
        for right_position in range(
            left_position + 1,
            len(selected_indices),
        ):
            left_idx = selected_indices[
                left_position
            ]
            right_idx = selected_indices[
                right_position
            ]

            left_runtime = runtime_clients[
                left_idx
            ]
            right_runtime = runtime_clients[
                right_idx
            ]

            left_vector = update_vectors[
                left_position
            ]
            right_vector = update_vectors[
                right_position
            ]

            left_centered = centered_vectors[
                left_position
            ]
            right_centered = centered_vectors[
                right_position
            ]

            left_norm = float(
                update_norms[
                    left_position
                ].item()
            )
            right_norm = float(
                update_norms[
                    right_position
                ].item()
            )

            raw_cosine = _safe_cosine(
                left_vector,
                right_vector,
            )
            centered_cosine = _safe_cosine(
                left_centered,
                right_centered,
            )

            difference_norm = float(
                torch.norm(
                    left_vector
                    - right_vector,
                    p=2,
                ).item()
            )

            mean_norm = 0.5 * (
                left_norm + right_norm
            )
            relative_l2 = (
                difference_norm
                / max(mean_norm, 1e-12)
            )

            left_meta = _client_extra_metadata(left_runtime, client_metadata)
            right_meta = _client_extra_metadata(right_runtime, client_metadata)
            left_community = left_meta.get("planted_community_id")
            right_community = right_meta.get("planted_community_id")
            left_task = left_meta.get("assigned_task")
            right_task = right_meta.get("assigned_task")

            raw_layer_cosines = []
            centered_layer_cosines = []
            parameter_numels = []

            for name in parameter_names:
                left_layer = round_updates[left_idx][name]
                right_layer = round_updates[right_idx][name]
                layer_mean = layer_means[name]

                raw_parameter_cosine = _safe_cosine(left_layer, right_layer)
                centered_parameter_cosine = _safe_cosine(
                    left_layer - layer_mean,
                    right_layer - layer_mean,
                )
                parameter_numel = int(left_layer.numel())
                left_parameter_norm = float(torch.norm(left_layer.reshape(-1), p=2).item())
                right_parameter_norm = float(torch.norm(right_layer.reshape(-1), p=2).item())
                parameter_difference_l2 = float(
                    torch.norm(
                        left_layer.reshape(-1) - right_layer.reshape(-1),
                        p=2,
                    ).item()
                )
                parameter_mean_norm = 0.5 * (
                    left_parameter_norm + right_parameter_norm
                )

                raw_layer_cosines.append(raw_parameter_cosine)
                centered_layer_cosines.append(centered_parameter_cosine)
                parameter_numels.append(parameter_numel)

                parameter_pair_rows.append(
                    {
                        "run_type": _exp_algorithm(),
                        "algorithm": _exp_algorithm(),
                        "subset_id": subset_id,
                        "subset_clients": subset_clients,
                        "seed": int(seed),
                        "round": int(round_idx),
                        "left_client_idx": int(left_idx),
                        "right_client_idx": int(right_idx),
                        "left_graph_id": str(left_runtime.client.graph_id),
                        "right_graph_id": str(right_runtime.client.graph_id),
                        "parameter_name": name,
                        "parameter_numel": parameter_numel,
                        "left_parameter_update_norm": left_parameter_norm,
                        "right_parameter_update_norm": right_parameter_norm,
                        "parameter_update_difference_l2": parameter_difference_l2,
                        "parameter_relative_update_l2": parameter_difference_l2 / max(parameter_mean_norm, 1e-12),
                        "raw_cosine": raw_parameter_cosine,
                        "global_centered_cosine": centered_parameter_cosine,
                        "analysis_only_left_assigned_task": left_task,
                        "analysis_only_right_assigned_task": right_task,
                        "analysis_only_left_planted_community_id": left_community,
                        "analysis_only_right_planted_community_id": right_community,
                        "analysis_only_same_task": int(
                            left_task is not None and right_task is not None
                            and str(left_task) == str(right_task)
                        ),
                        "analysis_only_same_community": int(
                            left_community is not None and right_community is not None
                            and str(left_community) == str(right_community)
                        ),
                    }
                )

            raw_layer_summary = (
                _finite_summary(
                    raw_layer_cosines
                )
            )
            centered_layer_summary = (
                _finite_summary(
                    centered_layer_cosines
                )
            )

            pair_rows.append(
                {
                    "run_type": (
                        _exp_algorithm()
                    ),
                    "algorithm": (
                        _exp_algorithm()
                    ),
                    "subset_id": subset_id,
                    "subset_clients": (
                        subset_clients
                    ),
                    "seed": int(seed),
                    "round": int(round_idx),
                    "left_client_idx": int(
                        left_idx
                    ),
                    "right_client_idx": int(
                        right_idx
                    ),
                    "left_graph_id": str(
                        left_runtime.client.graph_id
                    ),
                    "right_graph_id": str(
                        right_runtime.client.graph_id
                    ),
                    "left_dataset_id": str(
                        left_runtime.client.dataset_id
                    ),
                    "right_dataset_id": str(
                        right_runtime.client.dataset_id
                    ),
                    "left_update_norm": (
                        left_norm
                    ),
                    "right_update_norm": (
                        right_norm
                    ),
                    "update_norm_abs_difference": (
                        abs(
                            left_norm
                            - right_norm
                        )
                    ),
                    "update_norm_log_ratio": (
                        float(
                            abs(
                                math.log(
                                    max(
                                        left_norm,
                                        1e-12,
                                    )
                                    / max(
                                        right_norm,
                                        1e-12,
                                    )
                                )
                            )
                        )
                    ),
                    "raw_cosine": raw_cosine,
                    "global_centered_cosine": (
                        centered_cosine
                    ),
                    "update_difference_l2": (
                        difference_norm
                    ),
                    "relative_update_l2": (
                        relative_l2
                    ),
                    "layerwise_raw_cosine_mean": (
                        raw_layer_summary["mean"]
                    ),
                    "layerwise_raw_cosine_std": (
                        raw_layer_summary["std"]
                    ),
                    "layerwise_raw_cosine_min": (
                        raw_layer_summary["min"]
                    ),
                    "layerwise_raw_cosine_max": (
                        raw_layer_summary["max"]
                    ),
                    "layerwise_centered_cosine_mean": (
                        centered_layer_summary[
                            "mean"
                        ]
                    ),
                    "layerwise_centered_cosine_std": (
                        centered_layer_summary[
                            "std"
                        ]
                    ),
                    "layerwise_centered_cosine_min": (
                        centered_layer_summary[
                            "min"
                        ]
                    ),
                    "layerwise_centered_cosine_max": (
                        centered_layer_summary[
                            "max"
                        ]
                    ),
                    "parameterwise_raw_cosine_mean": raw_layer_summary["mean"],
                    "parameterwise_centered_cosine_mean": centered_layer_summary["mean"],
                    "numel_weighted_parameter_raw_cosine": _weighted_finite_mean(
                        raw_layer_cosines, parameter_numels
                    ),
                    "numel_weighted_parameter_centered_cosine": _weighted_finite_mean(
                        centered_layer_cosines, parameter_numels
                    ),
                    "analysis_only_left_assigned_task": left_task,
                    "analysis_only_right_assigned_task": right_task,
                    "analysis_only_left_planted_community_id": left_community,
                    "analysis_only_right_planted_community_id": right_community,
                    "analysis_only_left_base_graph_id": left_meta.get("base_graph_id"),
                    "analysis_only_right_base_graph_id": right_meta.get("base_graph_id"),
                    "analysis_only_same_task": int(
                        left_task is not None and right_task is not None
                        and str(left_task) == str(right_task)
                    ),
                    "analysis_only_same_community": int(
                        left_community is not None and right_community is not None
                        and str(left_community) == str(right_community)
                    ),
                }
            )

    expected_pairs = (
        len(selected_indices)
        * (len(selected_indices) - 1)
        // 2
    )

    print(
        "[BACKBONE AFFINITY PROBE]",
        f"round={round_idx}",
        f"clients={len(selected_indices)}",
        f"parameters={num_parameters}",
        f"unordered_pairs={expected_pairs}",
        (
            "start_backbone_max_l2_to_first="
            f"{start_max_distance:.8g}"
        ),
    )


def write_affinity_probe_tables(
    *,
    run_paths: AppleRunPaths,
    client_rows: Sequence[Dict],
    layer_rows: Sequence[Dict],
    pair_rows: Sequence[Dict],
    parameter_pair_rows: Sequence[Dict],
) -> None:
    paths = affinity_probe_output_paths(
        run_paths
    )

    pd.DataFrame(client_rows).to_csv(
        paths["clients"],
        index=False,
    )
    pd.DataFrame(layer_rows).to_csv(
        paths["layers"],
        index=False,
    )
    pd.DataFrame(pair_rows).to_csv(
        paths["pairs"],
        index=False,
    )
    pd.DataFrame(parameter_pair_rows).to_csv(
        paths["parameter_pairs"],
        index=False,
    )

# -----------------------------------------------------------------------------
# Main APPLE experiment
# -----------------------------------------------------------------------------
def run_apple_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: AppleRunPaths,
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
    selection_metrics: Optional[Sequence[str]] = None,
    selection_metric: Optional[str] = None,
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

    selection_metrics = normalize_selection_metrics(selection_metrics, selection_metric)

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
    expert_rho = float(cfg.get("apple_expert_rho", 0.8))
    expert_pseudo_count = float(cfg.get("apple_expert_pseudo_count", 1.0))
    use_head_ala = bool(cfg.get("apple_use_head_ala", False))
    ala_lr = float(cfg.get("apple_ala_lr", 1.0))
    ala_rand_percent = float(cfg.get("apple_ala_rand_percent", 20.0))
    ala_convergence_std = float(cfg.get("apple_ala_convergence_std", 0.1))
    ala_convergence_window = int(cfg.get("apple_ala_convergence_window", 10))
    ala_max_steps = int(cfg.get("apple_ala_max_steps", 100))
    ala_debug = bool(cfg.get("apple_ala_debug", False))

    dr_vectors = initialize_dr_vectors(
        num_clients,
        num_tasks,
        p0,
        init=dr_init,
        runtime_clients=runtime_clients,
        device=device,
        expert_rho=expert_rho,
        expert_pseudo_count=expert_pseudo_count,
    )
    ala_states = [FedALAClientState() for _ in range(num_clients)]
    current_ala_weights_by_client: Dict[int, Dict[str, torch.Tensor]] = {
        i: {} for i in range(num_clients)
    }

    print(
        "[fedavg-backbone+task-head apple post-ala]",
        "num_clients=",
        num_clients,
        "num_tasks=",
        num_tasks,
        "output_head=",
        cfg.get("output_head"),
        "mixing_mode=",
        cfg.get("apple_mixing_mode"),
        "task_dr_tensor_shape=",
        tuple(dr_matrix_cpu(dr_vectors).shape),
        "backbone=fedavg",
        "post_apple_ala_filter=",
        use_head_ala,
        "expert_rho=",
        expert_rho,
        "expert_pseudo_count=",
        expert_pseudo_count,
    )

    rows: List[Dict] = []
    dr_rows: List[Dict] = []

    affinity_client_rows: List[Dict] = []
    affinity_layer_rows: List[Dict] = []
    affinity_pair_rows: List[Dict] = []
    affinity_parameter_pair_rows: List[Dict] = []

    initial_core_states = core_state_dicts(core_models)
    initial_local_eval_states_by_client = {
        i: core_state_dicts(core_models) for i in range(num_clients)
    }
    initial_dr_matrix = dr_matrix_cpu(dr_vectors)
    initial_backbone_dr_matrix = dr_backbone_matrix_cpu(dr_vectors)

    def _initial_best_record():
        record = {
            "value": None,
            "round": -1,
            "server_core_states": copy.deepcopy(initial_core_states),
            "local_eval_states_by_client": copy.deepcopy(initial_local_eval_states_by_client),
            "dr_matrix": initial_dr_matrix.clone(),
            "backbone_dr_matrix": initial_backbone_dr_matrix.clone(),
            "ala_weights_by_client": {i: {} for i in range(num_clients)},
        }
        return record

    best_by_eval_mode = {
        eval_mode: {
            metric: _initial_best_record()
            for metric in selection_metrics
        }
        for eval_mode in ("full", "visible")
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

        if num_selected != num_clients:
            raise ValueError(
                "Backbone-affinity probe requires full client "
                "participation. Set CLIENT_FRACTION=1.0."
            )

        # Snapshot the server's core models at the start of the round.
        # This keeps the sequential Python loop faithful to APPLE's parallel round semantics:
        # every client trains using the same downloaded start-of-round cores.
        round_start_states = core_state_dicts(core_models)
        downloaded_core_models = [
            clone_model_from_state(state, cfg, ctx, device)
            for state in round_start_states
        ]

        updated_models: Dict[int, torch.nn.Module] = {}
        round_backbone_updates: Dict[
            int,
            Dict[str, torch.Tensor],
        ] = {}

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

            # In this POST-ALA variant, ALA is intentionally NOT learned before
            # local training. First we train the APPLE model normally, then we
            # use ALA as a post-training filter on the resulting APPLE-personalized
            # task heads.
            ala_weights_for_idx = None

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

            # Capture the receiver's backbone change after local
            # APPLE training and before any server aggregation.
            # PostALA is head-only, so this is the clean backbone
            # update produced by the local client.
            round_backbone_updates[idx] = (
                capture_backbone_parameter_updates(
                    local_model,
                    round_start_states[idx],
                )
            )

            # ------------------------------------------------------------
            # POST-APPLE ALA FILTER
            # ------------------------------------------------------------
            # Local APPLE training above updated:
            #   (1) the receiver client's model parameters, and
            #   (2) the receiver client's APPLE task donor weights.
            #
            # Now we materialize the APPLE-personalized task-head state and let
            # FedALA learn alpha between:
            #   old_local_state = the trained receiver-local state
            #   global_state    = the trained APPLE-personalized state
            #
            # The returned filtered state is loaded back into the client model.
            # Therefore the FedALA filter affects validation, checkpoint selection,
            # FedAvg backbone aggregation, and the next round.
            if use_head_ala:
                trained_local_state = clone_state_dict(local_model.state_dict())
                incoming_state = apple_personalized_state_dict(
                    models_for_mix,
                    idx,
                    dr_vectors[idx],
                )
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
                    {k: v.to(device=device) for k, v in ala_filtered_state.items()},
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
                append_apple_ala_diag_row(
                    rows,
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    round_idx=round_idx,
                    runtime=runtime,
                    client_idx=idx,
                    ala_stats=ala_stats,
                )

            round_local_eval_states_by_client[idx] = core_state_dicts(models_for_mix)

            for eval_mask_mode in ("full", "visible"):
                # Evaluate the materialized, post-filtered local model directly.
                # We do not apply another APPLE mix here; the ALA-filtered state
                # already contains the accepted APPLE task-head transfer.
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

        append_backbone_affinity_round(
            client_rows=affinity_client_rows,
            layer_rows=affinity_layer_rows,
            pair_rows=affinity_pair_rows,
            parameter_pair_rows=affinity_parameter_pair_rows,
            runtime_clients=runtime_clients,
            selected_indices=selected_indices,
            round_updates=round_backbone_updates,
            round_start_states=round_start_states,
            subset_id=subset_id,
            subset_clients=subset_clients_str,
            seed=seed,
            round_idx=round_idx,
            client_metadata=client_metadata,
        )

        # Save every round so diagnostic evidence survives a later
        # interruption during validation or final evaluation.
        write_affinity_probe_tables(
            run_paths=run_paths,
            client_rows=affinity_client_rows,
            layer_rows=affinity_layer_rows,
            pair_rows=affinity_pair_rows,
            parameter_pair_rows=affinity_parameter_pair_rows,
        )

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

        # FedAvg backbone aggregation:
        # average only non-task-head parameters using selected clients' train-node
        # counts, then copy the averaged backbone into every client model.
        selected_models_for_backbone = [core_models[idx] for idx in selected_indices]
        selected_runtimes_for_backbone = [
            runtime_clients[idx] for idx in selected_indices
        ]

        fedavg_backbone_state = fedavg_backbone_state_dict(
            selected_models_for_backbone,
            selected_runtimes_for_backbone,
        )

        load_backbone_state_into_models(core_models, fedavg_backbone_state)

        round_eval_infos_by_mode: Dict[str, List[Dict]] = {
            "full": [],
            "visible": [],
        }

        for eval_mask_mode in ("full", "visible"):
            for idx, runtime in enumerate(runtime_clients):
                # Server-round validation uses the materialized local model
                # after post-APPLE ALA filtering and FedAvg backbone aggregation.
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
            for selector_metric in selection_metrics:
                current_value = local_mean_scalar.get(selector_metric)
                best_record = best_by_eval_mode[eval_mask_mode][selector_metric]

                print(
                    "current local-update APPLE vs best:",
                    "eval_mask_mode:",
                    eval_mask_mode,
                    "selector_metric:",
                    selector_metric,
                    "current:",
                    current_value,
                    "best:",
                    best_record["value"],
                )

                if is_better_selection_value(
                    current_value,
                    initial_selection_value(selector_metric) if best_record["value"] is None else best_record["value"],
                    selector_metric,
                ):
                    best_record["value"] = float(current_value)
                    best_record["round"] = round_idx
                    best_record["local_eval_states_by_client"] = copy.deepcopy(
                        round_local_eval_states_by_client
                    )
                    best_record["server_core_states"] = core_state_dicts(core_models)
                    best_record["dr_matrix"] = dr_matrix_cpu(dr_vectors)
                    best_record["backbone_dr_matrix"] = dr_backbone_matrix_cpu(
                        dr_vectors
                    )
                    best_record["ala_weights_by_client"] = _copy_ala_weights_by_client(
                        round_local_eval_ala_weights_by_client
                    )

        run_elapsed = time.perf_counter() - run_start_time
        print(f"run time: {format_seconds(run_elapsed)}")
        print("=======================================================")
        print(f"==================== round {round_idx} / {rounds} ====================")
        print("=======================================================")

        # Keep GPU memory usage predictable across rounds.
        del downloaded_core_models

    checkpoint = {
        "selection_metric": "multi",
        "selection_metrics": list(selection_metrics),
        "selection_direction": "mixed",
        "best_round_by_eval_mode_and_metric": {
            mode: {metric: best_by_eval_mode[mode][metric]["round"] for metric in selection_metrics}
            for mode in ("full", "visible")
        },
        "best_value_by_eval_mode_and_metric": {
            mode: {metric: best_by_eval_mode[mode][metric]["value"] for metric in selection_metrics}
            for mode in ("full", "visible")
        },
        "cfg": cfg,
        "seed": seed,
        "subset_id": subset_id,
        "subset_clients": subset_clients_str,
        "apple_mixing_mode": cfg.get("apple_mixing_mode"),
        "output_head": cfg.get("output_head", "multi"),
        "num_tasks": num_tasks,
        "p0": p0.detach().cpu().clone(),
        "apple_dr_lr": dr_lr,
        "apple_mu": apple_mu,
        "apple_scheduler_type": scheduler_type,
        "apple_scheduler_fraction": scheduler_fraction,
        "apple_dr_init": dr_init,
        "apple_dr_constraint": dr_constraint,
        "apple_download_strategy": download_strategy,
        "apple_use_head_ala": use_head_ala,
        "apple_ala_lr": ala_lr,
        "apple_ala_rand_percent": ala_rand_percent,
        "apple_ala_convergence_std": ala_convergence_std,
        "apple_ala_convergence_window": ala_convergence_window,
        "apple_ala_max_steps": ala_max_steps,
        "apple_expert_rho": expert_rho,
        "apple_expert_pseudo_count": expert_pseudo_count,
        "eval_protocols": ["oracle_full", "realistic_visible", "realistic_selection_oracle"],
    }
    maybe_save_checkpoint(checkpoint, run_paths.ckpt_path)

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

    for selector_metric in selection_metrics:
        for job in final_eval_jobs:
            selection_protocol = job["selection_protocol"]
            selected_by = job["selected_by"]
            eval_mask_mode = job["eval_mask_mode"]

            selected_state = best_by_eval_mode[selected_by][selector_metric]
            selected_round = selected_state["round"]
            selected_dr_matrix = selected_state["dr_matrix"]
            selected_ala_weights_by_client = selected_state.get(
                "ala_weights_by_client",
                {},
            )

            if "backbone_dr_matrix" in selected_state:
                selected_backbone_dr_matrix = selected_state["backbone_dr_matrix"]
            else:
                selected_backbone_dr_matrix = torch.stack([p0.detach().cpu().clone() for _ in range(selected_dr_matrix.size(0))], dim=0)

            selected_dr_vectors = [
                AppleDRState(
                    backbone=torch.nn.Parameter(selected_backbone_dr_matrix[i].to(device).clone(), requires_grad=False),
                    task=torch.nn.Parameter(selected_dr_matrix[i].to(device).clone(), requires_grad=False),
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

                    # Final evaluation is also direct evaluation of the materialized
                    # post-filtered client model selected for this client.
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
                        phase=f"best_{selection_protocol}_{selector_metric}_{split_name}",
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
                        selection_metrics="|".join(selection_metrics),
                        selector_metric=selector_metric,
                        selector_direction=selection_direction(selector_metric),
                        selected_by_eval_mode=selected_by,
                        selected_round=selected_round,
                        selected_epoch=None,
                        best_val_metric_value=selected_state["value"],
                    )

                append_apple_mean_row(
                    rows,
                    phase=f"best_{selection_protocol}_{selector_metric}_{split_name}_mean",
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
                        # Keep final mean rows as diagnostics, but do not count them
                        # as per-client final scalar test rows in structural checks.
                        "mean_selection_protocol": selection_protocol,
                        "selection_protocol": None,
                        "selected_by": None,
                        "selection_metrics": "|".join(selection_metrics),
                        "selector_metric": selector_metric,
                        "selector_direction": selection_direction(selector_metric),
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
    print("checkpoint path:", run_paths.ckpt_path)
    print("best rounds by eval mode and selector:", {mode: {m: best_by_eval_mode[mode][m]["round"] for m in selection_metrics} for mode in ("full", "visible")})
    print("best values by eval mode and selector:", {mode: {m: best_by_eval_mode[mode][m]["value"] for m in selection_metrics} for mode in ("full", "visible")})

    final_rows = out_df[
        (out_df["task"].isna())
        & (out_df["split"] == "test")
        & (
            out_df["selection_protocol"].isin(
                [
                    "oracle_full",
                    "realistic_visible",
                    "realistic_selection_oracle",
                ]
            )
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

    write_affinity_probe_tables(
        run_paths=run_paths,
        client_rows=affinity_client_rows,
        layer_rows=affinity_layer_rows,
        pair_rows=affinity_pair_rows,
        parameter_pair_rows=affinity_parameter_pair_rows,
    )

    affinity_paths = affinity_probe_output_paths(
        run_paths
    )

    print(f"saved csv -> {run_paths.csv_path}")
    print(f"saved DR csv -> {run_paths.dr_csv_path}")
    print(
        "saved affinity client csv ->",
        affinity_paths["clients"],
    )
    print(
        "saved affinity layer csv ->",
        affinity_paths["layers"],
    )
    print(
        "saved affinity pair csv ->",
        affinity_paths["pairs"],
    )
    print(
        "saved affinity parameter-pair csv ->",
        affinity_paths["parameter_pairs"],
    )


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
    selection_metrics: Optional[Sequence[str]] = None,
    selection_metric: Optional[str] = None,
    client_metadata: Optional[Dict[str, Dict]] = None,
) -> AppleRunPaths:
    subset_id = _subset_clients_str(subset_clients)
    subset_file_tag = compact_subset_id_for_filename(subset_id)
    model_tag = _model_tag_from_cfg(cfg)

    run_paths = create_apple_run_paths(
        apple_root,
        subset_file_tag,
        rounds,
        local_epochs,
        dr_lr,
        apple_mu,
        scheduler_type,
        scheduler_fraction,
        model_tag,
        seed,
    )

    affinity_paths = affinity_probe_output_paths(
        run_paths
    )

    if (
        run_paths.csv_path.exists()
        and run_paths.dr_csv_path.exists()
        and affinity_paths["clients"].exists()
        and affinity_paths["layers"].exists()
        and affinity_paths["pairs"].exists()
        and affinity_paths["parameter_pairs"].exists()
    ):
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
        selection_metrics=selection_metrics,
        selection_metric=selection_metric,
        client_metadata=client_metadata,
    )
    return run_paths
