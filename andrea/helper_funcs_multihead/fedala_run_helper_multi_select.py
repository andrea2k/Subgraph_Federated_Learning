from __future__ import annotations

import math
import hashlib
import time
from dataclasses import dataclass, field
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
    fedavg_state_dict,
    make_model,
    make_optimizer,
    weighted_scalar_summary,
    normalize_selection_metrics,
    selection_direction,
    initial_selection_value,
    is_better_selection_value,
    maybe_save_checkpoint,
    tag_recent_eval_rows,
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
from utils.seed import set_seed

try:
    from torch.func import functional_call as _torch_functional_call
except Exception:  # pragma: no cover - older PyTorch fallback
    from torch.nn.utils.stateless import functional_call as _torch_functional_call


@dataclass(frozen=True)
class FedALARunPaths:
    csv_path: Path
    ckpt_path: Path
    ala_csv_path: Path


@dataclass
class FedALAClientState:
    """Client-side ALA state.

    weights maps parameter name -> alpha tensor. Alpha is clipped to [0, 1].

    alpha = 1 means fully accept the downloaded global parameter.
    alpha = 0 means preserve the old local parameter.
    """

    weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    start_phase: bool = True


# -----------------------------------------------------------------------------
# Filename helpers
# -----------------------------------------------------------------------------
def compact_subset_id_for_filename(subset_id: str, max_chars: int = 80) -> str:
    import hashlib

    raw = str(subset_id)
    if len(raw) <= max_chars and "/" not in raw and "\\" not in raw:
        return raw

    parts = [p for p in raw.split("|") if p]
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    if len(parts) >= 2:
        return f"n{len(parts)}_{parts[0]}_{parts[-1]}_{digest}"
    return f"subset_{digest}"


def safe_stem_for_suffixes(
    stem: str,
    suffixes: Sequence[str],
    *,
    max_filename_bytes: int = 255,
) -> str:
    """Return a deterministic filename stem safe for every requested suffix.

    Linux/macOS filesystems commonly cap a single filename component at 255
    bytes.  We preserve the original stem whenever it is already safe.  Only
    overlong stems are shortened, and a SHA-1 digest keeps the shortened name
    deterministic and collision-resistant.
    """
    stem = str(stem)
    suffixes = tuple(str(s) for s in suffixes)
    if not suffixes:
        raise ValueError("suffixes must not be empty")

    def _nbytes(value: str) -> int:
        return len(value.encode("utf-8"))

    if max(_nbytes(stem + suffix) for suffix in suffixes) <= int(max_filename_bytes):
        return stem

    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
    longest_suffix_bytes = max(_nbytes(suffix) for suffix in suffixes)
    digest_reserve = _nbytes("_" + digest)
    prefix_budget = int(max_filename_bytes) - longest_suffix_bytes - digest_reserve
    if prefix_budget < 1:
        raise RuntimeError(
            "Suffix leaves no room for a safe filename stem: "
            f"suffixes={suffixes!r}"
        )

    prefix = stem
    while prefix and _nbytes(prefix) > prefix_budget:
        prefix = prefix[:-1]
    prefix = prefix.rstrip("._-") or "run"

    safe_stem = f"{prefix}_{digest}"
    too_long = [
        safe_stem + suffix
        for suffix in suffixes
        if _nbytes(safe_stem + suffix) > int(max_filename_bytes)
    ]
    if too_long:
        raise RuntimeError(
            "Could not construct a safe filename component: "
            f"{too_long[0]!r}"
        )
    return safe_stem


# -----------------------------------------------------------------------------
# Paths and log manifest
# -----------------------------------------------------------------------------
def create_fedala_run_paths(
    fedala_root: str | Path,
    subset_id: str,
    rounds: int,
    local_epochs: int,
    mode: str,
    ala_lr: float,
    ala_rand_percent: float,
    model_tag: str,
    seed: int,
) -> FedALARunPaths:
    root = ensure_dir(fedala_root)
    # Multi-selection FedALA does not create/use andrea/checkpoints by default.
    ckpt_root = ensure_dir(fedala_root)
    clean_mode = str(mode).replace("-", "_")
    stem = (
        f"fedala_{clean_mode}"
        f"{subset_id}"
        f"_rounds{rounds}"
        f"_epoch{local_epochs}"
        f"_alalr{ala_lr}"
        f"_rand{ala_rand_percent}"
        f"_{model_tag}"
        f"_seed{seed}"
    )
    stem = safe_stem_for_suffixes(stem, (".csv", ".pt", "_ala.csv"))
    return FedALARunPaths(
        csv_path=root / f"{stem}.csv",
        ckpt_path=ckpt_root / f"{stem}.pt",
        ala_csv_path=root / f"{stem}_ala.csv",
    )


def build_fedala_log_row(
    *,
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: FedALARunPaths,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    mode: str,
    ala_lr: float,
    ala_rand_percent: float,
    ala_convergence_std: float,
    ala_convergence_window: int,
    ala_max_steps: int,
    selection_metrics: Optional[Sequence[str]] = None,
    selection_metric: Optional[str] = None,
) -> Dict:
    selection_metrics = normalize_selection_metrics(selection_metrics, selection_metric)
    subset_id = _subset_clients_str(subset_clients)
    dataset_ids = "|".join(str(client.dataset_id) for client in subset_clients)
    model_tag = _model_tag_from_cfg(cfg)
    algorithm = "fedala_fedavg" if mode == "all" else "fedala_head_only"

    return {
        "run_type": algorithm,
        "algorithm": algorithm,
        "subset_id": subset_id,
        "subset_clients": subset_id,
        "graph_id": "all",
        "dataset_id": dataset_ids,
        "num_clients": len(subset_clients),
        "client_fraction": client_fraction,
        "out_csv": str(run_paths.csv_path),
        "ckpt_path": str(run_paths.ckpt_path),
        "ala_csv_path": str(run_paths.ala_csv_path),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "selection_metric": "multi",
        "selection_metrics": "|".join(selection_metrics),
        "selection_direction": "mixed",
        "fedala_mode": mode,
        "fedala_ala_lr": ala_lr,
        "fedala_ala_rand_percent": ala_rand_percent,
        "fedala_ala_convergence_std": ala_convergence_std,
        "fedala_ala_convergence_window": ala_convergence_window,
        "fedala_ala_max_steps": ala_max_steps,
        "mcw": cfg.get("minority_class_weight", None),
        "num_layers": cfg["num_layers"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "dropout": cfg["dropout"],
        "hidden_dim": cfg["hidden_dim"],
        "use_ego_ids": cfg.get("use_ego_ids", True),
        "batch_size": cfg.get("batch_size", 64),
        "output_head": cfg.get("output_head", "multi"),
        "architecture": cfg.get("architecture", "multihead"),
        "seed": seed,
        "model_tag": model_tag,
    }


# -----------------------------------------------------------------------------
# Parameter selection and state helpers
# -----------------------------------------------------------------------------
def _task_idx_from_output_head_param_name(name: str) -> Optional[int]:
    prefix = "output_head.heads."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    first = rest.split(".", 1)[0]
    if not first.isdigit():
        return None
    return int(first)


def _is_task_head_param_name(name: str) -> bool:
    return _task_idx_from_output_head_param_name(name) is not None


def clone_state_dict_cpu(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def floating_named_parameters(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    return {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad and torch.is_floating_point(param)
    }


def select_ala_param_names(model: torch.nn.Module, mode: str) -> List[str]:
    named_params = floating_named_parameters(model)
    if mode == "all":
        return list(named_params.keys())
    if mode == "head_only":
        return [name for name in named_params.keys() if _is_task_head_param_name(name)]
    raise ValueError(f"Unknown FedALA mode={mode!r}; expected 'all' or 'head_only'.")


def state_distance_l2(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    names: Sequence[str],
) -> float:
    sq = 0.0
    for name in names:
        diff = a[name].float() - b[name].float()
        sq += float(torch.sum(diff * diff).item())
    return math.sqrt(max(sq, 0.0))


def summarize_weight_dict(
    weights: Dict[str, torch.Tensor],
    names: Sequence[str],
) -> Dict[str, float]:
    if not names:
        return {
            "ala_mean": float("nan"),
            "ala_std": float("nan"),
            "ala_min": float("nan"),
            "ala_max": float("nan"),
        }
    flat = torch.cat(
        [weights[name].detach().float().cpu().reshape(-1) for name in names]
    )
    return {
        "ala_mean": float(flat.mean().item()),
        "ala_std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "ala_min": float(flat.min().item()),
        "ala_max": float(flat.max().item()),
    }


def summarize_group_weights(
    weights: Dict[str, torch.Tensor],
    names: Sequence[str],
) -> Dict[str, float]:
    head_names = [name for name in names if _is_task_head_param_name(name)]
    backbone_names = [name for name in names if name not in set(head_names)]

    out: Dict[str, float] = {}
    for prefix, group_names in [("head", head_names), ("backbone", backbone_names)]:
        stats = summarize_weight_dict(weights, group_names)
        for key, value in stats.items():
            out[f"ala_{prefix}_{key.replace('ala_', '')}"] = value

    # Per-task head summaries. These are useful for q-label heterogeneity runs.
    by_task: Dict[int, List[str]] = {}
    for name in head_names:
        task_idx = _task_idx_from_output_head_param_name(name)
        if task_idx is not None:
            by_task.setdefault(task_idx, []).append(name)
    for task_idx, group_names in sorted(by_task.items()):
        stats = summarize_weight_dict(weights, group_names)
        out[f"ala_task{task_idx}_mean"] = stats["ala_mean"]
        out[f"ala_task{task_idx}_std"] = stats["ala_std"]
    return out


def build_mixed_state_dict(
    *,
    old_local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    weights: Dict[str, torch.Tensor],
    eligible_names: Sequence[str],
    mode: str,
) -> Dict[str, torch.Tensor]:
    """Construct the ALA initialization state.

    For mode='all':
        every selected learnable parameter is interpolated.
        buffers/non-floating entries come from the global model.

    For mode='head_only':
        task-head parameters are interpolated.
        all non-head parameters and buffers come from the global model.
    """

    eligible_set = set(eligible_names)
    out: Dict[str, torch.Tensor] = {}

    for key, global_value in global_state.items():
        if key in eligible_set:
            old_value = old_local_state[key].float()
            g_value = global_state[key].float()
            alpha = weights[key].detach().cpu().float().clamp(0.0, 1.0)
            mixed = old_value + alpha * (g_value - old_value)
            out[key] = mixed.to(global_value.dtype)
        else:
            out[key] = global_value.detach().cpu().clone()

    return out


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


def _build_functional_params(
    *,
    model: torch.nn.Module,
    old_local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    weight_vars: Dict[str, torch.nn.Parameter],
    eligible_names: Sequence[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    eligible_set = set(eligible_names)
    params: Dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if name in eligible_set:
            old_value = old_local_state[name].to(device=device, dtype=param.dtype)
            g_value = global_state[name].to(device=device, dtype=param.dtype)
            alpha = weight_vars[name].clamp(0.0, 1.0)
            params[name] = old_value + alpha * (g_value - old_value)
        else:
            params[name] = global_state[name].to(device=device, dtype=param.dtype)

    return params


def learn_ala_weights_for_client(
    *,
    client_idx: int,
    runtime,
    cfg: Dict,
    ctx: Dict,
    client_state: FedALAClientState,
    old_local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    mode: str,
    ala_lr: float,
    ala_rand_percent: float,
    ala_convergence_std: float,
    ala_convergence_window: int,
    ala_max_steps: int,
    device: torch.device,
    round_idx: int,
    debug: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Learn/update the client-side ALA weights and return the init state.

    This follows the FedALA idea: optimize only alpha on local data, with the old
    local model and downloaded global model frozen, then initialize the local
    model by alpha-interpolating old and global parameters.
    """

    template_model = make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)
    template_model.load_state_dict(global_state, strict=True)
    template_model.train()

    eligible_names = select_ala_param_names(template_model, mode)
    if not eligible_names:
        raise ValueError(
            f"FedALA mode={mode!r} selected zero parameters. "
            "For head_only, check that task-head params are named output_head.heads.*"
        )

    # Initialize unseen alpha tensors to 1, as in the FedALA implementation:
    # start from the downloaded global model and learn how much to keep.
    for name in eligible_names:
        if name not in client_state.weights:
            client_state.weights[name] = torch.ones_like(
                global_state[name], dtype=torch.float32
            )

    old_global_dist = state_distance_l2(old_local_state, global_state, eligible_names)

    stats: Dict[str, float] = {
        "round": float(round_idx),
        "client_idx": float(client_idx),
        "graph_id": (
            float(runtime.client.graph_id)
            if str(runtime.client.graph_id).isdigit()
            else float(client_idx)
        ),
        "old_global_l2": old_global_dist,
        "ala_steps": 0.0,
        "ala_first_loss": float("nan"),
        "ala_last_loss": float("nan"),
        "ala_loss_std_window": float("nan"),
        "ala_skipped_identical": 0.0,
    }

    if old_global_dist <= 1e-12:
        stats["ala_skipped_identical"] = 1.0
        mixed_state = build_mixed_state_dict(
            old_local_state=old_local_state,
            global_state=global_state,
            weights=client_state.weights,
            eligible_names=eligible_names,
            mode=mode,
        )
        stats.update(summarize_weight_dict(client_state.weights, eligible_names))
        stats.update(summarize_group_weights(client_state.weights, eligible_names))
        if debug:
            print(
                f"[FedALA DEBUG] round={round_idx} client={runtime.client.graph_id} "
                f"mode={mode} skip ALA because old/global selected-param distance is 0."
            )
        return mixed_state, stats

    weight_vars: Dict[str, torch.nn.Parameter] = {
        name: torch.nn.Parameter(
            client_state.weights[name].detach().to(device).float().clone()
        )
        for name in eligible_names
    }
    optimizer = torch.optim.SGD(list(weight_vars.values()), lr=float(ala_lr))

    try:
        loader_len = len(runtime.train_loader)
    except Exception:
        loader_len = int(ala_max_steps)
    max_batches = max(
        1, int(math.ceil(float(loader_len) * float(ala_rand_percent) / 100.0))
    )
    max_batches = min(max_batches, int(ala_max_steps))

    # Original FedALA learns until convergence at the start phase; in subsequent
    # rounds it only performs one pass over the sampled local data.
    hard_step_cap = int(ala_max_steps) if client_state.start_phase else max_batches
    was_start_phase = bool(client_state.start_phase)
    losses: List[float] = []
    step_count = 0
    converged = False

    buffers = {name: buf for name, buf in template_model.named_buffers()}

    while step_count < hard_step_cap and not converged:
        for batch in runtime.train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            x_in, y_seed, B = augment_batch_x_with_ego(
                batch,
                use_ego_ids=ctx["use_ego_ids"],
                ego_dim=ctx["ego_dim"],
            )
            edge_in, edge_attr_dict = unpack_batch_edges(batch)

            params = _build_functional_params(
                model=template_model,
                old_local_state=old_local_state,
                global_state=global_state,
                weight_vars=weight_vars,
                eligible_names=eligible_names,
                device=device,
            )

            out = functional_call_compat(
                template_model,
                params,
                buffers,
                args=(x_in, edge_in),
                kwargs={"edge_attr_dict": edge_attr_dict, "device": device},
            )
            out_seed = out[:B]
            label_mask_seed = get_seed_label_mask(batch, B)
            loss = masked_loss_from_logits(
                runtime.criterion,
                out_seed,
                y_seed.float(),
                label_mask_seed,
            )
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for param in weight_vars.values():
                    param.clamp_(0.0, 1.0)

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            step_count += 1

            if (
                client_state.start_phase
                and len(losses) >= int(ala_convergence_window)
                and float(np.std(losses[-int(ala_convergence_window) :]))
                < float(ala_convergence_std)
            ):
                converged = True
                break

            if step_count >= hard_step_cap:
                break

            # One sampled pass in non-start phase, like the FedALA code.
            if not client_state.start_phase and step_count >= max_batches:
                break

    for name in eligible_names:
        client_state.weights[name] = (
            weight_vars[name].detach().cpu().float().clamp(0.0, 1.0)
        )

    client_state.start_phase = False

    mixed_state = build_mixed_state_dict(
        old_local_state=old_local_state,
        global_state=global_state,
        weights=client_state.weights,
        eligible_names=eligible_names,
        mode=mode,
    )

    init_old_dist = state_distance_l2(mixed_state, old_local_state, eligible_names)
    init_global_dist = state_distance_l2(mixed_state, global_state, eligible_names)

    stats.update(summarize_weight_dict(client_state.weights, eligible_names))
    stats.update(summarize_group_weights(client_state.weights, eligible_names))
    stats.update(
        {
            "ala_steps": float(step_count),
            "ala_first_loss": float(losses[0]) if losses else float("nan"),
            "ala_last_loss": float(losses[-1]) if losses else float("nan"),
            "ala_loss_std_window": (
                float(np.std(losses[-int(ala_convergence_window) :]))
                if len(losses) >= int(ala_convergence_window)
                else float("nan")
            ),
            "init_old_l2": init_old_dist,
            "init_global_l2": init_global_dist,
            "eligible_param_count": float(len(eligible_names)),
            "eligible_param_elements": float(
                sum(int(client_state.weights[name].numel()) for name in eligible_names)
            ),
        }
    )

    if debug:
        print("\n" + "-" * 100)
        print(
            f"[FedALA DEBUG] round={round_idx} client={runtime.client.graph_id} "
            f"mode={mode} was_start_phase={was_start_phase}"
        )
        print(f"  selected parameter tensors: {len(eligible_names)}")
        print(f"  old-global L2 over selected params: {old_global_dist:.6f}")
        print(
            f"  ALA steps={step_count} first_loss={stats['ala_first_loss']:.6f} "
            f"last_loss={stats['ala_last_loss']:.6f} "
            f"std_window={stats['ala_loss_std_window']:.6f}"
        )
        print(
            f"  alpha mean={stats['ala_mean']:.4f} std={stats['ala_std']:.4f} "
            f"min={stats['ala_min']:.4f} max={stats['ala_max']:.4f}"
        )
        print(
            f"  alpha backbone mean={stats.get('ala_backbone_mean', float('nan')):.4f} | "
            f"head mean={stats.get('ala_head_mean', float('nan')):.4f}"
        )
        print(
            f"  init-old L2={init_old_dist:.6f} init-global L2={init_global_dist:.6f}"
        )
        print("  sample selected params:")
        for name in eligible_names[:12]:
            print(f"    {name}: shape={tuple(client_state.weights[name].shape)}")
        print("-" * 100)

    return mixed_state, stats


# -----------------------------------------------------------------------------
# Core experiment
# -----------------------------------------------------------------------------
def _evaluate_state_on_runtime(
    *,
    state: Dict[str, torch.Tensor],
    runtime,
    cfg: Dict,
    ctx: Dict,
    device: torch.device,
    split_name: str,
    eval_mask_mode: str = "full",
) -> Dict:
    model = make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)

    model.load_state_dict(state, strict=True)

    loader = {
        "train": runtime.train_loader,
        "val": runtime.val_loader,
        "test": runtime.test_loader,
    }[split_name]

    return evaluate_loader(
        model,
        loader,
        runtime.criterion,
        device,
        use_ego_ids=ctx["use_ego_ids"],
        ego_dim=ctx["ego_dim"],
        threshold=0.5,
        eval_mask_mode=eval_mask_mode,
    )


def _append_mean_eval_row(
    rows: List[Dict],
    *,
    run_type: str,
    algorithm: str,
    subset_id: str,
    subset_clients_str: str,
    seed: int,
    phase: str,
    split: str,
    round_idx: int,
    eval_infos: List[Dict],
) -> Dict[str, float]:
    mean_scalar = weighted_scalar_summary(eval_infos)
    rows.append(
        {
            "run_type": run_type,
            "algorithm": algorithm,
            "subset_id": subset_id,
            "subset_clients": subset_clients_str,
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
            "macro_minority_f1": mean_scalar.get("macro_minority_f1"),
            "micro_pr_auc": mean_scalar.get("micro_pr_auc"),
            "macro_pr_auc": mean_scalar.get("macro_pr_auc"),
            "pr_auc": None,
            "num_nodes": sum(
                int(metrics["counts"]["num_nodes"]) for metrics in eval_infos
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
    return mean_scalar


def run_fedala_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: FedALARunPaths,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    device: torch.device,
    mode: str,
    ala_lr: float,
    ala_rand_percent: float,
    ala_convergence_std: float,
    ala_convergence_window: int,
    ala_max_steps: int,
    selection_metrics: Optional[Sequence[str]] = None,
    selection_metric: Optional[str] = None,
    debug: bool = True,
) -> None:
    if not subset_clients:
        raise ValueError("subset_clients is empty")

    set_seed(seed)
    selection_metrics = normalize_selection_metrics(selection_metrics, selection_metric)

    subset_id = _subset_clients_str(subset_clients)
    subset_clients_str = subset_id
    algorithm = "fedala_fedavg" if mode == "all" else "fedala_head_only"

    ctx = build_model_context(subset_clients, cfg)
    runtime_clients = build_runtime_clients(subset_clients, cfg, device)

    global_model = make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)

    initial_global_state = clone_state_dict_cpu(global_model.state_dict())
    local_states: List[Dict[str, torch.Tensor]] = [
        clone_state_dict_cpu(initial_global_state) for _ in runtime_clients
    ]
    ala_states: List[FedALAClientState] = [FedALAClientState() for _ in runtime_clients]

    # Debug selected params once.
    if debug:
        eligible_names = select_ala_param_names(global_model, mode)
        print("\n" + "=" * 100)
        print("FEDALA RUN DEBUG SUMMARY")
        print("=" * 100)
        print("algorithm          :", algorithm)
        print("mode               :", mode)
        print("clients            :", subset_clients_str)
        print("rounds/local_epochs:", rounds, local_epochs)
        print("ala_lr             :", ala_lr)
        print("ala_rand_percent   :", ala_rand_percent)
        print("ala_max_steps      :", ala_max_steps)
        print("eligible tensors   :", len(eligible_names))
        print(
            "eligible elements  :",
            sum(int(global_model.state_dict()[n].numel()) for n in eligible_names),
        )
        print("first eligible names:")
        for name in eligible_names[:30]:
            print("  ", name, tuple(global_model.state_dict()[name].shape))
        print("=" * 100)

    rows: List[Dict] = []
    ala_rows: List[Dict] = []

    initial_global_for_selection = clone_state_dict_cpu(global_model.state_dict())
    initial_locals_for_selection = [clone_state_dict_cpu(state) for state in local_states]
    best_by_eval_mode = {
        eval_mode: {
            metric: {
                "value": initial_selection_value(metric),
                "round": -1,
                "global_state": clone_state_dict_cpu(initial_global_for_selection),
                "local_states": [clone_state_dict_cpu(state) for state in initial_locals_for_selection],
            }
            for metric in selection_metrics
        }
        for eval_mode in ("full", "visible")
    }

    num_clients = len(runtime_clients)
    generator = torch.Generator().manual_seed(seed)

    for round_idx in range(1, rounds + 1):
        run_start_time = time.perf_counter()

        num_selected = max(1, int(round(client_fraction * num_clients)))
        if num_selected >= num_clients:
            selected_indices = list(range(num_clients))
        else:
            permutation = torch.randperm(num_clients, generator=generator).tolist()
            selected_indices = permutation[:num_selected]

        if debug:
            print("\n" + "=" * 100)
            print(f"FEDALA ROUND {round_idx}/{rounds} | selected={selected_indices}")
            print("=" * 100)

        global_state_cpu = clone_state_dict_cpu(global_model.state_dict())
        selected_states: List[Dict[str, torch.Tensor]] = []
        selected_weights: List[int] = []

        for idx in selected_indices:
            runtime = runtime_clients[idx]
            old_local_state = local_states[idx]

            init_state, ala_stats = learn_ala_weights_for_client(
                client_idx=idx,
                runtime=runtime,
                cfg=cfg,
                ctx=ctx,
                client_state=ala_states[idx],
                old_local_state=old_local_state,
                global_state=global_state_cpu,
                mode=mode,
                ala_lr=ala_lr,
                ala_rand_percent=ala_rand_percent,
                ala_convergence_std=ala_convergence_std,
                ala_convergence_window=ala_convergence_window,
                ala_max_steps=ala_max_steps,
                device=device,
                round_idx=round_idx,
                debug=debug,
            )
            ala_stats.update(
                {
                    "run_type": algorithm,
                    "algorithm": algorithm,
                    "subset_id": subset_id,
                    "subset_clients": subset_clients_str,
                    "seed": seed,
                    "round": round_idx,
                    "client_idx": idx,
                    "graph_id": str(runtime.client.graph_id),
                    "dataset_id": str(runtime.client.dataset_id),
                    "fedala_mode": mode,
                }
            )
            ala_rows.append(ala_stats)

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
            local_model.load_state_dict(init_state, strict=True)
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
                    run_type=algorithm,
                    algorithm=algorithm,
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
                if debug:
                    print(
                        f"[FedALA TRAIN] round={round_idx} client={runtime.client.graph_id} "
                        f"local_epoch={local_epoch_idx} train_loss={train_loss:.6f}"
                    )

            trained_state = clone_state_dict_cpu(local_model.state_dict())
            local_states[idx] = trained_state
            selected_states.append(trained_state)
            selected_weights.append(runtime.num_train_nodes)

        # Server side: standard weighted FedAvg over selected trained local models.
        aggregated_state = fedavg_state_dict(selected_states, selected_weights)
        global_model.load_state_dict(aggregated_state, strict=True)

        # Diagnostic 1: global model after FedAvg, evaluated on every client.
        global_val_infos = []
        for runtime in runtime_clients:
            metrics = evaluate_loader(
                global_model,
                runtime.val_loader,
                runtime.criterion,
                device,
                use_ego_ids=ctx["use_ego_ids"],
                ego_dim=ctx["ego_dim"],
                threshold=0.5,
            )
            global_val_infos.append(metrics)
            append_eval_rows(
                rows,
                run_type=algorithm,
                algorithm=algorithm,
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                phase="global_val_client",
                split="val",
                graph_id=str(runtime.client.graph_id),
                dataset_id=str(runtime.client.dataset_id),
                metrics=metrics,
                round_idx=round_idx,
            )

        global_mean = _append_mean_eval_row(
            rows,
            run_type=algorithm,
            algorithm=algorithm,
            subset_id=subset_id,
            subset_clients_str=subset_clients_str,
            seed=seed,
            phase="global_val_mean",
            split="val",
            round_idx=round_idx,
            eval_infos=global_val_infos,
        )

        # Diagnostic 2: personalized local models, evaluated on their own client.
        # We evaluate two validation modes:
        #   full    -> oracle/full validation
        #   visible -> realistic/masked validation
        personalized_val_full_infos = []
        personalized_val_visible_infos = []

        for idx, runtime in enumerate(runtime_clients):
            metrics_full = _evaluate_state_on_runtime(
                state=local_states[idx],
                runtime=runtime,
                cfg=cfg,
                ctx=ctx,
                device=device,
                split_name="val",
                eval_mask_mode="full",
            )
            personalized_val_full_infos.append(metrics_full)
            append_eval_rows(
                rows,
                run_type=algorithm,
                algorithm=algorithm,
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                phase="personalized_val_full_client",
                split="val",
                graph_id=str(runtime.client.graph_id),
                dataset_id=str(runtime.client.dataset_id),
                metrics=metrics_full,
                round_idx=round_idx,
                eval_mask_mode="full",
            )
            tag_recent_eval_rows(rows, selection_metrics="|".join(selection_metrics))

            metrics_visible = _evaluate_state_on_runtime(
                state=local_states[idx],
                runtime=runtime,
                cfg=cfg,
                ctx=ctx,
                device=device,
                split_name="val",
                eval_mask_mode="visible",
            )
            personalized_val_visible_infos.append(metrics_visible)
            append_eval_rows(
                rows,
                run_type=algorithm,
                algorithm=algorithm,
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                phase="personalized_val_visible_client",
                split="val",
                graph_id=str(runtime.client.graph_id),
                dataset_id=str(runtime.client.dataset_id),
                metrics=metrics_visible,
                round_idx=round_idx,
                eval_mask_mode="visible",
            )
            tag_recent_eval_rows(rows, selection_metrics="|".join(selection_metrics))

        personalized_full_mean = _append_mean_eval_row(
            rows,
            run_type=algorithm,
            algorithm=algorithm,
            subset_id=subset_id,
            subset_clients_str=subset_clients_str,
            seed=seed,
            phase="personalized_val_full_mean",
            split="val",
            round_idx=round_idx,
            eval_infos=personalized_val_full_infos,
        )

        personalized_visible_mean = _append_mean_eval_row(
            rows,
            run_type=algorithm,
            algorithm=algorithm,
            subset_id=subset_id,
            subset_clients_str=subset_clients_str,
            seed=seed,
            phase="personalized_val_visible_mean",
            split="val",
            round_idx=round_idx,
            eval_infos=personalized_val_visible_infos,
        )

        if debug:
            print("\n[FedALA ROUND SUMMARY]")
            print(f"  round={round_idx}")
            print(
                f"  personalized FULL val    loss={personalized_full_mean['loss']:.6f} "
                f"micro_f1={personalized_full_mean['micro_f1']:.4f} "
                f"macro_pos_f1={personalized_full_mean['macro_pos_f1']:.4f} "
                f"micro_pr_auc={personalized_full_mean.get('micro_pr_auc', float('nan')):.4f}"
            )
            print(
                f"  personalized VISIBLE val loss={personalized_visible_mean['loss']:.6f} "
                f"micro_f1={personalized_visible_mean['micro_f1']:.4f} "
                f"macro_pos_f1={personalized_visible_mean['macro_pos_f1']:.4f} "
                f"micro_pr_auc={personalized_visible_mean.get('micro_pr_auc', float('nan')):.4f}"
            )

        for selector_metric in selection_metrics:
            current_full_value = personalized_full_mean.get(selector_metric)
            current_visible_value = personalized_visible_mean.get(selector_metric)

            if debug:
                print(
                    f"  full-selection {selector_metric}: current={current_full_value}, "
                    f"best={best_by_eval_mode['full'][selector_metric]['value']}"
                )
                print(
                    f"  visible-selection {selector_metric}: current={current_visible_value}, "
                    f"best={best_by_eval_mode['visible'][selector_metric]['value']}"
                )

            if is_better_selection_value(
                current_full_value,
                best_by_eval_mode["full"][selector_metric]["value"],
                selector_metric,
            ):
                best_by_eval_mode["full"][selector_metric]["value"] = float(current_full_value)
                best_by_eval_mode["full"][selector_metric]["round"] = round_idx
                best_by_eval_mode["full"][selector_metric]["global_state"] = clone_state_dict_cpu(global_model.state_dict())
                best_by_eval_mode["full"][selector_metric]["local_states"] = [
                    clone_state_dict_cpu(state) for state in local_states
                ]
                if debug:
                    print(
                        f"  NEW BEST FULL-selected {selector_metric} checkpoint at round {round_idx}"
                    )

            if is_better_selection_value(
                current_visible_value,
                best_by_eval_mode["visible"][selector_metric]["value"],
                selector_metric,
            ):
                best_by_eval_mode["visible"][selector_metric]["value"] = float(current_visible_value)
                best_by_eval_mode["visible"][selector_metric]["round"] = round_idx
                best_by_eval_mode["visible"][selector_metric]["global_state"] = clone_state_dict_cpu(global_model.state_dict())
                best_by_eval_mode["visible"][selector_metric]["local_states"] = [
                    clone_state_dict_cpu(state) for state in local_states
                ]
                if debug:
                    print(
                        f"  NEW BEST VISIBLE-selected {selector_metric} checkpoint at round {round_idx}"
                    )

        run_elapsed = time.perf_counter() - run_start_time
        print(f"run time: {format_seconds(run_elapsed)}")

    checkpoint = {
        "cfg": cfg,
        "seed": seed,
        "selection_metrics": list(selection_metrics),
        "best_round_by_eval_mode_and_metric": {
            mode_name: {
                metric: record["round"]
                for metric, record in mode_records.items()
            }
            for mode_name, mode_records in best_by_eval_mode.items()
        },
        "best_value_by_eval_mode_and_metric": {
            mode_name: {
                metric: record["value"]
                for metric, record in mode_records.items()
            }
            for mode_name, mode_records in best_by_eval_mode.items()
        },
        "subset_id": subset_id,
        "subset_clients": subset_clients_str,
        "fedala_mode": mode,
        "fedala_ala_lr": ala_lr,
        "fedala_ala_rand_percent": ala_rand_percent,
        "x_dim": ctx["x_dim"],
        "out_dim": ctx["out_dim"],
        "ego_dim": ctx["ego_dim"],
        "in_vocab": ctx["in_vocab"],
        "out_vocab": ctx["out_vocab"],
        "deg_fwd_hist": ctx["deg_fwd_hist"].cpu(),
        "deg_rev_hist": ctx["deg_rev_hist"].cpu(),
    }
    maybe_save_checkpoint(checkpoint, run_paths.ckpt_path)

    # -----------------------------------------------------------------------------
    # Final evaluation under the same 3 protocols as the other methods.
    #
    # oracle_full:
    #   select by full/oracle validation, evaluate full/oracle test
    #
    # realistic_visible:
    #   select by visible/masked validation, evaluate visible/masked test
    #
    # realistic_selection_oracle:
    #   select by visible/masked validation, evaluate full/oracle test
    # -----------------------------------------------------------------------------
    final_eval_jobs = [
        {
            "selection_protocol": "oracle_full",
            "selected_by": "full",
            "eval_mask_mode": "full",
        },
        {
            "selection_protocol": "realistic_visible",
            "selected_by": "visible",
            "eval_mask_mode": "visible",
        },
        {
            "selection_protocol": "realistic_selection_oracle",
            "selected_by": "visible",
            "eval_mask_mode": "full",
        },
    ]

    for selector_metric in selection_metrics:
        for job in final_eval_jobs:
            selection_protocol = job["selection_protocol"]
            selected_by = job["selected_by"]
            eval_mask_mode = job["eval_mask_mode"]

            selected_record = best_by_eval_mode[selected_by][selector_metric]
            selected_local_states = selected_record["local_states"]
            selected_round = selected_record["round"]
            best_value = selected_record["value"]

            for idx, runtime in enumerate(runtime_clients):
                metrics = _evaluate_state_on_runtime(
                    state=selected_local_states[idx],
                    runtime=runtime,
                    cfg=cfg,
                    ctx=ctx,
                    device=device,
                    split_name="test",
                    eval_mask_mode=eval_mask_mode,
                )

                append_eval_rows(
                    rows,
                    run_type=algorithm,
                    algorithm=algorithm,
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"best_{selection_protocol}_test",
                    split="test",
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
                    best_val_metric_value=best_value,
                )

    pd.DataFrame(rows).to_csv(run_paths.csv_path, index=False)
    pd.DataFrame(ala_rows).to_csv(run_paths.ala_csv_path, index=False)
    print(f"saved csv      -> {run_paths.csv_path}")
    print(f"saved ala csv  -> {run_paths.ala_csv_path}")


def run_fedala(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    fedala_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    device: torch.device,
    mode: str,
    ala_lr: float,
    ala_rand_percent: float,
    ala_convergence_std: float = 0.1,
    ala_convergence_window: int = 10,
    ala_max_steps: int = 100,
    selection_metrics: Optional[Sequence[str]] = None,
    selection_metric: Optional[str] = None,
    debug: bool = True,
) -> FedALARunPaths:
    selection_metrics = normalize_selection_metrics(selection_metrics, selection_metric)
    subset_id = _subset_clients_str(subset_clients)
    subset_file_tag = compact_subset_id_for_filename(subset_id)
    cfg = dict(cfg)
    cfg["selection_tag"] = "ms5" if len(selection_metrics) == 5 else "ms" + str(len(selection_metrics))
    model_tag = f"{_model_tag_from_cfg(cfg)}_ms"

    run_paths = create_fedala_run_paths(
        fedala_root,
        subset_file_tag,
        rounds,
        local_epochs,
        mode,
        ala_lr,
        ala_rand_percent,
        model_tag,
        seed,
    )

    if run_paths.csv_path.exists() and run_paths.ala_csv_path.exists():
        return run_paths

    run_fedala_experiment(
        subset_clients,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        device=device,
        mode=mode,
        ala_lr=ala_lr,
        ala_rand_percent=ala_rand_percent,
        ala_convergence_std=ala_convergence_std,
        ala_convergence_window=ala_convergence_window,
        ala_max_steps=ala_max_steps,
        selection_metrics=selection_metrics,
        debug=debug,
    )
    return run_paths
