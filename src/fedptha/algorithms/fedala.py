from __future__ import annotations
from fedptha.training.runtime import RunPaths, create_run_paths
from fedptha.training.state import personalized_call
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from fedptha.training.runtime import (
    _subset_clients_str,
    append_eval_rows,
    append_train_row,
    build_model_context,
    build_runtime_clients,
    fedavg_state_dict,
    make_model,
    make_optimizer,
    weighted_scalar_summary,
    is_better_selection_value,
    tag_recent_eval_rows,
)
from fedptha.data.clients import ClientData, format_seconds
from fedptha.training.steps import (
    augment_batch_x_with_ego,
    evaluate_loader,
    get_seed_label_mask,
    masked_loss_from_logits,
    train_epoch_neighbor,
    unpack_batch_edges,
)
from fedptha.training.random import set_seed


@dataclass
class FedALAClientState:
    """Client-side ALA state.

    weights maps parameter name -> alpha tensor. Alpha is clipped to [0, 1].

    alpha = 1 means fully accept the downloaded global parameter.
    alpha = 0 means preserve the old local parameter.
    """

    weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    start_phase: bool = True


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
    return {key: value.detach().cpu().clone() for (key, value) in state_dict.items()}


def floating_named_parameters(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    return {
        name: param
        for (name, param) in model.named_parameters()
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
    a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], names: Sequence[str]
) -> float:
    sq = 0.0
    for name in names:
        diff = a[name].float() - b[name].float()
        sq += float(torch.sum(diff * diff).item())
    return math.sqrt(max(sq, 0.0))


def summarize_weight_dict(
    weights: Dict[str, torch.Tensor], names: Sequence[str]
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
    weights: Dict[str, torch.Tensor], names: Sequence[str]
) -> Dict[str, float]:
    head_names = [name for name in names if _is_task_head_param_name(name)]
    backbone_names = [name for name in names if name not in set(head_names)]
    out: Dict[str, float] = {}
    for prefix, group_names in [("head", head_names), ("backbone", backbone_names)]:
        stats = summarize_weight_dict(weights, group_names)
        for key, value in stats.items():
            out[f"ala_{prefix}_{key.replace('ala_', '')}"] = value
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
            f"FedALA mode={mode!r} selected zero parameters. For head_only, check that task-head params are named output_head.heads.*"
        )
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
                f"[FedALA DEBUG] round={round_idx} client={runtime.client.graph_id} mode={mode} skip ALA because old/global selected-param distance is 0."
            )
        return (mixed_state, stats)
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
    hard_step_cap = int(ala_max_steps) if client_state.start_phase else max_batches
    was_start_phase = bool(client_state.start_phase)
    losses: List[float] = []
    step_count = 0
    converged = False
    buffers = {name: buf for (name, buf) in template_model.named_buffers()}
    while step_count < hard_step_cap and (not converged):
        for batch in runtime.train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            x_in, y_seed, B = augment_batch_x_with_ego(
                batch, use_ego_ids=ctx["use_ego_ids"], ego_dim=ctx["ego_dim"]
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
            out = personalized_call(
                template_model,
                params,
                buffers,
                args=(x_in, edge_in),
                kwargs={"edge_attr_dict": edge_attr_dict, "device": device},
            )
            out_seed = out[:B]
            label_mask_seed = get_seed_label_mask(batch, B)
            loss = masked_loss_from_logits(
                runtime.criterion, out_seed, y_seed.float(), label_mask_seed
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
                and (
                    float(np.std(losses[-int(ala_convergence_window) :]))
                    < float(ala_convergence_std)
                )
            ):
                converged = True
                break
            if step_count >= hard_step_cap:
                break
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
                sum(
                    (int(client_state.weights[name].numel()) for name in eligible_names)
                )
            ),
        }
    )
    if debug:
        print("\n" + "-" * 100)
        print(
            f"[FedALA DEBUG] round={round_idx} client={runtime.client.graph_id} mode={mode} was_start_phase={was_start_phase}"
        )
        print(f"  selected parameter tensors: {len(eligible_names)}")
        print(f"  old-global L2 over selected params: {old_global_dist:.6f}")
        print(
            f"  ALA steps={step_count} first_loss={stats['ala_first_loss']:.6f} last_loss={stats['ala_last_loss']:.6f} std_window={stats['ala_loss_std_window']:.6f}"
        )
        print(
            f"  alpha mean={stats['ala_mean']:.4f} std={stats['ala_std']:.4f} min={stats['ala_min']:.4f} max={stats['ala_max']:.4f}"
        )
        print(
            f"  alpha backbone mean={stats.get('ala_backbone_mean', float('nan')):.4f} | head mean={stats.get('ala_head_mean', float('nan')):.4f}"
        )
        print(
            f"  init-old L2={init_old_dist:.6f} init-global L2={init_global_dist:.6f}"
        )
        print("  sample selected params:")
        for name in eligible_names[:12]:
            print(f"    {name}: shape={tuple(client_state.weights[name].shape)}")
        print("-" * 100)
    return (mixed_state, stats)


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
                (int(metrics["counts"]["num_nodes"]) for metrics in eval_infos)
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
    run_paths: RunPaths,
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
    debug: bool = True,
) -> None:
    if not subset_clients:
        raise ValueError("subset_clients is empty")
    set_seed(seed)
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
            sum((int(global_model.state_dict()[n].numel()) for n in eligible_names)),
        )
        print("first eligible names:")
        for name in eligible_names[:30]:
            print("  ", name, tuple(global_model.state_dict()[name].shape))
        print("=" * 100)
    rows: List[Dict] = []
    ala_rows: List[Dict] = []
    initial_global_for_selection = clone_state_dict_cpu(global_model.state_dict())
    initial_locals_for_selection = [
        clone_state_dict_cpu(state) for state in local_states
    ]
    best_by_eval_mode = {
        eval_mode: {
            "value": float("-inf"),
            "round": -1,
            "global_state": clone_state_dict_cpu(initial_global_for_selection),
            "local_states": [
                clone_state_dict_cpu(state) for state in initial_locals_for_selection
            ],
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
                        f"[FedALA TRAIN] round={round_idx} client={runtime.client.graph_id} local_epoch={local_epoch_idx} train_loss={train_loss:.6f}"
                    )
            trained_state = clone_state_dict_cpu(local_model.state_dict())
            local_states[idx] = trained_state
            selected_states.append(trained_state)
            selected_weights.append(runtime.num_train_nodes)
        aggregated_state = fedavg_state_dict(selected_states, selected_weights)
        global_model.load_state_dict(aggregated_state, strict=True)
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
            tag_recent_eval_rows(rows, selection_metrics="macro_pr_auc")
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
            tag_recent_eval_rows(rows, selection_metrics="macro_pr_auc")
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
                f"  personalized FULL val    loss={personalized_full_mean['loss']:.6f} micro_f1={personalized_full_mean['micro_f1']:.4f} macro_pos_f1={personalized_full_mean['macro_pos_f1']:.4f} micro_pr_auc={personalized_full_mean.get('micro_pr_auc', float('nan')):.4f}"
            )
            print(
                f"  personalized VISIBLE val loss={personalized_visible_mean['loss']:.6f} micro_f1={personalized_visible_mean['micro_f1']:.4f} macro_pos_f1={personalized_visible_mean['macro_pos_f1']:.4f} micro_pr_auc={personalized_visible_mean.get('micro_pr_auc', float('nan')):.4f}"
            )
        current_full_value = personalized_full_mean.get("macro_pr_auc")
        current_visible_value = personalized_visible_mean.get("macro_pr_auc")
        if debug:
            print(
                f"  full-selection macro_pr_auc: current={current_full_value}, best={best_by_eval_mode['full']['value']}"
            )
            print(
                f"  visible-selection macro_pr_auc: current={current_visible_value}, best={best_by_eval_mode['visible']['value']}"
            )
        if is_better_selection_value(
            current_full_value, best_by_eval_mode["full"]["value"]
        ):
            best_by_eval_mode["full"]["value"] = float(current_full_value)
            best_by_eval_mode["full"]["round"] = round_idx
            best_by_eval_mode["full"]["global_state"] = clone_state_dict_cpu(
                global_model.state_dict()
            )
            best_by_eval_mode["full"]["local_states"] = [
                clone_state_dict_cpu(state) for state in local_states
            ]
            if debug:
                print(
                    f"  NEW BEST FULL-selected macro_pr_auc checkpoint at round {round_idx}"
                )
        if is_better_selection_value(
            current_visible_value, best_by_eval_mode["visible"]["value"]
        ):
            best_by_eval_mode["visible"]["value"] = float(current_visible_value)
            best_by_eval_mode["visible"]["round"] = round_idx
            best_by_eval_mode["visible"]["global_state"] = clone_state_dict_cpu(
                global_model.state_dict()
            )
            best_by_eval_mode["visible"]["local_states"] = [
                clone_state_dict_cpu(state) for state in local_states
            ]
            if debug:
                print(
                    f"  NEW BEST VISIBLE-selected macro_pr_auc checkpoint at round {round_idx}"
                )
        run_elapsed = time.perf_counter() - run_start_time
        print(f"run time: {format_seconds(run_elapsed)}")
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
    for job in final_eval_jobs:
        selection_protocol = job["selection_protocol"]
        selected_by = job["selected_by"]
        eval_mask_mode = job["eval_mask_mode"]
        selected_record = best_by_eval_mode[selected_by]
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
                selection_metrics="macro_pr_auc",
                selector_metric="macro_pr_auc",
                selector_direction="max",
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
    debug: bool = True,
) -> RunPaths:
    subset_id = _subset_clients_str(subset_clients)
    cfg = dict(cfg)
    cfg["selection_tag"] = "macro_pr_auc"
    run_paths = create_run_paths(fedala_root)
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
        debug=debug,
    )
    return run_paths
