from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import torch
from fedptha.training.state import (
    _named_parameter_dict,
    personalized_call,
    self_buffer_dict,
)


@dataclass
class RoutingState:
    """Task-head routing state for one receiving FedPTHA client.

    ``task_logits`` are unconstrained learnable preference scores. They are
    NEVER used directly as head coefficients. The actual task-head donor
    distribution is:

        p_i,t = rho_i,t * support_t
              + (1 - rho_i,t) * softmax(task_logits_i,t)

    Therefore every effective donor coefficient is non-negative and every task
    row sums to one. ``support`` and ``rho`` are computed only from visible
    training-label support. ``backbone`` is retained only for FedAvg/p0 logging
    compatibility and is not optimized or used for backbone mixing.
    """

    backbone: torch.nn.Parameter
    task_logits: torch.nn.Parameter
    support: torch.Tensor
    rho: torch.Tensor
    mode: str = "adaptive"


def effective_task_weights(dr_vector: RoutingState) -> torch.Tensor:
    if getattr(dr_vector, "mode", "adaptive") == "learned":
        result = _learned_effective_task_weights(dr_vector)
        return result
    if getattr(dr_vector, "mode", "adaptive") == "support":
        result = _support_effective_task_weights(dr_vector)
        return result
    """Return actual non-negative task-head donor coefficients [task, donor]."""
    learned = torch.softmax(dr_vector.task_logits, dim=1)
    support = dr_vector.support.to(device=learned.device, dtype=learned.dtype)
    rho = dr_vector.rho.to(device=learned.device, dtype=learned.dtype).unsqueeze(1)
    weights = rho * support + (1.0 - rho) * learned
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


def learned_task_weights(dr_vector: RoutingState) -> torch.Tensor:
    if getattr(dr_vector, "mode", "adaptive") == "support":
        result = _support_learned_task_weights(dr_vector)
        return result
    """Receiver-specific APPLE component A_i,t = softmax(z_i,t)."""
    return torch.softmax(dr_vector.task_logits, dim=1)


def routing_matrix_cpu(
    dr_vectors: Sequence[RoutingState],
) -> torch.Tensor:
    """Return EFFECTIVE routing tensor p with shape [receiver, task, donor]."""
    return torch.stack(
        [effective_task_weights(state).detach().cpu().clone() for state in dr_vectors],
        dim=0,
    )


def learned_matrix_cpu(
    dr_vectors: Sequence[RoutingState],
) -> torch.Tensor:
    """Return learned softmax component A with shape [receiver, task, donor]."""
    return torch.stack(
        [learned_task_weights(state).detach().cpu().clone() for state in dr_vectors],
        dim=0,
    )


def logits_matrix_cpu(
    dr_vectors: Sequence[RoutingState],
) -> torch.Tensor:
    return torch.stack(
        [state.task_logits.detach().cpu().clone() for state in dr_vectors], dim=0
    )


def support_matrix_cpu(
    dr_vectors: Sequence[RoutingState],
) -> torch.Tensor:
    return torch.stack(
        [state.support.detach().cpu().clone() for state in dr_vectors], dim=0
    )


def reliance_matrix_cpu(
    dr_vectors: Sequence[RoutingState],
) -> torch.Tensor:
    return torch.stack(
        [state.rho.detach().cpu().clone() for state in dr_vectors], dim=0
    )


def backbone_matrix_cpu(
    dr_vectors: Sequence[RoutingState],
) -> torch.Tensor:
    """Return backbone p0/debug matrix with shape [receiver, donor]."""
    return torch.stack(
        [state.backbone.detach().cpu().clone() for state in dr_vectors], dim=0
    )


def count_visible_positives(
    runtime_clients,
    num_tasks: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Visible positive TRAIN support [client, task]; no q/oracle metadata."""
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


def compute_support_prior(
    visible_pos: torch.Tensor,
    *,
    pseudo_count: float = 1.0,
) -> torch.Tensor:
    """S_t,j: normalized visible-positive task expertise [task, donor]."""
    if visible_pos.ndim != 2:
        raise ValueError(
            f"visible_pos must be [client, task], got {tuple(visible_pos.shape)}"
        )
    score = visible_pos.T + float(pseudo_count)
    if torch.any(score < 0):
        raise ValueError("support scores must be non-negative")
    return score / score.sum(dim=1, keepdim=True).clamp_min(1e-12)


def compute_support_reliance(
    visible_pos: torch.Tensor,
    *,
    pseudo_count: float = 1.0,
) -> torch.Tensor:
    """Adaptive support reliance rho [receiver, task].

    rho_i,t = 1 - (s_i,t + c) / (max_j s_j,t + c)

    The same small pseudo-count c used for support smoothing avoids an exact
    rho=1 when a receiver has zero positives, so the learnable APPLE component
    keeps a (small) gradient path. If all clients have zero support for a task,
    numerator and denominator match and rho=0: there is no evidence-based expert
    signal to enforce.
    """
    c = float(pseudo_count)
    if c < 0.0:
        raise ValueError(f"pseudo_count must be non-negative, got {c}")
    task_max = visible_pos.max(dim=0).values
    denom = (task_max + c).clamp_min(1e-12)
    reliability = (visible_pos + c) / denom.unsqueeze(0)
    return (1.0 - reliability).clamp(0.0, 1.0)


def initialize_routing(
    num_clients: int,
    num_tasks: int,
    p0: torch.Tensor,
    *,
    runtime_clients,
    device: torch.device,
    support_pseudo_count: float = 1.0,
    routing_mode: str = "adaptive",
) -> List[RoutingState]:
    if routing_mode == "learned":
        result = _learned_initialize_routing(
            num_clients,
            num_tasks,
            p0,
            runtime_clients=runtime_clients,
            device=device,
            support_pseudo_count=support_pseudo_count,
        )
        for state in result:
            state.mode = routing_mode
        return result
    if routing_mode == "support":
        result = _support_initialize_routing(
            num_clients,
            num_tasks,
            p0,
            runtime_clients=runtime_clients,
            device=device,
            support_pseudo_count=support_pseudo_count,
        )
        for state in result:
            state.mode = routing_mode
        return result
    """Initialize revised APPLE-PostALA routing state.

    The learned component starts from the original sample-size distribution p0
    represented as logits log(p0). Visible support is a separate persistent term;
    it is not merely an initialization and is never used as a proximal target.
    """
    visible_pos = count_visible_positives(
        runtime_clients, int(num_tasks), device=device
    )
    support = compute_support_prior(visible_pos, pseudo_count=support_pseudo_count)
    rho_all = compute_support_reliance(visible_pos, pseudo_count=support_pseudo_count)

    base_logits = torch.log(p0.detach().clone().clamp_min(1e-12))
    task_logits_base = base_logits.unsqueeze(0).repeat(int(num_tasks), 1)

    print("[SupportSimplex] visible positive support [client, task]:")
    print(visible_pos.detach().cpu())
    print("[SupportSimplex] S task-support distribution [task, donor]:")
    print(support.detach().cpu())
    print("[SupportSimplex] adaptive rho [receiver, task]:")
    print(rho_all.detach().cpu())

    out: List[RoutingState] = []
    for receiver_idx in range(num_clients):
        out.append(
            RoutingState(
                backbone=torch.nn.Parameter(p0.detach().clone(), requires_grad=False),
                task_logits=torch.nn.Parameter(task_logits_base.clone()),
                support=support.detach().clone(),
                rho=rho_all[receiver_idx].detach().clone(),
            )
        )
    return out


def make_optimizer(
    core_model: torch.nn.Module,
    dr_vector: RoutingState,
    cfg: Dict,
    dr_lr: float,
):
    if cfg.get("routing_mode", "adaptive") == "support":
        result = _support_make_optimizer(core_model, dr_vector, cfg, dr_lr)
        return result
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
                "params": [dr_vector.task_logits],
                "lr": float(dr_lr),
                "weight_decay": 0.0,
            },
        ]
    )


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
    dr_vector: RoutingState,
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

    task_weights = effective_task_weights(dr_vector)

    if task_weights.ndim != 2:
        raise ValueError(
            f"Expected effective task routing shape [num_tasks, num_clients], "
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
            alpha = (
                ala_weights[name]
                .to(
                    device=receiver_value.device,
                    dtype=receiver_value.dtype,
                )
                .clamp(0.0, 1.0)
            )
            acc = receiver_value + alpha * (acc - receiver_value)

        mixed[name] = acc

    return mixed


def personalized_forward(
    models_for_mix: Sequence[torch.nn.Module],
    client_idx: int,
    dr_vector: RoutingState,
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
    return personalized_call(
        receiver_model,
        params,
        buffers,
        args=(x_in, edge_in),
        kwargs={"edge_attr_dict": edge_attr_dict, "device": device},
    )


def _learned_effective_task_weights(dr_vector: RoutingState) -> torch.Tensor:
    """Return Learned-Simplex donor coefficients p_i,t = softmax(z_i,t)."""
    return torch.softmax(dr_vector.task_logits, dim=1)


def _support_effective_task_weights(dr_vector: RoutingState) -> torch.Tensor:
    """Return fixed Support-Only donor coefficients p_i,t = S_t."""
    support = dr_vector.support
    return support / support.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _learned_initialize_routing(
    num_clients: int,
    num_tasks: int,
    p0: torch.Tensor,
    *,
    runtime_clients,
    device: torch.device,
    support_pseudo_count: float = 1.0,
) -> List[RoutingState]:
    """Initialize the Learned-Simplex + PostALA routing ablation.

    The learned donor distribution starts from the same sample-size p0 used by
    the full method, represented as logits log(p0). No visible positive support
    is read or used. For output-schema compatibility only, ``support`` is a
    uniform task-by-donor matrix and ``rho`` is fixed to zero for every
    receiver/task, so the effective routing is exactly softmax(task_logits).
    """
    _ = runtime_clients
    _ = support_pseudo_count

    base_logits = torch.log(p0.detach().clone().clamp_min(1e-12))
    task_logits_base = base_logits.unsqueeze(0).repeat(int(num_tasks), 1)

    support = torch.full(
        (int(num_tasks), int(num_clients)),
        1.0 / float(num_clients),
        dtype=torch.float32,
        device=device,
    )
    rho_all = torch.zeros(
        (int(num_clients), int(num_tasks)),
        dtype=torch.float32,
        device=device,
    )

    print("[LearnedSimplex] visible-support prior DISABLED")
    print("[LearnedSimplex] task routing = softmax(task_logits)")
    print("[LearnedSimplex] task logits initialized from log(sample-size p0)")

    out: List[RoutingState] = []
    for receiver_idx in range(num_clients):
        out.append(
            RoutingState(
                backbone=torch.nn.Parameter(p0.detach().clone(), requires_grad=False),
                task_logits=torch.nn.Parameter(task_logits_base.clone()),
                support=support.detach().clone(),
                rho=rho_all[receiver_idx].detach().clone(),
            )
        )
    return out


def _support_initialize_routing(
    num_clients: int,
    num_tasks: int,
    p0: torch.Tensor,
    *,
    runtime_clients,
    device: torch.device,
    support_pseudo_count: float = 1.0,
) -> List[RoutingState]:
    """Initialize fixed visible-support routing for Support-Only + PostALA.

    For every receiver and task, the donor distribution is exactly the smoothed
    visible-positive TRAIN support prior S_t. No learned routing is optimized.
    """
    visible_pos = count_visible_positives(
        runtime_clients, int(num_tasks), device=device
    )
    support = compute_support_prior(visible_pos, pseudo_count=support_pseudo_count)
    # Clean support-only ablation: p_i,t = S_t for every receiver.
    rho_all = torch.ones(
        (int(num_clients), int(num_tasks)), device=device, dtype=support.dtype
    )
    task_logits_base = torch.log(support.detach().clone().clamp_min(1e-12))

    print("[SupportOnly] visible positive support [client, task]:")
    print(visible_pos.detach().cpu())
    print("[SupportOnly] fixed S task-support distribution [task, donor]:")
    print(support.detach().cpu())
    print("[SupportOnly] rho is fixed to one; learned task routing is disabled")

    out: List[RoutingState] = []
    for receiver_idx in range(num_clients):
        out.append(
            RoutingState(
                backbone=torch.nn.Parameter(p0.detach().clone(), requires_grad=False),
                task_logits=torch.nn.Parameter(
                    task_logits_base.clone(), requires_grad=False
                ),
                support=support.detach().clone(),
                rho=rho_all[receiver_idx].detach().clone(),
            )
        )
    return out


def _support_learned_task_weights(dr_vector: RoutingState) -> torch.Tensor:
    """Compatibility view: no learned routing exists; return the fixed S prior."""
    return effective_task_weights(dr_vector)


def _support_make_optimizer(
    core_model: torch.nn.Module,
    dr_vector: RoutingState,
    cfg: Dict,
    dr_lr: float,
):
    # Backbone parameters are optimized as ordinary local model parameters.
    # The backbone is later aggregated by FedAvg after local training.
    #
    # Support-only ablation: routing is fixed, so optimize model parameters only.
    # The backbone DR vector and task_logits exist only for schema compatibility.
    return torch.optim.Adam(
        [
            {
                "params": list(core_model.parameters()),
                "lr": float(cfg["lr"]),
                "weight_decay": float(cfg["weight_decay"]),
            }
        ]
    )
