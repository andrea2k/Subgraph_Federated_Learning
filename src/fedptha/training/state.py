from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import torch

from torch.func import functional_call


def core_state_dicts(
    core_models: Sequence[torch.nn.Module],
) -> List[Dict[str, torch.Tensor]]:
    return [clone_state_dict(model.state_dict()) for model in core_models]


def make_p0_from_train_sizes(runtime_clients, device: torch.device) -> torch.Tensor:
    sizes = torch.tensor(
        [float(runtime.num_train_nodes) for runtime in runtime_clients],
        dtype=torch.float32,
        device=device,
    )
    total = sizes.sum().clamp_min(1.0)
    return sizes / total


def _named_parameter_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return dict(model.named_parameters())


def self_buffer_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Use the receiving client's own buffers, especially BatchNorm statistics.

    APPLE's equations are over learnable model parameters. For PyTorch modules,
    state_dict also contains non-learnable buffers. Using the receiving client's
    buffers keeps BatchNorm state local instead of mixing running statistics with
    unconstrained, possibly negative DR coefficients.
    """
    return dict(model.named_buffers())


def personalized_call(model, params, buffers, args, kwargs=None):
    return functional_call(model, (params, buffers), args, kwargs or {}, strict=False)


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


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}
