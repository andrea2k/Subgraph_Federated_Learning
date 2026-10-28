from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["NodeOutputHead"]


class NodeOutputHead(nn.Module):
    """
    Per-node task output head.

    head_type="single" preserves the old shared MLP that emits all task logits.
    head_type="multi" uses one independent binary head per label column and
    concatenates the logits back to [num_nodes, num_tasks].
    """

    def __init__(self, hidden_dim: int, out_dim: int, head_type: str):
        super().__init__()
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        head_type = str(head_type).lower()
        if head_type not in {"single", "multi"}:
            raise ValueError(
                f"Unknown output_head={head_type!r}. Expected 'single' or 'multi'."
            )

        self.head_type = head_type
        self.out_dim = int(out_dim)

        if self.head_type == "single":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(out_dim)
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "single":
            return self.head(x)
        return torch.cat([head(x) for head in self.heads], dim=-1)
