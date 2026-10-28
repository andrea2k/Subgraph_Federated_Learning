#!/usr/bin/env python3
from __future__ import annotations
import os
import random
import hashlib
import numpy as np
import torch


def set_seed(
    seed: int,
    *,
    deterministic_torch: bool = True,
    set_python_hashseed: bool = False,
) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch.
    """
    if set_python_hashseed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # To set strict determinism for all operations, uncomment:
        # torch.use_deterministic_algorithms(True)


def derive_seed(base: int, name: str) -> int:
    """
    Derive a stable per-split/per-run seed from a base seed and a label.
    """
    h = hashlib.blake2b(f"{base}:{name}".encode(), digest_size=8).hexdigest()
    return int(h, 16) % (2**31 - 1)
