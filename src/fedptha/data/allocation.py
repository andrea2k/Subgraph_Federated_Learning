"""Deterministic integer allocation shared by generator and label masking."""

import math
from typing import Sequence, List


def allocation_counts(num_items: int, shares: Sequence[float]) -> List[int]:
    """
    Convert fractional shares into exact integer counts that sum to num_items.

    Example:
      num_items = 244
      shares = [0.5, 0.125, 0.125, 0.125, 0.125]
      raw = [122, 30.5, 30.5, 30.5, 30.5]

    This returns something like:
      [122, 31, 31, 30, 30]

    So no positive label is lost or duplicated.
    """
    n = int(num_items)
    raw = [float(s) * n for s in shares]
    counts = [int(math.floor(x)) for x in raw]

    remainder = n - int(sum(counts))
    if remainder < 0:
        raise ValueError(f"allocation counts overfull: n={n}, counts={counts}")

    # Add leftover items to largest fractional parts.
    frac_order = sorted(
        range(len(raw)),
        key=lambda i: (raw[i] - math.floor(raw[i]), -i),
        reverse=True,
    )

    for i in frac_order[:remainder]:
        counts[i] += 1

    if sum(counts) != n:
        raise AssertionError(f"counts do not sum to {n}: {counts}")

    return counts
