"""Deterministic dataset partitions shared by training and evaluation."""
from __future__ import annotations

import math
import random
from collections.abc import Mapping


_SPLIT_NAMES = ("train", "val", "test")


def split_dataset_indices(
    dataset_size: int,
    fractions: Mapping[str, float],
    seed: int = 42,
) -> dict[str, list[int]]:
    """Partition sample indices reproducibly using largest-remainder rounding."""
    if dataset_size < len(_SPLIT_NAMES):
        raise ValueError("At least three samples are required for train/val/test splits.")

    try:
        weights = [float(fractions[name]) for name in _SPLIT_NAMES]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Dataset splits must define train, val, and test fractions.") from exc
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("Train, val, and test split fractions must be positive and finite.")
    total = sum(weights)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Train, val, and test split fractions must sum to 1.")

    exact_sizes = [dataset_size * weight / total for weight in weights]
    sizes = [math.floor(size) for size in exact_sizes]
    remainder_order = sorted(
        range(len(_SPLIT_NAMES)),
        key=lambda index: exact_sizes[index] - sizes[index],
        reverse=True,
    )
    for index in remainder_order[: dataset_size - sum(sizes)]:
        sizes[index] += 1
    if any(size == 0 for size in sizes):
        raise ValueError("Dataset is too small to populate every configured split.")

    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    result = {}
    start = 0
    for name, size in zip(_SPLIT_NAMES, sizes):
        result[name] = indices[start : start + size]
        start += size
    return result
