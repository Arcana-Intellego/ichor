"""Train / validation split strategies for the second diversity sample.

CLI-pluggable via --split-strategy. Three options:

1. stratified_with_holdout (default):
    Top train_fraction (default 0.75) by acquisition alpha go to training,
    with high_holdout_fraction (default 0.10) of that top tier randomly
    held back into validation. Mid-tier val_mid_fraction (default 0.15) of
    the remaining candidates also go to validation. Provides validation
    coverage of the actual failure modes the model needs to capture.

2. random_80_20:
    Uniform random split with train_fraction (default 0.80).

3. pure_top_k:
    Top train_fraction to train, rest to val. Maximises training-set value
    but validation never sees failure modes; warned against in CLI help.

All strategies return a SplitResult with train_indices, val_indices, and
holdout_indices (a subset of val_indices flagged in the manifest). Indices
are into the candidate pool array; sort order within each is ascending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "SplitResult",
    "split_stratified_with_holdout",
    "split_random",
    "split_pure_top_k",
    "get_split_strategy",
    "AVAILABLE_STRATEGIES",
]


@dataclass(frozen=True)
class SplitResult:
    train_indices: List[int]
    val_indices: List[int]
    holdout_indices: List[int]
    strategy: str
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_val(self) -> int:
        return len(self.val_indices)

    @property
    def n_holdout(self) -> int:
        return len(self.holdout_indices)


def _validate_inputs(values, n_indices):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be 1D")
    if arr.size != n_indices:
        raise ValueError(f"len(values) {arr.size} != n_indices {n_indices}")
    return arr


def split_stratified_with_holdout(
    acquisition_values: Sequence[float],
    train_fraction: float = 0.75,
    val_mid_fraction: float = 0.15,
    high_holdout_fraction: float = 0.10,
    rng_seed: int = 0,
) -> SplitResult:
    arr = _validate_inputs(acquisition_values, len(acquisition_values))
    n = arr.size
    if n == 0:
        return SplitResult([], [], [], strategy="stratified_with_holdout")

    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be in [0, 1]")
    if not 0.0 <= val_mid_fraction <= 1.0:
        raise ValueError("val_mid_fraction must be in [0, 1]")
    if not 0.0 <= high_holdout_fraction <= 1.0:
        raise ValueError("high_holdout_fraction must be in [0, 1]")
    if train_fraction + val_mid_fraction > 1.0 + 1.0e-12:
        raise ValueError("train_fraction + val_mid_fraction must be <= 1")

    order = np.argsort(-arr, kind="stable")

    n_top_tier = max(1, int(round(train_fraction * n)))
    top_tier_indices = order[:n_top_tier].tolist()

    rng = np.random.default_rng(int(rng_seed))
    n_holdout = int(round(high_holdout_fraction * n_top_tier))
    n_holdout = max(0, min(n_holdout, n_top_tier))
    if n_top_tier > 0 and n_holdout >= n_top_tier:
        n_holdout = n_top_tier - 1
    if n_holdout > 0:
        holdout_choice = rng.choice(n_top_tier, size=n_holdout, replace=False)
        holdout_local = set(int(i) for i in holdout_choice)
    else:
        holdout_local = set()
    holdout_indices = [int(top_tier_indices[i]) for i in sorted(holdout_local)]
    train_indices = [
        int(top_tier_indices[i]) for i in range(n_top_tier) if i not in holdout_local
    ]

    n_remaining = n - n_top_tier
    n_val_mid = int(round(val_mid_fraction * n))
    n_val_mid = max(0, min(n_val_mid, n_remaining))
    val_mid_indices = [int(i) for i in order[n_top_tier : n_top_tier + n_val_mid]]
    low_remainder_indices = [int(i) for i in order[n_top_tier + n_val_mid :]]

    val_indices = sorted(val_mid_indices + low_remainder_indices + holdout_indices)

    return SplitResult(
        train_indices=sorted(train_indices),
        val_indices=val_indices,
        holdout_indices=sorted(holdout_indices),
        strategy="stratified_with_holdout",
        metadata={
            "train_fraction": float(train_fraction),
            "val_mid_fraction": float(val_mid_fraction),
            "high_holdout_fraction": float(high_holdout_fraction),
            "n_top_tier": int(n_top_tier),
            "n_low_remainder": int(len(low_remainder_indices)),
        },
    )


def split_random(
    acquisition_values: Sequence[float],
    train_fraction: float = 0.80,
    rng_seed: int = 0,
) -> SplitResult:
    arr = _validate_inputs(acquisition_values, len(acquisition_values))
    n = arr.size
    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be in [0, 1]")
    # the recorded strategy has to round-trip back through get_split_strategy,
    # so it stays the registry key rather than a fraction-bearing string. the
    # real train fraction still travels in metadata below.
    label = "random_80_20"
    if n == 0:
        return SplitResult([], [], [], strategy=label)

    rng = np.random.default_rng(int(rng_seed))
    perm = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    train_indices = sorted(int(i) for i in perm[:n_train])
    val_indices = sorted(int(i) for i in perm[n_train:])

    return SplitResult(
        train_indices=train_indices,
        val_indices=val_indices,
        holdout_indices=[],
        strategy=label,
        metadata={"train_fraction": float(train_fraction)},
    )


def split_pure_top_k(
    acquisition_values: Sequence[float],
    train_fraction: float = 0.80,
    rng_seed: int = 0,
) -> SplitResult:
    arr = _validate_inputs(acquisition_values, len(acquisition_values))
    n = arr.size
    if n == 0:
        return SplitResult([], [], [], strategy="pure_top_k")
    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be in [0, 1]")

    order = np.argsort(-arr, kind="stable")
    n_train = int(round(train_fraction * n))
    train_indices = sorted(int(i) for i in order[:n_train])
    val_indices = sorted(int(i) for i in order[n_train:])

    return SplitResult(
        train_indices=train_indices,
        val_indices=val_indices,
        holdout_indices=[],
        strategy="pure_top_k",
        metadata={"train_fraction": float(train_fraction)},
    )


AVAILABLE_STRATEGIES: Dict[str, Callable[..., SplitResult]] = {
    "stratified_with_holdout": split_stratified_with_holdout,
    "random_80_20": split_random,
    "pure_top_k": split_pure_top_k,
}


def get_split_strategy(name: str) -> Callable[..., SplitResult]:
    if name not in AVAILABLE_STRATEGIES:
        raise ValueError(
            f"unknown split strategy {name!r}; available: {sorted(AVAILABLE_STRATEGIES)}"
        )
    return AVAILABLE_STRATEGIES[name]
