"""Seed selection for the adversarial attack phase.

Half of the seeds are sampled uniformly at random from the current training
set; the other half are chosen as the top variance points by
posterior.variance(x) on the current GP model. The split mirrors the
explore-vs-exploit decomposition of expected information gain (random covers
the bulk of the support; variance-weighted concentrates on weak regions).

Current wiring: callers may pass "training_frame_ids" (parallel to
"training_atoms") that label each training point with its stable trajectory
frame_id, plus "forbidden_frame_ids" listing frames that must be excluded
(union of training-pool provenance and the recent-seeds cooldown cache).
:class:"SeedSelection" carries "frame_ids" for the picked positions so the
provenance ledger can record exactly which MD frames seeded which committed
pointdir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Sequence

import numpy as np

from ichor.core.atoms import Atoms


__all__ = ["SeedSelection", "select_seeds"]


def _finite_variances(values, *, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(context + " contains non-finite posterior variance")
    return arr


def _posterior_variances(posterior, points, *, chunk_size: Optional[int]):
    if not hasattr(posterior, "variances"):
        return [posterior.variance(point) for point in points]
    if chunk_size is None:
        return posterior.variances(points)
    try:
        return posterior.variances(points, chunk_size=int(chunk_size))
    except TypeError as exc:
        # Compatibility with older tests and light-weight posterior adapters
        # that implement variances(points) but not the new chunk-size keyword.
        if "chunk_size" not in str(exc):
            raise
        return posterior.variances(points)


@dataclass(frozen=True)
class SeedSelection:
    """Structured result of `select_seeds`."""

    seeds: List[Atoms]
    indices: List[int]
    bulk_indices: List[int]
    variance_indices: List[int]
    variances: List[float]
    frame_ids: List[Optional[int]] = field(default_factory=list)
    skipped_unknown_provenance: int = 0

    @property
    def n(self) -> int:
        return len(self.indices)


def select_seeds(
    training_atoms: Sequence[Atoms],
    posterior,
    n_seeds: int,
    *,
    bulk_fraction: float = 0.5,
    rng_seed: int = 0,
    training_frame_ids: Optional[Sequence[Optional[int]]] = None,
    forbidden_frame_ids: FrozenSet[int] = frozenset(),
    variance_chunk_size: Optional[int] = None,
) -> SeedSelection:
    """Return n_seeds seeds from training_atoms.

    Half (rounded) are uniform-random; the other half are the highest-variance
    points by posterior.variance(x) among the remaining training points.

    Parameters
    ----------
    training_atoms
        Sequence of training :class:Atoms.
    posterior
        Object exposing variance(atoms) -> float. The
        :class:ichor.core.adversarial.posterior.TotalEnergyPosterior already
        satisfies this contract.
    n_seeds
        Total number of seeds to return. If >= number of eligible positions,
        all eligible positions are returned.
    bulk_fraction
        Fraction (in [0, 1]) of n_seeds allocated to the random bulk pick.
        The remainder goes to the variance-weighted pick.
    rng_seed
        Seed for the random bulk selection. Variance ranking is deterministic
        for a given posterior.
    training_frame_ids
        Optional parallel sequence of stable trajectory frame_ids
        (one per training_atoms entry, with None for points without a
        frame_id). Used together with forbidden_frame_ids to filter the
        eligible-position set. When None (the legacy default) no filtering
        is performed and the returned frame_ids are all None.
    forbidden_frame_ids
        Stable trajectory frame_ids that MUST NOT be picked (the union of the
        training-pool provenance frame_ids and the recent-seeds cooldown
        cache). Positions whose training_frame_ids[i] is in this set are
        excluded before sampling.
    variance_chunk_size
        Optional chunk size passed to posterior.variances when the posterior
        supports that keyword.

    Returns
    -------
    SeedSelection
        seeds in the order: bulk first, then variance. frame_ids is
        parallel to seeds (None entries are preserved when the caller
        passed no frame_ids or when the picked training row has None).
    """
    if not 0.0 <= bulk_fraction <= 1.0:
        raise ValueError(f"bulk_fraction must be in [0, 1]; got {bulk_fraction}")

    n_total = len(training_atoms)
    if n_total <= 0:
        raise ValueError("training_atoms is empty")
    if n_seeds <= 0:
        return SeedSelection([], [], [], [], [], [])

    #build the parallel frame_ids vector. If the caller passed None, treat
    #every training row as having no frame_id (and therefore unfilterable).
    if training_frame_ids is None:
        fids: List[Optional[int]] = [None] * n_total
    else:
        if len(training_frame_ids) != n_total:
            raise ValueError(
                "training_frame_ids length "
                + str(len(training_frame_ids))
                + " != training_atoms length " + str(n_total)
            )
        fids = [int(f) if isinstance(f, int) else None for f in training_frame_ids]

    forbidden = frozenset(int(f) for f in forbidden_frame_ids)

    # Filter eligible positions. When the forbidden ledger is active, unknown provenance is
    # excluded instead of being silently eligible: otherwise old pointdirs without frame_id metadata
    # can bypass the anti-repeat guard and re-seed already-forbidden trajectory frames.
    skipped_unknown = sum(1 for fid in fids if fid is None) if forbidden else 0
    eligible: List[int] = [
        i for i in range(n_total)
        if (fids[i] is not None and fids[i] not in forbidden) or (not forbidden and fids[i] is None)
    ]
    n_eligible = len(eligible)
    if n_eligible <= 0:
        #forbidden set ate every training point; return empty selection.
        return SeedSelection([], [], [], [], [], [], skipped_unknown)

    if n_seeds >= n_eligible:
        all_idx = list(eligible)
        raw_variances = _posterior_variances(
            posterior,
            [training_atoms[i] for i in all_idx],
            chunk_size=variance_chunk_size,
        )
        variances = [
            float(v)
            for v in _finite_variances(raw_variances, context="seed selection")
        ]
        return SeedSelection(
            seeds=[training_atoms[i] for i in all_idx],
            indices=all_idx,
            bulk_indices=all_idx,
            variance_indices=[],
            variances=variances,
            frame_ids=[fids[i] for i in all_idx],
            skipped_unknown_provenance=skipped_unknown,
        )

    rng = np.random.default_rng(int(rng_seed))
    n_bulk = int(round(bulk_fraction * n_seeds))
    n_bulk = max(0, min(n_bulk, n_seeds))
    n_variance = n_seeds - n_bulk

    eligible_arr = np.asarray(eligible, dtype=int)
    if n_bulk > 0:
        bulk_pick = rng.choice(len(eligible_arr), size=n_bulk, replace=False)
        bulk_idx = [int(eligible_arr[p]) for p in bulk_pick]
    else:
        bulk_idx = []
    bulk_set = set(bulk_idx)

    remaining = [i for i in eligible if i not in bulk_set]
    # batched scan when the posterior supports it (the real GP). avoids a
    # python variance() call per pool frame on the login node.
    remaining_vars = np.asarray(
        _posterior_variances(
            posterior,
            [training_atoms[i] for i in remaining],
            chunk_size=variance_chunk_size,
        ),
        dtype=float,
    )
    remaining_vars = _finite_variances(remaining_vars, context="seed ranking")
    if remaining:
        order = np.argsort(-remaining_vars, kind="stable")
        variance_idx = [
            remaining[int(order[k])]
            for k in range(min(n_variance, len(remaining)))
        ]
    else:
        variance_idx = []

    all_idx = bulk_idx + variance_idx
    variances = [
        float(v)
        for v in _finite_variances(
            _posterior_variances(
                posterior,
                [training_atoms[i] for i in all_idx],
                chunk_size=variance_chunk_size,
            ),
            context="seed selection",
        )
    ]

    return SeedSelection(
        seeds=[training_atoms[i] for i in all_idx],
        indices=all_idx,
        bulk_indices=bulk_idx,
        variance_indices=variance_idx,
        variances=variances,
        frame_ids=[fids[i] for i in all_idx],
        skipped_unknown_provenance=skipped_unknown,
    )
