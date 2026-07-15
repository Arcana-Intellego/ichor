"""Phase-B anti-overlap case (d).

Cases (a)-(c) live in ichor.hpc.active_learning.daemon (seed-side):

  (a) Seed selection vs QM reference data     -> select_seeds + provenance index
  (b) Recent-seeds cooldown               -> recent_seeds.json
  (c) Post-ARIADNE flag (not filter)      -> _post_ariadne_array

Case (d) is the sampling-side de-duplicate that runs after Phase B diversity FPS produces a
candidate set: for each candidate, compute the aligned mass-weighted RMSD
to the nearest committed training point; drop the candidate if it falls
below the configured min-separation threshold.

Why a separate module? The seed side cases need access to the trajectory
pool + the daemon state. case (d) operates on a fully-formed candidate
list (post-FPS) and a snapshot of the current QM reference data. Same
distance metric, different inputs.

The effective min_separation is resolved from the daemon's geometry-novelty
protocol scale rather than a public campaign.yaml Angstrom threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from ichor.core.adversarial.geometry import aligned_mass_weighted_rmsd
from ichor.core.atoms import Atoms


__all__ = [
    "DedupReport",
    "filter_candidates_against_training",
    "min_distance_to_training",
]


@dataclass(frozen=True)
class DedupReport:
    """Outcome of a single anti-overlap pass."""

    kept_indices: Tuple[int, ...]
    dropped_indices: Tuple[int, ...]
    distances_to_nearest: Tuple[float, ...]   # parallel to candidates
    min_separation: float

    @property
    def n_kept(self) -> int:
        return len(self.kept_indices)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_indices)


def min_distance_to_training(candidate: Atoms, training: Sequence[Atoms]) -> float:
    """Return the smallest aligned mass-weighted RMSD from "candidate"
    to any point in "training".

    Returns "float("inf")" if "training" is empty -- the candidate is
    trivially acceptable.
    """
    if not training:
        return float("inf")
    best = float("inf")
    for t in training:
        d = float(aligned_mass_weighted_rmsd(candidate, t))
        if d < best:
            best = d
    return best


def filter_candidates_against_training(
    candidates: Sequence[Atoms],
    training: Sequence[Atoms],
    *,
    min_separation: float,
) -> DedupReport:
    """Drop candidates whose distance to the nearest training point is
    below 'min_separation'. Order-preserving on the kept indices.

    Parameters
    ----------
    candidates
        Sequence of candidate Atoms (typically the Phase-B FPS output).
    training
        Sequence of currently committed training Atoms.
    min_separation
        Minimum allowed aligned mass-weighted distance to the nearest
        training point. Candidates with d < min_separation are dropped.

    Returns
    -------
    DedupReport
    """
    if min_separation < 0.0:
        raise ValueError("min_separation must be >= 0")

    kept: List[int] = []
    dropped: List[int] = []
    distances: List[float] = []
    for i, cand in enumerate(candidates):
        d = min_distance_to_training(cand, training)
        distances.append(d)
        if d < float(min_separation):
            dropped.append(i)
        else:
            kept.append(i)
    return DedupReport(
        kept_indices=tuple(kept),
        dropped_indices=tuple(dropped),
        distances_to_nearest=tuple(distances),
        min_separation=float(min_separation),
    )


