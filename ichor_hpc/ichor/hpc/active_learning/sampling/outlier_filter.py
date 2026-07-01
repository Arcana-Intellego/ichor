"""Outlier filtering for the input trajectory (pre-Phase-A).

Two complementary filters:


1. Energy z-score: reject frames whose total / WFN energy lies more than
   z_threshold (default 3.0) standard deviations from the trajectory mean.
   Catches catastrophic SCF failures and equilibration artefacts.

2. Per-atom RMSD-from-mean z-score: reject frames where any single atom
   has |z| > z_threshold (default 4.0) in its mean-relative displacement.
   Catches single-atom-flies-away pathologies that pass the global RMSD
   filter because the rest of the molecule looks normal.

filter_initial_trajectory(frames, energies=None, ...) chains both filters
and returns an OutlierFilterResult with per-frame rejection reasons. The
active-learning daemon no longer applies this utility during trajectory-pool
import; it remains available for explicit offline/manual checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ichor.core.atoms import Atoms


__all__ = [
    "OutlierFilterResult",
    "filter_by_energy_zscore",
    "filter_by_per_atom_rmsd_zscore",
    "filter_initial_trajectory",
]


@dataclass
class OutlierFilterResult:
    kept_indices: List[int]
    rejected_indices: List[int]
    rejections_by_reason: List[Tuple[int, str]] = field(default_factory=list)
    energy_z_threshold: Optional[float] = None
    rmsd_z_threshold: Optional[float] = None

    @property
    def n_kept(self) -> int:
        return len(self.kept_indices)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected_indices)

    def as_json(self):
        return {
            "kept_count": self.n_kept,
            "rejected_count": self.n_rejected,
            "rejected": [
                {"index": int(i), "reason": str(r)} for i, r in self.rejections_by_reason
            ],
            "energy_z_threshold": self.energy_z_threshold,
            "rmsd_z_threshold": self.rmsd_z_threshold,
        }


def filter_by_energy_zscore(
    energies: Sequence[float],
    z_threshold: float = 3.0,
) -> Tuple[List[int], List[int]]:
    e = np.asarray(energies, dtype=float)
    if e.size == 0:
        return [], []
    finite = np.isfinite(e)
    # a non-finite energy (a blown-up SCF and the like) is an outlier by
    # definition. take those out before the mean/std, otherwise one NaN drags
    # the whole trajectory's statistics to NaN and every frame gets rejected.
    if not finite.any():
        return [], list(range(int(e.size)))
    mu = float(np.mean(e[finite]))
    sigma = float(np.std(e[finite]))
    if sigma <= 0.0:
        kept = [int(i) for i in np.where(finite)[0]]
        rejected = [int(i) for i in np.where(~finite)[0]]
        return kept, rejected
    z = np.full(e.shape, np.inf, dtype=float)
    z[finite] = (e[finite] - mu) / sigma
    keep_mask = finite & (np.abs(z) <= z_threshold)
    kept = [int(i) for i in np.where(keep_mask)[0]]
    rejected = [int(i) for i in np.where(~keep_mask)[0]]
    return kept, rejected




def filter_by_per_atom_rmsd_zscore(
    frames: Sequence[Atoms],
    z_threshold: float = 4.0,
) -> Tuple[List[int], List[int]]:
    n = len(frames)
    if n == 0:
        return [], []
    natoms = len(frames[0])
    coords = np.stack([np.asarray(f.coordinates, dtype=float) for f in frames], axis=0)
    # a frame carrying any non-finite coordinate is junk; keep it out of the
    # mean/std so it can't poison the per-atom statistics and sink every frame.
    # |z| is used so both tails of the displacement spread are caught.
    frame_finite = np.isfinite(coords).all(axis=(1, 2))
    if not frame_finite.any():
        return [], list(range(n))
    from .descriptors import kabsch_align

    ref_idx = int(np.where(frame_finite)[0][0])
    reference = coords[ref_idx]
    try:
        masses = np.asarray(frames[ref_idx].masses, dtype=float)
        if (
            masses.shape[0] != natoms
            or not np.isfinite(masses).all()
            or float(np.sum(masses)) <= 0.0
        ):
            masses = None
    except Exception:
        masses = None
    aligned = coords.copy()
    for i in np.where(frame_finite)[0]:
        aligned[int(i)] = kabsch_align(reference, coords[int(i)], weights=masses)

    mean_coords = aligned[frame_finite].mean(axis=0)
    diffs = aligned - mean_coords[None, :, :]
    per_atom_dist = np.linalg.norm(diffs, axis=2)
    good = per_atom_dist[frame_finite]
    mu = good.mean(axis=0)
    sigma = good.std(axis=0)
    sigma = np.where(sigma > 1.0e-30, sigma, 1.0)
    z = (per_atom_dist - mu[None, :]) / sigma[None, :]
    keep_mask = frame_finite & np.all(np.abs(z) <= z_threshold, axis=1)
    kept = [int(i) for i in np.where(keep_mask)[0]]
    rejected = [int(i) for i in np.where(~keep_mask)[0]]
    return kept, rejected






def filter_initial_trajectory(
    frames: Sequence[Atoms],
    energies: Optional[Sequence[float]] = None,
    energy_z_threshold: float = 3.0,
    rmsd_z_threshold: float = 4.0,
) -> OutlierFilterResult:
    n = len(frames)
    if n == 0:
        return OutlierFilterResult(
            kept_indices=[],
            rejected_indices=[],
            energy_z_threshold=energy_z_threshold,
            rmsd_z_threshold=rmsd_z_threshold,
        )
    rejected_reasons = []
    rejected_set = set()

    if energies is not None and len(energies) == n:
        _, energy_rej = filter_by_energy_zscore(energies, energy_z_threshold)
        for i in energy_rej:
            rejected_set.add(i)
            rejected_reasons.append((i, "energy_z"))

    _, rmsd_rej = filter_by_per_atom_rmsd_zscore(frames, rmsd_z_threshold)
    for i in rmsd_rej:
        if i not in rejected_set:
            rejected_reasons.append((i, "per_atom_rmsd_z"))
        rejected_set.add(i)

    kept = [i for i in range(n) if i not in rejected_set]
    return OutlierFilterResult(
        kept_indices=kept,
        rejected_indices=sorted(rejected_set),
        rejections_by_reason=sorted(rejected_reasons, key=lambda t: t[0]),
        energy_z_threshold=energy_z_threshold,
        rmsd_z_threshold=rmsd_z_threshold,
    )



