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

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ichor.core.adversarial.geometry import aligned_mass_weighted_rmsd
from ichor.core.adversarial.geometry import _aligned_mass_weighted_rmsd_arrays
from ichor.core.atoms import Atoms


__all__ = [
    "DedupReport",
    "filter_candidates_against_training",
    "min_distance_to_training",
]


class PhaseBNoveltyDistanceOracle:
    """Exact once-per-pair novelty distances for one Phase B candidate set."""

    def __init__(
        self,
        candidates: Sequence[Atoms],
        reference_data: Any,
        *,
        workers: int = 1,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> None:
        if not candidates:
            raise ValueError("Phase B novelty oracle requires candidates")
        self._candidates = tuple(candidates)
        self._candidate_coordinates = tuple(
            np.asarray(frame.coordinates, dtype=np.float64) for frame in candidates
        )
        self._candidate_identities = tuple(
            tuple(
                (str(atom.type), int(atom.index), str(atom.units.value))
                for atom in frame
            )
            for frame in candidates
        )
        expected_identity = self._candidate_identities[0]
        if any(identity != expected_identity for identity in self._candidate_identities):
            raise ValueError("Phase B candidates have incompatible atom identities")
        self._masses = np.asarray(candidates[0].masses, dtype=np.float64)
        if (
            self._masses.shape != (len(candidates[0]),)
            or not np.all(np.isfinite(self._masses))
            or np.any(self._masses <= 0.0)
        ):
            raise ValueError("Phase B novelty masses must be finite and positive")
        if hasattr(reference_data, "coordinates") and hasattr(
            reference_data, "atom_types"
        ):
            reference_coordinates = np.asarray(
                reference_data.coordinates,
                dtype=np.float64,
            )
            reference_identity = tuple(
                (
                    str(atom_type),
                    int(atom_index),
                    str(unit),
                )
                for atom_type, atom_index, unit in zip(
                    reference_data.atom_types,
                    reference_data.atom_indices,
                    reference_data.atom_units,
                )
            )
            if reference_coordinates.shape[0] and reference_identity != expected_identity:
                raise ValueError(
                    "Phase B references and candidates have incompatible atom identities"
                )
            reference_masses = tuple(float(value) for value in reference_data.masses)
            if reference_coordinates.shape[0] and reference_masses != tuple(
                float(value) for value in self._masses
            ):
                raise ValueError("Phase B reference and candidate masses disagree")
            if reference_coordinates.shape == (0, 0, 3):
                reference_coordinates = np.empty(
                    (0, len(candidates[0]), 3),
                    dtype=np.float64,
                )
            self._reference_coordinates = reference_coordinates
        else:
            references = tuple(reference_data)
            for frame in references:
                identity = tuple(
                    (str(atom.type), int(atom.index), str(atom.units.value))
                    for atom in frame
                )
                if identity != expected_identity:
                    raise ValueError(
                        "Phase B references and candidates have incompatible atom identities"
                    )
            self._reference_coordinates = (
                np.stack(
                    [np.asarray(frame.coordinates, dtype=np.float64) for frame in references],
                    axis=0,
                )
                if references
                else np.empty((0, len(candidates[0]), 3), dtype=np.float64)
            )
        if (
            self._reference_coordinates.ndim != 3
            or self._reference_coordinates.shape[1:] != (
                len(candidates[0]),
                3,
            )
            or not np.all(np.isfinite(self._reference_coordinates))
        ):
            raise ValueError("Phase B reference coordinate array is invalid")
        self._reference_minima = np.full(len(candidates), np.inf, dtype=np.float64)
        self._candidate_distances = np.zeros(
            (len(candidates), len(candidates)),
            dtype=np.float64,
        )
        self._compute(
            workers=max(1, int(workers)),
            progress_callback=progress_callback,
        )

    def _compute(
        self,
        *,
        workers: int,
        progress_callback: Optional[Callable[..., None]],
    ) -> None:
        try:
            from threadpoolctl import threadpool_limits

            thread_limit = threadpool_limits(limits=1)
        except Exception:
            thread_limit = nullcontext()

        def row(candidate_index: int) -> Tuple[int, float, np.ndarray]:
            candidate = self._candidate_coordinates[candidate_index]
            best = float("inf")
            for reference in self._reference_coordinates:
                distance = float(
                    _aligned_mass_weighted_rmsd_arrays(
                        candidate,
                        reference,
                        self._masses,
                    )
                )
                if not np.isfinite(distance):
                    raise ValueError(
                        "anti-overlap distance is non-finite for a non-empty training set"
                    )
                if distance < best:
                    best = distance
            directed = np.zeros(len(self._candidates), dtype=np.float64)
            for other_index, other in enumerate(self._candidate_coordinates):
                if other_index == candidate_index:
                    continue
                distance = float(
                    _aligned_mass_weighted_rmsd_arrays(
                        candidate,
                        other,
                        self._masses,
                    )
                )
                if not np.isfinite(distance):
                    raise ValueError("Phase B candidate distance is non-finite")
                directed[other_index] = distance
            return candidate_index, best, directed

        with thread_limit:
            with ThreadPoolExecutor(
                max_workers=max(1, min(int(workers), len(self._candidates)))
            ) as pool:
                futures = {
                    pool.submit(row, index): index
                    for index in range(len(self._candidates))
                }
                completed = 0
                rows = {}
                for future in as_completed(futures):
                    index, best, directed = future.result()
                    rows[index] = (best, directed)
                    completed += 1
                    if progress_callback is not None and (
                        completed == len(self._candidates) or completed % 8 == 0
                    ):
                        try:
                            progress_callback(
                                stage="novelty_distance_oracle",
                                completed=int(completed),
                                total=int(len(self._candidates)),
                                unit="candidate rows",
                            )
                        except Exception:
                            pass
        for index in range(len(self._candidates)):
            best, directed = rows[index]
            self._reference_minima[index] = best
            self._candidate_distances[index, :] = directed
        self._reference_minima.setflags(write=False)
        self._candidate_distances.setflags(write=False)

    @property
    def candidate_reference_pairs(self) -> int:
        return int(len(self._candidates) * self._reference_coordinates.shape[0])

    @property
    def directed_candidate_pairs(self) -> int:
        return int(len(self._candidates) * max(0, len(self._candidates) - 1))

    def nearest_distance(
        self,
        candidate_index: int,
        accepted_candidate_indices: Sequence[int] = (),
    ) -> float:
        index = int(candidate_index)
        if not 0 <= index < len(self._candidates):
            raise IndexError("Phase B novelty candidate index is out of range")
        best = float(self._reference_minima[index])
        for accepted_index in accepted_candidate_indices:
            distance = float(self._candidate_distances[index, int(accepted_index)])
            if not np.isfinite(distance):
                raise ValueError("Phase B candidate distance is non-finite")
            if distance < best:
                best = distance
        return best


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
        if not np.isfinite(d):
            raise ValueError(
                "anti-overlap distance is non-finite for a non-empty training set"
            )
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
