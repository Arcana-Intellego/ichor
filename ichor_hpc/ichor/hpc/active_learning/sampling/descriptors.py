"""Descriptors used by ICHOR's two exact diversity passes.

A Descriptor retains its dense ``pairwise_distance_matrix`` method for small
fixtures and compatibility. Production diversity execution evaluates the same
metric blockwise into an exact float64 condensed store, then exposes rows to
greedy furthest-point sampling without constructing an N by N matrix.

Three concrete descriptors are provided:

1. MassWeightedRMSDDescriptor: Phase A default. Kabsch-aligned mass-weighted
   RMSD; weights are sqrt(m_a) per Cartesian DOF, aligning the metric with
   the natural inner product on the configuration manifold.

2. HybridAlfRmsdDescriptor: Phase B recommended default. Blends mass-weighted
   RMSD with z-scored ALF-feature distance:

       d(a, b) = beta * RMSD_M(a, b) + (1 - beta) * ||phi(a) - phi(b)||_2

   The blend keeps geometric diversity while breaking failure-mode redundancy
   in the adversarial pool.

All descriptors take an ICHOR Atoms sequence as input; they do not know
about scheduler or trajectory file formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union
from pathlib import Path
import os

import numpy as np

from ichor.core.atoms import Atoms
from ichor.core.adversarial.geometry import kabsch_align as _core_kabsch_align

from ichor.hpc.active_learning.acquisition.rigid_projection import mass_vector_for


__all__ = [
    "Descriptor",
    "MassWeightedRMSDDescriptor",
    "HybridAlfRmsdDescriptor",
    "CondensedDistanceStore",
    "build_condensed_distance_store",
    "partition_descriptor_frames",
    "kabsch_align",
    "mass_weighted_rmsd",
    "build_descriptor_from_config",
]


class Descriptor(Protocol):
    name: str

    def pairwise_distance_matrix(self, frames: Sequence[Atoms]) -> np.ndarray: ...


def condensed_index(n: int, i: int, j: int) -> int:
    if i == j:
        raise ValueError("condensed distances do not store diagonal entries")
    if i > j:
        i, j = j, i
    if i < 0 or j >= int(n):
        raise IndexError("condensed-distance index is out of range")
    return int(n * i - i * (i + 1) // 2 + j - i - 1)


@dataclass
class CondensedDistanceStore:
    """Exact symmetric distance matrix backed by one condensed float64 vector."""

    n: int
    values: np.ndarray
    path: Optional[Path] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.n), int(self.n)

    def row(self, index: int) -> np.ndarray:
        i = int(index)
        if i < 0 or i >= int(self.n):
            raise IndexError("distance row is out of range")
        row = np.zeros(int(self.n), dtype=float)
        if i:
            js = np.arange(i, dtype=np.int64)
            indexes = self.n * js - js * (js + 1) // 2 + i - js - 1
            row[:i] = np.asarray(self.values[indexes], dtype=float)
        if i + 1 < int(self.n):
            start = condensed_index(int(self.n), i, i + 1)
            row[i + 1:] = np.asarray(
                self.values[start:start + (int(self.n) - i - 1)], dtype=float
            )
        return row

    def row_sums(self) -> np.ndarray:
        sums = np.zeros(int(self.n), dtype=float)
        for i in range(max(0, int(self.n) - 1)):
            start = condensed_index(int(self.n), i, i + 1)
            values = np.asarray(
                self.values[start:start + (int(self.n) - i - 1)], dtype=float
            )
            sums[i] += float(np.sum(values))
            sums[i + 1:] += values
        return sums

    def to_square(self) -> np.ndarray:
        matrix = np.zeros(self.shape, dtype=float)
        for i in range(int(self.n)):
            matrix[i, :] = self.row(i)
        return matrix

    def flush(self) -> None:
        flush = getattr(self.values, "flush", None)
        if callable(flush):
            flush()


def kabsch_align(reference, mobile, weights=None):
    """Use the single validated ICHOR weighted-alignment implementation."""
    return _core_kabsch_align(reference, mobile, weights=weights)


def mass_weighted_rmsd(reference: Atoms, mobile: Atoms) -> float:
    if len(reference) != len(mobile):
        raise ValueError("natoms differ")
    ref_xyz = np.asarray(reference.coordinates, dtype=float)
    mob_xyz = np.asarray(mobile.coordinates, dtype=float)
    masses = np.array([float(a.mass) for a in reference], dtype=float)
    aligned = kabsch_align(ref_xyz, mob_xyz, weights=masses)
    diff = aligned - ref_xyz
    per_atom_sq = np.sum(diff * diff, axis=1) * masses
    return float(np.sqrt(np.sum(per_atom_sq) / max(np.sum(masses), 1.0e-30)))


def _mass_weighted_rmsd_arrays(
    reference: np.ndarray,
    mobile: np.ndarray,
    masses: np.ndarray,
) -> float:
    aligned = kabsch_align(reference, mobile, weights=masses)
    diff = aligned - reference
    per_atom_sq = np.sum(diff * diff, axis=1) * masses
    return float(np.sqrt(np.sum(per_atom_sq) / max(np.sum(masses), 1.0e-30)))


@dataclass
class MassWeightedRMSDDescriptor:
    name: str = "rmsd_massweight"

    def pairwise_distance_matrix(self, frames):
        n = len(frames)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = mass_weighted_rmsd(frames[i], frames[j])
                D[i, j] = d
                D[j, i] = d
        return D


def _default_alf_feature_extractor():
    def extract(atoms):
        return _default_alf_features_and_cyclic_mask(atoms)[0]

    return extract


def _default_alf_features_and_cyclic_mask(
    atoms: Atoms,
) -> Tuple[np.ndarray, np.ndarray]:
    from ichor.core.calculators import calculate_alf_features
    from ichor.core.calculators import default_alf_calculator

    try:
        system_alf = atoms.alf(default_alf_calculator)
        features = []
        masks = []
        for atom in atoms:
            values = np.asarray(
                atom.features(calculate_alf_features, system_alf),
                dtype=float,
            ).reshape(-1)
            cyclic = np.zeros(values.size, dtype=bool)
            if len(atoms) > 3:
                cyclic[5::3] = True
            features.append(values)
            masks.append(cyclic)
    except Exception as exc:
        raise RuntimeError(
            "Phase B ALF feature extraction failed: " + str(exc)
        ) from exc
    if not features:
        raise ValueError("Phase B ALF feature extraction produced no features")
    flat = np.concatenate(features)
    cyclic_mask = np.concatenate(masks)
    if not np.all(np.isfinite(flat)):
        raise ValueError("Phase B ALF feature extraction produced non-finite values")
    return flat, cyclic_mask


def _normalise_hybrid_feature_matrix(
    values: np.ndarray,
    cyclic_mask: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Standardise linear ALF terms and encode azimuths on the unit circle."""
    matrix = np.asarray(values, dtype=float)
    mask = np.asarray(cyclic_mask, dtype=bool).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != mask.size:
        raise ValueError("hybrid ALF feature matrix/mask shape mismatch")
    columns = []
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        if mask[index]:
            columns.extend((np.sin(column), np.cos(column)))
            continue
        sigma = float(np.std(column))
        if sigma > float(epsilon):
            columns.append((column - float(np.mean(column))) / sigma)
        else:
            columns.append(np.zeros_like(column))
    return np.column_stack(columns) if columns else np.empty((len(matrix), 0))


def _frame_diagnostic_context(frame: Atoms) -> str:
    labels = []
    for atom in frame:
        label = getattr(atom, "name", None) or getattr(atom, "type", None)
        labels.append(str(label if label is not None else "?"))
    return "natoms=" + str(len(frame)) + " atom_labels=" + repr(labels)


@dataclass
class HybridAlfRmsdDescriptor:
    beta: float = 0.5
    name: str = "hybrid_alf_rmsd"
    feature_extractor: Optional[Callable[[Atoms], np.ndarray]] = None
    epsilon: float = 1.0e-12

    def pairwise_distance_matrix(self, frames):
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta out of range")
        n = len(frames)
        if n == 0:
            return np.zeros((0, 0), dtype=float)
        feats_z = _hybrid_features(self, frames)
        rmsd_mat = MassWeightedRMSDDescriptor().pairwise_distance_matrix(frames)
        sq = (feats_z[:, None, :] - feats_z[None, :, :]) ** 2
        feat_dist = np.sqrt(sq.sum(axis=2))
        return self.beta * rmsd_mat + (1.0 - self.beta) * feat_dist


_CONDENSED_WORKER_CONTEXT: Dict[str, Any] = {}


def _initialise_condensed_worker(context: Dict[str, Any]) -> None:
    global _CONDENSED_WORKER_CONTEXT
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    _CONDENSED_WORKER_CONTEXT = context


def _distance_for_pair(context: Dict[str, Any], i: int, j: int) -> float:
    positions = context["positions"]
    masses = context["masses"]
    rmsd = _mass_weighted_rmsd_arrays(positions[i], positions[j], masses)
    mode = str(context["mode"])
    if mode == "mass":
        value = rmsd
    else:
        features = context["features"]
        feature_distance = float(np.linalg.norm(features[i] - features[j]))
        beta = float(context["beta"])
        value = beta * rmsd + (1.0 - beta) * feature_distance
    return float(value)


def _condensed_row_block(task: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    context = _CONDENSED_WORKER_CONTEXT
    n = int(context["n"])
    first, stop = task
    start_index = condensed_index(n, int(first), int(first) + 1)
    count = sum(n - i - 1 for i in range(int(first), int(stop)))
    values = np.empty(int(count), dtype=np.float64)
    cursor = 0
    for i in range(int(first), int(stop)):
        for j in range(i + 1, n):
            values[cursor] = _distance_for_pair(context, i, j)
            cursor += 1
    return start_index, values


def _row_blocks(n: int, target_pairs: int = 100_000) -> Iterable[Tuple[int, int]]:
    first = 0
    accumulated = 0
    for i in range(max(0, int(n) - 1)):
        row_pairs = int(n) - i - 1
        if i > first and accumulated + row_pairs > int(target_pairs):
            yield first, i
            first = i
            accumulated = 0
        accumulated += row_pairs
    if first < int(n) - 1:
        yield first, int(n) - 1


def _hybrid_features(
    descriptor: HybridAlfRmsdDescriptor,
    frames: Sequence[Atoms],
) -> np.ndarray:
    if not 0.0 <= float(descriptor.beta) <= 1.0:
        raise ValueError("beta out of range")
    rows = []
    cyclic_mask: Optional[np.ndarray] = None
    expected: Optional[int] = None
    for index, frame in enumerate(frames):
        try:
            if descriptor.feature_extractor is None:
                row, observed_mask = _default_alf_features_and_cyclic_mask(frame)
            else:
                row = np.asarray(
                    descriptor.feature_extractor(frame), dtype=float
                ).reshape(-1)
                observed_mask = np.zeros(row.size, dtype=bool)
        except Exception as exc:
            raise RuntimeError(
                "hybrid_alf_rmsd feature extraction failed for frame "
                + str(index)
                + " ("
                + _frame_diagnostic_context(frame)
                + "): "
                + str(exc)
            ) from exc
        if row.size == 0 or not np.all(np.isfinite(row)):
            raise ValueError("hybrid_alf_rmsd produced invalid features")
        if expected is None:
            expected = int(row.size)
            cyclic_mask = observed_mask
        elif int(row.size) != expected:
            raise ValueError("hybrid_alf_rmsd feature length mismatch")
        elif not np.array_equal(cyclic_mask, observed_mask):
            raise ValueError("hybrid_alf_rmsd cyclic feature layout mismatch")
        rows.append(row)
    values = np.vstack(rows)
    return _normalise_hybrid_feature_matrix(
        values,
        np.asarray(cyclic_mask, dtype=bool),
        epsilon=float(descriptor.epsilon),
    )


def partition_descriptor_frames(
    descriptor: Descriptor,
    frames: Sequence[Atoms],
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Identify descriptor-singular frames without discarding valid peers."""
    kept: List[int] = []
    rejected: List[Dict[str, Any]] = []
    for index, frame in enumerate(frames):
        try:
            if isinstance(descriptor, HybridAlfRmsdDescriptor):
                if descriptor.feature_extractor is None:
                    _default_alf_features_and_cyclic_mask(frame)
                else:
                    values = np.asarray(
                        descriptor.feature_extractor(frame), dtype=float
                    ).reshape(-1)
                    if values.size == 0 or not np.all(np.isfinite(values)):
                        raise ValueError("custom descriptor features are invalid")
            else:
                coordinates = np.asarray(frame.coordinates, dtype=float)
                if coordinates.shape != (len(frame), 3) or not np.all(
                    np.isfinite(coordinates)
                ):
                    raise ValueError("frame coordinates are invalid")
        except Exception as exc:
            rejected.append({
                "candidate_index_zero_based": int(index),
                "reason": type(exc).__name__ + ": " + str(exc),
            })
            continue
        kept.append(int(index))
    return kept, rejected


def _condensed_context(
    descriptor: Descriptor,
    frames: Sequence[Atoms],
) -> Dict[str, Any]:
    if not frames:
        return {"n": 0, "mode": "mass", "positions": [], "masses": np.array([])}
    n_atoms = len(frames[0])
    if any(len(frame) != n_atoms for frame in frames):
        raise ValueError("diversity frames disagree on atom count")
    positions = [np.asarray(frame.coordinates, dtype=float) for frame in frames]
    masses = np.asarray([float(atom.mass) for atom in frames[0]], dtype=float)
    context: Dict[str, Any] = {
        "n": len(frames),
        "positions": positions,
        "masses": masses,
        "mode": "mass",
    }
    target: Any = descriptor
    if isinstance(target, HybridAlfRmsdDescriptor):
        context["features"] = _hybrid_features(target, frames)
        context["beta"] = float(target.beta)
        context["mode"] = "hybrid"
    elif not isinstance(target, MassWeightedRMSDDescriptor):
        raise TypeError(
            "condensed diversity execution does not support descriptor "
            + type(target).__name__
        )
    return context


def build_condensed_distance_store(
    descriptor: Descriptor,
    frames: Sequence[Atoms],
    *,
    path: Optional[Union[str, Path]] = None,
    workers: int = 1,
) -> CondensedDistanceStore:
    """Evaluate every exact pair without constructing an N by N array."""
    n = len(frames)
    n_pairs = int(n * (n - 1) // 2)
    if path is None or n_pairs == 0:
        values: np.ndarray = np.empty(n_pairs, dtype=np.float64)
        store_path = None if path is None else Path(path)
        if store_path is not None:
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.touch()
    else:
        store_path = Path(path)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        values = np.memmap(store_path, mode="w+", dtype=np.float64, shape=(n_pairs,))
    if n_pairs == 0:
        return CondensedDistanceStore(n=n, values=values, path=store_path)
    context = _condensed_context(descriptor, frames)
    blocks = list(_row_blocks(n))
    worker_count = max(1, min(int(workers), len(blocks)))
    if worker_count == 1:
        _initialise_condensed_worker(context)
        results = map(_condensed_row_block, blocks)
        for start, block_values in results:
            values[start:start + len(block_values)] = block_values
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialise_condensed_worker,
            initargs=(context,),
        ) as pool:
            for start, block_values in pool.map(_condensed_row_block, blocks):
                values[start:start + len(block_values)] = block_values
    store = CondensedDistanceStore(n=n, values=values, path=store_path)
    # Validate in bounded slices.  Calling isfinite() over a file-backed
    # vector materialises a second O(N^2) boolean array and defeats the
    # bounded-memory contract precisely for the largest pools.
    validation_chunk = 1_048_576
    for start in range(0, n_pairs, validation_chunk):
        chunk = np.asarray(values[start : start + validation_chunk])
        if not np.all(np.isfinite(chunk)) or np.any(chunk < 0.0):
            raise ValueError("condensed diversity distances are non-finite or negative")
    store.flush()
    return store




####################################################################################
# ------------------------------------------------------------------------
#                Factory: build a descriptor from CampaignConfig


def build_descriptor_from_config(config, *, posterior=None) -> "Descriptor":
    """Construct the descriptor instance named by config.phase_b.descriptor.

    Schema v2 makes "phase_b.beta" tunable; it was a dead in an earlier v.
    The returned descriptor instance carries the live beta /
    config-derived hyperparameters and is consumed by the production
    Phase B. For the dry-run executor, it is recorded
    in the per-pointdir provenance phase_b block instead of actually
    submitting a diversity job.
    """
    name = config.phase_b.descriptor
    if name == "rmsd_massweight":
        return MassWeightedRMSDDescriptor()
    if name == "hybrid_alf_rmsd":
        return HybridAlfRmsdDescriptor(beta=config.phase_b.beta)
    raise ValueError("unknown phase_b.descriptor: " + repr(name))
