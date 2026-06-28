"""Descriptors used by the POLUS wrapper for the two diversity passes.

A Descriptor implements one method, pairwise_distance_matrix(frames), which
returns an (N, N) symmetric float array of pairwise distances on whatever
metric the descriptor defines. The wrapper then runs greedy furthest-point
sampling over this matrix to choose a diverse subsample.

Three concrete descriptors are provided:

1. MassWeightedRMSDDescriptor: Phase A default. Kabsch-aligned mass-weighted
   RMSD; weights are sqrt(m_a) per Cartesian DOF, aligning the metric with
   the natural inner product on the configuration manifold.

2. HybridAlfRmsdDescriptor: Phase B recommended default. Blends mass-weighted
   RMSD with z-scored ALF-feature distance:

       d(a, b) = beta * RMSD_M(a, b) + (1 - beta) * ||phi(a) - phi(b)||_2

   The blend keeps geometric diversity while breaking failure-mode redundancy
   in the adversarial pool.

3. AcquisitionWeightedDescriptor: BatchBALD-style variance reweighting.
   Multiplies pairwise distances by sqrt(sigma2(i) * sigma2(j) / sigma_ref^4),
   so high-variance candidates are pulled apart by the FPS criterion.

All descriptors take an ICHOR Atoms sequence as input; they do not know
about POLUS internals or trajectory file formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, Sequence

import numpy as np

from ichor.core.atoms import Atoms

from ichor.hpc.active_learning.acquisition.rigid_projection import mass_vector_for


__all__ = [
    "Descriptor",
    "MassWeightedRMSDDescriptor",
    "HybridAlfRmsdDescriptor",
    "AcquisitionWeightedDescriptor",
    "kabsch_align",
    "mass_weighted_rmsd",
    "build_descriptor_from_config",
]


class Descriptor(Protocol):
    name: str

    def pairwise_distance_matrix(self, frames: Sequence[Atoms]) -> np.ndarray: ...


def kabsch_align(reference, mobile, weights=None):
    reference = np.asarray(reference, dtype=float)
    mobile = np.asarray(mobile, dtype=float)
    if reference.shape != mobile.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("shapes must match and be (N, 3)")
    if weights is None:
        w = np.ones(reference.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != reference.shape[0]:
            raise ValueError("weights size mismatch")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return mobile.copy()
    w_norm = w / w_sum
    ref_c = reference - (w_norm[:, None] * reference).sum(axis=0)
    mob_c = mobile - (w_norm[:, None] * mobile).sum(axis=0)
    H = (mob_c * w[:, None]).T @ ref_c
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    det = np.linalg.det(Vt.T @ U.T)
    D = np.eye(3)
    D[2, 2] = float(np.sign(det) if det != 0.0 else 1.0)
    R = Vt.T @ D @ U.T
    aligned = mob_c @ R.T + (w_norm[:, None] * reference).sum(axis=0)
    return aligned


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
    from ichor.core.calculators import calculate_alf_features
    from ichor.core.calculators import default_alf_calculator

    def extract(atoms):
        try:
            system_alf = atoms.alf(default_alf_calculator)
            features = [
                np.asarray(
                    atom.features(calculate_alf_features, system_alf),
                    dtype=float,
                ).reshape(-1)
                for atom in atoms
            ]
        except Exception as exc:
            raise RuntimeError(
                "Phase B ALF feature extraction failed: " + str(exc)
            ) from exc
        if not features:
            raise ValueError("Phase B ALF feature extraction produced no features")
        flat = np.concatenate(features)
        if not np.all(np.isfinite(flat)):
            raise ValueError("Phase B ALF feature extraction produced non-finite values")
        return flat

    return extract


@dataclass
class HybridAlfRmsdDescriptor:
    beta: float = 0.3
    name: str = "hybrid_alf_rmsd"
    feature_extractor: Optional[Callable[[Atoms], np.ndarray]] = None
    epsilon: float = 1.0e-12

    def pairwise_distance_matrix(self, frames):
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta out of range")
        n = len(frames)
        if n == 0:
            return np.zeros((0, 0), dtype=float)
        extractor = self.feature_extractor or _default_alf_feature_extractor()
        feature_rows = []
        feature_size = None
        for idx, frame in enumerate(frames):
            try:
                row = np.asarray(extractor(frame), dtype=float).reshape(-1)
            except Exception as exc:
                raise RuntimeError(
                    "hybrid_alf_rmsd feature extraction failed for frame "
                    + str(idx)
                    + ": "
                    + str(exc)
                ) from exc
            if row.size == 0:
                raise ValueError(
                    "hybrid_alf_rmsd feature extraction produced an empty "
                    + "feature vector for frame "
                    + str(idx)
                )
            if not np.all(np.isfinite(row)):
                raise ValueError(
                    "hybrid_alf_rmsd feature extraction produced non-finite "
                    + "values for frame "
                    + str(idx)
                )
            if feature_size is None:
                feature_size = int(row.size)
            elif int(row.size) != feature_size:
                raise ValueError(
                    "hybrid_alf_rmsd feature length mismatch: frame "
                    + str(idx)
                    + " has "
                    + str(int(row.size))
                    + " values, expected "
                    + str(feature_size)
                )
            feature_rows.append(row)
        feats = np.vstack(feature_rows)
        mu = feats.mean(axis=0, keepdims=True)
        sigma = feats.std(axis=0, keepdims=True)
        sigma = np.where(sigma > self.epsilon, sigma, 1.0)
        feats_z = (feats - mu) / sigma
        rmsd_mat = MassWeightedRMSDDescriptor().pairwise_distance_matrix(frames)
        sq = (feats_z[:, None, :] - feats_z[None, :, :]) ** 2
        feat_dist = np.sqrt(sq.sum(axis=2))
        return self.beta * rmsd_mat + (1.0 - self.beta) * feat_dist


@dataclass
class AcquisitionWeightedDescriptor:
    posterior: Any = None
    sigma_ref: Optional[float] = None
    base_descriptor: Optional[Descriptor] = None
    name: str = "acquisition_weighted"
    epsilon: float = 1.0e-12

    def pairwise_distance_matrix(self, frames):
        if self.posterior is None:
            raise ValueError("posterior is required")
        base = self.base_descriptor or MassWeightedRMSDDescriptor()
        base_matrix = np.asarray(base.pairwise_distance_matrix(frames), dtype=float)
        variances = np.array(
            [float(self.posterior.variance(f)) for f in frames],
            dtype=float,
        )
        sigma_ref_sq = (
            float(self.sigma_ref) ** 2
            if self.sigma_ref is not None
            else float(np.median(variances) + self.epsilon)
        )
        weights = np.sqrt(np.maximum(variances, 0.0) / max(sigma_ref_sq, self.epsilon))
        return base_matrix * np.sqrt(weights[:, None] * weights[None, :])




####################################################################################
# ------------------------------------------------------------------------
#                Factory: build a descriptor from CampaignConfig


def build_descriptor_from_config(config, *, posterior=None) -> "Descriptor":
    """Construct the descriptor instance named by config.phase_b.descriptor.

    Schema v2 makes "phase_b.beta" tunable; it was a dead in an earlier v.
    The returned descriptor instance carries the live beta /
    config-derived hyperparameters and is consumed by the production
    Phase-B (POLUS). For the dry-run executor, it is recorded
    in the per-pointdir provenance phase_b block instead of actually
    running POLUS.
    """
    name = config.phase_b.descriptor
    if name == "rmsd_massweight":
        return MassWeightedRMSDDescriptor()
    if name == "hybrid_alf_rmsd":
        return HybridAlfRmsdDescriptor(beta=config.phase_b.beta)
    if name == "acquisition_weighted":
        return AcquisitionWeightedDescriptor(
            posterior=posterior,
            base_descriptor=HybridAlfRmsdDescriptor(beta=config.phase_b.beta),
        )
    raise ValueError("unknown phase_b.descriptor: " + repr(name))




