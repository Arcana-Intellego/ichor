from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from ichor.core.atoms import Atoms

from .config import SubspaceConfig
from .geometry import Neighbour, aligned_mass_weighted_displacement, atoms_to_coordinates, mass_vector


@dataclass(frozen=True)
class LocalSubspace:
    seed_atoms: Atoms
    neighbours: List[Neighbour]
    covariance: np.ndarray
    basis: np.ndarray
    eigenvalues: np.ndarray
    mode_weights: np.ndarray
    mass_vector: np.ndarray
    active_covariance: np.ndarray
    neighbour_weights: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])

    @property
    def cartesian_directions(self) -> List[np.ndarray]:
        m_inv_sqrt = 1.0 / np.sqrt(self.mass_vector)
        return [(m_inv_sqrt * self.basis[:, i]) for i in range(self.dimension)]



def _gaussian_neighbour_weights(distances: np.ndarray, sigma: float | None) -> np.ndarray:
    if sigma is None:
        nonzero = distances[distances > 0.0]
        sigma = float(np.median(nonzero)) if nonzero.size else 1.0
    sigma = max(float(sigma), 1.0e-12)
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
        return np.ones_like(distances)
    return weights



def _canonicalise_basis(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    degeneracy_tolerance: float,
    sign_threshold: float = 1.0e-10,
) -> np.ndarray:
    """Return a canonicalised copy of 'basis' whose representation is
    invariant under 
    (a) per-column sign and 
    (b) arbitrary rotation within near-degenerate eigenvalue blocks.

    The canonicalisation has two stages applied in order:

    1. Within each block of consecutive columns whose eigenvalue gap is less
       than 'degeneracy_tolerance * max(eigenvalues)', apply a Procrustes
       rotation against the first 'k' Cartesian unit vectors. This picks the
       within-block rotation that maximises alignment with a fixed reference
       frame and is therefore reproducible across runs.

    2. For each column, multiply by -1 if the first component whose absolute
       value exceeds 'sign_threshold' is negative.

    The output basis spans the same subspace and has the same eigenvalues.
    Acquisition values that only depend on the subspace (not its individual
    columns) are unchanged; per-mode breakdowns become reproducible.
    """
    if basis.size == 0:
        return basis.copy()

    canonical = basis.copy()
    n_coords, r = canonical.shape
    lam_max = float(np.max(eigenvalues)) if eigenvalues.size else 1.0
    threshold = float(degeneracy_tolerance) * max(lam_max, 1.0e-30)

    # 1. Procrustes rotation within near-degenerate blocks.
    block_start = 0
    for i in range(1, r):
        if abs(float(eigenvalues[i]) - float(eigenvalues[i - 1])) >= threshold:
            _procrustes_align_block(canonical, block_start, i, n_coords)
            block_start = i
    _procrustes_align_block(canonical, block_start, r, n_coords)

    # 2. Sign convention column-by-column.
    for j in range(r):
        col = canonical[:, j]
        for i in range(n_coords):
            if abs(col[i]) > sign_threshold:
                if col[i] < 0.0:
                    canonical[:, j] = -col
                break

    return canonical



def _procrustes_align_block(basis: np.ndarray, start: int, end: int, n_coords: int) -> None:
    """In-place rotate columns [start:end] of 'basis' to align with the
    first 'k = end - start' Cartesian unit vectors via Procrustes.

    No-op for single-column blocks (the sign-convention stage handles those).
    """
    k = end - start
    if k <= 1:
        return
    block = basis[:, start:end]
    #Reference: first k Cartesian unit vectors (3N x k with identity prefix).
    ref = np.zeros((n_coords, k), dtype=float)
    diag_len = min(n_coords, k)
    ref[np.arange(diag_len), np.arange(diag_len)] = 1.0
    overlap = block.T @ ref
    try:
        U, _, Vt = np.linalg.svd(overlap, full_matrices=False)
    except np.linalg.LinAlgError:
        return
    rotation = U @ Vt
    basis[:, start:end] = block @ rotation



def build_local_subspace(seed_atoms: Atoms, neighbours: Sequence[Neighbour], config: SubspaceConfig) -> LocalSubspace:
    if not neighbours:
        raise ValueError("Cannot build a local subspace without neighbours.")

    mass_vec = mass_vector(seed_atoms)
    displacements = np.vstack([aligned_mass_weighted_displacement(seed_atoms, item.atoms) for item in neighbours])
    distances = np.array([item.aligned_distance for item in neighbours], dtype=float)
    weights = _gaussian_neighbour_weights(distances, config.gaussian_weight_sigma)

    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        weights = np.ones_like(weights, dtype=float)
        weight_sum = float(np.sum(weights))
    covariance = np.zeros((displacements.shape[1], displacements.shape[1]), dtype=float)
    for w, y in zip(weights, displacements):
        covariance += w * np.outer(y, y)
    covariance /= weight_sum
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = np.maximum(eigenvalues, 0.0)
    total = float(np.sum(positive))
    if total <= 0.0:
        raise ValueError("Degenerate local covariance: all eigenvalues are non-positive.")

    cumulative = np.cumsum(positive) / total
    r = int(np.searchsorted(cumulative, config.variance_capture, side="left") + 1)
    r = max(config.min_subspace_dim, r)
    r = min(config.max_subspace_dim, r, covariance.shape[0])

    basis = eigenvectors[:, :r].copy()
    active_eigs = positive[:r].copy()
    relative_regularization = float(config.covariance_regularization) * max(
        float(active_eigs[0]) if active_eigs.size else 0.0,
        1.0e-12,
    )
    covariance += relative_regularization * np.eye(covariance.shape[0], dtype=float)
    if config.canonicalise_basis:
        basis = _canonicalise_basis(basis, active_eigs, config.degeneracy_tolerance)
    active_cov = np.diag(active_eigs)
    mode_weights = active_eigs / max(float(np.sum(active_eigs)), 1.0e-12)

    return LocalSubspace(
        seed_atoms=seed_atoms.copy(),
        neighbours=[Neighbour(item.index, item.atoms.copy(), item.aligned_distance) for item in neighbours],
        covariance=covariance,
        basis=basis,
        eigenvalues=active_eigs,
        mode_weights=mode_weights,
        mass_vector=mass_vec,
        active_covariance=active_cov,
        neighbour_weights=weights,
    )



def active_coordinates(subspace: LocalSubspace, atoms: Atoms) -> np.ndarray:
    disp = aligned_mass_weighted_displacement(subspace.seed_atoms, atoms)
    return subspace.basis.T @ disp



def _mass_normalisation(subspace: LocalSubspace) -> float:
    total_mass = float(np.sum(np.asarray(subspace.seed_atoms.masses, dtype=float)))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        total_mass = float(len(subspace.seed_atoms))
    return float(np.sqrt(max(total_mass, 1.0e-12)))



def active_and_residual_displacement(subspace: LocalSubspace, atoms: Atoms) -> Dict[str, np.ndarray]:
    """Split aligned mass-weighted displacement into active and residual parts."""
    disp = aligned_mass_weighted_displacement(subspace.seed_atoms, atoms)
    xi = subspace.basis.T @ disp
    active = subspace.basis @ xi
    residual = disp - active
    return {
        "displacement": disp,
        "active": active,
        "residual": residual,
        "active_coordinates": xi,
    }



def fullspace_residual_distance(subspace: LocalSubspace, atoms: Atoms) -> float:
    """Return active-subspace-orthogonal displacement in RMSD-like Angstrom units."""
    parts = active_and_residual_displacement(subspace, atoms)
    return float(np.linalg.norm(parts["residual"]) / _mass_normalisation(subspace))



def local_neighbour_residual_scale(
    subspace: LocalSubspace,
    floor: float = 1.0e-12,
    min_scale: float = 1.0e-3,
) -> float:
    """Robust residual-distance scale from the seed-local neighbourhood."""
    values = []
    for neighbour in subspace.neighbours:
        try:
            values.append(fullspace_residual_distance(subspace, neighbour.atoms))
        except Exception:
            continue
    finite = np.asarray([v for v in values if np.isfinite(v) and v > 0.0], dtype=float)
    if finite.size:
        return float(max(np.median(finite) + float(floor), float(min_scale)))
    return float(max(float(floor), float(min_scale), 1.0e-12))



def whitened_distance_squared(subspace: LocalSubspace, atoms: Atoms, regularization: float = 1.0e-10) -> float:
    xi = active_coordinates(subspace, atoms)
    eig_max = float(np.max(np.diag(subspace.active_covariance))) if subspace.dimension else 1.0
    reg = float(regularization) * max(eig_max, 1.0e-12)
    metric = np.linalg.inv(subspace.active_covariance + reg * np.eye(subspace.dimension))
    return float(xi.T @ metric @ xi)



def directional_step_sizes(subspace: LocalSubspace, step_scale: float, min_step: float, max_step: float) -> np.ndarray:
    steps = step_scale * np.sqrt(np.maximum(subspace.eigenvalues, 0.0))
    return np.clip(steps, min_step, max_step)
