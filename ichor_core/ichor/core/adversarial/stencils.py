from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from ichor.core.atoms import Atoms

from .geometry import copy_atoms_with_flat_displacement
from .posterior import TotalEnergyPosterior


class PosteriorNumericsError(FloatingPointError):
    """Posterior covariance cannot support a trustworthy stencil variance."""


@dataclass(frozen=True)
class StencilEvaluation:
    offsets: np.ndarray
    coefficients: np.ndarray
    points: List[Atoms]
    mean: float
    variance: float

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))


@dataclass(frozen=True)
class DirectionalStencilBundle:
    offsets: np.ndarray
    points: List[Atoms]
    means: np.ndarray
    covariance: np.ndarray
    force: StencilEvaluation
    curvature: StencilEvaluation
    cubic: StencilEvaluation
    quartic: StencilEvaluation



def displaced_geometry(atoms: Atoms, direction_flat: np.ndarray, displacement: float) -> Atoms:
    return copy_atoms_with_flat_displacement(atoms, displacement * np.asarray(direction_flat, dtype=float))


def _stencil_from_coeffs(
    offsets: np.ndarray,
    coefficients: np.ndarray,
    points: Sequence[Atoms],
    means: np.ndarray,
    covariance: np.ndarray,
) -> StencilEvaluation:
    offsets_arr = np.asarray(offsets, dtype=float)
    coeffs_arr = np.asarray(coefficients, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    covariance_arr = np.asarray(covariance, dtype=float)
    n_points = int(coeffs_arr.size)
    if offsets_arr.shape != (n_points,) or means_arr.shape != (n_points,):
        raise PosteriorNumericsError(
            "stencil offsets, coefficients and means must have equal length"
        )
    if len(points) != n_points:
        raise PosteriorNumericsError("stencil point count does not match coefficients")
    if covariance_arr.shape != (n_points, n_points):
        raise PosteriorNumericsError(
            "stencil posterior covariance must be square with one row per point"
        )
    if not (
        np.all(np.isfinite(offsets_arr))
        and np.all(np.isfinite(coeffs_arr))
        and np.all(np.isfinite(means_arr))
        and np.all(np.isfinite(covariance_arr))
    ):
        raise PosteriorNumericsError("stencil inputs must be finite")
    covariance_scale = max(1.0, float(np.max(np.abs(covariance_arr))))
    symmetry_tolerance = (
        np.finfo(float).eps * max(1, n_points) * covariance_scale * 128.0
    )
    asymmetry = float(np.max(np.abs(covariance_arr - covariance_arr.T)))
    if asymmetry > symmetry_tolerance:
        raise PosteriorNumericsError(
            "stencil posterior covariance is materially asymmetric"
        )
    covariance_arr = 0.5 * (covariance_arr + covariance_arr.T)
    mean = float(coeffs_arr @ means_arr)
    variance = float(coeffs_arr @ covariance_arr @ coeffs_arr)
    variance_tolerance = (
        np.finfo(float).eps
        * max(1, n_points)
        * max(1.0, float(np.dot(coeffs_arr, coeffs_arr)))
        * covariance_scale
        * 256.0
    )
    if variance < -variance_tolerance:
        raise PosteriorNumericsError(
            "stencil posterior covariance produced a materially negative variance: "
            + repr(variance)
        )
    variance = max(0.0, variance)
    return StencilEvaluation(
        offsets=offsets_arr,
        coefficients=coeffs_arr,
        points=list(points),
        mean=mean,
        variance=variance,
    )



def evaluate_linear_stencil(
    posterior: TotalEnergyPosterior,
    atoms: Atoms,
    direction_flat: np.ndarray,
    offsets: Sequence[float],
    coefficients: Sequence[float],
    step: float,
) -> StencilEvaluation:
    offsets_arr = np.asarray(offsets, dtype=float)
    coeffs_arr = np.asarray(coefficients, dtype=float)
    step_f = float(step)
    if not np.isfinite(step_f) or step_f <= 0.0:
        raise ValueError("directional stencil step must be finite and positive")
    points = [displaced_geometry(atoms, direction_flat, float(offset) * step_f) for offset in offsets_arr]
    means = posterior.means(points)
    covariance = posterior.covariance_matrix(points)
    return _stencil_from_coeffs(offsets_arr, coeffs_arr, points, means, covariance)



def directional_all_stencils(
    posterior: TotalEnergyPosterior,
    atoms: Atoms,
    direction_flat: np.ndarray,
    step: float,
    *,
    prepared: bool = False,
) -> DirectionalStencilBundle:
    offsets, points = directional_stencil_points(
        atoms,
        direction_flat,
        step,
    )
    step_f = float(step)
    if bool(prepared) and hasattr(posterior, "prepare_points"):
        prepared_batch = posterior.prepare_points(points)
        return directional_stencils_from_prepared(
            prepared_batch,
            row_ids=prepared_batch.row_ids,
            points=points,
            step=step_f,
        )
    if hasattr(posterior, "means_and_covariance_matrix"):
        means_raw, covariance_raw = posterior.means_and_covariance_matrix(points)
    else:
        means_raw = posterior.means(points)
        covariance_raw = posterior.covariance_matrix(points)
    return _directional_stencils_from_values(
        offsets=offsets,
        points=points,
        means=means_raw,
        covariance=covariance_raw,
        step=step_f,
    )


def directional_stencil_points(
    atoms: Atoms,
    direction_flat: np.ndarray,
    step: float,
) -> tuple[np.ndarray, List[Atoms]]:
    """Build the five geometries shared by all directional stencils."""
    step_f = float(step)
    if not np.isfinite(step_f) or step_f <= 0.0:
        raise ValueError("directional stencil step must be finite and positive")
    offsets = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    points = [
        displaced_geometry(atoms, direction_flat, float(offset) * step_f)
        for offset in offsets
    ]
    return offsets, points


def directional_stencils_from_prepared(
    prepared_batch,
    *,
    row_ids: Sequence[int],
    points: Sequence[Atoms],
    step: float,
) -> DirectionalStencilBundle:
    """Evaluate one five-point stencil from a reusable posterior batch."""
    ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    if ids.shape != (5,) or len(points) != 5:
        raise ValueError("directional prepared stencil requires five points")
    return _directional_stencils_from_values(
        offsets=np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float),
        points=points,
        means=prepared_batch.means_by_index(ids),
        covariance=prepared_batch.cross_covariances_by_index(ids, ids),
        step=step,
    )


def _directional_stencils_from_values(
    *,
    offsets: np.ndarray,
    points: Sequence[Atoms],
    means: np.ndarray,
    covariance: np.ndarray,
    step: float,
) -> DirectionalStencilBundle:
    step_f = float(step)
    if not np.isfinite(step_f) or step_f <= 0.0:
        raise ValueError("directional stencil step must be finite and positive")
    means = np.asarray(means, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    coeff_force = np.asarray([0.0, -1.0, 0.0, 1.0, 0.0], dtype=float) / (2.0 * step_f)
    coeff_curvature = np.asarray([0.0, 1.0, -2.0, 1.0, 0.0], dtype=float) / (step_f ** 2)
    coeff_cubic = np.asarray([-1.0, 2.0, 0.0, -2.0, 1.0], dtype=float) / (2.0 * step_f ** 3)
    coeff_quartic = np.asarray([1.0, -4.0, 6.0, -4.0, 1.0], dtype=float) / (step_f ** 4)

    return DirectionalStencilBundle(
        offsets=offsets,
        points=list(points),
        means=means,
        covariance=covariance,
        force=_stencil_from_coeffs(offsets, coeff_force, points, means, covariance),
        curvature=_stencil_from_coeffs(offsets, coeff_curvature, points, means, covariance),
        cubic=_stencil_from_coeffs(offsets, coeff_cubic, points, means, covariance),
        quartic=_stencil_from_coeffs(offsets, coeff_quartic, points, means, covariance),
    )



def directional_force_stencil(posterior: TotalEnergyPosterior, atoms: Atoms, direction_flat: np.ndarray, step: float) -> StencilEvaluation:
    return evaluate_linear_stencil(
        posterior,
        atoms,
        direction_flat,
        offsets=(+1.0, -1.0),
        coefficients=np.array([1.0, -1.0], dtype=float) / (2.0 * step),
        step=step,
    )



def directional_curvature_stencil(posterior: TotalEnergyPosterior, atoms: Atoms, direction_flat: np.ndarray, step: float) -> StencilEvaluation:
    return evaluate_linear_stencil(
        posterior,
        atoms,
        direction_flat,
        offsets=(+1.0, 0.0, -1.0),
        coefficients=np.array([1.0, -2.0, 1.0], dtype=float) / (step**2),
        step=step,
    )



def directional_cubic_stencil(posterior: TotalEnergyPosterior, atoms: Atoms, direction_flat: np.ndarray, step: float) -> StencilEvaluation:
    return evaluate_linear_stencil(
        posterior,
        atoms,
        direction_flat,
        offsets=(-2.0, -1.0, +1.0, +2.0),
        coefficients=np.array([-1.0, 2.0, -2.0, 1.0], dtype=float) / (2.0 * step**3),
        step=step,
    )



def directional_quartic_stencil(posterior: TotalEnergyPosterior, atoms: Atoms, direction_flat: np.ndarray, step: float) -> StencilEvaluation:
    return evaluate_linear_stencil(
        posterior,
        atoms,
        direction_flat,
        offsets=(-2.0, -1.0, 0.0, +1.0, +2.0),
        coefficients=np.array([1.0, -4.0, 6.0, -4.0, 1.0], dtype=float) / (step**4),
        step=step,
    )
