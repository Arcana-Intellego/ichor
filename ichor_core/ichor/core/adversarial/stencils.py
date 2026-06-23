from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from ichor.core.atoms import Atoms

from .geometry import copy_atoms_with_flat_displacement
from .posterior import TotalEnergyPosterior


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
    mean = float(coeffs_arr @ means_arr)
    variance = float(coeffs_arr @ covariance_arr @ coeffs_arr)
    return StencilEvaluation(
        offsets=offsets_arr,
        coefficients=coeffs_arr,
        points=list(points),
        mean=mean,
        variance=max(variance, 0.0),
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
    points = [displaced_geometry(atoms, direction_flat, float(offset) * float(step)) for offset in offsets_arr]
    means = posterior.means(points)
    covariance = posterior.covariance_matrix(points)
    return _stencil_from_coeffs(offsets_arr, coeffs_arr, points, means, covariance)



def directional_all_stencils(
    posterior: TotalEnergyPosterior,
    atoms: Atoms,
    direction_flat: np.ndarray,
    step: float,
) -> DirectionalStencilBundle:
    step_f = float(step)
    offsets = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    points = [
        displaced_geometry(atoms, direction_flat, float(offset) * step_f)
        for offset in offsets
    ]
    means = np.asarray(posterior.means(points), dtype=float)
    covariance = np.asarray(posterior.covariance_matrix(points), dtype=float)

    coeff_force = np.asarray([0.0, -1.0, 0.0, 1.0, 0.0], dtype=float) / (2.0 * step_f)
    coeff_curvature = np.asarray([0.0, 1.0, -2.0, 1.0, 0.0], dtype=float) / (step_f ** 2)
    coeff_cubic = np.asarray([-1.0, 2.0, 0.0, -2.0, 1.0], dtype=float) / (2.0 * step_f ** 3)
    coeff_quartic = np.asarray([1.0, -4.0, 6.0, -4.0, 1.0], dtype=float) / (step_f ** 4)

    return DirectionalStencilBundle(
        offsets=offsets,
        points=points,
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
