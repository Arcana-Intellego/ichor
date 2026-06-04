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



def displaced_geometry(atoms: Atoms, direction_flat: np.ndarray, displacement: float) -> Atoms:
    return copy_atoms_with_flat_displacement(atoms, displacement * np.asarray(direction_flat, dtype=float))



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
    mean = float(coeffs_arr @ means)
    variance = float(coeffs_arr @ covariance @ coeffs_arr)
    return StencilEvaluation(
        offsets=offsets_arr,
        coefficients=coeffs_arr,
        points=points,
        mean=mean,
        variance=max(variance, 0.0),
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
