"""Finite-difference stencil coefficient checks."""
import numpy as np

from ichor.core.adversarial.stencils import directional_cubic_stencil
from ichor.core.atoms import Atom, Atoms


class _CubicPosterior:
    def means(self, points):
        return np.array([float(point.coordinates.reshape(-1)[0]) ** 3 for point in points])

    def covariance_matrix(self, points):
        return np.zeros((len(points), len(points)), dtype=float)


def test_directional_cubic_stencil_has_positive_third_derivative_sign():
    atoms = Atoms([Atom("H", 0.0, 0.0, 0.0)])
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    result = directional_cubic_stencil(
        _CubicPosterior(),
        atoms,
        direction,
        step=0.2,
    )

    assert abs(result.mean - 6.0) < 1.0e-10
