import numpy as np

from ichor.core.adversarial.stencils import (
    directional_all_stencils,
    directional_cubic_stencil,
    directional_curvature_stencil,
    directional_force_stencil,
    directional_quartic_stencil,
)
from ichor.core.atoms import Atom, Atoms


class _PolynomialPosterior:
    def __init__(self):
        self.n_means_calls = 0
        self.n_covariance_calls = 0
        self.n_moments_calls = 0

    def means(self, points):
        self.n_means_calls += 1
        values = []
        for point in points:
            x = float(point.coordinates.reshape(-1)[0])
            values.append(0.3 + 1.2 * x - 0.7 * x**2 + 0.4 * x**3 + 0.05 * x**4)
        return np.asarray(values, dtype=float)

    def covariance_matrix(self, points):
        self.n_covariance_calls += 1
        xs = np.asarray(
            [float(point.coordinates.reshape(-1)[0]) for point in points],
            dtype=float,
        )
        delta = xs[:, None] - xs[None, :]
        cov = np.exp(-(delta**2) / 0.7)
        cov += np.eye(len(points), dtype=float) * 0.05
        return cov

    def means_and_covariance_matrix(self, points):
        self.n_moments_calls += 1
        return self.means(points), self.covariance_matrix(points)


def _atoms():
    return Atoms([Atom("H", 0.4, 0.0, 0.0)])


def _direction():
    return np.asarray([1.0, 0.0, 0.0], dtype=float)


def _assert_same_evaluation(actual, expected):
    np.testing.assert_allclose(actual.mean, expected.mean, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(
        actual.variance,
        expected.variance,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(actual.std, expected.std, atol=1.0e-12, rtol=1.0e-12)


def test_directional_all_stencils_matches_legacy_force():
    posterior = _PolynomialPosterior()
    bundle = directional_all_stencils(posterior, _atoms(), _direction(), step=0.17)
    legacy = directional_force_stencil(_PolynomialPosterior(), _atoms(), _direction(), step=0.17)

    _assert_same_evaluation(bundle.force, legacy)


def test_directional_all_stencils_matches_legacy_curvature():
    posterior = _PolynomialPosterior()
    bundle = directional_all_stencils(posterior, _atoms(), _direction(), step=0.17)
    legacy = directional_curvature_stencil(_PolynomialPosterior(), _atoms(), _direction(), step=0.17)

    _assert_same_evaluation(bundle.curvature, legacy)


def test_directional_all_stencils_matches_legacy_cubic():
    posterior = _PolynomialPosterior()
    bundle = directional_all_stencils(posterior, _atoms(), _direction(), step=0.17)
    legacy = directional_cubic_stencil(_PolynomialPosterior(), _atoms(), _direction(), step=0.17)

    _assert_same_evaluation(bundle.cubic, legacy)


def test_directional_all_stencils_matches_legacy_quartic():
    posterior = _PolynomialPosterior()
    bundle = directional_all_stencils(posterior, _atoms(), _direction(), step=0.17)
    legacy = directional_quartic_stencil(_PolynomialPosterior(), _atoms(), _direction(), step=0.17)

    _assert_same_evaluation(bundle.quartic, legacy)


def test_directional_all_stencils_reuses_one_posterior_block():
    posterior = _PolynomialPosterior()

    bundle = directional_all_stencils(posterior, _atoms(), _direction(), step=0.17)

    assert posterior.n_moments_calls == 1
    assert posterior.n_means_calls == 1
    assert posterior.n_covariance_calls == 1
    assert bundle.means.shape == (5,)
    assert bundle.covariance.shape == (5, 5)
