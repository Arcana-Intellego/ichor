"""Batched TotalEnergyPosterior.variances() must equal per-frame variance().

A self-contained fake model exposes exactly the attribute surface the posterior
touches (kernel.k, r, lower_cholesky, predict, y, mean, x, ntrain, type, atom),
with a genuine PSD kernel so the linear algebra is real. We then check that the
vectorised diagonal matches the scalar path frame by frame, scaled and unscaled.
"""
import numpy as np
import pytest

from ichor.core.adversarial.posterior import TotalEnergyPosterior
from ichor.core.adversarial.stencils import directional_all_stencils
from ichor.core.atoms import Atom, Atoms


class _Kernel:
    def __init__(self, train_x, length=1.3):
        self._tx = np.asarray(train_x, dtype=float)
        self._l = float(length)

    def _rbf(self, A, B):
        A = np.atleast_2d(np.asarray(A, dtype=float))
        B = np.atleast_2d(np.asarray(B, dtype=float))
        d2 = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2.0 * A @ B.T
        return np.exp(-0.5 * d2 / self._l**2)

    def k(self, x1, x2):
        return self._rbf(x1, x2)


class _DiagOnlyKernel(_Kernel):
    def __init__(self, train_x, length=1.3):
        super().__init__(train_x, length=length)
        self.full_test_kernel_calls = 0
        self.k_diag_calls = 0

    def k_diag(self, x):
        self.k_diag_calls += 1
        return np.ones(np.atleast_2d(np.asarray(x, dtype=float)).shape[0])

    def k(self, x1, x2):
        if x1 is x2:
            self.full_test_kernel_calls += 1
            raise AssertionError("full candidate kernel block should not be built")
        return super().k(x1, x2)


class _Mean:
    def __init__(self):
        self.calls = 0

    def value(self, x):
        self.calls += 1
        return np.zeros((np.atleast_2d(np.asarray(x, dtype=float)).shape[0], 1))


class _Model:
    """Minimal stand-in for ichor Model, enough for TotalEnergyPosterior."""
    def __init__(self, atom, train_x, train_y, jitter=1e-6):
        self.atom = atom
        self.type = "iqa"
        self.x = np.asarray(train_x, dtype=float)
        self.y = np.asarray(train_y, dtype=float).reshape(-1, 1)
        self.ntrain = self.x.shape[0]
        self.kernel = _Kernel(self.x)
        self.mean = _Mean()
        K = self.kernel.k(self.x, self.x) + jitter * np.eye(self.ntrain)
        self.lower_cholesky = np.linalg.cholesky(K)

    def r(self, x):
        # cross-covariance train x query, shape (ntrain, nquery)
        return self.kernel.k(self.x, x)

    def predict(self, x):
        r = self.r(x)
        alpha = np.linalg.solve(self.lower_cholesky.T, np.linalg.solve(self.lower_cholesky, self.y))
        return (r.T @ alpha).reshape(-1)


class _DiagOnlyModel(_Model):
    def __init__(self, atom, train_x, train_y, jitter=1e-6):
        super().__init__(atom, train_x, train_y, jitter=jitter)
        self.kernel = _DiagOnlyKernel(self.x)
        K = self.kernel.k(self.x, self.x.copy()) + jitter * np.eye(self.ntrain)
        self.lower_cholesky = np.linalg.cholesky(K)


class _ScalarPredictOnlyModel(_Model):
    def predict(self, x):
        if np.atleast_2d(np.asarray(x, dtype=float)).shape[0] > 1:
            raise TypeError("batched predict unsupported")
        return super().predict(x)


class _ScalarQueryOnlyModel(_Model):
    def r(self, x):
        if np.atleast_2d(np.asarray(x, dtype=float)).shape[0] > 1:
            raise TypeError("batched query unsupported")
        return super().r(x)


class _Models:
    def __init__(self, models, feat_dim):
        self._models = models
        self._feat_dim = feat_dim

    def __iter__(self):
        return iter(self._models)

    def get_features_dict(self, x):
        # map a flat geometry vector to a per-atom feature row deterministically.
        arr = np.asarray(x.coordinates, dtype=float).reshape(-1)
        out = {}
        for m in self._models:
            # take feat_dim numbers, offset by a per-atom hash so atoms differ
            off = (abs(hash(m.atom)) % 7)
            v = np.array([arr[(off + k) % arr.size] for k in range(self._feat_dim)])
            out[m.atom] = v
        return out


def _make_posterior(scaled):
    rng = np.random.default_rng(0)
    feat_dim = 3
    train_x = rng.normal(size=(10, feat_dim))
    models = [
        _Model("O1", train_x, rng.normal(size=10)),
        _Model("H2", train_x + 0.5, rng.normal(size=10)),
        _Model("H3", train_x - 0.5, rng.normal(size=10)),
    ]
    post = TotalEnergyPosterior(_Models(models, feat_dim))
    post.scaled = scaled
    return post, rng


def _make_diag_only_posterior():
    rng = np.random.default_rng(2)
    feat_dim = 3
    train_x = rng.normal(size=(10, feat_dim))
    models = [
        _DiagOnlyModel("O1", train_x, rng.normal(size=10)),
        _DiagOnlyModel("H2", train_x + 0.5, rng.normal(size=10)),
    ]
    return TotalEnergyPosterior(_Models(models, feat_dim)), rng, models


def _make_scalar_predict_only_posterior():
    rng = np.random.default_rng(3)
    feat_dim = 3
    train_x = rng.normal(size=(10, feat_dim))
    models = [
        _ScalarPredictOnlyModel("O1", train_x, rng.normal(size=10)),
        _ScalarPredictOnlyModel("H2", train_x + 0.5, rng.normal(size=10)),
    ]
    return TotalEnergyPosterior(_Models(models, feat_dim)), rng


def _make_scalar_query_only_posterior():
    rng = np.random.default_rng(4)
    feat_dim = 3
    train_x = rng.normal(size=(10, feat_dim))
    models = [
        _ScalarQueryOnlyModel("O1", train_x, rng.normal(size=10)),
        _ScalarQueryOnlyModel("H2", train_x + 0.5, rng.normal(size=10)),
    ]
    return TotalEnergyPosterior(_Models(models, feat_dim)), rng


def _atoms(coords):
    types = ["O", "H", "H", "H"]
    cs = np.asarray(coords, dtype=float)
    return Atoms([Atom(types[i], *cs[i]) for i in range(cs.shape[0])])


def _points(rng, n):
    return [_atoms(rng.normal(size=(4, 3))) for _ in range(n)]


def test_variances_match_per_frame_unscaled():
    post, rng = _make_posterior(scaled=False)
    pts = _points(rng, 6)
    batched = np.asarray(post.variances(pts), dtype=float)
    per_frame = np.array([post.variance(p) for p in pts], dtype=float)
    np.testing.assert_allclose(batched, per_frame, atol=1e-10, rtol=1e-9)


def test_variances_match_per_frame_scaled():
    post, rng = _make_posterior(scaled=True)
    pts = _points(rng, 6)
    batched = np.asarray(post.variances(pts), dtype=float)
    per_frame = np.array([post.variance(p) for p in pts], dtype=float)
    np.testing.assert_allclose(batched, per_frame, atol=1e-10, rtol=1e-9)


def test_variances_empty():
    post, _ = _make_posterior(scaled=True)
    assert list(post.variances([])) == []


def test_variances_chunked_match_unchunked():
    post, rng = _make_posterior(scaled=True)
    pts = _points(rng, 9)
    chunked = np.asarray(post.variances(pts, chunk_size=2), dtype=float)
    unchunked = np.asarray(post.variances(pts), dtype=float)
    np.testing.assert_allclose(chunked, unchunked, atol=1e-10, rtol=1e-9)


def test_variances_use_kernel_diagonal_without_full_candidate_block():
    post, rng, models = _make_diag_only_posterior()
    pts = _points(rng, 5)
    values = post.variances(pts)
    assert values.shape == (5,)
    assert np.all(np.isfinite(values))
    assert all(model.kernel.k_diag_calls >= 1 for model in models)
    assert all(model.kernel.full_test_kernel_calls == 0 for model in models)


def test_scaled_signal_variance_is_cached_per_model_fit():
    post, rng = _make_posterior(scaled=True)
    pts = _points(rng, 4)

    post.variances(pts)
    first_calls = [model.mean.calls for model in post._property_models.values()]
    post.variances(pts)
    second_calls = [model.mean.calls for model in post._property_models.values()]

    assert first_calls == [1, 1, 1]
    assert second_calls == first_calls
    assert all(
        hasattr(model, "_ichor_al_signal_variance_cache")
        for model in post._property_models.values()
    )


def test_variances_reject_kernel_active_dims_out_of_range():
    post, rng, models = _make_diag_only_posterior()
    models[0].kernel.active_dims = [0, 3]

    with pytest.raises(ValueError, match="active_dims out of range"):
        post.variances(_points(rng, 2))


def test_variances_reject_non_integer_kernel_active_dims():
    post, rng, models = _make_diag_only_posterior()
    models[0].kernel.active_dims = [0, 1.2]

    with pytest.raises(ValueError, match="active_dims must be integer"):
        post.variances(_points(rng, 2))


def test_variances_reject_wrong_kernel_diagonal_shape():
    post, rng, models = _make_diag_only_posterior()

    def _bad_k_diag(x):
        n = np.atleast_2d(np.asarray(x, dtype=float)).shape[0]
        return np.ones((n, 2), dtype=float)

    models[0].kernel.k_diag = _bad_k_diag

    with pytest.raises(ValueError, match="kernel diagonal shape"):
        post.variances(_points(rng, 3))


def test_batched_means_match_per_frame_scalar():
    post, rng = _make_posterior(scaled=True)
    pts = _points(rng, 6)
    batched = np.asarray(post.means(pts), dtype=float)
    per_frame = np.array([post.mean(p) for p in pts], dtype=float)
    np.testing.assert_allclose(batched, per_frame, atol=1e-10, rtol=1e-9)


def test_batched_means_chunked_match_unchunked():
    post, rng = _make_posterior(scaled=True)
    pts = _points(rng, 9)
    chunked = np.asarray(post.means(pts, chunk_size=2), dtype=float)
    unchunked = np.asarray(post.means(pts), dtype=float)
    np.testing.assert_allclose(chunked, unchunked, atol=1e-10, rtol=1e-9)


def test_batched_means_empty():
    post, _ = _make_posterior(scaled=True)
    assert list(post.means([])) == []


def test_batched_means_fallback_for_scalar_predict_model():
    post, rng = _make_scalar_predict_only_posterior()
    pts = _points(rng, 5)
    values = post.means(pts)
    scalar = post.means(pts, prefer_batched=False)
    np.testing.assert_allclose(values, scalar, atol=1e-10, rtol=1e-9)
    assert post.diagnostics["n_means_scalar_fallbacks"] >= 1


def _assert_covariance_matrix_matches_scalar(scaled, n):
    post, rng = _make_posterior(scaled=scaled)
    pts = _points(rng, n)
    batched = np.asarray(post.covariance_matrix(pts), dtype=float)
    scalar = np.asarray(post.covariance_matrix(pts, prefer_batched=False), dtype=float)
    np.testing.assert_allclose(batched, scalar, atol=1e-10, rtol=1e-9)


def test_batched_covariance_matrix_matches_scalar_n1():
    _assert_covariance_matrix_matches_scalar(scaled=False, n=1)


def test_batched_covariance_matrix_matches_scalar_n2():
    _assert_covariance_matrix_matches_scalar(scaled=False, n=2)


def test_batched_covariance_matrix_matches_scalar_n5():
    _assert_covariance_matrix_matches_scalar(scaled=False, n=5)


def test_batched_covariance_matrix_matches_scalar_scaled():
    _assert_covariance_matrix_matches_scalar(scaled=True, n=5)


def test_batched_covariance_matrix_empty():
    post, _ = _make_posterior(scaled=True)
    cov = post.covariance_matrix([])
    assert cov.shape == (0, 0)


def test_batched_covariance_matrix_fallback_for_scalar_query_model():
    post, rng = _make_scalar_query_only_posterior()
    pts = _points(rng, 5)
    values = post.covariance_matrix(pts)
    scalar = post.covariance_matrix(pts, prefer_batched=False)
    np.testing.assert_allclose(values, scalar, atol=1e-10, rtol=1e-9)
    assert post.diagnostics["n_covariance_matrix_scalar_fallbacks"] >= 1


def test_fused_stencils_use_batched_posterior_blocks():
    post, rng = _make_posterior(scaled=True)
    atoms = _points(rng, 1)[0]
    direction = np.zeros(len(atoms) * 3, dtype=float)
    direction[0] = 1.0

    bundle = directional_all_stencils(post, atoms, direction, step=0.17)

    assert bundle.means.shape == (5,)
    assert bundle.covariance.shape == (5, 5)
    assert post.diagnostics["n_means_batched_calls"] >= 1
    assert post.diagnostics["n_covariance_matrix_batched_calls"] >= 1
    assert post.diagnostics["n_means_scalar_fallbacks"] == 0
    assert post.diagnostics["n_covariance_matrix_scalar_fallbacks"] == 0
