from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from ichor.core.atoms import Atoms
from ichor.core.models.models import Models


GeometryInput = Union[Atoms, Dict[str, np.ndarray], np.ndarray]
VARIANCE_NEGATIVE_TOLERANCE = 1.0e-10
POSTERIOR_CACHE_MAX_SIZE = 4096



def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("posterior features must be finite")
    if x.ndim == 1:
        return x[np.newaxis, :]
    return x


def _check_finite_array(values, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(label + " must be finite")
    return arr


def _check_variance_array(values, label: str) -> np.ndarray:
    arr = _check_finite_array(values, label)
    if np.any(arr < -VARIANCE_NEGATIVE_TOLERANCE):
        raise ValueError(label + " is materially negative")
    return arr


def _check_kernel_active_dims(model, nfeats: int, label: str) -> None:
    kernel = getattr(model, "kernel", None)
    active_dims = getattr(kernel, "active_dims", None)
    if active_dims is None:
        return
    arr = np.asarray(active_dims)
    if arr.size == 0:
        return
    try:
        raw_float = np.asarray(active_dims, dtype=float).reshape(-1)
        dims = np.asarray(active_dims, dtype=int).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(label + " kernel active_dims must be integer indices") from exc
    if dims.size != arr.reshape(-1).size:
        raise ValueError(label + " kernel active_dims shape is invalid")
    if not np.all(np.isfinite(raw_float)) or not np.allclose(raw_float, dims):
        raise ValueError(label + " kernel active_dims must be integer indices")
    if np.any(dims < 0) or np.any(dims >= int(nfeats)):
        raise ValueError(
            label
            + " kernel active_dims out of range for "
            + str(int(nfeats))
            + " features"
        )



def _estimated_signal_variance(model) -> float:
    y = _check_finite_array(model.y, "model.y").reshape((-1, 1))
    x = _check_finite_array(model.x, "model.x")
    mean = _check_finite_array(model.mean.value(x), "model mean").reshape((-1, 1))
    resid = y - mean
    whitened = np.linalg.solve(model.lower_cholesky, resid)
    tau2 = float((whitened.T @ whitened).reshape(-1)[0] / max(model.ntrain, 1))
    if not np.isfinite(tau2) or tau2 <= 0.0:
        return 1.0
    return tau2



def model_posterior_covariance(model, x1: np.ndarray, x2: np.ndarray, scaled: bool = True) -> np.ndarray:
    x1 = _ensure_2d(x1)
    x2 = _ensure_2d(x2)
    k12 = _check_finite_array(model.kernel.k(x1, x2), "kernel covariance")
    r1 = _check_finite_array(model.r(x1), "train-test covariance")
    r2 = _check_finite_array(model.r(x2), "train-test covariance")
    v1 = np.linalg.solve(model.lower_cholesky, r1)
    v2 = np.linalg.solve(model.lower_cholesky, r2)
    posterior = k12 - v1.T @ v2
    posterior = 0.5 * (posterior + posterior.T) if x1.shape == x2.shape and np.array_equal(x1, x2) else posterior
    if scaled:
        posterior = _estimated_signal_variance(model) * posterior
    return _check_finite_array(posterior, "posterior covariance")


@dataclass
class TotalEnergyPosterior:
    """Single-model posterior over the total IQA energy.

    This wrapper deliberately avoids any changes to ICHOR's existing 'Model' /
    'Models' API. It builds the total-energy posterior directly from the current
    'Models' object by summing the per-atom IQA models under an independence
    approximation across atoms.
    """

    models: Models
    property_name: str = "iqa"
    scaled: bool = True

    def __post_init__(self) -> None:
        property_models = [model for model in self.models if model.type == self.property_name]
        if not property_models:
            raise ValueError(f"No models of property {self.property_name!r} were found.")
        self._property_models: Dict[str, object] = {model.atom: model for model in property_models}
        self._model_identity = (
            id(self.models),
            tuple(sorted((str(model.atom), id(model)) for model in property_models)),
        )
        self._mean_cache: "OrderedDict[Tuple[object, Tuple[float, ...]], float]" = OrderedDict()
        self._cov_cache: "OrderedDict[Tuple[object, Tuple[float, ...], Tuple[float, ...]], float]" = OrderedDict()
        self.diagnostics = {
            "n_mean_scalar_calls": 0,
            "n_means_batched_calls": 0,
            "n_means_scalar_fallbacks": 0,
            "n_covariance_scalar_calls": 0,
            "n_covariance_matrix_batched_calls": 0,
            "n_covariance_matrix_scalar_fallbacks": 0,
        }

    @staticmethod
    def _cache_get(cache: OrderedDict, key):
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    @staticmethod
    def _cache_set(cache: OrderedDict, key, value) -> None:
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > POSTERIOR_CACHE_MAX_SIZE:
            cache.popitem(last=False)

    def _diagnostic_add(self, key: str, amount: int = 1) -> None:
        diagnostics = getattr(self, "diagnostics", None)
        if isinstance(diagnostics, dict):
            diagnostics[key] = int(diagnostics.get(key, 0)) + int(amount)

    def _features(self, x: GeometryInput) -> Dict[str, np.ndarray]:
        features = self.models.get_features_dict(x)
        missing = set(self._property_models) - set(features)
        if missing:
            raise KeyError(f"Missing features for atoms {sorted(missing)}")
        for atom, model in self._property_models.items():
            arr = _check_finite_array(features[atom], f"features for atom {atom}")
            arr2d = _ensure_2d(arr)
            nfeats = int(getattr(model, "nfeats", arr2d.shape[1]))
            if arr2d.shape[1] != nfeats:
                raise ValueError(
                    f"Feature dimension mismatch for atom {atom}: "
                    f"{arr2d.shape[1]} != {nfeats}"
                )
            _check_kernel_active_dims(model, nfeats, f"atom {atom}")
        return features

    @staticmethod
    def _geometry_key(x: GeometryInput) -> Tuple[float, ...]:
        if isinstance(x, dict):
            flat_parts: List[np.ndarray] = []
            for atom in sorted(x):
                flat_parts.append(np.asarray(x[atom], dtype=float).reshape(-1))
            arr = np.concatenate(flat_parts) if flat_parts else np.zeros(0, dtype=float)
        elif isinstance(x, Atoms):
            arr = np.asarray(x.coordinates, dtype=float).reshape(-1)
        else:
            arr = np.asarray(x, dtype=float).reshape(-1)
        return tuple(np.round(arr, 12))

    def mean(self, x: GeometryInput) -> float:
        self._diagnostic_add("n_mean_scalar_calls")
        key = (self._model_identity, self._geometry_key(x))
        cached = self._cache_get(self._mean_cache, key)
        if cached is not None:
            return cached
        features = self._features(x)
        total = 0.0
        for atom, model in self._property_models.items():
            total += float(np.asarray(model.predict(features[atom]), dtype=float).reshape(-1)[0])
        self._cache_set(self._mean_cache, key, total)
        return total

    @staticmethod
    def _atom_type_from_input(x: GeometryInput, atom: str) -> str:
        if isinstance(x, Atoms):
            try:
                return str(x[atom].type)
            except Exception:
                pass
        stripped = "".join(ch for ch in str(atom) if not ch.isdigit())
        return stripped or str(atom)

    def covariance(self, x1: GeometryInput, x2: GeometryInput) -> float:
        self._diagnostic_add("n_covariance_scalar_calls")
        key1 = self._geometry_key(x1)
        key2 = self._geometry_key(x2)
        cache_key = (
            (self._model_identity, key1, key2)
            if key1 <= key2
            else (self._model_identity, key2, key1)
        )
        cached = self._cache_get(self._cov_cache, cache_key)
        if cached is not None:
            return cached
        features1 = self._features(x1)
        features2 = self._features(x2)
        total = 0.0
        for atom, model in self._property_models.items():
            total += float(model_posterior_covariance(model, features1[atom], features2[atom], scaled=self.scaled)[0, 0])
        self._cache_set(self._cov_cache, cache_key, total)
        return total

    def variance(self, x: GeometryInput) -> float:
        return float(_check_variance_array([self.covariance(x, x)], "posterior variance")[0])

    def variance_components(self, x: GeometryInput) -> Tuple[float, Dict[str, float]]:
        """Return total predictive variance plus the per-atom contributions.

        This is the same calculation as ``variance(x)`` but exposes the
        atom-level diagonal terms so daemon-side error calibration can map
        per-atom realised IQA errors back to the uncertainty signal that
        selected the point. No extra model evaluations are performed by callers
        that only need the total.
        """
        features = self._features(x)
        per_atom: Dict[str, float] = {}
        total = 0.0
        for atom, model in self._property_models.items():
            value = float(
                model_posterior_covariance(
                    model,
                    features[atom],
                    features[atom],
                    scaled=self.scaled,
                )[0, 0]
            )
            per_atom[str(atom)] = value
            total += value
        return (
            float(_check_variance_array([total], "posterior variance")[0]),
            {
                atom: float(_check_variance_array([value], "atom posterior variance")[0])
                for atom, value in per_atom.items()
            },
        )

    @staticmethod
    def _stack_feature_rows(
        features: Sequence[Mapping[str, np.ndarray]],
        atom: str,
    ) -> np.ndarray:
        return np.asarray(
            [
                np.asarray(features[i][atom], dtype=float).reshape(-1)
                for i in range(len(features))
            ],
            dtype=float,
        )

    def atom_diagnostics(self, x: GeometryInput) -> Dict[str, Dict[str, float]]:
        """Per-atom prediction and uncertainty diagnostics for one geometry."""
        features = self._features(x)
        out: Dict[str, Dict[str, float]] = {}
        for atom, model in self._property_models.items():
            prediction = float(
                np.asarray(model.predict(features[atom]), dtype=float).reshape(-1)[0]
            )
            variance = float(
                model_posterior_covariance(
                    model,
                    features[atom],
                    features[atom],
                    scaled=self.scaled,
                )[0, 0]
            )
            _check_finite_array([prediction], "atom prediction")
            _check_variance_array([variance], "atom posterior variance")
            out[str(atom)] = {
                "predicted_iqa_ha": prediction,
                "raw_variance": variance,
                "atom_type": self._atom_type_from_input(x, str(atom)),
            }
        return out

    def variances(
        self,
        points: Sequence[GeometryInput],
        *,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """Predictive variance for many geometries at once -- the diagonal of
        the posterior covariance, summed over atoms under the same independence
        approximation variance() uses. One matmul per atom across the whole
        batch instead of a python call per frame, so a seed scan over a big
        pool is much cheaper. Returns one variance per input point, in order.

        Numerically identical to calling variance() on each point (it is the
        same per-atom posterior, just evaluated batched).
        """
        n = len(points)
        if n == 0:
            return np.zeros(0, dtype=float)
        if chunk_size is not None and int(chunk_size) > 0 and n > int(chunk_size):
            chunks = [
                self.variances(points[i:i + int(chunk_size)], chunk_size=None)
                for i in range(0, n, int(chunk_size))
            ]
            return _check_variance_array(
                np.concatenate(chunks) if chunks else np.zeros(0, dtype=float),
                "posterior variances",
            )
        feats = [self._features(p) for p in points]
        total = np.zeros(n, dtype=float)
        for atom, model in self._property_models.items():
            X = np.array(
                [np.asarray(feats[i][atom], dtype=float).reshape(-1) for i in range(n)]
            )
            # diagonal of model_posterior_covariance(model, X, X) without
            # forming the full n x n block: k_ii - sum_k v[k,i]^2.
            if hasattr(model.kernel, "k_diag"):
                k_diag = _check_finite_array(model.kernel.k_diag(X), "kernel diagonal").reshape(-1)
                if k_diag.shape != (n,):
                    raise ValueError(
                        f"kernel diagonal shape for atom {atom} must be {(n,)}, got {k_diag.shape}"
                    )
            else:
                k_diag = np.diag(
                    _check_finite_array(model.kernel.k(X, X), "kernel covariance")
                )
            r = _check_finite_array(model.r(X), "train-test covariance")
            v = np.linalg.solve(model.lower_cholesky, r)
            diag = k_diag - np.sum(v * v, axis=0)
            if self.scaled:
                diag = _estimated_signal_variance(model) * diag
            total = total + diag
        return _check_variance_array(total, "posterior variances")

    def _covariance_matrix_scalar(self, points: Sequence[GeometryInput]) -> np.ndarray:
        n = len(points)
        cov = np.zeros((n, n), dtype=float)
        for i in range(n):
            cov[i, i] = self.covariance(points[i], points[i])
            for j in range(i + 1, n):
                value = self.covariance(points[i], points[j])
                cov[i, j] = value
                cov[j, i] = value
        return _check_finite_array(cov, "posterior covariance matrix")

    def _covariance_matrix_batched(self, points: Sequence[GeometryInput]) -> np.ndarray:
        self._diagnostic_add("n_covariance_matrix_batched_calls")
        n = len(points)
        feats = [self._features(p) for p in points]
        return self._covariance_matrix_batched_from_features(feats, n=n)

    def _covariance_matrix_batched_from_features(
        self,
        feats: Sequence[Mapping[str, np.ndarray]],
        *,
        n: Optional[int] = None,
    ) -> np.ndarray:
        n = len(feats) if n is None else int(n)
        total_cov = np.zeros((n, n), dtype=float)
        for atom, model in self._property_models.items():
            X = self._stack_feature_rows(feats, atom)
            kxx = _check_finite_array(model.kernel.k(X, X), "kernel covariance")
            if kxx.shape != (n, n):
                raise ValueError(
                    f"kernel covariance shape for atom {atom} must be {(n, n)}, got {kxx.shape}"
                )
            r = _check_finite_array(model.r(X), "train-test covariance")
            expected_r_shape = (int(model.ntrain), n)
            if r.shape != expected_r_shape:
                raise ValueError(
                    f"train-test covariance shape for atom {atom} must be "
                    f"{expected_r_shape}, got {r.shape}"
                )
            v = np.linalg.solve(model.lower_cholesky, r)
            if v.shape != expected_r_shape:
                raise ValueError(
                    f"posterior solve shape for atom {atom} must be "
                    f"{expected_r_shape}, got {v.shape}"
                )
            atom_cov = kxx - v.T @ v
            if atom_cov.shape != (n, n):
                raise ValueError(
                    f"posterior covariance shape for atom {atom} must be {(n, n)}, got {atom_cov.shape}"
                )
            if self.scaled:
                atom_cov = _estimated_signal_variance(model) * atom_cov
            total_cov += atom_cov
        total_cov = 0.5 * (total_cov + total_cov.T)
        return _check_finite_array(total_cov, "posterior covariance matrix")

    def covariance_matrix(
        self,
        points: Sequence[GeometryInput],
        *,
        chunk_size: Optional[int] = None,
        prefer_batched: bool = True,
    ) -> np.ndarray:
        n = len(points)
        if n == 0:
            return np.zeros((0, 0), dtype=float)
        if not prefer_batched:
            return self._covariance_matrix_scalar(points)
        if chunk_size is not None and int(chunk_size) > 0 and n > int(chunk_size):
            self._diagnostic_add("n_covariance_matrix_scalar_fallbacks")
            return self._covariance_matrix_scalar(points)
        try:
            return self._covariance_matrix_batched(points)
        except Exception:
            self._diagnostic_add("n_covariance_matrix_scalar_fallbacks")
            return self._covariance_matrix_scalar(points)

    def cross_covariances(
        self,
        left: Sequence[GeometryInput],
        right: Sequence[GeometryInput],
        *,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """Posterior covariance between two geometry lists.

        This is intentionally rectangular. Seed selection uses it for
        candidate-vs-selected D-optimal gains without materialising an N x N
        covariance matrix for the whole trajectory pool.
        """
        n_left = len(left)
        n_right = len(right)
        if n_left == 0 or n_right == 0:
            return np.zeros((n_left, n_right), dtype=float)
        if chunk_size is not None and int(chunk_size) > 0 and n_left > int(chunk_size):
            chunks = [
                self.cross_covariances(
                    left[i:i + int(chunk_size)],
                    right,
                    chunk_size=None,
                )
                for i in range(0, n_left, int(chunk_size))
            ]
            return _check_finite_array(
                np.vstack(chunks) if chunks else np.zeros((0, n_right), dtype=float),
                "posterior cross-covariances",
            )
        cov = np.zeros((n_left, n_right), dtype=float)
        for i, x_left in enumerate(left):
            for j, x_right in enumerate(right):
                cov[i, j] = self.covariance(x_left, x_right)
        return _check_finite_array(cov, "posterior cross-covariances")

    def _means_scalar(self, points: Sequence[GeometryInput]) -> np.ndarray:
        return _check_finite_array(
            np.array([self.mean(point) for point in points], dtype=float),
            "posterior means",
        )

    def _means_batched(self, points: Sequence[GeometryInput]) -> np.ndarray:
        self._diagnostic_add("n_means_batched_calls")
        n = len(points)
        feats = [self._features(p) for p in points]
        return self._means_batched_from_features(feats, n=n)

    def _means_batched_from_features(
        self,
        feats: Sequence[Mapping[str, np.ndarray]],
        *,
        n: Optional[int] = None,
    ) -> np.ndarray:
        n = len(feats) if n is None else int(n)
        total = np.zeros(n, dtype=float)
        for atom, model in self._property_models.items():
            X = self._stack_feature_rows(feats, atom)
            pred = _check_finite_array(model.predict(X), "posterior means").reshape(-1)
            if pred.shape != (n,):
                raise ValueError(
                    f"model.predict(X) for atom {atom} must return {(n,)}, got {pred.shape}"
                )
            total += pred
        return _check_finite_array(total, "posterior means")

    def means(
        self,
        points: Sequence[GeometryInput],
        *,
        chunk_size: Optional[int] = None,
        prefer_batched: bool = True,
    ) -> np.ndarray:
        n = len(points)
        if n == 0:
            return np.zeros(0, dtype=float)
        if chunk_size is not None and int(chunk_size) > 0 and n > int(chunk_size):
            chunks = [
                self.means(
                    points[i:i + int(chunk_size)],
                    chunk_size=None,
                    prefer_batched=prefer_batched,
                )
                for i in range(0, n, int(chunk_size))
            ]
            return _check_finite_array(
                np.concatenate(chunks) if chunks else np.zeros(0, dtype=float),
                "posterior means",
            )
        if not prefer_batched:
            return self._means_scalar(points)
        try:
            return self._means_batched(points)
        except Exception:
            self._diagnostic_add("n_means_scalar_fallbacks")
            return self._means_scalar(points)

    def means_and_covariance_matrix(
        self,
        points: Sequence[GeometryInput],
        *,
        chunk_size: Optional[int] = None,
        prefer_batched: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return posterior means and covariance using one feature extraction pass.

        Fused directional stencils need both moments for the same small set of
        geometries. This helper preserves the existing public means() and
        covariance_matrix() methods while avoiding duplicate ALF feature work.
        """
        n = len(points)
        if n == 0:
            return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)
        if chunk_size is not None and int(chunk_size) > 0 and n > int(chunk_size):
            means_chunks = []
            cov = self.covariance_matrix(points, chunk_size=chunk_size, prefer_batched=prefer_batched)
            for i in range(0, n, int(chunk_size)):
                means_chunk, _cov_chunk = self.means_and_covariance_matrix(
                    points[i:i + int(chunk_size)],
                    chunk_size=None,
                    prefer_batched=prefer_batched,
                )
                means_chunks.append(means_chunk)
            means = np.concatenate(means_chunks) if means_chunks else np.zeros(0, dtype=float)
            return _check_finite_array(means, "posterior means"), cov
        if not prefer_batched:
            return self._means_scalar(points), self._covariance_matrix_scalar(points)
        try:
            self._diagnostic_add("n_means_batched_calls")
            self._diagnostic_add("n_covariance_matrix_batched_calls")
            feats = [self._features(p) for p in points]
            means = self._means_batched_from_features(feats, n=n)
            cov = self._covariance_matrix_batched_from_features(feats, n=n)
            return means, cov
        except Exception:
            self._diagnostic_add("n_means_scalar_fallbacks")
            self._diagnostic_add("n_covariance_matrix_scalar_fallbacks")
            return self._means_scalar(points), self._covariance_matrix_scalar(points)
