from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from ichor.core.atoms import Atoms
from ichor.core.models.models import Models


GeometryInput = Union[Atoms, Dict[str, np.ndarray], np.ndarray]
VARIANCE_NEGATIVE_TOLERANCE = 1.0e-10



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
        self._mean_cache: Dict[Tuple[float, ...], float] = {}
        self._cov_cache: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], float] = {}

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
        key = self._geometry_key(x)
        if key in self._mean_cache:
            return self._mean_cache[key]
        features = self._features(x)
        total = 0.0
        for atom, model in self._property_models.items():
            total += float(np.asarray(model.predict(features[atom]), dtype=float).reshape(-1)[0])
        self._mean_cache[key] = total
        return total

    def covariance(self, x1: GeometryInput, x2: GeometryInput) -> float:
        key1 = self._geometry_key(x1)
        key2 = self._geometry_key(x2)
        cache_key = (key1, key2) if key1 <= key2 else (key2, key1)
        if cache_key in self._cov_cache:
            return self._cov_cache[cache_key]
        features1 = self._features(x1)
        features2 = self._features(x2)
        total = 0.0
        for atom, model in self._property_models.items():
            total += float(model_posterior_covariance(model, features1[atom], features2[atom], scaled=self.scaled)[0, 0])
        self._cov_cache[cache_key] = total
        return total

    def variance(self, x: GeometryInput) -> float:
        return float(_check_variance_array([self.covariance(x, x)], "posterior variance")[0])

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

    def covariance_matrix(self, points: Sequence[GeometryInput]) -> np.ndarray:
        n = len(points)
        cov = np.zeros((n, n), dtype=float)
        for i in range(n):
            cov[i, i] = self.covariance(points[i], points[i])
            for j in range(i + 1, n):
                value = self.covariance(points[i], points[j])
                cov[i, j] = value
                cov[j, i] = value
        return cov

    def means(self, points: Sequence[GeometryInput]) -> np.ndarray:
        return np.array([self.mean(point) for point in points], dtype=float)
