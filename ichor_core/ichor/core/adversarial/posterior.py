from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular
from ichor.core.atoms import Atoms
from ichor.core.models.models import Models


GeometryInput = Union[Atoms, Dict[str, np.ndarray], np.ndarray]
VARIANCE_NEGATIVE_TOLERANCE = 1.0e-10
POSTERIOR_CACHE_MAX_SIZE = 4096
_VARIANCE_CLIPPED_COUNT = 0


def _solve_prepared_lower(factor: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """Solve a validated lower-triangular system for prepared batch work."""
    lower = _check_finite_array(factor, "prepared model Cholesky factor")
    rhs = _check_finite_array(right_hand_side, "prepared train-test covariance")
    if lower.ndim != 2 or lower.shape[0] != lower.shape[1]:
        raise ValueError("prepared model Cholesky factor must be square")
    if rhs.ndim != 2 or rhs.shape[0] != lower.shape[0]:
        raise ValueError("prepared train-test covariance shape mismatch")
    upper_scale = max(1.0, float(np.max(np.abs(lower))))
    upper_tolerance = (
        np.finfo(float).eps * max(1, int(lower.shape[0])) * upper_scale * 16.0
    )
    if np.any(np.abs(np.triu(lower, k=1)) > upper_tolerance):
        raise ValueError("prepared model Cholesky factor must be lower triangular")
    diagonal = np.diag(lower)
    if np.any(diagonal <= 0.0):
        raise ValueError("prepared model Cholesky diagonal must be positive")
    solved = solve_triangular(
        lower,
        rhs,
        lower=True,
        check_finite=False,
        overwrite_b=False,
    )
    return _check_finite_array(solved, "prepared posterior projection")



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
    global _VARIANCE_CLIPPED_COUNT
    arr = _check_finite_array(values, label)
    if np.any(arr < -VARIANCE_NEGATIVE_TOLERANCE):
        raise ValueError(label + " is materially negative")
    clipped = int(np.count_nonzero(arr < 0.0))
    if clipped:
        _VARIANCE_CLIPPED_COUNT += clipped
        arr = np.maximum(arr, 0.0)
    return arr


def variance_clip_diagnostics(*, reset: bool = False) -> Dict[str, int]:
    """Return the cumulative count of tolerated variances clipped to zero."""
    global _VARIANCE_CLIPPED_COUNT
    payload = {"clipped_tiny_negative_variances": int(_VARIANCE_CLIPPED_COUNT)}
    if reset:
        _VARIANCE_CLIPPED_COUNT = 0
    return payload


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


def _model_prior_covariance(model, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if hasattr(model, "prior_covariance"):
        return _check_finite_array(
            model.prior_covariance(x1, x2),
            "kernel covariance",
        )
    prefactor = float(getattr(model, "prefactor", 1.0))
    if not np.isfinite(prefactor) or prefactor <= 0.0:
        raise ValueError("model kernel prefactor must be finite and positive")
    return _check_finite_array(
        prefactor * model.kernel.k(x1, x2),
        "kernel covariance",
    )


def _model_prior_diagonal(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "prior_variance_diagonal"):
        return _check_finite_array(
            model.prior_variance_diagonal(x),
            "kernel diagonal",
        ).reshape(-1)
    prefactor = float(getattr(model, "prefactor", 1.0))
    if not np.isfinite(prefactor) or prefactor <= 0.0:
        raise ValueError("model kernel prefactor must be finite and positive")
    values = (
        model.kernel.k_diag(x)
        if hasattr(model.kernel, "k_diag")
        else np.diag(model.kernel.k(x, x))
    )
    return _check_finite_array(prefactor * values, "kernel diagonal").reshape(-1)



def _model_numeric_identity(model) -> Hashable:
    identity = getattr(model, "numeric_identity", None)
    if callable(identity):
        identity = identity()
    if identity is not None:
        return (type(model).__module__, type(model).__qualname__, str(identity))
    x_raw = getattr(model, "x")
    y_raw = getattr(model, "y")
    return (
        type(model).__module__,
        type(model).__qualname__,
        id(model),
        id(x_raw),
        np.asarray(x_raw).shape,
        id(y_raw),
        np.asarray(y_raw).shape,
        id(getattr(model, "kernel", None)),
        int(getattr(model, "ntrain", 0)),
        int(getattr(model, "nfeats", 0)),
        float(getattr(model, "jitter", 0.0)),
        float(getattr(model, "prefactor", 1.0)),
    )


def _model_lower_cholesky(model) -> np.ndarray:
    factor = _check_finite_array(model.lower_cholesky, "model Cholesky factor")
    expected = (int(model.ntrain), int(model.ntrain))
    if factor.shape != expected:
        raise ValueError(
            "model Cholesky factor shape must be "
            + repr(expected)
            + ", got "
            + repr(factor.shape)
        )
    return factor


def _estimated_signal_variance(model, *, lower_cholesky: Optional[np.ndarray] = None) -> float:
    y_raw = getattr(model, "y")
    x_raw = getattr(model, "x")
    cache_key = (
        _model_numeric_identity(model),
        id(y_raw),
        np.asarray(y_raw).shape,
        id(x_raw),
        np.asarray(x_raw).shape,
        int(getattr(model, "ntrain", 0)),
    )
    cached = getattr(model, "_ichor_al_signal_variance_cache", None)
    if (
        isinstance(cached, tuple)
        and len(cached) == 2
        and cached[0] == cache_key
    ):
        return float(cached[1])
    y = _check_finite_array(model.y, "model.y").reshape((-1, 1))
    x = _check_finite_array(model.x, "model.x")
    mean = _check_finite_array(model.mean.value(x), "model mean").reshape((-1, 1))
    resid = y - mean
    factor = (
        _model_lower_cholesky(model)
        if lower_cholesky is None
        else _check_finite_array(lower_cholesky, "model Cholesky factor")
    )
    whitened = np.linalg.solve(factor, resid)
    tau2 = float((whitened.T @ whitened).reshape(-1)[0] / max(model.ntrain, 1))
    if not np.isfinite(tau2) or tau2 <= 0.0:
        tau2 = 1.0
    try:
        setattr(model, "_ichor_al_signal_variance_cache", (cache_key, float(tau2)))
    except Exception:
        pass
    return tau2


def _estimated_signal_variance_prepared(
    model,
    *,
    lower_cholesky: np.ndarray,
) -> float:
    """Prepared-path signal scale using the validated triangular solver."""
    y_raw = getattr(model, "y")
    x_raw = getattr(model, "x")
    cache_key = (
        _model_numeric_identity(model),
        id(y_raw),
        np.asarray(y_raw).shape,
        id(x_raw),
        np.asarray(x_raw).shape,
        int(getattr(model, "ntrain", 0)),
    )
    cached = getattr(model, "_ichor_al_signal_variance_cache", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == cache_key:
        return float(cached[1])
    y = _check_finite_array(model.y, "model.y").reshape((-1, 1))
    x = _check_finite_array(model.x, "model.x")
    mean = _check_finite_array(model.mean.value(x), "model mean").reshape((-1, 1))
    whitened = _solve_prepared_lower(lower_cholesky, y - mean)
    tau2 = float((whitened.T @ whitened).reshape(-1)[0] / max(model.ntrain, 1))
    if not np.isfinite(tau2) or tau2 <= 0.0:
        tau2 = 1.0
    try:
        setattr(model, "_ichor_al_signal_variance_cache", (cache_key, float(tau2)))
    except Exception:
        pass
    return tau2



def model_posterior_covariance(model, x1: np.ndarray, x2: np.ndarray, scaled: bool = True) -> np.ndarray:
    x1 = _ensure_2d(x1)
    x2 = _ensure_2d(x2)
    k12 = _model_prior_covariance(model, x1, x2)
    r1 = _check_finite_array(model.r(x1), "train-test covariance")
    r2 = _check_finite_array(model.r(x2), "train-test covariance")
    factor = _model_lower_cholesky(model)
    v1 = np.linalg.solve(factor, r1)
    v2 = np.linalg.solve(factor, r2)
    posterior = k12 - v1.T @ v2
    posterior = 0.5 * (posterior + posterior.T) if x1.shape == x2.shape and np.array_equal(x1, x2) else posterior
    if scaled:
        posterior = _estimated_signal_variance(
            model,
            lower_cholesky=factor,
        ) * posterior
    return _check_finite_array(posterior, "posterior covariance")


class PreparedTotalEnergyPosteriorBatch:
    """Reusable posterior state for a fixed set of feature rows.

    The expensive train-query covariance and lower-Cholesky solve are performed
    once per atom. Subsequent variance and rectangular covariance requests use
    indexed slices of those immutable projections.
    """

    def __init__(
        self,
        *,
        posterior: "TotalEnergyPosterior",
        row_ids: np.ndarray,
        features: Mapping[str, np.ndarray],
        projections: Mapping[str, np.ndarray],
        means: np.ndarray,
        variances: np.ndarray,
        signal_variances: Mapping[str, float],
    ) -> None:
        ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
        if len(set(int(value) for value in ids)) != int(ids.size):
            raise ValueError("prepared posterior row IDs must be unique")
        self.posterior = posterior
        self.row_ids = ids
        self.features = {str(key): np.asarray(value) for key, value in features.items()}
        self.projections = {
            str(key): np.asarray(value) for key, value in projections.items()
        }
        self.means = _check_finite_array(means, "prepared posterior means").reshape(-1)
        self.variances = _check_variance_array(
            variances, "prepared posterior variances"
        ).reshape(-1)
        self.signal_variances = {
            str(key): float(value) for key, value in signal_variances.items()
        }
        if self.means.shape != ids.shape or self.variances.shape != ids.shape:
            raise ValueError("prepared posterior value count mismatch")
        expected_atoms = set(str(atom) for atom in posterior._property_models)
        if (
            set(self.features) != expected_atoms
            or set(self.projections) != expected_atoms
            or set(self.signal_variances) != expected_atoms
        ):
            raise ValueError("prepared posterior atom coverage mismatch")
        for atom, model in posterior._property_models.items():
            feature_shape = (
                int(ids.size),
                int(getattr(model, "nfeats", self.features[str(atom)].shape[1])),
            )
            projection_shape = (int(model.ntrain), int(ids.size))
            if self.features[str(atom)].shape != feature_shape:
                raise ValueError(
                    "prepared posterior feature shape mismatch for atom "
                    + str(atom)
                )
            if self.projections[str(atom)].shape != projection_shape:
                raise ValueError(
                    "prepared posterior projection shape mismatch for atom "
                    + str(atom)
                )
            signal = float(self.signal_variances[str(atom)])
            if not np.isfinite(signal) or signal <= 0.0:
                raise ValueError(
                    "prepared posterior signal variance is invalid for atom "
                    + str(atom)
                )
        self._positions = {int(value): index for index, value in enumerate(ids)}

    def _row_positions(self, row_ids: Sequence[int]) -> np.ndarray:
        try:
            return np.asarray(
                [self._positions[int(value)] for value in row_ids], dtype=np.int64
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError("prepared posterior row ID is unavailable") from exc

    def variances_by_index(self, row_ids: Sequence[int]) -> np.ndarray:
        positions = self._row_positions(row_ids)
        return _check_variance_array(
            self.variances[positions], "prepared indexed variances"
        )

    def means_by_index(self, row_ids: Sequence[int]) -> np.ndarray:
        positions = self._row_positions(row_ids)
        return _check_finite_array(
            self.means[positions], "prepared indexed means"
        )

    def cross_covariances_by_index(
        self,
        left_row_ids: Sequence[int],
        right_row_ids: Sequence[int],
    ) -> np.ndarray:
        left_positions = self._row_positions(left_row_ids)
        right_positions = self._row_positions(right_row_ids)
        total = np.zeros(
            (int(left_positions.size), int(right_positions.size)), dtype=float
        )
        for atom, model in self.posterior._property_models.items():
            feature_rows = self.features[str(atom)]
            left_features = np.asarray(feature_rows[left_positions], dtype=float)
            right_features = np.asarray(feature_rows[right_positions], dtype=float)
            prior = _model_prior_covariance(model, left_features, right_features)
            expected = (int(left_positions.size), int(right_positions.size))
            if prior.shape != expected:
                raise ValueError(
                    "prepared kernel cross-covariance shape for atom "
                    + str(atom)
                    + " must be "
                    + repr(expected)
                )
            projection = self.projections[str(atom)]
            left_projection = np.asarray(projection[:, left_positions], dtype=float)
            right_projection = np.asarray(projection[:, right_positions], dtype=float)
            atom_covariance = prior - left_projection.T @ right_projection
            atom_covariance *= float(self.signal_variances[str(atom)])
            total += atom_covariance
        return _check_finite_array(total, "prepared posterior cross-covariances")

    def covariance_matrix_by_index(self, row_ids: Sequence[int]) -> np.ndarray:
        covariance = self.cross_covariances_by_index(row_ids, row_ids)
        covariance = 0.5 * (covariance + covariance.T)
        return _check_finite_array(covariance, "prepared posterior covariance matrix")


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
        self._model_identity = tuple(
            sorted(
                (str(model.atom), _model_numeric_identity(model))
                for model in property_models
            )
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
            "n_prepared_batches": 0,
            "n_prepared_projection_solves": 0,
        }

    def _normalise_feature_arrays(
        self,
        feature_arrays: Mapping[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], int]:
        if not isinstance(feature_arrays, Mapping):
            raise TypeError("prepared posterior features must be a mapping")
        normalised: Dict[str, np.ndarray] = {}
        n_rows: Optional[int] = None
        for atom, model in self._property_models.items():
            if atom not in feature_arrays:
                raise KeyError("prepared posterior features missing atom " + str(atom))
            values = _check_finite_array(
                feature_arrays[atom], "prepared features for atom " + str(atom)
            )
            values = _ensure_2d(values)
            expected_features = int(getattr(model, "nfeats", values.shape[1]))
            if values.shape[1] != expected_features:
                raise ValueError(
                    "prepared feature dimension for atom "
                    + str(atom)
                    + " must be "
                    + str(expected_features)
                )
            _check_kernel_active_dims(
                model, expected_features, "prepared atom " + str(atom)
            )
            if n_rows is None:
                n_rows = int(values.shape[0])
            elif int(values.shape[0]) != n_rows:
                raise ValueError("prepared feature row counts do not match")
            normalised[str(atom)] = values
        return normalised, int(n_rows or 0)

    def prepare_feature_batch(
        self,
        feature_arrays: Mapping[str, np.ndarray],
        *,
        row_ids: Optional[Sequence[int]] = None,
        projection_directory: Optional[Union[str, Path]] = None,
        max_resident_projection_bytes: Optional[int] = None,
        projection_resume_columns: Optional[Mapping[str, int]] = None,
        projection_progress_callback: Optional[
            Callable[[str, int, int], None]
        ] = None,
    ) -> PreparedTotalEnergyPosteriorBatch:
        """Prepare one immutable feature batch for repeated indexed queries."""
        features, n_rows = self._normalise_feature_arrays(feature_arrays)
        ids = (
            np.arange(n_rows, dtype=np.int64)
            if row_ids is None
            else np.asarray(row_ids, dtype=np.int64).reshape(-1)
        )
        if ids.shape != (n_rows,):
            raise ValueError("prepared posterior row ID count mismatch")
        total_means = np.zeros(n_rows, dtype=float)
        total_variances = np.zeros(n_rows, dtype=float)
        projections: Dict[str, np.ndarray] = {}
        signal_variances: Dict[str, float] = {}
        total_projection_bytes = sum(
            int(model.ntrain) * int(n_rows) * np.dtype(np.float64).itemsize
            for model in self._property_models.values()
        )
        projection_limit = (
            None
            if max_resident_projection_bytes is None
            else max(1, int(max_resident_projection_bytes))
        )
        use_disk = (
            projection_limit is not None
            and total_projection_bytes > projection_limit
        )
        projection_root: Optional[Path] = None
        if use_disk:
            if projection_directory is None:
                raise ValueError(
                    "prepared posterior projections exceed the resident-memory "
                    "limit but no projection directory was supplied"
                )
            projection_root = Path(projection_directory)
            if projection_root.is_symlink():
                raise ValueError("prepared posterior projection directory is symlinked")
            projection_root.mkdir(parents=True, exist_ok=True)
        resume_columns = {
            str(atom): int(completed)
            for atom, completed in (projection_resume_columns or {}).items()
        }
        unknown_resume_atoms = set(resume_columns) - set(self._property_models)
        if unknown_resume_atoms:
            raise ValueError(
                "prepared projection resume contains unknown atoms "
                + repr(sorted(unknown_resume_atoms))
            )
        if resume_columns and not use_disk:
            raise ValueError(
                "prepared projection resume requires disk-backed projections"
            )

        for atom_position, (atom, model) in enumerate(self._property_models.items()):
            rows = features[str(atom)]
            predictions = _check_finite_array(
                model.predict(rows), "prepared model predictions"
            ).reshape(-1)
            if predictions.shape != (n_rows,):
                raise ValueError(
                    "prepared model prediction count mismatch for atom " + str(atom)
                )
            total_means += predictions
            prior_diagonal = _model_prior_diagonal(model, rows)
            if prior_diagonal.shape != (n_rows,):
                raise ValueError(
                    "prepared kernel diagonal count mismatch for atom " + str(atom)
                )
            expected = (int(model.ntrain), n_rows)
            factor = _model_lower_cholesky(model)
            signal = (
                _estimated_signal_variance_prepared(
                    model,
                    lower_cholesky=factor,
                )
                if self.scaled
                else 1.0
            )
            if use_disk:
                assert projection_root is not None
                projection_path = (
                    projection_root
                    / ("projection-" + str(atom_position).zfill(4) + ".npy")
                )
                completed_columns = int(resume_columns.get(str(atom), 0))
                if completed_columns < 0 or completed_columns > n_rows:
                    raise ValueError(
                        "prepared projection resume count is invalid for atom "
                        + str(atom)
                    )
                if completed_columns:
                    if projection_path.is_symlink() or not projection_path.is_file():
                        raise ValueError(
                            "prepared projection resume file is missing for atom "
                            + str(atom)
                        )
                    projection = np.lib.format.open_memmap(
                        projection_path,
                        mode="r+",
                    )
                    if (
                        projection.shape != expected
                        or projection.dtype != np.dtype(np.float64)
                    ):
                        raise ValueError(
                            "prepared projection resume array is invalid for atom "
                            + str(atom)
                        )
                else:
                    projection = np.lib.format.open_memmap(
                        projection_path,
                        mode="w+",
                        dtype=np.float64,
                        shape=expected,
                    )
                # Keep the train-query matrix and its solution comfortably below
                # the aggregate resident limit while retaining BLAS-sized blocks.
                bytes_per_column = max(1, int(model.ntrain)) * 8
                work_budget = max(1, int(projection_limit or 1) // 4)
                column_chunk = max(1, min(n_rows, work_budget // bytes_per_column))
                squared_norms = np.empty(n_rows, dtype=float)
                if completed_columns:
                    squared_norms[:completed_columns] = np.sum(
                        projection[:, :completed_columns]
                        * projection[:, :completed_columns],
                        axis=0,
                    )
                for start in range(completed_columns, n_rows, column_chunk):
                    stop = min(n_rows, start + column_chunk)
                    train_query = _check_finite_array(
                        model.r(rows[start:stop]),
                        "prepared train-test covariance",
                    )
                    expected_chunk = (int(model.ntrain), int(stop - start))
                    if train_query.shape != expected_chunk:
                        raise ValueError(
                            "prepared train-test covariance shape for atom "
                            + str(atom)
                            + " must be "
                            + repr(expected_chunk)
                        )
                    solved = _solve_prepared_lower(factor, train_query)
                    self._diagnostic_add("n_prepared_projection_solves")
                    projection[:, start:stop] = solved
                    squared_norms[start:stop] = np.sum(solved * solved, axis=0)
                    projection.flush()
                    if projection_progress_callback is not None:
                        projection_progress_callback(
                            str(atom), int(start), int(stop)
                        )
                projection.flush()
            else:
                train_query = _check_finite_array(
                    model.r(rows), "prepared train-test covariance"
                )
                if train_query.shape != expected:
                    raise ValueError(
                        "prepared train-test covariance shape for atom "
                        + str(atom)
                        + " must be "
                        + repr(expected)
                    )
                projection = _solve_prepared_lower(factor, train_query)
                self._diagnostic_add("n_prepared_projection_solves")
                squared_norms = np.sum(projection * projection, axis=0)
            atom_variance = signal * (prior_diagonal - squared_norms)
            total_variances += atom_variance
            projections[str(atom)] = projection
            signal_variances[str(atom)] = float(signal)
        self._diagnostic_add("n_prepared_batches")
        return PreparedTotalEnergyPosteriorBatch(
            posterior=self,
            row_ids=ids,
            features=features,
            projections=projections,
            means=total_means,
            variances=total_variances,
            signal_variances=signal_variances,
        )

    def prepare_points(
        self,
        points: Sequence[GeometryInput],
        *,
        row_ids: Optional[Sequence[int]] = None,
    ) -> PreparedTotalEnergyPosteriorBatch:
        features = [self._features(point) for point in points]
        arrays = {
            atom: self._stack_feature_rows(features, atom)
            for atom in self._property_models
        }
        return self.prepare_feature_batch(arrays, row_ids=row_ids)

    def variances_from_feature_arrays(
        self,
        feature_arrays: Mapping[str, np.ndarray],
        *,
        chunk_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        features, n_rows = self._normalise_feature_arrays(feature_arrays)
        if n_rows == 0:
            return np.zeros(0, dtype=float)
        size = n_rows if chunk_size is None else max(1, int(chunk_size))
        output = np.empty(n_rows, dtype=float)
        for start in range(0, n_rows, size):
            stop = min(n_rows, start + size)
            batch = self.prepare_feature_batch(
                {atom: values[start:stop] for atom, values in features.items()},
                row_ids=np.arange(start, stop, dtype=np.int64),
            )
            output[start:stop] = batch.variances
            if progress_callback is not None:
                progress_callback(int(stop), int(n_rows))
        return _check_variance_array(output, "prepared feature variances")

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
    def _geometry_key(x: GeometryInput) -> Hashable:
        if isinstance(x, dict):
            rows = []
            for atom in sorted(x):
                arr = _check_finite_array(x[atom], "posterior geometry mapping")
                rows.append(
                    (
                        str(atom),
                        tuple(arr.shape),
                        tuple(np.round(arr.reshape(-1), 12)),
                    )
                )
            return ("mapping", tuple(rows))
        if isinstance(x, Atoms):
            coordinates = _check_finite_array(
                x.coordinates,
                "posterior atom coordinates",
            )
            identities = tuple(
                (
                    str(getattr(atom, "name", "")),
                    str(getattr(atom, "type", "")),
                )
                for atom in x
            )
            return (
                "atoms",
                identities,
                tuple(coordinates.shape),
                tuple(np.round(coordinates.reshape(-1), 12)),
            )
        arr = _check_finite_array(x, "posterior geometry array")
        return (
            "array",
            tuple(arr.shape),
            tuple(np.round(arr.reshape(-1), 12)),
        )

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
        ordered_keys = sorted((key1, key2), key=repr)
        cache_key = (self._model_identity, ordered_keys[0], ordered_keys[1])
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
            k_diag = _model_prior_diagonal(model, X)
            if k_diag.shape != (n,):
                raise ValueError(
                    f"kernel diagonal shape for atom {atom} must be {(n,)}, got {k_diag.shape}"
                )
            r = _check_finite_array(model.r(X), "train-test covariance")
            factor = _model_lower_cholesky(model)
            v = np.linalg.solve(factor, r)
            diag = k_diag - np.sum(v * v, axis=0)
            if self.scaled:
                diag = _estimated_signal_variance(
                    model,
                    lower_cholesky=factor,
                ) * diag
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
            kxx = _model_prior_covariance(model, X, X)
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
            factor = _model_lower_cholesky(model)
            v = np.linalg.solve(factor, r)
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
                atom_cov = _estimated_signal_variance(
                    model,
                    lower_cholesky=factor,
                ) * atom_cov
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
        left_features = [self._features(point) for point in left]
        right_features = [self._features(point) for point in right]
        total = np.zeros((n_left, n_right), dtype=float)
        for atom, model in self._property_models.items():
            x_left = self._stack_feature_rows(left_features, atom)
            x_right = self._stack_feature_rows(right_features, atom)
            prior = _model_prior_covariance(model, x_left, x_right)
            expected_prior_shape = (n_left, n_right)
            if prior.shape != expected_prior_shape:
                raise ValueError(
                    f"kernel cross-covariance shape for atom {atom} must be "
                    f"{expected_prior_shape}, got {prior.shape}"
                )
            r_left = _check_finite_array(
                model.r(x_left),
                "left train-test covariance",
            )
            r_right = _check_finite_array(
                model.r(x_right),
                "right train-test covariance",
            )
            expected_left = (int(model.ntrain), n_left)
            expected_right = (int(model.ntrain), n_right)
            if r_left.shape != expected_left or r_right.shape != expected_right:
                raise ValueError(
                    f"rectangular train-test covariance shape mismatch for atom {atom}: "
                    f"{r_left.shape}/{r_right.shape} != "
                    f"{expected_left}/{expected_right}"
                )
            factor = _model_lower_cholesky(model)
            solved_left = np.linalg.solve(factor, r_left)
            solved_right = np.linalg.solve(factor, r_right)
            atom_covariance = prior - solved_left.T @ solved_right
            if self.scaled:
                atom_covariance = _estimated_signal_variance(
                    model,
                    lower_cholesky=factor,
                ) * atom_covariance
            total += atom_covariance
        return _check_finite_array(total, "posterior cross-covariances")

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
