"""Runtime contract checks for daemon-owned FEREBUS model commits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


VARIANCE_NEGATIVE_TOLERANCE = 1.0e-10


class ModelContractError(ValueError):
    """Raised when a staged or committed FEREBUS model set is not usable."""


@dataclass(frozen=True)
class FerebusTask:
    property: str
    atom: str
    alf_1_indexed: Tuple[int, int, int]
    config_path: Path
    expected_model_path: Path
    train_rows: Optional[int]

    @property
    def key(self) -> Tuple[str, str]:
        return (self.property, self.atom)

    @property
    def model_name(self) -> str:
        return self.expected_model_path.name

    @property
    def alf_zero_indexed(self) -> Tuple[int, int, int]:
        return tuple(int(x) - 1 for x in self.alf_1_indexed)


def _task_from_payload(payload: Mapping[str, Any]) -> FerebusTask:
    try:
        prop = str(payload["property"])
        atom = str(payload["atom"])
        raw_alf = tuple(int(x) for x in payload["alf_1_indexed"])
        config_path = Path(str(payload["config_path"]))
        model_path = Path(str(payload["expected_model_path"]))
    except Exception as exc:
        raise ModelContractError(
            "ferebus_manifest_task_invalid: "
            + type(exc).__name__
            + ": "
            + str(exc)
        ) from exc
    if len(raw_alf) != 3:
        raise ModelContractError("ferebus_manifest_task_invalid: ALF must have 3 indexes")
    if any(x <= 0 for x in raw_alf):
        raise ModelContractError("ferebus_manifest_task_invalid: ALF must be 1-indexed")
    rows = None
    row_counts = payload.get("row_counts") or {}
    if isinstance(row_counts, Mapping) and row_counts.get("train") is not None:
        rows = int(row_counts["train"])
    return FerebusTask(
        property=prop,
        atom=atom,
        alf_1_indexed=raw_alf,
        config_path=config_path,
        expected_model_path=model_path,
        train_rows=rows,
    )


def _manifest_tasks(manifest: Mapping[str, Any]) -> List[FerebusTask]:
    tasks_payload = manifest.get("tasks") or []
    if not isinstance(tasks_payload, list) or not tasks_payload:
        raise ModelContractError("ferebus_manifest_has_no_tasks")
    tasks = [_task_from_payload(t) for t in tasks_payload]
    seen = set()
    for task in tasks:
        if task.key in seen:
            raise ModelContractError(
                "ferebus_manifest_duplicate_task: "
                + task.property
                + "-"
                + task.atom
            )
        seen.add(task.key)
    expected_n = manifest.get("n_tasks")
    if expected_n is not None and int(expected_n) != len(tasks):
        raise ModelContractError(
            "ferebus_manifest_n_tasks_mismatch: "
            + str(expected_n)
            + "!="
            + str(len(tasks))
        )
    props = manifest.get("properties")
    if props is not None and set(str(p) for p in props) != {t.property for t in tasks}:
        raise ModelContractError("ferebus_manifest_properties_mismatch")
    atoms = manifest.get("atoms")
    if atoms is not None and set(str(a) for a in atoms) != {t.atom for t in tasks}:
        raise ModelContractError("ferebus_manifest_atoms_mismatch")
    return tasks


def _model_path_for(root: Path, task: FerebusTask, *, committed: bool) -> Path:
    if committed:
        return root / task.model_name
    return task.expected_model_path


def _config_path_for(root: Path, task: FerebusTask, *, committed: bool) -> Path:
    if committed:
        return root / ("ferebus_" + task.property + "_" + task.atom + ".config")
    return task.config_path


def _count_section_rows(path: Path, section: str) -> Optional[int]:
    marker = "[" + section + "]"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            in_section = False
            rows = 0
            for raw in handle:
                line = raw.strip()
                if not in_section:
                    if line == marker:
                        in_section = True
                    continue
                if not line:
                    break
                if line.startswith("["):
                    break
                rows += 1
            if in_section:
                return rows
    except OSError as exc:
        raise ModelContractError("model_file_unreadable: " + str(path)) from exc
    return None


def _declared_ntrain(path: Path) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if "number_of_training_points" not in line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ModelContractError("model_ntrain_unparseable")
                return int(parts[1])
    except OSError as exc:
        raise ModelContractError("model_file_unreadable: " + str(path)) from exc
    except ValueError as exc:
        raise ModelContractError("model_ntrain_unparseable") from exc
    return None


def _check_section_rows(path: Path, ntrain: int) -> None:
    for section in ("training_data.x", "training_data.y", "weights"):
        rows = _count_section_rows(path, section)
        if rows is None:
            raise ModelContractError("model_section_missing: " + section)
        if rows < ntrain:
            if section == "training_data.x":
                raise ModelContractError("model_truncated")
            raise ModelContractError("model_truncated:" + section)
        if rows > ntrain:
            raise ModelContractError("model_row_count_mismatch:" + section)


def _as_finite_array(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        raise ModelContractError(label + "_empty")
    if not np.all(np.isfinite(arr)):
        raise ModelContractError(label + "_non_finite")
    return arr


def _model_attr(model: Any, name: str) -> Any:
    try:
        value = getattr(model, name)
    except Exception as exc:
        raise ModelContractError(
            "model_parse_failed:" + name + ":" + type(exc).__name__ + ": " + str(exc)
        ) from exc
    if str(value) == "FileContents":
        raise ModelContractError("model_field_missing:" + name)
    return value


def _validate_model_object(model: Any, path: Path, task: FerebusTask, system: str) -> None:
    system_name = str(_model_attr(model, "system_name"))
    atom = str(_model_attr(model, "atom"))
    prop = str(_model_attr(model, "type"))
    if system_name != str(system):
        raise ModelContractError("model_metadata_mismatch:system")
    if atom != task.atom:
        raise ModelContractError("model_metadata_mismatch:atom")
    if prop != task.property:
        raise ModelContractError("model_metadata_mismatch:property")

    alf = tuple(int(x) for x in np.asarray(_model_attr(model, "ialf"), dtype=int).reshape(-1))
    if alf != task.alf_zero_indexed:
        raise ModelContractError("model_metadata_mismatch:ALF")

    ntrain = int(_model_attr(model, "ntrain"))
    nfeats = int(_model_attr(model, "nfeats"))
    if ntrain <= 0:
        raise ModelContractError("model_ntrain_invalid")
    if nfeats <= 0:
        raise ModelContractError("model_nfeats_invalid")
    if task.train_rows is not None and ntrain != int(task.train_rows):
        raise ModelContractError("model_metadata_mismatch:number_of_training_points")

    _check_section_rows(path, ntrain)

    x = _as_finite_array(_model_attr(model, "x"), "model_training_x")
    y = _as_finite_array(_model_attr(model, "y"), "model_training_y")
    weights = _as_finite_array(_model_attr(model, "weights"), "model_weights")
    if x.shape != (ntrain, nfeats):
        raise ModelContractError("model_training_x_shape_mismatch")
    if y.reshape(-1).shape[0] != ntrain:
        raise ModelContractError("model_training_y_shape_mismatch")
    if weights.reshape(-1).shape[0] != ntrain:
        raise ModelContractError("model_weights_shape_mismatch")

    kernel = _model_attr(model, "kernel")
    if kernel is None:
        raise ModelContractError("model_kernel_missing")
    params = getattr(kernel, "params", None)
    if params is not None:
        _as_finite_array(params, "model_kernel_params")
    active_dims = np.asarray(getattr(kernel, "active_dims", []), dtype=int).reshape(-1)
    if active_dims.size and (active_dims.min() < 0 or active_dims.max() >= nfeats):
        raise ModelContractError("model_kernel_active_dims_mismatch")

    _as_finite_array(_model_attr(model, "R"), "model_covariance")
    _as_finite_array(_model_attr(model, "lower_cholesky"), "model_cholesky")
    probe = x[: min(1, ntrain)]
    pred = _as_finite_array(model.predict(probe), "model_predict")
    var = _as_finite_array(model.variance(probe), "model_variance")
    if pred.reshape(-1).shape[0] != probe.shape[0]:
        raise ModelContractError("model_predict_shape_mismatch")
    if var.reshape(-1).shape[0] != probe.shape[0]:
        raise ModelContractError("model_variance_shape_mismatch")
    if np.any(var < -VARIANCE_NEGATIVE_TOLERANCE):
        raise ModelContractError("model_variance_negative")


def validate_ferebus_model_contract(
    root_dir: Path,
    *,
    committed: bool = False,
) -> None:
    """Validate a staged or committed FEREBUS model directory.

    Staged directories use task ``expected_model_path`` and ``config_path``. Committed model
    versions keep models/configs flat, so task paths are remapped to their committed basenames.
    """
    from ichor.core.models import Model, Models
    from . import input_staging as _stg

    root = Path(root_dir)
    if not root.is_dir():
        raise ModelContractError("ferebus_model_root_missing: " + str(root))
    manifest = _stg.read_ferebus_manifest(root)
    system = str(manifest.get("system"))
    tasks = _manifest_tasks(manifest)
    expected_models = set()
    model_paths: List[Tuple[FerebusTask, Path]] = []
    for task in tasks:
        config_path = _config_path_for(root, task, committed=committed)
        model_path = _model_path_for(root, task, committed=committed)
        expected_models.add(model_path.resolve())
        if not config_path.is_file():
            raise ModelContractError("ferebus_config_missing: " + str(config_path))
        if not model_path.is_file():
            raise ModelContractError("expected_model_missing: " + str(model_path))
        if model_path.stat().st_size <= 0:
            raise ModelContractError("model_file_empty")
        model_paths.append((task, model_path))

    for task, model_path in model_paths:
        declared_ntrain = _declared_ntrain(model_path)
        if declared_ntrain is not None:
            _check_section_rows(model_path, declared_ntrain)
        try:
            model = Model(model_path)
            _validate_model_object(model, model_path, task, system)
        except ModelContractError:
            raise
        except Exception as exc:
            raise ModelContractError(
                "model_parse_failed: "
                + str(model_path)
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

    actual_models = {p.resolve() for p in root.rglob("*.model")}
    extra_models = actual_models - expected_models
    if extra_models:
        raise ModelContractError(
            "unexpected_model_file: " + str(sorted(str(p) for p in extra_models)[0])
        )

    expected_keys = {task.key for task in tasks}
    if committed:
        try:
            models = Models(root)
            loaded = list(models)
        except Exception as exc:
            raise ModelContractError(
                "models_load_failed: " + type(exc).__name__ + ": " + str(exc)
            ) from exc
        loaded_keys = {(str(m.type), str(m.atom)) for m in loaded}
        if loaded_keys != expected_keys:
            raise ModelContractError(
                "models_coverage_mismatch: "
                + repr(sorted(loaded_keys))
                + "!="
                + repr(sorted(expected_keys))
            )


def _validate_variance_array(values: Any, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ModelContractError(context + "_non_finite")
    if np.any(arr < -VARIANCE_NEGATIVE_TOLERANCE):
        raise ModelContractError(context + "_negative")
    return arr


def smoke_total_energy_posterior(
    models_dir: Path,
    *,
    property_name: str,
    probe_frames: Sequence[Any],
) -> Any:
    """Load committed models and prove the total-energy posterior evaluates."""
    from ichor.core.adversarial.posterior import TotalEnergyPosterior
    from ichor.core.models import Models

    root = Path(models_dir)
    try:
        models = Models(root)
        posterior = TotalEnergyPosterior(models, property_name=str(property_name))
    except Exception as exc:
        raise ModelContractError(
            "posterior_load_failed: " + type(exc).__name__ + ": " + str(exc)
        ) from exc
    frames = list(probe_frames)
    if not frames:
        raise ModelContractError("posterior_smoke_has_no_probe_frames")
    try:
        _validate_variance_array(
            [posterior.variance(frames[0])],
            "posterior_variance",
        )
        _validate_variance_array(
            posterior.variances(frames[: min(3, len(frames))]),
            "posterior_variances",
        )
    except ModelContractError:
        raise
    except Exception as exc:
        raise ModelContractError(
            "posterior_smoke_failed: " + type(exc).__name__ + ": " + str(exc)
        ) from exc
    return posterior


def validate_reference_scales(scales: Mapping[str, Any]) -> Dict[str, float]:
    required = ("energy", "force", "omega", "anh", "anh_std")
    if not isinstance(scales, Mapping):
        raise ModelContractError("reference_scales_must_be_mapping")
    out: Dict[str, float] = {}
    for key in required:
        if key not in scales:
            raise ModelContractError("reference_scale_missing:" + key)
        try:
            value = float(scales[key])
        except (TypeError, ValueError) as exc:
            raise ModelContractError("reference_scale_invalid:" + key) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ModelContractError("reference_scale_non_finite_or_non_positive:" + key)
        out[key] = value
    return out
