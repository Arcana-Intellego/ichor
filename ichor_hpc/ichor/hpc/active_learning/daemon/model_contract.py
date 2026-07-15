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


def _resolve_under(root: Path, raw: Path, label: str) -> Path:
    base = Path(root).resolve(strict=False)
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise ModelContractError(label + "_path_unresolvable: " + str(path)) from exc
    if resolved != base and base not in resolved.parents:
        raise ModelContractError(label + "_path_escapes_staging: " + str(path))
    if path.is_symlink() or resolved.is_symlink():
        raise ModelContractError(label + "_path_is_symlink: " + str(path))
    return resolved


def _version_from_iteration_dir(root: Path) -> Optional[int]:
    name = Path(root).name
    prefix = "iteration-"
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix):])
    except ValueError:
        return None


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


def _validate_kernel_family(kernel: Any, family: str, nfeats: int) -> None:
    """Validate the exact native FEREBUS kernel topology for one public family."""
    from ichor.core.models.kernels import PeriodicKernel, RBF, RBFCyclic
    from ichor.core.models.kernels.kernel import KernelProd

    all_dimensions = np.arange(nfeats, dtype=int)
    if family == "rbf":
        if type(kernel) is not RBF:
            raise ModelContractError("model_kernel_family_mismatch:rbf")
        active = np.asarray(kernel.active_dims, dtype=int).reshape(-1)
        if not np.array_equal(active, all_dimensions):
            raise ModelContractError("model_kernel_active_dims_mismatch:rbf")
        if np.asarray(kernel.params, dtype=float).reshape(-1).size != nfeats:
            raise ModelContractError("model_kernel_parameter_count_mismatch:rbf")
        return

    if family != "periodic_rbf":
        raise ModelContractError("model_kernel_family_unsupported:" + repr(family))
    if type(kernel) is not KernelProd:
        raise ModelContractError("model_kernel_family_mismatch:periodic_rbf")
    if type(kernel.k1) is not RBFCyclic or type(kernel.k2) is not PeriodicKernel:
        raise ModelContractError("model_kernel_composition_mismatch:periodic_rbf")
    periodic_dimensions = np.asarray(
        [index for index in range(3, nfeats) if (index + 1) % 3 == 0],
        dtype=int,
    )
    periodic_dimension_set = set(periodic_dimensions.tolist())
    cyclic_dimensions = np.asarray(
        [index for index in range(nfeats) if index not in periodic_dimension_set],
        dtype=int,
    )
    if not np.array_equal(
        np.asarray(kernel.k1.active_dims, dtype=int).reshape(-1),
        cyclic_dimensions,
    ):
        raise ModelContractError("model_kernel_active_dims_mismatch:rbf_cyclic")
    if not np.array_equal(
        np.asarray(kernel.k2.active_dims, dtype=int).reshape(-1),
        periodic_dimensions,
    ):
        raise ModelContractError("model_kernel_active_dims_mismatch:periodic")
    if np.asarray(kernel.k1.params, dtype=float).reshape(-1).size != cyclic_dimensions.size:
        raise ModelContractError("model_kernel_parameter_count_mismatch:rbf_cyclic")
    periodic_params = getattr(kernel.k2, "_thetas", np.asarray([], dtype=float))
    if np.asarray(periodic_params, dtype=float).reshape(-1).size != periodic_dimensions.size:
        raise ModelContractError("model_kernel_parameter_count_mismatch:periodic")


def _validate_model_object(
    model: Any,
    path: Path,
    task: FerebusTask,
    system: str,
    *,
    kernel_family: Optional[str] = None,
) -> None:
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
        finite_params = _as_finite_array(params, "model_kernel_params")
        if np.any(finite_params < 0.0):
            raise ModelContractError("model_kernel_parameter_negative")
    active_dims = np.asarray(getattr(kernel, "active_dims", []), dtype=int).reshape(-1)
    if active_dims.size and (active_dims.min() < 0 or active_dims.max() >= nfeats):
        raise ModelContractError("model_kernel_active_dims_mismatch")
    prefactor = float(_model_attr(model, "kernel_prefactor"))
    if not np.isfinite(prefactor) or prefactor <= 0.0:
        raise ModelContractError("model_kernel_prefactor_invalid")
    jitter = float(_model_attr(model, "jitter"))
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ModelContractError("model_jitter_invalid")
    if kernel_family is not None:
        _validate_kernel_family(kernel, str(kernel_family), nfeats)

    # Admission must stay bounded in ntrain. Native FEREBUS has already built
    # and reported the dense training factorisation in its authenticated .perf
    # receipt; the daemon only probes the serialised prediction contract here.
    probe = x[: min(1, ntrain)]
    prior_diag = _as_finite_array(
        model.prior_variance_diagonal(probe),
        "model_prior_variance_diagonal",
    )
    cross = _as_finite_array(model.r(probe), "model_cross_covariance")
    if prior_diag.shape != (probe.shape[0],):
        raise ModelContractError("model_prior_variance_diagonal_shape_mismatch")
    if cross.shape != (ntrain, probe.shape[0]):
        raise ModelContractError("model_cross_covariance_shape_mismatch")
    if np.any(prior_diag < -VARIANCE_NEGATIVE_TOLERANCE):
        raise ModelContractError("model_prior_variance_negative")
    pred = _as_finite_array(model.predict(probe), "model_predict")
    if pred.reshape(-1).shape[0] != probe.shape[0]:
        raise ModelContractError("model_predict_shape_mismatch")


def validate_imported_model_file(
    path: Path,
    *,
    system: str,
    property_name: str,
    atom: str,
    alf_zero_indexed: Sequence[int],
    train_rows: int,
    prior_contract: Any = None,
) -> None:
    """Apply the complete runtime model contract during bootstrap admission."""
    from ichor.core.models import Model

    zero_indexed = tuple(int(value) for value in alf_zero_indexed)
    task = FerebusTask(
        property=str(property_name),
        atom=str(atom),
        alf_1_indexed=tuple(value + 1 for value in zero_indexed),  # type: ignore[arg-type]
        config_path=Path("<bootstrap>"),
        expected_model_path=Path(path),
        train_rows=int(train_rows),
    )
    try:
        model = Model(path)
        _validate_model_object(model, Path(path), task, str(system))
        if prior_contract is not None:
            from ..ferebus_prior import validate_model_prior_mean

            validate_model_prior_mean(
                model,
                contract=prior_contract,
                property_name=property_name,
                atom=atom,
                training_values=np.asarray(model.y, dtype=float).reshape(-1),
            )
    except ModelContractError:
        raise
    except Exception as exc:
        raise ModelContractError(
            "imported_model_validation_failed:"
            + type(exc).__name__
            + ": "
            + str(exc)
        ) from exc


def validate_ferebus_model_contract(
    root_dir: Path,
    *,
    committed: bool = False,
    expected_version: Optional[int] = None,
    trained_model_set: Any = None,
) -> None:
    """Validate a staged or committed FEREBUS model directory.

    Staged directories use paths relative to FEREBUS staging. Committed model
    versions use the authoritative trained-model set manifest.
    """
    from ichor.core.models import Model, Models
    from . import input_staging as _stg

    root = Path(root_dir)
    if not root.is_dir():
        raise ModelContractError("ferebus_model_root_missing: " + str(root))
    manifest = _stg.read_ferebus_manifest(
        root,
        verify_dataset_files=not committed,
    )
    from ..ferebus_prior import (
        contract_from_payload,
        validate_ferebus_config_contract,
        validate_model_prior_mean,
    )
    from ..versioning.manifest import sha256_file

    try:
        prior_contract = contract_from_payload(manifest.get("prior_mean_contract"))
    except Exception as exc:
        raise ModelContractError(
            "ferebus_prior_contract_invalid:" + type(exc).__name__ + ":" + str(exc)
        ) from exc
    model_set = trained_model_set
    if committed:
        if expected_version is None:
            expected_version = _version_from_iteration_dir(root)
        try:
            from ..versioning.trained_models import (
                resolve_trained_model_set,
            )

            if expected_version is None:
                raise ModelContractError("committed_model_version_unparseable")
            if model_set is None:
                model_set = resolve_trained_model_set(
                    root.parent.parent,
                    int(expected_version),
                    verification="deep",
                )
            if int(model_set.version) != int(expected_version):
                raise ModelContractError("trained_model_set_version_mismatch")
            if Path(model_set.root).resolve() != root.resolve():
                raise ModelContractError("trained_model_set_root_mismatch")
        except Exception as exc:
            raise ModelContractError(
                "trained_model_set_invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
    system = str(manifest.get("system"))
    kernel_contract = manifest.get("kernel_contract")
    if not isinstance(kernel_contract, Mapping):
        raise ModelContractError("ferebus_kernel_contract_missing")
    kernel_family = str(kernel_contract.get("family") or "")
    tasks = _manifest_tasks(manifest)
    if not committed:
        for raw_task in manifest.get("tasks", []):
            if not isinstance(raw_task, Mapping):
                raise ModelContractError("ferebus_manifest_task_invalid")
            for key in (
                "property_dir",
                "output_dir",
                "input_dir",
                "config_path",
                "training_csv",
                "int_validation_csv",
                "ext_validation_csv",
                "expected_model_path",
            ):
                if key in raw_task:
                    _resolve_under(root, Path(str(raw_task[key])), "ferebus_" + key)
    committed_tasks = (
        {} if model_set is None else {task.key: task for task in model_set.tasks}
    )
    expected_models = set()
    model_paths: List[Tuple[FerebusTask, Path, Path]] = []
    raw_tasks_by_key = {
        (str(item.get("property")), str(item.get("atom"))): item
        for item in manifest.get("tasks", [])
        if isinstance(item, Mapping)
    }
    for task in tasks:
        raw_task = raw_tasks_by_key.get(task.key, {})
        if committed:
            committed_task = committed_tasks.get(task.key)
            if committed_task is None:
                raise ModelContractError(
                    "committed_model_task_missing: "
                    + task.property
                    + "/"
                    + task.atom
                )
            config_path = committed_task.config.path
            model_path = committed_task.model.path
            training_path = committed_task.datasets["train"].path
        else:
            config_path = _config_path_for(root, task, committed=False)
            model_path = _model_path_for(root, task, committed=False)
            config_path = _resolve_under(root, config_path, "ferebus_config")
            model_path = _resolve_under(root, model_path, "ferebus_model")
            training_path = _resolve_under(
                root,
                Path(str(raw_task.get("training_csv") or "")),
                "ferebus_training_csv",
            )
        expected_models.add(model_path.resolve())
        if not config_path.is_file():
            raise ModelContractError("ferebus_config_missing: " + str(config_path))
        try:
            validate_ferebus_config_contract(config_path, prior_contract)
        except Exception as exc:
            raise ModelContractError(
                "ferebus_config_prior_contract_invalid:"
                + task.property
                + "/"
                + task.atom
                + ":"
                + str(exc)
            ) from exc
        generated = raw_tasks_by_key.get(task.key, {}).get("generated_config")
        if not isinstance(generated, Mapping):
            raise ModelContractError(
                "ferebus_generated_config_binding_missing:"
                + task.property
                + "/"
                + task.atom
            )
        if int(generated.get("size", -1)) != int(config_path.stat().st_size):
            raise ModelContractError("ferebus_generated_config_size_mismatch")
        if str(generated.get("sha256") or "") != sha256_file(config_path):
            raise ModelContractError("ferebus_generated_config_sha_mismatch")
        if not model_path.is_file():
            raise ModelContractError("expected_model_missing: " + str(model_path))
        if model_path.stat().st_size <= 0:
            raise ModelContractError("model_file_empty")
        model_paths.append((task, model_path, training_path))

    parsed_models: Dict[Tuple[str, str], Any] = {}
    for task, model_path, training_path in model_paths:
        raw_task = raw_tasks_by_key.get(task.key, {})
        declared_ntrain = _declared_ntrain(model_path)
        if declared_ntrain is not None:
            _check_section_rows(model_path, declared_ntrain)
        try:
            model = Model(model_path)
            _validate_model_object(
                model,
                model_path,
                task,
                system,
                kernel_family=kernel_family,
            )
            from .ferebus_dataset import iter_feature_target_chunks

            model_x = np.asarray(model.x, dtype=float)
            training_values = np.asarray(model.y, dtype=float).reshape(-1)
            offset = 0
            for training_features, targets in iter_feature_target_chunks(
                training_path,
                task.property,
            ):
                stop = offset + int(targets.shape[0])
                if (
                    stop > model_x.shape[0]
                    or training_features.shape != model_x[offset:stop].shape
                    or not np.allclose(
                        training_features,
                        model_x[offset:stop],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or not np.allclose(
                        targets,
                        training_values[offset:stop],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ):
                    raise ModelContractError("model_training_data_binding_mismatch")
                offset = stop
            if offset != model_x.shape[0] or offset != training_values.shape[0]:
                raise ModelContractError("model_training_data_binding_mismatch")
            validate_model_prior_mean(
                model,
                contract=prior_contract,
                property_name=task.property,
                atom=task.atom,
                training_values=training_values,
                expected_mean_ha=(
                    (raw_task.get("prior_mean") or {}).get(
                        "expected_mean_ha"
                    )
                    if committed
                    else None
                ),
            )
            parsed_models[task.key] = model
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

    model_bootstrap = manifest.get("model_bootstrap")
    if isinstance(model_bootstrap, Mapping):
        from ..strict_json import strict_json as json

        from ..versioning.manifest import sha256_file

        copied_manifest = root / "MODEL_BOOTSTRAP.json"
        if copied_manifest.is_symlink() or not copied_manifest.is_file():
            raise ModelContractError("model_bootstrap_manifest_missing")
        if sha256_file(copied_manifest) != str(
            model_bootstrap.get("manifest_sha256") or ""
        ):
            raise ModelContractError("model_bootstrap_manifest_sha_mismatch")
        try:
            bootstrap_payload = json.loads(
                copied_manifest.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ModelContractError(
                "model_bootstrap_manifest_unreadable"
            ) from exc
        if not isinstance(bootstrap_payload, Mapping):
            raise ModelContractError("model_bootstrap_manifest_invalid")
        historical_rows = int(
            model_bootstrap.get("historical_training_rows", -1)
        )
        records = bootstrap_payload.get("files")
        if historical_rows <= 0 or not isinstance(records, list):
            raise ModelContractError("model_bootstrap_manifest_invalid")
        campaign = root.parent.parent
        immutable_root = (
            campaign / ".DATA" / "ACTIVE_LEARNING" / "bootstrap_inputs"
        )
        baseline_by_key: Dict[Tuple[str, str], Any] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise ModelContractError("model_bootstrap_file_record_invalid")
            key = (str(record.get("property")), str(record.get("atom")))
            source = immutable_root / str(record.get("path") or "")
            try:
                source.resolve(strict=False).relative_to(immutable_root.resolve())
            except ValueError as exc:
                raise ModelContractError("model_bootstrap_path_escapes_inputs") from exc
            if source.is_symlink() or not source.is_file():
                raise ModelContractError("model_bootstrap_source_missing:" + repr(key))
            if sha256_file(source) != str(record.get("sha256") or ""):
                raise ModelContractError("model_bootstrap_source_sha_mismatch:" + repr(key))
            if key in baseline_by_key:
                raise ModelContractError("model_bootstrap_duplicate_task:" + repr(key))
            baseline_by_key[key] = Model(source)
        if set(baseline_by_key) != set(parsed_models):
            raise ModelContractError("model_bootstrap_task_coverage_mismatch")
        for key, trained in parsed_models.items():
            baseline = baseline_by_key[key]
            baseline_x = np.asarray(baseline.x, dtype=float)
            baseline_y = np.asarray(baseline.y, dtype=float).reshape(-1)
            trained_x = np.asarray(trained.x, dtype=float)
            trained_y = np.asarray(trained.y, dtype=float).reshape(-1)
            if baseline_x.shape[0] != historical_rows or baseline_y.size != historical_rows:
                raise ModelContractError("model_bootstrap_row_count_mismatch:" + repr(key))
            if trained_x.shape[0] < historical_rows or trained_y.size < historical_rows:
                raise ModelContractError("model_bootstrap_prefix_missing:" + repr(key))
            if not np.allclose(
                trained_x[:historical_rows],
                baseline_x,
                rtol=1.0e-12,
                atol=1.0e-12,
            ) or not np.allclose(
                trained_y[:historical_rows],
                baseline_y,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ModelContractError("model_bootstrap_prefix_changed:" + repr(key))

    actual_models = {p.resolve() for p in root.rglob("*.model")}
    extra_models = actual_models - expected_models
    if extra_models:
        raise ModelContractError(
            "unexpected_model_file: " + str(sorted(str(p) for p in extra_models)[0])
        )

    expected_keys = {task.key for task in tasks}
    if committed:
        try:
            models = Models.from_model_files(root, model_set.model_paths)
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
    from ..versioning.trained_models import load_trained_models

    root = Path(models_dir)
    try:
        version = _version_from_iteration_dir(root)
        if version is None:
            raise ModelContractError("posterior_model_version_unparseable")
        model_set, models = load_trained_models(
            root.parent.parent,
            version,
            verification="deep",
        )
        if model_set.root != root.resolve():
            raise ModelContractError("posterior_model_root_mismatch")
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
    for key, raw in scales.items():
        key_s = str(key)
        if key_s in out:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ModelContractError("reference_scale_invalid:" + key_s) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ModelContractError("reference_scale_non_finite_or_non_positive:" + key_s)
        out[key_s] = value
    return out
