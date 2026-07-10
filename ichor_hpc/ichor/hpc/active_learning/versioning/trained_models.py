"""Authoritative immutable trained-model snapshots."""

from __future__ import annotations

import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from ..layout import trained_models_dir
from .manifest import (
    MANIFEST_FILENAME,
    read_manifest,
    sha256_file,
    verify_manifest,
)
from .reference_data import (
    ReferenceDataVersioning,
    canonical_json_sha256,
)
from .versioned_directory import VersionedDirectory


TRAINED_MODEL_SET_FILENAME = "FEREBUS_TASK_ARTEFACTS.json"
TRAINED_MODEL_SET_SCHEMA_VERSION = 2
TRAINED_MODEL_STORAGE_MODE = "full_snapshot"
TRAINED_MODELS_COMMIT_LOCK = ".commit.lock"
TRAINED_MODEL_AUXILIARY_SUFFIXES = ("opt", "perf", "pred", "scurve", "sol")
SAFE_PATH_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class TrainedModelError(RuntimeError):
    """Raised when a committed trained-model set is not trustworthy."""


@dataclass(frozen=True)
class TrainedModelFile:
    relative_path: str
    path: Path
    size: int
    sha256: str

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "path": self.relative_path,
            "size": int(self.size),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TrainedModelTask:
    task_index: int
    property: str
    atom: str
    alf_1_indexed: Tuple[int, int, int]
    directory: str
    model: TrainedModelFile
    config: TrainedModelFile
    auxiliary: Mapping[str, Optional[TrainedModelFile]]

    @property
    def key(self) -> Tuple[str, str]:
        return (self.property, self.atom)

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "task_index": int(self.task_index),
            "property": self.property,
            "atom": self.atom,
            "alf_1_indexed": [int(value) for value in self.alf_1_indexed],
            "directory": self.directory,
            "model": self.model.identity_payload(),
            "config": self.config.identity_payload(),
            "auxiliary": {
                suffix: (
                    None if self.auxiliary[suffix] is None
                    else self.auxiliary[suffix].identity_payload()
                )
                for suffix in TRAINED_MODEL_AUXILIARY_SUFFIXES
            },
        }


@dataclass(frozen=True)
class TrainedModelSet:
    version: int
    campaign_uid: str
    system: str
    reference_data_version: int
    reference_data_head_manifest_sha256: str
    reference_data_view_sha256: str
    parent_version: Optional[int]
    parent_manifest_sha256: Optional[str]
    properties: Tuple[str, ...]
    atoms: Tuple[str, ...]
    tasks: Tuple[TrainedModelTask, ...]
    root_files: Tuple[TrainedModelFile, ...]
    model_set_sha256: str
    head_manifest_sha256: str
    root: Path

    @property
    def model_paths(self) -> Tuple[Path, ...]:
        return tuple(task.model.path for task in self.tasks)


class TrainedModelVersioning(VersionedDirectory):
    def update_current(self, target_version: int) -> None:
        committed = self.list_committed_versions()
        if not committed:
            raise FileNotFoundError("no committed trained-model versions exist")
        newest = max(committed)
        if int(target_version) != newest:
            raise TrainedModelError(
                "trained-model current pointer may only target newest version "
                + str(newest)
            )
        super().update_current(int(target_version))

    def resolve(
        self,
        version: int,
        *,
        verification: str = "deep",
    ) -> TrainedModelSet:
        return resolve_trained_model_set(
            Path(self.parent).parent,
            int(version),
            verification=verification,
            trained_models_root=self.parent,
        )


def trained_model_set_path(version_dir: Union[str, Path]) -> Path:
    return Path(version_dir) / TRAINED_MODEL_SET_FILENAME


def trained_models_commit_lock_path(campaign_dir: Union[str, Path]) -> Path:
    return trained_models_dir(campaign_dir) / TRAINED_MODELS_COMMIT_LOCK


@contextmanager
def trained_models_commit_lock(campaign_dir: Union[str, Path]) -> Iterator[None]:
    import portalocker

    path = trained_models_commit_lock_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(path), mode="a", timeout=0):
        yield


def _safe_int(value: Any, label: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise TrainedModelError(label + " must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise TrainedModelError(label + " must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainedModelError(label + " must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise TrainedModelError(label + " must be >= " + str(minimum))
    return parsed


def _safe_sha(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise TrainedModelError(label + " must be a lowercase SHA-256")
    return text


def _safe_token(value: Any, label: str) -> str:
    text = str(value or "")
    if not SAFE_PATH_TOKEN_RE.fullmatch(text):
        raise TrainedModelError(label + " is not a safe path token: " + repr(text))
    return text


def _safe_token_sequence(value: Any, label: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TrainedModelError(label + " must be a non-empty list")
    tokens = tuple(_safe_token(item, label) for item in value)
    folded = [token.casefold() for token in tokens]
    if len(folded) != len(set(folded)):
        raise TrainedModelError(label + " contains a case-insensitive collision")
    return tokens


def _safe_relative_path(value: Any, label: str) -> str:
    text = str(value or "")
    if "\\" in text:
        raise TrainedModelError(label + " must use POSIX separators")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TrainedModelError(label + " is not a safe relative path: " + repr(text))
    return path.as_posix()


def file_record(path: Union[str, Path], root: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(path)
    root_path = Path(root)
    if not file_path.is_file() or file_path.is_symlink():
        raise TrainedModelError("trained-model artefact is not a regular file: " + str(file_path))
    try:
        relative = file_path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError as exc:
        raise TrainedModelError("trained-model artefact escapes its root: " + str(file_path)) from exc
    return {
        "path": relative,
        "size": int(file_path.stat().st_size),
        "sha256": sha256_file(file_path),
    }


def _parse_file_record(
    root: Path,
    payload: Any,
    label: str,
    *,
    verification: str,
) -> TrainedModelFile:
    if not isinstance(payload, Mapping):
        raise TrainedModelError(label + " must be an object")
    relative = _safe_relative_path(payload.get("path"), label + ".path")
    expected_size = _safe_int(payload.get("size"), label + ".size", minimum=0)
    expected_sha = _safe_sha(payload.get("sha256"), label + ".sha256")
    path = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise TrainedModelError(label + " contains a symlink: " + str(current))
    if not path.is_file() or path.is_symlink():
        raise TrainedModelError(label + " is missing: " + str(path))
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise TrainedModelError(label + " escapes the model version") from exc
    if int(path.stat().st_size) != expected_size:
        raise TrainedModelError(label + " size mismatch: " + str(path))
    if verification == "deep" and sha256_file(path) != expected_sha:
        raise TrainedModelError(label + " SHA-256 mismatch: " + str(path))
    return TrainedModelFile(
        relative_path=relative,
        path=path.resolve(),
        size=expected_size,
        sha256=expected_sha,
    )


def _parse_task(
    root: Path,
    payload: Any,
    *,
    verification: str,
) -> TrainedModelTask:
    if not isinstance(payload, Mapping):
        raise TrainedModelError("trained-model task must be an object")
    task_index = _safe_int(payload.get("task_index"), "task_index", minimum=1)
    prop = _safe_token(payload.get("property"), "property")
    atom = _safe_token(payload.get("atom"), "atom")
    directory = _safe_relative_path(payload.get("directory"), "task.directory")
    expected_directory = PurePosixPath(prop, atom).as_posix()
    if directory != expected_directory:
        raise TrainedModelError(
            "task directory mismatch for " + prop + "/" + atom
        )
    raw_alf = payload.get("alf_1_indexed")
    if not isinstance(raw_alf, list) or len(raw_alf) != 3:
        raise TrainedModelError("task ALF must contain three indexes")
    alf = tuple(_safe_int(value, "task ALF", minimum=1) for value in raw_alf)
    model = _parse_file_record(
        root,
        payload.get("model"),
        prop + "/" + atom + ":model",
        verification=verification,
    )
    config = _parse_file_record(
        root,
        payload.get("config"),
        prop + "/" + atom + ":config",
        verification=verification,
    )
    task_root = PurePosixPath(directory)
    if PurePosixPath(model.relative_path).parent != task_root or not model.relative_path.endswith(".model"):
        raise TrainedModelError(prop + "/" + atom + ":model path is not canonical")
    expected_config_name = "ferebus_" + prop + "_" + atom + ".config"
    if PurePosixPath(config.relative_path) != task_root / expected_config_name:
        raise TrainedModelError(prop + "/" + atom + ":config path is not canonical")
    raw_auxiliary = payload.get("auxiliary")
    if not isinstance(raw_auxiliary, Mapping):
        raise TrainedModelError(prop + "/" + atom + ":auxiliary must be an object")
    if set(raw_auxiliary) != set(TRAINED_MODEL_AUXILIARY_SUFFIXES):
        raise TrainedModelError(prop + "/" + atom + ":auxiliary keys are invalid")
    auxiliary: Dict[str, Optional[TrainedModelFile]] = {}
    for suffix in TRAINED_MODEL_AUXILIARY_SUFFIXES:
        record = raw_auxiliary.get(suffix)
        if record is None:
            auxiliary[suffix] = None
            continue
        parsed = _parse_file_record(
            root,
            record,
            prop + "/" + atom + ":" + suffix,
            verification=verification,
        )
        expected_auxiliary_name = Path(model.relative_path).stem + "." + suffix
        if (
            PurePosixPath(parsed.relative_path).parent != task_root
            or PurePosixPath(parsed.relative_path).name != expected_auxiliary_name
        ):
            raise TrainedModelError(prop + "/" + atom + ":" + suffix + " path is not canonical")
        auxiliary[suffix] = parsed
    return TrainedModelTask(
        task_index=task_index,
        property=prop,
        atom=atom,
        alf_1_indexed=alf,
        directory=directory,
        model=model,
        config=config,
        auxiliary=auxiliary,
    )


def _validate_task_file_names(
    tasks: Sequence[TrainedModelTask],
    system: str,
) -> None:
    for task in tasks:
        expected_model = PurePosixPath(
            task.directory,
            system + "_" + task.property + "_" + task.atom + ".model",
        ).as_posix()
        if task.model.relative_path != expected_model:
            raise TrainedModelError(
                task.property + "/" + task.atom + ":model filename is not canonical"
            )


def _validate_root_file_records(root_files: Sequence[TrainedModelFile]) -> None:
    paths = [record.relative_path for record in root_files]
    if any("/" in path for path in paths):
        raise TrainedModelError("trained-model root file is not at the version root")
    if paths != sorted(paths):
        raise TrainedModelError("trained-model root files are not deterministically ordered")
    if len({path.casefold() for path in paths}) != len(paths):
        raise TrainedModelError("trained-model root files contain a case collision")


def _model_set_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "campaign_uid": payload.get("campaign_uid"),
        "system": payload.get("system"),
        "models_version": payload.get("models_version"),
        "reference_data_version": payload.get("reference_data_version"),
        "reference_data_head_manifest_sha256": payload.get(
            "reference_data_head_manifest_sha256"
        ),
        "reference_data_view_sha256": payload.get("reference_data_view_sha256"),
        "parent_version": payload.get("parent_version"),
        "parent_manifest_sha256": payload.get("parent_manifest_sha256"),
        "source_task_manifest": payload.get("source_task_manifest"),
        "quality_manifest": payload.get("quality_manifest"),
        "properties": payload.get("properties"),
        "atoms": payload.get("atoms"),
        "tasks": payload.get("tasks"),
    }


def build_trained_model_set_payload(
    *,
    campaign_uid: str,
    version: int,
    system: str,
    reference_data_head_manifest_sha256: str,
    reference_data_view_sha256: str,
    parent: Optional[TrainedModelSet],
    source_task_manifest: Mapping[str, Any],
    quality_manifest: Mapping[str, Any],
    properties: Sequence[str],
    atoms: Sequence[str],
    tasks: Sequence[Mapping[str, Any]],
    root_files: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    campaign_uid_text = str(campaign_uid or "")
    if not campaign_uid_text:
        raise TrainedModelError("campaign_uid is empty")
    system_token = _safe_token(system, "system")
    payload: Dict[str, Any] = {
        "schema_version": TRAINED_MODEL_SET_SCHEMA_VERSION,
        "storage_mode": TRAINED_MODEL_STORAGE_MODE,
        "campaign_uid": campaign_uid_text,
        "system": system_token,
        "models_version": int(version),
        "reference_data_version": int(version),
        "reference_data_head_manifest_sha256": _safe_sha(
            reference_data_head_manifest_sha256,
            "reference_data_head_manifest_sha256",
        ),
        "reference_data_view_sha256": _safe_sha(
            reference_data_view_sha256,
            "reference_data_view_sha256",
        ),
        "parent_version": None if parent is None else int(parent.version),
        "parent_manifest_sha256": (
            None if parent is None else str(parent.head_manifest_sha256)
        ),
        "source_task_manifest": dict(source_task_manifest),
        "quality_manifest": dict(quality_manifest),
        "properties": [str(value) for value in properties],
        "atoms": [str(value) for value in atoms],
        "n_tasks": int(len(tasks)),
        "tasks": [dict(value) for value in tasks],
        "root_files": [dict(value) for value in root_files],
    }
    payload["model_set_sha256"] = canonical_json_sha256(_model_set_identity(payload))
    return payload


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    def reject_non_finite_constant(value: str) -> None:
        raise ValueError("non-finite JSON constant: " + value)

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_non_finite_constant,
        )
    except (OSError, ValueError) as exc:
        raise TrainedModelError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise TrainedModelError(label + " must be a JSON object: " + str(path))
    return payload


def _validate_exact_inventory(
    root: Path,
    tasks: Sequence[TrainedModelTask],
    root_files: Sequence[TrainedModelFile],
    *,
    require_directory_manifest: bool = True,
) -> None:
    expected_task_files = set()
    expected_directories = set()
    for task in tasks:
        expected_directories.add(task.property)
        expected_directories.add(task.directory)
        expected_task_files.add(task.model.relative_path)
        expected_task_files.add(task.config.relative_path)
        expected_task_files.update(
            artefact.relative_path
            for artefact in task.auxiliary.values()
            if artefact is not None
        )
    actual_directories = set()
    actual_task_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise TrainedModelError("trained-model snapshot contains a symlink: " + str(path))
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file() and len(path.relative_to(root).parts) > 1:
            actual_task_files.add(relative)
    if actual_directories != expected_directories:
        unexpected = sorted(actual_directories - expected_directories)
        missing = sorted(expected_directories - actual_directories)
        raise TrainedModelError(
            "trained-model task directory inventory mismatch: unexpected="
            + repr(unexpected)
            + " missing="
            + repr(missing)
        )
    if actual_task_files != expected_task_files:
        unexpected = sorted(actual_task_files - expected_task_files)
        missing = sorted(expected_task_files - actual_task_files)
        raise TrainedModelError(
            "trained-model task file inventory mismatch: unexpected="
            + repr(unexpected)
            + " missing="
            + repr(missing)
        )
    expected_root_files = {artefact.relative_path for artefact in root_files}
    expected_root_files.add(TRAINED_MODEL_SET_FILENAME)
    if require_directory_manifest:
        expected_root_files.add(MANIFEST_FILENAME)
    actual_root_files = {
        path.name
        for path in root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_root_files != expected_root_files:
        raise TrainedModelError(
            "trained-model root file inventory mismatch: unexpected="
            + repr(sorted(actual_root_files - expected_root_files))
            + " missing="
            + repr(sorted(expected_root_files - actual_root_files))
        )


def _validate_source_and_quality(
    root: Path,
    payload: Mapping[str, Any],
    tasks: Sequence[TrainedModelTask],
    reference_view: Any,
) -> None:
    source = _read_json_object(root / "FEREBUS_TASKS.json", "FEREBUS task manifest")
    quality = _read_json_object(root / "FEREBUS_QUALITY.json", "FEREBUS quality manifest")
    if _safe_int(source.get("schema_version"), "FEREBUS task schema_version") != 3:
        raise TrainedModelError("unsupported committed FEREBUS task schema")
    if _safe_int(quality.get("schema_version"), "FEREBUS quality schema_version") != 3:
        raise TrainedModelError("unsupported committed FEREBUS quality schema")
    expected_keys = [task.key for task in tasks]
    source_tasks = source.get("tasks")
    if not isinstance(source_tasks, list):
        raise TrainedModelError("FEREBUS source tasks are invalid")
    source_keys = [
        (str(task.get("property")), str(task.get("atom")))
        for task in source_tasks
        if isinstance(task, Mapping)
    ]
    if source_keys != expected_keys:
        raise TrainedModelError("FEREBUS task manifest/model-set task order mismatch")
    if str(source.get("campaign_uid") or "") != str(payload.get("campaign_uid") or ""):
        raise TrainedModelError("FEREBUS task manifest campaign UID mismatch")
    if str(quality.get("campaign_uid") or "") != str(payload.get("campaign_uid") or ""):
        raise TrainedModelError("FEREBUS quality manifest campaign UID mismatch")
    if str(source.get("system") or "") != str(payload.get("system") or ""):
        raise TrainedModelError("FEREBUS task manifest system mismatch")
    if str(quality.get("system") or "") != str(payload.get("system") or ""):
        raise TrainedModelError("FEREBUS quality manifest system mismatch")
    expected_row_order = [entry.pointdir_name for entry in reference_view.entries]
    if list(source.get("pointdir_row_order") or []) != expected_row_order:
        raise TrainedModelError("FEREBUS task manifest reference-data row order mismatch")
    if _safe_int(
        source.get("n_reference_points"),
        "FEREBUS n_reference_points",
        minimum=1,
    ) != len(expected_row_order):
        raise TrainedModelError("FEREBUS task manifest reference-data count mismatch")
    expected_split_rows = {
        split: [
            index
            for index, entry in enumerate(reference_view.entries)
            if entry.split == split
        ]
        for split in ("train", "int_val", "ext_val")
    }
    dataset_path_fields = {
        "train": "training_csv",
        "int_val": "int_validation_csv",
        "ext_val": "ext_validation_csv",
    }
    for task in source_tasks:
        if not isinstance(task, Mapping):
            raise TrainedModelError("FEREBUS source task record is invalid")
        row_counts = task.get("row_counts") or {}
        if not isinstance(row_counts, Mapping):
            raise TrainedModelError("FEREBUS task manifest split counts are invalid")
        observed_counts = {
            split: _safe_int(
                row_counts.get(split),
                "FEREBUS " + split + " row count",
                minimum=0,
            )
            for split in expected_split_rows
        }
        expected_counts = {
            split: len(rows) for split, rows in expected_split_rows.items()
        }
        if observed_counts != expected_counts:
            raise TrainedModelError("FEREBUS task manifest split count mismatch")
        row_ids = task.get("row_ids") or {}
        if not isinstance(row_ids, Mapping):
            raise TrainedModelError("FEREBUS task manifest split rows are invalid")
        for split in expected_split_rows:
            if not isinstance(row_ids.get(split), list):
                raise TrainedModelError(
                    "FEREBUS task manifest " + split + " rows are invalid"
                )
        observed_rows = {
            split: [
                _safe_int(
                    value,
                    "FEREBUS " + split + " row index",
                    minimum=0,
                )
                for value in row_ids[split]
            ]
            for split in expected_split_rows
        }
        if observed_rows != expected_split_rows:
            raise TrainedModelError("FEREBUS task manifest split row mismatch")
        datasets = task.get("datasets")
        if not isinstance(datasets, Mapping) or set(datasets) != set(
            dataset_path_fields
        ):
            raise TrainedModelError("FEREBUS task dataset identities are invalid")
        for split, path_field in dataset_path_fields.items():
            record = datasets.get(split)
            if not isinstance(record, Mapping):
                raise TrainedModelError(
                    "FEREBUS " + split + " dataset identity is invalid"
                )
            if _safe_relative_path(
                record.get("path"),
                "FEREBUS " + split + " dataset path",
            ) != str(task.get(path_field) or ""):
                raise TrainedModelError(
                    "FEREBUS " + split + " dataset path mismatch"
                )
            if _safe_int(
                record.get("rows"),
                "FEREBUS " + split + " dataset rows",
                minimum=0,
            ) != expected_counts[split]:
                raise TrainedModelError(
                    "FEREBUS " + split + " dataset row count mismatch"
                )
            _safe_int(
                record.get("size"),
                "FEREBUS " + split + " dataset size",
                minimum=0,
            )
            _safe_sha(
                record.get("sha256"),
                "FEREBUS " + split + " dataset SHA-256",
            )
    quality_records = quality.get("records")
    if not isinstance(quality_records, list):
        raise TrainedModelError("FEREBUS quality records are invalid")
    quality_keys = [
        (str(record.get("property")), str(record.get("atom")))
        for record in quality_records
        if isinstance(record, Mapping)
    ]
    if quality_keys != expected_keys:
        raise TrainedModelError("FEREBUS quality/model-set task coverage mismatch")
    quality_by_key = dict(zip(quality_keys, quality_records))
    if quality.get("accepted") is not True:
        raise TrainedModelError("FEREBUS quality manifest is rejected")
    if _safe_sha(
        quality.get("source_task_manifest_sha256"),
        "source_task_manifest_sha256",
    ) != sha256_file(
        root / "FEREBUS_TASKS.json"
    ):
        raise TrainedModelError("FEREBUS quality/source task manifest SHA mismatch")
    for task in tasks:
        record = quality_by_key[task.key]
        if not isinstance(record, Mapping):
            raise TrainedModelError(
                task.property + "/" + task.atom + ":quality record is invalid"
            )
        if record.get("accepted") is not True:
            raise TrainedModelError(
                task.property + "/" + task.atom + ":quality record is rejected"
            )
        if str(record.get("model_path") or "") != task.model.relative_path:
            raise TrainedModelError(
                task.property + "/" + task.atom + ":quality model path mismatch"
            )
        if _safe_sha(
            record.get("model_sha256"),
            task.property + "/" + task.atom + ":quality model SHA",
        ) != task.model.sha256:
            raise TrainedModelError(
                task.property + "/" + task.atom + ":quality model SHA mismatch"
            )
        expected_counts = {
            split: len(rows) for split, rows in expected_split_rows.items()
        }
        try:
            quality_counts = {
                split: _safe_int(
                    (record.get("row_counts") or {})[split],
                    task.property + "/" + task.atom + ":quality " + split + " count",
                    minimum=0,
                )
                for split in expected_split_rows
            }
        except (KeyError, TypeError, ValueError, TrainedModelError) as exc:
            raise TrainedModelError(
                task.property + "/" + task.atom + ":quality row counts are invalid"
            ) from exc
        if quality_counts != expected_counts:
            raise TrainedModelError(
                task.property + "/" + task.atom + ":quality row count mismatch"
            )
    for field in (
        "reference_data_version",
        "reference_data_head_manifest_sha256",
        "reference_data_view_sha256",
    ):
        if str(source.get(field)) != str(payload.get(field)):
            raise TrainedModelError("FEREBUS task manifest " + field + " mismatch")
        if str(quality.get(field)) != str(payload.get(field)):
            raise TrainedModelError("FEREBUS quality manifest " + field + " mismatch")


def validate_trained_model_snapshot(
    campaign_dir: Union[str, Path],
    version_root: Union[str, Path],
    version: int,
    *,
    parent: Optional[TrainedModelSet],
    verification: str = "deep",
) -> TrainedModelSet:
    """Validate one unpublished full snapshot before its atomic rename."""
    if verification not in {"metadata", "deep"}:
        raise ValueError("trained-model verification must be metadata or deep")
    campaign = Path(campaign_dir)
    root = Path(version_root)
    expected_version = _safe_int(version, "models_version", minimum=0)
    payload = _read_json_object(
        trained_model_set_path(root),
        "trained-model set manifest",
    )
    if _safe_int(payload.get("schema_version"), "schema_version") != TRAINED_MODEL_SET_SCHEMA_VERSION:
        raise TrainedModelError("unsupported trained-model set schema")
    if str(payload.get("storage_mode")) != TRAINED_MODEL_STORAGE_MODE:
        raise TrainedModelError("trained-model storage_mode must be full_snapshot")
    if _safe_int(payload.get("models_version"), "models_version") != expected_version:
        raise TrainedModelError("trained-model version mismatch")
    campaign_uid = str(payload.get("campaign_uid") or "")
    if not campaign_uid:
        raise TrainedModelError("trained-model campaign UID is empty")
    system = _safe_token(payload.get("system"), "system")
    if parent is None:
        if expected_version != 0:
            raise TrainedModelError("non-bootstrap model snapshot has no parent")
        if payload.get("parent_version") is not None or payload.get("parent_manifest_sha256") is not None:
            raise TrainedModelError("bootstrap model snapshot must not have a parent")
    else:
        if expected_version != parent.version + 1:
            raise TrainedModelError("trained-model parent version is not contiguous")
        if campaign_uid != parent.campaign_uid:
            raise TrainedModelError("trained-model parent campaign UID mismatch")
        if _safe_int(payload.get("parent_version"), "parent_version") != parent.version:
            raise TrainedModelError("trained-model parent version mismatch")
        if _safe_sha(
            payload.get("parent_manifest_sha256"),
            "parent_manifest_sha256",
        ) != parent.head_manifest_sha256:
            raise TrainedModelError("trained-model parent manifest SHA mismatch")
    if _safe_int(payload.get("reference_data_version"), "reference_data_version") != expected_version:
        raise TrainedModelError("model/reference-data version mismatch")
    reference_view = ReferenceDataVersioning(
        campaign / "QM_REFERENCE_DATA"
    ).resolve(expected_version, verification=verification)
    if campaign_uid != reference_view.campaign_uid:
        raise TrainedModelError("model/reference-data campaign UID mismatch")
    if str(payload.get("reference_data_head_manifest_sha256")) != reference_view.head_manifest_sha256:
        raise TrainedModelError("model/reference-data head manifest SHA mismatch")
    if str(payload.get("reference_data_view_sha256")) != reference_view.cumulative_view_sha256:
        raise TrainedModelError("model/reference-data view SHA mismatch")
    properties = _safe_token_sequence(payload.get("properties"), "properties")
    atoms = _safe_token_sequence(payload.get("atoms"), "atoms")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise TrainedModelError("trained-model tasks must be a list")
    tasks = tuple(
        _parse_task(root, task, verification=verification) for task in raw_tasks
    )
    expected_keys = [(prop, atom) for prop in properties for atom in atoms]
    if [task.key for task in tasks] != expected_keys:
        raise TrainedModelError("trained-model tasks do not match property/atom product")
    if [task.task_index for task in tasks] != list(range(1, len(tasks) + 1)):
        raise TrainedModelError("trained-model task indexes are not contiguous")
    _validate_task_file_names(tasks, system)
    if _safe_int(payload.get("n_tasks"), "n_tasks", minimum=1) != len(tasks):
        raise TrainedModelError("trained-model task count mismatch")
    raw_root_files = payload.get("root_files")
    if not isinstance(raw_root_files, list):
        raise TrainedModelError("trained-model root_files must be a list")
    root_files = tuple(
        _parse_file_record(root, record, "root_file", verification=verification)
        for record in raw_root_files
    )
    _validate_root_file_records(root_files)
    source_record = _parse_file_record(
        root,
        payload.get("source_task_manifest"),
        "source_task_manifest",
        verification=verification,
    )
    quality_record = _parse_file_record(
        root,
        payload.get("quality_manifest"),
        "quality_manifest",
        verification=verification,
    )
    root_file_by_path = {record.relative_path: record for record in root_files}
    if source_record.relative_path != "FEREBUS_TASKS.json" or root_file_by_path.get(
        source_record.relative_path
    ) != source_record:
        raise TrainedModelError("source task manifest root-file record mismatch")
    if quality_record.relative_path != "FEREBUS_QUALITY.json" or root_file_by_path.get(
        quality_record.relative_path
    ) != quality_record:
        raise TrainedModelError("quality manifest root-file record mismatch")
    if _safe_sha(payload.get("model_set_sha256"), "model_set_sha256") != canonical_json_sha256(
        _model_set_identity(payload)
    ):
        raise TrainedModelError("trained-model set SHA mismatch")
    _validate_exact_inventory(
        root,
        tasks,
        root_files,
        require_directory_manifest=False,
    )
    _validate_source_and_quality(root, payload, tasks, reference_view)
    return TrainedModelSet(
        version=expected_version,
        campaign_uid=campaign_uid,
        system=system,
        reference_data_version=expected_version,
        reference_data_head_manifest_sha256=reference_view.head_manifest_sha256,
        reference_data_view_sha256=reference_view.cumulative_view_sha256,
        parent_version=None if parent is None else parent.version,
        parent_manifest_sha256=(
            None if parent is None else parent.head_manifest_sha256
        ),
        properties=properties,
        atoms=atoms,
        tasks=tasks,
        root_files=root_files,
        model_set_sha256=str(payload["model_set_sha256"]),
        head_manifest_sha256=sha256_file(trained_model_set_path(root)),
        root=root.resolve(),
    )


def resolve_trained_model_set(
    campaign_dir: Union[str, Path],
    models_version: int,
    *,
    verification: str = "deep",
    trained_models_root: Optional[Union[str, Path]] = None,
) -> TrainedModelSet:
    if verification not in {"metadata", "deep"}:
        raise ValueError("trained-model verification must be metadata or deep")
    campaign = Path(campaign_dir)
    root = (
        Path(trained_models_root)
        if trained_models_root is not None
        else trained_models_dir(campaign)
    )
    target_version = _safe_int(models_version, "models_version", minimum=0)
    versioning = TrainedModelVersioning(root)
    committed = versioning.list_committed_versions()
    expected_versions = list(range(target_version + 1))
    if [version for version in committed if version <= target_version] != expected_versions:
        raise TrainedModelError(
            "trained-model versions are not contiguous through " + str(target_version)
        )
    previous_manifest_sha: Optional[str] = None
    campaign_uid: Optional[str] = None
    resolved: Optional[TrainedModelSet] = None
    for version in expected_versions:
        version_root = versioning.iteration_path(version)
        directory_manifest = read_manifest(version_root)
        if verification == "deep" and version == target_version:
            verify_manifest(version_root, manifest=directory_manifest)
        manifest_path = trained_model_set_path(version_root)
        payload = _read_json_object(manifest_path, "trained-model set manifest")
        if _safe_int(payload.get("schema_version"), "schema_version") != TRAINED_MODEL_SET_SCHEMA_VERSION:
            raise TrainedModelError("unsupported trained-model set schema")
        if str(payload.get("storage_mode")) != TRAINED_MODEL_STORAGE_MODE:
            raise TrainedModelError("trained-model storage_mode must be full_snapshot")
        if _safe_int(payload.get("models_version"), "models_version") != version:
            raise TrainedModelError("trained-model version mismatch")
        uid = str(payload.get("campaign_uid") or "")
        if not uid or (campaign_uid is not None and uid != campaign_uid):
            raise TrainedModelError("trained-model campaign UID mismatch")
        campaign_uid = uid
        system = _safe_token(payload.get("system"), "system")
        parent_version = payload.get("parent_version")
        parent_sha = payload.get("parent_manifest_sha256")
        if version == 0:
            if parent_version is not None or parent_sha is not None:
                raise TrainedModelError("bootstrap model set must not have a parent")
        else:
            if _safe_int(parent_version, "parent_version") != version - 1:
                raise TrainedModelError("trained-model parent version mismatch")
            if _safe_sha(parent_sha, "parent_manifest_sha256") != previous_manifest_sha:
                raise TrainedModelError("trained-model parent manifest SHA mismatch")
        if _safe_int(payload.get("reference_data_version"), "reference_data_version") != version:
            raise TrainedModelError("model/reference-data version mismatch")
        if _safe_sha(
            payload.get("model_set_sha256"), "model_set_sha256"
        ) != canonical_json_sha256(_model_set_identity(payload)):
            raise TrainedModelError("trained-model set SHA mismatch")
        head_sha = sha256_file(manifest_path)
        if version != target_version:
            previous_manifest_sha = head_sha
            continue
        reference_view = ReferenceDataVersioning(
            campaign / "QM_REFERENCE_DATA"
        ).resolve(version, verification=verification)
        if uid != reference_view.campaign_uid:
            raise TrainedModelError("model/reference-data campaign UID mismatch")
        if str(payload.get("reference_data_head_manifest_sha256")) != reference_view.head_manifest_sha256:
            raise TrainedModelError("model/reference-data head manifest SHA mismatch")
        if str(payload.get("reference_data_view_sha256")) != reference_view.cumulative_view_sha256:
            raise TrainedModelError("model/reference-data view SHA mismatch")
        properties = _safe_token_sequence(payload.get("properties"), "properties")
        atoms = _safe_token_sequence(payload.get("atoms"), "atoms")
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list):
            raise TrainedModelError("trained-model tasks must be a list")
        tasks = tuple(
            _parse_task(version_root, task, verification=verification)
            for task in raw_tasks
        )
        expected_keys = [(prop, atom) for prop in properties for atom in atoms]
        if [task.key for task in tasks] != expected_keys:
            raise TrainedModelError("trained-model tasks do not match property/atom product")
        if [task.task_index for task in tasks] != list(range(1, len(tasks) + 1)):
            raise TrainedModelError("trained-model task indexes are not contiguous")
        _validate_task_file_names(tasks, system)
        if _safe_int(payload.get("n_tasks"), "n_tasks", minimum=1) != len(tasks):
            raise TrainedModelError("trained-model task count mismatch")
        raw_root_files = payload.get("root_files")
        if not isinstance(raw_root_files, list):
            raise TrainedModelError("trained-model root_files must be a list")
        root_files = tuple(
            _parse_file_record(
                version_root,
                record,
                "root_file",
                verification=verification,
            )
            for record in raw_root_files
        )
        _validate_root_file_records(root_files)
        source_record = _parse_file_record(
            version_root,
            payload.get("source_task_manifest"),
            "source_task_manifest",
            verification=verification,
        )
        quality_record = _parse_file_record(
            version_root,
            payload.get("quality_manifest"),
            "quality_manifest",
            verification=verification,
        )
        if source_record.relative_path != "FEREBUS_TASKS.json":
            raise TrainedModelError("source task manifest path is invalid")
        if quality_record.relative_path != "FEREBUS_QUALITY.json":
            raise TrainedModelError("quality manifest path is invalid")
        root_file_by_path = {record.relative_path: record for record in root_files}
        if root_file_by_path.get(source_record.relative_path) != source_record:
            raise TrainedModelError("source task manifest root-file record mismatch")
        if root_file_by_path.get(quality_record.relative_path) != quality_record:
            raise TrainedModelError("quality manifest root-file record mismatch")
        _validate_exact_inventory(version_root, tasks, root_files)
        _validate_source_and_quality(version_root, payload, tasks, reference_view)
        resolved = TrainedModelSet(
            version=version,
            campaign_uid=uid,
            system=system,
            reference_data_version=version,
            reference_data_head_manifest_sha256=reference_view.head_manifest_sha256,
            reference_data_view_sha256=reference_view.cumulative_view_sha256,
            parent_version=None if version == 0 else version - 1,
            parent_manifest_sha256=None if version == 0 else previous_manifest_sha,
            properties=properties,
            atoms=atoms,
            tasks=tasks,
            root_files=root_files,
            model_set_sha256=str(payload["model_set_sha256"]),
            head_manifest_sha256=head_sha,
            root=version_root.resolve(),
        )
        previous_manifest_sha = head_sha
    if resolved is None:
        raise TrainedModelError("trained-model set could not be resolved")
    return resolved


def load_trained_models(
    campaign_dir: Union[str, Path],
    models_version: int,
    *,
    verification: str = "deep",
):
    from ichor.core.models import Models

    model_set = resolve_trained_model_set(
        campaign_dir,
        models_version,
        verification=verification,
    )
    models = Models.from_model_files(model_set.root, model_set.model_paths)
    return model_set, models


def seal_trained_model_version(version_dir: Union[str, Path]) -> None:
    root = Path(version_dir)
    if not root.is_dir() or root.is_symlink():
        raise TrainedModelError("cannot seal missing trained-model version: " + str(root))
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise TrainedModelError("cannot seal symlinked trained-model artefact: " + str(path))
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    mode = stat.S_IMODE(root.stat().st_mode)
    os.chmod(root, mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    from ..daemon.state import _fsync_parent_dir

    _fsync_parent_dir(root)


__all__ = [
    "TRAINED_MODEL_SET_FILENAME",
    "TRAINED_MODEL_SET_SCHEMA_VERSION",
    "TRAINED_MODEL_AUXILIARY_SUFFIXES",
    "TrainedModelError",
    "TrainedModelFile",
    "TrainedModelTask",
    "TrainedModelSet",
    "TrainedModelVersioning",
    "trained_model_set_path",
    "trained_models_commit_lock_path",
    "trained_models_commit_lock",
    "file_record",
    "build_trained_model_set_payload",
    "validate_trained_model_snapshot",
    "resolve_trained_model_set",
    "load_trained_models",
    "seal_trained_model_version",
]
