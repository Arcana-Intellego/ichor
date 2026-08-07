"""Authoritative content-verified trained-model snapshots."""

from __future__ import annotations

from ..strict_json import strict_json as json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ..layout import trained_models_dir
from .manifest import (
    MANIFEST_FILENAME,
    read_manifest,
    sha256_file,
    verify_manifest,
)
from .reference_data import (
    REFERENCE_DATA_VERSION_FILENAME,
    REFERENCE_DATA_VERSION_SCHEMA_VERSION,
    ReferenceDataView,
    ReferenceDataVersioning,
    canonical_json_sha256,
    resolve_reference_data_chain,
)
from .versioned_directory import VersionedDirectory


TRAINED_MODEL_SET_FILENAME = "FEREBUS_TASK_ARTEFACTS.json"
TRAINED_MODEL_SET_SCHEMA_VERSION = 3
TRAINED_MODEL_STORAGE_MODE = "full_snapshot"
TRAINED_MODELS_COMMIT_LOCK = ".commit.lock"
TRAINED_MODEL_AUXILIARY_SUFFIXES = ("opt", "perf", "pred", "scurve", "sol")
TRAINED_MODEL_DATASET_SPLITS = ("train", "int_val", "ext_val")
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
    execution_receipt: TrainedModelFile
    datasets: Mapping[str, TrainedModelFile]
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
            "execution_receipt": self.execution_receipt.identity_payload(),
            "datasets": {
                split: self.datasets[split].identity_payload()
                for split in TRAINED_MODEL_DATASET_SPLITS
            },
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
    evidence_set_sha256: str
    head_manifest_sha256: str
    root: Path

    @property
    def model_paths(self) -> Tuple[Path, ...]:
        return tuple(task.model.path for task in self.tasks)


@dataclass(frozen=True)
class CurrentModelFileBinding:
    """One current model payload admitted for a SEED_SELECT consumer."""

    path: Path
    size: int
    mtime_ns: int
    sha256: str


def _verify_current_model_payloads(
    campaign_dir: Union[str, Path],
    model_set: TrainedModelSet,
    *,
    expected_bindings: Optional[Sequence[CurrentModelFileBinding]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[CurrentModelFileBinding, ...]:
    from ..daemon.filesystem import campaign_owned_path

    campaign = Path(campaign_dir).resolve()
    versioning = TrainedModelVersioning(trained_models_dir(campaign))
    expected_root = versioning.iteration_path(int(model_set.version)).resolve()
    if model_set.root.resolve() != expected_root:
        raise TrainedModelError("current model snapshot root is not canonical")
    expected_by_path = (
        None
        if expected_bindings is None
        else {binding.path.resolve(): binding for binding in expected_bindings}
    )
    bindings: List[CurrentModelFileBinding] = []
    seen = set()
    total = len(model_set.tasks)
    for position, task in enumerate(model_set.tasks, start=1):
        record = task.model
        path = campaign_owned_path(campaign, Path(record.path))
        if path.resolve() != (model_set.root / record.relative_path).resolve():
            raise TrainedModelError("current model payload path identity mismatch")
        try:
            path.resolve().relative_to(expected_root)
        except ValueError as exc:
            raise TrainedModelError("current model payload escapes its snapshot") from exc
        if path.is_symlink() or not path.is_file():
            raise TrainedModelError(
                "current model payload is missing or unsafe: " + str(path)
            )
        resolved = path.resolve()
        if resolved in seen:
            raise TrainedModelError("current model payload is duplicated")
        seen.add(resolved)
        stat = path.stat()
        if int(stat.st_size) != int(record.size):
            raise TrainedModelError("current model payload size mismatch: " + str(path))
        digest = sha256_file(path)
        if digest != str(record.sha256):
            raise TrainedModelError("current model payload hash mismatch: " + str(path))
        binding = CurrentModelFileBinding(
            path=resolved,
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            sha256=digest,
        )
        if expected_by_path is not None:
            expected = expected_by_path.get(resolved)
            if expected is None or (
                int(expected.size) != int(binding.size)
                or str(expected.sha256) != str(binding.sha256)
            ):
                raise TrainedModelError(
                    "current model payload changed after consumer admission: "
                    + str(path)
                )
        bindings.append(binding)
        if progress_callback is not None:
            try:
                progress_callback(
                    "model_authority",
                    {
                        "completed": int(position),
                        "total": int(total),
                    },
                )
            except Exception:
                pass
    if expected_by_path is not None and set(expected_by_path) != seen:
        raise TrainedModelError("current model payload inventory changed")
    return tuple(bindings)


def load_trained_models_from_snapshot(
    campaign_dir: Union[str, Path],
    models_version: int,
    *,
    snapshot: Any,
    expected_campaign_uid: str,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
):
    """Load only the current model payloads from an authority snapshot."""
    from ichor.core.models import Models

    campaign = Path(campaign_dir).resolve()
    version = _safe_int(models_version, "models_version", minimum=0)
    reference_versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
    model_versioning = TrainedModelVersioning(trained_models_dir(campaign))
    if reference_versioning.current_version() != version:
        raise TrainedModelError(
            "current reference-data pointer does not match the seed-selection model version"
        )
    if model_versioning.current_version() != version:
        raise TrainedModelError(
            "current trained-model pointer does not match the seed-selection model version"
        )
    reference_view = snapshot.reference_view(version)
    model_set = snapshot.model_set(version)
    if str(reference_view.campaign_uid) != str(expected_campaign_uid):
        raise TrainedModelError("reference-data campaign identity mismatch")
    if str(model_set.campaign_uid) != str(expected_campaign_uid):
        raise TrainedModelError("trained-model campaign identity mismatch")
    if (
        int(model_set.reference_data_version) != version
        or str(model_set.reference_data_head_manifest_sha256)
        != str(reference_view.head_manifest_sha256)
        or str(model_set.reference_data_view_sha256)
        != str(reference_view.cumulative_view_sha256)
    ):
        raise TrainedModelError("trained-model reference-data binding mismatch")
    if progress_callback is not None:
        try:
            progress_callback(
                "model_authority",
                {"completed": 0, "total": int(len(model_set.tasks))},
            )
        except Exception:
            pass
    bindings = _verify_current_model_payloads(
        campaign,
        model_set,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        try:
            progress_callback("models", {"completed": 0, "total": 1})
        except Exception:
            pass
    models = Models.from_model_files(model_set.root, model_set.model_paths)
    for binding in bindings:
        stat = binding.path.stat()
        if (
            int(stat.st_size) != int(binding.size)
            or int(stat.st_mtime_ns) != int(binding.mtime_ns)
        ):
            raise TrainedModelError(
                "current model payload changed while it was being parsed: "
                + str(binding.path)
            )
    if progress_callback is not None:
        try:
            progress_callback("models", {"completed": 1, "total": 1})
        except Exception:
            pass
    return model_set, models, bindings


def load_trained_models_for_ariadne_task(
    campaign_dir: Union[str, Path],
    models_version: int,
    *,
    task_map: Mapping[str, Any],
    expected_campaign_uid: str,
):
    """Load only task-bound current models without replaying history."""
    from ichor.core.models import Models

    campaign = Path(campaign_dir).resolve()
    version = _safe_int(models_version, "models_version", minimum=0)
    if _safe_int(
        task_map.get("models_version"),
        "task-map models_version",
        minimum=0,
    ) != version:
        raise TrainedModelError("task-map trained-model version mismatch")
    reference_versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
    model_versioning = TrainedModelVersioning(trained_models_dir(campaign))
    if reference_versioning.current_version() != version:
        raise TrainedModelError(
            "current reference-data pointer does not match the ARIADNE task"
        )
    if model_versioning.current_version() != version:
        raise TrainedModelError(
            "current trained-model pointer does not match the ARIADNE task"
        )
    root = model_versioning.iteration_path(version)
    manifest_path = trained_model_set_path(root)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise TrainedModelError("current trained-model manifest is missing or unsafe")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != _safe_sha(
        task_map.get("model_manifest_sha256"),
        "task-map model manifest SHA-256",
    ):
        raise TrainedModelError("current model manifest does not match the task map")
    payload = _read_json_object(manifest_path, "trained-model set manifest")
    if _safe_int(payload.get("schema_version"), "schema_version") != TRAINED_MODEL_SET_SCHEMA_VERSION:
        raise TrainedModelError("unsupported trained-model set schema")
    if str(payload.get("storage_mode")) != TRAINED_MODEL_STORAGE_MODE:
        raise TrainedModelError("trained-model storage_mode must be full_snapshot")
    if _safe_int(payload.get("models_version"), "models_version") != version:
        raise TrainedModelError("trained-model version mismatch")
    campaign_uid = str(payload.get("campaign_uid") or "")
    if campaign_uid != str(expected_campaign_uid):
        raise TrainedModelError("trained-model campaign UID mismatch")
    if str(task_map.get("campaign_uid") or "") != campaign_uid:
        raise TrainedModelError("task-map campaign UID mismatch")
    if _safe_int(payload.get("reference_data_version"), "reference_data_version") != version:
        raise TrainedModelError("model/reference-data version mismatch")
    if version == 0:
        if payload.get("parent_version") is not None or payload.get(
            "parent_manifest_sha256"
        ) is not None:
            raise TrainedModelError("bootstrap model snapshot parent is invalid")
    else:
        if _safe_int(payload.get("parent_version"), "parent_version") != version - 1:
            raise TrainedModelError("trained-model parent version mismatch")
        _safe_sha(payload.get("parent_manifest_sha256"), "parent_manifest_sha256")

    reference_root = reference_versioning.iteration_path(version)
    reference_manifest = reference_root / REFERENCE_DATA_VERSION_FILENAME
    if reference_manifest.is_symlink() or not reference_manifest.is_file():
        raise TrainedModelError("current reference-data manifest is missing or unsafe")
    reference_sha = sha256_file(reference_manifest)
    if str(payload.get("reference_data_head_manifest_sha256")) != reference_sha:
        raise TrainedModelError("model/reference-data head manifest SHA mismatch")
    reference_payload = _read_json_object(
        reference_manifest,
        "reference-data version manifest",
    )
    if _safe_int(reference_payload.get("schema_version"), "reference schema") != REFERENCE_DATA_VERSION_SCHEMA_VERSION:
        raise TrainedModelError("unsupported reference-data version schema")
    if _safe_int(
        reference_payload.get("reference_data_version"),
        "reference_data_version",
    ) != version:
        raise TrainedModelError("reference-data version mismatch")
    if str(reference_payload.get("campaign_uid") or "") != campaign_uid:
        raise TrainedModelError("reference-data campaign UID mismatch")
    reference_view_sha = _safe_sha(
        reference_payload.get("cumulative_view_sha256"),
        "reference-data cumulative view SHA-256",
    )
    if str(payload.get("reference_data_view_sha256")) != reference_view_sha:
        raise TrainedModelError("model/reference-data cumulative view SHA mismatch")

    system = _safe_token(payload.get("system"), "system")
    properties = _safe_token_sequence(payload.get("properties"), "properties")
    atoms = _safe_token_sequence(payload.get("atoms"), "atoms")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise TrainedModelError("trained-model tasks must be a list")
    tasks = tuple(
        _parse_task(root, record, verification="metadata")
        for record in raw_tasks
    )
    expected_keys = [(prop, atom) for prop in properties for atom in atoms]
    if [task.key for task in tasks] != expected_keys:
        raise TrainedModelError("trained-model tasks do not match property/atom product")
    if [task.task_index for task in tasks] != list(range(1, len(tasks) + 1)):
        raise TrainedModelError("trained-model task indexes are not contiguous")
    if _safe_int(payload.get("n_tasks"), "n_tasks", minimum=1) != len(tasks):
        raise TrainedModelError("trained-model task count mismatch")
    _validate_task_file_names(tasks, system)
    raw_root_files = payload.get("root_files")
    if not isinstance(raw_root_files, list):
        raise TrainedModelError("trained-model root_files must be a list")
    root_files = tuple(
        _parse_file_record(root, record, "root_file", verification="metadata")
        for record in raw_root_files
    )
    _validate_root_file_records(root_files)
    model_set_sha = _safe_sha(payload.get("model_set_sha256"), "model_set_sha256")
    if model_set_sha != canonical_json_sha256(_model_set_identity(payload)):
        raise TrainedModelError("trained-model set SHA mismatch")
    if model_set_sha != _safe_sha(
        task_map.get("model_set_sha256"),
        "task-map model-set SHA-256",
    ):
        raise TrainedModelError("scientific model set does not match the task map")
    evidence_set_sha = _safe_sha(
        payload.get("evidence_set_sha256"),
        "evidence_set_sha256",
    )
    if evidence_set_sha != canonical_json_sha256(_evidence_set_identity(payload)):
        raise TrainedModelError("trained-model evidence-set SHA mismatch")
    model_set = TrainedModelSet(
        version=version,
        campaign_uid=campaign_uid,
        system=system,
        reference_data_version=version,
        reference_data_head_manifest_sha256=reference_sha,
        reference_data_view_sha256=reference_view_sha,
        parent_version=None if version == 0 else version - 1,
        parent_manifest_sha256=(
            None if version == 0 else str(payload["parent_manifest_sha256"])
        ),
        properties=properties,
        atoms=atoms,
        tasks=tasks,
        root_files=root_files,
        model_set_sha256=model_set_sha,
        evidence_set_sha256=evidence_set_sha,
        head_manifest_sha256=manifest_sha,
        root=root.resolve(),
    )
    bindings = _verify_current_model_payloads(campaign, model_set)
    models = Models.from_model_files(model_set.root, model_set.model_paths)
    for binding in bindings:
        stat = binding.path.stat()
        if (
            int(stat.st_size) != int(binding.size)
            or int(stat.st_mtime_ns) != int(binding.mtime_ns)
        ):
            raise TrainedModelError(
                "current model payload changed while it was being parsed: "
                + str(binding.path)
            )
    return model_set, models, bindings


def assert_current_model_payloads_unchanged(
    campaign_dir: Union[str, Path],
    model_set: TrainedModelSet,
    bindings: Sequence[CurrentModelFileBinding],
) -> None:
    _verify_current_model_payloads(
        campaign_dir,
        model_set,
        expected_bindings=bindings,
    )


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
        reference_verification: Optional[str] = None,
    ) -> TrainedModelSet:
        return resolve_trained_model_set(
            Path(self.parent).parent,
            int(version),
            verification=verification,
            trained_models_root=self.parent,
            reference_verification=reference_verification,
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


def file_record(
    path: Union[str, Path],
    root: Union[str, Path],
    *,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
) -> Dict[str, Any]:
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
        "sha256": (
            sha256_file(file_path)
            if digest_file is None
            else digest_file(file_path, False)
        ),
    }


def _parse_file_record(
    root: Path,
    payload: Any,
    label: str,
    *,
    verification: str,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
) -> TrainedModelFile:
    if not isinstance(payload, Mapping):
        raise TrainedModelError(label + " must be an object")
    relative = _safe_relative_path(payload.get("path"), label + ".path")
    expected_size = _safe_int(payload.get("size"), label + ".size", minimum=0)
    expected_sha = _safe_sha(payload.get("sha256"), label + ".sha256")
    path = root.joinpath(*PurePosixPath(relative).parts)
    if verification != "authority":
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
    if verification == "deep":
        observed_sha = (
            digest_file(path, True)
            if digest_file is not None
            else sha256_file(path)
        )
        if observed_sha != expected_sha:
            raise TrainedModelError(label + " SHA-256 mismatch: " + str(path))
    return TrainedModelFile(
        relative_path=relative,
        path=(path.absolute() if verification == "authority" else path.resolve()),
        size=expected_size,
        sha256=expected_sha,
    )


def _parse_task(
    root: Path,
    payload: Any,
    *,
    verification: str,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
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
        digest_file=digest_file,
    )
    config = _parse_file_record(
        root,
        payload.get("config"),
        prop + "/" + atom + ":config",
        verification=verification,
        digest_file=digest_file,
    )
    execution_receipt = _parse_file_record(
        root,
        payload.get("execution_receipt"),
        prop + "/" + atom + ":execution_receipt",
        verification=verification,
        digest_file=digest_file,
    )
    task_root = PurePosixPath(directory)
    if PurePosixPath(model.relative_path).parent != task_root or not model.relative_path.endswith(".model"):
        raise TrainedModelError(prop + "/" + atom + ":model path is not canonical")
    if PurePosixPath(config.relative_path) != task_root / "ferebus.config":
        raise TrainedModelError(prop + "/" + atom + ":config path is not canonical")
    if PurePosixPath(execution_receipt.relative_path) != task_root / "FEREBUS_TASK_RECEIPT.json":
        raise TrainedModelError(
            prop + "/" + atom + ":execution receipt path is not canonical"
        )
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, Mapping) or set(raw_datasets) != set(
        TRAINED_MODEL_DATASET_SPLITS
    ):
        raise TrainedModelError(prop + "/" + atom + ":datasets keys are invalid")
    datasets: Dict[str, TrainedModelFile] = {}
    for split in TRAINED_MODEL_DATASET_SPLITS:
        parsed = _parse_file_record(
            root,
            raw_datasets.get(split),
            prop + "/" + atom + ":dataset:" + split,
            verification=verification,
            digest_file=digest_file,
        )
        dataset_path = PurePosixPath(parsed.relative_path)
        if dataset_path.parent != task_root / "datasets" or dataset_path.suffix != ".csv":
            raise TrainedModelError(
                prop + "/" + atom + ":" + split + " dataset path is not canonical"
            )
        datasets[split] = parsed
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
            digest_file=digest_file,
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
        execution_receipt=execution_receipt,
        datasets=datasets,
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
    """Return only prediction-affecting model structure and file bytes."""
    tasks = payload.get("tasks")
    scientific_tasks = []
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, Mapping):
                scientific_tasks.append(task)
                continue
            scientific_tasks.append({
                "task_index": task.get("task_index"),
                "property": task.get("property"),
                "atom": task.get("atom"),
                "alf_1_indexed": task.get("alf_1_indexed"),
                "directory": task.get("directory"),
                "model": task.get("model"),
                "config": task.get("config"),
            })
    return {
        "system": payload.get("system"),
        "properties": payload.get("properties"),
        "atoms": payload.get("atoms"),
        "tasks": scientific_tasks,
    }


def _evidence_set_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return complete immutable training and quality provenance evidence."""
    return {
        "schema_version": payload.get("schema_version"),
        "storage_mode": payload.get("storage_mode"),
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
        "quality_decision_manifest": payload.get("quality_decision_manifest"),
        "properties": payload.get("properties"),
        "atoms": payload.get("atoms"),
        "n_tasks": payload.get("n_tasks"),
        "tasks": payload.get("tasks"),
        "root_files": payload.get("root_files"),
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
    quality_decision_manifest: Mapping[str, Any],
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
        "quality_decision_manifest": dict(quality_decision_manifest),
        "properties": [str(value) for value in properties],
        "atoms": [str(value) for value in atoms],
        "n_tasks": int(len(tasks)),
        "tasks": [dict(value) for value in tasks],
        "root_files": [dict(value) for value in root_files],
    }
    payload["model_set_sha256"] = canonical_json_sha256(_model_set_identity(payload))
    payload["evidence_set_sha256"] = canonical_json_sha256(
        _evidence_set_identity(payload)
    )
    return payload


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), source=path)
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
        expected_task_files.add(task.execution_receipt.relative_path)
        expected_directories.add(task.directory + "/datasets")
        expected_task_files.update(
            dataset.relative_path for dataset in task.datasets.values()
        )
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
    decision = _read_json_object(
        root / "FEREBUS_QUALITY_DECISION.json",
        "FEREBUS quality decision",
    )
    if _safe_int(source.get("schema_version"), "FEREBUS task schema_version") != 5:
        raise TrainedModelError("unsupported committed FEREBUS task schema")
    if _safe_int(quality.get("schema_version"), "FEREBUS quality schema_version") != 4:
        raise TrainedModelError("unsupported committed FEREBUS quality schema")
    if _safe_int(decision.get("schema_version"), "FEREBUS decision schema_version") != 2:
        raise TrainedModelError("unsupported committed FEREBUS quality-decision schema")
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
    model_bootstrap = source.get("model_bootstrap")
    if model_bootstrap is None:
        historical_training_rows = 0
    elif isinstance(model_bootstrap, Mapping):
        historical_training_rows = _safe_int(
            model_bootstrap.get("historical_training_rows"),
            "FEREBUS historical_training_rows",
            minimum=1,
        )
        copied_model_manifest = root / "MODEL_BOOTSTRAP.json"
        if not copied_model_manifest.is_file() or copied_model_manifest.is_symlink():
            raise TrainedModelError(
                "committed model-bootstrap manifest is missing"
            )
        if sha256_file(copied_model_manifest) != _safe_sha(
            model_bootstrap.get("manifest_sha256"),
            "model-bootstrap manifest SHA-256",
        ):
            raise TrainedModelError("model-bootstrap manifest SHA mismatch")
    else:
        raise TrainedModelError("FEREBUS model_bootstrap record is invalid")
    expected_counts = {
        split: len(rows) + (
            historical_training_rows if split == "train" else 0
        )
        for split, rows in expected_split_rows.items()
    }
    dataset_path_fields = {
        "train": "training_csv",
        "int_val": "int_validation_csv",
        "ext_val": "ext_validation_csv",
    }
    committed_task_by_key = {task.key: task for task in tasks}
    for source_task in source_tasks:
        if not isinstance(source_task, Mapping):
            raise TrainedModelError("FEREBUS source task record is invalid")
        source_key = (
            str(source_task.get("property")),
            str(source_task.get("atom")),
        )
        committed_task = committed_task_by_key.get(source_key)
        if committed_task is None:
            raise TrainedModelError("FEREBUS source task has no committed model task")
        row_counts = source_task.get("row_counts") or {}
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
        if observed_counts != expected_counts:
            raise TrainedModelError("FEREBUS task manifest split count mismatch")
        if _safe_int(
            source_task.get("historical_training_rows", 0),
            "FEREBUS task historical_training_rows",
            minimum=0,
        ) != historical_training_rows:
            raise TrainedModelError(
                "FEREBUS task historical training-row count mismatch"
            )
        historical_ids = source_task.get("historical_training_row_ids", [])
        if not isinstance(historical_ids, list) or [
            _safe_int(
                value,
                "FEREBUS historical training-row index",
                minimum=0,
            )
            for value in historical_ids
        ] != list(range(historical_training_rows)):
            raise TrainedModelError(
                "FEREBUS task historical training-row IDs are invalid"
            )
        row_ids = source_task.get("row_ids") or {}
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
        datasets = source_task.get("datasets")
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
            ) != str(source_task.get(path_field) or ""):
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
            committed_dataset = committed_task.datasets[split]
            if (
                committed_dataset.relative_path != str(record.get("path") or "")
                or committed_dataset.size
                != _safe_int(
                    record.get("size"),
                    "FEREBUS " + split + " dataset size",
                    minimum=0,
                )
                or committed_dataset.sha256
                != _safe_sha(
                    record.get("sha256"),
                    "FEREBUS " + split + " dataset SHA-256",
                )
            ):
                raise TrainedModelError(
                    "FEREBUS " + split + " committed dataset binding mismatch"
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
    if quality.get("measurement_complete") is not True or quality.get(
        "measurement_errors"
    ) != []:
        raise TrainedModelError("committed FEREBUS raw measurements are incomplete")
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
    quality_binding = decision.get("quality")
    quality_path = root / "FEREBUS_QUALITY.json"
    if (
        not isinstance(quality_binding, Mapping)
        or quality_binding.get("path") != "FEREBUS_QUALITY.json"
        or _safe_int(
            quality_binding.get("size"),
            "FEREBUS decision quality size",
            minimum=0,
        )
        != int(quality_path.stat().st_size)
        or _safe_sha(
            quality_binding.get("sha256"),
            "FEREBUS decision quality SHA-256",
        )
        != sha256_file(quality_path)
    ):
        raise TrainedModelError("FEREBUS decision/raw-quality binding mismatch")
    evaluations = decision.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise TrainedModelError("FEREBUS decision has no evaluations")
    current_digest = _safe_sha(
        decision.get("current_evaluation_sha256"),
        "FEREBUS current evaluation SHA-256",
    )
    current = None
    from ..daemon.completion_receipts import canonical_sha256

    seen_evaluations = set()
    for raw_evaluation in evaluations:
        if not isinstance(raw_evaluation, Mapping):
            raise TrainedModelError("FEREBUS quality evaluation is invalid")
        material = dict(raw_evaluation)
        declared = _safe_sha(
            material.pop("evaluation_sha256", None),
            "FEREBUS evaluation SHA-256",
        )
        material.pop("evaluated_at_iso", None)
        if declared in seen_evaluations or declared != canonical_sha256(material):
            raise TrainedModelError("FEREBUS quality evaluation digest mismatch")
        seen_evaluations.add(declared)
        if declared == current_digest:
            current = raw_evaluation
    if current is None or current.get("accepted") is not True:
        raise TrainedModelError("committed FEREBUS promotion decision is rejected")
    decision_tasks = current.get("tasks")
    if not isinstance(decision_tasks, list) or [
        (str(item.get("property")), str(item.get("atom")))
        for item in decision_tasks
        if isinstance(item, Mapping)
    ] != expected_keys or any(
        not isinstance(item, Mapping) or item.get("accepted") is not True
        for item in decision_tasks
    ):
        raise TrainedModelError("FEREBUS promotion task decisions are invalid")

    execution = quality.get("task_execution")
    execution_records = execution.get("receipts") if isinstance(execution, Mapping) else None
    task_map_path = root / "FEREBUS_TASK_MAP.json"
    if (
        not isinstance(execution, Mapping)
        or execution.get("task_map_path") != "FEREBUS_TASK_MAP.json"
        or _safe_sha(
            execution.get("task_map_file_sha256"),
            "FEREBUS task-map file SHA-256",
        )
        != sha256_file(task_map_path)
        or _safe_int(
            execution.get("n_tasks"),
            "FEREBUS execution task count",
            minimum=1,
        )
        != len(tasks)
    ):
        raise TrainedModelError("FEREBUS task-map execution binding is invalid")
    if not isinstance(execution_records, list) or len(execution_records) != len(tasks):
        raise TrainedModelError("FEREBUS task-execution coverage is invalid")
    for task, execution_record in zip(tasks, execution_records):
        if not isinstance(execution_record, Mapping):
            raise TrainedModelError("FEREBUS task-execution record is invalid")
        if (
            _safe_int(
                execution_record.get("task_index"),
                "FEREBUS execution task index",
                minimum=1,
            )
            != task.task_index
            or execution_record.get("receipt_path")
            != task.execution_receipt.relative_path
            or _safe_sha(
                execution_record.get("receipt_sha256"),
                "FEREBUS execution receipt SHA-256",
            )
            != task.execution_receipt.sha256
            or execution_record.get("model_path") != task.model.relative_path
            or _safe_sha(
                execution_record.get("model_sha256"),
                "FEREBUS execution model SHA-256",
            )
            != task.model.sha256
            or execution_record.get("performance_path")
            != (
                None
                if task.auxiliary.get("perf") is None
                else task.auxiliary["perf"].relative_path
            )
            or (
                task.auxiliary.get("perf") is not None
                and _safe_sha(
                    execution_record.get("performance_sha256"),
                    "FEREBUS execution performance SHA-256",
                )
                != task.auxiliary["perf"].sha256
            )
        ):
            raise TrainedModelError("FEREBUS task-execution binding mismatch")
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
    reference_view: Optional[ReferenceDataView] = None,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
    require_directory_manifest: bool = False,
) -> TrainedModelSet:
    """Validate one unpublished full snapshot before its atomic rename."""
    if verification not in {"authority", "metadata", "deep"}:
        raise ValueError(
            "trained-model verification must be authority, metadata or deep"
        )
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
    if reference_view is None:
        reference_view = ReferenceDataVersioning(
            campaign / "QM_REFERENCE_DATA"
        ).resolve(
            expected_version,
            verification=verification,
            digest_file=digest_file,
        )
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
        _parse_task(
            root,
            task,
            verification=verification,
            digest_file=digest_file,
        )
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
            root,
            record,
            "root_file",
            verification=verification,
            digest_file=digest_file,
        )
        for record in raw_root_files
    )
    _validate_root_file_records(root_files)
    source_record = _parse_file_record(
        root,
        payload.get("source_task_manifest"),
        "source_task_manifest",
        verification=verification,
        digest_file=digest_file,
    )
    quality_record = _parse_file_record(
        root,
        payload.get("quality_manifest"),
        "quality_manifest",
        verification=verification,
        digest_file=digest_file,
    )
    quality_decision_record = _parse_file_record(
        root,
        payload.get("quality_decision_manifest"),
        "quality_decision_manifest",
        verification=verification,
        digest_file=digest_file,
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
    if (
        quality_decision_record.relative_path != "FEREBUS_QUALITY_DECISION.json"
        or root_file_by_path.get(quality_decision_record.relative_path)
        != quality_decision_record
    ):
        raise TrainedModelError("quality decision root-file record mismatch")
    if _safe_sha(payload.get("model_set_sha256"), "model_set_sha256") != canonical_json_sha256(
        _model_set_identity(payload)
    ):
        raise TrainedModelError("trained-model set SHA mismatch")
    if _safe_sha(
        payload.get("evidence_set_sha256"), "evidence_set_sha256"
    ) != canonical_json_sha256(_evidence_set_identity(payload)):
        raise TrainedModelError("trained-model evidence-set SHA mismatch")
    if verification == "authority":
        directory_manifest = read_manifest(root)
        expected_records = {
            record.relative_path: record.sha256
            for record in root_files
        }
        for task in tasks:
            task_records = (
                task.model,
                task.config,
                task.execution_receipt,
                *task.datasets.values(),
                *(
                    record
                    for record in task.auxiliary.values()
                    if record is not None
                ),
            )
            for record in task_records:
                expected_records[record.relative_path] = record.sha256
        expected_records[TRAINED_MODEL_SET_FILENAME] = sha256_file(
            trained_model_set_path(root)
        )
        if directory_manifest != expected_records:
            unexpected = sorted(set(directory_manifest) - set(expected_records))
            missing = sorted(set(expected_records) - set(directory_manifest))
            mismatched = sorted(
                path
                for path in set(directory_manifest) & set(expected_records)
                if directory_manifest[path] != expected_records[path]
            )
            raise TrainedModelError(
                "trained-model authority manifest mismatch: unexpected="
                + repr(unexpected)
                + " missing="
                + repr(missing)
                + " mismatched="
                + repr(mismatched)
            )
    else:
        _validate_exact_inventory(
            root,
            tasks,
            root_files,
            require_directory_manifest=require_directory_manifest,
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
        evidence_set_sha256=str(payload["evidence_set_sha256"]),
        head_manifest_sha256=sha256_file(trained_model_set_path(root)),
        root=root.resolve(),
    )


def resolve_trained_model_chain(
    campaign_dir: Union[str, Path],
    models_version: int,
    *,
    verification: str = "deep",
    trained_models_root: Optional[Union[str, Path]] = None,
    reference_views: Optional[Sequence[ReferenceDataView]] = None,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
    resolved_models_out: Optional[List[TrainedModelSet]] = None,
) -> Tuple[TrainedModelSet, ...]:
    if verification not in {"authority", "metadata", "deep"}:
        raise ValueError(
            "trained-model verification must be authority, metadata or deep"
        )
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
    if verification == "deep" and digest_file is None:
        digest_cache: Dict[str, str] = {}

        def cached_digest(path: Path, payload: bool) -> str:
            del payload
            key = str(Path(path).absolute())
            if key not in digest_cache:
                digest_cache[key] = sha256_file(path)
            return digest_cache[key]

        digest_file = cached_digest
    if reference_views is None:
        reference_views = resolve_reference_data_chain(
            campaign,
            target_version,
            verification=verification,
            digest_file=digest_file,
        )
    reference_by_version = {int(view.version): view for view in reference_views}
    if set(expected_versions) - set(reference_by_version):
        raise TrainedModelError("trained-model reference-data chain is incomplete")

    resolved: List[TrainedModelSet] = []
    parent: Optional[TrainedModelSet] = None
    for version in expected_versions:
        version_root = versioning.iteration_path(version)
        directory_manifest = read_manifest(version_root)
        if verification == "deep":
            verify_manifest(
                version_root,
                manifest=directory_manifest,
                digest_file=digest_file,
            )
        model_set = validate_trained_model_snapshot(
            campaign,
            version_root,
            version,
            parent=parent,
            verification=verification,
            reference_view=reference_by_version[version],
            digest_file=digest_file,
            require_directory_manifest=True,
        )
        resolved.append(model_set)
        if resolved_models_out is not None:
            resolved_models_out.append(model_set)
        parent = model_set
    return tuple(resolved)


def resolve_trained_model_set(
    campaign_dir: Union[str, Path],
    models_version: int,
    *,
    verification: str = "deep",
    trained_models_root: Optional[Union[str, Path]] = None,
    reference_verification: Optional[str] = None,
) -> TrainedModelSet:
    if verification not in {"authority", "metadata", "deep"}:
        raise ValueError(
            "trained-model verification must be authority, metadata or deep"
        )
    reference_level = (
        verification
        if reference_verification is None
        else str(reference_verification)
    )
    if reference_level not in {"authority", "metadata", "deep"}:
        raise ValueError(
            "reference-data verification must be authority, metadata or deep"
        )
    target_version = _safe_int(models_version, "models_version", minimum=0)
    campaign = Path(campaign_dir)
    reference_views = resolve_reference_data_chain(
        campaign,
        target_version,
        verification=reference_level,
    )
    return resolve_trained_model_chain(
        campaign,
        target_version,
        verification=verification,
        trained_models_root=trained_models_root,
        reference_views=reference_views,
    )[-1]


def load_trained_models(
    campaign_dir: Union[str, Path],
    models_version: int,
    *,
    verification: str = "deep",
    reference_verification: Optional[str] = None,
):
    from ichor.core.models import Models

    model_set = resolve_trained_model_set(
        campaign_dir,
        models_version,
        verification=verification,
        reference_verification=reference_verification,
    )
    models = Models.from_model_files(model_set.root, model_set.model_paths)
    return model_set, models


__all__ = [
    "TRAINED_MODEL_SET_FILENAME",
    "TRAINED_MODEL_SET_SCHEMA_VERSION",
    "TRAINED_MODEL_AUXILIARY_SUFFIXES",
    "TrainedModelError",
    "TrainedModelFile",
    "TrainedModelTask",
    "TrainedModelSet",
    "CurrentModelFileBinding",
    "TrainedModelVersioning",
    "trained_model_set_path",
    "trained_models_commit_lock_path",
    "trained_models_commit_lock",
    "file_record",
    "build_trained_model_set_payload",
    "validate_trained_model_snapshot",
    "resolve_trained_model_chain",
    "resolve_trained_model_set",
    "load_trained_models",
    "load_trained_models_from_snapshot",
    "load_trained_models_for_ariadne_task",
    "assert_current_model_payloads_unchanged",
]
