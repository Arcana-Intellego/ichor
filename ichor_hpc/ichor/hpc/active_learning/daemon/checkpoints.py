"""Content-addressed, verifiable campaign checkpoints.

Checkpoints are operational durability artefacts, not scientific inputs.  A
checkpoint is published only from an idle, deeply verified campaign boundary,
and restoration is permitted only into an empty target directory.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..config import CONFIG_SCHEMA_VERSION
from ..strict_json import strict_json as json
from .artifact_contracts import verify_state_referenced_artifacts
from .filesystem import lexical_absolute_path, reject_symlink_components
from .state import (
    SCHEMA_VERSION as STATE_SCHEMA_VERSION,
    CampaignPhase,
    atomic_write_json,
    read_state,
)
from .submission_intent import ACTIVE_STATUSES, inventory_intents


CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_COMPLETE_SCHEMA_VERSION = 1
CHECKPOINT_CURRENT_SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 8 * 1024 * 1024

_EXCLUDED_DIRECTORY_PARTS = {
    ".git",
    "__pycache__",
}
_EXCLUDED_OPERATIONAL_PREFIXES = (
    (".DATA", "CACHE"),
    (".DATA", "SCRATCH"),
    (".DATA", "SCRATCH_CLEANUP"),
    (".DATA", "STAGING"),
)
_EXCLUDED_FILE_NAMES = {
    "daemon.lock",
    "stop_control.lock",
    "ferebus_split_assignments.lock",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strict_json_file(path: Path, *, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " is not a regular file: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise ValueError(label + " must contain a JSON object")
    return payload


def normalise_checkpoint_destination(value: Union[str, Path]) -> Path:
    expanded = os.path.expandvars(os.fspath(Path(value).expanduser()))
    destination = lexical_absolute_path(expanded)
    reject_symlink_components(destination)
    return destination


def checkpoint_store(
    destination: Union[str, Path],
    campaign_uid: str,
) -> Path:
    uid = str(campaign_uid)
    if not uid or any(character not in "0123456789abcdef-" for character in uid.lower()):
        raise ValueError("campaign UID is unsafe for checkpoint storage")
    root = normalise_checkpoint_destination(destination) / uid
    reject_symlink_components(root)
    return root


def checkpoint_directory(store: Path, iteration: int) -> Path:
    if isinstance(iteration, bool) or int(iteration) < 0:
        raise ValueError("checkpoint iteration must be a non-negative integer")
    return store / "checkpoints" / ("iteration-" + str(int(iteration)).zfill(6))


def _excluded_relative_path(relative: Path) -> bool:
    parts = relative.parts
    if any(part in _EXCLUDED_DIRECTORY_PARTS for part in parts):
        return True
    if len(parts) >= 2 and parts[-2:] == ("6_TRAINED_MODELS", "iteration-staging"):
        return True
    if any(
        len(parts) >= len(prefix) and tuple(parts[: len(prefix)]) == prefix
        for prefix in _EXCLUDED_OPERATIONAL_PREFIXES
    ):
        return True
    if "iteration-staging" in parts:
        return True
    if any(part == "daemon.lease.d" or part.startswith("daemon.lease.d.stale.") for part in parts):
        return True
    name = relative.name
    if name in _EXCLUDED_FILE_NAMES:
        return True
    if name.endswith(".tmp") or ".tmp." in name or name.endswith(".partial"):
        return True
    return False


def _campaign_files(campaign_dir: Path) -> List[Path]:
    root = lexical_absolute_path(campaign_dir)
    reject_symlink_components(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("campaign root is not a regular directory: " + str(root))
    files: List[Path] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        relative_directory = current.relative_to(root)
        kept_directories: List[str] = []
        for name in sorted(directory_names):
            child = current / name
            relative = relative_directory / name
            if child.is_symlink():
                raise ValueError("checkpoint source contains a symlink: " + str(child))
            if not _excluded_relative_path(relative):
                kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            source = current / name
            relative = source.relative_to(root)
            if source.is_symlink():
                raise ValueError("checkpoint source contains a symlink: " + str(source))
            if _excluded_relative_path(relative):
                continue
            if not source.is_file():
                raise ValueError("checkpoint source entry is not a regular file: " + str(source))
            files.append(source)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _validate_idle_boundary(
    campaign_dir: Path,
    *,
    allow_active_lease: bool,
) -> Tuple[Any, int]:
    state = read_state(campaign_dir / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    if state.phase not in {CampaignPhase.SEED_SELECT, CampaignPhase.DONE}:
        raise ValueError(
            "checkpoint creation requires an idle SEED_SELECT or DONE boundary; "
            "observed " + state.phase.value
        )
    if any(value is not None for value in state.pending_jobs.values()):
        raise ValueError("checkpoint creation is blocked by pending scheduler jobs")
    inventory = inventory_intents(
        campaign_dir,
        expected_campaign_uid=str(state.campaign_uid),
    )
    if inventory["errors"]:
        raise ValueError("checkpoint creation found malformed submission intents")
    active = [
        item
        for item in inventory["records"]
        if str(item.get("status") or "") in ACTIVE_STATUSES
    ]
    if active:
        raise ValueError("checkpoint creation is blocked by active submission intents")
    from ..execution_identity import (
        read_active_environment_generation,
        read_execution_identity,
    )

    read_execution_identity(
        campaign_dir,
        expected_campaign_uid=str(state.campaign_uid),
    )
    read_active_environment_generation(
        campaign_dir,
        expected_campaign_uid=str(state.campaign_uid),
    )
    lease = campaign_dir / ".DATA" / "ACTIVE_LEARNING" / "daemon.lease.d"
    if lease.exists() and not allow_active_lease:
        raise ValueError(
            "checkpoint creation is blocked while a daemon lease exists; "
            "use the automatic boundary checkpoint or stop the daemon"
        )
    verify_state_referenced_artifacts(campaign_dir, state, strict_models=True)
    completed_iteration = (
        int(state.iteration) - 1
        if state.phase is CampaignPhase.SEED_SELECT
        else int(state.iteration)
    )
    if completed_iteration < 0:
        raise ValueError("campaign has no completed iteration to checkpoint")
    return state, completed_iteration


def _object_path(store: Path, digest: str) -> Path:
    return store / "objects" / str(digest)


def _verify_object(path: Path, *, digest: str, size: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint object is not a regular file: " + str(path))
    if path.stat().st_size != int(size):
        raise ValueError("checkpoint object size mismatch: " + str(path))
    if _sha256_file(path) != str(digest):
        raise ValueError("checkpoint object digest mismatch: " + str(path))


def _copy_object(source: Path, destination: Path, *, digest: str, size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(destination)
    if destination.exists():
        _verify_object(destination, digest=digest, size=size)
        return
    temporary = destination.with_name(destination.name + ".tmp." + uuid.uuid4().hex)
    try:
        copied_digest = hashlib.sha256()
        copied_size = 0
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            while True:
                chunk = source_handle.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                target_handle.write(chunk)
                copied_digest.update(chunk)
                copied_size += len(chunk)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        if copied_size != int(size) or copied_digest.hexdigest() != str(digest):
            raise ValueError("checkpoint source changed while it was copied: " + str(source))
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _verify_object(destination, digest=digest, size=size)


def _publish_current_pointer(
    store: Path,
    *,
    campaign_uid: str,
    iteration: int,
    manifest_path: Path,
    manifest_sha256: str,
) -> Dict[str, Any]:
    current = {
        "schema_version": CHECKPOINT_CURRENT_SCHEMA_VERSION,
        "campaign_uid": str(campaign_uid),
        "iteration": int(iteration),
        "manifest_path": manifest_path.relative_to(store).as_posix(),
        "manifest_sha256": str(manifest_sha256),
        "updated_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(store / "current.json", current)
    return current


def create_checkpoint(
    campaign_dir: Union[str, Path],
    destination: Union[str, Path],
    *,
    iteration: Optional[int] = None,
    verify_after_write: bool = True,
    allow_active_lease: bool = False,
) -> Dict[str, Any]:
    """Create or verify one immutable checkpoint at an idle campaign boundary."""
    campaign = lexical_absolute_path(campaign_dir)
    state, completed_iteration = _validate_idle_boundary(
        campaign,
        allow_active_lease=bool(allow_active_lease),
    )
    selected_iteration = completed_iteration if iteration is None else int(iteration)
    if selected_iteration != completed_iteration:
        raise ValueError(
            "checkpoint iteration must equal the latest completed iteration "
            + str(completed_iteration)
        )
    store = checkpoint_store(destination, str(state.campaign_uid))
    try:
        store.relative_to(campaign)
    except ValueError:
        pass
    else:
        raise ValueError("checkpoint destination must be outside the campaign root")
    checkpoint = checkpoint_directory(store, selected_iteration)
    manifest_path = checkpoint / "manifest.json"
    complete_path = checkpoint / "COMPLETE.json"
    if manifest_path.exists() or complete_path.exists():
        verified = verify_checkpoint(checkpoint)
        _publish_current_pointer(
            store,
            campaign_uid=str(state.campaign_uid),
            iteration=selected_iteration,
            manifest_path=manifest_path,
            manifest_sha256=str(verified["manifest_sha256"]),
        )
        return verified

    files = _campaign_files(campaign)
    records: List[Dict[str, Any]] = []
    missing_bytes = 0
    for source in files:
        digest = _sha256_file(source)
        size = source.stat().st_size
        relative = source.relative_to(campaign).as_posix()
        mode = stat.S_IMODE(source.stat().st_mode)
        record = {
            "path": relative,
            "sha256": digest,
            "size": int(size),
            "mode": int(mode),
        }
        records.append(record)
        if not _object_path(store, digest).exists():
            missing_bytes += int(size)
    store.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(store)
    free_bytes = shutil.disk_usage(store).free
    if missing_bytes > free_bytes:
        raise OSError(
            "checkpoint destination has insufficient free space: required="
            + str(missing_bytes)
            + " free="
            + str(free_bytes)
        )
    for source, record in zip(files, records):
        _copy_object(
            source,
            _object_path(store, str(record["sha256"])),
            digest=str(record["sha256"]),
            size=int(record["size"]),
        )
    for source, record in zip(files, records):
        if source.stat().st_size != int(record["size"]) or _sha256_file(source) != str(record["sha256"]):
            raise ValueError("checkpoint source changed before publication: " + str(source))

    manifest: Dict[str, Any] = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "campaign_uid": str(state.campaign_uid),
        "iteration": int(selected_iteration),
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "source_campaign_root": str(campaign),
        "state_phase": state.phase.value,
        "state_iteration": int(state.iteration),
        "campaign_schema_version": CONFIG_SCHEMA_VERSION,
        "state_schema_version": int(state.schema_version),
        "n_files": len(records),
        "total_bytes": sum(int(item["size"]) for item in records),
        "files": records,
    }
    checkpoints_root = checkpoint.parent
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(checkpoints_root)
    temporary_checkpoint = checkpoints_root / (
        "." + checkpoint.name + ".tmp." + uuid.uuid4().hex
    )
    temporary_checkpoint.mkdir(exist_ok=False)
    try:
        temporary_manifest_path = temporary_checkpoint / "manifest.json"
        temporary_complete_path = temporary_checkpoint / "COMPLETE.json"
        atomic_write_json(temporary_manifest_path, manifest)
        manifest_digest = _sha256_file(temporary_manifest_path)
        complete = {
            "schema_version": CHECKPOINT_COMPLETE_SCHEMA_VERSION,
            "campaign_uid": str(state.campaign_uid),
            "iteration": int(selected_iteration),
            "manifest_sha256": manifest_digest,
            "n_files": len(records),
            "completed_at_iso": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(temporary_complete_path, complete)
        _fsync_directory(temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint)
        _fsync_directory(checkpoints_root)
    finally:
        shutil.rmtree(temporary_checkpoint, ignore_errors=True)
    _publish_current_pointer(
        store,
        campaign_uid=str(state.campaign_uid),
        iteration=selected_iteration,
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
    )
    if verify_after_write:
        return verify_checkpoint(checkpoint)
    return {
        "ok": True,
        "store": str(store),
        "checkpoint": str(checkpoint),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "manifest": manifest,
        "complete": complete,
    }


def _resolve_checkpoint(value: Union[str, Path]) -> Tuple[Path, Path, Path]:
    supplied = lexical_absolute_path(value)
    reject_symlink_components(supplied)
    checkpoint = supplied.parent if supplied.name in {"manifest.json", "COMPLETE.json"} else supplied
    manifest_path = checkpoint / "manifest.json"
    complete_path = checkpoint / "COMPLETE.json"
    return checkpoint, manifest_path, complete_path


def verify_checkpoint(value: Union[str, Path]) -> Dict[str, Any]:
    checkpoint, manifest_path, complete_path = _resolve_checkpoint(value)
    manifest = _strict_json_file(manifest_path, label="checkpoint manifest")
    complete = _strict_json_file(complete_path, label="checkpoint completion record")
    if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("checkpoint manifest has an unsupported schema")
    if complete.get("schema_version") != CHECKPOINT_COMPLETE_SCHEMA_VERSION:
        raise ValueError("checkpoint completion record has an unsupported schema")
    if manifest.get("campaign_schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("checkpoint manifest campaign schema is unsupported")
    if manifest.get("state_schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("checkpoint manifest state schema is unsupported")
    digest = _sha256_file(manifest_path)
    if str(complete.get("manifest_sha256") or "") != digest:
        raise ValueError("checkpoint completion record does not bind the manifest")
    for key in ("campaign_uid", "iteration", "n_files"):
        if manifest.get(key) != complete.get(key):
            raise ValueError("checkpoint manifest/completion " + key + " mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != int(manifest.get("n_files", -1)):
        raise ValueError("checkpoint manifest file cardinality is invalid")
    seen = set()
    store = checkpoint.parent.parent
    if checkpoint.name != "iteration-" + str(int(manifest["iteration"])).zfill(6):
        raise ValueError("checkpoint directory does not match its iteration")
    if store.name != str(manifest["campaign_uid"]):
        raise ValueError("checkpoint store does not match its campaign UID")
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("checkpoint manifest file record is malformed")
        relative = item.get("path")
        digest_value = item.get("sha256")
        size = item.get("size")
        mode = item.get("mode")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise ValueError("checkpoint manifest file path is invalid or duplicated")
        path_object = Path(relative)
        if path_object.is_absolute() or ".." in path_object.parts or "\\" in relative:
            raise ValueError("checkpoint manifest file path is unsafe: " + relative)
        if (
            not isinstance(digest_value, str)
            or len(digest_value) != 64
            or any(character not in "0123456789abcdef" for character in digest_value)
        ):
            raise ValueError("checkpoint manifest object digest is malformed")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("checkpoint manifest object size is malformed")
        if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o7777:
            raise ValueError("checkpoint manifest file mode is malformed")
        seen.add(relative)
        total_bytes += size
        _verify_object(_object_path(store, digest_value), digest=digest_value, size=size)
    if total_bytes != int(manifest.get("total_bytes", -1)):
        raise ValueError("checkpoint manifest total byte count is invalid")
    return {
        "ok": True,
        "store": str(store),
        "checkpoint": str(checkpoint),
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest,
        "manifest": manifest,
        "complete": complete,
    }


def checkpoint_status(
    campaign_dir: Union[str, Path],
    destination: Union[str, Path],
) -> Dict[str, Any]:
    state = read_state(Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    store = checkpoint_store(destination, str(state.campaign_uid))
    current_path = store / "current.json"
    if not current_path.exists():
        return {
            "schema_version": 1,
            "campaign_uid": str(state.campaign_uid),
            "store": str(store),
            "status": "absent",
            "current": None,
        }
    current = _strict_json_file(current_path, label="checkpoint current pointer")
    if current.get("schema_version") != CHECKPOINT_CURRENT_SCHEMA_VERSION:
        raise ValueError("checkpoint current pointer has an unsupported schema")
    if str(current.get("campaign_uid") or "") != str(state.campaign_uid):
        raise ValueError("checkpoint current pointer campaign UID mismatch")
    relative = current.get("manifest_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("checkpoint current pointer manifest path is malformed")
    verified = verify_checkpoint(store / relative)
    if verified["manifest_sha256"] != str(current.get("manifest_sha256") or ""):
        raise ValueError("checkpoint current pointer digest mismatch")
    return {
        "schema_version": 1,
        "campaign_uid": str(state.campaign_uid),
        "store": str(store),
        "status": "verified",
        "current": current,
        "checkpoint": verified,
    }


def restore_checkpoint(
    value: Union[str, Path],
    target_empty_dir: Union[str, Path],
    *,
    apply: bool,
) -> Dict[str, Any]:
    verified = verify_checkpoint(value)
    target = lexical_absolute_path(target_empty_dir)
    reject_symlink_components(target)
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise ValueError("checkpoint restore target is not a regular directory")
        if any(target.iterdir()):
            raise ValueError("checkpoint restore target must be empty")
    if not apply:
        return {
            "schema_version": 1,
            "applied": False,
            "target": str(target),
            "checkpoint": verified,
        }
    if target.exists():
        target.rmdir()
    temporary = target.with_name(".restore-" + uuid.uuid4().hex[:12])
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        manifest = verified["manifest"]
        store = Path(verified["store"])
        for item in manifest["files"]:
            destination = temporary.joinpath(*Path(str(item["path"])).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = _object_path(store, str(item["sha256"]))
            with source.open("rb") as source_handle, destination.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=_HASH_CHUNK_BYTES)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            os.chmod(destination, int(item.get("mode", 0o600)))
            if destination.stat().st_size != int(item["size"]) or _sha256_file(destination) != str(item["sha256"]):
                raise ValueError("restored checkpoint file failed verification: " + str(destination))
        restored_state = read_state(
            temporary / ".DATA" / "ACTIVE_LEARNING" / "state.json"
        )
        if str(restored_state.campaign_uid) != str(manifest["campaign_uid"]):
            raise ValueError("restored campaign UID does not match checkpoint")
        from ..config import CampaignConfig

        CampaignConfig.from_yaml(temporary / "campaign.yaml")
        from ..execution_identity import (
            read_active_environment_generation,
            read_execution_identity,
        )

        read_execution_identity(
            temporary,
            expected_campaign_uid=str(restored_state.campaign_uid),
        )
        read_active_environment_generation(
            temporary,
            expected_campaign_uid=str(restored_state.campaign_uid),
        )
        if int(restored_state.reference_data_version) >= 0:
            from .quantum_acceptance_receipts import (
                restore_sealed_pointdir_permissions,
            )
            from ..versioning.reference_data import ReferenceDataVersioning

            reference_versioning = ReferenceDataVersioning(
                temporary / "QM_REFERENCE_DATA"
            )
            reference_index = reference_versioning.resolve(
                int(restored_state.reference_data_version),
                verification="index",
            )
            for entry in reference_index.entries:
                restore_sealed_pointdir_permissions(entry.pointdir_path)
        verify_state_referenced_artifacts(temporary, restored_state, strict_models=True)
        if int(restored_state.reference_data_version) >= 0:
            from .ferebus_row_cache import ensure_cumulative_row_caches

            reference_view = reference_versioning.resolve(
                int(restored_state.reference_data_version),
                verification="metadata",
            )
            ensure_cumulative_row_caches(temporary, reference_view)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema_version": 1,
        "applied": True,
        "target": str(target),
        "checkpoint": verified,
    }


__all__ = [
    "CHECKPOINT_COMPLETE_SCHEMA_VERSION",
    "CHECKPOINT_CURRENT_SCHEMA_VERSION",
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "checkpoint_directory",
    "checkpoint_status",
    "checkpoint_store",
    "create_checkpoint",
    "normalise_checkpoint_destination",
    "restore_checkpoint",
    "verify_checkpoint",
]
