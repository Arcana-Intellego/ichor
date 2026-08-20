"""Inventory and explicit clean-up for retained ARIADNE retry outputs."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..ariadne_seed_tree import (
    AriadneSeedTreeContext,
    AriadneSeedTreeError,
    classify_ariadne_seed_tree,
)
from ..layout import (
    active_iteration_dir,
    active_iteration_name,
    ariadne_seeds_dir,
)
from ..versioning.manifest import sha256_file
from .filesystem import operational_data_dir
from .state import _fsync_parent_dir, atomic_write_json
from ..strict_json import strict_json as json


QUARANTINE_SCHEMA_VERSION = 2
QUARANTINE_MANIFEST_FILENAME = "QUARANTINE.json"
MAX_RETAINED_QUARANTINE_ATTEMPTS = 256
MAX_RETAINED_QUARANTINE_BYTES = 100 * 1024**3
_ATOMIC_TEMP_RE = re.compile(r"^\.t-[0-9a-f]{12}$")


class AriadneQuarantineError(RuntimeError):
    """Raised when retained retry output ownership is invalid."""


def quarantine_root(campaign_dir: Path) -> Path:
    return (
        operational_data_dir(campaign_dir)
        / "ACTIVE_LEARNING"
        / "ariadne_retry_quarantine"
    )


def _contained_real_path(path: Path, root: Path, *, require_exists: bool = True) -> Path:
    root_resolved = root.resolve()
    current = path
    while True:
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise AriadneQuarantineError(
                    "ARIADNE quarantine path contains a symlink: " + str(current)
                )
        if current == root or current.parent == current:
            break
        current = current.parent
    if require_exists and not path.exists():
        raise AriadneQuarantineError(
            "ARIADNE quarantine path is missing: " + str(path)
        )
    try:
        path.resolve().relative_to(root_resolved)
    except ValueError as exc:
        raise AriadneQuarantineError(
            "ARIADNE quarantine path escapes its root: " + str(path)
        ) from exc
    return path


def _tree_evidence(path: Path) -> tuple[int, int, str]:
    total = 0
    n_files = 0
    records = []
    pending = [(Path(path), Path("."))]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            directory_before = directory.stat(follow_symlinks=False)
            if not stat.S_ISDIR(directory_before.st_mode):
                raise AriadneQuarantineError(
                    "ARIADNE quarantine tree entry is not a directory: "
                    + str(directory)
                )
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda item: item.name)
        except OSError as exc:
            raise AriadneQuarantineError(
                "ARIADNE quarantine tree is unreadable: " + str(directory)
            ) from exc
        for entry in entries:
            item = directory / entry.name
            relative = relative_directory / entry.name
            try:
                info = item.stat(follow_symlinks=False)
            except OSError as exc:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine entry is unreadable: " + str(item)
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise AriadneQuarantineError(
                    "ARIADNE quarantine contains a symlink: " + str(item)
                )
            if stat.S_ISDIR(info.st_mode):
                records.append(("d", relative.as_posix()))
                pending.append((item, relative))
                continue
            if not stat.S_ISREG(info.st_mode):
                raise AriadneQuarantineError(
                    "ARIADNE quarantine contains a special file: " + str(item)
                )
            digest = hashlib.sha256()
            size = 0
            try:
                with item.open("rb") as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                after = item.stat(follow_symlinks=False)
            except OSError as exc:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine file is unreadable: " + str(item)
                ) from exc
            identity = (
                int(info.st_mode),
                int(info.st_dev),
                int(info.st_ino),
                int(info.st_size),
                int(info.st_mtime_ns),
                int(info.st_ctime_ns),
            )
            after_identity = (
                int(after.st_mode),
                int(after.st_dev),
                int(after.st_ino),
                int(after.st_size),
                int(after.st_mtime_ns),
                int(after.st_ctime_ns),
            )
            if identity != after_identity or size != int(info.st_size):
                raise AriadneQuarantineError(
                    "ARIADNE quarantine file changed while it was read: "
                    + str(item)
                )
            total += size
            n_files += 1
            records.append(("f", relative.as_posix(), digest.hexdigest()))
        try:
            directory_after = directory.stat(follow_symlinks=False)
        except OSError as exc:
            raise AriadneQuarantineError(
                "ARIADNE quarantine directory disappeared: " + str(directory)
            ) from exc
        before_identity = (
            int(directory_before.st_mode),
            int(directory_before.st_dev),
            int(directory_before.st_ino),
            int(directory_before.st_size),
            int(directory_before.st_mtime_ns),
            int(directory_before.st_ctime_ns),
        )
        after_identity = (
            int(directory_after.st_mode),
            int(directory_after.st_dev),
            int(directory_after.st_ino),
            int(directory_after.st_size),
            int(directory_after.st_mtime_ns),
            int(directory_after.st_ctime_ns),
        )
        if before_identity != after_identity:
            raise AriadneQuarantineError(
                "ARIADNE quarantine tree changed while it was read: "
                + str(directory)
            )
    tree = hashlib.sha256()
    for record in sorted(records, key=lambda item: (item[1], item[0])):
        tree.update(repr(record).encode("utf-8"))
        tree.update(b"\n")
    return int(total), int(n_files), tree.hexdigest()


def _tree_size(path: Path) -> int:
    total = 0
    pending = [Path(path)]
    while pending:
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
            with entries:
                items = list(entries)
        except OSError as exc:
            raise AriadneQuarantineError(
                "ARIADNE quarantine tree is unreadable: " + str(directory)
            ) from exc
        for entry in items:
            item = directory / entry.name
            try:
                info = item.stat(follow_symlinks=False)
            except OSError as exc:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine entry is unreadable: " + str(item)
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise AriadneQuarantineError(
                    "ARIADNE quarantine contains a symlink: " + str(item)
                )
            if stat.S_ISDIR(info.st_mode):
                pending.append(item)
            elif stat.S_ISREG(info.st_mode):
                total += int(info.st_size)
            else:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine contains a special file: " + str(item)
                )
    return int(total)


def _campaign_relative_source(campaign_dir: Path, source: Path) -> str:
    campaign = Path(campaign_dir).resolve()
    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = campaign / candidate
    current = candidate
    while True:
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise AriadneQuarantineError(
                    "ARIADNE quarantine source contains a symlink: " + str(current)
                )
        if current == campaign or current.parent == current:
            break
        current = current.parent
    try:
        return candidate.resolve().relative_to(campaign).as_posix()
    except ValueError as exc:
        raise AriadneQuarantineError(
            "ARIADNE quarantine source escapes the campaign: " + str(source)
        ) from exc


def _write_manifest(
    campaign_dir: Path,
    attempt_dir: Path,
    *,
    iteration: int,
    source_paths: Sequence[Path],
    target_paths: Sequence[Path],
    status: str,
    tree_sha256_values: Optional[Sequence[str]] = None,
) -> Path:
    if len(source_paths) != len(target_paths) or not target_paths:
        raise AriadneQuarantineError(
            "ARIADNE quarantine manifest requires matched non-empty path lists"
        )
    if status not in {"prepared", "retained_failure"}:
        raise AriadneQuarantineError("ARIADNE quarantine status is invalid")
    if tree_sha256_values is not None and len(tree_sha256_values) != len(
        source_paths
    ):
        raise AriadneQuarantineError(
            "ARIADNE quarantine tree identities do not match its paths"
        )
    root = quarantine_root(Path(campaign_dir))
    _contained_real_path(Path(attempt_dir), root)
    entries = []
    for index, (source, target) in enumerate(zip(source_paths, target_paths)):
        target_path = Path(target)
        _contained_real_path(
            target_path,
            root,
            require_exists=status == "retained_failure",
        )
        if target_path.parent != Path(attempt_dir):
            raise AriadneQuarantineError(
                "ARIADNE quarantine target crosses attempts"
            )
        byte_source = target_path if target_path.exists() else Path(source)
        if not byte_source.is_dir() or byte_source.is_symlink():
            raise AriadneQuarantineError(
                "ARIADNE quarantine evidence source is not a real directory: "
                + str(byte_source)
            )
        entry = {
            "source_relative_path": _campaign_relative_source(
                Path(campaign_dir), Path(source)
            ),
            "retained_relative_path": str(
                target_path.resolve().relative_to(root.resolve()).as_posix()
            ),
            "bytes": _tree_size(byte_source),
        }
        if tree_sha256_values is not None:
            tree_sha256 = str(tree_sha256_values[index])
            if len(tree_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in tree_sha256
            ):
                raise AriadneQuarantineError(
                    "ARIADNE quarantine tree SHA-256 is invalid"
                )
            observed_tree_sha256 = _tree_evidence(byte_source)[2]
            if observed_tree_sha256 != tree_sha256:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine tree SHA-256 changed before publication"
                )
            entry["tree_sha256"] = tree_sha256
        entries.append(entry)
    payload = {
        "schema_version": QUARANTINE_SCHEMA_VERSION,
        "attempt_id": Path(attempt_dir).name,
        "iteration": int(iteration),
        "status": status,
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "total_bytes": int(sum(int(item["bytes"]) for item in entries)),
    }
    path = Path(attempt_dir) / QUARANTINE_MANIFEST_FILENAME
    atomic_write_json(path, payload)
    return path


def prepare_quarantine_manifest(
    campaign_dir: Path,
    attempt_dir: Path,
    *,
    iteration: int,
    source_paths: Sequence[Path],
    target_paths: Sequence[Path],
    tree_sha256_values: Optional[Sequence[str]] = None,
) -> Path:
    """Publish ownership evidence before moving the first retained output."""
    return _write_manifest(
        campaign_dir,
        attempt_dir,
        iteration=iteration,
        source_paths=source_paths,
        target_paths=target_paths,
        status="prepared",
        tree_sha256_values=tree_sha256_values,
    )


def write_quarantine_manifest(
    campaign_dir: Path,
    attempt_dir: Path,
    *,
    iteration: int,
    source_paths: Sequence[Path],
    target_paths: Sequence[Path],
    tree_sha256_values: Optional[Sequence[str]] = None,
) -> Path:
    return _write_manifest(
        campaign_dir,
        attempt_dir,
        iteration=iteration,
        source_paths=source_paths,
        target_paths=target_paths,
        status="retained_failure",
        tree_sha256_values=tree_sha256_values,
    )


def _read_manifest(path: Path, campaign_dir: Path) -> Dict[str, Any]:
    root = quarantine_root(campaign_dir)
    _contained_real_path(path, root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AriadneQuarantineError(
            "ARIADNE quarantine manifest is unreadable: " + str(path)
        ) from exc
    if not isinstance(payload, Mapping):
        raise AriadneQuarantineError("ARIADNE quarantine manifest must be an object")
    data = dict(payload)
    if data.get("schema_version") != QUARANTINE_SCHEMA_VERSION:
        raise AriadneQuarantineError("unsupported ARIADNE quarantine schema")
    attempt_id = data.get("attempt_id")
    if not isinstance(attempt_id, str) or attempt_id != path.parent.name:
        raise AriadneQuarantineError("ARIADNE quarantine attempt identity mismatch")
    iteration = data.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise AriadneQuarantineError("ARIADNE quarantine iteration is invalid")
    status = data.get("status")
    if status not in {"prepared", "retained_failure"}:
        raise AriadneQuarantineError("ARIADNE quarantine status is invalid")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise AriadneQuarantineError("ARIADNE quarantine entries are invalid")
    verified_bytes = 0
    pending_sources = 0
    declared_bytes = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise AriadneQuarantineError("ARIADNE quarantine entry must be an object")
        relative = entry.get("retained_relative_path")
        if not isinstance(relative, str) or not relative:
            raise AriadneQuarantineError("ARIADNE quarantine entry path is invalid")
        target = root / relative
        _contained_real_path(
            target,
            root,
            require_exists=status == "retained_failure",
        )
        if target.parent != path.parent:
            raise AriadneQuarantineError("ARIADNE quarantine entry crosses attempts")
        expected_bytes = entry.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise AriadneQuarantineError("ARIADNE quarantine byte count is invalid")
        declared_bytes += int(expected_bytes)
        source_relative = entry.get("source_relative_path")
        if not isinstance(source_relative, str) or not source_relative:
            raise AriadneQuarantineError("ARIADNE quarantine source path is invalid")
        source = campaign_dir / Path(source_relative)
        if _campaign_relative_source(campaign_dir, source) != source_relative:
            raise AriadneQuarantineError("ARIADNE quarantine source path is not canonical")
        source_exists = source.exists() or source.is_symlink()
        target_exists = target.exists() or target.is_symlink()
        if source_exists and target_exists:
            raise AriadneQuarantineError(
                "ARIADNE quarantine contains both source and retained evidence"
            )
        evidence_path = target if target_exists else source
        if evidence_path.is_symlink() or not evidence_path.is_dir():
            raise AriadneQuarantineError(
                "interrupted ARIADNE quarantine lost both source and target"
            )
        if _tree_size(evidence_path) != int(expected_bytes):
            raise AriadneQuarantineError(
                "ARIADNE quarantine evidence byte count mismatch"
            )
        expected_tree_sha256 = entry.get("tree_sha256")
        if expected_tree_sha256 is not None:
            if (
                not isinstance(expected_tree_sha256, str)
                or len(expected_tree_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_tree_sha256
                )
            ):
                raise AriadneQuarantineError(
                    "ARIADNE quarantine tree SHA-256 is invalid"
                )
            if _tree_evidence(evidence_path)[2] != expected_tree_sha256:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine evidence tree SHA-256 mismatch"
                )
        if target_exists:
            verified_bytes += int(expected_bytes)
        else:
            pending_sources += 1
    total_bytes = data.get("total_bytes")
    if (
        isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or int(total_bytes) != declared_bytes
    ):
        raise AriadneQuarantineError("ARIADNE quarantine total byte count mismatch")
    data["verified_bytes"] = int(verified_bytes)
    data["pending_sources"] = int(pending_sources)
    data["manifest_path"] = str(path)
    data["attempt_path"] = str(path.parent)
    return data


def inventory_quarantine(campaign_dir: Path) -> Dict[str, Any]:
    campaign = Path(campaign_dir)
    root = quarantine_root(campaign)
    if not root.exists():
        return {"attempts": [], "errors": [], "total_bytes": 0}
    _campaign_relative_source(campaign, root)
    _contained_real_path(root, root)
    attempts: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for path in sorted(root.glob("iteration-*/*")):
        if not path.is_dir() or path.is_symlink():
            errors.append({
                "path": str(path),
                "error": "quarantine attempt is not a real directory",
            })
            continue
        manifest = path / QUARANTINE_MANIFEST_FILENAME
        try:
            attempts.append(_read_manifest(manifest, campaign))
        except Exception as exc:
            errors.append({
                "path": str(path),
                "error": type(exc).__name__ + ": " + str(exc),
            })
    return {
        "attempts": attempts,
        "errors": errors,
        "total_bytes": int(sum(int(item["verified_bytes"]) for item in attempts)),
    }


def inventory_quarantine_authority(campaign_dir: Path) -> Dict[str, Any]:
    """Read bounded quarantine control evidence without walking retained trees."""
    campaign = Path(campaign_dir)
    root = quarantine_root(campaign)
    if not root.exists():
        return {"attempts": [], "errors": [], "total_bytes": 0}
    if root.is_symlink() or not root.is_dir():
        return {
            "attempts": [],
            "errors": [{
                "path": str(root),
                "error": "quarantine root is not a real directory",
            }],
            "total_bytes": 0,
        }
    try:
        _campaign_relative_source(campaign, root)
        _contained_real_path(root, root)
    except AriadneQuarantineError as exc:
        return {
            "attempts": [],
            "errors": [{"path": str(root), "error": str(exc)}],
            "total_bytes": 0,
        }
    attempts: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for path in sorted(root.glob("iteration-*/*")):
        if path.is_symlink() or not path.is_dir():
            errors.append({
                "path": str(path),
                "error": "quarantine attempt is not a real directory",
            })
            continue
        manifest_path = path / QUARANTINE_MANIFEST_FILENAME
        try:
            _contained_real_path(path, root)
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise AriadneQuarantineError(
                    "ARIADNE quarantine manifest is not a regular file"
                )
            before = manifest_path.stat(follow_symlinks=False)
            raw_manifest = manifest_path.read_bytes()
            after = manifest_path.stat(follow_symlinks=False)
            before_identity = (
                int(before.st_mode),
                int(before.st_dev),
                int(before.st_ino),
                int(before.st_size),
                int(before.st_mtime_ns),
                int(before.st_ctime_ns),
            )
            after_identity = (
                int(after.st_mode),
                int(after.st_dev),
                int(after.st_ino),
                int(after.st_size),
                int(after.st_mtime_ns),
                int(after.st_ctime_ns),
            )
            if before_identity != after_identity:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine manifest changed while it was read"
                )
            payload = json.loads(
                raw_manifest.decode("utf-8"),
                source=manifest_path,
            )
            if not isinstance(payload, Mapping):
                raise AriadneQuarantineError(
                    "ARIADNE quarantine manifest must be an object"
                )
            data = dict(payload)
            if data.get("schema_version") != QUARANTINE_SCHEMA_VERSION:
                raise AriadneQuarantineError(
                    "unsupported ARIADNE quarantine schema"
                )
            if data.get("attempt_id") != path.name:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine attempt identity mismatch"
                )
            iteration = data.get("iteration")
            if (
                isinstance(iteration, bool)
                or not isinstance(iteration, int)
                or iteration < 0
            ):
                raise AriadneQuarantineError(
                    "ARIADNE quarantine iteration is invalid"
                )
            if data.get("status") not in {"prepared", "retained_failure"}:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine status is invalid"
                )
            entries = data.get("entries")
            if not isinstance(entries, list) or not entries:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine entries are invalid"
                )
            declared_bytes = 0
            source_paths = set()
            retained_paths = set()
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise AriadneQuarantineError(
                        "ARIADNE quarantine entry must be an object"
                    )
                source_relative = entry.get("source_relative_path")
                if (
                    not isinstance(source_relative, str)
                    or not source_relative
                    or "\\" in source_relative
                    or Path(source_relative).is_absolute()
                    or any(
                        part in {"", ".", ".."}
                        for part in source_relative.split("/")
                    )
                    or source_relative in source_paths
                ):
                    raise AriadneQuarantineError(
                        "ARIADNE quarantine source path is invalid"
                    )
                source_paths.add(source_relative)
                raw_relative = entry.get("retained_relative_path")
                relative = (
                    raw_relative.replace("\\", "/")
                    if isinstance(raw_relative, str) and os.sep == "\\"
                    else raw_relative
                )
                expected_prefix = path.relative_to(root).as_posix() + "/"
                if (
                    not isinstance(relative, str)
                    or not relative.startswith(expected_prefix)
                    or (os.sep != "\\" and "\\" in relative)
                    or any(part in {"", ".", ".."} for part in relative.split("/"))
                    or relative in retained_paths
                ):
                    raise AriadneQuarantineError(
                        "ARIADNE quarantine entry path is invalid"
                    )
                retained_paths.add(relative)
                size = entry.get("bytes")
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise AriadneQuarantineError(
                        "ARIADNE quarantine byte count is invalid"
                    )
                tree_sha256 = entry.get("tree_sha256")
                if tree_sha256 is not None and (
                    not isinstance(tree_sha256, str)
                    or len(tree_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in tree_sha256
                    )
                ):
                    raise AriadneQuarantineError(
                        "ARIADNE quarantine tree SHA-256 is invalid"
                    )
                declared_bytes += int(size)
            if data.get("total_bytes") != declared_bytes:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine total byte count mismatch"
                )
            data["declared_bytes"] = declared_bytes
            data["manifest_sha256"] = hashlib.sha256(raw_manifest).hexdigest()
            data["manifest_path"] = str(manifest_path)
            data["attempt_path"] = str(path)
            attempts.append(data)
        except Exception as exc:
            errors.append({
                "path": str(path),
                "error": type(exc).__name__ + ": " + str(exc),
            })
    return {
        "attempts": attempts,
        "errors": errors,
        "total_bytes": int(
            sum(int(item.get("declared_bytes", 0)) for item in attempts)
        ),
    }


def ensure_quarantine_capacity(
    campaign_dir: Path,
    incoming_paths: Sequence[Path],
    *,
    max_attempts: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> Dict[str, int]:
    """Fail before a retry can make retained ARIADNE evidence unbounded."""
    attempt_limit = (
        MAX_RETAINED_QUARANTINE_ATTEMPTS
        if max_attempts is None
        else int(max_attempts)
    )
    byte_limit = (
        MAX_RETAINED_QUARANTINE_BYTES if max_bytes is None else int(max_bytes)
    )
    if attempt_limit < 1 or byte_limit < 1:
        raise AriadneQuarantineError(
            "ARIADNE quarantine limits must be positive integers"
        )
    inventory = inventory_quarantine(Path(campaign_dir))
    if inventory["errors"]:
        raise AriadneQuarantineError(
            "ARIADNE quarantine retention is blocked by invalid ownership evidence"
        )
    interrupted = [
        item
        for item in inventory["attempts"]
        if str(item.get("status")) == "prepared"
    ]
    if interrupted:
        raise AriadneQuarantineError(
            "ARIADNE quarantine contains an interrupted prepared attempt; "
            "review and clean it explicitly before retrying"
        )
    incoming_bytes = int(sum(_tree_size(Path(path)) for path in incoming_paths))
    projected_attempts = len(inventory["attempts"]) + (1 if incoming_paths else 0)
    projected_bytes = int(inventory["total_bytes"]) + incoming_bytes
    clean_hint = (
        " Run 'ichor-al-daemon reconcile --clean-ariadne-quarantine --apply' "
        "after reviewing the retained attempts."
    )
    if projected_attempts > attempt_limit:
        raise AriadneQuarantineError(
            "ARIADNE quarantine attempt limit would be exceeded "
            f"({projected_attempts} > {attempt_limit})." + clean_hint
        )
    if projected_bytes > byte_limit:
        raise AriadneQuarantineError(
            "ARIADNE quarantine byte limit would be exceeded "
            f"({projected_bytes} > {byte_limit})." + clean_hint
        )
    return {
        "existing_attempts": len(inventory["attempts"]),
        "existing_bytes": int(inventory["total_bytes"]),
        "incoming_bytes": incoming_bytes,
        "projected_attempts": projected_attempts,
        "projected_bytes": projected_bytes,
        "max_attempts": attempt_limit,
        "max_bytes": byte_limit,
    }


def _transaction_residue_attempt_id(
    *,
    campaign_uid: str,
    iteration: int,
    task_map_sha256: str,
    authority_identity: str,
) -> str:
    identity = str(authority_identity)
    if (
        not identity
        or any(ord(character) < 32 or ord(character) == 127 for character in identity)
    ):
        raise AriadneQuarantineError(
            "ARIADNE transaction-residue authority identity is invalid"
        )
    payload = "\0".join(
        (
            "ariadne_transaction_residue_v1",
            str(campaign_uid),
            str(int(iteration)),
            str(task_map_sha256),
            identity,
        )
    ).encode("utf-8")
    return "transaction-residue-" + hashlib.sha256(payload).hexdigest()[:32]


def _transaction_residue_attempts(
    campaign_dir: Path,
    *,
    iteration: int,
    intended_attempt_id: str,
) -> List[Dict[str, Any]]:
    inventory = inventory_quarantine(campaign_dir)
    if inventory["errors"]:
        raise AriadneQuarantineError(
            "ARIADNE quarantine retention is blocked by invalid ownership evidence"
        )
    iteration_attempts = [
        item
        for item in inventory["attempts"]
        if int(item["iteration"]) == int(iteration)
        and str(item["attempt_id"]).startswith("transaction-residue-")
    ]
    attempts = [
        item
        for item in iteration_attempts
        if str(item["attempt_id"]) == str(intended_attempt_id)
    ]
    if len(attempts) > 1:
        raise AriadneQuarantineError(
            "duplicate ARIADNE transaction-residue attempt identity"
        )
    foreign_prepared = [
        item
        for item in inventory["attempts"]
        if str(item.get("status")) == "prepared"
        and str(item.get("attempt_id")) != str(intended_attempt_id)
    ]
    if foreign_prepared:
        raise AriadneQuarantineError(
            "ARIADNE quarantine contains an unrelated interrupted prepared attempt"
        )
    return attempts


def _clean_deterministic_attempt_entries(
    campaign_dir: Path,
    attempt_dir: Path,
    *,
    allowed_names: Sequence[str],
) -> None:
    """Remove only interrupted atomic-manifest temps from one exact attempt."""
    campaign = Path(campaign_dir).resolve()
    root = quarantine_root(campaign)
    _campaign_relative_source(campaign, attempt_dir)
    _contained_real_path(attempt_dir, root)
    allowed = {str(name) for name in allowed_names}
    changed = False
    try:
        with os.scandir(attempt_dir) as scan:
            entries = sorted(scan, key=lambda item: item.name)
    except OSError as exc:
        raise AriadneQuarantineError(
            "ARIADNE transaction-residue attempt is unreadable"
        ) from exc
    for entry in entries:
        if entry.name in allowed:
            continue
        item = attempt_dir / entry.name
        try:
            info = item.stat(follow_symlinks=False)
        except OSError as exc:
            raise AriadneQuarantineError(
                "ARIADNE transaction-residue attempt entry is unreadable"
            ) from exc
        if (
            not _ATOMIC_TEMP_RE.fullmatch(entry.name)
            or not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
        ):
            raise AriadneQuarantineError(
                "ARIADNE transaction-residue attempt contains unknown evidence: "
                + entry.name
            )
        item.unlink()
        changed = True
    if changed:
        _fsync_parent_dir(attempt_dir / QUARANTINE_MANIFEST_FILENAME)


def _repair_pre_manifest_attempt(
    campaign_dir: Path,
    attempt_dir: Path,
    *,
    source_residue_present: bool,
) -> None:
    """Repair only a deterministic attempt interrupted before its manifest."""
    inspection = _inspect_pre_manifest_attempt(
        campaign_dir,
        attempt_dir,
        source_residue_present=source_residue_present,
    )
    if inspection is None:
        return
    for temporary_path in inspection["temporary_paths"]:
        Path(temporary_path).unlink()
    if inspection["temporary_paths"]:
        _fsync_parent_dir(attempt_dir / QUARANTINE_MANIFEST_FILENAME)
    try:
        attempt_dir.rmdir()
    except OSError as exc:
        raise AriadneQuarantineError(
            "ARIADNE transaction-residue pre-manifest attempt is not empty"
        ) from exc
    _fsync_parent_dir(attempt_dir)


def _inspect_pre_manifest_attempt(
    campaign_dir: Path,
    attempt_dir: Path,
    *,
    source_residue_present: bool,
) -> Optional[Dict[str, Any]]:
    """Classify one exact deterministic attempt without modifying it."""
    if not attempt_dir.exists() and not attempt_dir.is_symlink():
        return None
    if attempt_dir.is_symlink() or not attempt_dir.is_dir():
        raise AriadneQuarantineError(
            "ARIADNE transaction-residue attempt path is unsafe"
        )
    campaign = Path(campaign_dir).resolve()
    root = quarantine_root(campaign)
    _campaign_relative_source(campaign, attempt_dir)
    _contained_real_path(attempt_dir, root)
    manifest = attempt_dir / QUARANTINE_MANIFEST_FILENAME
    if manifest.exists() or manifest.is_symlink():
        return None
    if not source_residue_present:
        raise AriadneQuarantineError(
            "ARIADNE transaction-residue attempt lost its prepared manifest"
        )
    temporary_paths = []
    try:
        with os.scandir(attempt_dir) as scan:
            entries = sorted(scan, key=lambda item: item.name)
    except OSError as exc:
        raise AriadneQuarantineError(
            "ARIADNE transaction-residue attempt is unreadable"
        ) from exc
    for entry in entries:
        item = attempt_dir / entry.name
        try:
            info = item.stat(follow_symlinks=False)
        except OSError as exc:
            raise AriadneQuarantineError(
                "ARIADNE transaction-residue attempt entry is unreadable"
            ) from exc
        if (
            not _ATOMIC_TEMP_RE.fullmatch(entry.name)
            or not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
        ):
            raise AriadneQuarantineError(
                "ARIADNE transaction-residue attempt contains unknown evidence: "
                + entry.name
            )
        temporary_paths.append(str(item))
    return {
        "attempt_path": str(attempt_dir),
        "temporary_paths": temporary_paths,
    }


def _residues_from_manifest(
    campaign_dir: Path,
    task_map: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    tasks = {
        int(task["array_task_id"]): dict(task)
        for task in task_map.get("tasks", [])
        if isinstance(task, Mapping)
    }
    entries = []
    expected_source_root = ariadne_seeds_dir(
        active_iteration_dir(
            Path(campaign_dir),
            int(task_map.get("iteration", -1)),
        )
    )
    for raw in manifest.get("entries", []):
        source_relative = str(raw.get("source_relative_path") or "")
        source = Path(campaign_dir) / Path(source_relative)
        if (
            _campaign_relative_source(Path(campaign_dir), source)
            != source_relative
            or source.parent.resolve(strict=False)
            != expected_source_root.resolve(strict=False)
        ):
            raise AriadneQuarantineError(
                "ARIADNE quarantine residue source path is noncanonical"
            )
        try:
            from ..ariadne_seed_tree import (
                parse_ariadne_transaction_residue_name,
            )

            seed_id, task_id, pid = parse_ariadne_transaction_residue_name(
                source.name
            )
        except AriadneSeedTreeError as exc:
            raise AriadneQuarantineError(str(exc)) from exc
        task = tasks.get(task_id)
        if task is None or int(task.get("seed_id", -1)) != seed_id:
            raise AriadneQuarantineError(
                "ARIADNE quarantine residue no longer matches TASK_MAP"
            )
        retained_relative = str(raw.get("retained_relative_path") or "")
        if os.sep == "\\":
            retained_relative = retained_relative.replace("\\", "/")
        if retained_relative.rsplit("/", 1)[-1] != source.name:
            raise AriadneQuarantineError(
                "ARIADNE quarantine residue target identity is noncanonical"
            )
        entries.append(
            {
                "source": source,
                "seed_id": seed_id,
                "array_task_id": task_id,
                "pid": pid,
                "bytes": int(raw["bytes"]),
                "tree_sha256": raw.get("tree_sha256"),
            }
        )
    return entries


def retain_ariadne_transaction_residue(
    campaign_dir: Path,
    *,
    iteration: int,
    task_map: Mapping[str, Any],
    campaign_uid: str,
    authority_identity: str,
    context: Optional[AriadneSeedTreeContext] = None,
    expected_attempt_manifest_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Retain authenticated runner transaction residue without deleting evidence."""
    campaign = Path(campaign_dir).resolve()
    iter_dir = active_iteration_dir(campaign, int(iteration))
    task_map_path = iter_dir / "ariadne" / "TASK_MAP.json"
    if task_map_path.is_symlink() or not task_map_path.is_file():
        raise AriadneQuarantineError(
            "ARIADNE transaction-residue task map is unavailable"
        )
    from ..seed_identity import read_ariadne_task_map

    canonical_task_map = read_ariadne_task_map(
        iter_dir,
        expected_iteration=int(iteration),
    )
    if dict(canonical_task_map) != dict(task_map):
        raise AriadneQuarantineError(
            "ARIADNE transaction-residue task map changed after classification"
        )
    if str(canonical_task_map.get("campaign_uid") or "") != str(campaign_uid):
        raise AriadneQuarantineError(
            "ARIADNE transaction-residue task-map campaign UID mismatch"
        )
    task_map = canonical_task_map
    task_map_sha256 = sha256_file(task_map_path)
    current_context = context or classify_ariadne_seed_tree(iter_dir, task_map)
    intended_attempt_id = _transaction_residue_attempt_id(
        campaign_uid=str(campaign_uid),
        iteration=int(iteration),
        task_map_sha256=task_map_sha256,
        authority_identity=str(authority_identity),
    )
    intended_attempt_dir = (
        quarantine_root(campaign)
        / active_iteration_name(int(iteration))
        / intended_attempt_id
    )
    _repair_pre_manifest_attempt(
        campaign,
        intended_attempt_dir,
        source_residue_present=bool(current_context.residues),
    )
    attempts = _transaction_residue_attempts(
        campaign,
        iteration=int(iteration),
        intended_attempt_id=intended_attempt_id,
    )
    existing = attempts[0] if attempts else None

    if existing is not None:
        if expected_attempt_manifest_sha256 is not None and sha256_file(
            Path(str(existing["manifest_path"]))
        ) != str(expected_attempt_manifest_sha256):
            raise AriadneQuarantineError(
                "ARIADNE transaction-residue manifest changed after inspection"
            )
        manifest_entries = _residues_from_manifest(
            campaign,
            task_map,
            existing,
        )
        current_names = {residue.name for residue in current_context.residues}
        source_names = {entry["source"].name for entry in manifest_entries}
        if not current_names.issubset(source_names):
            raise AriadneQuarantineError(
                "ARIADNE transaction residue conflicts with its prepared quarantine"
            )
        attempt_dir = Path(str(existing["attempt_path"]))
        source_paths = [entry["source"] for entry in manifest_entries]
        target_paths = [attempt_dir / source.name for source in source_paths]
        _clean_deterministic_attempt_entries(
            campaign,
            attempt_dir,
            allowed_names=(
                QUARANTINE_MANIFEST_FILENAME,
                *[path.name for path in target_paths],
            ),
        )
        tree_sha256_values = [
            str(entry["tree_sha256"] or "") for entry in manifest_entries
        ]
        if any(not value for value in tree_sha256_values):
            raise AriadneQuarantineError(
                "legacy prepared quarantine cannot be replayed automatically"
            )
        if str(existing.get("status")) == "retained_failure":
            if current_names:
                raise AriadneQuarantineError(
                    "finalized ARIADNE quarantine still has source residue"
                )
            return {
                "changed": False,
                "attempt_id": str(existing["attempt_id"]),
                "attempt_path": str(attempt_dir),
                "manifest_path": str(existing["manifest_path"]),
                "retained_paths": [str(path) for path in target_paths],
                "residue_count": len(manifest_entries),
                "task_ids": sorted(
                    int(entry["array_task_id"])
                    for entry in manifest_entries
                ),
                "seed_ids": sorted(
                    int(entry["seed_id"]) for entry in manifest_entries
                ),
                "total_bytes": int(existing["total_bytes"]),
                "task_map_sha256": task_map_sha256,
            }
    elif current_context.residues:
        attempt_dir = intended_attempt_dir
        source_paths = [residue.path for residue in current_context.residues]
        target_paths = [attempt_dir / source.name for source in source_paths]
        tree_sha256_values = [_tree_evidence(source)[2] for source in source_paths]
    else:
        return {
            "changed": False,
            "attempt_id": None,
            "attempt_path": None,
            "manifest_path": None,
            "retained_paths": [],
            "residue_count": 0,
            "task_ids": [],
            "seed_ids": [],
            "total_bytes": 0,
        }

    inventory = inventory_quarantine(campaign)
    declared_existing_bytes = int(
        sum(int(item.get("total_bytes", 0)) for item in inventory["attempts"])
    )
    incoming_bytes = int(
        sum(_tree_size(source) for source in source_paths if source.exists())
    )
    projected_attempts = len(inventory["attempts"]) + (0 if existing else 1)
    projected_bytes = declared_existing_bytes + (0 if existing else incoming_bytes)
    if projected_attempts > MAX_RETAINED_QUARANTINE_ATTEMPTS:
        raise AriadneQuarantineError(
            "ARIADNE quarantine attempt limit would be exceeded"
        )
    if projected_bytes > MAX_RETAINED_QUARANTINE_BYTES:
        raise AriadneQuarantineError(
            "ARIADNE quarantine byte limit would be exceeded"
        )

    if existing is None:
        current_context.assert_unchanged(task_map)
        root = quarantine_root(campaign)
        _campaign_relative_source(campaign, attempt_dir.parent)
        _contained_real_path(
            attempt_dir.parent,
            root,
            require_exists=False,
        )
        attempt_dir.parent.mkdir(parents=True, exist_ok=True)
        if attempt_dir.parent.is_symlink():
            raise AriadneQuarantineError(
                "ARIADNE quarantine iteration root is a symlink"
            )
        attempt_dir.mkdir(exist_ok=False)
        prepare_quarantine_manifest(
            campaign,
            attempt_dir,
            iteration=int(iteration),
            source_paths=source_paths,
            target_paths=target_paths,
            tree_sha256_values=tree_sha256_values,
        )

    changed = False
    for source, target, expected_tree_sha256 in zip(
        source_paths,
        target_paths,
        tree_sha256_values,
    ):
        source_exists = source.exists() or source.is_symlink()
        target_exists = target.exists() or target.is_symlink()
        if source_exists and target_exists:
            raise AriadneQuarantineError(
                "ARIADNE transaction residue exists at source and target"
            )
        evidence = source if source_exists else target
        if evidence.is_symlink() or not evidence.is_dir():
            raise AriadneQuarantineError(
                "ARIADNE transaction residue was lost during quarantine"
            )
        observed_bytes, _n_files, observed_tree_sha256 = _tree_evidence(evidence)
        if observed_tree_sha256 != expected_tree_sha256:
            raise AriadneQuarantineError(
                "ARIADNE transaction residue changed during quarantine"
            )
        if source_exists:
            if int(source.stat(follow_symlinks=False).st_dev) != int(
                attempt_dir.parent.stat(follow_symlinks=False).st_dev
            ):
                raise AriadneQuarantineError(
                    "ARIADNE transaction residue quarantine crosses filesystems"
                )
            os.replace(source, target)
            _fsync_parent_dir(source)
            _fsync_parent_dir(target)
            changed = True
        target_bytes, _target_files, target_tree_sha256 = _tree_evidence(target)
        if (
            target_bytes != observed_bytes
            or target_tree_sha256 != expected_tree_sha256
        ):
            raise AriadneQuarantineError(
                "retained ARIADNE transaction residue failed validation"
            )

    manifest_path = write_quarantine_manifest(
        campaign,
        attempt_dir,
        iteration=int(iteration),
        source_paths=source_paths,
        target_paths=target_paths,
        tree_sha256_values=tree_sha256_values,
    )
    final = _read_manifest(manifest_path, campaign)
    residue_entries = _residues_from_manifest(campaign, task_map, final)
    return {
        "changed": bool(changed),
        "attempt_id": str(final["attempt_id"]),
        "attempt_path": str(attempt_dir),
        "manifest_path": str(manifest_path),
        "retained_paths": [str(path) for path in target_paths],
        "residue_count": len(residue_entries),
        "task_ids": sorted(
            int(entry["array_task_id"]) for entry in residue_entries
        ),
        "seed_ids": sorted(int(entry["seed_id"]) for entry in residue_entries),
        "total_bytes": int(final["total_bytes"]),
        "task_map_sha256": task_map_sha256,
    }


def clean_quarantine(
    campaign_dir: Path,
    *,
    attempt_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    inventory = inventory_quarantine(Path(campaign_dir))
    if inventory["errors"]:
        raise AriadneQuarantineError(
            "ARIADNE quarantine clean-up is blocked by invalid ownership evidence"
        )
    selected = None if attempt_ids is None else {str(value) for value in attempt_ids}
    removed = []
    root = quarantine_root(Path(campaign_dir))
    for item in inventory["attempts"]:
        if selected is not None and str(item["attempt_id"]) not in selected:
            continue
        attempt_path = Path(str(item["attempt_path"]))
        _contained_real_path(attempt_path, root)
        shutil.rmtree(attempt_path)
        removed.append(str(attempt_path))
    for iteration_dir in sorted(root.glob("iteration-*"), reverse=True):
        if iteration_dir.is_dir() and not any(iteration_dir.iterdir()):
            iteration_dir.rmdir()
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()
    return removed


__all__ = [
    "AriadneQuarantineError",
    "MAX_RETAINED_QUARANTINE_ATTEMPTS",
    "MAX_RETAINED_QUARANTINE_BYTES",
    "clean_quarantine",
    "ensure_quarantine_capacity",
    "inventory_quarantine",
    "inventory_quarantine_authority",
    "prepare_quarantine_manifest",
    "quarantine_root",
    "retain_ariadne_transaction_residue",
    "write_quarantine_manifest",
]
