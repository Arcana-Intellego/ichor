"""Inventory and explicit clean-up for retained ARIADNE retry outputs."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .filesystem import operational_data_dir
from .state import atomic_write_json
from ..strict_json import strict_json as json


QUARANTINE_SCHEMA_VERSION = 2
QUARANTINE_MANIFEST_FILENAME = "QUARANTINE.json"
MAX_RETAINED_QUARANTINE_ATTEMPTS = 256
MAX_RETAINED_QUARANTINE_BYTES = 100 * 1024**3


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


def _tree_size(path: Path) -> int:
    total = 0
    for root, directories, filenames in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in list(directories) + list(filenames):
            item = root_path / name
            if item.is_symlink():
                raise AriadneQuarantineError(
                    "ARIADNE quarantine contains a symlink: " + str(item)
                )
            if item.is_file():
                total += int(item.stat().st_size)
    return total


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
) -> Path:
    if len(source_paths) != len(target_paths) or not target_paths:
        raise AriadneQuarantineError(
            "ARIADNE quarantine manifest requires matched non-empty path lists"
        )
    if status not in {"prepared", "retained_failure"}:
        raise AriadneQuarantineError("ARIADNE quarantine status is invalid")
    root = quarantine_root(Path(campaign_dir))
    _contained_real_path(Path(attempt_dir), root)
    entries = []
    for source, target in zip(source_paths, target_paths):
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
        entries.append({
            "source_relative_path": _campaign_relative_source(
                Path(campaign_dir), Path(source)
            ),
            "retained_relative_path": str(
                target_path.resolve().relative_to(root.resolve())
            ),
            "bytes": _tree_size(byte_source),
        })
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
) -> Path:
    """Publish ownership evidence before moving the first retained output."""
    return _write_manifest(
        campaign_dir,
        attempt_dir,
        iteration=iteration,
        source_paths=source_paths,
        target_paths=target_paths,
        status="prepared",
    )


def write_quarantine_manifest(
    campaign_dir: Path,
    attempt_dir: Path,
    *,
    iteration: int,
    source_paths: Sequence[Path],
    target_paths: Sequence[Path],
) -> Path:
    return _write_manifest(
        campaign_dir,
        attempt_dir,
        iteration=iteration,
        source_paths=source_paths,
        target_paths=target_paths,
        status="retained_failure",
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
        if target.exists():
            actual_bytes = _tree_size(target)
            if actual_bytes != int(expected_bytes):
                raise AriadneQuarantineError(
                    "ARIADNE quarantine retained byte count mismatch"
                )
            verified_bytes += actual_bytes
            continue
        source_relative = entry.get("source_relative_path")
        if not isinstance(source_relative, str) or not source_relative:
            raise AriadneQuarantineError("ARIADNE quarantine source path is invalid")
        source = campaign_dir / Path(source_relative)
        if _campaign_relative_source(campaign_dir, source) != source_relative:
            raise AriadneQuarantineError("ARIADNE quarantine source path is not canonical")
        if not source.is_dir() or source.is_symlink():
            raise AriadneQuarantineError(
                "interrupted ARIADNE quarantine lost both source and target"
            )
        if _tree_size(source) != int(expected_bytes):
            raise AriadneQuarantineError(
                "interrupted ARIADNE quarantine source byte count mismatch"
            )
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
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8"),
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
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise AriadneQuarantineError(
                        "ARIADNE quarantine entry must be an object"
                    )
                relative = entry.get("retained_relative_path")
                expected_prefix = path.relative_to(root).as_posix() + "/"
                if (
                    not isinstance(relative, str)
                    or not relative.startswith(expected_prefix)
                    or "\\" in relative
                    or any(part in {"", ".", ".."} for part in relative.split("/"))
                ):
                    raise AriadneQuarantineError(
                        "ARIADNE quarantine entry path is invalid"
                    )
                size = entry.get("bytes")
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise AriadneQuarantineError(
                        "ARIADNE quarantine byte count is invalid"
                    )
                declared_bytes += int(size)
            if data.get("total_bytes") != declared_bytes:
                raise AriadneQuarantineError(
                    "ARIADNE quarantine total byte count mismatch"
                )
            data["declared_bytes"] = declared_bytes
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
    "write_quarantine_manifest",
]
