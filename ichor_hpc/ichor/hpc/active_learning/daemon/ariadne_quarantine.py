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


QUARANTINE_SCHEMA_VERSION = 1
QUARANTINE_MANIFEST_FILENAME = "QUARANTINE.json"


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


def write_quarantine_manifest(
    campaign_dir: Path,
    attempt_dir: Path,
    *,
    iteration: int,
    source_paths: Sequence[Path],
    target_paths: Sequence[Path],
) -> Path:
    if len(source_paths) != len(target_paths) or not target_paths:
        raise AriadneQuarantineError(
            "ARIADNE quarantine manifest requires matched non-empty path lists"
        )
    root = quarantine_root(Path(campaign_dir))
    _contained_real_path(Path(attempt_dir), root)
    entries = []
    for source, target in zip(source_paths, target_paths):
        _contained_real_path(Path(target), root)
        entries.append({
            "source_path": str(Path(source)),
            "retained_relative_path": str(Path(target).resolve().relative_to(root.resolve())),
            "bytes": _tree_size(Path(target)),
        })
    created = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": QUARANTINE_SCHEMA_VERSION,
        "attempt_id": Path(attempt_dir).name,
        "iteration": int(iteration),
        "status": "retained_failure",
        "created_at_iso": created,
        "entries": entries,
        "total_bytes": int(sum(int(item["bytes"]) for item in entries)),
    }
    path = Path(attempt_dir) / QUARANTINE_MANIFEST_FILENAME
    atomic_write_json(path, payload)
    return path


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
    if data.get("status") != "retained_failure":
        raise AriadneQuarantineError("ARIADNE quarantine status is invalid")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise AriadneQuarantineError("ARIADNE quarantine entries are invalid")
    verified_bytes = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise AriadneQuarantineError("ARIADNE quarantine entry must be an object")
        relative = entry.get("retained_relative_path")
        if not isinstance(relative, str) or not relative:
            raise AriadneQuarantineError("ARIADNE quarantine entry path is invalid")
        target = root / relative
        _contained_real_path(target, root)
        if target.parent != path.parent:
            raise AriadneQuarantineError("ARIADNE quarantine entry crosses attempts")
        verified_bytes += _tree_size(target)
    data["verified_bytes"] = int(verified_bytes)
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
    "clean_quarantine",
    "inventory_quarantine",
    "quarantine_root",
    "write_quarantine_manifest",
]
