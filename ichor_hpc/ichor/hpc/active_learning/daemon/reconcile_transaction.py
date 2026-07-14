"""Durable transaction evidence for operator reconciliation repairs."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..strict_json import strict_json as json
from .filesystem import campaign_owned_path, operational_path
from .state import _fsync_parent_dir, atomic_write_json
from ..versioning.versioned_directory import VersionedDirectory


RECONCILE_TRANSACTION_SCHEMA_VERSION = 1
_STATUSES = {"PREPARED", "MUTATING", "COMMITTING", "COMMITTED", "FAILED"}
_TERMINAL_STATUSES = {"COMMITTED", "FAILED"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(label + " must be an integer")
    if value < minimum:
        raise ValueError(label + " must be >= " + str(minimum))
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a non-empty string")
    return value


def _timestamp(value: Any, label: str) -> str:
    text = _required_text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(label + " must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(label + " must include a timezone")
    return text


def read_reconcile_transaction(path: Path) -> Dict[str, Any]:
    """Read one transaction without accepting ambiguous recovery evidence."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("reconcile transaction must be a regular file: " + str(candidate))
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("reconcile transaction is unreadable: " + str(candidate)) from exc
    if not isinstance(payload, dict):
        raise ValueError("reconcile transaction must be a JSON object")
    if _exact_int(payload.get("schema_version"), "schema_version", minimum=1) != RECONCILE_TRANSACTION_SCHEMA_VERSION:
        raise ValueError("unsupported reconcile transaction schema")
    transaction_id = _required_text(payload.get("transaction_id"), "transaction_id")
    if candidate.stem != transaction_id:
        raise ValueError("reconcile transaction filename does not match transaction_id")
    try:
        uuid.UUID(hex=transaction_id)
    except ValueError as exc:
        raise ValueError("reconcile transaction_id is invalid") from exc
    status = _required_text(payload.get("status"), "status")
    if status not in _STATUSES:
        raise ValueError("unknown reconcile transaction status: " + status)
    _timestamp(payload.get("created_at_iso"), "created_at_iso")
    _timestamp(payload.get("updated_at_iso"), "updated_at_iso")
    _required_text(payload.get("proposed_phase"), "proposed_phase")
    _exact_int(payload.get("proposed_iteration"), "proposed_iteration")
    for label in (
        "planned_operations",
        "intent_transitions",
        "completed_operations",
        "warnings",
    ):
        if not isinstance(payload.get(label), list):
            raise ValueError("reconcile transaction " + label + " must be a list")
    return payload


def inventory_reconcile_transactions(campaign_dir: Path) -> List[Dict[str, Any]]:
    """Return strict transaction summaries, preserving malformed evidence."""
    campaign = Path(campaign_dir).resolve()
    root = operational_path(campaign, "reconcile_transactions")
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        return [{"path": str(root), "status": "INVALID", "reason": "transaction root is unsafe"}]
    records: List[Dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.suffix != ".json":
            records.append({"path": str(path), "status": "INVALID", "reason": "unexpected transaction entry"})
            continue
        try:
            payload = read_reconcile_transaction(path)
        except Exception as exc:
            records.append({"path": str(path), "status": "INVALID", "reason": str(exc)})
            continue
        records.append(
            {
                "path": str(path),
                "transaction_id": str(payload["transaction_id"]),
                "status": str(payload["status"]),
                "updated_at_iso": str(payload["updated_at_iso"]),
                "proposed_phase": str(payload["proposed_phase"]),
                "proposed_iteration": int(payload["proposed_iteration"]),
            }
        )
    return records


@dataclass
class ReconcileTransaction:
    """Mutable handle whose complete history is atomically persisted."""

    campaign_dir: Path
    path: Path
    payload: Dict[str, Any]

    def _write(self) -> None:
        atomic_write_json(self.path, self.payload)

    def set_status(self, status: str, *, reason: Optional[str] = None) -> None:
        if status not in _STATUSES:
            raise ValueError("unknown reconcile transaction status: " + status)
        self.payload["status"] = status
        self.payload["updated_at_iso"] = _now_iso()
        if reason is not None:
            self.payload["status_reason"] = str(reason)
        self._write()

    def record_paths(self, operation: str, paths: Iterable[str]) -> None:
        normalised: List[str] = []
        for raw in paths:
            path = campaign_owned_path(self.campaign_dir, Path(raw))
            normalised.append(path.relative_to(self.campaign_dir).as_posix())
        if not normalised:
            return
        records = self.payload.setdefault("completed_operations", [])
        if not isinstance(records, list):
            raise ValueError("reconcile transaction completed_operations is malformed")
        records.append(
            {
                "operation": str(operation),
                "paths": normalised,
                "completed_at_iso": _now_iso(),
            }
        )
        self.payload["updated_at_iso"] = _now_iso()
        self._write()

    def record_pointer_snapshots(
        self,
        snapshots: Sequence[Mapping[str, Any]],
    ) -> None:
        self.payload["pointer_snapshots"] = [dict(item) for item in snapshots]
        self.payload["updated_at_iso"] = _now_iso()
        self._write()

    def add_warning(self, warning: str) -> None:
        warnings = self.payload.setdefault("warnings", [])
        if not isinstance(warnings, list):
            raise ValueError("reconcile transaction warnings are malformed")
        warnings.append(str(warning))
        self.payload["updated_at_iso"] = _now_iso()
        self._write()


def begin_reconcile_transaction(
    campaign_dir: Path,
    *,
    proposed_phase: str,
    proposed_iteration: int,
    planned_operations: Sequence[str],
    intent_transitions: Sequence[Mapping[str, Any]],
) -> ReconcileTransaction:
    campaign = Path(campaign_dir).resolve()
    root = operational_path(campaign, "reconcile_transactions")
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError as exc:
        raise OSError(
            "could not enforce private reconcile transaction permissions: "
            + str(root)
        ) from exc
    blocking = [
        record
        for record in inventory_reconcile_transactions(campaign)
        if record.get("status") not in _TERMINAL_STATUSES
    ]
    if blocking:
        raise RuntimeError(
            "cannot begin reconciliation while prior transaction evidence is "
            "incomplete or invalid: "
            + "; ".join(
                str(record.get("path")) + " (" + str(record.get("status")) + ")"
                for record in blocking[:5]
            )
        )
    transaction_id = uuid.uuid4().hex
    path = campaign_owned_path(campaign, root / (transaction_id + ".json"))
    now = _now_iso()
    payload: Dict[str, Any] = {
        "schema_version": RECONCILE_TRANSACTION_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "status": "PREPARED",
        "created_at_iso": now,
        "updated_at_iso": now,
        "proposed_phase": str(proposed_phase),
        "proposed_iteration": int(proposed_iteration),
        "planned_operations": [str(item) for item in planned_operations],
        "intent_transitions": [dict(item) for item in intent_transitions],
        "completed_operations": [],
        "warnings": [],
    }
    transaction = ReconcileTransaction(campaign, path, payload)
    transaction._write()
    return transaction


def snapshot_version_pointer(
    campaign_dir: Path,
    versioning: VersionedDirectory,
    *,
    label: str,
    requested_version: int,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir).resolve()
    campaign_owned_path(campaign, versioning.parent)
    previous = versioning.current_version()
    return {
        "label": str(label),
        "parent": campaign_owned_path(campaign, versioning.parent)
        .relative_to(campaign)
        .as_posix(),
        "prefix": str(versioning.prefix),
        "name_width": int(versioning.name_width),
        "previous_version": previous,
        "requested_version": int(requested_version),
    }


def restore_version_pointer(
    campaign_dir: Path,
    snapshot: Mapping[str, Any],
) -> None:
    campaign = Path(campaign_dir).resolve()
    parent_raw = snapshot.get("parent")
    if not isinstance(parent_raw, str) or not parent_raw:
        raise ValueError("pointer snapshot parent is missing")
    parent = campaign_owned_path(campaign, campaign / Path(parent_raw))
    versioning = VersionedDirectory(
        parent,
        prefix=str(snapshot.get("prefix") or "iteration"),
        name_width=int(snapshot.get("name_width") or 6),
    )
    previous = snapshot.get("previous_version")
    if previous is not None:
        versioning.update_current(int(previous))
        return

    link = versioning.current_link_path()
    pointer = versioning._pointer_path()
    campaign_owned_path(campaign, link.parent)
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise ValueError("current path is not a removable symlink: " + str(link))
    if pointer.is_symlink():
        raise ValueError("current pointer fallback must not be a symlink")
    if pointer.exists():
        if not pointer.is_file():
            raise ValueError("current pointer fallback is not a file: " + str(pointer))
        pointer.unlink()
    _fsync_parent_dir(parent)


__all__ = [
    "RECONCILE_TRANSACTION_SCHEMA_VERSION",
    "ReconcileTransaction",
    "begin_reconcile_transaction",
    "inventory_reconcile_transactions",
    "read_reconcile_transaction",
    "restore_version_pointer",
    "snapshot_version_pointer",
]
