"""Durable transaction evidence for user reconciliation repairs."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..strict_json import strict_json as json
from .filesystem import campaign_owned_path, operational_path
from .state import _fsync_parent_dir, atomic_write_json
from ..versioning.versioned_directory import VersionedDirectory


RECONCILE_TRANSACTION_SCHEMA_VERSION = 2
READABLE_RECONCILE_TRANSACTION_SCHEMA_VERSIONS = frozenset({1, 2})
_STATUSES = {"PREPARED", "MUTATING", "COMMITTING", "COMMITTED", "FAILED"}
_TERMINAL_STATUSES = {"COMMITTED", "FAILED"}
_ATOMIC_TEMP_RE = re.compile(r"^\.t-[0-9a-f]{12}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_KNOWN_V1_MUTATION_OPERATIONS = frozenset(
    {
        "archive_reconcile_evidence",
        "prepare_ferebus_candidate_recovery",
        "retire_completed_staging",
        "repair_current_pointers",
        "validate_recovery_contract",
        "write_recovered_state",
        "update_config_lock",
        "publish_intent_transitions",
    }
)


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


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_sha256(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(label + " must be SHA-256 or null")
    return text


def _relative_path(campaign: Path, path: Path) -> str:
    owned = campaign_owned_path(campaign, path)
    return owned.relative_to(campaign).as_posix()


def _managed_version_pointer_paths(
    campaign: Path,
    versioning: VersionedDirectory,
) -> Tuple[str, str]:
    """Return managed pointer identities without rejecting the POSIX link."""
    parent = campaign_owned_path(campaign, versioning.parent)
    pointers = (
        versioning.current_link_path(),
        versioning._pointer_path(),
    )
    relative_paths: List[str] = []
    for pointer in pointers:
        if pointer.parent != parent:
            raise ValueError("managed version pointer has an unexpected parent")
        relative_paths.append(pointer.relative_to(campaign).as_posix())
    return relative_paths[0], relative_paths[1]


def snapshot_small_file(
    campaign_dir: Path,
    path: Path,
    *,
    include_json: bool = False,
) -> Dict[str, Any]:
    """Capture bounded before/after evidence for one campaign control file."""
    campaign = Path(campaign_dir).resolve()
    candidate = campaign_owned_path(campaign, Path(path))
    record: Dict[str, Any] = {
        "path": candidate.relative_to(campaign).as_posix(),
        "exists": False,
        "size": 0,
        "sha256": None,
    }
    if not candidate.exists():
        return record
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("control path is not a regular file: " + str(candidate))
    record.update(
        {
            "exists": True,
            "size": int(candidate.stat().st_size),
            "sha256": _file_sha256(candidate),
        }
    )
    if include_json:
        payload = json.loads(candidate.read_text(encoding="utf-8"), source=candidate)
        if not isinstance(payload, dict):
            raise ValueError("control file must contain a JSON object: " + str(candidate))
        record["payload"] = payload
    return record


def _validate_file_snapshot(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(label + " must be an object")
    path = _required_text(value.get("path"), label + ".path")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(label + ".path must be campaign-relative")
    exists = value.get("exists")
    if not isinstance(exists, bool):
        raise ValueError(label + ".exists must be boolean")
    _exact_int(value.get("size"), label + ".size")
    digest = _optional_sha256(value.get("sha256"), label + ".sha256")
    if exists != (digest is not None):
        raise ValueError(label + " existence and digest disagree")
    return value


def _snapshot_matches(campaign: Path, snapshot: Mapping[str, Any]) -> bool:
    expected = _validate_file_snapshot(dict(snapshot), "file snapshot")
    observed = snapshot_small_file(
        campaign,
        campaign / Path(str(expected["path"])),
        include_json=False,
    )
    return all(
        observed.get(key) == expected.get(key)
        for key in ("path", "exists", "size", "sha256")
    )


def _validate_transaction_payload(
    payload: Any,
    *,
    expected_transaction_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("reconcile transaction must be a JSON object")
    schema = _exact_int(payload.get("schema_version"), "schema_version", minimum=1)
    if schema not in READABLE_RECONCILE_TRANSACTION_SCHEMA_VERSIONS:
        raise ValueError("unsupported reconcile transaction schema")
    transaction_id = _required_text(payload.get("transaction_id"), "transaction_id")
    if expected_transaction_id is not None and transaction_id != expected_transaction_id:
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
    if schema == 2:
        campaign_uid = payload.get("campaign_uid")
        if campaign_uid is not None and (
            not isinstance(campaign_uid, str) or not campaign_uid
        ):
            raise ValueError("campaign_uid must be a non-empty string or null")
        source = payload.get("source_evidence")
        if not isinstance(source, dict):
            raise ValueError("schema-v2 transaction source_evidence must be an object")
        _validate_file_snapshot(source.get("state"), "source_evidence.state")
        _optional_sha256(
            source.get("authority_anchor_sha256"),
            "source_evidence.authority_anchor_sha256",
        )
        mutation_plan = payload.get("mutation_plan")
        if not isinstance(mutation_plan, list) or not all(
            isinstance(item, dict) for item in mutation_plan
        ):
            raise ValueError("schema-v2 transaction mutation_plan must be a list of objects")
        operation_ids: set[str] = set()
        for item in mutation_plan:
            operation_id = _required_text(
                item.get("operation_id"),
                "mutation_plan.operation_id",
            )
            if operation_id in operation_ids:
                raise ValueError("schema-v2 mutation operation IDs are duplicated")
            operation_ids.add(operation_id)
            _required_text(item.get("kind"), "mutation_plan.kind")
            _required_text(item.get("name"), "mutation_plan.name")
            archive_identity = item.get("archive_identity")
            if archive_identity is not None and str(archive_identity) != transaction_id:
                raise ValueError("mutation archive identity must match transaction_id")
        commit_plan = payload.get("commit_plan")
        if commit_plan is not None and not isinstance(commit_plan, dict):
            raise ValueError("schema-v2 transaction commit_plan must be an object or null")
        if status == "COMMITTING" and not isinstance(commit_plan, dict):
            raise ValueError("COMMITTING schema-v2 transaction has no commit plan")
        if isinstance(commit_plan, dict):
            _timestamp(commit_plan.get("prepared_at_iso"), "commit_plan.prepared_at_iso")
            _optional_sha256(
                commit_plan.get("stable_authority_sha256"),
                "commit_plan.stable_authority_sha256",
            )
            if not isinstance(commit_plan.get("stable_authority_records"), list):
                raise ValueError("commit_plan stable authority records are missing")
            if not isinstance(commit_plan.get("state"), dict):
                raise ValueError("commit_plan state is missing")
            if not isinstance(commit_plan.get("pointers"), list):
                raise ValueError("commit_plan pointers must be a list")
            if not isinstance(commit_plan.get("intent_transitions"), list):
                raise ValueError("commit_plan intent transitions must be a list")
            if not isinstance(commit_plan.get("steps"), list):
                raise ValueError("commit_plan steps must be a list")
        resolution = payload.get("resolution")
        if resolution is not None and not isinstance(resolution, dict):
            raise ValueError("schema-v2 transaction resolution must be an object")
    return payload


def read_reconcile_transaction(path: Path) -> Dict[str, Any]:
    """Read one transaction without accepting ambiguous recovery evidence."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("reconcile transaction must be a regular file: " + str(candidate))
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("reconcile transaction is unreadable: " + str(candidate)) from exc
    return _validate_transaction_payload(
        payload,
        expected_transaction_id=candidate.stem,
    )


def _read_atomic_transaction_temp(path: Path) -> Dict[str, Any]:
    candidate = Path(path)
    if not _ATOMIC_TEMP_RE.fullmatch(candidate.name):
        raise ValueError("not a reconcile transaction atomic temporary file")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("atomic transaction temporary path is not a regular file")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"), source=candidate)
    except (OSError, ValueError) as exc:
        raise ValueError("atomic transaction temporary file is unreadable") from exc
    return _validate_transaction_payload(payload)


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
        if _ATOMIC_TEMP_RE.fullmatch(path.name):
            try:
                payload = _read_atomic_transaction_temp(path)
                target = root / (str(payload["transaction_id"]) + ".json")
                records.append(
                    {
                        "path": str(path),
                        "transaction_id": str(payload["transaction_id"]),
                        "schema_version": int(payload["schema_version"]),
                        "status": "ORPHAN",
                        "orphan_transaction_status": str(payload["status"]),
                        "canonical_exists": bool(target.exists()),
                        "updated_at_iso": str(payload["updated_at_iso"]),
                        "proposed_phase": str(payload["proposed_phase"]),
                        "proposed_iteration": int(payload["proposed_iteration"]),
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "path": str(path),
                        "status": "INVALID",
                        "reason": "invalid atomic transaction temporary file: " + str(exc),
                    }
                )
            continue
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
                "schema_version": int(payload["schema_version"]),
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

    def prepare_commit(self, commit_plan: Mapping[str, Any]) -> None:
        """Publish the complete authority plan atomically with COMMITTING."""
        if int(self.payload.get("schema_version", 1)) != 2:
            self.set_status("COMMITTING")
            return
        plan = dict(commit_plan)
        if not plan:
            raise ValueError("reconcile commit plan must not be empty")
        self.payload["commit_plan"] = plan
        self.payload["status"] = "COMMITTING"
        self.payload["updated_at_iso"] = _now_iso()
        self.payload.pop("last_error", None)
        self._write()

    def record_interruption(self, reason: str) -> None:
        """Record a retryable interruption without falsely terminalising it."""
        self.payload["last_error"] = {
            "reason": str(reason),
            "recorded_at_iso": _now_iso(),
        }
        self.payload["updated_at_iso"] = _now_iso()
        self._write()

    def resolve(
        self,
        *,
        status: str,
        disposition: str,
        reason: str,
    ) -> None:
        if status not in _TERMINAL_STATUSES:
            raise ValueError("resolved reconcile transaction must be terminal")
        self.payload["resolution"] = {
            "disposition": _required_text(disposition, "resolution disposition"),
            "reason": str(reason),
            "resolved_at_iso": _now_iso(),
        }
        self.payload["status"] = status
        self.payload["status_reason"] = str(reason)
        self.payload["updated_at_iso"] = _now_iso()
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
    campaign_uid: Optional[str] = None,
    source_authority_anchor_sha256: Optional[str] = None,
    source_state_identity: Optional[Mapping[str, Any]] = None,
    mutation_plan: Optional[Sequence[Mapping[str, Any]]] = None,
    transaction_id: Optional[str] = None,
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
    transaction_id = str(transaction_id or uuid.uuid4().hex)
    try:
        uuid.UUID(hex=transaction_id)
    except ValueError as exc:
        raise ValueError("reconcile transaction_id is invalid") from exc
    path = campaign_owned_path(campaign, root / (transaction_id + ".json"))
    if path.exists():
        raise FileExistsError("reconcile transaction already exists: " + str(path))
    now = _now_iso()
    state_identity = dict(
        source_state_identity
        or snapshot_small_file(
            campaign,
            operational_path(campaign, "state.json"),
            include_json=False,
        )
    )
    _validate_file_snapshot(state_identity, "source_state_identity")
    anchor = _optional_sha256(
        source_authority_anchor_sha256,
        "source_authority_anchor_sha256",
    )
    structured_mutations = [dict(item) for item in (mutation_plan or [])]
    if not structured_mutations:
        structured_mutations = [
            {
                "operation_id": "operation-" + str(index).zfill(4),
                "kind": "reconcile_operation",
                "name": str(name),
            }
            for index, name in enumerate(planned_operations)
        ]
    payload: Dict[str, Any] = {
        "schema_version": RECONCILE_TRANSACTION_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "status": "PREPARED",
        "created_at_iso": now,
        "updated_at_iso": now,
        "proposed_phase": str(proposed_phase),
        "proposed_iteration": int(proposed_iteration),
        "campaign_uid": None if campaign_uid is None else str(campaign_uid),
        "source_evidence": {
            "state": state_identity,
            "authority_anchor_sha256": anchor,
        },
        "mutation_plan": structured_mutations,
        "commit_plan": None,
        "planned_operations": [str(item) for item in planned_operations],
        "intent_transitions": [dict(item) for item in intent_transitions],
        "completed_operations": [],
        "warnings": [],
    }
    transaction = ReconcileTransaction(campaign, path, payload)
    transaction._write()
    return transaction


def _records_digest(records: Sequence[Sequence[Any]]) -> str:
    encoded = json.dumps(
        [list(record) for record in records],
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_authority_records(
    artifact_snapshot: Any,
    *,
    excluded_paths: Iterable[str],
) -> List[List[Any]]:
    excluded = {str(value) for value in excluded_paths}
    anchor_records = getattr(artifact_snapshot, "anchor_records", ())
    return [
        [str(path), int(size), str(digest)]
        for path, size, digest in anchor_records
        if str(path) not in excluded
    ]


def _planned_control_file(
    campaign: Path,
    path: Path,
    after_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "path": _relative_path(campaign, path),
        "before": snapshot_small_file(
            campaign,
            path,
            include_json=path.is_file() and not path.is_symlink(),
        ),
        "after_payload": dict(after_payload),
        "after_sha256": _payload_sha256(dict(after_payload)),
    }


def build_reconcile_commit_plan(
    campaign_dir: Path,
    *,
    transaction_id: str,
    proposed_state: Any,
    config: Optional[Any],
    intent_transitions: Sequence[Mapping[str, Any]],
    artifact_snapshot: Any,
) -> Dict[str, Any]:
    """Freeze every authoritative reconcile effect before publication."""
    campaign = Path(campaign_dir).resolve()
    prepared_at_iso = _now_iso()
    state_path = operational_path(campaign, "state.json")
    state_before = snapshot_small_file(
        campaign,
        state_path,
        include_json=state_path.is_file() and not state_path.is_symlink(),
    )
    state_after = proposed_state.to_dict()
    state_relative = _relative_path(campaign, state_path)

    pointer_records: List[Dict[str, Any]] = []
    for label, parent, target_version in (
        (
            "reference_data",
            campaign / "QM_REFERENCE_DATA",
            int(proposed_state.reference_data_version),
        ),
        (
            "trained_models",
            campaign / "TRAINED_MODELS",
            int(proposed_state.models_version),
        ),
    ):
        if target_version < 0:
            continue
        versioning = VersionedDirectory(parent)
        pointer_records.append(
            {
                "label": label,
                "parent": _relative_path(campaign, parent),
                "prefix": str(versioning.prefix),
                "name_width": int(versioning.name_width),
                "before_version": versioning.current_version(),
                "after_version": target_version,
            }
        )

    config_record: Optional[Dict[str, Any]] = None
    config_path = operational_path(campaign, "config_lock.json")
    if config is not None:
        from .config_lock import prepare_config_lock_update

        prepared_config = prepare_config_lock_update(
            campaign,
            config,
            campaign_uid=str(proposed_state.campaign_uid),
        )
        config_files: List[Dict[str, Any]] = []
        history_payload = prepared_config.get("history_payload")
        history_path = prepared_config.get("history_path")
        if isinstance(history_payload, Mapping) and history_path:
            config_files.append(
                _planned_control_file(
                    campaign,
                    Path(str(history_path)),
                    history_payload,
                )
            )
        lock_payload = prepared_config.get("lock_payload")
        if not isinstance(lock_payload, Mapping):
            raise ValueError("prepared config-lock payload is missing")
        config_files.append(
            _planned_control_file(
                campaign,
                Path(str(prepared_config["lock_path"])),
                lock_payload,
            )
        )
        config_record = {
            "files": config_files,
            "campaign_uid": str(proposed_state.campaign_uid),
        }

    from . import submission_intent as _submission_intent

    raw_intents: List[Dict[str, Any]] = [dict(raw) for raw in intent_transitions]
    seen_intents: set[Tuple[str, int]] = set()
    for item in raw_intents:
        phase = _required_text(item.get("phase"), "intent transition phase")
        iteration = _exact_int(item.get("iteration"), "intent transition iteration")
        if (phase, iteration) in seen_intents:
            raise ValueError("duplicate reconcile intent transition")
        seen_intents.add((phase, iteration))

    current_key = (proposed_state.phase.value, int(proposed_state.iteration))
    if current_key not in seen_intents:
        current = _submission_intent.load_intent(
            campaign,
            current_key[0],
            current_key[1],
            expected_campaign_uid=str(proposed_state.campaign_uid),
        )
        if isinstance(current, dict) and str(current.get("status") or "") == "FAILED":
            raw_intents.append(
                {
                    "phase": current_key[0],
                    "iteration": current_key[1],
                    "target_status": "SUPERSEDED",
                    "reason": "reconcile_apply_retry",
                }
            )

    planned_intents: List[Dict[str, Any]] = []
    for raw in raw_intents:
        item = dict(raw)
        phase = _required_text(item.get("phase"), "intent transition phase")
        iteration = _exact_int(item.get("iteration"), "intent transition iteration")
        if (
            (phase, iteration) == current_key
            and str(item.get("target_status") or "") == "FAILED"
        ):
            item["target_status"] = "SUPERSEDED"
            item["reason"] = "reconcile_apply_retry"
            item.pop("completion_receipt", None)
        path = _submission_intent.intent_path(campaign, phase, iteration)
        after_payload = _submission_intent.prepare_reconcile_terminal_transition(
            campaign,
            phase,
            iteration,
            target_status=_required_text(
                item.get("target_status"),
                "intent transition target_status",
            ),
            reason=str(item.get("reason") or ""),
            completion_receipt=(
                dict(item["completion_receipt"])
                if isinstance(item.get("completion_receipt"), Mapping)
                else None
            ),
            expected_campaign_uid=str(proposed_state.campaign_uid),
            updated_at_iso=prepared_at_iso,
        )
        item.update(_planned_control_file(campaign, path, after_payload))
        planned_intents.append(item)

    excluded = {
        state_relative,
        _relative_path(campaign, config_path),
        ".DATA/ACTIVE_LEARNING/reconcile_transactions/"
        + str(transaction_id)
        + ".json",
        *(
            str(record["path"])
            for record in planned_intents
        ),
    }
    if isinstance(config_record, Mapping):
        excluded.update(
            str(record["path"])
            for record in config_record.get("files") or []
        )
    for record in pointer_records:
        versioning = VersionedDirectory(
            campaign / Path(str(record["parent"])),
            prefix=str(record["prefix"]),
            name_width=int(record["name_width"]),
        )
        excluded.update(
            _managed_version_pointer_paths(
                campaign,
                versioning,
            )
        )
    stable_records = _stable_authority_records(
        artifact_snapshot,
        excluded_paths=excluded,
    )
    return {
        "prepared_at_iso": prepared_at_iso,
        "stable_authority_records": stable_records,
        "stable_authority_sha256": _records_digest(stable_records),
        "state": {
            "path": state_relative,
            "before": state_before,
            "after_payload": state_after,
            "after_sha256": _payload_sha256(state_after),
            "backup_path": state_relative + ".before-reconcile-" + transaction_id,
        },
        "pointers": pointer_records,
        "config_lock": config_record,
        "intent_transitions": planned_intents,
        "steps": [
            "repair_current_pointers",
            "write_recovered_state",
            "update_config_lock",
            "publish_intent_transitions",
        ],
    }


def _verify_stable_authority(campaign: Path, plan: Mapping[str, Any]) -> None:
    records = plan.get("stable_authority_records")
    if not isinstance(records, list):
        raise ValueError("reconcile commit plan has no stable authority records")
    expected_digest = _optional_sha256(
        plan.get("stable_authority_sha256"),
        "stable_authority_sha256",
    )
    if expected_digest != _records_digest(records):
        raise ValueError("reconcile commit plan stable authority digest is invalid")
    for raw in records:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("reconcile stable authority record is malformed")
        relative, size, digest = str(raw[0]), int(raw[1]), str(raw[2])
        path = campaign_owned_path(campaign, campaign / Path(relative))
        if path.is_symlink() or not path.is_file():
            raise ValueError("stable campaign authority is missing: " + relative)
        if int(path.stat().st_size) != size or _file_sha256(path) != digest:
            raise ValueError("stable campaign authority changed: " + relative)


def _pointer_state(campaign: Path, record: Mapping[str, Any]) -> str:
    parent = campaign_owned_path(campaign, campaign / Path(str(record["parent"])))
    current = VersionedDirectory(
        parent,
        prefix=str(record.get("prefix") or "iteration"),
        name_width=int(record.get("name_width") or 6),
    ).current_version()
    before = record.get("before_version")
    after = int(record["after_version"])
    if current == after:
        return "after"
    if current == before:
        return "before"
    return "other"


def _control_file_state(campaign: Path, record: Mapping[str, Any]) -> str:
    path = campaign_owned_path(campaign, campaign / Path(str(record["path"])))
    observed = snapshot_small_file(campaign, path)
    if observed.get("sha256") == record.get("after_sha256"):
        return "after"
    before = record.get("before")
    if isinstance(before, Mapping) and all(
        observed.get(key) == before.get(key)
        for key in ("path", "exists", "size", "sha256")
    ):
        return "before"
    return "other"


def _intent_state(campaign: Path, record: Mapping[str, Any]) -> str:
    return _control_file_state(campaign, record)


def _commit_plan_states(campaign: Path, plan: Mapping[str, Any]) -> List[str]:
    _verify_stable_authority(campaign, plan)
    state_record = plan.get("state")
    if not isinstance(state_record, dict):
        raise ValueError("reconcile commit plan state is missing")
    state_before = state_record.get("before")
    if not isinstance(state_before, Mapping):
        raise ValueError("reconcile commit plan source state is missing")
    state_path = campaign_owned_path(
        campaign,
        campaign / Path(str(state_record["path"])),
    )
    observed_state = snapshot_small_file(campaign, state_path)
    if observed_state.get("sha256") == state_record.get("after_sha256"):
        states = ["after"]
    elif all(
        observed_state.get(key) == state_before.get(key)
        for key in ("path", "exists", "size", "sha256")
    ):
        states = ["before"]
    else:
        states = ["other"]
    for record in plan.get("pointers") or []:
        states.append(_pointer_state(campaign, record))
    config_record = plan.get("config_lock")
    if isinstance(config_record, Mapping):
        for record in config_record.get("files") or []:
            states.append(_control_file_state(campaign, record))
    for record in plan.get("intent_transitions") or []:
        states.append(_intent_state(campaign, record))
    return states


def _read_current_state(campaign: Path) -> Optional[Any]:
    from .state import read_state

    path = operational_path(campaign, "state.json")
    if not path.exists():
        return None
    return read_state(path)


def _v1_mutation_plan_is_known(payload: Mapping[str, Any]) -> bool:
    operations = {str(value) for value in payload.get("planned_operations") or []}
    return bool(operations) and operations.issubset(_KNOWN_V1_MUTATION_OPERATIONS)


def _source_authority_matches(
    payload: Mapping[str, Any],
    artifact_snapshot: Optional[Any],
) -> bool:
    source = payload.get("source_evidence")
    if not isinstance(source, Mapping):
        return False
    expected = source.get("authority_anchor_sha256")
    if expected is None:
        return True
    if artifact_snapshot is None:
        return False
    transaction_path = (
        ".DATA/ACTIVE_LEARNING/reconcile_transactions/"
        + str(payload.get("transaction_id") or "")
        + ".json"
    )
    records = [
        [str(path), int(size), str(digest)]
        for path, size, digest in getattr(artifact_snapshot, "anchor_records", ())
        if str(path) != transaction_path
    ]
    return _records_digest(records) == str(expected)


def _v2_target_contract_is_valid(
    campaign: Path,
    payload: Mapping[str, Any],
) -> Tuple[bool, str]:
    try:
        commit_plan = payload.get("commit_plan")
        state_record = (
            commit_plan.get("state") if isinstance(commit_plan, Mapping) else None
        )
        target = (
            state_record.get("after_payload")
            if isinstance(state_record, Mapping)
            else None
        )
        if not isinstance(target, dict):
            return False, "target campaign state is unavailable"
        from .state import CampaignState
        from .recovery_contracts import recovery_contract_status

        state = CampaignState.from_dict(target)
        contract = recovery_contract_status(
            campaign,
            state,
            verification="authority",
            artifact_snapshot=None,
        )
        if not bool(contract.get("contract_ok", False)):
            return False, "recovered phase input contract is incomplete"
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)
    return True, "recovered phase input contract is complete"


def _v1_committed_state_is_complete(
    campaign: Path,
    payload: Mapping[str, Any],
    *,
    artifact_snapshot: Optional[Any],
) -> Tuple[bool, str]:
    try:
        state = _read_current_state(campaign)
        if state is None:
            return False, "campaign state is absent"
        if (
            state.phase.value != str(payload["proposed_phase"])
            or int(state.iteration) != int(payload["proposed_iteration"])
        ):
            return False, "campaign state does not match the recorded recovery target"
        if int(state.reference_data_version) >= 0:
            current = VersionedDirectory(campaign / "QM_REFERENCE_DATA").current_version()
            if current != int(state.reference_data_version):
                return False, "QM reference-data pointer does not match recovered state"
        if int(state.models_version) >= 0:
            current = VersionedDirectory(campaign / "TRAINED_MODELS").current_version()
            if current != int(state.models_version):
                return False, "trained-model pointer does not match recovered state"

        config_path = campaign / "campaign.yaml"
        lock_path = operational_path(campaign, "config_lock.json")
        if config_path.is_file() or lock_path.is_file():
            if not config_path.is_file() or not lock_path.is_file():
                return False, "campaign configuration and its lock are incomplete"
            from ..config import CampaignConfig
            from .config_lock import canonical_config, read_config_lock

            config = CampaignConfig.from_yaml(config_path)
            lock = read_config_lock(
                campaign,
                expected_campaign_uid=str(state.campaign_uid),
            )
            if lock.get("canonical_config") != canonical_config(config):
                return False, "campaign configuration lock is not the committed target"

        from . import submission_intent as _submission_intent

        for transition in payload.get("intent_transitions") or []:
            intent = _submission_intent.load_intent(
                campaign,
                str(transition["phase"]),
                int(transition["iteration"]),
                expected_campaign_uid=str(state.campaign_uid),
            )
            if not isinstance(intent, dict) or str(intent.get("status") or "") != str(
                transition.get("target_status") or ""
            ):
                return False, "submission-intent transition is incomplete"

        from .recovery_contracts import recovery_contract_status

        contract = recovery_contract_status(
            campaign,
            state,
            verification="authority",
            artifact_snapshot=artifact_snapshot,
        )
        if not bool(contract.get("contract_ok", False)):
            return False, "recovered phase input contract is incomplete"
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)
    return True, "recorded recovered state and authority are complete"


def _v1_untouched_commit_can_retire(
    campaign: Path,
    payload: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Prove a v1 commit stopped before state publication.

    V1 did not bind an exact state payload, so rollback is allowed only when
    the current state is still at a different phase/iteration and every
    pointer snapshot agrees with that state's recorded versions.
    """
    try:
        state = _read_current_state(campaign)
        if state is None:
            return False, "campaign state is absent"
        if (
            state.phase.value == str(payload["proposed_phase"])
            and int(state.iteration) == int(payload["proposed_iteration"])
        ):
            return False, "legacy transaction has no exact source-state proof"
        snapshots = payload.get("pointer_snapshots") or []
        if not isinstance(snapshots, list):
            return False, "legacy pointer snapshots are malformed"
        expected_versions = {
            "reference_data": int(state.reference_data_version),
            "trained_models": int(state.models_version),
        }
        for snapshot in snapshots:
            if not isinstance(snapshot, Mapping):
                return False, "legacy pointer snapshot is malformed"
            label = str(snapshot.get("label") or "")
            if label not in expected_versions:
                return False, "legacy pointer snapshot label is unknown"
            previous = snapshot.get("previous_version")
            previous_state = -1 if previous is None else int(previous)
            if expected_versions[label] != previous_state:
                return False, "legacy source state does not match pointer snapshot"
            parent = campaign_owned_path(
                campaign,
                campaign / Path(str(snapshot.get("parent") or "")),
            )
            current = VersionedDirectory(
                parent,
                prefix=str(snapshot.get("prefix") or "iteration"),
                name_width=int(snapshot.get("name_width") or 6),
            ).current_version()
            requested = int(snapshot.get("requested_version"))
            if current not in {previous, requested}:
                return False, "legacy pointer matches neither recorded value"
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)
    return True, "campaign state is unchanged and pointer changes are exactly reversible"


def inspect_reconcile_transaction_recovery(
    campaign_dir: Path,
    *,
    artifact_snapshot: Optional[Any] = None,
) -> Dict[str, Any]:
    """Classify one interrupted transaction without mutating campaign data."""
    campaign = Path(campaign_dir).resolve()
    records = inventory_reconcile_transactions(campaign)
    active = [
        dict(record)
        for record in records
        if str(record.get("status") or "") not in _TERMINAL_STATUSES
    ]
    if not active:
        return {"state": "none", "recoverable": False, "records": records}
    if len(active) == 2:
        orphan = next(
            (item for item in active if str(item.get("status") or "") == "ORPHAN"),
            None,
        )
        canonical = next(
            (
                item
                for item in active
                if str(item.get("status") or "") not in {"ORPHAN", "INVALID"}
            ),
            None,
        )
        if (
            isinstance(orphan, dict)
            and isinstance(canonical, dict)
            and bool(orphan.get("canonical_exists", False))
            and str(orphan.get("transaction_id") or "")
            == str(canonical.get("transaction_id") or "")
        ):
            return {
                "state": "recoverable",
                "recoverable": True,
                "action": "archive_redundant_orphan",
                "disposition": "completed_safe_cleanup",
                "reason": "an unpublished atomic transaction update remains beside its canonical record",
                "record": orphan,
                "records": records,
            }
    if len(active) != 1:
        return {
            "state": "blocked",
            "recoverable": False,
            "reason": "multiple incomplete or invalid reconcile records exist",
            "records": records,
        }
    record = active[0]
    status = str(record.get("status") or "")
    if status == "INVALID":
        return {
            "state": "blocked",
            "recoverable": False,
            "reason": str(record.get("reason") or "invalid transaction evidence"),
            "record": record,
            "records": records,
        }
    if status == "ORPHAN":
        orphan_status = str(record.get("orphan_transaction_status") or "")
        if bool(record.get("canonical_exists", False)):
            return {
                "state": "recoverable",
                "recoverable": True,
                "action": "archive_redundant_orphan",
                "disposition": "completed_safe_cleanup",
                "reason": "an unpublished atomic transaction update remains beside its canonical record",
                "record": record,
                "records": records,
            }
        if orphan_status == "PREPARED":
            return {
                "state": "recoverable",
                "recoverable": True,
                "action": "promote_prepared_orphan",
                "disposition": "abandoned_before_mutation",
                "reason": "transaction preparation was written but not atomically published",
                "record": record,
                "records": records,
            }
        return {
            "state": "blocked",
            "recoverable": False,
            "reason": "orphan transaction claims mutation without canonical authority",
            "record": record,
            "records": records,
        }

    path = Path(str(record["path"]))
    try:
        payload = read_reconcile_transaction(path)
    except Exception as exc:
        return {
            "state": "blocked",
            "recoverable": False,
            "reason": str(exc),
            "record": record,
            "records": records,
        }
    schema = int(payload["schema_version"])
    current_status = str(payload["status"])
    common = {
        "state": "recoverable",
        "recoverable": True,
        "record": record,
        "records": records,
        "transaction_id": str(payload["transaction_id"]),
        "schema_version": schema,
        "transaction_status": current_status,
        "phase": str(payload["proposed_phase"]),
        "iteration": int(payload["proposed_iteration"]),
    }
    if current_status == "PREPARED":
        if schema == 2 and not _snapshot_matches(
            campaign,
            payload["source_evidence"]["state"],
        ):
            return {
                **common,
                "state": "blocked",
                "recoverable": False,
                "reason": "campaign state changed after transaction preparation",
            }
        if schema == 2 and not _source_authority_matches(payload, artifact_snapshot):
            return {
                **common,
                "state": "blocked",
                "recoverable": False,
                "reason": "campaign authority changed after transaction preparation",
            }
        return {
            **common,
            "action": "abandon",
            "disposition": "abandoned_before_mutation",
            "reason": "the previous reconcile stopped before changing campaign data",
        }
    if current_status == "MUTATING":
        if schema == 1 and not _v1_mutation_plan_is_known(payload):
            return {
                **common,
                "state": "blocked",
                "recoverable": False,
                "reason": "legacy transaction contains an unknown cleanup operation",
            }
        if schema == 2 and not _snapshot_matches(
            campaign,
            payload["source_evidence"]["state"],
        ):
            return {
                **common,
                "state": "blocked",
                "recoverable": False,
                "reason": "authoritative state changed while temporary data was being handled",
            }
        if schema == 2 and not _source_authority_matches(payload, artifact_snapshot):
            return {
                **common,
                "state": "blocked",
                "recoverable": False,
                "reason": "campaign authority changed while temporary data was being handled",
            }
        try:
            _read_current_state(campaign)
        except Exception as exc:
            return {
                **common,
                "state": "blocked",
                "recoverable": False,
                "reason": "campaign state cannot validate safe cleanup: " + str(exc),
            }
        return {
            **common,
            "action": "abandon",
            "disposition": "completed_safe_cleanup",
            "reason": "temporary-data cleanup may be incomplete; campaign authority is unchanged",
        }
    if current_status != "COMMITTING":
        return {
            **common,
            "state": "blocked",
            "recoverable": False,
            "reason": "unsupported nonterminal transaction status",
        }
    if schema == 1:
        complete, reason = _v1_committed_state_is_complete(
            campaign,
            payload,
            artifact_snapshot=artifact_snapshot,
        )
        if complete:
            return {
                **common,
                "action": "adopt",
                "disposition": "adopted_committed_state",
                "reason": reason,
            }
        untouched, untouched_reason = _v1_untouched_commit_can_retire(
            campaign,
            payload,
        )
        if untouched:
            return {
                **common,
                "action": "rollback_v1_pointers",
                "disposition": "rolled_back_exactly",
                "reason": untouched_reason,
            }
        return {
            **common,
            "state": "blocked",
            "recoverable": False,
            "reason": reason + "; " + untouched_reason,
        }
    try:
        states = _commit_plan_states(campaign, payload["commit_plan"])
    except Exception as exc:
        return {
            **common,
            "state": "blocked",
            "recoverable": False,
            "reason": "commit evidence is inconsistent: " + str(exc),
        }
    if all(value == "after" for value in states):
        valid, reason = _v2_target_contract_is_valid(campaign, payload)
        if not valid:
            return {
                **common,
                "state": "blocked",
                "recoverable": False,
                "reason": reason,
            }
        return {
            **common,
            "action": "adopt",
            "disposition": "adopted_committed_state",
            "reason": "all recorded campaign changes were published",
        }
    if all(value == "before" for value in states):
        return {
            **common,
            "action": "abandon",
            "disposition": "rolled_back_exactly",
            "reason": "no recorded authoritative campaign change was published",
        }
    if all(value in {"before", "after"} for value in states):
        valid, reason = _v2_target_contract_is_valid(campaign, payload)
        if not valid:
            return {
                **common,
                "state": "blocked",
                "recoverable": False,
                "reason": reason,
            }
        return {
            **common,
            "action": "roll_forward",
            "disposition": "rolled_forward",
            "reason": "a partially published recovery can be completed from exact transaction evidence",
        }
    return {
        **common,
        "state": "blocked",
        "recoverable": False,
        "reason": "campaign authority matches neither the recorded before nor after state",
    }


def _archive_orphan_temp(campaign: Path, source: Path, transaction_id: str) -> Path:
    root = operational_path(campaign, "reconcile_transaction_orphans")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("reconcile transaction orphan archive is unsafe")
    try:
        os.chmod(root, 0o700)
    except OSError as exc:
        raise OSError("could not secure reconcile transaction orphan archive") from exc
    digest = _file_sha256(source)
    target = campaign_owned_path(
        campaign,
        root / (str(transaction_id) + "-" + digest + ".json.part"),
    )
    if target.exists():
        if target.is_symlink() or not target.is_file() or _file_sha256(target) != digest:
            raise ValueError("reconcile transaction orphan archive conflicts")
        source.unlink()
        _fsync_parent_dir(source)
        return target
    os.replace(source, target)
    _fsync_parent_dir(source)
    _fsync_parent_dir(target)
    return target


def publish_reconcile_intent_target(
    campaign: Path,
    record: Mapping[str, Any],
) -> None:
    from . import submission_intent as _submission_intent

    phase = str(record["phase"])
    iteration = int(record["iteration"])
    after_payload = record.get("after_payload")
    if not isinstance(after_payload, Mapping):
        raise ValueError("reconcile target intent payload is unavailable")
    if str(after_payload.get("status") or "") != str(record["target_status"]):
        raise ValueError("reconcile target intent status is inconsistent")
    _submission_intent.publish_prepared_reconcile_transition(
        campaign,
        phase,
        iteration,
        after_payload,
        expected_campaign_uid=str(after_payload.get("campaign_uid") or ""),
    )


def publish_reconcile_config_target(
    campaign: Path,
    record: Mapping[str, Any],
) -> None:
    from .config_lock import publish_prepared_config_lock_update

    lock_relative = _relative_path(
        campaign,
        operational_path(campaign, "config_lock.json"),
    )
    lock_record: Optional[Mapping[str, Any]] = None
    history_record: Optional[Mapping[str, Any]] = None
    for file_record in record.get("files") or []:
        if not isinstance(file_record, Mapping):
            raise ValueError("reconcile config-lock file plan is malformed")
        if str(file_record.get("path") or "") == lock_relative:
            lock_record = file_record
        elif history_record is None:
            history_record = file_record
        else:
            raise ValueError("reconcile config-lock plan has multiple history files")
    if not isinstance(lock_record, Mapping):
        raise ValueError("reconcile config-lock target is missing")
    prepared = {
        "lock_path": str(campaign / Path(str(lock_record["path"]))),
        "lock_payload": dict(lock_record["after_payload"]),
        "history_path": (
            None
            if history_record is None
            else str(campaign / Path(str(history_record["path"])))
        ),
        "history_payload": (
            None
            if history_record is None
            else dict(history_record["after_payload"])
        ),
    }
    publish_prepared_config_lock_update(
        campaign,
        prepared,
        expected_campaign_uid=str(record.get("campaign_uid") or ""),
    )


def _roll_forward_v2_commit(
    campaign: Path,
    transaction: ReconcileTransaction,
    *,
    artifact_snapshot: Optional[Any],
) -> None:
    from .recovery_contracts import recovery_contract_status
    from .state import CampaignState, atomic_write_json, write_state

    plan = transaction.payload.get("commit_plan")
    if not isinstance(plan, dict):
        raise ValueError("schema-v2 COMMITTING transaction has no commit plan")
    _verify_stable_authority(campaign, plan)

    for record in plan.get("pointers") or []:
        state = _pointer_state(campaign, record)
        if state == "other":
            raise ValueError("version pointer matches neither recorded state")
        if state == "before":
            parent = campaign_owned_path(
                campaign,
                campaign / Path(str(record["parent"])),
            )
            VersionedDirectory(
                parent,
                prefix=str(record.get("prefix") or "iteration"),
                name_width=int(record.get("name_width") or 6),
            ).update_current(int(record["after_version"]))

    state_record = plan.get("state")
    if not isinstance(state_record, dict):
        raise ValueError("reconcile commit state plan is missing")
    state_path = campaign_owned_path(
        campaign,
        campaign / Path(str(state_record["path"])),
    )
    observed_state = snapshot_small_file(campaign, state_path)
    if observed_state.get("sha256") != state_record.get("after_sha256"):
        before = state_record.get("before")
        if not isinstance(before, Mapping) or not _snapshot_matches(campaign, before):
            raise ValueError("campaign state matches neither recorded before nor after state")
        backup = campaign_owned_path(
            campaign,
            campaign / Path(str(state_record["backup_path"])),
        )
        if bool(before.get("exists", False)):
            before_payload = before.get("payload")
            if not isinstance(before_payload, dict):
                raise ValueError("reconcile source-state payload is unavailable")
            if backup.exists():
                if backup.is_symlink() or not backup.is_file():
                    raise ValueError("reconcile state backup is unsafe")
                if _file_sha256(backup) != str(before.get("sha256") or ""):
                    raise ValueError("reconcile state backup digest mismatch")
            else:
                atomic_write_json(backup, before_payload)
        after_payload = state_record.get("after_payload")
        if not isinstance(after_payload, dict):
            raise ValueError("reconcile target-state payload is unavailable")
        write_state(state_path, CampaignState.from_dict(after_payload))

    config_record = plan.get("config_lock")
    if isinstance(config_record, Mapping):
        config_states = [
            _control_file_state(campaign, record)
            for record in config_record.get("files") or []
        ]
        if not config_states or any(
            state not in {"before", "after"} for state in config_states
        ):
            raise ValueError(
                "config lock matches neither recorded before nor after state"
            )
        if any(state == "before" for state in config_states):
            publish_reconcile_config_target(campaign, config_record)

    for record in plan.get("intent_transitions") or []:
        intent_state = _intent_state(campaign, record)
        if intent_state == "other":
            raise ValueError("submission intent matches neither recorded state")
        if intent_state == "before":
            publish_reconcile_intent_target(campaign, record)

    states = _commit_plan_states(campaign, plan)
    if not states or any(value != "after" for value in states):
        raise ValueError("reconcile authority commit did not converge")
    recovered_state = CampaignState.from_dict(dict(state_record["after_payload"]))
    contract = recovery_contract_status(
        campaign,
        recovered_state,
        verification="authority",
        artifact_snapshot=None,
    )
    if not bool(contract.get("contract_ok", False)):
        raise ValueError("recovered phase input contract is incomplete after roll-forward")


def apply_reconcile_transaction_recovery(
    campaign_dir: Path,
    inspection: Mapping[str, Any],
    *,
    artifact_snapshot: Optional[Any] = None,
) -> Dict[str, Any]:
    """Apply one previously inspected transaction repair under operator lock."""
    campaign = Path(campaign_dir).resolve()
    if not bool(inspection.get("recoverable", False)):
        raise ValueError("reconcile transaction recovery is not safe to apply")
    action = str(inspection.get("action") or "")
    record = inspection.get("record")
    if not isinstance(record, Mapping):
        raise ValueError("reconcile transaction recovery record is missing")
    source = campaign_owned_path(campaign, Path(str(record["path"])))
    if action == "archive_redundant_orphan":
        archived = _archive_orphan_temp(
            campaign,
            source,
            str(record.get("transaction_id") or "unknown"),
        )
        return {
            "changed": True,
            "action": action,
            "archived_path": str(archived),
            "transaction_id": str(record.get("transaction_id") or ""),
            "transaction_status": str(
                record.get("orphan_transaction_status") or "ORPHAN"
            ),
            "disposition": str(
                inspection.get("disposition") or "completed_safe_cleanup"
            ),
            "reason": str(
                inspection.get("reason")
                or "redundant atomic transaction update was archived"
            ),
            "phase": str(record.get("proposed_phase") or ""),
            "iteration": int(record.get("proposed_iteration") or 0),
            "requires_reinspection": True,
        }
    if action == "promote_prepared_orphan":
        payload = _read_atomic_transaction_temp(source)
        if str(payload["status"]) != "PREPARED":
            raise ValueError("only a PREPARED orphan may be promoted")
        target = campaign_owned_path(
            campaign,
            source.parent / (str(payload["transaction_id"]) + ".json"),
        )
        if target.exists():
            raise FileExistsError("canonical reconcile transaction appeared during recovery")
        os.replace(source, target)
        _fsync_parent_dir(target)
        return {
            "changed": True,
            "action": action,
            "transaction_id": str(payload["transaction_id"]),
            "requires_reinspection": True,
        }

    payload = read_reconcile_transaction(source)
    transaction = ReconcileTransaction(campaign, source, payload)
    reason = str(inspection.get("reason") or "reconcile transaction recovered")
    disposition = str(inspection.get("disposition") or "")
    if action == "roll_forward":
        try:
            _roll_forward_v2_commit(
                campaign,
                transaction,
                artifact_snapshot=artifact_snapshot,
            )
        except Exception as exc:
            transaction.record_interruption(
                "roll-forward failed: " + type(exc).__name__ + ": " + str(exc)
            )
            raise
        transaction.resolve(
            status="COMMITTED",
            disposition=disposition,
            reason=reason,
        )
    elif action == "rollback_v1_pointers":
        for snapshot in payload.get("pointer_snapshots") or []:
            restore_version_pointer(campaign, snapshot)
        from .recovery_contracts import recovery_contract_status

        current_state = _read_current_state(campaign)
        if current_state is None:
            raise ValueError("campaign state is absent after legacy pointer rollback")
        contract = recovery_contract_status(
            campaign,
            current_state,
            verification="authority",
            artifact_snapshot=None,
        )
        if not bool(contract.get("contract_ok", False)):
            raise ValueError("legacy pointer rollback did not restore a valid campaign state")
        transaction.resolve(
            status="FAILED",
            disposition=disposition,
            reason=reason,
        )
    elif action == "adopt":
        transaction.resolve(
            status="COMMITTED",
            disposition=disposition,
            reason=reason,
        )
    elif action == "abandon":
        transaction.resolve(
            status="FAILED",
            disposition=disposition,
            reason=reason,
        )
    else:
        raise ValueError("unsupported reconcile transaction recovery action: " + action)
    return {
        "changed": True,
        "action": action,
        "transaction_id": str(payload["transaction_id"]),
        "transaction_status": str(payload["status"]),
        "disposition": disposition,
        "reason": reason,
        "phase": str(payload["proposed_phase"]),
        "iteration": int(payload["proposed_iteration"]),
        "requires_reinspection": True,
    }


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
    "READABLE_RECONCILE_TRANSACTION_SCHEMA_VERSIONS",
    "ReconcileTransaction",
    "apply_reconcile_transaction_recovery",
    "begin_reconcile_transaction",
    "build_reconcile_commit_plan",
    "inspect_reconcile_transaction_recovery",
    "inventory_reconcile_transactions",
    "publish_reconcile_config_target",
    "publish_reconcile_intent_target",
    "read_reconcile_transaction",
    "restore_version_pointer",
    "snapshot_small_file",
    "snapshot_version_pointer",
]
