"""Durable user stop requests for the active-learning daemon.

The command-line process must not rewrite ``state.json`` while the daemon may
be advancing it.  Stop requests therefore use a separate, atomically replaced
control manifest.  A short-lived lock serialises CLI and daemon updates; the
daemon remains the sole writer of campaign state.
"""
from __future__ import annotations

import copy
from ..strict_json import strict_json as json
import hashlib
import os
import socket
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Union

import portalocker

from .state import CampaignPhase, CampaignState, atomic_write_json
from .filesystem import operational_data_dir


STOP_REQUEST_SCHEMA_VERSION = 2
STOP_REQUEST_FILENAME = "stop_request.json"
STOP_CONTROL_LOCK_FILENAME = "stop_control.lock"
STOP_REQUEST_HISTORY_DIRNAME = "stop_request_history"
RESUME_TRANSACTION_FILENAME = "resume_transaction.json"
RESUME_TRANSACTION_HISTORY_DIRNAME = "resume_transaction_history"
RESUME_TRANSACTION_SCHEMA_VERSION = 1
STOP_MODES = frozenset({"immediate", "after_phase", "after_iteration"})
STOP_REQUEST_STATUSES = frozenset({"requested", "cancelling", "completed"})
_STATUS_TRANSITIONS = {
    "requested": frozenset({"requested", "completed"}),
    "cancelling": frozenset({"cancelling", "requested"}),
    "completed": frozenset({"completed"}),
}


class StopControlError(RuntimeError):
    """Raised when daemon stop control is malformed or conflicts."""


def _description_nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def describe_stop_request(
    request: Any,
    *,
    completed: Optional[bool] = None,
) -> str:
    """Return one stable, fail-safe description of a stop target."""
    if not isinstance(request, Mapping):
        return "stop target unavailable"
    mode = request.get("mode")
    if completed is None:
        completed = request.get("status") == "completed"
    if mode == "immediate":
        phase = request.get("observed_phase", request.get("phase"))
        iteration = _description_nonnegative_int(request.get("observed_iteration"))
        if iteration is None:
            iteration = _description_nonnegative_int(request.get("iteration"))
        if not isinstance(phase, str) or not phase or iteration is None:
            return "stop target unavailable"
        return (
            "immediate stop requested during "
            + phase
            + " in iteration "
            + str(iteration)
        )
    if mode == "after_phase":
        phase = request.get("target_phase")
        iteration = _description_nonnegative_int(request.get("target_iteration"))
        replacement_round = _description_nonnegative_int(
            request.get("target_replacement_round")
        )
        if (
            not isinstance(phase, str)
            or not phase
            or iteration is None
            or replacement_round is None
        ):
            return "stop target unavailable"
        prefix = "stopped" if completed else "stop requested"
        description = (
            prefix + " after phase " + phase + " in iteration " + str(iteration)
        )
        if replacement_round > 0:
            description += ", replacement round " + str(replacement_round)
        return description
    if mode == "after_iteration":
        iteration = _description_nonnegative_int(request.get("target_iteration"))
        if iteration is None:
            return "stop target unavailable"
        prefix = "stopped" if completed else "stop requested"
        return prefix + " after iteration " + str(iteration)
    return "stop target unavailable"


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_dir(campaign_dir: Union[str, Path]) -> Path:
    return operational_data_dir(campaign_dir)


def stop_request_path(campaign_dir: Union[str, Path]) -> Path:
    return _data_dir(campaign_dir) / STOP_REQUEST_FILENAME


def stop_control_lock_path(campaign_dir: Union[str, Path]) -> Path:
    return _data_dir(campaign_dir) / STOP_CONTROL_LOCK_FILENAME


def stop_request_history_dir(campaign_dir: Union[str, Path]) -> Path:
    return _data_dir(campaign_dir) / STOP_REQUEST_HISTORY_DIRNAME


def resume_transaction_path(campaign_dir: Union[str, Path]) -> Path:
    return _data_dir(campaign_dir) / RESUME_TRANSACTION_FILENAME


def resume_transaction_history_dir(campaign_dir: Union[str, Path]) -> Path:
    return _data_dir(campaign_dir) / RESUME_TRANSACTION_HISTORY_DIRNAME


def _validate_resume_transaction(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise StopControlError("resume transaction must be a JSON object")
    data = dict(payload)
    if data.get("schema_version") != RESUME_TRANSACTION_SCHEMA_VERSION:
        raise StopControlError("unsupported resume-transaction schema")
    transaction_id = _required_nonempty_string(data, "transaction_id")
    try:
        uuid.UUID(transaction_id)
    except (ValueError, AttributeError) as exc:
        raise StopControlError("resume transaction ID must be a UUID") from exc
    _required_nonempty_string(data, "campaign_uid")
    request_id = data.get("request_id")
    if request_id is not None:
        if not isinstance(request_id, str) or not request_id:
            raise StopControlError("resume transaction request_id must be a string or null")
        try:
            uuid.UUID(request_id)
        except (ValueError, AttributeError) as exc:
            raise StopControlError("resume transaction request_id must be a UUID") from exc
    _required_nonempty_string(data, "operation")
    status = _required_nonempty_string(data, "status")
    if status not in {"prepared", "control_archived", "state_written"}:
        raise StopControlError("invalid resume-transaction status")
    _validate_timestamp(data.get("created_at_iso"), "resume transaction created_at_iso")
    _validate_timestamp(data.get("updated_at_iso"), "resume transaction updated_at_iso")
    before_digest = _required_nonempty_string(data, "before_state_sha256")
    after_digest = _required_nonempty_string(data, "after_state_sha256")
    if any(
        len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
        for value in (before_digest, after_digest)
    ):
        raise StopControlError("resume transaction state digests must be SHA-256 hex")
    after_state = data.get("after_state")
    if not isinstance(after_state, Mapping):
        raise StopControlError("resume transaction after_state must be an object")
    validated_state = CampaignState.from_dict(dict(after_state)).to_dict()
    if _canonical_digest(validated_state) != after_digest:
        raise StopControlError("resume transaction target-state digest mismatch")
    archive_path = data.get("stop_request_archive_path")
    if archive_path is not None and (not isinstance(archive_path, str) or not archive_path):
        raise StopControlError(
            "resume transaction stop_request_archive_path must be a string or null"
        )
    if status in {"control_archived", "state_written"} and request_id is not None:
        if archive_path is None:
            raise StopControlError("resume transaction is missing stop-request archive evidence")
    return data


def read_resume_transaction(
    campaign_dir: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    path = resume_transaction_path(campaign_dir)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise StopControlError("resume transaction is not a regular file: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StopControlError("resume transaction is unreadable: " + str(path)) from exc
    return _validate_resume_transaction(payload)


def prepare_resume_transaction(
    campaign_dir: Union[str, Path],
    *,
    before_state: CampaignState,
    after_state: CampaignState,
    request_id: Optional[str],
    operation: str,
) -> Dict[str, Any]:
    before = CampaignState.from_dict(before_state.to_dict()).to_dict()
    after = CampaignState.from_dict(after_state.to_dict()).to_dict()
    now = _now_iso()
    candidate = {
        "schema_version": RESUME_TRANSACTION_SCHEMA_VERSION,
        "transaction_id": str(uuid.uuid4()),
        "campaign_uid": str(before_state.campaign_uid),
        "request_id": None if request_id is None else str(request_id),
        "operation": str(operation),
        "status": "prepared",
        "before_state_sha256": _canonical_digest(before),
        "after_state_sha256": _canonical_digest(after),
        "after_state": after,
        "stop_request_archive_path": None,
        "created_at_iso": now,
        "updated_at_iso": now,
    }
    validated = _validate_resume_transaction(candidate)
    with stop_control_lock(campaign_dir):
        existing = read_resume_transaction(campaign_dir)
        if existing is not None:
            if (
                str(existing.get("campaign_uid")) == str(before_state.campaign_uid)
                and existing.get("request_id") == candidate.get("request_id")
                and str(existing.get("operation")) == str(operation)
                and str(existing.get("after_state_sha256"))
                == str(candidate.get("after_state_sha256"))
            ):
                return existing
            raise StopControlError("a different resume transaction is already active")
        atomic_write_json(resume_transaction_path(campaign_dir), validated)
    return validated


def update_resume_transaction(
    campaign_dir: Union[str, Path],
    transaction_id: str,
    *,
    status: str,
    stop_request_archive_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    with stop_control_lock(campaign_dir):
        current = read_resume_transaction(campaign_dir)
        if current is None or str(current.get("transaction_id")) != str(transaction_id):
            raise StopControlError("resume transaction changed or disappeared")
        old_status = str(current.get("status"))
        allowed = {
            "prepared": {"prepared", "control_archived"},
            "control_archived": {"control_archived", "state_written"},
            "state_written": {"state_written"},
        }
        if str(status) not in allowed.get(old_status, set()):
            raise StopControlError(
                "illegal resume-transaction transition: "
                + old_status
                + " -> "
                + str(status)
            )
        updated = dict(current)
        updated["status"] = str(status)
        if stop_request_archive_path is not None:
            updated["stop_request_archive_path"] = str(stop_request_archive_path)
        updated["updated_at_iso"] = _now_iso()
        validated = _validate_resume_transaction(updated)
        atomic_write_json(resume_transaction_path(campaign_dir), validated)
        return validated


def archive_resume_transaction(
    campaign_dir: Union[str, Path],
    transaction_id: str,
) -> Path:
    with stop_control_lock(campaign_dir):
        current = read_resume_transaction(campaign_dir)
        if current is None or str(current.get("transaction_id")) != str(transaction_id):
            raise StopControlError("resume transaction changed or disappeared")
        if str(current.get("status")) != "state_written":
            raise StopControlError("resume transaction is not complete")
        history = resume_transaction_history_dir(campaign_dir)
        if history.is_symlink():
            raise StopControlError("resume-transaction history path is a symlink")
        history.mkdir(parents=True, exist_ok=True)
        target = history / (str(transaction_id) + ".json")
        atomic_write_json(target, current)
        resume_transaction_path(campaign_dir).unlink()
        return target


@contextmanager
def stop_control_lock(
    campaign_dir: Union[str, Path],
    *,
    timeout: float = 10.0,
) -> Iterator[None]:
    data = _data_dir(campaign_dir)
    data.mkdir(parents=True, exist_ok=True)
    lock_path = stop_control_lock_path(campaign_dir)
    if lock_path.is_symlink():
        raise StopControlError("stop-control lock path is a symlink")
    try:
        with portalocker.Lock(
            str(lock_path),
            mode="a",
            timeout=float(timeout),
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        ):
            yield
    except (portalocker.LockException, portalocker.AlreadyLocked) as exc:
        raise StopControlError("could not acquire the stop-control lock") from exc


def _required_nonempty_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise StopControlError("stop request " + key + " must be a non-empty string")
    return value


def _required_nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StopControlError("stop request " + key + " must be a non-negative integer")
    return int(value)


def _validate_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise StopControlError(label + " must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise StopControlError(label + " must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StopControlError(label + " must include a timezone")
    return value


def _validate_cancellation_summary(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise StopControlError(
            "stop request cancellation_summary must be an object or null"
        )
    for category in ("cancelled", "skipped", "failed"):
        items = value.get(category, [])
        if not isinstance(items, list):
            raise StopControlError(
                "stop request cancellation_summary."
                + category
                + " must be a list"
            )
        for item in items:
            if not isinstance(item, Mapping):
                raise StopControlError(
                    "stop request cancellation summary entries must be objects"
                )
            job_id = item.get("job_id")
            if not isinstance(job_id, str) or (
                category != "skipped" and not job_id
            ):
                raise StopControlError(
                    "stop request cancellation job_id must be a string"
                )
            if category != "cancelled":
                reason = item.get("reason")
                if not isinstance(reason, str) or not reason:
                    raise StopControlError(
                        "stop request cancellation failure/skip reason must be a string"
                    )
                continue
            phases = item.get("phases", [])
            if not isinstance(phases, list) or any(
                not isinstance(phase, str) or not phase for phase in phases
            ):
                raise StopControlError(
                    "stop request cancelled-job phases must be a string list"
                )
            keys = item.get("intent_keys", [])
            if not isinstance(keys, list):
                raise StopControlError(
                    "stop request cancelled-job intent_keys must be a list"
                )
            for key in keys:
                if not isinstance(key, Mapping):
                    raise StopControlError(
                        "stop request cancelled-job intent key must be an object"
                    )
                _required_nonempty_string(key, "phase")
                _required_nonnegative_int(key, "iteration")


def validate_stop_request(
    payload: Any,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise StopControlError("stop request must be a JSON object")
    data = dict(payload)
    if data.get("schema_version") != STOP_REQUEST_SCHEMA_VERSION:
        raise StopControlError("unsupported stop-request schema")
    request_id = _required_nonempty_string(data, "request_id")
    try:
        uuid.UUID(request_id)
    except (ValueError, AttributeError) as exc:
        raise StopControlError("stop request request_id must be a UUID") from exc
    campaign_uid = _required_nonempty_string(data, "campaign_uid")
    if expected_campaign_uid is not None and campaign_uid != str(expected_campaign_uid):
        raise StopControlError("stop request campaign UID mismatch")
    mode = _required_nonempty_string(data, "mode")
    if mode not in STOP_MODES:
        raise StopControlError("unsupported stop-request mode " + repr(mode))
    status = _required_nonempty_string(data, "status")
    if status not in STOP_REQUEST_STATUSES:
        raise StopControlError("unsupported stop-request status " + repr(status))
    _validate_timestamp(data.get("requested_at_iso"), "stop request requested_at_iso")
    _required_nonempty_string(data, "requested_by")
    phase = _required_nonempty_string(data, "observed_phase")
    try:
        CampaignPhase(phase)
    except ValueError as exc:
        raise StopControlError("stop request observed_phase is invalid") from exc
    observed_iteration = _required_nonnegative_int(data, "observed_iteration")
    observed_round = _required_nonnegative_int(data, "observed_replacement_round")
    if not isinstance(data.get("phase_started_at_request"), bool):
        raise StopControlError("stop request phase_started_at_request must be boolean")
    if not isinstance(data.get("cancel_jobs_requested"), bool):
        raise StopControlError("stop request cancel_jobs_requested must be boolean")
    if data["cancel_jobs_requested"] and mode != "immediate":
        raise StopControlError("job cancellation is valid only for immediate stop mode")
    if mode == "after_phase":
        target_phase = _required_nonempty_string(data, "target_phase")
        try:
            CampaignPhase(target_phase)
        except ValueError as exc:
            raise StopControlError("stop request target_phase is invalid") from exc
        target_iteration = _required_nonnegative_int(data, "target_iteration")
        target_round = _required_nonnegative_int(data, "target_replacement_round")
        if (
            target_phase != phase
            or target_iteration != observed_iteration
            or target_round != observed_round
        ):
            raise StopControlError("after-phase target does not match the observed phase")
    elif mode == "after_iteration":
        target = _required_nonnegative_int(data, "target_iteration")
        if target < observed_iteration:
            raise StopControlError("after-iteration target predates the observed iteration")
        if data.get("target_phase") is not None or data.get("target_replacement_round") is not None:
            raise StopControlError("after-iteration request must not contain a phase target")
    else:
        if any(
            data.get(key) is not None
            for key in ("target_phase", "target_iteration", "target_replacement_round")
        ):
            raise StopControlError("immediate stop request must not contain a target")
    _validate_cancellation_summary(data.get("cancellation_summary"))
    completion = data.get("completion_receipt")
    if completion is not None and not isinstance(completion, Mapping):
        raise StopControlError("stop request completion_receipt must be an object or null")
    for key in ("completed_at_iso", "completion_reason"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise StopControlError("stop request " + key + " must be a string or null")
    completed_at = data.get("completed_at_iso")
    completion_reason = data.get("completion_reason")
    if status == "requested":
        if completed_at is not None or completion_reason is not None or completion is not None:
            raise StopControlError("requested stop cannot contain completion metadata")
    elif status == "cancelling":
        if mode != "immediate" or not data["cancel_jobs_requested"]:
            raise StopControlError(
                "cancelling status requires immediate mode with job cancellation"
            )
        if completed_at is not None or completion_reason is not None or completion is not None:
            raise StopControlError("cancelling stop cannot contain completion metadata")
    else:
        _validate_timestamp(completed_at, "stop request completed_at_iso")
        if not isinstance(completion_reason, str) or not completion_reason:
            raise StopControlError("completed stop requires a completion_reason")
        if (
            mode != "immediate"
            and bool(data.get("phase_started_at_request"))
            and (not isinstance(completion, Mapping) or not completion)
        ):
            raise StopControlError(
                "completed boundary stop requires a completion receipt"
            )
    return data


def read_stop_request(
    campaign_dir: Union[str, Path],
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    path = stop_request_path(campaign_dir)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise StopControlError("stop request is not a regular file: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StopControlError("stop request is unreadable: " + str(path)) from exc
    return validate_stop_request(
        payload,
        expected_campaign_uid=expected_campaign_uid,
    )


def _request_identity(payload: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        payload.get("mode"),
        payload.get("target_phase"),
        payload.get("target_iteration"),
        payload.get("target_replacement_round"),
        bool(payload.get("cancel_jobs_requested", False)),
    )


def build_stop_request(
    state: CampaignState,
    *,
    mode: str,
    target_iteration: Optional[int] = None,
    phase_started: bool = False,
    cancel_jobs: bool = False,
    requested_by: Optional[str] = None,
) -> Dict[str, Any]:
    mode_value = str(mode)
    if mode_value not in STOP_MODES:
        raise StopControlError("unsupported stop-request mode " + repr(mode_value))
    if bool(cancel_jobs) and mode_value != "immediate":
        raise StopControlError("--cancel-jobs requires immediate stop mode")
    observed_iteration = int(state.iteration)
    target_value: Optional[int] = None
    target_phase: Optional[str] = None
    target_round: Optional[int] = None
    if mode_value == "after_phase":
        target_phase = state.phase.value
        target_value = observed_iteration
        target_round = int(state.replacement_round)
    elif mode_value == "after_iteration":
        target_value = observed_iteration if target_iteration is None else int(target_iteration)
        if target_value < observed_iteration:
            raise StopControlError("after-iteration target must be >= current iteration")
        if target_value > int(state.max_iterations):
            raise StopControlError(
                "after-iteration target exceeds campaign.max_iterations"
            )
    payload = {
        "schema_version": STOP_REQUEST_SCHEMA_VERSION,
        "request_id": str(uuid.uuid4()),
        "campaign_uid": str(state.campaign_uid),
        "mode": mode_value,
        "status": "cancelling" if bool(cancel_jobs) else "requested",
        "requested_at_iso": _now_iso(),
        "requested_by": str(requested_by or (socket.gethostname() + ":" + str(os.getpid()))),
        "observed_phase": state.phase.value,
        "observed_iteration": observed_iteration,
        "observed_replacement_round": int(state.replacement_round),
        "phase_started_at_request": bool(phase_started),
        "target_phase": target_phase,
        "target_iteration": target_value,
        "target_replacement_round": target_round,
        "cancel_jobs_requested": bool(cancel_jobs),
        "cancellation_summary": None,
        "completed_at_iso": None,
        "completion_reason": None,
        "completion_receipt": None,
    }
    return validate_stop_request(payload, expected_campaign_uid=str(state.campaign_uid))


def install_stop_request(
    campaign_dir: Union[str, Path],
    request: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    candidate = validate_stop_request(request)
    path = stop_request_path(campaign_dir)
    with stop_control_lock(campaign_dir):
        existing = read_stop_request(
            campaign_dir,
            expected_campaign_uid=str(candidate["campaign_uid"]),
        )
        if existing is not None:
            if _request_identity(existing) == _request_identity(candidate):
                return existing, "existing"
            if str(existing.get("status")) == "cancelling":
                raise StopControlError(
                    "a Slurm cancellation request is already in progress; "
                    "rerun the same immediate --cancel-jobs request"
                )
            if str(candidate["mode"]) != "immediate":
                raise StopControlError(
                    "a different stop request is already active; cancel it or use --immediate"
                )
            _archive_locked(campaign_dir, existing, status="superseded")
        atomic_write_json(path, candidate)
        return dict(candidate), "created"


def update_stop_request(
    campaign_dir: Union[str, Path],
    request_id: str,
    **updates: Any,
) -> Optional[Dict[str, Any]]:
    with stop_control_lock(campaign_dir):
        current = read_stop_request(campaign_dir)
        if current is None or str(current.get("request_id")) != str(request_id):
            return None
        updated = dict(current)
        updated.update(updates)
        new_status = str(updated.get("status"))
        old_status = str(current.get("status"))
        if new_status not in _STATUS_TRANSITIONS.get(old_status, frozenset()):
            raise StopControlError(
                "illegal stop-request transition: "
                + old_status
                + " -> "
                + new_status
            )
        validated = validate_stop_request(
            updated,
            expected_campaign_uid=str(current["campaign_uid"]),
        )
        atomic_write_json(stop_request_path(campaign_dir), validated)
        return validated


def complete_stop_request(
    campaign_dir: Union[str, Path],
    request_id: str,
    *,
    reason: str,
    completion_receipt: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    return update_stop_request(
        campaign_dir,
        request_id,
        status="completed",
        completed_at_iso=_now_iso(),
        completion_reason=str(reason),
        completion_receipt=(
            None if completion_receipt is None else dict(completion_receipt)
        ),
    )


def _archive_locked(
    campaign_dir: Union[str, Path],
    request: Mapping[str, Any],
    *,
    status: str,
) -> Path:
    history = stop_request_history_dir(campaign_dir)
    if history.is_symlink():
        raise StopControlError("stop-request history path is a symlink")
    history.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(dict(request))
    payload["archived_status"] = str(status)
    payload["archived_at_iso"] = _now_iso()
    target = history / (str(request["request_id"]) + ".json")
    atomic_write_json(target, payload)
    return target


def archive_and_clear_stop_request(
    campaign_dir: Union[str, Path],
    *,
    status: str,
    expected_request_id: Optional[str] = None,
) -> Optional[Path]:
    with stop_control_lock(campaign_dir):
        current = read_stop_request(campaign_dir)
        if current is None:
            return None
        if (
            expected_request_id is not None
            and str(current.get("request_id")) != str(expected_request_id)
        ):
            raise StopControlError(
                "stop request changed while it was being archived"
            )
        target = _archive_locked(campaign_dir, current, status=status)
        stop_request_path(campaign_dir).unlink()
        return target


def stop_request_summary(request: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if request is None:
        return None
    return {
        "request_id": request.get("request_id"),
        "mode": request.get("mode"),
        "status": request.get("status"),
        "requested_at_iso": request.get("requested_at_iso"),
        "observed_phase": request.get("observed_phase"),
        "observed_iteration": request.get("observed_iteration"),
        "observed_replacement_round": request.get("observed_replacement_round"),
        "target_phase": request.get("target_phase"),
        "target_iteration": request.get("target_iteration"),
        "target_replacement_round": request.get("target_replacement_round"),
        "phase_started_at_request": request.get("phase_started_at_request"),
        "cancel_jobs_requested": request.get("cancel_jobs_requested"),
        "completed_at_iso": request.get("completed_at_iso"),
        "completion_reason": request.get("completion_reason"),
        "completion_receipt": request.get("completion_receipt"),
    }


__all__ = [
    "RESUME_TRANSACTION_FILENAME",
    "RESUME_TRANSACTION_SCHEMA_VERSION",
    "STOP_MODES",
    "STOP_REQUEST_FILENAME",
    "STOP_REQUEST_SCHEMA_VERSION",
    "StopControlError",
    "archive_and_clear_stop_request",
    "archive_resume_transaction",
    "build_stop_request",
    "complete_stop_request",
    "describe_stop_request",
    "install_stop_request",
    "prepare_resume_transaction",
    "read_resume_transaction",
    "read_stop_request",
    "stop_control_lock",
    "stop_request_history_dir",
    "stop_request_path",
    "stop_request_summary",
    "update_resume_transaction",
    "update_stop_request",
    "validate_stop_request",
]
