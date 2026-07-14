"""Durable operator stop requests for the active-learning daemon.

The command-line process must not rewrite ``state.json`` while the daemon may
be advancing it.  Stop requests therefore use a separate, atomically replaced
control manifest.  A short-lived lock serialises CLI and daemon updates; the
daemon remains the sole writer of campaign state.
"""
from __future__ import annotations

import copy
from ..strict_json import strict_json as json
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


STOP_REQUEST_SCHEMA_VERSION = 1
STOP_REQUEST_FILENAME = "stop_request.json"
STOP_CONTROL_LOCK_FILENAME = "stop_control.lock"
STOP_REQUEST_HISTORY_DIRNAME = "stop_request_history"
STOP_MODES = frozenset({"immediate", "after_phase", "after_iteration"})
STOP_REQUEST_STATUSES = frozenset({"requested", "cancelling", "completed"})


class StopControlError(RuntimeError):
    """Raised when daemon stop control is malformed or conflicts."""


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
    _required_nonempty_string(data, "requested_at_iso")
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
    "STOP_MODES",
    "STOP_REQUEST_FILENAME",
    "STOP_REQUEST_SCHEMA_VERSION",
    "StopControlError",
    "archive_and_clear_stop_request",
    "build_stop_request",
    "complete_stop_request",
    "install_stop_request",
    "read_stop_request",
    "stop_control_lock",
    "stop_request_history_dir",
    "stop_request_path",
    "stop_request_summary",
    "update_stop_request",
    "validate_stop_request",
]
