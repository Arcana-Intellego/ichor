"""Durable pre-sbatch submission-intent sidecars.

The daemon writes one of these before handing control to an executor that may
call ``sbatch``. If the process dies after the scheduler accepts the job but
before ``state.json`` records the JobID, the next start can see that a submit
attempt was in progress and must run the adoption check before resubmitting.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .job_names import live_job_name
from .state import atomic_write_json


INTENT_SCHEMA_VERSION = 1
INTENT_DIR_NAME = "submission_intents"
INTENT_HISTORY_DIR_NAME = "history"
ACTIVE_STATUSES = frozenset({"PRE_SUBMIT", "SUBMITTED", "ADOPTED"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_seconds(start: Optional[str], end: Optional[str]) -> Optional[float]:
    if not start or not end:
        return None
    try:
        start_dt = datetime.fromisoformat(str(start))
        end_dt = datetime.fromisoformat(str(end))
    except ValueError:
        return None
    return max(0.0, float((end_dt - start_dt).total_seconds()))


def intent_dir(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / INTENT_DIR_NAME


def intent_path(campaign_dir: Union[str, Path], phase_name: str, iteration: int) -> Path:
    safe_phase = str(phase_name).replace("/", "_").replace("\\", "_")
    return intent_dir(campaign_dir) / (
        safe_phase + "-" + str(int(iteration)).zfill(6) + ".json"
    )


def expected_job_name(
    campaign_uid: Optional[str],
    phase_name: str,
    iteration: int,
    *,
    replacement_round: int = 0,
    attempt_sequence: Optional[int] = None,
    attempt_id: Optional[str] = None,
) -> str:
    return live_job_name(
        campaign_uid,
        phase_name,
        iteration,
        replacement_round=replacement_round,
        attempt_sequence=attempt_sequence,
        attempt_id=attempt_id,
    )


def load_intent(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    path = intent_path(campaign_dir, phase_name, iteration)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("submission intent must be a JSON object: " + str(path))
    if int(data.get("schema_version", -1)) != INTENT_SCHEMA_VERSION:
        raise ValueError("unsupported submission intent schema: " + str(path))
    recorded_phase = data.get("phase")
    if str(recorded_phase) != str(phase_name):
        raise ValueError(
            "submission intent phase mismatch for "
            + str(path)
            + ": expected "
            + str(phase_name)
            + " got "
            + repr(recorded_phase)
        )
    try:
        recorded_iteration = int(data.get("iteration"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "submission intent iteration is malformed for " + str(path)
        ) from exc
    if recorded_iteration != int(iteration):
        raise ValueError(
            "submission intent iteration mismatch for "
            + str(path)
            + ": expected "
            + str(int(iteration))
            + " got "
            + str(recorded_iteration)
        )
    status = data.get("status")
    if status is not None and not isinstance(status, str):
        raise ValueError("submission intent status must be a string: " + str(path))
    expected = data.get("expected_job_name")
    if expected is not None and not isinstance(expected, str):
        raise ValueError(
            "submission intent expected_job_name must be a string: " + str(path)
        )
    identity = data.get("submission_identity")
    if identity is not None:
        if not isinstance(identity, str) or not identity:
            raise ValueError("submission intent identity must be a non-empty string")
        try:
            sequence = int(data.get("attempt_sequence"))
            replacement_round = int(data.get("replacement_round", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("submission intent attempt identity is malformed") from exc
        attempt_id = str(data.get("attempt_id") or "")
        if sequence <= 0 or replacement_round < 0 or not attempt_id:
            raise ValueError("submission intent attempt identity is invalid")
        recomputed = expected_job_name(
            data.get("campaign_uid"),
            phase_name,
            int(iteration),
            replacement_round=replacement_round,
            attempt_sequence=sequence,
            attempt_id=attempt_id,
        )
        if expected != recomputed:
            raise ValueError("submission intent expected job name does not match identity")
    return data


def load_active_intent(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    data = load_intent(campaign_dir, phase_name, iteration)
    if data is None:
        return None
    if str(data.get("status")) in ACTIVE_STATUSES:
        return data
    return None


def _write_payload(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now_iso()
    payload["updated_iso"] = now
    payload["updated_at_iso"] = now
    atomic_write_json(path, payload)
    return payload


def write_pre_submit_intent(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    phase_name: str,
    iteration: int,
    replacement_round: int = 0,
) -> Dict[str, Any]:
    path = intent_path(campaign_dir, phase_name, iteration)
    previous = load_intent(campaign_dir, phase_name, iteration)
    previous_sequence = 0
    if previous is not None:
        try:
            previous_sequence = max(0, int(previous.get("attempt_sequence", 0)))
        except (TypeError, ValueError):
            previous_sequence = 0
        previous_attempt = str(previous.get("attempt_id") or "legacy")
        safe_attempt = "".join(ch for ch in previous_attempt if ch.isalnum())[:32]
        if not safe_attempt:
            safe_attempt = "legacy"
        history_path = (
            intent_dir(campaign_dir)
            / INTENT_HISTORY_DIR_NAME
            / (
                str(phase_name).replace("/", "_").replace("\\", "_")
                + "-"
                + str(int(iteration)).zfill(6)
                + "-"
                + safe_attempt
                + ".json"
            )
        )
        history_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(history_path, previous)
    attempt_sequence = previous_sequence + 1
    attempt_id = uuid.uuid4().hex
    round_number = max(0, int(replacement_round))
    identity = (
        "r"
        + str(round_number).zfill(4)
        + "-a"
        + str(attempt_sequence).zfill(4)
        + "-"
        + attempt_id[:8]
    )
    payload: Dict[str, Any] = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "attempt_sequence": int(attempt_sequence),
        "replacement_round": int(round_number),
        "submission_identity": identity,
        "campaign_uid": str(campaign_uid),
        "phase": str(phase_name),
        "iteration": int(iteration),
        "expected_job_name": expected_job_name(
            campaign_uid,
            phase_name,
            iteration,
            replacement_round=round_number,
            attempt_sequence=attempt_sequence,
            attempt_id=attempt_id,
        ),
        "status": "PRE_SUBMIT",
        "job_id": None,
        "created_iso": _now_iso(),
    }
    return _write_payload(path, payload)


def update_intent_status(
    campaign_dir: Union[str, Path],
    *,
    phase_name: str,
    iteration: int,
    status: str,
    job_id: Optional[str] = None,
    reason: Optional[str] = None,
    expected_tasks: Optional[int] = None,
    job_ids_seen: Optional[Any] = None,
    submission_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = intent_path(campaign_dir, phase_name, iteration)
    data = load_intent(campaign_dir, phase_name, iteration) or {
        "schema_version": INTENT_SCHEMA_VERSION,
        "attempt_id": uuid.uuid4().hex,
        "phase": str(phase_name),
        "iteration": int(iteration),
        "created_iso": _now_iso(),
    }
    data["status"] = str(status)
    lifecycle = data.get("queue_lifecycle")
    if not isinstance(lifecycle, dict):
        lifecycle = {}
    if job_id is not None:
        data["job_id"] = str(job_id)
        seen = data.get("job_ids_seen", [])
        if not isinstance(seen, list):
            seen = []
        if str(job_id) not in [str(x) for x in seen]:
            seen.append(str(job_id))
        data["job_ids_seen"] = seen
    if reason is not None:
        data["reason"] = str(reason)
    if expected_tasks is not None:
        data["expected_tasks"] = int(expected_tasks)
    if submission_metadata:
        metadata = dict(submission_metadata)
        data["submission_metadata"] = metadata
        array_recovery = metadata.get("array_recovery")
        if isinstance(array_recovery, dict):
            data["array_recovery"] = dict(array_recovery)
            logical_total = array_recovery.get("logical_total")
            retry_count = array_recovery.get("n_retry")
            try:
                if logical_total is not None:
                    data["logical_expected_tasks"] = int(logical_total)
            except (TypeError, ValueError):
                pass
            try:
                if retry_count is not None:
                    data["retry_expected_tasks"] = int(retry_count)
            except (TypeError, ValueError):
                pass
    if job_ids_seen is not None:
        data["job_ids_seen"] = [str(x) for x in list(job_ids_seen)]
    if str(status) == "SUBMITTED":
        submitted_at = data.get("submitted_at_iso") or _now_iso()
        data["submitted_at_iso"] = str(submitted_at)
        lifecycle.setdefault("submitted_at_iso", str(submitted_at))
    if str(status) == "ADOPTED":
        adopted_at = data.get("adopted_at_iso") or _now_iso()
        data["adopted_at_iso"] = str(adopted_at)
        lifecycle.setdefault("adopted_at_iso", str(adopted_at))
    if str(status) == "COMPLETED":
        completed_at = data.get("completed_at_iso") or _now_iso()
        data["completed_at_iso"] = str(completed_at)
        lifecycle.setdefault("completed_at_iso", str(completed_at))
    data["queue_lifecycle"] = lifecycle
    return _write_payload(path, data)


def record_queue_lifecycle(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    event: str,
    *,
    job_id: Optional[str] = None,
    status: Optional[str] = None,
    n_expected: Optional[int] = None,
    n_observed: Optional[int] = None,
    n_missing: Optional[int] = None,
    rows_sample: Optional[Any] = None,
) -> Dict[str, Any]:
    data = load_intent(campaign_dir, phase_name, iteration)
    if data is None:
        return {"changed_keys": [], "intent": None}
    if job_id is not None:
        recorded_job = data.get("job_id")
        if recorded_job is not None and str(recorded_job) != str(job_id):
            return {"changed_keys": [], "intent": data}
    lifecycle = data.get("queue_lifecycle")
    if not isinstance(lifecycle, dict):
        lifecycle = {}
    changed: list[str] = []
    now = _now_iso()

    def set_once(key: str, value: Any) -> None:
        if key not in lifecycle:
            lifecycle[key] = value
            changed.append(key)

    event_name = str(event)
    if event_name == "first_sacct":
        set_once("first_sacct_at_iso", now)
        if status is not None:
            set_once("first_sacct_status", str(status))
    elif event_name == "first_squeue":
        set_once("first_squeue_at_iso", now)
        if status is not None:
            set_once("first_squeue_status", str(status))
        if rows_sample is not None:
            set_once("first_squeue_rows_sample", list(rows_sample)[:5])
    elif event_name == "terminal":
        set_once("terminal_at_iso", now)
        if status is not None:
            set_once("terminal_status", str(status))
    elif event_name == "postprocess_started":
        lifecycle["postprocess_started_at_iso"] = now
        changed.append("postprocess_started_at_iso")
    elif event_name == "postprocess_finished":
        lifecycle["postprocess_finished_at_iso"] = now
        changed.append("postprocess_finished_at_iso")
    else:
        set_once(event_name + "_at_iso", now)
    if n_expected is not None:
        lifecycle["n_expected"] = int(n_expected)
    if n_observed is not None:
        lifecycle["n_observed"] = int(n_observed)
    if n_missing is not None:
        lifecycle["n_missing"] = int(n_missing)

    submitted_at = lifecycle.get("submitted_at_iso") or data.get("submitted_at_iso")
    first_seen = lifecycle.get("first_squeue_at_iso") or lifecycle.get("first_sacct_at_iso")
    queue_wait = _duration_seconds(
        str(submitted_at) if submitted_at else None,
        str(first_seen) if first_seen else None,
    )
    if queue_wait is not None:
        lifecycle["queue_wait_seconds"] = queue_wait
    postprocess_seconds = _duration_seconds(
        str(lifecycle.get("postprocess_started_at_iso") or ""),
        str(lifecycle.get("postprocess_finished_at_iso") or ""),
    )
    if postprocess_seconds is not None:
        lifecycle["postprocess_seconds"] = postprocess_seconds
    data["queue_lifecycle"] = lifecycle
    updated = _write_payload(intent_path(campaign_dir, phase_name, iteration), data)
    return {"changed_keys": changed, "intent": updated}


def mark_submitted(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    job_id: str,
    *,
    expected_tasks: Optional[int] = None,
    submission_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return update_intent_status(
        campaign_dir, phase_name=phase_name, iteration=iteration,
        status="SUBMITTED", job_id=str(job_id), expected_tasks=expected_tasks,
        submission_metadata=submission_metadata,
    )


def mark_adopted(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    job_id: str,
    *,
    expected_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    return update_intent_status(
        campaign_dir, phase_name=phase_name, iteration=iteration,
        status="ADOPTED", job_id=str(job_id), expected_tasks=expected_tasks,
    )


def mark_completed(campaign_dir: Union[str, Path], phase_name: str, iteration: int) -> Dict[str, Any]:
    return update_intent_status(
        campaign_dir, phase_name=phase_name, iteration=iteration,
        status="COMPLETED",
    )


def mark_failed(campaign_dir: Union[str, Path], phase_name: str, iteration: int, reason: str) -> Dict[str, Any]:
    return update_intent_status(
        campaign_dir, phase_name=phase_name, iteration=iteration,
        status="FAILED", reason=reason,
    )


def mark_superseded(campaign_dir: Union[str, Path], phase_name: str, iteration: int, reason: str) -> Dict[str, Any]:
    return update_intent_status(
        campaign_dir, phase_name=phase_name, iteration=iteration,
        status="SUPERSEDED", reason=reason,
    )
