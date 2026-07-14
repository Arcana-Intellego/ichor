"""Durable pre-sbatch submission-intent sidecars.

The daemon writes one of these before handing control to an executor that may
call ``sbatch``. If the process dies after the scheduler accepts the job but
before ``state.json`` records the JobID, the next start can see that a submit
attempt was in progress and must run the adoption check before resubmitting.
"""
from __future__ import annotations

from ..strict_json import strict_json as json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .job_names import live_job_name
from .state import CampaignPhase, atomic_write_json
from .filesystem import operational_path


INTENT_SCHEMA_VERSION = 2
INTENT_DIR_NAME = "submission_intents"
INTENT_HISTORY_DIR_NAME = "history"
ACTIVE_STATUSES = frozenset({"PRE_SUBMIT", "SUBMITTED", "ADOPTED"})
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "SUPERSEDED"})
INTENT_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
_STATUS_TRANSITIONS = {
    "PRE_SUBMIT": frozenset({"PRE_SUBMIT", "SUBMITTED", "ADOPTED", "FAILED", "SUPERSEDED"}),
    "SUBMITTED": frozenset({"SUBMITTED", "ADOPTED", "COMPLETED", "FAILED", "SUPERSEDED"}),
    "ADOPTED": frozenset({"ADOPTED", "COMPLETED", "FAILED", "SUPERSEDED"}),
    "COMPLETED": frozenset({"COMPLETED"}),
    "FAILED": frozenset({"FAILED", "SUPERSEDED"}),
    "SUPERSEDED": frozenset({"SUPERSEDED"}),
}


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


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(label + " must be an exact JSON integer")
    if value < minimum:
        raise ValueError(label + " must be >= " + str(minimum))
    return value


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(label + " must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(label + " must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(label + " must include a timezone")
    return value


def _validate_intent_payload(
    data: Any,
    *,
    path: Path,
    phase_name: str,
    iteration: int,
    expected_campaign_uid: Optional[str],
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("submission intent must be a JSON object: " + str(path))
    if _exact_int(data.get("schema_version"), "submission intent schema_version") != INTENT_SCHEMA_VERSION:
        raise ValueError("unsupported submission intent schema: " + str(path))
    if str(data.get("phase") or "") != str(phase_name):
        raise ValueError("submission intent phase mismatch: " + str(path))
    if str(phase_name) not in {phase.value for phase in CampaignPhase}:
        raise ValueError("submission intent phase is unknown: " + str(phase_name))
    recorded_iteration = _exact_int(
        data.get("iteration"), "submission intent iteration"
    )
    if recorded_iteration != int(iteration):
        raise ValueError("submission intent iteration mismatch: " + str(path))
    campaign_uid = data.get("campaign_uid")
    if not isinstance(campaign_uid, str) or not campaign_uid:
        raise ValueError("submission intent campaign_uid must be non-empty")
    if expected_campaign_uid is not None and campaign_uid != str(expected_campaign_uid):
        raise ValueError("submission intent campaign UID mismatch")
    attempt_id = data.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or len(attempt_id) != 32
        or any(character not in "0123456789abcdef" for character in attempt_id)
    ):
        raise ValueError("submission intent attempt_id is invalid")
    sequence = _exact_int(
        data.get("attempt_sequence"), "submission intent attempt_sequence", minimum=1
    )
    replacement_round = _exact_int(
        data.get("replacement_round"), "submission intent replacement_round"
    )
    identity = data.get("submission_identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError("submission intent submission_identity must be non-empty")
    expected_name = data.get("expected_job_name")
    if not isinstance(expected_name, str) or not expected_name:
        raise ValueError("submission intent expected_job_name must be non-empty")
    recomputed = expected_job_name(
        campaign_uid,
        phase_name,
        recorded_iteration,
        replacement_round=replacement_round,
        attempt_sequence=sequence,
        attempt_id=attempt_id,
    )
    if expected_name != recomputed:
        raise ValueError("submission intent expected job name does not match identity")
    status = data.get("status")
    if status not in INTENT_STATUSES:
        raise ValueError("submission intent status is unknown: " + repr(status))
    job_id = data.get("job_id")
    if job_id is not None and (not isinstance(job_id, str) or not job_id):
        raise ValueError("submission intent job_id must be non-empty or null")
    if status in {"SUBMITTED", "ADOPTED", "COMPLETED"} and not job_id:
        raise ValueError("submission intent status " + status + " requires job_id")
    if status == "PRE_SUBMIT" and job_id is not None:
        raise ValueError("PRE_SUBMIT intent cannot already contain job_id")
    if status in {"FAILED", "SUPERSEDED"} and (
        not isinstance(data.get("reason"), str) or not str(data.get("reason")).strip()
    ):
        raise ValueError("terminal submission intent requires a reason")
    _timestamp(data.get("created_iso"), "submission intent created_iso")
    _timestamp(data.get("updated_iso"), "submission intent updated_iso")
    _timestamp(data.get("updated_at_iso"), "submission intent updated_at_iso")
    for label in ("submitted_at_iso", "adopted_at_iso", "completed_at_iso"):
        if data.get(label) is not None:
            _timestamp(data[label], "submission intent " + label)
    expected_tasks = data.get("expected_tasks")
    if expected_tasks is not None:
        data["expected_tasks"] = _exact_int(
            expected_tasks, "submission intent expected_tasks", minimum=1
        )
    job_ids_seen = data.get("job_ids_seen", [])
    if not isinstance(job_ids_seen, list) or any(
        not isinstance(value, str) or not value for value in job_ids_seen
    ):
        raise ValueError("submission intent job_ids_seen must be a list of job IDs")
    if len(job_ids_seen) != len(set(job_ids_seen)):
        raise ValueError("submission intent job_ids_seen contains duplicates")
    if job_id is not None and job_id not in job_ids_seen:
        raise ValueError("submission intent job_id is absent from job_ids_seen")
    decision_contract = data.get("decision_contract")
    if decision_contract is not None:
        data["decision_contract"] = _validated_decision_contract(decision_contract)
    return data


def _validated_decision_contract(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("submission intent decision_contract must be an object")
    contract = dict(value)
    threshold = contract.get("failure_threshold_fraction")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("submission intent failure_threshold_fraction is malformed")
    parsed_threshold = float(threshold)
    if not math.isfinite(parsed_threshold) or not 0.0 <= parsed_threshold <= 1.0:
        raise ValueError(
            "submission intent failure_threshold_fraction must be finite and in [0, 1]"
        )
    config_digest = contract.get("config_sha256")
    if (
        not isinstance(config_digest, str)
        or len(config_digest) != 64
        or any(ch not in "0123456789abcdef" for ch in config_digest)
    ):
        raise ValueError("submission intent decision config_sha256 is invalid")
    contract["failure_threshold_fraction"] = parsed_threshold
    return contract


def intent_dir(campaign_dir: Union[str, Path]) -> Path:
    return operational_path(campaign_dir, INTENT_DIR_NAME)


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
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    path = intent_path(campaign_dir, phase_name, iteration)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = _validate_intent_payload(
        data,
        path=path,
        phase_name=str(phase_name),
        iteration=int(iteration),
        expected_campaign_uid=expected_campaign_uid,
    )
    for key in (
        "resource_resolution_path",
        "resource_formula_version",
        "scratch_path_template",
    ):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("submission intent " + key + " must be a non-empty string")
    digest = data.get("resource_resolution_sha256")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
    ):
        raise ValueError("submission intent resource_resolution_sha256 is invalid")
    return data


def load_active_intent(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    data = load_intent(
        campaign_dir,
        phase_name,
        iteration,
        expected_campaign_uid=expected_campaign_uid,
    )
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
    expected_tasks: Optional[int] = None,
    decision_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(campaign_uid, str) or not campaign_uid:
        raise ValueError("submission intent campaign_uid must be a non-empty string")
    if not isinstance(phase_name, str) or not phase_name:
        raise ValueError("submission intent phase must be a non-empty string")
    iteration_value = _exact_int(iteration, "submission intent iteration")
    round_number = _exact_int(
        replacement_round,
        "submission intent replacement_round",
    )
    path = intent_path(campaign_dir, phase_name, iteration_value)
    previous = load_intent(campaign_dir, phase_name, iteration_value)
    previous_sequence = 0
    if previous is not None:
        if str(previous.get("status")) not in TERMINAL_STATUSES:
            raise ValueError("refusing to replace an active submission intent")
        previous_sequence = _exact_int(
            previous.get("attempt_sequence"),
            "submission intent attempt_sequence",
            minimum=1,
        )
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
                + str(iteration_value).zfill(6)
                + "-"
                + safe_attempt
                + ".json"
            )
        )
        history_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(history_path, previous)
    attempt_sequence = previous_sequence + 1
    attempt_id = uuid.uuid4().hex
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
        "campaign_uid": campaign_uid,
        "phase": phase_name,
        "iteration": iteration_value,
        "expected_job_name": expected_job_name(
            campaign_uid,
            phase_name,
            iteration_value,
            replacement_round=round_number,
            attempt_sequence=attempt_sequence,
            attempt_id=attempt_id,
        ),
        "status": "PRE_SUBMIT",
        "job_id": None,
        "created_iso": _now_iso(),
    }
    if expected_tasks is not None:
        parsed_expected = _exact_int(
            expected_tasks, "submission intent expected_tasks", minimum=1
        )
        payload["expected_tasks"] = parsed_expected
    if decision_contract is not None:
        payload["decision_contract"] = _validated_decision_contract(decision_contract)
    return _write_payload(path, payload)


def snapshotted_failure_threshold_fraction(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> float:
    """Return the immutable batch-failure threshold for one submission."""
    intent = load_intent(
        campaign_dir,
        phase_name,
        iteration,
        expected_campaign_uid=expected_campaign_uid,
    )
    if not isinstance(intent, dict):
        raise ValueError("submission intent is unavailable for batch decision")
    contract = intent.get("decision_contract")
    if not isinstance(contract, dict):
        raise ValueError("submission intent has no decision_contract snapshot")
    return float(contract["failure_threshold_fraction"])


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
    completion_receipt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = intent_path(campaign_dir, phase_name, iteration)
    data = load_intent(campaign_dir, phase_name, iteration)
    if data is None:
        raise FileNotFoundError("submission intent does not exist")
    if not str(data.get("campaign_uid") or ""):
        from .filesystem import operational_path

        state_path = operational_path(campaign_dir, "state.json")
        if state_path.is_file() and not state_path.is_symlink():
            try:
                state_payload = json.loads(state_path.read_text(encoding="utf-8"))
                state_uid = str(state_payload.get("campaign_uid") or "")
                if state_uid:
                    data["campaign_uid"] = state_uid
            except (OSError, ValueError, AttributeError):
                pass
    new_status = str(status)
    if new_status not in INTENT_STATUSES:
        raise ValueError("unknown submission intent status: " + repr(new_status))
    previous_status = str(data.get("status"))
    if new_status not in _STATUS_TRANSITIONS[previous_status]:
        raise ValueError(
            "illegal submission intent transition: "
            + previous_status
            + " -> "
            + new_status
        )
    data["status"] = new_status
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
        data["expected_tasks"] = _exact_int(
            expected_tasks, "submission intent expected_tasks", minimum=1
        )
    if submission_metadata:
        metadata = dict(submission_metadata)
        data["submission_metadata"] = metadata
        array_recovery = metadata.get("array_recovery")
        if isinstance(array_recovery, dict):
            data["array_recovery"] = dict(array_recovery)
            logical_total = array_recovery.get("logical_total")
            retry_count = array_recovery.get("n_retry")
            if logical_total is not None:
                data["logical_expected_tasks"] = _exact_int(
                    logical_total,
                    "submission intent logical_expected_tasks",
                    minimum=1,
                )
            if retry_count is not None:
                data["retry_expected_tasks"] = _exact_int(
                    retry_count,
                    "submission intent retry_expected_tasks",
                    minimum=1,
                )
    if job_ids_seen is not None:
        data["job_ids_seen"] = [str(x) for x in list(job_ids_seen)]
    if completion_receipt is not None:
        data["completion_receipt"] = dict(completion_receipt)
    if new_status == "SUBMITTED":
        submitted_at = data.get("submitted_at_iso") or _now_iso()
        data["submitted_at_iso"] = str(submitted_at)
        lifecycle.setdefault("submitted_at_iso", str(submitted_at))
    if new_status == "ADOPTED":
        adopted_at = data.get("adopted_at_iso") or _now_iso()
        data["adopted_at_iso"] = str(adopted_at)
        lifecycle.setdefault("adopted_at_iso", str(adopted_at))
    if new_status == "COMPLETED":
        completed_at = data.get("completed_at_iso") or _now_iso()
        data["completed_at_iso"] = str(completed_at)
        lifecycle.setdefault("completed_at_iso", str(completed_at))
    data["queue_lifecycle"] = lifecycle
    updated = _write_payload(path, data)
    return _validate_intent_payload(
        updated,
        path=path,
        phase_name=str(phase_name),
        iteration=int(iteration),
        expected_campaign_uid=None,
    )


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
        lifecycle["n_expected"] = _exact_int(
            n_expected, "submission lifecycle n_expected", minimum=1
        )
    if n_observed is not None:
        lifecycle["n_observed"] = _exact_int(
            n_observed, "submission lifecycle n_observed"
        )
    if n_missing is not None:
        lifecycle["n_missing"] = _exact_int(
            n_missing, "submission lifecycle n_missing"
        )

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
    path = intent_path(campaign_dir, phase_name, iteration)
    updated = _write_payload(path, data)
    updated = _validate_intent_payload(
        updated,
        path=path,
        phase_name=str(phase_name),
        iteration=int(iteration),
        expected_campaign_uid=None,
    )
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


def bind_resource_resolution(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    *,
    path: str,
    sha256: str,
    formula_version: str,
    scratch_path_template: str,
    expected_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    """Bind immutable resource and scheduler evidence before submission."""
    intent = load_active_intent(campaign_dir, phase_name, int(iteration))
    if intent is None or str(intent.get("status")) != "PRE_SUBMIT":
        raise ValueError("resource resolution requires an active PRE_SUBMIT intent")
    digest = str(sha256)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("resource-resolution SHA-256 is invalid")
    updates = {
        "resource_resolution_path": str(path),
        "resource_resolution_sha256": digest,
        "resource_formula_version": str(formula_version),
        "scratch_path_template": str(scratch_path_template),
    }
    for key, value in updates.items():
        previous = intent.get(key)
        if previous is not None and previous != value:
            raise ValueError("submission intent " + key + " is already bound differently")
        intent[key] = value
    if expected_tasks is not None:
        if isinstance(expected_tasks, bool):
            raise ValueError("submission intent expected_tasks is malformed")
        parsed_expected_tasks = _exact_int(
            expected_tasks, "submission intent expected_tasks", minimum=1
        )
        # The initial PRE_SUBMIT value can describe the unrecovered logical
        # array.  Once staging has produced a dense retry array, this field
        # must snapshot the task count that Slurm will actually report.
        intent["expected_tasks"] = parsed_expected_tasks
    target = intent_path(campaign_dir, phase_name, int(iteration))
    updated = _write_payload(target, intent)
    return _validate_intent_payload(
        updated,
        path=target,
        phase_name=str(phase_name),
        iteration=int(iteration),
        expected_campaign_uid=None,
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


def mark_completed(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    *,
    completion_receipt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return update_intent_status(
        campaign_dir, phase_name=phase_name, iteration=iteration,
        status="COMPLETED",
        completion_receipt=completion_receipt,
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
