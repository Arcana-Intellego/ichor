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
ACTIVE_STATUSES = frozenset({"PRE_SUBMIT", "SUBMITTED", "ADOPTED"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def intent_dir(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / INTENT_DIR_NAME


def intent_path(campaign_dir: Union[str, Path], phase_name: str, iteration: int) -> Path:
    safe_phase = str(phase_name).replace("/", "_").replace("\\", "_")
    return intent_dir(campaign_dir) / (
        safe_phase + "-" + str(int(iteration)).zfill(4) + ".json"
    )


def expected_job_name(campaign_uid: Optional[str], phase_name: str, iteration: int) -> str:
    return live_job_name(campaign_uid, phase_name, iteration)


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
) -> Dict[str, Any]:
    path = intent_path(campaign_dir, phase_name, iteration)
    payload: Dict[str, Any] = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "attempt_id": uuid.uuid4().hex,
        "campaign_uid": str(campaign_uid),
        "phase": str(phase_name),
        "iteration": int(iteration),
        "expected_job_name": expected_job_name(campaign_uid, phase_name, iteration),
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
    if job_ids_seen is not None:
        data["job_ids_seen"] = [str(x) for x in list(job_ids_seen)]
    return _write_payload(path, data)


def mark_submitted(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    job_id: str,
    *,
    expected_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    return update_intent_status(
        campaign_dir, phase_name=phase_name, iteration=iteration,
        status="SUBMITTED", job_id=str(job_id), expected_tasks=expected_tasks,
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
