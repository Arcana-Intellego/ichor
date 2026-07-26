"""Durable pre-sbatch submission-intent sidecars.

The daemon writes one of these before handing control to an executor that may
call ``sbatch``. If the process dies after the scheduler accepts the job but
before ``state.json`` records the JobID, the next start can see that a submit
attempt was in progress and must run the adoption check before resubmitting.
"""
from __future__ import annotations

import hashlib
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from ..strict_json import strict_json as json
from ..submit.slurm_contracts import validate_parent_job_id
from .filesystem import operational_path
from .job_names import live_job_name
from .state import CampaignPhase, CampaignState, atomic_write_json


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

_SCALAR_SUBMISSION_PHASES = frozenset({
    "PHASE_A_DIVERSITY",
    "PHASE_B_DIVERSITY",
})

_POSTPROCESS_SOURCE_KEYS = frozenset({
    "campaign_uid",
    "phase",
    "iteration",
    "attempt_id",
    "submission_identity",
    "job_id",
    "environment_generation",
    "environment_generation_digest_sha256",
    "logical_total",
    "logical_task_set_sha256",
    "decision_contract",
    "source_sha256",
})
_AIMALL_POSTPROCESS_PHASES = frozenset({
    CampaignPhase.INITIAL_AIMALL.value,
    CampaignPhase.AIMALL.value,
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value,
    CampaignPhase.REPLACEMENT_AIMALL.value,
})
_POSTPROCESS_SOURCE_PHASES = (
    frozenset({CampaignPhase.ARIADNE_ARRAY.value})
    | _AIMALL_POSTPROCESS_PHASES
)


def submission_kind_for_phase(phase_name: str) -> str:
    return "scalar" if str(phase_name) in _SCALAR_SUBMISSION_PHASES else "array"


def _validate_job_identity(value: Any, scheduler_identity_kind: str) -> str:
    text = str(value)
    if scheduler_identity_kind == "slurm":
        return validate_parent_job_id(text)
    if scheduler_identity_kind == "sge":
        from ..submit.sge import validate_sge_parent_job_id

        return validate_sge_parent_job_id(text)
    if scheduler_identity_kind != "synthetic":
        raise ValueError("submission intent scheduler identity kind is invalid")
    if (
        not text
        or text.isdigit()
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", text) is None
    ):
        raise ValueError("synthetic scheduler identity is invalid")
    return text


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
    scheduler_identity_kind = data.get("scheduler_identity_kind")
    if scheduler_identity_kind not in {"slurm", "sge", "synthetic"}:
        raise ValueError(
            "submission intent scheduler_identity_kind must be slurm, sge, or synthetic"
        )
    recomputed = expected_job_name(
        campaign_uid,
        phase_name,
        recorded_iteration,
        replacement_round=replacement_round,
        attempt_sequence=sequence,
        attempt_id=attempt_id,
        scheduler_identity_kind=str(scheduler_identity_kind),
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
    if job_id is not None:
        try:
            data["job_id"] = _validate_job_identity(
                job_id, str(scheduler_identity_kind)
            )
        except ValueError as exc:
            raise ValueError("submission intent job_id is invalid") from exc
    submission_kind = data.get("submission_kind")
    expected_kind = submission_kind_for_phase(str(phase_name))
    if submission_kind != expected_kind:
        raise ValueError(
            "submission intent submission_kind must be " + expected_kind
        )
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
    for seen_job_id in job_ids_seen:
        try:
            _validate_job_identity(seen_job_id, str(scheduler_identity_kind))
        except ValueError as exc:
            raise ValueError("submission intent job_ids_seen contains an invalid JobID") from exc
    if job_id is not None and job_id not in job_ids_seen:
        raise ValueError("submission intent job_id is absent from job_ids_seen")
    decision_contract = data.get("decision_contract")
    if decision_contract is not None:
        data["decision_contract"] = _validated_decision_contract(decision_contract)
    environment_generation = data.get("environment_generation")
    environment_digest = data.get("environment_generation_digest_sha256")
    if (environment_generation is None) != (environment_digest is None):
        raise ValueError(
            "submission intent environment generation and digest must be paired"
        )
    if environment_generation is not None:
        data["environment_generation"] = _exact_int(
            environment_generation,
            "submission intent environment_generation",
        )
        if (
            not isinstance(environment_digest, str)
            or len(environment_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in environment_digest)
        ):
            raise ValueError(
                "submission intent environment generation digest is invalid"
            )
    postprocess_source = data.get("postprocess_source")
    if postprocess_source is not None:
        source = _validated_postprocess_source(
            postprocess_source
        )
        if str(phase_name) not in _POSTPROCESS_SOURCE_PHASES:
            raise ValueError(
                "postprocess_source is invalid for this submission phase"
            )
        if str(source["phase"]) != str(phase_name):
            raise ValueError("postprocess source phase differs from its intent")
        if data.get("job_id") is not None:
            raise ValueError("postprocess intent must remain jobless")
        if data.get("expected_tasks") != int(source["logical_total"]):
            raise ValueError(
                "postprocess intent expected_tasks mismatch"
            )
        if data.get("decision_contract") != source["decision_contract"]:
            raise ValueError(
                "postprocess intent decision contract differs from its producer"
            )
        data["postprocess_source"] = source
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


def _canonical_postprocess_source_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("source_sha256", None)
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validated_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(label + " must be a SHA-256 digest")
    return value


def _validated_postprocess_source(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("submission intent postprocess_source must be an object")
    source = dict(value)
    if set(source) != _POSTPROCESS_SOURCE_KEYS:
        missing = sorted(_POSTPROCESS_SOURCE_KEYS - set(source))
        unexpected = sorted(set(source) - _POSTPROCESS_SOURCE_KEYS)
        raise ValueError(
            "submission intent postprocess_source keys are invalid; missing="
            + repr(missing)
            + ", unexpected="
            + repr(unexpected)
        )
    campaign_uid = source.get("campaign_uid")
    if not isinstance(campaign_uid, str) or not campaign_uid:
        raise ValueError("postprocess source campaign_uid must be non-empty")
    if source.get("phase") not in _POSTPROCESS_SOURCE_PHASES:
        raise ValueError("postprocess source phase is unsupported")
    source["iteration"] = _exact_int(
        source.get("iteration"), "postprocess source iteration"
    )
    attempt_id = source.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or len(attempt_id) != 32
        or any(character not in "0123456789abcdef" for character in attempt_id)
    ):
        raise ValueError("postprocess source attempt_id is invalid")
    submission_identity = source.get("submission_identity")
    if not isinstance(submission_identity, str) or not submission_identity:
        raise ValueError("postprocess source submission_identity must be non-empty")
    source["job_id"] = validate_parent_job_id(str(source.get("job_id") or ""))
    source["environment_generation"] = _exact_int(
        source.get("environment_generation"),
        "postprocess source environment_generation",
    )
    source["environment_generation_digest_sha256"] = _validated_sha256(
        source.get("environment_generation_digest_sha256"),
        "postprocess source environment generation digest",
    )
    source["logical_total"] = _exact_int(
        source.get("logical_total"),
        "postprocess source logical_total",
        minimum=1,
    )
    source["logical_task_set_sha256"] = _validated_sha256(
        source.get("logical_task_set_sha256"),
        "postprocess source logical task-set digest",
    )
    source["decision_contract"] = _validated_decision_contract(
        source.get("decision_contract")
    )
    recorded_digest = _validated_sha256(
        source.get("source_sha256"),
        "postprocess source digest",
    )
    if recorded_digest != _canonical_postprocess_source_sha256(source):
        raise ValueError("postprocess source digest does not match its content")
    source["source_sha256"] = recorded_digest
    return source


def _validate_postprocess_source_environment(
    campaign_dir: Union[str, Path],
    source: Mapping[str, Any],
) -> None:
    from ..execution_identity import read_environment_generation

    generation = read_environment_generation(
        campaign_dir,
        generation=int(source["environment_generation"]),
        expected_campaign_uid=str(source["campaign_uid"]),
    )
    if str(generation["digest_sha256"]) != str(
        source["environment_generation_digest_sha256"]
    ):
        raise ValueError(
            "postprocess source environment generation digest mismatch"
        )


def resolve_ariadne_postprocess_source(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    iteration: int,
    logical_total: int,
    logical_task_set_sha256: str,
    intent: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve one full-array producer through local postprocess retries."""
    expected_iteration = _exact_int(iteration, "postprocess source iteration")
    expected_total = _exact_int(
        logical_total, "postprocess source logical_total", minimum=1
    )
    expected_task_digest = _validated_sha256(
        logical_task_set_sha256,
        "postprocess source logical task-set digest",
    )
    current = (
        dict(intent)
        if isinstance(intent, Mapping)
        else load_intent(
            campaign_dir,
            CampaignPhase.ARIADNE_ARRAY.value,
            expected_iteration,
            expected_campaign_uid=str(campaign_uid),
        )
    )
    if not current:
        raise ValueError("all-complete ARIADNE producer intent is unavailable")
    if str(current.get("campaign_uid") or "") != str(campaign_uid):
        raise ValueError("all-complete ARIADNE producer campaign UID mismatch")
    if str(current.get("phase") or "") != CampaignPhase.ARIADNE_ARRAY.value:
        raise ValueError("all-complete ARIADNE producer phase mismatch")
    if int(current.get("iteration", -1)) != expected_iteration:
        raise ValueError("all-complete ARIADNE producer iteration mismatch")

    status = str(current.get("status") or "")
    reason = str(current.get("reason") or "")
    nested = current.get("postprocess_source")
    if nested is not None:
        if status not in {"PRE_SUBMIT", "FAILED", "SUPERSEDED"}:
            raise ValueError(
                "postprocess source wrapper must be PRE_SUBMIT, FAILED or SUPERSEDED"
            )
        if status == "SUPERSEDED" and reason != "reconcile_apply_retry":
            raise ValueError(
                "superseded postprocess source wrapper has an unsupported reason"
            )
        if status == "PRE_SUBMIT" and current.get("job_id") is not None:
            raise ValueError("jobless postprocess intent unexpectedly owns a JobID")
        source = _validated_postprocess_source(nested)
    else:
        if status != "FAILED" and not (
            status == "SUPERSEDED" and reason == "reconcile_apply_retry"
        ):
            raise ValueError(
                "all-complete ARIADNE producer must be FAILED or "
                "SUPERSEDED by reconcile_apply_retry"
            )
        job_id = current.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("all-complete ARIADNE producer has no scheduler JobID")
        recovery = current.get("array_recovery")
        metadata = current.get("submission_metadata")
        if not isinstance(recovery, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError(
                "all-complete ARIADNE producer lacks retry ownership evidence"
            )
        raw_counts = (
            recovery.get("logical_total"),
            recovery.get("n_reuse"),
            recovery.get("n_retry"),
            current.get("logical_expected_tasks"),
            current.get("retry_expected_tasks"),
            current.get("expected_tasks"),
        )
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in raw_counts
        ):
            raise ValueError("all-complete ARIADNE producer task counts are malformed")
        if tuple(int(item) for item in raw_counts) != (
            expected_total,
            0,
            expected_total,
            expected_total,
            expected_total,
            expected_total,
        ):
            raise ValueError(
                "all-complete ARIADNE outputs are not bound to one full-array "
                "producer attempt"
            )
        if str(metadata.get("logical_task_set_sha256") or "") != expected_task_digest:
            raise ValueError("all-complete ARIADNE producer task-set digest mismatch")
        source = {
            "campaign_uid": str(campaign_uid),
            "phase": CampaignPhase.ARIADNE_ARRAY.value,
            "iteration": expected_iteration,
            "attempt_id": str(current.get("attempt_id") or ""),
            "submission_identity": str(current.get("submission_identity") or ""),
            "job_id": str(job_id),
            "environment_generation": current.get("environment_generation"),
            "environment_generation_digest_sha256": current.get(
                "environment_generation_digest_sha256"
            ),
            "logical_total": expected_total,
            "logical_task_set_sha256": expected_task_digest,
            "decision_contract": current.get("decision_contract"),
        }
        source["source_sha256"] = _canonical_postprocess_source_sha256(source)
        source = _validated_postprocess_source(source)

    expected_identity = (
        str(campaign_uid),
        CampaignPhase.ARIADNE_ARRAY.value,
        expected_iteration,
        expected_total,
        expected_task_digest,
    )
    observed_identity = (
        str(source["campaign_uid"]),
        str(source["phase"]),
        int(source["iteration"]),
        int(source["logical_total"]),
        str(source["logical_task_set_sha256"]),
    )
    if observed_identity != expected_identity:
        raise ValueError(
            "postprocess source does not match the current ARIADNE task set"
        )
    _validate_postprocess_source_environment(campaign_dir, source)
    return source


def aimall_postprocess_task_contract(
    campaign_dir: Union[str, Path],
    *,
    phase_name: str,
    iteration: int,
    replacement_round: int = 0,
    require_unpublished: bool = True,
) -> Dict[str, Any]:
    """Validate the filtered Gaussian handoff owned by one AIMAll attempt."""
    phase = str(phase_name)
    if phase not in _AIMALL_POSTPROCESS_PHASES:
        raise ValueError("AIMAll postprocess phase is unsupported: " + phase)
    iteration_value = _exact_int(iteration, "AIMAll postprocess iteration")
    replacement_value = _exact_int(
        replacement_round,
        "AIMAll postprocess replacement round",
    )
    campaign = Path(campaign_dir)
    if "REPLACEMENT" in phase:
        from ..replacement_sampling import replacement_round_dir

        context = "bootstrap" if phase.startswith("INITIAL_") else "active"
        staging = replacement_round_dir(
            campaign,
            context=context,
            iteration=(0 if context == "bootstrap" else iteration_value),
            replacement_round=replacement_value,
        )
    else:
        from ..layout import staging_phase_dir

        staging = staging_phase_dir(campaign, phase, iteration_value)
    gaussian_phase = (
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value
        if phase == CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value
        else CampaignPhase.REPLACEMENT_GAUSSIAN.value
        if phase == CampaignPhase.REPLACEMENT_AIMALL.value
        else CampaignPhase.INITIAL_GAUSSIAN.value
        if phase == CampaignPhase.INITIAL_AIMALL.value
        else CampaignPhase.GAUSSIAN.value
    )
    from . import input_staging as _stg

    accepted, manifest = _stg.read_quantum_acceptance_manifest(
        staging,
        expected_phase=gaussian_phase,
        expected_iteration=iteration_value,
        require_nonempty=False,
        points_membership=_stg.POINTS_MEMBERSHIP_ACCEPTED_ONLY,
    )
    if not accepted:
        raise ValueError("AIMAll postprocess source has no accepted Gaussian tasks")
    aimall_publication = _stg.quantum_acceptance_manifest_path(
        staging,
        phase_name=phase,
    )
    if (
        bool(require_unpublished)
        and (aimall_publication.exists() or aimall_publication.is_symlink())
    ):
        raise ValueError(
            "AIMAll postprocess source already has an acceptance publication"
        )
    logical_total = len(accepted)
    task_digest = hashlib.sha256(
        ",".join(str(task_id) for task_id in range(logical_total)).encode(
            "ascii"
        )
    ).hexdigest()
    return {
        "phase": phase,
        "iteration": iteration_value,
        "replacement_round": replacement_value,
        "staging": str(Path(staging)),
        "gaussian_phase": gaussian_phase,
        "gaussian_n_total": int(manifest["n_total"]),
        "logical_total": int(logical_total),
        "logical_task_set_sha256": task_digest,
    }


def aimall_intent_claims_completed_array(intent: Mapping[str, Any]) -> bool:
    """Return whether an AIMAll intent claims a complete producer array."""
    if str(intent.get("phase") or "") not in _AIMALL_POSTPROCESS_PHASES:
        return False
    if isinstance(intent.get("postprocess_source"), Mapping):
        return True
    expected = intent.get("expected_tasks")
    lifecycle = intent.get("queue_lifecycle")
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected < 1
        or not isinstance(lifecycle, Mapping)
    ):
        return False
    return (
        str(lifecycle.get("terminal_status") or "") == "COMPLETED"
        and lifecycle.get("n_expected") == expected
        and lifecycle.get("n_observed") == expected
        and lifecycle.get("n_missing") == 0
    )


def resolve_aimall_postprocess_source(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    phase_name: str,
    iteration: int,
    replacement_round: int = 0,
    intent: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve one scheduler-complete AIMAll producer for local postprocessing."""
    contract = aimall_postprocess_task_contract(
        campaign_dir,
        phase_name=str(phase_name),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
    )
    phase = str(contract["phase"])
    expected_iteration = int(contract["iteration"])
    expected_round = int(contract["replacement_round"])
    expected_total = int(contract["logical_total"])
    expected_task_digest = str(contract["logical_task_set_sha256"])
    current = (
        dict(intent)
        if isinstance(intent, Mapping)
        else load_intent(
            campaign_dir,
            phase,
            expected_iteration,
            expected_campaign_uid=str(campaign_uid),
        )
    )
    if not current:
        raise ValueError("scheduler-complete AIMAll producer intent is unavailable")
    observed_identity = (
        str(current.get("campaign_uid") or ""),
        str(current.get("phase") or ""),
        int(current.get("iteration", -1)),
        int(current.get("replacement_round", -1)),
    )
    expected_identity = (
        str(campaign_uid),
        phase,
        expected_iteration,
        expected_round,
    )
    if observed_identity != expected_identity:
        raise ValueError("scheduler-complete AIMAll producer identity mismatch")

    status = str(current.get("status") or "")
    reason = str(current.get("reason") or "")
    nested = current.get("postprocess_source")
    if nested is not None:
        if status not in {"PRE_SUBMIT", "FAILED", "SUPERSEDED"}:
            raise ValueError(
                "AIMAll postprocess source wrapper has an unsupported status"
            )
        if status == "SUPERSEDED" and reason != "reconcile_apply_retry":
            raise ValueError(
                "superseded AIMAll postprocess wrapper has an unsupported reason"
            )
        if current.get("job_id") is not None:
            raise ValueError("AIMAll postprocess wrapper unexpectedly owns a JobID")
        source = _validated_postprocess_source(nested)
    else:
        if status != "FAILED" and not (
            status == "SUPERSEDED" and reason == "reconcile_apply_retry"
        ):
            raise ValueError(
                "scheduler-complete AIMAll producer must be FAILED or "
                "SUPERSEDED by reconcile_apply_retry"
            )
        scheduler_kind = str(
            current.get("scheduler_identity_kind") or ""
        ).strip().lower()
        job_id = _validate_job_identity(current.get("job_id"), scheduler_kind)
        if current.get("submission_kind") != "array":
            raise ValueError("scheduler-complete AIMAll producer is not an array")
        if current.get("expected_tasks") != expected_total:
            raise ValueError(
                "scheduler-complete AIMAll producer expected-task count mismatch"
            )
        metadata = current.get("submission_metadata")
        if (
            not isinstance(metadata, Mapping)
            or str(metadata.get("logical_task_set_sha256") or "")
            != expected_task_digest
        ):
            raise ValueError(
                "scheduler-complete AIMAll producer task-set digest mismatch"
            )
        lifecycle = current.get("queue_lifecycle")
        if not isinstance(lifecycle, Mapping):
            raise ValueError(
                "scheduler-complete AIMAll producer lacks queue lifecycle evidence"
            )
        terminal_identity = (
            str(lifecycle.get("terminal_status") or ""),
            lifecycle.get("n_expected"),
            lifecycle.get("n_observed"),
            lifecycle.get("n_missing"),
        )
        if terminal_identity != (
            "COMPLETED",
            expected_total,
            expected_total,
            0,
        ):
            raise ValueError(
                "AIMAll scheduler lifecycle does not prove complete task ownership"
            )
        source = {
            "campaign_uid": str(campaign_uid),
            "phase": phase,
            "iteration": expected_iteration,
            "attempt_id": str(current.get("attempt_id") or ""),
            "submission_identity": str(
                current.get("submission_identity") or ""
            ),
            "job_id": str(job_id),
            "environment_generation": current.get("environment_generation"),
            "environment_generation_digest_sha256": current.get(
                "environment_generation_digest_sha256"
            ),
            "logical_total": expected_total,
            "logical_task_set_sha256": expected_task_digest,
            "decision_contract": current.get("decision_contract"),
        }
        source["source_sha256"] = _canonical_postprocess_source_sha256(source)
        source = _validated_postprocess_source(source)

    source_identity = (
        str(source["campaign_uid"]),
        str(source["phase"]),
        int(source["iteration"]),
        int(source["logical_total"]),
        str(source["logical_task_set_sha256"]),
    )
    if source_identity != (
        str(campaign_uid),
        phase,
        expected_iteration,
        expected_total,
        expected_task_digest,
    ):
        raise ValueError(
            "AIMAll postprocess source does not match the filtered task set"
        )
    if current.get("expected_tasks") != expected_total:
        raise ValueError("AIMAll postprocess wrapper expected-task count mismatch")
    if current.get("decision_contract") != source["decision_contract"]:
        raise ValueError("AIMAll postprocess wrapper decision contract mismatch")
    _validate_postprocess_source_environment(campaign_dir, source)
    return source


def ariadne_producer_environment_binding(
    campaign_dir: Union[str, Path],
    intent: Mapping[str, Any],
    *,
    expected_campaign_uid: str,
    expected_iteration: int,
) -> Dict[str, Any]:
    """Return the environment that produced ARIADNE seed predictions."""
    if (
        str(intent.get("campaign_uid") or "") != str(expected_campaign_uid)
        or str(intent.get("phase") or "") != CampaignPhase.ARIADNE_ARRAY.value
        or int(intent.get("iteration", -1)) != int(expected_iteration)
    ):
        raise ValueError("ARIADNE producer intent identity mismatch")
    nested = intent.get("postprocess_source")
    if nested is not None:
        source = _validated_postprocess_source(nested)
        if (
            str(source["campaign_uid"]) != str(expected_campaign_uid)
            or int(source["iteration"]) != int(expected_iteration)
        ):
            raise ValueError("ARIADNE postprocess source identity mismatch")
        _validate_postprocess_source_environment(campaign_dir, source)
        return {
            "generation": int(source["environment_generation"]),
            "generation_digest_sha256": str(
                source["environment_generation_digest_sha256"]
            ),
        }
    generation = intent.get("environment_generation")
    digest = intent.get("environment_generation_digest_sha256")
    if generation is None and digest is None:
        from .error_calibration_contract import active_environment_binding

        unbound = active_environment_binding(campaign_dir)
        if not bool(unbound.get("bound", False)):
            return {
                "generation": int(unbound["generation"]),
                "generation_digest_sha256": str(
                    unbound["generation_digest_sha256"]
                ),
            }
        raise ValueError(
            "ARIADNE producer intent lacks an environment generation binding"
        )
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("ARIADNE producer environment generation is malformed")
    _validated_sha256(digest, "ARIADNE producer environment generation digest")
    source = {
        "campaign_uid": str(expected_campaign_uid),
        "environment_generation": int(generation),
        "environment_generation_digest_sha256": str(digest),
    }
    _validate_postprocess_source_environment(campaign_dir, source)
    return {
        "generation": int(generation),
        "generation_digest_sha256": str(digest),
    }


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
    scheduler_identity_kind: str = "slurm",
) -> str:
    name = live_job_name(
        campaign_uid,
        phase_name,
        iteration,
        replacement_round=replacement_round,
        attempt_sequence=attempt_sequence,
        attempt_id=attempt_id,
    )
    if str(scheduler_identity_kind) == "sge":
        from ..submit.sge import sge_safe_job_name

        return sge_safe_job_name(name)
    return name


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
    for key in ("script_binding_path", "submitted_script_path"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("submission intent " + key + " must be a non-empty string")
    for key in ("script_binding_sha256", "submitted_script_sha256"):
        value = data.get(key)
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            raise ValueError("submission intent " + key + " is invalid")
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


def inventory_intents(
    campaign_dir: Union[str, Path],
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Read every current intent without silently dropping malformed ownership."""
    root = intent_dir(campaign_dir)
    records = []
    errors = []
    if not root.is_dir():
        return {"records": records, "errors": errors}
    pattern = re.compile(r"^(.+)-([0-9]{6})\.json$")
    for path in sorted(root.glob("*.json")):
        match = pattern.fullmatch(path.name)
        if match is None:
            errors.append({"path": str(path), "error": "invalid intent filename"})
            continue
        phase_name = match.group(1)
        iteration = int(match.group(2))
        try:
            payload = load_intent(
                campaign_dir,
                phase_name,
                iteration,
                expected_campaign_uid=expected_campaign_uid,
            )
            if payload is None:
                raise ValueError("intent disappeared during inventory")
            records.append(payload)
        except Exception as exc:
            errors.append({
                "path": str(path),
                "error": type(exc).__name__ + ": " + str(exc),
            })
    return {"records": records, "errors": errors}


def classify_completed_unsubmitted_intents(
    campaign_dir: Union[str, Path],
    state: Any,
    *,
    intents: Optional[Sequence[Mapping[str, Any]]] = None,
    completion_receipts: Optional[Sequence[Mapping[str, Any]]] = None,
    valid_reference_data_versions: Optional[Sequence[int]] = None,
    valid_model_versions: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Classify jobless intents conclusively covered by completion receipts.

    The helper is read-only.  It intentionally requires both an exact receipt
    identity match and evidence that campaign state and committed versions have
    advanced beyond the scheduler-free source phase.
    """
    campaign = Path(campaign_dir)
    errors: list[Dict[str, str]] = []
    if intents is None:
        intent_inventory = inventory_intents(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        intents = tuple(intent_inventory.get("records", []))
        errors.extend(dict(item) for item in intent_inventory.get("errors", []))
    if completion_receipts is None:
        from .completion_receipts import inventory_completion_receipts

        receipt_inventory = inventory_completion_receipts(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        completion_receipts = tuple(receipt_inventory.get("records", []))
        errors.extend(dict(item) for item in receipt_inventory.get("errors", []))

    if valid_reference_data_versions is None:
        from ..versioning.reference_data import ReferenceDataVersioning

        valid_reference_data_versions = ReferenceDataVersioning(
            campaign / "QM_REFERENCE_DATA"
        ).list_committed_versions()
    if valid_model_versions is None:
        from ..versioning.trained_models import TrainedModelVersioning

        valid_model_versions = TrainedModelVersioning(
            campaign / "TRAINED_MODELS"
        ).list_committed_versions()
    reference_versions = {int(value) for value in valid_reference_data_versions}
    model_versions = {int(value) for value in valid_model_versions}

    candidates = [
        dict(intent)
        for intent in intents
        if str(intent.get("status") or "") == "PRE_SUBMIT"
        and intent.get("job_id") is None
        and str(intent.get("campaign_uid") or "") == str(state.campaign_uid)
    ]
    repairs: list[Dict[str, Any]] = []
    for intent in candidates:
        phase = str(intent.get("phase") or "")
        iteration = int(intent.get("iteration", 0))
        replacement_round = int(intent.get("replacement_round", 0))
        submission_identity = str(intent.get("submission_identity") or "")
        if (
            state.phase.value == phase
            and int(state.iteration) == iteration
            and int(getattr(state, "replacement_round", 0)) == replacement_round
        ):
            continue

        matches: list[Mapping[str, Any]] = []
        for record in completion_receipts:
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if (
                str(payload.get("campaign_uid") or "") != str(state.campaign_uid)
                or str(payload.get("phase") or "") != phase
                or int(payload.get("iteration", -1)) != iteration
                or int(payload.get("replacement_round", -1)) != replacement_round
                or str(payload.get("submission_identity") or "")
                != submission_identity
                or payload.get("job_id") is not None
            ):
                continue
            intent_tasks = intent.get("expected_tasks")
            receipt_tasks = payload.get("expected_tasks")
            if (
                intent_tasks is not None
                and receipt_tasks is not None
                and int(intent_tasks) != int(receipt_tasks)
            ):
                continue
            matches.append(record)
        if len(matches) > 1:
            errors.append(
                {
                    "path": phase + "@" + str(iteration),
                    "error": (
                        "multiple completion receipts match jobless submission "
                        "intent " + submission_identity
                    ),
                }
            )
            continue
        if not matches:
            continue

        record = matches[0]
        payload = dict(record["payload"])
        try:
            after = CampaignState.from_dict(dict(payload["state_after"]))
        except Exception as exc:
            errors.append(
                {
                    "path": str(record.get("path") or ""),
                    "error": "completion receipt state_after is invalid: " + str(exc),
                }
            )
            continue
        if int(state.iteration) < int(after.iteration):
            continue
        if any(
            int(getattr(state, field_name)) < int(getattr(after, field_name))
            for field_name in (
                "reference_data_version",
                "validation_set_version",
                "models_version",
            )
        ):
            continue
        if (
            int(after.reference_data_version) >= 0
            and int(after.reference_data_version) not in reference_versions
        ):
            continue
        if int(after.models_version) >= 0 and int(after.models_version) not in model_versions:
            continue
        reference = record.get("reference")
        if not isinstance(reference, Mapping):
            errors.append(
                {
                    "path": str(record.get("path") or ""),
                    "error": "completion receipt reference is missing",
                }
            )
            continue
        repairs.append(
            {
                "phase": phase,
                "iteration": iteration,
                "replacement_round": replacement_round,
                "submission_identity": submission_identity,
                "expected_tasks": intent.get("expected_tasks"),
                "reason": "phase_completed_without_scheduler_submission",
                "target_status": "SUPERSEDED",
                "completion_receipt": dict(reference),
                "state_after": after.to_dict(),
            }
        )
    repairs.sort(
        key=lambda item: (
            int(item["iteration"]),
            str(item["phase"]),
            int(item["replacement_round"]),
            str(item["submission_identity"]),
        )
    )
    return {"repairs": repairs, "errors": errors}


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
    postprocess_source: Optional[Dict[str, Any]] = None,
    scheduler_identity_kind: str = "slurm",
    environment_generation: Optional[int] = None,
    environment_generation_digest_sha256: Optional[str] = None,
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
            scheduler_identity_kind=str(scheduler_identity_kind),
        ),
        "status": "PRE_SUBMIT",
        "submission_kind": submission_kind_for_phase(phase_name),
        "scheduler_identity_kind": str(scheduler_identity_kind),
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
    if postprocess_source is not None:
        payload["postprocess_source"] = _validated_postprocess_source(
            postprocess_source
        )
    if (environment_generation is None) != (
        environment_generation_digest_sha256 is None
    ):
        raise ValueError(
            "submission intent environment generation and digest must be paired"
        )
    if environment_generation is not None:
        payload["environment_generation"] = _exact_int(
            environment_generation,
            "submission intent environment_generation",
        )
        digest = str(environment_generation_digest_sha256)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "submission intent environment generation digest is invalid"
            )
        payload["environment_generation_digest_sha256"] = digest
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
        data["job_id"] = _validate_job_identity(
            job_id,
            str(data.get("scheduler_identity_kind")),
        )
        seen = data.get("job_ids_seen", [])
        if not isinstance(seen, list):
            seen = []
        if str(data["job_id"]) not in [str(x) for x in seen]:
            seen.append(str(data["job_id"]))
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
        data["job_ids_seen"] = [
            _validate_job_identity(x, str(data.get("scheduler_identity_kind")))
            for x in list(job_ids_seen)
        ]
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


def bind_submission_script(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    *,
    script_path: str,
    script_sha256: str,
    binding_path: str,
    binding_sha256: str,
) -> Dict[str, Any]:
    """Bind final script bytes before the corresponding sbatch call."""
    intent = load_active_intent(campaign_dir, phase_name, int(iteration))
    if intent is None or str(intent.get("status")) != "PRE_SUBMIT":
        raise ValueError("script binding requires an active PRE_SUBMIT intent")
    updates = {
        "submitted_script_path": str(script_path),
        "submitted_script_sha256": str(script_sha256),
        "script_binding_path": str(binding_path),
        "script_binding_sha256": str(binding_sha256),
    }
    for key in ("submitted_script_sha256", "script_binding_sha256"):
        digest = updates[key]
        if (
            len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise ValueError("submission intent " + key + " is invalid")
    for key, value in updates.items():
        previous = intent.get(key)
        if previous is not None and previous != value:
            raise ValueError("submission intent " + key + " is already bound differently")
        intent[key] = value
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


def mark_superseded(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    reason: str,
    *,
    completion_receipt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return update_intent_status(
        campaign_dir, phase_name=phase_name, iteration=iteration,
        status="SUPERSEDED", reason=reason,
        completion_receipt=completion_receipt,
    )


def prepare_reconcile_terminal_transition(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    *,
    target_status: str,
    reason: str,
    completion_receipt: Optional[Dict[str, Any]] = None,
    expected_campaign_uid: Optional[str] = None,
    updated_at_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the exact terminal intent payload used by reconcile commit."""
    path = intent_path(campaign_dir, phase_name, iteration)
    data = load_intent(
        campaign_dir,
        phase_name,
        iteration,
        expected_campaign_uid=expected_campaign_uid,
    )
    if not isinstance(data, dict):
        raise FileNotFoundError("submission intent does not exist")
    target = str(target_status)
    if target not in {"FAILED", "SUPERSEDED"}:
        raise ValueError("unsupported reconcile intent target: " + target)
    previous = str(data.get("status") or "")
    if target not in _STATUS_TRANSITIONS.get(previous, set()):
        raise ValueError(
            "illegal submission intent transition: " + previous + " -> " + target
        )
    prepared = dict(data)
    prepared["status"] = target
    prepared["reason"] = str(reason)
    if completion_receipt is not None:
        prepared["completion_receipt"] = dict(completion_receipt)
    now = str(updated_at_iso or _now_iso())
    _timestamp(now, "reconcile intent update time")
    prepared["updated_iso"] = now
    prepared["updated_at_iso"] = now
    return _validate_intent_payload(
        prepared,
        path=path,
        phase_name=str(phase_name),
        iteration=int(iteration),
        expected_campaign_uid=expected_campaign_uid,
    )


def publish_prepared_reconcile_transition(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    payload: Mapping[str, Any],
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Publish one exact terminal intent payload prepared by reconcile."""
    path = intent_path(campaign_dir, phase_name, iteration)
    prepared = _validate_intent_payload(
        dict(payload),
        path=path,
        phase_name=str(phase_name),
        iteration=int(iteration),
        expected_campaign_uid=expected_campaign_uid,
    )
    atomic_write_json(path, prepared)
    loaded = load_intent(
        campaign_dir,
        phase_name,
        iteration,
        expected_campaign_uid=expected_campaign_uid,
    )
    if not isinstance(loaded, dict) or loaded != prepared:
        raise ValueError("published reconcile intent payload did not round-trip")
    return loaded
