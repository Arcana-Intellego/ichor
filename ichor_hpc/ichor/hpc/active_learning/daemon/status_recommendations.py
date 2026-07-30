"""User recommendations for ``ichor-al-daemon status``.

The daemon state is a compact machine contract, but the status command is a
user interface. Keep the decision table here so recommendations stay
specific and testable instead of collapsing into one generic "run reconcile"
message.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .state import CampaignPhase
from .lease import evaluate_lease_liveness
from .presentation_assessment import assess_campaign_presentation


@dataclass
class StatusRecommendation:
    code: str
    severity: str
    primary: str
    why: str
    command: Optional[str] = None
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not self.command:
            payload.pop("command", None)
        if not self.details:
            payload.pop("details", None)
        return payload


def recommendation_dicts(
    recommendations: Iterable[StatusRecommendation],
) -> List[Dict[str, Any]]:
    return [recommendation.to_dict() for recommendation in recommendations]


def _cmd(campaign_dir: Path, command: str) -> str:
    return (
        "ichor-al-daemon "
        + command
        + " --campaign-dir "
        + shlex.quote(str(campaign_dir))
    )


def _resume_cmd(campaign_dir: Path) -> str:
    return _cmd(campaign_dir, "resume")


def _execution_mode(payload: Dict[str, Any]) -> Optional[str]:
    value = payload.get("_presentation_execution_mode")
    return str(value) if value in {"live", "dry_run"} else None


def _daemon_is_active(payload: Dict[str, Any]) -> bool:
    return bool(
        payload.get("lock_held") is True
        or payload.get("background_pid_alive") is True
        or _lease_is_fresh(payload)
    )


def _phase_command(campaign_dir: Path, payload: Dict[str, Any]) -> str:
    """Return a command valid for the campaign's recorded execution identity."""
    phase = _phase(payload)
    mode = _execution_mode(payload)
    if mode is not None or (
        phase != CampaignPhase.INIT.value
        and payload.get("_presentation_execution_identity_checked") is not True
    ):
        return _resume_cmd(campaign_dir)
    if phase == CampaignPhase.INIT.value:
        return _cmd(campaign_dir, "start")
    return _reconcile_cmd(campaign_dir)


def _reconcile_cmd(campaign_dir: Path, *, apply: bool = False) -> str:
    command = _cmd(campaign_dir, "reconcile")
    if apply:
        command += " --apply"
    return command


def _journal_cmd(campaign_dir: Path) -> str:
    return _cmd(campaign_dir, "journal")


def _lease_is_fresh(payload: Dict[str, Any]) -> bool:
    return evaluate_lease_liveness(
        payload.get("lease_heartbeat"),
        stale_seconds=int(payload.get("lease_stale_seconds", 900)),
        clock_skew_tolerance_seconds=int(
            payload.get("clock_skew_tolerance_seconds", 60)
        ),
    ).fresh


def _active_pending_jobs(payload: Dict[str, Any]) -> Dict[str, str]:
    return dict(assess_campaign_presentation(payload).scheduler.pending_jobs)


def _active_submission_intents(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    scheduler = assess_campaign_presentation(payload).scheduler
    return [
        dict(item)
        for item in scheduler.scheduler_intents + scheduler.local_intents
    ]


def _scheduler_name(payload: Dict[str, Any]) -> str:
    recorded = {
        str(intent.get("scheduler_identity_kind") or "").strip().lower()
        for intent in _active_submission_intents(payload)
        if intent.get("job_id") and intent.get("scheduler_identity_kind")
    }
    kind = next(iter(recorded)) if len(recorded) == 1 else str(
        payload.get("_presentation_scheduler_kind") or ""
    ).strip().lower()
    if not kind:
        try:
            from .cluster_profile import require_cluster_profile

            profile = require_cluster_profile()
            kind = str(
                profile.config[profile.machine]["hpc"]["scheduler"]
            ).strip().lower()
        except Exception:
            kind = "slurm"
    return "Sun Grid Engine" if kind == "sge" else "Slurm"


def _short_error(value: Any, *, limit: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _plain_error(value: Any, *, limit: int = 180) -> str:
    text = re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):\s*",
        "",
        str(value or "").strip(),
    )
    return _short_error(" ".join(text.split()), limit=limit)


def _contract_error(payload: Dict[str, Any]) -> str:
    contract = payload.get("state_artifact_contract_status")
    if isinstance(contract, dict):
        return _short_error(contract.get("error"))
    return ""


def _artifact_errors(payload: Dict[str, Any]) -> List[str]:
    status = payload.get("artifact_manifest_status")
    if not isinstance(status, dict):
        return []
    errors: List[str] = []
    if status.get("error"):
        errors.append(str(status.get("error")))
    for label in ("reference_data", "models"):
        item = status.get(label)
        if isinstance(item, dict):
            errors.extend(str(error) for error in (item.get("errors") or []))
    return errors


def _artifact_problem(payload: Dict[str, Any]) -> bool:
    phase = _phase(payload)
    if phase not in {
        CampaignPhase.SEED_SELECT.value,
        CampaignPhase.ARIADNE_ARRAY.value,
        CampaignPhase.PHASE_B_DIVERSITY.value,
        CampaignPhase.SPLIT.value,
        CampaignPhase.GAUSSIAN.value,
        CampaignPhase.AIMALL.value,
        CampaignPhase.ALLOCATION_CHECK.value,
        CampaignPhase.REPLACEMENT_GAUSSIAN.value,
        CampaignPhase.REPLACEMENT_AIMALL.value,
        CampaignPhase.REFERENCE_COMMIT.value,
        CampaignPhase.FEREBUS.value,
        CampaignPhase.STOP_CHECK.value,
        CampaignPhase.DONE.value,
    }:
        return False
    status = payload.get("artifact_manifest_status")
    if not isinstance(status, dict):
        return False
    if status.get("error"):
        return True
    for label in ("reference_data", "models"):
        item = status.get(label)
        if isinstance(item, dict) and item.get("ok") is False:
            return True
    return False


def _state_contract_problem(payload: Dict[str, Any]) -> bool:
    status = payload.get("state_artifact_contract_status")
    return isinstance(status, dict) and status.get("ok") is False


def _phase(payload: Dict[str, Any]) -> str:
    return str(payload.get("phase") or "")


def _allocation_check_transition_recommendation(
    campaign_dir: Path,
    payload: Dict[str, Any],
) -> Optional[StatusRecommendation]:
    if _phase(payload) not in {
        CampaignPhase.INITIAL_ALLOCATION_CHECK.value,
        CampaignPhase.ALLOCATION_CHECK.value,
    }:
        return None
    evidence = payload.get("_presentation_allocation_check_transition")
    if not isinstance(evidence, dict):
        return None
    if not bool(evidence.get("safe", False)):
        return StatusRecommendation(
            code="allocation_check_environment_blocked",
            severity="required",
            primary="preview recovery before attempting to resume",
            why=(
                "the allocation-check recovery boundary is not safe: "
                + _short_error(evidence.get("reason"))
            ),
            command=_reconcile_cmd(campaign_dir),
        )
    if not (
        str(payload.get("background_startup_state") or "") == "failed"
        and str(payload.get("background_startup_stage") or "")
        == "environment_transition"
    ):
        return None
    pending = int(evidence.get("pending_tasks") or 0)
    sample_state = str(evidence.get("replacement_sample_state") or "")
    if sample_state in {"missing_rebuildable", "partial_rebuildable"}:
        work = (
            "rebuild the missing replacement sample and continue with "
            + str(pending)
            + " pending replacement task"
            + ("" if pending == 1 else "s")
        )
    else:
        work = "continue the verified allocation check"
    return StatusRecommendation(
        code="allocation_check_environment_retry",
        severity="required",
        primary="resume the daemon; the allocation-check boundary is safe",
        why=(
            "the previous start stopped while changing software environments; "
            "the next start can "
            + work
        ),
        command=_resume_cmd(campaign_dir),
    )


def _scalar_diversity_transition_recommendation(
    campaign_dir: Path,
    payload: Dict[str, Any],
) -> Optional[StatusRecommendation]:
    phase = _phase(payload)
    if phase not in {
        CampaignPhase.PHASE_A_DIVERSITY.value,
        CampaignPhase.PHASE_B_DIVERSITY.value,
    }:
        return None
    evidence = payload.get("_presentation_diversity_transition")
    if not isinstance(evidence, Mapping) or not bool(
        evidence.get("safe", False)
    ):
        return None
    if not (
        str(payload.get("background_startup_state") or "") == "failed"
        and str(payload.get("background_startup_stage") or "")
        == "environment_transition"
    ):
        return None
    historical_job = str(evidence.get("producer_job_id") or "")
    terminal_status = str(evidence.get("scheduler_terminal_status") or "")
    if historical_job:
        detail = (
            "the historical scalar job "
            + historical_job
            + " is conclusively terminal"
            + (" (" + terminal_status.lower() + ")" if terminal_status else "")
        )
    else:
        detail = "the failed attempt never acquired scheduler ownership"
    if phase == CampaignPhase.PHASE_B_DIVERSITY.value:
        detail += (
            "; "
            + str(int(evidence.get("ariadne_accepted_tasks") or 0))
            + " accepted ARIADNE results remain authoritative and "
            + str(int(evidence.get("ariadne_rejected_tasks") or 0))
            + " rejected results remain excluded"
        )
    return StatusRecommendation(
        code="phase_" + phase.lower() + "_ready",
        severity="required",
        primary="resume the daemon; the scalar diversity retry boundary is safe",
        why=detail,
        command=_resume_cmd(campaign_dir),
    )


def _phase_name(value: str) -> str:
    try:
        return CampaignPhase(value).value
    except Exception:
        return value or "UNKNOWN"


def _halt_reason(payload: Dict[str, Any]) -> str:
    context = payload.get("lifecycle_context")
    if isinstance(context, dict) and context.get("message"):
        return str(context.get("message"))
    event = payload.get("latest_halt_event")
    if isinstance(event, dict):
        return str(event.get("reason") or "")
    return str(payload.get("halt_reason") or "")


def _stale_pid_recommendation(campaign: Path, payload: Dict[str, Any]) -> List[StatusRecommendation]:
    if payload.get("background_pid") is None:
        return []
    if payload.get("background_pid_alive") is True:
        return []
    return [
        StatusRecommendation(
            code="stale_background_pid",
            severity="watch",
            primary="background PID metadata is stale; it will be replaced on the next background launch",
            why="the recorded background PID is not alive",
            command=_phase_command(campaign, payload),
            details=["pid=" + str(payload.get("background_pid"))],
        )
    ]


def _ownership_blockers(
    campaign: Path,
    payload: Dict[str, Any],
) -> List[StatusRecommendation]:
    probe_errors: List[str] = []
    if (
        payload.get("lock_probe_error")
        or ("lock_held" in payload and payload.get("lock_held") is None)
    ):
        probe_errors.append(
            _short_error(payload.get("lock_probe_error"))
            or "the daemon lock could not be checked"
        )
    if payload.get("lease_probe_error"):
        probe_errors.append(_short_error(payload.get("lease_probe_error")))
    if probe_errors:
        return [
            StatusRecommendation(
                code="runtime_probe_failed",
                severity="blocked",
                primary="inspect daemon ownership before starting or recovering the campaign",
                why="status could not establish whether another daemon owns this campaign",
                command=_cmd(campaign, "status") + " --verbose",
                details=probe_errors,
            )
        ]
    return []


def _reconcile_transaction_recommendation(
    campaign: Path,
    payload: Dict[str, Any],
) -> Optional[StatusRecommendation]:
    if _daemon_is_active(payload):
        return None
    recovery = payload.get("_presentation_reconcile_transaction_recovery")
    if not isinstance(recovery, dict):
        return None
    state = str(recovery.get("state") or "")
    reason = _short_error(recovery.get("reason"))
    if state == "recoverable" and bool(recovery.get("recoverable", False)):
        return StatusRecommendation(
            code="reconcile_transaction_recoverable",
            severity="required",
            primary="preview recovery from the interrupted reconcile before continuing",
            why=reason or "the previous reconcile did not finish recording its result",
            command=_reconcile_cmd(campaign),
        )
    if state == "blocked":
        return StatusRecommendation(
            code="reconcile_transaction_manual_review",
            severity="blocked",
            primary="review the interrupted reconcile evidence before continuing",
            why=reason or "the interrupted reconcile cannot be recovered automatically",
            command=_reconcile_cmd(campaign),
        )
    return None


def _config_review_recommendation(
    campaign: Path,
    payload: Dict[str, Any],
) -> Optional[StatusRecommendation]:
    assessment = assess_campaign_presentation(payload)
    count = assessment.config_allowed_count + assessment.config_blocked_count
    if (
        assessment.config_state in {"allowed", "blocked"}
        and _daemon_is_active(payload)
    ):
        return StatusRecommendation(
            code="config_change_pending_running",
            severity="watch",
            primary=(
                "the daemon is still using its locked configuration; "
                "review the on-disk change after it stops"
            ),
            why=(
                str(count)
                + " configuration setting"
                + ("" if count == 1 else "s")
                + " differ on disk and do not affect the running process"
            ),
            command=_journal_cmd(campaign) + " --last-n 40",
            details=[
                "after the daemon stops, run " + _reconcile_cmd(campaign)
            ],
        )
    if assessment.config_state == "allowed":
        return StatusRecommendation(
            code="config_change_reconcile_required",
            severity="required",
            primary="preview and apply the pending configuration change before resuming",
            why=(
                str(count)
                + " configuration setting"
                + ("" if count == 1 else "s")
                + " differ from the campaign lock"
            ),
            command=_reconcile_cmd(campaign),
        )
    if assessment.config_state == "blocked":
        return StatusRecommendation(
            code="config_change_blocked",
            severity="blocked",
            primary="review or revert the blocked configuration change before continuing",
            why=(
                str(assessment.config_blocked_count)
                + " configuration setting"
                + ("" if assessment.config_blocked_count == 1 else "s")
                + " cannot be changed at the current campaign position"
            ),
            command=_reconcile_cmd(campaign),
        )
    if assessment.config_state == "invalid":
        return StatusRecommendation(
            code="config_lock_review_failed",
            severity="blocked",
            primary="inspect the campaign configuration and its lock before continuing",
            why=(
                _short_error(assessment.config_error)
                or "the configuration-lock comparison could not be completed"
            ),
            command=_reconcile_cmd(campaign),
        )
    return None


def _scheduler_recovery_recommendation(
    campaign: Path,
    payload: Dict[str, Any],
) -> Optional[StatusRecommendation]:
    assessment = assess_campaign_presentation(payload)
    scheduler = assessment.scheduler
    if scheduler.recovery_state == "invalid":
        recovery = payload.get("_presentation_scheduler_recovery")
        return StatusRecommendation(
            code="scheduler_recovery_invalid",
            severity="blocked",
            primary="preview recovery and inspect the invalid scheduler evidence",
            why=_short_error(
                recovery.get("error")
                if isinstance(recovery, dict)
                else "scheduler recovery evidence is malformed"
            ),
            command=_reconcile_cmd(campaign),
        )
    if (
        not scheduler.has_terminal_recovery
        or scheduler.has_unresolved_scheduler_work
        or scheduler.has_local_work
    ):
        return None
    if scheduler.recovery_state == "validated":
        primary = (
            "resume the campaign to continue with "
            + str(scheduler.reusable_outputs)
            + " validated output"
            + ("" if scheduler.reusable_outputs == 1 else "s")
            + " and "
            + str(scheduler.retry_tasks)
            + " retry task"
            + ("" if scheduler.retry_tasks == 1 else "s")
        )
    else:
        primary = (
            "resume the campaign to validate "
            + str(scheduler.completed_candidates)
            + " scheduler-completed output"
            + ("" if scheduler.completed_candidates == 1 else "s")
            + " and retry unfinished work"
        )
    return StatusRecommendation(
        code="scheduler_terminal_recovery_resume",
        severity="required",
        primary=primary,
        why=(
            "authenticated terminal scheduler evidence proves the old job is "
            "no longer active"
        ),
        command=_resume_cmd(campaign),
    )


def _background_startup_failure_recommendation(
    campaign: Path,
    payload: Dict[str, Any],
) -> Optional[StatusRecommendation]:
    if _daemon_is_active(payload) or str(
        payload.get("background_startup_state") or ""
    ) != "failed":
        return None
    stage = str(payload.get("background_startup_stage") or "startup")
    failure = _plain_error(payload.get("background_startup_failure"))
    if stage in {"environment_transition", "config_lock"}:
        primary = "preview recovery before retrying the failed daemon startup"
        command = _reconcile_cmd(campaign)
    elif stage in {
        "campaign_validation",
        "backend_preflight",
        "environment_preflight",
    }:
        primary = "run preflight and fix the failed startup check"
        command = _cmd(campaign, "preflight")
    else:
        primary = "review the failed startup details before trying again"
        command = _journal_cmd(campaign) + " --last-n 40"
    return StatusRecommendation(
        code="background_startup_failed",
        severity="required",
        primary=primary,
        why=(
            "background startup stopped during "
            + stage.replace("_", " ")
            + (": " + failure if failure else "")
        ),
        command=command,
    )


def _daemon_running_recommendation(
    campaign: Path,
    payload: Dict[str, Any],
) -> Optional[StatusRecommendation]:
    if not _daemon_is_active(payload):
        return None
    evidence: List[str] = []
    if payload.get("lock_held") is True:
        evidence.append("campaign lock held")
    if _lease_is_fresh(payload):
        evidence.append("recent daemon heartbeat")
    if payload.get("background_pid_alive") is True:
        evidence.append("background process is alive")
    return StatusRecommendation(
        code="daemon_running",
        severity="watch",
        primary="the daemon is running; monitor it instead of starting another",
        why=", ".join(evidence),
        command=_journal_cmd(campaign) + " --last-n 40",
    )


def _job_recommendations(campaign: Path, payload: Dict[str, Any]) -> List[StatusRecommendation]:
    recommendations: List[StatusRecommendation] = []
    daemon_active = _daemon_is_active(payload)
    scheduler_name = _scheduler_name(payload)
    pending = _active_pending_jobs(payload)
    if pending:
        recommendations.append(
            StatusRecommendation(
                code="pending_state_job",
                severity="watch",
                primary=(
                    scheduler_name + " work is recorded; monitor the running daemon"
                    if daemon_active
                    else "resume the daemon so it can monitor and postprocess the recorded "
                    + scheduler_name
                    + " work"
                ),
                why="campaign state contains active " + scheduler_name + " job IDs",
                command=(
                    _journal_cmd(campaign) + " --last-n 40"
                    if daemon_active
                    else _resume_cmd(campaign)
                ),
                details=[phase + "=" + job_id for phase, job_id in sorted(pending.items())],
            )
        )
    intents = _active_submission_intents(payload)
    scheduler_intents = [intent for intent in intents if intent.get("job_id")]
    local_intents = [intent for intent in intents if not intent.get("job_id")]
    if scheduler_intents:
        details = []
        for intent in scheduler_intents[:5]:
            details.append(
                str(intent.get("phase", "?"))
                + "@"
                + str(intent.get("iteration", "?"))
                + " "
                + str(intent.get("status", "?"))
                + (
                    " job_id=" + str(intent.get("job_id"))
                    if intent.get("job_id") is not None
                    else ""
                )
            )
        recommendations.append(
            StatusRecommendation(
                code="active_submission_intent",
                severity="watch",
                primary=(
                    "submitted "
                    + scheduler_name
                    + " work is recorded; monitor the running daemon"
                    if daemon_active
                    else "resume the daemon so it can adopt or postprocess the submitted "
                    + scheduler_name
                    + " work"
                ),
                why="saved submission information contains a "
                + scheduler_name
                + " job ID",
                command=(
                    _journal_cmd(campaign) + " --last-n 40"
                    if daemon_active
                    else _resume_cmd(campaign)
                ),
                details=details,
            )
        )
    if local_intents:
        details = [
            str(intent.get("phase", "?"))
            + " iteration "
            + str(intent.get("iteration", "?"))
            + " "
            + str(intent.get("status", "?"))
            for intent in local_intents[:5]
        ]
        recommendations.append(
            StatusRecommendation(
                code="local_submission_intent",
                severity="watch",
                primary=(
                    "local phase work is in progress; monitor the daemon"
                    if daemon_active
                    else "resume the daemon to continue the prepared local phase work"
                ),
                why="saved phase information has no scheduler job attached",
                command=(
                    _journal_cmd(campaign) + " --last-n 40"
                    if daemon_active
                    else _resume_cmd(campaign)
                ),
                details=details,
            )
        )
    return recommendations


def _scheduler_uncertain_resume_recommendation(
    campaign: Path,
    payload: Dict[str, Any],
) -> Optional[StatusRecommendation]:
    if _phase(payload) != CampaignPhase.HALTED.value:
        return None
    context = payload.get("lifecycle_context")
    if not isinstance(context, dict):
        return None
    if context.get("scheduler_uncertain") is not True:
        return None
    if str(context.get("source") or "") != "daemon":
        return None
    pending = _active_pending_jobs(payload)
    if len(pending) != 1:
        return None
    phase, job_id = next(iter(pending.items()))
    if phase != str(context.get("from_phase") or ""):
        return None
    if job_id != str(context.get("job_id") or ""):
        return None
    intents = [
        intent
        for intent in _active_submission_intents(payload)
        if str(intent.get("phase") or "") == phase
        and str(intent.get("job_id") or "") == job_id
        and str(intent.get("status") or "") in {"SUBMITTED", "ADOPTED"}
        and intent.get("iteration") == context.get("iteration")
    ]
    if len(intents) != 1:
        return None
    scheduler_name = _scheduler_name(payload)
    return StatusRecommendation(
        code="halted_scheduler_uncertain",
        severity="required",
        primary=(
            "inspect "
            + scheduler_name
            + " accounting, then resume to re-poll the preserved job; "
            "resume will not resubmit while scheduler ownership remains recorded"
        ),
        why=_short_error(context.get("message")),
        command=_resume_cmd(campaign),
        details=[phase + "=" + job_id],
    )


def _halt_recommendation(campaign: Path, payload: Dict[str, Any]) -> StatusRecommendation:
    reason = _halt_reason(payload)
    context = payload.get("lifecycle_context")
    reason_code = (
        str(context.get("reason_code") or "")
        if isinstance(context, dict)
        else ""
    )
    upper = reason.upper()
    if reason_code == "mandatory_custom_bootstrap_failed":
        return StatusRecommendation(
            code="halted_mandatory_custom_bootstrap_failed",
            severity="blocked",
            primary=(
                "inspect the failed supplied bootstrap calculation; correct the "
                "user input or start a new campaign because mandatory custom "
                "geometries cannot be replaced"
            ),
            why=_short_error(reason),
            command=_journal_cmd(campaign) + " --event-type halt --last-n 5",
        )
    if reason_code == "replacement_reserve_exhausted":
        from .aimall_quality_revalidation import (
            has_aimall_quality_revalidation_candidates,
        )

        if has_aimall_quality_revalidation_candidates(campaign):
            return StatusRecommendation(
                code="halted_replacement_reserve_exhausted",
                severity="required",
                primary=(
                    "preview recovery of AIMAll points rejected by the old INT parser"
                ),
                why=(
                    "the vacant allocation slots have the exact parser-related "
                    "rejection reason and may be revalidated without rerunning a backend"
                ),
                command=_reconcile_cmd(campaign),
                details=[
                    "reconcile will verify the original AIMAll task evidence and locked quality gates",
                    "no scheduler job is submitted during this correction",
                ],
            )
        return StatusRecommendation(
            code="halted_replacement_reserve_exhausted",
            severity="blocked",
            primary=(
                "the exact allocation cannot be completed in place; start a new "
                "campaign with a larger reserve or a smaller required batch"
            ),
            why=_short_error(reason),
            command=_journal_cmd(campaign) + " --event-type halt --last-n 5",
            details=[
                "reconcile cannot invent candidates outside the immutable point-allocation ledger",
                "do not delete or rewrite the exhausted allocation by hand",
            ],
        )
    if reason_code == "ferebus_quality_failed":
        return StatusRecommendation(
            code="halted_ferebus_quality_failed",
            severity="required",
            primary=(
                "inspect FEREBUS quality evidence, then either re-evaluate "
                "justified thresholds or explicitly retrain"
            ),
            why=_short_error(reason),
            command=_reconcile_cmd(campaign),
            details=[
                "threshold-only edits reuse hash-bound model evidence",
                "after a safe preview, reconcile --retrain-ferebus --apply archives the rejected candidate before retraining",
            ],
        )
    if "SEED_POOL_EXHAUSTED" in upper:
        return StatusRecommendation(
            code="halted_seed_pool_exhausted",
            severity="blocked",
            primary="start a new campaign with a larger pool or smaller bootstrap/seed budget",
            why="seed selection exhausted eligible trajectory frames: " + _short_error(reason),
            command=_journal_cmd(campaign) + " --event-type halt --last-n 5",
            details=[
                "changing bootstrap point-allocation sizes after bootstrap has committed cannot remove already-labelled provenance",
                "review committed-frame and recent-seed exclusions before initialising a replacement campaign",
            ],
        )
    if "BACKEND_SUBMISSION_FAILED" in upper:
        if "RESOURCE IMPLEMENTATION ICHOR PACKAGE TREE HAS DRIFTED" in upper:
            return StatusRecommendation(
                code="halted_backend_submission_failed",
                severity="required",
                primary=(
                    "ensure all daemon and "
                    + _scheduler_name(payload)
                    + " work is stopped, reinstall the "
                    "current ICHOR checkout, then preview recovery"
                ),
                why=(
                    "the editable ICHOR installation changed after this work was prepared"
                ),
                command=_reconcile_cmd(campaign),
            )
        if "RESOURCE EVIDENCE" in upper or "HANDOFF" in upper:
            return StatusRecommendation(
                code="halted_backend_submission_failed",
                severity="required",
                primary="preview recovery of the phase input evidence",
                why=(
                    "the phase stopped while validating its published input "
                    "evidence: "
                    + _short_error(reason)
                ),
                command=_reconcile_cmd(campaign),
            )
        return StatusRecommendation(
            code="halted_backend_submission_failed",
            severity="required",
            primary="fix the configured backend or profile problem, then preview recovery",
            why="backend submission failed before the phase could run: " + _short_error(reason),
            command=_reconcile_cmd(campaign),
        )
    if any(token in upper for token in ("NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "REVOKED")):
        return StatusRecommendation(
            code="halted_scheduler_transient",
            severity="required",
            primary="run reconcile; if retry budget remains, restart the daemon afterwards",
            why="the latest halt looks like a transient scheduler failure: " + _short_error(reason),
            command=_reconcile_cmd(campaign),
        )
    if any(token in upper for token in ("OUT_OF_MEMORY", "TIMEOUT", "CANCELLED", "SCRIPT")):
        return StatusRecommendation(
            code="halted_scheduler_hard_failure",
            severity="blocked",
            primary="inspect "
            + _scheduler_name(payload)
            + " output/error logs before reconciling or restarting",
            why="the latest halt looks like a hard job failure: " + _short_error(reason),
            command=_journal_cmd(campaign) + " --event-type halt --last-n 5",
        )
    if any(
        token in reason
        for token in (
            "committed_",
            "manifest",
            "artefact",
            "artifact",
            "contract",
            "reference_data_version_invalid",
            "models_version_invalid",
        )
    ):
        return StatusRecommendation(
            code="halted_contract_failure",
            severity="required",
            primary="run reconcile and inspect any blocking artefacts before restarting",
            why="the latest halt is an artefact/manifest contract failure: " + _short_error(reason),
            command=_reconcile_cmd(campaign),
        )
    if "campaign.yaml changed" in reason or "config lock" in reason:
        return StatusRecommendation(
            code="halted_config_changed",
            severity="required",
            primary="review the config lock diff and apply reconcile only if the changes are safe",
            why="the daemon halted because configuration changed mid-campaign",
            command=_reconcile_cmd(campaign),
        )
    return StatusRecommendation(
        code="halted_unknown",
        severity="required",
        primary="inspect the latest halt reason, then run reconcile before restarting",
        why="campaign phase is HALTED" + (": " + _short_error(reason) if reason else ""),
        command=_journal_cmd(campaign) + " --event-type halt --last-n 5",
    )


def _contract_recommendation(campaign: Path, payload: Dict[str, Any]) -> StatusRecommendation:
    phase = _phase(payload)
    error = _contract_error(payload)
    lower = error.lower()
    if phase == CampaignPhase.INITIAL_FEREBUS.value:
        return StatusRecommendation(
            code="initial_ferebus_bootstrap_contract_problem",
            severity="required",
            primary="run reconcile; initial FEREBUS cannot start until the initial AIMAll handoff is valid",
            why=error or "the INITIAL_FEREBUS bootstrap handoff is not valid",
            command=_reconcile_cmd(campaign),
        )
    if phase == CampaignPhase.STOP_CHECK.value:
        return StatusRecommendation(
            code="stop_check_no_committed_pair",
            severity="required",
            primary="run reconcile; STOP_CHECK needs the latest coherent committed reference-data/model pair",
            why=error or "state/artifact contract failed at STOP_CHECK",
            command=_reconcile_cmd(campaign),
        )
    if "skew" in lower:
        return StatusRecommendation(
            code="version_skew",
            severity="required",
            primary="run reconcile; reference-data and model versions in state are not mutually coherent",
            why=error,
            command=_reconcile_cmd(campaign),
        )
    if "reference-data" in lower or "reference_data" in lower:
        return StatusRecommendation(
            code="reference_data_missing",
            severity="required",
            primary="run reconcile; the reference-data version in state is missing or invalid",
            why=error,
            command=_reconcile_cmd(campaign),
        )
    if "model" in lower:
        return StatusRecommendation(
            code="models_missing",
            severity="required",
            primary="run reconcile; the model version referenced by state is missing or invalid",
            why=error,
            command=_reconcile_cmd(campaign),
        )
    return StatusRecommendation(
        code="state_artifact_contract_invalid",
        severity="required",
        primary="run reconcile; the state/artefact contract is invalid",
        why=error or "state/artifact contract failed",
        command=_reconcile_cmd(campaign),
    )


def _artifact_recommendation(campaign: Path, payload: Dict[str, Any]) -> StatusRecommendation:
    errors = [_short_error(error) for error in _artifact_errors(payload)]
    why = errors[0] if errors else "committed artefact checks failed"
    return StatusRecommendation(
        code="committed_artifact_invalid",
        severity="required",
        primary="run reconcile; committed reference-data/model artefacts are inconsistent with status checks",
        why=why,
        command=_reconcile_cmd(campaign),
        details=errors[1:5],
    )


_PHASE_ACTIONS: Dict[str, tuple[str, str]] = {
    CampaignPhase.INIT.value: (
        "choose live or dry-run mode and start the campaign",
        "campaign is initialised and no daemon work is active",
    ),
    CampaignPhase.PHASE_A_DIVERSITY.value: (
        "start the daemon to submit or postprocess initial ICHOR diversity sampling",
        "the next phase is PHASE_A_DIVERSITY",
    ),
    CampaignPhase.INITIAL_GAUSSIAN.value: (
        "start the daemon to submit or postprocess initial Gaussian jobs",
        "the next phase is INITIAL_GAUSSIAN",
    ),
    CampaignPhase.INITIAL_AIMALL.value: (
        "start the daemon to submit or postprocess initial AIMAll jobs",
        "the next phase is INITIAL_AIMALL",
    ),
    CampaignPhase.INITIAL_ALLOCATION_CHECK.value: (
        "start the daemon to verify bootstrap slot completion or allocate reserve replacements",
        "the next phase is INITIAL_ALLOCATION_CHECK",
    ),
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value: (
        "start the daemon to label the allocated bootstrap replacements with Gaussian",
        "the next phase is INITIAL_REPLACEMENT_GAUSSIAN",
    ),
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value: (
        "start the daemon to postprocess the bootstrap replacements with AIMAll",
        "the next phase is INITIAL_REPLACEMENT_AIMALL",
    ),
    CampaignPhase.INITIAL_FEREBUS.value: (
        "start the daemon to build or postprocess initial FEREBUS models",
        "the next phase is INITIAL_FEREBUS",
    ),
    CampaignPhase.SEED_SELECT.value: (
        "start the daemon to select ARIADNE seeds",
        "the next phase is SEED_SELECT",
    ),
    CampaignPhase.ARIADNE_ARRAY.value: (
        "start the daemon to submit or postprocess ARIADNE jobs",
        "the next phase is ARIADNE_ARRAY",
    ),
    CampaignPhase.PHASE_B_DIVERSITY.value: (
        "start the daemon to run Phase B diversity filtering/selection",
        "the next phase is PHASE_B_DIVERSITY",
    ),
    CampaignPhase.SPLIT.value: (
        "start the daemon to verify the exact pre-QM slot allocation",
        "the next phase is SPLIT",
    ),
    CampaignPhase.GAUSSIAN.value: (
        "start the daemon to submit or postprocess active Gaussian jobs",
        "the next phase is GAUSSIAN",
    ),
    CampaignPhase.AIMALL.value: (
        "start the daemon to submit or postprocess active AIMAll jobs",
        "the next phase is AIMALL",
    ),
    CampaignPhase.ALLOCATION_CHECK.value: (
        "start the daemon to verify active slot completion or allocate reserve replacements",
        "the next phase is ALLOCATION_CHECK",
    ),
    CampaignPhase.REPLACEMENT_GAUSSIAN.value: (
        "start the daemon to label the allocated active replacements with Gaussian",
        "the next phase is REPLACEMENT_GAUSSIAN",
    ),
    CampaignPhase.REPLACEMENT_AIMALL.value: (
        "start the daemon to postprocess the active replacements with AIMAll",
        "the next phase is REPLACEMENT_AIMALL",
    ),
    CampaignPhase.REFERENCE_COMMIT.value: (
        "start the daemon to publish accepted AIMAll pointdirs and cached FEREBUS rows",
        "the next phase is REFERENCE_COMMIT",
    ),
    CampaignPhase.FEREBUS.value: (
        "start the daemon to retrain or postprocess FEREBUS models",
        "the next phase is FEREBUS",
    ),
    CampaignPhase.STOP_CHECK.value: (
        "start the daemon for a tick to decide whether to continue or finish",
        "the campaign is at STOP_CHECK with a valid state/artefact contract",
    ),
}


def _phase_recommendation(campaign: Path, payload: Dict[str, Any]) -> StatusRecommendation:
    phase = _phase(payload)
    if phase == CampaignPhase.DONE.value:
        context = payload.get("lifecycle_context")
        reason_code = (
            str(context.get("reason_code") or "")
            if isinstance(context, dict)
            else ""
        )
        message = (
            str(context.get("message") or "")
            if isinstance(context, dict)
            else ""
        )
        if reason_code == "scientific_convergence":
            return StatusRecommendation(
                code="campaign_completed_scientific_convergence",
                severity="info",
                primary="campaign reached its scientific convergence criterion; no restart is needed",
                why=message or "state phase is DONE after scientific convergence",
                command=_journal_cmd(campaign) + " --event-type campaign_completed",
            )
        if reason_code == "max_iterations_reached":
            return StatusRecommendation(
                code="campaign_completed_max_iterations",
                severity="info",
                primary="campaign reached its configured iteration limit; review quality before extending it",
                why=message or "state phase is DONE at max_iterations",
                command=_journal_cmd(campaign) + " --event-type campaign_completed",
                details=[
                    "increase campaign.max_iterations through reconcile, then use "
                    "resume --reopen-converged only for a deliberate extension"
                ],
            )
        return StatusRecommendation(
            code="campaign_done",
            severity="info",
            primary="campaign is complete; no restart is needed",
            why="state phase is DONE",
            command=_journal_cmd(campaign),
        )
    aimall_recovery = payload.get(
        "_presentation_aimall_postprocess_recovery"
    )
    if isinstance(aimall_recovery, dict):
        total = int(aimall_recovery.get("logical_total") or 0)
        return StatusRecommendation(
            code="phase_" + phase.lower() + "_ready",
            severity="info",
            primary=(
                "resume the daemon to validate "
                + str(total)
                + " existing AIMAll output"
                + ("" if total == 1 else "s")
                + " locally; no AIMAll array will be resubmitted"
            ),
            why=(
                "the original scheduler lifecycle proves every AIMAll task "
                "completed, while local acceptance publication is still pending"
            ),
            command=_resume_cmd(campaign),
        )
    action = _PHASE_ACTIONS.get(phase)
    if action is not None:
        primary, why = action
        mode = _execution_mode(payload)
        if (
            phase != CampaignPhase.INIT.value
            and mode is None
            and payload.get("_presentation_execution_identity_checked") is True
        ):
            return StatusRecommendation(
                code="execution_identity_unavailable",
                severity="required",
                primary="preview recovery before continuing; the campaign execution mode could not be established",
                why="the campaign has progressed but its execution identity is missing or unreadable",
                command=_reconcile_cmd(campaign),
            )
        if mode is not None:
            primary = primary.replace("start the daemon", "resume the daemon")
        return StatusRecommendation(
            code="phase_" + phase.lower() + "_ready",
            severity="info",
            primary=primary,
            why=why,
            command=_phase_command(campaign, payload),
            details=(
                [
                    _cmd(campaign, "start") + " --mode dry_run",
                    "choose live only after preflight passes",
                ]
                if phase == CampaignPhase.INIT.value and mode is None
                else []
            ),
        )
    return StatusRecommendation(
        code="phase_unknown",
        severity="required",
        primary="run reconcile; status does not recognise the current phase",
        why="unknown phase " + repr(_phase_name(phase)),
        command=_reconcile_cmd(campaign),
    )


def build_status_recommendations(
    campaign_dir: Any,
    payload: Dict[str, Any],
    journal_path: Optional[Any] = None,
) -> List[StatusRecommendation]:
    """Return ordered user recommendations for a status payload."""
    del journal_path  # reserved for future journal-dependent detail expansion
    campaign = Path(campaign_dir)
    status_error = str(payload.get("status_error") or "")
    if (
        status_error == "state_missing"
        and payload.get("campaign_yaml_exists") is False
    ):
        return [
            StatusRecommendation(
                code="campaign_missing",
                severity="required",
                primary="initialise the campaign before starting the daemon",
                why="campaign.yaml and state.json are both missing",
                command=_cmd(campaign, "init"),
            )
        ]
    if status_error == "state_schema_invalid":
        return [
            StatusRecommendation(
                code="state_schema_invalid",
                severity="required",
                primary="run reconcile; do not restart until state.json is repaired",
                why=_short_error(payload.get("state_error")),
                command=_reconcile_cmd(campaign),
            )
        ]
    if status_error == "state_unreadable":
        return [
            StatusRecommendation(
                code="state_unreadable",
                severity="blocked",
                primary="check that state.json is readable, then preview recovery",
                why=_short_error(payload.get("state_error")),
                command=_reconcile_cmd(campaign),
                details=[_cmd(campaign, "status") + " --verbose"],
            )
        ]
    config_status = payload.get("campaign_config_status")
    if isinstance(config_status, dict) and config_status.get("ok") is False:
        if _daemon_is_active(payload):
            return [
                StatusRecommendation(
                    code="config_change_pending_running",
                    severity="watch",
                    primary=(
                        "the daemon is still using its locked configuration; "
                        "repair campaign.yaml before the next start"
                    ),
                    why=_short_error(config_status.get("error")),
                    command=_journal_cmd(campaign) + " --last-n 40",
                    details=[
                        "after the daemon stops, run "
                        + _cmd(campaign, "config-check")
                        + " --human"
                    ],
                )
            ]
        return [
            StatusRecommendation(
                code="campaign_config_invalid",
                severity="required",
                primary="repair campaign.yaml before assessing pool feasibility or restarting",
                why=_short_error(config_status.get("error")),
                command=_cmd(campaign, "config-check") + " --human",
            )
        ]
    if payload.get("partial_array_recovery_error"):
        return [
            StatusRecommendation(
                code="partial_array_recovery_invalid",
                severity="required",
                primary="run reconcile and inspect the malformed partial-array ledger",
                why=_short_error(payload.get("partial_array_recovery_error")),
                command=_reconcile_cmd(campaign),
            )
        ]
    if payload.get("journal_error"):
        return [
            StatusRecommendation(
                code="journal_corrupt",
                severity="required",
                primary="run reconcile and inspect the corrupt journal segment",
                why=_short_error(payload.get("journal_error")),
                command=_reconcile_cmd(campaign),
            )
        ]
    if payload.get("submission_intent_errors"):
        return [
            StatusRecommendation(
                code="submission_intent_invalid",
                severity="required",
                primary="run reconcile before cancellation, resubmission, or restart",
                why="one or more scheduler-ownership intents are malformed",
                command=_reconcile_cmd(campaign),
                details=[
                    _short_error(item.get("path")) + ": " + _short_error(item.get("error"))
                    for item in list(payload.get("submission_intent_errors") or [])[:8]
                    if isinstance(item, dict)
                ],
            )
        ]
    if payload.get("stop_control_error"):
        return [
            StatusRecommendation(
                code="stop_control_invalid",
                severity="required",
                primary="inspect and reconcile the malformed user stop control before continuing",
                why=_short_error(payload.get("stop_control_error")),
                command=_reconcile_cmd(campaign),
            )
        ]
    feasibility = payload.get("pool_feasibility")
    feasibility_error = (
        str(feasibility.get("error") or "")
        if isinstance(feasibility, dict)
        else ""
    )
    if (
        isinstance(feasibility, dict)
        and feasibility.get("ok") is False
        and "FileNotFoundError" not in feasibility_error
        and not (
            _daemon_is_active(payload)
            and assess_campaign_presentation(payload).config_state
            in {"allowed", "blocked", "invalid"}
        )
    ):
        return [
            StatusRecommendation(
                code="pool_feasibility_failed",
                severity="blocked",
                primary="fix bootstrap/seed sizing or import a larger trajectory pool before starting",
                why=str(feasibility.get("error") or feasibility.get("expression") or "trajectory pool is infeasible"),
                command=_cmd(campaign, "config-check") + " --human",
                details=[
                    "pool_n_frames=" + str(feasibility.get("pool_n_frames")),
                    "required_pool_frames=" + str(feasibility.get("required_pool_frames")),
                ],
            )
        ]

    if status_error == "state_missing":
        if bool(payload.get("fresh_init_safe", False)):
            if bool(payload.get("campaign_yaml_exists", False)):
                return [
                    StatusRecommendation(
                        code="state_missing_fresh_init",
                        severity="required",
                        primary="bootstrap the fresh campaign before starting the daemon",
                        why=(
                            "campaign.yaml exists, but "
                            ".DATA/ACTIVE_LEARNING/state.json has not been created"
                        ),
                        command=_cmd(campaign, "init"),
                    )
                ]
            return [
                StatusRecommendation(
                    code="campaign_missing",
                    severity="required",
                    primary="initialise the campaign before starting the daemon",
                    why="campaign.yaml and state.json are both missing",
                    command=_cmd(campaign, "init"),
                )
            ]
        return [
            StatusRecommendation(
                code="state_missing",
                severity="required",
                primary="run reconcile; missing state cannot be fresh-initialised safely",
                why=(
                    "no .DATA/ACTIVE_LEARNING/state.json exists, but the campaign "
                    "contains stateful run artefacts"
                ),
                command=_reconcile_cmd(campaign),
            )
        ]
    ownership = _ownership_blockers(campaign, payload)
    if ownership:
        return ownership

    stale_pid = _stale_pid_recommendation(campaign, payload)
    transaction_recovery = _reconcile_transaction_recommendation(
        campaign,
        payload,
    )
    if transaction_recovery is not None:
        return [transaction_recovery] + stale_pid

    scheduler_uncertain = _scheduler_uncertain_resume_recommendation(
        campaign,
        payload,
    )
    if scheduler_uncertain is not None:
        return [scheduler_uncertain] + stale_pid

    if _phase(payload) == CampaignPhase.HALTED.value:
        return [_halt_recommendation(campaign, payload)] + stale_pid

    stop_disposition = payload.get("_presentation_stop_disposition")
    if (
        isinstance(stop_disposition, Mapping)
        and str(stop_disposition.get("kind") or "") == "unreachable"
    ):
        return [
            StatusRecommendation(
                code="stop_control_invalid",
                severity="required",
                primary="preview reconcile before restarting the campaign",
                why=str(
                    stop_disposition.get("reason")
                    or "the recorded stop boundary no longer matches campaign state"
                ),
                command=_reconcile_cmd(campaign),
            )
        ] + stale_pid

    allocation_transition = _allocation_check_transition_recommendation(
        campaign,
        payload,
    )
    if allocation_transition is not None:
        return [allocation_transition] + stale_pid

    if _state_contract_problem(payload):
        return [_contract_recommendation(campaign, payload)] + stale_pid

    if _artifact_problem(payload):
        return [_artifact_recommendation(campaign, payload)] + stale_pid

    config_review = _config_review_recommendation(campaign, payload)
    if config_review is not None:
        return [config_review] + stale_pid

    scheduler_recovery = _scheduler_recovery_recommendation(campaign, payload)
    if scheduler_recovery is not None:
        return [scheduler_recovery] + stale_pid

    diversity_transition = _scalar_diversity_transition_recommendation(
        campaign,
        payload,
    )
    if diversity_transition is not None:
        return [diversity_transition] + stale_pid

    startup_failure = _background_startup_failure_recommendation(
        campaign,
        payload,
    )
    if startup_failure is not None:
        return [startup_failure] + stale_pid

    stop_request = payload.get("stop_request")
    if isinstance(stop_request, dict) and not payload.get("shutdown_requested"):
        from .stop_control import describe_stop_request

        if str(stop_request.get("status")) == "cancelling":
            return [
                StatusRecommendation(
                    code="user_stop_cancellation_incomplete",
                    severity="required",
                    primary="rerun the immediate stop command to finish recorded "
                    + _scheduler_name(payload)
                    + " cancellation",
                    why=(
                        "the daemon is waiting for the CLI cancellation summary "
                        "for request " + str(stop_request.get("request_id"))
                    ),
                    command=(
                        _cmd(campaign, "stop")
                        + " --immediate --cancel-jobs"
                    ),
                )
            ]
        request_completed = str(stop_request.get("status")) == "completed"
        daemon_active = _daemon_is_active(payload)
        stop_description = describe_stop_request(
            stop_request,
            completed=request_completed,
        )
        aimall_recovery = payload.get(
            "_presentation_aimall_postprocess_recovery"
        )
        if isinstance(aimall_recovery, dict):
            total = int(aimall_recovery.get("logical_total") or 0)
            return [
                StatusRecommendation(
                    code="user_stop_draining",
                    severity="required",
                    primary=(
                        "resume the daemon to validate "
                        + str(total)
                        + " existing AIMAll output"
                        + ("" if total == 1 else "s")
                        + " locally; no AIMAll array will be resubmitted"
                    ),
                    why=(
                        stop_description
                        + "; the request remains active and will be honoured "
                        "after this iteration genuinely completes"
                    ),
                    command=_cmd(campaign, "resume"),
                )
            ]
        return [
            StatusRecommendation(
                code="user_stop_draining",
                severity="watch" if daemon_active else "required",
                primary=(
                    "resume the campaign to finalise the completed user stop"
                    if request_completed
                    else (
                        "wait for the daemon to reach the requested stop boundary"
                        if daemon_active
                        else "resume the daemon so it can honour the pending stop boundary"
                    )
                ),
                why=stop_description,
                command=(
                    _cmd(campaign, "resume")
                    if request_completed or not daemon_active
                    else _journal_cmd(campaign) + " --last-n 40"
                ),
                details=[
                    "request_id=" + str(stop_request.get("request_id")),
                    "use resume --cancel-stop-request only to withdraw this request",
                ],
            )
        ]

    if payload.get("shutdown_requested"):
        context = payload.get("lifecycle_context")
        stopped = isinstance(context, dict) and str(
            context.get("disposition") or ""
        ) == "stopped"
        return [
            StatusRecommendation(
                code="shutdown_requested",
                severity="required",
                primary=(
                    "resume the campaign to clear the explicit user stop"
                    if stopped
                    else "preview recovery before clearing an unclassified stop request"
                ),
                why=(
                    str(context.get("message"))
                    if stopped
                    else "state.json has shutdown_requested=true without a stopped lifecycle"
                ),
                command=(
                    _cmd(campaign, "resume")
                    if stopped
                    else _reconcile_cmd(campaign)
                ),
            )
        ] + stale_pid

    daemon_running = _daemon_running_recommendation(campaign, payload)
    if daemon_running is not None:
        return [daemon_running] + stale_pid

    jobs = _job_recommendations(campaign, payload)
    if jobs:
        return jobs + stale_pid

    return [_phase_recommendation(campaign, payload)] + stale_pid
