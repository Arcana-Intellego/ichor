"""Operator recommendations for ``ichor-al-daemon status``.

The daemon state is a compact machine contract, but the status command is an
operator interface. Keep the decision table here so recommendations stay
specific and testable instead of collapsing into one generic "run reconcile"
message.
"""
from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .state import CampaignPhase
from .lease import evaluate_lease_liveness


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


def _start_cmd(campaign_dir: Path) -> str:
    return _cmd(campaign_dir, "start") + " --mode live"


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
    pending = payload.get("pending_jobs")
    if not isinstance(pending, dict):
        return {}
    return {
        str(phase): str(job_id)
        for phase, job_id in pending.items()
        if job_id not in (None, "", False)
    }


def _active_submission_intents(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    intents = payload.get("active_submission_intents")
    if not isinstance(intents, list):
        return []
    return [item for item in intents if isinstance(item, dict)]


def _short_error(value: Any, *, limit: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


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
        CampaignPhase.APPEND.value,
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
            command=_start_cmd(campaign),
            details=["pid=" + str(payload.get("background_pid"))],
        )
    ]


def _runtime_blockers(campaign: Path, payload: Dict[str, Any]) -> List[StatusRecommendation]:
    recommendations: List[StatusRecommendation] = []
    if payload.get("lock_held") is None:
        recommendations.append(
            StatusRecommendation(
                code="runtime_probe_failed",
                severity="blocked",
                primary="inspect the daemon lock before starting or reconciling",
                why="status could not determine whether the foreground lock is held",
                command=_cmd(campaign, "status") + " --verbose",
                details=[_short_error(payload.get("lock_probe_error"))],
            )
        )
    if payload.get("lease_probe_error"):
        recommendations.append(
            StatusRecommendation(
                code="runtime_probe_failed",
                severity="blocked",
                primary="inspect the daemon lease before starting or reconciling",
                why="status could not read the daemon lease heartbeat",
                command=_cmd(campaign, "status") + " --verbose",
                details=[_short_error(payload.get("lease_probe_error"))],
            )
        )
    if payload.get("lock_held") is True:
        recommendations.append(
            StatusRecommendation(
                code="daemon_running_lock",
                severity="watch",
                primary="a daemon appears to own the foreground lock; monitor it instead of starting another",
                why="the daemon lock is currently held",
                command=_journal_cmd(campaign),
            )
        )
    if _lease_is_fresh(payload):
        recommendations.append(
            StatusRecommendation(
                code="daemon_running_lease",
                severity="watch",
                primary="a daemon lease heartbeat is fresh; monitor the running daemon",
                why="the lease heartbeat is recent",
                command=_journal_cmd(campaign),
            )
        )
    if payload.get("background_pid_alive") is True:
        recommendations.append(
            StatusRecommendation(
                code="daemon_running_background",
                severity="watch",
                primary="the background daemon is running; monitor journal/status or stop it intentionally",
                why="the recorded background PID is alive",
                command=_journal_cmd(campaign),
                details=["pid=" + str(payload.get("background_pid"))],
            )
        )
    return recommendations


def _job_recommendations(campaign: Path, payload: Dict[str, Any]) -> List[StatusRecommendation]:
    recommendations: List[StatusRecommendation] = []
    pending = _active_pending_jobs(payload)
    if pending:
        recommendations.append(
            StatusRecommendation(
                code="pending_state_job",
                severity="watch",
                primary="a Slurm job is recorded in state; wait for daemon postprocess or stop it intentionally",
                why="state.json still contains active pending job IDs",
                command=_journal_cmd(campaign),
                details=[phase + "=" + job_id for phase, job_id in sorted(pending.items())],
            )
        )
    intents = _active_submission_intents(payload)
    if intents:
        details = []
        for intent in intents[:5]:
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
                primary="a submission intent is still active; let the daemon adopt/postprocess it or stop it intentionally",
                why="submission intent files show active scheduler work",
                command=_journal_cmd(campaign),
                details=details,
            )
        )
    return recommendations


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
                "operator input or start a new campaign because mandatory custom "
                "geometries cannot be replaced"
            ),
            why=_short_error(reason),
            command=_journal_cmd(campaign) + " --event-type halt --last-n 5",
        )
    if reason_code == "replacement_reserve_exhausted":
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
                "use reconcile --retrain-ferebus --apply to discard no output silently",
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
        return StatusRecommendation(
            code="halted_backend_submission_failed",
            severity="required",
            primary="fix the configured backend/profile problem, then apply reconcile before restarting",
            why="backend submission failed before the phase could run: " + _short_error(reason),
            command=_reconcile_cmd(campaign, apply=True),
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
            primary="inspect Slurm output/error logs before reconciling or restarting",
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
        "start the daemon to begin the campaign",
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
    CampaignPhase.APPEND.value: (
        "start the daemon to append accepted AIMAll pointdirs to the QM reference data",
        "the next phase is APPEND",
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
    action = _PHASE_ACTIONS.get(phase)
    if action is not None:
        primary, why = action
        return StatusRecommendation(
            code="phase_" + phase.lower() + "_ready",
            severity="info",
            primary=primary,
            why=why,
            command=_start_cmd(campaign),
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
    """Return ordered operator recommendations for a status payload."""
    del journal_path  # reserved for future journal-dependent detail expansion
    campaign = Path(campaign_dir)
    config_status = payload.get("campaign_config_status")
    if isinstance(config_status, dict) and config_status.get("ok") is False:
        return [
            StatusRecommendation(
                code="campaign_config_invalid",
                severity="required",
                primary="repair campaign.yaml before assessing pool feasibility or restarting",
                why=_short_error(config_status.get("error")),
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
                primary="inspect and reconcile the malformed operator stop control before continuing",
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
    ):
        return [
            StatusRecommendation(
                code="pool_feasibility_failed",
                severity="blocked",
                primary="fix bootstrap/seed sizing or import a larger trajectory pool before starting",
                why=str(feasibility.get("error") or feasibility.get("expression") or "trajectory pool is infeasible"),
                details=[
                    "pool_n_frames=" + str(feasibility.get("pool_n_frames")),
                    "required_pool_frames=" + str(feasibility.get("required_pool_frames")),
                ],
            )
        ]

    status_error = str(payload.get("status_error") or "")
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
                command=_reconcile_cmd(campaign, apply=True),
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

    stop_request = payload.get("stop_request")
    if isinstance(stop_request, dict) and not payload.get("shutdown_requested"):
        if str(stop_request.get("status")) == "cancelling":
            return [
                StatusRecommendation(
                    code="operator_stop_cancellation_incomplete",
                    severity="required",
                    primary="rerun the immediate stop command to finish recorded Slurm cancellation",
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
        daemon_active = (
            payload.get("lock_held") is True
            or payload.get("background_pid_alive") is True
            or _lease_is_fresh(payload)
        )
        target = "the next daemon tick"
        if stop_request.get("target_phase") is not None:
            target = (
                str(stop_request.get("target_phase"))
                + "@"
                + str(stop_request.get("target_iteration"))
            )
        elif stop_request.get("target_iteration") is not None:
            target = "iteration " + str(stop_request.get("target_iteration"))
        return [
            StatusRecommendation(
                code="operator_stop_draining",
                severity="watch" if daemon_active else "required",
                primary=(
                    "resume the campaign to finalise the completed operator stop"
                    if request_completed
                    else (
                        "wait for the daemon to reach the requested stop boundary"
                        if daemon_active
                        else "resume the daemon so it can honour the pending stop boundary"
                    )
                ),
                why=(
                    str(stop_request.get("mode"))
                    + " stop request is "
                    + str(stop_request.get("status"))
                    + "; target is "
                    + target
                ),
                command=(
                    _cmd(campaign, "resume")
                    if request_completed or not daemon_active
                    else _journal_cmd(campaign)
                ),
                details=[
                    "request_id=" + str(stop_request.get("request_id")),
                    "use resume --cancel-stop-request only to withdraw this request",
                ],
            )
        ]

    runtime = _runtime_blockers(campaign, payload)
    if runtime:
        return runtime

    stale_pid = _stale_pid_recommendation(campaign, payload)

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
                    "resume the campaign to clear the explicit operator stop"
                    if stopped
                    else "run reconcile before clearing an unclassified stop request"
                ),
                why=(
                    str(context.get("message"))
                    if stopped
                    else "state.json has shutdown_requested=true without a stopped lifecycle"
                ),
                command=(
                    _cmd(campaign, "resume")
                    if stopped
                    else _reconcile_cmd(campaign, apply=True)
                ),
            )
        ] + stale_pid

    jobs = _job_recommendations(campaign, payload)
    if jobs:
        return jobs + stale_pid

    if _phase(payload) == CampaignPhase.HALTED.value:
        return [_halt_recommendation(campaign, payload)] + stale_pid

    if _state_contract_problem(payload):
        return [_contract_recommendation(campaign, payload)] + stale_pid

    if _artifact_problem(payload):
        return [_artifact_recommendation(campaign, payload)] + stale_pid

    return [_phase_recommendation(campaign, payload)] + stale_pid
