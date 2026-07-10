"""Operator recommendations for ``ichor-al-daemon status``.

The daemon state is a compact machine contract, but the status command is an
operator interface. Keep the decision table here so recommendations stay
specific and testable instead of collapsing into one generic "run reconcile"
message.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .state import CampaignPhase


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
    return "ichor-al-daemon " + command + " --campaign-dir " + str(campaign_dir)


def _start_cmd(campaign_dir: Path) -> str:
    return _cmd(campaign_dir, "start") + " --live"


def _reconcile_cmd(campaign_dir: Path, *, apply: bool = False) -> str:
    command = _cmd(campaign_dir, "reconcile")
    if apply:
        command += " --apply"
    return command


def _journal_cmd(campaign_dir: Path) -> str:
    return _cmd(campaign_dir, "journal")


def _lease_is_fresh(heartbeat: Any, *, stale_seconds: int = 900) -> bool:
    if not isinstance(heartbeat, dict):
        return False
    try:
        age = time.time() - float(heartbeat.get("time"))
    except Exception:
        return False
    return age <= float(stale_seconds)


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
    for label in ("training", "models"):
        item = status.get(label)
        if isinstance(item, dict):
            errors.extend(str(error) for error in (item.get("errors") or []))
    return errors


def _artifact_problem(payload: Dict[str, Any]) -> bool:
    phase = _phase(payload)
    if phase not in {
        CampaignPhase.SEED_SELECT.value,
        CampaignPhase.ARIADNE_ARRAY.value,
        CampaignPhase.PHASE_B_POLUS.value,
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
    for label in ("training", "models"):
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
    if _lease_is_fresh(payload.get("lease_heartbeat")):
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
    upper = reason.upper()
    if "SEED_POOL_EXHAUSTED" in upper:
        return StatusRecommendation(
            code="halted_seed_pool_exhausted",
            severity="blocked",
            primary="start a new campaign with a larger pool or smaller bootstrap/seed budget",
            why="seed selection exhausted eligible trajectory frames: " + _short_error(reason),
            command=_journal_cmd(campaign) + " --event-type halt --last-n 5",
            details=[
                "changing bootstrap point-allocation sizes after bootstrap has committed cannot remove already-labelled provenance",
                "for plumbing-only debugging, anti_overlap.skip_training_seeds=false can allow reseeding",
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
            "training_version_invalid",
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
            primary="run reconcile; STOP_CHECK needs the latest coherent committed training/model pair",
            why=error or "state/artifact contract failed at STOP_CHECK",
            command=_reconcile_cmd(campaign),
        )
    if "skew" in lower:
        return StatusRecommendation(
            code="version_skew",
            severity="required",
            primary="run reconcile; training and model versions in state are not mutually coherent",
            why=error,
            command=_reconcile_cmd(campaign),
        )
    if "training" in lower:
        return StatusRecommendation(
            code="training_missing",
            severity="required",
            primary="run reconcile; the training version referenced by state is missing or invalid",
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
        primary="run reconcile; committed training/model artefacts are inconsistent with status checks",
        why=why,
        command=_reconcile_cmd(campaign),
        details=errors[1:5],
    )


_PHASE_ACTIONS: Dict[str, tuple[str, str]] = {
    CampaignPhase.INIT.value: (
        "start the daemon to begin the campaign",
        "campaign is initialised and no daemon work is active",
    ),
    CampaignPhase.PHASE_A_POLUS.value: (
        "start the daemon to submit or postprocess initial POLUS diversity sampling",
        "the next phase is PHASE_A_POLUS",
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
    CampaignPhase.PHASE_B_POLUS.value: (
        "start the daemon to run Phase B POLUS filtering/selection",
        "the next phase is PHASE_B_POLUS",
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
        "start the daemon to append accepted AIMAll pointdirs to the training set",
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
                        command="ichor-al-daemon init --campaign-dir " + str(campaign),
                    )
                ]
            return [
                StatusRecommendation(
                    code="campaign_missing",
                    severity="required",
                    primary="initialise the campaign before starting the daemon",
                    why="campaign.yaml and state.json are both missing",
                    command="ichor-al-daemon init --campaign-dir " + str(campaign),
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

    runtime = _runtime_blockers(campaign, payload)
    if runtime:
        return runtime

    stale_pid = _stale_pid_recommendation(campaign, payload)

    if payload.get("shutdown_requested"):
        return [
            StatusRecommendation(
                code="shutdown_requested",
                severity="required",
                primary="clear the stop request with reconcile before restarting",
                why="state.json has shutdown_requested=true",
                command=_reconcile_cmd(campaign, apply=True),
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
