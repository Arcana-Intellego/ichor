"""Shared read-only evidence assessment for daemon CLI presentation.

The daemon commands inspect overlapping campaign evidence.  This module
normalises that evidence before a command chooses human wording, so stale
state entries cannot override newer authenticated terminal records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Tuple


_TERMINAL_RECOVERY_STATES = frozenset(
    {"awaiting_validation", "legacy_unverified", "validated"}
)
_CONFIG_REVIEW_STATES = frozenset(
    {"unchanged", "allowed", "blocked", "invalid", "unavailable"}
)
_ENVIRONMENT_DISPOSITIONS = frozenset(
    {
        "unbound_first_start",
        "current",
        "rebindable_on_resume",
        "reconcile_required",
        "invalid",
        "ownership_blocked",
    }
)


@dataclass(frozen=True)
class EnvironmentLaunchAssessment:
    """Authenticated environment-generation evidence used for launch advice."""

    disposition: str
    reason: str
    generation: int

    @property
    def launchable(self) -> bool:
        return self.disposition in {
            "unbound_first_start",
            "current",
            "rebindable_on_resume",
        }

    @property
    def advances_on_resume(self) -> bool:
        return self.disposition == "rebindable_on_resume"

    @property
    def requires_reconcile(self) -> bool:
        return self.disposition == "reconcile_required"


@dataclass(frozen=True)
class ReconcileLaunchAssessment:
    """Whether interrupted reconcile evidence permits campaign launch."""

    disposition: str
    reason: str
    state_backup_status: str

    @property
    def launchable(self) -> bool:
        return self.disposition == "none"

    @property
    def requires_reconcile(self) -> bool:
        return self.disposition == "recoverable"


@dataclass(frozen=True)
class OperatorFailureAssessment:
    """Plain-language classification of an internal operator-facing failure."""

    family: str
    summary: str
    action: str


def classify_operator_failure(value: Any) -> OperatorFailureAssessment:
    """Classify implementation and backend failures without exposing internals."""
    text = " ".join(str(value or "").split())
    upper = text.upper()
    if any(
        token in upper
        for token in (
            "ICHOR PACKAGE TREE HAS DRIFTED",
            "ICHOR PACKAGE-TREE HAS DRIFTED",
            "ICHOR SOURCE",
            "SOURCE TREE HAS DRIFTED",
        )
    ):
        return OperatorFailureAssessment(
            family="ichor_source_drift",
            summary="the installed ICHOR code differs from the recorded producer code",
            action=(
                "stop active work, reinstall the current ICHOR checkout, then "
                "preview reconcile"
            ),
        )
    if "ARIADNE" in upper and any(
        token in upper
        for token in ("NATIVE CODE HAS DRIFTED", "NATIVE MODULE", "ABI")
    ):
        return OperatorFailureAssessment(
            family="ariadne_native_drift",
            summary="the installed ARIADNE native module differs from the recorded producer",
            action=(
                "rebuild and reinstall ARIADNE in the configured environment, "
                "then preview reconcile"
            ),
        )
    if "FEREBUS" in upper and any(
        token in upper
        for token in (
            "STAGING RECOVERY",
            "PRODUCER STAGING",
            "PRODUCER TASK MAP",
            "PREPARED FEREBUS TASK MAP IS MISSING",
            "FEREBUS TASK MAP IS MISSING",
            "FEREBUS_TASK_MAP.JSON",
        )
    ):
        return OperatorFailureAssessment(
            family="interrupted_ferebus_staging",
            summary=(
                "FEREBUS runtime preparation was interrupted while recoverable "
                "producer evidence remained"
            ),
            action=(
                "preview reconcile so the recorded FEREBUS producer staging "
                "can be restored before resume"
            ),
        )
    if any(
        token in upper
        for token in (
            "DEPENDENCY ENVIRONMENT HAS DRIFTED",
            "DEPENDENCY_ENVIRONMENT_CHANGED",
            "PYTHON DEPENDENC",
            "PYTHON ENVIRONMENT HAS DRIFTED",
        )
    ):
        return OperatorFailureAssessment(
            family="dependency_environment_drift",
            summary="the configured Python dependency environment differs from the recorded producer",
            action=(
                "restore or reinstall the configured Python environment, then "
                "preview reconcile"
            ),
        )
    if any(
        token in upper
        for token in (
            "IDENTITY KIND IS UNSUPPORTED",
            "IDENTITY KIND IS INVALID",
            "LEGACY IDENTITY",
            "LEGACY EVIDENCE",
            "PRODUCER EQUIVALENCE",
            "IMPLEMENTATION IDENTITY IS INCOMPLETE",
        )
    ):
        return OperatorFailureAssessment(
            family="legacy_identity_insufficient",
            summary="the historical implementation evidence is insufficient to prove output reuse",
            action=(
                "preview reconcile; affected tasks may need retry, but the "
                "campaign is not corrupted"
            ),
        )
    if any(
        token in upper
        for token in (
            "BACKEND",
            "EXECUTABLE IS NOT",
            "EXECUTABLE WAS NOT",
            "EXECUTABLE MISSING",
            "MACHINE PROFILE",
            "CLUSTER PROFILE",
            "MODULE STACK",
            "MODULE SETUP",
        )
    ):
        return OperatorFailureAssessment(
            family="backend_profile_failure",
            summary="the configured backend or machine profile could not be used",
            action="fix the configured backend or profile problem, then preview reconcile",
        )
    return OperatorFailureAssessment(
        family="unknown",
        summary="the campaign stopped because an operation could not be completed safely",
        action="preview reconcile and inspect the technical evidence",
    )


def config_review_evidence(
    review: Any,
    *,
    allow_unbound: bool = False,
) -> Dict[str, Any]:
    """Return a compact, presentation-only view of a config-lock review."""
    allowed = tuple(getattr(review, "allowed_changes", ()) or ())
    blocked = tuple(getattr(review, "blocked_changes", ()) or ())
    lock_existed = bool(getattr(review, "lock_existed", True))
    only_missing_lock = bool(blocked) and all(
        str(getattr(change, "path", "")) == "config_lock"
        for change in blocked
    )
    if allow_unbound and not lock_existed and only_missing_lock:
        state = "unavailable"
        blocked = ()
    elif blocked:
        state = "blocked"
    elif allowed:
        state = "allowed"
    else:
        state = "unchanged"
    return {
        "state": state,
        "n_allowed": len(allowed),
        "n_blocked": len(blocked),
        "allowed_paths": [str(change.path) for change in allowed],
        "blocked_paths": [str(change.path) for change in blocked],
    }


def invalid_config_review_evidence(error: Any) -> Dict[str, Any]:
    """Return presentation evidence for an unreadable config-lock review."""
    return {
        "state": "invalid",
        "n_allowed": 0,
        "n_blocked": 0,
        "allowed_paths": [],
        "blocked_paths": [],
        "error": str(error),
    }


@dataclass(frozen=True)
class SchedulerWorkAssessment:
    """Scheduler ownership after authenticated terminal evidence is applied."""

    terminal_job_ids: FrozenSet[str]
    pending_jobs: Tuple[Tuple[str, str], ...]
    scheduler_intents: Tuple[Mapping[str, Any], ...]
    local_intents: Tuple[Mapping[str, Any], ...]
    recovery_state: str
    completed_candidates: int
    retry_tasks: int
    reusable_outputs: int

    @property
    def pending_job_count(self) -> int:
        return len(self.pending_jobs)

    @property
    def scheduler_intent_count(self) -> int:
        return len(self.scheduler_intents)

    @property
    def local_intent_count(self) -> int:
        return len(self.local_intents)

    @property
    def has_unresolved_scheduler_work(self) -> bool:
        return bool(self.pending_jobs or self.scheduler_intents)

    @property
    def has_local_work(self) -> bool:
        return bool(self.local_intents)

    @property
    def has_terminal_recovery(self) -> bool:
        return self.recovery_state in _TERMINAL_RECOVERY_STATES


@dataclass(frozen=True)
class CampaignPresentationAssessment:
    """Normalised facts shared by status, preflight and reconcile wording."""

    scheduler: SchedulerWorkAssessment
    environment: EnvironmentLaunchAssessment
    reconcile: ReconcileLaunchAssessment
    config_state: str
    config_allowed_count: int
    config_blocked_count: int
    config_error: str

    @property
    def config_requires_reconcile(self) -> bool:
        return self.config_state == "allowed"

    @property
    def config_blocks_progress(self) -> bool:
        return self.config_state in {"blocked", "invalid"}

    @property
    def launch_evidence_ready(self) -> bool:
        return self.environment.launchable and self.reconcile.launchable


def _environment_launch_assessment(
    payload: Mapping[str, Any],
) -> EnvironmentLaunchAssessment:
    evidence = payload.get("_presentation_environment_generation")
    if not isinstance(evidence, Mapping):
        return EnvironmentLaunchAssessment(
            disposition="invalid",
            reason="execution environment evidence is unavailable",
            generation=-1,
        )
    disposition = str(evidence.get("disposition") or "")
    if not disposition:
        if evidence.get("error"):
            disposition = "invalid"
        elif evidence.get("config_matches") is True:
            disposition = "current"
        elif evidence.get("config_matches") is False:
            disposition = "reconcile_required"
        else:
            disposition = "invalid"
    if disposition not in _ENVIRONMENT_DISPOSITIONS:
        disposition = "invalid"
    try:
        generation = int(evidence.get("generation", -1))
    except (TypeError, ValueError):
        generation = -1
        disposition = "invalid"
    reason = str(
        evidence.get("reason")
        or evidence.get("error")
        or {
            "unbound_first_start": "the fresh campaign will create its first execution identity",
            "current": "the active environment generation matches the campaign configuration",
            "rebindable_on_resume": (
                "startup can safely create a generation bound to the current configuration"
            ),
            "reconcile_required": "campaign recovery is required before the environment can advance",
            "invalid": "execution environment evidence is invalid",
            "ownership_blocked": "active or uncertain ownership prevents an environment transition",
        }[disposition]
    )
    return EnvironmentLaunchAssessment(
        disposition=disposition,
        reason=reason,
        generation=generation,
    )


def _reconcile_launch_assessment(
    payload: Mapping[str, Any],
) -> ReconcileLaunchAssessment:
    evidence = payload.get("_presentation_reconcile_transaction_recovery")
    if not isinstance(evidence, Mapping):
        return ReconcileLaunchAssessment(
            disposition="none",
            reason="",
            state_backup_status="",
        )
    state = str(evidence.get("state") or "")
    if state in {"", "none", "recovered"}:
        disposition = "none"
    elif state == "recoverable" and bool(evidence.get("recoverable", False)):
        disposition = "recoverable"
    else:
        disposition = "blocked"
    return ReconcileLaunchAssessment(
        disposition=disposition,
        reason=str(evidence.get("reason") or ""),
        state_backup_status=str(evidence.get("state_backup_status") or ""),
    )


def _matches_current_state(
    payload: Mapping[str, Any],
    recovery: Mapping[str, Any],
) -> bool:
    for key in ("phase", "iteration", "replacement_round"):
        expected = payload.get(key)
        observed = recovery.get(key)
        if expected is None:
            continue
        if observed is None:
            return False
        try:
            if key in {"iteration", "replacement_round"}:
                if int(expected) != int(observed):
                    return False
            elif str(expected) != str(observed):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _terminal_recovery(
    payload: Mapping[str, Any],
) -> Tuple[str, FrozenSet[str], int, int, int]:
    recovery = payload.get("_presentation_scheduler_recovery")
    if not isinstance(recovery, Mapping):
        return "none", frozenset(), 0, 0, 0
    state = str(recovery.get("state") or "invalid")
    if state not in _TERMINAL_RECOVERY_STATES:
        return state, frozenset(), 0, 0, 0
    if not _matches_current_state(payload, recovery):
        return "invalid", frozenset(), 0, 0, 0
    raw_job_ids = recovery.get("original_job_ids") or []
    if not isinstance(raw_job_ids, (list, tuple)):
        return "invalid", frozenset(), 0, 0, 0
    terminal_job_ids = frozenset(
        str(job_id)
        for job_id in raw_job_ids
        if str(job_id)
    )
    if not terminal_job_ids:
        return "invalid", frozenset(), 0, 0, 0
    try:
        completed = int(recovery.get("n_scheduler_completed") or 0)
        retry = int(
            recovery.get(
                (
                    "n_retry"
                    if state in {"legacy_unverified", "validated"}
                    else "n_scheduler_retry"
                )
            )
            or 0
        )
        reusable = int(recovery.get("n_reusable") or 0)
    except (TypeError, ValueError):
        return "invalid", frozenset(), 0, 0, 0
    if min(completed, retry, reusable) < 0:
        return "invalid", frozenset(), 0, 0, 0
    return state, terminal_job_ids, completed, retry, reusable


def assess_campaign_presentation(
    payload: Mapping[str, Any],
) -> CampaignPresentationAssessment:
    """Combine bounded presentation evidence without reading or writing files."""
    (
        recovery_state,
        terminal_job_ids,
        completed_candidates,
        retry_tasks,
        reusable_outputs,
    ) = _terminal_recovery(payload)

    pending = payload.get("pending_jobs")
    pending_jobs = tuple(
        (str(phase), str(job_id))
        for phase, job_id in (
            pending.items() if isinstance(pending, Mapping) else ()
        )
        if job_id not in (None, "", False)
        and str(job_id) not in terminal_job_ids
    )

    intents = payload.get("active_submission_intents")
    scheduler_intents = []
    local_intents = []
    for intent in intents if isinstance(intents, (list, tuple)) else ():
        if not isinstance(intent, Mapping):
            continue
        job_id = str(intent.get("job_id") or "")
        if job_id:
            if job_id not in terminal_job_ids:
                scheduler_intents.append(intent)
        else:
            local_intents.append(intent)

    config = payload.get("_presentation_config_review")
    if isinstance(config, Mapping):
        config_state = str(config.get("state") or "unavailable")
        if config_state not in _CONFIG_REVIEW_STATES:
            config_state = "invalid"
        try:
            config_allowed_count = int(config.get("n_allowed") or 0)
            config_blocked_count = int(config.get("n_blocked") or 0)
        except (TypeError, ValueError):
            config_state = "invalid"
            config_allowed_count = 0
            config_blocked_count = 0
        if min(config_allowed_count, config_blocked_count) < 0:
            config_state = "invalid"
            config_allowed_count = 0
            config_blocked_count = 0
        elif config_state == "allowed" and config_allowed_count == 0:
            config_state = "invalid"
        elif config_state == "blocked" and config_blocked_count == 0:
            config_state = "invalid"
        elif (
            config_state in {"unchanged", "unavailable"}
            and (config_allowed_count or config_blocked_count)
        ):
            config_state = "invalid"
        config_error = str(config.get("error") or "")
    else:
        config_state = "unavailable"
        config_allowed_count = 0
        config_blocked_count = 0
        config_error = ""

    return CampaignPresentationAssessment(
        scheduler=SchedulerWorkAssessment(
            terminal_job_ids=terminal_job_ids,
            pending_jobs=pending_jobs,
            scheduler_intents=tuple(scheduler_intents),
            local_intents=tuple(local_intents),
            recovery_state=recovery_state,
            completed_candidates=completed_candidates,
            retry_tasks=retry_tasks,
            reusable_outputs=reusable_outputs,
        ),
        environment=_environment_launch_assessment(payload),
        reconcile=_reconcile_launch_assessment(payload),
        config_state=config_state,
        config_allowed_count=config_allowed_count,
        config_blocked_count=config_blocked_count,
        config_error=config_error,
    )


__all__ = [
    "CampaignPresentationAssessment",
    "EnvironmentLaunchAssessment",
    "OperatorFailureAssessment",
    "ReconcileLaunchAssessment",
    "SchedulerWorkAssessment",
    "assess_campaign_presentation",
    "classify_operator_failure",
    "config_review_evidence",
    "invalid_config_review_evidence",
]
