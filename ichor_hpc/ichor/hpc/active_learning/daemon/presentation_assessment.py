"""Shared read-only evidence assessment for daemon CLI presentation.

The daemon commands inspect overlapping campaign evidence.  This module
normalises that evidence before a command chooses human wording, so stale
state entries cannot override newer authenticated terminal records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Tuple


_TERMINAL_RECOVERY_STATES = frozenset({"awaiting_validation", "validated"})
_CONFIG_REVIEW_STATES = frozenset(
    {"unchanged", "allowed", "blocked", "invalid", "unavailable"}
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
                    if state == "validated"
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
        config_state=config_state,
        config_allowed_count=config_allowed_count,
        config_blocked_count=config_blocked_count,
        config_error=config_error,
    )


__all__ = [
    "CampaignPresentationAssessment",
    "SchedulerWorkAssessment",
    "assess_campaign_presentation",
    "config_review_evidence",
    "invalid_config_review_evidence",
]
