"""Authority for restarting a daemon solely to re-poll preserved work."""
from __future__ import annotations

import getpass
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from . import submission_intent
from .state import CampaignPhase, CampaignState
from .stop_control import (
    read_resume_transaction,
    read_resume_transaction_history,
    resume_transaction_path,
)


_SCHEDULER_REPOLL_OPERATIONS = frozenset(
    {
        "resume_scheduler_uncertain_cancel_stop",
        "resume_scheduler_uncertain_completed_stop",
        "resume_scheduler_uncertain_with_pending_stop",
    }
)


class PreservedSchedulerRepollError(ValueError):
    """Raised when apparent preserved-job authority is contradictory."""


@dataclass(frozen=True)
class PreservedSchedulerRepollAuthority:
    campaign_uid: str
    phase: str
    iteration: int
    replacement_round: int
    job_id: str
    submission_identity: str
    attempt_id: str
    expected_tasks: int
    expected_job_name: str
    expected_owner: str
    scheduler_identity_kind: str
    environment_generation: int
    environment_generation_digest_sha256: str
    transaction_id: str
    transaction_path: str
    transaction_sha256: str
    target_state_sha256: str

    def to_evidence(self) -> Dict[str, Any]:
        return {
            "campaign_uid": self.campaign_uid,
            "phase": self.phase,
            "iteration": self.iteration,
            "replacement_round": self.replacement_round,
            "job_id": self.job_id,
            "submission_identity": self.submission_identity,
            "attempt_id": self.attempt_id,
            "expected_tasks": self.expected_tasks,
            "expected_job_name": self.expected_job_name,
            "expected_owner": self.expected_owner,
            "scheduler_identity_kind": self.scheduler_identity_kind,
            "environment_generation": self.environment_generation,
            "environment_generation_digest_sha256": (
                self.environment_generation_digest_sha256
            ),
            "transaction_id": self.transaction_id,
            "transaction_path": self.transaction_path,
            "transaction_sha256": self.transaction_sha256,
            "target_state_sha256": self.target_state_sha256,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _transaction_candidates(
    campaign: Path,
) -> Tuple[Tuple[Path, Mapping[str, Any]], ...]:
    candidates = list(read_resume_transaction_history(campaign))
    active = read_resume_transaction(campaign)
    if active is not None and str(active.get("status") or "") == "state_written":
        active_id = str(active["transaction_id"])
        duplicate = next(
            (
                payload
                for _path, payload in candidates
                if str(payload["transaction_id"]) == active_id
            ),
            None,
        )
        if duplicate is None:
            candidates.append((resume_transaction_path(campaign), active))
        elif dict(duplicate) != dict(active):
            raise PreservedSchedulerRepollError(
                "resume transaction history conflicts with its active copy"
            )
    return tuple(candidates)


def _transaction_target_matches_state(
    transaction: Mapping[str, Any],
    state: CampaignState,
) -> bool:
    """Match restored ownership while allowing accounting counters to advance."""
    target = CampaignState.from_dict(dict(transaction["after_state"])).to_dict()
    current = CampaignState.from_dict(state.to_dict()).to_dict()
    target.pop("sacct_empty_streak", None)
    current.pop("sacct_empty_streak", None)
    return target == current


def resolve_preserved_scheduler_repoll_authority(
    campaign_dir: Union[str, Path],
    state: CampaignState,
) -> Optional[PreservedSchedulerRepollAuthority]:
    """Return exact re-poll authority, or ``None`` for ordinary ownership."""
    campaign = Path(campaign_dir).expanduser().resolve()
    if state.phase in {CampaignPhase.HALTED, CampaignPhase.DONE}:
        return None
    pending = tuple(
        (str(phase), str(job_id))
        for phase, job_id in state.pending_jobs.items()
        if job_id not in (None, "")
    )
    if len(pending) != 1:
        return None
    pending_phase, pending_job_id = pending[0]
    if pending_phase != state.phase.value:
        raise PreservedSchedulerRepollError(
            "preserved pending-job phase differs from campaign state"
        )

    matching_transactions = []
    for path, transaction in _transaction_candidates(campaign):
        if str(transaction.get("operation") or "") not in (
            _SCHEDULER_REPOLL_OPERATIONS
        ):
            continue
        if str(transaction.get("campaign_uid") or "") != str(
            state.campaign_uid
        ):
            continue
        if not _transaction_target_matches_state(transaction, state):
            continue
        matching_transactions.append((path, transaction))
    if not matching_transactions:
        return None
    transaction_path_value, transaction = max(
        matching_transactions,
        key=lambda item: (
            str(item[1].get("updated_at_iso") or ""),
            str(item[1].get("transaction_id") or ""),
        ),
    )

    inventory = submission_intent.inventory_intents(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    if inventory["errors"]:
        raise PreservedSchedulerRepollError(
            "submission-intent ownership contains malformed evidence"
        )
    active_intents = [
        record
        for record in inventory["records"]
        if str(record.get("status") or "")
        in submission_intent.ACTIVE_STATUSES
    ]
    if len(active_intents) != 1:
        raise PreservedSchedulerRepollError(
            "preserved re-poll requires exactly one active submission intent"
        )
    intent = active_intents[0]
    observed = (
        str(intent.get("phase") or ""),
        int(intent.get("iteration", -1)),
        int(intent.get("replacement_round", -1)),
        str(intent.get("job_id") or ""),
        str(intent.get("status") or ""),
    )
    expected = (
        state.phase.value,
        int(state.iteration),
        int(state.replacement_round),
        pending_job_id,
        str(intent.get("status") or ""),
    )
    if observed != expected or observed[-1] not in {"SUBMITTED", "ADOPTED"}:
        raise PreservedSchedulerRepollError(
            "preserved re-poll submission intent differs from campaign state"
        )
    expected_tasks = intent.get("expected_tasks")
    if (
        isinstance(expected_tasks, bool)
        or not isinstance(expected_tasks, int)
        or expected_tasks < 1
    ):
        raise PreservedSchedulerRepollError(
            "preserved re-poll task count is invalid"
        )
    if str(intent.get("submission_kind") or "") == "array":
        task_ids = submission_intent.intent_submitted_logical_task_ids(
            campaign,
            intent,
        )
        if len(task_ids) != int(expected_tasks):
            raise PreservedSchedulerRepollError(
                "preserved re-poll task map differs from its intent"
            )
    elif int(expected_tasks) != 1:
        raise PreservedSchedulerRepollError(
            "preserved scalar re-poll must contain exactly one task"
        )

    generation_number = intent.get("environment_generation")
    generation_digest = intent.get("environment_generation_digest_sha256")
    if (
        isinstance(generation_number, bool)
        or not isinstance(generation_number, int)
        or not isinstance(generation_digest, str)
    ):
        raise PreservedSchedulerRepollError(
            "preserved re-poll intent lacks its producer environment"
        )
    from ..execution_identity import read_environment_generation

    generation = read_environment_generation(
        campaign,
        generation=int(generation_number),
        expected_campaign_uid=str(state.campaign_uid),
    )
    if str(generation.get("digest_sha256") or "") != generation_digest:
        raise PreservedSchedulerRepollError(
            "preserved re-poll producer environment digest mismatch"
        )

    return PreservedSchedulerRepollAuthority(
        campaign_uid=str(state.campaign_uid),
        phase=state.phase.value,
        iteration=int(state.iteration),
        replacement_round=int(state.replacement_round),
        job_id=pending_job_id,
        submission_identity=str(intent["submission_identity"]),
        attempt_id=str(intent["attempt_id"]),
        expected_tasks=int(expected_tasks),
        expected_job_name=str(intent["expected_job_name"]),
        expected_owner=getpass.getuser(),
        scheduler_identity_kind=str(intent["scheduler_identity_kind"]),
        environment_generation=int(generation_number),
        environment_generation_digest_sha256=str(generation_digest),
        transaction_id=str(transaction["transaction_id"]),
        transaction_path=str(transaction_path_value),
        transaction_sha256=_file_sha256(transaction_path_value),
        target_state_sha256=str(transaction["after_state_sha256"]),
    )


def validate_preserved_scheduler_repoll_continuation(
    campaign_dir: Union[str, Path],
    state: CampaignState,
    authority: PreservedSchedulerRepollAuthority,
) -> None:
    """Recheck preserved ownership after an allowed environment transition."""
    campaign = Path(campaign_dir).expanduser().resolve()
    transaction_path_value = Path(authority.transaction_path)
    try:
        transaction_path_value.relative_to(campaign)
    except ValueError as exc:
        raise PreservedSchedulerRepollError(
            "preserved re-poll transaction escaped the campaign"
        ) from exc
    if transaction_path_value.is_symlink() or not transaction_path_value.is_file():
        raise PreservedSchedulerRepollError(
            "preserved re-poll transaction changed or disappeared"
        )
    if _file_sha256(transaction_path_value) != authority.transaction_sha256:
        raise PreservedSchedulerRepollError(
            "preserved re-poll transaction content changed"
        )
    pending = tuple(
        (str(phase), str(job_id))
        for phase, job_id in state.pending_jobs.items()
        if job_id not in (None, "")
    )
    expected_pending = ((authority.phase, authority.job_id),)
    if (
        str(state.campaign_uid) != authority.campaign_uid
        or state.phase.value != authority.phase
        or int(state.iteration) != authority.iteration
        or int(state.replacement_round) != authority.replacement_round
        or pending != expected_pending
    ):
        raise PreservedSchedulerRepollError(
            "campaign ownership changed before preserved scheduler re-poll"
        )
    inventory = submission_intent.inventory_intents(
        campaign,
        expected_campaign_uid=authority.campaign_uid,
    )
    if inventory["errors"]:
        raise PreservedSchedulerRepollError(
            "submission-intent evidence changed before preserved re-poll"
        )
    active = [
        record
        for record in inventory["records"]
        if str(record.get("status") or "")
        in submission_intent.ACTIVE_STATUSES
    ]
    if len(active) != 1:
        raise PreservedSchedulerRepollError(
            "active scheduler ownership changed before preserved re-poll"
        )
    intent = active[0]
    observed = (
        str(intent.get("phase") or ""),
        int(intent.get("iteration", -1)),
        int(intent.get("replacement_round", -1)),
        str(intent.get("job_id") or ""),
        str(intent.get("submission_identity") or ""),
        str(intent.get("attempt_id") or ""),
        int(intent.get("expected_tasks", -1)),
        str(intent.get("expected_job_name") or ""),
        str(intent.get("scheduler_identity_kind") or ""),
        int(intent.get("environment_generation", -1)),
        str(intent.get("environment_generation_digest_sha256") or ""),
    )
    expected = (
        authority.phase,
        authority.iteration,
        authority.replacement_round,
        authority.job_id,
        authority.submission_identity,
        authority.attempt_id,
        authority.expected_tasks,
        authority.expected_job_name,
        authority.scheduler_identity_kind,
        authority.environment_generation,
        authority.environment_generation_digest_sha256,
    )
    if observed != expected:
        raise PreservedSchedulerRepollError(
            "submission intent changed before preserved scheduler re-poll"
        )


__all__ = [
    "PreservedSchedulerRepollAuthority",
    "PreservedSchedulerRepollError",
    "resolve_preserved_scheduler_repoll_authority",
    "validate_preserved_scheduler_repoll_continuation",
]
