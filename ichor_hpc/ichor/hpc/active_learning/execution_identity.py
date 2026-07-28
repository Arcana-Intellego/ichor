"""Immutable campaign execution mode and environment generation records."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
from .strict_json import strict_json as json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .config import CampaignConfig
from .daemon.config_lock import config_fingerprint
from .daemon.state import (
    CampaignPhase,
    atomic_write_json,
    read_state,
    write_state,
)
from .daemon.filesystem import campaign_owned_path, operational_path


EXECUTION_IDENTITY_SCHEMA_VERSION = 1
ENVIRONMENT_GENERATION_SCHEMA_VERSION = 1
ENVIRONMENT_CURRENT_SCHEMA_VERSION = 1
VALID_EXECUTION_MODES = frozenset({"live", "dry_run"})

_REBINDABLE_REFERENCE_COMMIT_STATES = frozenset(
    {
        "prepared",
        "partially_moved",
        "cache_incomplete",
        "publication_incomplete",
    }
)


def _scheduler_cancellation_transition_boundary(
    campaign: Path,
    state: Any,
) -> Optional[Dict[str, Any]]:
    """Recognise exact terminal cancellation evidence without parsing outputs."""
    from .daemon.phase_executor import SBATCH_PHASES
    from .daemon.scheduler_recovery import scheduler_terminal_recoveries
    from .daemon.submission_intent import intent_attempt_records

    phase = CampaignPhase(state.phase)
    if phase.value not in SBATCH_PHASES:
        return None
    recoveries = scheduler_terminal_recoveries(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase=phase.value,
        iteration=int(state.iteration),
        replacement_round=int(getattr(state, "replacement_round", 0)),
    )
    if not recoveries:
        return None
    attempts = [
        record
        for record in intent_attempt_records(
            campaign,
            phase.value,
            int(state.iteration),
            expected_campaign_uid=str(state.campaign_uid),
        )
        if int(record.get("replacement_round", 0))
        == int(getattr(state, "replacement_round", 0))
    ]
    if not attempts:
        return None
    latest_attempt = max(
        attempts,
        key=lambda record: int(record.get("attempt_sequence", 0)),
    )
    recovery_identities = {
        str(recovery["intent"].get("submission_identity") or "")
        for recovery in recoveries
    }
    if (
        str(latest_attempt.get("submission_identity") or "")
        not in recovery_identities
    ):
        return None
    from .daemon.cluster_profile import profile_value

    current_scheduler = str(
        profile_value("hpc", "scheduler", default="slurm") or "slurm"
    ).strip().lower()
    recorded_schedulers = {
        str(
            recovery["intent"].get("scheduler_identity_kind") or "slurm"
        ).strip().lower()
        for recovery in recoveries
    }
    if recorded_schedulers != {current_scheduler}:
        raise ExecutionIdentityError(
            "scheduler kind cannot change during "
            + phase.value
            + " recovery: recorded "
            + ", ".join(sorted(recorded_schedulers))
            + ", active "
            + current_scheduler
        )
    latest_by_task: Dict[int, Mapping[str, Any]] = {}
    for recovery in recoveries:
        for outcome in recovery["receipt"].get("outcomes", []):
            latest_by_task[int(outcome["logical_task_id"])] = outcome
    completed = {
        task_id
        for task_id, outcome in latest_by_task.items()
        if str(outcome.get("status") or "") == "COMPLETED"
        and outcome.get("exit_code") == [0, 0]
    }
    retry = set(latest_by_task).difference(completed)
    return {
        "transition_kind": "scheduler_cancellation_recovery",
        "scheduler_terminal_receipts": len(recoveries),
        "scheduler_completed_candidates": len(completed),
        "scheduler_retry_candidates": len(retry),
    }


def _validate_ferebus_transition_boundary(campaign: Path, state: Any) -> None:
    """Require a clean, committed-data-only FEREBUS submission boundary."""
    from .daemon.recovery_contracts import phase_recovery_contract_error
    from .layout import qm_reference_data_dir, trained_models_dir
    from .versioning.reference_data import ReferenceDataVersioning
    from .versioning.trained_models import TrainedModelVersioning

    phase = CampaignPhase(state.phase)
    staging = trained_models_dir(campaign) / "iteration-staging"
    if staging.exists() or staging.is_symlink():
        raise ExecutionIdentityError(
            "environment transition at "
            + phase.value
            + " requires reconcile to archive TRAINED_MODELS/iteration-staging first"
        )

    iteration = int(state.iteration)
    reference_version = int(state.reference_data_version)
    model_version = int(state.models_version)
    if phase is CampaignPhase.INITIAL_FEREBUS:
        expected = (0, 0, -1)
        observed = (iteration, reference_version, model_version)
        if observed != expected:
            raise ExecutionIdentityError(
                "INITIAL_FEREBUS environment transition requires iteration/reference/model "
                "versions 0/0/-1"
            )
    elif phase is CampaignPhase.FEREBUS:
        if (
            reference_version < 1
            or iteration != reference_version
            or model_version != reference_version - 1
        ):
            raise ExecutionIdentityError(
                "FEREBUS environment transition requires iteration N, reference version N, "
                "and incumbent model version N-1"
            )
    else:  # pragma: no cover - guarded by the caller
        raise ExecutionIdentityError("invalid FEREBUS environment-transition phase")

    reference_current = ReferenceDataVersioning(
        qm_reference_data_dir(campaign)
    ).current_version()
    model_current = TrainedModelVersioning(
        trained_models_dir(campaign)
    ).current_version()
    expected_model_current = None if model_version < 0 else model_version
    if reference_current != reference_version or model_current != expected_model_current:
        raise ExecutionIdentityError(
            "environment transition at "
            + phase.value
            + " requires committed current pointers to match campaign state"
        )

    contract_error = phase_recovery_contract_error(
        campaign,
        state,
        verification="authority",
    )
    if contract_error is not None:
        raise ExecutionIdentityError(
            "environment transition at "
            + phase.value
            + " failed its recovery contract: "
            + str(contract_error)
        )


def inspect_allocation_check_transition_boundary(
    campaign: Path,
    state: Any,
) -> Dict[str, Any]:
    """Inspect a scheduler-free allocation-check recovery boundary."""
    from .daemon.recovery_contracts import phase_recovery_contract_error
    from .layout import qm_reference_data_dir, trained_models_dir
    from .point_allocation import (
        pending_attempts,
        point_allocation_path,
        read_point_allocation,
    )
    from .replacement_sampling import inspect_replacement_sample_recovery
    from .versioning.reference_data import ReferenceDataVersioning
    from .versioning.trained_models import TrainedModelVersioning

    try:
        phase = CampaignPhase(state.phase)
        if phase not in {
            CampaignPhase.INITIAL_ALLOCATION_CHECK,
            CampaignPhase.ALLOCATION_CHECK,
        }:
            raise ValueError("phase is not an allocation check")
        iteration = int(state.iteration)
        context = (
            "bootstrap"
            if phase is CampaignPhase.INITIAL_ALLOCATION_CHECK
            else "active"
        )
        reference_current = ReferenceDataVersioning(
            qm_reference_data_dir(campaign)
        ).current_version()
        model_current = TrainedModelVersioning(
            trained_models_dir(campaign)
        ).current_version()
        if context == "bootstrap":
            observed = (
                iteration,
                int(state.reference_data_version),
                int(state.validation_set_version),
                int(state.models_version),
                reference_current,
                model_current,
            )
            if observed != (0, -1, -1, -1, None, None):
                raise ValueError(
                    "bootstrap allocation check requires iteration zero with no "
                    "committed data or model pointer"
                )
        else:
            expected_version = iteration - 1
            observed = (
                int(state.reference_data_version),
                int(state.validation_set_version),
                int(state.models_version),
                reference_current,
                model_current,
            )
            if iteration < 1 or observed != (expected_version,) * 5:
                raise ValueError(
                    "active allocation check requires reference data, validation "
                    "data and models through iteration N-1"
                )

        contract_error = phase_recovery_contract_error(
            campaign,
            state,
            verification="authority",
        )
        if contract_error is not None:
            raise ValueError(str(contract_error))
        allocation = read_point_allocation(
            point_allocation_path(
                campaign,
                context=context,
                iteration=iteration,
            ),
            expected_campaign_uid=str(state.campaign_uid),
            expected_context=context,
            expected_iteration=iteration,
        )
        pending = pending_attempts(allocation)
        sample_state = "not_required"
        replacement_round = int(getattr(state, "replacement_round", 0))
        if pending:
            rounds = {int(record.get("round", -1)) for record in pending}
            if len(rounds) != 1 or min(rounds) <= 0:
                raise ValueError(
                    "point allocation has incompatible pending replacement rounds"
                )
            pending_round = next(iter(rounds))
            if replacement_round != pending_round:
                raise ValueError(
                    "campaign replacement round does not match pending point allocation"
                )
            sample = inspect_replacement_sample_recovery(
                campaign,
                context=context,
                iteration=iteration,
                replacement_round=pending_round,
                expected_campaign_uid=str(state.campaign_uid),
            )
            sample_state = str(sample.get("state") or "")
            if sample_state == "conflicting":
                raise ValueError(
                    "replacement sample evidence conflicts with pending allocation: "
                    + str(sample.get("reason") or "unknown conflict")
                )
        return {
            "safe": True,
            "reason": None,
            "transition_kind": "allocation_check_recovery",
            "context": context,
            "replacement_round": replacement_round,
            "pending_tasks": int(len(pending)),
            "replacement_sample_state": sample_state,
            "allocation_complete": bool(
                (allocation.get("summary") or {}).get("complete", False)
            ),
        }
    except Exception as exc:
        return {
            "safe": False,
            "reason": str(exc),
            "transition_kind": "allocation_check_recovery",
        }


def _validate_allocation_check_transition_boundary(
    campaign: Path,
    state: Any,
) -> Dict[str, Any]:
    result = inspect_allocation_check_transition_boundary(campaign, state)
    if not bool(result.get("safe", False)):
        raise ExecutionIdentityError(
            "environment transition at allocation check is unsafe: "
            + str(result.get("reason") or "unknown allocation evidence")
        )
    return {
        key: value
        for key, value in result.items()
        if key not in {"safe", "reason"}
    }


def _validate_ariadne_retry_transition_boundary(
    campaign: Path,
    state: Any,
    *,
    task_scan: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate an all-retry or all-complete ARIADNE recovery boundary."""
    from .daemon.array_recovery import scan_array_tasks
    from .daemon.recovery_contracts import phase_recovery_contract_error

    phase = CampaignPhase(state.phase)
    if phase is not CampaignPhase.ARIADNE_ARRAY:  # pragma: no cover - caller guard
        raise ExecutionIdentityError(
            "invalid ARIADNE environment-transition phase"
        )

    contract_error = phase_recovery_contract_error(
        campaign,
        state,
        verification="authority",
    )
    if contract_error is not None:
        raise ExecutionIdentityError(
            "environment transition at ARIADNE_ARRAY failed its recovery contract: "
            + str(contract_error)
        )

    try:
        scan = (
            dict(task_scan)
            if isinstance(task_scan, Mapping)
            else scan_array_tasks(
                campaign,
                phase,
                int(state.iteration),
                force_resubmit=False,
            )
        )
        observed_phase = str(scan["phase"])
        observed_iteration = int(scan["iteration"])
        logical_total = int(scan["logical_total"])
        n_complete = int(scan["n_complete"])
        n_reuse = int(scan["n_reuse"])
        n_retry = int(scan["n_retry"])
        retry_task_ids = [int(task_id) for task_id in scan["retry_task_ids"]]
        all_complete = bool(scan["all_complete"])
    except Exception as exc:
        raise ExecutionIdentityError(
            "environment transition at ARIADNE_ARRAY could not validate retry tasks: "
            + type(exc).__name__
            + ": "
            + str(exc)[:200]
        ) from exc

    if observed_phase != phase.value or observed_iteration != int(state.iteration):
        raise ExecutionIdentityError(
            "environment transition at ARIADNE_ARRAY found inconsistent task-scan "
            "phase or iteration identity"
        )
    if logical_total <= 0:
        raise ExecutionIdentityError(
            "environment transition at ARIADNE_ARRAY requires a non-empty logical "
            "task set"
        )
    expected_task_ids = list(range(logical_total))
    if n_complete == 0 and n_reuse == 0 and not all_complete:
        if n_retry != logical_total or retry_task_ids != expected_task_ids:
            raise ExecutionIdentityError(
                "environment transition at ARIADNE_ARRAY requires the retry set to "
                "cover every logical task exactly once"
            )
        return {
            "transition_kind": "ariadne_all_retry",
            "logical_total": int(logical_total),
        }

    if not (
        n_complete == logical_total
        and n_reuse == logical_total
        and n_retry == 0
        and not retry_task_ids
        and all_complete
    ):
        raise ExecutionIdentityError(
            "environment transition at ARIADNE_ARRAY requires either every logical "
            "task to be retried or every logical task to be complete for "
            "postprocessing; observed "
            + str(n_reuse)
            + " reusable task(s) out of "
            + str(logical_total)
        )

    from .daemon.ariadne_publication import classify_ariadne_publication
    from .daemon.submission_intent import resolve_ariadne_postprocess_source

    publication = classify_ariadne_publication(
        campaign,
        int(state.iteration),
        expected_campaign_uid=str(state.campaign_uid),
    )
    publication_state = str(publication.get("state") or "")
    replayable_states = {
        "absent",
        "incomplete",
        "stale_results_binding",
        "archive_incomplete",
    }
    if publication_state == "complete" and not bool(
        publication.get("accepted", False)
    ):
        raise ExecutionIdentityError(
            "environment transition at an all-complete ARIADNE boundary is "
            "blocked by a complete rejected batch decision"
        )
    if publication_state not in replayable_states and not (
        publication_state == "complete"
        and bool(publication.get("accepted", False))
    ):
        raise ExecutionIdentityError(
            "environment transition at an all-complete ARIADNE boundary requires "
            "derived publication to be absent, recoverably incomplete, or an "
            "accepted uncommitted publication; observed "
            + (publication_state or "unknown")
            + ": "
            + str(publication.get("reason") or "")
        )
    expected_task_set_sha256 = hashlib.sha256(
        ",".join(str(task_id) for task_id in expected_task_ids).encode("ascii")
    ).hexdigest()
    try:
        source = resolve_ariadne_postprocess_source(
            campaign,
            campaign_uid=str(state.campaign_uid),
            iteration=int(state.iteration),
            logical_total=int(logical_total),
            logical_task_set_sha256=expected_task_set_sha256,
        )
    except Exception as exc:
        raise ExecutionIdentityError(
            "environment transition at an all-complete ARIADNE boundary could "
            "not validate the producer contract: "
            + type(exc).__name__
            + ": "
            + str(exc)[:200]
        ) from exc
    return {
        "transition_kind": "ariadne_postprocess_only",
        "logical_total": int(logical_total),
        "producer_submission_identity": str(source["submission_identity"]),
        "producer_job_id": str(source["job_id"]),
        "producer_environment_generation": int(source["environment_generation"]),
        "producer_environment_generation_digest_sha256": str(
            source["environment_generation_digest_sha256"]
        ),
        "publication_state": publication_state,
        "postprocess_source": dict(source),
    }


def _validate_aimall_postprocess_transition_boundary(
    campaign: Path,
    state: Any,
) -> Dict[str, Any]:
    """Validate scheduler-free replay of one completed AIMAll array."""
    from .daemon.recovery_contracts import phase_recovery_contract_error
    from .daemon.submission_intent import resolve_aimall_postprocess_source
    from .layout import qm_reference_data_dir, trained_models_dir
    from .versioning.reference_data import ReferenceDataVersioning
    from .versioning.trained_models import TrainedModelVersioning

    phase = CampaignPhase(state.phase)
    if phase.value not in {
        CampaignPhase.INITIAL_AIMALL.value,
        CampaignPhase.AIMALL.value,
        CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value,
        CampaignPhase.REPLACEMENT_AIMALL.value,
    }:
        raise ExecutionIdentityError(
            "invalid AIMAll postprocess environment-transition phase"
        )
    iteration = int(state.iteration)
    reference_current = ReferenceDataVersioning(
        qm_reference_data_dir(campaign)
    ).current_version()
    model_current = TrainedModelVersioning(
        trained_models_dir(campaign)
    ).current_version()
    if phase in {
        CampaignPhase.INITIAL_AIMALL,
        CampaignPhase.INITIAL_REPLACEMENT_AIMALL,
    }:
        if (
            iteration != 0
            or int(state.reference_data_version) != -1
            or int(state.models_version) != -1
        ):
            raise ExecutionIdentityError(
                "initial AIMAll postprocess recovery requires bootstrap iteration "
                "zero with no committed reference-data or model version"
            )
        if (reference_current, model_current) != (None, None):
            raise ExecutionIdentityError(
                "initial AIMAll postprocess recovery requires no committed "
                "current reference-data or model pointer"
            )
    else:
        expected_version = iteration - 1
        if iteration < 1 or (
            int(state.reference_data_version),
            int(state.models_version),
        ) != (expected_version, expected_version):
            raise ExecutionIdentityError(
                "AIMAll postprocess recovery requires iteration N with "
                "reference-data and model versions N-1"
            )
        if (reference_current, model_current) != (
            expected_version,
            expected_version,
        ):
            raise ExecutionIdentityError(
                "AIMAll postprocess recovery requires committed current pointers "
                "to match campaign state"
            )

    contract_error = phase_recovery_contract_error(
        campaign,
        state,
        verification="authority",
    )
    if contract_error is not None:
        raise ExecutionIdentityError(
            "AIMAll postprocess recovery failed its phase contract: "
            + str(contract_error)
        )
    try:
        source = resolve_aimall_postprocess_source(
            campaign,
            campaign_uid=str(state.campaign_uid),
            phase_name=phase.value,
            iteration=iteration,
            replacement_round=int(getattr(state, "replacement_round", 0)),
        )
    except Exception as exc:
        raise ExecutionIdentityError(
            "AIMAll postprocess recovery could not validate the completed "
            "producer contract: "
            + type(exc).__name__
            + ": "
            + str(exc)[:200]
        ) from exc
    return {
        "transition_kind": "aimall_postprocess_only",
        "logical_total": int(source["logical_total"]),
        "producer_submission_identity": str(source["submission_identity"]),
        "producer_job_id": str(source["job_id"]),
        "producer_environment_generation": int(source["environment_generation"]),
        "producer_environment_generation_digest_sha256": str(
            source["environment_generation_digest_sha256"]
        ),
        "postprocess_source": dict(source),
    }


def _validate_gaussian_postprocess_transition_boundary(
    campaign: Path,
    state: Any,
) -> Dict[str, Any]:
    """Validate scheduler-free replay of one completed Gaussian array."""
    from .daemon.recovery_contracts import phase_recovery_contract_error
    from .daemon.submission_intent import resolve_gaussian_postprocess_source
    from .layout import qm_reference_data_dir, trained_models_dir
    from .versioning.reference_data import ReferenceDataVersioning
    from .versioning.trained_models import TrainedModelVersioning

    phase = CampaignPhase(state.phase)
    if phase.value not in {
        CampaignPhase.INITIAL_GAUSSIAN.value,
        CampaignPhase.GAUSSIAN.value,
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value,
        CampaignPhase.REPLACEMENT_GAUSSIAN.value,
    }:
        raise ExecutionIdentityError(
            "invalid Gaussian postprocess environment-transition phase"
        )
    iteration = int(state.iteration)
    reference_current = ReferenceDataVersioning(
        qm_reference_data_dir(campaign)
    ).current_version()
    model_current = TrainedModelVersioning(
        trained_models_dir(campaign)
    ).current_version()
    if phase in {
        CampaignPhase.INITIAL_GAUSSIAN,
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
    }:
        expected_state = (0, -1, -1)
        observed_state = (
            iteration,
            int(state.reference_data_version),
            int(state.models_version),
        )
        if observed_state != expected_state or (
            reference_current,
            model_current,
        ) != (None, None):
            raise ExecutionIdentityError(
                "initial Gaussian postprocess recovery requires bootstrap "
                "iteration zero with no committed data or model pointer"
            )
    else:
        expected_version = iteration - 1
        if iteration < 1 or (
            int(state.reference_data_version),
            int(state.models_version),
            reference_current,
            model_current,
        ) != (
            expected_version,
            expected_version,
            expected_version,
            expected_version,
        ):
            raise ExecutionIdentityError(
                "Gaussian postprocess recovery requires data and model versions N-1"
            )
    contract_error = phase_recovery_contract_error(
        campaign,
        state,
        verification="authority",
    )
    if contract_error is not None:
        raise ExecutionIdentityError(
            "Gaussian postprocess recovery failed its phase contract: "
            + str(contract_error)
        )
    try:
        source = resolve_gaussian_postprocess_source(
            campaign,
            campaign_uid=str(state.campaign_uid),
            phase_name=phase.value,
            iteration=iteration,
            replacement_round=int(getattr(state, "replacement_round", 0)),
        )
    except Exception as exc:
        raise ExecutionIdentityError(
            "Gaussian postprocess recovery could not validate the completed "
            "producer contract: "
            + type(exc).__name__
            + ": "
            + str(exc)[:200]
        ) from exc
    return {
        "transition_kind": "gaussian_postprocess_only",
        "logical_total": int(source["logical_total"]),
        "producer_submission_identity": str(source["submission_identity"]),
        "producer_job_id": str(source["job_id"]),
        "producer_environment_generation": int(source["environment_generation"]),
        "producer_environment_generation_digest_sha256": str(
            source["environment_generation_digest_sha256"]
        ),
        "postprocess_source": dict(source),
    }


def _validate_phase_b_transition_boundary(
    campaign: Path,
    state: Any,
    *,
    intent_records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate a clean Phase B retry that owns no scheduler work or output."""
    from .daemon.recovery_contracts import (
        ariadne_results_recovery_summary,
        phase_recovery_contract_error,
    )
    from .layout import (
        active_allocation_dir,
        active_iteration_dir,
        active_phase_b_dir,
        qm_reference_data_dir,
        trained_models_dir,
    )
    from .versioning.reference_data import ReferenceDataVersioning
    from .versioning.trained_models import TrainedModelVersioning

    phase = CampaignPhase(state.phase)
    if phase is not CampaignPhase.PHASE_B_DIVERSITY:  # pragma: no cover
        raise ExecutionIdentityError(
            "invalid Phase B environment-transition phase"
        )
    iteration = int(state.iteration)
    expected_version = iteration - 1
    if iteration < 1 or (
        int(state.reference_data_version),
        int(state.models_version),
    ) != (expected_version, expected_version):
        raise ExecutionIdentityError(
            "PHASE_B_DIVERSITY environment transition requires iteration N "
            "with reference-data and model versions N-1"
        )
    reference_current = ReferenceDataVersioning(
        qm_reference_data_dir(campaign)
    ).current_version()
    model_current = TrainedModelVersioning(
        trained_models_dir(campaign)
    ).current_version()
    if (reference_current, model_current) != (
        expected_version,
        expected_version,
    ):
        raise ExecutionIdentityError(
            "PHASE_B_DIVERSITY environment transition requires committed "
            "reference-data and model pointers to match campaign state"
        )
    contract_error = phase_recovery_contract_error(
        campaign,
        state,
        verification="authority",
    )
    if contract_error is not None:
        raise ExecutionIdentityError(
            "environment transition at PHASE_B_DIVERSITY failed its recovery "
            "contract: "
            + str(contract_error)
        )

    iteration_root = active_iteration_dir(campaign, iteration)
    phase_b_root = active_phase_b_dir(iteration_root)
    if phase_b_root.is_symlink():
        raise ExecutionIdentityError(
            "PHASE_B_DIVERSITY environment transition found a symlinked Phase B output root"
        )
    if phase_b_root.exists():
        if not phase_b_root.is_dir():
            raise ExecutionIdentityError(
                "PHASE_B_DIVERSITY environment transition found an invalid Phase B output root"
            )
        entries = sorted(path.name for path in phase_b_root.iterdir())
        if entries:
            raise ExecutionIdentityError(
                "PHASE_B_DIVERSITY environment transition refuses partial Phase B "
                "output; inspect "
                + str(phase_b_root)
                + " (entries: "
                + ", ".join(entries[:5])
                + (", ..." if len(entries) > 5 else "")
                + ")"
            )
    allocation_path = active_allocation_dir(iteration_root) / "POINT_ALLOCATION.json"
    if allocation_path.exists() or allocation_path.is_symlink():
        raise ExecutionIdentityError(
            "PHASE_B_DIVERSITY environment transition refuses an existing active "
            "point-allocation manifest: "
            + str(allocation_path)
        )

    matching_intents = [
        record
        for record in intent_records
        if str(record.get("phase") or "") == phase.value
        and int(record.get("iteration", -1)) == iteration
    ]
    for intent in matching_intents:
        status = str(intent.get("status") or "")
        reason = str(intent.get("reason") or "")
        if intent.get("job_id") not in (None, "", False):
            raise ExecutionIdentityError(
                "PHASE_B_DIVERSITY environment transition requires its historical "
                "retry intent to be jobless"
            )
        if status == "FAILED":
            continue
        if status == "SUPERSEDED" and reason == "reconcile_apply_retry":
            continue
        raise ExecutionIdentityError(
            "PHASE_B_DIVERSITY environment transition found an incompatible "
            "historical intent status: "
            + (status or "missing")
        )

    summary = ariadne_results_recovery_summary(
        campaign,
        iteration,
        str(state.campaign_uid),
        verification="authority",
    )
    return {
        "transition_kind": "phase_b_pre_submission_retry",
        "ariadne_expected_tasks": int(summary["expected_tasks"]),
        "ariadne_accepted_tasks": int(summary["accepted_tasks"]),
        "ariadne_rejected_tasks": int(summary["rejected_tasks"]),
        "ariadne_tasks_resubmitted": 0,
    }


_ENVIRONMENT_FINGERPRINT_KEYS = (
    "python_executable",
    "python_version",
    "ichor_git",
    "ichor_package_tree_sha256",
    "dependencies",
    "pyferebus",
    "ariadne",
    "ferebus_executable",
    "machine_profile",
    "loaded_modules",
    "native_library_paths",
    "campaign_schema_version",
)
_ENVIRONMENT_GENERATION_KEYS = frozenset({
    "schema_version",
    "generation",
    "campaign_uid",
    "created_at_iso",
    "host",
    "operator",
    "python_executable",
    "python_version",
    "ichor_git",
    "ichor_package_tree_sha256",
    "dependencies",
    "pyferebus",
    "ariadne",
    "ferebus_executable",
    "machine_profile",
    "loaded_modules",
    "native_library_paths",
    "campaign_schema_version",
    "campaign_config_sha256",
    "config_lock_sha256",
    "campaign_dir",
    "environment_fingerprint_sha256",
    "digest_sha256",
})
_EXECUTION_IDENTITY_KEYS = frozenset({
    "schema_version",
    "campaign_uid",
    "mode",
    "campaign_random_seed",
    "campaign_schema_version",
    "initial_config_sha256",
    "environment_generation",
    "environment_generation_digest_sha256",
    "created_at_iso",
    "digest_sha256",
})


class ExecutionIdentityError(ValueError):
    """Raised when launch mode or environment identity is inconsistent."""


def execution_identity_path(campaign_dir: Union[str, Path]) -> Path:
    return operational_path(campaign_dir, "execution_identity.json")


def environment_generations_dir(campaign_dir: Union[str, Path]) -> Path:
    return operational_path(campaign_dir, "environment_generations")


def environment_current_path(campaign_dir: Union[str, Path]) -> Path:
    return operational_path(campaign_dir, "environment_current.json")


def _canonical_digest(payload: Dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("digest_sha256", None)
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _environment_fingerprint_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only execution-affecting fields from a generation record.

    Campaign configuration is governed by the config-lock contract.  Host,
    operator, timestamps, generation numbers and paths are provenance rather
    than execution identity, so they must not make a login-node restart look
    like software drift.
    """
    return {key: payload.get(key) for key in _ENVIRONMENT_FINGERPRINT_KEYS}


def _environment_fingerprint(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        _environment_fingerprint_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExecutionIdentityError(label + " must be a lowercase SHA-256 digest")
    return value


def _exact_non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ExecutionIdentityError(label + " must be a non-negative integer")
    return int(value)


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    observed = set(payload)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ExecutionIdentityError(label + " fields are invalid: " + "; ".join(details))


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionIdentityError(label + " must be a non-empty string")
    return value


def _validate_timestamp(value: Any, label: str) -> str:
    text = _non_empty_string(value, label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ExecutionIdentityError(label + " must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ExecutionIdentityError(label + " must include a timezone")
    return text


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(roots: Iterable[Path]) -> str:
    """Return a bounded source identity without traversing package trees."""
    resolved_roots = sorted(
        {path.resolve() for path in roots if path.exists()},
        key=lambda item: str(item),
    )
    repositories: Dict[str, Dict[str, Any]] = {}
    fallback_roots: List[Dict[str, Any]] = []
    package_distributions = importlib.metadata.packages_distributions()
    for root in resolved_roots:
        try:
            repository_text = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(root),
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            repository = Path(repository_text).resolve()
            relative = root.relative_to(repository).as_posix()
        except (OSError, ValueError, subprocess.SubprocessError):
            distributions = sorted(
                package_distributions.get(root.name, [])
            )
            fallback_roots.append(
                {
                    "root": str(root),
                    "distributions": [
                        {
                            "name": name,
                            "version": _distribution_version(name),
                        }
                        for name in distributions
                    ],
                    "initialiser_sha256": _sha256_file(root / "__init__.py"),
                }
            )
            continue
        key = str(repository)
        record = repositories.setdefault(
            key,
            {
                "repository": key,
                "git": _git_identity(repository),
                "package_roots": [],
            },
        )
        record["package_roots"].append(relative)
    for record in repositories.values():
        record["package_roots"] = sorted(set(record["package_roots"]))
    payload = {
        "repositories": [repositories[key] for key in sorted(repositories)],
        "installed_roots": fallback_roots,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_identity(repo_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "commit": None,
        "tree": None,
        "tracked_tree_clean": None,
        "tracked_diff_sha256": None,
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        result = {
            "commit": commit,
            "tree": tree,
            "tracked_tree_clean": not bool(status.strip()),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _git_repository_root(path: Path) -> Optional[Path]:
    try:
        text = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(Path(path)),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(text).resolve() if text else None


def _ichor_package_roots() -> List[Path]:
    """Locate installed package roots without assuming an editable checkout."""
    roots = set()
    for import_name in ("ichor.core", "ichor.hpc", "ichor.cli"):
        spec = importlib.util.find_spec(import_name)
        if spec is None or spec.origin is None:
            continue
        origin = Path(spec.origin).resolve()
        if origin.name == "__init__.py" and len(origin.parents) >= 2:
            roots.add(origin.parents[1])
        else:
            roots.add(origin.parent)
    return sorted(roots, key=lambda value: str(value))


def _distribution_version(name: str) -> Optional[str]:
    candidates = [name, name.replace("_", "-"), name.replace("-", "_")]
    for candidate in candidates:
        try:
            return str(importlib.metadata.version(candidate))
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _module_identity(name: str) -> Dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        spec = None
    path = None if spec is None else spec.origin
    file_path = None if path in (None, "built-in") else Path(str(path)).resolve()
    package_roots = []
    if spec is not None and spec.submodule_search_locations:
        package_roots = [Path(value).resolve() for value in spec.submodule_search_locations]
    return {
        "module": name,
        "path": None if file_path is None else str(file_path),
        "sha256": None if file_path is None else _sha256_file(file_path),
        "package_tree_sha256": (
            None if not package_roots else _tree_hash(package_roots)
        ),
        "version": _distribution_version(name),
    }


def _json_safe_probe(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _ariadne_identity() -> Dict[str, Any]:
    identity = _module_identity("ariadne")
    identity["import_ok"] = False
    identity["abi_probe"] = None
    identity["abi_probe_error"] = None
    try:
        module = importlib.import_module("ariadne")
        identity["import_ok"] = True
        from .acquisition.ariadne_abi import probe_ariadne_module

        identity["abi_probe"] = _json_safe_probe(probe_ariadne_module(module))
    except Exception as exc:
        identity["abi_probe_error"] = type(exc).__name__ + ": " + str(exc)
    return identity


def _active_profile_identity() -> Dict[str, Any]:
    machine = os.environ.get("ICHOR_MACHINE")
    profile = None
    try:
        global_variables = importlib.import_module("ichor.hpc.global_variables")
        machine = getattr(global_variables, "MACHINE", None) or machine
        config = getattr(global_variables, "ICHOR_CONFIG", None)
        if isinstance(config, dict) and machine in config:
            profile = config[machine]
    except Exception:
        profile = None
    digest = None
    if isinstance(profile, dict):
        digest = hashlib.sha256(
            json.dumps(
                profile,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    software = profile.get("software", {}) if isinstance(profile, dict) else {}
    python_modules = (
        software.get("python", {}).get("modules", [])
        if isinstance(software.get("python", {}), dict)
        else []
    )
    ariadne_modules = (
        software.get("ariadne_runtime", {}).get("modules", [])
        if isinstance(software.get("ariadne_runtime", {}), dict)
        else []
    )
    return {
        "name": machine,
        "digest_sha256": digest,
        "module_sequence": {
            "purge_first": True,
            "python_modules": list(python_modules or []),
            "ariadne_runtime_modules": list(ariadne_modules or []),
        },
    }


def _installed_dependencies() -> list[Dict[str, str]]:
    rows = []
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip()
        if name:
            rows.append({"name": name, "version": str(distribution.version)})
    return sorted(rows, key=lambda row: (row["name"].lower(), row["version"]))


def _profile_ferebus_executable() -> Optional[str]:
    try:
        from .daemon.cluster_profile import expanded_profile_value

        value = expanded_profile_value(
            "software",
            "ferebus",
            "executable_path",
            default=None,
        )
    except Exception:
        value = None
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _configured_ferebus_identity() -> Dict[str, Any]:
    configured = os.environ.get("FEREBUS_PATH") or _profile_ferebus_executable()
    resolved = Path(configured).expanduser() if configured else None
    if resolved is None:
        located = shutil.which("ferebus")
        resolved = Path(located) if located else None
    return {
        "path": None if resolved is None else str(resolved.resolve()),
        "sha256": None if resolved is None else _sha256_file(resolved.resolve()),
    }


def capture_environment_generation(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    config: CampaignConfig,
    generation: int = 0,
) -> Dict[str, Any]:
    """Capture the exact Python/package/native identity for one generation."""
    campaign = Path(campaign_dir).resolve()
    package_roots = _ichor_package_roots()
    repo_root = _git_repository_root(Path(__file__).resolve().parent)
    git_identity = (
        _git_identity(repo_root)
        if repo_root is not None
        else _git_identity(Path(__file__).resolve().parent)
    )
    payload: Dict[str, Any] = {
        "schema_version": ENVIRONMENT_GENERATION_SCHEMA_VERSION,
        "generation": int(generation),
        "campaign_uid": str(campaign_uid),
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "operator": os.environ.get("USER") or os.environ.get("USERNAME") or "",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "ichor_git": git_identity,
        "ichor_package_tree_sha256": _tree_hash(package_roots),
        "dependencies": _installed_dependencies(),
        "pyferebus": _module_identity("pyferebus"),
        "ariadne": _ariadne_identity(),
        "ferebus_executable": _configured_ferebus_identity(),
        "machine_profile": _active_profile_identity(),
        "loaded_modules": [
            value for value in os.environ.get("LOADEDMODULES", "").split(":") if value
        ],
        "native_library_paths": {
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "LIBRARY_PATH": os.environ.get("LIBRARY_PATH", ""),
        },
        "campaign_schema_version": int(config.schema_version),
        "campaign_config_sha256": config_fingerprint(config.to_dict()),
        "config_lock_sha256": _sha256_file(
            campaign / ".DATA" / "ACTIVE_LEARNING" / "config_lock.json"
        ),
        "campaign_dir": str(campaign),
    }
    payload["environment_fingerprint_sha256"] = _environment_fingerprint(payload)
    payload["digest_sha256"] = _canonical_digest(payload)
    return payload


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExecutionIdentityError(label + " is not a regular file: " + str(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExecutionIdentityError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(value, dict):
        raise ExecutionIdentityError(label + " must contain a JSON object")
    return value


def _validate_generation_payload(
    payload: Dict[str, Any],
    *,
    expected_generation: int,
    expected_campaign_uid: str,
    path: Path,
) -> Dict[str, Any]:
    _require_exact_keys(
        payload,
        _ENVIRONMENT_GENERATION_KEYS,
        "environment generation",
    )
    if payload.get("schema_version") != ENVIRONMENT_GENERATION_SCHEMA_VERSION:
        raise ExecutionIdentityError(
            "unsupported environment generation schema: " + str(path)
        )
    generation = _exact_non_negative_int(
        payload.get("generation"), "environment generation"
    )
    if generation != int(expected_generation):
        raise ExecutionIdentityError("environment generation number mismatch")
    campaign_uid = _non_empty_string(
        payload.get("campaign_uid"),
        "environment generation campaign UID",
    )
    if campaign_uid != str(expected_campaign_uid):
        raise ExecutionIdentityError("environment generation campaign UID mismatch")
    _validate_timestamp(
        payload.get("created_at_iso"),
        "environment generation creation time",
    )
    for key in ("host", "operator"):
        if not isinstance(payload.get(key), str):
            raise ExecutionIdentityError(
                "environment generation " + key + " must be a string"
            )
    python_executable = Path(
        _non_empty_string(
            payload.get("python_executable"),
            "environment generation Python executable",
        )
    )
    if not python_executable.is_absolute():
        raise ExecutionIdentityError(
            "environment generation Python executable must be absolute"
        )
    _non_empty_string(
        payload.get("python_version"),
        "environment generation Python version",
    )
    _sha256_digest(
        payload.get("ichor_package_tree_sha256"),
        "ICHOR package-tree digest",
    )
    for key in (
        "ichor_git",
        "pyferebus",
        "ariadne",
        "ferebus_executable",
        "machine_profile",
    ):
        if not isinstance(payload.get(key), Mapping):
            raise ExecutionIdentityError(
                "environment generation " + key + " must be an object"
            )
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("name"), str)
        or not row["name"]
        or not isinstance(row.get("version"), str)
        for row in dependencies
    ):
        raise ExecutionIdentityError(
            "environment generation dependencies must contain name/version objects"
        )
    loaded_modules = payload.get("loaded_modules")
    if not isinstance(loaded_modules, list) or any(
        not isinstance(module, str) or not module for module in loaded_modules
    ):
        raise ExecutionIdentityError(
            "environment generation loaded_modules must be a string list"
        )
    native_paths = payload.get("native_library_paths")
    if not isinstance(native_paths, Mapping) or set(native_paths) != {
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
    } or any(not isinstance(value, str) for value in native_paths.values()):
        raise ExecutionIdentityError(
            "environment generation native-library paths are invalid"
        )
    if _exact_non_negative_int(
        payload.get("campaign_schema_version"),
        "environment generation campaign schema",
    ) < 1:
        raise ExecutionIdentityError(
            "environment generation campaign schema must be >= 1"
        )
    _sha256_digest(
        payload.get("campaign_config_sha256"),
        "environment generation campaign-config digest",
    )
    config_lock_digest = payload.get("config_lock_sha256")
    if config_lock_digest is not None:
        _sha256_digest(config_lock_digest, "environment generation config-lock digest")
    campaign_dir = Path(
        _non_empty_string(
            payload.get("campaign_dir"),
            "environment generation campaign directory",
        )
    )
    if not campaign_dir.is_absolute():
        raise ExecutionIdentityError(
            "environment generation campaign directory must be absolute"
        )
    recorded_fingerprint = _sha256_digest(
        payload.get("environment_fingerprint_sha256"),
        "environment fingerprint",
    )
    if recorded_fingerprint != _environment_fingerprint(payload):
        raise ExecutionIdentityError("environment generation fingerprint mismatch")
    recorded_digest = _sha256_digest(
        payload.get("digest_sha256"), "environment generation digest"
    )
    if recorded_digest != _canonical_digest(payload):
        raise ExecutionIdentityError("environment generation digest mismatch")
    return payload


def read_active_environment_generation(
    campaign_dir: Union[str, Path],
    *,
    expected_campaign_uid: str,
) -> Dict[str, Any]:
    """Read and fully validate the active generation and its pointer."""
    campaign = Path(campaign_dir).resolve()
    current_path = environment_current_path(campaign)
    current = _read_json_object(current_path, "active environment pointer")
    _require_exact_keys(
        current,
        frozenset({
            "schema_version",
            "generation",
            "generation_path",
            "generation_digest_sha256",
        }),
        "active environment pointer",
    )
    if current.get("schema_version") != ENVIRONMENT_CURRENT_SCHEMA_VERSION:
        raise ExecutionIdentityError("unsupported active environment pointer schema")
    generation = _exact_non_negative_int(
        current.get("generation"), "active environment generation"
    )
    expected_relative = (
        Path(".DATA")
        / "ACTIVE_LEARNING"
        / "environment_generations"
        / ("generation-" + str(generation).zfill(6) + ".json")
    )
    recorded_relative = current.get("generation_path")
    if not isinstance(recorded_relative, str) or not recorded_relative:
        raise ExecutionIdentityError("active environment generation path is missing")
    if Path(recorded_relative) != expected_relative:
        raise ExecutionIdentityError("active environment generation path is not canonical")
    generation_path = campaign_owned_path(campaign, campaign / recorded_relative)
    payload = _validate_generation_payload(
        _read_json_object(generation_path, "environment generation"),
        expected_generation=generation,
        expected_campaign_uid=str(expected_campaign_uid),
        path=generation_path,
    )
    pointer_digest = _sha256_digest(
        current.get("generation_digest_sha256"),
        "active environment generation digest",
    )
    if pointer_digest != payload["digest_sha256"]:
        raise ExecutionIdentityError("active environment pointer digest mismatch")
    return {
        "pointer": current,
        "generation": payload,
        "generation_path": generation_path,
    }


def read_environment_generation(
    campaign_dir: Union[str, Path],
    *,
    generation: int,
    expected_campaign_uid: str,
) -> Dict[str, Any]:
    """Read one immutable historical environment generation by number."""
    campaign = Path(campaign_dir).resolve()
    generation_number = _exact_non_negative_int(
        generation, "environment generation"
    )
    path = campaign_owned_path(
        campaign,
        environment_generations_dir(campaign)
        / ("generation-" + str(generation_number).zfill(6) + ".json"),
    )
    return _validate_generation_payload(
        _read_json_object(path, "environment generation"),
        expected_generation=generation_number,
        expected_campaign_uid=str(expected_campaign_uid),
        path=path,
    )


def environment_status(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    config: CampaignConfig,
) -> Dict[str, Any]:
    """Compare the current process and native backends with the active generation."""
    active = read_active_environment_generation(
        campaign_dir,
        expected_campaign_uid=str(campaign_uid),
    )
    generation = active["generation"]
    observed = capture_environment_generation(
        campaign_dir,
        campaign_uid=str(campaign_uid),
        config=config,
        generation=int(generation["generation"]),
    )
    expected_fields = _environment_fingerprint_payload(generation)
    observed_fields = _environment_fingerprint_payload(observed)
    changed_fields: List[str] = [
        key
        for key in _ENVIRONMENT_FINGERPRINT_KEYS
        if expected_fields.get(key) != observed_fields.get(key)
    ]
    matches = (
        generation["environment_fingerprint_sha256"]
        == observed["environment_fingerprint_sha256"]
    )
    return {
        "schema_version": 1,
        "campaign_uid": str(campaign_uid),
        "generation": int(generation["generation"]),
        "generation_digest_sha256": str(generation["digest_sha256"]),
        "expected_environment_fingerprint_sha256": str(
            generation["environment_fingerprint_sha256"]
        ),
        "observed_environment_fingerprint_sha256": str(
            observed["environment_fingerprint_sha256"]
        ),
        "matches": bool(matches),
        "changed_fields": changed_fields,
        "active_generation_path": str(active["generation_path"]),
    }


def assert_environment_unchanged(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    config: CampaignConfig,
) -> Dict[str, Any]:
    status = environment_status(
        campaign_dir,
        campaign_uid=str(campaign_uid),
        config=config,
    )
    if not bool(status["matches"]):
        fields = ", ".join(status["changed_fields"]) or "unknown fields"
        raise ExecutionIdentityError(
            "active execution environment has drifted: " + fields
        )
    return status


def read_execution_identity(
    campaign_dir: Union[str, Path],
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    path = execution_identity_path(campaign_dir)
    payload = _read_json_object(path, "execution identity")
    _require_exact_keys(payload, _EXECUTION_IDENTITY_KEYS, "execution identity")
    if payload.get("schema_version") != EXECUTION_IDENTITY_SCHEMA_VERSION:
        raise ExecutionIdentityError("unsupported execution identity schema")
    if str(payload.get("digest_sha256") or "") != _canonical_digest(payload):
        raise ExecutionIdentityError("execution identity digest mismatch")
    campaign_uid = _non_empty_string(
        payload.get("campaign_uid"),
        "execution identity campaign UID",
    )
    if expected_campaign_uid is not None and campaign_uid != str(expected_campaign_uid):
        raise ExecutionIdentityError("execution identity campaign UID mismatch")
    if str(payload.get("mode") or "") not in VALID_EXECUTION_MODES:
        raise ExecutionIdentityError("execution identity mode is invalid")
    _exact_non_negative_int(
        payload.get("campaign_random_seed"),
        "execution identity random seed",
    )
    if _exact_non_negative_int(
        payload.get("campaign_schema_version"),
        "execution identity campaign schema",
    ) < 1:
        raise ExecutionIdentityError("execution identity campaign schema must be >= 1")
    _sha256_digest(
        payload.get("initial_config_sha256"),
        "execution identity initial-config digest",
    )
    if _exact_non_negative_int(
        payload.get("environment_generation"),
        "execution identity initial environment generation",
    ) != 0:
        raise ExecutionIdentityError(
            "execution identity initial environment generation must be zero"
        )
    initial_generation_digest = _sha256_digest(
        payload.get("environment_generation_digest_sha256"),
        "execution identity initial environment digest",
    )
    _validate_timestamp(
        payload.get("created_at_iso"),
        "execution identity creation time",
    )
    initial_generation_path = environment_generations_dir(campaign_dir) / (
        "generation-000000.json"
    )
    initial_generation = _validate_generation_payload(
        _read_json_object(initial_generation_path, "initial environment generation"),
        expected_generation=0,
        expected_campaign_uid=campaign_uid,
        path=initial_generation_path,
    )
    if initial_generation_digest != initial_generation["digest_sha256"]:
        raise ExecutionIdentityError(
            "execution identity initial environment digest mismatch"
        )
    return payload


def ensure_execution_identity(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    config: CampaignConfig,
    requested_mode: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """Create the first identity or require an exact mode match thereafter."""
    campaign = Path(campaign_dir).resolve()
    path = execution_identity_path(campaign)
    if path.is_file() or path.is_symlink():
        payload = read_execution_identity(
            campaign,
            expected_campaign_uid=str(campaign_uid),
        )
        stored_mode = str(payload.get("mode") or "")
        if payload.get("campaign_random_seed") != int(
            config.campaign.reproducibility_seed
        ):
            raise ExecutionIdentityError(
                "campaign.reproducibility_seed differs from the immutable "
                "execution identity"
            )
        if payload.get("campaign_schema_version") != int(config.schema_version):
            raise ExecutionIdentityError(
                "campaign schema differs from the immutable execution identity"
            )
        if requested_mode is not None and requested_mode != stored_mode:
            raise ExecutionIdentityError(
                "campaign execution mode is permanently bound to "
                + stored_mode
                + "; requested "
                + requested_mode
            )
        return stored_mode, payload

    if requested_mode not in VALID_EXECUTION_MODES:
        raise ExecutionIdentityError(
            "the first start requires --mode live or --mode dry_run"
        )
    generations = environment_generations_dir(campaign)
    generations.mkdir(parents=True, exist_ok=True)
    generation = capture_environment_generation(
        campaign,
        campaign_uid=str(campaign_uid),
        config=config,
        generation=0,
    )
    generation_path = generations / "generation-000000.json"
    atomic_write_json(generation_path, generation)
    current = {
        "schema_version": ENVIRONMENT_CURRENT_SCHEMA_VERSION,
        "generation": 0,
        "generation_path": str(generation_path.relative_to(campaign)),
        "generation_digest_sha256": str(generation["digest_sha256"]),
    }
    atomic_write_json(environment_current_path(campaign), current)
    payload = {
        "schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
        "campaign_uid": str(campaign_uid),
        "mode": str(requested_mode),
        "campaign_random_seed": int(config.campaign.reproducibility_seed),
        "campaign_schema_version": int(config.schema_version),
        "initial_config_sha256": config_fingerprint(config.to_dict()),
        "environment_generation": 0,
        "environment_generation_digest_sha256": str(generation["digest_sha256"]),
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    payload["digest_sha256"] = _canonical_digest(payload)
    atomic_write_json(path, payload)
    return str(requested_mode), payload


def advance_environment_generation(
    campaign_dir: Union[str, Path],
    *,
    config: CampaignConfig,
    live_preflight_ok: bool = False,
    scheduler_ownership_clear: bool = False,
) -> Dict[str, Any]:
    """Advance a drifted environment at a verified safe transition boundary.

    Generation publication is ordered so interruption remains fail-closed:
    write the immutable generation, clear derived reference scales in state,
    then advance the current pointer.  Repeating the command after any partial
    attempt is safe.  An unpublished, valid REFERENCE_COMMIT transaction is a
    recovery boundary because it is local, resumable and owns no scheduler job.
    A fully retryable ARIADNE array is safe because no completed task output
    crosses the boundary.  A single-generation, all-complete array is safe when
    only its derived batch publication remains to be rebuilt.
    """
    campaign = Path(campaign_dir).resolve()
    state = read_state(operational_path(campaign, "state.json"))
    identity = read_execution_identity(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    active_generation = active["generation"]
    candidate = capture_environment_generation(
        campaign,
        campaign_uid=str(state.campaign_uid),
        config=config,
        generation=int(active_generation["generation"]) + 1,
    )
    if (
        candidate["environment_fingerprint_sha256"]
        == active_generation["environment_fingerprint_sha256"]
    ):
        return {
            "schema_version": 1,
            "changed": False,
            "generation": int(active_generation["generation"]),
            "generation_digest_sha256": str(active_generation["digest_sha256"]),
            "message": "active environment already matches the current process",
        }
    if identity["mode"] == "live" and not bool(live_preflight_ok):
        raise ExecutionIdentityError(
            "automatic live environment transition requires a successful backend preflight"
        )
    transition_context: Dict[str, Any] = {
        "transition_kind": "idle_boundary",
    }
    phase_b_transition_pending = False
    aimall_transition_pending = False
    gaussian_transition_pending = False
    allocation_check_transition_pending = False
    scheduler_cancel_transition = _scheduler_cancellation_transition_boundary(
        campaign,
        state,
    )
    if scheduler_cancel_transition is not None:
        transition_context = scheduler_cancel_transition
    elif state.phase is CampaignPhase.REFERENCE_COMMIT:
        from .daemon.reference_commit import classify_reference_commit

        context = "bootstrap" if int(state.iteration) == 0 else "active"
        reference_commit_recovery = classify_reference_commit(
            campaign,
            context=context,
            iteration=int(state.iteration),
            verification="authority",
        )
        recovery_state = str(reference_commit_recovery.get("state") or "")
        if recovery_state not in _REBINDABLE_REFERENCE_COMMIT_STATES:
            detail = str(reference_commit_recovery.get("reason") or recovery_state)
            raise ExecutionIdentityError(
                "environment transition at REFERENCE_COMMIT requires a valid "
                "unpublished recovery transaction; observed "
                + detail
            )
        ledger = reference_commit_recovery.get("ledger")
        if not isinstance(ledger, Mapping):
            raise ExecutionIdentityError(
                "environment transition at REFERENCE_COMMIT lacks a valid transaction ledger"
            )
        expected_reference_version = int(state.iteration) - 1
        observed_identity = (
            str(ledger.get("campaign_uid") or ""),
            int(ledger.get("iteration", -1)),
            int(ledger.get("reference_data_version", -1)),
            str(ledger.get("context") or ""),
            int(state.reference_data_version),
        )
        expected_identity = (
            str(state.campaign_uid),
            int(state.iteration),
            int(state.iteration),
            context,
            expected_reference_version,
        )
        if observed_identity != expected_identity:
            raise ExecutionIdentityError(
                "environment transition at REFERENCE_COMMIT found inconsistent "
                "campaign, iteration, or reference-version identity"
            )
        transition_context = {"transition_kind": "reference_commit_recovery"}
    elif state.phase in {CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS}:
        _validate_ferebus_transition_boundary(campaign, state)
        transition_context = {"transition_kind": "ferebus_retry"}
    elif state.phase is CampaignPhase.ARIADNE_ARRAY:
        transition_context = _validate_ariadne_retry_transition_boundary(
            campaign,
            state,
        )
    elif state.phase in {
        CampaignPhase.INITIAL_ALLOCATION_CHECK,
        CampaignPhase.ALLOCATION_CHECK,
    }:
        allocation_check_transition_pending = True
    elif state.phase in {
        CampaignPhase.INITIAL_GAUSSIAN,
        CampaignPhase.GAUSSIAN,
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
        CampaignPhase.REPLACEMENT_GAUSSIAN,
    }:
        from .daemon.submission_intent import (
            gaussian_intent_claims_completed_array,
            load_intent,
        )

        gaussian_intent = load_intent(
            campaign,
            state.phase.value,
            int(state.iteration),
        )
        if not (
            isinstance(gaussian_intent, Mapping)
            and gaussian_intent_claims_completed_array(gaussian_intent)
        ):
            raise ExecutionIdentityError(
                "environment transition requires an idle SEED_SELECT, "
                "STOP_CHECK or DONE boundary"
            )
        gaussian_transition_pending = True
    elif state.phase in {
        CampaignPhase.INITIAL_AIMALL,
        CampaignPhase.AIMALL,
        CampaignPhase.INITIAL_REPLACEMENT_AIMALL,
        CampaignPhase.REPLACEMENT_AIMALL,
    }:
        from .daemon.submission_intent import (
            aimall_intent_claims_completed_array,
            load_intent,
        )

        aimall_intent = load_intent(
            campaign,
            state.phase.value,
            int(state.iteration),
        )
        if not (
            isinstance(aimall_intent, Mapping)
            and aimall_intent_claims_completed_array(aimall_intent)
        ):
            raise ExecutionIdentityError(
                "environment transition requires an idle SEED_SELECT, "
                "STOP_CHECK or DONE boundary"
            )
        aimall_transition_pending = True
    elif state.phase is CampaignPhase.PHASE_B_DIVERSITY:
        phase_b_transition_pending = True
    elif state.phase not in {
        CampaignPhase.SEED_SELECT,
        CampaignPhase.STOP_CHECK,
        CampaignPhase.DONE,
    }:
        raise ExecutionIdentityError(
            "environment transition requires an idle SEED_SELECT, STOP_CHECK or DONE "
            "boundary, "
            "a verified unpublished REFERENCE_COMMIT recovery transaction, or a "
            "clean pre-submission FEREBUS retry boundary, or a fully retryable "
            "ARIADNE retry or postprocess-only recovery boundary, a "
            "scheduler-complete AIMAll postprocess boundary, or a clean "
            "scheduler-complete Gaussian postprocess boundary, or a clean "
            "pre-submission Phase B recovery boundary, or a verified "
            "allocation-check recovery boundary"
        )
    if any(value is not None for value in state.pending_jobs.values()):
        raise ExecutionIdentityError(
            "environment transition is blocked by pending scheduler ownership"
        )
    if not bool(scheduler_ownership_clear):
        raise ExecutionIdentityError(
            "environment transition requires a conclusive scheduler ownership check"
        )

    from .daemon.artifact_contracts import verify_state_referenced_artifacts
    from .daemon.config_lock import review_config_changes
    from .daemon.submission_intent import ACTIVE_STATUSES, inventory_intents
    from .daemon.reconcile_transaction import (
        inspect_reconcile_transaction_recovery,
    )

    review = review_config_changes(
        campaign,
        config,
        state,
        initialise_missing=False,
    )
    if review.changed:
        raise ExecutionIdentityError(
            "campaign configuration differs from its lock; reconcile it before restart"
        )
    inventory = inventory_intents(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    if inventory["errors"]:
        raise ExecutionIdentityError(
            "environment transition is blocked by malformed submission intents"
        )
    active_intents = [
        record
        for record in inventory["records"]
        if str(record.get("status")) in ACTIVE_STATUSES
    ]
    if active_intents:
        raise ExecutionIdentityError(
            "environment transition is blocked by active submission intents"
        )
    from .daemon.artifact_snapshot import build_committed_artifact_snapshot

    snapshot = build_committed_artifact_snapshot(
        campaign,
        verification_level="authority",
    )
    transaction_recovery = inspect_reconcile_transaction_recovery(
        campaign,
        artifact_snapshot=snapshot,
    )
    if str(transaction_recovery.get("state") or "") != "none":
        if bool(transaction_recovery.get("recoverable", False)):
            raise ExecutionIdentityError(
                "environment transition is waiting for ordinary reconcile to recover "
                "an interrupted transaction"
            )
        raise ExecutionIdentityError(
            "environment transition is blocked by ambiguous reconcile transaction evidence"
        )
    verify_state_referenced_artifacts(
        campaign,
        state,
        strict_models=True,
        verification="authority",
        snapshot=snapshot,
    )
    if phase_b_transition_pending:
        transition_context = _validate_phase_b_transition_boundary(
            campaign,
            state,
            intent_records=inventory["records"],
        )
    if aimall_transition_pending:
        transition_context = _validate_aimall_postprocess_transition_boundary(
            campaign,
            state,
        )
    if gaussian_transition_pending:
        transition_context = _validate_gaussian_postprocess_transition_boundary(
            campaign,
            state,
        )
    if allocation_check_transition_pending:
        transition_context = _validate_allocation_check_transition_boundary(
            campaign,
            state,
        )

    generation_number = int(candidate["generation"])
    generations_root = environment_generations_dir(campaign)
    generations_root.mkdir(parents=True, exist_ok=True)
    if scheduler_cancel_transition is not None:
        refreshed_transition = _scheduler_cancellation_transition_boundary(
            campaign,
            state,
        )
        if refreshed_transition != scheduler_cancel_transition:
            raise ExecutionIdentityError(
                "scheduler-cancellation recovery evidence changed during "
                "environment transition"
            )
    if phase_b_transition_pending:
        refreshed_intents = inventory_intents(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        if refreshed_intents["errors"] or any(
            str(record.get("status")) in ACTIVE_STATUSES
            for record in refreshed_intents["records"]
        ):
            raise ExecutionIdentityError(
                "PHASE_B_DIVERSITY environment transition found submission-intent "
                "ownership after its safety check"
            )
        transition_context = _validate_phase_b_transition_boundary(
            campaign,
            state,
            intent_records=refreshed_intents["records"],
        )
    if aimall_transition_pending:
        refreshed_intents = inventory_intents(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        if refreshed_intents["errors"] or any(
            str(record.get("status")) in ACTIVE_STATUSES
            for record in refreshed_intents["records"]
        ):
            raise ExecutionIdentityError(
                "AIMAll postprocess environment transition found submission-intent "
                "ownership after its safety check"
            )
        transition_context = _validate_aimall_postprocess_transition_boundary(
            campaign,
            state,
        )
    if gaussian_transition_pending:
        refreshed_intents = inventory_intents(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        if refreshed_intents["errors"] or any(
            str(record.get("status")) in ACTIVE_STATUSES
            for record in refreshed_intents["records"]
        ):
            raise ExecutionIdentityError(
                "Gaussian postprocess environment transition found submission-intent "
                "ownership after its safety check"
            )
        transition_context = _validate_gaussian_postprocess_transition_boundary(
            campaign,
            state,
        )
    if allocation_check_transition_pending:
        refreshed_intents = inventory_intents(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        if refreshed_intents["errors"] or any(
            str(record.get("status")) in ACTIVE_STATUSES
            for record in refreshed_intents["records"]
        ):
            raise ExecutionIdentityError(
                "allocation-check environment transition found submission-intent "
                "ownership after its safety check"
            )
        transition_context = _validate_allocation_check_transition_boundary(
            campaign,
            state,
        )
    while True:
        generation_path = generations_root / (
            "generation-" + str(generation_number).zfill(6) + ".json"
        )
        if not generation_path.exists() and not generation_path.is_symlink():
            atomic_write_json(generation_path, candidate)
            break
        existing = _validate_generation_payload(
            _read_json_object(generation_path, "environment generation"),
            expected_generation=generation_number,
            expected_campaign_uid=str(state.campaign_uid),
            path=generation_path,
        )
        if (
            existing["environment_fingerprint_sha256"]
            == candidate["environment_fingerprint_sha256"]
        ):
            candidate = existing
            break
        generation_number += 1
        candidate = capture_environment_generation(
            campaign,
            campaign_uid=str(state.campaign_uid),
            config=config,
            generation=generation_number,
        )

    state.reference_scales = None
    state.reference_scales_iteration = -1
    state.reference_scales_models_version = -1
    state.reference_scales_model_manifest_sha256 = None
    write_state(operational_path(campaign, "state.json"), state)

    current = {
        "schema_version": ENVIRONMENT_CURRENT_SCHEMA_VERSION,
        "generation": generation_number,
        "generation_path": str(generation_path.relative_to(campaign)),
        "generation_digest_sha256": str(candidate["digest_sha256"]),
    }
    atomic_write_json(environment_current_path(campaign), current)
    try:
        from .daemon.journal import append_event

        journal_transition_context = {
            key: value
            for key, value in transition_context.items()
            if key != "postprocess_source"
        }
        append_event(
            operational_path(campaign, "journal.ndjson"),
            "environment_generation_advanced",
            max_bytes=int(config.runtime.journal_max_bytes),
            retained_files=int(config.runtime.journal_retained_files),
            lock_timeout_seconds=int(config.runtime.ledger_lock_timeout_seconds),
            previous_generation=int(active_generation["generation"]),
            generation=generation_number,
            previous_generation_digest_sha256=str(active_generation["digest_sha256"]),
            generation_digest_sha256=str(candidate["digest_sha256"]),
            phase=state.phase.value,
            iteration=int(state.iteration),
            changed_fields=[
                key
                for key in _ENVIRONMENT_FINGERPRINT_KEYS
                if active_generation.get(key) != candidate.get(key)
            ],
            **journal_transition_context,
        )
    except Exception:
        pass
    return {
        "schema_version": 1,
        "changed": True,
        "previous_generation": int(active_generation["generation"]),
        "generation": generation_number,
        "generation_digest_sha256": str(candidate["digest_sha256"]),
        "generation_path": str(generation_path),
        **transition_context,
    }


def rebind_environment(
    campaign_dir: Union[str, Path],
    *,
    config: CampaignConfig,
    live_preflight_ok: bool = False,
    scheduler_ownership_clear: bool = False,
) -> Dict[str, Any]:
    """Compatibility wrapper for internal callers; no CLI command exposes it."""
    return advance_environment_generation(
        campaign_dir,
        config=config,
        live_preflight_ok=live_preflight_ok,
        scheduler_ownership_clear=scheduler_ownership_clear,
    )


__all__ = [
    "ENVIRONMENT_GENERATION_SCHEMA_VERSION",
    "EXECUTION_IDENTITY_SCHEMA_VERSION",
    "ExecutionIdentityError",
    "VALID_EXECUTION_MODES",
    "assert_environment_unchanged",
    "advance_environment_generation",
    "capture_environment_generation",
    "ensure_execution_identity",
    "environment_current_path",
    "environment_generations_dir",
    "execution_identity_path",
    "inspect_allocation_check_transition_boundary",
    "read_active_environment_generation",
    "read_execution_identity",
]
