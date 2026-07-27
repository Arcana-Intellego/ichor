"""Phase-aware recovery contracts for active-learning reconcile.

The daemon state machine is intentionally simple, but recovery is only safe
when the selected re-entry phase has the producer artefacts that phase
consumes.  These helpers keep that phase-specific knowledge out of the CLI
and out of the broad committed-version checks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..strict_json import strict_json as json
from ..acquisition.trajectory_pool import TrajectoryPool
from ..versioning.reference_data import ReferenceDataVersioning
from ..handoff_manifests import (
    read_ariadne_results_manifest,
    read_phase_a_sample_manifest,
)
from ..layout import active_learning_dir
from . import input_staging as _stg
from .artifact_contracts import (
    verify_committed_model_version,
    verify_committed_reference_data_version,
)
from .state import CampaignPhase, CampaignState


class RecoveryContractError(RuntimeError):
    """Raised when a proposed recovery phase lacks its input handoff."""


@dataclass(frozen=True)
class RecoveryDecision:
    phase: CampaignPhase
    iteration: int
    reason: str
    trusted_artifact: Optional[str] = None
    replacement_round: int = 0


@dataclass(frozen=True)
class RecoveryHandoff:
    decision: RecoveryDecision
    priority: int
    kind: str


def iteration_dir(campaign_dir: Union[str, Path], iteration: int) -> Path:
    from ..layout import active_iteration_dir

    return active_iteration_dir(campaign_dir, int(iteration))


def active_iteration_committed(state: CampaignState, iteration: int) -> bool:
    """Return true when active iteration ``iteration`` is fully committed.

    Version 0 is bootstrap. Active iteration ``i`` commits reference-data and
    model version ``i``.
    """
    try:
        reference_data_version = int(getattr(state, "reference_data_version", -1))
        models_version = int(getattr(state, "models_version", -1))
    except (TypeError, ValueError):
        return False
    return min(reference_data_version, models_version) >= int(iteration)


def active_iteration_reference_data_committed(
    state: CampaignState,
    iteration: int,
) -> bool:
    """Return true once REFERENCE_COMMIT has published this iteration's QM delta."""
    try:
        reference_data_version = int(
            getattr(state, "reference_data_version", -1)
        )
    except (TypeError, ValueError):
        return False
    return reference_data_version >= int(iteration)


def _has_version(versions: Sequence[int], version: int) -> bool:
    return int(version) in {int(v) for v in versions}


def _ok(func, *args, **kwargs) -> bool:
    try:
        func(*args, **kwargs)
        return True
    except Exception:
        return False


def _error(func, *args, **kwargs) -> Optional[str]:
    try:
        func(*args, **kwargs)
        return None
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)[:220]


def _require_pool(campaign: Path, *, verification: str = "metadata") -> None:
    if verification == "authority":
        from ..acquisition.trajectory_pool import (
            POOL_MANIFEST_FILENAME,
            POOL_SUBDIR,
            TrajectoryPoolManifest,
        )

        manifest_path = campaign / POOL_SUBDIR / POOL_MANIFEST_FILENAME
        manifest = TrajectoryPoolManifest.from_dict(
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                source=manifest_path,
            )
        )
    elif verification in {"metadata", "deep"}:
        manifest = TrajectoryPool.load(campaign).manifest
    else:
        raise RecoveryContractError("pool verification level is invalid")
    if manifest.natoms <= 0:
        raise RecoveryContractError("trajectory pool atom count is not positive")


def _require_phase_a(campaign: Path, *, verification: str = "metadata") -> None:
    from ..layout import bootstrap_selection_dir

    root = bootstrap_selection_dir(campaign)
    if verification == "authority":
        from ..handoff_manifests import (
            PHASE_A_SAMPLE_SCHEMA_VERSION,
            phase_a_sample_manifest_path,
        )
        from ..sampling.diversity_contract import selector_contract_matches

        path = phase_a_sample_manifest_path(root)
        payload = json.loads(path.read_text(encoding="utf-8"), source=path)
        if not isinstance(payload, dict):
            raise RecoveryContractError("Phase A sample manifest must be an object")
        n_select = payload.get("n_select")
        if (
            payload.get("schema_version") != PHASE_A_SAMPLE_SCHEMA_VERSION
            or payload.get("phase") != CampaignPhase.PHASE_A_DIVERSITY.value
            or payload.get("iteration") != 0
            or isinstance(n_select, bool)
            or not isinstance(n_select, int)
            or n_select <= 0
            or not selector_contract_matches(payload.get("selector"))
        ):
            raise RecoveryContractError("Phase A sample authority is invalid")
        return
    if verification not in {"metadata", "deep"}:
        raise RecoveryContractError("Phase A verification level is invalid")
    read_phase_a_sample_manifest(root, require_nonempty=True)


def _require_quantum_acceptance_authority(
    staging: Path,
    *,
    expected_phase: str,
    expected_iteration: int,
    require_nonempty: bool,
) -> None:
    from .input_staging import (
        QUANTUM_ACCEPTANCE_SCHEMA_VERSION,
        _validate_pointdir_basename,
        quantum_acceptance_manifest_path,
    )

    path = quantum_acceptance_manifest_path(
        staging,
        phase_name=expected_phase,
    )
    payload = json.loads(path.read_text(encoding="utf-8"), source=path)
    if not isinstance(payload, dict):
        raise RecoveryContractError("quantum acceptance manifest must be an object")
    accepted = payload.get("accepted_pointdirs")
    rejected = payload.get("rejected", [])
    total = payload.get("n_total")
    if (
        payload.get("schema_version") != QUANTUM_ACCEPTANCE_SCHEMA_VERSION
        or payload.get("phase") != expected_phase
        or payload.get("iteration") != int(expected_iteration)
        or not isinstance(accepted, list)
        or not isinstance(rejected, list)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total != len(accepted) + len(rejected)
        or (require_nonempty and not accepted)
    ):
        raise RecoveryContractError("quantum acceptance authority is invalid")
    names = [_validate_pointdir_basename(name) for name in accepted]
    for record in rejected:
        if not isinstance(record, dict) or set(record) != {"pointdir", "reason"}:
            raise RecoveryContractError("quantum rejection authority is invalid")
        names.append(_validate_pointdir_basename(record.get("pointdir")))
        if not isinstance(record.get("reason"), str) or not str(
            record.get("reason")
        ).strip():
            raise RecoveryContractError("quantum rejection reason is invalid")
    if len(names) != len(set(names)):
        raise RecoveryContractError("quantum acceptance dispositions are duplicated")


def _require_all_rejected_gaussian_coverage(
    campaign: Path,
    staging: Path,
    *,
    expected_phase: str,
    expected_iteration: int,
    context: str,
    replacement_round: int = 0,
    expected_campaign_uid: Optional[str] = None,
) -> None:
    """Bind a zero-accepted Gaussian publication to its allocation tasks."""
    from .input_staging import (
        _validate_pointdir_basename,
        quantum_acceptance_manifest_path,
    )
    from ..layout import staging_pointdir_name
    from ..point_allocation import point_allocation_path, read_point_allocation

    path = quantum_acceptance_manifest_path(
        staging,
        phase_name=expected_phase,
    )
    payload = json.loads(path.read_text(encoding="utf-8"), source=path)
    accepted = list(payload.get("accepted_pointdirs") or [])
    if accepted:
        return
    allocation_iteration = 0 if str(context) == "bootstrap" else int(
        expected_iteration
    )
    allocation = read_point_allocation(
        point_allocation_path(
            campaign,
            context=str(context),
            iteration=allocation_iteration,
        ),
        expected_campaign_uid=expected_campaign_uid,
        expected_context=str(context),
        expected_iteration=allocation_iteration,
    )
    if int(replacement_round) > 0:
        from ..replacement_sampling import (
            read_replacement_sample_strict,
            replacement_round_dir,
        )

        sample = read_replacement_sample_strict(
            campaign,
            context=str(context),
            iteration=allocation_iteration,
            replacement_round=int(replacement_round),
            expected_campaign_uid=str(allocation["campaign_uid"]),
        )
        expected_staging = replacement_round_dir(
            campaign,
            context=str(context),
            iteration=allocation_iteration,
            replacement_round=int(replacement_round),
        )
        if expected_staging.resolve(strict=False) != staging.resolve(strict=False):
            raise RecoveryContractError(
                "Gaussian replacement publication uses the wrong staging bucket"
            )
        expected_names = [
            staging_pointdir_name(int(record["pointdir_index"]))
            for record in sample["records"]
        ]
    else:
        expected_names = []
        for task_index, slot in enumerate(allocation["slots"]):
            attempts = [
                attempt
                for attempt in list(slot.get("attempts") or [])
                if int(attempt.get("round", -1)) == 0
            ]
            if len(attempts) != 1:
                raise RecoveryContractError(
                    "Gaussian allocation does not contain one primary task per slot"
                )
            expected_name = staging_pointdir_name(int(task_index))
            recorded_name = str(attempts[0].get("pointdir_name") or "")
            if recorded_name and (
                _validate_pointdir_basename(recorded_name) != expected_name
            ):
                raise RecoveryContractError(
                    "Gaussian allocation primary task identity is inconsistent"
                )
            expected_names.append(expected_name)
    rejected_names = [
        _validate_pointdir_basename(str(record.get("pointdir") or ""))
        for record in list(payload.get("rejected") or [])
        if isinstance(record, dict)
    ]
    if (
        int(payload.get("n_total", -1)) != len(expected_names)
        or rejected_names != expected_names
    ):
        raise RecoveryContractError(
            "all-rejected Gaussian publication does not exactly cover its "
            "allocation producer tasks"
        )


def _require_point_allocation(
    campaign: Path,
    *,
    context: str,
    iteration: int,
    expected_campaign_uid: Optional[str] = None,
    complete: Optional[bool] = None,
) -> Dict[str, Any]:
    from ..point_allocation import point_allocation_path, read_point_allocation

    path = point_allocation_path(
        campaign,
        context=str(context),
        iteration=int(iteration),
    )
    payload = read_point_allocation(
        path,
        expected_campaign_uid=expected_campaign_uid,
        expected_context=str(context),
        expected_iteration=int(iteration),
    )
    is_complete = bool((payload.get("summary") or {}).get("complete", False))
    if complete is not None and is_complete != bool(complete):
        raise RecoveryContractError(
            "point allocation is "
            + ("complete" if is_complete else "incomplete")
            + " but this phase requires it to be "
            + ("complete" if complete else "incomplete")
        )
    return payload


def _require_replacement_sample(
    campaign: Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
    expected_campaign_uid: Optional[str] = None,
) -> Path:
    from ..replacement_sampling import (
        read_replacement_sample_strict,
        replacement_round_dir,
    )

    path = replacement_round_dir(
        campaign,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
    )
    read_replacement_sample_strict(
        campaign,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        expected_campaign_uid=expected_campaign_uid,
    )
    return path


def _require_allocation_check_ready(
    campaign: Path,
    *,
    context: str,
    iteration: int,
    expected_replacement_round: int = 0,
    expected_campaign_uid: Optional[str] = None,
) -> None:
    from ..point_allocation import pending_attempts
    from ..replacement_sampling import inspect_replacement_sample_recovery

    payload = _require_point_allocation(
        campaign,
        context=str(context),
        iteration=int(iteration),
        expected_campaign_uid=expected_campaign_uid,
    )
    pending = pending_attempts(payload)
    if pending:
        rounds = {int(record.get("round", -1)) for record in pending}
        if rounds == {0}:
            raise RecoveryContractError(
                "primary QM outcomes have not yet been recorded in point allocation"
            )
        if len(rounds) != 1 or min(rounds) <= 0:
            raise RecoveryContractError(
                "point allocation has pending attempts from incompatible replacement rounds"
            )
        replacement_round = next(iter(rounds))
        if int(expected_replacement_round) != int(replacement_round):
            raise RecoveryContractError(
                "campaign replacement round does not match pending point allocation"
            )
        sample = inspect_replacement_sample_recovery(
            campaign,
            context=str(context),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            expected_campaign_uid=expected_campaign_uid,
        )
        if str(sample.get("state") or "") == "conflicting":
            raise RecoveryContractError(
                "replacement sample evidence conflicts with pending allocation: "
                + str(sample.get("reason") or "unknown conflict")
            )


def _require_replacement_gaussian_handoff(
    campaign: Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
    verification: str = "metadata",
    expected_campaign_uid: Optional[str] = None,
) -> None:
    round_dir = _require_replacement_sample(
        campaign,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        expected_campaign_uid=expected_campaign_uid,
    )
    phase = (
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN
        if context == "bootstrap"
        else CampaignPhase.REPLACEMENT_GAUSSIAN
    )
    if verification == "authority":
        _require_quantum_acceptance_authority(
            round_dir,
            expected_phase=phase.value,
            expected_iteration=int(iteration),
            require_nonempty=False,
        )
    else:
        _stg.read_quantum_acceptance_manifest(
            round_dir,
            expected_phase=phase.value,
            expected_iteration=int(iteration),
            require_nonempty=False,
            points_membership=_stg.POINTS_MEMBERSHIP_PRODUCER_OR_ACCEPTED,
        )
    _require_all_rejected_gaussian_coverage(
        campaign,
        round_dir,
        expected_phase=phase.value,
        expected_iteration=int(iteration),
        context=str(context),
        replacement_round=int(replacement_round),
        expected_campaign_uid=expected_campaign_uid,
    )


def _require_initial_quantum(
    campaign: Path,
    phase: CampaignPhase,
    iteration: int,
    *,
    verification: str = "metadata",
    expected_campaign_uid: Optional[str] = None,
) -> None:
    staging = campaign / ".DATA" / "STAGING" / "initial"
    require_nonempty = phase is not CampaignPhase.INITIAL_GAUSSIAN
    if verification == "authority":
        _require_quantum_acceptance_authority(
            staging,
            expected_phase=phase.value,
            expected_iteration=int(iteration),
            require_nonempty=require_nonempty,
        )
    else:
        _stg.read_quantum_acceptance_manifest(
            staging,
            expected_phase=phase.value,
            expected_iteration=int(iteration),
            require_nonempty=require_nonempty,
            points_membership=(
                _stg.POINTS_MEMBERSHIP_PRODUCER_OR_ACCEPTED
                if phase is CampaignPhase.INITIAL_GAUSSIAN
                else _stg.POINTS_MEMBERSHIP_ALL_DISPOSITIONS
            ),
        )
    if phase is CampaignPhase.INITIAL_GAUSSIAN:
        _require_all_rejected_gaussian_coverage(
            campaign,
            staging,
            expected_phase=phase.value,
            expected_iteration=int(iteration),
            context="bootstrap",
            expected_campaign_uid=expected_campaign_uid,
        )


def _require_initial_ferebus_input(
    campaign: Path,
    iteration: int,
    reference_data_version: int,
    model_version: int,
    *,
    verification: str = "metadata",
) -> None:
    try:
        _require_initial_quantum(
            campaign,
            CampaignPhase.INITIAL_AIMALL,
            int(iteration),
            verification=verification,
        )
        return
    except Exception as handoff_error:
        if int(reference_data_version) == 0 and int(model_version) < 0:
            try:
                _require_reference_data_version(campaign, 0)
                return
            except Exception as training_error:
                raise RecoveryContractError(
                    "INITIAL_FEREBUS requires either a valid initial AIMAll "
                    "handoff or committed bootstrap reference-data version 0; "
                    "initial handoff error: "
                    + type(handoff_error).__name__
                    + ": "
                    + str(handoff_error)[:120]
                    + "; training error: "
                    + type(training_error).__name__
                    + ": "
                    + str(training_error)[:120]
                ) from training_error
        raise


def _require_iter_quantum(
    campaign: Path,
    phase: CampaignPhase,
    iteration: int,
    *,
    verification: str = "metadata",
    expected_campaign_uid: Optional[str] = None,
) -> None:
    staging = campaign / ".DATA" / "STAGING" / ("iter_" + str(int(iteration)))
    require_nonempty = phase is not CampaignPhase.GAUSSIAN
    if verification == "authority":
        _require_quantum_acceptance_authority(
            staging,
            expected_phase=phase.value,
            expected_iteration=int(iteration),
            require_nonempty=require_nonempty,
        )
    else:
        _stg.read_quantum_acceptance_manifest(
            staging,
            expected_phase=phase.value,
            expected_iteration=int(iteration),
            require_nonempty=require_nonempty,
            points_membership=(
                _stg.POINTS_MEMBERSHIP_PRODUCER_OR_ACCEPTED
                if phase is CampaignPhase.GAUSSIAN
                else _stg.POINTS_MEMBERSHIP_ALL_DISPOSITIONS
            ),
        )
    if phase is CampaignPhase.GAUSSIAN:
        _require_all_rejected_gaussian_coverage(
            campaign,
            staging,
            expected_phase=phase.value,
            expected_iteration=int(iteration),
            context="active",
            expected_campaign_uid=expected_campaign_uid,
        )


def _require_seeds(campaign: Path, iteration: int) -> None:
    from ..handoff_manifests import load_seeds_picked

    load_seeds_picked(iteration_dir(campaign, iteration), expected_iteration=int(iteration))


def _authority_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryContractError(label + " must be an integer")
    result = int(value)
    if result < int(minimum):
        raise RecoveryContractError(
            label + " must be >= " + str(int(minimum))
        )
    return result


def _authority_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryContractError(label + " is not a regular file: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecoveryContractError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise RecoveryContractError(label + " must be a JSON object")
    return dict(payload)


def _authority_handoff_reference(
    root: Path,
    raw: Any,
    label: str,
) -> Path:
    """Normalise a handoff reference lexically without inspecting its payload."""
    if not isinstance(raw, str) or not raw.strip():
        raise RecoveryContractError(label + " path is missing")
    base = Path(os.path.abspath(os.path.normpath(str(root))))
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    normalised = Path(os.path.abspath(os.path.normpath(str(candidate))))
    if normalised != base and base not in normalised.parents:
        raise RecoveryContractError(label + " path escapes its handoff root")
    return normalised


def _require_ariadne_results_authority(
    campaign: Path,
    iteration: int,
    expected_campaign_uid: Optional[str],
) -> Dict[str, Any]:
    """Validate the bounded ARIADNE authority without opening seed payloads."""
    from ..handoff_manifests import (
        ARIADNE_RESULTS_SCHEMA_VERSION,
        ariadne_results_path,
        read_ariadne_batch_decision,
    )
    from ..ariadne_outputs import (
        SEED_OUTPUT_MANIFEST_FILENAME,
        SEED_RESULT_FILENAME,
    )
    from ..seed_identity import read_ariadne_task_map
    from ..versioning.provenance import PROVENANCE_FILENAME

    idir = iteration_dir(campaign, iteration)
    path = ariadne_results_path(idir)
    payload = _authority_json_object(path, "ARIADNE results manifest")
    if _authority_integer(
        payload.get("schema_version"),
        "ARIADNE results schema_version",
        minimum=1,
    ) != ARIADNE_RESULTS_SCHEMA_VERSION:
        raise RecoveryContractError("unsupported ARIADNE results manifest schema")
    if _authority_integer(
        payload.get("iteration"), "ARIADNE results iteration", minimum=1
    ) != int(iteration):
        raise RecoveryContractError("ARIADNE results manifest iteration mismatch")
    campaign_uid = str(payload.get("campaign_uid") or "")
    if not campaign_uid:
        raise RecoveryContractError("ARIADNE results campaign UID is missing")
    if expected_campaign_uid is not None and campaign_uid != str(
        expected_campaign_uid
    ):
        raise RecoveryContractError("ARIADNE results campaign UID mismatch")
    accepted = payload.get("accepted")
    rejected = payload.get("rejected")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise RecoveryContractError(
            "ARIADNE results accepted/rejected records must be lists"
        )
    expected_n = _authority_integer(
        payload.get("expected_n"), "ARIADNE expected_n", minimum=1
    )
    if _authority_integer(
        payload.get("n_accepted"), "ARIADNE n_accepted"
    ) != len(accepted):
        raise RecoveryContractError("ARIADNE n_accepted does not match its records")
    if _authority_integer(
        payload.get("n_rejected"), "ARIADNE n_rejected"
    ) != len(rejected):
        raise RecoveryContractError("ARIADNE n_rejected does not match its records")
    if expected_n != len(accepted) + len(rejected) or not accepted:
        raise RecoveryContractError("ARIADNE results counts are inconsistent")

    task_map = read_ariadne_task_map(idir, expected_iteration=int(iteration))
    if str(task_map.get("campaign_uid") or "") != campaign_uid:
        raise RecoveryContractError("ARIADNE results/task-map campaign UID mismatch")
    if int(task_map.get("n_tasks", -1)) != expected_n:
        raise RecoveryContractError("ARIADNE results/task-map count mismatch")
    task_by_seed = {
        int(task["seed_id"]): dict(task) for task in list(task_map.get("tasks") or [])
    }
    ariadne_root = path.parent
    seen = set()
    for label, records in (("accepted", accepted), ("rejected", rejected)):
        for record in records:
            if not isinstance(record, dict):
                raise RecoveryContractError(
                    label + " ARIADNE record must be an object"
                )
            seed_id = _authority_integer(
                record.get("seed_id"), label + " ARIADNE seed_id", minimum=1
            )
            task = task_by_seed.get(seed_id)
            if task is None or seed_id in seen:
                raise RecoveryContractError(
                    "invalid or duplicate " + label + " ARIADNE seed_id"
                )
            seen.add(seed_id)
            if str(record.get("seed_uid") or "") != str(task.get("seed_uid") or ""):
                raise RecoveryContractError(label + " ARIADNE seed UID mismatch")
            if record.get("array_task_id") is not None and _authority_integer(
                record.get("array_task_id"),
                label + " ARIADNE array_task_id",
                minimum=0,
            ) != int(task.get("array_task_id", -1)):
                raise RecoveryContractError(
                    label + " ARIADNE array task ID mismatch"
                )
            expected_seed_dir = _authority_handoff_reference(
                ariadne_root,
                str(task.get("seed_directory") or "").removeprefix("ariadne/"),
                label + " ARIADNE task-map seed directory",
            )
            expected_paths = {
                "seed_dir": expected_seed_dir,
                "result_json": expected_seed_dir / SEED_RESULT_FILENAME,
                "provenance_json": expected_seed_dir / PROVENANCE_FILENAME,
                "output_manifest": expected_seed_dir
                / SEED_OUTPUT_MANIFEST_FILENAME,
            }
            for path_key, expected_path in expected_paths.items():
                raw_path = record.get(path_key)
                if label == "accepted" or raw_path not in (None, ""):
                    observed_path = _authority_handoff_reference(
                        ariadne_root,
                        raw_path,
                        label + " ARIADNE " + path_key,
                    )
                    if observed_path != expected_path:
                        raise RecoveryContractError(
                            label
                            + " ARIADNE "
                            + path_key
                            + " does not match its canonical task path"
                        )
            if label == "accepted":
                safety = record.get("landing_safety")
                if not isinstance(safety, dict) or safety.get("accepted") is not True:
                    raise RecoveryContractError(
                        "accepted ARIADNE record lacks accepted landing safety"
                    )
    if seen != set(task_by_seed):
        raise RecoveryContractError("ARIADNE results do not cover every task-map seed")

    read_ariadne_batch_decision(
        idir,
        expected_iteration=int(iteration),
        expected_campaign_uid=campaign_uid,
        require_accepted=True,
    )
    return payload


def _require_ariadne_results(
    campaign: Path,
    iteration: int,
    expected_campaign_uid: Optional[str] = None,
    *,
    verification: str = "metadata",
) -> Dict[str, Any]:
    from ..config import CampaignConfig
    from .config_lock import canonical_config, config_fingerprint
    from ..handoff_manifests import read_ariadne_batch_decision

    if verification == "authority":
        payload = _require_ariadne_results_authority(
            campaign,
            int(iteration),
            expected_campaign_uid,
        )
    else:
        payload = read_ariadne_results_manifest(
            iteration_dir(campaign, iteration),
            expected_iteration=int(iteration),
            require_nonempty=True,
        )
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    read_ariadne_batch_decision(
        iteration_dir(campaign, iteration),
        expected_iteration=int(iteration),
        expected_campaign_uid=expected_campaign_uid,
        expected_config_sha256=config_fingerprint(canonical_config(config)),
        require_accepted=True,
    )
    return payload


def ariadne_results_recovery_summary(
    campaign_dir: Union[str, Path],
    iteration: int,
    expected_campaign_uid: Optional[str] = None,
    *,
    verification: str = "authority",
) -> Dict[str, Any]:
    """Return bounded recovery counts for an accepted ARIADNE handoff."""
    campaign = Path(campaign_dir)
    payload = _require_ariadne_results(
        campaign,
        int(iteration),
        expected_campaign_uid,
        verification=verification,
    )
    root = iteration_dir(campaign, int(iteration)) / "ariadne"
    missing_rejected_outputs = 0
    for record in list(payload.get("rejected") or []):
        raw_seed_dir = record.get("seed_dir") if isinstance(record, dict) else None
        if raw_seed_dir in (None, ""):
            missing_rejected_outputs += 1
            continue
        seed_dir = _authority_handoff_reference(
            root,
            raw_seed_dir,
            "rejected ARIADNE seed_dir",
        )
        if not seed_dir.exists():
            missing_rejected_outputs += 1
    return {
        "expected_tasks": int(payload["expected_n"]),
        "accepted_tasks": int(payload["n_accepted"]),
        "rejected_tasks": int(payload["n_rejected"]),
        "missing_rejected_outputs": int(missing_rejected_outputs),
        "tasks_resubmitted": 0,
    }


def _count_xyz_frames(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RecoveryContractError("sample xyz unreadable: " + str(path)) from exc
    pos = 0
    n_frames = 0
    while pos < len(lines):
        if not lines[pos].strip():
            pos += 1
            continue
        try:
            natoms = int(lines[pos].strip())
        except ValueError as exc:
            raise RecoveryContractError(
                "sample xyz frame atom count is not an integer at line "
                + str(pos + 1)
            ) from exc
        if natoms < 0:
            raise RecoveryContractError("sample xyz frame atom count is negative")
        frame_end = pos + 2 + natoms
        if frame_end > len(lines):
            raise RecoveryContractError("sample xyz frame is truncated")
        pos = frame_end
        n_frames += 1
    if n_frames <= 0:
        raise RecoveryContractError("sample xyz contains no frames: " + str(path))
    return n_frames


def _require_phase_b_authority(
    campaign: Path,
    iteration: int,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate Phase B authority without reading XYZ or per-seed payloads."""
    from ..handoff_manifests import (
        PHASE_B_SELECTION_SCHEMA_VERSION,
        ariadne_results_path,
        phase_b_selection_path,
    )
    from ..point_allocation import read_point_allocation
    from ..sampling.diversity_contract import selector_contract_matches
    from ..versioning.manifest import sha256_file
    from .filesystem import campaign_owned_path

    idir = iteration_dir(campaign, iteration)
    path = phase_b_selection_path(idir)
    payload = _authority_json_object(path, "Phase B selection manifest")
    if _authority_integer(
        payload.get("schema_version"), "Phase B schema_version", minimum=1
    ) != PHASE_B_SELECTION_SCHEMA_VERSION:
        raise RecoveryContractError("unsupported Phase B selection manifest schema")
    if _authority_integer(
        payload.get("iteration"), "Phase B iteration", minimum=1
    ) != int(iteration):
        raise RecoveryContractError("Phase B selection manifest iteration mismatch")
    campaign_uid = str(payload.get("campaign_uid") or "")
    if not campaign_uid:
        raise RecoveryContractError("Phase B campaign UID is missing")
    if expected_campaign_uid is not None and campaign_uid != str(
        expected_campaign_uid
    ):
        raise RecoveryContractError("Phase B campaign UID mismatch")
    if str(payload.get("status") or "") != "complete":
        raise RecoveryContractError("Phase B selection is not complete")
    considered = payload.get("considered")
    final = payload.get("final")
    if not isinstance(considered, list) or not isinstance(final, list):
        raise RecoveryContractError("Phase B considered/final records must be lists")
    if not final:
        raise RecoveryContractError("Phase B final records are empty")
    if payload.get("n_considered") is not None and _authority_integer(
        payload.get("n_considered"), "Phase B n_considered"
    ) != len(considered):
        raise RecoveryContractError("Phase B n_considered mismatch")
    if payload.get("n_kept") is not None and _authority_integer(
        payload.get("n_kept"), "Phase B n_kept"
    ) != len(final):
        raise RecoveryContractError("Phase B n_kept mismatch")

    source_raw = payload.get("source_ariadne_manifest")
    if not isinstance(source_raw, str) or not source_raw:
        raise RecoveryContractError("Phase B source ARIADNE manifest is missing")
    source = campaign_owned_path(campaign, idir / source_raw)
    expected_source = campaign_owned_path(campaign, ariadne_results_path(idir))
    if source != expected_source or not source.is_file():
        raise RecoveryContractError("Phase B source ARIADNE manifest is noncanonical")
    if str(payload.get("source_ariadne_manifest_sha256") or "") != sha256_file(
        source
    ):
        raise RecoveryContractError("Phase B source ARIADNE manifest hash mismatch")
    ariadne = _require_ariadne_results_authority(
        campaign,
        int(iteration),
        campaign_uid,
    )
    accepted_by_seed = {
        int(record["seed_id"]): dict(record)
        for record in list(ariadne.get("accepted") or [])
    }

    allocation = payload.get("point_allocation")
    if not isinstance(allocation, dict):
        raise RecoveryContractError("Phase B point-allocation binding is missing")
    allocation_raw = allocation.get("manifest")
    if not isinstance(allocation_raw, str) or not allocation_raw:
        raise RecoveryContractError("Phase B point-allocation path is missing")
    allocation_path = campaign_owned_path(campaign, idir / allocation_raw)
    allocation_payload = read_point_allocation(
        allocation_path,
        expected_campaign_uid=campaign_uid,
        expected_context="active",
        expected_iteration=int(iteration),
    )
    if str(allocation.get("slot_assignment_sha256") or "") != str(
        allocation_payload.get("slot_assignment_sha256") or ""
    ):
        raise RecoveryContractError("Phase B point-allocation assignment mismatch")
    if not selector_contract_matches(payload.get("selector")):
        raise RecoveryContractError("Phase B selector contract is invalid")

    considered_kept = set()
    considered_seeds = set()
    for rank, record in enumerate(considered, start=1):
        if not isinstance(record, dict):
            raise RecoveryContractError("Phase B considered record must be an object")
        if _authority_integer(
            record.get("considered_rank"), "Phase B considered_rank", minimum=1
        ) != rank:
            raise RecoveryContractError("Phase B considered ranks are not contiguous")
        seed_id = _authority_integer(
            record.get("seed_id"), "Phase B considered seed_id", minimum=1
        )
        source_record = accepted_by_seed.get(seed_id)
        if source_record is None or seed_id in considered_seeds:
            raise RecoveryContractError("Phase B considered seed mapping is invalid")
        considered_seeds.add(seed_id)
        if str(record.get("seed_uid") or "") != str(
            source_record.get("seed_uid") or ""
        ):
            raise RecoveryContractError("Phase B considered seed UID mismatch")
        kept = record.get("kept_after_dedup")
        if kept is not True and kept is not False:
            raise RecoveryContractError("Phase B kept_after_dedup must be Boolean")
        final_rank = record.get("final_rank")
        if kept:
            considered_kept.add(
                _authority_integer(
                    final_rank, "Phase B considered final_rank", minimum=1
                )
            )
        elif final_rank is not None:
            raise RecoveryContractError(
                "dropped Phase B considered record has a final rank"
            )

    final_ranks = set()
    final_considered = set()
    for record in final:
        if not isinstance(record, dict):
            raise RecoveryContractError("Phase B final record must be an object")
        final_rank = _authority_integer(
            record.get("final_rank"), "Phase B final_rank", minimum=1
        )
        considered_rank = _authority_integer(
            record.get("considered_rank"),
            "Phase B final considered_rank",
            minimum=1,
        )
        if final_rank in final_ranks or considered_rank in final_considered:
            raise RecoveryContractError("Phase B final ranks are duplicated")
        final_ranks.add(final_rank)
        final_considered.add(considered_rank)
        if record.get("kept_after_dedup") is not True:
            raise RecoveryContractError("Phase B final record is not marked kept")
        seed_id = _authority_integer(
            record.get("seed_id"), "Phase B final seed_id", minimum=1
        )
        source_record = accepted_by_seed.get(seed_id)
        if source_record is None or str(record.get("seed_uid") or "") != str(
            source_record.get("seed_uid") or ""
        ):
            raise RecoveryContractError("Phase B final seed mapping is invalid")
    expected_ranks = set(range(1, len(final) + 1))
    if final_ranks != expected_ranks or considered_kept != expected_ranks:
        raise RecoveryContractError("Phase B final rank coverage is invalid")
    if final_considered != {
        int(record["considered_rank"])
        for record in considered
        if record.get("kept_after_dedup") is True
    }:
        raise RecoveryContractError("Phase B final/considered mapping is invalid")
    return payload


def _phase_b_final_count(
    campaign: Path,
    iteration: int,
    expected_campaign_uid: Optional[str] = None,
    *,
    verification: str = "metadata",
) -> int:
    idir = iteration_dir(campaign, iteration)
    if verification == "authority":
        manifest = _require_phase_b_authority(
            campaign,
            int(iteration),
            expected_campaign_uid,
        )
    else:
        from ..handoff_manifests import validate_phase_b_handoff

        manifest = validate_phase_b_handoff(
            idir,
            expected_iteration=int(iteration),
            expected_campaign_uid=expected_campaign_uid,
        )
    n_final = len(list(manifest.get("final") or []))
    return int(n_final)


def _require_phase_b(
    campaign: Path,
    iteration: int,
    expected_campaign_uid: Optional[str] = None,
    *,
    verification: str = "metadata",
) -> None:
    _phase_b_final_count(
        campaign,
        int(iteration),
        expected_campaign_uid=expected_campaign_uid,
        verification=verification,
    )


def _require_split(
    campaign: Path,
    iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> None:
    from ..layout import active_allocation_dir
    from ..point_allocation import point_allocation_path

    idir = iteration_dir(campaign, iteration)
    path = active_allocation_dir(idir) / "SPLIT_RECEIPT.json"
    if not path.is_file():
        raise FileNotFoundError("SPLIT_RECEIPT.json missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecoveryContractError("SPLIT_RECEIPT.json unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise RecoveryContractError("SPLIT_RECEIPT.json must be a JSON object")
    if int(data.get("schema_version", -1)) != 3:
        raise RecoveryContractError("SPLIT_RECEIPT.json schema 3 is required")
    if int(data.get("iteration")) != int(iteration):
        raise RecoveryContractError("SPLIT_RECEIPT.json iteration mismatch")
    if str(data.get("strategy")) != "exact_pre_qm_point_allocation":
        raise RecoveryContractError("SPLIT_RECEIPT.json strategy is invalid")
    allocation = _require_point_allocation(
        campaign,
        context="active",
        iteration=int(iteration),
        expected_campaign_uid=expected_campaign_uid,
    )
    slots = data.get("slots")
    if not isinstance(slots, list) or len(slots) != int(allocation["targets"]["total"]):
        raise RecoveryContractError("SPLIT_RECEIPT.json slots do not match point allocation")
    if str(data.get("slot_assignment_sha256") or "") != str(
        allocation.get("slot_assignment_sha256") or ""
    ):
        raise RecoveryContractError("SPLIT_RECEIPT.json assignment hash mismatch")
    allocation_path = Path(
        str(data.get("point_allocation_manifest") or "")
    )
    if not allocation_path.is_absolute():
        allocation_path = idir / allocation_path
    if allocation_path.resolve() != point_allocation_path(
        campaign,
        context="active",
        iteration=int(iteration),
    ).resolve():
        raise RecoveryContractError("SPLIT_RECEIPT.json allocation path mismatch")
    expected = {
        int(slot["slot_id"]): (
            str(slot["split"]),
            str(slot["attempts"][0]["candidate_id"]),
        )
        for slot in allocation["slots"]
    }
    observed: Dict[int, Tuple[str, str]] = {}
    for record in slots:
        if not isinstance(record, dict):
            raise RecoveryContractError("SPLIT_RECEIPT.json slot record is invalid")
        slot_id = int(record.get("slot_id", -1))
        if slot_id in observed:
            raise RecoveryContractError("SPLIT_RECEIPT.json slot IDs contain duplicates")
        observed[slot_id] = (
            str(record.get("split") or ""),
            str(record.get("candidate_id") or ""),
        )
    if observed != expected:
        raise RecoveryContractError("SPLIT_RECEIPT.json does not reproduce point allocation")


def _require_reference_data_version(
    campaign: Path,
    version: int,
    *,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> None:
    verify_committed_reference_data_version(
        campaign,
        int(version),
        verification=verification,
        snapshot=artifact_snapshot,
    )


def _require_model_version(
    campaign: Path,
    version: int,
    *,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> None:
    verify_committed_model_version(
        campaign,
        int(version),
        verification=verification,
        snapshot=artifact_snapshot,
    )


def _require_active_iteration_committed(state: CampaignState, iteration: int) -> None:
    if not active_iteration_committed(state, iteration):
        raise RecoveryContractError(
            "active iteration "
            + str(int(iteration))
            + " is not fully committed; STOP_CHECK would skip unfinished work"
        )


def _require_ferebus_needed(
    campaign: Path,
    reference_data_version: int,
    model_version: int,
    *,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> None:
    if int(reference_data_version) < 0:
        raise RecoveryContractError("FEREBUS requires a non-negative reference-data version")
    _require_reference_data_version(
        campaign,
        int(reference_data_version),
        verification=verification,
        artifact_snapshot=artifact_snapshot,
    )
    if int(model_version) >= int(reference_data_version):
        raise RecoveryContractError(
            "FEREBUS model version "
            + str(int(model_version))
            + " is already at or ahead of reference-data version "
            + str(int(reference_data_version))
        )


def _active_iteration_for_reference_data_version(reference_data_version: int) -> int:
    return int(reference_data_version)


def _allocation_recovery_decision(
    campaign: Path,
    *,
    context: str,
    iteration: int,
    expected_campaign_uid: Optional[str] = None,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> Optional[RecoveryDecision]:
    from ..point_allocation import pending_attempts

    try:
        allocation = _require_point_allocation(
            campaign,
            context=str(context),
            iteration=int(iteration),
            expected_campaign_uid=expected_campaign_uid,
        )
    except Exception:
        return None
    summary = dict(allocation.get("summary") or {})
    from ..point_allocation import point_allocation_path

    allocation_artifact = point_allocation_path(
        campaign,
        context=str(context),
        iteration=int(iteration),
    ).relative_to(campaign).as_posix()
    if bool(summary.get("complete", False)):
        if _ok(
            verify_committed_reference_data_version,
            campaign,
            int(iteration),
            verification=verification,
            snapshot=artifact_snapshot,
        ):
            # The allocation remains as authoritative evidence after its
            # pointdirs have moved. Do not rewind an already published
            # reference version back into REFERENCE_COMMIT.
            return None
        return RecoveryDecision(
            CampaignPhase.REFERENCE_COMMIT,
            int(iteration),
            "REFERENCE_COMMIT: exact point allocation is complete",
            allocation_artifact,
        )
    pending = pending_attempts(allocation)
    if not pending:
        return RecoveryDecision(
            CampaignPhase.INITIAL_ALLOCATION_CHECK
            if context == "bootstrap"
            else CampaignPhase.ALLOCATION_CHECK,
            int(iteration),
            "point-allocation check: labelled slots are underfilled and no QM attempt is pending",
            allocation_artifact,
        )
    rounds = {int(record.get("round", -1)) for record in pending}
    if len(rounds) != 1:
        return None
    replacement_round = next(iter(rounds))
    if replacement_round <= 0:
        return None
    from ..replacement_sampling import inspect_replacement_sample_recovery

    sample = inspect_replacement_sample_recovery(
        campaign,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        expected_campaign_uid=expected_campaign_uid,
    )
    sample_state = str(sample.get("state") or "")
    if sample_state in {"missing_rebuildable", "partial_rebuildable"}:
        return RecoveryDecision(
            CampaignPhase.INITIAL_ALLOCATION_CHECK
            if context == "bootstrap"
            else CampaignPhase.ALLOCATION_CHECK,
            int(iteration),
            "point-allocation check: pending replacement allocation needs sample repair",
            allocation_artifact,
            replacement_round=int(replacement_round),
        )
    if sample_state != "valid":
        return RecoveryDecision(
            CampaignPhase.INITIAL_ALLOCATION_CHECK
            if context == "bootstrap"
            else CampaignPhase.ALLOCATION_CHECK,
            int(iteration),
            "point-allocation check: replacement sample evidence requires review",
            allocation_artifact,
            replacement_round=int(replacement_round),
        )
    round_dir = Path(str(sample["round_dir"]))
    gaussian_phase = (
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN
        if context == "bootstrap"
        else CampaignPhase.REPLACEMENT_GAUSSIAN
    )
    aimall_phase = (
        CampaignPhase.INITIAL_REPLACEMENT_AIMALL
        if context == "bootstrap"
        else CampaignPhase.REPLACEMENT_AIMALL
    )
    if _ok(
        _stg.read_quantum_acceptance_manifest,
        round_dir,
        expected_phase=gaussian_phase.value,
        expected_iteration=int(iteration),
        require_nonempty=False,
        points_membership=_stg.POINTS_MEMBERSHIP_PRODUCER_OR_ACCEPTED,
    ):
        return RecoveryDecision(
            aimall_phase,
            int(iteration),
            aimall_phase.value + ": valid replacement Gaussian handoff exists",
            str(round_dir.relative_to(campaign)),
            replacement_round=int(replacement_round),
        )
    return RecoveryDecision(
        gaussian_phase,
        int(iteration),
        gaussian_phase.value + ": replacement sample is ready",
        str(round_dir.relative_to(campaign)),
        replacement_round=int(replacement_round),
    )


def _require_ferebus_iteration(iteration: int, reference_data_version: int) -> None:
    expected = _active_iteration_for_reference_data_version(reference_data_version)
    if int(iteration) != expected:
        raise RecoveryContractError(
            "FEREBUS iteration "
            + str(int(iteration))
            + " does not match reference-data version "
            + str(int(reference_data_version))
        )


def protected_staging_handoff(
    campaign_dir: Union[str, Path],
    *,
    iteration: int,
    expected_campaign_uid: Optional[str] = None,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> Optional[RecoveryDecision]:
    """Return the consumer phase for a valid active quantum staging handoff."""
    campaign = Path(campaign_dir)
    allocation_decision = _allocation_recovery_decision(
        campaign,
        context="active",
        iteration=int(iteration),
        expected_campaign_uid=expected_campaign_uid,
        verification=verification,
        artifact_snapshot=artifact_snapshot,
    )
    if allocation_decision is not None and allocation_decision.phase in {
        CampaignPhase.ALLOCATION_CHECK,
        CampaignPhase.REPLACEMENT_GAUSSIAN,
        CampaignPhase.REPLACEMENT_AIMALL,
        CampaignPhase.REFERENCE_COMMIT,
    }:
        return RecoveryDecision(
            allocation_decision.phase,
            allocation_decision.iteration,
            allocation_decision.reason,
            ".DATA/STAGING/iter_" + str(int(iteration)),
            replacement_round=int(allocation_decision.replacement_round),
        )
    if _ok(
        _require_iter_quantum,
        campaign,
        CampaignPhase.AIMALL,
        int(iteration),
        verification=verification,
    ):
        return RecoveryDecision(
            CampaignPhase.AIMALL,
            int(iteration),
            "AIMALL: acceptance handoff exists and point-allocation recording must be verified",
            ".DATA/STAGING/iter_" + str(int(iteration)),
        )
    if _ok(
        _require_iter_quantum,
        campaign,
        CampaignPhase.GAUSSIAN,
        int(iteration),
        verification=verification,
    ):
        return RecoveryDecision(
            CampaignPhase.AIMALL,
            int(iteration),
            "AIMALL: valid iterative Gaussian handoff exists",
            ".DATA/STAGING/iter_" + str(int(iteration)),
        )
    return None


def staging_handoff_decisions(
    campaign_dir: Union[str, Path],
    state: CampaignState,
    *,
    include_committed: bool = False,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> List[RecoveryDecision]:
    """Return valid quantum staging handoffs that must not be archived."""
    campaign = Path(campaign_dir)
    decisions: List[RecoveryDecision] = []
    reference_data_version = int(
        getattr(state, "reference_data_version", -1)
    )
    if include_committed or reference_data_version < 0:
        allocation_decision = _allocation_recovery_decision(
            campaign,
            context="bootstrap",
            iteration=0,
            expected_campaign_uid=str(state.campaign_uid),
            verification=verification,
            artifact_snapshot=artifact_snapshot,
        )
        if allocation_decision is not None:
            decisions.append(
                RecoveryDecision(
                    allocation_decision.phase,
                    allocation_decision.iteration,
                    allocation_decision.reason,
                    ".DATA/STAGING/initial",
                    replacement_round=int(allocation_decision.replacement_round),
                )
            )
        elif _ok(
            _require_initial_quantum,
            campaign,
            CampaignPhase.INITIAL_AIMALL,
            0,
            verification=verification,
        ):
            decisions.append(
                RecoveryDecision(
                    CampaignPhase.INITIAL_AIMALL,
                    0,
                    "INITIAL_AIMALL: acceptance handoff exists and point-allocation recording must be verified",
                    ".DATA/STAGING/initial",
                )
            )
        elif _ok(
            _require_initial_quantum,
            campaign,
            CampaignPhase.INITIAL_GAUSSIAN,
            0,
            verification=verification,
        ):
            decisions.append(
                RecoveryDecision(
                    CampaignPhase.INITIAL_AIMALL,
                    0,
                    "INITIAL_AIMALL: valid initial Gaussian handoff exists",
                    ".DATA/STAGING/initial",
                )
            )
    staging = campaign / ".DATA" / "STAGING"
    if not staging.is_dir():
        return decisions
    for bucket in sorted(staging.glob("iter_*")):
        if not bucket.is_dir() or bucket.is_symlink():
            continue
        suffix = bucket.name[len("iter_"):]
        try:
            iteration = int(suffix)
        except ValueError:
            continue
        if not include_committed and active_iteration_committed(state, iteration):
            continue
        decision = protected_staging_handoff(
            campaign,
            iteration=iteration,
            expected_campaign_uid=str(state.campaign_uid),
            verification=verification,
            artifact_snapshot=artifact_snapshot,
        )
        if decision is not None:
            decisions.append(decision)
    return decisions


def _best_active_iteration_handoff(
    campaign: Path,
    iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> Optional[RecoveryHandoff]:
    allocation_decision = _allocation_recovery_decision(
        campaign,
        context="active",
        iteration=int(iteration),
        expected_campaign_uid=expected_campaign_uid,
        verification=verification,
        artifact_snapshot=artifact_snapshot,
    )
    if allocation_decision is not None:
        return RecoveryHandoff(
            allocation_decision,
            50,
            "point_allocation",
        )
    if _ok(
        _require_split,
        campaign,
        int(iteration),
        expected_campaign_uid=expected_campaign_uid,
    ):
        from ..layout import active_allocation_dir

        return RecoveryHandoff(
            RecoveryDecision(
                CampaignPhase.GAUSSIAN,
                int(iteration),
                "GAUSSIAN: valid split handoff exists",
                str(
                    active_allocation_dir(iteration_dir(campaign, iteration))
                    / "SPLIT_RECEIPT.json"
                ),
            ),
            40,
            "split",
        )
    if _ok(
        _require_phase_b,
        campaign,
        int(iteration),
        expected_campaign_uid=expected_campaign_uid,
        verification=verification,
    ):
        from ..handoff_manifests import phase_b_selection_path

        return RecoveryHandoff(
            RecoveryDecision(
                CampaignPhase.SPLIT,
                int(iteration),
                "SPLIT: valid Phase B handoff exists",
                str(phase_b_selection_path(iteration_dir(campaign, iteration))),
            ),
            30,
            "phase_b",
        )
    if _ok(
        _require_ariadne_results,
        campaign,
        int(iteration),
        verification=verification,
    ):
        from ..handoff_manifests import ariadne_results_path

        return RecoveryHandoff(
            RecoveryDecision(
                CampaignPhase.PHASE_B_DIVERSITY,
                int(iteration),
                "PHASE_B_DIVERSITY: valid ARIADNE results handoff exists",
                str(ariadne_results_path(iteration_dir(campaign, iteration))),
            ),
            20,
            "ariadne_results",
        )
    if _ok(_require_seeds, campaign, int(iteration)):
        from ..handoff_manifests import seeds_picked_path

        return RecoveryHandoff(
            RecoveryDecision(
                CampaignPhase.ARIADNE_ARRAY,
                int(iteration),
                "ARIADNE_ARRAY: valid seed-selection handoff exists",
                str(seeds_picked_path(iteration_dir(campaign, iteration))),
            ),
            10,
            "seeds",
        )
    return None


def active_iteration_handoff_decisions(
    campaign_dir: Union[str, Path],
    state: CampaignState,
    *,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> List[RecoveryDecision]:
    """Return the furthest valid AL handoff for each uncommitted iteration."""
    campaign = Path(campaign_dir)
    root = active_learning_dir(campaign)
    if not root.is_dir():
        return []
    decisions: List[RecoveryDecision] = []
    from ..layout import parse_active_iteration_name

    for path in sorted(root.glob("iteration-*")):
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            iteration = parse_active_iteration_name(path.name)
        except ValueError:
            continue
        if active_iteration_committed(state, iteration):
            continue
        if active_iteration_reference_data_committed(state, iteration):
            # All per-iteration handoffs through REFERENCE_COMMIT are now historical.
            # Recovery must evaluate the reference-data/model skew and select
            # FEREBUS rather than replaying a completed allocation.
            continue
        handoff = _best_active_iteration_handoff(
            campaign,
            iteration,
            expected_campaign_uid=str(state.campaign_uid),
            verification=verification,
            artifact_snapshot=artifact_snapshot,
        )
        if handoff is not None:
            decisions.append(handoff.decision)
    return decisions


def _single_decision_or_none(decisions: Sequence[RecoveryDecision]) -> Optional[RecoveryDecision]:
    if len(decisions) == 1:
        return decisions[0]
    return None


def _phase_contract_checks(
    campaign: Path,
    state: CampaignState,
    *,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> List[Tuple[str, Callable[[], None]]]:
    phase = CampaignPhase(state.phase)
    iteration = int(getattr(state, "iteration", 0))
    reference_data_version = int(getattr(state, "reference_data_version", -1))
    model_version = int(getattr(state, "models_version", -1))

    checks: Dict[CampaignPhase, List[Tuple[str, Callable[[], None]]]] = {
        CampaignPhase.PHASE_A_DIVERSITY: [
            (
                "trajectory pool",
                lambda: _require_pool(campaign, verification=verification),
            ),
        ],
        CampaignPhase.INITIAL_GAUSSIAN: [
            (
                "Phase A sample",
                lambda: _require_phase_a(campaign, verification=verification),
            ),
        ],
        CampaignPhase.INITIAL_AIMALL: [
            (
                "initial Gaussian handoff",
                lambda: _require_initial_quantum(
                    campaign,
                    CampaignPhase.INITIAL_GAUSSIAN,
                    iteration,
                    verification=verification,
                    expected_campaign_uid=str(state.campaign_uid),
                ),
            ),
        ],
        CampaignPhase.INITIAL_ALLOCATION_CHECK: [
            (
                "bootstrap point allocation",
                lambda: _require_allocation_check_ready(
                    campaign,
                    context="bootstrap",
                    iteration=0,
                    expected_replacement_round=int(
                        getattr(state, "replacement_round", 0)
                    ),
                    expected_campaign_uid=str(state.campaign_uid),
                ),
            ),
        ],
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN: [
            (
                "bootstrap replacement sample",
                lambda: _require_replacement_sample(
                    campaign,
                    context="bootstrap",
                    iteration=0,
                    replacement_round=int(getattr(state, "replacement_round", 0)),
                    expected_campaign_uid=str(state.campaign_uid),
                ),
            ),
        ],
        CampaignPhase.INITIAL_REPLACEMENT_AIMALL: [
            (
                "bootstrap replacement Gaussian handoff",
                lambda: _require_replacement_gaussian_handoff(
                    campaign,
                    context="bootstrap",
                    iteration=0,
                    replacement_round=int(getattr(state, "replacement_round", 0)),
                    verification=verification,
                    expected_campaign_uid=str(state.campaign_uid),
                ),
            ),
        ],
        CampaignPhase.INITIAL_FEREBUS: [
            (
                "committed bootstrap reference-data version 0",
                lambda: _require_reference_data_version(
                    campaign,
                    0,
                    verification=verification,
                    artifact_snapshot=artifact_snapshot,
                ),
            ),
        ],
        CampaignPhase.REFERENCE_COMMIT: [
            (
                "complete point allocation",
                lambda: _require_point_allocation(
                    campaign,
                    context="bootstrap" if iteration == 0 else "active",
                    iteration=iteration,
                    expected_campaign_uid=str(state.campaign_uid),
                    complete=True,
                ),
            ),
        ],
        CampaignPhase.SEED_SELECT: [
            (
                "committed reference-data version " + str(reference_data_version),
                lambda: _require_reference_data_version(
                    campaign,
                    reference_data_version,
                    verification=verification,
                    artifact_snapshot=artifact_snapshot,
                ),
            ),
            (
                "committed model version " + str(model_version),
                lambda: _require_model_version(
                    campaign,
                    model_version,
                    verification=verification,
                    artifact_snapshot=artifact_snapshot,
                ),
            ),
            (
                "trajectory pool",
                lambda: _require_pool(campaign, verification=verification),
            ),
        ],
        CampaignPhase.ARIADNE_ARRAY: [
            ("seed_selection/SELECTION.json", lambda: _require_seeds(campaign, iteration)),
        ],
        CampaignPhase.PHASE_B_DIVERSITY: [
            (
                "ariadne/RESULTS.json",
                lambda: _require_ariadne_results(
                    campaign,
                    iteration,
                    str(state.campaign_uid),
                    verification=verification,
                ),
            ),
        ],
        CampaignPhase.SPLIT: [
            (
                "Phase B selection/sample",
                lambda: _require_phase_b(
                    campaign,
                    iteration,
                    str(state.campaign_uid),
                    verification=verification,
                ),
            ),
        ],
        CampaignPhase.GAUSSIAN: [
            (
                "Phase B selection/sample",
                lambda: _require_phase_b(
                    campaign,
                    iteration,
                    str(state.campaign_uid),
                    verification=verification,
                ),
            ),
            (
                "allocation/SPLIT_RECEIPT.json",
                lambda: _require_split(
                    campaign,
                    iteration,
                    expected_campaign_uid=str(state.campaign_uid),
                ),
            ),
        ],
        CampaignPhase.AIMALL: [
            (
                "iterative Gaussian handoff",
                lambda: _require_iter_quantum(
                    campaign,
                    CampaignPhase.GAUSSIAN,
                    iteration,
                    verification=verification,
                    expected_campaign_uid=str(state.campaign_uid),
                ),
            ),
        ],
        CampaignPhase.ALLOCATION_CHECK: [
            (
                "active point allocation",
                lambda: _require_allocation_check_ready(
                    campaign,
                    context="active",
                    iteration=iteration,
                    expected_replacement_round=int(
                        getattr(state, "replacement_round", 0)
                    ),
                    expected_campaign_uid=str(state.campaign_uid),
                ),
            ),
        ],
        CampaignPhase.REPLACEMENT_GAUSSIAN: [
            (
                "active replacement sample",
                lambda: _require_replacement_sample(
                    campaign,
                    context="active",
                    iteration=iteration,
                    replacement_round=int(getattr(state, "replacement_round", 0)),
                    expected_campaign_uid=str(state.campaign_uid),
                ),
            ),
        ],
        CampaignPhase.REPLACEMENT_AIMALL: [
            (
                "active replacement Gaussian handoff",
                lambda: _require_replacement_gaussian_handoff(
                    campaign,
                    context="active",
                    iteration=iteration,
                    replacement_round=int(getattr(state, "replacement_round", 0)),
                    verification=verification,
                    expected_campaign_uid=str(state.campaign_uid),
                ),
            ),
        ],
        CampaignPhase.FEREBUS: [
            (
                "committed reference-data version ahead of model version",
                lambda: _require_ferebus_needed(
                    campaign,
                    reference_data_version,
                    model_version,
                    verification=verification,
                    artifact_snapshot=artifact_snapshot,
                ),
            ),
            (
                "FEREBUS iteration matches reference-data version",
                lambda: _require_ferebus_iteration(iteration, reference_data_version),
            ),
        ],
        CampaignPhase.STOP_CHECK: [
            (
                "active iteration fully committed",
                lambda: _require_active_iteration_committed(state, iteration),
            ),
        ],
    }
    return list(checks.get(phase, []))


def _existing_phase_recovery(
    campaign: Path,
    state: CampaignState,
    *,
    valid_reference_data_versions: Sequence[int],
    valid_model_versions: Sequence[int],
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> Optional[RecoveryDecision]:
    phase = CampaignPhase(state.phase)
    iteration = int(getattr(state, "iteration", 0))
    if phase in {CampaignPhase.INIT, CampaignPhase.HALTED, CampaignPhase.DONE}:
        return None
    if phase is CampaignPhase.STOP_CHECK:
        if active_iteration_committed(state, iteration):
            return RecoveryDecision(
                CampaignPhase.STOP_CHECK,
                iteration,
                "STOP_CHECK: active iteration is fully committed",
            )
        return None
    err = phase_recovery_contract_error(
        campaign,
        state,
        verification=verification,
        artifact_snapshot=artifact_snapshot,
    )
    if err is None:
        return RecoveryDecision(
            phase,
            iteration,
            phase.value + ": existing state phase has a valid input contract",
        )
    return None


def select_recovery_phase(
    campaign_dir: Union[str, Path],
    state: CampaignState,
    *,
    valid_reference_data_versions: Sequence[int],
    valid_model_versions: Sequence[int],
    existing_loaded: bool,
    last_phase: Optional[str] = None,
    last_iteration: Optional[int] = None,
    last_phase_retryable: bool = False,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> Optional[RecoveryDecision]:
    """Choose the furthest safe re-entry phase from producer contracts."""
    campaign = Path(campaign_dir)
    iteration = int(getattr(state, "iteration", 0))
    if last_iteration is not None and not existing_loaded:
        try:
            iteration = int(last_iteration)
        except (TypeError, ValueError):
            pass

    reference_data_version = int(getattr(state, "reference_data_version", -1))
    model_version = int(getattr(state, "models_version", -1))

    # Bootstrap/no-committed-version path.
    if not valid_reference_data_versions and not valid_model_versions:
        allocation_decision = _allocation_recovery_decision(
            campaign,
            context="bootstrap",
            iteration=0,
            expected_campaign_uid=str(state.campaign_uid),
            verification=verification,
            artifact_snapshot=artifact_snapshot,
        )
        if allocation_decision is not None:
            return allocation_decision
        if _ok(
            _require_initial_quantum,
            campaign,
            CampaignPhase.INITIAL_AIMALL,
            iteration,
            verification=verification,
        ):
            return RecoveryDecision(
                CampaignPhase.INITIAL_AIMALL,
                iteration,
                "INITIAL_AIMALL: acceptance exists and point-allocation recording must be verified",
                ".DATA/STAGING/initial",
            )
        if _ok(
            _require_initial_quantum,
            campaign,
            CampaignPhase.INITIAL_GAUSSIAN,
            iteration,
            verification=verification,
        ):
            return RecoveryDecision(
                CampaignPhase.INITIAL_AIMALL,
                iteration,
                "INITIAL_AIMALL: valid initial Gaussian handoff exists without committed models",
                ".DATA/STAGING/initial",
            )
        if _ok(_require_phase_a, campaign, verification=verification):
            return RecoveryDecision(
                CampaignPhase.INITIAL_GAUSSIAN,
                iteration,
                "INITIAL_GAUSSIAN: valid Phase A sample exists without committed models",
                ".DATA/BOOTSTRAP/selection/SELECTION.json",
            )
        if (
            bool(last_phase_retryable)
            and str(last_phase or "") == CampaignPhase.PHASE_A_DIVERSITY.value
            and _ok(_require_pool, campaign, verification=verification)
        ):
            return RecoveryDecision(
                CampaignPhase.PHASE_A_DIVERSITY,
                iteration,
                "PHASE_A_DIVERSITY: retryable pre-bootstrap phase has a valid trajectory pool input",
                "pool.xyz",
            )
        if existing_loaded:
            try:
                existing_phase = CampaignPhase(state.phase)
            except Exception:
                existing_phase = CampaignPhase.HALTED
            if existing_phase is CampaignPhase.PHASE_A_DIVERSITY and _ok(
                _require_pool,
                campaign,
                verification=verification,
            ):
                return RecoveryDecision(
                    CampaignPhase.PHASE_A_DIVERSITY,
                    iteration,
                    "PHASE_A_DIVERSITY: existing phase has a valid trajectory pool input",
                    "pool.xyz",
                )
        return None

    staging_decision = _single_decision_or_none(
        staging_handoff_decisions(
            campaign,
            state,
            verification=verification,
            artifact_snapshot=artifact_snapshot,
        )
    )
    if staging_decision is not None:
        return staging_decision

    handoff_decision = _single_decision_or_none(
        active_iteration_handoff_decisions(
            campaign,
            state,
            verification=verification,
            artifact_snapshot=artifact_snapshot,
        )
    )
    if handoff_decision is not None:
        return handoff_decision

    if reference_data_version == 0 and model_version < 0:
        if _has_version(valid_reference_data_versions, 0):
            return RecoveryDecision(
                CampaignPhase.INITIAL_FEREBUS,
                0,
                "INITIAL_FEREBUS: exact point allocation is complete and committed bootstrap training exists without model version 0",
                str(
                    ReferenceDataVersioning(
                        campaign / "QM_REFERENCE_DATA"
                    ).iteration_path(0).relative_to(campaign)
                ),
            )

    if reference_data_version == 0 and model_version == 0:
        return RecoveryDecision(
            CampaignPhase.SEED_SELECT,
            1,
            "SEED_SELECT: bootstrap reference data and models are committed",
            ".DATA/BOOTSTRAP/BOOTSTRAP_MANIFEST.json",
        )

    if reference_data_version == model_version and reference_data_version >= 1:
        canonical_iteration = _active_iteration_for_reference_data_version(reference_data_version)
        if iteration != canonical_iteration and active_iteration_committed(
            state,
            canonical_iteration,
        ):
            return RecoveryDecision(
                CampaignPhase.STOP_CHECK,
                canonical_iteration,
                "STOP_CHECK: corrected over-advanced iteration from committed version mapping",
            )

    if (
        reference_data_version >= 0
        and model_version >= 0
        and active_iteration_committed(state, iteration)
    ):
        return RecoveryDecision(
            CampaignPhase.STOP_CHECK,
            iteration,
            "STOP_CHECK: active iteration is fully committed",
        )

    if reference_data_version >= 0 and reference_data_version > model_version:
        if _has_version(valid_reference_data_versions, reference_data_version):
            return RecoveryDecision(
                CampaignPhase.FEREBUS,
                int(reference_data_version),
                "FEREBUS: committed reference data is one version ahead of committed models",
                str(
                    ReferenceDataVersioning(
                        campaign / "QM_REFERENCE_DATA"
                    ).iteration_path(reference_data_version)
                ),
            )

    if existing_loaded:
        existing = _existing_phase_recovery(
            campaign,
            state,
            valid_reference_data_versions=valid_reference_data_versions,
            valid_model_versions=valid_model_versions,
            verification=verification,
            artifact_snapshot=artifact_snapshot,
        )
        if existing is not None:
            return existing

    current_handoff = _best_active_iteration_handoff(
        campaign,
        iteration,
        expected_campaign_uid=str(state.campaign_uid),
        verification=verification,
        artifact_snapshot=artifact_snapshot,
    )
    if current_handoff is not None:
        return current_handoff.decision
    if (
        reference_data_version >= 0
        and model_version >= 0
        and reference_data_version == model_version
        and _has_version(valid_reference_data_versions, reference_data_version)
        and _has_version(valid_model_versions, model_version)
        and _ok(_require_pool, campaign, verification=verification)
    ):
        return RecoveryDecision(
            CampaignPhase.SEED_SELECT,
            iteration,
            "SEED_SELECT: coherent committed models and trajectory pool are ready",
            "pool.xyz",
        )
    return None


def phase_recovery_contract_error(
    campaign_dir: Union[str, Path],
    state: CampaignState,
    *,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> Optional[str]:
    """Return a phase-specific input error for ``state``, or ``None``."""
    campaign = Path(campaign_dir)
    for label, check in _phase_contract_checks(
        campaign,
        state,
        verification=verification,
        artifact_snapshot=artifact_snapshot,
    ):
        error = _error(check)
        if error is not None:
            return label + ": " + error
    return None


def recovery_contract_status(
    campaign_dir: Union[str, Path],
    state: CampaignState,
    *,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a user-facing phase input contract summary."""
    campaign = Path(campaign_dir)
    phase = CampaignPhase(state.phase)
    iteration = int(getattr(state, "iteration", 0))
    required_inputs: List[str] = []
    trusted_inputs: List[str] = []
    missing_or_invalid_inputs: List[str] = []
    for label, check in _phase_contract_checks(
        campaign,
        state,
        verification=verification,
        artifact_snapshot=artifact_snapshot,
    ):
        required_inputs.append(label)
        error = _error(check)
        if error is None:
            trusted_inputs.append(label)
        else:
            missing_or_invalid_inputs.append(label + ": " + error)

    protected_artifacts = []
    try:
        for decision in staging_handoff_decisions(
            campaign,
            state,
            verification=verification,
            artifact_snapshot=artifact_snapshot,
        ):
            protected_artifacts.append({
                "phase": decision.phase.value,
                "iteration": int(decision.iteration),
                "path": str(decision.trusted_artifact or ""),
                "replacement_round": int(decision.replacement_round),
            })
    except Exception as exc:
        protected_artifacts.append({
            "phase": "UNKNOWN",
            "iteration": iteration,
            "path": (
                "staging inventory failed: "
                + type(exc).__name__
                + ": "
                + str(exc)[:160]
            ),
        })

    trusted_handoffs = []
    try:
        handoff_decisions = active_iteration_handoff_decisions(
            campaign,
            state,
            verification=verification,
            artifact_snapshot=artifact_snapshot,
        )
    except Exception as exc:
        handoff_decisions = []
        missing_or_invalid_inputs.append(
            "active-iteration handoff inventory: "
            + type(exc).__name__
            + ": "
            + str(exc)[:160]
        )
    for decision in handoff_decisions:
        trusted_handoffs.append({
            "phase": decision.phase.value,
            "iteration": int(decision.iteration),
            "path": str(decision.trusted_artifact or ""),
            "replacement_round": int(decision.replacement_round),
        })

    if not required_inputs and phase in {CampaignPhase.HALTED, CampaignPhase.DONE}:
        missing_or_invalid_inputs.append(
            "phase " + phase.value + " is not a runnable recovery phase"
        )

    return {
        "selected_phase": phase.value,
        "iteration": iteration,
        "contract_ok": not missing_or_invalid_inputs,
        "required_inputs": required_inputs,
        "trusted_inputs": trusted_inputs,
        "missing_or_invalid_inputs": missing_or_invalid_inputs,
        "trusted_handoffs": trusted_handoffs,
        "protected_artifacts": protected_artifacts,
    }


def validate_phase_recovery_contract(
    campaign_dir: Union[str, Path],
    state: CampaignState,
    *,
    verification: str = "metadata",
    artifact_snapshot: Optional[Any] = None,
) -> None:
    error = phase_recovery_contract_error(
        campaign_dir,
        state,
        verification=verification,
        artifact_snapshot=artifact_snapshot,
    )
    if error is not None:
        raise RecoveryContractError(
            "phase "
            + CampaignPhase(state.phase).value
            + " input contract invalid: "
            + error
        )
