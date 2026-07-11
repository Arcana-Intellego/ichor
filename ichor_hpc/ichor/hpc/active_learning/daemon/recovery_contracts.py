"""Phase-aware recovery contracts for active-learning reconcile.

The daemon state machine is intentionally simple, but recovery is only safe
when the selected re-entry phase has the producer artefacts that phase
consumes.  These helpers keep that phase-specific knowledge out of the CLI
and out of the broad committed-version checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    """Return true once APPEND has committed this iteration's QM delta."""
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


def _require_pool(campaign: Path) -> None:
    pool = TrajectoryPool.load(campaign)
    if pool.manifest.natoms <= 0:
        raise RecoveryContractError("trajectory pool atom count is not positive")


def _require_phase_a(campaign: Path) -> None:
    from ..layout import bootstrap_selection_dir

    read_phase_a_sample_manifest(
        bootstrap_selection_dir(campaign),
        require_nonempty=True,
    )


def _require_point_allocation(
    campaign: Path,
    *,
    context: str,
    iteration: int,
    complete: Optional[bool] = None,
) -> Dict[str, Any]:
    from ..point_allocation import point_allocation_path, read_point_allocation

    path = point_allocation_path(
        campaign,
        context=str(context),
        iteration=int(iteration),
    )
    payload = read_point_allocation(path)
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
    )
    return path


def _require_allocation_check_ready(
    campaign: Path,
    *,
    context: str,
    iteration: int,
) -> None:
    from ..point_allocation import pending_attempts

    payload = _require_point_allocation(
        campaign,
        context=str(context),
        iteration=int(iteration),
    )
    pending = pending_attempts(payload)
    if pending:
        rounds = {int(record.get("round", -1)) for record in pending}
        if rounds == {0}:
            raise RecoveryContractError(
                "primary QM outcomes have not yet been recorded in point allocation"
            )


def _require_replacement_gaussian_handoff(
    campaign: Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
) -> None:
    round_dir = _require_replacement_sample(
        campaign,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
    )
    phase = (
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN
        if context == "bootstrap"
        else CampaignPhase.REPLACEMENT_GAUSSIAN
    )
    _stg.read_quantum_acceptance_manifest(
        round_dir,
        expected_phase=phase.value,
        expected_iteration=int(iteration),
        require_nonempty=False,
        require_points_file_membership=True,
    )


def _require_initial_quantum(campaign: Path, phase: CampaignPhase, iteration: int) -> None:
    _stg.read_quantum_acceptance_manifest(
        campaign / ".DATA" / "STAGING" / "initial",
        expected_phase=phase.value,
        expected_iteration=int(iteration),
        require_nonempty=True,
        require_points_file_membership=True,
    )


def _require_initial_ferebus_input(
    campaign: Path,
    iteration: int,
    reference_data_version: int,
    model_version: int,
) -> None:
    try:
        _require_initial_quantum(
            campaign,
            CampaignPhase.INITIAL_AIMALL,
            int(iteration),
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


def _require_iter_quantum(campaign: Path, phase: CampaignPhase, iteration: int) -> None:
    _stg.read_quantum_acceptance_manifest(
        campaign / ".DATA" / "STAGING" / ("iter_" + str(int(iteration))),
        expected_phase=phase.value,
        expected_iteration=int(iteration),
        require_nonempty=True,
        require_points_file_membership=True,
    )


def _require_seeds(campaign: Path, iteration: int) -> None:
    from ..handoff_manifests import load_seeds_picked

    load_seeds_picked(iteration_dir(campaign, iteration), expected_iteration=int(iteration))


def _require_ariadne_results(
    campaign: Path,
    iteration: int,
    expected_campaign_uid: Optional[str] = None,
) -> None:
    from ..config import CampaignConfig
    from .config_lock import canonical_config, config_fingerprint
    from ..handoff_manifests import read_ariadne_batch_decision

    read_ariadne_results_manifest(
        iteration_dir(campaign, iteration),
        expected_iteration=int(iteration),
        require_nonempty=True,
        accept_legacy_missing_landing_safety=False,
    )
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    read_ariadne_batch_decision(
        iteration_dir(campaign, iteration),
        expected_iteration=int(iteration),
        expected_campaign_uid=expected_campaign_uid,
        expected_config_sha256=config_fingerprint(canonical_config(config)),
        require_accepted=True,
    )


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


def _phase_b_final_count(
    campaign: Path,
    iteration: int,
    expected_campaign_uid: Optional[str] = None,
) -> int:
    idir = iteration_dir(campaign, iteration)
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
) -> None:
    _phase_b_final_count(
        campaign,
        int(iteration),
        expected_campaign_uid=expected_campaign_uid,
    )


def _require_split(campaign: Path, iteration: int) -> None:
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


def _require_reference_data_version(campaign: Path, version: int) -> None:
    verify_committed_reference_data_version(campaign, int(version))


def _require_model_version(campaign: Path, version: int) -> None:
    verify_committed_model_version(campaign, int(version))


def _require_active_iteration_committed(state: CampaignState, iteration: int) -> None:
    if not active_iteration_committed(state, iteration):
        raise RecoveryContractError(
            "active iteration "
            + str(int(iteration))
            + " is not fully committed; STOP_CHECK would skip unfinished work"
        )


def _require_ferebus_needed(campaign: Path, reference_data_version: int, model_version: int) -> None:
    if int(reference_data_version) < 0:
        raise RecoveryContractError("FEREBUS requires a non-negative reference-data version")
    _require_reference_data_version(campaign, int(reference_data_version))
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
) -> Optional[RecoveryDecision]:
    from ..point_allocation import pending_attempts

    try:
        allocation = _require_point_allocation(
            campaign,
            context=str(context),
            iteration=int(iteration),
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
        return RecoveryDecision(
            CampaignPhase.INITIAL_FEREBUS if context == "bootstrap" else CampaignPhase.APPEND,
            int(iteration),
            ("INITIAL_FEREBUS" if context == "bootstrap" else "APPEND")
            + ": exact point allocation is complete",
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
    try:
        round_dir = _require_replacement_sample(
            campaign,
            context=str(context),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
        )
    except Exception:
        return RecoveryDecision(
            CampaignPhase.INITIAL_ALLOCATION_CHECK
            if context == "bootstrap"
            else CampaignPhase.ALLOCATION_CHECK,
            int(iteration),
            "point-allocation check: pending replacement allocation needs sample repair",
            allocation_artifact,
            replacement_round=int(replacement_round),
        )
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
        require_points_file_membership=True,
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
) -> Optional[RecoveryDecision]:
    """Return the consumer phase for a valid active quantum staging handoff."""
    campaign = Path(campaign_dir)
    allocation_decision = _allocation_recovery_decision(
        campaign,
        context="active",
        iteration=int(iteration),
    )
    if allocation_decision is not None and allocation_decision.phase in {
        CampaignPhase.ALLOCATION_CHECK,
        CampaignPhase.REPLACEMENT_GAUSSIAN,
        CampaignPhase.REPLACEMENT_AIMALL,
        CampaignPhase.APPEND,
    }:
        return RecoveryDecision(
            allocation_decision.phase,
            allocation_decision.iteration,
            allocation_decision.reason,
            ".DATA/STAGING/iter_" + str(int(iteration)),
            replacement_round=int(allocation_decision.replacement_round),
        )
    if _ok(_require_iter_quantum, campaign, CampaignPhase.AIMALL, int(iteration)):
        return RecoveryDecision(
            CampaignPhase.AIMALL,
            int(iteration),
            "AIMALL: acceptance handoff exists and point-allocation recording must be verified",
            ".DATA/STAGING/iter_" + str(int(iteration)),
        )
    if _ok(_require_iter_quantum, campaign, CampaignPhase.GAUSSIAN, int(iteration)):
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
) -> List[RecoveryDecision]:
    """Return valid quantum staging handoffs that must not be archived."""
    campaign = Path(campaign_dir)
    decisions: List[RecoveryDecision] = []
    models_version = int(getattr(state, "models_version", -1))
    if include_committed or models_version < 0:
        allocation_decision = _allocation_recovery_decision(
            campaign,
            context="bootstrap",
            iteration=0,
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
        elif _ok(_require_initial_quantum, campaign, CampaignPhase.INITIAL_AIMALL, 0):
            decisions.append(
                RecoveryDecision(
                    CampaignPhase.INITIAL_AIMALL,
                    0,
                    "INITIAL_AIMALL: acceptance handoff exists and point-allocation recording must be verified",
                    ".DATA/STAGING/initial",
                )
            )
        elif _ok(_require_initial_quantum, campaign, CampaignPhase.INITIAL_GAUSSIAN, 0):
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
        decision = protected_staging_handoff(campaign, iteration=iteration)
        if decision is not None:
            decisions.append(decision)
    return decisions


def _best_active_iteration_handoff(campaign: Path, iteration: int) -> Optional[RecoveryHandoff]:
    allocation_decision = _allocation_recovery_decision(
        campaign,
        context="active",
        iteration=int(iteration),
    )
    if allocation_decision is not None:
        return RecoveryHandoff(
            allocation_decision,
            50,
            "point_allocation",
        )
    if _ok(_require_split, campaign, int(iteration)):
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
    if _ok(_require_phase_b, campaign, int(iteration)):
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
    if _ok(_require_ariadne_results, campaign, int(iteration)):
        from ..handoff_manifests import ariadne_results_path

        return RecoveryHandoff(
            RecoveryDecision(
                CampaignPhase.PHASE_B_POLUS,
                int(iteration),
                "PHASE_B_POLUS: valid ARIADNE results handoff exists",
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
            # All per-iteration handoffs through APPEND are now historical.
            # Recovery must evaluate the reference-data/model skew and select
            # FEREBUS rather than replaying a completed allocation.
            continue
        handoff = _best_active_iteration_handoff(campaign, iteration)
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
) -> List[Tuple[str, Callable[[], None]]]:
    phase = CampaignPhase(state.phase)
    iteration = int(getattr(state, "iteration", 0))
    reference_data_version = int(getattr(state, "reference_data_version", -1))
    model_version = int(getattr(state, "models_version", -1))

    checks: Dict[CampaignPhase, List[Tuple[str, Callable[[], None]]]] = {
        CampaignPhase.PHASE_A_POLUS: [
            ("trajectory pool", lambda: _require_pool(campaign)),
        ],
        CampaignPhase.INITIAL_GAUSSIAN: [
            ("Phase A sample", lambda: _require_phase_a(campaign)),
        ],
        CampaignPhase.INITIAL_AIMALL: [
            (
                "initial Gaussian handoff",
                lambda: _require_initial_quantum(
                    campaign,
                    CampaignPhase.INITIAL_GAUSSIAN,
                    iteration,
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
                ),
            ),
        ],
        CampaignPhase.INITIAL_FEREBUS: [
            (
                "complete bootstrap point allocation",
                lambda: _require_point_allocation(
                    campaign, context="bootstrap", iteration=0, complete=True,
                ),
            ),
        ],
        CampaignPhase.SEED_SELECT: [
            (
                "committed reference-data version " + str(reference_data_version),
                lambda: _require_reference_data_version(campaign, reference_data_version),
            ),
            (
                "committed model version " + str(model_version),
                lambda: _require_model_version(campaign, model_version),
            ),
            ("trajectory pool", lambda: _require_pool(campaign)),
        ],
        CampaignPhase.ARIADNE_ARRAY: [
            ("seed_selection/SELECTION.json", lambda: _require_seeds(campaign, iteration)),
        ],
        CampaignPhase.PHASE_B_POLUS: [
            (
                "ariadne/RESULTS.json",
                lambda: _require_ariadne_results(
                    campaign,
                    iteration,
                    str(state.campaign_uid),
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
                ),
            ),
            ("allocation/SPLIT_RECEIPT.json", lambda: _require_split(campaign, iteration)),
        ],
        CampaignPhase.AIMALL: [
            (
                "iterative Gaussian handoff",
                lambda: _require_iter_quantum(
                    campaign,
                    CampaignPhase.GAUSSIAN,
                    iteration,
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
                ),
            ),
        ],
        CampaignPhase.APPEND: [
            (
                "complete active point allocation",
                lambda: _require_point_allocation(
                    campaign,
                    context="active",
                    iteration=iteration,
                    complete=True,
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
    err = phase_recovery_contract_error(campaign, state)
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
        )
        if allocation_decision is not None:
            return allocation_decision
        if _ok(_require_initial_quantum, campaign, CampaignPhase.INITIAL_AIMALL, iteration):
            return RecoveryDecision(
                CampaignPhase.INITIAL_AIMALL,
                iteration,
                "INITIAL_AIMALL: acceptance exists and point-allocation recording must be verified",
                ".DATA/STAGING/initial",
            )
        if _ok(_require_initial_quantum, campaign, CampaignPhase.INITIAL_GAUSSIAN, iteration):
            return RecoveryDecision(
                CampaignPhase.INITIAL_AIMALL,
                iteration,
                "INITIAL_AIMALL: valid initial Gaussian handoff exists without committed models",
                ".DATA/STAGING/initial",
            )
        if _ok(_require_phase_a, campaign):
            return RecoveryDecision(
                CampaignPhase.INITIAL_GAUSSIAN,
                iteration,
                "INITIAL_GAUSSIAN: valid Phase A sample exists without committed models",
                "BOOTSTRAP/selection/SELECTION.json",
            )
        if (
            bool(last_phase_retryable)
            and str(last_phase or "") == CampaignPhase.PHASE_A_POLUS.value
            and _ok(_require_pool, campaign)
        ):
            return RecoveryDecision(
                CampaignPhase.PHASE_A_POLUS,
                iteration,
                "PHASE_A_POLUS: retryable pre-bootstrap phase has a valid trajectory pool input",
                ".DATA/TRAJECTORY/pool.xyz",
            )
        if existing_loaded:
            try:
                existing_phase = CampaignPhase(state.phase)
            except Exception:
                existing_phase = CampaignPhase.HALTED
            if existing_phase is CampaignPhase.PHASE_A_POLUS and _ok(_require_pool, campaign):
                return RecoveryDecision(
                    CampaignPhase.PHASE_A_POLUS,
                    iteration,
                    "PHASE_A_POLUS: existing phase has a valid trajectory pool input",
                    ".DATA/TRAJECTORY/pool.xyz",
                )
        return None

    staging_decision = _single_decision_or_none(staging_handoff_decisions(campaign, state))
    if staging_decision is not None:
        return staging_decision

    handoff_decision = _single_decision_or_none(active_iteration_handoff_decisions(campaign, state))
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
            "BOOTSTRAP/BOOTSTRAP_MANIFEST.json",
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
        )
        if existing is not None:
            return existing

    current_handoff = _best_active_iteration_handoff(campaign, iteration)
    if current_handoff is not None:
        return current_handoff.decision
    if (
        reference_data_version >= 0
        and model_version >= 0
        and reference_data_version == model_version
        and _has_version(valid_reference_data_versions, reference_data_version)
        and _has_version(valid_model_versions, model_version)
        and _ok(_require_pool, campaign)
    ):
        return RecoveryDecision(
            CampaignPhase.SEED_SELECT,
            iteration,
            "SEED_SELECT: coherent committed models and trajectory pool are ready",
            ".DATA/TRAJECTORY/pool.xyz",
        )
    return None


def phase_recovery_contract_error(
    campaign_dir: Union[str, Path],
    state: CampaignState,
) -> Optional[str]:
    """Return a phase-specific input error for ``state``, or ``None``."""
    campaign = Path(campaign_dir)
    for label, check in _phase_contract_checks(campaign, state):
        error = _error(check)
        if error is not None:
            return label + ": " + error
    return None


def recovery_contract_status(
    campaign_dir: Union[str, Path],
    state: CampaignState,
) -> Dict[str, Any]:
    """Return an operator-facing phase input contract summary."""
    campaign = Path(campaign_dir)
    phase = CampaignPhase(state.phase)
    iteration = int(getattr(state, "iteration", 0))
    required_inputs: List[str] = []
    trusted_inputs: List[str] = []
    missing_or_invalid_inputs: List[str] = []
    for label, check in _phase_contract_checks(campaign, state):
        required_inputs.append(label)
        error = _error(check)
        if error is None:
            trusted_inputs.append(label)
        else:
            missing_or_invalid_inputs.append(label + ": " + error)

    protected_artifacts = []
    try:
        for decision in staging_handoff_decisions(campaign, state):
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
        handoff_decisions = active_iteration_handoff_decisions(campaign, state)
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
) -> None:
    error = phase_recovery_contract_error(campaign_dir, state)
    if error is not None:
        raise RecoveryContractError(
            "phase "
            + CampaignPhase(state.phase).value
            + " input contract invalid: "
            + error
        )
