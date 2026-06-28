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
from typing import List, Optional, Sequence, Union

from ..acquisition.trajectory_pool import TrajectoryPool
from ..handoff_manifests import (
    read_ariadne_results_manifest,
    read_phase_a_sample_manifest,
    read_phase_b_selection_manifest,
)
from . import input_staging as _stg
from .artifact_contracts import (
    verify_committed_model_version,
    verify_committed_training_version,
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


@dataclass(frozen=True)
class RecoveryHandoff:
    decision: RecoveryDecision
    priority: int
    kind: str


def iteration_dir(campaign_dir: Union[str, Path], iteration: int) -> Path:
    return (
        Path(campaign_dir)
        / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(iteration)).zfill(4))
    )


def active_iteration_committed(state: CampaignState, iteration: int) -> bool:
    """Return true when active iteration ``iteration`` is fully committed.

    Version 0 is the bootstrap set.  Active iteration ``i`` is represented by
    training/model version ``i + 1`` once APPEND and FEREBUS have both
    completed.
    """
    try:
        training_version = int(getattr(state, "training_set_version", -1))
        models_version = int(getattr(state, "models_version", -1))
    except (TypeError, ValueError):
        return False
    return min(training_version, models_version) >= int(iteration) + 1


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
    read_phase_a_sample_manifest(
        campaign / "3_DIVERSITY_SAMPLING" / "initial",
        require_nonempty=True,
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
    training_version: int,
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
        if int(training_version) == 0 and int(model_version) < 0:
            try:
                _require_training_version(campaign, 0)
                return
            except Exception as training_error:
                raise RecoveryContractError(
                    "INITIAL_FEREBUS requires either a valid initial AIMAll "
                    "handoff or committed bootstrap training version 0; "
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


def _require_ariadne_results(campaign: Path, iteration: int) -> None:
    read_ariadne_results_manifest(
        iteration_dir(campaign, iteration),
        expected_iteration=int(iteration),
        require_nonempty=True,
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


def _phase_b_final_count(campaign: Path, iteration: int) -> int:
    idir = iteration_dir(campaign, iteration)
    manifest = read_phase_b_selection_manifest(
        idir,
        expected_iteration=int(iteration),
        require_nonempty=True,
    )
    sample = idir / "phase_b_SAMPLE.xyz"
    if not sample.is_file():
        raise FileNotFoundError("Phase B sample xyz missing: " + str(sample))
    n_frames = _count_xyz_frames(sample)
    n_final = len(list(manifest.get("final") or []))
    if int(n_frames) != int(n_final):
        raise RecoveryContractError(
            "Phase B sample frame count "
            + str(int(n_frames))
            + " does not match final record count "
            + str(int(n_final))
        )
    return int(n_final)


def _require_phase_b(campaign: Path, iteration: int) -> None:
    _phase_b_final_count(campaign, int(iteration))


def _require_split(campaign: Path, iteration: int) -> None:
    path = iteration_dir(campaign, iteration) / "split.json"
    if not path.is_file():
        raise FileNotFoundError("split.json missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecoveryContractError("split.json unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise RecoveryContractError("split.json must be a JSON object")
    if int(data.get("iteration")) != int(iteration):
        raise RecoveryContractError("split.json iteration mismatch")
    train = data.get("train_indices")
    val = data.get("val_indices")
    holdout = data.get("holdout_indices")
    if not isinstance(train, list) or not isinstance(val, list) or not isinstance(holdout, list):
        raise RecoveryContractError("split.json train/val/holdout indices must be lists")
    def _index_set(values: Sequence[object], label: str) -> set:
        out = set()
        for value in values:
            if isinstance(value, bool):
                raise RecoveryContractError("split.json " + label + " indices must be integers")
            try:
                idx = int(value)
            except (TypeError, ValueError) as exc:
                raise RecoveryContractError("split.json " + label + " indices must be integers") from exc
            out.add(idx)
        if len(out) != len(values):
            raise RecoveryContractError("split.json " + label + " indices contain duplicates")
        return out

    train_set = _index_set(train, "train")
    val_set = _index_set(val, "validation")
    holdout_set = _index_set(holdout, "holdout")
    if train_set & val_set:
        raise RecoveryContractError("split.json train and validation indices overlap")
    if not holdout_set.issubset(val_set):
        raise RecoveryContractError("split.json holdout indices must be a subset of validation")
    n_final = _phase_b_final_count(campaign, int(iteration))
    allowed = set(range(int(n_final)))
    assigned = train_set | val_set
    if not assigned.issubset(allowed):
        raise RecoveryContractError("split.json indices are outside Phase B final record range")
    if assigned != allowed:
        raise RecoveryContractError("split.json train/validation indices do not cover every Phase B final record")


def _require_training_version(campaign: Path, version: int) -> None:
    verify_committed_training_version(campaign, int(version))


def _require_model_version(campaign: Path, version: int) -> None:
    verify_committed_model_version(campaign, int(version))


def _require_active_iteration_committed(state: CampaignState, iteration: int) -> None:
    if not active_iteration_committed(state, iteration):
        raise RecoveryContractError(
            "active iteration "
            + str(int(iteration))
            + " is not fully committed; STOP_CHECK would skip unfinished work"
        )


def _require_ferebus_needed(campaign: Path, training_version: int, model_version: int) -> None:
    if int(training_version) < 0:
        raise RecoveryContractError("FEREBUS requires a non-negative training version")
    _require_training_version(campaign, int(training_version))
    if int(model_version) >= int(training_version):
        raise RecoveryContractError(
            "FEREBUS model version "
            + str(int(model_version))
            + " is already at or ahead of training version "
            + str(int(training_version))
        )


def _active_iteration_for_training_version(training_version: int) -> int:
    return max(0, int(training_version) - 1)


def _require_ferebus_iteration(iteration: int, training_version: int) -> None:
    expected = _active_iteration_for_training_version(training_version)
    if int(iteration) != expected:
        raise RecoveryContractError(
            "FEREBUS iteration "
            + str(int(iteration))
            + " does not match training version "
            + str(int(training_version))
        )


def protected_staging_handoff(
    campaign_dir: Union[str, Path],
    *,
    iteration: int,
) -> Optional[RecoveryDecision]:
    """Return the consumer phase for a valid active quantum staging handoff."""
    campaign = Path(campaign_dir)
    if _ok(_require_iter_quantum, campaign, CampaignPhase.AIMALL, int(iteration)):
        return RecoveryDecision(
            CampaignPhase.APPEND,
            int(iteration),
            "APPEND: valid iterative AIMAll handoff exists",
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
        if _ok(_require_initial_quantum, campaign, CampaignPhase.INITIAL_AIMALL, 0):
            decisions.append(
                RecoveryDecision(
                    CampaignPhase.INITIAL_FEREBUS,
                    0,
                    "INITIAL_FEREBUS: valid initial AIMAll handoff exists",
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
    if _ok(_require_split, campaign, int(iteration)):
        return RecoveryHandoff(
            RecoveryDecision(
                CampaignPhase.GAUSSIAN,
                int(iteration),
                "GAUSSIAN: valid split handoff exists",
                str(iteration_dir(campaign, iteration) / "split.json"),
            ),
            40,
            "split",
        )
    if _ok(_require_phase_b, campaign, int(iteration)):
        return RecoveryHandoff(
            RecoveryDecision(
                CampaignPhase.SPLIT,
                int(iteration),
                "SPLIT: valid Phase B handoff exists",
                str(iteration_dir(campaign, iteration) / "PHASE_B_SELECTION.json"),
            ),
            30,
            "phase_b",
        )
    if _ok(_require_ariadne_results, campaign, int(iteration)):
        return RecoveryHandoff(
            RecoveryDecision(
                CampaignPhase.PHASE_B_POLUS,
                int(iteration),
                "PHASE_B_POLUS: valid ARIADNE results handoff exists",
                str(iteration_dir(campaign, iteration) / "ARIADNE_RESULTS.json"),
            ),
            20,
            "ariadne_results",
        )
    if _ok(_require_seeds, campaign, int(iteration)):
        return RecoveryHandoff(
            RecoveryDecision(
                CampaignPhase.ARIADNE_ARRAY,
                int(iteration),
                "ARIADNE_ARRAY: valid seed-selection handoff exists",
                str(iteration_dir(campaign, iteration) / "seeds_picked.json"),
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
    root = campaign / "7_ACTIVE_LEARNING"
    if not root.is_dir():
        return []
    decisions: List[RecoveryDecision] = []
    for path in sorted(root.glob("iteration-*")):
        if not path.is_dir() or path.is_symlink():
            continue
        suffix = path.name[len("iteration-"):]
        try:
            iteration = int(suffix)
        except ValueError:
            continue
        if active_iteration_committed(state, iteration):
            continue
        handoff = _best_active_iteration_handoff(campaign, iteration)
        if handoff is not None:
            decisions.append(handoff.decision)
    return decisions


def _single_decision_or_none(decisions: Sequence[RecoveryDecision]) -> Optional[RecoveryDecision]:
    if len(decisions) == 1:
        return decisions[0]
    return None


def _existing_phase_recovery(
    campaign: Path,
    state: CampaignState,
    *,
    valid_training_versions: Sequence[int],
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
    valid_training_versions: Sequence[int],
    valid_model_versions: Sequence[int],
    existing_loaded: bool,
    last_phase: Optional[str] = None,
    last_iteration: Optional[int] = None,
) -> Optional[RecoveryDecision]:
    """Choose the furthest safe re-entry phase from producer contracts."""
    campaign = Path(campaign_dir)
    iteration = int(getattr(state, "iteration", 0))
    if last_iteration is not None and not existing_loaded:
        try:
            iteration = int(last_iteration)
        except (TypeError, ValueError):
            pass

    training_version = int(getattr(state, "training_set_version", -1))
    model_version = int(getattr(state, "models_version", -1))

    # Bootstrap/no-committed-version path.
    if not valid_training_versions and not valid_model_versions:
        if _ok(_require_initial_quantum, campaign, CampaignPhase.INITIAL_AIMALL, iteration):
            return RecoveryDecision(
                CampaignPhase.INITIAL_FEREBUS,
                iteration,
                "INITIAL_FEREBUS: valid initial AIMAll handoff exists without committed models",
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
                "3_DIVERSITY_SAMPLING/initial/PHASE_A_SAMPLE.json",
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

    if training_version == 0 and model_version < 0:
        if _has_version(valid_training_versions, 0):
            return RecoveryDecision(
                CampaignPhase.INITIAL_FEREBUS,
                0,
                "INITIAL_FEREBUS: committed bootstrap training exists without model version 0",
                "5_TRAINING/iteration-0000",
            )

    if training_version == model_version and training_version >= 1:
        canonical_iteration = _active_iteration_for_training_version(training_version)
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
        training_version >= 0
        and model_version >= 0
        and active_iteration_committed(state, iteration)
    ):
        return RecoveryDecision(
            CampaignPhase.STOP_CHECK,
            iteration,
            "STOP_CHECK: active iteration is fully committed",
        )

    if training_version >= 0 and training_version > model_version:
        if _has_version(valid_training_versions, training_version):
            return RecoveryDecision(
                CampaignPhase.FEREBUS,
                max(0, training_version - 1),
                "FEREBUS: committed training is one version ahead of committed models",
                "5_TRAINING/iteration-" + str(training_version).zfill(4),
            )

    if existing_loaded:
        existing = _existing_phase_recovery(
            campaign,
            state,
            valid_training_versions=valid_training_versions,
            valid_model_versions=valid_model_versions,
        )
        if existing is not None:
            return existing

    current_handoff = _best_active_iteration_handoff(campaign, iteration)
    if current_handoff is not None:
        return current_handoff.decision
    if (
        training_version >= 0
        and model_version >= 0
        and training_version == model_version
        and _has_version(valid_training_versions, training_version)
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
    phase = CampaignPhase(state.phase)
    iteration = int(getattr(state, "iteration", 0))
    training_version = int(getattr(state, "training_set_version", -1))
    model_version = int(getattr(state, "models_version", -1))

    checks = {
        CampaignPhase.PHASE_A_POLUS: lambda: _require_pool(campaign),
        CampaignPhase.INITIAL_GAUSSIAN: lambda: _require_phase_a(campaign),
        CampaignPhase.INITIAL_AIMALL: lambda: _require_initial_quantum(
            campaign,
            CampaignPhase.INITIAL_GAUSSIAN,
            iteration,
        ),
        CampaignPhase.INITIAL_FEREBUS: lambda: _require_initial_ferebus_input(
            campaign,
            iteration,
            training_version,
            model_version,
        ),
        CampaignPhase.SEED_SELECT: lambda: (
            _require_training_version(campaign, training_version),
            _require_model_version(campaign, model_version),
            _require_pool(campaign),
        ),
        CampaignPhase.ARIADNE_ARRAY: lambda: _require_seeds(campaign, iteration),
        CampaignPhase.PHASE_B_POLUS: lambda: _require_ariadne_results(campaign, iteration),
        CampaignPhase.SPLIT: lambda: _require_phase_b(campaign, iteration),
        CampaignPhase.GAUSSIAN: lambda: (
            _require_phase_b(campaign, iteration),
            _require_split(campaign, iteration),
        ),
        CampaignPhase.AIMALL: lambda: _require_iter_quantum(
            campaign,
            CampaignPhase.GAUSSIAN,
            iteration,
        ),
        CampaignPhase.APPEND: lambda: _require_iter_quantum(
            campaign,
            CampaignPhase.AIMALL,
            iteration,
        ),
        CampaignPhase.FEREBUS: lambda: (
            _require_ferebus_needed(campaign, training_version, model_version),
            _require_ferebus_iteration(iteration, training_version),
        ),
        CampaignPhase.STOP_CHECK: lambda: _require_active_iteration_committed(
            state,
            iteration,
        ),
    }
    check = checks.get(phase)
    if check is None:
        return None
    return _error(check)


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
