"""Committed-artifact contract checks used before daemon consumers glob files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from ..versioning.reference_data import ReferenceDataVersioning
from ..versioning.trained_models import resolve_trained_model_set
from ..layout import QM_REFERENCE_DATA_DIRNAME, TRAINED_MODELS_DIRNAME
from .state import CampaignPhase


class CommittedArtifactError(RuntimeError):
    """Raised when a state-referenced committed artifact cannot be trusted."""


_TRAINING_REQUIRED = {
    CampaignPhase.SEED_SELECT,
    CampaignPhase.ARIADNE_ARRAY,
    CampaignPhase.PHASE_B_POLUS,
    CampaignPhase.SPLIT,
    CampaignPhase.GAUSSIAN,
    CampaignPhase.AIMALL,
    CampaignPhase.ALLOCATION_CHECK,
    CampaignPhase.REPLACEMENT_GAUSSIAN,
    CampaignPhase.REPLACEMENT_AIMALL,
    CampaignPhase.APPEND,
    CampaignPhase.FEREBUS,
    CampaignPhase.STOP_CHECK,
}

_MODELS_REQUIRED = {
    CampaignPhase.SEED_SELECT,
    CampaignPhase.ARIADNE_ARRAY,
    CampaignPhase.PHASE_B_POLUS,
    CampaignPhase.SPLIT,
    CampaignPhase.GAUSSIAN,
    CampaignPhase.AIMALL,
    CampaignPhase.ALLOCATION_CHECK,
    CampaignPhase.REPLACEMENT_GAUSSIAN,
    CampaignPhase.REPLACEMENT_AIMALL,
    CampaignPhase.APPEND,
    CampaignPhase.STOP_CHECK,
}

_COHERENT_TRAINING_MODEL_REQUIRED = {
    CampaignPhase.SEED_SELECT,
    CampaignPhase.ARIADNE_ARRAY,
    CampaignPhase.PHASE_B_POLUS,
    CampaignPhase.SPLIT,
    CampaignPhase.GAUSSIAN,
    CampaignPhase.AIMALL,
    CampaignPhase.ALLOCATION_CHECK,
    CampaignPhase.REPLACEMENT_GAUSSIAN,
    CampaignPhase.REPLACEMENT_AIMALL,
    CampaignPhase.APPEND,
    CampaignPhase.STOP_CHECK,
}


def verify_committed_reference_data_version(
    campaign_dir: Union[str, Path],
    version: int,
    *,
    reference_data_dir_name: str = QM_REFERENCE_DATA_DIRNAME,
) -> None:
    try:
        ReferenceDataVersioning(
            Path(campaign_dir) / reference_data_dir_name
        ).resolve(int(version), verification="deep")
    except Exception as exc:
        raise CommittedArtifactError(
            "reference_data_version_invalid:"
            + str(int(version))
            + ": "
            + type(exc).__name__
            + ": "
            + str(exc)
        ) from exc


def verify_committed_model_version(
    campaign_dir: Union[str, Path],
    version: int,
    *,
    models_dir_name: str = TRAINED_MODELS_DIRNAME,
) -> None:
    try:
        model_set = resolve_trained_model_set(
            campaign_dir,
            int(version),
            verification="deep",
            trained_models_root=Path(campaign_dir) / models_dir_name,
        )
        from .model_contract import validate_ferebus_model_contract

        validate_ferebus_model_contract(
            model_set.root,
            committed=True,
            expected_version=int(version),
            trained_model_set=model_set,
        )
    except Exception as exc:
        raise CommittedArtifactError(
            "models_version_invalid:"
            + str(int(version))
            + ": "
            + type(exc).__name__
            + ": "
            + str(exc)
        ) from exc


def verify_state_referenced_artifacts(
    campaign_dir: Union[str, Path],
    state: Any,
    *,
    reference_data_dir_name: str = QM_REFERENCE_DATA_DIRNAME,
    models_dir_name: str = TRAINED_MODELS_DIRNAME,
    strict_models: bool = True,
) -> None:
    phase = CampaignPhase(state.phase)
    if phase in (CampaignPhase.DONE, CampaignPhase.HALTED):
        return
    campaign = Path(campaign_dir)

    train_version = int(getattr(state, "reference_data_version", -1))
    if phase is CampaignPhase.INITIAL_FEREBUS and train_version < 0:
        try:
            from ..point_allocation import point_allocation_path, read_point_allocation

            allocation = read_point_allocation(
                point_allocation_path(
                    campaign,
                    context="bootstrap",
                    iteration=0,
                )
            )
            if not bool((allocation.get("summary") or {}).get("complete", False)):
                raise ValueError("bootstrap point allocation is incomplete")
        except Exception as exc:
            raise CommittedArtifactError(
                "initial_ferebus_point_allocation_invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

    if phase in _TRAINING_REQUIRED and train_version < 0:
        raise CommittedArtifactError(
            "phase "
            + phase.value
            + " requires a committed reference-data version but state has "
            + str(train_version)
        )
    if train_version >= 0:
        train_dir = ReferenceDataVersioning(
            campaign / reference_data_dir_name
        ).iteration_path(train_version)
        if train_dir.is_dir():
            verify_committed_reference_data_version(
                campaign,
                train_version,
                reference_data_dir_name=reference_data_dir_name,
            )
        elif phase in _TRAINING_REQUIRED:
            raise CommittedArtifactError(
                "state references missing committed reference-data version "
                + str(train_version)
                + " required by phase "
                + phase.value
                + ": "
                + str(train_dir)
            )

    model_version = int(getattr(state, "models_version", -1))
    if strict_models and phase in _MODELS_REQUIRED and model_version < 0:
        raise CommittedArtifactError(
            "phase "
            + phase.value
            + " requires a committed model version but state has "
            + str(model_version)
        )
    if model_version >= 0:
        from ..versioning.trained_models import TrainedModelVersioning

        model_dir = TrainedModelVersioning(
            campaign / models_dir_name
        ).iteration_path(model_version)
        if model_dir.is_dir():
            if strict_models:
                verify_committed_model_version(
                    campaign,
                    model_version,
                    models_dir_name=models_dir_name,
                )
        elif strict_models and phase in _MODELS_REQUIRED:
            raise CommittedArtifactError(
                "state references missing committed model version "
                + str(model_version)
                + " required by phase "
                + phase.value
                + ": "
                + str(model_dir)
            )
    if strict_models and phase in _COHERENT_TRAINING_MODEL_REQUIRED:
        if train_version != model_version:
            raise CommittedArtifactError(
                "state reference-data/model version skew for phase "
                + phase.value
                + ": reference_data_version="
                + str(train_version)
                + ", models_version="
                + str(model_version)
            )
    if strict_models and min(train_version, model_version) >= 0:
        completed_through = min(train_version, model_version)
        if phase is CampaignPhase.STOP_CHECK and int(state.iteration) == completed_through:
            completed_through -= 1
        try:
            from ..versioning.sampling_iterations import verify_sampling_chain

            verify_sampling_chain(
                campaign,
                completed_through,
                expected_campaign_uid=str(state.campaign_uid),
            )
        except Exception as exc:
            raise CommittedArtifactError(
                "sampling_iteration_chain_invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc


def artifact_manifest_status(
    campaign_dir: Union[str, Path],
    state: Any,
    *,
    reference_data_dir_name: str = QM_REFERENCE_DATA_DIRNAME,
    models_dir_name: str = TRAINED_MODELS_DIRNAME,
    strict_models: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"reference_data": {}, "models": {}}
    checks: List[tuple] = [
        (
            "reference_data",
            int(getattr(state, "reference_data_version", -1)),
            lambda version: verify_committed_reference_data_version(
                campaign_dir,
                version,
                reference_data_dir_name=reference_data_dir_name,
            ),
        ),
    ]
    if strict_models:
        checks.append(
            (
                "models",
                int(getattr(state, "models_version", -1)),
                lambda version: verify_committed_model_version(
                    campaign_dir,
                    version,
                    models_dir_name=models_dir_name,
                ),
            )
        )
    else:
        out["models"] = {
            "version": int(getattr(state, "models_version", -1)),
            "strict": False,
            "ok": None,
            "errors": [],
        }
    for key, version, checker in checks:
        item = {"version": version, "strict": True, "ok": None, "errors": []}
        if version < 0:
            item["ok"] = True
        else:
            try:
                checker(version)
                item["ok"] = True
            except Exception as exc:
                item["ok"] = False
                item["errors"].append(type(exc).__name__ + ": " + str(exc))
        out[key] = item
    return out


def state_artifact_contract_status(
    campaign_dir: Union[str, Path],
    state: Any,
    *,
    reference_data_dir_name: str = QM_REFERENCE_DATA_DIRNAME,
    models_dir_name: str = TRAINED_MODELS_DIRNAME,
    strict_models: bool = True,
) -> Dict[str, Any]:
    """Return the full state/artefact contract status for operator output.

    ``artifact_manifest_status`` reports the independent training and model
    version checks. This helper reports the combined producer/consumer
    contract that the live daemon enforces before phase entry, including
    bootstrap ``INITIAL_FEREBUS`` handoffs and reference-data/model version skew.
    """
    try:
        phase = CampaignPhase(state.phase).value
    except Exception:
        phase = str(getattr(state, "phase", "UNKNOWN"))
    payload: Dict[str, Any] = {
        "ok": None,
        "phase": phase,
        "reference_data_version": int(getattr(state, "reference_data_version", -1)),
        "models_version": int(getattr(state, "models_version", -1)),
        "strict_models": bool(strict_models),
        "errors": [],
    }
    try:
        verify_state_referenced_artifacts(
            campaign_dir,
            state,
            reference_data_dir_name=reference_data_dir_name,
            models_dir_name=models_dir_name,
            strict_models=strict_models,
        )
        payload["ok"] = True
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        payload["ok"] = False
        payload["error"] = error
        payload["errors"] = [error]
    return payload
