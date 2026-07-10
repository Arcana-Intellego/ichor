"""Committed-artifact contract checks used before daemon consumers glob files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from ..versioning.versioned_directory import VersionedDirectory
from ..versioning.reference_data import ReferenceDataVersioning
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


def _versioning(campaign_dir: Union[str, Path], dirname: str) -> VersionedDirectory:
    return VersionedDirectory(Path(campaign_dir) / dirname)


def verify_committed_reference_data_version(
    campaign_dir: Union[str, Path],
    version: int,
    *,
    reference_data_dir_name: str = "QM_REFERENCE_DATA",
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
    models_dir_name: str = "6_TRAINED_MODELS",
) -> None:
    try:
        v_models = _versioning(campaign_dir, models_dir_name)
        v_models.verify_committed(int(version))
        from .model_contract import validate_ferebus_model_contract

        validate_ferebus_model_contract(
            v_models.iteration_path(int(version)),
            committed=True,
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
    reference_data_dir_name: str = "QM_REFERENCE_DATA",
    models_dir_name: str = "6_TRAINED_MODELS",
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
        train_dir = (
            campaign
            / reference_data_dir_name
            / ("iteration-" + str(train_version).zfill(4))
        )
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
        model_dir = (
            campaign
            / models_dir_name
            / ("iteration-" + str(model_version).zfill(4))
        )
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


def artifact_manifest_status(
    campaign_dir: Union[str, Path],
    state: Any,
    *,
    reference_data_dir_name: str = "QM_REFERENCE_DATA",
    models_dir_name: str = "6_TRAINED_MODELS",
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
    reference_data_dir_name: str = "QM_REFERENCE_DATA",
    models_dir_name: str = "6_TRAINED_MODELS",
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
