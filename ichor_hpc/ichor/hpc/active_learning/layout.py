"""Canonical active-learning campaign directory names."""

from __future__ import annotations

from pathlib import Path
from typing import Union


QM_REFERENCE_DATA_DIRNAME = "QM_REFERENCE_DATA"
LEGACY_TRAINING_DIRNAME = "5_TRAINING"
TRAINED_MODELS_DIRNAME = "TRAINED_MODELS"
LEGACY_TRAINED_MODELS_DIRNAME = "6_TRAINED_MODELS"
ACTIVE_LEARNING_DIRNAME = "7_ACTIVE_LEARNING"
COMMITTED_VERSION_NAME_WIDTH = 6


def qm_reference_data_dir(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / QM_REFERENCE_DATA_DIRNAME


def trained_models_dir(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / TRAINED_MODELS_DIRNAME


def reject_legacy_training_layout(campaign_dir: Union[str, Path]) -> None:
    legacy = Path(campaign_dir) / LEGACY_TRAINING_DIRNAME
    if legacy.exists():
        raise RuntimeError(
            "unsupported legacy reference-data layout at "
            + str(legacy)
            + "; start a fresh campaign using "
            + QM_REFERENCE_DATA_DIRNAME
        )


def reject_legacy_model_layout(campaign_dir: Union[str, Path]) -> None:
    campaign = Path(campaign_dir)
    legacy = campaign / LEGACY_TRAINED_MODELS_DIRNAME
    canonical = trained_models_dir(campaign)
    if legacy.exists():
        qualifier = " alongside " + str(canonical) if canonical.exists() else ""
        raise RuntimeError(
            "unsupported legacy trained-model layout at "
            + str(legacy)
            + qualifier
            + "; start a fresh campaign using "
            + TRAINED_MODELS_DIRNAME
        )


def reject_legacy_campaign_layout(campaign_dir: Union[str, Path]) -> None:
    reject_legacy_training_layout(campaign_dir)
    reject_legacy_model_layout(campaign_dir)


__all__ = [
    "QM_REFERENCE_DATA_DIRNAME",
    "LEGACY_TRAINING_DIRNAME",
    "TRAINED_MODELS_DIRNAME",
    "LEGACY_TRAINED_MODELS_DIRNAME",
    "ACTIVE_LEARNING_DIRNAME",
    "COMMITTED_VERSION_NAME_WIDTH",
    "qm_reference_data_dir",
    "trained_models_dir",
    "reject_legacy_training_layout",
    "reject_legacy_model_layout",
    "reject_legacy_campaign_layout",
]
