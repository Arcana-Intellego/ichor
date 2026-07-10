"""Canonical active-learning campaign directory names."""

from __future__ import annotations

from pathlib import Path
from typing import Union


QM_REFERENCE_DATA_DIRNAME = "QM_REFERENCE_DATA"
LEGACY_TRAINING_DIRNAME = "5_TRAINING"
TRAINED_MODELS_DIRNAME = "6_TRAINED_MODELS"
ACTIVE_LEARNING_DIRNAME = "7_ACTIVE_LEARNING"


def qm_reference_data_dir(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / QM_REFERENCE_DATA_DIRNAME


def reject_legacy_training_layout(campaign_dir: Union[str, Path]) -> None:
    legacy = Path(campaign_dir) / LEGACY_TRAINING_DIRNAME
    if legacy.exists():
        raise RuntimeError(
            "unsupported legacy reference-data layout at "
            + str(legacy)
            + "; start a fresh campaign using "
            + QM_REFERENCE_DATA_DIRNAME
        )


__all__ = [
    "QM_REFERENCE_DATA_DIRNAME",
    "LEGACY_TRAINING_DIRNAME",
    "TRAINED_MODELS_DIRNAME",
    "ACTIVE_LEARNING_DIRNAME",
    "qm_reference_data_dir",
    "reject_legacy_training_layout",
]
