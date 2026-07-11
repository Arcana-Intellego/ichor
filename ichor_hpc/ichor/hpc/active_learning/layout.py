"""Canonical active-learning campaign paths and identifiers."""

from __future__ import annotations

import re
from numbers import Integral
from pathlib import Path
from typing import Union


QM_REFERENCE_DATA_DIRNAME = "QM_REFERENCE_DATA"
LEGACY_TRAINING_DIRNAME = "5_TRAINING"
TRAINED_MODELS_DIRNAME = "TRAINED_MODELS"
LEGACY_TRAINED_MODELS_DIRNAME = "6_TRAINED_MODELS"
BOOTSTRAP_DIRNAME = "BOOTSTRAP"
LEGACY_BOOTSTRAP_DIRNAME = "3_DIVERSITY_SAMPLING"
ACTIVE_LEARNING_DIRNAME = "ACTIVE_LEARNING"
LEGACY_ACTIVE_LEARNING_DIRNAME = "7_ACTIVE_LEARNING"
COMMITTED_VERSION_NAME_WIDTH = 6
ACTIVE_ITERATION_NAME_WIDTH = 6
SEED_ID_NAME_WIDTH = 6

_ACTIVE_ITERATION_RE = re.compile(r"^iteration-([0-9]{6,})$")
_SEED_DIRECTORY_RE = re.compile(r"^seed-([0-9]{6,})$")


def _positive_identifier(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(label + " must be an integer")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(label + " must be >= 1")
    return parsed


def qm_reference_data_dir(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / QM_REFERENCE_DATA_DIRNAME


def trained_models_dir(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / TRAINED_MODELS_DIRNAME


def bootstrap_dir(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / BOOTSTRAP_DIRNAME


def bootstrap_selection_dir(campaign_dir: Union[str, Path]) -> Path:
    return bootstrap_dir(campaign_dir) / "selection"


def bootstrap_allocation_dir(campaign_dir: Union[str, Path]) -> Path:
    return bootstrap_dir(campaign_dir) / "allocation"


def active_learning_dir(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / ACTIVE_LEARNING_DIRNAME


def active_iteration_name(iteration: int) -> str:
    value = _positive_identifier(iteration, "active iteration")
    return "iteration-" + str(value).zfill(ACTIVE_ITERATION_NAME_WIDTH)


def parse_active_iteration_name(name: str) -> int:
    match = _ACTIVE_ITERATION_RE.fullmatch(str(name))
    if match is None:
        raise ValueError("invalid active-iteration directory name: " + repr(name))
    value = int(match.group(1))
    if value < 1 or active_iteration_name(value) != str(name):
        raise ValueError("noncanonical active-iteration directory name: " + repr(name))
    return value


def active_iteration_dir(
    campaign_dir: Union[str, Path],
    iteration: int,
) -> Path:
    return active_learning_dir(campaign_dir) / active_iteration_name(iteration)


def active_protocol_dir(iteration_dir: Union[str, Path]) -> Path:
    return Path(iteration_dir) / "protocol"


def active_seed_selection_dir(iteration_dir: Union[str, Path]) -> Path:
    return Path(iteration_dir) / "seed_selection"


def active_ariadne_dir(iteration_dir: Union[str, Path]) -> Path:
    return Path(iteration_dir) / "ariadne"


def active_phase_b_dir(iteration_dir: Union[str, Path]) -> Path:
    return Path(iteration_dir) / "phase_b"


def active_allocation_dir(iteration_dir: Union[str, Path]) -> Path:
    return Path(iteration_dir) / "allocation"


def active_calibration_dir(iteration_dir: Union[str, Path]) -> Path:
    return Path(iteration_dir) / "calibration"


def seed_directory_name(seed_id: int) -> str:
    value = _positive_identifier(seed_id, "seed_id")
    return "seed-" + str(value).zfill(SEED_ID_NAME_WIDTH)


def parse_seed_directory_name(name: str) -> int:
    match = _SEED_DIRECTORY_RE.fullmatch(str(name))
    if match is None:
        raise ValueError("invalid seed directory name: " + repr(name))
    value = int(match.group(1))
    if value < 1 or seed_directory_name(value) != str(name):
        raise ValueError("noncanonical seed directory name: " + repr(name))
    return value


def ariadne_seeds_dir(iteration_dir: Union[str, Path]) -> Path:
    return active_ariadne_dir(iteration_dir) / "seeds"


def ariadne_seed_dir(
    iteration_dir: Union[str, Path],
    seed_id: int,
) -> Path:
    return ariadne_seeds_dir(iteration_dir) / seed_directory_name(seed_id)


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


def reject_legacy_sampling_layout(campaign_dir: Union[str, Path]) -> None:
    campaign = Path(campaign_dir)
    legacy_roots = (
        campaign / LEGACY_BOOTSTRAP_DIRNAME,
        campaign / LEGACY_ACTIVE_LEARNING_DIRNAME,
    )
    canonical_roots = (
        bootstrap_dir(campaign),
        active_learning_dir(campaign),
    )
    for legacy in legacy_roots:
        if legacy.exists() or legacy.is_symlink():
            qualifier = (
                " alongside a canonical root"
                if any(root.exists() or root.is_symlink() for root in canonical_roots)
                else ""
            )
            raise RuntimeError(
                "unsupported legacy sampling layout at "
                + str(legacy)
                + qualifier
                + "; start a fresh campaign using "
                + BOOTSTRAP_DIRNAME
                + " and "
                + ACTIVE_LEARNING_DIRNAME
            )
    active_root = active_learning_dir(campaign)
    if active_root.is_symlink():
        raise RuntimeError("ACTIVE_LEARNING root must not be a symlink")
    if active_root.exists() and not active_root.is_dir():
        raise RuntimeError("ACTIVE_LEARNING root is not a directory")
    if active_root.is_dir():
        for child in active_root.iterdir():
            if child.name.startswith("iteration-"):
                if child.is_symlink() or not child.is_dir():
                    raise RuntimeError(
                        "invalid active-iteration entry: " + str(child)
                    )
                try:
                    parse_active_iteration_name(child.name)
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc


def reject_legacy_campaign_layout(campaign_dir: Union[str, Path]) -> None:
    reject_legacy_training_layout(campaign_dir)
    reject_legacy_model_layout(campaign_dir)
    reject_legacy_sampling_layout(campaign_dir)


__all__ = [
    "QM_REFERENCE_DATA_DIRNAME",
    "LEGACY_TRAINING_DIRNAME",
    "TRAINED_MODELS_DIRNAME",
    "LEGACY_TRAINED_MODELS_DIRNAME",
    "BOOTSTRAP_DIRNAME",
    "LEGACY_BOOTSTRAP_DIRNAME",
    "ACTIVE_LEARNING_DIRNAME",
    "LEGACY_ACTIVE_LEARNING_DIRNAME",
    "COMMITTED_VERSION_NAME_WIDTH",
    "ACTIVE_ITERATION_NAME_WIDTH",
    "SEED_ID_NAME_WIDTH",
    "qm_reference_data_dir",
    "trained_models_dir",
    "bootstrap_dir",
    "bootstrap_selection_dir",
    "bootstrap_allocation_dir",
    "active_learning_dir",
    "active_iteration_name",
    "parse_active_iteration_name",
    "active_iteration_dir",
    "active_protocol_dir",
    "active_seed_selection_dir",
    "active_ariadne_dir",
    "active_phase_b_dir",
    "active_allocation_dir",
    "active_calibration_dir",
    "seed_directory_name",
    "parse_seed_directory_name",
    "ariadne_seeds_dir",
    "ariadne_seed_dir",
    "reject_legacy_training_layout",
    "reject_legacy_model_layout",
    "reject_legacy_sampling_layout",
    "reject_legacy_campaign_layout",
]
