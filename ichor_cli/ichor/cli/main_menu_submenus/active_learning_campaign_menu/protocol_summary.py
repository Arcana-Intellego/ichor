"""Readable summaries of the active-learning sampling protocol.

The summary is intentionally read-only. It gives an operator one compact view
of the knobs that materially affect seed choice, adversarial landing, and
Phase-B safety before a CSF4 daemon launch.
"""
from __future__ import annotations

from pathlib import Path

from ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu import (
    format_field_value,
)
from ichor.hpc.active_learning.config import CampaignConfig


def _line(label: str, value) -> str:
    return "-- " + label + ": " + format_field_value(value) + "\n"


def format_sampling_protocol_summary(config: CampaignConfig) -> str:
    seed = config.seed_selection
    calib = config.error_calibration
    acq = config.acquisition
    resources = config.resources
    gaussian = config.gaussian
    spectral = acq.spectral
    energy = acq.calibrated_energy
    fullspace = acq.fullspace_confinement
    stencils = acq.stencils
    safety = config.adversarial_safety
    gates = config.quality_gates
    phase_b = config.phase_b
    anti = config.anti_overlap

    lines = ["Sampling protocol summary:\n"]
    lines.append(_line("seed_selection.strategy", seed.strategy))
    lines.append(_line("seed_selection.bulk_fraction", seed.bulk_fraction))
    lines.append(
        _line(
            "seed_selection.d_optimal",
            "pool_multiplier="
            + str(seed.d_optimal_pool_multiplier)
            + ", score_power="
            + str(seed.d_optimal_score_power),
        )
    )
    lines.append(_line("error_calibration.enabled", calib.enabled))
    lines.append(_line("error_calibration.mode", calib.mode))
    lines.append(_line("error_calibration.apply_strength", calib.apply_strength))
    lines.append(_line("error_calibration.model_version_policy", calib.model_version_policy))
    lines.append(
        _line("error_calibration.min_records_to_apply", calib.min_records_to_apply)
    )
    lines.append(_line("acquisition.property_name", acq.property_name))
    lines.append(_line("acquisition.spectral.enabled", spectral.enabled))
    lines.append(_line("acquisition.spectral.mode", spectral.mode))
    lines.append(_line("acquisition.spectral.mode_weighting", spectral.mode_weighting))
    lines.append(_line("acquisition.spectral.lambda_spectral", spectral.lambda_spectral))
    lines.append(_line("acquisition.calibrated_energy.utility", energy.utility))
    lines.append(_line("acquisition.calibrated_energy.band_low_ha", energy.band_low_ha))
    lines.append(
        _line("acquisition.calibrated_energy.band_high_ha", energy.band_high_ha)
    )
    lines.append(
        _line(
            "acquisition.calibrated_energy.fallback_to_raw_variance",
            energy.fallback_to_raw_variance,
        )
    )
    lines.append(_line("acquisition.fullspace_confinement.enabled", fullspace.enabled))
    lines.append(
        _line(
            "acquisition.fullspace_confinement.lambda_residual",
            fullspace.lambda_residual,
        )
    )
    lines.append(
        _line("acquisition.fullspace_confinement.lambda_rmsd", fullspace.lambda_rmsd)
    )
    lines.append(
        _line(
            "acquisition.fullspace_confinement.residual_scale",
            fullspace.residual_scale,
        )
    )
    lines.append(
        _line(
            "acquisition.fullspace_confinement.rmsd_scale_ang",
            fullspace.rmsd_scale_ang,
        )
    )
    lines.append(
        _line(
            "acquisition.fullspace_confinement.min_residual_scale_ang",
            fullspace.min_residual_scale_ang,
        )
    )
    lines.append(
        _line(
            "acquisition.fullspace_confinement.failure_penalty",
            fullspace.failure_penalty,
        )
    )
    lines.append(
        _line(
            "acquisition.stencils.negative_curvature_policy",
            stencils.negative_curvature_policy,
        )
    )
    lines.append(
        _line(
            "acquisition.stencils.lambda_negative_curvature",
            stencils.lambda_negative_curvature,
        )
    )
    lines.append(_line("resources.partition", resources.partition))
    lines.append(_line("resources.mem_per_cpu", resources.mem_per_cpu))
    lines.append(
        _line("resources.array_concurrency_limit", resources.array_concurrency_limit)
    )
    lines.append(_line("gaussian.nproc", gaussian.nproc))
    lines.append(_line("gaussian.memory_mode", gaussian.memory_mode))
    lines.append(
        _line("gaussian.memory_fraction_of_slurm", gaussian.memory_fraction_of_slurm)
    )
    lines.append(_line("adversarial_safety.enabled", safety.enabled))
    lines.append(
        _line("adversarial_safety.reject_unsafe_landings", safety.reject_unsafe_landings)
    )
    lines.append(
        _line("adversarial_safety.max_whitened_distance", safety.max_whitened_distance)
    )
    lines.append(
        _line(
            "quality_gates.ariadne_max_displacement_ang",
            gates.ariadne_max_displacement_ang,
        )
    )
    lines.append(
        _line(
            "quality_gates.ariadne_min_pair_distance_ang",
            gates.ariadne_min_pair_distance_ang,
        )
    )
    lines.append(_line("phase_b.descriptor", phase_b.descriptor))
    lines.append(_line("phase_b.min_separation", phase_b.min_separation))
    lines.append(
        _line(
            "anti_overlap.post_ariadne_whitened_distance",
            str(anti.min_post_ariadne_whitened_distance)
            + " .. "
            + str(anti.max_post_ariadne_whitened_distance),
        )
    )
    return "".join(lines)


def format_saved_sampling_protocol_summary(campaign_dir: str | Path) -> str:
    campaign = Path(campaign_dir)
    yaml_path = campaign / "campaign.yaml"
    if not yaml_path.exists():
        return "No campaign.yaml at " + str(yaml_path)
    try:
        config = CampaignConfig.from_yaml(yaml_path)
    except Exception as exc:
        return "Failed to load " + str(yaml_path) + ": " + str(exc)
    return format_sampling_protocol_summary(config)
