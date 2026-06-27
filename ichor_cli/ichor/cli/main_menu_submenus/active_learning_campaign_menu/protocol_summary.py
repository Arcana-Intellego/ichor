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


def _walltime_summary(resources) -> str:
    phases = (
        ("POLUS", "PHASE_A_POLUS"),
        ("Gaussian", "INITIAL_GAUSSIAN"),
        ("AIMAll", "INITIAL_AIMALL"),
        ("ARIADNE", "ARIADNE_ARRAY"),
        ("FEREBUS", "INITIAL_FEREBUS"),
    )
    return ", ".join(
        label + "=" + str(int(resources.walltime_for(phase))) + "h"
        for label, phase in phases
    )


def format_sampling_protocol_summary(config: CampaignConfig) -> str:
    seed = config.seed_selection
    calib = config.error_calibration
    acq = config.acquisition
    resources = config.resources
    gaussian = config.gaussian
    aimall = config.aimall
    ariadne = config.ariadne
    spectral = acq.spectral
    energy = acq.calibrated_energy
    fullspace = acq.fullspace_confinement
    size_norm = acq.size_normalisation
    movement_band = acq.movement_band
    movement_utility = acq.movement_utility
    driver = acq.driver
    stencils = acq.stencils
    safety = config.adversarial_safety
    gates = config.quality_gates
    phase_b = config.phase_b
    anti = config.anti_overlap
    runtime = config.runtime

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
    lines.append(_line("acquisition.size_normalisation.enabled", size_norm.enabled))
    lines.append(
        _line(
            "acquisition.size_normalisation",
            "energy="
            + str(size_norm.energy_mode)
            + ", whitened="
            + str(size_norm.whitened_distance_mode)
            + ", chemistry="
            + str(size_norm.chemistry_barrier_mode),
        )
    )
    lines.append(_line("acquisition.movement_band.enabled", movement_band.enabled))
    lines.append(
        _line(
            "acquisition.movement_band",
            "metric="
            + str(movement_band.metric)
            + ", local="
            + str(movement_band.local_statistic)
            + ", floors/caps="
            + str(movement_band.hard_min_floor_ang)
            + "/"
            + str(movement_band.target_low_floor_ang)
            + "/"
            + str(movement_band.target_peak_floor_ang)
            + "/"
            + str(movement_band.target_high_cap_ang)
            + "/"
            + str(movement_band.hard_max_cap_ang),
        )
    )
    lines.append(_line("acquisition.movement_utility.enabled", movement_utility.enabled))
    lines.append(
        _line(
            "acquisition.movement_utility",
            "lambda="
            + str(movement_utility.lambda_move)
            + ", direction="
            + str(movement_utility.direction),
        )
    )
    lines.append(_line("acquisition.driver.enabled", driver.enabled))
    lines.append(_line("acquisition.driver.objective", driver.objective))
    lines.append(_line("acquisition.driver.gradient_backend", driver.gradient_backend))
    lines.append(_line("acquisition.driver.include_stencils", driver.include_stencils))
    lines.append(
        _line(
            "acquisition.driver.analytic_terms",
            "movement="
            + str(driver.analytic_movement)
            + ", distance="
            + str(driver.analytic_whitened_distance)
            + ", pair_barriers="
            + str(driver.analytic_pair_barriers)
            + ", fullspace="
            + str(driver.analytic_fullspace_rmsd)
            + ", fd_energy="
            + str(driver.finite_difference_energy),
        )
    )
    lines.append(
        _line(
            "acquisition.driver.weights",
            "energy="
            + str(driver.lambda_energy)
            + ", movement="
            + str(driver.lambda_movement)
            + ", distance="
            + str(driver.lambda_distance)
            + ", fullspace="
            + str(driver.lambda_fullspace)
            + ", chemistry="
            + str(driver.lambda_chemistry),
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
    lines.append(
        _line(
            "acquisition.stencils.weak_mode_gating_enabled",
            stencils.weak_mode_gating_enabled,
        )
    )
    lines.append(
        _line(
            "acquisition.stencils.weak_mode_omega_band",
            "low_fraction="
            + str(stencils.weak_mode_omega_low_fraction)
            + ", high_fraction="
            + str(stencils.weak_mode_omega_high_fraction)
            + ", abs_floor="
            + str(stencils.weak_mode_abs_omega_floor),
        )
    )
    lines.append(
        _line(
            "acquisition.stencils.weak_mode_penalty",
            stencils.weak_mode_penalty,
        )
    )
    lines.append(
        _line(
            "acquisition.stencils.anharmonic_caps",
            "per_mode="
            + str(stencils.max_anharmonic_mode_score)
            + ", total="
            + str(stencils.max_anharmonic_total_score),
        )
    )
    lines.append(_line("ariadne.optimiser", ariadne.optimiser))
    lines.append(_line("ariadne.hessian_model", ariadne.hessian_model))
    lines.append(_line("ariadne.fallback_to_ds", ariadne.fallback_to_ds))
    lines.append(
        _line(
            "ariadne.trqn_objective_scaling",
            "mode="
            + str(ariadne.trqn_scale_mode)
            + ", target="
            + str(ariadne.trqn_target_initial_grad_norm)
            + ", target_rms="
            + str(ariadne.trqn_target_initial_grad_rms)
            + ", retry_target="
            + str(ariadne.trqn_retry_target_initial_grad_norm)
            + ", retry_target_rms="
            + str(ariadne.trqn_retry_target_initial_grad_rms)
            + ", under_move_target_rms="
            + str(ariadne.trqn_under_move_target_initial_grad_rms)
            + ", scale=["
            + str(ariadne.trqn_min_objective_scale)
            + ", "
            + str(ariadne.trqn_max_objective_scale)
            + "]",
        )
    )
    lines.append(
        _line(
            "ariadne.trqn_retry_on_no_proposal",
            ariadne.trqn_retry_on_no_proposal,
        )
    )
    lines.append(
        _line("ariadne.trqn_backtransform_mode", ariadne.trqn_backtransform_mode)
    )
    lines.append(
        _line("ariadne.trqn_geodesic_bt_mode", ariadne.trqn_geodesic_bt_mode)
    )
    lines.append(
        _line(
            "ariadne.trqn_backtransform_numerics",
            "dt="
            + str(ariadne.trqn_geodesic_dt)
            + ", tol="
            + str(ariadne.trqn_geodesic_tol)
            + ", ic_tol="
            + str(ariadne.trqn_bt_ic_tol)
            + ", max_iter="
            + str(ariadne.trqn_max_backtransform_iter)
            + ", trust_min="
            + str(ariadne.trqn_trust_min),
        )
    )
    lines.append(_line("resources.partition", resources.partition))
    lines.append(_line("resources.default_walltime_hours", resources.default_walltime_hours))
    lines.append(_line("resources.effective_phase_walltimes", _walltime_summary(resources)))
    lines.append(
        _line(
            "resources.phase_cpus",
            "POLUS="
            + str(resources.polus_cpus_per_task)
            + ", Gaussian="
            + str(resources.gaussian_cpus_per_task)
            + ", AIMAll="
            + str(resources.aimall_cpus_per_task)
            + ", ARIADNE="
            + str(resources.ariadne_cpus_per_task)
            + ", FEREBUS="
            + str(resources.ferebus_cpus_per_task),
        )
    )
    lines.append(
        _line(
            "resources.phase_mem_per_cpu",
            "POLUS="
            + str(resources.polus_mem_per_cpu)
            + ", Gaussian="
            + str(resources.gaussian_mem_per_cpu)
            + ", AIMAll="
            + str(resources.aimall_mem_per_cpu)
            + ", ARIADNE="
            + str(resources.ariadne_mem_per_cpu)
            + ", FEREBUS="
            + str(resources.ferebus_mem_per_cpu),
        )
    )
    lines.append(_line("resources.gaussian_memory_mode", resources.gaussian_memory_mode))
    lines.append(_line("resources.gaussian_link0_mem", resources.gaussian_link0_mem))
    lines.append(
        _line(
            "resources.gaussian_memory_fraction_of_slurm",
            resources.gaussian_memory_fraction_of_slurm,
        )
    )
    lines.append(
        _line("resources.array_concurrency_limit", resources.array_concurrency_limit)
    )
    lines.append(
        _line(
            "resources.fail_on_memory_estimate_exceeds_request",
            resources.fail_on_memory_estimate_exceeds_request,
        )
    )
    lines.append(_line("resources.gradient_parallel_backend", resources.gradient_parallel_backend))
    lines.append(_line("resources.aimall_cpus_per_task", resources.aimall_cpus_per_task))
    lines.append(_line("aimall.naat", aimall.naat))
    lines.append(_line("aimall.encomp", aimall.encomp))
    lines.append(_line("aimall.boaq", aimall.boaq))
    lines.append(_line("aimall.iasmesh", aimall.iasmesh))
    lines.append(_line("gaussian.method", gaussian.method))
    lines.append(_line("gaussian.basis_set", gaussian.basis_set))
    lines.append(
        _line(
            "runtime.poll_sacct_missing_max_ticks",
            runtime.poll_sacct_missing_max_ticks,
        )
    )
    lines.append(
        _line(
            "runtime.halt_on_tick_exception",
            runtime.halt_on_tick_exception,
        )
    )
    lines.append(_line("adversarial_safety.enabled", safety.enabled))
    lines.append(
        _line("adversarial_safety.reject_unsafe_landings", safety.reject_unsafe_landings)
    )
    lines.append(
        _line("adversarial_safety.max_whitened_distance", safety.max_whitened_distance)
    )
    lines.append(_line("adversarial_safety.enforce_movement_band", safety.enforce_movement_band))
    lines.append(_line("adversarial_safety.under_move_retry", safety.under_move_retry))
    lines.append(_line("adversarial_safety.reject_over_moved", safety.reject_over_moved))
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
