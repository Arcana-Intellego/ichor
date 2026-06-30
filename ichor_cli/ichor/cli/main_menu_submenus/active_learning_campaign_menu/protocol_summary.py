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


def _latest_geometry_novelty_payload(campaign_dir: Path):
    try:
        from ichor.hpc.active_learning.geometry_novelty import (
            read_geometry_novelty_scale,
        )
    except Exception as exc:
        return None, "unavailable (" + type(exc).__name__ + ": " + str(exc) + ")"
    base = Path(campaign_dir) / "7_ACTIVE_LEARNING"
    if not base.is_dir():
        return None, "not written yet"
    candidates = sorted(base.glob("iteration-*/GEOMETRY_NOVELTY_SCALE.json"))
    if not candidates:
        return None, "not written yet"
    latest = candidates[-1].parent
    try:
        payload = read_geometry_novelty_scale(latest)
    except Exception as exc:
        return None, "unreadable (" + type(exc).__name__ + ": " + str(exc) + ")"
    return payload, "latest_sidecar"


def _latest_geometry_novelty_scale_summary(campaign_dir: Path) -> str:
    payload, status = _latest_geometry_novelty_payload(campaign_dir)
    if payload is None:
        return str(status)
    return (
        "iteration="
        + str(payload.get("iteration"))
        + ", scale_angstrom="
        + str(payload.get("scale_angstrom"))
        + ", fallback_used="
        + str(payload.get("fallback_used"))
        + ", n_values="
        + str(payload.get("n_values"))
    )


def _geometry_novelty_resolved_lines(
    config: CampaignConfig,
    campaign_dir: str | Path | None,
) -> list[str]:
    try:
        from ichor.hpc.active_learning.geometry_novelty import (
            resolve_geometry_novelty_consumers,
        )
        from ichor.hpc.active_learning.geometry_protocol import (
            FULLSPACE_RMSD_SCALE_MULTIPLIER,
            GEOMETRY_NOVELTY_SCORE_TRANSFORM,
            MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION,
            MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION,
            PHASE_B_MIN_SEPARATION_SCALE,
            movement_band_fractions,
        )
    except Exception as exc:
        return [
            _line(
                "geometry_novelty.resolved_consumers",
                "unavailable (" + type(exc).__name__ + ": " + str(exc) + ")",
            )
        ]

    payload = None
    source = "configured_fallback"
    if campaign_dir is not None:
        payload, source = _latest_geometry_novelty_payload(Path(campaign_dir))
    if payload is None and bool(config.geometry_novelty.enabled):
        payload = {
            "schema_version": 1,
            "scale_angstrom": float(config.geometry_novelty.fallback_scale_angstrom),
        }
        source = "configured_fallback"

    try:
        resolved = resolve_geometry_novelty_consumers(config, payload)
    except Exception as exc:
        return [
            _line(
                "geometry_novelty.resolved_consumers",
                "unreadable (" + type(exc).__name__ + ": " + str(exc) + ")",
            )
        ]

    phase_b = dict(resolved.get("phase_b") or {})
    movement_band = dict(resolved.get("movement_band") or {})
    movement_utility = dict(resolved.get("movement_utility") or {})
    fullspace = dict(resolved.get("fullspace_confinement") or {})
    return [
        _line(
            "geometry_novelty.protocol.phase_b_scale",
            PHASE_B_MIN_SEPARATION_SCALE,
        ),
        _line(
            "geometry_novelty.protocol.movement_band_fractions",
            movement_band_fractions(),
        ),
        _line(
            "geometry_novelty.protocol.movement_utility_softness_fractions",
            str(MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION)
            + "/"
            + str(MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION),
        ),
        _line(
            "geometry_novelty.protocol.fullspace_rmsd_multiplier",
            FULLSPACE_RMSD_SCALE_MULTIPLIER,
        ),
        _line(
            "geometry_novelty.protocol.score_transform",
            GEOMETRY_NOVELTY_SCORE_TRANSFORM,
        ),
        _line(
            "geometry_novelty.resolved_source",
            str(source)
            + ", mode="
            + str(resolved.get("threshold_mode"))
            + ", scale_angstrom="
            + str(resolved.get("scale_angstrom")),
        ),
        _line(
            "geometry_novelty.resolved_phase_b",
            "mode="
            + str(phase_b.get("threshold_mode"))
            + ", min_separation_angstrom="
            + str(phase_b.get("effective_min_separation_angstrom"))
            + ", coefficient="
            + str(phase_b.get("min_separation_scaled")),
        ),
        _line(
            "geometry_novelty.resolved_movement_band",
            "mode="
            + str(movement_band.get("threshold_mode"))
            + ", min/low/peak/high/max_angstrom="
            + str(movement_band.get("hard_min_angstrom"))
            + "/"
            + str(movement_band.get("target_low_angstrom"))
            + "/"
            + str(movement_band.get("target_peak_angstrom"))
            + "/"
            + str(movement_band.get("target_high_angstrom"))
            + "/"
            + str(movement_band.get("hard_max_angstrom")),
        ),
        _line(
            "geometry_novelty.resolved_movement_utility",
            "mode="
            + str(movement_utility.get("threshold_mode"))
            + ", low/high_softness_angstrom="
            + str(movement_utility.get("low_softness_angstrom"))
            + "/"
            + str(movement_utility.get("high_softness_angstrom")),
        ),
        _line(
            "geometry_novelty.resolved_fullspace_confinement",
            "mode="
            + str(fullspace.get("threshold_mode"))
            + ", rmsd_scale_angstrom="
            + str(fullspace.get("rmsd_scale_angstrom")),
        ),
    ]


def _hours_label(value) -> str:
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return str(value) + "h"
    if hours.is_integer():
        return str(int(hours)) + "h"
    minutes = int(round(hours * 60.0))
    return str(hours) + "h (" + str(minutes) + "m)"


def _walltime_summary(resources) -> str:
    phases = (
        ("POLUS", "PHASE_A_POLUS"),
        ("Gaussian", "INITIAL_GAUSSIAN"),
        ("AIMAll", "INITIAL_AIMALL"),
        ("ARIADNE", "ARIADNE_ARRAY"),
        ("FEREBUS", "INITIAL_FEREBUS"),
    )
    return ", ".join(
        label + "=" + _hours_label(resources.walltime_for(phase))
        for label, phase in phases
    )


def _backend_effective_summary(resources, field_name: str) -> str:
    phases = (
        ("POLUS", "PHASE_A_POLUS"),
        ("Gaussian", "INITIAL_GAUSSIAN"),
        ("AIMAll", "INITIAL_AIMALL"),
        ("ARIADNE", "ARIADNE_ARRAY"),
        ("FEREBUS", "INITIAL_FEREBUS"),
    )
    values = []
    for label, phase in phases:
        if field_name == "partition":
            effective = resources.partition_for(phase)
        elif field_name == "cpus_per_task":
            effective = resources.cpus_for(phase)
        elif field_name == "mem_per_cpu":
            effective = resources.mem_per_cpu_for(phase)
        else:
            effective = resources.walltime_for(phase)
        explicit = getattr(getattr(resources, resources.backend_for_phase(phase)), field_name)
        values.append(label + "=" + format_field_value(explicit) + " -> " + str(effective))
    return ", ".join(values)


def format_sampling_protocol_summary(
    config: CampaignConfig,
    campaign_dir: str | Path | None = None,
) -> str:
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
    driver = acq.driver
    stencils = acq.stencils
    safety = config.adversarial_safety
    gates = config.quality_gates
    phase_b = config.phase_b
    novelty = config.geometry_novelty
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
    lines.append(_line("error_calibration.max_records", calib.max_records))
    lines.append(
        _line(
            "error_calibration.max_model_age_iterations",
            calib.max_model_age_iterations,
        )
    )
    lines.append(_line("error_calibration.monotone_estimator", calib.monotone_estimator))
    lines.append(_line("error_calibration.quantile", calib.quantile))
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
            "acquisition.fullspace_confinement.public_fields",
            "enabled/lambda_residual/lambda_rmsd; scale and failure policy are protocol constants",
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
    lines.append(_line("geometry_novelty.protocol.movement_band", "internal"))
    lines.append(_line("geometry_novelty.protocol.movement_utility", "internal"))
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
    lines.append(_line("resources.defaults.partition", resources.defaults.partition))
    lines.append(_line("resources.defaults.walltime_hours", resources.defaults.walltime_hours))
    lines.append(_line("resources.defaults.cpus_per_task", resources.defaults.cpus_per_task))
    lines.append(_line("resources.defaults.mem_per_cpu", resources.defaults.mem_per_cpu))
    lines.append(
        _line(
            "resources.backend_partitions",
            _backend_effective_summary(resources, "partition"),
        )
    )
    lines.append(_line("resources.effective_phase_walltimes", _walltime_summary(resources)))
    lines.append(
        _line(
            "resources.backend_cpus",
            _backend_effective_summary(resources, "cpus_per_task"),
        )
    )
    lines.append(
        _line(
            "resources.backend_mem_per_cpu",
            _backend_effective_summary(resources, "mem_per_cpu"),
        )
    )
    lines.append(_line("resources.gaussian.memory_mode", resources.gaussian.memory_mode))
    lines.append(_line("resources.gaussian.link0_mem", resources.gaussian.link0_mem))
    lines.append(
        _line(
            "resources.gaussian.memory_fraction_of_slurm",
            resources.gaussian.memory_fraction_of_slurm,
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
    lines.append(_line("resources.aimall.effective_cpus_per_task", resources.cpus_for("AIMALL")))
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
        _line(
            "adversarial_safety.accept_legacy_missing_landing_safety",
            safety.accept_legacy_missing_landing_safety,
        )
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
    lines.append(_line("geometry_novelty.enabled", novelty.enabled))
    lines.append(
        _line(
            "geometry_novelty.scale",
            "source="
            + str(novelty.scale_source)
            + ", statistic="
            + str(novelty.statistic)
            + ", fallback_scale_angstrom="
            + str(novelty.fallback_scale_angstrom)
            + ", floor="
            + str(novelty.scale_floor_angstrom),
        )
    )
    if campaign_dir is not None:
        lines.append(
            _line(
                "geometry_novelty.latest_sidecar",
                _latest_geometry_novelty_scale_summary(Path(campaign_dir)),
            )
        )
    lines.extend(_geometry_novelty_resolved_lines(config, campaign_dir))
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
    return format_sampling_protocol_summary(config, campaign_dir=campaign)
