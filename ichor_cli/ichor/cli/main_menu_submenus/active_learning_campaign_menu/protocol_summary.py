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
from ichor.hpc.active_learning.layout import active_learning_dir


def _line(label: str, value) -> str:
    return "-- " + label + ": " + format_field_value(value) + "\n"


def _latest_geometry_novelty_payload(campaign_dir: Path):
    try:
        from ichor.hpc.active_learning.geometry_novelty import (
            read_geometry_novelty_scale,
        )
    except Exception as exc:
        return None, "unavailable (" + type(exc).__name__ + ": " + str(exc) + ")"
    base = active_learning_dir(campaign_dir)
    if not base.is_dir():
        return None, "not written yet"
    candidates = sorted(
        base.glob("iteration-*/protocol/GEOMETRY_NOVELTY_SCALE.json")
    )
    if not candidates:
        return None, "not written yet"
    latest = candidates[-1].parent.parent
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
            + ", resolution="
            + str(resolved.get("scale_resolution_mode"))
            + ", scale_angstrom="
            + str(resolved.get("scale_angstrom")),
        ),
        _line(
            "geometry_novelty.resolved_phase_b",
            "mode="
            + str(phase_b.get("threshold_mode"))
            + ", resolution="
            + str(phase_b.get("scale_resolution_mode"))
            + ", min_separation_angstrom="
            + str(phase_b.get("effective_min_separation_angstrom"))
            + ", coefficient="
            + str(phase_b.get("min_separation_scaled")),
        ),
        _line(
            "geometry_novelty.resolved_movement_band",
            "mode="
            + str(movement_band.get("threshold_mode"))
            + ", resolution="
            + str(movement_band.get("scale_resolution_mode"))
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
            + ", resolution="
            + str(movement_utility.get("scale_resolution_mode"))
            + ", low/high_softness_angstrom="
            + str(movement_utility.get("low_softness_angstrom"))
            + "/"
            + str(movement_utility.get("high_softness_angstrom")),
        ),
        _line(
            "geometry_novelty.resolved_fullspace_confinement",
            "mode="
            + str(fullspace.get("threshold_mode"))
            + ", resolution="
            + str(fullspace.get("scale_resolution_mode"))
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


def _pool_feasibility_summary(
    config: CampaignConfig,
    campaign_dir: str | Path | None,
) -> list[str]:
    if campaign_dir is None:
        if bool(config.anti_overlap.skip_training_seeds):
            required = (
                int(config.point_allocation.bootstrap_total_size)
                + int(config.campaign.max_iterations)
                * int(config.seed_selection.n_seeds_per_iteration)
            )
            expression = (
                str(config.point_allocation.bootstrap_total_size)
                + " + "
                + str(config.campaign.max_iterations)
                + " * "
                + str(config.seed_selection.n_seeds_per_iteration)
                + " = "
                + str(required)
            )
        else:
            required = int(config.point_allocation.bootstrap_total_size)
            expression = str(required)
        return [
            _line("pool_feasibility.imported_pool", "not available"),
            _line("pool_feasibility.required_frames", str(required) + " (" + expression + ")"),
        ]
    try:
        from ichor.hpc.active_learning.daemon.pool_feasibility import (
            evaluate_pool_feasibility,
        )

        result = evaluate_pool_feasibility(Path(campaign_dir), config)
        return [
            _line("pool_feasibility.status", "ok" if result.ok else "failed"),
            _line("pool_feasibility.pool_n_frames", result.pool_n_frames),
            _line("pool_feasibility.bootstrap_anchor_count", result.bootstrap_anchor_count),
            _line("pool_feasibility.bootstrap_pool_frame_count", result.bootstrap_pool_frame_count),
            _line("pool_feasibility.required_pool_frames", result.required_pool_frames),
            _line("pool_feasibility.expression", result.expression),
        ]
    except Exception as exc:
        return [
            _line(
                "pool_feasibility.status",
                "unavailable (" + type(exc).__name__ + ": " + str(exc) + ")",
            )
        ]


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
    resources = config.resources
    gaussian = config.gaussian
    aimall = config.aimall
    runtime = config.runtime
    geometry_payload = None
    geometry_source = "profile fallback preview"
    sampling_protocol_resolved_path = None
    if campaign_dir is not None:
        geometry_payload, geometry_source = _latest_geometry_novelty_payload(
            Path(campaign_dir)
        )
    preview_iteration = 1
    if isinstance(geometry_payload, dict):
        try:
            preview_iteration = max(1, int(geometry_payload.get("iteration", 1)))
        except (TypeError, ValueError):
            preview_iteration = 1
    try:
        from ichor.hpc.active_learning.sampling_protocol import (
            preview_sampling_protocol,
            sampling_protocol_audit_path,
            sampling_protocol_resolved_path,
        )

        resolved = preview_sampling_protocol(
            config,
            campaign_dir=campaign_dir,
            iteration=preview_iteration,
            geometry_scale_payload=geometry_payload,
        )
        resolved_error = None
    except Exception as exc:
        resolved = None
        resolved_error = type(exc).__name__ + ": " + str(exc)

    lines = ["Sampling protocol summary:\n"]
    lines.append(
        _line(
            "campaign.sampling_aggressiveness",
            config.campaign.sampling_aggressiveness,
        )
    )
    if resolved is None:
        lines.append(_line("sampling_protocol.resolution", "unavailable (" + str(resolved_error) + ")"))
    else:
        phase_b = dict(resolved.phase_b or {})
        safety = resolved.adversarial_safety
        gates = resolved.quality_gates
        acq_cfg = resolved.acquisition_config
        ariadne_run = resolved.ariadne_run_config
        scale_model = dict(resolved.scale_model_payload or {})
        movement = acq_cfg.movement_band
        scale = resolved.resolved_geometry_scale_angstrom
        movement_text = "unavailable"
        if scale is not None:
            movement_text = (
                "scale="
                + str(scale)
                + ", min/peak/max="
                + str(float(movement.hard_min_fraction) * float(scale))
                + "/"
                + str(float(movement.target_peak_fraction) * float(scale))
                + "/"
                + str(float(movement.hard_max_fraction) * float(scale))
            )
        lines.append(_line("sampling_protocol.geometry_scale_source", geometry_source))
        lines.append(_line("sampling_protocol.resolved_geometry_scale_angstrom", scale))
        geom_scale = scale_model.get("geometry_motion_scale", {})
        rmsd_scale = scale_model.get("aligned_rmsd_scale", {})
        residual_scale = scale_model.get("residual_fullspace_scale", {})
        mobility = scale_model.get("per_atom_mobility_scales", {})
        pair_ref = scale_model.get("pair_distance_reference", {})
        dimensionless = scale_model.get("dimensionless_preset", {})
        lines.append(
            _line(
                "sampling_protocol.scale_model",
                "schema="
                + str(scale_model.get("schema_version"))
                + ", model_version="
                + str(scale_model.get("model_version"))
                + ", geometry_source="
                + str(geom_scale.get("source"))
                + ", history_records="
                + str((scale_model.get("history") or {}).get("n_records")),
            )
        )
        lines.append(
            _line(
                "sampling_protocol.scale_model.geometry_motion_scale",
                str(geom_scale.get("value_angstrom"))
                + " Angstrom, fallback="
                + str(geom_scale.get("fallback_used")),
            )
        )
        lines.append(
            _line(
                "sampling_protocol.scale_model.aligned_rmsd_scale",
                str(rmsd_scale.get("value_angstrom"))
                + " Angstrom, source="
                + str(rmsd_scale.get("source")),
            )
        )
        lines.append(
            _line(
                "sampling_protocol.scale_model.residual_fullspace_scale",
                str(residual_scale.get("value_angstrom"))
                + " Angstrom, source="
                + str(residual_scale.get("source")),
            )
        )
        lines.append(
            _line(
                "sampling_protocol.scale_model.per_atom_mobility",
                "mode="
                + str(mobility.get("mode"))
                + ", source="
                + str(mobility.get("source"))
                + ", n_values="
                + str(len(mobility.get("values_angstrom") or [])),
            )
        )
        lines.append(
            _line(
                "sampling_protocol.scale_model.pair_reference",
                str(pair_ref.get("reference_min_pair_distance_angstrom"))
                + " Angstrom, mode="
                + str(pair_ref.get("mode")),
            )
        )
        lines.append(
            _line(
                "sampling_protocol.dimensionless_landing_gates",
                "max_scaled_atom_move="
                + str(dimensionless.get("max_scaled_atom_move"))
                + ", max_scaled_rmsd="
                + str(dimensionless.get("max_scaled_rmsd"))
                + ", max_scaled_residual="
                + str(dimensionless.get("max_scaled_fullspace_residual"))
                + ", max_scaled_whitened="
                + str(dimensionless.get("max_scaled_whitened_distance"))
                + ", chemistry_cap="
                + str(dimensionless.get("normalised_chemistry_penalty_cap")),
            )
        )
        lines.append(_line("sampling_protocol.resolved_movement_band", movement_text))
        lines.append(
            _line(
                "sampling_protocol.resolved_phase_b",
                "descriptor="
                + str(phase_b.get("descriptor"))
                + ", beta="
                + str(phase_b.get("beta"))
                + ", min_separation_angstrom="
                + str(phase_b.get("effective_min_separation_angstrom"))
                + ", coefficient="
                + str(phase_b.get("min_separation_scaled"))
                + ", scale_model_source="
                + str(phase_b.get("scale_model_source")),
            )
        )
        lines.append(
            _line(
                "sampling_protocol.resolved_safety",
                "max_whitened_distance="
                + str(safety.max_whitened_distance)
                + ", backtrack_points="
                + str(safety.backtrack_points)
                + ", reject_unsafe="
                + str(safety.reject_unsafe_landings)
                + ", movement_band="
                + str(safety.enforce_movement_band),
            )
        )
        lines.append(
            _line(
                "sampling_protocol.resolved_quality_gates",
                "max_displacement_ang="
                + str(gates.ariadne_max_displacement_ang)
                + ", min_pair_distance_policy=scale_model_ratio("
                + str(gates.ariadne_min_pair_distance_ang)
                + " Angstrom)",
            )
        )
        lines.append(
            _line(
                "sampling_protocol.resolved_acquisition_risk",
                "lambda_distance="
                + str(acq_cfg.weights.lambda_distance)
                + ", lambda_residual="
                + str(acq_cfg.fullspace_confinement.lambda_residual)
                + ", lambda_rmsd="
                + str(acq_cfg.fullspace_confinement.lambda_rmsd)
                + ", residual_scale="
                + str(acq_cfg.fullspace_confinement.fixed_residual_scale_ang)
                + ", rmsd_scale="
                + str(acq_cfg.fullspace_confinement.rmsd_scale_ang),
            )
        )
        lines.append(
            _line(
                "sampling_protocol.resolved_ariadne",
                "legacy_profile_delta0="
                + str(ariadne_run.delta0)
                + ", legacy_profile_delta_max="
                + str(ariadne_run.delta_max)
                + ", target_rms="
                + str(ariadne_run.trqn_target_initial_grad_rms)
                + ", under_move_target_rms="
                + str(ariadne_run.trqn_under_move_target_initial_grad_rms),
            )
        )
        trust_policy = scale_model.get("trust_radius_policy", {})
        lines.append(
            _line(
                "sampling_protocol.size_normalised_trust_radius",
                "enabled="
                + str(trust_policy.get("enabled"))
                + ", normalisation="
                + str(trust_policy.get("normalisation"))
                + ", aggressiveness_multiplier="
                + str(trust_policy.get("aggressiveness_multiplier"))
                + ", retry_factor_max="
                + str(trust_policy.get("under_move_feedback_max_factor")),
            )
        )
        lines.append(
            _line(
                "sampling_protocol.hidden_overrides_detected",
                len(resolved.hidden_overrides_detected),
            )
        )
    if campaign_dir is not None:
        try:
            from ichor.hpc.active_learning.layout import active_iteration_dir

            example_iteration = active_iteration_dir(Path(campaign_dir), 1)
            manifest_path = sampling_protocol_resolved_path(
                example_iteration
            )
            lines.append(_line("sampling_protocol.resolved_manifest_example", manifest_path))
            audit_path = sampling_protocol_audit_path(
                example_iteration
            )
            lines.append(_line("sampling_protocol.audit_manifest_example", audit_path))
        except Exception:
            pass
    lines.append(
        _line(
            "point_allocation.bootstrap_training_size",
            config.point_allocation.bootstrap_training_size,
        )
    )
    lines.append(
        _line(
            "point_allocation.bootstrap_internal_validation_size",
            config.point_allocation.bootstrap_internal_validation_size,
        )
    )
    lines.append(
        _line(
            "point_allocation.bootstrap_external_validation_size",
            config.point_allocation.bootstrap_external_validation_size,
        )
    )
    lines.append(
        _line(
            "point_allocation.bootstrap_total_size",
            int(config.point_allocation.bootstrap_total_size),
        )
    )
    lines.append(
        _line(
            "point_allocation.batch_training_size",
            config.point_allocation.batch_training_size,
        )
    )
    lines.append(
        _line(
            "point_allocation.batch_internal_validation_size",
            config.point_allocation.batch_internal_validation_size,
        )
    )
    lines.append(
        _line(
            "point_allocation.batch_total_size",
            int(config.point_allocation.batch_total_size),
        )
    )
    lines.append(_line("campaign.source_path", config.campaign.source_path))
    lines.append(_line("campaign.anchor_path", config.campaign.anchor_path))
    lines.append(_line("point_allocation.anchor", config.point_allocation.anchor))
    if bool(getattr(config.point_allocation, "anchor", False)):
        lines.append(_line("point_allocation.anchor_xyz", config.campaign.anchor_path))
    lines.append(
        _line(
            "seed_selection.n_seeds_per_iteration",
            seed.n_seeds_per_iteration,
        )
    )
    lines.append(
        _line(
            "point_allocation.seed_surplus",
            int(seed.n_seeds_per_iteration)
            - int(config.point_allocation.batch_total_size),
        )
    )
    lines.append(_line("campaign.max_iterations", config.campaign.max_iterations))
    lines.extend(_pool_feasibility_summary(config, campaign_dir))
    lines.append(_line("seed_selection.strategy", seed.strategy))
    lines.append(_line("seed_selection.bulk_fraction", seed.bulk_fraction))
    lines.append(
        _line(
            "seed_selection.d_optimal",
            "pool_multiplier="
            + str(seed.d_optimal_pool_multiplier)
            + ", score_power="
            + str(seed.d_optimal_score_power)
            + ", degenerate_policy="
            + str(seed.d_optimal_degenerate_policy),
        )
    )
    lines.append(_line("error_calibration.enabled", calib.enabled))
    lines.append(_line("error_calibration.mode", calib.mode))
    lines.append(_line("error_calibration.apply_strength", calib.apply_strength))
    lines.append(_line("error_calibration.model_version_policy", calib.model_version_policy))
    lines.append(
        _line("error_calibration.min_records_to_apply", calib.min_records_to_apply)
    )
    lines.append(
        _line(
            "error_calibration.min_model_versions_to_apply",
            calib.min_model_versions_to_apply,
        )
    )
    lines.append(
        _line(
            "error_calibration.aggressiveness_match_required",
            calib.aggressiveness_match_required,
        )
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
