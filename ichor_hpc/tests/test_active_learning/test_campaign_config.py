"""Tests for ichor.hpc.active_learning.config -- schema v5."""
from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import (
    AriadneConfigBlock,
    CONFIG_SCHEMA_VERSION,
    CampaignConfig,
    ConfigValidationError,
)
from ichor.hpc.active_learning.geometry_protocol import (
    FULLSPACE_FAILURE_PENALTY,
    FULLSPACE_FIXED_RESIDUAL_SCALE_ANGSTROM,
    FULLSPACE_MIN_RESIDUAL_SCALE_ANGSTROM,
    FULLSPACE_RESIDUAL_SCALE,
    FULLSPACE_RMSD_SCALE_MULTIPLIER,
    MOVEMENT_BAND_HARD_MAX_FRACTION,
    MOVEMENT_BAND_HARD_MIN_FRACTION,
    MOVEMENT_BAND_TARGET_HIGH_FRACTION,
    MOVEMENT_BAND_TARGET_LOW_FRACTION,
    MOVEMENT_BAND_TARGET_PEAK_FRACTION,
    MOVEMENT_UTILITY_BAND_FRACTION,
    MOVEMENT_UTILITY_DIRECTION,
    MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION,
    MOVEMENT_UTILITY_LAMBDA_MOVE,
    MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION,
    MOVEMENT_UTILITY_PROGRESS_FRACTION,
)


def test_schema_version_is_five():
    assert CONFIG_SCHEMA_VERSION == 5


def test_default_campaign_config_is_valid():
    c = CampaignConfig()
    c._validate()
    assert c.quality_gates.require_readable_aimall_geometry is True
    assert c.quality_gates.require_finite_iqa is True
    assert c.quality_gates.require_finite_integration_error is True
    assert c.runtime.lease_stale_seconds == 900
    assert c.runtime.postprocess_settle_attempts == 3
    assert c.runtime.postprocess_settle_seconds == 10
    assert c.runtime.transient_phase_retry_max == 1
    assert c.runtime.poll_sacct_unknown_max_ticks == 3
    assert c.runtime.poll_sacct_missing_max_ticks == 12
    assert c.runtime.halt_on_tick_exception is True
    assert c.seed_selection.variance_chunk_size == 512
    assert c.seed_selection.strategy == "hybrid_variance"
    assert c.seed_selection.d_optimal_pool_multiplier == 8
    assert c.seed_selection.d_optimal_jitter == 1.0e-12
    assert c.seed_selection.d_optimal_novelty_floor == 1.0e-12
    assert c.seed_selection.d_optimal_score_power == 1.0
    assert c.geometry_novelty.enabled is True
    assert c.geometry_novelty.scale_source == "local_motion"
    assert c.geometry_novelty.statistic == "median"
    assert c.geometry_novelty.scale_floor_angstrom == 1.0e-3
    assert c.geometry_novelty.history_window_iterations == 5
    assert c.geometry_novelty.fallback_scale_angstrom == 0.05
    assert c.resources.defaults.partition == "multicore"
    assert c.resources.defaults.walltime_hours == 24
    assert c.resources.defaults.cpus_per_task == "auto"
    assert c.resources.defaults.mem_per_cpu == "auto"
    assert c.resources.partition_for("PHASE_A_POLUS") == "multicore"
    assert c.resources.polus.walltime_hours == 2
    assert c.resources.gaussian.walltime_hours == 24
    assert c.resources.cpus_for("GAUSSIAN") == "auto"
    assert c.resources.mem_per_cpu_for("AIMALL") == "auto"
    assert c.resources.array_concurrency_limit is None
    assert c.resources.gaussian.memory_mode == "slurm_env"
    assert c.resources.gaussian.memory_fraction_of_slurm == 0.85
    assert c.aimall.encomp == 3
    assert c.aimall.nogui is True
    assert c.aimall.naat == "auto"
    assert c.aimall.boaq == "auto"
    assert c.aimall.iasmesh == "fine"
    assert c.quality_gates.ariadne_max_displacement_ang == 1.25
    assert c.quality_gates.ariadne_min_pair_distance_ang == 0.60
    assert c.adversarial_safety.accept_legacy_missing_landing_safety is False
    assert c.max_acquisition_grad_per_ang is None
    assert c.effective_max_acquisition_grad_per_ang() == 50.0
    assert c.error_calibration.enabled is True
    assert c.error_calibration.mode == "record_only"
    assert c.error_calibration.min_records_to_apply == 100
    assert c.error_calibration.n_bins == 10
    assert c.error_calibration.min_bin_records == 8
    assert c.error_calibration.max_records == 5000
    assert c.error_calibration.max_model_age_iterations == 10
    assert c.error_calibration.monotone_estimator is True
    assert c.error_calibration.quantile == 0.75
    assert c.error_calibration.apply_strength == 0.0
    assert c.error_calibration.group_by_atom_type is True
    assert c.error_calibration.group_by_landing_policy is False
    assert c.error_calibration.model_version_policy == "current"
    assert c.error_calibration.output_units == "ha"
    assert c.acquisition.spectral.enabled is True
    assert c.acquisition.spectral.mode == "blend"
    assert c.acquisition.spectral.mode_weighting == "inverse_frequency"
    assert c.acquisition.spectral.lambda_spectral == 1.5
    assert c.acquisition.calibrated_energy.utility == "banded"
    assert c.acquisition.calibrated_energy.fallback_to_raw_variance is True
    assert c.acquisition.fullspace_confinement.enabled is True
    assert c.acquisition.fullspace_confinement.lambda_residual == 0.5
    assert c.acquisition.fullspace_confinement.lambda_rmsd == 0.25
    assert c.acquisition.stencils.negative_curvature_policy == "ignore"
    assert c.acquisition.stencils.lambda_negative_curvature == 1.0
    assert c.acquisition.stencils.weak_mode_gating_enabled is True
    assert c.acquisition.stencils.weak_mode_omega_low_fraction == 0.05
    assert c.acquisition.stencils.weak_mode_omega_high_fraction == 0.15
    assert c.acquisition.stencils.weak_mode_abs_omega_floor == 1.0e-4
    assert c.acquisition.stencils.weak_mode_penalty == 2.0
    assert c.acquisition.stencils.max_anharmonic_mode_score == 6.0
    assert c.acquisition.stencils.max_anharmonic_total_score == 15.0
    assert c.acquisition.subspace.canonicalise_basis is True


@pytest.mark.parametrize("name", ["WATER", "nh3_batch_01", "C6H6-AL"])
def test_system_name_accepts_filename_safe_tokens(name):
    payload = CampaignConfig().to_dict()
    payload["system_name"] = name
    cfg = CampaignConfig.from_dict(payload)
    assert cfg.system_name == name


@pytest.mark.parametrize(
    "name",
    ["", " ", "my system", "../x", "x/y", "x\\y", "x;y", '"x"', "$x", "x\n"],
)
def test_system_name_rejects_unsafe_tokens(name):
    payload = CampaignConfig().to_dict()
    payload["system_name"] = name
    with pytest.raises(ConfigValidationError, match="system_name"):
        CampaignConfig.from_dict(payload)


def _set_path(payload, path, value):
    cur = payload
    parts = path.split(".")
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


@pytest.mark.parametrize(
    "path,value",
    [
        ("resources.defaults.walltime_hours", 0),
        ("resources.defaults.walltime_hours", -1),
        ("resources.polus.walltime_hours", 0),
        ("resources.gaussian.walltime_hours", -1),
        ("resources.aimall.walltime_hours", 0),
        ("resources.ariadne.walltime_hours", -1),
        ("resources.ferebus.walltime_hours", 0),
        ("resources.defaults.cpus_per_task", 0),
        ("resources.polus.cpus_per_task", 0),
        ("resources.gaussian.cpus_per_task", 0),
        ("resources.aimall.cpus_per_task", 0),
        ("resources.ariadne.cpus_per_task", 0),
        ("resources.ferebus.cpus_per_task", 0),
    ],
)
def test_scheduler_positive_resource_fields_validated(path, value):
    payload = CampaignConfig().to_dict()
    _set_path(payload, path, value)
    with pytest.raises(ConfigValidationError, match="resources"):
        CampaignConfig.from_dict(payload)


def test_walltime_accepts_fractional_hours():
    payload = CampaignConfig().to_dict()
    payload["resources"]["polus"]["walltime_hours"] = 0.25
    cfg = CampaignConfig.from_dict(payload)
    assert cfg.resources.walltime_for("PHASE_A_POLUS") == 0.25


def test_walltime_rejects_non_numeric_values():
    payload = CampaignConfig().to_dict()
    payload["resources"]["ferebus"]["walltime_hours"] = "2"
    with pytest.raises(ConfigValidationError, match="ferebus.walltime_hours"):
        CampaignConfig.from_dict(payload)


def test_backend_resources_inherit_defaults_and_override_per_phase():
    cfg = CampaignConfig()
    assert cfg.resources.ferebus.walltime_hours is None
    assert cfg.resources.walltime_for("INITIAL_FEREBUS") == cfg.resources.defaults.walltime_hours
    cfg.resources.defaults.walltime_hours = 12
    cfg.resources.polus.walltime_hours = 1
    cfg.resources.gaussian.walltime_hours = 2
    cfg.resources.aimall.walltime_hours = 3
    cfg.resources.ariadne.walltime_hours = 4
    cfg.resources.ferebus.walltime_hours = 5
    cfg.resources.polus.partition = "interactive"
    assert cfg.resources.walltime_for("PHASE_A_POLUS") == 1
    assert cfg.resources.walltime_for("GAUSSIAN") == 2
    assert cfg.resources.walltime_for("INITIAL_AIMALL") == 3
    assert cfg.resources.walltime_for("ARIADNE_ARRAY") == 4
    assert cfg.resources.walltime_for("FEREBUS") == 5
    assert cfg.resources.partition_for("PHASE_A_POLUS") == "interactive"
    assert cfg.resources.partition_for("GAUSSIAN") == "multicore"


@pytest.mark.parametrize("partition", ["multicore", "serial-debug", "gpu:shared", "csf4.test"])
def test_scheduler_partition_accepts_safe_tokens(partition):
    payload = CampaignConfig().to_dict()
    payload["resources"]["defaults"]["partition"] = partition
    assert CampaignConfig.from_dict(payload).resources.partition == partition


@pytest.mark.parametrize("partition", ["", " ", "multi core", "../queue", "q;rm", "$QUEUE", "q\n"])
def test_scheduler_partition_rejects_unsafe_tokens(partition):
    payload = CampaignConfig().to_dict()
    payload["resources"]["defaults"]["partition"] = partition
    with pytest.raises(ConfigValidationError, match="resources.defaults.partition"):
        CampaignConfig.from_dict(payload)


@pytest.mark.parametrize("mem", ["auto", "1K", "4000M", "4G", "2T"])
def test_slurm_memory_accepts_csf4_style_units(mem):
    payload = CampaignConfig().to_dict()
    payload["resources"]["gaussian"]["mem_per_cpu"] = mem
    payload["resources"]["gaussian"]["link0_mem"] = "1MB"
    assert CampaignConfig.from_dict(payload).resources.gaussian_mem_per_cpu == mem


@pytest.mark.parametrize("mem", ["", "Auto", "0G", "4GB", "4 G", "fourG", "4G;rm"])
def test_slurm_memory_rejects_invalid_units(mem):
    payload = CampaignConfig().to_dict()
    payload["resources"]["gaussian"]["mem_per_cpu"] = mem
    with pytest.raises(ConfigValidationError, match="resources.gaussian.mem_per_cpu"):
        CampaignConfig.from_dict(payload)


@pytest.mark.parametrize("mem", ["8GB", "8000MB", "8G", "500MW"])
def test_gaussian_memory_accepts_gaussian_style_units(mem):
    payload = CampaignConfig().to_dict()
    payload["resources"]["gaussian"]["link0_mem"] = mem
    assert CampaignConfig.from_dict(payload).resources.gaussian_link0_mem == mem


@pytest.mark.parametrize("mem", ["", "0GB", "8 GB", "eightGB", "8GB;rm"])
def test_gaussian_memory_rejects_invalid_units(mem):
    payload = CampaignConfig().to_dict()
    payload["resources"]["gaussian"]["link0_mem"] = mem
    with pytest.raises(ConfigValidationError, match="gaussian.link0_mem"):
        CampaignConfig.from_dict(payload)


def test_gaussian_cpus_must_be_positive_or_auto():
    payload = CampaignConfig().to_dict()
    payload["resources"]["gaussian"]["cpus_per_task"] = 0
    with pytest.raises(ConfigValidationError, match="gaussian.cpus_per_task"):
        CampaignConfig.from_dict(payload)


def test_gaussian_slurm_env_cpus_are_backend_specific():
    payload = CampaignConfig().to_dict()
    payload["resources"]["gaussian"]["cpus_per_task"] = 3
    cfg = CampaignConfig.from_dict(payload)
    assert cfg.resources.gaussian_cpus_per_task == 3


def test_gaussian_link0_mem_must_leave_slurm_headroom():
    payload = CampaignConfig().to_dict()
    payload["resources"]["gaussian"]["memory_mode"] = "link0"
    payload["resources"]["gaussian"]["mem_per_cpu"] = "4G"
    payload["resources"]["gaussian"]["cpus_per_task"] = 1
    payload["resources"]["gaussian"]["link0_mem"] = "4GB"
    with pytest.raises(ConfigValidationError, match="gaussian.link0_mem"):
        CampaignConfig.from_dict(payload)


def test_schema_v3_resources_migrate_to_current_grouped_schema():
    payload = {
        "schema_version": 3,
        "system_name": "MIGRATE",
        "resources": {
            "partition": "multicore",
            "default_walltime_hours": 12,
            "polus_walltime_hours": 1,
            "gaussian_cpus_per_task": 6,
            "gaussian_mem_per_cpu": "4G",
            "gaussian_memory_mode": "link0",
            "gaussian_link0_mem": "8GB",
        },
    }
    cfg = CampaignConfig.from_dict(payload)
    assert cfg.schema_version == 5
    assert cfg.resources.defaults.partition == "multicore"
    assert cfg.resources.defaults.walltime_hours == 12
    assert cfg.resources.polus.walltime_hours == 1
    assert cfg.resources.gaussian.cpus_per_task == 6
    assert cfg.resources.gaussian.mem_per_cpu == "4G"
    assert cfg.resources.gaussian.memory_mode == "link0"


def test_schema_v4_rejects_old_flat_resource_fields():
    payload = CampaignConfig().to_dict()
    payload["resources"]["partition"] = "serial"
    with pytest.raises(ConfigValidationError, match="unknown keys"):
        CampaignConfig.from_dict(payload)


def test_aimall_fields_validated():
    payload = CampaignConfig().to_dict()
    payload["aimall"]["encomp"] = 0
    with pytest.raises(ConfigValidationError, match="aimall.encomp"):
        CampaignConfig.from_dict(payload)
    payload = CampaignConfig().to_dict()
    payload["aimall"]["nogui"] = "yes"
    with pytest.raises(ConfigValidationError, match="aimall.nogui"):
        CampaignConfig.from_dict(payload)


def test_preferred_acquisition_gradient_clamp_overrides_legacy_default():
    payload = CampaignConfig().to_dict()
    payload["max_acquisition_grad_per_ang"] = 7.5
    cfg = CampaignConfig.from_dict(payload)
    assert cfg.effective_max_acquisition_grad_per_ang() == 7.5


def test_legacy_force_clamp_alias_still_supported():
    payload = CampaignConfig().to_dict()
    payload["max_force_per_atom_ha_per_ang"] = 6.0
    cfg = CampaignConfig.from_dict(payload)
    assert cfg.max_acquisition_grad_per_ang is None
    assert cfg.effective_max_acquisition_grad_per_ang() == 6.0


def test_acquisition_gradient_clamp_rejects_ambiguous_alias_values():
    payload = CampaignConfig().to_dict()
    payload["max_acquisition_grad_per_ang"] = 7.5
    payload["max_force_per_atom_ha_per_ang"] = 6.0
    with pytest.raises(ConfigValidationError, match="conflicts"):
        CampaignConfig.from_dict(payload)


def test_acquisition_gradient_clamp_must_be_positive():
    payload = CampaignConfig().to_dict()
    payload["max_acquisition_grad_per_ang"] = 0.0
    with pytest.raises(ConfigValidationError, match="max_acquisition_grad_per_ang"):
        CampaignConfig.from_dict(payload)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("mode", "apply", "error_calibration.mode"),
        ("min_records_to_apply", 0, "min_records_to_apply"),
        ("n_bins", 0, "n_bins"),
        ("min_bin_records", 0, "min_bin_records"),
        ("max_records", 0, "max_records"),
        ("max_model_age_iterations", -1, "max_model_age_iterations"),
        ("monotone_estimator", "yes", "monotone_estimator"),
        ("quantile", 0.0, "quantile"),
        ("quantile", 1.1, "quantile"),
        ("apply_strength", -0.1, "apply_strength"),
        ("apply_strength", 1.1, "apply_strength"),
        ("group_by_atom_type", "yes", "group_by_atom_type"),
        ("group_by_landing_policy", "no", "group_by_landing_policy"),
        ("model_version_policy", "recent", "model_version_policy"),
        ("output_units", "kjmol", "output_units"),
    ],
)
def test_error_calibration_validation_rejects_bad_values(field, value, match):
    payload = CampaignConfig().to_dict()
    payload["error_calibration"][field] = value
    with pytest.raises(ConfigValidationError, match=match):
        CampaignConfig.from_dict(payload)


def test_error_calibration_apply_mode_roundtrips():
    payload = CampaignConfig().to_dict()
    payload["error_calibration"]["mode"] = "apply_to_acquisition"
    payload["error_calibration"]["apply_strength"] = 0.5
    cfg = CampaignConfig.from_dict(payload)
    assert cfg.error_calibration.mode == "apply_to_acquisition"
    assert cfg.error_calibration.apply_strength == 0.5


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("spectral", "mode"), "yes", "acquisition.spectral.mode"),
        (("spectral", "mode_weighting"), "heavy", "acquisition.spectral.mode_weighting"),
        (("spectral", "omega_floor"), 0.0, "omega_floor"),
        (("spectral", "max_modes"), 0, "max_modes"),
        (("calibrated_energy", "utility"), "linear", "calibrated_energy.utility"),
        (("calibrated_energy", "band_high_ha"), -1.0, "band_high_ha"),
        (("stencils", "negative_curvature_policy"), "reward", "negative_curvature_policy"),
        (("stencils", "weak_mode_gating_enabled"), "yes", "weak_mode_gating_enabled"),
        (("stencils", "weak_mode_omega_low_fraction"), -0.1, "weak_mode_omega_low_fraction"),
        (("stencils", "weak_mode_abs_omega_floor"), 0.0, "weak_mode_abs_omega_floor"),
        (("stencils", "max_anharmonic_mode_score"), 0.0, "max_anharmonic_mode_score"),
        (("stencils", "max_anharmonic_total_score"), 0.0, "max_anharmonic_total_score"),
    ],
)
def test_mature_acquisition_config_rejects_bad_values(path, value, match):
    payload = CampaignConfig().to_dict()
    block, field = path
    payload["acquisition"][block][field] = value
    with pytest.raises(ConfigValidationError, match=match):
        CampaignConfig.from_dict(payload)


def test_weak_mode_threshold_band_must_be_ordered():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["stencils"]["weak_mode_omega_low_fraction"] = 0.20
    payload["acquisition"]["stencils"]["weak_mode_omega_high_fraction"] = 0.10
    with pytest.raises(ConfigValidationError, match="weak_mode_omega_high_fraction"):
        CampaignConfig.from_dict(payload)


def test_mature_acquisition_config_bridge_roundtrips_to_core():
    payload = CampaignConfig().to_dict()
    payload["geometry_novelty"]["fallback_scale_angstrom"] = 0.04
    payload["acquisition"]["spectral"]["mode"] = "record_only"
    payload["acquisition"]["spectral"]["max_modes"] = 3
    payload["acquisition"]["calibrated_energy"]["band_low_ha"] = 0.01
    payload["acquisition"]["calibrated_energy"]["band_high_ha"] = 0.10
    payload["acquisition"]["stencils"]["negative_curvature_policy"] = "penalise"
    payload["acquisition"]["stencils"]["lambda_negative_curvature"] = 2.0
    payload["acquisition"]["stencils"]["weak_mode_gating_enabled"] = False
    payload["acquisition"]["stencils"]["weak_mode_omega_low_fraction"] = 0.02
    payload["acquisition"]["stencils"]["weak_mode_omega_high_fraction"] = 0.12
    payload["acquisition"]["stencils"]["weak_mode_abs_omega_floor"] = 0.0002
    payload["acquisition"]["stencils"]["weak_mode_penalty"] = 3.0
    payload["acquisition"]["stencils"]["max_anharmonic_mode_score"] = 4.0
    payload["acquisition"]["stencils"]["max_anharmonic_total_score"] = 9.0
    cfg = CampaignConfig.from_dict(payload)
    core = cfg.to_acquisition_config()
    assert core.spectral.mode == "record_only"
    assert core.spectral.max_modes == 3
    assert core.calibrated_energy.band_low_ha == 0.01
    assert core.calibrated_energy.band_high_ha == 0.10
    assert core.fullspace_confinement.residual_scale == FULLSPACE_RESIDUAL_SCALE
    assert core.fullspace_confinement.fixed_residual_scale_ang == FULLSPACE_FIXED_RESIDUAL_SCALE_ANGSTROM
    assert core.fullspace_confinement.rmsd_scale_ang == pytest.approx(
        FULLSPACE_RMSD_SCALE_MULTIPLIER * 0.04
    )
    assert core.fullspace_confinement.min_residual_scale_ang == FULLSPACE_MIN_RESIDUAL_SCALE_ANGSTROM
    assert core.fullspace_confinement.failure_penalty == FULLSPACE_FAILURE_PENALTY
    assert core.movement_band.hard_min_fraction == MOVEMENT_BAND_HARD_MIN_FRACTION
    assert core.movement_band.target_low_fraction == MOVEMENT_BAND_TARGET_LOW_FRACTION
    assert core.movement_band.target_peak_fraction == MOVEMENT_BAND_TARGET_PEAK_FRACTION
    assert core.movement_band.target_high_fraction == MOVEMENT_BAND_TARGET_HIGH_FRACTION
    assert core.movement_band.hard_max_fraction == MOVEMENT_BAND_HARD_MAX_FRACTION
    assert core.movement_utility.direction == MOVEMENT_UTILITY_DIRECTION
    assert core.movement_utility.lambda_move == MOVEMENT_UTILITY_LAMBDA_MOVE
    assert core.movement_utility.band_fraction == MOVEMENT_UTILITY_BAND_FRACTION
    assert core.movement_utility.progress_fraction == MOVEMENT_UTILITY_PROGRESS_FRACTION
    assert core.movement_utility.low_softness_ang == pytest.approx(
        MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION * 0.04
    )
    assert core.movement_utility.high_softness_ang == pytest.approx(
        MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION * 0.04
    )
    assert core.stencils.negative_curvature_policy == "penalise"
    assert core.stencils.lambda_negative_curvature == 2.0
    assert core.stencils.weak_mode_gating_enabled is False
    assert core.stencils.weak_mode_omega_low_fraction == 0.02
    assert core.stencils.weak_mode_omega_high_fraction == 0.12
    assert core.stencils.weak_mode_abs_omega_floor == 0.0002
    assert core.stencils.weak_mode_penalty == 3.0
    assert core.stencils.max_anharmonic_mode_score == 4.0
    assert core.stencils.max_anharmonic_total_score == 9.0


def test_gradient_parallel_backend_rejects_unknown_values():
    payload = CampaignConfig().to_dict()
    payload["resources"]["gradient_parallel_backend"] = "forkbomb"
    with pytest.raises(ConfigValidationError, match="gradient_parallel_backend"):
        CampaignConfig.from_dict(payload)


def test_dict_roundtrip_preserves_nested_fields():
    c = CampaignConfig(max_iterations=3)
    c.phase_b.descriptor = "hybrid_alf_rmsd"
    c.split.strategy = "random_80_20"
    c.ferebus.warmstart = "never"
    c2 = CampaignConfig.from_dict(c.to_dict())
    assert c2.max_iterations == 3
    assert c2.phase_b.descriptor == "hybrid_alf_rmsd"
    assert c2.split.strategy == "random_80_20"
    assert c2.ferebus.warmstart == "never"


def test_yaml_roundtrip(tmp_path):
    c = CampaignConfig(max_iterations=7)
    c.ferebus.kernel = "rbf_per"
    p = tmp_path / "campaign.yaml"
    c.to_yaml(p)
    c2 = CampaignConfig.from_yaml(p)
    assert c2.max_iterations == 7
    assert c2.ferebus.kernel == "rbf_per"


def test_unknown_top_level_key_rejected():
    payload = CampaignConfig().to_dict()
    payload["mystery_field"] = 1
    with pytest.raises(ConfigValidationError, match="unknown keys"):
        CampaignConfig.from_dict(payload)


def test_unknown_nested_key_rejected_with_path():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["weights"]["lambda_extra"] = 1.0
    with pytest.raises(ConfigValidationError, match="acquisition.weights"):
        CampaignConfig.from_dict(payload)


def test_aimall_grid_and_naat_values_roundtrip():
    payload = CampaignConfig().to_dict()
    payload["aimall"]["naat"] = 4
    payload["aimall"]["boaq"] = "auto_gs2"
    payload["aimall"]["iasmesh"] = "medium"
    loaded = CampaignConfig.from_dict(payload)

    assert loaded.aimall.naat == 4
    assert loaded.aimall.boaq == "auto_gs2"
    assert loaded.aimall.iasmesh == "medium"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("naat", "sometimes", "aimall.naat"),
        ("naat", 99, "aimall.naat"),
        ("boaq", "unknown_grid", "aimall.boaq"),
        ("iasmesh", "tiny", "aimall.iasmesh"),
    ],
)
def test_aimall_grid_and_naat_validation(field, value, message):
    payload = CampaignConfig().to_dict()
    payload["aimall"][field] = value

    with pytest.raises(ConfigValidationError, match=message):
        CampaignConfig.from_dict(payload)


def test_unknown_ariadne_key_rejected_with_path():
    payload = CampaignConfig().to_dict()
    payload["ariadne"]["mystery"] = 99
    with pytest.raises(ConfigValidationError, match="ariadne"):
        CampaignConfig.from_dict(payload)


def test_wrong_schema_version_rejected():
    payload = CampaignConfig().to_dict()
    payload["schema_version"] = 1
    with pytest.raises(ConfigValidationError, match="schema_version"):
        CampaignConfig.from_dict(payload)


def test_invalid_descriptor_rejected():
    payload = CampaignConfig().to_dict()
    payload["phase_b"]["descriptor"] = "not_a_real_descriptor"
    with pytest.raises(ConfigValidationError):
        CampaignConfig.from_dict(payload)


@pytest.mark.parametrize(
    "path,value",
    [
        ("geometry_novelty.scale_source", "global_magic"),
        ("geometry_novelty.statistic", "mean"),
        ("geometry_novelty.scale_floor_angstrom", 0.0),
        ("geometry_novelty.history_window_iterations", -1),
        ("geometry_novelty.fallback_scale_angstrom", 0.0),
    ],
)
def test_geometry_novelty_config_validated(path, value):
    payload = CampaignConfig().to_dict()
    _set_path(payload, path, value)
    with pytest.raises(ConfigValidationError, match=path.split(".")[0]):
        CampaignConfig.from_dict(payload)


def test_schema_v4_geometry_knobs_migrate_out_of_public_config():
    payload = CampaignConfig().to_dict()
    payload["schema_version"] = 4
    payload["phase_b"]["min_separation"] = 0.20
    payload["phase_b"]["min_separation_scaled"] = 10.0
    payload["geometry_novelty"]["score_transform"] = "exponential"
    payload["acquisition"]["movement_band"] = {
        "target_peak_fraction": 0.20,
    }
    payload["acquisition"]["movement_utility"] = {
        "lambda_move": 2.0,
    }
    payload["acquisition"]["fullspace_confinement"]["rmsd_scale_ang"] = 99.0

    cfg = CampaignConfig.from_dict(payload)

    assert cfg.schema_version == 5
    assert not hasattr(cfg.phase_b, "min_separation")
    assert not hasattr(cfg.phase_b, "min_separation_scaled")
    assert not hasattr(cfg.geometry_novelty, "score_transform")
    assert not hasattr(cfg.acquisition, "movement_band")
    assert not hasattr(cfg.acquisition, "movement_utility")
    assert not hasattr(cfg.acquisition.fullspace_confinement, "rmsd_scale_ang")


def test_schema_v5_rejects_removed_geometry_knobs():
    payload = CampaignConfig().to_dict()
    payload["phase_b"]["min_separation_scaled"] = 0.5

    with pytest.raises(ConfigValidationError, match="unknown keys"):
        CampaignConfig.from_dict(payload)


def test_invalid_split_strategy_rejected():
    payload = CampaignConfig().to_dict()
    payload["split"]["strategy"] = "invalid_split"
    with pytest.raises(ConfigValidationError):
        CampaignConfig.from_dict(payload)


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/csf3_first_live_iter/campaign.yaml",
        "examples/csf4_first_live_iter/campaign.yaml",
        "examples/csf3_long_campaign/campaign.yaml",
        "examples/csf4_long_campaign/campaign.yaml",
        "examples/dry_run_water_tetramer/campaign.yaml",
    ],
)
def test_shipped_active_learning_examples_are_resource_safe(relative_path):
    repo = Path(__file__).resolve().parents[3]
    cfg = CampaignConfig.from_yaml(repo / relative_path)

    assert cfg.schema_version == CONFIG_SCHEMA_VERSION
    assert cfg.resources.defaults.partition == "multicore"
    assert cfg.resources.partition_for("PHASE_A_POLUS") == "multicore"
    assert cfg.resources.cpus_for("PHASE_A_POLUS") == "auto"
    assert cfg.resources.cpus_for("GAUSSIAN") == "auto"
    if "first_live_iter" in relative_path:
        assert cfg.resources.cpus_for("AIMALL") == "auto"
        assert cfg.resources.cpus_for("ARIADNE_ARRAY") == "auto"
        assert cfg.aimall.naat == "auto"
        assert cfg.aimall.boaq == "auto_gs2"
        assert cfg.aimall.iasmesh == "medium"
    assert cfg.resources.gaussian.memory_mode == "slurm_env"
    if "first_live_iter" in relative_path and cfg.resources.array_concurrency_limit is not None:
        assert cfg.runtime.poll_sacct_missing_max_ticks >= 3


def test_first_live_examples_match_canonical_template_exactly():
    repo = Path(__file__).resolve().parents[3]
    template = (repo / "ichor_hpc/ichor/hpc/active_learning/templates/campaign.yaml").read_text(
        encoding="utf-8"
    )
    assert (repo / "examples/csf3_first_live_iter/campaign.yaml").read_text(
        encoding="utf-8"
    ) == template
    assert (repo / "examples/csf4_first_live_iter/campaign.yaml").read_text(
        encoding="utf-8"
    ) == template


def test_invalid_warmstart_rejected():
    payload = CampaignConfig().to_dict()
    payload["ferebus"]["warmstart"] = "sometimes_maybe"
    with pytest.raises(ConfigValidationError):
        CampaignConfig.from_dict(payload)


def test_fraction_ranges_validated():
    cases = [
        ("split", "train_fraction"),
        ("split", "val_mid_fraction"),
        ("split", "high_holdout_fraction"),
        ("seed_selection", "bulk_fraction"),
        ("phase_b", "beta"),
    ]
    for block, field in cases:
        payload = CampaignConfig().to_dict()
        payload[block][field] = 1.5
        with pytest.raises(ConfigValidationError):
            CampaignConfig.from_dict(payload)


def test_failure_threshold_fraction_validated():
    payload = CampaignConfig().to_dict()
    payload["failure_threshold_fraction"] = 1.5
    with pytest.raises(ConfigValidationError):
        CampaignConfig.from_dict(payload)


def test_max_iterations_must_be_positive():
    payload = CampaignConfig().to_dict()
    payload["max_iterations"] = 0
    with pytest.raises(ConfigValidationError):
        CampaignConfig.from_dict(payload)


def test_batch_sizing_cap_must_be_at_least_floor():
    payload = CampaignConfig().to_dict()
    payload["batch_sizing"]["floor"] = 10
    payload["batch_sizing"]["cap"] = 5
    with pytest.raises(ConfigValidationError):
        CampaignConfig.from_dict(payload)


def test_anti_overlap_max_must_exceed_min():
    payload = CampaignConfig().to_dict()
    payload["anti_overlap"]["min_post_ariadne_whitened_distance"] = 5.0
    payload["anti_overlap"]["max_post_ariadne_whitened_distance"] = 1.0
    with pytest.raises(ConfigValidationError, match="max_post_ariadne"):
        CampaignConfig.from_dict(payload)


def test_invalid_gradient_mode_rejected():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["gradient"]["mode"] = "wave_function_magic"
    with pytest.raises(ConfigValidationError, match="gradient.mode"):
        CampaignConfig.from_dict(payload)


def test_invalid_mode_weighting_policy_rejected():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["mode_weighting_policy"] = "bogus"
    with pytest.raises(ConfigValidationError, match="mode_weighting_policy"):
        CampaignConfig.from_dict(payload)


def test_ariadne_block_defaults():
    ab = AriadneConfigBlock()
    assert ab.optimiser == "trust_region_qn"
    assert ab.hessian_model == "SCHLEGEL"
    assert ab.fallback_to_ds is True
    assert ab.trqn_scale_mode == "adaptive_initial_gradient_rms"
    assert ab.trqn_target_initial_grad_norm == pytest.approx(0.01)
    assert ab.trqn_retry_target_initial_grad_norm == pytest.approx(0.003)
    assert ab.trqn_target_initial_grad_rms == pytest.approx(2.0e-4)
    assert ab.trqn_retry_target_initial_grad_rms == pytest.approx(4.0e-4)
    assert ab.trqn_under_move_target_initial_grad_rms == pytest.approx(6.0e-4)
    assert ab.trqn_under_move_retry is True
    assert ab.trqn_under_move_retry_max == 1
    assert ab.trqn_min_objective_scale == pytest.approx(1.0e-8)
    assert ab.trqn_max_objective_scale == pytest.approx(1.0)
    assert ab.trqn_fixed_objective_scale == pytest.approx(1.0)
    assert ab.trqn_retry_on_no_proposal is True
    assert ab.trqn_backtransform_mode == "geodesic"
    assert ab.trqn_geodesic_bt_mode == "dense"
    assert ab.trqn_geodesic_dt == pytest.approx(1.0e-2)
    assert ab.trqn_geodesic_tol == pytest.approx(1.0e-8)
    assert ab.trqn_bt_ic_tol == pytest.approx(1.0e-6)
    assert ab.trqn_max_backtransform_iter == 50
    assert ab.trqn_trust_min == pytest.approx(1.0e-4)


def test_ferebus_fraction_sum_must_be_one():
    payload = CampaignConfig().to_dict()
    payload["ferebus"]["train_fraction"] = 0.7
    payload["ferebus"]["int_val_fraction"] = 0.1
    payload["ferebus"]["ext_val_fraction"] = 0.1
    with pytest.raises(ConfigValidationError, match="sum to 1.0"):
        CampaignConfig.from_dict(payload)


def test_ferebus_fraction_range_validated():
    payload = CampaignConfig().to_dict()
    payload["ferebus"]["train_fraction"] = -0.1
    payload["ferebus"]["int_val_fraction"] = 0.2
    payload["ferebus"]["ext_val_fraction"] = 0.9
    with pytest.raises(ConfigValidationError, match="ferebus.train_fraction"):
        CampaignConfig.from_dict(payload)


def test_acquisition_property_must_be_trained_property():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["property_name"] = "iqa"
    payload["ferebus"]["properties"] = ["q00"]
    with pytest.raises(ConfigValidationError, match="must be present"):
        CampaignConfig.from_dict(payload)


def test_non_iqa_active_acquisition_rejected_even_if_trained():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["property_name"] = "q00"
    payload["ferebus"]["properties"] = ["iqa", "q00"]
    with pytest.raises(ConfigValidationError, match="must be 'iqa'"):
        CampaignConfig.from_dict(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("lease_stale_seconds", 0),
        ("postprocess_settle_attempts", 0),
        ("postprocess_settle_seconds", -1),
        ("transient_phase_retry_max", -1),
        ("poll_sacct_unknown_max_ticks", -1),
    ],
)
def test_runtime_fields_validated(field, value):
    payload = CampaignConfig().to_dict()
    payload["runtime"][field] = value
    with pytest.raises(ConfigValidationError, match="runtime"):
        CampaignConfig.from_dict(payload)


def test_quality_gate_thresholds_are_optional_but_nonnegative():
    payload = CampaignConfig().to_dict()
    payload["quality_gates"]["max_abs_integration_error"] = -1.0
    with pytest.raises(ConfigValidationError, match="quality_gates.max_abs_integration_error"):
        CampaignConfig.from_dict(payload)


def test_seed_selection_variance_chunk_size_must_be_positive():
    payload = CampaignConfig().to_dict()
    payload["seed_selection"]["variance_chunk_size"] = 0
    with pytest.raises(ConfigValidationError, match="variance_chunk_size"):
        CampaignConfig.from_dict(payload)


def test_seed_selection_d_optimal_fields_validate_and_roundtrip():
    payload = CampaignConfig().to_dict()
    payload["seed_selection"]["strategy"] = "d_optimal"
    payload["seed_selection"]["d_optimal_pool_multiplier"] = 4
    payload["seed_selection"]["d_optimal_jitter"] = 1.0e-10
    payload["seed_selection"]["d_optimal_novelty_floor"] = 0.0
    payload["seed_selection"]["d_optimal_score_power"] = 0.5

    cfg = CampaignConfig.from_dict(payload)

    assert cfg.seed_selection.strategy == "d_optimal"
    assert cfg.seed_selection.d_optimal_pool_multiplier == 4
    assert cfg.seed_selection.d_optimal_jitter == 1.0e-10
    assert cfg.seed_selection.d_optimal_novelty_floor == 0.0
    assert cfg.seed_selection.d_optimal_score_power == 0.5


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("strategy", "maxvol_magic", "seed_selection.strategy"),
        ("d_optimal_pool_multiplier", 0, "d_optimal_pool_multiplier"),
        ("d_optimal_jitter", 0.0, "d_optimal_jitter"),
        ("d_optimal_novelty_floor", -1.0, "d_optimal_novelty_floor"),
        ("d_optimal_score_power", -0.1, "d_optimal_score_power"),
    ],
)
def test_seed_selection_d_optimal_fields_reject_bad_values(field, value, match):
    payload = CampaignConfig().to_dict()
    payload["seed_selection"][field] = value
    with pytest.raises(ConfigValidationError, match=match):
        CampaignConfig.from_dict(payload)


def test_split_train_and_mid_validation_fraction_sum_validated():
    payload = CampaignConfig().to_dict()
    payload["split"]["train_fraction"] = 0.9
    payload["split"]["val_mid_fraction"] = 0.2
    with pytest.raises(ConfigValidationError, match="train_fraction \\+ split.val_mid_fraction"):
        CampaignConfig.from_dict(payload)


def test_barrier_new_safety_terms_validated():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["barrier"]["nonbonded_expansion_delta"] = -0.1
    with pytest.raises(ConfigValidationError, match="nonbonded_expansion_delta"):
        CampaignConfig.from_dict(payload)
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["barrier"]["angle_upper_scale"] = 0.1
    with pytest.raises(ConfigValidationError, match="angle_upper_scale"):
        CampaignConfig.from_dict(payload)
