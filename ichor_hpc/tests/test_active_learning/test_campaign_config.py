"""Tests for ichor.hpc.active_learning.config -- schema v2 (M12).

The legacy v1 flat-field tests have been retired wholesale; v2 introduces
nested blocks and the v1 schema-version is rejected.
"""
import pytest

from ichor.hpc.active_learning.config import (
    AriadneConfigBlock,
    CONFIG_SCHEMA_VERSION,
    CampaignConfig,
    ConfigValidationError,
)


def test_schema_version_is_two():
    assert CONFIG_SCHEMA_VERSION == 2


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
    assert c.seed_selection.variance_chunk_size == 512
    assert c.seed_selection.strategy == "hybrid_variance"
    assert c.seed_selection.d_optimal_pool_multiplier == 8
    assert c.seed_selection.d_optimal_jitter == 1.0e-12
    assert c.seed_selection.d_optimal_novelty_floor == 1.0e-12
    assert c.seed_selection.d_optimal_score_power == 1.0
    assert c.resources.mem_per_cpu == "auto"
    assert c.resources.array_concurrency_limit is None
    assert c.gaussian.memory_mode == "slurm_env"
    assert c.gaussian.memory_fraction_of_slurm == 0.85
    assert c.aimall.encomp == 3
    assert c.aimall.nogui is True
    assert c.quality_gates.ariadne_max_displacement_ang == 1.25
    assert c.quality_gates.ariadne_min_pair_distance_ang == 0.60
    assert c.max_acquisition_grad_per_ang is None
    assert c.effective_max_acquisition_grad_per_ang() == 50.0
    assert c.error_calibration.enabled is True
    assert c.error_calibration.mode == "record_only"
    assert c.error_calibration.min_records_to_apply == 100
    assert c.error_calibration.n_bins == 10
    assert c.error_calibration.min_bin_records == 8
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
    assert c.acquisition.fullspace_confinement.residual_scale == "local_neighbour_median"
    assert c.acquisition.fullspace_confinement.rmsd_scale_ang == 0.50
    assert c.acquisition.fullspace_confinement.min_residual_scale_ang == 1.0e-3
    assert c.acquisition.fullspace_confinement.failure_penalty == 1.0e6
    assert c.acquisition.stencils.negative_curvature_policy == "ignore"
    assert c.acquisition.stencils.lambda_negative_curvature == 1.0


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


@pytest.mark.parametrize(
    "field,value",
    [
        ("walltime_hours", 0),
        ("walltime_hours", -1),
        ("cpus_per_task", 0),
        ("ntasks", 0),
        ("ariadne_cpus_per_task", 0),
    ],
)
def test_scheduler_positive_integer_fields_validated(field, value):
    payload = CampaignConfig().to_dict()
    payload["resources"][field] = value
    with pytest.raises(ConfigValidationError, match="resources"):
        CampaignConfig.from_dict(payload)


def test_scheduler_integer_fields_reject_non_integer_values():
    payload = CampaignConfig().to_dict()
    payload["resources"]["walltime_hours"] = "24"
    with pytest.raises(ConfigValidationError, match="walltime_hours"):
        CampaignConfig.from_dict(payload)


@pytest.mark.parametrize("partition", ["multicore", "serial-debug", "gpu:shared", "csf4.test"])
def test_scheduler_partition_accepts_safe_tokens(partition):
    payload = CampaignConfig().to_dict()
    payload["resources"]["partition"] = partition
    assert CampaignConfig.from_dict(payload).resources.partition == partition


@pytest.mark.parametrize("partition", ["", " ", "multi core", "../queue", "q;rm", "$QUEUE", "q\n"])
def test_scheduler_partition_rejects_unsafe_tokens(partition):
    payload = CampaignConfig().to_dict()
    payload["resources"]["partition"] = partition
    with pytest.raises(ConfigValidationError, match="resources.partition"):
        CampaignConfig.from_dict(payload)


@pytest.mark.parametrize("mem", ["auto", "1K", "4000M", "4G", "2T"])
def test_slurm_memory_accepts_csf4_style_units(mem):
    payload = CampaignConfig().to_dict()
    payload["resources"]["mem_per_cpu"] = mem
    payload["gaussian"]["mem"] = "1MB"
    assert CampaignConfig.from_dict(payload).resources.mem_per_cpu == mem


@pytest.mark.parametrize("mem", ["", "Auto", "0G", "4GB", "4 G", "fourG", "4G;rm"])
def test_slurm_memory_rejects_invalid_units(mem):
    payload = CampaignConfig().to_dict()
    payload["resources"]["mem_per_cpu"] = mem
    with pytest.raises(ConfigValidationError, match="resources.mem_per_cpu"):
        CampaignConfig.from_dict(payload)


@pytest.mark.parametrize("mem", ["8GB", "8000MB", "8G", "500MW"])
def test_gaussian_memory_accepts_gaussian_style_units(mem):
    payload = CampaignConfig().to_dict()
    payload["gaussian"]["mem"] = mem
    assert CampaignConfig.from_dict(payload).gaussian.mem == mem


@pytest.mark.parametrize("mem", ["", "0GB", "8 GB", "eightGB", "8GB;rm"])
def test_gaussian_memory_rejects_invalid_units(mem):
    payload = CampaignConfig().to_dict()
    payload["gaussian"]["mem"] = mem
    with pytest.raises(ConfigValidationError, match="gaussian.mem"):
        CampaignConfig.from_dict(payload)


def test_gaussian_nproc_must_be_positive():
    payload = CampaignConfig().to_dict()
    payload["gaussian"]["nproc"] = 0
    with pytest.raises(ConfigValidationError, match="gaussian.nproc"):
        CampaignConfig.from_dict(payload)


def test_gaussian_slurm_env_nproc_is_independent_of_general_phase_cpus():
    payload = CampaignConfig().to_dict()
    payload["resources"]["cpus_per_task"] = 1
    payload["gaussian"]["nproc"] = 3
    cfg = CampaignConfig.from_dict(payload)
    assert cfg.gaussian.nproc == 3


def test_gaussian_link0_mem_must_leave_slurm_headroom():
    payload = CampaignConfig().to_dict()
    payload["gaussian"]["memory_mode"] = "link0"
    payload["resources"]["mem_per_cpu"] = "4G"
    payload["gaussian"]["nproc"] = 1
    payload["gaussian"]["mem"] = "4GB"
    with pytest.raises(ConfigValidationError, match="gaussian.mem"):
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
        (("fullspace_confinement", "residual_scale"), "global", "residual_scale"),
        (("fullspace_confinement", "rmsd_scale_ang"), 0.0, "rmsd_scale_ang"),
        (("fullspace_confinement", "min_residual_scale_ang"), 0.0, "min_residual_scale_ang"),
        (("fullspace_confinement", "failure_penalty"), 0.0, "failure_penalty"),
        (("stencils", "negative_curvature_policy"), "reward", "negative_curvature_policy"),
    ],
)
def test_mature_acquisition_config_rejects_bad_values(path, value, match):
    payload = CampaignConfig().to_dict()
    block, field = path
    payload["acquisition"][block][field] = value
    with pytest.raises(ConfigValidationError, match=match):
        CampaignConfig.from_dict(payload)


def test_mature_acquisition_config_bridge_roundtrips_to_core():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["spectral"]["mode"] = "record_only"
    payload["acquisition"]["spectral"]["max_modes"] = 3
    payload["acquisition"]["calibrated_energy"]["band_low_ha"] = 0.01
    payload["acquisition"]["calibrated_energy"]["band_high_ha"] = 0.10
    payload["acquisition"]["fullspace_confinement"]["residual_scale"] = "fixed"
    payload["acquisition"]["fullspace_confinement"]["fixed_residual_scale_ang"] = 0.2
    payload["acquisition"]["fullspace_confinement"]["min_residual_scale_ang"] = 0.003
    payload["acquisition"]["fullspace_confinement"]["failure_penalty"] = 123.0
    payload["acquisition"]["stencils"]["negative_curvature_policy"] = "penalise"
    payload["acquisition"]["stencils"]["lambda_negative_curvature"] = 2.0
    cfg = CampaignConfig.from_dict(payload)
    core = cfg.to_acquisition_config()
    assert core.spectral.mode == "record_only"
    assert core.spectral.max_modes == 3
    assert core.calibrated_energy.band_low_ha == 0.01
    assert core.calibrated_energy.band_high_ha == 0.10
    assert core.fullspace_confinement.residual_scale == "fixed"
    assert core.fullspace_confinement.fixed_residual_scale_ang == 0.2
    assert core.fullspace_confinement.min_residual_scale_ang == 0.003
    assert core.fullspace_confinement.failure_penalty == 123.0
    assert core.stencils.negative_curvature_policy == "penalise"
    assert core.stencils.lambda_negative_curvature == 2.0


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


def test_invalid_split_strategy_rejected():
    payload = CampaignConfig().to_dict()
    payload["split"]["strategy"] = "invalid_split"
    with pytest.raises(ConfigValidationError):
        CampaignConfig.from_dict(payload)


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
    assert ab.fallback_to_ds is True


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
