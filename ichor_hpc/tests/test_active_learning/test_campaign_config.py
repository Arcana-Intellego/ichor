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
    assert c.resources.mem_per_cpu == "8G"
    assert c.aimall.encomp == 3
    assert c.aimall.nogui is True
    assert c.quality_gates.ariadne_max_displacement_ang == 1.25
    assert c.quality_gates.ariadne_min_pair_distance_ang == 0.60
    assert c.max_acquisition_grad_per_ang is None
    assert c.effective_max_acquisition_grad_per_ang() == 50.0


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


@pytest.mark.parametrize("mem", ["1K", "4000M", "4G", "2T"])
def test_slurm_memory_accepts_csf4_style_units(mem):
    payload = CampaignConfig().to_dict()
    payload["resources"]["mem_per_cpu"] = mem
    payload["gaussian"]["mem"] = "1MB"
    assert CampaignConfig.from_dict(payload).resources.mem_per_cpu == mem


@pytest.mark.parametrize("mem", ["", "0G", "4GB", "4 G", "fourG", "4G;rm"])
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


def test_gaussian_nproc_must_not_exceed_allocated_cpus():
    payload = CampaignConfig().to_dict()
    payload["resources"]["cpus_per_task"] = 2
    payload["gaussian"]["nproc"] = 3
    with pytest.raises(ConfigValidationError, match="gaussian.nproc"):
        CampaignConfig.from_dict(payload)


def test_gaussian_mem_must_not_exceed_slurm_task_allocation():
    payload = CampaignConfig().to_dict()
    payload["resources"]["mem_per_cpu"] = "4G"
    payload["resources"]["cpus_per_task"] = 1
    payload["gaussian"]["mem"] = "8GB"
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
