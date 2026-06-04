"""M12 schema-v2 specific tests: translator round-trip, dead-letter cleanup
proof, generic parser path-qualified error messages, subspace-dim guard.
"""
import pytest

from ichor.hpc.active_learning.config import (
    AcquisitionConfigBlock,
    CampaignConfig,
    ConfigValidationError,
)
from ichor.hpc.active_learning.config_dataclass import (
    DataclassParseError,
    parse_dataclass_block,
)


# --- generic parser ---------------------------------------------------


def test_parser_rejects_unknown_acquisition_key_with_full_path():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["barrier"]["clash_lamba"] = 1.0  # typo
    with pytest.raises(ConfigValidationError) as exc:
        CampaignConfig.from_dict(payload)
    assert "acquisition.barrier" in str(exc.value)


def test_parser_rejects_unknown_subspace_key_with_full_path():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["neighbor_count"] = 100  # US spelling typo
    with pytest.raises(ConfigValidationError) as exc:
        CampaignConfig.from_dict(payload)
    assert "acquisition.subspace" in str(exc.value)
    assert "neighbor_count" in str(exc.value)


def test_parser_int_type_mismatch_path_qualified():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["neighbour_count"] = "fifty"
    with pytest.raises(ConfigValidationError) as exc:
        CampaignConfig.from_dict(payload)
    assert "neighbour_count" in str(exc.value)


def test_parser_bool_type_mismatch_path_qualified():
    payload = CampaignConfig().to_dict()
    payload["anti_overlap"]["skip_training_seeds"] = "yes"
    with pytest.raises(ConfigValidationError) as exc:
        CampaignConfig.from_dict(payload)
    assert "skip_training_seeds" in str(exc.value)


def test_ferebus_properties_default_and_multipole_roundtrip():
    c = CampaignConfig()
    assert c.ferebus.properties == ["iqa"]
    payload = c.to_dict()
    payload["ferebus"]["properties"] = ["iqa", "q00", "q10", "q44s"]
    loaded = CampaignConfig.from_dict(payload)
    assert loaded.ferebus.properties == ["iqa", "q00", "q10", "q44s"]


@pytest.mark.parametrize("properties", [
    [],
    ["iqa", "iqa"],
    ["not_a_property"],
    ["iqa", 7],
])
def test_ferebus_properties_validation(properties):
    payload = CampaignConfig().to_dict()
    payload["ferebus"]["properties"] = properties
    with pytest.raises(ConfigValidationError, match="ferebus.properties"):
        CampaignConfig.from_dict(payload)


def test_acquisition_uniform_posterior_fallback_default_and_validation():
    cfg = CampaignConfig()
    assert cfg.acquisition.allow_uniform_posterior_fallback is False
    payload = cfg.to_dict()
    payload["acquisition"]["allow_uniform_posterior_fallback"] = True
    loaded = CampaignConfig.from_dict(payload)
    assert loaded.acquisition.allow_uniform_posterior_fallback is True
    payload["acquisition"]["allow_uniform_posterior_fallback"] = "yes"
    with pytest.raises(ConfigValidationError):
        CampaignConfig.from_dict(payload)


def test_adversarial_safety_defaults_roundtrip():
    cfg = CampaignConfig()
    assert cfg.adversarial_safety.enabled is True
    assert cfg.adversarial_safety.reject_unsafe_landings is True
    assert cfg.adversarial_safety.backtrack_points == 16
    assert cfg.adversarial_safety.max_whitened_distance == pytest.approx(10.0)
    payload = cfg.to_dict()
    payload["adversarial_safety"]["max_whitened_distance"] = 7.5
    loaded = CampaignConfig.from_dict(payload)
    assert loaded.adversarial_safety.max_whitened_distance == pytest.approx(7.5)


def test_adversarial_safety_validation_rejects_bad_values():
    payload = CampaignConfig().to_dict()
    payload["adversarial_safety"]["enabled"] = "yes"
    with pytest.raises(ConfigValidationError, match="adversarial_safety.enabled"):
        CampaignConfig.from_dict(payload)
    payload = CampaignConfig().to_dict()
    payload["adversarial_safety"]["backtrack_points"] = 0
    with pytest.raises(ConfigValidationError, match="backtrack_points"):
        CampaignConfig.from_dict(payload)
    payload = CampaignConfig().to_dict()
    payload["adversarial_safety"]["min_whitened_distance"] = 2.0
    payload["adversarial_safety"]["max_whitened_distance"] = 2.0
    with pytest.raises(ConfigValidationError, match="max_whitened_distance"):
        CampaignConfig.from_dict(payload)


# --- translator: to_acquisition_config --------------------------------


def test_to_acquisition_config_propagates_nested_values():
    c = CampaignConfig()
    c.acquisition.weights.lambda_force = 2.5
    c.acquisition.subspace.neighbour_count = 73
    c.acquisition.barrier.use_connectivity_barrier = False
    ac = c.to_acquisition_config()
    assert ac.weights.lambda_force == 2.5
    assert ac.subspace.neighbour_count == 73
    assert ac.barrier.use_connectivity_barrier is False


def test_subspace_dim_guard_fires_with_cartesian_fd():
    c = CampaignConfig()
    c.acquisition.subspace.max_subspace_dim = 10
    assert c.acquisition.gradient.mode == "cartesian_fd"
    with pytest.raises(ConfigValidationError, match="max_subspace_dim"):
        c.to_acquisition_config()


def test_subspace_dim_guard_passes_with_active_fd():
    c = CampaignConfig()
    c.acquisition.subspace.max_subspace_dim = 10
    c.acquisition.gradient.mode = "active_fd"
    ac = c.to_acquisition_config()
    assert ac.subspace.max_subspace_dim == 10
    assert ac.gradient.mode == "active_fd"


# --- translator: to_ariadne_run_config --------------------------------


def test_to_ariadne_run_config_propagates_values():
    c = CampaignConfig()
    c.ariadne.max_iter = 350
    c.ariadne.gradf_tol = 1.0e-5
    c.ariadne.delta0 = 0.07
    rc = c.to_ariadne_run_config()
    assert rc.max_iter == 350
    assert rc.gradf_tol == pytest.approx(1.0e-5)
    assert rc.delta0 == pytest.approx(0.07)


# --- dead-letter cleanup proof: every previously-orphaned field is read --


def test_phase_b_beta_reaches_descriptor_factory():
    from ichor.hpc.active_learning.sampling.descriptors import (
        HybridAlfRmsdDescriptor,
        build_descriptor_from_config,
    )
    c = CampaignConfig()
    c.phase_b.beta = 0.42
    d = build_descriptor_from_config(c)
    assert isinstance(d, HybridAlfRmsdDescriptor)
    assert d.beta == pytest.approx(0.42)


def test_split_val_mid_and_high_holdout_consumed_by_executor(tmp_path):
    import json
    from types import SimpleNamespace
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import CampaignPhase

    c = CampaignConfig()
    c.split.val_mid_fraction = 0.22
    c.split.high_holdout_fraction = 0.07
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=c)
    ex.submit_or_run(SimpleNamespace(iteration=0), CampaignPhase.SPLIT)
    split_json = tmp_path / "7_ACTIVE_LEARNING" / "iteration-0000" / "split.json"
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    assert payload["val_mid_fraction"] == pytest.approx(0.22)
    assert payload["high_holdout_fraction"] == pytest.approx(0.07)


def test_seed_selection_bulk_fraction_consumed_by_executor(tmp_path):
    """bulk_fraction was a dead-letter in v1; in v2 it flows into
    select_seeds via _inline_seed_select."""
    import json
    from types import SimpleNamespace
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import CampaignPhase
    from ichor.hpc.active_learning.acquisition.trajectory_pool import (
        TrajectoryPool,
    )
    from pathlib import Path

    FIXTURE = (
        Path(__file__).resolve().parent / "fixtures" / "water_tetramer.xyz"
    )
    c = CampaignConfig()
    c.seed_selection.n_seeds_per_iteration = 4
    c.seed_selection.bulk_fraction = 0.0  # pure variance pick
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=c)
    TrajectoryPool.import_from(FIXTURE, tmp_path)
    ex.submit_or_run(SimpleNamespace(iteration=0), CampaignPhase.SEED_SELECT)
    picked = json.loads(
        (tmp_path / "7_ACTIVE_LEARNING" / "iteration-0000" / "seeds_picked.json")
        .read_text(encoding="utf-8")
    )
    # bulk_fraction=0 means every pick is variance-based; bulk_indices empty.
    assert picked["bulk_indices"] == []
    assert len(picked["variance_indices"]) == 4


# --- M15 F4: subspace-dim cross-validation -----------------------------


def test_min_subspace_dim_exceeding_max_rejected():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["min_subspace_dim"] = 8
    payload["acquisition"]["subspace"]["max_subspace_dim"] = 6
    with pytest.raises(ConfigValidationError, match="min_subspace_dim"):
        CampaignConfig.from_dict(payload)


def test_neighbour_count_below_max_subspace_dim_rejected():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["neighbour_count"] = 4
    payload["acquisition"]["subspace"]["max_subspace_dim"] = 6
    with pytest.raises(ConfigValidationError, match="neighbour_count"):
        CampaignConfig.from_dict(payload)


def test_neighbour_count_zero_rejected():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["neighbour_count"] = 0
    with pytest.raises(ConfigValidationError, match="neighbour_count"):
        CampaignConfig.from_dict(payload)
