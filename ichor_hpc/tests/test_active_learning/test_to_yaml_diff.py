"""Sparse and dense campaign YAML serialisation tests."""
import yaml
import pytest

from ichor.hpc.active_learning.config import (
    CONFIG_SCHEMA_VERSION,
    CampaignConfig,
    diff_against_defaults,
)


# --- diff_against_defaults helper ---


def test_defaults_only_diff_has_just_schema_version():
    d = diff_against_defaults(CampaignConfig())
    assert d == {"schema_version": CONFIG_SCHEMA_VERSION}


def test_single_nested_edit_serialises_minimally():
    c = CampaignConfig()
    c.acquisition.weights.lambda_force = 7.5
    d = diff_against_defaults(c)
    assert d == {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "acquisition": {"weights": {"lambda_force": 7.5}},
    }


def test_campaign_identity_alias_serialises_in_campaign_block():
    c = CampaignConfig()
    c.max_iterations = 999
    d = diff_against_defaults(c)
    assert d == {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "campaign": {"max_iterations": 999},
    }


def test_diff_is_round_trip_loadable(tmp_path):
    c = CampaignConfig()
    c.acquisition.subspace.mode_weighting_policy = "inverse_frequency"
    c.acquisition.weights.lambda_force = 2.5
    c.point_allocation.batch_training_size = 5
    p = tmp_path / "campaign.yaml"
    c.to_yaml(p)
    text = p.read_text(encoding="utf-8")
    # Sanity: the diff is small, NOT 100+ lines like the legacy dense dump.
    assert text.count("\n") < 20
    # Loadable + equals original
    c2 = CampaignConfig.from_yaml(p)
    assert c2.acquisition.subspace.mode_weighting_policy == "inverse_frequency"
    assert c2.acquisition.weights.lambda_force == 2.5
    assert c2.point_allocation.batch_training_size == 5
    assert c2 == c


def test_to_yaml_dense_still_dumps_everything(tmp_path):
    p = tmp_path / "campaign.yaml"
    CampaignConfig().to_yaml_dense(p)
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    # Should have all the nested blocks fully populated
    assert "acquisition" in payload
    assert "subspace" in payload["acquisition"]
    assert "weights" in payload["acquisition"]
    assert payload["acquisition"]["weights"]["lambda_force"] == 1.0


def test_save_load_round_trip_preserves_config(tmp_path):
    """Regression: even with the new diff serialisation, a CampaignConfig
    that's been through to_yaml -> from_yaml must equal the original."""
    c = CampaignConfig()
    c.phase_b.beta = 0.7
    c.point_allocation.batch_internal_validation_size = 2
    c.ferebus.kernel = "rbf_per"
    p = tmp_path / "c.yaml"
    c.to_yaml(p)
    c2 = CampaignConfig.from_yaml(p)
    assert c2 == c


def test_active_fd_gradient_settings_round_trip_through_yaml(tmp_path):
    c = CampaignConfig()
    c.acquisition.gradient.mode = "active_fd"
    c.acquisition.gradient.active_step = 3.0e-3
    c.acquisition.gradient.regularization = 4.0e-9

    p = tmp_path / "campaign.yaml"
    c.to_yaml(p)
    c2 = CampaignConfig.from_yaml(p)

    assert c2.acquisition.gradient.mode == "active_fd"
    assert c2.acquisition.gradient.active_step == pytest.approx(3.0e-3)
    assert c2.acquisition.gradient.regularization == pytest.approx(4.0e-9)
    assert c2 == c


def test_empty_diff_yaml_is_minimal(tmp_path):
    """A pristine config produces a tiny YAML (just schema_version)."""
    p = tmp_path / "c.yaml"
    CampaignConfig().to_yaml(p)
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert payload == {"schema_version": CONFIG_SCHEMA_VERSION}
