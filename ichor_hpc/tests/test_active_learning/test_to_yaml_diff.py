"""M15 F7 tests: to_yaml writes only operator-edited fields, so a
subsequent --preset overlay isn't silently neutered."""
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


def test_top_level_edit_appears_at_root():
    c = CampaignConfig()
    c.max_iterations = 999
    d = diff_against_defaults(c)
    assert d == {"schema_version": CONFIG_SCHEMA_VERSION, "max_iterations": 999}


def test_diff_is_round_trip_loadable(tmp_path):
    c = CampaignConfig()
    c.acquisition.subspace.mode_weighting_policy = "inverse_frequency"
    c.acquisition.weights.lambda_force = 2.5
    c.split.train_fraction = 0.8
    p = tmp_path / "campaign.yaml"
    c.to_yaml(p)
    text = p.read_text(encoding="utf-8")
    # Sanity: the diff is small, NOT 100+ lines like the legacy dense dump.
    assert text.count("\n") < 20
    # Loadable + equals original
    c2 = CampaignConfig.from_yaml(p)
    assert c2.acquisition.subspace.mode_weighting_policy == "inverse_frequency"
    assert c2.acquisition.weights.lambda_force == 2.5
    assert c2.split.train_fraction == 0.8
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


# --- the CRITICAL F7 regression: preset survives save-then-overlay ---


def test_preset_survives_menu_save_then_reload_with_preset(tmp_path):
    """Repro of the M13-defeating bug:

    1. Operator runs `ichor-al-daemon start --preset spectroscopy_focused`.
    2. Opens the menu, changes ONE field (max_iterations), saves to disk.
    3. Restarts with `--preset spectroscopy_focused --config campaign.yaml`.

    Pre-F7: the saved YAML covered EVERY key, so deep_merge gave campaign.yaml
    the win on every preset-tuned value. Preset became a no-op silently.

    Post-F7: the saved YAML carries only the operator's max_iterations edit,
    so the preset's mode_weighting_policy, lambda_frequency, etc. survive.
    """
    from ichor.hpc.active_learning.preset_loader import apply_preset

    # Stage 1: start with the preset (no campaign.yaml yet).
    sparse_initial = {"schema_version": CONFIG_SCHEMA_VERSION}
    effective_initial, _ = apply_preset("spectroscopy_focused", sparse_initial)
    c1 = CampaignConfig.from_dict(effective_initial)

    # Sanity: the preset gave us inverse_frequency + max_subspace_dim=10
    assert c1.acquisition.subspace.mode_weighting_policy == "inverse_frequency"
    assert c1.acquisition.subspace.max_subspace_dim == 10
    assert c1.acquisition.weights.lambda_frequency == 2.5

    # Stage 2: operator changes one field via the menu and Saves.
    c1.max_iterations = 250
    saved_path = tmp_path / "campaign.yaml"
    c1.to_yaml(saved_path)

    # Stage 3: restart -- load campaign.yaml + apply preset on top.
    import yaml as _yaml
    with open(saved_path, "r", encoding="utf-8") as f:
        sparse_after_save = _yaml.safe_load(f) or {}
    effective_after, _ = apply_preset("spectroscopy_focused", sparse_after_save)
    c2 = CampaignConfig.from_dict(effective_after)

    # The operator's max_iterations edit must survive.
    assert c2.max_iterations == 250

    # AND the preset's spectroscopy choices must ALSO survive (this is the
    # regression that motivated F7).
    assert c2.acquisition.subspace.mode_weighting_policy == "inverse_frequency"
    assert c2.acquisition.subspace.max_subspace_dim == 10
    assert c2.acquisition.weights.lambda_frequency == 2.5
    assert c2.acquisition.gradient.mode == "active_fd"


def test_save_load_round_trip_preserves_config(tmp_path):
    """Regression: even with the new diff serialisation, a CampaignConfig
    that's been through to_yaml -> from_yaml must equal the original."""
    c = CampaignConfig()
    c.phase_b.beta = 0.7
    c.split.strategy = "random_80_20"
    c.ferebus.kernel = "rbf_per"
    p = tmp_path / "c.yaml"
    c.to_yaml(p)
    c2 = CampaignConfig.from_yaml(p)
    assert c2 == c


def test_empty_diff_yaml_is_minimal(tmp_path):
    """A pristine config produces a tiny YAML (just schema_version)."""
    p = tmp_path / "c.yaml"
    CampaignConfig().to_yaml(p)
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert payload == {"schema_version": CONFIG_SCHEMA_VERSION}
