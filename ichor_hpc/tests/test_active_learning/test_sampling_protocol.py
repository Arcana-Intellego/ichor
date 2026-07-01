import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.geometry_protocol import PHASE_B_MIN_SEPARATION_SCALE
from ichor.hpc.active_learning.sampling_protocol import (
    hidden_sampling_overrides,
    phase_b_min_separation_from_resolved,
    preview_sampling_protocol,
    read_sampling_protocol_resolved,
    resolve_sampling_protocol,
    sampling_protocol_resolved_path,
)


def test_level_five_preview_matches_current_balanced_defaults():
    cfg = CampaignConfig()

    resolved = preview_sampling_protocol(cfg)

    assert resolved.sampling_aggressiveness == 5
    assert resolved.resolved_geometry_scale_angstrom == pytest.approx(0.05)
    assert resolved.phase_b["min_separation_scaled"] == pytest.approx(
        PHASE_B_MIN_SEPARATION_SCALE
    )
    assert resolved.phase_b["effective_min_separation_angstrom"] == pytest.approx(
        PHASE_B_MIN_SEPARATION_SCALE * 0.05
    )
    assert resolved.adversarial_safety.max_whitened_distance == pytest.approx(10.0)
    assert resolved.adversarial_safety.backtrack_points == 16
    assert resolved.quality_gates.ariadne_max_displacement_ang == pytest.approx(1.25)
    assert resolved.quality_gates.ariadne_min_pair_distance_ang == pytest.approx(0.60)
    assert resolved.acquisition_config.weights.lambda_distance == pytest.approx(1.0)
    assert resolved.acquisition_config.fullspace_confinement.lambda_residual == pytest.approx(
        0.5
    )
    assert resolved.ariadne_run_config.delta0 == pytest.approx(0.10)
    assert resolved.ariadne_run_config.delta_max == pytest.approx(0.40)


def test_aggressiveness_profiles_move_from_conservative_to_exploratory():
    conservative = CampaignConfig()
    conservative.sampling_protocol.sampling_aggressiveness = 1
    exploratory = CampaignConfig()
    exploratory.sampling_protocol.sampling_aggressiveness = 10

    low = preview_sampling_protocol(conservative)
    high = preview_sampling_protocol(exploratory)

    assert high.profile.geometry_fallback_scale_angstrom > low.profile.geometry_fallback_scale_angstrom
    assert high.profile.max_whitened_distance > low.profile.max_whitened_distance
    assert high.profile.lambda_distance < low.profile.lambda_distance
    assert high.profile.ariadne_max_displacement_ang > low.profile.ariadne_max_displacement_ang
    assert high.ariadne_run_config.delta_max > low.ariadne_run_config.delta_max


def test_hidden_low_level_overrides_are_reported_not_applied():
    cfg = CampaignConfig()
    cfg.acquisition.weights.lambda_distance = 99.0
    cfg.geometry_novelty.fallback_scale_angstrom = 9.0
    cfg.phase_b.beta = 0.9
    cfg.quality_gates.ariadne_min_pair_distance_ang = 0.2

    overrides = hidden_sampling_overrides(cfg)
    paths = {entry["path"] for entry in overrides}

    assert "acquisition.weights.lambda_distance" in paths
    assert "geometry_novelty.fallback_scale_angstrom" in paths
    assert "phase_b.beta" in paths
    assert "quality_gates.ariadne_min_pair_distance_ang" in paths

    resolved = preview_sampling_protocol(cfg)

    assert resolved.acquisition_config.weights.lambda_distance == pytest.approx(1.0)
    assert resolved.geometry_scale_payload["scale_angstrom"] == pytest.approx(0.05)
    assert resolved.phase_b["beta"] == pytest.approx(CampaignConfig().phase_b.beta)
    assert resolved.quality_gates.ariadne_min_pair_distance_ang == pytest.approx(0.60)


def test_resolver_writes_round_trippable_manifest(tmp_path):
    cfg = CampaignConfig()

    resolved = resolve_sampling_protocol(tmp_path, cfg, iteration=2)

    iter_dir = tmp_path / "7_ACTIVE_LEARNING" / "iteration-0002"
    expected_path = sampling_protocol_resolved_path(iter_dir)
    assert resolved.manifest_path == expected_path
    assert expected_path.exists()

    payload = read_sampling_protocol_resolved(iter_dir, expected_iteration=2)
    assert payload["schema_version"] == 1
    assert payload["sampling_aggressiveness"] == 5
    assert payload["resolved_phase_b"]["effective_min_separation_angstrom"] == pytest.approx(
        PHASE_B_MIN_SEPARATION_SCALE * 0.05
    )
    assert payload["resolved_adversarial_safety"]["max_whitened_distance"] == pytest.approx(
        10.0
    )

    threshold, mode = phase_b_min_separation_from_resolved(resolved)
    assert threshold == pytest.approx(PHASE_B_MIN_SEPARATION_SCALE * 0.05)
    assert mode in {"absolute", "scaled_geometry_novelty"}
