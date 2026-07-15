"""Daemon-owned ARIADNE convergence policy tests."""
import pytest

from ichor.hpc.active_learning.acquisition.ariadne_local_runner import (
    _resolved_convergence,
)
from ichor.hpc.active_learning.acquisition.ariadne_runner import (
    AriadneConvergenceConfig,
    AriadneRunConfig,
    resolve_ariadne_convergence,
)


def test_fixed_convergence_preserves_editable_base_thresholds():
    config = AriadneConvergenceConfig(
        objective_change_tolerance=2.0e-6,
        gradient_rms_tolerance_per_ang=3.0e-4,
        gradient_max_tolerance_per_ang=4.0e-4,
        step_rms_tolerance_ang=5.0e-3,
        step_max_tolerance_ang=6.0e-3,
    )

    resolved = resolve_ariadne_convergence(
        config,
        initial_acquisition_score=8.0,
        initial_trust_scale_ang=0.2,
    )

    assert resolved["mode"] == "fixed"
    assert resolved["score_multiplier"] == 1.0
    assert resolved["length_multiplier"] == 1.0
    assert resolved["gradient_multiplier"] == 1.0
    assert resolved["effective_thresholds"] == {
        "objective_change_tolerance": 2.0e-6,
        "gradient_rms_tolerance_per_ang": 3.0e-4,
        "gradient_max_tolerance_per_ang": 4.0e-4,
        "step_rms_tolerance_ang": 5.0e-3,
        "step_max_tolerance_ang": 6.0e-3,
    }


def test_scale_adaptive_convergence_freezes_documented_multipliers():
    resolved = resolve_ariadne_convergence(
        AriadneConvergenceConfig(mode="scale_adaptive"),
        initial_acquisition_score=2.0,
        initial_trust_scale_ang=0.1,
    )

    assert resolved["score_multiplier"] == pytest.approx(2.0)
    assert resolved["length_multiplier"] == pytest.approx(2.0)
    assert resolved["gradient_multiplier"] == pytest.approx(1.0)
    assert resolved["effective_thresholds"]["objective_change_tolerance"] == pytest.approx(2.0e-6)
    assert resolved["effective_thresholds"]["step_max_tolerance_ang"] == pytest.approx(3.6e-3)


def test_scale_adaptive_runner_rejects_missing_frozen_seed_evidence():
    run_config = AriadneRunConfig(
        convergence=AriadneConvergenceConfig(mode="scale_adaptive"),
        resolved_convergence=None,
    )

    with pytest.raises(RuntimeError, match="requires frozen seed-scale evidence"):
        _resolved_convergence(run_config)


def test_convergence_resolver_rejects_zero_accepted_step_streak():
    with pytest.raises(ValueError, match="positive integer"):
        resolve_ariadne_convergence(
            AriadneConvergenceConfig(consecutive_accepted_steps=0),
            initial_acquisition_score=1.0,
            initial_trust_scale_ang=0.05,
        )
