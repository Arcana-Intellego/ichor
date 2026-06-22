"""Tests for ichor.hpc.active_learning.acquisition.ariadne_runner.

We exercise the mock path comprehensively. The live path is checked only for
its NotImplementedError sentinel (it lands in M9).
"""
import json

import numpy as np
import pytest

from ichor.core.atoms import Atom, Atoms

from ichor.hpc.active_learning.acquisition.ariadne_runner import (
    AriadneRunConfig,
    AriadneRunResult,
    ariadne_result_usability_payload,
    optimise_seed,
)


def _water() -> Atoms:
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])


def test_mock_returns_run_result():
    seed = _water()
    out = optimise_seed(models=None, seed=seed, trajectory=[seed], mock=True)
    assert isinstance(out, AriadneRunResult)
    assert out.mock is True
    assert out.return_code == 0
    assert out.fell_back_to_ds is False
    assert out.n_evaluations >= 4
    assert out.wall_seconds > 0.0


def test_mock_alpha_trajectory_monotone_non_decreasing():
    seed = _water()
    out = optimise_seed(models=None, seed=seed, trajectory=[seed],
                       run_config=AriadneRunConfig(rng_seed=123), mock=True)
    assert len(out.alpha_trajectory) >= 2
    diffs = np.diff(out.alpha_trajectory)
    assert np.all(diffs >= -1.0e-12), f"alpha not monotone non-decreasing: {diffs}"


def test_mock_grad_trajectory_monotone_non_increasing():
    seed = _water()
    out = optimise_seed(models=None, seed=seed, trajectory=[seed],
                       run_config=AriadneRunConfig(rng_seed=456), mock=True)
    assert len(out.grad_norm_trajectory) >= 2
    diffs = np.diff(out.grad_norm_trajectory)
    assert np.all(diffs <= 1.0e-12), f"grad not monotone non-increasing: {diffs}"


def test_mock_is_deterministic_for_same_seed():
    seed = _water()
    cfg = AriadneRunConfig(rng_seed=99)
    a = optimise_seed(models=None, seed=seed, trajectory=[seed], run_config=cfg, mock=True)
    b = optimise_seed(models=None, seed=seed, trajectory=[seed], run_config=cfg, mock=True)
    assert a.alpha_trajectory == b.alpha_trajectory
    assert a.grad_norm_trajectory == b.grad_norm_trajectory
    np.testing.assert_array_equal(
        np.asarray(a.final_atoms.coordinates),
        np.asarray(b.final_atoms.coordinates),
    )


def test_mock_final_atoms_differ_from_seed_by_perturbation():
    seed = _water()
    cfg = AriadneRunConfig(rng_seed=1, mock_perturbation_angstrom=1.0e-3)
    out = optimise_seed(models=None, seed=seed, trajectory=[seed], run_config=cfg, mock=True)
    seed_arr = np.asarray(seed.coordinates)
    final_arr = np.asarray(out.final_atoms.coordinates)
    # Coordinates moved, but not by much (per-coord stddev = perturbation).
    assert not np.allclose(final_arr, seed_arr)
    # |delta| per coordinate is ~ 1e-3 stddev * sqrt(2) at most a few sigma.
    assert np.max(np.abs(final_arr - seed_arr)) < 1.0e-2


def test_mock_atom_types_preserved():
    seed = _water()
    out = optimise_seed(models=None, seed=seed, trajectory=[seed], mock=True)
    assert [a.type for a in out.final_atoms] == [a.type for a in seed]


def test_mock_to_dict_is_json_serialisable():
    seed = _water()
    out = optimise_seed(models=None, seed=seed, trajectory=[seed], mock=True)
    d = out.to_dict()
    s = json.dumps(d)        # must not raise
    assert "alpha_trajectory" in d
    assert d["mock"] is True
    assert d["atom_types"] == ["O", "H", "H"]
    assert d["trqn_backtransform_mode"] == "geodesic"
    assert d["trqn_geodesic_bt_mode"] == "dense"
    assert d["seed_coordinates"] == d["initial_coordinates"]
    assert d["optimiser_initial_coordinates"] == d["initial_coordinates"]
    assert d["seed_alpha"] == d["optimiser_initial_alpha"]
    assert d["optimiser_initial_origin"] == "seed_fallback"
    assert d["warm_start_alpha_delta_from_seed"] == 0.0


def test_to_dict_separates_seed_from_warm_started_initial_geometry():
    seed = _water()
    warm = Atoms([
        Atom("O", 0.10, 0.0, 0.0),
        Atom("H", 1.06, 0.0, 0.0),
        Atom("H", -0.14, 0.93, 0.0),
    ])
    out = AriadneRunResult(
        initial_atoms=seed,
        seed_atoms=seed,
        seed_alpha=1.0,
        optimiser_initial_atoms=warm,
        optimiser_initial_alpha=1.25,
        optimiser_initial_origin="gradient_band_warm_start",
        warm_start_alpha_delta_from_seed=0.25,
        final_atoms=warm,
        alpha_trajectory=[1.25, 1.4],
    )

    d = out.to_dict()

    assert d["initial_coordinates"] == np.asarray(seed.coordinates).tolist()
    assert d["seed_coordinates"] == np.asarray(seed.coordinates).tolist()
    assert d["optimiser_initial_coordinates"] == np.asarray(warm.coordinates).tolist()
    assert d["alpha_initial"] == 1.25
    assert d["seed_alpha"] == 1.0
    assert d["optimiser_initial_alpha"] == 1.25
    assert d["optimiser_initial_origin"] == "gradient_band_warm_start"
    assert d["warm_start_alpha_delta_from_seed"] == 0.25


def test_mock_to_dict_contains_landing_safety_payload():
    seed = _water()
    out = optimise_seed(models=None, seed=seed, trajectory=[seed], mock=True)
    d = out.to_dict()
    assert d["landing_safety"]["accepted"] is True
    assert d["landing_safety"]["policy"] == "mock_final"
    assert d["landing_candidates"][0]["origin"] == "mock_final"
    assert "raw_final_coordinates" in d


def test_mock_selection_diagnostics_surface_movement_and_initial_fields():
    seed = _water()
    out = optimise_seed(models=None, seed=seed, trajectory=[seed], mock=True)
    diag = out.to_dict()["selection_diagnostics"]

    for key in (
        "movement_rmsd_ang",
        "movement_band_min_ang",
        "movement_band_peak_ang",
        "movement_band_max_ang",
        "movement_utility_score",
        "movement_progress_score",
        "movement_direction_source",
        "seed_alpha",
        "optimiser_initial_alpha",
        "optimiser_initial_origin",
        "warm_start_alpha_delta_from_seed",
    ):
        assert key in diag


def test_to_dict_includes_optional_optimiser_diagnostics():
    seed = _water()
    out = AriadneRunResult(
        initial_atoms=seed,
        final_atoms=seed,
        optimiser_diagnostics={
            "schema_version": 1,
            "last_return_code_reason": "max_iterations_no_trial_evaluations",
        },
    )

    d = out.to_dict()

    assert d["optimiser_diagnostics"]["schema_version"] == 1
    assert (
        d["optimiser_diagnostics"]["last_return_code_reason"]
        == "max_iterations_no_trial_evaluations"
    )
    json.dumps(d)


def test_safe_max_iteration_result_is_task_usable():
    seed = _water()
    out = AriadneRunResult(
        initial_atoms=seed,
        final_atoms=seed,
        alpha_trajectory=[1.0, 2.0],
        grad_norm_trajectory=[3.0, 2.5],
        return_code=1,
        landing_safety={
            "accepted": True,
            "policy": "salvaged_iterate",
            "selected_origin": "accepted_iterate",
            "reasons": [],
        },
        optimiser_diagnostics={"last_return_code_reason": "max_iterations"},
    )

    usability = ariadne_result_usability_payload(out.to_dict())

    assert usability["usable"] is True
    assert usability["task_exit_code"] == 0
    assert usability["reason"] == "safe_landing_after_max_iterations"
    assert usability["optimiser_converged"] is False


def test_unsafe_max_iteration_result_is_not_task_usable():
    seed = _water()
    out = AriadneRunResult(
        initial_atoms=seed,
        final_atoms=seed,
        return_code=1,
        landing_safety={
            "accepted": False,
            "policy": "rejected",
            "reasons": ["no_safe_non_seed_landing"],
        },
    )

    usability = ariadne_result_usability_payload(out.to_dict())

    assert usability["usable"] is False
    assert usability["task_exit_code"] == 4
    assert usability["reason"] == "no_safe_non_seed_landing"


def test_backtransform_failure_with_safe_landing_is_task_usable():
    seed = _water()
    out = AriadneRunResult(
        initial_atoms=seed,
        final_atoms=seed,
        alpha_trajectory=[3.8, 4.1],
        grad_norm_trajectory=[65.0, 65.2],
        return_code=2,
        landing_safety={
            "accepted": True,
            "policy": "salvaged_iterate",
            "selected_origin": "accepted_iterate",
            "reasons": [],
            "metrics": {
                "movement_rmsd_ang": 0.025,
                "movement_band_min_ang": 0.014,
                "movement_band_max_ang": 0.177,
            },
        },
        optimiser_diagnostics={
            "last_return_code_reason": "trqn_no_proposal_backtransform_fail",
        },
    )

    usability = ariadne_result_usability_payload(out.to_dict())

    assert usability["usable"] is True
    assert usability["task_exit_code"] == 0
    assert usability["optimiser_converged"] is False
    assert usability["reason"] == "safe_landing_after_backtransform_failure"
    assert usability["safe_landing_salvaged_after_optimiser_failure"] is True


def test_backtransform_failure_still_honours_landing_safety_rejection():
    seed = _water()
    out = AriadneRunResult(
        initial_atoms=seed,
        final_atoms=seed,
        return_code=2,
        landing_safety={
            "accepted": False,
            "policy": "rejected",
            "selected_origin": "raw_final",
            "reasons": ["ariadne_landing_under_moved"],
        },
        optimiser_diagnostics={
            "last_return_code_reason": "trqn_no_proposal_backtransform_fail",
        },
    )

    usability = ariadne_result_usability_payload(out.to_dict())

    assert usability["usable"] is False
    assert usability["task_exit_code"] == 4
    assert usability["reason"] == "ariadne_landing_under_moved"


def test_backtransform_failure_still_honours_seed_fallback_policy():
    seed = _water()
    out = AriadneRunResult(
        initial_atoms=seed,
        final_atoms=seed,
        return_code=2,
        landing_safety={
            "accepted": True,
            "policy": "seed_fallback",
            "selected_origin": "seed_fallback",
            "reasons": [],
        },
        optimiser_diagnostics={
            "last_return_code_reason": "trqn_no_proposal_backtransform_fail",
        },
    )

    usability = ariadne_result_usability_payload(
        out.to_dict(),
        allow_seed_fallback=False,
    )

    assert usability["usable"] is False
    assert usability["task_exit_code"] == 4
    assert usability["reason"] == "seed_fallback_not_allowed"


def test_return_code_two_unknown_reason_is_not_task_usable():
    seed = _water()
    out = AriadneRunResult(
        initial_atoms=seed,
        final_atoms=seed,
        return_code=2,
        landing_safety={
            "accepted": True,
            "policy": "salvaged_iterate",
            "selected_origin": "accepted_iterate",
            "reasons": [],
        },
        optimiser_diagnostics={"last_return_code_reason": "optimiser_error"},
    )

    usability = ariadne_result_usability_payload(out.to_dict())

    assert usability["usable"] is False
    assert usability["task_exit_code"] == 4
    assert usability["reason"] == "ariadne_return_code_2"


def test_return_code_two_nonfinite_geometry_is_not_task_usable():
    seed = _water()
    out = AriadneRunResult(
        initial_atoms=seed,
        final_atoms=seed,
        return_code=2,
        landing_safety={
            "accepted": True,
            "policy": "salvaged_iterate",
            "selected_origin": "accepted_iterate",
            "reasons": [],
        },
        optimiser_diagnostics={
            "last_return_code_reason": "trqn_no_proposal_backtransform_fail",
        },
    )
    payload = out.to_dict()
    payload["final_coordinates"] = [[float("nan"), 0.0, 0.0]]

    usability = ariadne_result_usability_payload(payload)

    assert usability["usable"] is False
    assert usability["task_exit_code"] == 4
    assert usability["reason"] == "nonfinite_final_geometry"


def test_alpha_initial_and_final_properties():
    seed = _water()
    out = optimise_seed(models=None, seed=seed, trajectory=[seed], mock=True)
    assert out.alpha_initial == out.alpha_trajectory[0]
    assert out.alpha_final == out.alpha_trajectory[-1]


def test_live_path_needs_real_models():
    seed = _water()
    # the live path now actually composes the posterior + acquisition
    # + calculator + ARIADNE pipeline. it needs trained FEREBUS models;
    # if we hand it None for models, the posterior construction barfs
    # immediately. if models are valid, it would then try to import
    # ariadne which is not on PYTHONPATH off-cluster. either way we
    # expect an Exception -- the live path is for cluster use, not
    # for off-cluster plumbing tests.
    with pytest.raises(Exception):
        optimise_seed(models=None, seed=seed, trajectory=[seed], mock=False)
