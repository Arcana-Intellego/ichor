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


def test_mock_to_dict_contains_landing_safety_payload():
    seed = _water()
    out = optimise_seed(models=None, seed=seed, trajectory=[seed], mock=True)
    d = out.to_dict()
    assert d["landing_safety"]["accepted"] is True
    assert d["landing_safety"]["policy"] == "mock_final"
    assert d["landing_candidates"][0]["origin"] == "mock_final"
    assert "raw_final_coordinates" in d


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
