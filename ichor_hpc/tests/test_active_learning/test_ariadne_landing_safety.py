from types import SimpleNamespace

import numpy as np

from ichor.core.adversarial.subspace import LocalSubspace
from ichor.core.atoms import Atom, Atoms
from ichor.hpc.active_learning.acquisition.ariadne_runner import (
    _copy_atoms_with_coords,
    _select_safe_landing,
)


def _one_atom(x=0.0):
    return Atoms([Atom("O", float(x), 0.0, 0.0)])


class _Posterior:
    def mean(self, atoms):
        return float(np.asarray(atoms.coordinates, dtype=float)[0, 0])


class _FakeAcquisition:
    def __init__(self, seed):
        self.config = SimpleNamespace(
            subspace=SimpleNamespace(covariance_regularization=1.0e-10),
        )
        self.subspace = LocalSubspace(
            seed_atoms=seed.copy(),
            neighbours=[],
            covariance=np.eye(3),
            basis=np.array([[1.0], [0.0], [0.0]]),
            eigenvalues=np.array([1.0]),
            mode_weights=np.array([1.0]),
            mass_vector=np.ones(3),
            active_covariance=np.array([[1.0]]),
            neighbour_weights=np.array([]),
        )
        self.posterior = _Posterior()

    def components(self, atoms):
        x = float(np.asarray(atoms.coordinates, dtype=float)[0, 0])
        return SimpleNamespace(
            total=x,
            informativeness_score=x,
            risk_penalty_score=0.0,
            mean_energy=x,
            energy_variance=abs(x),
            raw_energy_risk=abs(x),
            energy_risk=abs(x),
            banded_energy_risk=None,
            calibrated_expected_iqa_error_ha=None,
            calibration_applied=False,
            spectral_frequency_risk=0.0,
            legacy_frequency_risk=0.0,
            fullspace_residual_distance=0.0,
            fullspace_residual_penalty=0.0,
            aligned_rmsd_ang=abs(x),
            aligned_rmsd_penalty=0.0,
            observable_score=x,
            outlier_penalty_score=0.0,
            fallback_reasons=[],
            mode_evaluations=[],
            chemistry_penalty=0.0,
            distance_penalty=x * x,
        )


def _safety(**overrides):
    values = {
        "enabled": True,
        "reject_unsafe_landings": True,
        "salvage_safe_iterate": True,
        "backtrack_to_safe_landing": True,
        "backtrack_points": 4,
        "allow_seed_fallback": False,
        "min_whitened_distance": 0.0,
        "max_whitened_distance": 10.0,
        "enforce_min_whitened_distance": False,
        "max_predicted_energy_delta_ha": None,
        "max_energy_variance": None,
        "max_chemistry_penalty": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _select(raw_x, candidate_xs, safety):
    seed = _one_atom(0.0)
    acquisition = _FakeAcquisition(seed)
    raw = _copy_atoms_with_coords(seed, np.array([[raw_x, 0.0, 0.0]]))
    positions = [np.array([[0.0, 0.0, 0.0]])]
    positions.extend(np.array([[x, 0.0, 0.0]]) for x in candidate_xs)
    return _select_safe_landing(
        acquisition=acquisition,
        seed_atoms=seed,
        raw_final_atoms=raw,
        opt_candidate_positions=positions,
        opt_candidate_alphas=[0.0] + [float(x) for x in candidate_xs],
        opt_candidate_grad_norms=[0.0 for _ in positions],
        alpha_trajectory=[0.0, float(raw_x)],
        safety_config=safety,
        quality_gates=SimpleNamespace(
            ariadne_max_displacement_ang=None,
            ariadne_min_pair_distance_ang=None,
        ),
    )


def test_safe_raw_final_selected_unchanged():
    out = _select(raw_x=1.0, candidate_xs=[], safety=_safety())
    assert out["landing_safety"]["accepted"] is True
    assert out["landing_safety"]["policy"] == "raw_final"
    assert np.asarray(out["selected_atoms"].coordinates)[0, 0] == 1.0


def test_unsafe_raw_final_salvages_safe_iterate():
    out = _select(
        raw_x=3.0,
        candidate_xs=[1.0],
        safety=_safety(
            max_predicted_energy_delta_ha=1.5,
            backtrack_to_safe_landing=False,
        ),
    )
    assert out["landing_safety"]["accepted"] is True
    assert out["landing_safety"]["policy"] == "salvaged_iterate"
    assert np.asarray(out["selected_atoms"].coordinates)[0, 0] == 1.0


def test_unsafe_raw_final_backtracks_when_no_safe_iterate():
    out = _select(
        raw_x=3.0,
        candidate_xs=[],
        safety=_safety(max_predicted_energy_delta_ha=1.5),
    )
    assert out["landing_safety"]["accepted"] is True
    assert out["landing_safety"]["policy"] == "backtracked"
    assert np.asarray(out["selected_atoms"].coordinates)[0, 0] < 3.0
    assert out["landing_safety"]["metrics"]["predicted_energy_delta_ha"] <= 1.5


def test_no_safe_non_seed_landing_rejects_seed():
    out = _select(
        raw_x=3.0,
        candidate_xs=[],
        safety=_safety(
            max_predicted_energy_delta_ha=0.1,
            backtrack_to_safe_landing=False,
        ),
    )
    assert out["landing_safety"]["accepted"] is False
    assert "no_safe_non_seed_landing" in out["landing_safety"]["reasons"]


def test_seed_equivalent_raw_final_rejects_when_seed_fallback_disabled():
    out = _select(raw_x=0.0, candidate_xs=[], safety=_safety())

    safety = out["landing_safety"]
    assert safety["accepted"] is False
    assert safety["policy"] == "unsafe_raw_final"
    assert "no_safe_non_seed_landing" in safety["reasons"]
    assert "ariadne_landing_is_seed" in safety["reasons"]
    assert "seed_fallback_disabled" in safety["reasons"]
    assert safety["raw_final"]["accepted"] is False
    assert safety["raw_final"]["metrics"]["seed_equivalent"] is True


def test_seed_equivalent_raw_final_accepts_only_when_seed_fallback_enabled():
    out = _select(
        raw_x=0.0,
        candidate_xs=[],
        safety=_safety(allow_seed_fallback=True),
    )

    safety = out["landing_safety"]
    assert safety["accepted"] is True
    assert safety["policy"] == "seed_fallback"
    assert safety["selected_origin"] == "seed_fallback"
    assert "ariadne_landing_is_seed" in safety["record_only_reasons"]
    assert safety["metrics"]["seed_equivalent"] is True
