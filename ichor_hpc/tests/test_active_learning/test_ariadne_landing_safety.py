from types import SimpleNamespace

import numpy as np
import pytest

from ichor.core.adversarial.subspace import LocalSubspace
from ichor.core.atoms import Atom, Atoms
from ichor.hpc.active_learning.acquisition.ariadne_runner import (
    AriadneRunConfig,
    _copy_atoms_with_coords,
    _landing_needs_under_move_retry,
    _rank_landing_candidate,
    _select_safe_gradient_band_warm_start,
    _select_safe_landing,
    _size_normalised_trust_radius,
    _under_move_trust_feedback,
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

    def components(self, atoms, **_kwargs):
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
            calibrated_expected_iqa_error_ha_per_sqrt_atom=None,
            calibration_applied=False,
            spectral_frequency_risk=0.0,
            legacy_frequency_risk=0.0,
            fullspace_residual_distance=0.0,
            fullspace_residual_penalty=0.0,
            aligned_rmsd_ang=abs(x),
            aligned_rmsd_penalty=0.0,
            movement_metric="aligned_active_rmsd",
            movement_rmsd_ang=abs(x),
            movement_progress_ang=x,
            movement_band_min_ang=0.1,
            movement_band_low_ang=0.2,
            movement_band_peak_ang=0.5,
            movement_band_high_ang=2.0,
            movement_band_max_ang=10.0,
            movement_utility_score=0.0,
            movement_band_score=1.0,
            movement_progress_score=1.0,
            movement_direction_source="test",
            n_effective_movement_atoms=1.0,
            observable_score=x,
            outlier_penalty_score=0.0,
            fallback_reasons=[],
            mode_evaluations=[],
            chemistry_penalty=0.0,
            distance_penalty=x * x,
        )

    def gradient_band_probe_atoms(self):
        return (
            ("gradient_band_probe_min", _one_atom(0.05)),
            ("gradient_band_probe_peak", _one_atom(0.5)),
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
        "enforce_movement_band": True,
        "reject_over_moved": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _select(raw_x, candidate_xs, safety, origins=None, initial_x=0.0, scale_model=None):
    seed = _one_atom(0.0)
    acquisition = _FakeAcquisition(seed)
    raw = _copy_atoms_with_coords(seed, np.array([[raw_x, 0.0, 0.0]]))
    positions = [np.array([[initial_x, 0.0, 0.0]])]
    positions.extend(np.array([[x, 0.0, 0.0]]) for x in candidate_xs)
    return _select_safe_landing(
        acquisition=acquisition,
        seed_atoms=seed,
        raw_final_atoms=raw,
        opt_candidate_positions=positions,
        opt_candidate_alphas=[float(initial_x)] + [float(x) for x in candidate_xs],
        opt_candidate_grad_norms=[0.0 for _ in positions],
        opt_candidate_origins=origins,
        alpha_trajectory=[float(initial_x), float(raw_x)],
        safety_config=safety,
        quality_gates=SimpleNamespace(
            ariadne_max_displacement_ang=None,
            ariadne_min_pair_distance_ang=None,
        ),
        scale_model=scale_model,
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


def test_safe_geometry_with_lower_alpha_is_rejected():
    out = _select(
        raw_x=-1.0,
        candidate_xs=[],
        safety=_safety(backtrack_to_safe_landing=False),
    )

    safety = out["landing_safety"]
    assert safety["accepted"] is False
    assert "no_safe_non_seed_landing" in safety["reasons"]
    assert "ariadne_landing_acquisition_not_improved" in safety["reasons"]
    assert safety["raw_final"]["accepted"] is False
    assert safety["raw_final"]["metrics"]["improves_acquisition"] is False
    assert safety["raw_final"]["metrics"]["alpha_delta_from_initial"] < 0.0


def test_dimensionless_scale_gate_rejects_over_scaled_atom_move():
    scale_model = {
        "schema_version": 1,
        "model_version": 2,
        "geometry_motion_scale": {"value_angstrom": 0.10},
        "aligned_rmsd_scale": {"value_angstrom": 0.10},
        "residual_fullspace_scale": {"value_angstrom": 1.0},
        "per_atom_mobility_scales": {
            "mode": "uniform",
            "values_angstrom": [0.10],
        },
        "pair_distance_reference": {
            "reference_min_pair_distance_angstrom": 1.0,
            "ratio_floor": 0.0,
        },
        "dimensionless_policy": {
            "max_scaled_atom_move": 2.0,
            "max_scaled_rmsd": 100.0,
            "max_scaled_fullspace_residual": 100.0,
            "max_scaled_whitened_distance": 100.0,
            "pair_ratio_floor": 0.0,
            "normalised_chemistry_penalty_cap": 100.0,
        },
    }

    out = _select(
        raw_x=0.5,
        candidate_xs=[],
        safety=_safety(backtrack_to_safe_landing=False),
        scale_model=scale_model,
    )

    safety = out["landing_safety"]
    assert safety["accepted"] is False
    assert "ariadne_scaled_atom_move_threshold_exceeded" in safety["reasons"]
    assert safety["raw_final"]["metrics"]["max_per_atom_mobility_ratio"] == pytest.approx(5.0)


def test_lower_alpha_raw_final_can_only_fall_back_to_seed_explicitly():
    out = _select(
        raw_x=-1.0,
        candidate_xs=[],
        safety=_safety(
            allow_seed_fallback=True,
            backtrack_to_safe_landing=False,
        ),
    )

    safety = out["landing_safety"]
    assert safety["accepted"] is True
    assert safety["policy"] == "seed_fallback"
    assert safety["selected_origin"] == "seed_fallback"
    assert np.asarray(out["selected_atoms"].coordinates)[0, 0] == 0.0
    assert safety["raw_final"]["accepted"] is False
    assert (
        "ariadne_landing_acquisition_not_improved"
        in safety["raw_final"]["reasons"]
    )


def test_gradient_band_warm_start_origin_is_not_seed_fallback():
    out = _select(
        raw_x=0.0,
        candidate_xs=[],
        safety=_safety(),
        origins=["gradient_band_warm_start"],
        initial_x=0.5,
    )

    safety = out["landing_safety"]
    assert safety["accepted"] is True
    assert safety["selected_origin"] == "gradient_band_warm_start"
    assert safety["policy"] == "gradient_band_warm_start"
    assert "seed_fallback_disabled" not in safety["reasons"]


def test_safe_gradient_band_warm_start_skips_under_moved_probe():
    seed = _one_atom(0.0)
    acquisition = _FakeAcquisition(seed)

    positions, origin, records = _select_safe_gradient_band_warm_start(
        acquisition=acquisition,
        seed_atoms=seed,
        safety_config=_safety(),
        quality_gates=SimpleNamespace(
            ariadne_max_displacement_ang=None,
            ariadne_min_pair_distance_ang=None,
        ),
    )

    assert origin == "gradient_band_warm_start"
    assert np.asarray(positions)[0, 0] == 0.5
    assert records[0]["accepted"] is False
    assert "ariadne_landing_under_moved" in records[0]["reasons"]
    assert records[1]["selected_for_warm_start"] is True


def test_under_move_retry_reads_top_level_landing_candidates():
    payload = {
        "landing_safety": {"accepted": False, "reasons": ["no_safe_non_seed_landing"]},
        "landing_candidates": [{
            "accepted": False,
            "reasons": ["ariadne_landing_under_moved"],
        }],
    }

    assert _landing_needs_under_move_retry(
        payload,
        safety_config=_safety(),
        run_config=AriadneRunConfig(),
    ) is True


def test_safe_landing_rank_uses_full_total_without_hidden_reweighting():
    high_information_lower_total = {
        "alpha": 100.0,
        "informativeness_score": 100.0,
        "metrics": {"total_score": 5.0, "movement_band_score": 1.0},
    }
    lower_information_higher_total = {
        "alpha": 1.0,
        "informativeness_score": 1.0,
        "metrics": {"total_score": 6.0, "movement_band_score": 0.0},
    }

    assert _rank_landing_candidate(lower_information_higher_total) > (
        _rank_landing_candidate(high_information_lower_total)
    )


def _trust_acquisition(n_atoms, *, localised=False):
    atoms = Atoms(
        [Atom("H", float(index), 0.0, 0.0) for index in range(n_atoms)]
    )
    if localised:
        basis = np.zeros((3 * n_atoms, 1), dtype=float)
        basis[0, 0] = 1.0
    else:
        basis = np.eye(3 * n_atoms, dtype=float)
    dimension = basis.shape[1]
    subspace = LocalSubspace(
        seed_atoms=atoms,
        neighbours=[],
        covariance=np.eye(3 * n_atoms),
        basis=basis,
        eigenvalues=np.ones(dimension),
        mode_weights=np.ones(dimension) / float(dimension),
        mass_vector=np.ones(3 * n_atoms),
        active_covariance=np.eye(dimension),
        neighbour_weights=np.array([]),
    )
    return SimpleNamespace(seed_atoms=atoms, subspace=subspace)


def _trust_scale_model(n_atoms):
    return {
        "geometry_motion_scale": {"value_angstrom": 0.1},
        "per_atom_mobility_scales": {
            "values_angstrom": [0.1] * n_atoms,
        },
        "trust_radius_policy": {
            "enabled": True,
            "normalisation": "weighted_mobility_sqrt_effective_atoms",
            "aggressiveness_multiplier": 1.5,
            "max_to_initial_ratio": 4.0,
            "under_move_feedback_min_factor": 1.0,
            "under_move_feedback_max_factor": 2.0,
        },
    }


def test_trust_radius_preserves_per_effective_atom_motion_across_system_sizes():
    per_atom_trust = []
    for n_atoms in (3, 30):
        resolved, diagnostics = _size_normalised_trust_radius(
            _trust_acquisition(n_atoms),
            AriadneRunConfig(delta0=0.1, delta_max=0.4),
            _trust_scale_model(n_atoms),
        )
        per_atom_trust.append(
            resolved.delta0
            / np.sqrt(diagnostics["n_effective_movement_atoms"])
        )
        assert resolved.delta_max == pytest.approx(4.0 * resolved.delta0)

    assert per_atom_trust == pytest.approx([0.15, 0.15])


def test_trust_radius_uses_active_atoms_not_total_molecule_size():
    resolved, diagnostics = _size_normalised_trust_radius(
        _trust_acquisition(40, localised=True),
        AriadneRunConfig(delta0=0.1, delta_max=0.4),
        _trust_scale_model(40),
    )

    assert diagnostics["n_atoms"] == 40
    assert diagnostics["n_effective_movement_atoms"] == pytest.approx(1.0)
    assert resolved.delta0 == pytest.approx(0.15)


def test_under_move_trust_feedback_is_bounded_by_policy():
    factor, diagnostics = _under_move_trust_feedback(
        {
            "landing_candidates": [
                {
                    "reasons": ["ariadne_landing_under_moved"],
                    "metrics": {
                        "movement_rmsd_ang": 0.1,
                        "movement_band_peak_ang": 0.5,
                    },
                }
            ]
        },
        {"weighted_per_atom_mobility_angstrom": 0.1},
        _trust_scale_model(1),
    )

    assert factor == pytest.approx(2.0)
    assert diagnostics["factor_bounds"] == [1.0, 2.0]
