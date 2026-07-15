import pytest
import numpy as np
from types import SimpleNamespace

from ichor.core.adversarial.acquisition import ModeEvaluation, SeedLocalAdversarialAcquisition
from ichor.core.adversarial.config import (
    AcquisitionConfig,
    CalibratedEnergyConfig,
    FullspaceConfinementConfig,
    SpectralConfig,
    StencilConfig,
)
from ichor.core.adversarial.subspace import LocalSubspace, fullspace_residual_distance
from ichor.core.atoms import Atom, Atoms


def _mode(index, omega):
    return ModeEvaluation(
        index=index,
        force_std=1.0,
        curvature_mean=omega * omega,
        curvature_std=0.1,
        omega=omega,
        omega_std=0.5,
        cubic_mean=0.0,
        cubic_std=0.0,
        quartic_mean=0.0,
        quartic_std=0.0,
        anharmonicity=0.0,
        anharmonicity_std=0.0,
    )


def _stub_acq(config=None):
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    acq.config = config or AcquisitionConfig()
    acq.error_calibration_model = None
    acq.error_calibration_apply_strength = 0.0
    return acq


class _Posterior:
    def mean(self, atoms):
        return -1.0

    def variance(self, atoms):
        return 0.002


def _component_stub(mode, config=None):
    acq = _stub_acq(config)
    acq.reference_scales = {
        "energy": 0.0019333022952096493,
        "force": 0.06403051147076265,
        "omega": 0.5080184931098937,
        "anh": 0.08533774843328841,
        "anh_std": 3.925176277165896,
        "spectral": 0.5080184931098937,
    }
    acq.posterior = _Posterior()
    acq.subspace = SimpleNamespace(mode_weights=np.array([1.0]))
    acq.seed_atoms = Atoms([Atom("O", 0.0, 0.0, 0.0)])
    acq.barrier_state = None
    acq._mode_metrics = lambda atoms, mean_energy=None: (mode,)
    acq._fullspace_confinement_metrics = lambda atoms: {
        "residual_penalty": 0.0,
        "rmsd_penalty": 0.0,
        "fallback_reasons": [],
        "residual_distance": 0.0,
        "aligned_rmsd_ang": 0.0,
    }
    return acq


def test_spectral_inverse_frequency_weights_favour_soft_modes():
    acq = _stub_acq(
        AcquisitionConfig(
            spectral=SpectralConfig(
                enabled=True,
                mode="blend",
                mode_weighting="inverse_frequency",
                omega_floor=1.0e-6,
            )
        )
    )
    weights = acq._spectral_mode_weights((_mode(0, 0.1), _mode(1, 10.0)))
    assert weights[0] > weights[1]
    assert sum(weights) == pytest.approx(1.0)


def test_banded_energy_utility_rewards_middle_uncertainty():
    acq = _stub_acq(
        AcquisitionConfig(
            calibrated_energy=CalibratedEnergyConfig(
                utility="banded",
                band_low_ha_per_sqrt_atom=0.1,
                band_high_ha_per_sqrt_atom=1.0,
                low_softness_ha_per_sqrt_atom=0.05,
                high_softness_ha_per_sqrt_atom=0.05,
            )
        )
    )
    low, _, _ = acq._energy_utility(0.01, 0.1)
    middle, _, _ = acq._energy_utility(0.5, 0.1)
    high, _, _ = acq._energy_utility(5.0, 0.1)
    assert middle > low
    assert middle > high


def test_fullspace_residual_distance_zero_for_active_displacement(monkeypatch):
    seed = Atoms([Atom("C", 0.0, 0.0, 0.0)])
    subspace = LocalSubspace(
        seed_atoms=seed,
        neighbours=[],
        covariance=np.eye(3),
        basis=np.array([[1.0], [0.0], [0.0]]),
        eigenvalues=np.array([1.0]),
        mode_weights=np.array([1.0]),
        mass_vector=np.ones(3),
        active_covariance=np.eye(1),
        neighbour_weights=np.ones(0),
    )
    from ichor.core.adversarial import subspace as subspace_mod

    monkeypatch.setattr(
        subspace_mod,
        "aligned_mass_weighted_displacement",
        lambda reference, mobile: np.array([2.0, 0.0, 0.0]),
    )
    assert fullspace_residual_distance(subspace, seed) == pytest.approx(0.0)

    monkeypatch.setattr(
        subspace_mod,
        "aligned_mass_weighted_displacement",
        lambda reference, mobile: np.array([0.0, 2.0, 0.0]),
    )
    assert fullspace_residual_distance(subspace, seed) > 0.0


def test_negative_curvature_is_not_rewarded_by_default():
    acq = _stub_acq()
    positive = acq._curvature_floor(4.0)
    negative = acq._curvature_floor(-4.0)
    assert negative < positive
    mode = ModeEvaluation(0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert acq._negative_curvature_penalty((mode,), (1.0,)) == 0.0


def test_negative_curvature_contributes_no_frequency_or_anharmonic_reward(
    monkeypatch,
):
    monkeypatch.setattr(
        "ichor.core.adversarial.acquisition.whitened_distance_squared",
        lambda *args, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        "ichor.core.adversarial.acquisition.chemistry_barrier_value",
        lambda *args, **kwargs: 0.0,
    )
    unstable = ModeEvaluation(
        index=0,
        force_std=0.0,
        curvature_mean=-4.0,
        curvature_std=100.0,
        omega=2.0,
        omega_std=100.0,
        cubic_mean=10.0,
        cubic_std=10.0,
        quartic_mean=10.0,
        quartic_std=10.0,
        anharmonicity=100.0,
        anharmonicity_std=100.0,
    )

    breakdown = _component_stub(unstable).components(
        Atoms([Atom("O", 0.0, 0.0, 0.0)]),
        include_movement=False,
    )

    assert breakdown.legacy_frequency_risk == 0.0
    assert breakdown.spectral_frequency_risk == 0.0
    assert breakdown.anharmonic_risk == 0.0


def test_negative_curvature_penalise_mode_is_bounded():
    acq = _stub_acq(
        AcquisitionConfig(
            stencils=StencilConfig(
                negative_curvature_policy="penalise",
                lambda_negative_curvature=2.0,
            )
        )
    )
    penalty = acq._negative_curvature_penalty(
        (ModeEvaluation(0, 0.0, -100.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        (1.0,),
    )
    assert 0.0 < penalty <= 1.0


def test_weak_mode_reliability_thresholds_are_smooth():
    acq = _stub_acq()
    acq.reference_scales = {"omega": 0.5080184931098937}

    low, high = acq._weak_mode_thresholds()

    assert low == pytest.approx(0.025400924655494686)
    assert high == pytest.approx(0.07620277396648405)
    assert acq._weak_mode_reliability(0.010463511968800517)[0] == 0.0
    assert acq._weak_mode_reliability(0.10)[0] == 1.0
    mid = acq._weak_mode_reliability((low + high) / 2.0)[0]
    assert 0.0 < mid < 1.0


def test_weak_mode_gating_blocks_singular_anharmonic_reward(monkeypatch):
    monkeypatch.setattr(
        "ichor.core.adversarial.acquisition.whitened_distance_squared",
        lambda *args, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        "ichor.core.adversarial.acquisition.chemistry_barrier_value",
        lambda *args, **kwargs: 0.0,
    )
    pathological = ModeEvaluation(
        index=0,
        force_std=0.09086389753555923,
        curvature_mean=0.010463511968800517 ** 2,
        curvature_std=1.0,
        omega=0.010463511968800517,
        omega_std=25.66494931803662,
        cubic_mean=0.0,
        cubic_std=0.0,
        quartic_mean=0.0,
        quartic_std=0.0,
        anharmonicity=246.10215196571232,
        anharmonicity_std=15105.939507675506,
    )
    atoms = Atoms([Atom("O", 0.0, 0.0, 0.0)])
    gated = _component_stub(pathological).components(atoms, include_movement=False)
    ungated_cfg = AcquisitionConfig(
        stencils=StencilConfig(weak_mode_gating_enabled=False)
    )
    ungated = _component_stub(pathological, ungated_cfg).components(
        atoms,
        include_movement=False,
    )

    assert gated.mode_evaluations[0].weak_mode_reliability == 0.0
    assert gated.anharmonic_risk < 1.0
    assert gated.weak_mode_penalty_score > 0.0
    assert ungated.anharmonic_risk > 20.0
    assert gated.total < ungated.total


def test_finite_frequency_anharmonic_signal_survives_gating(monkeypatch):
    monkeypatch.setattr(
        "ichor.core.adversarial.acquisition.whitened_distance_squared",
        lambda *args, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        "ichor.core.adversarial.acquisition.chemistry_barrier_value",
        lambda *args, **kwargs: 0.0,
    )
    mode = ModeEvaluation(
        index=0,
        force_std=0.1,
        curvature_mean=0.2 ** 2,
        curvature_std=0.2,
        omega=0.2,
        omega_std=0.5,
        cubic_mean=0.0,
        cubic_std=0.0,
        quartic_mean=0.0,
        quartic_std=0.0,
        anharmonicity=0.5,
        anharmonicity_std=10.0,
    )

    breakdown = _component_stub(mode).components(
        Atoms([Atom("O", 0.0, 0.0, 0.0)]),
        include_movement=False,
    )

    assert breakdown.mode_evaluations[0].weak_mode_reliability == 1.0
    assert breakdown.anharmonic_risk > 0.0
    assert breakdown.weak_mode_penalty_score == 0.0


def test_fullspace_confinement_failure_returns_unsafe_penalty(monkeypatch):
    seed = Atoms([Atom("C", 0.0, 0.0, 0.0)])
    acq = _stub_acq(
        AcquisitionConfig(
            fullspace_confinement=FullspaceConfinementConfig(
                failure_penalty=123.0,
            )
        )
    )
    acq.seed_atoms = seed
    acq.subspace = object()
    monkeypatch.setattr(
        "ichor.core.adversarial.acquisition.fullspace_residual_distance",
        lambda subspace, atoms: (_ for _ in ()).throw(RuntimeError("bad geometry")),
    )

    metrics = acq._fullspace_confinement_metrics(seed)

    assert metrics["residual_penalty"] == 123.0
    assert metrics["rmsd_penalty"] == 123.0
    assert metrics["fallback_reasons"]
