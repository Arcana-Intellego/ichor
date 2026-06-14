import pytest
import numpy as np

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
                band_low_ha=0.1,
                band_high_ha=1.0,
                low_softness_ha=0.05,
                high_softness_ha=0.05,
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
    assert negative == pytest.approx(positive)
    mode = ModeEvaluation(0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert acq._negative_curvature_penalty((mode,), (1.0,)) == 0.0


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
