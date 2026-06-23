from types import SimpleNamespace

import numpy as np

import ichor.core.adversarial.acquisition as acquisition_mod
from ichor.core.adversarial.acquisition import (
    AcquisitionBreakdown,
    SeedLocalAdversarialAcquisition,
)
from ichor.core.adversarial.config import AcquisitionConfig
from ichor.core.adversarial.config import DriverConfig
from ichor.core.atoms import Atom, Atoms


def _water_like() -> Atoms:
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])


def _stub_acq():
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    acq.config = AcquisitionConfig()

    def value(atoms, *, objective="full"):
        scale = 3.0 if objective == "cheap_driver" else 1.0
        coords = np.asarray(atoms.coordinates, dtype=float).reshape(-1)
        return 0.5 * scale * float(np.dot(coords, coords))

    acq.value = value  # type: ignore[assignment]
    acq.mode_directions = [np.eye(9, dtype=float)[0], np.eye(9, dtype=float)[3]]
    return acq


def test_cartesian_gradient_respects_objective_keyword():
    acq = _stub_acq()
    atoms = _water_like()

    full = acq.gradient(atoms, mode="cartesian_fd", objective="full")
    driver = acq.gradient(atoms, mode="cartesian_fd", objective="cheap_driver")

    np.testing.assert_allclose(driver, 3.0 * full, atol=1.0e-6)


def test_active_gradient_respects_objective_keyword():
    acq = _stub_acq()
    atoms = _water_like()

    full = acq.gradient(atoms, mode="active_fd", objective="full")
    driver = acq.gradient(atoms, mode="active_fd", objective="cheap_driver")

    np.testing.assert_allclose(driver, 3.0 * full, atol=1.0e-6)


def test_components_dispatches_to_driver_objective():
    acq = _stub_acq()
    atoms = _water_like()
    sentinel = AcquisitionBreakdown(
        total=12.0,
        informativeness_score=13.0,
        risk_penalty_score=1.0,
        energy_risk=2.0,
        force_risk=0.0,
        frequency_risk=0.0,
        anharmonic_risk=0.0,
        distance_penalty=0.0,
        chemistry_penalty=0.0,
        mean_energy=0.0,
        energy_variance=0.0,
    )
    acq._driver_components = lambda atoms, include_movement=True: sentinel

    assert acq.components(atoms, objective="cheap_driver") is sentinel


def test_cheap_driver_can_dispatch_to_hybrid_geometry_gradient():
    acq = _stub_acq()
    acq.config = AcquisitionConfig(
        driver=DriverConfig(gradient_backend="hybrid_geometry")
    )
    atoms = _water_like()
    expected = np.arange(9, dtype=float).reshape(3, 3)
    calls = []

    def hybrid(atoms_arg, *, mode):
        calls.append(mode)
        return expected

    acq._hybrid_driver_geometry_gradient = hybrid  # type: ignore[assignment]

    out = acq.gradient(atoms, mode="active_fd", objective="cheap_driver")

    np.testing.assert_allclose(out, expected)
    assert calls == ["active_fd"]


def test_hybrid_gradient_validation_falls_back_to_fd():
    acq = _stub_acq()
    acq.config = AcquisitionConfig(
        driver=DriverConfig(
            gradient_backend="hybrid_geometry",
            finite_difference_energy=False,
            analytic_movement=True,
            analytic_whitened_distance=False,
            analytic_pair_barriers=False,
            analytic_fullspace_rmsd=False,
            analytic_validation=True,
            analytic_validation_tol_cosine=0.5,
        )
    )
    atoms = _water_like()
    fd = np.ones((3, 3), dtype=float)
    acq._movement_utility_gradient = lambda atoms_arg: -fd  # type: ignore[assignment]
    acq._cartesian_finite_difference_gradient = (  # type: ignore[assignment]
        lambda atoms_arg, objective="full": fd
    )

    out = acq.gradient(atoms, mode="cartesian_fd", objective="cheap_driver")

    np.testing.assert_allclose(out, fd)
    diag = acq._last_driver_gradient_diagnostics
    assert diag["driver_gradient_fallback"] is True
    assert diag["driver_gradient_fallback_reason"] == "analytic_validation_cosine_below_tolerance"


def test_whitened_distance_gradient_uses_value_regularisation_scale():
    acq = _stub_acq()
    acq.config = AcquisitionConfig()
    acq.config = AcquisitionConfig()
    acq.subspace = type("Subspace", (), {})()
    acq.subspace.dimension = 3
    acq.subspace.basis = np.eye(3, dtype=float)
    acq.subspace.active_covariance = np.diag([100.0, 2.0, 1.0])
    disp = np.array([1.0, 2.0, 3.0], dtype=float)
    masses = np.ones(1, dtype=float)
    acq._mass_weighted_displacement_and_masses = (  # type: ignore[assignment]
        lambda atoms_arg: (disp, masses)
    )
    acq._mass_weighted_gradient_to_cartesian = (  # type: ignore[assignment]
        lambda grad_mw, masses_arg: np.asarray(grad_mw, dtype=float).reshape(1, 3)
    )

    out = acq._whitened_distance_gradient(_water_like())

    reg = acq.config.subspace.covariance_regularization * 100.0
    expected = 2.0 * np.linalg.inv(
        acq.subspace.active_covariance + reg * np.eye(3)
    ) @ disp
    np.testing.assert_allclose(out.reshape(-1), expected)


def _minimal_real_component_acq(driver_backend="fd"):
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    acq.config = AcquisitionConfig(
        driver=DriverConfig(gradient_backend=driver_backend),
    )
    acq.posterior = SimpleNamespace(
        mean=lambda atoms: 0.0,
        variance=lambda atoms: 1.0,
    )
    acq.error_calibration_model = None
    acq.error_calibration_apply_strength = 0.0
    acq.reference_scales = {
        "energy": 1.0,
        "force": 1.0,
        "omega": 1.0,
        "anh": 1.0,
        "anh_std": 1.0,
        "spectral": 1.0,
    }
    acq.subspace = SimpleNamespace(dimension=1)
    acq.barrier_state = object()
    acq._fullspace_confinement_metrics = lambda atoms: {
        "residual_distance": 0.0,
        "residual_penalty": 0.0,
        "aligned_rmsd_ang": 0.0,
        "rmsd_penalty": 0.0,
        "fallback_reasons": [],
    }
    acq.movement_metrics = lambda atoms: {}
    acq._mode_metrics = lambda atoms, mean_energy=None: ()
    acq._effective_mode_weights = lambda mode_evals: ()
    acq._spectral_mode_weights = lambda mode_evals: ()
    acq._negative_curvature_penalty = lambda mode_evals, weights: 0.0
    return acq


def test_driver_components_keep_angle_barrier_for_fd_backend(monkeypatch):
    calls = []

    def fake_barrier(*args, include_angles=True, **kwargs):
        calls.append(include_angles)
        return 2.0 if include_angles else 1.0

    monkeypatch.setattr(acquisition_mod, "chemistry_barrier_value", fake_barrier)
    monkeypatch.setattr(acquisition_mod, "whitened_distance_squared", lambda *args: 0.0)
    acq = _minimal_real_component_acq(driver_backend="fd")

    out = acq._driver_components(_water_like(), include_movement=False)

    assert calls == [True]
    assert out.chemistry_penalty == 2.0
    assert "cheap_driver_hybrid_omits_angle_barrier" not in out.fallback_reasons


def test_driver_components_omit_angle_barrier_for_hybrid_backend(monkeypatch):
    calls = []

    def fake_barrier(*args, include_angles=True, **kwargs):
        calls.append(include_angles)
        return 2.0 if include_angles else 1.0

    monkeypatch.setattr(acquisition_mod, "chemistry_barrier_value", fake_barrier)
    monkeypatch.setattr(acquisition_mod, "whitened_distance_squared", lambda *args: 0.0)
    acq = _minimal_real_component_acq(driver_backend="hybrid_geometry")

    out = acq._driver_components(_water_like(), include_movement=False)

    assert calls == [False]
    assert out.chemistry_penalty == 1.0
    assert "cheap_driver_hybrid_omits_angle_barrier" in out.fallback_reasons


def test_full_components_always_keep_angle_barrier(monkeypatch):
    calls = []

    def fake_barrier(*args, include_angles=True, **kwargs):
        calls.append(include_angles)
        return 2.0 if include_angles else 1.0

    monkeypatch.setattr(acquisition_mod, "chemistry_barrier_value", fake_barrier)
    monkeypatch.setattr(acquisition_mod, "whitened_distance_squared", lambda *args: 0.0)
    acq = _minimal_real_component_acq(driver_backend="hybrid_geometry")

    out = acq.components(_water_like(), include_movement=False, objective="full")

    assert calls == [True]
    assert out.chemistry_penalty == 2.0
