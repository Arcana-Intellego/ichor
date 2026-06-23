import numpy as np

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
