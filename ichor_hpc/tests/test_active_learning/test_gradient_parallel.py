"""The parallel acquisition-gradient driver must be numerically identical to
the serial loop, on every backend. Process uses real fork on Linux and falls
back to serial here on Windows -- either way it must match."""
from types import SimpleNamespace

import numpy as np

from ichor.core.atoms import Atom, Atoms
from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
from ichor.hpc.active_learning.acquisition.parallel_gradient import (
    compute_active_gradient,
    compute_cartesian_gradient,
    inside_gradient_worker,
    last_gradient_parallel_diagnostics,
    resolve_workers,
)
from ichor.hpc.active_learning.acquisition.ase_calculator import (
    AdversarialASECalculator,
)


def _stub_acq():
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    acq.config = SimpleNamespace(
        gradient=SimpleNamespace(
            mode="cartesian_fd",
            cartesian_step=1.0e-3,
            active_step=1.0e-3,
            regularization=0.0,
            cartesian_step_floor=0.0,
            ghost_mass_threshold=0.0,
        )
    )
    # smooth deterministic surrogate so the FD is well-defined and reproducible.
    acq.value = lambda atoms, objective="full": float(
        np.sum(np.asarray(atoms.coordinates, dtype=float) ** 2)
    )
    return acq


def _water():
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])


def test_parallel_gradient_matches_serial():
    acq = _stub_acq()
    atoms = _water()
    serial = acq._cartesian_finite_difference_gradient(atoms)
    for backend in ("serial", "thread", "process"):
        g = compute_cartesian_gradient(acq, atoms, workers=4, backend=backend)
        np.testing.assert_allclose(g, serial, atol=1.0e-12, err_msg=backend)


def test_parallel_gradient_recovers_known_surrogate():
    # value = sum(x^2) -> d/dx_i = 2 x_i; central differences recover that.
    acq = _stub_acq()
    atoms = _water()
    g = compute_cartesian_gradient(acq, atoms, workers=2, backend="thread")
    expected = 2.0 * np.asarray(atoms.coordinates, dtype=float)
    np.testing.assert_allclose(g, expected, atol=1.0e-6)


def test_active_parallel_gradient_matches_serial():
    acq = _stub_acq()
    atoms = _water()
    acq.mode_directions = [
        np.eye(9, dtype=float)[0],
        np.eye(9, dtype=float)[3],
    ]
    serial = acq._active_finite_difference_gradient(atoms)

    for backend in ("serial", "thread", "process"):
        g = compute_active_gradient(acq, atoms, workers=4, backend=backend)
        np.testing.assert_allclose(g, serial, atol=1.0e-12, err_msg=backend)


def test_active_thread_backend_falls_back_to_serial():
    acq = _stub_acq()
    atoms = _water()
    acq.mode_directions = [np.eye(9, dtype=float)[0], np.eye(9, dtype=float)[1]]

    compute_active_gradient(acq, atoms, workers=4, backend="thread")
    diag = last_gradient_parallel_diagnostics()

    assert diag["gradient_backend"] == "serial"
    assert diag["parallel_fallback_reason"] == "thread_backend_disabled_for_active_fd"


def test_gradient_worker_count_respects_slurm_and_task_count(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert resolve_workers(8, n_tasks=6) == 2
    assert resolve_workers(8, n_tasks=1) == 1
    monkeypatch.setenv("ICHOR_GRADIENT_WORKERS", "5")
    assert resolve_workers(None, n_tasks=6) == 2


def test_disable_gradient_mp_forces_active_serial(monkeypatch):
    acq = _stub_acq()
    atoms = _water()
    acq.mode_directions = [np.eye(9, dtype=float)[0], np.eye(9, dtype=float)[1]]

    monkeypatch.setenv("ICHOR_DISABLE_GRADIENT_MP", "1")
    compute_active_gradient(acq, atoms, workers=4, backend="process")
    diag = last_gradient_parallel_diagnostics()

    assert diag["gradient_backend"] == "serial"
    assert diag["parallel_fallback_reason"] == "disabled_by_environment"


def test_nested_gradient_worker_guard_forces_active_serial(monkeypatch):
    acq = _stub_acq()
    atoms = _water()
    acq.mode_directions = [np.eye(9, dtype=float)[0], np.eye(9, dtype=float)[1]]

    monkeypatch.setenv("ICHOR_GRADIENT_WORKER", "1")
    assert inside_gradient_worker() is True
    compute_active_gradient(acq, atoms, workers=4, backend="process")
    diag = last_gradient_parallel_diagnostics()

    assert diag["gradient_backend"] == "serial"
    assert diag["parallel_fallback_reason"] == "inside_gradient_worker"


def test_calculator_dispatches_active_fd_to_parallel_backend(monkeypatch):
    atoms = _water()
    acq = _stub_acq()
    acq.config.gradient.mode = "active_fd"
    calls = []

    def fake_active_gradient(
        acquisition,
        incoming_atoms,
        *,
        backend,
        workers=None,
        objective="full",
    ):
        calls.append((acquisition, incoming_atoms, backend, workers, objective))
        return np.ones_like(np.asarray(incoming_atoms.coordinates, dtype=float))

    monkeypatch.setattr(
        "ichor.hpc.active_learning.acquisition.parallel_gradient.compute_active_gradient",
        fake_active_gradient,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.acquisition.parallel_gradient.last_gradient_parallel_diagnostics",
        lambda: {
            "gradient_backend": "process",
            "workers_requested": None,
            "workers_used": 2,
            "parallel_fallback_reason": None,
        },
    )

    calc = AdversarialASECalculator(
        acq,
        project_rigid=False,
        gradient_backend="process",
    )
    calc._hartree = lambda: 1.0
    forces = calc.get_forces(atoms)

    assert len(calls) == 1
    assert calls[0][2] == "process"
    assert calls[0][4] == "full"
    np.testing.assert_allclose(forces, np.ones_like(forces))
    diag = calc.gradient_diagnostics()
    assert diag["gradient_backend"] == "process"
    assert diag["workers_used"] == 2


def test_calculator_passes_driver_objective_to_active_backend(monkeypatch):
    atoms = _water()
    acq = _stub_acq()
    acq.config.gradient.mode = "active_fd"
    calls = []

    def fake_active_gradient(
        acquisition,
        incoming_atoms,
        *,
        backend,
        workers=None,
        objective="full",
    ):
        calls.append(objective)
        return np.ones_like(np.asarray(incoming_atoms.coordinates, dtype=float))

    monkeypatch.setattr(
        "ichor.hpc.active_learning.acquisition.parallel_gradient.compute_active_gradient",
        fake_active_gradient,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.acquisition.parallel_gradient.last_gradient_parallel_diagnostics",
        lambda: {"gradient_backend": "process", "workers_used": 2},
    )

    calc = AdversarialASECalculator(
        acq,
        project_rigid=False,
        gradient_backend="process",
        objective="cheap_driver",
    )
    calc._hartree = lambda: 1.0
    calc.get_forces(atoms)

    assert calls == ["cheap_driver"]
    assert calc.gradient_diagnostics()["gradient_objective"] == "cheap_driver"
