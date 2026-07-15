"""Tests for task-owned active-subspace gradient workers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
from ichor.core.atoms import Atom, Atoms
from ichor.hpc.active_learning.acquisition.ase_calculator import (
    AdversarialASECalculator,
)
from ichor.hpc.active_learning.acquisition.parallel_gradient import (
    ActiveGradientWorkerPool,
    _THREAD_ENV_NAMES,
    _capped_worker_thread_env,
    compute_active_gradient,
    inside_gradient_worker,
    last_gradient_parallel_diagnostics,
    resolve_workers,
)


def _stub_acq():
    acquisition = object.__new__(SeedLocalAdversarialAcquisition)
    acquisition.config = SimpleNamespace(
        gradient=SimpleNamespace(active_step=1.0e-3, regularization=0.0)
    )
    acquisition.mode_directions = [
        np.eye(9, dtype=float)[0],
        np.eye(9, dtype=float)[3],
    ]
    acquisition.value = lambda atoms, objective="full": float(
        np.sum(np.asarray(atoms.coordinates, dtype=float) ** 2)
    )
    return acquisition


def _water():
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])


def test_active_process_gradient_matches_serial():
    acquisition = _stub_acq()
    atoms = _water()
    serial = acquisition._active_finite_difference_gradient(atoms)
    parallel = compute_active_gradient(
        acquisition,
        atoms,
        workers=4,
        backend="process",
    )
    np.testing.assert_allclose(parallel, serial, atol=1.0e-12)


@pytest.mark.parametrize("backend", ["thread", "cartesian", "unknown"])
def test_removed_gradient_backends_are_rejected(backend):
    with pytest.raises(ValueError, match="serial.*process"):
        compute_active_gradient(_stub_acq(), _water(), backend=backend)


def test_removed_objective_argument_is_rejected():
    with pytest.raises(TypeError, match="objective"):
        compute_active_gradient(
            _stub_acq(),
            _water(),
            backend="serial",
            objective="cheap_driver",
        )


def test_gradient_worker_count_respects_slurm_and_task_count(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert resolve_workers(8, n_tasks=6) == 2
    assert resolve_workers(8, n_tasks=1) == 1
    monkeypatch.setenv("ICHOR_GRADIENT_WORKERS", "5")
    assert resolve_workers(None, n_tasks=6) == 2


def test_disable_gradient_mp_forces_active_serial(monkeypatch):
    monkeypatch.setenv("ICHOR_DISABLE_GRADIENT_MP", "1")
    compute_active_gradient(
        _stub_acq(), _water(), workers=4, backend="process"
    )
    diagnostics = last_gradient_parallel_diagnostics()
    assert diagnostics["gradient_backend"] == "serial"
    assert diagnostics["parallel_fallback_reason"] == "disabled_by_environment"


def test_worker_thread_env_cap_overrides_and_restores_parent(monkeypatch):
    for name in _THREAD_ENV_NAMES:
        monkeypatch.setenv(name, "8")
    with _capped_worker_thread_env():
        for name in _THREAD_ENV_NAMES:
            assert os.environ[name] == "1"
    for name in _THREAD_ENV_NAMES:
        assert os.environ[name] == "8"


def test_nested_gradient_worker_guard_forces_active_serial(monkeypatch):
    monkeypatch.setenv("ICHOR_GRADIENT_WORKER", "1")
    assert inside_gradient_worker() is True
    compute_active_gradient(
        _stub_acq(), _water(), workers=4, backend="process"
    )
    diagnostics = last_gradient_parallel_diagnostics()
    assert diagnostics["gradient_backend"] == "serial"
    assert diagnostics["parallel_fallback_reason"] == "inside_gradient_worker"


def test_calculator_reuses_one_task_owned_process_pool(monkeypatch):
    calls = {"created": 0, "gradient": 0, "closed": 0}

    class FakePool:
        def __init__(self, acquisition):
            calls["created"] += 1

        def gradient(self, atoms):
            calls["gradient"] += 1
            return np.ones_like(np.asarray(atoms.coordinates, dtype=float))

        def close(self):
            calls["closed"] += 1

    monkeypatch.setattr(
        "ichor.hpc.active_learning.acquisition.parallel_gradient."
        "ActiveGradientWorkerPool",
        FakePool,
    )
    calculator = AdversarialASECalculator(
        _stub_acq(),
        project_rigid=False,
        gradient_backend="process",
    )
    calculator._hartree = lambda: 1.0
    atoms = _water()
    calculator.get_forces(atoms)
    moved = Atoms([
        Atom(atom.type, *(np.asarray(atom.coordinates) + [0.01, 0.0, 0.0]))
        for atom in atoms
    ])
    calculator.get_forces(moved)
    calculator.close()

    assert calls == {"created": 1, "gradient": 2, "closed": 1}
    assert calculator.gradient_diagnostics()["gradient_objective"] == "full"


def test_calculator_rejects_thread_backend():
    with pytest.raises(ValueError, match="serial.*process"):
        AdversarialASECalculator(_stub_acq(), gradient_backend="thread")


def test_failed_persistent_pool_falls_back_to_serial_for_future_calls():
    class BrokenPool:
        def map(self, *args, **kwargs):
            raise RuntimeError("worker failed")

        def shutdown(self, **kwargs):
            return None

    acquisition = _stub_acq()
    pool = ActiveGradientWorkerPool.__new__(ActiveGradientWorkerPool)
    pool._acquisition = acquisition
    pool._chunksize = 1
    pool._pool = BrokenPool()
    pool._closed = False
    pool._owns_worker_acquisition = False
    pool._fallback_reason = None
    pool._workers_requested = 2
    pool._workers_used = 2

    first = pool.gradient(_water())
    second = pool.gradient(_water())

    expected = acquisition._active_finite_difference_gradient(_water())
    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected)
    assert pool.diagnostics["parallel_fallback_reason"].startswith(
        "persistent_process_exception"
    )
