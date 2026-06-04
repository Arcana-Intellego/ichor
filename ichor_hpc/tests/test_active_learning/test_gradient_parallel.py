"""The parallel acquisition-gradient driver must be numerically identical to
the serial loop, on every backend. Process uses real fork on Linux and falls
back to serial here on Windows -- either way it must match."""
from types import SimpleNamespace

import numpy as np

from ichor.core.atoms import Atom, Atoms
from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
from ichor.hpc.active_learning.acquisition.parallel_gradient import (
    compute_cartesian_gradient,
)


def _stub_acq():
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    acq.config = SimpleNamespace(
        gradient=SimpleNamespace(
            cartesian_step=1.0e-3,
            cartesian_step_floor=0.0,
            ghost_mass_threshold=0.0,
        )
    )
    # smooth deterministic surrogate so the FD is well-defined and reproducible.
    acq.value = lambda atoms: float(
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
