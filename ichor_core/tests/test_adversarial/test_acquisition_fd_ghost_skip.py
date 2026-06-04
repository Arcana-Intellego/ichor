"""Tests for cartesian FD floor and ghost-atom skip.

It exercises 'SeedLocalAdversarialAcquisition._cartesian_finite_difference_gradient'
via a constructed-by-__new__ instance (no GP setup) plus a stub 'value'
function. This isolates the FD bookkeeping logic from any posterior machinery.
"""
import numpy as np
import pytest

from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
from ichor.core.adversarial.config import AcquisitionConfig, GradientConfig
from ichor.core.adversarial.geometry import coordinates_to_atoms
from ichor.core.atoms import Atom, Atoms


def _water() -> Atoms:
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])


def _make_stub_acquisition(config: AcquisitionConfig, value_func):
    """Build a SeedLocalAdversarialAcquisition by-passing __init__ so we can
    drive _cartesian_finite_difference_gradient without GP setup."""
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    acq.config = config
    #bind methods as plain attributes (simplest possible stubbing surface)
    acq.value = value_func
    #_atoms_from_flat is the existing helper - reuse its semantics.
    def _atoms_from_flat(flat, template):
        return coordinates_to_atoms(template, np.asarray(flat, dtype=float).reshape((-1, 3)))
    acq._atoms_from_flat = _atoms_from_flat
    return acq


def test_default_config_perturbs_every_dof():
    """Every Cartesian DOF receives a nonzero gradient
    contribution from a value function that depends on all coordinates."""
    atoms = _water()
    #Use a value function that depends nontrivially on every coordinate, so
    #the central FD gradient is nonzero for every DOF.
    def value(a):
        c = np.asarray(a.coordinates).reshape(-1)
        return float(np.sum(c * (np.arange(c.size, dtype=float) + 1.0)))

    cfg = AcquisitionConfig(gradient=GradientConfig(cartesian_step=1.0e-3))
    acq = _make_stub_acquisition(cfg, value)
    grad = acq._cartesian_finite_difference_gradient(atoms)

    assert grad.shape == (3, 3)
    assert np.all(np.abs(grad) > 0.0)
    expected = np.arange(9, dtype=float) + 1.0
    np.testing.assert_allclose(grad.reshape(-1), expected, atol=1.0e-8)


def test_ghost_threshold_zeros_low_mass_atoms():
    """When ghost_mass_threshold is set above H mass but below O mass, the
    gradient rows for the two hydrogens must be exactly zero, while the
    oxygen row remains the linear FD of the value function."""
    atoms = _water()
    #Same coord-dependent value function as above.
    coeffs = np.arange(9, dtype=float) + 1.0
    def value(a):
        c = np.asarray(a.coordinates).reshape(-1)
        return float(np.sum(c * coeffs))

    cfg = AcquisitionConfig(gradient=GradientConfig(
        cartesian_step=1.0e-3,
        ghost_mass_threshold=2.0, # masks both H atoms (mass ~1.008)
    ))
    acq = _make_stub_acquisition(cfg, value)
    grad = acq._cartesian_finite_difference_gradient(atoms)

    #Oxygen (index 0) is heavier than the threshold: gradient = its coeffs.
    np.testing.assert_allclose(grad[0], coeffs[0:3], atol=1.0e-8)
    #Both hydrogens (indices 1, 2) are below the threshold: zero rows.
    np.testing.assert_array_equal(grad[1], np.zeros(3))
    np.testing.assert_array_equal(grad[2], np.zeros(3))


def test_step_floor_applies_when_step_below_floor():
    """When cartesian_step_floor exceeds cartesian_step, the effective step
    is the floor. We verify this by observing the (nominal) step magnitude
    via a value function that records the perturbation magnitude."""
    atoms = _water()
    recorded_perturbations = []
    base_coords = np.asarray(atoms.coordinates).reshape(-1).copy()

    def value(a):
        c = np.asarray(a.coordinates).reshape(-1)
        #The maximum per-DOF |delta from base| is the FD step that was used.
        recorded_perturbations.append(float(np.max(np.abs(c - base_coords))))
        return float(np.sum(c))

    cfg = AcquisitionConfig(gradient=GradientConfig(
        cartesian_step=1.0e-8,            #tiny causes catastrophic cancellation
        cartesian_step_floor=1.0e-3,      #opt-in floor
    ))
    acq = _make_stub_acquisition(cfg, value)
    acq._cartesian_finite_difference_gradient(atoms)

    #Every perturbation magnitude observed by the value function must be
    #at the floor (1e-3), not at the tiny cartesian_step (1e-8).
    assert len(recorded_perturbations) > 0
    assert all(abs(p - 1.0e-3) < 1.0e-12 for p in recorded_perturbations)


def test_step_floor_no_op_when_floor_zero():
    """Default floor of 0.0 leaves cartesian_step untouched (no behaviour change)."""
    atoms = _water()
    base_coords = np.asarray(atoms.coordinates).reshape(-1).copy()
    observed_steps = []

    def value(a):
        c = np.asarray(a.coordinates).reshape(-1)
        observed_steps.append(float(np.max(np.abs(c - base_coords))))
        return float(np.sum(c))

    cfg = AcquisitionConfig(gradient=GradientConfig(
        cartesian_step=1.0e-4,
        cartesian_step_floor=0.0,         # default
    ))
    acq = _make_stub_acquisition(cfg, value)
    acq._cartesian_finite_difference_gradient(atoms)

    assert all(abs(p - 1.0e-4) < 1.0e-15 for p in observed_steps)
