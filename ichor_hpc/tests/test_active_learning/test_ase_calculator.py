"""Tests for ichor.hpc.active_learning.acquisition.ase_calculator.

The calculator wraps a SeedLocalAdversarialAcquisition. To keep the tests
fast we bypass the GP setup using `__new__` + stubbed value/gradient closures,
exactly as the M2 FD tests do.
"""
import math
from collections import Counter

import numpy as np
import pytest

from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
from ichor.core.atoms import Atom, Atoms

from ichor.hpc.active_learning.acquisition.ase_calculator import (
    AdversarialASECalculator,
    ase_atoms_to_ichor_atoms,
)
from ichor.hpc.active_learning.acquisition.rigid_projection import (
    project_out_rigid,
)


HARTREE_EV = None  # populated lazily in fixtures to avoid forcing ase at import


def _hartree_ev() -> float:
    from ase.units import Hartree
    return float(Hartree)


def _water() -> Atoms:
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])


def _stub_acquisition(value_fn, gradient_fn) -> SeedLocalAdversarialAcquisition:
    """Return a SeedLocalAdversarialAcquisition with manually-set value/grad."""
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)

    def _gradient(atoms, mode=None):
        return gradient_fn(atoms)

    acq.value = value_fn
    acq.gradient = _gradient
    return acq


def test_energy_is_negative_alpha_in_ev():
    atoms = _water()
    H = _hartree_ev()
    alpha = 1.23

    acq = _stub_acquisition(lambda a: alpha, lambda a: np.zeros((3, 3)))
    calc = AdversarialASECalculator(
        acq,
        project_rigid=False,
        max_force_per_atom_ha_per_ang=0.0,
    )
    energy = calc.get_potential_energy(atoms)

    assert energy == pytest.approx(-alpha * H, rel=1.0e-12)


def test_forces_are_positive_alpha_gradient_in_ev_per_angstrom():
    atoms = _water()
    H = _hartree_ev()
    # Use a vibrational gradient (orthogonal to rigid modes) so rigid
    # projection wouldn't change the answer in the next test.
    raw = np.zeros(9); raw[0] = 1.0; raw[4] = -2.0; raw[8] = 0.5
    # Strip the rigid part with the Euclidean dual projector used for scalar
    # gradients. A mass-weighted vector projection is not the covector
    # condition required by rigid invariance.
    from ichor.hpc.active_learning.acquisition.rigid_projection import (
        rigid_basis,
    )
    B = rigid_basis(atoms)
    grad_vib_flat = raw - B @ (B.T @ raw)
    grad_vib = grad_vib_flat.reshape(3, 3)

    acq = _stub_acquisition(lambda a: 0.0, lambda a: grad_vib)
    calc = AdversarialASECalculator(
        acq,
        project_rigid=True,
        max_force_per_atom_ha_per_ang=0.0,
    )
    forces = calc.get_forces(atoms)

    # forces (eV/A) = +grad(alpha) (acquisition units/A) * Hartree pseudo-scale
    expected = grad_vib * H
    np.testing.assert_allclose(forces, expected, atol=1.0e-10)


def test_rigid_projection_strips_translation_from_forces():
    atoms = _water()
    H = _hartree_ev()
    # Pure translation in Cartesian: every atom gets the same gradient row.
    t = np.array([1.0, 2.0, -0.5])
    grad = np.tile(t, (3, 1))

    acq = _stub_acquisition(lambda a: 0.0, lambda a: grad)

    calc_no_proj = AdversarialASECalculator(
        acq, project_rigid=False, max_force_per_atom_ha_per_ang=0.0,
    )
    f_no_proj = calc_no_proj.get_forces(atoms)
    # Without projection, the per-atom pseudo-force equals t * H.
    np.testing.assert_allclose(f_no_proj, grad * H, atol=1.0e-10)

    calc_proj = AdversarialASECalculator(
        acq, project_rigid=True, max_force_per_atom_ha_per_ang=0.0,
    )
    f_proj = calc_proj.get_forces(atoms)
    # With projection, the pure-translation gradient is entirely in the
    # rigid subspace -> projected force is ~0.
    np.testing.assert_allclose(f_proj, np.zeros((3, 3)), atol=1.0e-10)


def test_per_atom_acquisition_gradient_clamp_and_counter():
    atoms = _water()
    H = _hartree_ev()
    # Vibrational gradient with one row well above the clamp.
    # Build via projection so rigid component is gone before clamping.
    raw = np.array([
        [100.0, 0.0, 0.0],   # |grad| = 100 acquisition units/A; should clamp to 5
        [-50.0, 1.0, 0.0],   # vibrational; small after projection
        [0.0, 0.0, 0.0],
    ])
    grad_after_proj = project_out_rigid(raw, atoms)

    # Wire calculator to return raw (so calc itself runs projection).
    acq = _stub_acquisition(lambda a: 0.0, lambda a: raw)
    counter = Counter()
    calc = AdversarialASECalculator(
        acq,
        project_rigid=True,
        max_acquisition_grad_per_ang=5.0,
        clamp_counter=counter,
    )
    forces = calc.get_forces(atoms)

    # Per-atom norms after clamp must all be <= 5 acquisition units/A * Hartree.
    per_atom_ev = np.linalg.norm(forces, axis=1)
    assert np.all(per_atom_ev <= 5.0 * H + 1.0e-9)
    # At least one row should have been clamped (counter > 0).
    assert counter.get("per_atom_acquisition_grad", 0) > 0


def test_deprecated_force_clamp_alias_still_applies():
    atoms = _water()
    H = _hartree_ev()
    raw = np.array([[100.0, 0.0, 0.0], [-50.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    acq = _stub_acquisition(lambda a: 0.0, lambda a: raw)
    calc = AdversarialASECalculator(
        acq,
        project_rigid=True,
        max_force_per_atom_ha_per_ang=5.0,
    )
    forces = calc.get_forces(atoms)
    assert np.all(np.linalg.norm(forces, axis=1) <= 5.0 * H + 1.0e-9)


def test_value_and_gradient_cache_avoids_double_compute():
    """A paired energy + forces call on the same coordinates must evaluate
    the underlying acquisition exactly once."""
    atoms = _water()
    call_counter = {"value": 0, "grad": 0}

    def v(a):
        call_counter["value"] += 1
        return 1.5

    def g(a):
        call_counter["grad"] += 1
        return np.zeros((3, 3))

    acq = _stub_acquisition(v, g)
    calc = AdversarialASECalculator(
        acq, project_rigid=False, max_force_per_atom_ha_per_ang=0.0,
    )

    e = calc.get_potential_energy(atoms)
    f = calc.get_forces(atoms)
    assert call_counter["value"] == 1
    assert call_counter["grad"] == 1


def test_gradient_diagnostics_distinguish_raw_projected_and_capped_norms():
    atoms = _water()
    raw = np.array([
        [100.0, 0.0, 0.0],
        [-50.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    acq = _stub_acquisition(lambda a: 0.0, lambda a: raw)
    calc = AdversarialASECalculator(
        acq,
        project_rigid=True,
        max_acquisition_grad_per_ang=5.0,
    )

    calc.get_forces(atoms)
    diagnostics = calc.gradient_diagnostics()

    assert diagnostics["last_gradient_norm_raw"] == pytest.approx(
        np.linalg.norm(raw)
    )
    assert diagnostics["last_gradient_norm_post_rigid"] <= diagnostics[
        "last_gradient_norm_raw"
    ]
    assert diagnostics["last_gradient_norm_post_cap"] <= 5.0 * np.sqrt(len(atoms))
    assert diagnostics["last_gradient_norm"] == pytest.approx(
        diagnostics["last_gradient_norm_post_cap"]
    )


def test_ase_atoms_to_ichor_atoms_roundtrip():
    """ASE atoms in -> ichor Atoms with matching elements/coords."""
    pytest.importorskip("ase")
    import ase
    atoms_ase = ase.Atoms("OH2", positions=[
        (0.0, 0.0, 0.0),
        (0.96, 0.0, 0.0),
        (-0.24, 0.93, 0.0),
    ])
    out = ase_atoms_to_ichor_atoms(atoms_ase)
    assert [a.type for a in out] == ["O", "H", "H"]
    np.testing.assert_allclose(
        np.asarray(out.coordinates),
        atoms_ase.get_positions(),
        atol=1.0e-12,
    )
