"""Tests for ichor.hpc.active_learning.acquisition.rigid_projection."""
import math

import numpy as np
import pytest

from ichor.core.atoms import Atom, Atoms

from ichor.hpc.active_learning.acquisition.rigid_projection import (
    mass_vector_for,
    project_out_rigid,
    rigid_basis,
)


def _water() -> Atoms:
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])


def _co2() -> Atoms:
    # Linear molecule along x axis. One rotational column collapses to zero.
    return Atoms([
        Atom("O", -1.16, 0.0, 0.0),
        Atom("C", 0.0, 0.0, 0.0),
        Atom("O", 1.16, 0.0, 0.0),
    ])


def test_rigid_basis_water_has_rank_6():
    atoms = _water()
    B = rigid_basis(atoms)
    # 9 Cartesian coordinates, 6 rigid modes for a non-linear molecule.
    assert B.shape == (9, 6)
    # Orthonormal columns (post-SVD).
    np.testing.assert_allclose(B.T @ B, np.eye(6), atol=1.0e-10)


def test_rigid_basis_linear_co2_has_rank_5():
    atoms = _co2()
    B = rigid_basis(atoms)
    assert B.shape == (9, 5)
    np.testing.assert_allclose(B.T @ B, np.eye(5), atol=1.0e-10)


def test_rigid_basis_single_atom_has_rank_3():
    atoms = Atoms([Atom("H", 0.0, 0.0, 0.0)])
    B = rigid_basis(atoms)
    # 3 translations only.
    assert B.shape == (3, 3)
    np.testing.assert_allclose(B.T @ B, np.eye(3), atol=1.0e-10)


def test_mass_vector_repeats_per_coordinate():
    atoms = _water()
    M = mass_vector_for(atoms)
    assert M.shape == (9,)
    expected = np.repeat([atoms[0].mass, atoms[1].mass, atoms[2].mass], 3)
    np.testing.assert_allclose(M, expected, atol=1.0e-12)


def test_pure_translation_projected_out():
    atoms = _water()
    # A pure translation: same translation vector replicated for every atom.
    t = np.array([0.3, -0.1, 0.07])
    grad = np.tile(t, len(atoms))  # shape (9,)
    out = project_out_rigid(grad, atoms)
    np.testing.assert_allclose(out, np.zeros_like(grad), atol=1.0e-10)


def test_pure_rotation_projected_out():
    atoms = _water()
    coords = np.asarray(atoms.coordinates, dtype=float)
    centroid = coords.mean(axis=0)
    rel = coords - centroid
    # Infinitesimal rotation about z axis: g_a = e_z x r_rel_a
    e_z = np.array([0.0, 0.0, 1.0])
    rot_grad = np.array([np.cross(e_z, rel[i]) for i in range(len(atoms))]).reshape(-1)
    out = project_out_rigid(rot_grad, atoms)
    np.testing.assert_allclose(out, np.zeros_like(rot_grad), atol=1.0e-10)


def test_pure_vibrational_gradient_preserved():
    """A gradient that is mass-weighted-orthogonal to all rigid modes must
    pass through `project_out_rigid` unchanged."""
    atoms = _water()
    rng = np.random.default_rng(0)
    M = mass_vector_for(atoms)
    B = rigid_basis(atoms)

    # Build a vibrational gradient: take a random vector and explicitly
    # remove its rigid component in the M-inner-product. Result is by
    # construction in the vibrational subspace; projecting again is a no-op.
    raw = rng.standard_normal(9)
    BTMB = B.T @ (M[:, None] * B)
    coeffs = np.linalg.solve(BTMB, B.T @ (M * raw))
    pure_vib = raw - B @ coeffs

    out = project_out_rigid(pure_vib, atoms)
    np.testing.assert_allclose(out, pure_vib, atol=1.0e-10)


def test_mixed_gradient_decomposed_correctly():
    """For any g, project_out_rigid(g) must (i) lie in the vibrational
    subspace and (ii) reproduce the same total norm structure when
    re-projected (idempotence)."""
    atoms = _water()
    M = mass_vector_for(atoms)
    B = rigid_basis(atoms)
    rng = np.random.default_rng(1)
    grad = rng.standard_normal(9)
    proj = project_out_rigid(grad, atoms)

    # Component along rigid modes (M-projection) must be zero.
    rigid_component_M = B.T @ (M * proj)
    np.testing.assert_allclose(rigid_component_M, np.zeros(B.shape[1]), atol=1.0e-10)

    # Idempotence.
    np.testing.assert_allclose(project_out_rigid(proj, atoms), proj, atol=1.0e-10)


def test_project_handles_2d_shape():
    """If grad is shaped (N, 3) (the ASE convention), the projected output
    must keep the same shape."""
    atoms = _water()
    rng = np.random.default_rng(2)
    grad = rng.standard_normal((3, 3))
    out = project_out_rigid(grad, atoms)
    assert out.shape == grad.shape


def test_project_invariant_to_centroid_translation():
    """Projecting from a rotated/translated reference frame must give the
    same result. We translate the whole molecule by an arbitrary vector,
    apply the same gradient, and check the projector output is unchanged."""
    atoms_a = _water()
    atoms_b = Atoms([
        Atom(a.type, *(np.asarray(a.coordinates) + np.array([10.0, -3.0, 1.5])))
        for a in atoms_a
    ])
    rng = np.random.default_rng(3)
    grad = rng.standard_normal(9)
    out_a = project_out_rigid(grad, atoms_a)
    out_b = project_out_rigid(grad, atoms_b)
    # Translation only affects the rotation columns of B, which themselves
    # depend on r_rel (centroid-relative). Centroid-relative coords are the
    # SAME for both molecules so the projector is identical.
    np.testing.assert_allclose(out_a, out_b, atol=1.0e-10)
