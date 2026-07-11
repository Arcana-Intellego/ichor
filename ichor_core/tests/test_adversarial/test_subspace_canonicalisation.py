"""M2 subspace-basis canonicalisation in adversarial.subspace.

These tests target the private helper '_canonicalise_basis' directly so that
the verification does not require constructing a full GP posterior. The
helper's contract: given an orthonormal basis of an r-dimensional eigenspace
together with its eigenvalues, return a representation that is 
(i) sign-stable per column and 
(ii) rotation-stable within any near-degenerate eigenvalue block.
"""
import numpy as np
import pytest

from ichor.core.adversarial.config import SubspaceConfig
from ichor.core.adversarial.subspace import _canonicalise_basis


def _orthonormalise(M):
    Q, _ = np.linalg.qr(M)
    return Q


def test_core_subspace_config_defaults_to_canonical_basis():
    assert SubspaceConfig().canonicalise_basis is True


def test_canonicalise_is_idempotent():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((9, 4))
    basis = _orthonormalise(raw)
    eigs = np.array([4.0, 3.0, 2.0, 1.0])

    once = _canonicalise_basis(basis, eigs, degeneracy_tolerance=1.0e-3)
    twice = _canonicalise_basis(once, eigs, degeneracy_tolerance=1.0e-3)
    np.testing.assert_allclose(once, twice, atol=1.0e-12)


def test_sign_convention_is_applied():
    #Construct a 2-column basis with deliberately negative leading components.
    basis = np.zeros((6, 2))
    basis[:, 0] = np.array([-0.6, 0.8, 0.0, 0.0, 0.0, 0.0])
    basis[:, 1] = np.array([0.0, 0.0, -0.5, 0.0, np.sqrt(0.75), 0.0])
    eigs = np.array([5.0, 1.0])

    out = _canonicalise_basis(basis, eigs, degeneracy_tolerance=1.0e-3)
    # first non-trivial component of each column must be positive.
    for j in range(out.shape[1]):
        col = out[:, j]
        i = np.argmax(np.abs(col) > 1e-10)
        assert col[i] > 0


def test_basis_stable_under_sign_flip_of_input():
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((9, 3))
    basis = _orthonormalise(raw)
    eigs = np.array([3.0, 2.0, 1.0])

    out_a = _canonicalise_basis(basis, eigs, degeneracy_tolerance=1.0e-3)

    #Flip the sign of column output must be identical after canonicalisation.
    flipped = basis.copy()
    flipped[:, 1] *= -1.0
    out_b = _canonicalise_basis(flipped, eigs, degeneracy_tolerance=1.0e-3)

    np.testing.assert_allclose(out_a, out_b, atol=1.0e-12)


def test_basis_stable_under_rotation_within_degenerate_block():
    rng = np.random.default_rng(2)
    #3 columns; the first two share a single eigenvalue, the third differs.
    raw = rng.standard_normal((9, 3))
    basis = _orthonormalise(raw)
    eigs = np.array([2.0, 2.0, 1.0])  # first two degenerate

    #Apply an arbitrary 2x2 rotation to columns 0-1.
    theta = 0.789
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    rotated = basis.copy()
    rotated[:, 0:2] = basis[:, 0:2] @ R

    out_a = _canonicalise_basis(basis,   eigs, degeneracy_tolerance=1.0e-3)
    out_b = _canonicalise_basis(rotated, eigs, degeneracy_tolerance=1.0e-3)

    #The Procrustes stage maps both inputs to the same canonical
    #in-block representation up to a permitted sign flip per column.
    for j in range(out_a.shape[1]):
        diff_same = np.linalg.norm(out_a[:, j] - out_b[:, j])
        diff_flip = np.linalg.norm(out_a[:, j] + out_b[:, j])
        assert min(diff_same, diff_flip) < 1.0e-9


def test_subspace_invariance_under_basis_choice():
    """Even when two raw bases differ by an arbitrary in-block rotation, the
    spanned subspace and its outer-product projector must agree after
    canonicalisation."""
    rng = np.random.default_rng(3)
    raw = rng.standard_normal((9, 4))
    basis = _orthonormalise(raw)
    eigs = np.array([2.0, 2.0, 2.0, 1.0])  #first three degenerate

    rot = _orthonormalise(rng.standard_normal((3, 3)))
    rotated = basis.copy()
    rotated[:, 0:3] = basis[:, 0:3] @ rot

    out_a = _canonicalise_basis(basis,   eigs, degeneracy_tolerance=1.0e-3)
    out_b = _canonicalise_basis(rotated, eigs, degeneracy_tolerance=1.0e-3)

    #Projector onto the subspace must be invariant.
    P_a = out_a @ out_a.T
    P_b = out_b @ out_b.T
    np.testing.assert_allclose(P_a, P_b, atol=1.0e-10)


def test_canonicalisation_preserves_orthonormality():
    rng = np.random.default_rng(4)
    raw = rng.standard_normal((9, 5))
    basis = _orthonormalise(raw)
    eigs = np.array([4.0, 4.0, 3.0, 2.0, 1.0])  #one degenerate pair at the top

    out = _canonicalise_basis(basis, eigs, degeneracy_tolerance=1.0e-3)
    gram = out.T @ out
    np.testing.assert_allclose(gram, np.eye(5), atol=1.0e-10)


def test_degenerate_block_is_stable_when_leading_cartesian_probes_have_zero_rank():
    basis = np.zeros((8, 2), dtype=float)
    basis[5, 0] = 1.0
    basis[7, 1] = 1.0
    eigenvalues = np.array([2.0, 2.0])
    angle = 0.731
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    canonical = _canonicalise_basis(
        basis,
        eigenvalues,
        degeneracy_tolerance=1.0e-3,
    )
    rotated = _canonicalise_basis(
        basis @ rotation,
        eigenvalues,
        degeneracy_tolerance=1.0e-3,
    )

    np.testing.assert_allclose(canonical, rotated, atol=1.0e-12)
