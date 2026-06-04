"""Mass-weighted rigid-body projector for Cartesian acquisition gradients.

Given a scalar acquisition f(x) that is invariant under simultaneous
translations and rigid rotations of all atoms, the natural inner product on
its Cartesian gradient is mass-weighted:

    <u, v>_M = u^T M v       with  M = diag(m_1, m_1, m_1, m_2, m_2, m_2, ...)

The mass-weighted orthogonal projector onto the rigid-body subspace
spanned by columns of B (3N x k, k = 6 for non-linear, k = 5 for linear,
k = 3 for a single atom) is

    P_rigid = B (B^T M B)^-1 B^T M

and "project_out_rigid(g, atoms) = (I - P_rigid) g" strips the rigid
component from any Cartesian gradient. Doing this at the ASE-calculator
boundary keeps ARIADNE's curvature gate clean of FD noise that lives in
directions the optimiser is supposed to ignore.

The implementation auto-detects collinear / single-atom geometries by SVD
rank thresholding so that no spurious zero-mode is left in B.
"""
from __future__ import annotations

import numpy as np
from ichor.core.atoms import Atoms


__all__ = ["rigid_basis", "mass_vector_for", "project_out_rigid"]


def mass_vector_for(atoms: Atoms) -> np.ndarray:
    """Return the 3N vector of per-Cartesian-coordinate masses [m1, m1, m1, m2, ...]."""
    masses = np.array([float(a.mass) for a in atoms], dtype=float)
    return np.repeat(masses, 3)


def rigid_basis(atoms: Atoms, linear_tol: float = 1.0e-10) -> np.ndarray:
    """Return an SVD-orthonormalised 3N x k basis of infinitesimal rigid motions.

    Columns 0..2 represent translations (e_x, e_y, e_z replicated per atom);
    columns 3..5 represent rotations about the centroid via the cross product
    r_a^rel x e_axis. For collinear or single-atom geometries, one or more
    rotation columns are rank-deficient; those are detected via the smallest
    singular value of the 6-column ansatz and dropped, so the returned basis
    is always an orthonormal spanning set of the true rigid subspace.
    """
    n = len(atoms)
    coords = np.asarray(atoms.coordinates, dtype=float)
    if coords.shape != (n, 3):
        raise ValueError(f"unexpected coordinates shape {coords.shape}")
    centroid = coords.mean(axis=0)
    rel = coords - centroid

    B = np.zeros((3 * n, 6), dtype=float)
    #translations
    for j in range(3):
        B[3 * np.arange(n) + j, j] = 1.0
    #rotations: row block for atom i along axis j_axis is e_axis x r_rel_i
    for j_axis in range(3):
        e = np.zeros(3)
        e[j_axis] = 1.0
        for i in range(n):
            B[3 * i : 3 * i + 3, 3 + j_axis] = np.cross(e, rel[i])

    #determine effective rank via SVD.
    U, S, _ = np.linalg.svd(B, full_matrices=False)
    if S.size == 0 or S[0] == 0.0:
        return np.zeros((3 * n, 0), dtype=float)
    cutoff = float(linear_tol) * float(S[0])
    rank = int(np.sum(S > cutoff))
    return np.ascontiguousarray(U[:, :rank])


def project_out_rigid(
    grad: np.ndarray,
    atoms: Atoms,
    *,
    rcond: float = 1.0e-12,
) -> np.ndarray:
    """Project rigid translations + rotations out of a Cartesian gradient.

    "grad" may be flat "(3N,)" or shaped (N, 3); the projected gradient
    is returned with the same shape. The projection is mass-weighted:

        g_phys = g - B (B^T M B)^-1 B^T M g

    where B is the orthonormal rigid basis from :func:"rigid_basis" and M is
    the diagonal mass matrix from :func:"mass_vector_for". The Gram matrix
    "B^T M B" is k x k SPD; we solve it with "np.linalg.solve" and fall
    back to a least-squares solve if it is rank-deficient.
    """
    g = np.asarray(grad, dtype=float)
    orig_shape = g.shape
    flat = g.reshape(-1).astype(float, copy=True)
    M = mass_vector_for(atoms)
    if flat.shape[0] != M.shape[0]:
        raise ValueError(
            f"gradient size {flat.shape[0]} does not match 3*natoms {M.shape[0]}"
        )

    B = rigid_basis(atoms)
    if B.shape[1] == 0:
        return flat.reshape(orig_shape)

    BTMB = B.T @ (M[:, None] * B)
    rhs = B.T @ (M * flat)
    try:
        coeffs = np.linalg.solve(BTMB, rhs)
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(BTMB, rhs, rcond=rcond)
    proj_flat = flat - B @ coeffs
    return proj_flat.reshape(orig_shape)
