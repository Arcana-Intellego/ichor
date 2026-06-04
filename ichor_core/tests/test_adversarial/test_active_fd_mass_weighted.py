""" Protective regression for _active_finite_difference_gradient.

I thought of replacing the Moore-Penrose pseudoinverse
projection 'D (D^T D)^{-1} rhs' with a mass-weighted Gram solve. Implementing
that "fix" exposed -- via the full-rank test below -- that the original
formula is actually CORRECT: for full-rank D it exactly recovers the
Cartesian gradient regardless of mass heterogeneity, while the proposed
mass-weighted version divides by mass (multiplies by M^{-1}).

These tests pin the original Cartesian-least-squares projection behaviour so
a future refactor that reintroduces the mass-weighted variant immediately
fails. See acquisition.py:_active_finite_difference_gradient docstring for
the full reasoning.
"""
import numpy as np
import pytest

from ichor.core.atoms import Atom, Atoms
from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
from ichor.core.adversarial.config import AcquisitionConfig
from ichor.core.adversarial.geometry import mass_vector


def _two_atom(symbol_a: str, symbol_b: str) -> Atoms:
    return Atoms([
        Atom(symbol_a, 0.0, 0.0, 0.0),
        Atom(symbol_b, 1.0, 0.0, 0.0),
    ])


def _make_acq(value_fn, mode_directions):
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    acq.config = AcquisitionConfig()
    object.__setattr__(acq.config.gradient, "regularization", 0.0)
    acq.mode_directions = mode_directions
    acq.value = value_fn  #type: ignore[assignment]
    return acq


def test_full_rank_recovers_cartesian_gradient_for_heterogeneous_masses():
    """Pin behaviour: with a full-rank mass-weighted basis (3N modes), the
    FD gradient must recover the analytic Cartesian gradient EXACTLY for
    any mass ratio. This is the regression that catches a future "mass-
    weighted Gram" rewrite -- it would divide every component by m[i] and
    fail this test loudly."""
    atoms = _two_atom("H", "Br")   #1.008 vs 79.9 amu

    #V(x) = 0.5 * sum(x_i^2)  =>  Cartesian grad = x
    def value(a):
        coords = np.asarray(a.coordinates, dtype=float).reshape(-1)
        return 0.5 * float(np.dot(coords, coords))

    M = mass_vector(atoms)
    sqrtMinv = 1.0 / np.sqrt(M)
    mode_directions = [sqrtMinv * np.eye(6)[i] for i in range(6)]

    acq = _make_acq(value, mode_directions)
    g = acq._active_finite_difference_gradient(atoms)

    expected = np.asarray(atoms.coordinates, dtype=float)
    np.testing.assert_allclose(g, expected, atol=1e-6)


def test_full_rank_recovery_independent_of_mass_ratio():
    """The Cartesian-least-squares projection is invariant under mass
    rescaling for full-rank bases. Run with two very different mass ratios
    and verify the recovered gradient is identical."""
    for symbols in (("H", "H"), ("H", "Br"), ("C", "Zn")):
        atoms = Atoms([
            Atom(symbols[0], 0.5, 0.0, 0.0),
            Atom(symbols[1], 2.5, 0.0, 0.0),
        ])

        def value(a):
            coords = np.asarray(a.coordinates, dtype=float).reshape(-1)
            return 0.5 * float(np.dot(coords, coords))

        M = mass_vector(atoms)
        sqrtMinv = 1.0 / np.sqrt(M)
        mode_directions = [sqrtMinv * np.eye(6)[i] for i in range(6)]

        acq = _make_acq(value, mode_directions)
        g = acq._active_finite_difference_gradient(atoms)

        np.testing.assert_allclose(
            g, np.asarray(atoms.coordinates, dtype=float), atol=1e-6,
            err_msg=f"failed for {symbols}",
        )


def test_subset_basis_is_cartesian_least_squares_projection():
    """With a SUBSET basis (1 mode), the result is the orthogonal Cartesian
    projection of the gradient onto the subspace -- mathematically what
    ARIADNE wants because its descent step lives in Cartesian coordinates.

    We compute the expected projection analytically:
        g_proj = D (D^T D)^{-1} D^T g
    and confirm the FD gradient matches it.
    """
    atoms = _two_atom("H", "Br")
    M = mass_vector(atoms)
    sqrtMinv = 1.0 / np.sqrt(M)

    #Single coupled mode in mass-weighted space.
    b = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    d = sqrtMinv * b

    #V(x) = x[0] + x[3]  =>  Cartesian grad = [1, 0, 0, 1, 0, 0]
    def value(a):
        coords = np.asarray(a.coordinates, dtype=float).reshape(-1)
        return float(coords[0] + coords[3])

    acq = _make_acq(value, [d])
    g_recon = acq._active_finite_difference_gradient(atoms).reshape(-1)

    #Analytic projection
    g_true = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    D = d.reshape(-1, 1)
    proj = D @ np.linalg.solve(D.T @ D, D.T @ g_true)

    np.testing.assert_allclose(g_recon, proj.reshape(-1), atol=1e-6)
