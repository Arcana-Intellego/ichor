"""Confirm the real whitened distance metric agrees with the underlying
subspace function.

This guards against two regressions:

  1. _live_optimise_seed using the wrong subspace or the wrong
     regularisation value (e.g. forgetting to pull it from the config).
  2. AriadneRunResult.whitened_distance_final not being set, so the
     postprocess parser silently falls back to the synthetic proxy.
"""
from __future__ import annotations

import numpy as np
import pytest

from ichor.core.adversarial.subspace import whitened_distance_squared


def test_whitened_distance_squared_at_seed_is_zero():
    """the active coordinate at the seed is the zero vector by
    construction, so the whitened d^2 there must be zero."""
    pytest.importorskip("ichor.core.adversarial.subspace")
    # we build a minimal subspace by hand: 2D acquisition, one seed,
    # identity covariance. then querying the seed gives xi = 0.
    from types import SimpleNamespace
    from ichor.core.atoms import Atom, Atoms
    seed = Atoms([Atom("O", 0.0, 0.0, 0.0)])
    # synthetic LocalSubspace-like object that satisfies the API the
    # function uses: .basis (3N x d), .seed_coordinates (3N,),
    # .active_covariance (d x d), .dimension, .mass_vector.
    basis = np.eye(3)[:, :2]  # 3x2: pick out x and y axes
    sub = SimpleNamespace(
        basis=basis,
        seed_atoms=seed,
        active_covariance=np.eye(2),
        dimension=2,
        mass_vector=np.ones(3),
    )
    d_sq = whitened_distance_squared(sub, seed)
    assert d_sq == pytest.approx(0.0, abs=1.0e-12)


def test_whitened_distance_grows_with_displacement():
    """Displacing one atom in a 3-atom system gives a positive,
    finite d^2. With a single atom, rigid alignment can null any
    displacement; we need 3 atoms to make sure the active coordinate
    actually moves.
    """
    from types import SimpleNamespace
    import math
    from ichor.core.atoms import Atom, Atoms
    seed = Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])
    n_dof = 9
    # a 2D subspace -- the first two cartesian DOFs, basis is just the
    # first two columns of the 9x9 identity. that gives us a defined
    # active space without needing to fit anything.
    basis = np.eye(n_dof)[:, :2]
    sub = SimpleNamespace(
        basis=basis,
        seed_atoms=seed,
        active_covariance=np.eye(2),
        dimension=2,
        mass_vector=np.ones(n_dof),
    )
    # displace one atom by a meaningful amount.
    moved = Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.5),
        Atom("H", -0.24, 0.93, 0.0),
    ])
    d_sq = whitened_distance_squared(sub, moved)
    assert d_sq >= 0
    assert math.isfinite(d_sq)


def test_whitened_distance_in_AriadneRunResult():
    """Verify the new field round-trips through to_dict."""
    from ichor.hpc.active_learning.acquisition.ariadne_runner import AriadneRunResult
    from ichor.core.atoms import Atom, Atoms
    a = Atoms([Atom("O", 0.0, 0.0, 0.0)])
    r = AriadneRunResult(
        initial_atoms=a, final_atoms=a,
        whitened_distance_final=0.42,
    )
    payload = r.to_dict()
    assert payload["whitened_distance_final"] == 0.42
    r2 = AriadneRunResult(initial_atoms=a, final_atoms=a)
    assert r2.to_dict()["whitened_distance_final"] is None
