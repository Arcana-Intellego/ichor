"""Capped softplus_sq in the chemistry barrier.

The clash term in chemistry_barrier_value is

    lambda_clash * softplus((safe_d - d) / delta) ** 2

For d -> 0 the argument grows without bound. softplus(x) ~ x for large x,
so the squared term grows like x^2 -> the per-pair contribution diverges
linearly in time as ARIADNE drifts into the clash. With cap=C, the argument
is clamped to C, so the contribution saturates at softplus(C)^2.
"""
import math
import numpy as np
import pytest

from ichor.core.adversarial.barrier import _softplus, _softplus_sq


def test_softplus_sq_default_unchanged():
    """Without cap, the function must reproduce prior behaviour exactly."""
    delta = 0.05
    for u in (-1.0, 0.0, 0.01, 0.1, 1.0, 5.0):
        expected = float(_softplus(u / delta)) ** 2
        got = _softplus_sq(u, delta)
        assert math.isclose(got, expected, rel_tol=1.0e-12, abs_tol=1.0e-15)


def test_softplus_sq_cap_saturates():
    """With cap=C, the per-pair contribution saturates at softplus(C)^2."""
    delta = 0.05
    cap = 10.0
    saturation = float(_softplus(cap)) ** 2

    #below the cap: behaviour identical to the uncapped form.
    for u_over_delta in (-1.0, 0.0, 1.0, 5.0, 9.5):
        u = u_over_delta * delta
        got_cap = _softplus_sq(u, delta, cap=cap)
        got_nocap = _softplus_sq(u, delta)
        assert math.isclose(got_cap, got_nocap, rel_tol=1.0e-12)

    #At or above the cap: contribution clamped.
    for u_over_delta in (10.0, 50.0, 1_000.0, 1.0e6):
        u = u_over_delta * delta
        got = _softplus_sq(u, delta, cap=cap)
        assert math.isclose(got, saturation, rel_tol=1.0e-12)


def test_softplus_sq_cap_monotone_below_cap():
    """For u/delta in [0, cap], the capped value should still be monotone
    non-decreasing in u (sanity check that gradient sign wasn't broken)."""
    delta = 0.05
    cap = 10.0
    us = np.linspace(0.0, cap * delta, 50)
    vals = [_softplus_sq(u, delta, cap=cap) for u in us]
    diffs = np.diff(vals)
    assert np.all(diffs >= -1.0e-15)


def test_softplus_sq_cap_finite_for_extreme_arguments():
    """The cap must yield a finite, well-defined value even for cosmically
    large arguments (defends against NaN/Inf in the ARIADNE callback)."""
    delta = 0.05
    cap = 10.0
    for u in (1.0e3, 1.0e6, 1.0e12, 1.0e18):
        v = _softplus_sq(u, delta, cap=cap)
        assert math.isfinite(v)
        assert v > 0.0


def test_barrier_value_with_cap_versus_without_at_severe_clash():
    """End-to-end: building a ChemistryBarrierState on a water dimer-like
    geometry, then evaluating the barrier at a deeply collided configuration,
    the *capped* configuration must produce a strictly smaller (and finite)
    value than the uncapped one."""
    from ichor.core.adversarial.barrier import (
        build_chemistry_barrier_state,
        chemistry_barrier_value,
    )
    from ichor.core.adversarial.config import BarrierConfig
    from ichor.core.atoms import Atom, Atoms

    #Tiny H2 dimer-style seed; "neighbour" is the same to avoid having to
    #build a posterior in this unit test.
    seed = Atoms([Atom("H", 0.0, 0.0, 0.0), Atom("H", 0.74, 0.0, 0.0)])
    neighbours = [Atoms([Atom("H", 0.0, 0.0, 0.0), Atom("H", 0.75, 0.0, 0.0)])]

    class _StubPosterior:
        def mean(self, atoms):
            return 0.0

    cfg_uncapped = BarrierConfig(softplus_cap=None)
    cfg_capped = BarrierConfig(softplus_cap=10.0)

    state_uncapped = build_chemistry_barrier_state(seed, neighbours, _StubPosterior(), cfg_uncapped)
    state_capped = build_chemistry_barrier_state(seed, neighbours, _StubPosterior(), cfg_capped)

    #Geometry with a severe clash: the two H atoms practically on top.
    clashed = Atoms([Atom("H", 0.0, 0.0, 0.0), Atom("H", 0.01, 0.0, 0.0)])

    v_uncapped = chemistry_barrier_value(clashed, state_uncapped, mean_energy=0.0)
    v_capped = chemistry_barrier_value(clashed, state_capped, mean_energy=0.0)

    assert math.isfinite(v_uncapped)
    assert math.isfinite(v_capped)
    assert v_capped < v_uncapped
