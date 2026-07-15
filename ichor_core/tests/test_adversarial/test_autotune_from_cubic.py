"""Per-mode FD step autotune from cubic estimate.

Strategy: bypass the full GP / TotalEnergyPosterior construction by
instantiating just enough of SeedLocalAdversarialAcquisition to
exercise _mode_metrics + _refine_steps_from_cubic + the cache path.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from ichor.core.atoms import Atom, Atoms
from ichor.core.adversarial.acquisition import (
    ModeEvaluation,
    SeedLocalAdversarialAcquisition,
)
from ichor.core.adversarial.config import AcquisitionConfig


def _make_acq(autotune=False, default_step=0.1):
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    cfg = AcquisitionConfig()
    object.__setattr__(cfg.stencils, "autotune_from_cubic", autotune)
    object.__setattr__(cfg.stencils, "min_step", 1e-4)
    object.__setattr__(cfg.stencils, "max_step", 0.5)
    acq.config = cfg
    acq.tuned_mode_steps = None
    return acq


def _make_evaluation(cubic_mean, force_mean=1.0, force_std=1.0):
    """Synthetic ModeEvaluation for refine-tests."""
    return ModeEvaluation(
        index=0, force_mean=force_mean, force_std=force_std,
        curvature_mean=1.0, curvature_std=0.1,
        omega=1.0, omega_std=0.1, cubic_mean=cubic_mean, cubic_std=0.01,
        quartic_mean=0.0, quartic_std=0.0,
        anharmonicity=0.0, anharmonicity_std=0.0,
    )


# --- _refine_steps_from_cubic math ---

def test_refine_steps_targets_one_percent_error():
    """Target: eps^2 * |cubic| / 6 = 0.01 * |grad|.
    Solve: eps = sqrt(0.06 * grad / |cubic|).
    """
    acq = _make_acq(autotune=True)
    grad_mag, cubic_mag = 1.0, 1.0
    expected = math.sqrt(0.06 * grad_mag / cubic_mag)
    tuned = acq._refine_steps_from_cubic([
        _make_evaluation(cubic_mean=cubic_mag, force_mean=grad_mag),
    ])
    assert tuned[0] == pytest.approx(expected, rel=1e-9)


def test_refine_steps_clamps_to_max():
    """Tiny cubic + big gradient -> target step is huge; clamp to max_step."""
    acq = _make_acq(autotune=True)
    # 0.01 * 100 / 1e-9 = 1e9 -> eps target = ~31000 -> clamp to max_step.
    tuned = acq._refine_steps_from_cubic([
        _make_evaluation(cubic_mean=1e-9, force_mean=100.0),
    ])
    assert tuned[0] == acq.config.stencils.max_step


def test_refine_steps_clamps_to_min():
    """Huge cubic -> target step is tiny; clamp to min_step."""
    acq = _make_acq(autotune=True)
    # 0.01 * 1e-9 / 100 = 1e-13 -> eps target ~ 3e-7 -> clamp to min_step.
    tuned = acq._refine_steps_from_cubic([
        _make_evaluation(cubic_mean=100.0, force_mean=1e-9),
    ])
    assert tuned[0] == acq.config.stencils.min_step


def test_refine_handles_zero_cubic_safely():
    """Cubic = 0 should not crash (denominator is floored)."""
    acq = _make_acq(autotune=True)
    tuned = acq._refine_steps_from_cubic([
        _make_evaluation(cubic_mean=0.0, force_std=1.0),
    ])
    #should clamp to max_step (target would be huge).
    assert tuned[0] == acq.config.stencils.max_step


def test_refine_per_mode_independent():
    """Different cubics per mode -> different tuned eps."""
    acq = _make_acq(autotune=True)
    evals = [
        _make_evaluation(cubic_mean=1.0, force_mean=1.0),
        _make_evaluation(cubic_mean=4.0, force_mean=1.0),
    ]
    tuned = acq._refine_steps_from_cubic(evals)
    assert tuned[0] == pytest.approx(math.sqrt(0.06), rel=1e-9)
    assert tuned[1] == pytest.approx(math.sqrt(0.015), rel=1e-9)


def test_refine_steps_do_not_depend_on_force_uncertainty():
    acq = _make_acq(autotune=True)
    low_std = _make_evaluation(2.0, force_mean=0.5, force_std=1.0e-9)
    high_std = _make_evaluation(2.0, force_mean=0.5, force_std=1.0e9)
    assert acq._refine_steps_from_cubic([low_std])[0] == pytest.approx(
        acq._refine_steps_from_cubic([high_std])[0]
    )


# --- end-to-end: _mode_metrics cache behaviour ---


class _AnharmonicPotential1D:
    """Synthetic 1D anharmonic posterior: V(x) = 0.5*x^2 + alpha*x^3.

    Implements the small posterior surface that stencils.evaluate_linear_stencil
    expects: means(points) batched mean values + covariance_matrix(points)
    batched covariance. variance(x) is used by _build_reference_scales.
    """

    def __init__(self, alpha=0.0, noise=1.0e-6):
        self.alpha = alpha
        self.noise = noise

    def _v(self, atoms):
        x = float(np.asarray(atoms.coordinates, dtype=float)[0, 0])
        return 0.5 * x * x + self.alpha * x * x * x

    def mean(self, atoms):
        return self._v(atoms)

    def means(self, points):
        # 'points' is a sequence of Atoms instances.
        return np.array([self._v(a) for a in points], dtype=float)

    def variance(self, atoms):
        return float(self.noise)

    def covariance_matrix(self, points):
        n = len(points)
        return float(self.noise) * np.eye(n)


def _make_acq_with_synthetic_posterior(alpha=0.0, autotune=False, default_step=0.1):
    acq = _make_acq(autotune=autotune, default_step=default_step)
    acq.posterior = _AnharmonicPotential1D(alpha=alpha)
    # 1 mode aligned with atom-1 x-coord. Mass-weighted: d = 1/sqrt(m) * e_x.
    atoms = Atoms([Atom("H", 0.0, 0.0, 0.0)])
    acq.mode_directions = [np.array([1.0, 0.0, 0.0]) / math.sqrt(atoms[0].mass)]
    acq.mode_steps = np.array([default_step], dtype=float)
    return acq, atoms


def test_autotune_cache_populated_on_first_call():
    acq, atoms = _make_acq_with_synthetic_posterior(alpha=2.0, autotune=True)
    assert acq.tuned_mode_steps is None
    acq._mode_metrics(atoms)
    assert acq.tuned_mode_steps is not None
    assert len(acq.tuned_mode_steps) == 1


def test_autotune_cache_persists_across_calls():
    acq, atoms = _make_acq_with_synthetic_posterior(alpha=2.0, autotune=True)
    acq._mode_metrics(atoms)
    first_tuned = acq.tuned_mode_steps.copy()
    #Call again at same geometry -- cache should be reused, not recomputed.
    acq._mode_metrics(atoms)
    np.testing.assert_array_equal(acq.tuned_mode_steps, first_tuned)


def test_autotune_off_leaves_cache_alone():
    acq, atoms = _make_acq_with_synthetic_posterior(alpha=2.0, autotune=False)
    acq._mode_metrics(atoms)
    assert acq.tuned_mode_steps is None


def test_autotune_first_call_pays_2x_stencil_cost():
    """Counts how many times the posterior.mean is called. First autotune
    call must run the stencils twice (baseline + refined). Subsequent
    calls run them once (cache hit)."""
    acq, atoms = _make_acq_with_synthetic_posterior(alpha=1.0, autotune=True)
    counter = {"n": 0}
    original_means = acq.posterior.means

    def counting_means(points):
        counter["n"] += len(points)
        return original_means(points)

    acq.posterior.means = counting_means
    #First call: 2 passes over the stencils -> ~2x baseline cost.
    acq._mode_metrics(atoms)
    first_call_count = counter["n"]
    counter["n"] = 0
    #Second call: cache hit -> ~1x baseline cost.
    acq._mode_metrics(atoms)
    second_call_count = counter["n"]
    #The first call must be strictly more expensive than the second.
    assert first_call_count > second_call_count
    #It should be roughly 2x (allow some slack for variance).
    assert first_call_count >= 1.5 * second_call_count


def test_autotune_refined_step_differs_from_default_for_anharmonic():
    """For a strongly anharmonic system (large alpha), the refined step
    must differ from the default step -- otherwise autotune is a no-op."""
    acq, atoms = _make_acq_with_synthetic_posterior(
        alpha=10.0, autotune=True, default_step=0.05,
    )
    acq._mode_metrics(atoms)
    refined = float(acq.tuned_mode_steps[0])
    default = float(acq.mode_steps[0])
    #Refined step should be strictly different from default for an
    #anharmonic potential (cubic != 0).
    assert refined != default
    #And it should be within the configured bounds.
    assert acq.config.stencils.min_step <= refined <= acq.config.stencils.max_step
