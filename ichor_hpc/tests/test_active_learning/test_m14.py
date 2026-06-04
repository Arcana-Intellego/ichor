"""M14 tests: STOP_CHECK alpha-trend triggers + phase-B anti-overlap (d)."""
from types import SimpleNamespace

import numpy as np
import pytest

from ichor.core.atoms import Atom, Atoms
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
from ichor.hpc.active_learning.daemon.state import (
    CampaignState,
    fresh_campaign_state,
)
from ichor.hpc.active_learning.sampling.anti_overlap import (
    DedupReport,
    filter_candidates_against_training,
    min_distance_to_training,
)


def _water(offset=(0.0, 0.0, 0.0)):
    dx, dy, dz = offset
    return Atoms([
        Atom("O", dx, dy, dz),
        Atom("H", dx + 0.96, dy, dz),
        Atom("H", dx - 0.24, dy + 0.93, dz),
    ])


def _stretched_water():
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 1.20, 0.0, 0.0),
        Atom("H", -0.30, 1.20, 0.0),
    ])


# --- STOP_CHECK alpha-trend ------------------------------------------


def test_stop_check_streak_rule_fires(tmp_path):
    cfg = CampaignConfig()
    cfg.stop.alpha0_streak_threshold = 0.05
    cfg.stop.alpha0_streak_length = 3
    cfg.stop.min_iterations_before_stop = 0
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=cfg)
    state = fresh_campaign_state(max_iterations=20)
    for alpha in (0.04, 0.03, 0.02):
        state.last_acquisition_alpha0 = alpha
        state.iteration += 1
        updates = ex._inline_stop_check(state)
        state.alpha_history = updates["alpha_history"]
        state.shutdown_requested = bool(updates.get("shutdown_requested", False))
    assert state.shutdown_requested is True


def test_stop_check_streak_does_not_fire_above_threshold(tmp_path):
    cfg = CampaignConfig()
    cfg.stop.alpha0_streak_threshold = 0.05
    cfg.stop.alpha0_streak_length = 3
    cfg.stop.min_iterations_before_stop = 0
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=cfg)
    state = fresh_campaign_state(max_iterations=20)
    # One iteration is over threshold; streak broken.
    for alpha in (0.04, 0.06, 0.02):
        state.last_acquisition_alpha0 = alpha
        state.iteration += 1
        updates = ex._inline_stop_check(state)
        state.alpha_history = updates["alpha_history"]
        state.shutdown_requested = bool(updates.get("shutdown_requested", False))
    assert state.shutdown_requested is False


def test_stop_check_high_plateau_does_not_converge(tmp_path):
    # A50: a flat-but-HIGH alpha is the loop stuck at a bad level, NOT converged. the old code
    # treated any plateau as convergence and shut down here; it must not now.
    cfg = CampaignConfig()
    cfg.stop.alpha0_streak_threshold = 1.0e-9   # disable the streak rule, isolate the plateau rule
    cfg.stop.alpha0_streak_length = 100
    cfg.stop.rel_alpha_improvement_min = 0.05
    cfg.stop.rel_alpha_improvement_window = 3
    cfg.stop.min_iterations_before_stop = 0
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=cfg)
    state = fresh_campaign_state(max_iterations=20)
    # each step changes ~1% (flat) but the level stays ~1.0 (high).
    for alpha in (1.0, 0.99, 0.98, 0.97):
        state.last_acquisition_alpha0 = alpha
        state.iteration += 1
        updates = ex._inline_stop_check(state)
        state.alpha_history = updates["alpha_history"]
        state.shutdown_requested = bool(updates.get("shutdown_requested", False))
    assert state.shutdown_requested is False


def test_stop_check_low_plateau_converges(tmp_path):
    # the plateau rule SHOULD fire when the alpha is both flat AND low.
    cfg = CampaignConfig()
    cfg.stop.alpha0_streak_threshold = 0.05     # the "low" bar the plateau rule reuses
    cfg.stop.alpha0_streak_length = 100         # high, so the streak rule alone never fires here
    cfg.stop.rel_alpha_improvement_min = 0.05
    cfg.stop.rel_alpha_improvement_window = 3
    cfg.stop.min_iterations_before_stop = 0
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=cfg)
    state = fresh_campaign_state(max_iterations=20)
    # flat (~1% steps) AND low (well under 0.05).
    for alpha in (0.010, 0.0099, 0.0098, 0.0097):
        state.last_acquisition_alpha0 = alpha
        state.iteration += 1
        updates = ex._inline_stop_check(state)
        state.alpha_history = updates["alpha_history"]
        state.shutdown_requested = bool(updates.get("shutdown_requested", False))
    assert state.shutdown_requested is True


def test_stop_check_rising_alpha_does_not_converge(tmp_path):
    # A50: a rising alpha (model getting worse / new hard region) must never trigger a stop, even
    # with the streak rule disabled. the old clamped rule read a rise as zero-improvement = stalled.
    cfg = CampaignConfig()
    cfg.stop.alpha0_streak_threshold = 1.0e-9   # disable the streak rule
    cfg.stop.alpha0_streak_length = 100
    cfg.stop.rel_alpha_improvement_min = 0.05
    cfg.stop.rel_alpha_improvement_window = 3
    cfg.stop.min_iterations_before_stop = 0
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=cfg)
    state = fresh_campaign_state(max_iterations=20)
    for alpha in (0.5, 0.6, 0.7, 0.8):  # rising ~20% a step
        state.last_acquisition_alpha0 = alpha
        state.iteration += 1
        updates = ex._inline_stop_check(state)
        state.alpha_history = updates["alpha_history"]
        state.shutdown_requested = bool(updates.get("shutdown_requested", False))
    assert state.shutdown_requested is False


def test_stop_check_min_iterations_before_stop_blocks_early(tmp_path):
    cfg = CampaignConfig()
    cfg.stop.alpha0_streak_threshold = 0.05
    cfg.stop.alpha0_streak_length = 2
    cfg.stop.min_iterations_before_stop = 50   # never reached in this test
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=cfg)
    state = fresh_campaign_state(max_iterations=10)
    for alpha in (0.01, 0.01, 0.01):
        state.last_acquisition_alpha0 = alpha
        state.iteration += 1
        updates = ex._inline_stop_check(state)
        state.alpha_history = updates["alpha_history"]
        state.shutdown_requested = bool(updates.get("shutdown_requested", False))
    assert state.shutdown_requested is False


def test_alpha_history_round_trips_in_state():
    st = CampaignState()
    assert st.alpha_history == []
    st.alpha_history = [0.1, 0.05, 0.02]
    payload = st.to_dict()
    st2 = CampaignState.from_dict(payload)
    assert st2.alpha_history == [0.1, 0.05, 0.02]


def test_stop_check_caps_history_length(tmp_path):
    cfg = CampaignConfig()
    cfg.stop.alpha0_streak_length = 3
    cfg.stop.rel_alpha_improvement_window = 2
    cfg.stop.min_iterations_before_stop = 1000
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=cfg)
    state = fresh_campaign_state(max_iterations=100)
    for alpha in [0.5] * 20:
        state.last_acquisition_alpha0 = alpha
        state.iteration += 1
        updates = ex._inline_stop_check(state)
        state.alpha_history = updates["alpha_history"]
    # cap = max(3, 2) + 1 = 4
    assert len(state.alpha_history) <= 4


# --- Phase-B anti-overlap (case d) ------------------------------------


def test_min_distance_to_empty_training_is_infinity():
    w = _water()
    assert min_distance_to_training(w, []) == float("inf")


def test_min_distance_picks_closest():
    candidate = _water((0.0, 0.0, 0.0))
    training = [
        _water((10.0, 0.0, 0.0)),   # far
        _water((0.0, 0.0, 0.0)),    # identical -> distance 0
        _water((5.0, 0.0, 0.0)),
    ]
    d = min_distance_to_training(candidate, training)
    assert d == pytest.approx(0.0, abs=1.0e-9)


def test_filter_keeps_far_candidates_drops_close_ones():
    training = [_water((0.0, 0.0, 0.0))]
    candidates = [
        _water((0.0, 0.0, 0.0)),    # dist 0 -> drop
        _stretched_water(),         # different shape -> keep
        _water((0.0001, 0.0, 0.0)), # nearly identical -> drop
    ]
    rep = filter_candidates_against_training(
        candidates, training, min_separation=0.1,
    )
    assert isinstance(rep, DedupReport)
    assert rep.kept_indices == (1,)
    assert set(rep.dropped_indices) == {0, 2}
    assert rep.n_kept == 1
    assert rep.n_dropped == 2


def test_filter_with_zero_training_keeps_everything():
    candidates = [_water((0.0, 0.0, 0.0)), _water((1.0, 0.0, 0.0))]
    rep = filter_candidates_against_training(candidates, [], min_separation=1.0)
    assert rep.kept_indices == (0, 1)
    assert rep.dropped_indices == ()


def test_filter_negative_min_separation_raises():
    with pytest.raises(ValueError):
        filter_candidates_against_training(
            [_water()], [_water()], min_separation=-1.0,
        )


def test_filter_min_separation_zero_keeps_everything():
    """A boundary case: zero min_separation never drops, even identical
    geometries (their distance is also 0 but the rule is `d < min`)."""
    training = [_water()]
    candidates = [_water((0.0, 0.0, 0.0))]
    rep = filter_candidates_against_training(
        candidates, training, min_separation=0.0,
    )
    assert rep.kept_indices == (0,)
