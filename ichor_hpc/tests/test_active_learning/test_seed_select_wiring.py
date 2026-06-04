"""M11 wiring tests: SEED_SELECT picks against the trajectory pool with
forbidden frame_ids excluded, and the post-ARIADNE anti-overlap flag fires
when the synthetic descent stays inside / leaves the trust region.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.dry_run_executor import (
    ANTI_OVERLAP_MAX_WHITENED_DISTANCE,
    ANTI_OVERLAP_MIN_WHITENED_DISTANCE,
    DryRunPhaseExecutor,
)
from ichor.hpc.active_learning.daemon.live_executor import LiveBackendsPhaseExecutor
from ichor.hpc.active_learning.daemon.phase_executor import BackendSubmissionError
from ichor.hpc.active_learning.daemon.state import CampaignPhase
from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.versioning.provenance import (
    append_recent_seeds,
    append_to_index,
    load_recent_seed_frame_ids,
    load_recent_seeds_payload,
    read_provenance,
)


FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "water_tetramer.xyz"
)


def _read_journal_events(campaign_dir):
    journal_path = (
        campaign_dir / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    )
    if not journal_path.is_file():
        return []
    return [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- (case a) training-set + (case b) recent-seeds cooldown wiring -----


def test_inline_seed_select_with_no_pool_falls_back_to_placeholder(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 3
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    ex.submit_or_run(SimpleNamespace(iteration=0), CampaignPhase.SEED_SELECT)
    seeds_path = cd / "7_ACTIVE_LEARNING" / "iteration-0000" / "seeds.xyz"
    assert seeds_path.is_file()
    assert "pool_available=False" in seeds_path.read_text(encoding="utf-8")
    assert not (cd / "7_ACTIVE_LEARNING" / "iteration-0000"
                / "seeds_picked.json").is_file()
    events = _read_journal_events(cd)
    picked = [e for e in events if e.get("event") == "seed_selected"]
    assert picked
    assert picked[-1]["pool_available"] is False
    assert picked[-1]["n_picked"] == 0


def test_inline_seed_select_with_pool_writes_seeds_picked_json(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 4
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    ex.submit_or_run(SimpleNamespace(iteration=0), CampaignPhase.SEED_SELECT)
    seeds_picked = cd / "7_ACTIVE_LEARNING" / "iteration-0000" / "seeds_picked.json"
    assert seeds_picked.is_file()
    payload = json.loads(seeds_picked.read_text(encoding="utf-8"))
    assert payload["iteration"] == 0
    assert payload["n_picked"] == 4
    assert len(payload["frame_ids"]) == 4
    pool = TrajectoryPool.load(cd)
    valid = set(pool.frame_ids())
    for fid in payload["frame_ids"]:
        assert fid in valid


def test_inline_seed_select_skips_training_pool_frame_ids(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 3
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    pool = TrajectoryPool.load(cd)
    forbidden = list(pool.frame_ids())[:5]
    for fid in forbidden:
        append_to_index(
            cd, iteration=-1,
            pointdir_name=f"POINT_dummy_{fid}.pointdir",
            seed_frame_id=int(fid),
        )
    ex.submit_or_run(SimpleNamespace(iteration=0), CampaignPhase.SEED_SELECT)
    seeds_picked = cd / "7_ACTIVE_LEARNING" / "iteration-0000" / "seeds_picked.json"
    payload = json.loads(seeds_picked.read_text(encoding="utf-8"))
    for fid in payload["frame_ids"]:
        assert fid not in forbidden
    events = _read_journal_events(cd)
    picked = [e for e in events if e.get("event") == "seed_selected"]
    assert picked[-1]["forbidden_set_size"] == 5


def test_inline_seed_select_skips_recent_cooldown_frames(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 2
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    pool = TrajectoryPool.load(cd)
    recent = list(pool.frame_ids())[:3]
    append_recent_seeds(cd, iteration=-1, frame_ids=recent, cooldown=3)
    ex.submit_or_run(SimpleNamespace(iteration=0), CampaignPhase.SEED_SELECT)
    payload = json.loads(
        (cd / "7_ACTIVE_LEARNING" / "iteration-0000" / "seeds_picked.json")
        .read_text(encoding="utf-8")
    )
    for fid in payload["frame_ids"]:
        assert fid not in recent
    cache = load_recent_seeds_payload(cd)
    assert cache["history"][-1]["iteration"] == 0


def test_inline_seed_select_updates_recent_seeds_cache_on_each_run(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 2
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    for it in range(4):
        ex.submit_or_run(SimpleNamespace(iteration=it), CampaignPhase.SEED_SELECT)
    cache = load_recent_seeds_payload(cd)
    iters = [e["iteration"] for e in cache["history"]]
    assert iters == [1, 2, 3]


def test_inline_seed_select_uses_configured_recent_seed_cooldown(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 1
    cfg.anti_overlap.recent_seeds_cooldown = 1
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    ex.submit_or_run(SimpleNamespace(iteration=0), CampaignPhase.SEED_SELECT)
    ex.submit_or_run(SimpleNamespace(iteration=1), CampaignPhase.SEED_SELECT)
    cache = load_recent_seeds_payload(cd)
    assert cache["cooldown"] == 1
    assert [e["iteration"] for e in cache["history"]] == [1]


def test_seed_select_reentry_repairs_recent_seed_history(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 2
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    state = SimpleNamespace(iteration=0)
    ex.submit_or_run(state, CampaignPhase.SEED_SELECT)
    recent_path = cd / ".DATA" / "ACTIVE_LEARNING" / "recent_seeds.json"
    recent_path.unlink()

    ex.submit_or_run(state, CampaignPhase.SEED_SELECT)
    cache = load_recent_seeds_payload(cd)
    assert [e["iteration"] for e in cache["history"]] == [0]


def test_live_seed_selection_requires_models_by_default(tmp_path):
    cfg = CampaignConfig()
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )
    state = SimpleNamespace(iteration=0, models_version=0)
    with pytest.raises(BackendSubmissionError, match="committed models"):
        ex._seed_selection_posterior(state, [object()])


def test_live_seed_selection_uniform_fallback_requires_explicit_config(tmp_path):
    cfg = CampaignConfig()
    cfg.acquisition.allow_uniform_posterior_fallback = True
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )
    state = SimpleNamespace(iteration=0, models_version=0)
    posterior = ex._seed_selection_posterior(state, [object()])
    assert posterior.variance(object()) == 1.0


def test_seed_pool_exhaustion_halts_when_no_eligible_frames(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 1
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    pool = TrajectoryPool.load(cd)
    append_recent_seeds(cd, iteration=-1, frame_ids=list(pool.frame_ids()), cooldown=3)
    with pytest.raises(BackendSubmissionError, match="seed_pool_exhausted"):
        ex.submit_or_run(SimpleNamespace(iteration=0), CampaignPhase.SEED_SELECT)


def test_partial_seed_batch_halts_instead_of_silent_smaller_batch(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    pool = TrajectoryPool.load(cd)
    cfg.seed_selection.n_seeds_per_iteration = pool.n_frames() + 1
    with pytest.raises(BackendSubmissionError, match="only .* eligible"):
        ex.submit_or_run(SimpleNamespace(iteration=0), CampaignPhase.SEED_SELECT)


# --- (case c) post-ARIADNE anti-overlap flag ---------------------------


def test_anti_overlap_passes_when_distance_within_band(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 2
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    state = SimpleNamespace(iteration=0)
    ex.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    pool_dir = cd / "7_ACTIVE_LEARNING" / "iteration-0000" / "pool"
    seed_dirs = sorted(d for d in pool_dir.iterdir() if d.is_dir())
    assert seed_dirs
    any_flagged = False
    for sd in seed_dirs:
        data = read_provenance(sd)
        if data["anti_overlap"]["flag"] is not None:
            any_flagged = True
    events = _read_journal_events(cd)
    flagged_events = [e for e in events if e.get("event") == "anti_overlap_flagged"]
    assert (len(flagged_events) > 0) == any_flagged


def test_synthetic_whitened_distance_returns_none_when_alpha_missing(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    R = SimpleNamespace(alpha_initial=None, alpha_final=None)
    assert ex._synthetic_whitened_distance(R) is None
    R = SimpleNamespace(alpha_initial=0.0, alpha_final=None)
    assert ex._synthetic_whitened_distance(R) is None
    R = SimpleNamespace(alpha_initial=0.5, alpha_final=0.8)
    assert ex._synthetic_whitened_distance(R) == pytest.approx(0.3)


def test_anti_overlap_thresholds_have_documented_defaults():
    assert ANTI_OVERLAP_MIN_WHITENED_DISTANCE == 0.01
    assert ANTI_OVERLAP_MAX_WHITENED_DISTANCE == 10.0


def test_post_ariadne_uses_picked_seed_frame_ids_when_present(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 3
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    state = SimpleNamespace(iteration=0, campaign_uid="uid")
    ex.submit_or_run(state, CampaignPhase.SEED_SELECT)
    picked = json.loads(
        (cd / "7_ACTIVE_LEARNING" / "iteration-0000" / "seeds_picked.json")
        .read_text(encoding="utf-8")
    )
    ex.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    pool_dir = cd / "7_ACTIVE_LEARNING" / "iteration-0000" / "pool"
    seed_dirs = sorted(d for d in pool_dir.iterdir() if d.is_dir())
    expected = picked["frame_ids"][:len(seed_dirs)]
    for sd, exp_fid in zip(seed_dirs, expected):
        data = read_provenance(sd)
        assert data["seed"]["frame_id"] == exp_fid
