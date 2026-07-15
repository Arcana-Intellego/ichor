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
    DryRunPhaseExecutor,
)
from ichor.hpc.active_learning.daemon.live_executor import LiveBackendsPhaseExecutor
from ichor.hpc.active_learning.daemon.phase_executor import BackendSubmissionError
from ichor.hpc.active_learning.daemon.state import CampaignPhase
from ichor.hpc.active_learning.daemon.submission_intent import (
    write_pre_submit_intent,
)
from ichor.hpc.active_learning.daemon.config_lock import (
    canonical_config,
    config_fingerprint,
)
from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.handoff_manifests import (
    ariadne_task_map_path,
    load_seeds_picked,
    read_seed_selection_diagnostics,
    write_seed_selection_diagnostics,
)
from ichor.hpc.active_learning.layout import (
    active_iteration_dir,
    active_seed_selection_dir,
    ariadne_seeds_dir,
)
from ichor.hpc.active_learning.versioning.provenance import (
    append_recent_seeds,
    append_to_index,
    load_recent_seed_frame_ids,
    load_recent_seeds_payload,
    load_training_seed_frame_ids,
    read_provenance,
)
from ichor_hpc.tests.quantum_test_support import prepare_dry_submitted_phase


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


def _active_state(iteration=1):
    return SimpleNamespace(
        iteration=int(iteration),
        campaign_uid="seed-wiring-test",
        reference_data_version=0,
        models_version=0,
        replacement_round=0,
    )


def _select(ex, *, iteration=1):
    models_root = ex.campaign_dir / "TRAINED_MODELS" / "iteration-000000"
    if not models_root.is_dir():
        bootstrap_state = SimpleNamespace(
            iteration=0,
            campaign_uid="seed-wiring-test",
            reference_data_version=-1,
            models_version=-1,
            replacement_round=0,
        )
        ex._post_phase_a_diversity(bootstrap_state)
        prepare_dry_submitted_phase(
            ex, bootstrap_state, CampaignPhase.INITIAL_GAUSSIAN
        )
        ex._post_initial_gaussian(bootstrap_state)
        prepare_dry_submitted_phase(
            ex, bootstrap_state, CampaignPhase.INITIAL_AIMALL
        )
        ex._post_initial_aimall(bootstrap_state)
        ex._post_initial_ferebus(bootstrap_state)
    state = _active_state(iteration)
    return state, ex.submit_or_run(state, CampaignPhase.SEED_SELECT)


def _stage_ariadne_intent(ex, state):
    write_pre_submit_intent(
        ex.campaign_dir,
        campaign_uid=str(state.campaign_uid),
        phase_name=CampaignPhase.ARIADNE_ARRAY.value,
        iteration=int(state.iteration),
        decision_contract={
            "failure_threshold_fraction": float(
                ex.config.runtime.failure_threshold_fraction
            ),
            "config_sha256": config_fingerprint(canonical_config(ex.config)),
        },
    )


# --- (case a) training-set + (case b) recent-seeds cooldown wiring -----


def test_dry_seed_select_bootstraps_a_missing_trajectory_pool(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 3
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    _state, _result = _select(ex)
    iter_dir = active_iteration_dir(cd, 1)
    seeds_path = active_seed_selection_dir(iter_dir) / "seeds.xyz"
    assert seeds_path.is_file()
    assert (active_seed_selection_dir(iter_dir) / "SELECTION.json").is_file()
    assert TrajectoryPool.load(cd).n_frames() >= 3
    events = _read_journal_events(cd)
    picked = [e for e in events if e.get("event") == "seed_selected"]
    assert picked
    assert picked[-1]["pool_available"] is True
    assert picked[-1]["n_picked"] == 3


def test_seed_selection_permanently_excludes_pool_matched_bootstrap_geometry(
    tmp_path,
):
    cd = tmp_path / "campaign"
    cd.mkdir()
    pool = TrajectoryPool.import_from(FIXTURE, cd)
    frame = pool.frame(0)
    bootstrap = cd / "bootstrap"
    bootstrap.mkdir()
    lines = [str(len(frame)), "operator bootstrap frame 0"]
    lines.extend(
        str(atom.type)
        + " "
        + format(float(atom.x), ".16g")
        + " "
        + format(float(atom.y), ".16g")
        + " "
        + format(float(atom.z), ".16g")
        for atom in frame
    )
    (bootstrap / "training_set_bootstrap.xyz").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    cfg = CampaignConfig()
    cfg.campaign.custom_bootstrap = True
    cfg.seed_selection.n_seeds_per_iteration = 3
    cfg.seed_selection.exclude_committed_seed_frames = False
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)

    _state, _result = _select(ex)

    payload = load_seeds_picked(
        active_iteration_dir(cd, 1),
        expected_iteration=1,
    )
    assert 0 not in payload["frame_ids"]
    assert payload["bootstrap_forbidden_set_size"] == 1
    assert payload["diagnostics"]["bootstrap_forbidden_set_size"] == 1


def test_inline_seed_select_with_pool_writes_seeds_picked_json(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 4
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    _state, _result = _select(ex)
    seeds_picked = active_seed_selection_dir(
        active_iteration_dir(cd, 1)
    ) / "SELECTION.json"
    assert seeds_picked.is_file()
    payload = json.loads(seeds_picked.read_text(encoding="utf-8"))
    assert payload["iteration"] == 1
    assert payload["n_picked"] == 4
    assert len(payload["frame_ids"]) == 4
    pool = TrajectoryPool.load(cd)
    valid = set(pool.frame_ids())
    for fid in payload["frame_ids"]:
        assert fid in valid


def test_d_optimal_seed_select_writes_diagnostics_manifest(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 4
    cfg.seed_selection.bulk_fraction = 0.0
    cfg.seed_selection.strategy = "d_optimal"
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)

    _state, _result = _select(ex)

    iter_dir = active_iteration_dir(cd, 1)
    picked = load_seeds_picked(iter_dir, expected_iteration=1)
    assert all(
        rec["selection_origin"] == "d_optimal"
        for rec in picked["seed_records"]
    )
    assert picked["d_optimal_indices"] == picked["indices"]
    assert all("d_optimal_gain" in rec for rec in picked["seed_records"])

    diagnostics = read_seed_selection_diagnostics(iter_dir, expected_iteration=1)
    assert diagnostics["strategy"] == "d_optimal"
    assert diagnostics["n_picked"] == 4
    assert len(diagnostics["selected"]) == 4


def test_seed_selection_diagnostics_manifest_roundtrips(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 1
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    _state, _result = _select(ex)
    iter_dir = active_iteration_dir(cd, 1)
    path = write_seed_selection_diagnostics(
        iter_dir,
        {
            "iteration": 1,
            "strategy": "d_optimal",
            "n_picked": 1,
            "selected": [
                {
                    "seed_id": 1,
                    "selection_origin": "d_optimal",
                    "d_optimal_gain": 1.0,
                }
            ],
        },
    )

    assert path.name == "SELECTION.json"
    data = read_seed_selection_diagnostics(iter_dir, expected_iteration=1)
    assert data["selected"][0]["selection_origin"] == "d_optimal"


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
            cd, iteration=0,
            pointdir_name=f"POINT_dummy_{fid}.pointdir",
            seed_frame_id=int(fid),
            trajectory_sha256=pool.sha256,
        )
    _state, _result = _select(ex)
    seeds_picked = active_seed_selection_dir(
        active_iteration_dir(cd, 1)
    ) / "SELECTION.json"
    payload = json.loads(seeds_picked.read_text(encoding="utf-8"))
    for fid in payload["frame_ids"]:
        assert fid not in forbidden
    events = _read_journal_events(cd)
    picked = [e for e in events if e.get("event") == "seed_selected"]
    expected_forbidden = load_training_seed_frame_ids(
        cd,
        reference_data_dir=cd / "QM_REFERENCE_DATA",
        expected_trajectory_sha256=pool.sha256,
    )
    assert set(forbidden).issubset(expected_forbidden)
    assert picked[-1]["forbidden_set_size"] == len(expected_forbidden)


def test_seed_frame_index_batch_upsert_is_idempotent_and_conflict_safe(tmp_path):
    from ichor.hpc.active_learning.versioning.provenance import (
        ProvenanceError,
        load_index,
        upsert_index_records,
    )

    campaign = tmp_path / "campaign"
    record = {
        "iteration": 1,
        "pointdir_name": "POINT_0001.pointdir",
        "seed_frame_id": 17,
        "trajectory_sha256": "a" * 64,
    }
    upsert_index_records(campaign, records=[record, dict(record)])
    upsert_index_records(campaign, records=[dict(record)])

    assert load_index(campaign)["records"] == [record]
    with pytest.raises(ProvenanceError, match="conflicting seed-frame index record"):
        upsert_index_records(
            campaign,
            records=[dict(record, seed_frame_id=18)],
        )


def test_inline_seed_select_skips_recent_cooldown_frames(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 2
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    pool = TrajectoryPool.load(cd)
    recent = list(pool.frame_ids())[:3]
    append_recent_seeds(
        cd,
        iteration=0,
        frame_ids=recent,
        trajectory_sha256=pool.sha256,
        cooldown=3,
    )
    _state, _result = _select(ex)
    payload = json.loads(
        (
            active_seed_selection_dir(active_iteration_dir(cd, 1))
            / "SELECTION.json"
        )
        .read_text(encoding="utf-8")
    )
    for fid in payload["frame_ids"]:
        assert fid not in recent
    cache = load_recent_seeds_payload(cd)
    assert cache["history"][-1]["iteration"] == 1


def test_inline_seed_select_updates_recent_seeds_cache_on_each_run(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 2
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    for it in range(1, 5):
        _select(ex, iteration=it)
    cache = load_recent_seeds_payload(cd)
    iters = [e["iteration"] for e in cache["history"]]
    assert iters == [4]


def test_inline_seed_select_uses_configured_recent_seed_cooldown(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 1
    cfg.seed_selection.recent_seed_cooldown_iterations = 1
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    _select(ex, iteration=1)
    _select(ex, iteration=2)
    cache = load_recent_seeds_payload(cd)
    assert cache["cooldown"] == 1
    assert [e["iteration"] for e in cache["history"]] == [2]


def test_seed_select_reentry_repairs_recent_seed_history(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 2
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    state, _result = _select(ex)
    recent_path = cd / ".DATA" / "ACTIVE_LEARNING" / "recent_seeds.json"
    recent_path.unlink()

    ex.submit_or_run(state, CampaignPhase.SEED_SELECT)
    cache = load_recent_seeds_payload(cd)
    assert [e["iteration"] for e in cache["history"]] == [1]


def test_seed_select_reentry_repairs_incomplete_published_handoff(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 2
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    state, _result = _select(ex)
    iter_dir = active_iteration_dir(cd, 1)
    task_map = ariadne_task_map_path(iter_dir)
    seeds_xyz = active_seed_selection_dir(iter_dir) / "seeds.xyz"
    task_map.unlink()
    seeds_xyz.unlink()

    ex.submit_or_run(state, CampaignPhase.SEED_SELECT)

    assert task_map.is_file()
    assert seeds_xyz.is_file()
    assert "active iteration 1 seed 1" in seeds_xyz.read_text(encoding="utf-8")


def test_live_seed_selection_requires_models_by_default(tmp_path):
    cfg = CampaignConfig()
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )
    state = _active_state()
    with pytest.raises(BackendSubmissionError, match="committed models"):
        ex._seed_selection_posterior(state, [object()])


def test_live_seed_select_requires_imported_trajectory_pool(tmp_path):
    cfg = CampaignConfig()
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )
    state = _active_state()

    with pytest.raises(BackendSubmissionError, match="imported trajectory pool"):
        ex.submit_or_run(state, CampaignPhase.SEED_SELECT)


def test_live_seed_selection_never_uses_uniform_posterior_fallback(tmp_path):
    cfg = CampaignConfig()
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )
    state = _active_state()
    with pytest.raises(BackendSubmissionError, match="committed models"):
        ex._seed_selection_posterior(state, [object()])


def test_seed_pool_exhaustion_halts_when_no_eligible_frames(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 1
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    pool = TrajectoryPool.load(cd)
    append_recent_seeds(
        cd,
        iteration=0,
        frame_ids=list(pool.frame_ids()),
        trajectory_sha256=pool.sha256,
        cooldown=3,
    )
    with pytest.raises(BackendSubmissionError, match="seed_pool_exhausted"):
        _select(ex)


def test_partial_seed_batch_halts_instead_of_silent_smaller_batch(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    pool = TrajectoryPool.load(cd)
    cfg.seed_selection.n_seeds_per_iteration = pool.n_frames() + 1
    with pytest.raises(BackendSubmissionError, match="only .* eligible"):
        _select(ex)


# --- (case c) post-ARIADNE anti-overlap flag ---------------------------


def test_anti_overlap_passes_when_distance_within_band(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 2
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    state, _result = _select(ex)
    _stage_ariadne_intent(ex, state)
    ex.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    pool_dir = ariadne_seeds_dir(active_iteration_dir(cd, 1))
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


def test_hidden_anti_overlap_bounds_cannot_override_sampling_policy(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 2
    cfg.anti_overlap.min_post_ariadne_whitened_distance = 100.0
    cfg.anti_overlap.max_post_ariadne_whitened_distance = 200.0
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    state, _result = _select(ex)
    _stage_ariadne_intent(ex, state)
    ex.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])

    pool_dir = ariadne_seeds_dir(active_iteration_dir(cd, 1))
    seed_dirs = sorted(d for d in pool_dir.iterdir() if d.is_dir())
    assert seed_dirs
    for sd in seed_dirs:
        data = read_provenance(sd)
        assert data["anti_overlap"]["flag"] is None


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
    c = CampaignConfig()
    assert c.anti_overlap.min_post_ariadne_whitened_distance == 0.01
    assert c.anti_overlap.max_post_ariadne_whitened_distance == 10.0


def test_post_ariadne_uses_picked_seed_frame_ids_when_present(tmp_path):
    cd = tmp_path / "campaign"
    cfg = CampaignConfig()
    cfg.seed_selection.n_seeds_per_iteration = 3
    ex = DryRunPhaseExecutor(campaign_dir=cd, config=cfg)
    TrajectoryPool.import_from(FIXTURE, cd)
    state, _result = _select(ex)
    picked = load_seeds_picked(active_iteration_dir(cd, 1), expected_iteration=1)
    _stage_ariadne_intent(ex, state)
    ex.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    pool_dir = ariadne_seeds_dir(active_iteration_dir(cd, 1))
    seed_dirs = sorted(d for d in pool_dir.iterdir() if d.is_dir())
    expected = picked["frame_ids"][:len(seed_dirs)]
    for sd, exp_fid in zip(seed_dirs, expected):
        data = read_provenance(sd)
        assert data["seed"]["frame_id"] == exp_fid
