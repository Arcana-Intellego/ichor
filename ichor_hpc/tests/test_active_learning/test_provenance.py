"""Tests for ichor.hpc.active_learning.versioning.provenance (M10).

Two layers: unit tests for the per-pointdir sidecar (write / read / enrich)
and the flat index file (ensure / append / load / training_seed_frame_ids),
plus one end-to-end integration test that drives the dry-run executor
through 2 iterations and verifies the full provenance flow including
journal events.
"""
import json
from pathlib import Path

import pytest

from ichor.hpc.active_learning.versioning.provenance import (
    INDEX_SCHEMA_VERSION,
    PROVENANCE_FILENAME,
    PROVENANCE_SCHEMA_VERSION,
    ProvenanceError,
    SEED_FRAME_ID_INDEX_FILENAME,
    append_to_index,
    ensure_index,
    enrich_with_anti_overlap,
    enrich_with_ariadne,
    enrich_with_point_allocation,
    enrich_with_phase_b,
    load_index,
    load_training_seed_frame_ids,
    repair_index_from_committed_pointdirs,
    read_provenance,
    write_seed_provenance,
)


# --- per-pointdir sidecar ---------------------------------------------


def test_write_seed_provenance_creates_full_schema_skeleton(tmp_path):
    pdir = tmp_path / "POINT_0000.pointdir"
    write_seed_provenance(
        pdir,
        campaign_uid="abc-123",
        iteration=5,
        trajectory_sha256="deadbeef",
        seed_frame_id=42,
        seed_selection_origin="variance",
        seed_variance_at_selection=1.23e-4,
        subspace_neighbour_frame_ids=[42, 41, 99],
        subspace_dimension=3,
        subspace_eigenvalues=[1.0, 0.5, 0.2],
        mode_weighting_policy="inverse_frequency",
    )
    data = read_provenance(pdir)
    assert data["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert data["campaign_uid"] == "abc-123"
    assert data["iteration"] == 5
    assert data["trajectory_sha256"] == "deadbeef"
    assert data["seed"]["frame_id"] == 42
    assert data["seed"]["selection_origin"] == "variance"
    assert data["subspace"]["neighbour_frame_ids"] == [42, 41, 99]
    assert data["subspace"]["mode_weighting_policy"] == "inverse_frequency"
    # The non-yet-enriched blocks are nulled out.
    assert data["ariadne"] is None
    assert data["anti_overlap"] is None
    assert data["phase_b"] is None


def test_enrich_with_ariadne_preserves_earlier_blocks(tmp_path):
    pdir = tmp_path / "p"
    write_seed_provenance(
        pdir, campaign_uid="X", iteration=1, trajectory_sha256="",
        seed_frame_id=7, seed_selection_origin="bulk",
        seed_variance_at_selection=None,
        subspace_neighbour_frame_ids=[7], subspace_dimension=1,
        subspace_eigenvalues=[1.0],
    )
    enrich_with_ariadne(
        pdir, alpha_initial=0.1, alpha_final=2.0, n_evaluations=50,
        fell_back_to_ds=False, wall_seconds=12.5, return_code=0,
    )
    data = read_provenance(pdir)
    assert data["ariadne"]["alpha_final"] == 2.0
    assert data["ariadne"]["return_code"] == 0
    # Earlier seed block survives.
    assert data["seed"]["frame_id"] == 7
    assert data["subspace"]["dimension"] == 1


def test_enrich_with_anti_overlap_records_flag(tmp_path):
    pdir = tmp_path / "p"
    write_seed_provenance(
        pdir, campaign_uid="X", iteration=1, trajectory_sha256="",
        seed_frame_id=None, seed_selection_origin="bulk",
        seed_variance_at_selection=None,
        subspace_neighbour_frame_ids=[], subspace_dimension=0,
        subspace_eigenvalues=[],
    )
    enrich_with_anti_overlap(pdir,
        min_whitened_distance_to_training=0.005,
        passed=False,
        flag="moved_too_little",
    )
    data = read_provenance(pdir)
    assert data["anti_overlap"]["passed"] is False
    assert data["anti_overlap"]["flag"] == "moved_too_little"
    assert data["anti_overlap"]["min_whitened_distance_to_training"] == pytest.approx(0.005)


def test_enrich_with_phase_b_records_diversity_rank(tmp_path):
    pdir = tmp_path / "p"
    write_seed_provenance(
        pdir, campaign_uid="X", iteration=1, trajectory_sha256="",
        seed_frame_id=None, seed_selection_origin="bulk",
        seed_variance_at_selection=None,
        subspace_neighbour_frame_ids=[], subspace_dimension=0,
        subspace_eigenvalues=[],
    )
    enrich_with_phase_b(pdir,
        selected_after_fps=True,
        diversity_rank=4,
        descriptor_used="hybrid_alf_rmsd",
    )
    data = read_provenance(pdir)
    assert data["phase_b"]["diversity_rank"] == 4
    assert data["phase_b"]["descriptor_used"] == "hybrid_alf_rmsd"


def test_read_provenance_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_provenance(tmp_path / "no_such_dir")


def test_read_provenance_raises_on_schema_mismatch(tmp_path):
    pdir = tmp_path / "p"
    pdir.mkdir()
    (pdir / PROVENANCE_FILENAME).write_text(
        json.dumps({"schema_version": 999}),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="schema_version"):
        read_provenance(pdir)


def test_enrich_raises_when_no_initial_provenance(tmp_path):
    pdir = tmp_path / "p"
    pdir.mkdir()
    with pytest.raises(FileNotFoundError):
        enrich_with_ariadne(
            pdir, alpha_initial=0.0, alpha_final=0.0,
            n_evaluations=0, fell_back_to_ds=False, wall_seconds=0.0,
        )


# --- flat index file ---------------------------------------------------


def test_ensure_index_creates_empty_payload(tmp_path):
    p = ensure_index(tmp_path)
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema_version"] == INDEX_SCHEMA_VERSION
    assert data["records"] == []


def test_ensure_index_is_idempotent(tmp_path):
    p1 = ensure_index(tmp_path)
    append_to_index(tmp_path, iteration=0, pointdir_name="POINT_0000.pointdir", seed_frame_id=7)
    p2 = ensure_index(tmp_path)
    assert p1 == p2
    data = json.loads(p2.read_text(encoding="utf-8"))
    assert len(data["records"]) == 1
    assert data["records"][0]["seed_frame_id"] == 7


def test_append_to_index_grows_records(tmp_path):
    for i in range(3):
        append_to_index(
            tmp_path, iteration=i,
            pointdir_name=f"POINT_{i:04d}.pointdir",
            seed_frame_id=10 * i,
        )
    data = load_index(tmp_path)
    assert len(data["records"]) == 3
    assert [r["seed_frame_id"] for r in data["records"]] == [0, 10, 20]
    assert [r["pointdir_name"] for r in data["records"]] == [
        "POINT_0000.pointdir", "POINT_0001.pointdir", "POINT_0002.pointdir",
    ]


def test_append_to_index_accepts_none_frame_id(tmp_path):
    append_to_index(
        tmp_path, iteration=0,
        pointdir_name="POINT_0000.pointdir",
        seed_frame_id=None,
    )
    data = load_index(tmp_path)
    assert data["records"][0]["seed_frame_id"] is None


def test_load_index_returns_empty_when_file_missing(tmp_path):
    data = load_index(tmp_path)
    assert data["schema_version"] == INDEX_SCHEMA_VERSION
    assert data["records"] == []


def test_load_index_raises_on_corrupted_json(tmp_path):
    idx_dir = tmp_path / ".DATA" / "ACTIVE_LEARNING"
    idx_dir.mkdir(parents=True)
    (idx_dir / SEED_FRAME_ID_INDEX_FILENAME).write_text("not json {", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="failed to parse"):
        load_index(tmp_path)


def test_load_index_raises_on_schema_mismatch(tmp_path):
    idx_dir = tmp_path / ".DATA" / "ACTIVE_LEARNING"
    idx_dir.mkdir(parents=True)
    (idx_dir / SEED_FRAME_ID_INDEX_FILENAME).write_text(
        json.dumps({"schema_version": 999, "records": []}),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="schema_version"):
        load_index(tmp_path)


def test_load_index_raises_when_records_not_list(tmp_path):
    idx_dir = tmp_path / ".DATA" / "ACTIVE_LEARNING"
    idx_dir.mkdir(parents=True)
    (idx_dir / SEED_FRAME_ID_INDEX_FILENAME).write_text(
        json.dumps({"schema_version": INDEX_SCHEMA_VERSION, "records": "nope"}),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="records"):
        load_index(tmp_path)


def test_load_training_seed_frame_ids_skips_none(tmp_path):
    append_to_index(tmp_path, iteration=0, pointdir_name="A", seed_frame_id=5)
    append_to_index(tmp_path, iteration=0, pointdir_name="B", seed_frame_id=None)
    append_to_index(tmp_path, iteration=1, pointdir_name="C", seed_frame_id=12)
    append_to_index(tmp_path, iteration=1, pointdir_name="D", seed_frame_id=5)
    s = load_training_seed_frame_ids(tmp_path)
    assert isinstance(s, set)
    assert s == {5, 12}


def test_load_training_seed_frame_ids_empty_for_fresh_campaign(tmp_path):
    assert load_training_seed_frame_ids(tmp_path) == set()


def test_repair_index_from_committed_pointdirs_adds_missing_records(tmp_path):
    from ichor.hpc.active_learning.daemon.input_staging import (
        commit_reference_data_delta,
    )
    from ichor.hpc.active_learning.point_allocation import (
        create_point_allocation,
        pending_attempts,
        point_allocation_path,
        record_quantum_results,
    )

    training = tmp_path / "QM_REFERENCE_DATA"
    allocation_path = point_allocation_path(
        tmp_path,
        context="bootstrap",
        iteration=0,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="X",
        context="bootstrap",
        iteration=0,
        targets={"train": 1, "int_val": 0, "ext_val": 0, "total": 1},
        primary_candidates=[
            {
                "candidate_id": "candidate-42",
                "frame_id": 42,
                "pointdir_name": "POINT_0000.pointdir",
            }
        ],
        reserve_candidates=[],
    )
    attempt = pending_attempts(allocation)[0]
    pdir = tmp_path / ".DATA" / "STAGING" / "initial" / "POINT_0000.pointdir"
    write_seed_provenance(
        pdir,
        campaign_uid="X",
        iteration=0,
        trajectory_sha256="",
        seed_frame_id=42,
        seed_selection_origin="variance",
        seed_variance_at_selection=None,
        subspace_neighbour_frame_ids=[42],
        subspace_dimension=1,
        subspace_eigenvalues=[1.0],
    )
    enrich_with_point_allocation(
        pdir,
        candidate_id=str(attempt["candidate_id"]),
        context="bootstrap",
        slot_id=int(attempt["slot_id"]),
        split=str(attempt["split"]),
    )
    record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": str(attempt["candidate_id"]),
                "accepted": True,
                "pointdir": str(pdir),
            }
        ],
    )
    commit_reference_data_delta(
        tmp_path,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )

    assert repair_index_from_committed_pointdirs(tmp_path, training) == 1
    assert repair_index_from_committed_pointdirs(tmp_path, training) == 0
    data = load_index(tmp_path)
    assert data["records"] == [
        {
            "iteration": 0,
            "pointdir_name": "POINT_000000.pointdir",
            "seed_frame_id": 42,
        }
    ]


# --- end-to-end integration (dry-run executor drives full chain) -------


def test_full_provenance_chain_through_dry_run_executor(tmp_path):
    """Drive the dry-run executor through ARIADNE_ARRAY -> PHASE_B_POLUS ->
    APPEND for a single iteration and verify the full provenance + index
    flow lands as documented in the M10 design.

    Specifically:
      * each seed_NNNN/ pool subdir gets a .provenance.json after ARIADNE,
      * PHASE_B enrich populates the phase_b block in-place,
      * APPEND copies the provenance sidecar into the committed pointdir
        AND emits one record per pointdir into seed_frame_id_index.json,
      * a reference_data_committed journal event is emitted.
    """
    from types import SimpleNamespace

    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import CampaignPhase
    from ichor.hpc.active_learning.versioning.reference_data import (
        ReferenceDataVersioning,
    )

    campaign_dir = tmp_path / "campaign"
    cfg = CampaignConfig(max_iterations=2)
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.seed_selection.n_seeds_per_iteration = 3
    ex = DryRunPhaseExecutor(campaign_dir=campaign_dir, config=cfg)
    state = SimpleNamespace(
        iteration=0,
        campaign_uid="uid-e2e",
        replacement_round=0,
        reference_data_version=-1,
        models_version=-1,
    )

    ex.postprocess(state, CampaignPhase.PHASE_A_POLUS, observations=[])
    ex.postprocess(state, CampaignPhase.INITIAL_GAUSSIAN, observations=[])
    ex.postprocess(state, CampaignPhase.INITIAL_AIMALL, observations=[])
    ex.submit_or_run(state, CampaignPhase.INITIAL_ALLOCATION_CHECK)
    bootstrap = ex.postprocess(
        state,
        CampaignPhase.INITIAL_FEREBUS,
        observations=[],
    )
    state.reference_data_version = bootstrap.state_updates["reference_data_version"]
    state.models_version = bootstrap.state_updates["models_version"]

    ex.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    iter_dir = campaign_dir / "7_ACTIVE_LEARNING" / "iteration-0000"
    pool = iter_dir / "pool"
    assert pool.is_dir()
    seed_dirs = sorted(d for d in pool.iterdir() if d.is_dir())
    assert seed_dirs, "ARIADNE produced no seed subdirs"
    for sd in seed_dirs:
        data = read_provenance(sd)
        assert data["seed"]["selection_origin"] in {"bulk", "variance"}
        assert data["ariadne"] is not None
        assert "alpha_final" in data["ariadne"]
        assert data["anti_overlap"] is not None
        assert data["anti_overlap"]["passed"] is True
        assert data["phase_b"] is None

    ex.postprocess(state, CampaignPhase.PHASE_B_POLUS, observations=[])
    selected_after_fps = 0
    for sd in seed_dirs:
        data = read_provenance(sd)
        assert data["phase_b"] is not None
        selected_after_fps += int(bool(data["phase_b"]["selected_after_fps"]))
        assert data["phase_b"]["descriptor_used"] == cfg.phase_b.descriptor
    assert selected_after_fps == cfg.point_allocation.batch_total_size

    ex.postprocess(state, CampaignPhase.GAUSSIAN, observations=[])
    ex.postprocess(state, CampaignPhase.AIMALL, observations=[])
    ex.submit_or_run(state, CampaignPhase.ALLOCATION_CHECK)
    ex.submit_or_run(state, CampaignPhase.APPEND)
    v = ReferenceDataVersioning(campaign_dir / "QM_REFERENCE_DATA")
    assert v.current_version() == 1
    committed_iter = v.iteration_path(1)
    committed_delta = sorted(committed_iter.glob("*.pointdir"))
    assert len(committed_delta) == cfg.point_allocation.batch_total_size
    view = v.resolve(1, verification="deep")
    committed_pdirs = [entry.pointdir_path for entry in view.entries]
    assert len(committed_pdirs) == (
        cfg.point_allocation.bootstrap_total_size
        + cfg.point_allocation.batch_total_size
    )
    active_pdirs = [
        pointdir
        for pointdir in committed_pdirs
        if read_provenance(pointdir)["point_allocation"]["context"] == "active"
    ]
    assert len(active_pdirs) == cfg.point_allocation.batch_total_size
    assert active_pdirs == committed_delta
    for pdir in committed_pdirs:
        data = read_provenance(pdir)
        if data["point_allocation"]["context"] == "active":
            assert data["ariadne"] is not None
            assert data["anti_overlap"] is not None
            assert data["phase_b"] is not None
        else:
            assert data["point_allocation"]["context"] == "bootstrap"
            assert data["ariadne"] is None
            assert data["anti_overlap"] is None
            assert data["phase_b"] is None

    idx_data = load_index(campaign_dir)
    assert len(idx_data["records"]) == len(active_pdirs)
    for rec in idx_data["records"]:
        assert rec["iteration"] == 1
        assert rec["pointdir_name"].startswith("POINT_")
        assert rec["seed_frame_id"] is None

    assert load_training_seed_frame_ids(campaign_dir) == set()

    journal_path = (
        campaign_dir / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    )
    assert journal_path.is_file()
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    commits = [e for e in events if e.get("event") == "reference_data_committed"]
    assert commits, "no reference_data_committed event emitted"
    assert any(e.get("reference_data_version") == 1 for e in commits)


def test_full_provenance_chain_two_iterations_grows_index_monotonically(tmp_path):
    """Two passes through ARIADNE_ARRAY -> PHASE_B_POLUS -> APPEND must
    produce a monotonically-growing index. Catches regressions where the
    APPEND phase resets the index, or where iteration numbers in records
    drift."""
    from types import SimpleNamespace

    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import CampaignPhase

    campaign_dir = tmp_path / "campaign"
    cfg = CampaignConfig(max_iterations=3)
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.seed_selection.n_seeds_per_iteration = 2
    ex = DryRunPhaseExecutor(campaign_dir=campaign_dir, config=cfg)

    bootstrap_state = SimpleNamespace(
        iteration=0,
        campaign_uid="uid",
        replacement_round=0,
        reference_data_version=-1,
        models_version=-1,
    )
    ex.postprocess(bootstrap_state, CampaignPhase.PHASE_A_POLUS, observations=[])
    ex.postprocess(bootstrap_state, CampaignPhase.INITIAL_GAUSSIAN, observations=[])
    ex.postprocess(bootstrap_state, CampaignPhase.INITIAL_AIMALL, observations=[])
    ex.submit_or_run(bootstrap_state, CampaignPhase.INITIAL_ALLOCATION_CHECK)
    ex.postprocess(
        bootstrap_state,
        CampaignPhase.INITIAL_FEREBUS,
        observations=[],
    )

    # Simulate the daemon's state machine: each loop iteration updates
    # reference_data_version from the APPEND return so the M15 F2 idempotency
    # guard in _inline_append sees a fresh expected-next per loop.
    reference_data_version = 0
    for it in (0, 1):
        state_ns = SimpleNamespace(
            iteration=it,
            campaign_uid="uid",
            replacement_round=0,
            reference_data_version=reference_data_version,
            models_version=reference_data_version,
        )
        ex.postprocess(state_ns, CampaignPhase.ARIADNE_ARRAY, observations=[])
        ex.postprocess(state_ns, CampaignPhase.PHASE_B_POLUS, observations=[])
        ex.postprocess(state_ns, CampaignPhase.GAUSSIAN, observations=[])
        ex.postprocess(state_ns, CampaignPhase.AIMALL, observations=[])
        ex.submit_or_run(state_ns, CampaignPhase.ALLOCATION_CHECK)
        result = ex.submit_or_run(state_ns, CampaignPhase.APPEND)
        # PhaseResult; pull the updated reference_data_version out of it.
        if hasattr(result, "state_updates") and result.state_updates:
            reference_data_version = int(
                result.state_updates.get("reference_data_version", reference_data_version)
            )

    idx_data = load_index(campaign_dir)
    iterations_in_index = sorted({r["iteration"] for r in idx_data["records"]})
    assert iterations_in_index == [1, 2]
    pairs = [(r["iteration"], r["pointdir_name"]) for r in idx_data["records"]]
    assert len(set(pairs)) == len(pairs)



# --- recent_seeds cooldown cache (M11) ---------------------------------


from ichor.hpc.active_learning.versioning.provenance import (
    DEFAULT_RECENT_SEEDS_COOLDOWN,
    RECENT_SEEDS_FILENAME,
    RECENT_SEEDS_SCHEMA_VERSION,
    append_recent_seeds,
    load_recent_seed_frame_ids,
    load_recent_seeds_payload,
)


def test_load_recent_seed_frame_ids_empty_for_fresh_campaign(tmp_path):
    assert load_recent_seed_frame_ids(tmp_path) == set()


def test_load_recent_seeds_payload_returns_empty_on_missing(tmp_path):
    data = load_recent_seeds_payload(tmp_path)
    assert data["schema_version"] == RECENT_SEEDS_SCHEMA_VERSION
    assert data["history"] == []


def test_append_recent_seeds_grows_and_rolls_off_oldest(tmp_path):
    append_recent_seeds(tmp_path, iteration=0, frame_ids=[1, 2], cooldown=3)
    append_recent_seeds(tmp_path, iteration=1, frame_ids=[3], cooldown=3)
    append_recent_seeds(tmp_path, iteration=2, frame_ids=[4, 5], cooldown=3)
    data = load_recent_seeds_payload(tmp_path)
    assert [e["iteration"] for e in data["history"]] == [0, 1, 2]
    # 4th append rolls iteration 0 off.
    append_recent_seeds(tmp_path, iteration=3, frame_ids=[6], cooldown=3)
    data = load_recent_seeds_payload(tmp_path)
    assert [e["iteration"] for e in data["history"]] == [1, 2, 3]
    s = load_recent_seed_frame_ids(tmp_path)
    assert s == {3, 4, 5, 6}


def test_append_recent_seeds_strips_none_frame_ids(tmp_path):
    append_recent_seeds(
        tmp_path, iteration=0,
        frame_ids=[1, None, 2, None, 3],
        cooldown=3,
    )
    data = load_recent_seeds_payload(tmp_path)
    assert data["history"][0]["frame_ids"] == [1, 2, 3]


def test_load_recent_seeds_payload_raises_on_corruption(tmp_path):
    p = tmp_path / ".DATA" / "ACTIVE_LEARNING"
    p.mkdir(parents=True)
    (p / RECENT_SEEDS_FILENAME).write_text("not json {", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="failed to parse"):
        load_recent_seeds_payload(tmp_path)


def test_load_recent_seeds_payload_raises_on_schema_drift(tmp_path):
    p = tmp_path / ".DATA" / "ACTIVE_LEARNING"
    p.mkdir(parents=True)
    (p / RECENT_SEEDS_FILENAME).write_text(
        json.dumps({"schema_version": 999, "cooldown": 3, "history": []}),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="schema_version"):
        load_recent_seeds_payload(tmp_path)


def test_default_cooldown_constant_is_three():
    assert DEFAULT_RECENT_SEEDS_COOLDOWN == 3


# --- M15 F5: cooldown=0 and negative-cooldown handling -----------------


def test_append_recent_seeds_cooldown_zero_empties_history(tmp_path):
    """cooldown=0 must trim to an empty history after every append (operator
    saying 'do not remember any seeds'). Pre-M15 the lst[-0:] slice silently
    returned the whole list, retaining everything."""
    append_recent_seeds(tmp_path, iteration=0, frame_ids=[1, 2], cooldown=0)
    data = load_recent_seeds_payload(tmp_path)
    assert data["history"] == []
    assert data["cooldown"] == 0
    append_recent_seeds(tmp_path, iteration=1, frame_ids=[3, 4], cooldown=0)
    data = load_recent_seeds_payload(tmp_path)
    assert data["history"] == []
    assert load_recent_seed_frame_ids(tmp_path) == set()


def test_append_recent_seeds_clamps_negative_cooldown_to_zero(tmp_path):
    append_recent_seeds(tmp_path, iteration=0, frame_ids=[1], cooldown=-5)
    data = load_recent_seeds_payload(tmp_path)
    assert data["cooldown"] == 0
    assert data["history"] == []


def test_load_training_seed_frame_ids_self_heals_from_sidecars(tmp_path):
    # the flat index can fall short of what's committed if a crash hit between
    # commit() and the index-append loop. passing training_dir should union the
    # committed pointdir sidecars back in so a truncated index can't make
    # SEED_SELECT re-pick an already-trained frame.
    from ichor.hpc.active_learning.daemon.input_staging import (
        commit_reference_data_delta,
    )
    from ichor.hpc.active_learning.point_allocation import (
        create_point_allocation,
        pending_attempts,
        point_allocation_path,
        record_quantum_results,
    )

    campaign = tmp_path
    training = campaign / "QM_REFERENCE_DATA"
    allocation_path = point_allocation_path(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="u",
        context="bootstrap",
        iteration=0,
        targets={"train": 1, "int_val": 1, "ext_val": 0, "total": 2},
        primary_candidates=[
            {
                "candidate_id": "candidate-0",
                "frame_id": 11,
                "pointdir_name": "POINT_0000.pointdir",
            },
            {
                "candidate_id": "candidate-1",
                "frame_id": 22,
                "pointdir_name": "POINT_0001.pointdir",
            },
        ],
        reserve_candidates=[],
    )
    results = []
    for attempt in pending_attempts(allocation):
        pd = (
            campaign
            / ".DATA"
            / "STAGING"
            / "initial"
            / str(attempt["pointdir_name"])
        )
        pd.mkdir(parents=True)
        write_seed_provenance(
            pd,
            campaign_uid="u", iteration=0, trajectory_sha256="s",
            seed_frame_id=int(attempt["frame_id"]), seed_selection_origin="bulk",
            seed_variance_at_selection=None,
            subspace_neighbour_frame_ids=[], subspace_dimension=0,
            subspace_eigenvalues=[],
        )
        enrich_with_point_allocation(
            pd,
            candidate_id=str(attempt["candidate_id"]),
            context="bootstrap",
            slot_id=int(attempt["slot_id"]),
            split=str(attempt["split"]),
        )
        results.append(
            {
                "candidate_id": str(attempt["candidate_id"]),
                "accepted": True,
                "pointdir": str(pd),
            }
        )
    record_quantum_results(allocation_path, results)
    commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    # index only knows about the first point (simulating a crash before the
    # second append landed).
    ensure_index(campaign)
    append_to_index(
        campaign,
        iteration=0,
        pointdir_name="POINT_000000.pointdir",
        seed_frame_id=11,
    )

    # index-only view misses 22
    assert load_training_seed_frame_ids(campaign) == {11}
    # self-healing view recovers it from the sidecar
    assert load_training_seed_frame_ids(
        campaign,
        reference_data_dir=training,
    ) == {11, 22}
