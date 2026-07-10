"""M16 Day 2: unit tests for _parse_quantum_postprocess.

Strategy: drive the parser directly with hand-constructed states + phases,
overriding _quantum_staging_path to point at the fixture pack rather than
materialising a campaign filesystem.
"""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon import live_executor as live_executor_mod
from ichor.hpc.active_learning.daemon.live_executor import (
    FEREBUS_TASK_ARTEFACTS_MANIFEST,
    LIVE_POSTPROCESS_IMPLEMENTED,
    LiveBackendsPhaseExecutor,
    _ariadne_optional_diagnostic_warnings,
    _write_ferebus_task_artefact_layout,
    clean_stale_ariadne_seed_outputs,
)
from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.daemon.phase_executor import (
    BackendSubmissionError,
    PhaseResult,
)
from ichor.hpc.active_learning.daemon.state import CampaignPhase
from ichor.hpc.active_learning.point_allocation import (
    create_point_allocation,
    pending_attempts,
    point_allocation_path,
    record_quantum_results,
)
from ichor.hpc.active_learning.versioning.provenance import (
    enrich_with_point_allocation,
    write_seed_provenance,
)
from ichor.hpc.active_learning.versioning.training_set import TrainingSetVersioning


FIXTURES = (
    Path(__file__).resolve().parent / "fixtures" / "live_outputs"
)


def _make_executor(tmp_path, failure_threshold=0.5):
    cfg = CampaignConfig()
    cfg.failure_threshold_fraction = failure_threshold
    return LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )


def _bind_staging(ex, fixture_subdir):
    source = Path(fixture_subdir)
    if source.is_dir():
        target = Path(ex.campaign_dir) / ".DATA" / "TEST_STAGING" / source.name
        if target.exists():
            shutil.rmtree(str(target), ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(source), str(target))
        pointdirs = sorted(target.glob("POINT_*.pointdir"))
        if pointdirs:
            stg.write_points_file(target, pointdirs)
    else:
        target = source
    ex._quantum_staging_path = lambda state, phase_name: target
    return target


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


def _seed_point_allocation(campaign, staging, *, context, iteration):
    pointdirs = sorted(Path(staging).glob("POINT_*.pointdir"))
    candidates = [
        {
            "candidate_id": (
                str(context) + "-" + str(int(iteration)) + "-" + pointdir.name
            ),
            "frame_id": index,
            "pointdir_name": pointdir.name,
        }
        for index, pointdir in enumerate(pointdirs)
    ]
    allocation_path = point_allocation_path(
        campaign,
        context=context,
        iteration=iteration,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="m16-test",
        context=context,
        iteration=iteration,
        targets={
            "train": len(candidates),
            "int_val": 0,
            "ext_val": 0,
            "total": len(candidates),
        },
        primary_candidates=candidates,
        reserve_candidates=[],
    )
    attempts_by_name = {
        str(attempt["pointdir_name"]): attempt
        for attempt in pending_attempts(allocation)
    }
    for pointdir in pointdirs:
        attempt = attempts_by_name[pointdir.name]
        write_seed_provenance(
            pointdir,
            campaign_uid="m16-test",
            iteration=iteration,
            trajectory_sha256="0" * 64,
            seed_frame_id=attempt.get("frame_id"),
            seed_selection_origin="live_parser_fixture",
            seed_variance_at_selection=None,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
        )
        enrich_with_point_allocation(
            pointdir,
            candidate_id=str(attempt["candidate_id"]),
            context=context,
            slot_id=int(attempt["slot_id"]),
            split=str(attempt["split"]),
        )
    return allocation_path, allocation


def _complete_point_allocation(campaign, staging, *, context, iteration):
    allocation_path, allocation = _seed_point_allocation(
        campaign,
        staging,
        context=context,
        iteration=iteration,
    )
    pointdirs = {
        pointdir.name: pointdir
        for pointdir in sorted(Path(staging).glob("POINT_*.pointdir"))
    }
    record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": str(attempt["candidate_id"]),
                "accepted": True,
                "pointdir": str(pointdirs[str(attempt["pointdir_name"])]),
            }
            for attempt in pending_attempts(allocation)
        ],
    )
    return allocation_path


def test_ariadne_optional_scale_diagnostics_are_warnings_only():
    payload = {
        "optimiser_diagnostics": {
            "trqn_scale_mode": "adaptive_initial_gradient",
            "trqn_objective_scale": "not-a-number",
            "trqn_initial_raw_grad_norm": 12.0,
            "trqn_retry_scaled_grad_norm": float("inf"),
        }
    }

    warnings = _ariadne_optional_diagnostic_warnings(payload)

    assert "trqn_objective_scale_not_numeric" in warnings
    assert "trqn_retry_scaled_grad_norm_not_finite" in warnings


def test_ariadne_optional_scale_diagnostics_accept_old_results():
    assert _ariadne_optional_diagnostic_warnings({}) == []
    assert _ariadne_optional_diagnostic_warnings({"optimiser_diagnostics": {}}) == []


def test_four_quantum_phases_registered_as_live():
    for ph in ("INITIAL_GAUSSIAN", "GAUSSIAN", "INITIAL_AIMALL", "AIMALL"):
        assert ph in LIVE_POSTPROCESS_IMPLEMENTED


def test_handlers_dict_dispatches_quantum_phases(tmp_path):
    ex = _make_executor(tmp_path)
    handlers = ex._live_postprocess_handlers()
    for ph in ("INITIAL_GAUSSIAN", "GAUSSIAN", "INITIAL_AIMALL", "AIMALL"):
        assert handlers[ph] == ex._parse_quantum_postprocess


def test_initial_gaussian_happy_path(tmp_path):
    ex = _make_executor(tmp_path)
    _bind_staging(ex, FIXTURES / "initial_quantum")
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_quantum_postprocess(
        state, CampaignPhase("INITIAL_GAUSSIAN"), observations=[],
    )
    assert isinstance(result, PhaseResult)
    assert result.is_complete is True
    assert result.failure_reason is None
    assert result.state_updates == {}
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded
    assert succeeded[-1]["phase"] == "INITIAL_GAUSSIAN"
    assert succeeded[-1]["n_kept"] == 4
    assert succeeded[-1]["n_rejected"] == 0


def test_iter_gaussian_happy_path(tmp_path):
    ex = _make_executor(tmp_path)
    _bind_staging(ex, FIXTURES / "iter_quantum")
    state = SimpleNamespace(iteration=3, campaign_uid="m16-test", training_set_version=0)
    result = ex._parse_quantum_postprocess(
        state, CampaignPhase("GAUSSIAN"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None


def test_initial_aimall_happy_path(tmp_path):
    ex = _make_executor(tmp_path)
    staging = _bind_staging(ex, FIXTURES / "initial_quantum")
    _seed_point_allocation(
        ex.campaign_dir,
        staging,
        context="bootstrap",
        iteration=0,
    )
    stg.write_quantum_acceptance_manifest(
        staging,
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        accepted=sorted(staging.glob("POINT_*.pointdir")),
        rejected=[],
    )
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_quantum_postprocess(
        state, CampaignPhase("INITIAL_AIMALL"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded[-1]["phase"] == "INITIAL_AIMALL"
    manifest = json.loads((staging / stg.QUANTUM_ACCEPTANCE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["phase"] == "INITIAL_AIMALL"
    assert manifest["iteration"] == 0
    assert len(manifest["accepted_pointdirs"]) == 4


def test_stage_aimall_inputs_writes_resolved_naat_metadata(tmp_path):
    campaign = tmp_path / "campaign"
    staging = campaign / ".DATA" / "STAGING" / "initial"
    shutil.copytree(str(FIXTURES / "initial_quantum"), str(staging))
    accepted = sorted(staging.glob("POINT_*.pointdir"))[:1]
    (accepted[0] / "input.wfn").write_text("synthetic wfn\n", encoding="utf-8")
    stg.write_points_file(staging, sorted(staging.glob("POINT_*.pointdir")))
    stg.write_quantum_acceptance_manifest(
        staging,
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        accepted=accepted,
        rejected=[],
    )
    cfg = CampaignConfig()
    cfg.resources.aimall_cpus_per_task = 8
    cfg.aimall.naat = "auto"

    staged_dir, n_points = stg.stage_aimall_inputs(
        campaign, cfg, "INITIAL_AIMALL", 0,
    )

    assert staged_dir == staging
    assert n_points == 1
    task = json.loads(
        (accepted[0] / stg.AIMALL_TASK_METADATA).read_text(encoding="utf-8")
    )
    assert task["atom_count"] == 3
    assert task["nproc"] == 8
    assert task["naat"] == 3


def test_aimall_parser_only_consumes_gaussian_accepted_pointdirs(tmp_path):
    ex = _make_executor(tmp_path)
    staging = _bind_staging(ex, FIXTURES / "initial_quantum")
    _seed_point_allocation(
        ex.campaign_dir,
        staging,
        context="bootstrap",
        iteration=0,
    )
    accepted = [staging / "POINT_0000.pointdir", staging / "POINT_0001.pointdir"]
    stg.write_quantum_acceptance_manifest(
        staging,
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        accepted=accepted,
        rejected=[
            ("POINT_0002.pointdir", "wfn_gjf_geometry_mismatch"),
            ("POINT_0003.pointdir", "wfn_gjf_geometry_mismatch"),
        ],
    )
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")

    result = ex._parse_quantum_postprocess(
        state, CampaignPhase("INITIAL_AIMALL"), observations=[],
    )

    assert result.failure_reason is None
    manifest = json.loads((staging / stg.QUANTUM_ACCEPTANCE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["accepted_pointdirs"] == ["POINT_0000.pointdir", "POINT_0001.pointdir"]


def test_partial_rejection_below_threshold_still_succeeds(tmp_path):
    ex = _make_executor(tmp_path, failure_threshold=0.5)
    staging = _bind_staging(ex, FIXTURES / "iter_quantum_scf_failure")
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_quantum_postprocess(
        state, CampaignPhase("GAUSSIAN"), observations=[],
    )
    assert result.failure_reason is None
    manifest = json.loads((staging / stg.QUANTUM_ACCEPTANCE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["phase"] == "GAUSSIAN"
    assert len(manifest["accepted_pointdirs"]) == 1
    assert manifest["rejected"][0]["reason"] == "scf_nonconvergence_or_crash"
    events = _read_journal_events(tmp_path / "campaign")
    rejected = [e for e in events if e.get("event") == "quantum_output_rejected"]
    assert rejected
    assert rejected[-1]["reason"] == "scf_nonconvergence_or_crash"


def test_high_gaussian_rejection_is_deferred_to_allocation_replacement(tmp_path):
    ex = _make_executor(tmp_path, failure_threshold=0.3)
    staging = _bind_staging(ex, FIXTURES / "iter_quantum_scf_failure")
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_quantum_postprocess(
        state, CampaignPhase("GAUSSIAN"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None
    manifest = json.loads(
        (staging / stg.QUANTUM_ACCEPTANCE_MANIFEST).read_text(encoding="utf-8")
    )
    assert len(manifest["accepted_pointdirs"]) == 1
    assert len(manifest["rejected"]) == 1


def test_missing_staging_dir_sets_failure_reason(tmp_path):
    ex = _make_executor(tmp_path)
    _bind_staging(ex, tmp_path / "ghost_staging_that_does_not_exist")
    state = SimpleNamespace(iteration=2, campaign_uid="m16-test", training_set_version=0)
    result = ex._parse_quantum_postprocess(
        state, CampaignPhase("GAUSSIAN"), observations=[],
    )
    assert result.failure_reason is not None
    assert "no_pointdirs_in_staging" in result.failure_reason


def test_gaussian_phase_uses_gaussian_validator(tmp_path):
    ex = _make_executor(tmp_path)
    validators = ex._validators_for("INITIAL_GAUSSIAN")
    from ichor.hpc.active_learning.daemon.live_executor import (
        validate_gaussian_completed,
    )
    assert validate_gaussian_completed in validators


def test_aimall_phase_uses_aimall_validator(tmp_path):
    ex = _make_executor(tmp_path)
    validators = ex._validators_for("AIMALL")
    from ichor.hpc.active_learning.daemon.live_executor import (
        validate_aimall_completed,
    )
    assert validate_aimall_completed in validators


def test_unknown_phase_raises(tmp_path):
    ex = _make_executor(tmp_path)
    with pytest.raises(ValueError, match="unknown quantum phase"):
        ex._validators_for("MAGIC_QUANTUM_PHASE")


def test_initial_phase_uses_initial_staging(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=0)
    path = ex._quantum_staging_path(state, "INITIAL_GAUSSIAN")
    assert path.name == "initial"
    assert path.parent.name == "STAGING"


def test_iter_phase_uses_iter_n_staging(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=7)
    path = ex._quantum_staging_path(state, "GAUSSIAN")
    assert path.name == "iter_7"
    assert path.parent.name == "STAGING"


def test_postprocess_dispatches_to_quantum_handler(tmp_path):
    ex = _make_executor(tmp_path)
    _bind_staging(ex, FIXTURES / "initial_quantum")
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex.postprocess(state, CampaignPhase("INITIAL_GAUSSIAN"), observations=[])
    assert result.is_complete is True
    assert result.failure_reason is None


def test_commit_initial_training_set_rejects_incomplete_allocation(tmp_path):
    campaign = tmp_path / "campaign"
    initial = campaign / ".DATA" / "STAGING" / "initial"
    good = initial / "POINT_0001.pointdir"
    bad = initial / "POINT_0000.pointdir"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    (good / "accepted.txt").write_text("good\n", encoding="utf-8")
    (bad / "rejected.txt").write_text("bad\n", encoding="utf-8")
    stg.write_points_file(initial, [bad, good])
    stg.write_quantum_acceptance_manifest(
        initial,
        phase_name="INITIAL_AIMALL",
        iteration=0,
        accepted=[good],
        rejected=[("POINT_0000.pointdir", "missing_atomicfiles_dir")],
    )
    allocation_path, allocation = _seed_point_allocation(
        campaign,
        initial,
        context="bootstrap",
        iteration=0,
    )
    attempts = pending_attempts(allocation)
    accepted_id = next(
        str(attempt["candidate_id"])
        for attempt in attempts
        if str(attempt["pointdir_name"]) == good.name
    )
    record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": str(attempt["candidate_id"]),
                "accepted": str(attempt["candidate_id"]) == accepted_id,
                "pointdir": str(initial / str(attempt["pointdir_name"])),
                "reason": (
                    None
                    if str(attempt["candidate_id"]) == accepted_id
                    else "missing_atomicfiles_dir"
                ),
            }
            for attempt in attempts
        ],
    )

    with pytest.raises(ValueError, match="point allocation is incomplete"):
        stg.commit_initial_training_set(campaign)


def test_initial_aimall_reader_migrates_legacy_gaussian_alias(tmp_path):
    campaign = tmp_path / "campaign"
    initial = campaign / ".DATA" / "STAGING" / "initial"
    pointdir = initial / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True)
    stg.write_points_file(initial, [pointdir])
    legacy = {
        "schema_version": 1,
        "phase": "INITIAL_GAUSSIAN",
        "iteration": 0,
        "accepted_pointdirs": ["POINT_0000.pointdir"],
        "rejected": [],
        "n_total": 1,
    }
    (initial / "accepted_pointdirs.json").write_text(
        json.dumps(legacy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pointdirs, manifest = stg.read_quantum_acceptance_manifest(
        initial,
        expected_phase="INITIAL_GAUSSIAN",
        expected_iteration=0,
        require_points_file_membership=True,
    )

    assert [p.name for p in pointdirs] == ["POINT_0000.pointdir"]
    assert manifest["phase"] == "INITIAL_GAUSSIAN"
    migrated = initial / "accepted_pointdirs.INITIAL_GAUSSIAN.json"
    assert migrated.is_file()
    assert json.loads(migrated.read_text(encoding="utf-8")) == legacy


def test_commit_initial_training_set_rejects_missing_point_allocation(tmp_path):
    campaign = tmp_path / "campaign"
    v_train = TrainingSetVersioning(campaign / "5_TRAINING")
    staging = v_train.stage(source_version=None, target_version=0)
    (staging / "POINT_0000.pointdir").mkdir()
    v_train.commit(0)

    with pytest.raises(FileNotFoundError, match="point-allocation manifest missing"):
        stg.commit_initial_training_set(campaign)


def test_live_append_requires_complete_point_allocation(tmp_path):
    ex = _make_executor(tmp_path)
    v = TrainingSetVersioning(ex.campaign_dir / "5_TRAINING")
    staging = v.stage(source_version=None, target_version=0)
    (staging / "POINT_0000.pointdir").mkdir()
    v.commit(0)
    v.update_current(0)
    live_staging = stg.bucket_dir(ex.campaign_dir, "APPEND", 0)
    (live_staging / "POINT_0000.pointdir").mkdir(parents=True)
    _seed_point_allocation(
        ex.campaign_dir,
        live_staging,
        context="active",
        iteration=0,
    )
    state = SimpleNamespace(iteration=0, campaign_uid="uid", training_set_version=0)

    with pytest.raises(BackendSubmissionError, match="complete point allocation"):
        ex._inline_append(state)


def test_live_append_commits_global_pointdir_names_from_manifest(tmp_path):
    ex = _make_executor(tmp_path)
    v = TrainingSetVersioning(ex.campaign_dir / "5_TRAINING")
    base = v.stage(source_version=None, target_version=0)
    old_point = base / "POINT_0000.pointdir"
    old_point.mkdir()
    (old_point / "old.txt").write_text("old\n", encoding="utf-8")
    v.commit(0)
    v.update_current(0)

    live_staging = stg.bucket_dir(ex.campaign_dir, "APPEND", 0)
    new_point = live_staging / "POINT_0000.pointdir"
    new_point.mkdir(parents=True)
    (new_point / "new.txt").write_text("new\n", encoding="utf-8")
    stg.write_points_file(live_staging, [new_point])
    stg.write_quantum_acceptance_manifest(
        live_staging,
        phase_name="AIMALL",
        iteration=0,
        accepted=[new_point],
        rejected=[],
    )
    _complete_point_allocation(
        ex.campaign_dir,
        live_staging,
        context="active",
        iteration=0,
    )
    state = SimpleNamespace(iteration=0, campaign_uid="uid", training_set_version=0)

    result = ex._inline_append(state)

    assert result["training_set_version"] == 1
    committed = v.iteration_path(1)
    assert (committed / "POINT_0000.pointdir" / "old.txt").is_file()
    assert (committed / "POINT_0001.pointdir" / "new.txt").is_file()
    assert not (committed / "POINT_0000.pointdir" / "new.txt").exists()


# ==================================================================
# M16 Day 3: FEREBUS / ARIADNE / POLUS parser tests
# ==================================================================


def test_all_nine_sbatch_phases_registered_as_live():
    """After M16 Day 3, every SBATCH phase has a live parser."""
    for ph in (
        "INITIAL_GAUSSIAN", "GAUSSIAN",
        "INITIAL_AIMALL", "AIMALL",
        "INITIAL_FEREBUS", "FEREBUS",
        "ARIADNE_ARRAY",
        "PHASE_A_POLUS", "PHASE_B_POLUS",
    ):
        assert ph in LIVE_POSTPROCESS_IMPLEMENTED


def test_handlers_dict_dispatches_all_day3_phases(tmp_path):
    ex = _make_executor(tmp_path)
    handlers = ex._live_postprocess_handlers()
    assert handlers["INITIAL_FEREBUS"] == ex._parse_ferebus_postprocess
    assert handlers["FEREBUS"] == ex._parse_ferebus_postprocess
    assert handlers["ARIADNE_ARRAY"] == ex._parse_ariadne_array_postprocess
    assert handlers["PHASE_A_POLUS"] == ex._parse_polus_postprocess
    assert handlers["PHASE_B_POLUS"] == ex._parse_polus_postprocess


# --- FEREBUS parser tests ----------------------------------------


def _seed_models_staging(campaign_dir):
    """Create a manifest-backed pyferebus staging tree with one expected model."""
    target = campaign_dir / "6_TRAINED_MODELS" / "iteration-staging"
    target.mkdir(parents=True, exist_ok=True)
    model_dir = target / "iqa" / "O1"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "ferebus.config").write_text("name WATER\nproperties [\"iqa\"]\n", encoding="utf-8")
    model = model_dir / "WATER_iqa_O1.model"
    _write_loadable_model(model, atom="O1")
    train_csv = model_dir / "WATER_O1_TRAINING_SET.csv"
    int_csv = model_dir / "WATER_O1_INT_VALIDATION_SET.csv"
    ext_csv = model_dir / "WATER_O1_EXT_VALIDATION_SET.csv"
    _write_metric_csv(train_csv, 5)
    _write_metric_csv(int_csv, 2)
    _write_metric_csv(ext_csv, 2)
    for suffix in ("opt", "perf", "pred", "scurve", "sol"):
        (model_dir / ("WATER_iqa_O1." + suffix)).write_text(
            suffix + "\n",
            encoding="utf-8",
        )
    (target / "FEREBUS_QUALITY.json").write_text(
        json.dumps({
            "schema_version": 1,
            "training_version": 0,
            "accepted": True,
            "summary": {},
            "tasks": [],
        }),
        encoding="utf-8",
    )
    (target / stg.FEREBUS_TASK_MANIFEST).write_text(
        json.dumps({
            "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
            "system": "WATER",
            "training_version": 0,
            "properties": ["iqa"],
            "atoms": ["O1"],
            "n_tasks": 1,
            "tasks": [{
                "task_index": 1,
                "property": "iqa",
                "atom": "O1",
                "alf_1_indexed": [1, 2, 3],
                "config_path": str(model_dir / "ferebus.config"),
                "expected_model_path": str(model),
                "training_csv": str(train_csv),
                "int_validation_csv": str(int_csv),
                "ext_validation_csv": str(ext_csv),
                "row_counts": {"train": 5, "int_val": 2, "ext_val": 2},
            }],
        }),
        encoding="utf-8",
    )
    for name in (stg.FEREBUS_JOB_DETAILS, "commands", "list.txt", "runFerebus.sh"):
        (target / name).write_text(name + "\n", encoding="utf-8")
    return target


def _write_metric_csv(path, n_rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("f1,f2,f3,iqa\n")
        for i in range(int(n_rows)):
            f.write(f"{0.1+i*0.1},{0.2+i*0.1},{0.3+i*0.1},0.0\n")


def _write_loadable_model(
    path,
    *,
    atom="O1",
    prop="iqa",
    system="WATER",
    alf=(1, 2, 3),
    ntrain=5,
    nfeats=3,
):
    rows = [
        [0.1 + i * 0.1 + j * 0.01 for j in range(nfeats)]
        for i in range(ntrain)
    ]
    lines = [
        "# jitter 1.0e-6",
        "# likelihood -1.0",
        "",
        "[system]",
        "name " + system,
        "atom " + atom,
        "property " + prop,
        "ALF " + " ".join(str(int(x)) for x in alf),
        "",
        "[dimensions]",
        "number_of_atoms 3",
        "number_of_features " + str(nfeats),
        "number_of_training_points " + str(ntrain),
        "",
        "[mean]",
        "type zero",
        "",
        "[kernels]",
        "number_of_kernels 1",
        "composition k1",
        "",
        "[kernel.k1]",
        "type rbf",
        "number_of_dimensions " + str(nfeats),
        "active_dimensions " + " ".join(str(i + 1) for i in range(nfeats)),
        "thetas " + " ".join("1.0" for _ in range(nfeats)),
        "",
        "[training_data]",
        "units.x " + " ".join("bohr" for _ in range(nfeats)),
        "units.y Ha",
        "",
        "[training_data.x]",
    ]
    lines += [" ".join(str(v) for v in row) for row in rows]
    lines += ["", "[training_data.y]"]
    lines += [str(-75.0 - i * 0.01) for i in range(ntrain)]
    lines += ["", "[weights]"]
    lines += ["0.0" for _ in range(ntrain)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ferebus_parser_happy_path_commits_models_version(tmp_path):
    ex = _make_executor(tmp_path)
    _seed_models_staging(tmp_path / "campaign")
    state = SimpleNamespace(iteration=3, campaign_uid="m16-test", training_set_version=0)
    result = ex._parse_ferebus_postprocess(
        state, CampaignPhase("FEREBUS"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None
    # validation_set_version now advances alongside models_version each FEREBUS commit (A12)
    assert result.state_updates == {"models_version": 0, "validation_set_version": 0}
    # The committed iteration-0000 directory should now hold the .model file.
    committed_dir = (
        tmp_path / "campaign" / "6_TRAINED_MODELS" / "iteration-0000"
    )
    assert committed_dir.is_dir()
    assert (committed_dir / "WATER_iqa_O1.model").is_file()
    assert (committed_dir / stg.FEREBUS_TASK_MANIFEST).is_file()
    assert (committed_dir / "ferebus_iqa_O1.config").is_file()
    artefact_dir = committed_dir / "task_artefacts" / "iqa" / "O1"
    assert artefact_dir.is_dir()
    assert not (artefact_dir / "WATER_iqa_O1.model").exists()
    assert not (artefact_dir / "ferebus.config").exists()
    assert (artefact_dir / "WATER_iqa_O1.opt").read_text(encoding="utf-8") == "opt\n"
    artefact_manifest = json.loads(
        (committed_dir / "task_artefacts" / FEREBUS_TASK_ARTEFACTS_MANIFEST)
        .read_text(encoding="utf-8")
    )
    assert artefact_manifest["schema_version"] == 1
    assert artefact_manifest["tasks"] == [{
        "property": "iqa",
        "atom": "O1",
        "directory": "task_artefacts/iqa/O1",
        "canonical_model": "WATER_iqa_O1.model",
        "canonical_config": "ferebus_iqa_O1.config",
        "files": {
            "opt": "task_artefacts/iqa/O1/WATER_iqa_O1.opt",
            "perf": "task_artefacts/iqa/O1/WATER_iqa_O1.perf",
            "pred": "task_artefacts/iqa/O1/WATER_iqa_O1.pred",
            "scurve": "task_artefacts/iqa/O1/WATER_iqa_O1.scurve",
            "sol": "task_artefacts/iqa/O1/WATER_iqa_O1.sol",
        },
    }]
    events = _read_journal_events(tmp_path / "campaign")
    assert any(e.get("event") == "models_committed" for e in events)


def test_ferebus_task_artefact_layout_supports_properties_and_missing_files(tmp_path):
    staging = tmp_path / "campaign" / "6_TRAINED_MODELS" / "iteration-staging"
    committed = tmp_path / "campaign" / "6_TRAINED_MODELS" / "iteration-0000"
    staging.mkdir(parents=True)
    committed.mkdir(parents=True)
    tasks = []
    for prop, atom in (("iqa", "O1"), ("q00", "O1")):
        task_dir = staging / prop / atom
        task_dir.mkdir(parents=True)
        model = task_dir / ("WATER_" + prop + "_" + atom + ".model")
        config = task_dir / "ferebus.config"
        model.write_text("model\n", encoding="utf-8")
        config.write_text("config\n", encoding="utf-8")
        (task_dir / ("WATER_" + prop + "_" + atom + ".opt")).write_text(
            prop + "\n",
            encoding="utf-8",
        )
        tasks.append({
            "property": prop,
            "atom": atom,
            "expected_model_path": str(model),
            "config_path": str(config),
        })

    _write_ferebus_task_artefact_layout(
        staging,
        committed,
        {"schema_version": 1, "training_version": 0, "tasks": tasks},
    )

    assert (committed / "task_artefacts" / "iqa" / "O1" / "WATER_iqa_O1.opt").is_file()
    assert (committed / "task_artefacts" / "q00" / "O1" / "WATER_q00_O1.opt").is_file()
    assert not (committed / "task_artefacts" / "iqa" / "O1" / "WATER_iqa_O1.model").exists()
    assert not (committed / "task_artefacts" / "q00" / "O1" / "ferebus.config").exists()
    artefact_manifest = json.loads(
        (committed / "task_artefacts" / FEREBUS_TASK_ARTEFACTS_MANIFEST)
        .read_text(encoding="utf-8")
    )
    assert artefact_manifest["n_tasks"] == 2
    assert artefact_manifest["tasks"][0]["files"]["perf"] is None
    assert artefact_manifest["tasks"][1]["property"] == "q00"


def test_ferebus_task_artefact_layout_rejects_unsafe_tokens(tmp_path):
    staging = tmp_path / "campaign" / "6_TRAINED_MODELS" / "iteration-staging"
    committed = tmp_path / "campaign" / "6_TRAINED_MODELS" / "iteration-0000"
    staging.mkdir(parents=True)
    committed.mkdir(parents=True)
    model = staging / "WATER_iqa_O1.model"
    config = staging / "ferebus.config"
    model.write_text("model\n", encoding="utf-8")
    config.write_text("config\n", encoding="utf-8")

    with pytest.raises(BackendSubmissionError, match="safe path token"):
        _write_ferebus_task_artefact_layout(
            staging,
            committed,
            {
                "schema_version": 1,
                "training_version": 0,
                "tasks": [{
                    "property": "iqa/bad",
                    "atom": "O1",
                    "expected_model_path": str(model),
                    "config_path": str(config),
                }],
            },
        )


def test_initial_ferebus_also_commits_training_set_version_zero(tmp_path):
    ex = _make_executor(tmp_path)
    _seed_models_staging(tmp_path / "campaign")
    # Put one synthetic pointdir in the initial-quantum staging so the parser
    # has something to copy into 5_TRAINING/iteration-0.
    initial_staging = tmp_path / "campaign" / ".DATA" / "STAGING" / "initial"
    initial_staging.mkdir(parents=True, exist_ok=True)
    src_pdir = FIXTURES / "initial_quantum" / "POINT_0000.pointdir"
    dst_pdir = initial_staging / "POINT_0000.pointdir"
    dst_pdir.mkdir(parents=True, exist_ok=True)
    for child in src_pdir.iterdir():
        if child.is_file():
            (dst_pdir / child.name).write_bytes(child.read_bytes())
        elif child.is_dir():
            sub = dst_pdir / child.name
            sub.mkdir(parents=True, exist_ok=True)
            for s in child.iterdir():
                if s.is_file():
                    (sub / s.name).write_bytes(s.read_bytes())
    stg.write_points_file(initial_staging, [dst_pdir])
    stg.write_quantum_acceptance_manifest(
        initial_staging,
        phase_name="INITIAL_AIMALL",
        iteration=0,
        accepted=[dst_pdir],
        rejected=[],
    )
    _complete_point_allocation(
        ex.campaign_dir,
        initial_staging,
        context="bootstrap",
        iteration=0,
    )

    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_ferebus_postprocess(
        state, CampaignPhase("INITIAL_FEREBUS"), observations=[],
    )
    assert result.failure_reason is None
    assert result.state_updates["models_version"] == 0
    assert result.state_updates["training_set_version"] == 0
    train_dir = tmp_path / "campaign" / "5_TRAINING" / "iteration-0000"
    assert train_dir.is_dir()
    assert (train_dir / "POINT_0000.pointdir").is_dir()


def test_ferebus_parser_rejects_empty_staging(tmp_path):
    ex = _make_executor(tmp_path)
    empty = tmp_path / "campaign" / "6_TRAINED_MODELS" / "iteration-staging"
    empty.mkdir(parents=True, exist_ok=True)
    state = SimpleNamespace(iteration=2, campaign_uid="m16-test", training_set_version=0)
    result = ex._parse_ferebus_postprocess(
        state, CampaignPhase("FEREBUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ferebus_staging_invalid" in result.failure_reason


def test_ferebus_parser_rejects_missing_staging(tmp_path):
    ex = _make_executor(tmp_path)
    # Do not create the staging dir.
    state = SimpleNamespace(iteration=2, campaign_uid="m16-test", training_set_version=0)
    result = ex._parse_ferebus_postprocess(
        state, CampaignPhase("FEREBUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ferebus_staging_invalid" in result.failure_reason



# --- ARIADNE parser tests ----------------------------------------


def _write_seeds_picked(campaign_dir, iteration, n_seeds):
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool

    campaign_dir.mkdir(parents=True, exist_ok=True)
    pool_source = campaign_dir / "_test_pool.xyz"
    frames = []
    for frame_id in range(int(n_seeds)):
        offset = 0.01 * float(frame_id)
        frames.extend([
            "3",
            "frame " + str(frame_id),
            "O " + str(offset) + " 0.0 0.0",
            "H " + str(0.96 + offset) + " 0.0 0.0",
            "H " + str(-0.24 + offset) + " 0.93 0.0",
        ])
    pool_source.write_text("\n".join(frames) + "\n", encoding="utf-8")
    pool = TrajectoryPool.import_from(
        pool_source,
        campaign_dir,
        overwrite=True,
    )
    iter_dir = (
        campaign_dir / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(iteration).zfill(4))
    )
    iter_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = list(range(int(n_seeds)))
    (iter_dir / "seeds_picked.json").write_text(
        json.dumps({
            "schema_version": 1,
            "iteration": int(iteration),
            "n_picked": int(n_seeds),
            "frame_ids": frame_ids,
            "indices": frame_ids,
            "bulk_indices": frame_ids,
            "variance_indices": [],
            "variances": [0.0 for _ in frame_ids],
            "seed_records": [
                {
                    "seed_index": int(i),
                    "frame_id": int(i),
                    "selection_index": int(i),
                    "selection_origin": "bulk",
                    "variance_at_selection": 0.0,
                }
                for i in frame_ids
            ],
            "trajectory_sha256": str(pool.sha256),
        }),
        encoding="utf-8",
    )
    return iter_dir


def _seed_ariadne_pool(campaign_dir, iteration, *, n_seeds=3):
    """Copy fixture per-seed result.json files into the iter_dir / pool."""
    _write_seeds_picked(campaign_dir, iteration, n_seeds)
    pool_dir = (
        campaign_dir / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(iteration).zfill(4)) / "pool"
    )
    pool_dir.mkdir(parents=True, exist_ok=True)
    src = FIXTURES / "ariadne_pool"
    seeds_manifest = json.loads(
        (
            campaign_dir / "7_ACTIVE_LEARNING"
            / ("iteration-" + str(iteration).zfill(4))
            / "seeds_picked.json"
        ).read_text(encoding="utf-8")
    )
    trajectory_sha256 = str(seeds_manifest["trajectory_sha256"])
    for i in range(n_seeds):
        name = "seed_" + str(i).zfill(4)
        target = pool_dir / name
        target.mkdir(exist_ok=True)
        result_src = src / name / "result.json"
        if result_src.is_file():
            payload = json.loads(result_src.read_text(encoding="utf-8"))
            payload["iteration"] = int(iteration)
            payload["seed_index"] = int(i)
            payload["seed_frame_id"] = int(i)
            payload["trajectory_sha256"] = trajectory_sha256
            payload.setdefault("task_success", True)
            payload.setdefault(
                "landing_safety",
                {
                    "accepted": True,
                    "policy": "raw_final",
                    "selected_origin": "raw_final",
                    "reasons": [],
                    "record_only_reasons": [],
                    "metrics": {
                        "max_displacement_ang": 0.02,
                        "min_pair_distance_ang": 0.90,
                    },
                },
            )
            (target / "result.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
    return pool_dir


def test_ariadne_parser_happy_path(tmp_path):
    ex = _make_executor(tmp_path)
    _seed_ariadne_pool(tmp_path / "campaign", iteration=4)
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test")
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None
    # Fixture has alpha_final values 2.13, 1.87, 2.40. Max is 2.40 (or 2.13 if
    # only seed_0000 in fixture). Just assert > 0 to be fixture-robust.
    assert result.state_updates["last_acquisition_alpha0"] > 0.0
    assert "last_n_anti_overlap_flagged" in result.state_updates
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded
    assert succeeded[-1]["phase"] == "ARIADNE_ARRAY"


def test_ariadne_parser_ignores_hidden_raw_ariadne_quality_gate_override(tmp_path):
    ex = _make_executor(tmp_path)
    ex.config.quality_gates.ariadne_max_displacement_ang = 1.0e-8
    ex.config.anti_overlap.enforce_post_ariadne = True
    ex.config.anti_overlap.max_post_ariadne_whitened_distance = 1.0e-8
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    result_path = pool / "seed_0000" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["task_success"] = True
    payload["landing_safety"] = {
        "accepted": True,
        "policy": "raw_final",
        "selected_origin": "raw_final",
        "reasons": [],
        "record_only_reasons": [],
        "metrics": {
            "max_displacement_ang": 0.01,
            "min_pair_distance_ang": 0.95,
        },
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test")

    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )

    assert result.is_complete is True
    assert result.failure_reason is None
    iter_dir = tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0004"
    manifest = json.loads((iter_dir / "ARIADNE_RESULTS.json").read_text(encoding="utf-8"))
    assert manifest["n_accepted"] == 1
    assert manifest["n_rejected"] == 0


def test_ariadne_parser_uses_result_resolved_protocol_manifest(tmp_path):
    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    iter_dir = tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0004"
    protocol_path = iter_dir / "SAMPLING_PROTOCOL_RESOLVED.json"
    protocol_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 4,
                "sampling_aggressiveness": 5,
                "resolved_quality_gates": {
                    "ariadne_max_displacement_ang": 1.0e-8,
                    "ariadne_min_pair_distance_ang": 0.60,
                },
                "resolved_adversarial_safety": {
                    "accept_legacy_missing_landing_safety": False,
                    "allow_seed_fallback": False,
                },
                "sampling_scale_model": {
                    "schema_version": 1,
                    "iteration": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    result_path = pool / "seed_0000" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["task_success"] = True
    payload["sampling_protocol"] = {
        "sampling_aggressiveness": 5,
        "resolved_manifest": str(protocol_path.resolve()),
    }
    payload["landing_safety"] = {
        "accepted": True,
        "policy": "raw_final",
        "selected_origin": "raw_final",
        "reasons": [],
        "record_only_reasons": [],
        "metrics": {
            "max_displacement_ang": 0.01,
            "min_pair_distance_ang": 0.95,
        },
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test")

    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )

    assert result.failure_reason == "ariadne_no_seed_results_parsed: 1"
    manifest = json.loads((iter_dir / "ARIADNE_RESULTS.json").read_text(encoding="utf-8"))
    assert manifest["n_accepted"] == 0
    assert manifest["rejected"][0]["reason"] == "ariadne_max_displacement_threshold_exceeded"
    audit = json.loads((iter_dir / "ARIADNE_LANDING_AUDIT.json").read_text(encoding="utf-8"))
    replay = audit["seeds"][0]["sampling_protocol_replay"]
    assert replay["used_exact_sampling_protocol"] is True
    assert replay["sampling_protocol_source"] == "result_manifest"


def test_ariadne_parser_reconstructs_missing_seed_provenance(tmp_path):
    from ichor.hpc.active_learning.versioning.provenance import (
        PROVENANCE_FILENAME,
        read_provenance,
    )

    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    seed_dir = pool / "seed_0000"
    prov_path = seed_dir / PROVENANCE_FILENAME
    assert not prov_path.exists()

    state = SimpleNamespace(iteration=4, campaign_uid="m16-test")
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )

    assert result.failure_reason is None
    assert prov_path.is_file()
    prov = read_provenance(seed_dir)
    assert prov["seed"]["frame_id"] == 0
    assert prov["ariadne"] is not None
    assert prov["anti_overlap"] is not None

    iter_dir = tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0004"
    manifest = json.loads((iter_dir / "ARIADNE_RESULTS.json").read_text(encoding="utf-8"))
    assert manifest["n_accepted"] == 1
    assert manifest["n_rejected"] == 0
    assert manifest["accepted"][0]["provenance_json"] == str(prov_path.resolve())
    audit = json.loads((iter_dir / "ARIADNE_LANDING_AUDIT.json").read_text(encoding="utf-8"))
    assert len(audit["seeds"]) == 1
    assert audit["seeds"][0]["provenance_reconstructed"] is True
    assert audit["summary"]["handoff_accepted"] == 1
    events = _read_journal_events(tmp_path / "campaign")
    assert any(e.get("event") == "ariadne_provenance_reconstructed" for e in events)


def test_ariadne_parser_accepts_safe_max_iteration_result(tmp_path):
    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    result_path = pool / "seed_0000" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["return_code"] = 1
    payload["optimiser_diagnostics"] = {
        "schema_version": 1,
        "last_return_code_reason": "max_iterations",
    }
    payload["landing_safety"] = {
        "accepted": True,
        "policy": "salvaged_iterate",
        "selected_origin": "accepted_iterate",
        "reasons": [],
        "record_only_reasons": [],
        "metrics": {},
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test")

    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )

    assert result.is_complete is True
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    salvaged = [
        e for e in events
        if e.get("event") == "ariadne_task_salvaged_from_nonzero_exit"
    ]
    assert salvaged
    assert salvaged[-1]["return_code"] == 1
    assert salvaged[-1]["reason"] == "safe_landing_after_max_iterations"


def test_ariadne_parser_rejects_explicit_unsuccessful_task(tmp_path):
    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    result_path = pool / "seed_0000" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["return_code"] = 0
    payload["task_success"] = False
    payload["task_success_reason"] = "runner_failed_after_result_write"
    payload["landing_safety"] = {
        "accepted": True,
        "policy": "raw_final",
        "selected_origin": "raw_final",
        "reasons": [],
        "record_only_reasons": [],
        "metrics": {},
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test")

    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )

    assert result.failure_reason == "ariadne_no_seed_results_parsed: 1"
    iter_dir = tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0004"
    manifest = json.loads((iter_dir / "ARIADNE_RESULTS.json").read_text(encoding="utf-8"))
    assert manifest["n_accepted"] == 0
    assert manifest["n_rejected"] == 1
    assert manifest["rejected"][0]["reason"] == (
        "ariadne_unusable:runner_failed_after_result_write"
    )
    audit = json.loads(
        (iter_dir / "ARIADNE_LANDING_AUDIT.json").read_text(encoding="utf-8")
    )
    assert audit["seeds"][0]["handoff_accepted"] is False
    assert audit["seeds"][0]["handoff_rejection_reason"] == (
        "ariadne_unusable:runner_failed_after_result_write"
    )


def test_ariadne_parser_missing_pool_dir(tmp_path):
    ex = _make_executor(tmp_path)
    # Do not seed the pool dir.
    _write_seeds_picked(tmp_path / "campaign", 4, 3)
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test")
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ariadne_pool_missing" in result.failure_reason


def test_ariadne_parser_no_seeds_in_pool(tmp_path):
    ex = _make_executor(tmp_path)
    _write_seeds_picked(tmp_path / "campaign", 4, 1)
    iter_dir = (
        tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0004" / "pool"
    )
    iter_dir.mkdir(parents=True, exist_ok=True)
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test")
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ariadne_no_seed_results_parsed" in result.failure_reason


def test_ariadne_parser_handles_missing_result_json(tmp_path):
    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4)
    # Remove one result.json so the parser sees a half-missing pool.
    (pool / "seed_0001" / "result.json").unlink()
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test")
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    # Two seeds still have a valid result -> succeeds.
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    rejected = [e for e in events if e.get("event") == "quantum_output_rejected"]
    assert any(e.get("reason") == "missing_result_json" for e in rejected)


def test_ariadne_parser_all_results_unreadable_fails(tmp_path):
    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4)
    for seed_dir in pool.iterdir():
        if seed_dir.is_dir():
            r = seed_dir / "result.json"
            if r.is_file():
                r.unlink()
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test")
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ariadne_no_seed_results_parsed" in result.failure_reason


def test_clean_stale_ariadne_seed_outputs_removes_only_results(tmp_path):
    campaign = tmp_path / "campaign"
    pool = _seed_ariadne_pool(campaign, iteration=4)
    seed_dir = pool / "seed_0000"
    tmp_result = seed_dir / "result.json.1234.abcd.tmp"
    tmp_result.write_text("partial", encoding="utf-8")
    trace = seed_dir / "ARIADNE_TRACE.jsonl"
    trace.write_text('{"event":"old"}\n', encoding="utf-8")
    protected = pool.parent / "seeds_picked.json"
    assert protected.is_file()

    removed = clean_stale_ariadne_seed_outputs(campaign, 4)

    assert str(seed_dir / "result.json") in removed
    assert str(tmp_result) in removed
    assert str(trace) in removed
    assert not (seed_dir / "result.json").exists()
    assert not tmp_result.exists()
    assert not trace.exists()
    assert protected.is_file()
    assert seed_dir.is_dir()


def test_clean_stale_ariadne_seed_outputs_rejects_non_regular_result(tmp_path):
    campaign = tmp_path / "campaign"
    pool = _seed_ariadne_pool(campaign, iteration=4)
    result = pool / "seed_0000" / "result.json"
    result.unlink()
    result.mkdir()

    with pytest.raises(BackendSubmissionError, match="non-regular ARIADNE output"):
        clean_stale_ariadne_seed_outputs(campaign, 4)


class _FakeSbatch:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")


def test_partial_array_recovery_journal_payload_does_not_duplicate_phase(
    tmp_path,
    monkeypatch,
):
    cfg = CampaignConfig()
    runner = _FakeSbatch()
    campaign = tmp_path / "campaign"
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=cfg,
        backend_check=False,
        sbatch_runner=runner,
    )
    monkeypatch.setattr(ex, "_array_size_after_staging", lambda _phase, _state: 3)
    monkeypatch.setattr(
        ex,
        "_write_real_script",
        lambda _phase, _state, _array_size, array_task_map=None: campaign / "job.sh",
    )
    monkeypatch.setattr(
        live_executor_mod,
        "prepare_retry_submission",
        lambda _campaign, _phase, _iteration: {
            "phase": "GAUSSIAN",
            "iteration": 4,
            "logical_total": 3,
            "n_complete": 1,
            "n_reuse": 1,
            "n_retry": 2,
            "force_resubmit": False,
            "retry_task_ids": [1, 2],
            "path": str(campaign / "ledger.json"),
            "retry_task_file": None,
        },
    )

    result = ex.submit_or_run(
        SimpleNamespace(iteration=4, campaign_uid="m16-test"),
        CampaignPhase("GAUSSIAN"),
    )

    assert result.submitted_job_id == "12345"
    events = _read_journal_events(campaign)
    prepared = [
        event for event in events
        if event.get("event") == "partial_array_recovery_prepared"
    ]
    assert prepared
    assert prepared[-1]["phase"] == "GAUSSIAN"
    assert prepared[-1]["iteration"] == 4
    assert prepared[-1]["n_retry"] == 2


def test_partial_array_recovery_postprocess_only_journal_payload_is_safe(
    tmp_path,
    monkeypatch,
):
    cfg = CampaignConfig()
    campaign = tmp_path / "campaign"
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=cfg,
        backend_check=False,
        sbatch_runner=_FakeSbatch(),
    )
    sentinel = PhaseResult(is_complete=True, state_updates={"ok": True})
    monkeypatch.setattr(ex, "_array_size_after_staging", lambda _phase, _state: 3)
    monkeypatch.setattr(ex, "postprocess", lambda _state, _phase, _obs: sentinel)
    monkeypatch.setattr(
        live_executor_mod,
        "prepare_retry_submission",
        lambda _campaign, _phase, _iteration: {
            "phase": "GAUSSIAN",
            "iteration": 4,
            "logical_total": 3,
            "n_complete": 3,
            "n_reuse": 3,
            "n_retry": 0,
            "force_resubmit": False,
            "retry_task_ids": [],
            "path": str(campaign / "ledger.json"),
            "retry_task_file": None,
        },
    )

    result = ex.submit_or_run(
        SimpleNamespace(iteration=4, campaign_uid="m16-test"),
        CampaignPhase("GAUSSIAN"),
    )

    assert result is sentinel
    events = _read_journal_events(campaign)
    postprocess_ready = [
        event for event in events
        if event.get("event") == "partial_array_recovery_postprocess_only"
    ]
    assert postprocess_ready
    assert postprocess_ready[-1]["phase"] == "GAUSSIAN"
    assert postprocess_ready[-1]["iteration"] == 4
    assert postprocess_ready[-1]["n_retry"] == 0


def test_ariadne_submit_reuses_complete_existing_results(tmp_path):
    from ichor.hpc.active_learning.versioning.provenance import (
        PROVENANCE_FILENAME,
        read_provenance,
    )

    cfg = CampaignConfig()
    runner = _FakeSbatch()
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
        sbatch_runner=runner,
    )
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4)
    stale_result = pool / "seed_0000" / "result.json"
    assert stale_result.is_file()

    result = ex.submit_or_run(
        SimpleNamespace(iteration=4, campaign_uid="m16-test"),
        CampaignPhase("ARIADNE_ARRAY"),
    )

    assert result.is_complete is True
    assert result.submitted_job_id is None
    assert not runner.calls
    assert stale_result.is_file()
    prov_path = pool / "seed_0000" / PROVENANCE_FILENAME
    assert prov_path.is_file()
    prov = read_provenance(pool / "seed_0000")
    assert prov["seed"]["frame_id"] == 0
    events = _read_journal_events(tmp_path / "campaign")
    reused = [
        event
        for event in events
        if event.get("event") == "partial_array_recovery_postprocess_only"
    ]
    assert reused
    assert reused[-1]["phase"] == "ARIADNE_ARRAY"


def test_ariadne_submit_rejects_invalid_existing_seed_provenance(tmp_path):
    from ichor.hpc.active_learning.versioning.provenance import write_seed_provenance

    cfg = CampaignConfig()
    runner = _FakeSbatch()
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
        sbatch_runner=runner,
    )
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    write_seed_provenance(
        pool / "seed_0000",
        campaign_uid="wrong-campaign",
        iteration=4,
        trajectory_sha256="0" * 64,
        seed_frame_id=0,
        seed_selection_origin="bulk",
        seed_variance_at_selection=0.0,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
        mode_weighting_policy="variance",
    )

    with pytest.raises(BackendSubmissionError, match="ARIADNE seed provenance invalid"):
        ex.submit_or_run(
            SimpleNamespace(iteration=4, campaign_uid="m16-test"),
            CampaignPhase("ARIADNE_ARRAY"),
        )
    assert not runner.calls



# --- POLUS parser tests ------------------------------------------


def _seed_phase_a_sample(campaign_dir, *, n_frames=2):
    """Copy fixture POLUS Phase-A xyz into 3_DIVERSITY_SAMPLING/initial/."""
    from ichor.hpc.active_learning.handoff_manifests import write_phase_a_sample_manifest

    target = campaign_dir / "3_DIVERSITY_SAMPLING" / "initial"
    target.mkdir(parents=True, exist_ok=True)
    src = FIXTURES / "polus_phase_a" / ("initial-SAMPLE-" + str(n_frames) + ".xyz")
    dst = target / src.name
    dst.write_bytes(src.read_bytes())
    index = target / ("initial-INDEX-" + str(n_frames) + ".dat")
    index.write_text("\n".join(str(i) for i in range(n_frames)) + "\n", encoding="utf-8")
    allocation_path = point_allocation_path(
        campaign_dir,
        context="bootstrap",
        iteration=0,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="m16-test",
        context="bootstrap",
        iteration=0,
        targets={
            "train": int(n_frames),
            "int_val": 0,
            "ext_val": 0,
            "total": int(n_frames),
        },
        primary_candidates=[
            {
                "candidate_id": "phase-a-candidate-" + str(i),
                "frame_id": int(i),
            }
            for i in range(int(n_frames))
        ],
        reserve_candidates=[],
    )
    primary = [
        {
            **slot["attempts"][0],
            "slot_id": int(slot["slot_id"]),
            "split": str(slot["split"]),
        }
        for slot in allocation["slots"]
    ]
    write_phase_a_sample_manifest(target, {
        "phase": "PHASE_A_POLUS",
        "iteration": -1,
        "sample_xyz": str(dst.resolve()),
        "index_path": str(index.resolve()),
        "n_select": int(n_frames),
        "n_frames": int(n_frames),
        "selected_indices": [int(i) for i in range(n_frames)],
        "descriptor": "rmsd_massweight",
        "n_pool_frames": int(n_frames),
        "bootstrap_total_size": int(n_frames),
        "point_allocation": {
            "manifest": str(allocation_path.resolve()),
            "targets": dict(allocation["targets"]),
            "primary": primary,
            "reserve_frame_ids": [],
            "reserve_count": 0,
        },
        "reserve_after_bootstrap": 0,
        "trajectory_sha256": "0" * 64,
        "source_pool_manifest": "",
    })
    return target


def _seed_phase_b_sample(campaign_dir, iteration):
    """Copy fixture POLUS Phase-B xyz into the iter_dir."""
    iter_dir = (
        campaign_dir / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(iteration).zfill(4))
    )
    iter_dir.mkdir(parents=True, exist_ok=True)
    src = FIXTURES / "polus_phase_b" / "phase_b_SAMPLE.xyz"
    dst = iter_dir / "phase_b_SAMPLE.xyz"
    dst.write_bytes(src.read_bytes())
    return iter_dir


def _write_phase_b_manifest(iter_dir, *, n_final=1, write_sample=True):
    from ichor.hpc.active_learning.handoff_manifests import (
        PHASE_B_SELECTION_SCHEMA_VERSION,
        write_phase_b_selection_manifest,
    )
    from ichor.hpc.active_learning.versioning.provenance import (
        PROVENANCE_FILENAME,
        enrich_with_point_allocation,
        write_seed_provenance,
    )

    records = []
    xyz_lines = []
    pool_dir = iter_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    for i in range(int(n_final)):
        coords = [
            [0.0, 0.0, 0.0],
            [0.96 + i * 0.01, 0.0, 0.0],
            [-0.24, 0.93 + i * 0.01, 0.0],
        ]
        seed_dir = pool_dir / ("seed_" + str(i).zfill(4))
        seed_dir.mkdir(parents=True, exist_ok=True)
        result_path = seed_dir / "result.json"
        result_path.write_text(
            json.dumps({
                "iteration": int(iter_dir.name.split("-")[-1]),
                "seed_index": int(i),
                "seed_frame_id": int(i),
                "trajectory_sha256": "0" * 64,
                "atom_types": ["O", "H", "H"],
                "final_coordinates": coords,
                "alpha_initial": 0.0,
                "alpha_final": 1.0,
                "alpha_trajectory": [0.0, 1.0],
                "n_evaluations": 1,
                "return_code": 0,
                "wall_seconds": 1.0,
            }),
            encoding="utf-8",
        )
        prov_path = seed_dir / PROVENANCE_FILENAME
        write_seed_provenance(
            seed_dir,
            campaign_uid="m16-test",
            iteration=int(iter_dir.name.split("-")[-1]),
            trajectory_sha256="0" * 64,
            seed_frame_id=i,
            seed_selection_origin="variance",
            seed_variance_at_selection=0.001,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
            mode_weighting_policy="variance",
        )
        records.append({
            "candidate_id": "phase-b-candidate-" + str(i),
            "seed_index": int(i),
            "seed_dir": str(seed_dir.resolve()),
            "result_json": str(result_path.resolve()),
            "provenance_json": str(prov_path.resolve()),
            "seed_frame_id": int(i),
            "selection_index": int(i),
            "selection_origin": "variance",
            "variance_at_selection": 0.001,
            "alpha_final": 1.0,
            "candidate_index": int(i),
            "raw_index": int(i),
            "final_index": int(i),
            "kept_after_dedup": True,
            "drop_reason": None,
        })
        xyz_lines.extend([
            "3",
            "seed_" + str(i).zfill(4),
            "O " + " ".join(str(x) for x in coords[0]),
            "H " + " ".join(str(x) for x in coords[1]),
            "H " + " ".join(str(x) for x in coords[2]),
        ])
    iteration = int(iter_dir.name.split("-")[-1])
    campaign = iter_dir.parent.parent
    allocation_path = point_allocation_path(
        campaign,
        context="active",
        iteration=iteration,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="m16-test",
        context="active",
        iteration=iteration,
        targets={
            "train": int(n_final),
            "int_val": 0,
            "ext_val": 0,
            "total": int(n_final),
        },
        primary_candidates=[
            {
                "candidate_id": str(record["candidate_id"]),
                "seed_index": int(record["seed_index"]),
                "frame_id": int(record["seed_frame_id"]),
            }
            for record in records
        ],
        reserve_candidates=[],
    )
    slot_by_candidate = {
        str(slot["attempts"][0]["candidate_id"]): slot
        for slot in allocation["slots"]
    }
    for record in records:
        slot = slot_by_candidate[str(record["candidate_id"])]
        enrich_with_point_allocation(
            Path(record["seed_dir"]),
            candidate_id=str(record["candidate_id"]),
            context="active",
            slot_id=int(slot["slot_id"]),
            split=str(slot["split"]),
        )
    if write_sample:
        (iter_dir / "phase_b_SAMPLE.xyz").write_text(
            "\n".join(xyz_lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    write_phase_b_selection_manifest(iter_dir, {
        "schema_version": PHASE_B_SELECTION_SCHEMA_VERSION,
        "iteration": int(iter_dir.name.split("-")[-1]),
        "descriptor": "hybrid_alf_rmsd",
        "source_ariadne_manifest": "",
        "point_allocation": {
            "manifest": str(allocation_path.resolve()),
            "targets": dict(allocation["targets"]),
            "reserve": [],
            "reserve_count": 0,
        },
        "n_candidates": int(n_final),
        "n_selected_raw": int(n_final),
        "n_kept": int(n_final),
        "raw": records,
        "final": records,
        "dedup": {
            "n_candidates": int(n_final),
            "n_kept": int(n_final),
            "n_dropped": 0,
            "min_separation": 0.05,
        },
    })
    return records


def test_polus_phase_a_happy_path(tmp_path):
    ex = _make_executor(tmp_path)
    _seed_phase_a_sample(tmp_path / "campaign", n_frames=2)
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_polus_postprocess(
        state, CampaignPhase("PHASE_A_POLUS"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded
    assert succeeded[-1]["phase"] == "PHASE_A_POLUS"
    assert succeeded[-1]["n_frames"] == 2


def test_polus_phase_a_missing_outdir(tmp_path):
    ex = _make_executor(tmp_path)
    # 3_DIVERSITY_SAMPLING dir does NOT yet exist. The executor __post_init__
    # in fact creates it, so synthesise a stricter no-dir scenario by binding
    # diversity_dir_name to a known-empty alternate.
    import shutil
    target = tmp_path / "campaign" / "3_DIVERSITY_SAMPLING" / "initial"
    if target.exists():
        shutil.rmtree(target.parent)
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_polus_postprocess(
        state, CampaignPhase("PHASE_A_POLUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "phase_a" in result.failure_reason


def test_polus_phase_a_no_sample_in_outdir(tmp_path):
    ex = _make_executor(tmp_path)
    # The diversity dir is created at __post_init__ time but no sample yet.
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    target = tmp_path / "campaign" / "3_DIVERSITY_SAMPLING" / "initial"
    target.mkdir(parents=True, exist_ok=True)
    result = ex._parse_polus_postprocess(
        state, CampaignPhase("PHASE_A_POLUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "phase_a_sample_manifest_invalid" in result.failure_reason


def test_polus_phase_a_uses_manifest_not_lexical_last(tmp_path):
    ex = _make_executor(tmp_path)
    outdir = _seed_phase_a_sample(tmp_path / "campaign", n_frames=2)
    stale = outdir / "initial-SAMPLE-999.xyz"
    stale.write_text("garbage no frames here", encoding="utf-8")
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_polus_postprocess(
        state, CampaignPhase("PHASE_A_POLUS"), observations=[],
    )
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded[-1]["sample_path"].endswith("initial-SAMPLE-2.xyz")


def test_polus_phase_a_manifest_count_mismatch_fails(tmp_path):
    from ichor.hpc.active_learning.handoff_manifests import write_phase_a_sample_manifest

    ex = _make_executor(tmp_path)
    outdir = _seed_phase_a_sample(tmp_path / "campaign", n_frames=2)
    sample = outdir / "initial-SAMPLE-2.xyz"
    index = outdir / "initial-INDEX-2.dat"
    existing = json.loads((outdir / "PHASE_A_SAMPLE.json").read_text(encoding="utf-8"))
    allocation = dict(existing["point_allocation"])
    allocation["primary"] = list(allocation["primary"][:1])
    write_phase_a_sample_manifest(outdir, {
        "phase": "PHASE_A_POLUS",
        "iteration": -1,
        "sample_xyz": str(sample.resolve()),
        "index_path": str(index.resolve()),
        "n_select": 1,
        "n_frames": 1,
        "selected_indices": [0],
        "descriptor": "rmsd_massweight",
        "point_allocation": allocation,
    })
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_polus_postprocess(
        state, CampaignPhase("PHASE_A_POLUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "phase_a_sample_count_mismatch" in result.failure_reason


def test_polus_phase_b_happy_path(tmp_path):
    ex = _make_executor(tmp_path)
    iter_dir = _seed_phase_b_sample(tmp_path / "campaign", iteration=5)
    _write_phase_b_manifest(iter_dir, n_final=2)
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_polus_postprocess(
        state, CampaignPhase("PHASE_B_POLUS"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded
    assert succeeded[-1]["phase"] == "PHASE_B_POLUS"


def test_polus_phase_b_missing_sample(tmp_path):
    ex = _make_executor(tmp_path)
    iter_dir = (
        tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0005"
    )
    iter_dir.mkdir(parents=True, exist_ok=True)
    _write_phase_b_manifest(iter_dir, n_final=1, write_sample=False)
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_polus_postprocess(
        state, CampaignPhase("PHASE_B_POLUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "phase_b_sample_missing" in result.failure_reason


def test_polus_phase_b_raw_sample_only_is_rejected(tmp_path):
    ex = _make_executor(tmp_path)
    iter_dir = (
        tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0005"
    )
    iter_dir.mkdir(parents=True, exist_ok=True)
    src = FIXTURES / "polus_phase_b" / "phase_b_SAMPLE.xyz"
    (iter_dir / "phase_b_SAMPLE_raw.xyz").write_bytes(src.read_bytes())
    _write_phase_b_manifest(iter_dir, n_final=1, write_sample=False)
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_polus_postprocess(
        state, CampaignPhase("PHASE_B_POLUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "phase_b_sample_missing" in result.failure_reason


def test_polus_phase_b_enriches_seed_provenance(tmp_path):
    """When Phase-B parser runs and the pool seed dirs have provenance,
    it enriches each seed with phase_b block."""
    from ichor.hpc.active_learning.versioning.provenance import (
        read_provenance, write_seed_provenance,
    )
    ex = _make_executor(tmp_path)
    iter_dir = _seed_phase_b_sample(tmp_path / "campaign", iteration=5)
    _write_phase_b_manifest(iter_dir, n_final=2)
    pool_dir = iter_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    # Write a synthetic seed_0000 with provenance.
    seed_dir = pool_dir / "seed_0000"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_seed_provenance(
        seed_dir,
        campaign_uid="m16-test",
        iteration=5,
        trajectory_sha256="0" * 64,
        seed_frame_id=42,
        seed_selection_origin="variance",
        seed_variance_at_selection=0.001,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
        mode_weighting_policy="variance",
    )
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_polus_postprocess(
        state, CampaignPhase("PHASE_B_POLUS"), observations=[],
    )
    assert result.failure_reason is None
    prov = read_provenance(seed_dir)
    assert prov["phase_b"] is not None
    assert prov["phase_b"]["selected_after_fps"] is True
    assert prov["phase_b"]["diversity_rank"] == 0


def test_polus_phase_b_unreadable_sample_fails(tmp_path):
    ex = _make_executor(tmp_path)
    iter_dir = (
        tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0005"
    )
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "phase_b_SAMPLE.xyz").write_text("garbage no frames here", encoding="utf-8")
    _write_phase_b_manifest(iter_dir, n_final=1, write_sample=False)
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_polus_postprocess(
        state, CampaignPhase("PHASE_B_POLUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "polus_sample_unreadable_or_empty" in result.failure_reason


# --- count_xyz_frames helper -------------------------------------


def test_count_xyz_frames_counts_correctly(tmp_path):
    ex = _make_executor(tmp_path)
    sample = tmp_path / "test.xyz"
    sample.write_text(
        "3\nframe 0\nO 0 0 0\nH 1 0 0\nH 0 1 0\n"
        "3\nframe 1\nO 0 0 0\nH 1.1 0 0\nH 0 1 0\n",
        encoding="utf-8",
    )
    assert ex._count_xyz_frames(sample) == 2


def test_count_xyz_frames_empty_file_returns_zero(tmp_path):
    ex = _make_executor(tmp_path)
    sample = tmp_path / "test.xyz"
    sample.write_text("", encoding="utf-8")
    assert ex._count_xyz_frames(sample) == 0


def test_count_xyz_frames_unreadable_file_returns_none(tmp_path):
    ex = _make_executor(tmp_path)
    # Nonexistent path.
    assert ex._count_xyz_frames(tmp_path / "does-not-exist.xyz") is None
