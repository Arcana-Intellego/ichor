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
from ichor.hpc.active_learning.daemon.live_executor import (
    LIVE_POSTPROCESS_IMPLEMENTED,
    LiveBackendsPhaseExecutor,
    clean_stale_ariadne_seed_outputs,
)
from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.daemon.phase_executor import (
    BackendSubmissionError,
    PhaseResult,
)
from ichor.hpc.active_learning.daemon.state import CampaignPhase
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


def test_above_threshold_rejection_sets_failure_reason(tmp_path):
    ex = _make_executor(tmp_path, failure_threshold=0.3)
    _bind_staging(ex, FIXTURES / "iter_quantum_scf_failure")
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_quantum_postprocess(
        state, CampaignPhase("GAUSSIAN"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is not None
    assert "too_many_rejected" in result.failure_reason
    assert "1/2" in result.failure_reason


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


def test_commit_initial_training_set_uses_aimall_acceptance_manifest(tmp_path):
    campaign = tmp_path / "campaign"
    initial = campaign / ".DATA" / "STAGING" / "initial"
    good = initial / "POINT_0001.pointdir"
    bad = initial / "POINT_0000.pointdir"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    (good / "accepted.txt").write_text("good\n", encoding="utf-8")
    (bad / "rejected.txt").write_text("bad\n", encoding="utf-8")
    stg.write_quantum_acceptance_manifest(
        initial,
        phase_name="INITIAL_AIMALL",
        iteration=0,
        accepted=[good],
        rejected=[("POINT_0000.pointdir", "missing_atomicfiles_dir")],
    )

    assert stg.commit_initial_training_set(campaign) is True
    committed = campaign / "5_TRAINING" / "iteration-0000"
    assert (committed / "POINT_0001.pointdir" / "accepted.txt").is_file()
    assert not (committed / "POINT_0000.pointdir").exists()


def test_live_append_requires_aimall_acceptance_manifest(tmp_path):
    ex = _make_executor(tmp_path)
    v = TrainingSetVersioning(ex.campaign_dir / "5_TRAINING")
    staging = v.stage(source_version=None, target_version=0)
    (staging / "POINT_0000.pointdir").mkdir()
    v.commit(0)
    v.update_current(0)
    live_staging = stg.bucket_dir(ex.campaign_dir, "APPEND", 0)
    (live_staging / "POINT_0000.pointdir").mkdir(parents=True)
    state = SimpleNamespace(iteration=0, campaign_uid="uid", training_set_version=0)

    with pytest.raises(BackendSubmissionError, match="acceptance manifest"):
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
    stg.write_quantum_acceptance_manifest(
        live_staging,
        phase_name="AIMALL",
        iteration=0,
        accepted=[new_point],
        rejected=[],
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
    state = SimpleNamespace(iteration=3, campaign_uid="m16-test")
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
    events = _read_journal_events(tmp_path / "campaign")
    assert any(e.get("event") == "models_committed" for e in events)


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
    stg.write_quantum_acceptance_manifest(
        initial_staging,
        phase_name="INITIAL_AIMALL",
        iteration=0,
        accepted=[dst_pdir],
        rejected=[],
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
    state = SimpleNamespace(iteration=2, campaign_uid="m16-test")
    result = ex._parse_ferebus_postprocess(
        state, CampaignPhase("FEREBUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ferebus_staging_invalid" in result.failure_reason



# --- ARIADNE parser tests ----------------------------------------


def _write_seeds_picked(campaign_dir, iteration, n_seeds):
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
            "trajectory_sha256": "0" * 64,
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


def test_ariadne_submit_cleans_stale_results_before_sbatch(tmp_path):
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

    assert result.submitted_job_id == "12345"
    assert runner.calls
    assert not stale_result.exists()
    events = _read_journal_events(tmp_path / "campaign")
    cleaned = [e for e in events if e.get("event") == "ariadne_stale_outputs_cleaned"]
    assert cleaned
    assert cleaned[-1]["removed"] >= 1



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
        "trajectory_sha256": "0" * 64,
        "source_pool_manifest": "",
        "initial_train_size": int(n_frames),
        "initial_val_size": 0,
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


def _write_phase_b_manifest(iter_dir, *, n_final=1):
    from ichor.hpc.active_learning.handoff_manifests import (
        PHASE_B_SELECTION_SCHEMA_VERSION,
        write_phase_b_selection_manifest,
    )
    from ichor.hpc.active_learning.versioning.provenance import (
        PROVENANCE_FILENAME,
        write_seed_provenance,
    )

    records = []
    pool_dir = iter_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    for i in range(int(n_final)):
        seed_dir = pool_dir / ("seed_" + str(i).zfill(4))
        seed_dir.mkdir(parents=True, exist_ok=True)
        result_path = seed_dir / "result.json"
        result_path.write_text(
            json.dumps({
                "iteration": int(iter_dir.name.split("-")[-1]),
                "seed_index": int(i),
                "seed_frame_id": int(i),
                "atom_types": ["O", "H", "H"],
                "final_coordinates": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
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
    write_phase_b_selection_manifest(iter_dir, {
        "schema_version": PHASE_B_SELECTION_SCHEMA_VERSION,
        "iteration": int(iter_dir.name.split("-")[-1]),
        "descriptor": "hybrid_alf_rmsd",
        "source_ariadne_manifest": "",
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
    write_phase_a_sample_manifest(outdir, {
        "phase": "PHASE_A_POLUS",
        "iteration": -1,
        "sample_xyz": str(sample.resolve()),
        "index_path": str(index.resolve()),
        "n_select": 1,
        "n_frames": 1,
        "selected_indices": [0],
        "descriptor": "rmsd_massweight",
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
    _write_phase_b_manifest(iter_dir, n_final=1)
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
    _write_phase_b_manifest(iter_dir, n_final=1)
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
    _write_phase_b_manifest(iter_dir, n_final=1)
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
