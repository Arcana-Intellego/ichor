"""M15 F14: live-mode end-to-end smoke tests.

Two flavours:

  1. `test_live_refuses_unimplemented_postprocess_*` -- POSITIVE test of
     the F1 refuse guard. These actually run on a host with `sbatch` on
     PATH (any cluster), driving the daemon through the entry phase and
     asserting that hitting the FIRST unimplemented postprocess raises
     NotImplementedError cleanly rather than silently calling the dry-run
     stub writer.

  2. `test_live_one_iter_water_tetramer_after_parsers_land` -- SKELETON
     for the success path. Skipped until M16 lands real parsers that
     register themselves in LIVE_POSTPROCESS_IMPLEMENTED. Unskipping the
     test on a CSF4 worker proves the first real iteration completes.
"""
import pytest

from ichor.hpc.active_learning.daemon.phase_executor import SBATCH_PHASES
from ichor.hpc.active_learning.daemon.state import CampaignPhase


# --- positive test of F1 refusal ---


@pytest.mark.live
def test_live_refuses_unimplemented_postprocess_for_initial_gaussian(tmp_path):
    """Live executor must raise NotImplementedError when asked to postprocess
    a SBATCH phase that isn't registered in LIVE_POSTPROCESS_IMPLEMENTED.

    Without this guard (the pre-F1 state), the daemon would silently fall
    through to DryRunPhaseExecutor.postprocess and overwrite real Gaussian
    outputs with dry-run placeholder text. Catastrophic.
    """
    pytest.importorskip("ichor.hpc.active_learning.daemon.live_executor")
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.live_executor import (
        LIVE_POSTPROCESS_IMPLEMENTED,
        LiveBackendNotAvailableError,
        LiveBackendsPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.preflight import check_backends
    from types import SimpleNamespace

    avail = check_backends()
    if not avail.sbatch:
        pytest.skip("sbatch not on PATH; live-mode smoke needs a cluster")

    cfg = CampaignConfig()
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign", config=cfg,
    )
    state = SimpleNamespace(iteration=0, campaign_uid="live-smoke")

    # all the SBATCH parsers have since landed, so INITIAL_GAUSSIAN is REGISTERED now and must NOT
    # refuse -- it dispatches to the real quantum parser, which (with no staging tree here) just
    # reports a failure_reason rather than raising NotImplementedError. the refuse guard itself is
    # still exercised for genuinely-unregistered phases by the test below.
    # (A21a -- inverted from the obsolete pre-M16 assertion.)
    assert "INITIAL_GAUSSIAN" in LIVE_POSTPROCESS_IMPLEMENTED
    result = ex.postprocess(state, CampaignPhase("INITIAL_GAUSSIAN"), observations=[])
    assert result is not None  # a PhaseResult, not a raised NotImplementedError


@pytest.mark.live
def test_live_postprocess_refused_for_all_unimplemented_sbatch_phases(tmp_path):
    """Every SBATCH phase not in LIVE_POSTPROCESS_IMPLEMENTED must refuse.
    Catches a future regression where one parser lands but its registration
    in LIVE_POSTPROCESS_IMPLEMENTED is forgotten."""
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.live_executor import (
        LIVE_POSTPROCESS_IMPLEMENTED,
        LiveBackendsPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.preflight import check_backends
    from types import SimpleNamespace

    avail = check_backends()
    if not avail.sbatch:
        pytest.skip("sbatch not on PATH; live-mode smoke needs a cluster")

    cfg = CampaignConfig()
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign", config=cfg,
    )
    state = SimpleNamespace(iteration=0, campaign_uid="live-smoke")

    for phase_name in sorted(SBATCH_PHASES):
        if phase_name in LIVE_POSTPROCESS_IMPLEMENTED:
            continue
        with pytest.raises(NotImplementedError):
            ex.postprocess(state, CampaignPhase(phase_name), observations=[])


# --- success-path skeleton (skipped until M16) ---




# ==================================================================
# M16 Day 4 P11: end-to-end live executor smoke test
# ==================================================================


def _copy_tree(src_dir, dst_dir):
    """Recursive copy used by _seed_fixture_for_phase. Avoid shutil.copytree
    here so the destination can already exist and we just overlay."""
    from pathlib import Path
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        if child.is_file():
            (dst_dir / child.name).write_bytes(child.read_bytes())
        elif child.is_dir():
            _copy_tree(child, dst_dir / child.name)


def _ensure_live_quantum_contract_names(staging_dir):
    """Older fixtures use molecule-prefixed names; live staging expects input.*."""
    from pathlib import Path

    root = Path(staging_dir)
    for pointdir in root.glob("POINT_*.pointdir"):
        for suffix, canonical in (
            ("*.wfn", "input.wfn"),
            ("*.gaussianoutput", "input.gau"),
            ("*.gjf", "input.gjf"),
        ):
            target = pointdir / canonical
            matches = sorted(pointdir.glob(suffix))
            noncanonical_matches = [old for old in matches if old.name != canonical]
            if noncanonical_matches:
                target.write_bytes(noncanonical_matches[0].read_bytes())
            elif matches and not target.exists():
                target.write_bytes(matches[0].read_bytes())
            for old in matches:
                if old.name != canonical:
                    old.unlink()


def _prune_unlisted_pointdirs(staging_dir):
    """Keep fixture staging aligned with the submitted POINTS.txt array."""
    from pathlib import Path
    import shutil

    root = Path(staging_dir)
    points = root / "POINTS.txt"
    if not points.is_file():
        return
    allowed = {Path(line.strip()).name for line in points.read_text(encoding="utf-8").splitlines() if line.strip()}
    for pointdir in root.glob("POINT_*.pointdir"):
        if pointdir.name not in allowed:
            shutil.rmtree(pointdir)


def _live_smoke_fixtures():
    """Return the absolute path to the fixture pack."""
    from pathlib import Path
    return Path(__file__).resolve().parent / "fixtures" / "live_outputs"


def _write_loadable_ferebus_model(path, *, atom, alf, ntrain=5, nfeats=3):
    rows = [
        [0.1 + i * 0.1 + j * 0.01 for j in range(nfeats)]
        for i in range(ntrain)
    ]
    lines = [
        "# jitter 1.0e-6",
        "# likelihood -1.0",
        "",
        "[system]",
        "name WATER",
        "atom " + atom,
        "property iqa",
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
        "units.x bohr bohr radians",
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


def _seed_pyferebus_manifest_staging(campaign_dir, training_version=0):
    """Seed the pyferebus-owned staging layout expected by live validators."""
    import json
    import shutil
    from pathlib import Path

    from ichor.hpc.active_learning.daemon import input_staging as stg

    target = Path(campaign_dir) / "6_TRAINED_MODELS" / "iteration-staging"
    if target.exists():
        shutil.rmtree(target)
    specs = {
        "O1": (1, 2, 3),
        "H2": (2, 1, 3),
        "H3": (3, 1, 2),
    }
    tasks = []
    for idx, (atom, alf) in enumerate(specs.items(), start=1):
        model_dir = target / "iqa" / atom
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_dir / "ferebus.config"
        config_path.write_text("name WATER\nproperties [\"iqa\"]\n", encoding="utf-8")
        model_path = model_dir / ("WATER_iqa_" + atom + ".model")
        _write_loadable_ferebus_model(model_path, atom=atom, alf=alf)
        train_csv = model_dir / ("WATER_" + atom + "_TRAINING_SET.csv")
        int_csv = model_dir / ("WATER_" + atom + "_INT_VALIDATION_SET.csv")
        ext_csv = model_dir / ("WATER_" + atom + "_EXT_VALIDATION_SET.csv")
        _write_ferebus_metric_csv(train_csv, 5)
        _write_ferebus_metric_csv(int_csv, 2)
        _write_ferebus_metric_csv(ext_csv, 2)
        tasks.append({
            "task_index": idx,
            "property": "iqa",
            "atom": atom,
            "alf_1_indexed": list(alf),
            "config_path": str(config_path),
            "expected_model_path": str(model_path),
            "training_csv": str(train_csv),
            "int_validation_csv": str(int_csv),
            "ext_validation_csv": str(ext_csv),
            "row_counts": {"train": 5, "int_val": 2, "ext_val": 2},
        })
    (target / stg.FEREBUS_TASK_MANIFEST).write_text(
        json.dumps({
            "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
            "system": "WATER",
            "training_version": int(training_version),
            "properties": ["iqa"],
            "atoms": list(specs),
            "n_tasks": len(tasks),
            "tasks": tasks,
        }),
        encoding="utf-8",
    )
    for name in (stg.FEREBUS_JOB_DETAILS, "commands", "list.txt", "runFerebus.sh"):
        (target / name).write_text(name + "\n", encoding="utf-8")
    return target


def _write_ferebus_metric_csv(path, n_rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("f1,f2,f3,iqa\n")
        for i in range(int(n_rows)):
            f.write(f"{0.1+i*0.1},{0.2+i*0.1},{0.3+i*0.1},0.0\n")


def _ensure_live_trajectory_pool(campaign_dir):
    """Import a tiny pool so live SEED_SELECT can emit seeds_picked.json."""
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool

    source = _live_smoke_fixtures() / "polus_phase_a" / "initial-SAMPLE-2.xyz"
    TrajectoryPool.import_from(
        source,
        campaign_dir,
        overwrite=True,
        outlier_filter_enabled=False,
    )


def _patch_ferebus_submit_for_live_smoke(monkeypatch, campaign_dir, call_log):
    """Patch only the expensive pyferebus/FEREBUS submit seam for fixture tests."""
    from pathlib import Path

    from ichor.hpc.active_learning.daemon import input_staging as stg
    from ichor.hpc.active_learning.submit import pyferebus_wrap
    from ichor.hpc.active_learning.submit.pyferebus_wrap import FerebusSubmission

    calls = {"n": 0}

    def fake_stage(campaign_dir_arg, config, training_version, *, is_initial=False):
        staging = _seed_pyferebus_manifest_staging(
            campaign_dir_arg,
            training_version=0 if is_initial else int(training_version),
        )
        return staging, 1

    def fake_submit(jd_file, working_directory, **kwargs):
        working = Path(working_directory)
        script = working / "runFerebus.sh"
        script.write_text("#!/bin/sh\n# fixture pyferebus submit\n", encoding="utf-8")
        phase_name = "INITIAL_FEREBUS" if calls["n"] == 0 else "FEREBUS"
        if phase_name == "INITIAL_FEREBUS":
            _ensure_live_trajectory_pool(campaign_dir)
        call_log.append((phase_name, 0))
        calls["n"] += 1
        assert kwargs["overwrite_workdir"] is False
        assert kwargs["move_dataset_files"] is True
        assert str(kwargs["expected_job_name"]).endswith("-" + phase_name + "-0")
        assert int(kwargs["expected_tasks"]) >= 1
        return FerebusSubmission(
            job_id=str(19000 + calls["n"]),
            cluster=None,
            submission_script=script,
            working_dir=working,
            transfer_learning=False,
        )

    monkeypatch.setattr(stg, "stage_ferebus_inputs", fake_stage)
    monkeypatch.setattr(pyferebus_wrap, "submit_ferebus", fake_submit)


def _count_xyz_frames(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    count = 0
    while i < len(lines):
        try:
            natoms = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        i += 2 + natoms
        count += 1
    return count


def _annotate_ariadne_fixture_results(campaign_dir, iteration):
    import json as _json

    iter_dir = (
        campaign_dir / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(iteration)).zfill(4))
    )
    seeds_path = iter_dir / "seeds_picked.json"
    if not seeds_path.is_file():
        return
    picked = _json.loads(seeds_path.read_text(encoding="utf-8"))
    records = picked.get("seed_records") or []
    pool_dir = iter_dir / "pool"
    for rec in records:
        seed_index = int(rec["seed_index"])
        result_path = pool_dir / ("seed_" + str(seed_index).zfill(4)) / "result.json"
        if not result_path.is_file():
            continue
        data = _json.loads(result_path.read_text(encoding="utf-8"))
        data["iteration"] = int(iteration)
        data["seed_index"] = seed_index
        data["seed_frame_id"] = int(rec["frame_id"])
        data["trajectory_sha256"] = str(picked.get("trajectory_sha256", ""))
        result_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")


def _write_phase_b_selection_for_live_smoke(campaign_dir, iteration):
    import json as _json
    from pathlib import Path
    from ichor.hpc.active_learning.handoff_manifests import (
        PHASE_B_SELECTION_SCHEMA_VERSION,
        read_ariadne_results_manifest,
        write_phase_b_selection_manifest,
    )

    iter_dir = (
        campaign_dir / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(iteration)).zfill(4))
    )
    final_sample = iter_dir / "phase_b_SAMPLE.xyz"
    n_final = _count_xyz_frames(final_sample)
    ariadne_manifest = read_ariadne_results_manifest(
        iter_dir,
        expected_iteration=int(iteration),
    )
    records = []
    xyz_lines = []
    for final_index, source in enumerate(ariadne_manifest["accepted"][:n_final]):
        rec = dict(source)
        rec["candidate_index"] = int(final_index)
        rec["raw_index"] = int(final_index)
        rec["final_index"] = int(final_index)
        rec["kept_after_dedup"] = True
        rec["drop_reason"] = None
        records.append(rec)
        result = _json.loads(Path(str(rec["result_json"])).read_text(encoding="utf-8"))
        atom_types = [str(x) for x in result["atom_types"]]
        coords = result["final_coordinates"]
        xyz_lines.append(str(len(atom_types)))
        xyz_lines.append("fixture phase_b final " + str(final_index))
        for atom, coord in zip(atom_types, coords):
            xyz_lines.append(
                "{atom} {x:.6f} {y:.6f} {z:.6f}".format(
                    atom=atom,
                    x=float(coord[0]),
                    y=float(coord[1]),
                    z=float(coord[2]),
                )
            )
    final_sample.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
    write_phase_b_selection_manifest(iter_dir, {
        "schema_version": PHASE_B_SELECTION_SCHEMA_VERSION,
        "iteration": int(iteration),
        "descriptor": "hybrid_alf_rmsd",
        "source_ariadne_manifest": str((iter_dir / "ARIADNE_RESULTS.json").resolve()),
        "n_candidates": int(len(ariadne_manifest["accepted"])),
        "n_selected_raw": int(len(records)),
        "n_kept": int(len(records)),
        "raw": records,
        "final": records,
        "dedup": {
            "n_candidates": int(len(records)),
            "n_kept": int(len(records)),
            "n_dropped": 0,
            "min_separation": 0.05,
        },
    })


def _live_smoke_seed_for_phase(campaign_dir, phase_name, iteration):
    """Pre-seed the canonical staging dir for the given phase + iteration.

    Called by the mocked sbatch_runner so each phase finds its expected
    artefacts at postprocess time. Returns silently for unknown phases.
    """
    from pathlib import Path
    fixtures = _live_smoke_fixtures()
    campaign_dir = Path(campaign_dir)
    iter4 = "iteration-" + str(int(iteration)).zfill(4)

    if phase_name == "PHASE_A_POLUS":
        from ichor.hpc.active_learning.handoff_manifests import write_phase_a_sample_manifest

        target = campaign_dir / "3_DIVERSITY_SAMPLING" / "initial"
        target.mkdir(parents=True, exist_ok=True)
        src = fixtures / "polus_phase_a"
        for f in src.iterdir():
            if f.is_file():
                (target / f.name).write_bytes(f.read_bytes())
        sample = target / "initial-SAMPLE-2.xyz"
        index = target / "initial-INDEX-2.dat"
        if not index.is_file():
            index.write_text("0\n1\n", encoding="utf-8")
        write_phase_a_sample_manifest(target, {
            "phase": "PHASE_A_POLUS",
            "iteration": -1,
            "sample_xyz": str(sample.resolve()),
            "index_path": str(index.resolve()),
            "n_select": 2,
            "n_frames": 2,
            "selected_indices": [0, 1],
            "descriptor": "rmsd_massweight",
            "n_pool_frames": 2,
            "trajectory_sha256": "0" * 64,
            "source_pool_manifest": "",
            "initial_train_size": 1,
            "initial_val_size": 1,
        })
    elif phase_name in ("INITIAL_GAUSSIAN", "INITIAL_AIMALL"):
        target = campaign_dir / ".DATA" / "STAGING" / "initial"
        _copy_tree(fixtures / "initial_quantum", target)
        _ensure_live_quantum_contract_names(target)
        _prune_unlisted_pointdirs(target)
    elif phase_name in ("GAUSSIAN", "AIMALL"):
        target = (
            campaign_dir / ".DATA" / "STAGING" / ("iter_" + str(int(iteration)))
        )
        _copy_tree(fixtures / "iter_quantum", target)
        _ensure_live_quantum_contract_names(target)
        _prune_unlisted_pointdirs(target)
    elif phase_name in ("INITIAL_FEREBUS", "FEREBUS"):
        _seed_pyferebus_manifest_staging(
            campaign_dir,
            training_version=0 if phase_name == "INITIAL_FEREBUS" else 1,
        )
    elif phase_name == "ARIADNE_ARRAY":
        target = campaign_dir / "7_ACTIVE_LEARNING" / iter4 / "pool"
        _copy_tree(fixtures / "ariadne_pool", target)
        _annotate_ariadne_fixture_results(campaign_dir, iteration)
    elif phase_name == "PHASE_B_POLUS":
        target = campaign_dir / "7_ACTIVE_LEARNING" / iter4
        target.mkdir(parents=True, exist_ok=True)
        src = fixtures / "polus_phase_b"
        for f in src.iterdir():
            if f.is_file():
                (target / f.name).write_bytes(f.read_bytes())
        _write_phase_b_selection_for_live_smoke(campaign_dir, iteration)


def _build_live_smoke_sbatch_runner(campaign_dir, call_log):
    """Return a callable suitable for LiveBackendsPhaseExecutor.sbatch_runner.

    The returned callable seeds the staging dir for the inferred phase
    before returning a fake CompletedProcess with a synthetic JobID.
    """
    from pathlib import Path
    from types import SimpleNamespace

    def runner(cmd, **kwargs):
        script = Path(cmd[-1])
        stem = script.stem  # e.g. "PHASE_A_POLUS-0"
        try:
            phase_name, _, iter_part = stem.rpartition("-")
            iteration = int(iter_part)
        except ValueError:
            phase_name = stem
            iteration = 0
        call_log.append((phase_name, iteration))
        _live_smoke_seed_for_phase(campaign_dir, phase_name, iteration)
        n = len(call_log)
        return SimpleNamespace(
            stdout=str(10000 + n) + "\n",
            stderr="",
            returncode=0,
        )
    return runner


def _build_live_smoke_poller():
    """Return a fake sacct poller that says every job COMPLETED.

    Drop-in for Daemon.sacct_poller; safe to call for any JobID.
    """
    from ichor.hpc.active_learning.submit.sacct_poll import (
        JobObservation, JobStatus,
    )

    def poller(job_id, **kwargs):
        expected = int(kwargs.get("expected_task_count") or 1)
        if expected <= 1:
            return [JobObservation(
                job_id=str(job_id),
                status=JobStatus.COMPLETED,
                exit_code=(0, 0),
                elapsed_seconds=0,
            )]
        return [
            JobObservation(
                job_id=str(job_id) + "_" + str(idx),
                status=JobStatus.COMPLETED,
                exit_code=(0, 0),
                elapsed_seconds=0,
            )
            for idx in range(expected)
        ]
    return poller



@pytest.mark.live
def test_live_one_iter_water_tetramer_after_parsers_land(tmp_path, monkeypatch):
    """End-to-end live executor smoke test.

    After M16 Day 3, all 9 SBATCH phases have real parsers registered in
    LIVE_POSTPROCESS_IMPLEMENTED. This test drives Daemon.run() with a
    LiveBackendsPhaseExecutor whose sbatch_runner is a phase-aware mock
    that pre-seeds the canonical staging dirs from the fixture pack, and
    a stub sacct poller that says every job COMPLETED.

    Asserts:
      * Daemon finishes a 1-iteration campaign with rc == 0.
      * No NotImplementedError raised by any live postprocess.
      * Every SBATCH phase fired sbatch (a non-empty call log).
      * Final state.phase is DONE and training_set_version >= 0.
      * Final state.iteration shows the iteration loop ran at least once.
      * journal contains at least one phase_succeeded_live event.
    """
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.live_executor import (
        LIVE_POSTPROCESS_IMPLEMENTED,
        LiveBackendsPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.phase_executor import SBATCH_PHASES
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, read_state,
    )
    from ichor.hpc.active_learning.daemon.journal import iter_events

    # Sanity: all SBATCH phases must be implemented after M16 Day 3.
    assert SBATCH_PHASES.issubset(LIVE_POSTPROCESS_IMPLEMENTED)

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cfg = CampaignConfig(max_iterations=1, poll_interval_seconds=1)
    cfg.batch_sizing.floor = 2
    cfg.seed_selection.n_seeds_per_iteration = 2

    call_log = []
    _patch_ferebus_submit_for_live_smoke(monkeypatch, campaign, call_log)
    sbatch_runner = _build_live_smoke_sbatch_runner(campaign, call_log)
    poller = _build_live_smoke_poller()

    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=cfg,
        sbatch_runner=sbatch_runner,
        backend_check=False,
    )
    d = Daemon(
        campaign_dir=campaign,
        config=cfg,
        executor=ex,
        sacct_poller=poller,
        sleep_fn=lambda s: None,
    )
    rc = d.run(max_ticks=500, catch_keyboard_interrupt=False)
    assert rc == 0, "daemon exited non-zero: " + str(rc)

    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.DONE, (
        "expected DONE, got " + str(state.phase)
    )
    # max_iterations=1 runs the iteration loop exactly once at iter 0
    # then transitions DONE. iteration stays at 0 by convention.
    assert int(state.iteration) >= 0
    # INITIAL_FEREBUS commits initial models 0 + iteration-1 FEREBUS
    # commits models 1, so both versions should be at least 1.
    assert state.training_set_version >= 1, (
        "training_set_version=" + str(state.training_set_version)
        + "; expected at least 1 (initial + 1 iter)"
    )
    assert state.models_version >= 1, (
        "models_version=" + str(state.models_version)
        + "; expected at least 1 (INITIAL_FEREBUS + FEREBUS)"
    )

    # Every SBATCH phase fired sbatch at least once.
    phases_called = sorted({p for p, _ in call_log})
    assert set(phases_called) == set(SBATCH_PHASES), (
        "unexpected SBATCH coverage; got "
        + str(phases_called) + " missing "
        + str(sorted(set(SBATCH_PHASES) - set(phases_called)))
    )

    # Live parsers emitted phase_succeeded_live events.
    journal_path = (
        campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    )
    events = list(iter_events(journal_path))
    live_succeeded = [
        e for e in events if e.get("event") == "phase_succeeded_live"
    ]
    assert live_succeeded, (
        "no phase_succeeded_live events emitted; live parsers may have "
        "silently fallen through. Saw events: "
        + str(sorted({e.get("event") for e in events}))
    )



def _build_no_wd_sbatch_runner(campaign_dir, call_log):
    """Like the standard live-smoke runner but strips the
    whitened_distance_final key from each ARIADNE seed result.json
    immediately after the standard seeder writes them.

    Used by the fallback-path test to verify the live parser copes when
    the field is missing -- older runs, hand-crafted JSON, or any
    fixture pack predating Phase B.
    """
    import json as _json
    from pathlib import Path
    from types import SimpleNamespace

    def runner(cmd, **kwargs):
        script = Path(cmd[-1])
        stem = script.stem
        try:
            phase_name, _, iter_part = stem.rpartition("-")
            iteration = int(iter_part)
        except ValueError:
            phase_name = stem
            iteration = 0
        call_log.append((phase_name, iteration))
        _live_smoke_seed_for_phase(campaign_dir, phase_name, iteration)

        # for ARIADNE_ARRAY, strip the whitened_distance_final key from
        # every per-seed result.json so the parser is forced through the
        # synthetic fallback branch.
        if phase_name == "ARIADNE_ARRAY":
            iter4 = "iteration-" + str(iteration).zfill(4)
            pool = (
                campaign_dir / "7_ACTIVE_LEARNING"
                / iter4 / "pool"
            )
            if pool.is_dir():
                for sd in pool.iterdir():
                    rj = sd / "result.json"
                    if rj.is_file():
                        try:
                            data = _json.loads(rj.read_text(encoding="utf-8"))
                            data.pop("whitened_distance_final", None)
                            rj.write_text(_json.dumps(data, indent=2), encoding="utf-8")
                        except (OSError, ValueError):
                            pass

        n = len(call_log)
        return SimpleNamespace(
            stdout=str(10000 + n) + chr(10),
            stderr="",
            returncode=0,
        )
    return runner


@pytest.mark.live
def test_live_one_iter_whitened_distance_fallback(tmp_path, monkeypatch):
    """Same end-to-end flow as the main F14 test, but each ARIADNE
    result.json has whitened_distance_final stripped. The daemon should
    still complete cleanly -- the live parser falls back to the synthetic
    alpha-delta proxy in _synthetic_whitened_distance.
    """
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.live_executor import (
        LiveBackendsPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, read_state,
    )

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cfg = CampaignConfig(max_iterations=1, poll_interval_seconds=1)
    cfg.batch_sizing.floor = 2
    cfg.seed_selection.n_seeds_per_iteration = 2

    call_log = []
    _patch_ferebus_submit_for_live_smoke(monkeypatch, campaign, call_log)
    runner = _build_no_wd_sbatch_runner(campaign, call_log)
    poller = _build_live_smoke_poller()

    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign, config=cfg,
        sbatch_runner=runner, backend_check=False,
    )
    d = Daemon(
        campaign_dir=campaign, config=cfg,
        executor=ex, sacct_poller=poller,
        sleep_fn=lambda s: None,
    )
    rc = d.run(max_ticks=500, catch_keyboard_interrupt=False)
    assert rc == 0, "daemon exited non-zero: " + str(rc)

    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.DONE
    # the daemon still progresses; the fallback path is exercised inside
    # _parse_ariadne_array_postprocess without us needing to inspect the
    # actual whitened-distance values here.


def _build_both_phase_b_sbatch_runner(campaign_dir, call_log):
    """Like the standard live-smoke runner but after PHASE_B_POLUS
    standard seeding, additionally writes a phase_b_SAMPLE_raw.xyz with
    a deliberately different frame count.

    Used by the dedup-naming test to verify that the parser picks the
    dedup-filtered file (phase_b_SAMPLE.xyz) rather than the raw one.
    """
    from pathlib import Path
    from types import SimpleNamespace

    def runner(cmd, **kwargs):
        script = Path(cmd[-1])
        stem = script.stem
        try:
            phase_name, _, iter_part = stem.rpartition("-")
            iteration = int(iter_part)
        except ValueError:
            phase_name = stem
            iteration = 0
        call_log.append((phase_name, iteration))
        _live_smoke_seed_for_phase(campaign_dir, phase_name, iteration)

        if phase_name == "PHASE_B_POLUS":
            iter4 = "iteration-" + str(iteration).zfill(4)
            iter_dir = campaign_dir / "7_ACTIVE_LEARNING" / iter4
            # The fixture pack copies a 2-frame phase_b_SAMPLE.xyz from
            # polus_phase_b. add a 4-frame raw file alongside it -- if the
            # parser preferred the raw one, the journal would record 4
            # frames; if it prefers SAMPLE.xyz it records 2.
            raw = iter_dir / "phase_b_SAMPLE_raw.xyz"
            xyz_lines = []
            for k in range(4):
                xyz_lines.append("3")
                xyz_lines.append("raw frame " + str(k))
                xyz_lines.append("O 0.0 0.0 0.0")
                xyz_lines.append("H 0.96 0.0 0.0")
                xyz_lines.append("H -0.24 0.93 0.0")
            raw.write_text(chr(10).join(xyz_lines) + chr(10), encoding="utf-8")

        n = len(call_log)
        return SimpleNamespace(
            stdout=str(10000 + n) + chr(10),
            stderr="",
            returncode=0,
        )
    return runner


@pytest.mark.live
def test_live_one_iter_prefers_dedup_filtered_sample(tmp_path, monkeypatch):
    """When both phase_b_SAMPLE.xyz and phase_b_SAMPLE_raw.xyz exist
    (with different frame counts), the live parser should pick the
    dedup-filtered SAMPLE.xyz -- that is the canonical name the daemon
    main always writes; raw is only retained as a debugging artefact.
    """
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.live_executor import (
        LiveBackendsPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, read_state,
    )
    from ichor.hpc.active_learning.daemon.journal import iter_events

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cfg = CampaignConfig(max_iterations=1, poll_interval_seconds=1)
    cfg.batch_sizing.floor = 2
    cfg.seed_selection.n_seeds_per_iteration = 2

    call_log = []
    _patch_ferebus_submit_for_live_smoke(monkeypatch, campaign, call_log)
    runner = _build_both_phase_b_sbatch_runner(campaign, call_log)
    poller = _build_live_smoke_poller()

    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign, config=cfg,
        sbatch_runner=runner, backend_check=False,
    )
    d = Daemon(
        campaign_dir=campaign, config=cfg,
        executor=ex, sacct_poller=poller,
        sleep_fn=lambda s: None,
    )
    rc = d.run(max_ticks=500, catch_keyboard_interrupt=False)
    assert rc == 0
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.DONE

    # inspect the journal for PHASE_B_POLUS phase_succeeded_live event.
    # its sample_path field should end with phase_b_SAMPLE.xyz (the
    # dedup-filtered file from the fixture), NOT phase_b_SAMPLE_raw.xyz.
    journal_path = (
        campaign / ".DATA" / "ACTIVE_LEARNING"
        / "journal.ndjson"
    )
    events = list(iter_events(journal_path))
    phase_b_events = [
        e for e in events
        if e.get("event") == "phase_succeeded_live"
        and e.get("phase") == "PHASE_B_POLUS"
    ]
    assert phase_b_events, "no PHASE_B_POLUS phase_succeeded_live event"
    last = phase_b_events[-1]
    sample_path = str(last.get("sample_path"))
    assert sample_path.endswith("phase_b_SAMPLE.xyz"), (
        "expected SAMPLE.xyz but got " + sample_path
    )
    # also confirm the fixture-supplied frame count (2) was parsed, not the
    # 4-frame raw file we deliberately also wrote.
    assert int(last.get("n_frames", 0)) == 2
