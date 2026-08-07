"""M16 Day 2: unit tests for _parse_quantum_postprocess.

Strategy: drive the parser directly with hand-constructed states + phases,
overriding _quantum_staging_path to point at the fixture pack rather than
materialising a campaign filesystem.
"""
import hashlib
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
from ichor.hpc.active_learning.daemon import submission_intent
from ichor.hpc.active_learning.daemon.phase_executor import (
    BackendSubmissionError,
    PhaseResult,
)
from ichor.hpc.active_learning.daemon.state import CampaignPhase, atomic_write_json
from ichor.hpc.active_learning.daemon.ariadne_publication import (
    ariadne_publication_archive_root,
)
from ichor.hpc.active_learning.daemon.script_bundles import (
    prepare_attempt_bundle,
    write_attempt_script,
    write_script_binding,
)
from ichor.hpc.active_learning.daemon.ferebus_quality import (
    FEREBUS_QUALITY_SCHEMA_VERSION,
)
from ichor.hpc.active_learning.handoff_manifests import (
    ariadne_landing_audit_path,
    ariadne_results_path,
    seeds_picked_path,
)
from ichor.hpc.active_learning.layout import (
    active_ariadne_dir,
    active_iteration_dir,
    ariadne_seed_dir,
    ariadne_seeds_dir,
    bootstrap_selection_dir,
)
from ichor.hpc.active_learning.sampling_protocol import sampling_protocol_resolved_path
from ichor.hpc.active_learning.point_allocation import (
    create_point_allocation,
    pending_attempts,
    point_allocation_path,
    read_point_allocation,
    record_quantum_results,
)
from ichor.hpc.active_learning.versioning.provenance import (
    enrich_with_point_allocation,
    write_seed_provenance,
)
from ichor.hpc.active_learning.versioning.versioned_directory import VersionedDirectory
from ichor.hpc.active_learning.versioning.reference_data import (
    ReferenceDataVersioning,
)
from ichor.hpc.active_learning.versioning.manifest import ManifestMismatchError
from ichor.hpc.active_learning.versioning.trained_models import (
    TrainedModelVersioning,
)


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


def test_aimall_postprocess_source_bypasses_staging_and_scheduler_submission(
    tmp_path,
    monkeypatch,
):
    executor = _make_executor(tmp_path)
    state = SimpleNamespace(
        campaign_uid="aimall-postprocess-test",
        iteration=14,
        replacement_round=0,
    )
    source = {
        "logical_total": 149,
        "job_id": "17888108",
    }
    monkeypatch.setattr(
        submission_intent,
        "load_active_intent",
        lambda *_args, **_kwargs: {"postprocess_source": {"source": "fixture"}},
    )
    monkeypatch.setattr(
        submission_intent,
        "resolve_aimall_postprocess_source",
        lambda *_args, **_kwargs: dict(source),
    )
    expected = PhaseResult(is_complete=True)
    monkeypatch.setattr(
        executor,
        "postprocess",
        lambda observed_state, observed_phase, observations: (
            expected
            if (
                observed_state is state
                and observed_phase is CampaignPhase.AIMALL
                and observations == []
            )
            else (_ for _ in ()).throw(
                AssertionError("unexpected postprocess invocation")
            )
        ),
    )
    executor.sbatch_runner = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("scheduler submission must not run")
    )
    monkeypatch.setattr(
        executor,
        "_stage_phase_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("AIMAll input staging must not run")
        ),
        raising=False,
    )

    result = executor.submit_or_run(state, CampaignPhase.AIMALL)

    assert result is expected


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


def _install_quantum_intent(ex, state, phase):
    phase_name = phase.value if isinstance(phase, CampaignPhase) else str(phase)
    staging = ex._quantum_staging_path(state, phase_name)
    expected_tasks = max(1, len(list(Path(staging).glob("POINT_*.pointdir"))))
    submission_intent.write_pre_submit_intent(
        ex.campaign_dir,
        campaign_uid=str(state.campaign_uid),
        phase_name=phase_name,
        iteration=int(state.iteration),
        expected_tasks=expected_tasks,
    )
    submission_intent.mark_submitted(
        ex.campaign_dir,
        phase_name,
        int(state.iteration),
        "900001",
        expected_tasks=expected_tasks,
    )


def _parse_quantum_fixture(ex, state, phase):
    if "AIMALL" in phase.value:
        from ichor.hpc.active_learning.daemon.state import atomic_write_json
        from ichor.hpc.active_learning.versioning.manifest import sha256_file
        from ichor.hpc.active_learning.daemon.quantum_task_receipts import (
            GAUSSIAN_TASK_RECEIPT,
            write_quantum_task_receipt,
        )
        from ichor.core.files.point_directory import PointDirectory

        staging = ex._quantum_staging_path(state, phase.value)
        gaussian_phase = (
            "INITIAL_GAUSSIAN" if phase.value.startswith("INITIAL_") else "GAUSSIAN"
        )
        _install_quantum_intent(ex, state, CampaignPhase(gaussian_phase))
        for task_index, pointdir in enumerate(sorted(staging.glob("POINT_*.pointdir"))):
            wfns = sorted(pointdir.glob("*.wfn"))
            if len(wfns) != 1:
                continue
            write_quantum_task_receipt(
                ex.campaign_dir,
                pointdir,
                phase_name=gaussian_phase,
                iteration=int(state.iteration),
                logical_task_id=task_index,
            )
            gaussian_receipt = pointdir / GAUSSIAN_TASK_RECEIPT
            receipt_path, receipt = stg.rewrite_wfn_for_aimall(
                wfns[0],
                method=str(ex.config.gaussian.method),
                phase_name=phase.value,
                iteration=int(state.iteration),
                task_index=task_index,
                source_acceptance_sha256="0" * 64,
            )
            atomic_write_json(
                pointdir / stg.AIMALL_TASK_METADATA,
                {
                    "schema_version": stg.AIMALL_TASK_METADATA_SCHEMA_VERSION,
                    "pointdir": pointdir.name,
                    "task_index": task_index,
                    "gaussian_logical_task_id": task_index,
                    "atom_count": 3,
                    "expected_atom_names": list(PointDirectory(pointdir).atoms.names),
                    "primitive_count": 1,
                    "nproc": 1,
                    "naat": 1,
                    "electronic_method": str(receipt["method"]),
                    "wfn_sha256": str(receipt["wfn"]["after_sha256"]),
                    "wfn_method_receipt": {
                        "path": receipt_path.name,
                        "sha256": sha256_file(receipt_path),
                    },
                    "gaussian_task_receipt": {
                        "path": gaussian_receipt.name,
                        "sha256": sha256_file(gaussian_receipt),
                    },
                    "gjf_sha256": sha256_file(next(pointdir.glob("*.gjf"))),
                    "resource_resolution": {"fixture": True},
                },
            )
    _install_quantum_intent(ex, state, phase)
    return ex._parse_quantum_postprocess(state, phase, observations=[])


def _seed_point_allocation(
    campaign,
    staging,
    *,
    context,
    iteration,
    targets=None,
):
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
        targets=(
            dict(targets)
            if targets is not None
            else {
                "train": len(candidates),
                "int_val": 0,
                "ext_val": 0,
                "total": len(candidates),
            }
        ),
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
            seed_id=(
                int(attempt["slot_id"]) + 1
                if str(context) == "active"
                else None
            ),
            seed_uid=(
                format(int(attempt["slot_id"]) + 1, "064x")
                if str(context) == "active"
                else None
            ),
            array_task_id_zero_based=(
                int(attempt["slot_id"])
                if str(context) == "active"
                else None
            ),
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
            allocation_slot_assignment_sha256=str(
                allocation["slot_assignment_sha256"]
            ),
        )
    return allocation_path, allocation


def _complete_point_allocation(
    campaign,
    staging,
    *,
    context,
    iteration,
    targets=None,
):
    allocation_path, allocation = _seed_point_allocation(
        campaign,
        staging,
        context=context,
        iteration=iteration,
        targets=targets,
    )
    pointdirs = {
        pointdir.name: pointdir
        for pointdir in sorted(Path(staging).glob("POINT_*.pointdir"))
    }
    from ichor.hpc.active_learning.daemon.quantum_quality import (
        write_quantum_quality_manifest,
    )
    from ichor_hpc.tests.quantum_test_support import (
        attach_synthetic_quantum_acceptance,
        synthetic_quantum_quality_record,
    )

    phase_name = (
        CampaignPhase.INITIAL_AIMALL.value
        if str(context) == "bootstrap"
        else CampaignPhase.AIMALL.value
    )
    quality_records = [
        synthetic_quantum_quality_record(pointdir.name)
        for pointdir in pointdirs.values()
    ]
    quality_path = write_quantum_quality_manifest(
        Path(staging),
        phase_name=phase_name,
        iteration=int(iteration),
        records=quality_records,
        gates={},
    )
    attempts = {
        str(attempt["pointdir_name"]): attempt
        for attempt in pending_attempts(allocation)
    }
    result_evidence = {}
    for pointdir, quality_record in zip(pointdirs.values(), quality_records):
        result_evidence[pointdir.name] = attach_synthetic_quantum_acceptance(
            campaign,
            pointdir,
            phase_name=phase_name,
            iteration=int(iteration),
            quality_manifest=quality_path,
            quality_record=quality_record,
        )
    record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": str(attempt["candidate_id"]),
                "accepted": True,
                "pointdir": str(pointdirs[str(attempt["pointdir_name"])]),
                "quality_manifest": str(quality_path.resolve()),
                **result_evidence[str(attempt["pointdir_name"])],
            }
            for attempt in pending_attempts(allocation)
        ],
    )
    return allocation_path


def _commit_bootstrap_reference_data(campaign):
    initial = Path(campaign) / ".DATA" / "STAGING" / "initial"
    for index in range(9):
        pointdir = initial / ("POINT_" + str(index).zfill(4) + ".pointdir")
        pointdir.mkdir(parents=True, exist_ok=True)
        (pointdir / "input.gjf").write_text("# synthetic\n", encoding="utf-8")
    _complete_point_allocation(
        campaign,
        initial,
        context="bootstrap",
        iteration=0,
        targets={"train": 5, "int_val": 2, "ext_val": 2, "total": 9},
    )
    stg.commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    return ReferenceDataVersioning(
        Path(campaign) / "QM_REFERENCE_DATA"
    ).resolve(0, verification="deep")


def _commit_active_reference_data(campaign, *, iteration=1):
    live = Path(campaign) / ".DATA" / "STAGING" / ("iter_" + str(iteration))
    pointdir = live / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    (pointdir / "input.gjf").write_text("# synthetic\n", encoding="utf-8")
    stg.write_points_file(live, [pointdir])
    stg.write_quantum_acceptance_manifest(
        live,
        phase_name="AIMALL",
        iteration=iteration,
        accepted=[pointdir],
        rejected=[],
    )
    _complete_point_allocation(
        campaign,
        live,
        context="active",
        iteration=iteration,
        targets={"train": 1, "int_val": 0, "ext_val": 0, "total": 1},
    )
    stg.commit_reference_data_delta(
        campaign,
        reference_data_version=iteration,
        context="active",
        iteration=iteration,
    )
    return ReferenceDataVersioning(
        Path(campaign) / "QM_REFERENCE_DATA"
    ).resolve(iteration, verification="deep")


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
    result = _parse_quantum_fixture(ex, state, CampaignPhase("INITIAL_GAUSSIAN"))
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
    state = SimpleNamespace(iteration=3, campaign_uid="m16-test", reference_data_version=0)
    result = _parse_quantum_fixture(ex, state, CampaignPhase("GAUSSIAN"))
    assert result.is_complete is True
    assert result.failure_reason is None


def test_initial_aimall_happy_path(tmp_path, monkeypatch):
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
    from ichor.hpc.active_learning.daemon import quantum_quality as quality_module

    evaluated = []
    real_evaluate = quality_module.evaluate_aimall_pointdir

    def track_evaluation(pointdir, *args, **kwargs):
        evaluated.append(Path(pointdir.path).name)
        return real_evaluate(pointdir, *args, **kwargs)

    monkeypatch.setattr(
        quality_module,
        "evaluate_aimall_pointdir",
        track_evaluation,
    )
    result = _parse_quantum_fixture(ex, state, CampaignPhase("INITIAL_AIMALL"))
    assert result.is_complete is True
    assert result.failure_reason is None
    assert sorted(evaluated) == sorted(path.name for path in staging.glob("*.pointdir"))
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded[-1]["phase"] == "INITIAL_AIMALL"
    manifest = json.loads((staging / stg.QUANTUM_ACCEPTANCE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["phase"] == "INITIAL_AIMALL"
    assert manifest["iteration"] == 0
    assert len(manifest["accepted_pointdirs"]) == 4


def test_aimall_visibility_lag_retries_before_publishing_scientific_evidence(
    tmp_path,
):
    ex = _make_executor(tmp_path)
    staging = _bind_staging(ex, FIXTURES / "initial_quantum")
    _seed_point_allocation(
        ex.campaign_dir,
        staging,
        context="bootstrap",
        iteration=0,
    )
    pointdirs = sorted(staging.glob("POINT_*.pointdir"))
    stg.write_quantum_acceptance_manifest(
        staging,
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        accepted=pointdirs,
        rejected=[],
    )
    missing_int = next(pointdirs[0].glob("*_atomicfiles/*.int"))
    missing_int.unlink()
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")

    result = _parse_quantum_fixture(ex, state, CampaignPhase("INITIAL_AIMALL"))

    assert result.failure_reason.startswith(
        "aimall_outputs_not_settled_missing_or_unreadable"
    )
    current = json.loads(
        (staging / stg.QUANTUM_ACCEPTANCE_MANIFEST).read_text(encoding="utf-8")
    )
    assert current["phase"] == "INITIAL_GAUSSIAN"
    assert not (staging / "quantum_quality.json").exists()
    assert not any(
        pointdir.joinpath("AIMALL_COMPLETION_RECEIPT.json").exists()
        for pointdir in pointdirs
    )


def test_stage_aimall_inputs_writes_resolved_naat_metadata(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    staging = campaign / ".DATA" / "STAGING" / "initial"
    shutil.copytree(str(FIXTURES / "initial_quantum"), str(staging))
    all_pointdirs = sorted(staging.glob("POINT_*.pointdir"))
    accepted = all_pointdirs[1:2]
    source_gjf = next(accepted[0].glob("*.gjf"))
    source_wfn = next(accepted[0].glob("*.wfn"))
    shutil.copy2(
        source_gjf,
        accepted[0] / "input.gjf",
    )
    shutil.copy2(
        source_wfn,
        accepted[0] / "input.wfn",
    )
    source_gjf.unlink()
    source_wfn.unlink()
    write_seed_provenance(
        accepted[0],
        campaign_uid="m16-test",
        iteration=0,
        trajectory_sha256="0" * 64,
        seed_frame_id=0,
        seed_id=None,
        seed_uid=None,
        array_task_id_zero_based=None,
        seed_selection_origin="live_parser_fixture",
        seed_variance_at_selection=None,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
    )
    stg.write_points_file(staging, all_pointdirs)
    stg.write_quantum_acceptance_manifest(
        staging,
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        accepted=accepted,
        rejected=[
            (pointdir.name, "fixture_not_selected")
            for pointdir in all_pointdirs
            if pointdir not in accepted
        ],
    )
    cfg = CampaignConfig()
    cfg.resources.aimall_cpus_per_task = 8
    cfg.aimall.naat = "auto"
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid="m16-test",
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        expected_tasks=4,
    )
    submission_intent.mark_submitted(
        campaign,
        "INITIAL_GAUSSIAN",
        0,
        "900001",
        expected_tasks=4,
    )
    from ichor.hpc.active_learning.daemon.quantum_task_receipts import (
        write_quantum_task_receipt,
    )

    write_quantum_task_receipt(
        campaign,
        accepted[0],
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        logical_task_id=1,
    )

    original_atomic_write_json = stg.atomic_write_json
    interrupted = {"raised": False}

    def interrupt_before_task_metadata(path, payload):
        if (
            Path(path).name == stg.AIMALL_TASK_METADATA
            and not interrupted["raised"]
        ):
            interrupted["raised"] = True
            raise OSError("injected AIMAll task-metadata interruption")
        return original_atomic_write_json(path, payload)

    monkeypatch.setattr(stg, "atomic_write_json", interrupt_before_task_metadata)
    with pytest.raises(
        OSError,
        match="injected AIMAll task-metadata interruption",
    ):
        stg.stage_aimall_inputs(
            campaign,
            cfg,
            "INITIAL_AIMALL",
            0,
        )
    assert [
        Path(line).name
        for line in (staging / "POINTS.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ] == [pointdir.name for pointdir in all_pointdirs]
    prepared_before_replay = {
        name: (accepted[0] / name).read_bytes()
        for name in ("input.wfn", stg.WFN_METHOD_RECEIPT)
    }

    monkeypatch.setattr(stg, "atomic_write_json", original_atomic_write_json)
    progress = []
    staged_dir, n_points = stg.stage_aimall_inputs(
        campaign,
        cfg,
        "INITIAL_AIMALL",
        0,
        progress_callback=lambda **payload: progress.append(payload),
    )

    assert staged_dir == staging
    assert n_points == 1
    assert {
        name: (accepted[0] / name).read_bytes()
        for name in prepared_before_replay
    } == prepared_before_replay
    task = json.loads(
        (accepted[0] / stg.AIMALL_TASK_METADATA).read_text(encoding="utf-8")
    )
    assert task["atom_count"] == 3
    assert task["nproc"] == 8
    assert task["naat"] == 3
    assert task["gaussian_logical_task_id"] == 1
    assert task["electronic_method"] == "B3LYP"
    assert task["wfn_sha256"]
    assert task["wfn_method_receipt"]["path"] == stg.WFN_METHOD_RECEIPT

    from ichor.hpc.active_learning.daemon.resource_solver import (
        resolve_phase_resources,
    )

    resolved = resolve_phase_resources(
        phase_name="INITIAL_AIMALL",
        config=cfg,
        campaign_dir=campaign,
        iteration=0,
        staging_dir=staging,
        array_size=1,
        require_evidence=True,
    )
    assert resolved.extra["aimall_task_naat"] == [3]
    assert resolved.extra["evidence"]["aimall_task_naat"] == [3]
    assert [record["stage"] for record in progress] == [
        "aimall_input_validation",
        "aimall_input_validation",
        "aimall_task_staging",
        "aimall_task_staging",
    ]
    assert [record["completed"] for record in progress] == [0, 1, 0, 1]

    immutable_before = {
        name: (accepted[0] / name).read_bytes()
        for name in (
            "input.wfn",
            stg.WFN_METHOD_RECEIPT,
            stg.AIMALL_TASK_METADATA,
        )
    }
    stg.write_points_file(staging, all_pointdirs)

    def refuse_task_metadata_rewrite(path, payload):
        if Path(path).name == stg.AIMALL_TASK_METADATA:
            raise AssertionError("valid AIMAll task metadata was rewritten")
        return original_atomic_write_json(path, payload)

    monkeypatch.setattr(stg, "atomic_write_json", refuse_task_metadata_rewrite)
    replay_dir, replay_count = stg.stage_aimall_inputs(
        campaign,
        cfg,
        "INITIAL_AIMALL",
        0,
    )
    assert replay_dir == staged_dir
    assert replay_count == 1
    assert {
        name: (accepted[0] / name).read_bytes()
        for name in immutable_before
    } == immutable_before
    assert [
        Path(line).name
        for line in (staging / "POINTS.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ] == [pointdir.name for pointdir in accepted]

    task_metadata_path = accepted[0] / stg.AIMALL_TASK_METADATA
    task_metadata_path.unlink()
    with pytest.raises(ValueError, match="lacks task metadata"):
        stg.stage_aimall_inputs(
            campaign,
            cfg,
            "INITIAL_AIMALL",
            0,
        )

    task_metadata_path.write_bytes(
        immutable_before[stg.AIMALL_TASK_METADATA]
    )
    (accepted[0] / stg.WFN_METHOD_RECEIPT).unlink()
    stg.write_points_file(staging, all_pointdirs)
    with pytest.raises(
        ValueError,
        match="task metadata exists without its WFN method receipt",
    ):
        stg.stage_aimall_inputs(
            campaign,
            cfg,
            "INITIAL_AIMALL",
            0,
        )


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
    stg.write_points_file(staging, accepted)
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")

    result = _parse_quantum_fixture(ex, state, CampaignPhase("INITIAL_AIMALL"))

    assert result.failure_reason is None
    manifest = json.loads((staging / stg.QUANTUM_ACCEPTANCE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["accepted_pointdirs"] == ["POINT_0000.pointdir", "POINT_0001.pointdir"]


def test_aimall_allocation_join_accepts_absent_gaussian_rejections(tmp_path):
    ex = _make_executor(tmp_path)
    staging = _bind_staging(ex, FIXTURES / "initial_quantum")
    allocation_path, _allocation = _seed_point_allocation(
        ex.campaign_dir,
        staging,
        context="bootstrap",
        iteration=0,
    )
    accepted = [staging / "POINT_0000.pointdir", staging / "POINT_0001.pointdir"]
    rejected_names = ["POINT_0002.pointdir", "POINT_0003.pointdir"]
    stg.write_quantum_acceptance_manifest(
        staging,
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        accepted=accepted,
        rejected=[
            (name, "submitted_pointdir_missing")
            for name in rejected_names
        ],
    )
    stg.write_points_file(staging, accepted)
    for name in rejected_names:
        shutil.rmtree(staging / name)
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")

    result = _parse_quantum_fixture(ex, state, CampaignPhase("INITIAL_AIMALL"))

    assert result.failure_reason is None
    allocation = read_point_allocation(
        allocation_path,
        expected_campaign_uid="m16-test",
        expected_context="bootstrap",
        expected_iteration=0,
    )
    assert allocation["summary"]["accepted_total"] == 2
    assert allocation["summary"]["vacant_slots"] == 2


def test_partial_rejection_below_threshold_still_succeeds(tmp_path):
    ex = _make_executor(tmp_path, failure_threshold=0.5)
    staging = _bind_staging(ex, FIXTURES / "iter_quantum_scf_failure")
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = _parse_quantum_fixture(ex, state, CampaignPhase("GAUSSIAN"))
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
    result = _parse_quantum_fixture(ex, state, CampaignPhase("GAUSSIAN"))
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
    state = SimpleNamespace(iteration=2, campaign_uid="m16-test", reference_data_version=0)
    result = _parse_quantum_fixture(ex, state, CampaignPhase("GAUSSIAN"))
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
    phase = CampaignPhase("INITIAL_GAUSSIAN")
    _install_quantum_intent(ex, state, phase)
    result = ex.postprocess(state, phase, observations=[])
    assert result.is_complete is True
    assert result.failure_reason is None


def test_bootstrap_reference_commit_rejects_incomplete_allocation(tmp_path):
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
    quality_manifest = initial / "quantum_quality.json"
    quality_manifest.write_text("{}\n", encoding="utf-8", newline="\n")
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
                "quality_manifest": (
                    str(quality_manifest.resolve())
                    if str(attempt["candidate_id"]) == accepted_id
                    else None
                ),
            }
            for attempt in attempts
        ],
    )

    with pytest.raises(
        ValueError,
        match="reference commit requires a complete point allocation",
    ):
        stg.commit_reference_data_delta(
            campaign,
            reference_data_version=0,
            context="bootstrap",
            iteration=0,
        )


def test_initial_aimall_reader_rejects_legacy_gaussian_alias(tmp_path):
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

    with pytest.raises(FileNotFoundError, match="quantum acceptance manifest missing"):
        stg.read_quantum_acceptance_manifest(
            initial,
            expected_phase="INITIAL_GAUSSIAN",
            expected_iteration=0,
            require_points_file_membership=True,
        )


def test_bootstrap_reference_commit_rejects_missing_point_allocation(tmp_path):
    campaign = tmp_path / "campaign"

    with pytest.raises(FileNotFoundError, match="point-allocation manifest missing"):
        stg.commit_reference_data_delta(
            campaign,
            reference_data_version=0,
            context="bootstrap",
            iteration=0,
        )


def test_reference_commit_requires_complete_point_allocation(tmp_path):
    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    live_staging = stg.bucket_dir(ex.campaign_dir, "REFERENCE_COMMIT", 1)
    (live_staging / "POINT_0000.pointdir").mkdir(parents=True)
    _seed_point_allocation(
        ex.campaign_dir,
        live_staging,
        context="active",
        iteration=1,
    )
    state = SimpleNamespace(
        iteration=1,
        campaign_uid="m16-test",
        reference_data_version=0,
    )

    with pytest.raises(BackendSubmissionError, match="complete point allocation"):
        ex._inline_reference_commit(state)


def test_reference_commit_publishes_global_pointdir_names_from_manifest(tmp_path):
    ex = _make_executor(tmp_path)
    bootstrap_view = _commit_bootstrap_reference_data(ex.campaign_dir)
    v = ReferenceDataVersioning(ex.campaign_dir / "QM_REFERENCE_DATA")

    live_staging = stg.bucket_dir(ex.campaign_dir, "REFERENCE_COMMIT", 1)
    new_point = live_staging / "POINT_0000.pointdir"
    new_point.mkdir(parents=True)
    (new_point / "new.txt").write_text("new\n", encoding="utf-8")
    stg.write_points_file(live_staging, [new_point])
    stg.write_quantum_acceptance_manifest(
        live_staging,
        phase_name="AIMALL",
        iteration=1,
        accepted=[new_point],
        rejected=[],
    )
    _complete_point_allocation(
        ex.campaign_dir,
        live_staging,
        context="active",
        iteration=1,
    )
    state = SimpleNamespace(
        iteration=1,
        campaign_uid="m16-test",
        reference_data_version=0,
    )

    result = ex._inline_reference_commit(state)

    assert result["reference_data_version"] == 1
    committed = v.iteration_path(1)
    assert len(bootstrap_view.entries) == 9
    assert not list(committed.glob("POINT_00000[0-8].pointdir"))
    assert (committed / "POINT_000009.pointdir" / "new.txt").is_file()
    resolved = v.resolve(1, verification="deep")
    assert [entry.pointdir_name for entry in resolved.entries] == [
        "POINT_" + str(index).zfill(6) + ".pointdir"
        for index in range(10)
    ]


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
        "PHASE_A_DIVERSITY", "PHASE_B_DIVERSITY",
    ):
        assert ph in LIVE_POSTPROCESS_IMPLEMENTED


def test_handlers_dict_dispatches_all_day3_phases(tmp_path):
    ex = _make_executor(tmp_path)
    handlers = ex._live_postprocess_handlers()
    assert handlers["INITIAL_FEREBUS"] == ex._parse_ferebus_postprocess
    assert handlers["FEREBUS"] == ex._parse_ferebus_postprocess
    assert handlers["ARIADNE_ARRAY"] == ex._parse_ariadne_array_postprocess
    assert handlers["PHASE_A_DIVERSITY"] == ex._parse_diversity_postprocess
    assert handlers["PHASE_B_DIVERSITY"] == ex._parse_diversity_postprocess


# --- FEREBUS parser tests ----------------------------------------


def _seed_models_staging(
    campaign_dir,
    properties=("iqa",),
    *,
    reference_version=0,
):
    """Create manifest-backed pyferebus staging with parseable models."""
    import hashlib

    from ichor.hpc.active_learning.daemon.ferebus_task_runner import (
        write_preexisting_model_receipts,
    )
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.submit.pyferebus_wrap import (
        _write_structured_task_map,
    )
    from ichor.hpc.active_learning.versioning.manifest import sha256_file
    from ichor.hpc.active_learning.versioning.reference_data import (
        canonical_json_sha256,
    )
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.ferebus_prior import (
        resolve_ferebus_prior_contract,
        validate_ferebus_config_contract,
    )

    prior = resolve_ferebus_prior_contract(CampaignConfig())

    reference_view = ReferenceDataVersioning(
        campaign_dir / "QM_REFERENCE_DATA"
    ).resolve(reference_version, verification="deep")
    target = campaign_dir / "TRAINED_MODELS" / "iteration-staging"
    target.mkdir(parents=True, exist_ok=True)
    row_ids = {
        split: [
            index
            for index, entry in enumerate(reference_view.entries)
            if entry.split == split
        ]
        for split in ("train", "int_val", "ext_val")
    }
    row_counts = {split: len(values) for split, values in row_ids.items()}
    source_rows = [
        {
            "source_row_index": index,
            "pointdir_name": entry.pointdir_name,
            "introduced_in_version": entry.introduced_in_version,
            "split": entry.split,
            "provenance_sha256": entry.provenance_sha256,
        }
        for index, entry in enumerate(reference_view.entries)
    ]
    split_identity_rows = {
        split: [source_rows[index] for index in indexes]
        for split, indexes in row_ids.items()
    }
    row_identity_payload = {
        "schema_version": stg.FEREBUS_ROW_IDENTITIES_SCHEMA_VERSION,
        "campaign_uid": reference_view.campaign_uid,
        "reference_data_version": int(reference_version),
        "reference_data_view_sha256": reference_view.cumulative_view_sha256,
        "source_rows": source_rows,
        "source_rows_sha256": canonical_json_sha256(source_rows),
        "splits": {
            split: {
                "rows": rows,
                "n_rows": len(rows),
                "row_identity_sha256": canonical_json_sha256(rows),
            }
            for split, rows in split_identity_rows.items()
        },
    }
    row_identity_path = target / stg.FEREBUS_ROW_IDENTITIES
    atomic_write_json(row_identity_path, row_identity_payload)
    split_payload = {
        "schema_version": 7,
        "allocation_policy": "exact_per_reference_data_version",
        "historical_training_rows": 0,
        "assignments": {
            row["pointdir_name"]: {
                "split": row["split"],
                "first_seen_reference_data_version": row["introduced_in_version"],
                "assignment_version": 4,
                "allocation_manifest_sha256": "c" * 64,
                "provenance_sha256": row["provenance_sha256"],
            }
            for row in source_rows
        },
        "version_allocations": {},
    }
    split_path = target / stg.FEREBUS_SPLIT_SNAPSHOT
    atomic_write_json(split_path, split_payload)
    tasks = []
    for task_index, prop in enumerate(properties, start=1):
        model_dir = target / prop / "O1"
        datasets = model_dir / "datasets"
        datasets.mkdir(parents=True, exist_ok=True)
        config = model_dir / "ferebus.config"
        config.write_text(
            "name = \"WATER\"\nproperty = \"" + prop + "\"\n"
            + "mean_type = 21\n"
            + 'level_of_theory = "' + prior.level_of_theory + '"\n'
            + "iqaDeviationFactor = 1.0\nscaling = 1\n"
            + "scale_feats = 1\nscale_prop = 0\n",
            encoding="utf-8",
        )
        parsed_config = validate_ferebus_config_contract(config, prior)
        model = model_dir / ("WATER_" + prop + "_O1.model")
        _write_loadable_model(
            model,
            atom="O1",
            prop=prop,
            ntrain=row_counts["train"],
        )
        train_csv = datasets / "WATER_O1_TRAINING_SET.csv"
        int_csv = datasets / "WATER_O1_INT_VALIDATION_SET.csv"
        ext_csv = datasets / "WATER_O1_EXT_VALIDATION_SET.csv"
        _write_metric_csv(train_csv, row_counts["train"], prop=prop)
        _write_metric_csv(int_csv, row_counts["int_val"], prop=prop)
        _write_metric_csv(ext_csv, row_counts["ext_val"], prop=prop)
        suffixes = (
            ("opt", "perf", "pred", "scurve", "sol")
            if prop == "iqa"
            else ("opt", "perf")
        )
        for suffix in suffixes:
            (model_dir / ("WATER_" + prop + "_O1." + suffix)).write_text(
                (
                    "RMSE 0.0\nMAE 0.0\n"
                    "covariance_condition_number 1.0\n"
                    if suffix == "perf"
                    else suffix + "\n"
                ),
                encoding="utf-8",
            )
        task_dir = prop + "/O1"
        input_dir = task_dir + "/datasets"
        dataset_records = {
            split: {
                "path": path.relative_to(target).as_posix(),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": row_counts[split],
                "row_identity_sha256": row_identity_payload["splits"][split][
                    "row_identity_sha256"
                ],
                "row_identity_count": row_counts[split],
            }
            for split, path in (
                ("train", train_csv),
                ("int_val", int_csv),
                ("ext_val", ext_csv),
            )
        }
        tasks.append({
            "task_index": task_index,
            "property": prop,
            "atom": "O1",
            "prior_mean": prior.task_payload(
                prop,
                "O1",
                training_values=[0.0] * row_counts["train"],
                training_dataset_sha256=sha256_file(train_csv),
            ),
            "alf_1_indexed": [1, 2, 3],
            "alf_cli": "1_2_3",
            "property_dir": prop,
            "output_dir": task_dir,
            "input_dir": input_dir,
            "config_path": task_dir + "/ferebus.config",
            "expected_model_path": task_dir + "/WATER_" + prop + "_O1.model",
            "training_csv": input_dir + "/WATER_O1_TRAINING_SET.csv",
            "int_validation_csv": input_dir + "/WATER_O1_INT_VALIDATION_SET.csv",
            "ext_validation_csv": input_dir + "/WATER_O1_EXT_VALIDATION_SET.csv",
            "command_args": [
                "-c", task_dir + "/ferebus.config",
                "-I", input_dir,
                "-O", task_dir,
                "-P", prop,
                "-A", "O1",
                "-ALF", "1_2_3",
            ],
            "row_counts": dict(row_counts),
            "historical_training_rows": 0,
            "historical_training_row_ids": [],
            "row_ids": {key: list(value) for key, value in row_ids.items()},
            "datasets": dataset_records,
            "degenerate_property_stats": False,
            "generated_config": {
                "path": task_dir + "/ferebus.config",
                "size": config.stat().st_size,
                "sha256": sha256_file(config),
                "parsed_contract": parsed_config,
                "prior_mean_contract_sha256": prior.contract_sha256,
            },
        })
    task_payload = {
        "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
        "campaign_uid": reference_view.campaign_uid,
        "system": "WATER",
        "reference_data_version": int(reference_version),
        "reference_data_head_manifest_sha256": reference_view.head_manifest_sha256,
        "reference_data_view_sha256": reference_view.cumulative_view_sha256,
        "n_reference_points": len(reference_view.entries),
        "pointdir_row_order": [entry.pointdir_name for entry in reference_view.entries],
        "properties": list(properties),
        "atoms": ["O1"],
        "n_atoms": 1,
        "n_tasks": len(tasks),
        "prior_mean_contract": prior.to_dict(),
        "kernel_contract": {
            "family": "rbf",
            "backend_token": "rbf",
            "loss": "huber",
            "constant_noise": True,
            "full_ard": True,
            "feature_scaling": True,
            "property_scaling": False,
            "kernel_prefactor_mode": 2,
        },
        "row_identity_snapshot": {
            "path": stg.FEREBUS_ROW_IDENTITIES,
            "size": row_identity_path.stat().st_size,
            "sha256": sha256_file(row_identity_path),
            "source_rows_sha256": row_identity_payload["source_rows_sha256"],
        },
        "split_ledger": {
            "path": stg.FEREBUS_SPLIT_SNAPSHOT,
            "size": split_path.stat().st_size,
            "sha256": sha256_file(split_path),
            "counts": dict(row_counts),
            "version_allocation": {},
            "allocation_policy": "exact_per_reference_data_version",
            "allocation_manifest": "test",
            "allocation_manifest_sha256": "c" * 64,
            "forced_splits": {
                row["pointdir_name"]: row["split"] for row in source_rows
            },
        },
        "degenerate_property_stats": [],
        "tasks": tasks,
    }
    task_path = target / stg.FEREBUS_TASK_MANIFEST
    atomic_write_json(task_path, task_payload)
    _write_structured_task_map(
        target,
        executable="ferebus",
        execution_kind="synthetic_dry_run",
        performance_required=True,
    )
    write_preexisting_model_receipts(
        target,
        execution_kind="synthetic_dry_run",
    )
    for name in (stg.FEREBUS_JOB_DETAILS, "commands", "list.txt", "runFerebus.sh"):
        (target / name).write_text(name + "\n", encoding="utf-8")
    return target


def _write_metric_csv(path, n_rows, *, prop="iqa"):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("f1,f2,f3," + prop + "\n")
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
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.ferebus_prior import (
        resolve_ferebus_prior_contract,
    )

    prior_mean = resolve_ferebus_prior_contract(
        CampaignConfig()
    ).expected_mean_ha(prop, atom)
    rows = [
        [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1][:nfeats]
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
        "type constant",
        "value " + repr(prior_mean),
        "",
        "[kernels]",
        "number_of_kernels 1",
        "composition k1",
        "prefactor 1.0",
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
    lines += ["0.0" for _ in range(ntrain)]
    lines += ["", "[weights]"]
    lines += ["0.0" for _ in range(ntrain)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ferebus_parser_happy_path_commits_models_version(tmp_path):
    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    _seed_models_staging(tmp_path / "campaign")
    state = SimpleNamespace(iteration=3, campaign_uid="m16-test", reference_data_version=0)
    result = ex._parse_ferebus_postprocess(
        state, CampaignPhase("FEREBUS"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None
    # validation_set_version now advances alongside models_version each FEREBUS commit (A12)
    assert result.state_updates == {"models_version": 0, "validation_set_version": 0}
    # The committed snapshot keeps each task's files together.
    committed_dir = (
        tmp_path / "campaign" / "TRAINED_MODELS" / "iteration-000000"
    )
    assert committed_dir.is_dir()
    task_dir = committed_dir / "iqa" / "O1"
    assert (task_dir / "WATER_iqa_O1.model").is_file()
    assert (committed_dir / stg.FEREBUS_TASK_MANIFEST).is_file()
    assert not (committed_dir / "commands").exists()
    assert not (committed_dir / "list.txt").exists()
    assert (task_dir / "ferebus.config").is_file()
    assert (
        task_dir / "datasets" / "WATER_O1_TRAINING_SET.csv"
    ).is_file()
    assert (task_dir / "WATER_iqa_O1.opt").read_text(encoding="utf-8") == "opt\n"
    assert not (committed_dir / "task_artefacts").exists()
    artefact_manifest = json.loads(
        (committed_dir / FEREBUS_TASK_ARTEFACTS_MANIFEST)
        .read_text(encoding="utf-8")
    )
    assert artefact_manifest["schema_version"] == 3
    assert artefact_manifest["storage_mode"] == "full_snapshot"
    assert artefact_manifest["models_version"] == 0
    assert artefact_manifest["tasks"][0]["directory"] == "iqa/O1"
    assert artefact_manifest["tasks"][0]["model"]["path"] == (
        "iqa/O1/WATER_iqa_O1.model"
    )
    assert artefact_manifest["tasks"][0]["config"]["path"] == (
        "iqa/O1/ferebus.config"
    )
    assert artefact_manifest["tasks"][0]["datasets"]["train"]["path"] == (
        "iqa/O1/datasets/WATER_O1_TRAINING_SET.csv"
    )
    assert artefact_manifest["tasks"][0]["auxiliary"]["opt"]["path"] == (
        "iqa/O1/WATER_iqa_O1.opt"
    )
    import os

    assert os.access(task_dir, os.W_OK)
    assert os.access(task_dir / "WATER_iqa_O1.model", os.W_OK)
    events = _read_journal_events(tmp_path / "campaign")
    assert any(e.get("event") == "models_committed" for e in events)


def test_ariadne_task_model_loader_avoids_historical_chain_resolution(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelError,
        load_trained_models_for_ariadne_task,
    )

    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    _seed_models_staging(ex.campaign_dir)
    result = ex._parse_ferebus_postprocess(
        SimpleNamespace(
            iteration=0,
            campaign_uid="m16-test",
            reference_data_version=0,
        ),
        CampaignPhase("FEREBUS"),
        observations=[],
    )
    assert result.failure_reason is None

    versioning = TrainedModelVersioning(ex.campaign_dir / "TRAINED_MODELS")
    committed = versioning.resolve(
        0,
        verification="metadata",
        reference_verification="metadata",
    )
    task_map = {
        "campaign_uid": "m16-test",
        "models_version": 0,
        "model_manifest_sha256": committed.head_manifest_sha256,
        "model_set_sha256": committed.model_set_sha256,
    }

    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError("ARIADNE task loading must not replay history")

    monkeypatch.setattr(TrainedModelVersioning, "resolve", forbidden_resolve)
    monkeypatch.setattr(ReferenceDataVersioning, "resolve", forbidden_resolve)

    model_set, models, bindings = load_trained_models_for_ariadne_task(
        ex.campaign_dir,
        0,
        task_map=task_map,
        expected_campaign_uid="m16-test",
    )

    assert model_set.head_manifest_sha256 == committed.head_manifest_sha256
    assert model_set.model_set_sha256 == committed.model_set_sha256
    assert len(models) == 1
    assert [binding.path for binding in bindings] == list(model_set.model_paths)
    with pytest.raises(
        TrainedModelError,
        match="task-map trained-model version mismatch",
    ):
        load_trained_models_for_ariadne_task(
            ex.campaign_dir,
            0,
            task_map={**task_map, "models_version": 1},
            expected_campaign_uid="m16-test",
        )


def test_ferebus_postprocess_snapshot_avoids_historical_chain_resolution(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon.artifact_snapshot import (
        build_committed_artifact_snapshot,
    )
    from ichor.hpc.active_learning.daemon.ferebus_quality import (
        enrich_task_receipt_with_quality,
    )
    from ichor.hpc.active_learning.daemon.ferebus_model_admission import (
        enrich_task_receipt_with_model_admission,
    )
    import ichor.hpc.active_learning.daemon.ferebus_model_admission as admission_module
    import ichor.hpc.active_learning.daemon.model_contract as model_contract_module
    import ichor.hpc.active_learning.daemon.ferebus_quality as quality_module
    import ichor.hpc.active_learning.versioning.trained_models as models_module
    from ichor.hpc.active_learning.versioning.reference_data import (
        ReferenceDataVersioning,
    )
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelVersioning,
    )
    from contextlib import contextmanager

    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    _seed_models_staging(ex.campaign_dir)
    bootstrap = ex._parse_ferebus_postprocess(
        SimpleNamespace(
            iteration=0,
            campaign_uid="m16-test",
            reference_data_version=0,
        ),
        CampaignPhase("FEREBUS"),
        observations=[],
    )
    assert bootstrap.failure_reason is None
    staging = ex.campaign_dir / "TRAINED_MODELS" / "iteration-staging"
    if staging.exists():
        shutil.rmtree(staging)
    _commit_active_reference_data(ex.campaign_dir, iteration=1)
    staging = _seed_models_staging(
        ex.campaign_dir,
        reference_version=1,
    )
    enrich_task_receipt_with_quality(staging, 0)
    enrich_task_receipt_with_model_admission(staging, 0)
    ex._committed_artifact_snapshot = build_committed_artifact_snapshot(
        ex.campaign_dir,
        verification_level="authority",
    )

    def forbidden_resolve(*args, **kwargs):
        raise AssertionError("FEREBUS hot path must not replay authority chains")

    def forbidden_incumbent_prediction(*args, **kwargs):
        raise AssertionError(
            "FEREBUS hot path must reuse authenticated incumbent metrics"
        )

    def forbidden_broad_anchor_replay(*args, **kwargs):
        raise AssertionError(
            "FEREBUS hot path must use its narrow reference/incumbent guard"
        )

    lock_state = {"held": False}
    real_evaluate = quality_module.evaluate_ferebus_quality
    real_lock = models_module.trained_models_commit_lock
    real_model_contract = model_contract_module.validate_ferebus_model_contract

    def no_legacy_admission(*args, **kwargs):
        if kwargs.get("validation_context") is None:
            raise AssertionError(
                "inline task admission must avoid full login-node model replay"
            )
        return real_model_contract(*args, **kwargs)

    def checked_evaluate(*args, **kwargs):
        assert lock_state["held"] is False
        return real_evaluate(*args, **kwargs)

    @contextmanager
    def observed_lock(*args, **kwargs):
        with real_lock(*args, **kwargs):
            lock_state["held"] = True
            try:
                yield
            finally:
                lock_state["held"] = False

    monkeypatch.setattr(TrainedModelVersioning, "resolve", forbidden_resolve)
    monkeypatch.setattr(ReferenceDataVersioning, "resolve", forbidden_resolve)
    monkeypatch.setattr(
        quality_module,
        "_local_incumbent_metric",
        forbidden_incumbent_prediction,
    )
    monkeypatch.setattr(quality_module, "evaluate_ferebus_quality", checked_evaluate)
    monkeypatch.setattr(models_module, "trained_models_commit_lock", observed_lock)
    monkeypatch.setattr(
        model_contract_module,
        "validate_ferebus_model_contract",
        no_legacy_admission,
    )
    monkeypatch.setattr(
        admission_module,
        "_run_cache_subprocess",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inline admission must not invoke legacy fallback")
        ),
    )
    monkeypatch.setattr(
        type(ex._committed_artifact_snapshot),
        "assert_anchors_unchanged",
        forbidden_broad_anchor_replay,
    )
    result = ex._parse_ferebus_postprocess(
        SimpleNamespace(
            iteration=1,
            campaign_uid="m16-test",
            reference_data_version=1,
        ),
        CampaignPhase("FEREBUS"),
        observations=[],
    )

    assert result.failure_reason is None
    assert result.state_updates["models_version"] == 1
    summary = next(
        event
        for event in reversed(_read_journal_events(ex.campaign_dir))
        if event.get("event") == "ferebus_quality_summary"
        and event.get("iteration") == 1
    )
    assert summary["n_inline_quality_measurements"] == 1
    assert summary["n_incumbent_metrics_reused"] == 1
    assert summary["n_incumbent_metrics_computed"] == 0


def test_ferebus_task_artefact_layout_supports_properties_and_missing_files(tmp_path):
    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    _seed_models_staging(ex.campaign_dir, properties=("iqa", "q00"))
    result = ex._parse_ferebus_postprocess(
        SimpleNamespace(iteration=0, campaign_uid="m16-test", reference_data_version=0),
        CampaignPhase("FEREBUS"),
        observations=[],
    )
    assert result.failure_reason is None
    committed = tmp_path / "campaign" / "TRAINED_MODELS" / "iteration-000000"
    assert (committed / "iqa" / "O1" / "WATER_iqa_O1.model").is_file()
    assert (committed / "iqa" / "O1" / "ferebus.config").is_file()
    assert (committed / "q00" / "O1" / "WATER_q00_O1.model").is_file()
    assert (committed / "q00" / "O1" / "ferebus.config").is_file()
    assert not (committed / "task_artefacts").exists()
    artefact_manifest = json.loads(
        (committed / FEREBUS_TASK_ARTEFACTS_MANIFEST).read_text(encoding="utf-8")
    )
    assert artefact_manifest["n_tasks"] == 2
    assert artefact_manifest["tasks"][1]["auxiliary"]["perf"] is not None
    assert artefact_manifest["tasks"][1]["property"] == "q00"


def test_rejected_ferebus_candidate_is_quarantined_without_model_commit(tmp_path):
    from ichor.hpc.active_learning.daemon.ferebus_candidate_recovery import (
        discover_recovery_candidate,
    )

    ex = _make_executor(tmp_path)
    ex.config.quality_gates.ferebus_max_ext_rmse_ha = 0.1
    _commit_bootstrap_reference_data(ex.campaign_dir)
    staging = _seed_models_staging(ex.campaign_dir)

    result = ex._parse_ferebus_postprocess(
        SimpleNamespace(iteration=0, campaign_uid="m16-test", reference_data_version=0),
        CampaignPhase("FEREBUS"),
        observations=[],
    )

    result.validate(stage="postprocess", phase_name="FEREBUS")
    assert result.failure_reason.startswith("ferebus_quality_failed:")
    assert not staging.exists(), result.failure_reason
    assert not (
        ex.campaign_dir / "TRAINED_MODELS" / "iteration-000000"
    ).exists()
    quarantined_quality = list(
        (
            ex.campaign_dir
            / "TRAINED_MODELS"
            / "rejected-candidates"
            / "reference-000000"
        ).glob("*/FEREBUS_QUALITY.json")
    )
    assert len(quarantined_quality) == 1
    assert (
        discover_recovery_candidate(
            ex.campaign_dir,
            expected_campaign_uid="m16-test",
            reference_data_version=0,
        )
        is None
    )


def test_incomplete_ferebus_measurement_preserves_raw_staging(tmp_path):
    from ichor.hpc.active_learning.daemon.ferebus_candidate_recovery import (
        read_recovery_request,
    )
    from ichor.hpc.active_learning.daemon.ferebus_task_runner import (
        write_preexisting_model_receipts,
    )

    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    staging = _seed_models_staging(ex.campaign_dir)
    perf = staging / "iqa" / "O1" / "WATER_iqa_O1.perf"
    perf.write_text(
        "RMSE 0.0\nMAE 0.0\nunknown_metric 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    write_preexisting_model_receipts(
        staging,
        execution_kind="synthetic_dry_run",
    )

    result = ex._parse_ferebus_postprocess(
        SimpleNamespace(
            iteration=0,
            campaign_uid="m16-test",
            reference_data_version=0,
        ),
        CampaignPhase("INITIAL_FEREBUS"),
        observations=[],
    )

    result.validate(stage="postprocess", phase_name="INITIAL_FEREBUS")
    assert result.failure_reason.startswith(
        "ferebus_quality_measurement_incomplete:"
    )
    assert staging.is_dir()
    assert not (staging / "FEREBUS_QUALITY.json").exists()
    assert not (staging / "FEREBUS_QUALITY_DECISION.json").exists()
    assert not (
        ex.campaign_dir / "TRAINED_MODELS" / "rejected-candidates"
    ).exists()
    attempts = list(
        (
            ex.campaign_dir
            / ".DATA"
            / "ACTIVE_LEARNING"
            / "ferebus_quality_attempts"
        ).glob("reference-000000/*/*.json")
    )
    assert len(attempts) == 1
    request = read_recovery_request(
        ex.campaign_dir,
        expected_campaign_uid="m16-test",
    )
    assert request is not None
    assert request["status"] == "measurement_incomplete"


def test_legacy_measurement_quarantine_is_copied_for_inline_recovery(tmp_path):
    from ichor.hpc.active_learning.daemon.ferebus_candidate_recovery import (
        discover_recovery_candidate,
        materialise_recovery_candidate,
        prepare_recovery_request,
    )
    from ichor.hpc.active_learning.daemon.state import atomic_write_json

    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    staging = _seed_models_staging(ex.campaign_dir)
    quarantine = (
        ex.campaign_dir
        / "TRAINED_MODELS"
        / "rejected-candidates"
        / "reference-000000"
        / ("a" * 16)
    )
    quarantine.parent.mkdir(parents=True)
    shutil.move(str(staging), str(quarantine))
    measurement_error = (
        "ferebus_quality_metric_failed:ValueError:"
        "FEREBUS performance receipt is missing "
        "['covariance_condition_number']"
    )
    atomic_write_json(
        quarantine / "FEREBUS_QUALITY.json",
        {
            "schema_version": 4,
            "campaign_uid": "m16-test",
            "reference_data_version": 0,
            "measurement_complete": False,
            "measurement_errors": [measurement_error],
        },
    )
    from ichor.hpc.active_learning.daemon.completion_receipts import (
        canonical_sha256,
    )
    from ichor.hpc.active_learning.versioning.manifest import sha256_file

    quality_path = quarantine / "FEREBUS_QUALITY.json"
    evaluation = {
        "accepted": False,
        "reasons": [
            "ferebus_aggregate_metric_missing",
            measurement_error,
        ],
    }
    evaluation_digest = canonical_sha256(evaluation)
    evaluation["evaluation_sha256"] = evaluation_digest
    atomic_write_json(
        quarantine / "FEREBUS_QUALITY_DECISION.json",
        {
            "schema_version": 2,
            "campaign_uid": "m16-test",
            "reference_data_version": 0,
            "quality": {
                "path": "FEREBUS_QUALITY.json",
                "size": quality_path.stat().st_size,
                "sha256": sha256_file(quality_path),
            },
            "evaluations": [evaluation],
            "current_evaluation_sha256": evaluation_digest,
        },
    )
    source_bytes = {
        path.relative_to(quarantine).as_posix(): path.read_bytes()
        for path in quarantine.rglob("*")
        if path.is_file()
    }

    candidate = discover_recovery_candidate(
        ex.campaign_dir,
        expected_campaign_uid="m16-test",
        reference_data_version=0,
    )
    assert candidate is not None
    prepare_recovery_request(
        ex.campaign_dir,
        candidate=candidate,
        campaign_uid="m16-test",
        phase="INITIAL_FEREBUS",
        iteration=0,
        reference_data_version=0,
    )
    recovered = materialise_recovery_candidate(
        ex.campaign_dir,
        campaign_uid="m16-test",
        phase="INITIAL_FEREBUS",
        iteration=0,
        reference_data_version=0,
    )

    assert recovered == staging
    assert recovered.is_dir()
    assert not (recovered / "FEREBUS_QUALITY.json").exists()
    assert not (recovered / "FEREBUS_QUALITY_DECISION.json").exists()
    assert (recovered / "iqa" / "O1" / "WATER_iqa_O1.perf").is_file()
    assert source_bytes == {
        path.relative_to(quarantine).as_posix(): path.read_bytes()
        for path in quarantine.rglob("*")
        if path.is_file()
    }


def test_relative_regression_quarantine_reuses_quality_and_commits_without_job(
    tmp_path,
):
    from ichor.hpc.active_learning.daemon.completion_receipts import (
        canonical_sha256,
    )
    from ichor.hpc.active_learning.daemon.ferebus_candidate_recovery import (
        discover_recovery_candidate,
        materialise_recovery_candidate,
        prepare_recovery_request,
    )
    from ichor.hpc.active_learning.daemon.ferebus_quality import (
        FEREBUS_QUALITY_DECISION_POLICY,
        evaluate_ferebus_quality,
        evaluate_ferebus_quality_decision,
        read_ferebus_quality_decision,
        write_ferebus_quality_manifest,
    )
    from ichor.hpc.active_learning.daemon.ferebus_task_runner import (
        write_preexisting_model_receipts,
    )
    from ichor.hpc.active_learning.versioning.manifest import sha256_file

    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    _seed_models_staging(ex.campaign_dir)
    bootstrap_result = ex._parse_ferebus_postprocess(
        SimpleNamespace(
            iteration=0,
            campaign_uid="m16-test",
            reference_data_version=0,
        ),
        CampaignPhase("FEREBUS"),
        observations=[],
    )
    assert bootstrap_result.failure_reason is None
    bootstrap_staging = ex.campaign_dir / "TRAINED_MODELS" / "iteration-staging"
    if bootstrap_staging.exists():
        shutil.rmtree(bootstrap_staging)
    _commit_active_reference_data(ex.campaign_dir, iteration=1)
    staging = _seed_models_staging(
        ex.campaign_dir,
        reference_version=1,
    )
    candidate_model = staging / "iqa" / "O1" / "WATER_iqa_O1.model"
    model_lines = candidate_model.read_text(encoding="utf-8").splitlines()
    weights_index = model_lines.index("[weights]")
    for index in range(weights_index + 1, len(model_lines)):
        if model_lines[index]:
            model_lines[index] = "-1000.0"
    candidate_model.write_text(
        "\n".join(model_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_preexisting_model_receipts(
        staging,
        execution_kind="synthetic_dry_run",
    )
    quality = evaluate_ferebus_quality(staging, ex.config.quality_gates)
    advisory = evaluate_ferebus_quality_decision(
        quality,
        ex.config.quality_gates,
    )
    assert advisory["accepted"] is True
    assert advisory["warnings"]
    quality_path = write_ferebus_quality_manifest(staging, quality)

    legacy = dict(advisory)
    legacy.pop("decision_policy", None)
    legacy.pop("n_warned", None)
    legacy.pop("n_warnings", None)
    legacy["accepted"] = False
    legacy["reasons"] = list(legacy.pop("warnings"))
    legacy_tasks = []
    for task in legacy["tasks"]:
        old_task = dict(task)
        old_task["reasons"] = list(old_task.pop("warnings"))
        old_task["accepted"] = not bool(old_task["reasons"])
        legacy_tasks.append(old_task)
    legacy["tasks"] = legacy_tasks
    legacy["n_accepted"] = sum(
        1 for task in legacy_tasks if task["accepted"]
    )
    legacy["n_rejected"] = len(legacy_tasks) - legacy["n_accepted"]
    legacy["config_sha256"] = "legacy-config"
    evaluation_sha = canonical_sha256(legacy)
    legacy["evaluation_sha256"] = evaluation_sha
    decision_path = staging / "FEREBUS_QUALITY_DECISION.json"
    atomic_write_json(
        decision_path,
        {
            "schema_version": 2,
            "campaign_uid": "m16-test",
            "reference_data_version": 1,
            "quality": {
                "path": "FEREBUS_QUALITY.json",
                "size": quality_path.stat().st_size,
                "sha256": sha256_file(quality_path),
            },
            "evaluations": [legacy],
            "current_evaluation_sha256": evaluation_sha,
        },
    )
    candidate_digest = hashlib.sha256(
        (sha256_file(quality_path) + ":" + evaluation_sha).encode("ascii")
    ).hexdigest()
    quarantine = (
        ex.campaign_dir
        / "TRAINED_MODELS"
        / "rejected-candidates"
        / "reference-000001"
        / candidate_digest
    )
    quarantine.parent.mkdir(parents=True)
    shutil.move(str(staging), str(quarantine))
    source_bytes = {
        path.relative_to(quarantine).as_posix(): path.read_bytes()
        for path in quarantine.rglob("*")
        if path.is_file()
    }

    candidate = discover_recovery_candidate(
        ex.campaign_dir,
        expected_campaign_uid="m16-test",
        reference_data_version=1,
    )
    assert candidate is not None
    assert candidate["candidate_kind"] == "quality_rejected_relative_regression"
    prepare_recovery_request(
        ex.campaign_dir,
        candidate=candidate,
        campaign_uid="m16-test",
        phase="FEREBUS",
        iteration=1,
        reference_data_version=1,
    )
    recovered = materialise_recovery_candidate(
        ex.campaign_dir,
        campaign_uid="m16-test",
        phase="FEREBUS",
        iteration=1,
        reference_data_version=1,
    )
    assert recovered == staging
    assert (recovered / "FEREBUS_QUALITY.json").read_bytes() == source_bytes[
        "FEREBUS_QUALITY.json"
    ]

    result = ex._parse_ferebus_postprocess(
        SimpleNamespace(
            iteration=1,
            campaign_uid="m16-test",
            reference_data_version=1,
        ),
        CampaignPhase("FEREBUS"),
        observations=[],
    )

    result.validate(stage="submit", phase_name="FEREBUS")
    assert result.failure_reason is None
    assert result.submitted_job_id is None
    assert result.state_updates["models_version"] == 1
    committed = ex.campaign_dir / "TRAINED_MODELS" / "iteration-000001"
    decision = read_ferebus_quality_decision(
        committed,
        require_accepted=True,
        verify_current_config=False,
    )
    assert decision["current_evaluation"]["decision_policy"] == (
        FEREBUS_QUALITY_DECISION_POLICY
    )
    assert decision["current_evaluation"]["warnings"]
    assert len(decision["evaluations"]) == 2
    assert source_bytes == {
        path.relative_to(quarantine).as_posix(): path.read_bytes()
        for path in quarantine.rglob("*")
        if path.is_file()
    }


def test_submit_ferebus_reprocesses_active_recovery_without_sbatch(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import ferebus_candidate_recovery

    ex = _make_executor(tmp_path)
    staging = ex.campaign_dir / "TRAINED_MODELS" / "iteration-staging"
    staging.mkdir(parents=True)
    recovery = {
        "status": "prepared",
        "candidate_id": "candidate-a",
    }
    statuses = []
    monkeypatch.setattr(
        ferebus_candidate_recovery,
        "read_recovery_request",
        lambda *_args, **_kwargs: dict(recovery),
    )
    monkeypatch.setattr(
        ferebus_candidate_recovery,
        "materialise_recovery_candidate",
        lambda *_args, **_kwargs: staging,
    )
    monkeypatch.setattr(
        ferebus_candidate_recovery,
        "update_recovery_status",
        lambda *_args, **kwargs: statuses.append(kwargs["status"]),
    )
    monkeypatch.setattr(
        ex,
        "_parse_ferebus_postprocess",
        lambda *_args, **_kwargs: PhaseResult(
            is_complete=True,
            state_updates={"models_version": 0},
        ),
    )

    result = ex._submit_ferebus_phase(
        SimpleNamespace(
            iteration=0,
            campaign_uid="m16-test",
            reference_data_version=0,
        ),
        "INITIAL_FEREBUS",
    )

    result.validate(stage="submit", phase_name="INITIAL_FEREBUS")
    assert result.failure_reason is None
    assert result.state_updates == {"models_version": 0}
    assert statuses == ["accepted"]


def test_recovery_can_record_terminal_status_after_staging_is_moved(tmp_path):
    from ichor.hpc.active_learning.daemon.ferebus_candidate_recovery import (
        prepare_staging_recovery_request,
        read_recovery_request,
        update_recovery_status,
    )
    from ichor.hpc.active_learning.daemon.state import atomic_write_json

    campaign = tmp_path / "campaign"
    staging = campaign / "TRAINED_MODELS" / "iteration-staging"
    staging.mkdir(parents=True)
    atomic_write_json(staging / "FEREBUS_TASKS.json", {"tasks": []})
    atomic_write_json(staging / "FEREBUS_TASK_MAP.json", {"tasks": []})
    attempt = campaign / ".DATA" / "ACTIVE_LEARNING" / "attempt.json"
    attempt.parent.mkdir(parents=True)
    atomic_write_json(attempt, {"schema_version": 1})
    prepare_staging_recovery_request(
        campaign,
        campaign_uid="recovery-test",
        phase="INITIAL_FEREBUS",
        iteration=0,
        reference_data_version=0,
        staging_dir=staging,
        quality_attempt_path=attempt,
    )
    quarantine = campaign / "TRAINED_MODELS" / "rejected-candidate"
    staging.rename(quarantine)

    update_recovery_status(
        campaign,
        campaign_uid="recovery-test",
        status="rejected",
        last_error="quality threshold failed",
    )

    request = read_recovery_request(
        campaign,
        expected_campaign_uid="recovery-test",
    )
    assert request is not None
    assert request["status"] == "rejected"
    assert request["last_error"] == "quality threshold failed"


def test_trained_model_resolver_rejects_post_commit_model_tamper(tmp_path):
    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    _seed_models_staging(ex.campaign_dir)
    result = ex._parse_ferebus_postprocess(
        SimpleNamespace(iteration=0, campaign_uid="m16-test", reference_data_version=0),
        CampaignPhase("FEREBUS"),
        observations=[],
    )
    assert result.failure_reason is None
    versioning = TrainedModelVersioning(ex.campaign_dir / "TRAINED_MODELS")
    model_set = versioning.resolve(0, verification="deep")
    model = model_set.tasks[0].model.path
    model.write_text(
        model.read_text(encoding="utf-8") + "\n# tamper\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestMismatchError):
        versioning.resolve(0, verification="deep")


def test_trained_model_resolver_rejects_post_commit_dataset_tamper(tmp_path):
    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    _seed_models_staging(ex.campaign_dir)
    result = ex._parse_ferebus_postprocess(
        SimpleNamespace(iteration=0, campaign_uid="m16-test", reference_data_version=0),
        CampaignPhase("FEREBUS"),
        observations=[],
    )
    assert result.failure_reason is None
    versioning = TrainedModelVersioning(ex.campaign_dir / "TRAINED_MODELS")
    model_set = versioning.resolve(0, verification="deep")
    dataset = model_set.tasks[0].datasets["train"].path
    dataset.write_text(
        dataset.read_text(encoding="utf-8") + "0.9,0.9,0.9,0.0\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ManifestMismatchError):
        versioning.resolve(0, verification="deep")


def test_ferebus_task_artefact_layout_rejects_unsafe_tokens(tmp_path):
    staging = tmp_path / "campaign" / "TRAINED_MODELS" / "iteration-staging"
    committed = tmp_path / "campaign" / "TRAINED_MODELS" / "iteration-000000.staging"
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
                "reference_data_version": 0,
                "tasks": [{
                    "property": "iqa/bad",
                    "atom": "O1",
                    "expected_model_path": str(model),
                    "config_path": str(config),
                }],
            },
            models_version=0,
            parent_model_set=None,
        )


def test_initial_ferebus_consumes_published_reference_data_version_zero(tmp_path):
    ex = _make_executor(tmp_path)
    _commit_bootstrap_reference_data(ex.campaign_dir)
    _seed_phase_a_sample(ex.campaign_dir, n_frames=9)
    _seed_models_staging(tmp_path / "campaign")

    state = SimpleNamespace(
        iteration=0,
        campaign_uid="m16-test",
        reference_data_version=0,
    )
    result = ex._parse_ferebus_postprocess(
        state, CampaignPhase("INITIAL_FEREBUS"), observations=[],
    )
    assert result.failure_reason is None
    assert result.state_updates["models_version"] == 0
    assert "reference_data_version" not in result.state_updates
    train_dir = tmp_path / "campaign" / "QM_REFERENCE_DATA" / "iteration-000000"
    assert train_dir.is_dir()
    assert (train_dir / "POINT_000000.pointdir").is_dir()


def test_initial_ferebus_rejects_missing_published_reference_version(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(
        iteration=0,
        campaign_uid="m16-test",
        reference_data_version=-1,
    )

    result = ex._parse_ferebus_postprocess(
        state,
        CampaignPhase("INITIAL_FEREBUS"),
        observations=[],
    )

    assert result.failure_reason == "initial_ferebus_requires_reference_data_version_zero"


def test_ferebus_parser_rejects_empty_staging(tmp_path):
    ex = _make_executor(tmp_path)
    empty = tmp_path / "campaign" / "TRAINED_MODELS" / "iteration-staging"
    empty.mkdir(parents=True, exist_ok=True)
    state = SimpleNamespace(iteration=2, campaign_uid="m16-test", reference_data_version=0)
    result = ex._parse_ferebus_postprocess(
        state, CampaignPhase("FEREBUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ferebus_staging_invalid" in result.failure_reason


def test_ferebus_parser_rejects_missing_staging(tmp_path):
    ex = _make_executor(tmp_path)
    # Do not create the staging dir.
    state = SimpleNamespace(iteration=2, campaign_uid="m16-test", reference_data_version=0)
    result = ex._parse_ferebus_postprocess(
        state, CampaignPhase("FEREBUS"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ferebus_staging_invalid" in result.failure_reason



# --- ARIADNE parser tests ----------------------------------------


def _write_seeds_picked(campaign_dir, iteration, n_seeds):
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.handoff_manifests import (
        build_seed_selection_manifest,
        seeds_picked_path,
    )
    from ichor.hpc.active_learning.layout import active_iteration_dir
    from ichor.hpc.active_learning.seed_identity import write_ariadne_task_map
    from ichor.hpc.active_learning.daemon.submission_intent import (
        write_pre_submit_intent,
    )
    from ichor.hpc.active_learning.daemon.config_lock import (
        canonical_config,
        config_fingerprint,
        write_config_lock,
    )

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
    iter_dir = active_iteration_dir(campaign_dir, int(iteration))
    frame_ids = list(range(int(n_seeds)))
    payload = build_seed_selection_manifest(
        campaign_uid="m16-test",
        campaign_random_seed=0,
        iteration=int(iteration),
        models_version=0,
        model_manifest_sha256="c" * 64,
        model_set_sha256="d" * 64,
        trajectory_sha256=str(pool.sha256),
        selection_strategy="hybrid_variance",
        seed_records=[
            {
                "seed_id": i + 1,
                "frame_id": i,
                "pool_row_index_zero_based": i,
                "selection_origin": "bulk",
                "variance_at_selection": 0.0,
            }
            for i in frame_ids
        ],
    )
    selection_path = seeds_picked_path(iter_dir)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, payload)
    write_ariadne_task_map(iter_dir, payload)
    decision_config = (
        CampaignConfig.from_yaml(campaign_dir / "campaign.yaml")
        if (campaign_dir / "campaign.yaml").is_file()
        else CampaignConfig()
    )
    write_config_lock(
        campaign_dir,
        decision_config,
        campaign_uid="m16-test",
        reason="test_ariadne_intent",
    )
    write_pre_submit_intent(
        campaign_dir,
        campaign_uid="m16-test",
        phase_name=CampaignPhase.ARIADNE_ARRAY.value,
        iteration=int(iteration),
        expected_tasks=int(n_seeds),
        decision_contract={
            "failure_threshold_fraction": 0.5,
            "config_sha256": config_fingerprint(
                canonical_config(decision_config)
            ),
        },
    )
    return iter_dir


def _seed_ariadne_pool(campaign_dir, iteration, *, n_seeds=3):
    """Copy fixture results into canonical, integrity-bound seed outputs."""
    from ichor.hpc.active_learning.ariadne_outputs import (
        write_optimisation_trajectory,
        write_seed_output_manifest,
    )
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.handoff_manifests import load_seeds_picked
    from ichor.hpc.active_learning.layout import ariadne_seed_dir, ariadne_seeds_dir
    from ichor.hpc.active_learning.sampling_protocol import resolve_sampling_protocol
    from ichor.hpc.active_learning.versioning.manifest import sha256_file

    iter_dir = _write_seeds_picked(campaign_dir, iteration, n_seeds)
    pool_dir = ariadne_seeds_dir(iter_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    src = FIXTURES / "ariadne_pool"
    seeds_manifest = load_seeds_picked(
        iter_dir,
        expected_iteration=int(iteration),
    )
    trajectory_sha256 = str(seeds_manifest["trajectory_sha256"])
    campaign_config_path = Path(campaign_dir) / "campaign.yaml"
    campaign_config = (
        CampaignConfig.from_yaml(campaign_config_path)
        if campaign_config_path.is_file()
        else CampaignConfig()
    )
    resolved_protocol = resolve_sampling_protocol(
        campaign_dir,
        campaign_config,
        iteration=int(iteration),
    )

    def protocol_binding(path):
        resolved = path.resolve()
        return (
            resolved.relative_to(campaign_dir.resolve()).as_posix(),
            sha256_file(resolved),
        )

    resolved_manifest, resolved_manifest_sha256 = protocol_binding(
        resolved_protocol.manifest_path
    )
    scale_manifest, scale_manifest_sha256 = protocol_binding(
        resolved_protocol.scale_model_path
    )
    audit_manifest, audit_manifest_sha256 = protocol_binding(
        resolved_protocol.audit_manifest_path
    )
    for i in range(n_seeds):
        source_name = "seed_" + str(i).zfill(4)
        seed_id = i + 1
        seed_uid = str(seeds_manifest["seed_records"][i]["seed_uid"])
        target = ariadne_seed_dir(iter_dir, seed_id)
        target.mkdir(exist_ok=True)
        result_src = src / source_name / "result.json"
        if result_src.is_file():
            payload = json.loads(result_src.read_text(encoding="utf-8"))
            payload["iteration"] = int(iteration)
            payload["seed_id"] = seed_id
            payload["seed_uid"] = seed_uid
            payload["array_task_id"] = i
            payload["seed_frame_id"] = int(i)
            payload["trajectory_sha256"] = trajectory_sha256
            offset = 0.01 * float(i)
            initial_coordinates = [
                [offset, 0.0, 0.0],
                [0.96 + offset, 0.0, 0.0],
                [-0.24 + offset, 0.93, 0.0],
            ]
            payload["initial_coordinates"] = [
                list(row) for row in initial_coordinates
            ]
            payload["seed_coordinates"] = [
                list(row) for row in initial_coordinates
            ]
            payload["sampling_protocol"] = {
                "sampling_aggressiveness": int(
                    resolved_protocol.sampling_aggressiveness
                ),
                "resolved_manifest": resolved_manifest,
                "resolved_manifest_sha256": resolved_manifest_sha256,
                "scale_model_manifest": scale_manifest,
                "scale_model_manifest_sha256": scale_manifest_sha256,
                "audit_manifest": audit_manifest,
                "audit_manifest_sha256": audit_manifest_sha256,
            }
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
            atomic_write_json(target / "result.json", payload)
            coordinates = payload.get("final_coordinates") or [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ]
            write_optimisation_trajectory(
                target,
                atom_types=payload.get("atom_types") or ["O", "H", "H"],
                coordinate_frames=[coordinates],
                alpha_values=[payload.get("alpha_final")],
                gradient_norms=[0.0],
                origins=["raw_final"],
            )
            task_success = bool(payload.get("task_success", True))
            write_seed_output_manifest(
                target,
                campaign_uid="m16-test",
                iteration=int(iteration),
                seed_id=seed_id,
                seed_uid=seed_uid,
                array_task_id=i,
                task_success=task_success,
                task_exit_code=int(
                    payload.get("task_exit_code", 0 if task_success else 4)
                ),
            )
    return pool_dir


def _rewrite_seed_result(seed_dir, payload):
    from ichor.hpc.active_learning.ariadne_outputs import write_seed_output_manifest
    from ichor.hpc.active_learning.daemon.state import atomic_write_json

    atomic_write_json(seed_dir / "result.json", payload)
    task_success = bool(payload.get("task_success", True))
    write_seed_output_manifest(
        seed_dir,
        campaign_uid="m16-test",
        iteration=int(payload["iteration"]),
        seed_id=int(payload["seed_id"]),
        seed_uid=str(payload["seed_uid"]),
        array_task_id=int(payload["array_task_id"]),
        task_success=task_success,
        task_exit_code=int(
            payload.get("task_exit_code", 0 if task_success else 4)
        ),
    )


def test_ariadne_parser_happy_path(tmp_path):
    ex = _make_executor(tmp_path)
    _seed_ariadne_pool(tmp_path / "campaign", iteration=4)
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0)
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None
    # Fixture has alpha_final values 2.13, 1.87, 2.40. Max is 2.40 (or 2.13 if
    # only one historical source seed in the fixture). Just assert > 0 to be
    # fixture-robust.
    assert result.state_updates["last_acquisition_alpha0"] > 0.0
    assert "last_n_anti_overlap_flagged" in result.state_updates
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded
    assert succeeded[-1]["phase"] == "ARIADNE_ARRAY"


def test_ariadne_parser_ignores_hidden_raw_ariadne_quality_gate_override(tmp_path):
    ex = _make_executor(tmp_path)
    ex.config.quality_gates.ariadne_max_displacement_angstrom = 1.0e-8
    ex.config.anti_overlap.enforce_post_ariadne = True
    ex.config.anti_overlap.max_post_ariadne_whitened_distance = 1.0e-8
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    seed_dir = ariadne_seed_dir(active_iteration_dir(tmp_path / "campaign", 4), 1)
    result_path = seed_dir / "result.json"
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
    _rewrite_seed_result(seed_dir, payload)
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0)

    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )

    assert result.is_complete is True
    assert result.failure_reason is None
    iter_dir = active_iteration_dir(tmp_path / "campaign", 4)
    manifest = json.loads(ariadne_results_path(iter_dir).read_text(encoding="utf-8"))
    assert manifest["n_accepted"] == 1
    assert manifest["n_rejected"] == 0


def test_ariadne_parser_uses_result_resolved_protocol_manifest(tmp_path):
    from ichor.hpc.active_learning.versioning.manifest import sha256_file

    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    iter_dir = active_iteration_dir(tmp_path / "campaign", 4)
    protocol_path = sampling_protocol_resolved_path(iter_dir)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["resolved_quality_gates"]["ariadne_max_displacement_ang"] = 1.0e-8
    protocol_path.write_text(
        json.dumps(protocol),
        encoding="utf-8",
        newline="\n",
    )
    seed_dir = ariadne_seed_dir(iter_dir, 1)
    result_path = seed_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["task_success"] = True
    payload["sampling_protocol"]["sampling_aggressiveness"] = 5
    payload["sampling_protocol"]["resolved_manifest"] = str(
        protocol_path.resolve()
    )
    payload["sampling_protocol"]["resolved_manifest_sha256"] = sha256_file(
        protocol_path
    )
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
    _rewrite_seed_result(seed_dir, payload)
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0)

    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )

    assert result.failure_reason == "ariadne_no_usable_seed_results: 1"
    manifest = json.loads(ariadne_results_path(iter_dir).read_text(encoding="utf-8"))
    assert manifest["n_accepted"] == 0
    assert manifest["rejected"][0]["reason"] == "ariadne_max_displacement_threshold_exceeded"
    audit = json.loads(ariadne_landing_audit_path(iter_dir).read_text(encoding="utf-8"))
    replay = audit["seeds"][0]["sampling_protocol_replay"]
    assert replay["used_exact_sampling_protocol"] is True
    assert replay["sampling_protocol_source"] == "result_manifest"


def test_ariadne_parser_rejects_sampling_protocol_hash_drift(tmp_path):
    ex = _make_executor(tmp_path)
    _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    iter_dir = active_iteration_dir(tmp_path / "campaign", 4)
    protocol_path = sampling_protocol_resolved_path(iter_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["sampling_aggressiveness"] = int(
        protocol["sampling_aggressiveness"]
    ) + 1
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    state = SimpleNamespace(
        iteration=4,
        campaign_uid="m16-test",
        models_version=0,
    )
    result = ex._parse_ariadne_array_postprocess(
        state,
        CampaignPhase("ARIADNE_ARRAY"),
        observations=[],
    )

    assert result.failure_reason == "ariadne_no_usable_seed_results: 1"
    manifest = json.loads(
        ariadne_results_path(iter_dir).read_text(encoding="utf-8")
    )
    assert "resolved_manifest SHA-256 mismatch" in manifest["rejected"][0][
        "reason"
    ]


def test_ariadne_parser_reconstructs_missing_seed_provenance(tmp_path):
    from ichor.hpc.active_learning.versioning.provenance import (
        PROVENANCE_FILENAME,
        read_provenance,
    )

    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    iter_dir = active_iteration_dir(tmp_path / "campaign", 4)
    seed_dir = ariadne_seed_dir(iter_dir, 1)
    prov_path = seed_dir / PROVENANCE_FILENAME
    assert not prov_path.exists()

    state = SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0)
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )

    assert result.failure_reason is None
    assert prov_path.is_file()
    prov = read_provenance(seed_dir)
    assert prov["seed"]["frame_id"] == 0
    assert prov["ariadne"] is not None
    assert prov["anti_overlap"] is not None

    manifest = json.loads(ariadne_results_path(iter_dir).read_text(encoding="utf-8"))
    assert manifest["n_accepted"] == 1
    assert manifest["n_rejected"] == 0
    assert manifest["accepted"][0]["provenance_json"].endswith(
        "seeds/seed-000001/provenance.json"
    )
    audit = json.loads(ariadne_landing_audit_path(iter_dir).read_text(encoding="utf-8"))
    assert len(audit["seeds"]) == 1
    assert audit["seeds"][0]["provenance_created"] is True
    assert audit["summary"]["handoff_accepted"] == 1
    events = _read_journal_events(tmp_path / "campaign")
    assert any(e.get("event") == "ariadne_provenance_reconstructed" for e in events)


def test_ariadne_parser_accepts_safe_max_iteration_result(tmp_path):
    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4, n_seeds=1)
    seed_dir = ariadne_seed_dir(active_iteration_dir(tmp_path / "campaign", 4), 1)
    result_path = seed_dir / "result.json"
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
    _rewrite_seed_result(seed_dir, payload)
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0)

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
    seed_dir = ariadne_seed_dir(active_iteration_dir(tmp_path / "campaign", 4), 1)
    result_path = seed_dir / "result.json"
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
    _rewrite_seed_result(seed_dir, payload)
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0)

    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )

    assert result.failure_reason == "ariadne_no_usable_seed_results: 1"
    iter_dir = active_iteration_dir(tmp_path / "campaign", 4)
    manifest = json.loads(ariadne_results_path(iter_dir).read_text(encoding="utf-8"))
    assert manifest["n_accepted"] == 0
    assert manifest["n_rejected"] == 1
    assert manifest["rejected"][0]["reason"] == (
        "ariadne_unusable:runner_failed_after_result_write"
    )
    audit = json.loads(
        ariadne_landing_audit_path(iter_dir).read_text(encoding="utf-8")
    )
    assert audit["seeds"][0]["handoff_accepted"] is False
    assert audit["seeds"][0]["handoff_rejection_reason"] == (
        "ariadne_unusable:runner_failed_after_result_write"
    )
    summary = next(
        event
        for event in _read_journal_events(tmp_path / "campaign")
        if event.get("event") == "ariadne_landing_summary"
    )
    assert summary["n_expected"] == 1
    assert summary["n_accepted"] == 0
    assert summary["n_rejected"] == 1
    assert summary["dominant_rejection_reasons"] == [
        {
            "reason": "ariadne_unusable:runner_failed_after_result_write",
            "count": 1,
        }
    ]


def test_ariadne_parser_missing_pool_dir(tmp_path):
    ex = _make_executor(tmp_path)
    # Do not seed the pool dir.
    _write_seeds_picked(tmp_path / "campaign", 4, 3)
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0)
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ariadne_seeds_directory_missing" in result.failure_reason


def test_ariadne_parser_no_seeds_in_pool(tmp_path):
    ex = _make_executor(tmp_path)
    _write_seeds_picked(tmp_path / "campaign", 4, 1)
    iter_dir = active_iteration_dir(tmp_path / "campaign", 4)
    ariadne_seeds_dir(iter_dir).mkdir(parents=True, exist_ok=True)
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0)
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ariadne_no_usable_seed_results" in result.failure_reason


def test_ariadne_parser_handles_missing_result_json(tmp_path):
    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4)
    # Remove one result.json so the parser sees a half-missing pool.
    (ariadne_seed_dir(active_iteration_dir(tmp_path / "campaign", 4), 2) / "result.json").unlink()
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0)
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    # Two seeds still have a valid result -> succeeds.
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    rejected = [
        event
        for event in events
        if event.get("event") == "ariadne_task_rejected_invalid_output"
    ]
    assert any("result.json" in str(event.get("reason")) for event in rejected)


def test_ariadne_parser_all_results_unreadable_fails(tmp_path):
    ex = _make_executor(tmp_path)
    pool = _seed_ariadne_pool(tmp_path / "campaign", iteration=4)
    for seed_dir in pool.iterdir():
        if seed_dir.is_dir():
            r = seed_dir / "result.json"
            if r.is_file():
                r.unlink()
    state = SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0)
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "ariadne_no_usable_seed_results" in result.failure_reason


def test_clean_stale_ariadne_seed_outputs_quarantines_selected_task(tmp_path):
    campaign = tmp_path / "campaign"
    pool = _seed_ariadne_pool(campaign, iteration=4)
    iter_dir = active_iteration_dir(campaign, 4)
    seed_dir = ariadne_seed_dir(iter_dir, 1)
    tmp_result = seed_dir / "result.json.1234.abcd.tmp"
    tmp_result.write_text("partial", encoding="utf-8")
    trace = seed_dir / "trajectory" / "trace.jsonl"
    trace.write_text('{"event":"old"}\n', encoding="utf-8")
    protected = seeds_picked_path(iter_dir)
    assert protected.is_file()

    removed = clean_stale_ariadne_seed_outputs(
        campaign,
        4,
        retry_array_task_ids=[0],
    )

    assert len(removed) == 1
    quarantined = Path(removed[0])
    assert (quarantined / "result.json").is_file()
    assert (quarantined / tmp_result.name).is_file()
    assert (quarantined / "trajectory" / trace.name).is_file()
    assert protected.is_file()
    assert not seed_dir.exists()


def test_clean_stale_ariadne_seed_outputs_rejects_non_regular_result(tmp_path):
    campaign = tmp_path / "campaign"
    pool = _seed_ariadne_pool(campaign, iteration=4)
    seed_dir = ariadne_seed_dir(active_iteration_dir(campaign, 4), 1)
    shutil.rmtree(seed_dir)
    seed_dir.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(BackendSubmissionError, match="non-directory ARIADNE output"):
        clean_stale_ariadne_seed_outputs(
            campaign,
            4,
            retry_array_task_ids=[0],
        )


class _FakeSbatch:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")


@pytest.mark.parametrize("scheduler_kind", ["slurm", "sge"])
def test_submission_environment_guard_runs_before_scheduler_acceptance(
    tmp_path,
    monkeypatch,
    scheduler_kind,
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
    ex.scheduler_identity_kind = scheduler_kind
    ex._scheduler_backend = live_executor_mod.get_scheduler_backend(
        scheduler_kind
    )
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid="binding-guard-test",
        phase_name="GAUSSIAN",
        iteration=4,
        expected_tasks=1,
        scheduler_identity_kind=scheduler_kind,
    )
    monkeypatch.setattr(ex, "_array_size_after_staging", lambda *_args: 1)

    def write_bound_fixture_script(
        phase,
        state,
        array_size,
        array_task_map=None,
    ):
        phase_name = phase.value if isinstance(phase, CampaignPhase) else str(phase)
        intent = submission_intent.load_active_intent(
            campaign,
            phase_name,
            state.iteration,
        )
        bundle = prepare_attempt_bundle(
            campaign,
            phase_name,
            state.iteration,
            str(intent["submission_identity"]),
            array_size=array_size,
            max_log_files_per_directory=5000,
            source_array_task_map=array_task_map,
        )
        script = write_attempt_script(bundle, "#!/bin/bash\ntrue\n")
        binding = write_script_binding(bundle)
        submission_intent.bind_submission_script(
            campaign,
            phase_name,
            state.iteration,
            script_path=str(script.resolve()),
            script_sha256=str(binding["script_sha256"]),
            binding_path=str(binding["path"]),
            binding_sha256=str(binding["sha256"]),
        )
        return script

    monkeypatch.setattr(ex, "_write_real_script", write_bound_fixture_script)
    monkeypatch.setattr(
        live_executor_mod,
        "prepare_retry_submission",
        lambda *_args, **_kwargs: {
            "phase": "GAUSSIAN",
            "iteration": 4,
            "logical_total": 1,
            "n_complete": 0,
            "n_reuse": 0,
            "n_retry": 1,
            "force_resubmit": False,
            "retry_task_ids": [0],
            "path": str(campaign / "ledger.json"),
            "retry_task_file": None,
        },
    )
    ex._submission_environment_guard = lambda _intent: (_ for _ in ()).throw(
        ValueError("stale campaign configuration")
    )

    with pytest.raises(
        BackendSubmissionError,
        match="submission environment binding failed",
    ):
        ex.submit_or_run(
            SimpleNamespace(iteration=4, campaign_uid="binding-guard-test"),
            CampaignPhase.GAUSSIAN,
        )

    assert runner.calls == []


def test_phase_b_resource_contract_failure_has_precise_pre_submit_message(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=CampaignConfig(),
        backend_check=False,
        sbatch_runner=_FakeSbatch(),
    )
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid="phase-b-resource-message",
        phase_name="PHASE_B_DIVERSITY",
        iteration=19,
        expected_tasks=1,
    )
    monkeypatch.setattr(
        live_executor_mod,
        "resolve_phase_resources",
        lambda **_kwargs: (_ for _ in ()).throw(
            live_executor_mod.ResourceEvidenceInvalid(
                "PHASE_B_DIVERSITY",
                "ARIADNE decision contract is not bound to a producer environment",
            )
        ),
    )

    with pytest.raises(
        BackendSubmissionError,
        match="Phase B resource evidence validation failed",
    ):
        ex._write_real_script(
            "PHASE_B_DIVERSITY",
            SimpleNamespace(
                iteration=19,
                campaign_uid="phase-b-resource-message",
                replacement_round=0,
                models_version=18,
            ),
        )


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
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid="m16-test",
        phase_name="GAUSSIAN",
        iteration=4,
        expected_tasks=3,
    )
    monkeypatch.setattr(ex, "_array_size_after_staging", lambda _phase, _state: 3)

    def write_bound_fixture_script(
        phase,
        state,
        array_size,
        array_task_map=None,
    ):
        phase_name = phase.value if isinstance(phase, CampaignPhase) else str(phase)
        intent = submission_intent.load_active_intent(
            campaign,
            phase_name,
            state.iteration,
        )
        assert intent is not None
        bundle = prepare_attempt_bundle(
            campaign,
            phase_name,
            state.iteration,
            str(intent["submission_identity"]),
            array_size=array_size,
            max_log_files_per_directory=5000,
            source_array_task_map=array_task_map,
        )
        script = write_attempt_script(bundle, "#!/bin/bash\ntrue\n")
        binding = write_script_binding(bundle)
        submission_intent.bind_submission_script(
            campaign,
            phase_name,
            state.iteration,
            script_path=str(script.resolve()),
            script_sha256=str(binding["script_sha256"]),
            binding_path=str(binding["path"]),
            binding_sha256=str(binding["sha256"]),
        )
        return script

    monkeypatch.setattr(
        ex,
        "_write_real_script",
        write_bound_fixture_script,
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
    iter_dir = active_iteration_dir(tmp_path / "campaign", 4)
    seed_dir = ariadne_seed_dir(iter_dir, 1)
    stale_result = seed_dir / "result.json"
    assert stale_result.is_file()

    result = ex.submit_or_run(
        SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0),
        CampaignPhase("ARIADNE_ARRAY"),
    )

    assert result.is_complete is True
    assert result.submitted_job_id is None
    assert result.failure_reason is None
    assert not runner.calls
    assert stale_result.is_file()
    prov_path = seed_dir / PROVENANCE_FILENAME
    assert prov_path.is_file()
    prov = read_provenance(seed_dir)
    assert prov["seed"]["frame_id"] == 0
    events = _read_journal_events(tmp_path / "campaign")
    reused = [
        event
        for event in events
        if event.get("event") == "partial_array_recovery_postprocess_only"
    ]
    assert reused
    assert reused[-1]["phase"] == "ARIADNE_ARRAY"


def test_ariadne_postprocess_only_archives_incomplete_batch_publication(tmp_path):
    cfg = CampaignConfig()
    runner = _FakeSbatch()
    campaign = tmp_path / "campaign"
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=cfg,
        backend_check=False,
        sbatch_runner=runner,
    )
    _seed_ariadne_pool(campaign, iteration=4)
    iteration_dir = active_iteration_dir(campaign, 4)
    ariadne_root = active_ariadne_dir(iteration_dir)
    atomic_write_json(
        ariadne_root / "RESULTS.json",
        {"stale": True},
    )

    result = ex.submit_or_run(
        SimpleNamespace(iteration=4, campaign_uid="m16-test", models_version=0),
        CampaignPhase("ARIADNE_ARRAY"),
    )

    assert result.is_complete is True
    assert result.failure_reason is None
    assert not runner.calls
    archives = list(ariadne_publication_archive_root(campaign, 4).iterdir())
    assert len(archives) == 1
    assert (archives[0] / "RESULTS.json").is_file()
    events = _read_journal_events(campaign)
    archived = [
        event
        for event in events
        if event.get("event") == "ariadne_publication_archived"
    ]
    assert archived
    assert archived[-1]["reason"] == "ariadne_postprocess_only_recovery"


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
    seed_dir = ariadne_seed_dir(active_iteration_dir(tmp_path / "campaign", 4), 1)
    selection = json.loads(seeds_picked_path(active_iteration_dir(tmp_path / "campaign", 4)).read_text(encoding="utf-8"))
    write_seed_provenance(
        seed_dir,
        campaign_uid="wrong-campaign",
        iteration=4,
        trajectory_sha256="0" * 64,
        seed_frame_id=0,
        seed_id=1,
        seed_uid=str(selection["seed_records"][0]["seed_uid"]),
        array_task_id_zero_based=0,
        seed_selection_origin="bulk",
        seed_variance_at_selection=0.0,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
        mode_weighting_policy="variance",
    )

    result = ex.submit_or_run(
        SimpleNamespace(
            iteration=4,
            campaign_uid="m16-test",
            models_version=0,
        ),
        CampaignPhase("ARIADNE_ARRAY"),
    )

    assert result.failure_reason == "ariadne_no_usable_seed_results: 1"
    manifest = json.loads(
        ariadne_results_path(
            active_iteration_dir(tmp_path / "campaign", 4)
        ).read_text(encoding="utf-8")
    )
    assert "ariadne_provenance_missing" in manifest["rejected"][0]["reason"]
    assert "campaign_uid mismatch" in manifest["rejected"][0]["reason"]
    assert not runner.calls



# --- POLUS parser tests ------------------------------------------


def _seed_phase_a_sample(campaign_dir, *, n_frames=2):
    """Publish a canonical Phase-A handoff from the legacy-named fixture."""
    from ichor.hpc.active_learning.handoff_manifests import write_phase_a_sample_manifest
    from ichor.hpc.active_learning.layout import bootstrap_selection_dir
    from ichor.hpc.active_learning.sampling.diversity_contract import (
        diversity_selector_contract,
    )
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool

    target = bootstrap_selection_dir(campaign_dir)
    target.mkdir(parents=True, exist_ok=True)
    dst = target / "selected.xyz"
    index = target / "selected_indices.dat"
    allocation_path = point_allocation_path(
        campaign_dir,
        context="bootstrap",
        iteration=0,
    )
    if allocation_path.is_file():
        from ichor.hpc.active_learning.point_allocation import read_point_allocation

        allocation = read_point_allocation(allocation_path)
    else:
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
    selected_indices = [int(record["frame_id"]) for record in primary]
    source_lines = (
        FIXTURES / "polus_phase_a" / "initial-SAMPLE-2.xyz"
    ).read_text(encoding="utf-8").splitlines()
    frame_size = int(source_lines[0]) + 2
    frames = [
        source_lines[offset : offset + frame_size]
        for offset in range(0, len(source_lines), frame_size)
    ]
    selected_frames = []
    for frame_id in selected_indices:
        frame = list(frames[frame_id % len(frames)])
        frame[1] = "frame " + str(frame_id)
        selected_frames.append(frame)
    dst.write_text(
        "\n".join(line for frame in selected_frames for line in frame) + "\n",
        encoding="utf-8",
    )
    pool_source = campaign_dir / "phase_a_pool_source.xyz"
    pool_source.write_text(
        "\n".join(line for frame in selected_frames for line in frame) + "\n",
        encoding="utf-8",
    )
    pool = TrajectoryPool.import_from(
        pool_source,
        campaign_dir,
        overwrite=True,
    )
    index.write_text(
        "\n".join(str(frame_id) for frame_id in selected_indices) + "\n",
        encoding="utf-8",
    )
    write_phase_a_sample_manifest(target, {
        "phase": "PHASE_A_DIVERSITY",
        "iteration": 0,
        "sample_xyz": str(dst.resolve()),
        "index_path": str(index.resolve()),
        "n_select": int(n_frames),
        "n_frames": int(n_frames),
        "selected_indices": selected_indices,
        "descriptor": "rmsd_massweight",
        "selector": diversity_selector_contract(),
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
        "trajectory_sha256": str(pool.sha256),
        "source_pool_manifest": ".DATA/TRAJECTORY/pool.manifest.json",
    })
    return target


def _seed_phase_b_sample(campaign_dir, iteration):
    """Return the canonical active iteration used by Phase B fixtures."""
    return active_iteration_dir(campaign_dir, int(iteration))


def _write_phase_b_manifest(iter_dir, *, n_final=1, write_sample=True):
    from ichor.hpc.active_learning.daemon.state import (
        DEFAULT_STATE_FILENAME,
        CampaignPhase as _CampaignPhase,
        fresh_campaign_state,
        write_state,
    )
    from ichor.hpc.active_learning.handoff_manifests import (
        phase_b_selection_path,
        read_phase_b_selection_manifest,
    )
    from ichor.hpc.active_learning.layout import active_phase_b_dir
    from ichor.hpc.active_learning.sampling.diversity import main as polus_main

    iteration = int(iter_dir.name.split("-")[-1])
    campaign = iter_dir.parent.parent
    cfg = CampaignConfig(max_iterations=max(1, iteration))
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.point_allocation.batch_training_size = int(n_final)
    cfg.point_allocation.batch_internal_validation_size = 0
    cfg.seed_selection.n_seeds_per_iteration = int(n_final)
    cfg.to_yaml(campaign / "campaign.yaml")
    state = fresh_campaign_state(max_iterations=max(1, iteration))
    state.campaign_uid = "m16-test"
    state.phase = _CampaignPhase.ARIADNE_ARRAY
    state.iteration = iteration
    state.reference_data_version = 0
    state.models_version = 0
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(state_path, state)
    _seed_ariadne_pool(campaign, iteration=iteration, n_seeds=int(n_final))
    for seed_index in range(int(n_final)):
        seed_dir = ariadne_seed_dir(iter_dir, seed_index + 1)
        result_path = seed_dir / "result.json"
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        coordinates = [list(row) for row in result_payload["final_coordinates"]]
        coordinates[1][0] += 0.60 * seed_index
        coordinates[2][1] += 0.40 * seed_index
        result_payload["final_coordinates"] = coordinates
        result_payload["landing_safety"]["metrics"][
            "max_displacement_ang"
        ] = 0.60 * seed_index
        _rewrite_seed_result(seed_dir, result_payload)
    executor = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=cfg,
        backend_check=False,
    )
    parsed = executor._parse_ariadne_array_postprocess(
        state,
        _CampaignPhase.ARIADNE_ARRAY,
        observations=[],
    )
    assert parsed.failure_reason is None
    state.phase = _CampaignPhase.PHASE_B_DIVERSITY
    write_state(state_path, state)
    rc = polus_main([
        "--descriptor",
        "rmsd_massweight",
        "--iteration",
        str(iteration),
        "--campaign-dir",
        str(campaign),
    ])
    assert rc == 0
    payload = read_phase_b_selection_manifest(
        iter_dir,
        expected_iteration=iteration,
    )
    if not write_sample:
        selected = active_phase_b_dir(iter_dir) / "selected.xyz"
        selected.unlink()
    assert phase_b_selection_path(iter_dir).is_file()
    return list(payload["final"])


def test_polus_phase_a_happy_path(tmp_path):
    ex = _make_executor(tmp_path)
    _seed_phase_a_sample(tmp_path / "campaign", n_frames=2)
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_diversity_postprocess(
        state, CampaignPhase("PHASE_A_DIVERSITY"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded
    assert succeeded[-1]["phase"] == "PHASE_A_DIVERSITY"
    assert succeeded[-1]["n_frames"] == 2


def test_polus_phase_a_missing_outdir(tmp_path):
    ex = _make_executor(tmp_path)
    # The executor creates the daemon bootstrap root, so remove it to exercise
    # the missing-output path explicitly.
    import shutil
    target = bootstrap_selection_dir(tmp_path / "campaign")
    if target.exists():
        shutil.rmtree(target.parent)
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_diversity_postprocess(
        state, CampaignPhase("PHASE_A_DIVERSITY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "phase_a" in result.failure_reason


def test_polus_phase_a_no_sample_in_outdir(tmp_path):
    ex = _make_executor(tmp_path)
    # The diversity dir is created at __post_init__ time but no sample yet.
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    target = bootstrap_selection_dir(tmp_path / "campaign")
    target.mkdir(parents=True, exist_ok=True)
    result = ex._parse_diversity_postprocess(
        state, CampaignPhase("PHASE_A_DIVERSITY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "phase_a_sample_manifest_invalid" in result.failure_reason


def test_polus_phase_a_uses_manifest_not_lexical_last(tmp_path):
    ex = _make_executor(tmp_path)
    outdir = _seed_phase_a_sample(tmp_path / "campaign", n_frames=2)
    stale = outdir / "retired-sample-name.xyz"
    stale.write_text("garbage no frames here", encoding="utf-8")
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_diversity_postprocess(
        state, CampaignPhase("PHASE_A_DIVERSITY"), observations=[],
    )
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded[-1]["sample_path"].endswith("selected.xyz")


def test_polus_phase_a_manifest_count_mismatch_fails(tmp_path):
    from ichor.hpc.active_learning.handoff_manifests import write_phase_a_sample_manifest

    ex = _make_executor(tmp_path)
    outdir = _seed_phase_a_sample(tmp_path / "campaign", n_frames=2)
    sample = outdir / "selected.xyz"
    index = outdir / "selected_indices.dat"
    existing = json.loads((outdir / "SELECTION.json").read_text(encoding="utf-8"))
    allocation = dict(existing["point_allocation"])
    allocation["primary"] = list(allocation["primary"][:1])
    write_phase_a_sample_manifest(outdir, {
        "phase": "PHASE_A_DIVERSITY",
        "iteration": 0,
        "sample_xyz": str(sample.resolve()),
        "index_path": str(index.resolve()),
        "n_select": 1,
        "n_frames": 1,
        "selected_indices": [0],
        "descriptor": "rmsd_massweight",
        "selector": dict(existing["selector"]),
        "source_pool_manifest": ".DATA/TRAJECTORY/pool.manifest.json",
        "point_allocation": allocation,
    })
    state = SimpleNamespace(iteration=0, campaign_uid="m16-test")
    result = ex._parse_diversity_postprocess(
        state, CampaignPhase("PHASE_A_DIVERSITY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "Phase A sample XYZ frame count mismatch" in result.failure_reason


def test_polus_phase_b_happy_path(tmp_path):
    ex = _make_executor(tmp_path)
    iter_dir = _seed_phase_b_sample(tmp_path / "campaign", iteration=5)
    _write_phase_b_manifest(iter_dir, n_final=2)
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_diversity_postprocess(
        state, CampaignPhase("PHASE_B_DIVERSITY"), observations=[],
    )
    assert result.is_complete is True
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    succeeded = [e for e in events if e.get("event") == "phase_succeeded_live"]
    assert succeeded
    assert succeeded[-1]["phase"] == "PHASE_B_DIVERSITY"


def test_polus_phase_b_recovery_defers_success_reporting_until_adoption(
    tmp_path,
):
    ex = _make_executor(tmp_path)
    iter_dir = _seed_phase_b_sample(tmp_path / "campaign", iteration=5)
    _write_phase_b_manifest(iter_dir, n_final=2)
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")

    result = ex._parse_diversity_postprocess(
        state,
        CampaignPhase("PHASE_B_DIVERSITY"),
        observations=[],
        emit_success_events=False,
    )

    assert result.is_complete is True
    assert result.failure_reason is None
    events = _read_journal_events(tmp_path / "campaign")
    assert not [
        event
        for event in events
        if event.get("event") == "phase_succeeded_live"
        and event.get("phase") == "PHASE_B_DIVERSITY"
    ]
    deferred = result.journal_events
    assert deferred[-1]["event"] == "phase_succeeded_live"
    assert deferred[-1]["phase"] == "PHASE_B_DIVERSITY"


def test_polus_phase_b_missing_sample(tmp_path):
    ex = _make_executor(tmp_path)
    iter_dir = active_iteration_dir(tmp_path / "campaign", 5)
    _write_phase_b_manifest(iter_dir, n_final=1, write_sample=False)
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_diversity_postprocess(
        state, CampaignPhase("PHASE_B_DIVERSITY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "phase_b_handoff_invalid" in result.failure_reason


def test_polus_phase_b_raw_sample_only_is_rejected(tmp_path):
    ex = _make_executor(tmp_path)
    iter_dir = active_iteration_dir(tmp_path / "campaign", 5)
    _write_phase_b_manifest(iter_dir, n_final=1, write_sample=False)
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_diversity_postprocess(
        state, CampaignPhase("PHASE_B_DIVERSITY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "phase_b_handoff_invalid" in result.failure_reason


def test_polus_phase_b_enriches_seed_provenance(tmp_path):
    """When Phase-B parser runs and the pool seed dirs have provenance,
    it enriches each seed with phase_b block."""
    from ichor.hpc.active_learning.versioning.provenance import (
        read_provenance, write_seed_provenance,
    )
    ex = _make_executor(tmp_path)
    iter_dir = _seed_phase_b_sample(tmp_path / "campaign", iteration=5)
    records = _write_phase_b_manifest(iter_dir, n_final=2)
    seed_dir = Path(records[0]["seed_dir"])
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_diversity_postprocess(
        state, CampaignPhase("PHASE_B_DIVERSITY"), observations=[],
    )
    assert result.failure_reason is None
    prov = read_provenance(seed_dir)
    assert prov["phase_b"] is not None
    assert prov["phase_b"]["selected_after_fps"] is True
    assert prov["phase_b"]["diversity_rank"] == 1


def test_polus_phase_b_unreadable_sample_fails(tmp_path):
    ex = _make_executor(tmp_path)
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.handoff_manifests import phase_b_selection_path
    from ichor.hpc.active_learning.layout import active_phase_b_dir
    from ichor.hpc.active_learning.versioning.manifest import sha256_file

    iter_dir = active_iteration_dir(tmp_path / "campaign", 5)
    _write_phase_b_manifest(iter_dir, n_final=1, write_sample=True)
    sample = active_phase_b_dir(iter_dir) / "selected.xyz"
    sample.write_text("garbage no frames here", encoding="utf-8")
    manifest_path = phase_b_selection_path(iter_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_xyz"]["size"] = int(sample.stat().st_size)
    manifest["selected_xyz"]["sha256"] = sha256_file(sample)
    atomic_write_json(manifest_path, manifest)
    state = SimpleNamespace(iteration=5, campaign_uid="m16-test")
    result = ex._parse_diversity_postprocess(
        state, CampaignPhase("PHASE_B_DIVERSITY"), observations=[],
    )
    assert result.failure_reason is not None
    assert "phase_b_handoff_invalid" in result.failure_reason


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
