from __future__ import annotations

from pathlib import Path

import pytest

from ichor.hpc.active_learning.daemon import array_recovery
from ichor.hpc.active_learning.daemon.live_executor import build_sbatch_script
from ichor.hpc.active_learning.config import CampaignConfig


def _write_points(campaign: Path, phase: str, iteration: int, n: int) -> None:
    bucket = "initial" if phase.startswith("INITIAL_") else "iter_" + str(iteration)
    staging = campaign / ".DATA" / "STAGING" / bucket
    staging.mkdir(parents=True)
    pointdirs = []
    for idx in range(n):
        pointdir = staging / ("POINT_" + str(idx).zfill(4) + ".pointdir")
        pointdir.mkdir()
        pointdirs.append(pointdir)
    (staging / "POINTS.txt").write_text(
        "\n".join(str(path.resolve()) for path in pointdirs) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_prepare_retry_submission_writes_only_incomplete_task_ids(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    phase = "INITIAL_GAUSSIAN"
    _write_points(campaign, phase, 0, 4)

    def fake_validate(_campaign, _phase, _iteration, task_id):
        return (int(task_id) in {0, 2}, "" if int(task_id) in {0, 2} else "missing", "out")

    monkeypatch.setattr(array_recovery, "_validate_task", fake_validate)
    monkeypatch.setattr(
        array_recovery,
        "logical_task_ids",
        lambda *_args, **_kwargs: [0, 1, 2, 3],
    )

    payload = array_recovery.prepare_retry_submission(campaign, phase, 0)

    assert payload["logical_total"] == 4
    assert payload["n_reuse"] == 2
    assert payload["retry_task_ids"] == [1, 3]
    retry_file = Path(payload["retry_task_file"])
    assert retry_file.read_text(encoding="utf-8").splitlines() == ["1", "3"]


def test_force_resubmit_array_ignores_reusable_outputs(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    phase = "GAUSSIAN"
    _write_points(campaign, phase, 7, 3)

    monkeypatch.setattr(
        array_recovery,
        "_validate_task",
        lambda _campaign, _phase, _iteration, task_id: (True, "", "out"),
    )
    monkeypatch.setattr(
        array_recovery,
        "logical_task_ids",
        lambda *_args, **_kwargs: [0, 1, 2],
    )

    payload = array_recovery.prepare_retry_submission(
        campaign,
        phase,
        7,
        force_resubmit=True,
    )

    assert payload["force_resubmit"] is True
    assert payload["n_reuse"] == 0
    assert payload["retry_task_ids"] == [0, 1, 2]


def test_bound_scan_builds_contract_and_intent_index_once(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.daemon import (
        quantum_task_contracts,
        submission_intent,
    )
    from ichor.hpc.active_learning.daemon.quantum_task_contracts import (
        QuantumLogicalTask,
        QuantumTaskContract,
    )
    from ichor.hpc.active_learning.daemon.state import (
        fresh_campaign_state,
        write_state,
    )

    campaign = tmp_path / "campaign"
    phase = "GAUSSIAN"
    staging = campaign / ".DATA" / "STAGING" / "iter_4"
    staging.mkdir(parents=True)
    state = fresh_campaign_state(max_iterations=10)
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    tasks = []
    for task_id in range(4):
        pointdir = staging / ("POINT_" + str(task_id).zfill(4) + ".pointdir")
        pointdir.mkdir()
        (pointdir / "GAUSSIAN_TASK_RECEIPT.json").write_text(
            "{}",
            encoding="utf-8",
        )
        tasks.append(
            QuantumLogicalTask(
                logical_task_id=task_id,
                pointdir_name=pointdir.name,
                pointdir=pointdir,
                producer_logical_task_id=task_id,
                candidate_id="candidate-" + str(task_id),
            )
        )
    (staging / "POINTS.txt").write_text(
        "\n".join(str(task.pointdir.resolve()) for task in tasks) + "\n",
        encoding="utf-8",
    )
    contract = QuantumTaskContract(
        campaign_uid=str(state.campaign_uid),
        phase=phase,
        iteration=4,
        replacement_round=0,
        staging_dir=staging,
        tasks=tuple(tasks),
    )
    counts = {"contract": 0, "intents": 0, "task_maps": 0}

    def build_contract(*_args, **_kwargs):
        counts["contract"] += 1
        return contract

    intent = {
        "attempt_id": "attempt-1",
        "submission_identity": "r0000-a0001-test",
        "job_id": "123",
        "status": "COMPLETED",
        "submission_kind": "array",
        "replacement_round": 0,
        "submission_metadata": {},
    }

    def attempts(*_args, **_kwargs):
        counts["intents"] += 1
        return (intent,)

    def submitted(*_args, **_kwargs):
        counts["task_maps"] += 1
        return tuple(range(4))

    monkeypatch.setattr(
        quantum_task_contracts,
        "quantum_task_contract",
        build_contract,
    )
    monkeypatch.setattr(submission_intent, "intent_attempt_records", attempts)
    monkeypatch.setattr(
        submission_intent,
        "intent_submitted_logical_task_ids",
        submitted,
    )
    monkeypatch.setattr(
        array_recovery,
        "logical_task_ids",
        lambda *_args, **_kwargs: pytest.fail(
            "bound scan must use its prepared task contract"
        ),
    )
    monkeypatch.setattr(
        array_recovery,
        "_validate_task",
        lambda *_args, **_kwargs: (True, "", str(tasks[_args[3]].pointdir)),
    )
    progress = []

    payload = array_recovery.prepare_retry_submission(
        campaign,
        phase,
        4,
        progress_callback=lambda stage, **values: progress.append(
            (stage, values.get("completed"), values.get("total"))
        ),
    )

    assert payload["n_reuse"] == 4
    assert counts == {"contract": 1, "intents": 1, "task_maps": 1}
    assert progress[0] == ("array_recovery_validation", 0, 4)
    assert progress[-1] == ("array_recovery_validation", 4, 4)


def test_bound_scan_refuses_control_mutation_before_ledger(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.daemon import quantum_task_contracts
    from ichor.hpc.active_learning.daemon.quantum_task_contracts import (
        QuantumLogicalTask,
        QuantumTaskContract,
    )
    from ichor.hpc.active_learning.daemon.state import (
        fresh_campaign_state,
        write_state,
    )

    campaign = tmp_path / "campaign"
    staging = campaign / ".DATA" / "STAGING" / "iter_4"
    staging.mkdir(parents=True)
    state = fresh_campaign_state(max_iterations=10)
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    pointdir = staging / "POINT_0000.pointdir"
    pointdir.mkdir()
    points = staging / "POINTS.txt"
    points.write_text(str(pointdir.resolve()) + "\n", encoding="utf-8")
    contract = QuantumTaskContract(
        campaign_uid=str(state.campaign_uid),
        phase="GAUSSIAN",
        iteration=4,
        replacement_round=0,
        staging_dir=staging,
        tasks=(
            QuantumLogicalTask(
                logical_task_id=0,
                pointdir_name=pointdir.name,
                pointdir=pointdir,
                producer_logical_task_id=0,
                candidate_id="candidate-0",
            ),
        ),
    )
    monkeypatch.setattr(
        quantum_task_contracts,
        "quantum_task_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        array_recovery,
        "_validate_task",
        lambda *_args, **_kwargs: (False, "missing", str(pointdir)),
    )
    changed = False

    def progress(_stage, **values):
        nonlocal changed
        if values.get("completed") == 1 and not changed:
            changed = True
            points.write_text(str(pointdir.resolve()) + "\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="authority changed during validation"):
        array_recovery.prepare_retry_submission(
            campaign,
            "GAUSSIAN",
            4,
            progress_callback=progress,
        )

    assert not array_recovery.array_ledger_path(
        campaign,
        "GAUSSIAN",
        4,
    ).exists()


def test_bound_scan_refuses_receipt_mutation_before_ledger(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.daemon import quantum_task_contracts
    from ichor.hpc.active_learning.daemon.quantum_task_contracts import (
        QuantumLogicalTask,
        QuantumTaskContract,
    )
    from ichor.hpc.active_learning.daemon.state import (
        fresh_campaign_state,
        write_state,
    )

    campaign = tmp_path / "campaign"
    staging = campaign / ".DATA" / "STAGING" / "iter_4"
    staging.mkdir(parents=True)
    state = fresh_campaign_state(max_iterations=10)
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    pointdir = staging / "POINT_0000.pointdir"
    pointdir.mkdir()
    points = staging / "POINTS.txt"
    points.write_text(str(pointdir.resolve()) + "\n", encoding="utf-8")
    receipt = pointdir / "GAUSSIAN_TASK_RECEIPT.json"
    receipt.write_text("{}", encoding="utf-8")
    contract = QuantumTaskContract(
        campaign_uid=str(state.campaign_uid),
        phase="GAUSSIAN",
        iteration=4,
        replacement_round=0,
        staging_dir=staging,
        tasks=(
            QuantumLogicalTask(
                logical_task_id=0,
                pointdir_name=pointdir.name,
                pointdir=pointdir,
                producer_logical_task_id=0,
                candidate_id="candidate-0",
            ),
        ),
    )
    monkeypatch.setattr(
        quantum_task_contracts,
        "quantum_task_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        array_recovery,
        "_validate_task",
        lambda *_args, **_kwargs: (True, "", str(pointdir)),
    )

    def progress(_stage, **values):
        if values.get("completed") == 1:
            receipt.write_text('{"changed":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="authority changed during validation"):
        array_recovery.prepare_retry_submission(
            campaign,
            "GAUSSIAN",
            4,
            progress_callback=progress,
        )

    assert not array_recovery.array_ledger_path(
        campaign,
        "GAUSSIAN",
        4,
    ).exists()


def test_missing_middle_quantum_task_is_independently_retryable(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import quantum_task_contracts
    from ichor.hpc.active_learning.daemon.quantum_task_contracts import (
        QuantumLogicalTask,
        QuantumTaskContract,
    )

    campaign = tmp_path / "campaign"
    staging = campaign / ".DATA" / "STAGING" / "iter_4"
    staging.mkdir(parents=True)
    tasks = []
    for task_id in range(3):
        pointdir = staging / ("POINT_" + str(task_id).zfill(4) + ".pointdir")
        if task_id != 1:
            pointdir.mkdir()
        tasks.append(
            QuantumLogicalTask(
                logical_task_id=task_id,
                pointdir_name=pointdir.name,
                pointdir=pointdir,
                producer_logical_task_id=task_id,
                candidate_id="candidate-" + str(task_id),
            )
        )
    contract = QuantumTaskContract(
        campaign_uid="campaign-uid",
        phase="GAUSSIAN",
        iteration=4,
        replacement_round=0,
        staging_dir=staging,
        tasks=tuple(tasks),
    )
    monkeypatch.setattr(
        quantum_task_contracts,
        "quantum_task_contract",
        lambda *_args, **_kwargs: contract,
    )

    assert array_recovery.logical_task_ids(campaign, "GAUSSIAN", 4) == [0, 1, 2]
    missing = array_recovery._pointdir_for_task(
        campaign,
        "GAUSSIAN",
        4,
        1,
    )
    assert missing == tasks[1].pointdir
    assert array_recovery._validate_quantum_task_path(
        campaign,
        "GAUSSIAN",
        4,
        1,
        missing,
    )[:2] == (False, "pointdir_missing")


def test_missing_aimall_task_rewinds_only_its_gaussian_producer(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import (
        input_staging,
        quantum_task_contracts,
    )
    from ichor.hpc.active_learning.daemon.quantum_task_contracts import (
        QuantumLogicalTask,
        QuantumTaskContract,
    )

    campaign = tmp_path / "campaign"
    staging = campaign / ".DATA" / "STAGING" / "iter_4"
    staging.mkdir(parents=True)
    names = [
        "POINT_0000.pointdir",
        "POINT_0001.pointdir",
        "POINT_0002.pointdir",
    ]
    (staging / names[1]).mkdir()
    input_staging.write_points_file(
        staging,
        [staging / names[0], staging / names[1]],
    )
    input_staging.write_quantum_acceptance_manifest(
        staging,
        phase_name="GAUSSIAN",
        iteration=4,
        accepted=[staging / names[0], staging / names[1]],
        rejected=[(names[2], "submitted_pointdir_missing")],
    )

    gaussian_tasks = tuple(
        QuantumLogicalTask(
            logical_task_id=index,
            pointdir_name=name,
            pointdir=staging / name,
            producer_logical_task_id=index,
            candidate_id="candidate-" + str(index),
        )
        for index, name in enumerate(names)
    )
    aimall_tasks = tuple(
        QuantumLogicalTask(
            logical_task_id=index,
            pointdir_name=name,
            pointdir=staging / name,
            producer_logical_task_id=index,
            candidate_id="candidate-" + str(index),
        )
        for index, name in enumerate(names[:2])
    )

    def contract(_campaign, phase_name, _iteration, **_kwargs):
        tasks = gaussian_tasks if phase_name == "GAUSSIAN" else aimall_tasks
        return QuantumTaskContract(
            campaign_uid="campaign-uid",
            phase=phase_name,
            iteration=4,
            replacement_round=0,
            staging_dir=staging,
            tasks=tasks,
        )

    monkeypatch.setattr(
        quantum_task_contracts,
        "quantum_task_contract",
        contract,
    )
    monkeypatch.setattr(
        array_recovery,
        "_validate_quantum_task_path",
        lambda *_args, **_kwargs: (True, "", str(staging / names[1])),
    )

    recovery = array_recovery.scan_aimall_upstream_gaussian_recovery(
        campaign,
        "AIMALL",
        4,
    )

    assert recovery is not None
    assert recovery["phase"] == "GAUSSIAN"
    assert recovery["retry_task_ids"] == [0]
    assert recovery["n_complete"] == 2
    assert recovery["tasks"][2]["reason"] == "prior_gaussian_rejection_preserved"


def test_sbatch_retry_map_uses_logical_task_id_for_ariadne(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    config = CampaignConfig()
    retry_map = campaign / ".DATA" / "ACTIVE_LEARNING" / "RETRY_TASKS.ARIADNE_ARRAY.0002.txt"
    retry_map.parent.mkdir(parents=True)
    retry_map.write_text("4\n9\n", encoding="utf-8", newline="\n")

    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.live_executor._configured_scheduler",
        lambda: "slurm",
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.live_executor._configured_max_array_task_id",
        lambda: None,
    )

    script = build_sbatch_script(
        phase_name="ARIADNE_ARRAY",
        iteration=2,
        campaign_dir=campaign,
        config=config,
        array_size=2,
        array_task_map=retry_map,
    )

    assert "ICHOR_LOGICAL_ARRAY_TASK_ID" in script
    assert "retry task map SHA-256 mismatch" in script
    assert "--array-task-id $ICHOR_LOGICAL_ARRAY_TASK_ID" in script
    assert "#SBATCH --array=0-1" in script


def _patch_ariadne_recovery_identity(monkeypatch):
    task_map = {
        "campaign_uid": "campaign-uid",
        "tasks": [
            {
                "array_task_id": 0,
                "seed_id": 1,
                "seed_uid": "seed-uid",
            }
        ],
    }
    monkeypatch.setattr(
        "ichor.hpc.active_learning.seed_identity.read_ariadne_task_map",
        lambda _iter_dir, expected_iteration=None: task_map,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.seed_identity.task_for_array_task_id",
        lambda payload, task_id: payload["tasks"][int(task_id)],
    )


def test_ariadne_recovery_reuses_only_successful_zero_exit_output(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    _patch_ariadne_recovery_identity(monkeypatch)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.ariadne_outputs.validate_seed_output",
        lambda *_args, **_kwargs: {
            "task_success": True,
            "task_exit_code": 0,
        },
    )

    reusable, reason, _path = array_recovery._validate_ariadne_task(
        campaign,
        1,
        0,
    )

    assert reusable is True
    assert reason == ""


def test_ariadne_recovery_rejects_hash_valid_failed_output(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    _patch_ariadne_recovery_identity(monkeypatch)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.ariadne_outputs.validate_seed_output",
        lambda *_args, **_kwargs: {
            "task_success": False,
            "task_exit_code": 7,
        },
    )

    reusable, reason, _path = array_recovery._validate_ariadne_task(
        campaign,
        1,
        0,
    )

    assert reusable is False
    assert reason == "task_success_false"


def test_ariadne_recovery_rejects_nonzero_task_exit_code(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    _patch_ariadne_recovery_identity(monkeypatch)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.ariadne_outputs.validate_seed_output",
        lambda *_args, **_kwargs: {
            "task_success": True,
            "task_exit_code": 9,
        },
    )

    reusable, reason, _path = array_recovery._validate_ariadne_task(
        campaign,
        1,
        0,
    )

    assert reusable is False
    assert reason == "task_exit_code_9"


def test_ariadne_retry_archives_complete_seed_directory_and_partial_output(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.layout import (
        active_iteration_dir,
        ariadne_seed_dir,
    )

    campaign = tmp_path / "campaign"
    _patch_ariadne_recovery_identity(monkeypatch)
    iter_dir = active_iteration_dir(campaign, 1)
    seed_dir = ariadne_seed_dir(iter_dir, 1)
    seed_dir.mkdir(parents=True)
    (seed_dir / "result.json").write_text("{}\n", encoding="utf-8")
    partial = seed_dir.parent / ".seed-000001.partial-task-0-pid-12"
    partial.mkdir()
    (partial / "partial.txt").write_text("partial\n", encoding="utf-8")

    archived = array_recovery.archive_existing_array_task_outputs(
        campaign,
        "ARIADNE_ARRAY",
        1,
        task_ids=[0],
    )

    assert not seed_dir.exists()
    assert not partial.exists()
    assert len(archived) == 2
    assert all(Path(path).exists() for path in archived)

    replayed = array_recovery.archive_existing_array_task_outputs(
        campaign,
        "ARIADNE_ARRAY",
        1,
        task_ids=[0],
        archive_identity=Path(archived[0]).parents[1].name.split(
            "000001-", 1
        )[1],
    )
    assert replayed == archived


def test_ariadne_retry_archive_replay_rejects_recreated_source(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.layout import (
        active_iteration_dir,
        ariadne_seed_dir,
    )

    campaign = tmp_path / "campaign"
    _patch_ariadne_recovery_identity(monkeypatch)
    seed_dir = ariadne_seed_dir(active_iteration_dir(campaign, 1), 1)
    seed_dir.mkdir(parents=True)
    (seed_dir / "result.json").write_text("{}\n", encoding="utf-8")
    identity = "replay-conflict"

    array_recovery.archive_existing_array_task_outputs(
        campaign,
        "ARIADNE_ARRAY",
        1,
        task_ids=[0],
        archive_identity=identity,
    )
    seed_dir.mkdir(parents=True)
    (seed_dir / "result.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="archive destination both exist",
    ):
        array_recovery.archive_existing_array_task_outputs(
            campaign,
            "ARIADNE_ARRAY",
            1,
            task_ids=[0],
            archive_identity=identity,
        )


def test_aimall_retry_archives_only_aimall_derived_outputs(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    pointdir = (
        campaign
        / ".DATA"
        / "STAGING"
        / "initial"
        / "POINT_0000.pointdir"
    )
    pointdir.mkdir(parents=True)
    monkeypatch.setattr(
        array_recovery,
        "_bound_array_recovery_scan_context",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        array_recovery,
        "_pointdir_for_task",
        lambda *_args, **_kwargs: pointdir,
    )

    preserved = (
        "input.gjf",
        "input.wfn",
        "input.gaussianoutput",
        "provenance.json",
        "GAUSSIAN_TASK_RECEIPT.json",
        "WFN_METHOD_RECEIPT.json",
        "AIMALL_TASK.json",
    )
    for name in preserved:
        (pointdir / name).write_text("preserved\n", encoding="utf-8")

    derived = (
        "input.aim",
        "input.agp",
        "input.agpviz",
        "input.extout",
        "input.int",
        "input.mgp",
        "input.mgpviz",
        "input.sum",
        "input.sumviz",
        "AIMALL_COMPLETION_RECEIPT.json",
        "QUANTUM_ACCEPTANCE_RECEIPT.json",
    )
    for name in derived:
        (pointdir / name).write_text("stale\n", encoding="utf-8")
    atomic = pointdir / "input_atomicfiles"
    atomic.mkdir()
    (atomic / "h1.int").write_text("truncated\n", encoding="utf-8")

    archived = array_recovery.archive_existing_array_task_outputs(
        campaign,
        "INITIAL_AIMALL",
        0,
        task_ids=[0],
        archive_identity="aimall-structural-retry",
    )

    assert all((pointdir / name).is_file() for name in preserved)
    assert all(not (pointdir / name).exists() for name in derived)
    assert not atomic.exists()
    assert {Path(path).name for path in archived} == {
        *derived,
        "input_atomicfiles",
    }
    assert all(Path(path).exists() for path in archived)

    replayed = array_recovery.archive_existing_array_task_outputs(
        campaign,
        "INITIAL_AIMALL",
        0,
        task_ids=[0],
        archive_identity="aimall-structural-retry",
    )
    assert replayed == archived


def test_replacement_phases_support_round_specific_partial_recovery(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    round_dir = (
        campaign
        / ".DATA"
        / "STAGING"
        / "iter_2"
        / "replacement_round_0003"
    )
    round_dir.mkdir(parents=True)
    pointdirs = []
    for index in range(3):
        pointdir = round_dir / ("POINT_" + str(index + 4).zfill(4) + ".pointdir")
        pointdir.mkdir()
        pointdirs.append(pointdir)
    (round_dir / "POINTS.txt").write_text(
        "\n".join(str(path.resolve()) for path in pointdirs) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(
        array_recovery,
        "_replacement_identity",
        lambda *args, **kwargs: (3, round_dir),
    )
    monkeypatch.setattr(
        array_recovery,
        "_validate_task",
        lambda _campaign, _phase, _iteration, task_id: (
            int(task_id) == 1,
            "" if int(task_id) == 1 else "missing",
            "output",
        ),
    )
    monkeypatch.setattr(
        array_recovery,
        "logical_task_ids",
        lambda *_args, **_kwargs: [0, 1, 2],
    )

    payload = array_recovery.prepare_retry_submission(
        campaign,
        "REPLACEMENT_GAUSSIAN",
        2,
    )

    assert payload["replacement_round"] == 3
    assert payload["retry_task_ids"] == [0, 2]
    assert "-r0003.json" in payload["path"]
    assert ".r0003.txt" in payload["retry_task_file"]


def test_all_replacement_quantum_phases_advertise_partial_recovery():
    for phase in (
        "INITIAL_REPLACEMENT_GAUSSIAN",
        "INITIAL_REPLACEMENT_AIMALL",
        "REPLACEMENT_GAUSSIAN",
        "REPLACEMENT_AIMALL",
    ):
        assert array_recovery.supports_partial_array_recovery(phase)
