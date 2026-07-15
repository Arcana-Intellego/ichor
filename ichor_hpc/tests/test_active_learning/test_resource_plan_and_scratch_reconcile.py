from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning import cli
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.resource_plan import (
    build_resource_plan,
    format_resource_plan,
    plan_phase,
)
from ichor.hpc.active_learning.daemon.daemon import Daemon
from ichor.hpc.active_learning.daemon.journal import read_events
from ichor.hpc.active_learning.daemon.resource_records import (
    resolution_payload,
    write_resolution,
)
from ichor.hpc.active_learning.daemon.resource_solver import ResolvedPhaseResources
from ichor.hpc.active_learning.daemon.scratch import inventory, prepare_task_scratch
from ichor.hpc.active_learning.daemon.script_bundles import (
    prepare_attempt_bundle,
    write_attempt_script,
    write_script_binding,
)
from ichor.hpc.active_learning.daemon.submission_intent import (
    bind_resource_resolution,
    load_intent,
    mark_completed,
    mark_submitted,
    write_pre_submit_intent,
)
from ichor.hpc.active_learning.daemon.state import CampaignPhase, fresh_campaign_state


def _failed_scratch(campaign: Path, identity: str = "r0000-a0001-deadbeef") -> Path:
    from ichor.hpc.active_learning.daemon.scratch import finish_task_scratch

    payload = resolution_payload(
        campaign_uid="uid",
        phase_name="GAUSSIAN",
        iteration=1,
        attempt_id="attempt-id",
        submission_identity=identity,
        resolved=ResolvedPhaseResources(
            backend="gaussian",
            partition="multicore",
            ntasks=1,
            cpus_per_task=1,
            mem_per_cpu="4G",
            estimated_total_memory_gb=4.0,
            partition_memory_per_core_gb=4.0,
            cpus_raw=1,
            mem_per_cpu_raw="4G",
            cpu_reason="fixture",
            memory_reason="fixture",
        ),
        evidence={"source": "fixture"},
        scratch_path_template="fixture",
    )
    binding = write_resolution(campaign, payload)
    bundle = prepare_attempt_bundle(
        campaign,
        "GAUSSIAN",
        1,
        identity,
        array_size=1,
        max_log_files_per_directory=10,
    )
    write_attempt_script(bundle, "#!/bin/bash\ntrue\n")
    script_binding = write_script_binding(bundle)
    leaf = prepare_task_scratch(
        campaign,
        campaign_uid="uid",
        phase_name="GAUSSIAN",
        iteration=1,
        attempt_id="attempt-id",
        submission_identity=identity,
        job_id="100",
        array_task_id=0,
        resource_resolution_path=binding["path"],
        resource_resolution_sha256=binding["sha256"],
        script_binding_path=script_binding["path"],
        script_binding_sha256=script_binding["sha256"],
    )
    finish_task_scratch(leaf, success=False)
    return leaf


def test_scratch_inventory_reports_every_unowned_entry(tmp_path):
    root = tmp_path / ".DATA" / "SCRATCH"
    root.mkdir(parents=True)
    (root / "unexpected.txt").write_text("unknown\n", encoding="utf-8")
    (root / "BROKEN").mkdir()

    records = inventory(tmp_path)

    assert len(records) == 2
    assert {record["status"] for record in records} == {"invalid"}
    assert any("outside a task directory" in record["reason"] for record in records)
    assert any("component is malformed" in record["reason"] for record in records)


def test_scratch_reconcile_preview_then_apply(monkeypatch, tmp_path, capsys):
    leaf = _failed_scratch(tmp_path)
    monkeypatch.setattr(
        cli,
        "_scratch_scheduler_state",
        lambda _job_id, expected_task_count=None: (
            "inactive",
            "terminal fixture",
        ),
    )

    assert cli._cmd_reconcile_scratch(
        tmp_path,
        apply=False,
        json_output=False,
        selected_attempts=[],
    ) == 0
    assert leaf.is_dir()
    assert "Conclusively inactive" in capsys.readouterr().out

    assert cli._cmd_reconcile_scratch(
        tmp_path,
        apply=True,
        json_output=False,
        selected_attempts=[],
    ) == 0
    assert not leaf.exists()


def test_scratch_reconcile_refuses_selected_active_attempt(
    monkeypatch,
    tmp_path,
):
    leaf = _failed_scratch(tmp_path)
    monkeypatch.setattr(
        cli,
        "_scratch_scheduler_state",
        lambda _job_id, expected_task_count=None: (
            "active",
            "running fixture",
        ),
    )
    assert cli._cmd_reconcile_scratch(
        tmp_path,
        apply=True,
        json_output=False,
        selected_attempts=["attempt-id"],
    ) == 9
    assert leaf.is_dir()


def test_scratch_scheduler_state_keeps_missing_array_rows_inconclusive(
    monkeypatch,
):
    from ichor.hpc.active_learning.submit import sacct_poll

    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda _job_id: sacct_poll.JobQueueLookup(
            active=False,
            inconclusive=False,
        ),
    )
    monkeypatch.setattr(
        sacct_poll,
        "poll_job",
        lambda _job_id: [
            sacct_poll.JobObservation(
                job_id="100_0",
                status=sacct_poll.JobStatus.COMPLETED,
                exit_code=(0, 0),
                elapsed_seconds=1,
            )
        ],
    )

    status, reason = cli._scratch_scheduler_state(
        "100",
        expected_task_count=2,
    )

    assert status == "inconclusive"
    assert "missing 1 of 2 expected task rows" in reason


def test_resource_plan_reports_local_and_missing_future_evidence(tmp_path):
    local = plan_phase(tmp_path, CampaignConfig(), "SEED_SELECT", 1)
    assert local["status"] == "local"
    unavailable = plan_phase(tmp_path, CampaignConfig(), "PHASE_A_DIVERSITY", 0)
    assert unavailable["status"] == "evidence_not_yet_produced"


def _campaign_byte_inventory(root: Path):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def test_orphaned_resolution_is_not_reported_as_submitted(tmp_path):
    resolved = ResolvedPhaseResources(
        backend="diversity",
        partition="multicore",
        ntasks=1,
        cpus_per_task=1,
        mem_per_cpu="4G",
        estimated_total_memory_gb=4.0,
        partition_memory_per_core_gb=4.0,
        cpus_raw=1,
        mem_per_cpu_raw="4G",
        cpu_reason="fixture",
        memory_reason="fixture",
    )
    payload = resolution_payload(
        campaign_uid="uid",
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        attempt_id="orphan-attempt",
        submission_identity="r0000-a0001-orphaned",
        resolved=resolved,
        evidence={"source": "fixture"},
        scratch_path_template="fixture",
    )
    binding = write_resolution(tmp_path, payload)

    planned = plan_phase(tmp_path, CampaignConfig(), "PHASE_A_DIVERSITY", 0)

    assert planned["status"] == "evidence_not_yet_produced"
    assert planned["orphaned_resolutions"] == [str(Path(binding["path"]).resolve())]


def test_resource_plan_is_byte_for_byte_read_only(tmp_path):
    data = tmp_path / ".DATA" / "ACTIVE_LEARNING"
    data.mkdir(parents=True)
    (tmp_path / "campaign.yaml").write_text("schema_version: 13\n", encoding="utf-8")
    (data / "operator-note.txt").write_bytes(b"unchanged evidence\n")
    before = _campaign_byte_inventory(tmp_path)

    payload = build_resource_plan(
        tmp_path,
        CampaignConfig(),
        current_phase="INIT",
        current_iteration=0,
        all_phases=True,
    )

    assert payload["schema_version"] == 2
    assert _campaign_byte_inventory(tmp_path) == before


def test_resource_plan_prefers_immutable_submitted_resolution(tmp_path):
    intent = write_pre_submit_intent(
        tmp_path,
        campaign_uid="uid",
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        expected_tasks=1,
    )
    resolved = ResolvedPhaseResources(
        backend="diversity",
        partition="multicore",
        ntasks=1,
        cpus_per_task=10,
        mem_per_cpu="1G",
        estimated_total_memory_gb=5.0,
        partition_memory_per_core_gb=8.0,
        cpus_raw="auto",
        mem_per_cpu_raw="auto",
        cpu_reason="diversity_pairs_per_worker",
        memory_reason="diversity_condensed_distance_store",
        extra={
            "active_workers": 10,
            "memory_only_cpus": 0,
            "peak_allocation_gb": 10.0,
        },
    )
    payload = resolution_payload(
        campaign_uid="uid",
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        attempt_id=str(intent["attempt_id"]),
        submission_identity=str(intent["submission_identity"]),
        resolved=resolved,
        evidence={"source": "fixture", "n_frames": 10_000},
        scratch_path_template="fixture",
    )
    binding = write_resolution(tmp_path, payload)
    bind_resource_resolution(
        tmp_path,
        "PHASE_A_DIVERSITY",
        0,
        path=str(binding["path"]),
        sha256=str(binding["sha256"]),
        formula_version=str(binding["formula_version"]),
        scratch_path_template="fixture",
    )
    mark_submitted(tmp_path, "PHASE_A_DIVERSITY", 0, "123", expected_tasks=1)

    submitted = plan_phase(tmp_path, CampaignConfig(), "PHASE_A_DIVERSITY", 0)
    assert submitted["status"] == "submitted"
    assert submitted["resources"]["cpus_per_task"] == 10
    mark_completed(tmp_path, "PHASE_A_DIVERSITY", 0)
    completed = plan_phase(tmp_path, CampaignConfig(), "PHASE_A_DIVERSITY", 0)
    assert completed["status"] == "completed"


def test_resource_plan_rejects_drifted_submitted_resolution(tmp_path):
    intent = write_pre_submit_intent(
        tmp_path,
        campaign_uid="uid",
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        expected_tasks=1,
    )
    resolved = ResolvedPhaseResources(
        backend="diversity",
        partition="multicore",
        ntasks=1,
        cpus_per_task=1,
        mem_per_cpu="4G",
        estimated_total_memory_gb=4.0,
        partition_memory_per_core_gb=4.0,
        cpus_raw=1,
        mem_per_cpu_raw="4G",
        cpu_reason="fixture",
        memory_reason="fixture",
    )
    payload = resolution_payload(
        campaign_uid="uid",
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        attempt_id=str(intent["attempt_id"]),
        submission_identity=str(intent["submission_identity"]),
        resolved=resolved,
        evidence={"source": "fixture"},
        scratch_path_template="fixture",
    )
    binding = write_resolution(tmp_path, payload)
    bind_resource_resolution(
        tmp_path,
        "PHASE_A_DIVERSITY",
        0,
        path=str(binding["path"]),
        sha256="0" * 64,
        formula_version=str(binding["formula_version"]),
        scratch_path_template="fixture",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        plan_phase(tmp_path, CampaignConfig(), "PHASE_A_DIVERSITY", 0)


def test_all_resource_plan_keeps_expected_future_absence_informational(tmp_path):
    payload = build_resource_plan(
        tmp_path,
        CampaignConfig(),
        current_phase="INIT",
        current_iteration=0,
        all_phases=True,
    )
    statuses = {plan["phase"]: plan["status"] for plan in payload["plans"]}
    assert statuses["INIT"] == "local"
    assert statuses["PHASE_A_DIVERSITY"] == "evidence_not_yet_produced"
    assert "Resource plan" in format_resource_plan(payload)


def test_resource_plan_human_output_includes_formula_and_evidence_contract(tmp_path):
    payload = {
        "campaign_dir": str(tmp_path),
        "current_phase": "PHASE_A_DIVERSITY",
        "current_iteration": 0,
        "plans": [
            {
                "phase": "PHASE_A_DIVERSITY",
                "iteration": 0,
                "status": "submitted",
                "resources": {
                    "partition": "multicore",
                    "cpus_per_task": 10,
                    "mem_per_cpu": "1G",
                    "estimated_total_memory_gb": 6.0,
                    "cpu_reason": "diversity_pairs_per_worker",
                    "memory_reason": "diversity_condensed_distance_store",
                    "warnings": ["fixture warning"],
                    "extra": {
                        "active_workers": 10,
                        "memory_only_cpus": 0,
                        "array_size": None,
                        "array_concurrency": 1,
                        "per_task_allocation_gb": 10.0,
                        "peak_allocation_gb": 10.0,
                        "memory_estimate_safety_factor": 1.25,
                        "scratch_mode": "file_backed_condensed_distances",
                        "expected_scratch_bytes": 1000,
                        "scratch_requirement_exact": True,
                        "profile_limits": {
                            "partition_min_cpus": 1,
                            "partition_max_cpus": 64,
                            "partition_memory_per_core_gb": 8.0,
                        },
                        "campaign_filesystem": {
                            "path": str(tmp_path),
                            "free_bytes_at_resolution": 100000,
                        },
                    },
                },
                "evidence": {
                    "source": "trajectory_pool_manifest",
                    "pool": {"path": "pool.xyz", "sha256": "a" * 64},
                },
                "scratch_path_template": "scratch-template",
                "telemetry": {
                    "status": "final",
                    "path": "resource_usage_records.json",
                    "error": None,
                    "summary": {
                        "p95_rss_mib": 100.0,
                        "p95_elapsed_seconds": 20.0,
                        "recommended_memory_mib": 125.0,
                        "recommended_walltime_seconds": 30.0,
                        "n_missing_task_rows": 0,
                    },
                },
            }
        ],
    }
    output = format_resource_plan(payload)
    assert "cpu_formula=diversity_pairs_per_worker" in output
    assert "evidence_sha256 pool.xyz " + "a" * 64 in output
    assert "scratch_template=scratch-template" in output
    assert "scratch_mode=file_backed_condensed_distances" in output
    assert "profile_limits min_cpus=1 max_cpus=64" in output
    assert "campaign_filesystem=" in output
    assert "advisory memory_mib=125.0" in output


def test_resource_plan_parser_phase_and_all_are_mutually_exclusive():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "resource-plan",
                "--phase",
                "PHASE_A_DIVERSITY",
                "--all",
            ]
        )


def test_daemon_terminal_telemetry_is_advisory_and_journalled(tmp_path):
    state = fresh_campaign_state(campaign_uid="uid")
    state.phase = CampaignPhase.PHASE_A_DIVERSITY
    intent = write_pre_submit_intent(
        tmp_path,
        campaign_uid="uid",
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        expected_tasks=1,
    )
    mark_submitted(tmp_path, "PHASE_A_DIVERSITY", 0, "321", expected_tasks=1)
    calls = []

    def collector(campaign_dir, *, intent, history_limit):
        calls.append((Path(campaign_dir), dict(intent), int(history_limit)))
        return {"n_rows": 1, "p95_rss_mib": 100.0, "p95_elapsed_seconds": 4.0}

    daemon = Daemon(
        campaign_dir=tmp_path,
        config=CampaignConfig(),
        resource_usage_collector=collector,
    )
    daemon._collect_terminal_resource_usage(
        state,
        CampaignPhase.PHASE_A_DIVERSITY,
        "321",
    )
    assert calls[0][1]["attempt_id"] == intent["attempt_id"]
    events = list(
        read_events(tmp_path / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson")
    )
    assert events[-1]["event"] == "scheduler_usage_recorded"


def test_daemon_telemetry_failure_does_not_raise(tmp_path):
    state = fresh_campaign_state(campaign_uid="uid")
    state.phase = CampaignPhase.PHASE_A_DIVERSITY
    write_pre_submit_intent(
        tmp_path,
        campaign_uid="uid",
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        expected_tasks=1,
    )
    mark_submitted(tmp_path, "PHASE_A_DIVERSITY", 0, "654", expected_tasks=1)

    def collector(*_args, **_kwargs):
        raise RuntimeError("accounting lag")

    daemon = Daemon(
        campaign_dir=tmp_path,
        config=CampaignConfig(),
        resource_usage_collector=collector,
    )
    daemon._collect_terminal_resource_usage(
        state,
        CampaignPhase.PHASE_A_DIVERSITY,
        "654",
    )
    events = list(
        read_events(tmp_path / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson")
    )
    assert events[-1]["event"] == "scheduler_usage_warning"
def test_resource_binding_snapshots_dense_scheduler_task_count(tmp_path):
    intent = write_pre_submit_intent(
        tmp_path,
        campaign_uid="campaign-uid",
        phase_name="GAUSSIAN",
        iteration=2,
        expected_tasks=200,
    )
    bind_resource_resolution(
        tmp_path,
        "GAUSSIAN",
        2,
        path=str(tmp_path / "resolution.json"),
        sha256="a" * 64,
        formula_version="resource-formula-v1",
        scratch_path_template="scratch-template",
        expected_tasks=7,
    )

    rebound = load_intent(tmp_path, "GAUSSIAN", 2)
    assert rebound is not None
    assert rebound["attempt_id"] == intent["attempt_id"]
    assert rebound["expected_tasks"] == 7
