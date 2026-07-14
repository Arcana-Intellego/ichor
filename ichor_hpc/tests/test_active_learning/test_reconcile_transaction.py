"""Durable reconciliation transaction and pointer rollback contracts."""

from __future__ import annotations

import json
import pytest

from ichor.hpc.active_learning import cli
from ichor.hpc.active_learning.daemon.reconcile_transaction import (
    begin_reconcile_transaction,
    inventory_reconcile_transactions,
    read_reconcile_transaction,
    restore_version_pointer,
    snapshot_version_pointer,
)
from ichor.hpc.active_learning.daemon.reconcile import stateful_campaign_artifacts
from ichor.hpc.active_learning.daemon.submission_intent import (
    load_intent,
    mark_submitted,
    write_pre_submit_intent,
)
from ichor.hpc.active_learning.submit import sacct_poll
from ichor.hpc.active_learning.versioning.versioned_directory import VersionedDirectory


def test_reconcile_transaction_records_lossless_moves_and_status(tmp_path):
    campaign = tmp_path / "campaign"
    evidence = campaign / ".DATA" / "STAGING.before-reconcile" / "failure.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("retained\n", encoding="utf-8")

    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase="FEREBUS",
        proposed_iteration=1,
        planned_operations=["archive_reconcile_evidence"],
        intent_transitions=[],
    )
    transaction.set_status("MUTATING")
    transaction.record_paths("archive_staging", [str(evidence.parent)])
    transaction.set_status("COMMITTED")

    payload = json.loads(transaction.path.read_text(encoding="utf-8"))
    assert payload["status"] == "COMMITTED"
    assert payload["completed_operations"][0]["paths"] == [
        ".DATA/STAGING.before-reconcile"
    ]
    assert read_reconcile_transaction(transaction.path)["status"] == "COMMITTED"


def test_incomplete_reconcile_transaction_blocks_a_second_transaction(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    first = begin_reconcile_transaction(
        campaign,
        proposed_phase="INIT",
        proposed_iteration=0,
        planned_operations=["write state"],
        intent_transitions=[],
    )

    with pytest.raises(RuntimeError, match="prior transaction evidence"):
        begin_reconcile_transaction(
            campaign,
            proposed_phase="INIT",
            proposed_iteration=0,
            planned_operations=["write state"],
            intent_transitions=[],
        )

    assert inventory_reconcile_transactions(campaign)[0]["status"] == "PREPARED"
    assert any(
        "reconcile_transactions" in path
        for path in stateful_campaign_artifacts(campaign)
    )
    first.set_status("FAILED", reason="test repair abandoned")
    second = begin_reconcile_transaction(
        campaign,
        proposed_phase="INIT",
        proposed_iteration=0,
        planned_operations=["write state"],
        intent_transitions=[],
    )
    assert second.path != first.path


def test_reconcile_pointer_snapshot_restores_previous_binding(tmp_path):
    campaign = tmp_path / "campaign"
    versions = VersionedDirectory(campaign / "QM_REFERENCE_DATA")
    versions.iteration_path(0).mkdir(parents=True)
    versions.iteration_path(1).mkdir()
    versions.update_current(0)
    snapshot = snapshot_version_pointer(
        campaign,
        versions,
        label="reference_data",
        requested_version=1,
    )

    versions.update_current(1)
    restore_version_pointer(campaign, snapshot)

    assert versions.current_version() == 0


def test_terminal_intent_classification_is_proposal_only(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    write_pre_submit_intent(
        campaign,
        campaign_uid="reconcile-intent-test",
        phase_name="FEREBUS",
        iteration=0,
        expected_tasks=1,
    )
    mark_submitted(campaign, "FEREBUS", 0, "123", expected_tasks=1)
    active = load_intent(campaign, "FEREBUS", 0)
    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda _job_id: sacct_poll.JobQueueLookup(active=False, rows=[]),
    )
    monkeypatch.setattr(
        sacct_poll,
        "poll_job",
        lambda _job_id: [
            sacct_poll.JobObservation(
                job_id="123",
                status=sacct_poll.JobStatus.CANCELLED,
                exit_code=(1, 0),
                elapsed_seconds=1,
            )
        ],
    )

    resolved, blocking = cli._resolve_terminal_submission_intents_for_apply(
        campaign,
        [active],
    )

    assert blocking == []
    assert resolved[0]["target_status"] == "FAILED"
    assert load_intent(campaign, "FEREBUS", 0)["status"] == "SUBMITTED"
