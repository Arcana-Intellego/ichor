from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.daemon.ariadne_publication import (
    ARIADNE_PUBLICATION_ARCHIVE_FILENAME,
    archive_ariadne_publication,
    classify_ariadne_publication,
)
from ichor.hpc.active_learning.daemon.live_executor import (
    archive_stale_ariadne_publication,
)
from ichor.hpc.active_learning.daemon.phase_executor import BackendSubmissionError
from ichor.hpc.active_learning.daemon.state import CampaignPhase, atomic_write_json
from ichor.hpc.active_learning.cli import _perform_reconcile_apply_mutations
from ichor.hpc.active_learning.handoff_manifests import (
    ARIADNE_RESULTS_SCHEMA_VERSION,
    ariadne_batch_decision_path,
    ariadne_results_path,
    write_ariadne_batch_decision,
)
from ichor.hpc.active_learning.layout import active_ariadne_dir, active_iteration_dir


CAMPAIGN_UID = "12345678-1234-5678-1234-567812345678"


def _write_publication(
    campaign: Path,
    *,
    iteration: int = 1,
    accepted: bool = True,
) -> Path:
    iteration_dir = active_iteration_dir(campaign, iteration)
    root = active_ariadne_dir(iteration_dir)
    root.mkdir(parents=True)
    atomic_write_json(
        ariadne_results_path(iteration_dir),
        {
            "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
            "campaign_uid": CAMPAIGN_UID,
            "iteration": iteration,
            "expected_n": 1,
            "n_accepted": 1 if accepted else 0,
            "n_rejected": 0 if accepted else 1,
            "accepted": [{"seed_id": 1}] if accepted else [],
            "rejected": [] if accepted else [{"seed_id": 1, "reason": "test"}],
        },
    )
    atomic_write_json(root / "AUDIT.json", {"iteration": iteration})
    write_ariadne_batch_decision(
        iteration_dir,
        campaign_uid=CAMPAIGN_UID,
        iteration=iteration,
        config_sha256="config",
        failure_threshold_fraction=0.0,
        expected_n=1,
        n_accepted=1 if accepted else 0,
        n_rejected=0 if accepted else 1,
        accepted=accepted,
        reasons=[] if accepted else ["test rejection"],
    )
    return iteration_dir


def _make_results_stale(iteration_dir: Path) -> None:
    path = ariadne_results_path(iteration_dir)
    payload = path.read_text(encoding="utf-8")
    path.write_text(payload + "\n", encoding="utf-8", newline="\n")


def test_classifies_and_archives_decision_bound_to_earlier_results(tmp_path):
    iteration_dir = _write_publication(tmp_path)
    _make_results_stale(iteration_dir)

    classification = classify_ariadne_publication(
        tmp_path,
        1,
        expected_campaign_uid=CAMPAIGN_UID,
    )

    assert classification["state"] == "stale_results_binding"
    assert classification["archive_required"] is True
    archived = archive_ariadne_publication(
        tmp_path,
        1,
        reason="test_recovery",
        campaign_uid=CAMPAIGN_UID,
        submission_identity="r0000-a0001-test",
        classification=classification,
    )
    assert archived["changed"] is True
    assert classify_ariadne_publication(tmp_path, 1)["state"] == "absent"
    archive_dir = Path(archived["archive_dir"])
    assert sorted(path.name for path in archive_dir.iterdir()) == [
        "ARCHIVE.json",
        "ARIADNE_BATCH_DECISION.json",
        "AUDIT.json",
        "RESULTS.json",
    ]
    receipt = archive_dir / ARIADNE_PUBLICATION_ARCHIVE_FILENAME
    assert '"status": "complete"' in receipt.read_text(encoding="utf-8")


def test_archive_resumes_after_interrupted_move(tmp_path, monkeypatch):
    iteration_dir = _write_publication(tmp_path)
    _make_results_stale(iteration_dir)
    real_replace = os.replace
    calls = {"count": 0}

    def interrupted_replace(source, target):
        source_path = Path(source)
        if source_path.name in {
            "RESULTS.json",
            "ARIADNE_BATCH_DECISION.json",
            "AUDIT.json",
        }:
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("injected move failure")
        return real_replace(source, target)

    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.ariadne_publication.os.replace",
        interrupted_replace,
    )
    with pytest.raises(OSError, match="injected move failure"):
        archive_ariadne_publication(
            tmp_path,
            1,
            reason="test_interruption",
            campaign_uid=CAMPAIGN_UID,
        )

    classification = classify_ariadne_publication(tmp_path, 1)
    assert classification["state"] == "archive_incomplete"
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.ariadne_publication.os.replace",
        real_replace,
    )
    recovered = archive_ariadne_publication(
        tmp_path,
        1,
        reason="test_interruption",
        campaign_uid=CAMPAIGN_UID,
    )
    assert recovered["changed"] is True
    assert classify_ariadne_publication(tmp_path, 1)["state"] == "absent"


def test_complete_publication_is_only_archived_when_retry_forces_replacement(tmp_path):
    _write_publication(tmp_path)
    classification = classify_ariadne_publication(
        tmp_path,
        1,
        expected_campaign_uid=CAMPAIGN_UID,
    )
    assert classification["state"] == "complete"

    unchanged = archive_ariadne_publication(
        tmp_path,
        1,
        reason="no_retry",
        campaign_uid=CAMPAIGN_UID,
        classification=classification,
    )
    assert unchanged["changed"] is False
    forced = archive_ariadne_publication(
        tmp_path,
        1,
        reason="retry",
        campaign_uid=CAMPAIGN_UID,
        classification=classification,
        force=True,
    )
    assert forced["changed"] is True
    assert not ariadne_batch_decision_path(active_iteration_dir(tmp_path, 1)).exists()


def test_direct_postprocess_recovery_archives_accepted_complete_publication(tmp_path):
    _write_publication(tmp_path, accepted=True)
    state = SimpleNamespace(iteration=1, campaign_uid=CAMPAIGN_UID)

    archived = archive_stale_ariadne_publication(
        tmp_path,
        state,
        retry_task_ids=[],
    )

    assert archived["changed"] is True
    assert classify_ariadne_publication(tmp_path, 1)["state"] == "absent"


def test_direct_postprocess_recovery_preserves_rejected_complete_publication(tmp_path):
    _write_publication(tmp_path, accepted=False)
    state = SimpleNamespace(iteration=1, campaign_uid=CAMPAIGN_UID)
    classification = classify_ariadne_publication(tmp_path, 1)
    assert classification["state"] == "complete"
    assert classification["accepted"] is False

    with pytest.raises(BackendSubmissionError, match="preserved for user review"):
        archive_stale_ariadne_publication(
            tmp_path,
            state,
            retry_task_ids=[],
        )

    assert ariadne_batch_decision_path(active_iteration_dir(tmp_path, 1)).is_file()


def test_reconcile_mutation_archives_stale_publication(tmp_path):
    iteration_dir = _write_publication(tmp_path)
    _make_results_stale(iteration_dir)
    classification = classify_ariadne_publication(
        tmp_path,
        1,
        expected_campaign_uid=CAMPAIGN_UID,
    )
    recorded = []
    transaction = SimpleNamespace(
        record_paths=lambda operation, paths: recorded.append(
            (operation, list(paths))
        )
    )
    report = SimpleNamespace(
        proposed_state=SimpleNamespace(
            phase=CampaignPhase.ARIADNE_ARRAY,
            iteration=1,
            campaign_uid=CAMPAIGN_UID,
        ),
        ariadne_publication_recovery=classification,
        unsafe_reasons=[],
    )

    result = _perform_reconcile_apply_mutations(
        tmp_path,
        report,
        transaction=transaction,
        retrain_ferebus=False,
        force_resubmit_array=False,
        partial_array=None,
        force_array_phase=CampaignPhase.ARIADNE_ARRAY,
        force_array_iteration=1,
        archive_existing_array_outputs=False,
        data_staging_archive_mode=None,
    )

    assert len(result["archived_ariadne_publication"]) == 1
    assert recorded[0][0] == "archive_ariadne_publication"
    assert classify_ariadne_publication(tmp_path, 1)["state"] == "absent"


def test_reconcile_mutation_archives_accepted_uncommitted_publication(tmp_path):
    _write_publication(tmp_path, accepted=True)
    classification = classify_ariadne_publication(
        tmp_path,
        1,
        expected_campaign_uid=CAMPAIGN_UID,
    )
    classification["archive_for_replay"] = True
    transaction = SimpleNamespace(record_paths=lambda *_args, **_kwargs: None)
    report = SimpleNamespace(
        proposed_state=SimpleNamespace(
            phase=CampaignPhase.ARIADNE_ARRAY,
            iteration=1,
            campaign_uid=CAMPAIGN_UID,
        ),
        ariadne_publication_recovery=classification,
        unsafe_reasons=[],
    )

    result = _perform_reconcile_apply_mutations(
        tmp_path,
        report,
        transaction=transaction,
        retrain_ferebus=False,
        force_resubmit_array=False,
        partial_array=None,
        force_array_phase=CampaignPhase.ARIADNE_ARRAY,
        force_array_iteration=1,
        archive_existing_array_outputs=False,
        data_staging_archive_mode=None,
    )

    assert len(result["archived_ariadne_publication"]) == 1
    assert classify_ariadne_publication(tmp_path, 1)["state"] == "absent"
