"""Strict submission-intent lifecycle contracts."""

from __future__ import annotations

import json

import pytest

import ichor.hpc.active_learning.daemon.submission_intent as intent_module
from ichor.hpc.active_learning.daemon.submission_intent import (
    aimall_postprocess_task_contract,
    classify_completed_unsubmitted_intents,
    intent_path,
    load_intent,
    mark_completed,
    mark_failed,
    mark_superseded,
    resolve_gaussian_postprocess_source,
    write_pre_submit_intent,
)
from ichor.hpc.active_learning.daemon import input_staging
from ichor.hpc.active_learning.layout import staging_phase_dir
from ichor.hpc.active_learning.replacement_sampling import replacement_round_dir
from ichor.hpc.active_learning.daemon.completion_receipts import (
    inventory_completion_receipts,
    write_completion_receipt,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    CampaignState,
    fresh_campaign_state,
)


def test_submission_intent_rejects_fractional_task_count(tmp_path):
    with pytest.raises(ValueError, match="expected_tasks"):
        write_pre_submit_intent(
            tmp_path,
            campaign_uid="intent-test",
            phase_name="GAUSSIAN",
            iteration=1,
            expected_tasks=1.5,
        )


@pytest.mark.parametrize(
    (
        "phase",
        "gaussian_phase",
        "iteration",
        "replacement_round",
        "context",
    ),
    [
        ("INITIAL_AIMALL", "INITIAL_GAUSSIAN", 0, 0, None),
        ("AIMALL", "GAUSSIAN", 3, 0, None),
        (
            "INITIAL_REPLACEMENT_AIMALL",
            "INITIAL_REPLACEMENT_GAUSSIAN",
            0,
            2,
            "bootstrap",
        ),
        (
            "REPLACEMENT_AIMALL",
            "REPLACEMENT_GAUSSIAN",
            3,
            2,
            "active",
        ),
    ],
)
def test_aimall_postprocess_task_contract_covers_all_aimall_phases(
    tmp_path,
    phase,
    gaussian_phase,
    iteration,
    replacement_round,
    context,
):
    if context is None:
        staging = staging_phase_dir(tmp_path, phase, iteration)
    else:
        staging = replacement_round_dir(
            tmp_path,
            context=context,
            iteration=iteration,
            replacement_round=replacement_round,
        )
    accepted = [
        staging / "POINT_0001.pointdir",
        staging / "POINT_0003.pointdir",
    ]
    for pointdir in accepted:
        pointdir.mkdir(parents=True, exist_ok=True)
    input_staging.write_quantum_acceptance_manifest(
        staging,
        phase_name=gaussian_phase,
        iteration=iteration,
        accepted=accepted,
        rejected=[("POINT_0002.pointdir", "fixture rejection")],
    )
    input_staging.write_points_file(staging, accepted)

    contract = aimall_postprocess_task_contract(
        tmp_path,
        phase_name=phase,
        iteration=iteration,
        replacement_round=replacement_round,
    )

    assert contract["phase"] == phase
    assert contract["gaussian_phase"] == gaussian_phase
    assert contract["logical_total"] == 2
    assert contract["gaussian_n_total"] == 3


def test_gaussian_postprocess_source_requires_complete_scheduler_evidence(
    tmp_path,
    monkeypatch,
):
    task_digest = "1" * 64
    decision_contract = {
        "failure_threshold_fraction": 0.5,
        "config_sha256": "2" * 64,
    }
    contract = {
        "campaign_uid": "intent-test",
        "phase": "GAUSSIAN",
        "iteration": 3,
        "replacement_round": 0,
        "logical_total": 2,
        "logical_task_set_sha256": task_digest,
        "staging": str(tmp_path / "staging"),
    }
    monkeypatch.setattr(
        intent_module,
        "gaussian_postprocess_task_contract",
        lambda *_args, **_kwargs: dict(contract),
    )
    monkeypatch.setattr(
        intent_module,
        "_validate_postprocess_source_environment",
        lambda *_args, **_kwargs: None,
    )
    intent = {
        "status": "FAILED",
        "reason": "postprocess_exception",
        "campaign_uid": "intent-test",
        "phase": "GAUSSIAN",
        "iteration": 3,
        "replacement_round": 0,
        "submission_kind": "array",
        "scheduler_identity_kind": "slurm",
        "expected_tasks": 2,
        "job_id": "12345",
        "attempt_id": "a" * 32,
        "submission_identity": "r0000-a0001-test",
        "environment_generation": 4,
        "environment_generation_digest_sha256": "3" * 64,
        "decision_contract": decision_contract,
        "submission_metadata": {
            "logical_task_set_sha256": task_digest,
        },
        "queue_lifecycle": {
            "terminal_status": "COMPLETED",
            "n_expected": 2,
            "n_observed": 2,
            "n_missing": 0,
        },
    }

    source = resolve_gaussian_postprocess_source(
        tmp_path,
        campaign_uid="intent-test",
        phase_name="GAUSSIAN",
        iteration=3,
        intent=intent,
    )

    assert source["job_id"] == "12345"
    assert source["attempt_id"] == "a" * 32
    assert source["logical_total"] == 2

    incomplete = dict(intent)
    incomplete["queue_lifecycle"] = {
        **intent["queue_lifecycle"],
        "n_missing": 1,
    }
    with pytest.raises(ValueError, match="does not prove complete task ownership"):
        resolve_gaussian_postprocess_source(
            tmp_path,
            campaign_uid="intent-test",
            phase_name="GAUSSIAN",
            iteration=3,
            intent=incomplete,
        )


def test_submission_intent_rejects_unknown_status_on_read(tmp_path):
    write_pre_submit_intent(
        tmp_path,
        campaign_uid="intent-test",
        phase_name="GAUSSIAN",
        iteration=1,
        expected_tasks=1,
    )
    path = intent_path(tmp_path, "GAUSSIAN", 1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "SUBMITED"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="status"):
        load_intent(tmp_path, "GAUSSIAN", 1)


def test_submission_intent_rejects_illegal_transition(tmp_path):
    write_pre_submit_intent(
        tmp_path,
        campaign_uid="intent-test",
        phase_name="GAUSSIAN",
        iteration=1,
        expected_tasks=1,
    )

    with pytest.raises(ValueError, match="illegal submission intent transition"):
        mark_completed(tmp_path, "GAUSSIAN", 1)


def test_submission_intent_mutator_requires_predecessor(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mark_failed(tmp_path, "GAUSSIAN", 1, "fixture")


def test_jobless_pre_submit_intent_can_be_retired_with_completion_receipt(
    tmp_path,
):
    write_pre_submit_intent(
        tmp_path,
        campaign_uid="intent-test",
        phase_name="INITIAL_FEREBUS",
        iteration=0,
        expected_tasks=6,
    )
    reference = {
        "path": ".DATA/ACTIVE_LEARNING/phase_completions/" + "a" * 64 + ".json",
        "receipt_id": "a" * 64,
        "sha256": "b" * 64,
    }

    retired = mark_superseded(
        tmp_path,
        "INITIAL_FEREBUS",
        0,
        "phase_completed_without_scheduler_submission",
        completion_receipt=reference,
    )

    assert retired["status"] == "SUPERSEDED"
    assert retired["job_id"] is None
    assert retired["reason"] == "phase_completed_without_scheduler_submission"
    assert retired["completion_receipt"] == reference


def test_receipt_backed_jobless_intent_classification_is_exact_and_read_only(
    tmp_path,
):
    before = fresh_campaign_state(max_iterations=2, campaign_uid="intent-test")
    before.phase = CampaignPhase.INITIAL_FEREBUS
    before.reference_data_version = 0
    before.validation_set_version = 0
    intent = write_pre_submit_intent(
        tmp_path,
        campaign_uid=before.campaign_uid,
        phase_name=before.phase.value,
        iteration=0,
        expected_tasks=6,
    )
    after = CampaignState.from_dict(before.to_dict())
    after.phase = CampaignPhase.SEED_SELECT
    after.iteration = 1
    after.models_version = 0
    write_completion_receipt(
        tmp_path,
        campaign_uid=before.campaign_uid,
        phase=before.phase.value,
        iteration=0,
        replacement_round=0,
        config_sha256="c" * 64,
        state_before=before,
        state_after=after,
        next_phase=after.phase.value,
        next_iteration=after.iteration,
        state_updates={"models_version": 0},
        evidence=[],
        job_id=None,
        expected_tasks=6,
        submission_identity=str(intent["submission_identity"]),
    )
    halted = CampaignState.from_dict(after.to_dict())
    halted.phase = CampaignPhase.HALTED
    inventory = inventory_completion_receipts(
        tmp_path,
        expected_campaign_uid=halted.campaign_uid,
    )

    classified = classify_completed_unsubmitted_intents(
        tmp_path,
        halted,
        intents=[intent],
        completion_receipts=inventory["records"],
        valid_reference_data_versions=[0],
        valid_model_versions=[0],
    )

    assert classified["errors"] == []
    assert len(classified["repairs"]) == 1
    repair = classified["repairs"][0]
    assert repair["phase"] == CampaignPhase.INITIAL_FEREBUS.value
    assert repair["target_status"] == "SUPERSEDED"
    assert repair["completion_receipt"]["receipt_id"]
    assert load_intent(tmp_path, CampaignPhase.INITIAL_FEREBUS.value, 0)[
        "status"
    ] == "PRE_SUBMIT"

    source_phase = classify_completed_unsubmitted_intents(
        tmp_path,
        before,
        intents=[intent],
        completion_receipts=inventory["records"],
        valid_reference_data_versions=[0],
        valid_model_versions=[0],
    )
    assert source_phase["repairs"] == []
