"""Strict submission-intent lifecycle contracts."""

from __future__ import annotations

import getpass
import hashlib
import json

import pytest

import ichor.hpc.active_learning.daemon.submission_intent as intent_module
from ichor.hpc.active_learning.daemon.submission_intent import (
    aimall_postprocess_task_contract,
    bind_pre_submit_metadata,
    classify_repairable_accidental_postprocess_submission,
    classify_completed_unsubmitted_intents,
    classify_scalar_diversity_retry_intent,
    inventory_intents,
    intent_path,
    load_intent,
    mark_completed,
    mark_failed,
    mark_submitted,
    mark_superseded,
    prepare_reconcile_terminal_transition,
    publish_prepared_reconcile_transition,
    resolve_ariadne_postprocess_source,
    resolve_ariadne_terminal_postprocess_source,
    resolve_gaussian_postprocess_source,
    resolve_scalar_diversity_postprocess_source,
    write_pre_submit_intent,
)
from ichor.hpc.active_learning.daemon import input_staging
from ichor.hpc.active_learning.layout import staging_phase_dir
from ichor.hpc.active_learning.replacement_sampling import replacement_round_dir
from ichor.hpc.active_learning.daemon.completion_receipts import (
    inventory_completion_receipts,
    write_completion_receipt,
)
from ichor.hpc.active_learning.daemon.script_bundles import (
    prepare_attempt_bundle,
)
from ichor.hpc.active_learning.daemon.scheduler_recovery import (
    classify_terminal_scheduler_evidence,
    write_scheduler_terminal_receipt,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    CampaignState,
    fresh_campaign_state,
)
from ichor.hpc.active_learning.submit.sacct_poll import (
    JobObservation,
    JobStatus,
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


def test_pre_submit_metadata_binding_is_preserved_at_scheduler_acceptance(
    tmp_path,
):
    write_pre_submit_intent(
        tmp_path,
        campaign_uid="intent-test",
        phase_name="AIMALL",
        iteration=8,
        expected_tasks=1,
    )
    marker = {"kind": "fixture", "digest": "a" * 64}

    bound = bind_pre_submit_metadata(
        tmp_path,
        "AIMALL",
        8,
        {"aimall_structural_retry": marker},
    )
    submitted = mark_submitted(
        tmp_path,
        "AIMALL",
        8,
        "898505",
        expected_tasks=1,
        submission_metadata={
            "aimall_structural_retry": marker,
            "script_bundle": str(tmp_path / "bundle"),
        },
    )

    assert bound["submission_metadata"] == {
        "aimall_structural_retry": marker
    }
    assert submitted["submission_metadata"][
        "aimall_structural_retry"
    ] == marker


def test_scheduler_acceptance_cannot_replace_bound_pre_submit_metadata(
    tmp_path,
):
    write_pre_submit_intent(
        tmp_path,
        campaign_uid="intent-test",
        phase_name="AIMALL",
        iteration=8,
        expected_tasks=1,
    )
    bind_pre_submit_metadata(
        tmp_path,
        "AIMALL",
        8,
        {"aimall_structural_retry": {"task_ids": [0]}},
    )

    with pytest.raises(ValueError, match="contradicts"):
        mark_submitted(
            tmp_path,
            "AIMALL",
            8,
            "898505",
            expected_tasks=1,
            submission_metadata={
                "aimall_structural_retry": {"task_ids": [1]},
            },
        )


def test_rejected_postprocess_submission_does_not_mutate_intent(tmp_path):
    source = {
        "campaign_uid": "intent-test",
        "phase": CampaignPhase.ARIADNE_ARRAY.value,
        "iteration": 10,
        "attempt_id": "1" * 32,
        "submission_identity": "r0000-a0001-source",
        "job_id": "898513",
        "environment_generation": 7,
        "environment_generation_digest_sha256": "e" * 64,
        "logical_total": 3,
        "logical_task_set_sha256": hashlib.sha256(b"0,1,2").hexdigest(),
        "decision_contract": {
            "config_sha256": "c" * 64,
            "failure_threshold_fraction": 0.25,
        },
    }
    source["source_sha256"] = (
        intent_module._canonical_postprocess_source_sha256(source)
    )
    write_pre_submit_intent(
        tmp_path,
        campaign_uid="intent-test",
        phase_name=CampaignPhase.ARIADNE_ARRAY.value,
        iteration=10,
        expected_tasks=3,
        decision_contract=source["decision_contract"],
        postprocess_source=source,
        scheduler_identity_kind="sge",
        environment_generation=8,
        environment_generation_digest_sha256="f" * 64,
    )
    path = intent_path(tmp_path, CampaignPhase.ARIADNE_ARRAY.value, 10)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="postprocess intent must remain jobless"):
        mark_submitted(
            tmp_path,
            CampaignPhase.ARIADNE_ARRAY.value,
            10,
            "898514",
            expected_tasks=3,
        )

    assert path.read_bytes() == before
    assert load_intent(
        tmp_path,
        CampaignPhase.ARIADNE_ARRAY.value,
        10,
    )["status"] == "PRE_SUBMIT"


def test_reconcile_repairs_exact_accidental_ariadne_postprocess_submission(
    tmp_path,
    monkeypatch,
):
    phase = CampaignPhase.ARIADNE_ARRAY.value
    decision = {
        "config_sha256": "c" * 64,
        "failure_threshold_fraction": 0.25,
    }
    task_digest = hashlib.sha256(b"0,1,2").hexdigest()
    producer = write_pre_submit_intent(
        tmp_path,
        campaign_uid="intent-test",
        phase_name=phase,
        iteration=10,
        expected_tasks=3,
        decision_contract=decision,
        scheduler_identity_kind="sge",
        environment_generation=7,
        environment_generation_digest_sha256="e" * 64,
    )
    producer_bundle = prepare_attempt_bundle(
        tmp_path,
        phase,
        10,
        str(producer["submission_identity"]),
        array_size=3,
        max_log_files_per_directory=100,
        logical_task_ids=[0, 1, 2],
    )
    producer = mark_submitted(
        tmp_path,
        phase,
        10,
        "898513",
        expected_tasks=3,
        submission_metadata={
            "array_recovery": {
                "logical_total": 3,
                "n_complete": 0,
                "n_reuse": 0,
                "n_retry": 3,
            },
            "logical_task_set_sha256": task_digest,
            "script_bundle": str(producer_bundle.root),
        },
    )
    observations = [
        JobObservation(
            job_id="898513_" + str(task_id),
            status=(
                JobStatus.FAILED if task_id == 2 else JobStatus.COMPLETED
            ),
            exit_code=((139, 0) if task_id == 2 else (0, 0)),
            elapsed_seconds=24,
            job_name=str(producer["expected_job_name"]),
            owner=getpass.getuser(),
        )
        for task_id in range(3)
    ]
    terminal = classify_terminal_scheduler_evidence(
        tmp_path,
        producer,
        observations,
        queue_active=False,
    )
    write_scheduler_terminal_receipt(tmp_path, producer, terminal)
    mark_failed(tmp_path, phase, 10, "qacct parser failure")
    producer = mark_superseded(
        tmp_path,
        phase,
        10,
        intent_module.ARIADNE_TERMINAL_POSTPROCESS_REASON,
    )
    monkeypatch.setattr(
        intent_module,
        "_validate_postprocess_source_environment",
        lambda *_args, **_kwargs: None,
    )
    source = resolve_ariadne_postprocess_source(
        tmp_path,
        campaign_uid="intent-test",
        iteration=10,
        logical_total=3,
        logical_task_set_sha256=task_digest,
        intent=producer,
    )
    wrapper = write_pre_submit_intent(
        tmp_path,
        campaign_uid="intent-test",
        phase_name=phase,
        iteration=10,
        expected_tasks=3,
        decision_contract=decision,
        postprocess_source=source,
        scheduler_identity_kind="sge",
        environment_generation=8,
        environment_generation_digest_sha256="f" * 64,
    )
    bundle = prepare_attempt_bundle(
        tmp_path,
        phase,
        10,
        str(wrapper["submission_identity"]),
        array_size=3,
        max_log_files_per_directory=100,
        logical_task_ids=[0, 1, 2],
    )

    recovery = {
        "logical_total": 3,
        "n_complete": 0,
        "n_reuse": 0,
        "n_retry": 3,
    }
    malformed = dict(wrapper)
    malformed.update(
        status="SUBMITTED",
        job_id="898514",
        job_ids_seen=["898514"],
        submitted_at_iso="2026-08-19T10:19:19+00:00",
        queue_lifecycle={
            "submitted_at_iso": "2026-08-19T10:19:19+00:00"
        },
        expected_tasks=3,
        logical_expected_tasks=3,
        retry_expected_tasks=3,
        array_recovery=recovery,
        submission_metadata={
            "array_recovery": recovery,
            "logical_task_set_sha256": task_digest,
            "script_bundle": str(bundle.root),
        },
        resource_resolution_path="resource.json",
        resource_resolution_sha256="1" * 64,
        resource_formula_version="fixture-v1",
        scratch_path_template="/tmp/ichor/{job_id}/{task_id}",
        script_binding_path="binding.json",
        script_binding_sha256="2" * 64,
        submitted_script_path="job.sh",
        submitted_script_sha256="3" * 64,
    )
    path = intent_path(tmp_path, phase, 10)
    path.write_text(json.dumps(malformed), encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(ValueError, match="postprocess intent must remain jobless"):
        load_intent(tmp_path, phase, 10)
    repair = classify_repairable_accidental_postprocess_submission(
        tmp_path,
        phase,
        10,
        expected_campaign_uid="intent-test",
    )
    assert repair is not None
    assert repair["job_id"] == "898514"
    assert repair["source_job_id"] == "898513"
    assert "postprocess_source" not in repair["intent"]
    inventory = inventory_intents(
        tmp_path,
        expected_campaign_uid="intent-test",
    )
    assert inventory["errors"] == []
    assert inventory["repairs"][0]["repair_kind"] == (
        "accidental_ariadne_postprocess_submission"
    )
    assert inventory["records"][0]["job_id"] == "898514"
    assert path.read_bytes() == before

    current_observations = [
        JobObservation(
            job_id="898514_" + str(task_id),
            status=JobStatus.COMPLETED,
            exit_code=(0, 0),
            elapsed_seconds=24,
            job_name=str(repair["intent"]["expected_job_name"]),
            owner=getpass.getuser(),
        )
        for task_id in range(3)
    ]
    current_terminal = classify_terminal_scheduler_evidence(
        tmp_path,
        repair["intent"],
        current_observations,
        queue_active=False,
    )
    current_receipt = write_scheduler_terminal_receipt(
        tmp_path,
        repair["intent"],
        current_terminal,
    )
    assert current_receipt["n_completed"] == 3
    assert current_receipt["n_retry"] == 0

    prepared = prepare_reconcile_terminal_transition(
        tmp_path,
        phase,
        10,
        target_status="SUPERSEDED",
        reason=intent_module.ARIADNE_TERMINAL_POSTPROCESS_REASON,
        expected_campaign_uid="intent-test",
    )
    assert prepared["status"] == "SUPERSEDED"
    assert prepared["job_id"] == "898514"
    assert "postprocess_source" not in prepared
    published = publish_prepared_reconcile_transition(
        tmp_path,
        phase,
        10,
        prepared,
        expected_campaign_uid="intent-test",
    )
    assert published["job_id"] == "898514"
    assert published["status"] == "SUPERSEDED"

    repaired_source = resolve_ariadne_postprocess_source(
        tmp_path,
        campaign_uid="intent-test",
        iteration=10,
        logical_total=3,
        logical_task_set_sha256=task_digest,
        intent=published,
    )
    repaired_wrapper = write_pre_submit_intent(
        tmp_path,
        campaign_uid="intent-test",
        phase_name=phase,
        iteration=10,
        expected_tasks=3,
        decision_contract=decision,
        postprocess_source=repaired_source,
        scheduler_identity_kind="sge",
        environment_generation=9,
        environment_generation_digest_sha256="9" * 64,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.array_recovery.logical_task_ids",
        lambda *_args, **_kwargs: [0, 1, 2],
    )
    replay = resolve_ariadne_terminal_postprocess_source(
        tmp_path,
        campaign_uid="intent-test",
        iteration=10,
        intent=repaired_wrapper,
    )
    assert replay["postprocess_source"]["job_id"] == "898514"
    assert replay["n_scheduler_completed"] == 3
    assert replay["n_scheduler_failed"] == 0


def _scalar_retry_intent(
    *,
    phase="PHASE_B_DIVERSITY",
    scheduler="slurm",
    status="FAILED",
    reason="worker failed",
    job_id="17997698",
    terminal_status="FAILED",
):
    return {
        "campaign_uid": "intent-test",
        "phase": phase,
        "iteration": 17 if phase == "PHASE_B_DIVERSITY" else 0,
        "replacement_round": 0,
        "scheduler_identity_kind": scheduler,
        "submission_kind": "scalar",
        "expected_tasks": 1,
        "submission_identity": "r0000-a0001-test",
        "status": status,
        "reason": reason,
        "job_id": job_id,
        "job_ids_seen": [] if job_id is None else [job_id],
        "queue_lifecycle": {
            "terminal_status": terminal_status,
            "n_expected": 1,
            "n_observed": 1,
            "n_missing": 0,
        },
    }


@pytest.mark.parametrize(
    ("phase", "scheduler", "terminal_status"),
    [
        ("PHASE_A_DIVERSITY", "slurm", "COMPLETED"),
        ("PHASE_A_DIVERSITY", "sge", "FAILED"),
        ("PHASE_B_DIVERSITY", "slurm", "FAILED"),
        ("PHASE_B_DIVERSITY", "sge", "COMPLETED"),
    ],
)
def test_scalar_diversity_retry_accepts_exact_terminal_scheduler_evidence(
    phase,
    scheduler,
    terminal_status,
):
    intent = _scalar_retry_intent(
        phase=phase,
        scheduler=scheduler,
        terminal_status=terminal_status,
    )

    result = classify_scalar_diversity_retry_intent(
        intent,
        expected_campaign_uid="intent-test",
        expected_phase=phase,
        expected_iteration=int(intent["iteration"]),
        expected_replacement_round=0,
        expected_scheduler_kind=scheduler,
    )

    assert result["retry_evidence_kind"] == "terminal_scheduler_retry"
    assert result["producer_job_id"] == "17997698"
    assert result["scheduler_terminal_status"] == terminal_status


def test_scalar_diversity_retry_accepts_jobless_failed_attempt():
    intent = _scalar_retry_intent(job_id=None)

    result = classify_scalar_diversity_retry_intent(
        intent,
        expected_campaign_uid="intent-test",
        expected_phase="PHASE_B_DIVERSITY",
        expected_iteration=17,
        expected_replacement_round=0,
        expected_scheduler_kind="slurm",
    )

    assert result["retry_evidence_kind"] == "jobless_pre_submit_retry"
    assert result["producer_job_id"] is None


def test_scalar_diversity_postprocess_source_requires_completed_producer(
    tmp_path,
    monkeypatch,
):
    intent = _scalar_retry_intent(terminal_status="COMPLETED")
    intent.update(
        attempt_id="1" * 32,
        environment_generation=3,
        environment_generation_digest_sha256="2" * 64,
        decision_contract={
            "config_sha256": "3" * 64,
            "failure_threshold_fraction": 0.2,
        },
    )
    monkeypatch.setattr(
        intent_module,
        "_validate_postprocess_source_environment",
        lambda *_args, **_kwargs: None,
    )

    source = resolve_scalar_diversity_postprocess_source(
        tmp_path,
        campaign_uid="intent-test",
        phase_name="PHASE_B_DIVERSITY",
        iteration=17,
        scheduler_identity_kind="slurm",
        intent=intent,
    )

    assert source["job_id"] == "17997698"
    assert source["logical_total"] == 1
    assert source["logical_task_set_sha256"] == hashlib.sha256(b"0").hexdigest()

    intent["queue_lifecycle"]["terminal_status"] = "FAILED"
    with pytest.raises(ValueError, match="scheduler task did not complete"):
        resolve_scalar_diversity_postprocess_source(
            tmp_path,
            campaign_uid="intent-test",
            phase_name="PHASE_B_DIVERSITY",
            iteration=17,
            scheduler_identity_kind="slurm",
            intent=intent,
        )


def test_phase_a_postprocess_source_uses_same_scalar_contract(tmp_path, monkeypatch):
    intent = _scalar_retry_intent(terminal_status="COMPLETED")
    intent.update(
        phase="PHASE_A_DIVERSITY",
        iteration=0,
        attempt_id="1" * 32,
        environment_generation=3,
        environment_generation_digest_sha256="2" * 64,
        decision_contract={
            "config_sha256": "3" * 64,
            "failure_threshold_fraction": 0.2,
        },
    )
    monkeypatch.setattr(
        intent_module,
        "_validate_postprocess_source_environment",
        lambda *_args, **_kwargs: None,
    )

    source = resolve_scalar_diversity_postprocess_source(
        tmp_path,
        campaign_uid="intent-test",
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        scheduler_identity_kind="slurm",
        intent=intent,
    )

    assert source["phase"] == "PHASE_A_DIVERSITY"
    assert source["iteration"] == 0
    assert source["logical_task_set_sha256"] == hashlib.sha256(b"0").hexdigest()


def test_scalar_diversity_postprocess_source_survives_local_failure_wrapper(
    tmp_path,
    monkeypatch,
):
    producer = _scalar_retry_intent(terminal_status="COMPLETED")
    producer.update(
        attempt_id="1" * 32,
        environment_generation=3,
        environment_generation_digest_sha256="2" * 64,
        decision_contract={
            "config_sha256": "3" * 64,
            "failure_threshold_fraction": 0.2,
        },
    )
    monkeypatch.setattr(
        intent_module,
        "_validate_postprocess_source_environment",
        lambda *_args, **_kwargs: None,
    )
    source = resolve_scalar_diversity_postprocess_source(
        tmp_path,
        campaign_uid="intent-test",
        phase_name="PHASE_B_DIVERSITY",
        iteration=17,
        scheduler_identity_kind="slurm",
        intent=producer,
    )
    wrapper = {
        **producer,
        "attempt_id": "4" * 32,
        "submission_identity": "r0000-a0002-test",
        "status": "FAILED",
        "reason": "local Phase B validation failed",
        "job_id": None,
        "job_ids_seen": [],
        "queue_lifecycle": None,
        "postprocess_source": source,
    }
    monkeypatch.setattr(
        intent_module,
        "intent_attempt_records",
        lambda *_args, **_kwargs: [producer],
    )

    repeated = resolve_scalar_diversity_postprocess_source(
        tmp_path,
        campaign_uid="intent-test",
        phase_name="PHASE_B_DIVERSITY",
        iteration=17,
        scheduler_identity_kind="slurm",
        intent=wrapper,
    )

    assert repeated == source


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(status="SUBMITTED"), "safely terminal"),
        (
            lambda value: value["queue_lifecycle"].update(n_missing=1),
            "exactly one terminal task",
        ),
        (
            lambda value: value["queue_lifecycle"].update(
                terminal_status="CANCELLED"
            ),
            "exactly one terminal task",
        ),
        (
            lambda value: value.update(expected_tasks=2),
            "exactly one expected task",
        ),
        (
            lambda value: value.update(job_ids_seen=[]),
            "JobID history",
        ),
    ],
)
def test_scalar_diversity_retry_rejects_ambiguous_terminal_evidence(
    mutation,
    message,
):
    intent = _scalar_retry_intent()
    mutation(intent)

    with pytest.raises(ValueError, match=message):
        classify_scalar_diversity_retry_intent(
            intent,
            expected_campaign_uid="intent-test",
            expected_phase="PHASE_B_DIVERSITY",
            expected_iteration=17,
            expected_replacement_round=0,
            expected_scheduler_kind="slurm",
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
