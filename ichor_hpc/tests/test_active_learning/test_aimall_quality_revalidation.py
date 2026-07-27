"""Transactional recovery of AIMAll points rejected by the former INT parser."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ichor.hpc.active_learning.cli import (
    _reconcile_apply_command,
    _reconcile_decision_payload,
)
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.aimall_quality_revalidation import (
    AIMALL_QUALITY_REVALIDATION_REASON,
    apply_aimall_quality_revalidation,
    inspect_aimall_quality_revalidation,
)
from ichor.hpc.active_learning.daemon.array_recovery import refresh_array_ledger
from ichor.hpc.active_learning.daemon.completion_receipts import (
    evidence_records,
    receipt_reference,
    write_completion_receipt,
)
from ichor.hpc.active_learning.daemon.config_lock import (
    read_config_lock,
    write_config_lock,
)
from ichor.hpc.active_learning.daemon.quantum_acceptance_receipts import (
    QUANTUM_ACCEPTANCE_RECEIPT,
)
from ichor.hpc.active_learning.daemon.quantum_quality import (
    write_quantum_quality_manifest,
)
from ichor.hpc.active_learning.daemon.quantum_task_receipts import (
    write_quantum_task_receipt,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state,
    make_lifecycle_context,
    write_state,
)
from ichor.hpc.active_learning.daemon.status_recommendations import (
    build_status_recommendations,
)
from ichor.hpc.active_learning.daemon.submission_intent import (
    mark_completed,
    mark_submitted,
    write_pre_submit_intent,
)
from ichor.hpc.active_learning.point_allocation import (
    create_point_allocation,
    point_allocation_path,
    read_point_allocation,
    record_quantum_results,
)
from ichor.hpc.active_learning.versioning.provenance import (
    enrich_with_point_allocation,
    write_seed_provenance,
)
from ichor_hpc.tests.quantum_test_support import (
    attach_synthetic_quantum_acceptance,
    synthetic_quantum_quality_record,
)


_CAMPAIGN_UID = "8a515291-b7b4-4c61-85fb-c573b924e73c"


def _build_parser_rejection_campaign(
    tmp_path: Path,
    *,
    n_accepted: int = 0,
) -> tuple[Path, CampaignConfig, Path]:
    campaign = tmp_path / "campaign"
    control = campaign / ".DATA" / "ACTIVE_LEARNING"
    control.mkdir(parents=True)
    config = CampaignConfig(max_iterations=2)
    config.to_yaml(campaign / "campaign.yaml")
    write_config_lock(campaign, config, campaign_uid=_CAMPAIGN_UID)

    allocation_path = point_allocation_path(
        campaign,
        context="active",
        iteration=1,
    )
    n_total = int(n_accepted) + 2
    candidates = [
        {
            "candidate_id": "candidate-" + str(index),
            "frame_id": index,
            "pointdir_name": "POINT_" + str(index).zfill(4) + ".pointdir",
        }
        for index in range(n_total)
    ]
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid=_CAMPAIGN_UID,
        context="active",
        iteration=1,
        targets={"train": n_total, "int_val": 0, "ext_val": 0, "total": n_total},
        primary_candidates=candidates,
        reserve_candidates=[],
    )
    staging = campaign / ".DATA" / "STAGING" / "iter_1"
    staging.mkdir(parents=True)
    attempts = [
        dict(slot["attempts"][-1]) | {
            "slot_id": int(slot["slot_id"]),
            "split": str(slot["split"]),
        }
        for slot in allocation["slots"]
    ]

    accepted_records = []
    pointdirs = []
    for logical_task_id, attempt in enumerate(attempts):
        pointdir = staging / str(attempt["pointdir_name"])
        pointdir.mkdir()
        write_seed_provenance(
            pointdir,
            campaign_uid=_CAMPAIGN_UID,
            iteration=1,
            trajectory_sha256="a" * 64,
            seed_frame_id=int(attempt["frame_id"]),
            seed_id=logical_task_id + 1,
            seed_uid=format(logical_task_id + 1, "064x"),
            array_task_id_zero_based=logical_task_id,
            seed_selection_origin="variance",
            seed_variance_at_selection=1.0,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
        )
        enrich_with_point_allocation(
            pointdir,
            candidate_id=str(attempt["candidate_id"]),
            context="active",
            slot_id=int(attempt["slot_id"]),
            split=str(attempt["split"]),
            replacement_round=0,
            allocation_slot_assignment_sha256=str(
                allocation["slot_assignment_sha256"]
            ),
        )
        accepted_records.append(synthetic_quantum_quality_record(pointdir.name))
        pointdirs.append(pointdir)

    fixture_quality = write_quantum_quality_manifest(
        staging,
        phase_name=CampaignPhase.AIMALL.value,
        iteration=1,
        records=accepted_records,
        gates=config.quality_gates,
        manifest_path=staging / "fixture-accepted-quality.json",
    )
    for logical_task_id, (pointdir, record) in enumerate(
        zip(pointdirs, accepted_records)
    ):
        attach_synthetic_quantum_acceptance(
            campaign,
            pointdir,
            phase_name=CampaignPhase.AIMALL.value,
            iteration=1,
            quality_manifest=fixture_quality,
            quality_record=record,
        )
        (pointdir / QUANTUM_ACCEPTANCE_RECEIPT).unlink()
        task_path = pointdir / "AIMALL_TASK.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task.update(
            {
                "pointdir": pointdir.name,
                "task_index": logical_task_id,
            }
        )
        task_path.write_text(
            json.dumps(task, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    (staging / "POINTS.txt").write_text(
        "\n".join(str(path.resolve()) for path in pointdirs) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_pre_submit_intent(
        campaign,
        campaign_uid=_CAMPAIGN_UID,
        phase_name=CampaignPhase.AIMALL.value,
        iteration=1,
        expected_tasks=n_total,
    )
    intent = mark_submitted(
        campaign,
        CampaignPhase.AIMALL.value,
        1,
        "123456",
        expected_tasks=n_total,
    )
    for logical_task_id, pointdir in enumerate(pointdirs):
        completion_path = pointdir / "AIMALL_COMPLETION_RECEIPT.json"
        if completion_path.exists():
            completion_path.unlink()
        if logical_task_id < int(n_accepted):
            write_quantum_task_receipt(
                campaign,
                pointdir,
                phase_name=CampaignPhase.AIMALL.value,
                iteration=1,
                logical_task_id=logical_task_id,
            )
    task_contract = SimpleNamespace(
        campaign_uid=_CAMPAIGN_UID,
        staging_dir=staging,
        logical_total=n_total,
        pointdir_names=tuple(path.name for path in pointdirs),
        tasks=tuple(
            SimpleNamespace(
                logical_task_id=logical_task_id,
                pointdir=pointdir,
                pointdir_name=pointdir.name,
                producer_logical_task_id=logical_task_id,
                candidate_id=str(attempts[logical_task_id]["candidate_id"]),
            )
            for logical_task_id, pointdir in enumerate(pointdirs)
        ),
    )
    with patch(
        "ichor.hpc.active_learning.daemon.array_recovery.logical_task_ids",
        return_value=list(range(n_total)),
    ), patch(
        "ichor.hpc.active_learning.daemon.array_recovery._pointdir_for_task",
        side_effect=lambda _campaign, _phase, _iteration, task_id: (
            pointdirs[int(task_id)]
        ),
    ), patch(
        "ichor.hpc.active_learning.daemon.quantum_task_contracts."
        "quantum_task_contract",
        return_value=task_contract,
    ):
        array_ledger = refresh_array_ledger(
            campaign,
            CampaignPhase.AIMALL.value,
            1,
        )
    assert array_ledger["all_complete"] is False
    assert array_ledger["retry_task_ids"] == list(
        range(int(n_accepted), n_total)
    )

    old_records = []
    for index, record in enumerate(accepted_records):
        if index < int(n_accepted):
            old_records.append(record)
        else:
            old_records.append(
                {
                    "pointdir": str(record["pointdir"]),
                    "accepted": False,
                    "reasons": [AIMALL_QUALITY_REVALIDATION_REASON],
                }
            )
    old_quality = write_quantum_quality_manifest(
        staging,
        phase_name=CampaignPhase.AIMALL.value,
        iteration=1,
        records=old_records,
        gates=config.quality_gates,
    )
    quantum_results = []
    for index, (attempt, pointdir, quality_record) in enumerate(
        zip(attempts, pointdirs, old_records)
    ):
        result = {
            "candidate_id": str(attempt["candidate_id"]),
            "accepted": bool(index < int(n_accepted)),
            "pointdir": str(pointdir.resolve()),
            "quality_manifest": str(old_quality.resolve()),
        }
        if result["accepted"]:
            result.update(
                attach_synthetic_quantum_acceptance(
                    campaign,
                    pointdir,
                    phase_name=CampaignPhase.AIMALL.value,
                    iteration=1,
                    quality_manifest=old_quality,
                    quality_record=quality_record,
                )
            )
        else:
            result["reason"] = AIMALL_QUALITY_REVALIDATION_REASON
        quantum_results.append(result)
    record_quantum_results(allocation_path, quantum_results)

    before = fresh_campaign_state(
        max_iterations=2,
        campaign_uid=_CAMPAIGN_UID,
    )
    before.phase = CampaignPhase.AIMALL
    before.iteration = 1
    before.reference_data_version = 0
    before.validation_set_version = 0
    before.models_version = 0
    after = fresh_campaign_state(
        max_iterations=2,
        campaign_uid=_CAMPAIGN_UID,
    )
    after.phase = CampaignPhase.ALLOCATION_CHECK
    after.iteration = 1
    after.reference_data_version = 0
    after.validation_set_version = 0
    after.models_version = 0
    lock = read_config_lock(campaign, expected_campaign_uid=_CAMPAIGN_UID)
    completion_path = write_completion_receipt(
        campaign,
        campaign_uid=_CAMPAIGN_UID,
        phase=CampaignPhase.AIMALL.value,
        iteration=1,
        replacement_round=0,
        config_sha256=str(lock["fingerprint_sha256"]),
        state_before=before,
        state_after=after,
        next_phase=CampaignPhase.ALLOCATION_CHECK.value,
        next_iteration=1,
        state_updates={},
        evidence=evidence_records(campaign, [old_quality, Path(array_ledger["path"])]),
        job_id="123456",
        expected_tasks=n_total,
        submission_identity=str(intent["submission_identity"]),
    )
    mark_completed(
        campaign,
        CampaignPhase.AIMALL.value,
        1,
        completion_receipt=receipt_reference(campaign, completion_path),
    )
    halted = fresh_campaign_state(
        max_iterations=2,
        campaign_uid=_CAMPAIGN_UID,
    )
    halted.phase = CampaignPhase.HALTED
    halted.iteration = 1
    halted.reference_data_version = 0
    halted.validation_set_version = 0
    halted.models_version = 0
    halted.lifecycle_context = make_lifecycle_context(
        disposition="halted",
        reason_code="replacement_reserve_exhausted",
        message="replacement reserve exhausted",
        from_phase=CampaignPhase.ALLOCATION_CHECK,
        iteration=1,
        source="daemon",
    )
    write_state(control / "state.json", halted)
    return campaign, config, old_quality


def test_parser_revalidation_repairs_only_the_false_rejections(
    tmp_path,
    monkeypatch,
):
    campaign, config, old_quality = _build_parser_rejection_campaign(tmp_path)
    original_quality_bytes = old_quality.read_bytes()
    from ichor.hpc.active_learning.daemon import aimall_quality_revalidation as module
    from ichor.hpc.active_learning.daemon import error_calibration

    def prepare_reference(campaign_dir, **_kwargs):
        path = (
            Path(campaign_dir)
            / ".DATA"
            / "ACTIVE_LEARNING"
            / "reference_commit_transactions"
            / "active-iteration-000001.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(module, "prepare_reference_data_delta", prepare_reference)
    monkeypatch.setattr(
        error_calibration,
        "update_from_aimall_acceptance",
        lambda **kwargs: {
            "accepted_pointdirs": [
                Path(path).name for path in kwargs["accepted_pointdirs"]
            ]
        },
    )

    preview = inspect_aimall_quality_revalidation(campaign, config=config)
    assert preview["eligible"] is True
    assert preview["candidate_count"] == 2
    assert preview["all_accepted_on_revalidation"] is True
    assert old_quality.read_bytes() == original_quality_bytes

    result = apply_aimall_quality_revalidation(campaign, config=config)

    assert result["state"] == "complete"
    assert result["candidate_count"] == 2
    assert result["no_slurm_submission"] is True
    assert old_quality.read_bytes() == original_quality_bytes
    allocation = read_point_allocation(
        point_allocation_path(campaign, context="active", iteration=1),
        expected_campaign_uid=_CAMPAIGN_UID,
    )
    assert allocation["generation"] == 2
    assert allocation["summary"]["accepted_total"] == 2
    assert allocation["summary"]["deficit_total"] == 0
    assert allocation["applied_quantum_batches"][-1]["kind"] == (
        "quality_revalidation"
    )
    history = (
        point_allocation_path(campaign, context="active", iteration=1).parent
        / "history"
        / "generation-000001.json"
    )
    rejected = json.loads(history.read_text(encoding="utf-8"))
    assert rejected["summary"]["accepted_total"] == 0
    assert all(
        (path / QUANTUM_ACCEPTANCE_RECEIPT).is_file()
        for path in sorted((campaign / ".DATA" / "STAGING" / "iter_1").glob("*.pointdir"))
    )
    assert all(
        (path / "AIMALL_COMPLETION_RECEIPT.json").is_file()
        for path in sorted((campaign / ".DATA" / "STAGING" / "iter_1").glob("*.pointdir"))
    )

    replay = apply_aimall_quality_revalidation(campaign, config=config)
    assert replay["state"] == "complete"
    assert read_point_allocation(
        point_allocation_path(campaign, context="active", iteration=1)
    )["generation"] == 2


def test_status_and_reconcile_recommend_bounded_revalidation(tmp_path):
    campaign, _config, _old_quality = _build_parser_rejection_campaign(tmp_path)
    lifecycle = {
        "disposition": "halted",
        "reason_code": "replacement_reserve_exhausted",
        "message": "replacement reserve exhausted",
        "from_phase": CampaignPhase.ALLOCATION_CHECK.value,
        "iteration": 1,
        "source": "daemon",
        "scheduler_uncertain": False,
    }

    recommendations = build_status_recommendations(
        campaign,
        {
            "lock_held": False,
            "phase": CampaignPhase.HALTED.value,
            "iteration": 1,
            "lifecycle_context": lifecycle,
            "latest_halt_event": {"reason": "replacement reserve exhausted"},
        },
    )

    assert len(recommendations) == 1
    assert recommendations[0].code == "halted_replacement_reserve_exhausted"
    assert recommendations[0].severity == "required"
    assert recommendations[0].command.startswith("ichor-al-daemon reconcile ")
    assert str(campaign) in recommendations[0].command
    assert "new campaign" not in recommendations[0].primary

    proposed = fresh_campaign_state(max_iterations=2, campaign_uid=_CAMPAIGN_UID)
    proposed.phase = CampaignPhase.HALTED
    proposed.iteration = 1
    report = SimpleNamespace(
        proposed_state=proposed,
        unsafe_reasons=[".DATA/STAGING is non-empty"],
        blocking_artifacts=[".DATA/STAGING"],
        recovery_candidates=[],
        deep_verification_required=False,
        artifact_snapshot=None,
        aimall_quality_revalidation={
            "state": "eligible",
            "eligible": True,
            "candidate_count": 2,
            "all_accepted_on_revalidation": True,
        },
    )
    contract = {
        "selected_phase": CampaignPhase.HALTED.value,
        "contract_ok": False,
        "trusted_handoffs": [],
        "missing_or_invalid_inputs": ["phase HALTED is not runnable"],
    }

    command = _reconcile_apply_command(campaign, report)
    decision = _reconcile_decision_payload(campaign, report, contract)

    assert command.startswith("ichor-al-daemon reconcile ")
    assert str(campaign) in command
    assert command.endswith(" --apply")
    assert "--archive-staging" not in command
    assert decision["next_command"] == command
    assert decision["hard_blockers"] == []


def test_parser_revalidation_resumes_after_evidence_publication(tmp_path, monkeypatch):
    campaign, config, _old_quality = _build_parser_rejection_campaign(tmp_path)
    from ichor.hpc.active_learning.daemon import aimall_quality_revalidation as module
    from ichor.hpc.active_learning.daemon import error_calibration

    real_revalidate = module.revalidate_rejected_quantum_results

    def interrupt_allocation(*_args, **_kwargs):
        raise OSError("injected allocation publication interruption")

    monkeypatch.setattr(
        module,
        "revalidate_rejected_quantum_results",
        interrupt_allocation,
    )
    with pytest.raises(OSError, match="allocation publication interruption"):
        apply_aimall_quality_revalidation(campaign, config=config)

    interrupted = inspect_aimall_quality_revalidation(campaign, config=config)
    assert interrupted["state"] == "resumable"
    assert interrupted["candidate_count"] == 2
    assert read_point_allocation(
        point_allocation_path(campaign, context="active", iteration=1)
    )["generation"] == 1

    changed_int = (
        campaign
        / ".DATA"
        / "STAGING"
        / "iter_1"
        / "POINT_0000.pointdir"
        / "input_atomicfiles"
        / "o1.int"
    )
    original_int = changed_int.read_text(encoding="utf-8")
    changed_int.write_text(
        original_int.replace("Model: B3LYP", "Model: HF", 1),
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(
        module,
        "revalidate_rejected_quantum_results",
        real_revalidate,
    )
    with pytest.raises(ValueError, match="quality changed after revalidation"):
        apply_aimall_quality_revalidation(campaign, config=config)
    changed_int.write_text(original_int, encoding="utf-8", newline="\n")

    def prepare_reference(campaign_dir, **_kwargs):
        path = (
            Path(campaign_dir)
            / ".DATA"
            / "ACTIVE_LEARNING"
            / "reference_commit_transactions"
            / "active-iteration-000001.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(module, "prepare_reference_data_delta", prepare_reference)
    monkeypatch.setattr(
        error_calibration,
        "update_from_aimall_acceptance",
        lambda **_kwargs: {"status": "updated"},
    )

    completed = apply_aimall_quality_revalidation(campaign, config=config)

    assert completed["state"] == "complete"
    assert read_point_allocation(
        point_allocation_path(campaign, context="active", iteration=1)
    )["generation"] == 2
