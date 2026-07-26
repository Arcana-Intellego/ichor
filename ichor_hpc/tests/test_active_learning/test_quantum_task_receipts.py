"""Content binding for reusable Gaussian and AIMAll array tasks."""

from __future__ import annotations

import pytest

import ichor.hpc.active_learning.daemon.quantum_task_receipts as receipts_module
import ichor.hpc.active_learning.daemon.submission_intent as intent_module
from ichor.hpc.active_learning.daemon.quantum_task_receipts import (
    read_quantum_task_receipt,
    write_quantum_task_receipt,
    write_quantum_task_receipt_from_terminal_intent,
)
from ichor.hpc.active_learning.daemon.submission_intent import (
    load_intent,
    mark_completed,
    mark_submitted,
    write_pre_submit_intent,
)


def _submitted_gaussian(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    pointdir = campaign / ".DATA" / "STAGING" / "iter_1" / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True)
    (pointdir / "input.gjf").write_text("# fixture\n", encoding="utf-8")
    (pointdir / "input.gau").write_text("Normal termination\n", encoding="utf-8")
    (pointdir / "input.wfn").write_text("WFN fixture\n", encoding="utf-8")
    write_pre_submit_intent(
        campaign,
        campaign_uid="receipt-test",
        phase_name="GAUSSIAN",
        iteration=1,
        expected_tasks=1,
    )
    mark_submitted(
        campaign,
        "GAUSSIAN",
        1,
        "12345",
        expected_tasks=1,
    )
    return campaign, pointdir


def test_gaussian_task_receipt_binds_input_output_and_attempt(tmp_path):
    campaign, pointdir = _submitted_gaussian(tmp_path)

    write_quantum_task_receipt(
        campaign,
        pointdir,
        phase_name="GAUSSIAN",
        iteration=1,
        logical_task_id=0,
    )
    payload = read_quantum_task_receipt(
        pointdir,
        phase_name="GAUSSIAN",
        iteration=1,
        logical_task_id=0,
    )

    assert payload["job_id"] == "12345"
    assert {record["path"] for record in payload["inputs"]} == {"input.gjf"}
    assert {record["path"] for record in payload["outputs"]} == {
        "input.gau",
        "input.wfn",
    }


def test_quantum_task_receipt_rejects_input_drift(tmp_path):
    campaign, pointdir = _submitted_gaussian(tmp_path)
    write_quantum_task_receipt(
        campaign,
        pointdir,
        phase_name="GAUSSIAN",
        iteration=1,
        logical_task_id=0,
    )
    (pointdir / "input.gjf").write_text("# changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artefact binding mismatch"):
        read_quantum_task_receipt(
            pointdir,
            phase_name="GAUSSIAN",
            iteration=1,
            logical_task_id=0,
        )


def test_quantum_task_receipt_rejects_unbound_output_set_change(tmp_path):
    campaign, pointdir = _submitted_gaussian(tmp_path)
    write_quantum_task_receipt(
        campaign,
        pointdir,
        phase_name="GAUSSIAN",
        iteration=1,
        logical_task_id=0,
    )
    (pointdir / "second.gaussianoutput").write_text(
        "unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one GJF, one output and one WFN"):
        read_quantum_task_receipt(
            pointdir,
            phase_name="GAUSSIAN",
            iteration=1,
            logical_task_id=0,
        )


def test_quantum_task_receipt_rejects_fractional_identity(tmp_path):
    campaign, pointdir = _submitted_gaussian(tmp_path)

    with pytest.raises(ValueError, match="logical_task_id"):
        write_quantum_task_receipt(
            campaign,
            pointdir,
            phase_name="GAUSSIAN",
            iteration=1,
            logical_task_id=0.5,
        )


def test_quantum_task_receipt_can_be_reconstructed_from_completed_intent(tmp_path):
    campaign, pointdir = _submitted_gaussian(tmp_path)
    mark_completed(campaign, "GAUSSIAN", 1)
    intent = load_intent(
        campaign,
        "GAUSSIAN",
        1,
        expected_campaign_uid="receipt-test",
    )

    write_quantum_task_receipt_from_terminal_intent(
        pointdir,
        phase_name="GAUSSIAN",
        iteration=1,
        logical_task_id=0,
        intent=intent,
    )
    payload = read_quantum_task_receipt(
        pointdir,
        phase_name="GAUSSIAN",
        iteration=1,
        logical_task_id=0,
    )

    assert payload["attempt_id"] == intent["attempt_id"]
    assert payload["submission_identity"] == intent["submission_identity"]
    assert payload["job_id"] == "12345"


def test_aimall_postprocess_receipt_uses_original_scheduler_producer(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    pointdir = (
        campaign
        / ".DATA"
        / "STAGING"
        / "iter_14"
        / "POINT_0000.pointdir"
    )
    pointdir.mkdir(parents=True)
    for name in (
        "input.wfn",
        "AIMALL_TASK.json",
        "WFN_METHOD_RECEIPT.json",
        "GAUSSIAN_TASK_RECEIPT.json",
        "o1.int",
    ):
        (pointdir / name).write_text(name + "\n", encoding="utf-8")
    wrapper = {
        "status": "PRE_SUBMIT",
        "campaign_uid": "receipt-test",
        "replacement_round": 0,
        "postprocess_source": {"source": "fixture"},
    }
    producer = {
        "campaign_uid": "receipt-test",
        "attempt_id": "original-attempt",
        "submission_identity": "r0000-a0001-original",
        "job_id": "17888108",
        "logical_total": 149,
    }
    monkeypatch.setattr(
        receipts_module,
        "load_intent",
        lambda *_args, **_kwargs: dict(wrapper),
    )
    monkeypatch.setattr(
        intent_module,
        "resolve_aimall_postprocess_source",
        lambda *_args, **_kwargs: dict(producer),
    )

    receipt_path = write_quantum_task_receipt(
        campaign,
        pointdir,
        phase_name="AIMALL",
        iteration=14,
        logical_task_id=0,
    )
    original_bytes = receipt_path.read_bytes()
    repeated = write_quantum_task_receipt(
        campaign,
        pointdir,
        phase_name="AIMALL",
        iteration=14,
        logical_task_id=0,
    )
    payload = read_quantum_task_receipt(
        pointdir,
        phase_name="AIMALL",
        iteration=14,
        logical_task_id=0,
    )

    assert repeated == receipt_path
    assert repeated.read_bytes() == original_bytes
    assert payload["attempt_id"] == "original-attempt"
    assert payload["submission_identity"] == "r0000-a0001-original"
    assert payload["job_id"] == "17888108"
