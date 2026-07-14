"""Content binding for reusable Gaussian and AIMAll array tasks."""

from __future__ import annotations

import pytest

from ichor.hpc.active_learning.daemon.quantum_task_receipts import (
    read_quantum_task_receipt,
    write_quantum_task_receipt,
)
from ichor.hpc.active_learning.daemon.submission_intent import (
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
