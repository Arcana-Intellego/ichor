"""Accepted quantum bytes remain content-bound between AIMAll and commit."""

import json
import stat
from pathlib import Path

import pytest

from ichor.hpc.active_learning.daemon.quantum_acceptance_receipts import (
    read_quantum_acceptance_receipt,
)
from ichor.hpc.active_learning.daemon.quantum_quality import (
    write_quantum_quality_manifest,
)
from ichor.hpc.active_learning.versioning.provenance import (
    enrich_with_point_allocation,
    write_seed_provenance,
)
from ichor_hpc.tests.quantum_test_support import (
    attach_synthetic_quantum_acceptance,
    synthetic_quantum_quality_record,
)


def _accepted_fixture(tmp_path):
    campaign = tmp_path / "campaign"
    pointdir = campaign / ".DATA" / "STAGING" / "initial" / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True)
    write_seed_provenance(
        pointdir,
        campaign_uid="acceptance-test",
        iteration=0,
        trajectory_sha256="a" * 64,
        seed_frame_id=0,
        seed_selection_origin="phase_a_diversity",
        seed_variance_at_selection=None,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
    )
    enrich_with_point_allocation(
        pointdir,
        candidate_id="candidate-0",
        context="bootstrap",
        slot_id=0,
        split="train",
        allocation_slot_assignment_sha256="b" * 64,
    )
    record = synthetic_quantum_quality_record(pointdir.name)
    quality = write_quantum_quality_manifest(
        pointdir.parent,
        phase_name="INITIAL_AIMALL",
        iteration=0,
        records=[record],
        gates={},
    )
    attach_synthetic_quantum_acceptance(
        campaign,
        pointdir,
        phase_name="INITIAL_AIMALL",
        iteration=0,
        quality_manifest=quality,
        quality_record=record,
    )
    return campaign, pointdir, quality


def test_quantum_acceptance_receipt_detects_late_pointdir_mutation(tmp_path):
    campaign, pointdir, _quality = _accepted_fixture(tmp_path)
    wfn = pointdir / "input.wfn"
    wfn.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="(bytes|inventory) (have|has) changed"):
        read_quantum_acceptance_receipt(
            campaign,
            pointdir,
            expected_phase="INITIAL_AIMALL",
            expected_iteration=0,
            expected_candidate_id="candidate-0",
            expected_assignment_sha256="b" * 64,
        )


def test_quantum_acceptance_hashes_and_fsyncs_each_artefact_once(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import quantum_acceptance_receipts

    calls = []
    real_stream = quantum_acceptance_receipts._stream_hash_and_fsync

    def track(path):
        calls.append(Path(path))
        return real_stream(path)

    monkeypatch.setattr(
        quantum_acceptance_receipts,
        "_stream_hash_and_fsync",
        track,
    )
    campaign, pointdir, _quality = _accepted_fixture(tmp_path)
    receipt = read_quantum_acceptance_receipt(campaign, pointdir)

    expected = {
        pointdir / str(binding["path"])
        for binding in receipt["artefacts"]
    }
    assert set(calls) == expected
    assert len(calls) == len(expected)


def test_quantum_acceptance_receipt_detects_late_quality_mutation(tmp_path):
    campaign, pointdir, quality = _accepted_fixture(tmp_path)
    quality.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="quality manifest has changed"):
        read_quantum_acceptance_receipt(campaign, pointdir)


def test_quantum_acceptance_receipt_rejects_old_schema(tmp_path):
    campaign, pointdir, _quality = _accepted_fixture(tmp_path)
    receipt = pointdir / "QUANTUM_ACCEPTANCE_RECEIPT.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="unsupported quantum acceptance"):
        read_quantum_acceptance_receipt(campaign, pointdir)


def test_quantum_acceptance_receipt_v3_keeps_campaign_owned_files_writable(tmp_path):
    campaign, pointdir, _quality = _accepted_fixture(tmp_path)
    receipt = read_quantum_acceptance_receipt(campaign, pointdir)

    assert receipt["schema_version"] == 3
    assert receipt["integrity_policy"] == "sha256_inventory"
    assert isinstance(receipt["accepted_at_iso"], str)
    assert stat.S_IMODE(pointdir.stat().st_mode) & stat.S_IWUSR
    assert stat.S_IMODE((pointdir / "input.wfn").stat().st_mode) & stat.S_IWUSR


def test_schema_two_acceptance_receipt_remains_readable_when_writable(tmp_path):
    campaign, pointdir, _quality = _accepted_fixture(tmp_path)
    receipt_path = pointdir / "QUANTUM_ACCEPTANCE_RECEIPT.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    payload["sealed_at_iso"] = payload.pop("accepted_at_iso")
    payload.pop("integrity_policy")
    receipt_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    restored = read_quantum_acceptance_receipt(campaign, pointdir)
    assert restored["schema_version"] == 2
