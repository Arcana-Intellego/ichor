"""Delta-only QM reference-data storage and resolution contracts."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from ichor.hpc.active_learning.daemon.input_staging import (
    accepted_allocation_pointdirs,
    commit_reference_data_delta,
)
from ichor.hpc.active_learning.point_allocation import (
    accepted_attempts,
    create_point_allocation,
    pending_attempts,
    point_allocation_path,
    record_quantum_results,
)
from ichor.hpc.active_learning.versioning.provenance import (
    ProvenanceError,
    enrich_with_point_allocation,
    write_seed_provenance,
)
from ichor.hpc.active_learning.versioning.reference_data import (
    REFERENCE_DATA_VERSION_FILENAME,
    ReferenceDataError,
    ReferenceDataVersioning,
)
from ichor.hpc.active_learning.layout import (
    reject_legacy_campaign_layout,
    reject_legacy_training_layout,
)
from ichor_hpc.tests.quantum_test_support import (
    attach_synthetic_quantum_acceptance,
    synthetic_quantum_quality_record,
)


def _complete_allocation(
    campaign: Path,
    *,
    context: str,
    iteration: int,
    first_frame_id: int,
) -> None:
    allocation_path = point_allocation_path(
        campaign,
        context=context,
        iteration=iteration,
    )
    candidates = [
        {
            "candidate_id": context + "-candidate-" + str(first_frame_id + offset),
            "frame_id": first_frame_id + offset,
            "pointdir_name": "POINT_" + str(offset).zfill(4) + ".pointdir",
        }
        for offset in range(2)
    ]
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="campaign-uid",
        context=context,
        iteration=iteration,
        targets={"train": 1, "int_val": 1, "ext_val": 0, "total": 2},
        primary_candidates=candidates,
        reserve_candidates=[],
    )
    staging = (
        campaign / ".DATA" / "STAGING" / "initial"
        if context == "bootstrap"
        else campaign / ".DATA" / "STAGING" / ("iter_" + str(iteration))
    )
    results = []
    for attempt in pending_attempts(allocation):
        pointdir = staging / str(attempt["pointdir_name"])
        pointdir.mkdir(parents=True)
        (pointdir / "input.gjf").write_text("# synthetic\n", encoding="utf-8")
        write_seed_provenance(
            pointdir,
            campaign_uid="campaign-uid",
            iteration=iteration,
            trajectory_sha256="a" * 64,
            seed_frame_id=int(attempt["frame_id"]),
            seed_id=(
                int(attempt["slot_id"]) + 1
                if context == "active"
                else None
            ),
            seed_uid=(
                format(int(attempt["slot_id"]) + 1, "064x")
                if context == "active"
                else None
            ),
            array_task_id_zero_based=(
                int(attempt["slot_id"])
                if context == "active"
                else None
            ),
            seed_selection_origin="bootstrap" if context == "bootstrap" else "variance",
            seed_variance_at_selection=None,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
        )
        enrich_with_point_allocation(
            pointdir,
            candidate_id=str(attempt["candidate_id"]),
            context=context,
            slot_id=int(attempt["slot_id"]),
            split=str(attempt["split"]),
            replacement_round=int(attempt.get("round", 0)),
            allocation_slot_assignment_sha256=str(
                allocation["slot_assignment_sha256"]
            ),
        )
        results.append(
            {
                "candidate_id": str(attempt["candidate_id"]),
                "accepted": True,
                "pointdir": str(pointdir),
            }
        )
    from ichor.hpc.active_learning.daemon.quantum_quality import (
        write_quantum_quality_manifest,
    )

    quality_records = [
        synthetic_quantum_quality_record(Path(result["pointdir"]).name)
        for result in results
    ]
    quality_path = write_quantum_quality_manifest(
        staging,
        phase_name=("INITIAL_AIMALL" if context == "bootstrap" else "AIMALL"),
        iteration=iteration,
        records=quality_records,
        gates={},
    )
    for result, quality_record in zip(results, quality_records):
        result["quality_manifest"] = str(quality_path.resolve())
        result.update(
            attach_synthetic_quantum_acceptance(
                campaign,
                Path(result["pointdir"]),
                phase_name=("INITIAL_AIMALL" if context == "bootstrap" else "AIMALL"),
                iteration=iteration,
                quality_manifest=quality_path,
                quality_record=quality_record,
            )
        )
    completed = record_quantum_results(allocation_path, results)
    assert len(accepted_attempts(completed)) == 2


def _commit_two_versions(campaign: Path):
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    first, first_names, first_created = commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    _complete_allocation(
        campaign,
        context="active",
        iteration=1,
        first_frame_id=10,
    )
    second, second_names, second_created = commit_reference_data_delta(
        campaign,
        reference_data_version=1,
        context="active",
        iteration=1,
    )
    return first, first_names, first_created, second, second_names, second_created


def test_reference_data_versions_store_only_their_delta(tmp_path):
    campaign = tmp_path / "campaign"
    first, first_names, first_created, second, second_names, second_created = (
        _commit_two_versions(campaign)
    )
    root = campaign / "QM_REFERENCE_DATA"

    assert first_created is True
    assert second_created is True
    assert first_names == ["POINT_000000.pointdir", "POINT_000001.pointdir"]
    assert second_names == ["POINT_000002.pointdir", "POINT_000003.pointdir"]
    assert sorted(path.name for path in (root / "iteration-000000").glob("*.pointdir")) == first_names
    assert sorted(path.name for path in (root / "iteration-000001").glob("*.pointdir")) == second_names
    assert [entry.pointdir_name for entry in second.entries] == first_names + second_names
    assert len(first.entries) == 2
    assert len(second.entries) == 4
    assert second.head_manifest_sha256 != first.head_manifest_sha256


def test_committed_quantum_quality_paths_are_version_relative(tmp_path):
    campaign = tmp_path / "campaign"
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    version_dir = campaign / "QM_REFERENCE_DATA" / "iteration-000000"
    manifest = json.loads(
        (version_dir / REFERENCE_DATA_VERSION_FILENAME).read_text(encoding="utf-8")
    )

    evidence = manifest["quantum_quality_evidence"]
    assert evidence
    assert all(not Path(record["path"]).is_absolute() for record in evidence)
    assert all(not Path(record["source_path"]).is_absolute() for record in evidence)
    committed = json.loads(
        (version_dir / evidence[0]["path"]).read_text(encoding="utf-8")
    )
    accepted = [record for record in committed["records"] if record["accepted"]]
    assert all(record.get("committed_pointdir") for record in accepted)
    assert all(record.get("candidate_id") for record in accepted)


def test_committed_acceptance_receipts_survive_staging_removal(tmp_path):
    from ichor.hpc.active_learning.daemon.quantum_acceptance_receipts import (
        QUANTUM_ACCEPTANCE_RECEIPT,
        read_quantum_acceptance_receipt,
    )

    campaign = tmp_path / "campaign"
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    version_dir = campaign / "QM_REFERENCE_DATA" / "iteration-000000"
    for pointdir in sorted(version_dir.glob("POINT_*.pointdir")):
        receipt = json.loads(
            (pointdir / QUANTUM_ACCEPTANCE_RECEIPT).read_text(encoding="utf-8")
        )
        assert receipt["committed_pointdir"] == pointdir.name
        assert receipt["quality_manifest"]["path"].startswith(
            "QM_REFERENCE_DATA/iteration-000000/quality_evidence/"
        )

    shutil.rmtree(campaign / ".DATA" / "STAGING")
    view = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA").resolve(
        0,
        verification="deep",
    )
    assert len(view.entries) == 2
    for entry in view.entries:
        read_quantum_acceptance_receipt(
            campaign,
            entry.pointdir_path,
            expected_candidate_id=entry.candidate_id,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("iteration", 9), "provenance iteration mismatch"),
        (
            ("point_allocation.slot_assignment_sha256", "f" * 64),
            "slot_assignment_sha256 mismatch",
        ),
    ],
)
def test_allocation_join_rejects_stale_provenance_bindings(
    tmp_path, mutation, message
):
    campaign = tmp_path / "campaign"
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    pointdir = (
        campaign / ".DATA" / "STAGING" / "initial" / "POINT_0000.pointdir"
    )
    provenance_path = pointdir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    path, value = mutation
    if path == "iteration":
        provenance[path] = value
    else:
        provenance["point_allocation"]["slot_assignment_sha256"] = value
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ProvenanceError, match=message):
        accepted_allocation_pointdirs(
            campaign,
            context="bootstrap",
            iteration=0,
        )


def test_reference_data_hash_chain_detects_parent_manifest_tamper(tmp_path):
    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)
    manifest = (
        campaign
        / "QM_REFERENCE_DATA"
        / "iteration-000000"
        / REFERENCE_DATA_VERSION_FILENAME
    )
    os.chmod(manifest, stat.S_IMODE(manifest.stat().st_mode) | stat.S_IWUSR)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tampered"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReferenceDataError, match="parent manifest SHA mismatch"):
        ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA").resolve(
            1,
            verification="metadata",
        )


def test_reference_data_resolution_has_no_mutable_cache_side_effect(tmp_path):
    campaign = tmp_path / "campaign"
    *_, second, _, _ = _commit_two_versions(campaign)
    before = sorted(
        path.relative_to(campaign).as_posix()
        for path in campaign.rglob("*")
        if path.is_file()
    )

    rebuilt = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA").resolve(1)
    after = sorted(
        path.relative_to(campaign).as_posix()
        for path in campaign.rglob("*")
        if path.is_file()
    )

    assert rebuilt.cumulative_view_sha256 == second.cumulative_view_sha256
    assert after == before


def test_committed_reference_pointdirs_are_read_only(tmp_path):
    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)
    pointdir = (
        campaign
        / "QM_REFERENCE_DATA"
        / "iteration-000001"
        / "POINT_000002.pointdir"
    )
    file_path = pointdir / "input.gjf"

    assert stat.S_IMODE(pointdir.stat().st_mode) & stat.S_IWUSR == 0
    assert stat.S_IMODE(file_path.stat().st_mode) & stat.S_IWUSR == 0


def test_idempotent_older_commit_does_not_move_current_backwards(tmp_path):
    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)
    versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")

    replayed, names, created = commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )

    assert created is False
    assert replayed.version == 0
    assert names == ["POINT_000000.pointdir", "POINT_000001.pointdir"]
    assert versioning.current_version() == 1


def test_legacy_training_directory_is_rejected(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / "5_TRAINING").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="unsupported legacy reference-data layout"):
        reject_legacy_training_layout(campaign)


def test_legacy_trained_models_directory_is_rejected(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / "6_TRAINED_MODELS").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="unsupported legacy trained-model layout"):
        reject_legacy_campaign_layout(campaign)


def test_mixed_legacy_and_canonical_model_roots_are_rejected(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / "6_TRAINED_MODELS").mkdir(parents=True)
    (campaign / "TRAINED_MODELS").mkdir()

    with pytest.raises(RuntimeError, match="alongside"):
        reject_legacy_campaign_layout(campaign)
