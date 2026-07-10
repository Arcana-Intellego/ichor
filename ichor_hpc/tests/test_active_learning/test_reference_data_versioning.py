"""Delta-only QM reference-data storage and resolution contracts."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from ichor.hpc.active_learning.daemon.input_staging import (
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
    enrich_with_point_allocation,
    write_seed_provenance,
)
from ichor.hpc.active_learning.versioning.reference_data import (
    REFERENCE_DATA_VERSION_FILENAME,
    ReferenceDataError,
    ReferenceDataVersioning,
    reference_data_cache_path,
)
from ichor.hpc.active_learning.layout import (
    reject_legacy_campaign_layout,
    reject_legacy_training_layout,
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
        )
        results.append(
            {
                "candidate_id": str(attempt["candidate_id"]),
                "accepted": True,
                "pointdir": str(pointdir),
            }
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
        iteration=0,
        first_frame_id=10,
    )
    second, second_names, second_created = commit_reference_data_delta(
        campaign,
        reference_data_version=1,
        context="active",
        iteration=0,
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


def test_reference_data_cache_is_rebuilt_from_authoritative_manifests(tmp_path):
    campaign = tmp_path / "campaign"
    *_, second, _, _ = _commit_two_versions(campaign)
    cache = reference_data_cache_path(campaign)
    cache.write_text('{"untrusted": true}\n', encoding="utf-8")

    rebuilt = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA").resolve(1)
    cached = json.loads(cache.read_text(encoding="utf-8"))

    assert rebuilt.cumulative_view_sha256 == second.cumulative_view_sha256
    assert cached["cumulative_view_sha256"] == second.cumulative_view_sha256
    assert [record["pointdir_name"] for record in cached["entries"]] == [
        "POINT_000000.pointdir",
        "POINT_000001.pointdir",
        "POINT_000002.pointdir",
        "POINT_000003.pointdir",
    ]


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
