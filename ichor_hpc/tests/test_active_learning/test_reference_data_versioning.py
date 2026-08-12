"""Delta-only QM reference-data storage and resolution contracts."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
from ichor.hpc.active_learning.daemon.input_staging import (
    accepted_allocation_pointdirs,
    commit_reference_data_delta,
)
from ichor.hpc.active_learning.daemon.reference_commit import (
    classify_reference_commit,
    prepare_reference_data_delta,
)
from ichor.hpc.active_learning.daemon.staging_retirement import (
    StagingRetirementError,
    classify_completed_staging_buckets,
    retire_completed_staging_buckets,
    retired_staging_root,
)
from ichor.hpc.active_learning.daemon.state import fresh_campaign_state
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


def test_reference_index_resolution_does_not_walk_pointdir_payloads(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import quantum_acceptance_receipts
    from ichor.hpc.active_learning.versioning import reference_data

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
    monkeypatch.setattr(
        quantum_acceptance_receipts,
        "read_quantum_acceptance_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("index resolution walked acceptance payloads")
        ),
    )
    monkeypatch.setattr(
        reference_data,
        "_validate_provenance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("index resolution read pointdir provenance")
        ),
    )

    view = ReferenceDataVersioning(
        campaign / "QM_REFERENCE_DATA"
    ).resolve(0, verification="index")

    assert len(view.entries) == 2


def test_reference_commit_moves_without_copying_or_payload_rehash(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import reference_commit as commit_mod

    campaign = tmp_path / "campaign"
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    sources = sorted(
        (campaign / ".DATA" / "STAGING" / "initial").glob("*.pointdir")
    )
    hashed_paths = []
    resolve_modes = []
    quality_reads = 0
    progress_events = []
    real_sha256_file = commit_mod.sha256_file
    real_resolve = ReferenceDataVersioning.resolve
    real_read_quality = commit_mod.read_quantum_quality_manifest

    def track_sha256(path):
        hashed_paths.append(Path(path))
        return real_sha256_file(path)

    def track_resolve(self, version, *args, **kwargs):
        resolve_modes.append(str(kwargs.get("verification", "metadata")))
        return real_resolve(self, version, *args, **kwargs)

    def track_quality(*args, **kwargs):
        nonlocal quality_reads
        quality_reads += 1
        return real_read_quality(*args, **kwargs)

    monkeypatch.setattr(commit_mod, "sha256_file", track_sha256)
    monkeypatch.setattr(commit_mod, "read_quantum_quality_manifest", track_quality)
    monkeypatch.setattr(ReferenceDataVersioning, "resolve", track_resolve)
    monkeypatch.setattr(
        shutil,
        "copytree",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reference commit must not call copytree")
        ),
    )

    view, names, created = commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
        progress_callback=lambda event, payload: progress_events.append(
            (str(event), dict(payload))
        ),
    )

    assert created is True
    assert len(view.entries) == len(sources) == len(names)
    assert all(not source.exists() for source in sources)
    assert all(
        (campaign / "QM_REFERENCE_DATA" / "iteration-000000" / name).is_dir()
        for name in names
    )
    assert not {
        path.suffix.lower()
        for path in hashed_paths
    } & {".wfn", ".int", ".gau", ".gjf"}
    assert "deep" not in resolve_modes
    assert quality_reads == 1
    shard_progress = [
        payload
        for event, payload in progress_events
        if event == "reference_commit_shard_progress"
    ]
    assert shard_progress[-1]["processed_points"] == len(sources)
    assert shard_progress[-1]["total_points"] == len(sources)


def test_reference_commit_can_be_prepared_without_moving_pointdirs(tmp_path):
    campaign = tmp_path / "campaign"
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    staging = campaign / ".DATA" / "STAGING" / "initial"
    sources = sorted(staging.glob("*.pointdir"))

    ledger_path = prepare_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )

    transaction = classify_reference_commit(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    assert ledger_path.is_file()
    assert transaction["state"] == "prepared"
    assert all(path.is_dir() for path in sources)
    assert not list(
        (campaign / "QM_REFERENCE_DATA" / "iteration-000000.staging").glob(
            "*.pointdir"
        )
    )


def test_aimall_row_sidecar_uses_one_matrix_file_per_point(tmp_path):
    from ichor.hpc.active_learning.daemon.ferebus_row_cache import (
        FEREBUS_ROW_SHARD,
        FEREBUS_ROW_SHARD_ARRAY,
    )

    campaign = tmp_path / "campaign"
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )

    for pointdir in sorted(
        (campaign / ".DATA" / "STAGING" / "initial").glob("*.pointdir")
    ):
        task = json.loads((pointdir / "AIMALL_TASK.json").read_text(encoding="utf-8"))
        shard = campaign / str(task["ferebus_row_shard"]["directory"])
        assert sorted(path.name for path in shard.iterdir()) == sorted(
            [FEREBUS_ROW_SHARD, FEREBUS_ROW_SHARD_ARRAY]
        )


@pytest.mark.parametrize(
    ("missing_shards", "expected_reused"),
    ((1, 1), (2, 0)),
)
def test_reference_commit_repairs_missing_row_shards_serially(
    tmp_path,
    missing_shards,
    expected_reused,
):
    from ichor.hpc.active_learning.daemon import reference_commit as commit_mod
    from ichor.hpc.active_learning.daemon.ferebus_row_cache import (
        read_feature_contract,
        read_version_row_cache,
    )

    campaign = tmp_path / "campaign"
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    staging = campaign / ".DATA" / "STAGING" / "initial"
    for pointdir in sorted(staging.glob("*.pointdir"))[:missing_shards]:
        task = json.loads(
            (pointdir / "AIMALL_TASK.json").read_text(encoding="utf-8")
        )
        shard = campaign / str(task["ferebus_row_shard"]["directory"])
        shutil.rmtree(shard)

    commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )

    transaction = commit_mod.classify_reference_commit(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    assert transaction["ledger"]["shards_repaired"] == missing_shards
    assert transaction["ledger"]["shards_reused"] == expected_reused
    contract = read_feature_contract(campaign)
    cache, arrays = read_version_row_cache(
        campaign,
        str(contract["contract_sha256"]),
        0,
    )
    assert cache["n_rows"] == 2
    assert {array.shape[0] for array in arrays.values()} == {2}


def test_reference_commit_resumes_after_interrupted_point_move(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import reference_commit as commit_mod

    campaign = tmp_path / "campaign"
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    real_replace = commit_mod.os.replace
    point_moves = 0

    def interrupt_second_point(source, destination):
        nonlocal point_moves
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name.endswith(".pointdir")
            and destination_path.name.endswith(".pointdir")
        ):
            point_moves += 1
            if point_moves == 2:
                raise OSError("injected point-move interruption")
        return real_replace(source, destination)

    monkeypatch.setattr(commit_mod.os, "replace", interrupt_second_point)
    with pytest.raises(OSError, match="injected point-move interruption"):
        commit_reference_data_delta(
            campaign,
            reference_data_version=0,
            context="bootstrap",
            iteration=0,
        )

    transaction = commit_mod.classify_reference_commit(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    assert transaction["state"] == "partially_moved"
    assert transaction["ledger"]["moved_points"] == 1
    from ichor.hpc.active_learning.daemon.config_lock import (
        reference_data_staging_can_archive_for_reconcile,
    )
    from ichor.hpc.active_learning.daemon.state import fresh_campaign_state

    can_archive, reason = reference_data_staging_can_archive_for_reconcile(
        campaign,
        fresh_campaign_state(),
    )
    assert can_archive is False
    assert "resumable reference-commit transaction" in reason

    monkeypatch.setattr(commit_mod.os, "replace", real_replace)
    view, names, created = commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )

    assert created is True
    assert len(view.entries) == len(names) == 2
    assert commit_mod.classify_reference_commit(
        campaign,
        context="bootstrap",
        iteration=0,
    )["state"] == "complete"


def test_schema_two_writable_receipts_resume_zero_move_transaction(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import reference_commit as commit_mod
    from ichor_hpc.tests import quantum_test_support

    campaign = tmp_path / "campaign"
    write_v3_receipt = quantum_test_support.write_quantum_acceptance_receipt

    def write_v2_receipt(*args, **kwargs):
        receipt_path = write_v3_receipt(*args, **kwargs)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["schema_version"] = 2
        payload["sealed_at_iso"] = payload.pop("accepted_at_iso")
        payload.pop("integrity_policy")
        receipt_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return receipt_path

    monkeypatch.setattr(
        quantum_test_support,
        "write_quantum_acceptance_receipt",
        write_v2_receipt,
    )
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    pointdirs = sorted(
        (campaign / ".DATA" / "STAGING" / "initial").glob("*.pointdir")
    )
    for pointdir in pointdirs:
        for path in [*pointdir.rglob("*"), pointdir]:
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    real_replace = commit_mod.os.replace

    def deny_first_point_move(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name.endswith(".pointdir")
            and destination_path.name.endswith(".pointdir")
        ):
            raise PermissionError("simulated sealed-directory move failure")
        return real_replace(source, destination)

    monkeypatch.setattr(commit_mod.os, "replace", deny_first_point_move)
    with pytest.raises(PermissionError, match="sealed-directory move failure"):
        commit_reference_data_delta(
            campaign,
            reference_data_version=0,
            context="bootstrap",
            iteration=0,
        )

    transaction = commit_mod.classify_reference_commit(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    assert transaction["state"] == "prepared"
    assert transaction["ledger"]["status"] == "moving"
    assert transaction["ledger"]["moved_points"] == 0

    monkeypatch.setattr(commit_mod.os, "replace", real_replace)
    for pointdir in pointdirs:
        for path in [pointdir, *pointdir.rglob("*")]:
            mode = stat.S_IMODE(path.stat().st_mode)
            if path.is_dir():
                path.chmod(mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            else:
                path.chmod(mode | stat.S_IRUSR | stat.S_IWUSR)

    view, names, created = commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    assert created is True
    assert len(view.entries) == len(names) == len(pointdirs)
    assert commit_mod.classify_reference_commit(
        campaign,
        context="bootstrap",
        iteration=0,
    )["state"] == "complete"


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
    assert {record["pointdir"] for record in accepted} == {
        binding["source_pointdir"]
        for binding in evidence[0]["pointdir_bindings"]
    }
    assert all(
        binding.get("committed_pointdir") and binding.get("candidate_id")
        for binding in evidence[0]["pointdir_bindings"]
    )


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
    version_manifest = json.loads(
        (version_dir / REFERENCE_DATA_VERSION_FILENAME).read_text(encoding="utf-8")
    )
    bindings = {
        record["pointdir_name"]: record
        for record in version_manifest["added_pointdirs"]
    }
    for pointdir in sorted(version_dir.glob("POINT_*.pointdir")):
        receipt = json.loads(
            (pointdir / QUANTUM_ACCEPTANCE_RECEIPT).read_text(encoding="utf-8")
        )
        assert receipt["source_pointdir"] == bindings[pointdir.name]["source_pointdir"]
        assert "committed_pointdir" not in receipt

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
            expected_source_pointdir=entry.source_pointdir,
            validate_quality=False,
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
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tampered"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReferenceDataError, match="receipt identity mismatch"):
        ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA").resolve(
            1,
            verification="metadata",
        )


def test_reference_data_reader_rejects_schema_two_manifest(tmp_path):
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
    manifest = (
        campaign
        / "QM_REFERENCE_DATA"
        / "iteration-000000"
        / REFERENCE_DATA_VERSION_FILENAME
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ReferenceDataError, match="unsupported reference-data"):
        ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA").resolve(
            0,
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


def test_metadata_snapshot_skips_payload_hashing_but_deep_detects_tamper(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import artifact_snapshot as snapshot_mod

    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)
    target = next(
        (campaign / "QM_REFERENCE_DATA").glob(
            "iteration-*/POINT_*.pointdir/input.gjf"
        )
    )
    original = target.read_bytes()
    replacement = (b"!" if original[:1] != b"!" else b"?") + original[1:]
    assert len(replacement) == len(original)
    target.write_bytes(replacement)

    hashed = []
    real_sha256 = snapshot_mod.sha256_file

    def record_hash(path, **kwargs):
        hashed.append(Path(path).resolve())
        return real_sha256(path, **kwargs)

    monkeypatch.setattr(snapshot_mod, "sha256_file", record_hash)
    authority = snapshot_mod.build_committed_artifact_snapshot(
        campaign,
        verification_level="authority",
    )
    assert authority.valid_reference_data_versions == (0, 1)
    assert authority.payload_files_hashed == 0
    assert authority.verification_payload()["recursive_scan"] is False
    assert target.resolve() not in hashed

    metadata = snapshot_mod.build_committed_artifact_snapshot(
        campaign,
        verification_level="metadata",
    )
    assert metadata.valid_reference_data_versions == (0, 1)
    assert metadata.payload_files_hashed == 0
    assert target.resolve() not in hashed

    deep = snapshot_mod.build_committed_artifact_snapshot(
        campaign,
        verification_level="deep",
    )
    assert deep.valid_reference_data_versions == ()
    assert deep.reference_errors
    assert target.resolve() in hashed


def test_authority_snapshot_never_uses_recursive_filesystem_walkers(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import artifact_snapshot as snapshot_mod

    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)

    def refuse_recursive_walk(*_args, **_kwargs):
        raise AssertionError("authority verification attempted a recursive walk")

    monkeypatch.setattr(Path, "rglob", refuse_recursive_walk)
    monkeypatch.setattr(os, "walk", refuse_recursive_walk)

    snapshot = snapshot_mod.build_committed_artifact_snapshot(
        campaign,
        verification_level="authority",
    )

    assert snapshot.valid_reference_data_versions == (0, 1)
    assert snapshot.payload_files_hashed == 0
    assert snapshot.verification_payload()["recursive_scan"] is False


def test_deep_snapshot_hashes_each_scientific_payload_once(tmp_path, monkeypatch):
    from collections import Counter

    from ichor.hpc.active_learning.daemon import artifact_snapshot as snapshot_mod

    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)
    target = next(
        (campaign / "QM_REFERENCE_DATA").glob(
            "iteration-*/POINT_*.pointdir/input.gjf"
        )
    ).resolve()
    calls = Counter()
    real_sha256 = snapshot_mod.sha256_file

    def count_hash(path, **kwargs):
        calls[Path(path).resolve()] += 1
        return real_sha256(path, **kwargs)

    monkeypatch.setattr(snapshot_mod, "sha256_file", count_hash)
    snapshot = snapshot_mod.build_committed_artifact_snapshot(
        campaign,
        verification_level="deep",
    )

    assert snapshot.valid_reference_data_versions == (0, 1)
    assert snapshot.payload_files_hashed > 0
    assert calls[target] == 1


def test_snapshot_resolves_reference_chain_once_and_rechecks_anchors(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon.artifact_snapshot import (
        ArtefactSnapshotError,
        build_committed_artifact_snapshot,
    )

    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)
    calls = []
    original = ReferenceDataVersioning.resolve_chain

    def counted(self, version, **kwargs):
        calls.append(int(version))
        return original(self, version, **kwargs)

    monkeypatch.setattr(ReferenceDataVersioning, "resolve_chain", counted)
    snapshot = build_committed_artifact_snapshot(
        campaign,
        verification_level="metadata",
    )
    assert calls == [1]

    anchor = (
        campaign
        / "QM_REFERENCE_DATA"
        / "iteration-000000"
        / REFERENCE_DATA_VERSION_FILENAME
    )
    anchor.write_text(anchor.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtefactSnapshotError, match="changed"):
        snapshot.assert_anchors_unchanged(campaign)


def test_snapshot_rejects_completion_receipt_added_after_inspection(tmp_path):
    from ichor.hpc.active_learning.daemon.artifact_snapshot import (
        ArtefactSnapshotError,
        build_committed_artifact_snapshot,
    )

    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)
    snapshot = build_committed_artifact_snapshot(
        campaign,
        verification_level="authority",
    )

    receipts = campaign / ".DATA" / "ACTIVE_LEARNING" / "phase_completions"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / ("a" * 64 + ".json")).write_text(
        "{}\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ArtefactSnapshotError, match="inventory changed"):
        snapshot.assert_anchors_unchanged(campaign)


def test_snapshot_reference_head_guard_allows_submission_control_publication(
    tmp_path,
):
    from ichor.hpc.active_learning.daemon.artifact_snapshot import (
        ArtefactSnapshotError,
        build_committed_artifact_snapshot,
    )

    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)
    snapshot = build_committed_artifact_snapshot(
        campaign,
        verification_level="authority",
    )

    intents = campaign / ".DATA" / "ACTIVE_LEARNING" / "submission_intents"
    intents.mkdir(parents=True, exist_ok=True)
    (intents / "PHASE_B_DIVERSITY-000002.json").write_text(
        "{}\n",
        encoding="utf-8",
        newline="\n",
    )

    snapshot.assert_reference_head_unchanged(
        campaign,
        reference_version=1,
        purpose="Phase B submission",
    )

    head = (
        campaign
        / "QM_REFERENCE_DATA"
        / "iteration-000001"
        / REFERENCE_DATA_VERSION_FILENAME
    )
    head.write_text(head.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtefactSnapshotError, match="head changed"):
        snapshot.assert_reference_head_unchanged(
            campaign,
            reference_version=1,
            purpose="Phase B submission",
        )


def test_snapshot_stops_at_first_invalid_reference_without_rescanning_prefix(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon.artifact_snapshot import (
        build_committed_artifact_snapshot,
    )

    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)
    second_manifest = (
        campaign
        / "QM_REFERENCE_DATA"
        / "iteration-000001"
        / REFERENCE_DATA_VERSION_FILENAME
    )
    payload = json.loads(second_manifest.read_text(encoding="utf-8"))
    payload["parent_manifest_sha256"] = "0" * 64
    second_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ReferenceDataVersioning,
        "resolve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot rescanned a cumulative reference prefix")
        ),
    )

    snapshot = build_committed_artifact_snapshot(
        campaign,
        verification_level="metadata",
    )

    assert snapshot.valid_reference_data_versions == (0,)
    assert 1 in snapshot.reference_errors


def test_snapshot_reads_each_version_manifest_once_in_40_version_chain(
    tmp_path,
    monkeypatch,
):
    from collections import Counter

    from ichor.hpc.active_learning.daemon.artifact_snapshot import (
        build_committed_artifact_snapshot,
    )
    from ichor.hpc.active_learning.versioning import reference_data as reference_mod

    campaign = tmp_path / "campaign"
    for version in range(40):
        context = "bootstrap" if version == 0 else "active"
        _complete_allocation(
            campaign,
            context=context,
            iteration=version,
            first_frame_id=version * 10,
        )
        commit_reference_data_delta(
            campaign,
            reference_data_version=version,
            context=context,
            iteration=version,
        )

    reads = Counter()
    original = reference_mod._read_json_object

    def counted(path, label):
        source = Path(path)
        if source.name == REFERENCE_DATA_VERSION_FILENAME:
            reads[source.resolve()] += 1
        return original(path, label)

    monkeypatch.setattr(reference_mod, "_read_json_object", counted)
    snapshot = build_committed_artifact_snapshot(
        campaign,
        verification_level="metadata",
    )

    assert snapshot.valid_reference_data_versions == tuple(range(40))
    assert len(reads) == 40
    assert set(reads.values()) == {1}


def test_committed_reference_pointdirs_remain_owner_writable(tmp_path):
    campaign = tmp_path / "campaign"
    _commit_two_versions(campaign)
    pointdir = (
        campaign
        / "QM_REFERENCE_DATA"
        / "iteration-000001"
        / "POINT_000002.pointdir"
    )
    file_path = pointdir / "input.gjf"

    assert os.access(pointdir, os.W_OK)
    assert os.access(file_path, os.W_OK)


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


def test_reference_commit_retires_duplicate_only_staging(tmp_path):
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

    assert not (campaign / ".DATA" / "STAGING" / "initial").exists()
    retired = retired_staging_root(campaign)
    assert not retired.exists() or not list(retired.iterdir())


def test_completed_staging_with_rejected_payload_is_preserved(tmp_path):
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
    bucket = campaign / ".DATA" / "STAGING" / "initial"
    rejected = bucket / "POINT_9999.pointdir"
    rejected.mkdir(parents=True)
    failure = rejected / "failure.txt"
    failure.write_text("diagnostic\n", encoding="utf-8")
    unexpected = bucket / "unexpected.bin"
    unexpected.write_bytes(b"diagnostic payload")
    failure.chmod(0o400)
    unexpected.chmod(0o400)
    rejected.chmod(0o500)
    bucket.chmod(0o500)

    classification = classify_completed_staging_buckets(campaign)

    assert len(classification["eligible"]) == 1
    assert classification["eligible"][0]["action"] == "preserve"
    result = retire_completed_staging_buckets(
        campaign,
        classification=classification,
    )
    assert result["n_preserved"] == 1
    assert not bucket.exists()
    preserved = Path(result["preserved"][0])
    assert (preserved / "POINT_9999.pointdir" / "failure.txt").read_text(
        encoding="utf-8"
    ) == "diagnostic\n"
    assert (preserved / "unexpected.bin").read_bytes() == b"diagnostic payload"
    assert stat.S_IMODE(preserved.stat().st_mode) & stat.S_IWUSR
    assert stat.S_IMODE((preserved / "POINT_9999.pointdir").stat().st_mode) & stat.S_IWUSR
    assert stat.S_IMODE(
        (preserved / "POINT_9999.pointdir" / "failure.txt").stat().st_mode
    ) & stat.S_IWUSR
    assert stat.S_IMODE((preserved / "unexpected.bin").stat().st_mode) & stat.S_IWUSR


def test_changed_quality_residue_is_not_deleted(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.daemon import staging_retirement as retirement_mod

    campaign = tmp_path / "campaign"
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    monkeypatch.setattr(
        retirement_mod,
        "retire_completed_staging_buckets",
        lambda *_args, **_kwargs: {
            "n_retired": 0,
            "n_deleted": 0,
            "n_preserved": 0,
            "warnings": [],
        },
    )
    commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    quality = campaign / ".DATA" / "STAGING" / "initial" / "quantum_quality.json"
    payload = json.loads(quality.read_text(encoding="utf-8"))
    quality.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")

    classification = classify_completed_staging_buckets(campaign)

    assert classification["eligible"] == []
    assert len(classification["ambiguous"]) == 1
    assert "differs from committed authority" in classification["ambiguous"][0]["reason"]


def test_malformed_known_staging_metadata_remains_ambiguous(tmp_path):
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
    bucket = campaign / ".DATA" / "STAGING" / "initial"
    bucket.mkdir(parents=True)
    (bucket / "accepted_pointdirs.json").write_text("{}\n", encoding="utf-8")

    classification = classify_completed_staging_buckets(campaign)

    assert classification["eligible"] == []
    assert len(classification["ambiguous"]) == 1
    with pytest.raises(StagingRetirementError, match="ambiguous"):
        retire_completed_staging_buckets(
            campaign,
            classification=classification,
        )
    assert bucket.is_dir()


def test_retirement_deletion_interruption_leaves_retryable_tombstone(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import staging_retirement as retirement_mod

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
    bucket = campaign / ".DATA" / "STAGING" / "initial"
    bucket.mkdir(parents=True)
    classification = classify_completed_staging_buckets(campaign)
    original_rmtree = retirement_mod.shutil.rmtree

    def fail_delete(_path):
        raise OSError("injected deletion failure")

    monkeypatch.setattr(retirement_mod.shutil, "rmtree", fail_delete)
    result = retire_completed_staging_buckets(
        campaign,
        classification=classification,
    )

    assert result["n_deleted"] == 1
    assert result["warnings"]
    assert not bucket.exists()
    tombstones = list(retired_staging_root(campaign).glob(".deleting-*"))
    assert len(tombstones) == 1
    interrupted = classify_completed_staging_buckets(campaign)
    assert interrupted["pending_tombstones"] == [str(tombstones[0])]

    monkeypatch.setattr(retirement_mod.shutil, "rmtree", original_rmtree)
    replay = retire_completed_staging_buckets(campaign)
    assert replay["warnings"] == []
    assert not tombstones[0].exists()


def test_retirement_rename_failure_leaves_source_untouched(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.daemon import staging_retirement as retirement_mod

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
    bucket = campaign / ".DATA" / "STAGING" / "initial"
    bucket.mkdir(parents=True)
    classification = classify_completed_staging_buckets(campaign)
    original_replace = retirement_mod.os.replace

    def fail_rename(_source, _destination):
        raise OSError("injected rename failure")

    monkeypatch.setattr(retirement_mod.os, "replace", fail_rename)
    with pytest.raises(OSError, match="injected rename failure"):
        retire_completed_staging_buckets(
            campaign,
            classification=classification,
        )
    assert bucket.is_dir()
    assert not Path(classification["eligible"][0]["destination"]).exists()

    monkeypatch.setattr(retirement_mod.os, "replace", original_replace)
    result = retire_completed_staging_buckets(campaign)
    assert result["n_deleted"] == 1
    assert not bucket.exists()


def test_retirement_refuses_existing_source_and_destination(tmp_path):
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
    bucket = campaign / ".DATA" / "STAGING" / "initial"
    bucket.mkdir(parents=True)
    classification = classify_completed_staging_buckets(campaign)
    destination = Path(classification["eligible"][0]["destination"])
    destination.mkdir(parents=True)

    with pytest.raises(
        StagingRetirementError,
        match="source and retirement destination both exist",
    ):
        retire_completed_staging_buckets(
            campaign,
            classification=classification,
        )

    assert bucket.is_dir()
    assert destination.is_dir()


def test_reconcile_style_retirement_ignores_incomplete_future_bucket(tmp_path):
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
    completed = campaign / ".DATA" / "STAGING" / "initial"
    completed.mkdir(parents=True)
    active = campaign / ".DATA" / "STAGING" / "iter_1"
    (active / "POINT_0000.pointdir").mkdir(parents=True)
    classification = classify_completed_staging_buckets(campaign)

    assert [record["iteration"] for record in classification["eligible"]] == [0]
    assert [record["iteration"] for record in classification["ambiguous"]] == [1]
    result = retire_completed_staging_buckets(
        campaign,
        classification={
            "eligible": classification["eligible"],
            "ambiguous": [],
            "noncanonical": [],
        },
    )

    assert result["n_deleted"] == 1
    assert not completed.exists()
    assert active.is_dir()


def test_stop_check_catches_up_completed_staging(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.versioning import sampling_iterations

    campaign = tmp_path / "campaign"
    for context, iteration, frame_id in (
        ("bootstrap", 0, 0),
        ("active", 1, 10),
    ):
        _complete_allocation(
            campaign,
            context=context,
            iteration=iteration,
            first_frame_id=frame_id,
        )
        commit_reference_data_delta(
            campaign,
            reference_data_version=iteration,
            context=context,
            iteration=iteration,
        )
    residue = campaign / ".DATA" / "STAGING" / "iter_1"
    residue.mkdir(parents=True)
    monkeypatch.setattr(
        sampling_iterations,
        "finalise_active_iteration",
        lambda *_args, **_kwargs: campaign / "ITERATION_MANIFEST.json",
    )
    state = fresh_campaign_state(max_iterations=3)
    state.campaign_uid = "campaign-uid"
    state.iteration = 1
    state.reference_data_version = 1
    state.models_version = 1

    DryRunPhaseExecutor(campaign, CampaignConfig())._inline_stop_check(state)

    assert not residue.exists()
