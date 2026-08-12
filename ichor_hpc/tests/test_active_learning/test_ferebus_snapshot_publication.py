import os
from pathlib import Path

import pytest

import ichor.hpc.active_learning.daemon.ferebus_snapshot_publication as publication
from ichor.hpc.active_learning.daemon.ferebus_snapshot_publication import (
    FerebusSnapshotPublicationError,
    assert_committed_ferebus_snapshot,
    assert_ferebus_snapshot_unchanged,
    capture_ferebus_snapshot_publication,
    commit_prevalidated_ferebus_snapshot,
    publish_ferebus_snapshot_manifest,
    synchronise_ferebus_snapshot_directories,
)
from ichor.hpc.active_learning.daemon.live_executor import _StableDigestTracker
from ichor.hpc.active_learning.versioning.manifest import read_manifest, sha256_file


def _seed_snapshot(tmp_path):
    staging = tmp_path / "TRAINED_MODELS" / "iteration-000003.staging"
    (staging / "iqa" / "O1" / "datasets").mkdir(parents=True)
    files = {
        "FEREBUS_TASK_ARTEFACTS.json": b'{"schema_version": 1}\n',
        "iqa/O1/WATER_iqa_O1.model": b"model-payload\n",
        "iqa/O1/ferebus.config": b"config-payload\n",
        "iqa/O1/datasets/train.csv": b"a,b\n1,2\n",
    }
    tracker = _StableDigestTracker()
    for relative, payload in files.items():
        path = staging.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        tracker.remember(path, sha256_file(path))
    context = capture_ferebus_snapshot_publication(
        staging,
        digest_for_identity=tracker.digest_for_captured_identity,
    )
    return staging, tracker, context


def test_prevalidated_snapshot_commit_preserves_exact_manifest(tmp_path):
    staging, unused_tracker, context = _seed_snapshot(tmp_path)
    publish_ferebus_snapshot_manifest(context)
    assert_ferebus_snapshot_unchanged(context)
    synchronise_ferebus_snapshot_directories(context)

    target = staging.with_name("iteration-000003")
    commit_prevalidated_ferebus_snapshot(context, target)
    reusable = assert_committed_ferebus_snapshot(context, target)

    assert read_manifest(target) == dict(context.manifest)
    assert not staging.exists()
    assert reusable is (os.name != "nt")


def test_prevalidated_snapshot_rejects_same_size_replacement(tmp_path):
    staging, unused_tracker, context = _seed_snapshot(tmp_path)
    publish_ferebus_snapshot_manifest(context)
    model = staging / "iqa/O1/WATER_iqa_O1.model"
    before = model.stat()
    model.write_bytes(b"changed-data!\n")
    os.utime(model, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(
        FerebusSnapshotPublicationError,
        match="file (inventory|content) changed",
    ):
        assert_ferebus_snapshot_unchanged(context)


def test_prevalidated_snapshot_rejects_inventory_change(tmp_path):
    staging, unused_tracker, context = _seed_snapshot(tmp_path)
    publish_ferebus_snapshot_manifest(context)
    (staging / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(
        FerebusSnapshotPublicationError,
        match="file inventory changed",
    ):
        commit_prevalidated_ferebus_snapshot(
            context,
            staging.with_name("iteration-000003"),
        )


def test_stable_posix_publication_does_not_rehash_payloads(
    tmp_path,
    monkeypatch,
):
    staging, unused_tracker, context = _seed_snapshot(tmp_path)
    if any(entry.stat_identity[1] == 0 for entry in context.files):
        pytest.skip("filesystem has no stable inode identity")
    publish_ferebus_snapshot_manifest(context)
    monkeypatch.setattr(publication.os, "name", "posix")
    monkeypatch.setattr(
        publication,
        "sha256_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stable publication rehashed a payload")
        ),
    )

    assert_ferebus_snapshot_unchanged(context)
    target = staging.with_name("iteration-000003")
    commit_prevalidated_ferebus_snapshot(context, target)
    assert assert_committed_ferebus_snapshot(context, target) is True


def test_snapshot_capture_rejects_symlink(tmp_path):
    staging, tracker, unused_context = _seed_snapshot(tmp_path)
    link = staging / "unsafe-link"
    try:
        link.symlink_to(staging / "FEREBUS_TASK_ARTEFACTS.json")
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(FerebusSnapshotPublicationError, match="symlink"):
        capture_ferebus_snapshot_publication(
            staging,
            digest_for_identity=tracker.digest_for_captured_identity,
        )
