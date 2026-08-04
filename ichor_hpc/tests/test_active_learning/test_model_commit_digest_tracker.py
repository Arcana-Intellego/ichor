import hashlib
import os

from ichor.hpc.active_learning.daemon.live_executor import _StableDigestTracker


def test_digest_tracker_preserves_digest_across_unchanged_directory_rename(
    tmp_path,
):
    staged = tmp_path / "iteration-staging"
    staged.mkdir()
    source = staged / "model.bin"
    source.write_bytes(b"immutable-model")
    tracker = _StableDigestTracker()
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    assert tracker.digest(source) == expected
    captured = tracker.capture_rename_tree(staged)
    committed = tmp_path / "iteration-000001"
    staged.rename(committed)
    tracker.bind_renamed_tree(captured, committed)

    assert tracker.digest(committed / "model.bin") == expected


def test_digest_tracker_never_reuses_same_size_mutation_with_restored_mtime(
    tmp_path,
):
    staged = tmp_path / "iteration-staging"
    staged.mkdir()
    source = staged / "model.bin"
    source.write_bytes(b"before")
    before_stat = source.stat()
    tracker = _StableDigestTracker()
    original = tracker.digest(source)
    captured = tracker.capture_rename_tree(staged)

    source.write_bytes(b"after!")
    os.utime(
        source,
        ns=(int(before_stat.st_atime_ns), int(before_stat.st_mtime_ns)),
    )
    committed = tmp_path / "iteration-000001"
    staged.rename(committed)
    try:
        tracker.bind_renamed_tree(captured, committed)
    except ValueError:
        return

    observed = tracker.digest(committed / "model.bin")
    assert observed != original
    assert observed == hashlib.sha256(b"after!").hexdigest()
