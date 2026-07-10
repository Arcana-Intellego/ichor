"""Tests for the generic atomic versioned-directory protocol."""
import os
import shutil
from pathlib import Path

import pytest

from ichor.hpc.active_learning.versioning.manifest import (
    MANIFEST_FILENAME,
    ManifestMismatchError,
)
from ichor.hpc.active_learning.versioning.versioned_directory import (
    CURRENT_LINK_NAME,
    STAGING_SUFFIX,
    VersionedDirectory,
)
from ichor.hpc.active_learning.versioning.trained_models import (
    TrainedModelError,
    TrainedModelVersioning,
)


def _new_setup(tmp_path: Path) -> VersionedDirectory:
    parent = tmp_path / "QM_REFERENCE_DATA"
    parent.mkdir()
    return VersionedDirectory(parent)


def test_iteration_name_pads_width():
    v = VersionedDirectory(Path("."), prefix="iter", name_width=5)
    assert v.iteration_name(7) == "iter-00007"
    assert v.staging_name(7) == "iter-00007.staging"


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (9, "iteration-000009"),
        (9999, "iteration-009999"),
        (10000, "iteration-010000"),
        (999999, "iteration-999999"),
        (1000000, "iteration-1000000"),
    ],
)
def test_default_iteration_name_uses_six_digit_minimum(version, expected):
    assert VersionedDirectory(Path(".")).iteration_name(version) == expected


def test_iteration_name_rejects_negative_and_boolean_versions():
    versioning = VersionedDirectory(Path("."))
    with pytest.raises(ValueError, match="non-negative integer"):
        versioning.iteration_name(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        versioning.iteration_name(True)
    with pytest.raises(ValueError, match="non-negative integer"):
        versioning.iteration_name(1.5)


def test_stage_creates_empty_initial_iteration(tmp_path):
    v = _new_setup(tmp_path)
    s = v.stage(source_version=None, target_version=0)
    assert s.exists()
    assert s.name.endswith(STAGING_SUFFIX)
    assert list(s.iterdir()) == []


def test_stage_copies_from_source_iteration(tmp_path):
    v = _new_setup(tmp_path)
    src = v.iteration_path(0)
    src.mkdir()
    (src / "a.txt").write_text("from source")
    s = v.stage(source_version=0, target_version=1)
    assert (s / "a.txt").read_text() == "from source"


def test_stage_raises_when_staging_already_exists(tmp_path):
    v = _new_setup(tmp_path)
    v.stage(None, 0)
    with pytest.raises(FileExistsError):
        v.stage(None, 0)


def test_stage_raises_when_source_missing(tmp_path):
    v = _new_setup(tmp_path)
    with pytest.raises(FileNotFoundError):
        v.stage(source_version=99, target_version=1)


def test_commit_writes_manifest_and_renames(tmp_path):
    v = _new_setup(tmp_path)
    s = v.stage(None, 0)
    (s / "a.txt").write_text("hi")
    manifest = v.commit(0)
    assert "a.txt" in manifest
    assert v.iteration_path(0).is_dir()
    assert not v.staging_path(0).exists()
    assert (v.iteration_path(0) / MANIFEST_FILENAME).exists()


def test_commit_raises_when_staging_missing(tmp_path):
    v = _new_setup(tmp_path)
    with pytest.raises(FileNotFoundError):
        v.commit(0)


def test_commit_is_idempotent_when_target_already_committed(tmp_path):
    v = _new_setup(tmp_path)
    v.stage(None, 0); first = v.commit(0)
    # a re-run that finds the target already committed should be a no-op that
    # returns the stored manifest, not a crash. simulate a leftover staging dir
    # from the interrupted run and confirm it gets cleaned up.
    shutil.copytree(str(v.iteration_path(0)), str(v.staging_path(0)))
    again = v.commit(0)
    assert again == first
    assert not v.staging_path(0).exists()


def test_verify_committed_passes_on_intact(tmp_path):
    v = _new_setup(tmp_path)
    s = v.stage(None, 0)
    (s / "a.txt").write_text("hi")
    v.commit(0)
    v.verify_committed(0)  # should not raise


def test_verify_committed_detects_post_commit_tamper(tmp_path):
    v = _new_setup(tmp_path)
    s = v.stage(None, 0)
    (s / "a.txt").write_text("hi")
    v.commit(0)
    (v.iteration_path(0) / "a.txt").write_text("EVIL")
    with pytest.raises(ManifestMismatchError):
        v.verify_committed(0)


def test_list_committed_versions_orders_ascending(tmp_path):
    v = _new_setup(tmp_path)
    for i in (3, 0, 7, 1):
        s = v.stage(None, i)
        (s / "x.txt").write_text(str(i))
        v.commit(i)
    assert v.list_committed_versions() == [0, 1, 3, 7]


def test_list_committed_versions_excludes_staging(tmp_path):
    v = _new_setup(tmp_path)
    v.stage(None, 0); v.commit(0)
    v.stage(None, 1)   # left as staging
    assert v.list_committed_versions() == [0]


def test_list_committed_versions_rejects_noncanonical_padding(tmp_path):
    versioning = _new_setup(tmp_path)
    (versioning.parent / "iteration-0000").mkdir()
    with pytest.raises(ValueError, match="non-canonical version directory"):
        versioning.list_committed_versions()


def test_list_dangling_staging_returns_only_staging_dirs(tmp_path):
    v = _new_setup(tmp_path)
    v.stage(None, 0)
    v.stage(None, 1)
    # And a committed one which must NOT appear.
    v.stage(None, 2); v.commit(2)
    dangling = v.list_dangling_staging()
    assert sorted(p.name for p in dangling) == [
        "iteration-000000.staging",
        "iteration-000001.staging",
    ]


def test_list_dangling_staging_rejects_noncanonical_padding(tmp_path):
    versioning = _new_setup(tmp_path)
    (versioning.parent / "iteration-0000.staging").mkdir()
    with pytest.raises(ValueError, match="non-canonical version staging"):
        versioning.list_dangling_staging()


def test_recover_dangling_staging_removes_them(tmp_path):
    v = _new_setup(tmp_path)
    v.stage(None, 0)
    v.stage(None, 1)
    removed = v.recover_dangling_staging()
    assert sorted(p.name for p in removed) == [
        "iteration-000000.staging",
        "iteration-000001.staging",
    ]
    assert not v.staging_path(0).exists()
    assert not v.staging_path(1).exists()


def test_recover_dangling_staging_with_delete_false_does_not_remove(tmp_path):
    v = _new_setup(tmp_path)
    v.stage(None, 0)
    removed = v.recover_dangling_staging(delete=False)
    assert len(removed) == 1
    assert v.staging_path(0).exists()


def test_update_current_and_read_back(tmp_path):
    v = _new_setup(tmp_path)
    v.stage(None, 0); v.commit(0)
    v.update_current(0)
    assert v.current_version() == 0


def test_current_pointer_rejects_path_bearing_target(tmp_path):
    versioning = _new_setup(tmp_path)
    versioning.stage(None, 0)
    versioning.commit(0)
    (versioning.parent / ".current.pointer").write_text(
        "../iteration-000000\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version basename"):
        versioning.current_version()


def test_commit_rejects_non_directory_target(tmp_path):
    versioning = _new_setup(tmp_path)
    versioning.stage(None, 0)
    versioning.iteration_path(0).write_text("collision\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a regular directory"):
        versioning.commit(0)


def test_update_current_overwrites_previous(tmp_path):
    v = _new_setup(tmp_path)
    v.stage(None, 0); v.commit(0)
    v.stage(None, 1); v.commit(1)
    v.update_current(0)
    assert v.current_version() == 0
    v.update_current(1)
    assert v.current_version() == 1


def test_trained_model_current_pointer_cannot_move_backwards(tmp_path):
    versioning = TrainedModelVersioning(tmp_path / "TRAINED_MODELS")
    versioning.iteration_path(0).mkdir(parents=True)
    versioning.iteration_path(1).mkdir()

    with pytest.raises(TrainedModelError, match="only target newest version 1"):
        versioning.update_current(0)

    versioning.update_current(1)
    assert versioning.current_version() == 1


def test_update_current_raises_when_target_missing(tmp_path):
    v = _new_setup(tmp_path)
    with pytest.raises(FileNotFoundError):
        v.update_current(99)


def test_full_lifecycle(tmp_path):
    """End-to-end: stage iter 0, commit, point current; then stage iter 1
    from iter 0, commit, repoint current."""
    v = _new_setup(tmp_path)
    s0 = v.stage(None, 0)
    (s0 / "first.txt").write_text("alpha")
    v.commit(0)
    v.update_current(0)

    s1 = v.stage(0, 1)
    assert (s1 / "first.txt").read_text() == "alpha"
    (s1 / "second.txt").write_text("beta")
    v.commit(1)
    v.update_current(1)

    assert v.list_committed_versions() == [0, 1]
    assert v.current_version() == 1
    v.verify_committed(0)
    v.verify_committed(1)


def test_simulated_kill_between_stage_and_commit_recovers_cleanly(tmp_path):
    """Daemon dies between stage() and commit(). Next start finds dangling
    staging and removes it; committed iterations remain intact."""
    v = _new_setup(tmp_path)
    v.stage(None, 0); v.commit(0)
    # Now mid-iteration crash
    s = v.stage(0, 1)
    (s / "partial.txt").write_text("incomplete write")
    # Daemon restarts
    cleaned = v.recover_dangling_staging()
    assert len(cleaned) == 1
    # Iter 0 still verifies; iter 1 was discarded.
    v.verify_committed(0)
    assert not v.iteration_path(1).exists()
    assert not v.staging_path(1).exists()
