"""Tests for ichor.hpc.active_learning.versioning.manifest."""
import json
import os
from pathlib import Path

import pytest

from ichor.hpc.active_learning.versioning.manifest import (
    MANIFEST_FILENAME,
    ManifestMismatchError,
    compute_directory_manifest,
    read_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)


def _populate(root: Path):
    (root / "a.txt").write_text("hello")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("world")
    (root / "sub" / "nested").mkdir()
    (root / "sub" / "nested" / "c.bin").write_bytes(b"\x00\x01\x02\x03")
    return root


def test_sha256_file_matches_python_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "f.bin"
    data = b"some binary content " * 100
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()


def test_compute_directory_manifest_keys_use_forward_slashes(tmp_path):
    _populate(tmp_path)
    m = compute_directory_manifest(tmp_path)
    keys = list(m.keys())
    assert all("\\" not in k for k in keys)
    assert "a.txt" in m
    assert "sub/b.txt" in m
    assert "sub/nested/c.bin" in m


def test_compute_directory_manifest_skips_manifest_file(tmp_path):
    _populate(tmp_path)
    (tmp_path / MANIFEST_FILENAME).write_text("{}")
    m = compute_directory_manifest(tmp_path)
    assert MANIFEST_FILENAME not in m


def test_compute_directory_manifest_skips_symlinks(tmp_path):
    _populate(tmp_path)
    try:
        os.symlink("a.txt", str(tmp_path / "link.txt"))
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not supported on this platform")
    m = compute_directory_manifest(tmp_path)
    assert "link.txt" not in m


def test_compute_directory_manifest_excludes_user_paths(tmp_path):
    _populate(tmp_path)
    m = compute_directory_manifest(tmp_path, exclude=("a.txt",))
    assert "a.txt" not in m


def test_compute_directory_manifest_raises_for_non_directory(tmp_path):
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("hi")
    with pytest.raises(NotADirectoryError):
        compute_directory_manifest(file_path)


def test_write_and_read_manifest_roundtrip(tmp_path):
    _populate(tmp_path)
    m = compute_directory_manifest(tmp_path)
    written = write_manifest(tmp_path, m)
    assert written == tmp_path / MANIFEST_FILENAME
    loaded = read_manifest(tmp_path)
    assert loaded == m


def test_read_manifest_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path)


def test_verify_manifest_passes_on_intact_tree(tmp_path):
    _populate(tmp_path)
    m = compute_directory_manifest(tmp_path)
    write_manifest(tmp_path, m)
    missing, mismatched = verify_manifest(tmp_path)
    assert missing == []
    assert mismatched == []


def test_verify_manifest_detects_tampered_file(tmp_path):
    _populate(tmp_path)
    m = compute_directory_manifest(tmp_path)
    write_manifest(tmp_path, m)
    (tmp_path / "a.txt").write_text("TAMPERED")
    with pytest.raises(ManifestMismatchError):
        verify_manifest(tmp_path)


def test_verify_manifest_detects_missing_file(tmp_path):
    _populate(tmp_path)
    m = compute_directory_manifest(tmp_path)
    write_manifest(tmp_path, m)
    (tmp_path / "a.txt").unlink()
    with pytest.raises(ManifestMismatchError):
        verify_manifest(tmp_path)


def test_verify_manifest_non_strict_returns_diff(tmp_path):
    _populate(tmp_path)
    m = compute_directory_manifest(tmp_path)
    write_manifest(tmp_path, m)
    (tmp_path / "a.txt").write_text("TAMPERED")
    missing, mismatched = verify_manifest(tmp_path, strict=False)
    assert missing == []
    assert mismatched == ["a.txt"]


def test_verify_manifest_ignores_extra_files_outside_manifest(tmp_path):
    _populate(tmp_path)
    m = compute_directory_manifest(tmp_path)
    write_manifest(tmp_path, m)
    # Adding a file NOT listed in the manifest is harmless.
    (tmp_path / "added_later.txt").write_text("not in manifest")
    missing, mismatched = verify_manifest(tmp_path, strict=False)
    assert missing == []
    assert mismatched == []
