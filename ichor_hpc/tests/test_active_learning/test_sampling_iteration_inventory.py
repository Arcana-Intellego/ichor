from pathlib import Path

import pytest

from ichor.hpc.active_learning.versioning import sampling_iterations as sampling


def test_iteration_inventory_hashes_each_file_once_and_reports_bytes(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "iteration"
    (root / "ariadne").mkdir(parents=True)
    expected = {}
    for index in range(24):
        path = root / "ariadne" / ("result-" + str(index).zfill(3) + ".json")
        path.write_bytes((str(index) * (index + 1)).encode("ascii"))
        expected[path] = 0
    original = sampling.sha256_file

    def counted(path):
        expected[Path(path)] += 1
        return original(path)

    monkeypatch.setattr(sampling, "sha256_file", counted)
    progress = []
    capture = sampling._inventory_capture(
        root,
        "ITERATION_MANIFEST.json",
        progress_callback=lambda **fields: progress.append(dict(fields)),
    )

    assert set(expected.values()) == {1}
    assert [record["path"] for record in capture.files] == sorted(
        record["path"] for record in capture.files
    )
    assert progress[-1]["completed"] == len(expected)
    assert progress[-1]["bytes_completed"] == sum(
        path.stat().st_size for path in expected
    )


def test_iteration_inventory_rejects_concurrent_file_replacement(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "iteration"
    root.mkdir()
    target = root / "payload.bin"
    target.write_bytes(b"before")
    original = sampling.sha256_file

    def replace_after_hash(path):
        digest = original(path)
        Path(path).write_bytes(b"after-content")
        return digest

    monkeypatch.setattr(sampling, "sha256_file", replace_after_hash)
    with pytest.raises(sampling.SamplingIterationError, match="changed while hashing"):
        sampling._inventory_capture(root, "ITERATION_MANIFEST.json")


def test_iteration_inventory_recheck_rejects_new_root_entry(tmp_path):
    root = tmp_path / "iteration"
    root.mkdir()
    (root / "payload.bin").write_bytes(b"stable")
    capture = sampling._inventory_capture(root, "ITERATION_MANIFEST.json")

    (root / "late-entry.bin").write_bytes(b"late")

    with pytest.raises(sampling.SamplingIterationError, match="root entries changed"):
        capture.recheck()
