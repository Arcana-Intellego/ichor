"""Input-staging path and symlink hardening tests."""
import json
import os

import pytest

from ichor.hpc.active_learning.daemon import input_staging as stg


def test_safe_path_token_rejects_path_traversal_and_shell_chars():
    for value in ("../O1", "O/1", "O 1", "O1;rm", "$O1", ""):
        with pytest.raises(ValueError, match="safe path token"):
            stg.validate_safe_path_token("atom", value)


def test_copytree_no_symlinks_rejects_symlinked_pointdir_child(tmp_path):
    src = tmp_path / "POINT_0000.pointdir"
    src.mkdir()
    target = tmp_path / "target.txt"
    target.write_text("data\n", encoding="utf-8")
    link = src / "linked.txt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip("symlink creation unavailable on this host: " + str(exc))

    with pytest.raises(ValueError, match="symlink"):
        stg._copytree_no_symlinks(src, tmp_path / "copy.pointdir")


def test_quantum_acceptance_manifest_reports_malformed_n_total(tmp_path):
    path = tmp_path / stg.QUANTUM_ACCEPTANCE_MANIFEST
    path.write_text(
            json.dumps({
                "schema_version": stg.QUANTUM_ACCEPTANCE_SCHEMA_VERSION,
                "phase": "GAUSSIAN",
                "iteration": 0,
                "accepted_pointdirs": [],
                "rejected": [],
                "n_total": "not-an-integer",
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="n_total is not an integer"):
        stg.read_quantum_acceptance_manifest(
            tmp_path,
            expected_phase="GAUSSIAN",
            expected_iteration=0,
        )
