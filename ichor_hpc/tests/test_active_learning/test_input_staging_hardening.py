"""Input-staging path and symlink hardening tests."""
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
