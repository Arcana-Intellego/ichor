import shutil
from pathlib import Path

import pytest

from ichor.hpc.active_learning.daemon.aimall_output_validation import (
    AIMALL_AUTHORITY_INVALID,
    AIMALL_OUTPUT_INVALID,
    AIMALL_STRUCTURAL_INVALID_EXIT_CODE,
    assess_aimall_output,
    main,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "live_outputs"
    / "iter_quantum"
    / "POINT_0000.pointdir"
)


def _copy_pointdir(tmp_path: Path) -> Path:
    target = tmp_path / "POINT_0000.pointdir"
    shutil.copytree(FIXTURE, target)
    return target


def test_shared_aimall_validator_accepts_complete_output(tmp_path):
    pointdir = _copy_pointdir(tmp_path)

    assessment = assess_aimall_output(pointdir)

    assert assessment.valid is True
    assert assessment.reason == ""
    assert main(["--pointdir", str(pointdir)]) == 0


def test_shared_aimall_validator_returns_reserved_exit_for_truncated_int(
    tmp_path,
):
    pointdir = _copy_pointdir(tmp_path)
    int_path = next(pointdir.rglob("h3.int"))
    text = int_path.read_text(encoding="utf-8")
    int_path.write_text(text.split("Total time", 1)[0], encoding="utf-8")

    first = assess_aimall_output(pointdir)
    second = assess_aimall_output(pointdir)

    assert first.category == AIMALL_OUTPUT_INVALID
    assert first.reason == "int_file_incomplete:h3.int"
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    assert main(["--pointdir", str(pointdir)]) == (
        AIMALL_STRUCTURAL_INVALID_EXIT_CODE
    )


def test_shared_aimall_validator_rejects_partial_and_misidentified_ints(
    tmp_path,
):
    pointdir = _copy_pointdir(tmp_path)
    int_path = next(pointdir.rglob("h3.int"))
    int_path.unlink()

    partial = assess_aimall_output(pointdir)

    assert partial.category == AIMALL_OUTPUT_INVALID
    assert partial.reason == "missing_or_partial_int_set_2_of_3"

    pointdir = _copy_pointdir(tmp_path / "renamed")
    int_path = next(pointdir.rglob("h3.int"))
    int_path.rename(int_path.with_name("x3.int"))

    renamed = assess_aimall_output(pointdir)

    assert renamed.category == AIMALL_OUTPUT_INVALID
    assert renamed.reason == "int_atom_identity_mismatch"


def test_shared_aimall_validator_rejects_parser_failure_after_terminal_marker(
    tmp_path,
):
    pointdir = _copy_pointdir(tmp_path)
    int_path = next(pointdir.rglob("h3.int"))
    int_path.write_text("Total time\n", encoding="utf-8")

    assessment = assess_aimall_output(pointdir)

    assert assessment.category == AIMALL_OUTPUT_INVALID
    assert assessment.reason == "int_parse_failure"


def test_shared_aimall_validator_rejects_ambiguous_atomic_directories(
    tmp_path,
):
    pointdir = _copy_pointdir(tmp_path)
    atomic = next(
        path for path in pointdir.iterdir() if path.name.endswith("_atomicfiles")
    )
    shutil.copytree(atomic, pointdir / "duplicate_atomicfiles")

    assessment = assess_aimall_output(pointdir)

    assert assessment.category == AIMALL_OUTPUT_INVALID
    assert assessment.reason == "missing_or_ambiguous_atomicfiles_directory"


def test_shared_aimall_validator_rejects_symlinked_atomic_authority(tmp_path):
    pointdir = _copy_pointdir(tmp_path)
    atomic = next(
        path for path in pointdir.iterdir() if path.name.endswith("_atomicfiles")
    )
    moved = pointdir / "real-output"
    atomic.rename(moved)
    try:
        atomic.symlink_to(moved, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")

    assessment = assess_aimall_output(pointdir)

    assert assessment.category == AIMALL_AUTHORITY_INVALID
    assert assessment.reason == "atomicfiles_directory_symlinked"
