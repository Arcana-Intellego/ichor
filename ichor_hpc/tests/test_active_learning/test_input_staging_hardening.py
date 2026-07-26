"""Input-staging path and symlink hardening tests."""
import json
import os

import pytest

from ichor.hpc.active_learning.daemon import input_staging as stg


def test_safe_path_token_rejects_path_traversal_and_shell_chars():
    for value in ("../O1", "O/1", "O 1", "O1;rm", "$O1", ""):
        with pytest.raises(ValueError, match="safe path token"):
            stg.validate_safe_path_token("atom", value)


def test_pointdir_tree_validation_rejects_symlinked_child(tmp_path):
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
        stg._reject_symlink_tree(src)


def test_quantum_acceptance_manifest_reports_malformed_n_total(tmp_path):
    path = stg.quantum_acceptance_manifest_path(
        tmp_path,
        phase_name="GAUSSIAN",
    )
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


def test_quantum_acceptance_rejects_contradictory_dispositions(tmp_path):
    pointdir = tmp_path / "POINT_0000.pointdir"
    pointdir.mkdir()

    with pytest.raises(ValueError, match="duplicate dispositions"):
        stg.write_quantum_acceptance_manifest(
            tmp_path,
            phase_name="AIMALL",
            iteration=1,
            accepted=[pointdir],
            rejected=[(pointdir.name, "failed")],
        )


def test_quantum_acceptance_membership_modes_distinguish_filtered_consumer(
    tmp_path,
):
    accepted = [
        tmp_path / "POINT_0000.pointdir",
        tmp_path / "POINT_0002.pointdir",
    ]
    for pointdir in accepted:
        pointdir.mkdir()
    stg.write_points_file(
        tmp_path,
        [accepted[0], tmp_path / "POINT_0001.pointdir", accepted[1]],
    )
    stg.write_quantum_acceptance_manifest(
        tmp_path,
        phase_name="GAUSSIAN",
        iteration=1,
        accepted=accepted,
        rejected=[("POINT_0001.pointdir", "gaussian_failed")],
    )

    _, producer = stg.read_quantum_acceptance_manifest(
        tmp_path,
        expected_phase="GAUSSIAN",
        expected_iteration=1,
        points_membership=stg.POINTS_MEMBERSHIP_ALL_DISPOSITIONS,
    )
    assert (
        producer["points_membership_state"]
        == stg.POINTS_MEMBERSHIP_ALL_DISPOSITIONS
    )

    stg.write_points_file(tmp_path, accepted)
    _, consumer = stg.read_quantum_acceptance_manifest(
        tmp_path,
        expected_phase="GAUSSIAN",
        expected_iteration=1,
        points_membership=stg.POINTS_MEMBERSHIP_ACCEPTED_ONLY,
    )
    assert (
        consumer["points_membership_state"]
        == stg.POINTS_MEMBERSHIP_ACCEPTED_ONLY
    )
    _, replay = stg.read_quantum_acceptance_manifest(
        tmp_path,
        expected_phase="GAUSSIAN",
        expected_iteration=1,
        points_membership=stg.POINTS_MEMBERSHIP_PRODUCER_OR_ACCEPTED,
    )
    assert (
        replay["points_membership_state"]
        == stg.POINTS_MEMBERSHIP_ACCEPTED_ONLY
    )

    with pytest.raises(
        ValueError,
        match="dispositions do not exactly cover POINTS.txt",
    ):
        stg.read_quantum_acceptance_manifest(
            tmp_path,
            expected_phase="GAUSSIAN",
            expected_iteration=1,
            points_membership=stg.POINTS_MEMBERSHIP_ALL_DISPOSITIONS,
        )


def test_quantum_acceptance_accepted_only_rejects_subset_and_reordering(
    tmp_path,
):
    accepted = [
        tmp_path / "POINT_0000.pointdir",
        tmp_path / "POINT_0002.pointdir",
    ]
    for pointdir in accepted:
        pointdir.mkdir()
    stg.write_quantum_acceptance_manifest(
        tmp_path,
        phase_name="GAUSSIAN",
        iteration=14,
        accepted=accepted,
        rejected=[("POINT_0001.pointdir", "gaussian_failed")],
    )

    for points in ([accepted[0]], list(reversed(accepted))):
        stg.write_points_file(tmp_path, points)
        with pytest.raises(
            ValueError,
            match="authoritative order",
        ):
            stg.read_quantum_acceptance_manifest(
                tmp_path,
                expected_phase="GAUSSIAN",
                expected_iteration=14,
                points_membership=stg.POINTS_MEMBERSHIP_ACCEPTED_ONLY,
            )


def test_quantum_acceptance_boolean_membership_alias_remains_compatible(
    tmp_path,
):
    pointdir = tmp_path / "POINT_0000.pointdir"
    pointdir.mkdir()
    stg.write_points_file(tmp_path, [pointdir])
    stg.write_quantum_acceptance_manifest(
        tmp_path,
        phase_name="GAUSSIAN",
        iteration=1,
        accepted=[pointdir],
        rejected=[],
    )

    _, payload = stg.read_quantum_acceptance_manifest(
        tmp_path,
        expected_phase="GAUSSIAN",
        expected_iteration=1,
        require_points_file_membership=True,
    )

    assert (
        payload["points_membership_state"]
        == stg.POINTS_MEMBERSHIP_ALL_DISPOSITIONS
    )
