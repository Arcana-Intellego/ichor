import json

import pytest

from ichor.hpc.active_learning.handoff_manifests import (
    ARIADNE_RESULTS_SCHEMA_VERSION,
    HandoffManifestError,
    ariadne_candidate_frames,
    read_ariadne_results_manifest,
    validate_ariadne_result,
    write_ariadne_results_manifest,
)


def _result_payload(*, trajectory_sha256=None):
    payload = {
        "iteration": 0,
        "seed_index": 0,
        "seed_frame_id": 2,
        "atom_types": ["O", "H", "H"],
        "final_coordinates": [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        "alpha_trajectory": [0.0, 1.0],
        "alpha_initial": 0.0,
        "alpha_final": 1.0,
        "n_evaluations": 2,
        "return_code": 0,
        "wall_seconds": 1.5,
        "fell_back_to_ds": False,
        "whitened_distance_final": 0.5,
    }
    if trajectory_sha256 is not None:
        payload["trajectory_sha256"] = str(trajectory_sha256)
    return payload


def _seed_record():
    return {"seed_index": 0, "frame_id": 2}


def test_validate_ariadne_result_accepts_matching_trajectory_sha():
    payload = _result_payload(trajectory_sha256="a" * 64)

    out = validate_ariadne_result(
        payload,
        expected_iteration=0,
        seed_record=_seed_record(),
        expected_trajectory_sha256="a" * 64,
    )

    assert out["trajectory_sha256"] == "a" * 64
    assert out["legacy_missing_trajectory_sha256"] is False


def test_validate_ariadne_result_rejects_wrong_trajectory_sha():
    payload = _result_payload(trajectory_sha256="b" * 64)

    with pytest.raises(HandoffManifestError, match="wrong_trajectory_sha256"):
        validate_ariadne_result(
            payload,
            expected_iteration=0,
            seed_record=_seed_record(),
            expected_trajectory_sha256="a" * 64,
        )


def test_validate_ariadne_result_accepts_legacy_missing_result_sha():
    payload = _result_payload()

    out = validate_ariadne_result(
        payload,
        expected_iteration=0,
        seed_record=_seed_record(),
        expected_trajectory_sha256="a" * 64,
    )

    assert out["trajectory_sha256"] == "a" * 64
    assert out["legacy_missing_trajectory_sha256"] is True


def test_read_ariadne_manifest_accepts_legacy_result_sha_from_manifest(tmp_path):
    iter_dir = tmp_path / "iteration-0000"
    seed_dir = iter_dir / "pool" / "seed_0000"
    seed_dir.mkdir(parents=True)
    result_path = seed_dir / "result.json"
    result_path.write_text(json.dumps(_result_payload()), encoding="utf-8")
    provenance_path = seed_dir / ".provenance.json"
    provenance_path.write_text("{}", encoding="utf-8")

    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "iteration": 0,
        "trajectory_sha256": "a" * 64,
        "expected_n": 1,
        "n_accepted": 1,
        "n_rejected": 0,
        "accepted": [{
            "seed_index": 0,
            "seed_frame_id": 2,
            "seed_dir": str(seed_dir.resolve()),
            "result_json": str(result_path.resolve()),
            "provenance_json": str(provenance_path.resolve()),
            "return_code": 0,
        }],
        "rejected": [],
    })

    manifest = read_ariadne_results_manifest(iter_dir, expected_iteration=0)

    assert manifest["accepted"][0]["trajectory_sha256"] == "a" * 64
    assert manifest["accepted"][0]["legacy_missing_trajectory_sha256"] is True
    _, frames, records = ariadne_candidate_frames(iter_dir, expected_iteration=0)
    assert len(frames) == 1
    assert records[0]["legacy_missing_trajectory_sha256"] is True


def test_read_ariadne_manifest_still_rejects_wrong_result_sha(tmp_path):
    iter_dir = tmp_path / "iteration-0000"
    seed_dir = iter_dir / "pool" / "seed_0000"
    seed_dir.mkdir(parents=True)
    result_path = seed_dir / "result.json"
    result_path.write_text(
        json.dumps(_result_payload(trajectory_sha256="b" * 64)),
        encoding="utf-8",
    )
    provenance_path = seed_dir / ".provenance.json"
    provenance_path.write_text("{}", encoding="utf-8")

    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "iteration": 0,
        "trajectory_sha256": "a" * 64,
        "expected_n": 1,
        "n_accepted": 1,
        "n_rejected": 0,
        "accepted": [{
            "seed_index": 0,
            "seed_frame_id": 2,
            "seed_dir": str(seed_dir.resolve()),
            "result_json": str(result_path.resolve()),
            "provenance_json": str(provenance_path.resolve()),
            "return_code": 0,
        }],
        "rejected": [],
    })

    with pytest.raises(HandoffManifestError, match="wrong_trajectory_sha256"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=0)
