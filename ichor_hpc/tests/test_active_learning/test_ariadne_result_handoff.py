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


def _landing_safety(*, accepted=True):
    return {
        "accepted": bool(accepted),
        "policy": "raw_final" if accepted else "rejected",
        "selected_origin": "raw_final" if accepted else "rejected",
        "selected_candidate_index": 0,
        "reasons": [] if accepted else ["landing_safety_rejected"],
        "record_only_reasons": [],
        "metrics": {"whitened_distance": 0.5},
        "raw_final": {},
        "n_candidates_evaluated": 1,
        "n_safe_candidates": 1 if accepted else 0,
    }


def _result_payload(
    *,
    trajectory_sha256=None,
    include_landing_safety=True,
    landing_safety_accepted=True,
):
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
    if include_landing_safety:
        payload["landing_safety"] = _landing_safety(
            accepted=bool(landing_safety_accepted),
        )
    return payload


def _seed_record():
    return {"seed_index": 0, "frame_id": 2}


def _write_manifest_for_result(tmp_path, result_payload):
    iter_dir = tmp_path / "iteration-0000"
    seed_dir = iter_dir / "pool" / "seed_0000"
    seed_dir.mkdir(parents=True)
    result_path = seed_dir / "result.json"
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
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
            "return_code": int(result_payload.get("return_code", 0)),
        }],
        "rejected": [],
    })
    return iter_dir


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


def test_validate_ariadne_result_rejects_missing_landing_safety_by_default():
    payload = _result_payload(include_landing_safety=False)

    with pytest.raises(HandoffManifestError, match="missing_landing_safety"):
        validate_ariadne_result(
            payload,
            expected_iteration=0,
            seed_record=_seed_record(),
        )


def test_validate_ariadne_result_accepts_missing_landing_safety_for_legacy():
    payload = _result_payload(include_landing_safety=False)

    out = validate_ariadne_result(
        payload,
        expected_iteration=0,
        seed_record=_seed_record(),
        accept_legacy_missing_landing_safety=True,
    )

    assert out["return_code"] == 0


def test_validate_ariadne_result_rejects_unsafe_landing_safety():
    payload = _result_payload(landing_safety_accepted=False)

    with pytest.raises(HandoffManifestError, match="landing_safety_rejected"):
        validate_ariadne_result(
            payload,
            expected_iteration=0,
            seed_record=_seed_record(),
        )


def test_read_ariadne_manifest_accepts_legacy_result_sha_from_manifest(tmp_path):
    iter_dir = _write_manifest_for_result(tmp_path, _result_payload())

    manifest = read_ariadne_results_manifest(iter_dir, expected_iteration=0)

    assert manifest["accepted"][0]["trajectory_sha256"] == "a" * 64
    assert manifest["accepted"][0]["legacy_missing_trajectory_sha256"] is True
    assert manifest["accepted"][0]["landing_safety"]["accepted"] is True
    _, frames, records = ariadne_candidate_frames(iter_dir, expected_iteration=0)
    assert len(frames) == 1
    assert records[0]["legacy_missing_trajectory_sha256"] is True
    assert records[0]["landing_safety"]["accepted"] is True


def test_read_ariadne_manifest_rejects_missing_landing_safety_by_default(tmp_path):
    iter_dir = _write_manifest_for_result(
        tmp_path,
        _result_payload(include_landing_safety=False),
    )

    with pytest.raises(HandoffManifestError, match="missing_landing_safety"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=0)


def test_read_ariadne_manifest_accepts_missing_landing_safety_for_legacy(tmp_path):
    iter_dir = _write_manifest_for_result(
        tmp_path,
        _result_payload(include_landing_safety=False),
    )

    manifest = read_ariadne_results_manifest(
        iter_dir,
        expected_iteration=0,
        accept_legacy_missing_landing_safety=True,
    )
    _, frames, records = ariadne_candidate_frames(
        iter_dir,
        expected_iteration=0,
        accept_legacy_missing_landing_safety=True,
    )

    assert len(manifest["accepted"]) == 1
    assert len(frames) == 1
    assert records[0]["seed_index"] == 0


def test_read_ariadne_manifest_rejects_unsafe_landing_safety(tmp_path):
    iter_dir = _write_manifest_for_result(
        tmp_path,
        _result_payload(landing_safety_accepted=False),
    )

    with pytest.raises(HandoffManifestError, match="landing_safety_rejected"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=0)


def test_read_ariadne_manifest_still_rejects_wrong_result_sha(tmp_path):
    iter_dir = _write_manifest_for_result(
        tmp_path,
        _result_payload(trajectory_sha256="b" * 64),
    )

    with pytest.raises(HandoffManifestError, match="wrong_trajectory_sha256"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=0)


def test_read_ariadne_manifest_wraps_malformed_integer_fields(tmp_path):
    iter_dir = tmp_path / "iteration-0000"
    seed_dir = iter_dir / "pool" / "seed_0000"
    seed_dir.mkdir(parents=True)
    result_path = seed_dir / "result.json"
    result_path.write_text(json.dumps(_result_payload()), encoding="utf-8")
    provenance_path = seed_dir / ".provenance.json"
    provenance_path.write_text("{}", encoding="utf-8")

    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "iteration": "not-an-int",
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

    with pytest.raises(HandoffManifestError, match="ARIADNE results iteration"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=0)
