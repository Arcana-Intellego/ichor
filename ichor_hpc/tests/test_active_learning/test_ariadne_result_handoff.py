import json
from pathlib import Path

import pytest

from ichor.hpc.active_learning.ariadne_outputs import (
    AriadneOutputError,
    SEED_OUTPUT_MANIFEST_FILENAME,
    read_and_validate_optimisation_trajectory,
    write_optimisation_trajectory,
    write_seed_output_manifest,
)
from ichor.hpc.active_learning.daemon.state import atomic_write_json
from ichor.hpc.active_learning.handoff_manifests import (
    ARIADNE_RESULTS_SCHEMA_VERSION,
    HandoffManifestError,
    ariadne_candidate_frames,
    build_seed_selection_manifest,
    read_ariadne_results_manifest,
    seeds_picked_path,
    validate_ariadne_result,
    write_ariadne_results_manifest,
    write_ariadne_batch_decision,
    write_ariadne_landing_audit,
)
from ichor.hpc.active_learning.historical_ariadne import (
    read_historical_ariadne_records,
)
from ichor.hpc.active_learning.layout import active_ariadne_dir, ariadne_seed_dir
from ichor.hpc.active_learning.seed_identity import (
    SeedIdentityError,
    read_ariadne_task_map,
    write_ariadne_task_map,
)
from ichor.hpc.active_learning.versioning.manifest import sha256_file
from ichor.hpc.active_learning.versioning.provenance import write_seed_provenance


TRAJECTORY_SHA = "a" * 64
MODEL_SHA = "c" * 64
CAMPAIGN_UID = "canonical-handoff-test"


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
    seed_uid,
    *,
    trajectory_sha256=TRAJECTORY_SHA,
    include_landing_safety=True,
    landing_safety_accepted=True,
):
    payload = {
        "iteration": 1,
        "seed_id": 1,
        "seed_uid": str(seed_uid),
        "array_task_id": 0,
        "seed_frame_id": 2,
        "trajectory_sha256": str(trajectory_sha256),
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
    if include_landing_safety:
        payload["landing_safety"] = _landing_safety(
            accepted=bool(landing_safety_accepted),
        )
    return payload


def _selection_payload():
    return build_seed_selection_manifest(
        campaign_uid=CAMPAIGN_UID,
        campaign_random_seed=0,
        iteration=1,
        models_version=0,
        model_manifest_sha256=MODEL_SHA,
        model_set_sha256=MODEL_SHA,
        trajectory_sha256=TRAJECTORY_SHA,
        selection_strategy="hybrid_variance",
        seed_records=[{
            "seed_id": 1,
            "frame_id": 2,
            "pool_row_index_zero_based": 2,
            "selection_origin": "bulk",
            "variance_at_selection": 0.1,
        }],
    )


def _write_canonical_handoff(
    tmp_path,
    *,
    result_trajectory_sha=TRAJECTORY_SHA,
    include_landing_safety=True,
    landing_safety_accepted=True,
):
    iter_dir = tmp_path / "iteration-000001"
    selection = _selection_payload()
    selection_path = seeds_picked_path(iter_dir)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, selection)
    task_map_path = write_ariadne_task_map(iter_dir, selection)
    seed_uid = str(selection["seed_records"][0]["seed_uid"])

    seed_dir = ariadne_seed_dir(iter_dir, 1)
    seed_dir.mkdir(parents=True, exist_ok=True)
    result_path = seed_dir / "result.json"
    result = _result_payload(
        seed_uid,
        trajectory_sha256=result_trajectory_sha,
        include_landing_safety=include_landing_safety,
        landing_safety_accepted=landing_safety_accepted,
    )
    atomic_write_json(result_path, result)
    write_optimisation_trajectory(
        seed_dir,
        atom_types=result["atom_types"],
        coordinate_frames=[result["final_coordinates"]],
        alpha_values=[1.0],
        gradient_norms=[0.0],
        origins=["raw_final"],
    )
    output_manifest = write_seed_output_manifest(
        seed_dir,
        campaign_uid=CAMPAIGN_UID,
        iteration=1,
        seed_id=1,
        seed_uid=seed_uid,
        array_task_id=0,
        task_success=bool(landing_safety_accepted),
        task_exit_code=0 if landing_safety_accepted else 5,
    )
    provenance_path = write_seed_provenance(
        seed_dir,
        campaign_uid=CAMPAIGN_UID,
        iteration=1,
        trajectory_sha256=TRAJECTORY_SHA,
        seed_frame_id=2,
        seed_id=1,
        seed_uid=seed_uid,
        array_task_id_zero_based=0,
        seed_selection_origin="bulk",
        seed_variance_at_selection=0.1,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
        mode_weighting_policy="variance",
    )
    ariadne_root = active_ariadne_dir(iter_dir)
    accepted_record = {
        "seed_id": 1,
        "seed_uid": seed_uid,
        "array_task_id": 0,
        "seed_frame_id": 2,
        "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
        "result_json": result_path.relative_to(ariadne_root).as_posix(),
        "provenance_json": Path(provenance_path).relative_to(ariadne_root).as_posix(),
        "output_manifest": output_manifest.relative_to(ariadne_root).as_posix(),
        "return_code": 0,
        "landing_safety": result.get("landing_safety"),
        "result_sha256": sha256_file(result_path),
        "provenance_sha256": sha256_file(provenance_path),
        "output_manifest_sha256": sha256_file(output_manifest),
    }
    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "campaign_uid": CAMPAIGN_UID,
        "iteration": 1,
        "trajectory_sha256": TRAJECTORY_SHA,
        "task_map": {
            "path": task_map_path.relative_to(ariadne_root).as_posix(),
            "sha256": sha256_file(task_map_path),
        },
        "expected_n": 1,
        "n_accepted": 1,
        "n_rejected": 0,
        "accepted": [accepted_record],
        "rejected": [],
    })
    write_ariadne_landing_audit(iter_dir, {
        "iteration": 1,
        "summary": {
            "accepted": 1,
            "salvaged": 0,
            "backtracked": 0,
            "rejected": 0,
            "rejection_reasons": {},
            "policies": {"raw_final": 1},
        },
        "seeds": [{
            "seed_id": 1,
            "seed_uid": seed_uid,
            "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
            "result_json": result_path.relative_to(ariadne_root).as_posix(),
            "landing_safety": result.get("landing_safety"),
            "handoff_accepted": True,
            "movement": {"aligned_rmsd_ang": 0.125},
        }],
    })
    write_ariadne_batch_decision(
        iter_dir,
        campaign_uid=CAMPAIGN_UID,
        iteration=1,
        config_sha256="test-config",
        failure_threshold_fraction=0.0,
        expected_n=1,
        n_accepted=1,
        n_rejected=0,
        accepted=True,
        reasons=[],
    )
    return iter_dir, selection, result_path


def test_historical_reader_requires_agreeing_results_and_audit(tmp_path):
    iter_dir, selection, _ = _write_canonical_handoff(tmp_path)

    records = read_historical_ariadne_records(
        iter_dir,
        expected_iteration=1,
    )

    assert len(records) == 1
    assert records[0]["seed_uid"] == selection["seed_records"][0]["seed_uid"]
    assert records[0]["movement"]["aligned_rmsd_ang"] == pytest.approx(0.125)


def test_trajectory_reader_rejects_semantic_drift_with_rebound_hash(tmp_path):
    seed_dir = tmp_path / "seed-000001"
    write_optimisation_trajectory(
        seed_dir,
        atom_types=["H"],
        coordinate_frames=[[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]]],
        alpha_values=[1.0, 2.0],
        gradient_norms=[0.5, 0.25],
        origins=["accepted", "accepted"],
    )
    trajectory = seed_dir / "trajectory"
    metrics_path = trajectory / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    records[1]["frame_number"] = 1
    metrics_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest_path = trajectory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["metrics_jsonl"]["size"] = metrics_path.stat().st_size
    manifest["files"]["metrics_jsonl"]["sha256"] = sha256_file(metrics_path)
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(AriadneOutputError, match="frame numbers"):
        read_and_validate_optimisation_trajectory(seed_dir)


def test_validate_ariadne_result_accepts_canonical_identity():
    selection = _selection_payload()
    seed_uid = selection["seed_records"][0]["seed_uid"]
    payload = _result_payload(seed_uid)

    out = validate_ariadne_result(
        payload,
        expected_iteration=1,
        seed_record={"seed_id": 1, "seed_uid": seed_uid, "frame_id": 2},
        expected_trajectory_sha256=TRAJECTORY_SHA,
    )

    assert out["seed_id"] == 1
    assert out["array_task_id"] == 0
    assert out["trajectory_sha256"] == TRAJECTORY_SHA


def test_validate_ariadne_result_binds_claimed_seed_to_authoritative_pool_frame():
    selection = _selection_payload()
    seed_uid = selection["seed_records"][0]["seed_uid"]
    authoritative = [
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0],
    ]
    payload = _result_payload(seed_uid)
    payload["initial_coordinates"] = [list(row) for row in authoritative]
    payload["seed_coordinates"] = [list(row) for row in authoritative]

    validate_ariadne_result(
        payload,
        expected_iteration=1,
        seed_record={"seed_id": 1, "seed_uid": seed_uid, "frame_id": 2},
        expected_initial_coordinates=authoritative,
    )

    payload["initial_coordinates"][1][0] = 1.25
    with pytest.raises(
        HandoffManifestError,
        match="initial_coordinates_authoritative_seed_mismatch",
    ):
        validate_ariadne_result(
            payload,
            expected_iteration=1,
            seed_record={"seed_id": 1, "seed_uid": seed_uid, "frame_id": 2},
            expected_initial_coordinates=authoritative,
        )


def test_validate_ariadne_result_rejects_wrong_trajectory_sha():
    selection = _selection_payload()
    seed_uid = selection["seed_records"][0]["seed_uid"]
    payload = _result_payload(seed_uid, trajectory_sha256="b" * 64)

    with pytest.raises(HandoffManifestError, match="wrong_trajectory_sha256"):
        validate_ariadne_result(
            payload,
            expected_iteration=1,
            seed_record={"seed_id": 1, "seed_uid": seed_uid, "frame_id": 2},
            expected_trajectory_sha256=TRAJECTORY_SHA,
        )


def test_validate_ariadne_result_rejects_missing_landing_safety():
    selection = _selection_payload()
    seed_uid = selection["seed_records"][0]["seed_uid"]
    payload = _result_payload(seed_uid, include_landing_safety=False)

    with pytest.raises(HandoffManifestError, match="missing_landing_safety"):
        validate_ariadne_result(
            payload,
            expected_iteration=1,
            seed_record={"seed_id": 1, "seed_uid": seed_uid, "frame_id": 2},
        )


def test_validate_ariadne_result_rejects_unsafe_landing():
    selection = _selection_payload()
    seed_uid = selection["seed_records"][0]["seed_uid"]
    payload = _result_payload(seed_uid, landing_safety_accepted=False)

    with pytest.raises(HandoffManifestError, match="landing_safety_rejected"):
        validate_ariadne_result(
            payload,
            expected_iteration=1,
            seed_record={"seed_id": 1, "seed_uid": seed_uid, "frame_id": 2},
        )


def test_read_ariadne_manifest_and_reconstruct_candidates(tmp_path):
    iter_dir, selection, _ = _write_canonical_handoff(tmp_path)

    manifest = read_ariadne_results_manifest(iter_dir, expected_iteration=1)
    _, frames, records = ariadne_candidate_frames(iter_dir, expected_iteration=1)

    assert manifest["accepted"][0]["seed_id"] == 1
    assert manifest["accepted"][0]["seed_uid"] == selection["seed_records"][0]["seed_uid"]
    assert len(frames) == 1
    assert records[0]["seed_id"] == 1
    assert records[0]["landing_safety"]["accepted"] is True


def test_ariadne_batch_decision_must_match_immutable_result_counts(tmp_path):
    iter_dir, _, _ = _write_canonical_handoff(tmp_path)

    with pytest.raises(HandoffManifestError, match="counts do not match"):
        write_ariadne_batch_decision(
            iter_dir,
            campaign_uid=CAMPAIGN_UID,
            iteration=1,
            config_sha256="different-policy",
            failure_threshold_fraction=0.0,
            expected_n=1,
            n_accepted=0,
            n_rejected=1,
            accepted=False,
            reasons=["synthetic rejection"],
        )


def test_ariadne_batch_decision_boolean_is_derived_from_policy(tmp_path):
    iter_dir, _, _ = _write_canonical_handoff(tmp_path)

    with pytest.raises(HandoffManifestError, match="accepted flag disagrees"):
        write_ariadne_batch_decision(
            iter_dir,
            campaign_uid=CAMPAIGN_UID,
            iteration=1,
            config_sha256="different-policy",
            failure_threshold_fraction=0.0,
            expected_n=1,
            n_accepted=1,
            n_rejected=0,
            accepted=False,
            reasons=["synthetic inconsistent decision"],
        )


def test_read_ariadne_manifest_rejects_wrong_result_trajectory_sha(tmp_path):
    iter_dir, _, _ = _write_canonical_handoff(
        tmp_path,
        result_trajectory_sha="b" * 64,
    )

    with pytest.raises(HandoffManifestError, match="wrong_trajectory_sha256"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=1)


def test_read_ariadne_manifest_rejects_tampered_result(tmp_path):
    iter_dir, _, result_path = _write_canonical_handoff(tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["alpha_final"] = 99.0
    atomic_write_json(result_path, payload)

    with pytest.raises(Exception, match="(size|hash) mismatch"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=1)


def test_seed_output_writer_rejects_inconsistent_success_status(tmp_path):
    _iter_dir, selection, result_path = _write_canonical_handoff(tmp_path)

    with pytest.raises(AriadneOutputError, match="must agree"):
        write_seed_output_manifest(
            result_path.parent,
            campaign_uid=CAMPAIGN_UID,
            iteration=1,
            seed_id=1,
            seed_uid=str(selection["seed_records"][0]["seed_uid"]),
            array_task_id=0,
            task_success=False,
            task_exit_code=0,
        )


def test_accepted_ariadne_record_rejects_failed_output_status(tmp_path):
    iter_dir, _, result_path = _write_canonical_handoff(tmp_path)
    output_path = result_path.parent / SEED_OUTPUT_MANIFEST_FILENAME
    output = json.loads(output_path.read_text(encoding="utf-8"))
    output["task_success"] = False
    output["task_exit_code"] = 5
    atomic_write_json(output_path, output)

    with pytest.raises(HandoffManifestError, match="records task failure"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=1)


def test_ariadne_results_rejects_task_map_binding_drift(tmp_path):
    iter_dir, _, _ = _write_canonical_handoff(tmp_path)
    results_path = active_ariadne_dir(iter_dir) / "RESULTS.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["task_map"]["sha256"] = "f" * 64
    atomic_write_json(results_path, results)

    with pytest.raises(HandoffManifestError, match="task-map hash mismatch"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=1)


def test_task_map_makes_scheduler_and_seed_numbering_explicit(tmp_path):
    iter_dir, selection, _ = _write_canonical_handoff(tmp_path)

    task_map = read_ariadne_task_map(iter_dir, expected_iteration=1)

    assert task_map["n_tasks"] == 1
    assert task_map["tasks"][0]["array_task_id"] == 0
    assert task_map["tasks"][0]["seed_id"] == 1
    assert task_map["tasks"][0]["seed_uid"] == selection["seed_records"][0][
        "seed_uid"
    ]
    assert task_map["tasks"][0]["seed_directory"] == (
        "ariadne/seeds/seed-000001"
    )


def test_task_map_rejects_selection_manifest_drift(tmp_path):
    iter_dir, _, _ = _write_canonical_handoff(tmp_path)
    selection_path = seeds_picked_path(iter_dir)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["diagnostic_tamper"] = True
    atomic_write_json(selection_path, selection)

    with pytest.raises(SeedIdentityError, match="selection (size|SHA-256) mismatch"):
        read_ariadne_task_map(iter_dir, expected_iteration=1)


def test_ariadne_manifest_rejects_tampered_trajectory(tmp_path):
    iter_dir, _, _ = _write_canonical_handoff(tmp_path)
    trajectory = (
        iter_dir
        / "ariadne"
        / "seeds"
        / "seed-000001"
        / "trajectory"
        / "trajectory.xyz"
    )
    trajectory.write_text(
        trajectory.read_text(encoding="utf-8") + "# tamper\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="(size|hash) mismatch"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=1)


def test_ariadne_manifest_rejects_noncanonical_orphan_seed_directory(tmp_path):
    iter_dir, _, _ = _write_canonical_handoff(tmp_path)
    (active_ariadne_dir(iter_dir) / "seeds" / "seed_0001").mkdir()

    with pytest.raises(HandoffManifestError, match="invalid seed directory name"):
        read_ariadne_results_manifest(iter_dir, expected_iteration=1)


def test_read_ariadne_manifest_wraps_malformed_iteration(tmp_path):
    iter_dir = tmp_path / "iteration-000001"
    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "iteration": "not-an-int",
        "accepted": [],
        "rejected": [],
    })

    with pytest.raises(HandoffManifestError, match="ARIADNE results iteration"):
        read_ariadne_results_manifest(
            iter_dir,
            expected_iteration=1,
            require_nonempty=False,
        )
