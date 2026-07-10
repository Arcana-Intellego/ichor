"""Replacement-sample producer/consumer contract tests."""
from __future__ import annotations

import json

import pytest

from ichor.hpc.active_learning.point_allocation import (
    allocation_manifest_sha256,
    create_point_allocation,
    pending_attempts,
    record_quantum_results,
)
from ichor.hpc.active_learning.replacement_sampling import (
    _active_frame,
    read_replacement_sample,
)


def _replacement_fixture(tmp_path):
    allocation_path = tmp_path / "allocation" / "POINT_ALLOCATION.json"
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="replacement-test",
        context="active",
        iteration=0,
        targets={"train": 1, "int_val": 0, "ext_val": 0, "total": 1},
        primary_candidates=[{"candidate_id": "candidate-primary"}],
        reserve_candidates=[],
    )
    round_dir = tmp_path / "staging" / "replacement_round_0001"
    round_dir.mkdir(parents=True)
    sample = round_dir / "replacement-SAMPLE.xyz"
    sample.write_text(
        "1\nreplacement frame 0\nH 0.0 0.0 0.0\n",
        encoding="utf-8",
        newline="\n",
    )
    payload = {
        "schema_version": 1,
        "context": "active",
        "iteration": 0,
        "replacement_round": 1,
        "sample_xyz": str(sample.resolve()),
        "n_candidates": 1,
        "records": [
            {
                "candidate_id": "candidate-replacement",
                "sample_index": 0,
                "pointdir_index": 1,
                "slot_id": 0,
                "split": "train",
                "round": 1,
            }
        ],
        "point_allocation_manifest": str(allocation_path.resolve()),
        "point_allocation_generation": 0,
        "point_allocation_sha256": allocation_manifest_sha256(allocation_path),
    }
    manifest = round_dir / "REPLACEMENT_SAMPLE.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return round_dir, allocation_path, allocation, payload


def test_replacement_sample_reader_validates_current_allocation(tmp_path):
    round_dir, _allocation_path, _allocation, _payload = _replacement_fixture(tmp_path)

    loaded = read_replacement_sample(round_dir, verify_allocation=True)

    assert loaded["n_candidates"] == 1
    assert loaded["records"][0]["split"] == "train"


def test_replacement_sample_reader_rejects_record_index_drift(tmp_path):
    round_dir, _allocation_path, _allocation, payload = _replacement_fixture(tmp_path)
    payload["records"][0]["sample_index"] = 1
    (round_dir / "REPLACEMENT_SAMPLE.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="indexes must be contiguous"):
        read_replacement_sample(round_dir)


def test_replacement_sample_reader_detects_allocation_generation_drift(tmp_path):
    round_dir, allocation_path, allocation, _payload = _replacement_fixture(tmp_path)
    attempt = pending_attempts(allocation)[0]
    record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": str(attempt["candidate_id"]),
                "accepted": False,
                "pointdir": "/synthetic/rejected.pointdir",
                "reason": "synthetic failure",
            }
        ],
    )

    with pytest.raises(ValueError, match="generation has changed"):
        read_replacement_sample(round_dir, verify_allocation=True)


def test_active_replacement_rejects_non_finite_coordinates(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "atom_types": ["H"],
                "final_coordinates": [[float("inf"), 0.0, 0.0]],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite coordinates"):
        _active_frame({"result_json": str(result)})
