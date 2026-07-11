"""Replacement-sample producer/consumer contract tests."""
from __future__ import annotations

import json

import pytest

from ichor.core.atoms import Atom, Atoms
from ichor.hpc.active_learning.point_allocation import (
    allocate_replacements,
    allocation_manifest_sha256,
    create_point_allocation,
    pending_attempts,
    point_allocation_path,
    record_quantum_results,
)
from ichor.hpc.active_learning.replacement_sampling import (
    _active_frame,
    _write_xyz,
    prepare_replacement_round,
    read_replacement_sample,
    read_replacement_sample_strict,
    replacement_round_dir,
)


def _replacement_fixture(tmp_path):
    allocation_path = tmp_path / "allocation" / "POINT_ALLOCATION.json"
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="replacement-test",
        context="active",
        iteration=1,
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
        "iteration": 1,
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


def _canonical_bootstrap_replacement(campaign):
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool

    source = campaign / "pool-source.xyz"
    source.write_text(
        "1\nframe 0\nH 0.0 0.0 0.0\n"
        "1\nframe 1\nH 0.1 0.0 0.0\n",
        encoding="utf-8",
        newline="\n",
    )
    TrajectoryPool.import_from(source, campaign, overwrite=True)
    allocation_path = point_allocation_path(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="replacement-test",
        context="bootstrap",
        iteration=0,
        targets={"train": 1, "int_val": 0, "ext_val": 0, "total": 1},
        primary_candidates=[{"candidate_id": "primary", "frame_id": 0}],
        reserve_candidates=[
            {"candidate_id": "reserve", "frame_id": 1, "reserve_rank": 0}
        ],
    )
    primary = pending_attempts(allocation)[0]
    rejected = record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": primary["candidate_id"],
                "accepted": False,
                "pointdir": "/synthetic/primary.pointdir",
                "reason": "fixture rejection",
            }
        ],
    )
    allocate_replacements(
        allocation_path,
        replacement_round=1,
        expected_generation=int(rejected["generation"]),
    )
    payload = prepare_replacement_round(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
    )
    return allocation_path, payload


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


def test_strict_replacement_reader_joins_current_pending_attempts(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _allocation_path, payload = _canonical_bootstrap_replacement(campaign)

    loaded = read_replacement_sample_strict(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
    )

    assert loaded["records"] == payload["records"]


def test_strict_replacement_reader_rejects_plausible_record_drift(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _allocation_path, payload = _canonical_bootstrap_replacement(campaign)
    round_dir = replacement_round_dir(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
    )
    payload["records"][0]["split"] = "int_val"
    (round_dir / "REPLACEMENT_SAMPLE.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="does not match allocation attempt"):
        read_replacement_sample_strict(
            campaign,
            context="bootstrap",
            iteration=0,
            replacement_round=1,
        )


@pytest.mark.parametrize(
    "phase_name",
    ["INITIAL_REPLACEMENT_GAUSSIAN", "INITIAL_REPLACEMENT_AIMALL"],
)
def test_daemon_infers_replacement_expected_tasks_from_exact_round(
    tmp_path,
    phase_name,
):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.state import CampaignPhase, fresh_campaign_state

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _canonical_bootstrap_replacement(campaign)
    round_dir = replacement_round_dir(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
    )
    pointdir = round_dir / "POINT_0001.pointdir"
    pointdir.mkdir()
    (round_dir / "POINTS.txt").write_text(
        str(pointdir.resolve()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    state = fresh_campaign_state()
    state.replacement_round = 1
    state.phase = CampaignPhase(phase_name)
    daemon = Daemon(campaign_dir=campaign, config=CampaignConfig())

    assert daemon._infer_expected_tasks_from_artifacts(state, state.phase) == 1


def test_replacement_xyz_writer_uses_atomic_text_helper(tmp_path, monkeypatch):
    calls = []

    def fake_atomic_write(path, text):
        calls.append((path, text))

    monkeypatch.setattr(
        "ichor.hpc.active_learning.replacement_sampling.atomic_write_text",
        fake_atomic_write,
    )
    _write_xyz(
        [Atoms([Atom("H", 0, 0, 0)])],
        tmp_path / "replacement-SAMPLE.xyz",
    )

    assert len(calls) == 1
    assert calls[0][0] == tmp_path / "replacement-SAMPLE.xyz"
    assert calls[0][1].endswith("\n")
