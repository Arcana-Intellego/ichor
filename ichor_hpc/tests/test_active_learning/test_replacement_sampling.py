"""Replacement-sample producer/consumer contract tests."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ichor.core.atoms import Atom, Atoms
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.input_staging import stage_gaussian_inputs
from ichor.hpc.active_learning.daemon.scheduler_contracts import (
    infer_expected_tasks_from_artifacts,
)
from ichor.hpc.active_learning.daemon.recovery_contracts import (
    phase_recovery_contract_error,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state,
)
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
    _bootstrap_frames,
    _write_xyz,
    ensure_replacement_sample_strict,
    inspect_replacement_sample_recovery,
    prepare_replacement_round,
    read_replacement_sample,
    read_replacement_sample_strict,
    replacement_round_dir,
)
from ichor.hpc.active_learning.versioning.manifest import sha256_file


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
        "schema_version": 2,
        "context": "active",
        "iteration": 1,
        "replacement_round": 1,
        "sample_xyz": {
            "path": "replacement-SAMPLE.xyz",
            "size": sample.stat().st_size,
            "sha256": sha256_file(sample),
        },
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
    pool = TrajectoryPool.load(campaign)
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
            {
                "candidate_id": "reserve",
                "frame_id": 1,
                "reserve_rank": 0,
                "pool_sha256": pool.sha256,
            }
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
        _active_frame(
            {
                "result_json": str(result),
                "result_sha256": sha256_file(result),
            }
        )


def test_active_replacement_rejects_tampered_result(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "atom_types": ["H"],
                "final_coordinates": [[0.0, 0.0, 0.0]],
                "landing_safety": {"accepted": True},
            }
        ),
        encoding="utf-8",
    )
    declared = sha256_file(result)
    result.write_text(result.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _active_frame(
            {
                "result_json": str(result),
                "result_sha256": declared,
                "landing_safety": {"accepted": True},
            }
        )


def test_active_replacement_rejects_unsafe_landing(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "atom_types": ["H"],
                "final_coordinates": [[0.0, 0.0, 0.0]],
                "landing_safety": {"accepted": False},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not explicitly safe"):
        _active_frame(
            {
                "result_json": str(result),
                "result_sha256": sha256_file(result),
                "landing_safety": {"accepted": False},
            }
        )


def test_bootstrap_replacement_rejects_pool_drift(tmp_path):
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    source = tmp_path / "source.xyz"
    source.write_text(
        "1\nframe 0\nH 0.0 0.0 0.0\n",
        encoding="utf-8",
        newline="\n",
    )
    TrajectoryPool.import_from(source, campaign, overwrite=True)

    with pytest.raises(ValueError, match="different trajectory pool"):
        _bootstrap_frames(
            campaign,
            [
                {
                    "frame_id": 0,
                    "pool_sha256": "0" * 64,
                }
            ],
        )


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
    assert loaded["records"][0]["pointdir_index"] == 1


def test_missing_replacement_sample_is_classified_and_rebuilt_without_reallocation(
    tmp_path,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    allocation_path, _payload = _canonical_bootstrap_replacement(campaign)
    round_dir = replacement_round_dir(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
    )
    (round_dir / "replacement-SAMPLE.xyz").unlink()
    (round_dir / "REPLACEMENT_SAMPLE.json").unlink()
    before = allocation_path.read_bytes()

    inspected = inspect_replacement_sample_recovery(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
        expected_campaign_uid="replacement-test",
    )

    assert inspected["state"] == "missing_rebuildable"
    assert inspected["n_candidates"] == 1
    assert not (round_dir / "replacement-SAMPLE.xyz").exists()
    assert not (round_dir / "REPLACEMENT_SAMPLE.json").exists()

    rebuilt = ensure_replacement_sample_strict(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
        expected_campaign_uid="replacement-test",
    )

    assert rebuilt["n_candidates"] == 1
    assert allocation_path.read_bytes() == before
    assert (round_dir / "replacement-SAMPLE.xyz").is_file()
    assert (round_dir / "REPLACEMENT_SAMPLE.json").is_file()


def test_partial_replacement_sample_rebuild_preserves_exact_surviving_file(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _allocation_path, _payload = _canonical_bootstrap_replacement(campaign)
    round_dir = replacement_round_dir(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
    )
    sample = round_dir / "replacement-SAMPLE.xyz"
    manifest = round_dir / "REPLACEMENT_SAMPLE.json"
    sample_before = sample.read_bytes()
    manifest.unlink()

    inspected = inspect_replacement_sample_recovery(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
        expected_campaign_uid="replacement-test",
    )
    rebuilt = ensure_replacement_sample_strict(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
        expected_campaign_uid="replacement-test",
    )

    assert inspected["state"] == "partial_rebuildable"
    assert sample.read_bytes() == sample_before
    assert manifest.is_file()
    assert rebuilt["n_candidates"] == 1


def test_conflicting_partial_replacement_sample_is_not_overwritten(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _allocation_path, _payload = _canonical_bootstrap_replacement(campaign)
    round_dir = replacement_round_dir(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
    )
    sample = round_dir / "replacement-SAMPLE.xyz"
    manifest = round_dir / "REPLACEMENT_SAMPLE.json"
    manifest.unlink()
    sample.write_text(
        sample.read_text(encoding="utf-8").replace(
            "0.100000000000",
            "0.200000000000",
        ),
        encoding="utf-8",
        newline="\n",
    )
    conflicting = sample.read_bytes()

    inspected = inspect_replacement_sample_recovery(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
        expected_campaign_uid="replacement-test",
    )

    assert inspected["state"] == "conflicting"
    with pytest.raises(ValueError, match="conflicts with pending allocation"):
        ensure_replacement_sample_strict(
            campaign,
            context="bootstrap",
            iteration=0,
            replacement_round=1,
            expected_campaign_uid="replacement-test",
        )
    assert sample.read_bytes() == conflicting
    assert not manifest.exists()


def test_allocation_check_contract_accepts_missing_exactly_rebuildable_sample(
    tmp_path,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _allocation_path, _payload = _canonical_bootstrap_replacement(campaign)
    round_dir = replacement_round_dir(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
    )
    (round_dir / "replacement-SAMPLE.xyz").unlink()
    (round_dir / "REPLACEMENT_SAMPLE.json").unlink()
    state = fresh_campaign_state(campaign_uid="replacement-test")
    state.phase = CampaignPhase.INITIAL_ALLOCATION_CHECK
    state.iteration = 0
    state.replacement_round = 1

    assert phase_recovery_contract_error(campaign, state) is None

    state.replacement_round = 2
    assert "does not match pending point allocation" in str(
        phase_recovery_contract_error(campaign, state)
    )


@pytest.mark.parametrize("verification", ["metadata", "authority", "deep"])
def test_replacement_gaussian_recovery_contract_accepts_strict_sample(
    tmp_path,
    verification,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _canonical_bootstrap_replacement(campaign)
    state = fresh_campaign_state(campaign_uid="replacement-test")
    state.phase = CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN
    state.iteration = 0
    state.replacement_round = 1

    assert phase_recovery_contract_error(
        campaign,
        state,
        verification=verification,
    ) is None


def test_replacement_gaussian_staging_preserves_strict_sample(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _allocation_path, _payload = _canonical_bootstrap_replacement(campaign)
    round_dir = replacement_round_dir(
        campaign,
        context="bootstrap",
        iteration=0,
        replacement_round=1,
    )
    sample = round_dir / "replacement-SAMPLE.xyz"
    manifest = round_dir / "REPLACEMENT_SAMPLE.json"
    before = {
        sample: sample.read_bytes(),
        manifest: manifest.read_bytes(),
    }

    staging, count = stage_gaussian_inputs(
        campaign,
        CampaignConfig(),
        "INITIAL_REPLACEMENT_GAUSSIAN",
        0,
        sample,
        campaign_uid="replacement-test",
    )
    repeated_staging, repeated_count = stage_gaussian_inputs(
        campaign,
        CampaignConfig(),
        "INITIAL_REPLACEMENT_GAUSSIAN",
        0,
        sample,
        campaign_uid="replacement-test",
    )

    assert staging == round_dir
    assert repeated_staging == round_dir
    assert count == repeated_count == 1
    assert (round_dir / "POINT_0001.pointdir" / "input.gjf").is_file()
    assert {path: path.read_bytes() for path in before} == before


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


def test_replacement_sample_reader_rejects_tampered_xyz(tmp_path):
    round_dir, _allocation_path, _allocation, _payload = _replacement_fixture(tmp_path)
    sample = round_dir / "replacement-SAMPLE.xyz"
    sample.write_text(
        sample.read_text(encoding="utf-8").replace("0.0 0.0 0.0", "0.1 0.0 0.0"),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_replacement_sample(round_dir)


def test_replacement_sample_reader_rejects_fractional_integer(tmp_path):
    round_dir, _allocation_path, _allocation, payload = _replacement_fixture(tmp_path)
    payload["replacement_round"] = 1.5
    (round_dir / "REPLACEMENT_SAMPLE.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact JSON integer"):
        read_replacement_sample(round_dir)


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
    if phase_name == "INITIAL_REPLACEMENT_AIMALL":
        from ichor.hpc.active_learning.daemon.input_staging import (
            write_quantum_acceptance_manifest,
        )

        write_quantum_acceptance_manifest(
            round_dir,
            phase_name="INITIAL_REPLACEMENT_GAUSSIAN",
            iteration=0,
            accepted=[pointdir],
            rejected=[],
        )
    state = fresh_campaign_state()
    state.replacement_round = 1
    state.phase = CampaignPhase(phase_name)
    daemon = Daemon(campaign_dir=campaign, config=CampaignConfig())

    assert daemon._infer_expected_tasks_from_artifacts(state, state.phase) == 1


def test_replacement_aimall_task_count_uses_filtered_gaussian_handoff(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import quantum_task_contracts

    staging = tmp_path / "replacement_round_0001"
    staging.mkdir()

    def contract(_campaign, phase_name, _iteration, **_kwargs):
        if phase_name == "REPLACEMENT_GAUSSIAN":
            names = ("POINT_0004.pointdir", "POINT_0005.pointdir")
        elif phase_name == "REPLACEMENT_AIMALL":
            names = ("POINT_0004.pointdir",)
        else:
            raise AssertionError("unexpected phase " + str(phase_name))
        return SimpleNamespace(
            staging_dir=staging,
            pointdir_names=names,
            logical_total=len(names),
        )

    monkeypatch.setattr(
        quantum_task_contracts,
        "quantum_task_contract",
        contract,
    )

    assert infer_expected_tasks_from_artifacts(
        tmp_path,
        phase="REPLACEMENT_AIMALL",
        iteration=3,
        replacement_round=1,
    ) == 1


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
