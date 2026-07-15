"""Tests for the live executor reference-scales override.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.phase_executor import BackendSubmissionError
from ichor.hpc.active_learning.daemon.live_executor import LiveBackendsPhaseExecutor
from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
from ichor.hpc.active_learning.reference_scale_snapshot import (
    read_reference_scale_snapshot,
)


def _make_executor(tmp_path):
    return LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=CampaignConfig(),
        backend_check=False,
    )


def _state(
    iteration=0,
    models_version=-1,
    ref_scales=None,
    ref_iter=-1,
    ref_models_version=-1,
    ref_manifest_sha=None,
):
    return SimpleNamespace(
        iteration=iteration,
        models_version=models_version,
        reference_scales=ref_scales,
        reference_scales_iteration=ref_iter,
        reference_scales_models_version=ref_models_version,
        reference_scales_model_manifest_sha256=ref_manifest_sha,
        campaign_uid="reference-scale-test",
    )


def test_live_override_distinct_from_dry():
    live_m = LiveBackendsPhaseExecutor._maybe_refresh_reference_scales
    dry_m = DryRunPhaseExecutor._maybe_refresh_reference_scales
    assert live_m is not dry_m


def test_no_models_committed_is_noop(tmp_path):
    ex = _make_executor(tmp_path)
    state = _state(iteration=0, models_version=-1)
    refreshed = ex._maybe_refresh_reference_scales(state)
    assert refreshed is False
    assert state.reference_scales is None
    assert state.reference_scales_iteration == -1


def test_missing_models_dir_halts_by_default(tmp_path):
    ex = _make_executor(tmp_path)
    state = _state(iteration=0, models_version=0)
    with pytest.raises(BackendSubmissionError, match="committed models"):
        ex._maybe_refresh_reference_scales(state)


def test_uniform_posterior_fallback_is_not_public_configuration():
    assert not hasattr(
        CampaignConfig().acquisition,
        "allow_uniform_posterior_fallback",
    )


def test_happy_path_writes_sidecar_and_journals(tmp_path):
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool

    ex = _make_executor(tmp_path)
    models_dir = ex.campaign_dir / "TRAINED_MODELS" / "iteration-000000"
    models_dir.mkdir(parents=True, exist_ok=True)
    from ichor.hpc.active_learning.versioning.trained_models import (
        TRAINED_MODEL_SET_FILENAME,
    )

    (models_dir / TRAINED_MODEL_SET_FILENAME).write_text(
        '{"test": true}\n',
        encoding="utf-8",
    )
    pool_xyz = ex.campaign_dir / "pool.xyz"
    pool_xyz.parent.mkdir(parents=True, exist_ok=True)
    xyz_content = "3" + chr(10)
    xyz_content += "frame 0" + chr(10)
    xyz_content += "O 0.0 0.0 0.0" + chr(10)
    xyz_content += "H 0.96 0.0 0.0" + chr(10)
    xyz_content += "H -0.24 0.93 0.0" + chr(10)
    pool_xyz.write_text(xyz_content, encoding="utf-8")
    TrajectoryPool.import_from(pool_xyz, ex.campaign_dir)
    fake_scales = {
        "energy": 0.001,
        "force": 0.5,
        "omega": 1500.0,
        "anh": 0.1,
        "anh_std": 0.02,
    }
    fake_acq = SimpleNamespace(reference_scales=fake_scales)
    target = "ichor.core.adversarial.acquisition.SeedLocalAdversarialAcquisition"
    contract = (
        "ichor.hpc.active_learning.daemon.artifact_contracts."
        "verify_committed_model_version"
    )
    with patch(contract, return_value=None):
        with patch(target, return_value=fake_acq):
            with patch(
                "ichor.hpc.active_learning.versioning.trained_models."
                "load_trained_models",
                return_value=(SimpleNamespace(), SimpleNamespace()),
            ):
                state = _state(iteration=1, models_version=0)
                refreshed = ex._maybe_refresh_reference_scales(state)
    assert refreshed is True
    assert state.reference_scales == fake_scales
    assert state.reference_scales_iteration == 1
    sidecar = (
        ex.campaign_dir / "ACTIVE_LEARNING"
        / "iteration-000001" / "protocol" / "reference_scales.json"
    )
    assert sidecar.is_file()
    persisted = read_reference_scale_snapshot(sidecar, expected_iteration=1)
    assert persisted["values"] == fake_scales
    assert persisted["source_iteration"] == 1
    assert persisted["models_version"] == 0
    assert state.reference_scales_model_manifest_sha256 == persisted[
        "model_set_manifest_sha256"
    ]
    journal = (
        ex.campaign_dir / ".DATA"
        / "ACTIVE_LEARNING" / "journal.ndjson"
    )
    assert journal.is_file()
    events = []
    for ln in journal.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            events.append(json.loads(ln))
    matching = [e for e in events if e.get("event") == "reference_scales_computed"]
    assert matching
    assert matching[-1]["n_keys"] == 5
    assert matching[-1]["models_version"] == 0


def test_dry_path_still_returns_synthetic_stub(tmp_path):
    cfg = CampaignConfig()
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path / "c", config=cfg)
    state = _state(iteration=1, models_version=0)
    refreshed = ex._maybe_refresh_reference_scales_dry(state)
    assert refreshed is True
    assert set(state.reference_scales.keys()) == {
        "energy", "force", "omega", "anh", "anh_std",
    }
    sidecar = (
        ex.campaign_dir
        / "ACTIVE_LEARNING"
        / "iteration-000001"
        / "protocol"
        / "reference_scales.json"
    )
    snapshot = read_reference_scale_snapshot(sidecar, expected_iteration=1)
    assert snapshot["values"] == state.reference_scales


def test_cached_scales_are_materialised_for_non_refresh_iteration(tmp_path):
    ex = _make_executor(tmp_path)
    scales = {
        "energy": 0.001,
        "force": 0.01,
        "omega": 1.0,
        "anh": 1.0,
        "anh_std": 1.0,
    }
    digest = "a" * 64
    state = _state(
        iteration=2,
        models_version=1,
        ref_scales=scales,
        ref_iter=1,
        ref_models_version=0,
        ref_manifest_sha=digest,
    )
    assert ex._maybe_refresh_reference_scales(state) is False
    sidecar = (
        ex.campaign_dir
        / "ACTIVE_LEARNING"
        / "iteration-000002"
        / "protocol"
        / "reference_scales.json"
    )
    snapshot = read_reference_scale_snapshot(sidecar, expected_iteration=2)
    assert snapshot["source_iteration"] == 1
    assert snapshot["models_version"] == 0
    assert snapshot["model_set_manifest_sha256"] == digest
