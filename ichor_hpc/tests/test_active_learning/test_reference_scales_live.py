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


def _make_executor(tmp_path):
    return LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=CampaignConfig(),
        backend_check=False,
    )


def _state(iteration=0, models_version=-1, ref_scales=None, ref_iter=-1):
    return SimpleNamespace(
        iteration=iteration,
        models_version=models_version,
        reference_scales=ref_scales,
        reference_scales_iteration=ref_iter,
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


def test_missing_models_dir_bails_only_with_explicit_uniform_fallback(tmp_path):
    cfg = CampaignConfig()
    cfg.acquisition.allow_uniform_posterior_fallback = True
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )
    state = _state(iteration=0, models_version=0)
    assert ex._maybe_refresh_reference_scales(state) is False


def test_happy_path_writes_sidecar_and_journals(tmp_path):
    ex = _make_executor(tmp_path)
    models_dir = ex.campaign_dir / "TRAINED_MODELS" / "iteration-000000"
    models_dir.mkdir(parents=True, exist_ok=True)
    pool_xyz = ex.campaign_dir / ".DATA" / "TRAJECTORY" / "pool.xyz"
    pool_xyz.parent.mkdir(parents=True, exist_ok=True)
    xyz_content = "3" + chr(10)
    xyz_content += "frame 0" + chr(10)
    xyz_content += "O 0.0 0.0 0.0" + chr(10)
    xyz_content += "H 0.96 0.0 0.0" + chr(10)
    xyz_content += "H -0.24 0.93 0.0" + chr(10)
    pool_xyz.write_text(xyz_content, encoding="utf-8")
    import hashlib as _h
    sha = _h.sha256(pool_xyz.read_bytes()).hexdigest()
    manifest_path = pool_xyz.parent / "pool.manifest.json"
    manifest = {
        "schema_version": 1,
        "source_path": str(pool_xyz),
        "canonical_path": str(pool_xyz),
        "sha256": sha,
        "n_frames": 1,
        "natoms": 3,
        "atom_types": ["O", "H", "H"],
        "masses": [15.999, 1.008, 1.008],
        "imported_iso": "",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
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
                state = _state(iteration=0, models_version=0)
                refreshed = ex._maybe_refresh_reference_scales(state)
    assert refreshed is True
    assert state.reference_scales == fake_scales
    assert state.reference_scales_iteration == 0
    sidecar = (
        ex.campaign_dir / "7_ACTIVE_LEARNING"
        / "iteration-0000" / "reference_scales.json"
    )
    assert sidecar.is_file()
    persisted = json.loads(sidecar.read_text(encoding="utf-8"))
    assert persisted == fake_scales
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
    state = _state(iteration=0, models_version=0)
    refreshed = ex._maybe_refresh_reference_scales_dry(state)
    assert refreshed is True
    assert set(state.reference_scales.keys()) == {
        "energy", "force", "omega", "anh", "anh_std",
    }
