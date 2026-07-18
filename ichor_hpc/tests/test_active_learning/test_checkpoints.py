from __future__ import annotations

import os
from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
import ichor.hpc.active_learning.execution_identity as execution_identity_module
from ichor.hpc.active_learning.execution_identity import (
    _canonical_digest,
    _environment_fingerprint,
    ensure_execution_identity,
)
from ichor.hpc.active_learning.daemon import checkpoints
from ichor.hpc.active_learning.daemon.daemon import Daemon, TickStatus
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state,
    write_state,
)


def _idle_campaign(tmp_path: Path, monkeypatch) -> Path:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = CampaignConfig()
    config.to_yaml(campaign / "campaign.yaml")
    state = fresh_campaign_state(campaign_uid="abc123")
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = -1
    state.models_version = -1
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)

    def fake_generation(*args, campaign_uid, config, generation=0, **kwargs):
        generation_campaign = Path(args[0] if args else campaign).resolve()
        payload = {
            "schema_version": 1,
            "generation": int(generation),
            "campaign_uid": str(campaign_uid),
            "created_at_iso": "2026-01-01T00:00:00+00:00",
            "host": "test-host",
            "operator": "test-operator",
            "python_executable": str((Path.cwd() / "python-test").resolve()),
            "campaign_schema_version": int(config.schema_version),
            "python_version": "3.11.checkpoint-test",
            "ichor_git": {},
            "ichor_package_tree_sha256": "0" * 64,
            "dependencies": [],
            "pyferebus": {},
            "ariadne": {},
            "ferebus_executable": {},
            "machine_profile": {},
            "loaded_modules": [],
            "native_library_paths": {
                "LD_LIBRARY_PATH": "",
                "LIBRARY_PATH": "",
            },
            "campaign_config_sha256": "1" * 64,
            "config_lock_sha256": None,
            "campaign_dir": str(generation_campaign),
        }
        payload["environment_fingerprint_sha256"] = _environment_fingerprint(
            payload
        )
        payload["digest_sha256"] = _canonical_digest(payload)
        return payload

    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        fake_generation,
    )
    ensure_execution_identity(
        campaign,
        campaign_uid=state.campaign_uid,
        config=config,
        requested_mode="dry_run",
    )
    (campaign / "authoritative.bin").write_bytes(b"authoritative-bytes")
    scratch = campaign / ".DATA" / "SCRATCH" / "GAUSSIAN"
    scratch.mkdir(parents=True)
    (scratch / "temporary.bin").write_bytes(b"scratch")
    staging = campaign / ".DATA" / "STAGING"
    staging.mkdir(parents=True)
    (staging / "incomplete.bin").write_bytes(b"staging")
    monkeypatch.setattr(
        checkpoints,
        "verify_state_referenced_artifacts",
        lambda *_args, **_kwargs: None,
    )
    return campaign


def test_checkpoint_deduplicates_verifies_and_restores(tmp_path, monkeypatch):
    campaign = _idle_campaign(tmp_path, monkeypatch)
    destination = tmp_path / "checkpoint-store"
    destination.mkdir()

    created = checkpoints.create_checkpoint(campaign, destination)
    (Path(created["store"]) / "current.json").unlink()
    repeated = checkpoints.create_checkpoint(campaign, destination)

    assert created["ok"] is True
    assert repeated["manifest_sha256"] == created["manifest_sha256"]
    assert (Path(created["store"]) / "current.json").is_file()
    manifest = created["manifest"]
    relative_paths = {item["path"] for item in manifest["files"]}
    assert "authoritative.bin" in relative_paths
    assert ".DATA/ACTIVE_LEARNING/execution_identity.json" in relative_paths
    assert ".DATA/ACTIVE_LEARNING/environment_current.json" in relative_paths
    assert (
        ".DATA/ACTIVE_LEARNING/environment_generations/generation-000000.json"
        in relative_paths
    )
    assert not any(path.startswith(".DATA/SCRATCH/") for path in relative_paths)
    assert not any(path.startswith(".DATA/STAGING/") for path in relative_paths)
    object_paths = list((Path(created["store"]) / "objects").iterdir())
    assert len(object_paths) == len({item["sha256"] for item in manifest["files"]})

    with monkeypatch.context() as context:
        context.setattr(
            checkpoints,
            "_verify_object",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("authority status hashed a checkpoint object")
            ),
        )
        authority = checkpoints.checkpoint_authority_status(
            campaign,
            destination,
        )
    assert authority["status"] == "authority_verified"
    assert authority["current"]["iteration"] == 0

    status = checkpoints.checkpoint_status(campaign, destination)
    assert status["status"] == "verified"
    assert status["current"]["iteration"] == 0

    target = tmp_path / "restored"
    preview = checkpoints.restore_checkpoint(
        created["checkpoint"],
        target,
        apply=False,
    )
    assert preview["applied"] is False
    assert not target.exists()

    restored = checkpoints.restore_checkpoint(
        created["checkpoint"],
        target,
        apply=True,
    )
    assert restored["applied"] is True
    assert (target / "authoritative.bin").read_bytes() == b"authoritative-bytes"
    assert not (target / ".DATA" / "SCRATCH").exists()
    for record in manifest["files"]:
        restored_path = target.joinpath(*Path(record["path"]).parts)
        assert restored_path.stat().st_size == int(record["size"])
        assert checkpoints._sha256_file(restored_path) == record["sha256"]


def test_checkpoint_restore_rebuilds_excluded_ferebus_row_cache(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon.ferebus_row_cache import (
        FEREBUS_ROW_CACHE,
        read_feature_contract,
        row_cache_path,
    )
    from ichor.hpc.active_learning.daemon.input_staging import (
        commit_reference_data_delta,
    )
    from ichor.hpc.active_learning.daemon.state import read_state
    from ichor_hpc.tests.test_active_learning.test_reference_data_versioning import (
        _complete_allocation,
    )

    campaign = _idle_campaign(tmp_path, monkeypatch)
    _complete_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        first_frame_id=0,
    )
    commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state = read_state(state_path)
    state.reference_data_version = 0
    write_state(state_path, state)
    destination = tmp_path / "checkpoint-store"
    destination.mkdir()

    created = checkpoints.create_checkpoint(campaign, destination)
    assert not any(
        item["path"].startswith(".DATA/CACHE/")
        for item in created["manifest"]["files"]
    )
    target = tmp_path / "restored"
    checkpoints.restore_checkpoint(created["checkpoint"], target, apply=True)

    contract = read_feature_contract(target)
    cache_manifest = (
        row_cache_path(target, str(contract["contract_sha256"]), 0)
        / FEREBUS_ROW_CACHE
    )
    assert cache_manifest.is_file()
    restored_pointdir = next(
        (target / "QM_REFERENCE_DATA" / "iteration-000000").glob("*.pointdir")
    )
    assert os.access(restored_pointdir, os.W_OK)
    assert os.access(restored_pointdir / "input.wfn", os.W_OK)


def test_checkpoint_rejects_corrupt_object(tmp_path, monkeypatch):
    campaign = _idle_campaign(tmp_path, monkeypatch)
    destination = tmp_path / "checkpoint-store"
    destination.mkdir()
    created = checkpoints.create_checkpoint(campaign, destination)
    first = created["manifest"]["files"][0]
    object_path = Path(created["store"]) / "objects" / first["sha256"]
    object_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="object (size|digest) mismatch"):
        checkpoints.verify_checkpoint(created["checkpoint"])


def test_checkpoint_restore_requires_empty_target(tmp_path, monkeypatch):
    campaign = _idle_campaign(tmp_path, monkeypatch)
    destination = tmp_path / "checkpoint-store"
    destination.mkdir()
    created = checkpoints.create_checkpoint(campaign, destination)
    target = tmp_path / "target"
    target.mkdir()
    (target / "existing.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        checkpoints.restore_checkpoint(
            created["checkpoint"],
            target,
            apply=True,
        )


def test_checkpoint_publication_failure_leaves_no_partial_checkpoint(
    tmp_path,
    monkeypatch,
):
    campaign = _idle_campaign(tmp_path, monkeypatch)
    destination = tmp_path / "checkpoint-store"
    destination.mkdir()
    real_replace = checkpoints.os.replace

    def fail_checkpoint_publication(source, target):
        if Path(target).name == "iteration-000000" and Path(source).is_dir():
            raise OSError("injected checkpoint publication failure")
        return real_replace(source, target)

    monkeypatch.setattr(checkpoints.os, "replace", fail_checkpoint_publication)

    with pytest.raises(OSError, match="injected checkpoint publication failure"):
        checkpoints.create_checkpoint(campaign, destination)

    store = checkpoints.checkpoint_store(destination, "abc123")
    checkpoint_root = store / "checkpoints"
    assert not (checkpoint_root / "iteration-000000").exists()
    assert not list(checkpoint_root.glob(".iteration-000000.tmp.*"))


def test_checkpoint_restore_publication_failure_leaves_no_partial_target(
    tmp_path,
    monkeypatch,
):
    campaign = _idle_campaign(tmp_path, monkeypatch)
    destination = tmp_path / "checkpoint-store"
    destination.mkdir()
    created = checkpoints.create_checkpoint(campaign, destination)
    target = tmp_path / "restored"
    real_replace = checkpoints.os.replace

    def fail_restore_publication(source, destination_path):
        if Path(destination_path) == target and Path(source).name.startswith(
            ".restore-"
        ):
            raise OSError("injected restore publication failure")
        return real_replace(source, destination_path)

    monkeypatch.setattr(checkpoints.os, "replace", fail_restore_publication)
    with pytest.raises(OSError, match="injected restore publication failure"):
        checkpoints.restore_checkpoint(
            created["checkpoint"],
            target,
            apply=True,
        )

    assert not target.exists()
    assert not list(tmp_path.glob(".restore-*"))


def test_checkpoint_source_symlink_is_rejected(tmp_path, monkeypatch):
    campaign = _idle_campaign(tmp_path, monkeypatch)
    destination = tmp_path / "checkpoint-store"
    destination.mkdir()
    source = campaign / "authoritative.bin"
    link = campaign / "linked.bin"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("file symlinks are unavailable on this test host")

    with pytest.raises(ValueError, match="contains a symlink"):
        checkpoints.create_checkpoint(campaign, destination)


def test_manual_checkpoint_refuses_active_daemon_lease(tmp_path, monkeypatch):
    campaign = _idle_campaign(tmp_path, monkeypatch)
    destination = tmp_path / "checkpoint-store"
    destination.mkdir()
    lease = campaign / ".DATA" / "ACTIVE_LEARNING" / "daemon.lease.d"
    lease.mkdir()

    with pytest.raises(ValueError, match="daemon lease"):
        checkpoints.create_checkpoint(campaign, destination)


def test_required_automatic_checkpoint_failure_blocks_seed_selection(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = CampaignConfig()
    config.retention.checkpoint_destination = str(tmp_path / "checkpoint-store")
    config.retention.checkpoint_required = True
    state = fresh_campaign_state(campaign_uid="abc123")
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    daemon = Daemon(campaign_dir=campaign, config=config)
    reasons = []

    monkeypatch.setattr(
        checkpoints,
        "create_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("destination offline")),
    )
    monkeypatch.setattr(
        daemon,
        "_halt",
        lambda _state, _phase, reason: reasons.append(reason) or TickStatus.HALTED,
    )

    result = daemon._checkpoint_before_seed_selection(state)

    assert result == TickStatus.HALTED
    assert reasons and "destination offline" in reasons[0]
