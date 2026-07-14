from __future__ import annotations

import json

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.execution_identity import (
    ExecutionIdentityError,
    ensure_execution_identity,
    environment_current_path,
    environment_generations_dir,
    execution_identity_path,
)


def _fake_generation(*args, campaign_uid, config, generation=0, **kwargs):
    return {
        "schema_version": 1,
        "generation": int(generation),
        "campaign_uid": str(campaign_uid),
        "campaign_schema_version": int(config.schema_version),
        "digest_sha256": "a" * 64,
    }


def test_first_start_requires_explicit_mode(tmp_path):
    with pytest.raises(ExecutionIdentityError, match="first start requires --mode"):
        ensure_execution_identity(
            tmp_path,
            campaign_uid="uid-1",
            config=CampaignConfig(),
            requested_mode=None,
        )

    assert not execution_identity_path(tmp_path).exists()


def test_first_start_binds_mode_seed_and_environment_generation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    config.campaign.random_seed = 42

    mode, payload = ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-2",
        config=config,
        requested_mode="dry_run",
    )

    assert mode == "dry_run"
    assert payload["campaign_random_seed"] == 42
    assert execution_identity_path(tmp_path).is_file()
    assert (environment_generations_dir(tmp_path) / "generation-000000.json").is_file()
    current = json.loads(environment_current_path(tmp_path).read_text(encoding="utf-8"))
    assert current["generation"] == 0
    assert current["generation_digest_sha256"] == "a" * 64


def test_bound_mode_cannot_change(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-3",
        config=config,
        requested_mode="live",
    )

    with pytest.raises(ExecutionIdentityError, match="permanently bound to live"):
        ensure_execution_identity(
            tmp_path,
            campaign_uid="uid-3",
            config=config,
            requested_mode="dry_run",
        )


def test_bound_random_seed_cannot_change(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-seed",
        config=config,
        requested_mode="dry_run",
    )
    config.campaign.random_seed = 7

    with pytest.raises(ExecutionIdentityError, match="random_seed"):
        ensure_execution_identity(
            tmp_path,
            campaign_uid="uid-seed",
            config=config,
            requested_mode=None,
        )


def test_tampered_execution_identity_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-4",
        config=config,
        requested_mode="dry_run",
    )
    path = execution_identity_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mode"] = "live"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExecutionIdentityError, match="digest mismatch"):
        ensure_execution_identity(
            tmp_path,
            campaign_uid="uid-4",
            config=config,
            requested_mode=None,
        )
