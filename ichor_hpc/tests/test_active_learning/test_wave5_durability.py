"""Focused regression tests for Wave 5 daemon durability contracts."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from ichor.hpc.active_learning.daemon.ariadne_quarantine import (
    AriadneQuarantineError,
    clean_quarantine,
    ensure_quarantine_capacity,
    inventory_quarantine,
    prepare_quarantine_manifest,
    quarantine_root,
    write_quarantine_manifest,
)
from ichor.hpc.active_learning.daemon.lease import evaluate_lease_liveness


def _heartbeat(timestamp: float, token: str = "a" * 32):
    return {
        "schema_version": 2,
        "owner_token": token,
        "time": timestamp,
        "pid": 123,
        "host": "login.example",
        "phase": "ARIADNE_ARRAY",
        "iteration": 4,
    }


def test_future_lease_beyond_clock_skew_is_inconclusive_not_fresh():
    now = time.time()
    result = evaluate_lease_liveness(
        _heartbeat(now + 120.0),
        stale_seconds=900,
        clock_skew_tolerance_seconds=60,
        now=now,
    )

    assert result.disposition == "clock_skew"
    assert not result.fresh
    assert not result.stale


def test_ariadne_quarantine_inventory_and_explicit_clean(tmp_path):
    campaign = tmp_path / "campaign"
    root = quarantine_root(campaign)
    attempt = root / "iteration-000004" / "attempt-a"
    retained = attempt / "seed-000007"
    retained.mkdir(parents=True)
    (retained / "result.json").write_bytes(b"{}\n")
    source = campaign / "7_ACTIVE_LEARNING" / "iteration-000004" / "seed-000007"
    write_quarantine_manifest(
        campaign,
        attempt,
        iteration=4,
        source_paths=[source],
        target_paths=[retained],
    )

    inventory = inventory_quarantine(campaign)
    assert inventory["errors"] == []
    assert inventory["total_bytes"] == 3
    assert inventory["attempts"][0]["attempt_id"] == "attempt-a"

    removed = clean_quarantine(campaign, attempt_ids=["attempt-a"])
    assert removed == [str(attempt)]
    assert not root.exists()


def test_ariadne_quarantine_capacity_bounds_attempts_and_bytes(tmp_path):
    campaign = tmp_path / "campaign"
    root = quarantine_root(campaign)
    attempt = root / "iteration-000004" / "attempt-a"
    retained = attempt / "seed-000007"
    retained.mkdir(parents=True)
    (retained / "result.json").write_bytes(b"{}\n")
    write_quarantine_manifest(
        campaign,
        attempt,
        iteration=4,
        source_paths=[campaign / "source-a"],
        target_paths=[retained],
    )
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "result.json").write_bytes(b"data")

    with pytest.raises(AriadneQuarantineError, match="attempt limit"):
        ensure_quarantine_capacity(
            campaign,
            [incoming],
            max_attempts=1,
            max_bytes=100,
        )
    with pytest.raises(AriadneQuarantineError, match="byte limit"):
        ensure_quarantine_capacity(
            campaign,
            [incoming],
            max_attempts=2,
            max_bytes=6,
        )


def test_interrupted_prepared_quarantine_is_inventoryable_and_cleanable(tmp_path):
    campaign = tmp_path / "campaign"
    source_a = campaign / "ACTIVE_LEARNING" / "seed-a"
    source_b = campaign / "ACTIVE_LEARNING" / "seed-b"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)
    (source_a / "result.json").write_bytes(b"aaa")
    (source_b / "result.json").write_bytes(b"bbbb")
    attempt = quarantine_root(campaign) / "iteration-000004" / "attempt-b"
    attempt.mkdir(parents=True)
    target_a = attempt / "seed-a"
    target_b = attempt / "seed-b"
    prepare_quarantine_manifest(
        campaign,
        attempt,
        iteration=4,
        source_paths=[source_a, source_b],
        target_paths=[target_a, target_b],
    )
    shutil.move(str(source_a), str(target_a))

    inventory = inventory_quarantine(campaign)
    assert inventory["errors"] == []
    assert inventory["attempts"][0]["status"] == "prepared"
    assert inventory["attempts"][0]["verified_bytes"] == 3
    assert inventory["attempts"][0]["pending_sources"] == 1
    with pytest.raises(AriadneQuarantineError, match="interrupted prepared"):
        ensure_quarantine_capacity(campaign, [source_b])

    clean_quarantine(campaign, attempt_ids=["attempt-b"])
    assert source_b.is_dir()
    assert not target_a.exists()
    assert not quarantine_root(campaign).exists()
