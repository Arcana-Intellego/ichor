"""Focused regression tests for Wave 5 daemon durability contracts."""
from __future__ import annotations

import time
from pathlib import Path

from ichor.hpc.active_learning.daemon.ariadne_quarantine import (
    clean_quarantine,
    inventory_quarantine,
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
