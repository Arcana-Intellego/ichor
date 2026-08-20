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
    inventory_quarantine_authority,
    prepare_quarantine_manifest,
    quarantine_root,
    retain_ariadne_transaction_residue,
    write_quarantine_manifest,
)
from ichor.hpc.active_learning.daemon.lease import evaluate_lease_liveness
from ichor.hpc.active_learning.daemon.state import atomic_write_json
from ichor.hpc.active_learning.handoff_manifests import (
    build_seed_selection_manifest,
    seeds_picked_path,
)
from ichor.hpc.active_learning.layout import active_iteration_dir
from ichor.hpc.active_learning.seed_identity import write_ariadne_task_map


_CAMPAIGN_UID = "ariadne-quarantine-durability"


def _write_ariadne_task_map(
    campaign: Path,
    *,
    iteration: int = 4,
    n_tasks: int = 1,
):
    iter_dir = active_iteration_dir(campaign, iteration)
    selection = build_seed_selection_manifest(
        campaign_uid=_CAMPAIGN_UID,
        campaign_random_seed=1,
        iteration=iteration,
        models_version=iteration - 1,
        model_manifest_sha256="a" * 64,
        model_set_sha256="b" * 64,
        trajectory_sha256="c" * 64,
        selection_strategy="hybrid_variance",
        seed_records=[
            {
                "seed_id": index + 1,
                "frame_id": 10 + index,
                "pool_row_index_zero_based": 9 + index,
                "selection_origin": "bulk",
                "variance_at_selection": 0.5,
            }
            for index in range(n_tasks)
        ],
    )
    selection_path = seeds_picked_path(iter_dir)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, selection)
    write_ariadne_task_map(iter_dir, selection)
    from ichor.hpc.active_learning.seed_identity import read_ariadne_task_map

    return iter_dir, read_ariadne_task_map(
        iter_dir,
        expected_iteration=iteration,
    )


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
    source.mkdir(parents=True)
    (source / "result.json").write_bytes(b"new retry output\n")

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


def test_residue_move_replay(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import ariadne_quarantine

    campaign = tmp_path / "campaign"
    iter_dir, task_map = _write_ariadne_task_map(campaign)
    residue = (
        iter_dir
        / "ariadne"
        / "seeds"
        / ".seed-000001.partial-task-0-pid-12"
    )
    residue.mkdir(parents=True)
    (residue / "partial.json").write_bytes(b"partial\n")
    changed_task_map = {
        **task_map,
        "tasks": [dict(task) for task in task_map["tasks"]],
    }
    changed_task_map["tasks"][0]["seed_uid"] = "changed-after-classification"
    with pytest.raises(AriadneQuarantineError, match="changed after classification"):
        retain_ariadne_transaction_residue(
            campaign,
            iteration=4,
            task_map=changed_task_map,
            campaign_uid=_CAMPAIGN_UID,
            authority_identity="completion-receipt-a",
        )
    real_replace = ariadne_quarantine.os.replace

    def interrupt_residue_move(source, target):
        if Path(source) == residue:
            raise OSError("injected residue move interruption")
        return real_replace(source, target)

    monkeypatch.setattr(
        ariadne_quarantine.os,
        "replace",
        interrupt_residue_move,
    )
    with pytest.raises(OSError, match="injected residue move interruption"):
        retain_ariadne_transaction_residue(
            campaign,
            iteration=4,
            task_map=task_map,
            campaign_uid=_CAMPAIGN_UID,
            authority_identity="completion-receipt-a",
        )

    authority = inventory_quarantine_authority(campaign)
    assert authority["errors"] == []
    assert authority["attempts"][0]["status"] == "prepared"
    assert residue.is_dir()

    target = Path(authority["attempts"][0]["attempt_path"]) / residue.name
    shutil.copytree(residue, target)
    with pytest.raises(
        AriadneQuarantineError,
        match="exists at source and target",
    ):
        retain_ariadne_transaction_residue(
            campaign,
            iteration=4,
            task_map=task_map,
            campaign_uid=_CAMPAIGN_UID,
            authority_identity="completion-receipt-a",
        )
    shutil.rmtree(target)

    monkeypatch.setattr(ariadne_quarantine.os, "replace", real_replace)
    replay = retain_ariadne_transaction_residue(
        campaign,
        iteration=4,
        task_map=task_map,
        campaign_uid=_CAMPAIGN_UID,
        authority_identity="completion-receipt-a",
    )

    assert replay["changed"] is True
    assert replay["residue_count"] == 1
    assert not residue.exists()
    assert Path(replay["retained_paths"][0]).is_dir()
    assert inventory_quarantine(campaign)["attempts"][0]["status"] == (
        "retained_failure"
    )


def test_residue_later_attempt(tmp_path):
    campaign = tmp_path / "campaign"
    iter_dir, task_map = _write_ariadne_task_map(campaign)
    seeds = iter_dir / "ariadne" / "seeds"
    first = seeds / ".seed-000001.partial-task-0-pid-12"
    first.mkdir(parents=True)
    (first / "partial.json").write_bytes(b"first\n")
    retain_ariadne_transaction_residue(
        campaign,
        iteration=4,
        task_map=task_map,
        campaign_uid=_CAMPAIGN_UID,
        authority_identity="producer-attempt-a",
    )
    second = seeds / ".seed-000001.partial-task-0-pid-13"
    second.mkdir()
    (second / "partial.json").write_bytes(b"second\n")

    retained = retain_ariadne_transaction_residue(
        campaign,
        iteration=4,
        task_map=task_map,
        campaign_uid=_CAMPAIGN_UID,
        authority_identity="producer-attempt-b",
    )

    assert retained["seed_ids"] == [1]
    inventory = inventory_quarantine(campaign)
    assert inventory["errors"] == []
    assert len(inventory["attempts"]) == 2


def test_residue_multi_move_replay_and_final_noop(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.daemon import ariadne_quarantine

    campaign = tmp_path / "campaign"
    monkeypatch.setattr(
        ariadne_quarantine,
        "quarantine_root",
        lambda _campaign: campaign / ".Q",
    )
    iter_dir, task_map = _write_ariadne_task_map(campaign, n_tasks=2)
    seeds = iter_dir / "ariadne" / "seeds"
    first = seeds / ".seed-000001.partial-task-0-pid-12"
    second = seeds / ".seed-000002.partial-task-1-pid-13"
    for path, payload in ((first, b"first\n"), (second, b"second\n")):
        path.mkdir(parents=True, exist_ok=True)
        (path / "partial.json").write_bytes(payload)

    real_replace = ariadne_quarantine.os.replace

    def interrupt_second_move(source, target):
        if Path(source) == second:
            raise OSError("injected second residue move interruption")
        return real_replace(source, target)

    monkeypatch.setattr(
        ariadne_quarantine.os,
        "replace",
        interrupt_second_move,
    )
    with pytest.raises(OSError, match="injected second residue move interruption"):
        retain_ariadne_transaction_residue(
            campaign,
            iteration=4,
            task_map=task_map,
            campaign_uid=_CAMPAIGN_UID,
            authority_identity="completion-receipt-multi",
        )

    assert not first.exists()
    assert second.is_dir()
    monkeypatch.setattr(ariadne_quarantine.os, "replace", real_replace)
    replay = retain_ariadne_transaction_residue(
        campaign,
        iteration=4,
        task_map=task_map,
        campaign_uid=_CAMPAIGN_UID,
        authority_identity="completion-receipt-multi",
    )
    manifest_path = Path(replay["manifest_path"])
    manifest_bytes = manifest_path.read_bytes()

    final_replay = retain_ariadne_transaction_residue(
        campaign,
        iteration=4,
        task_map=task_map,
        campaign_uid=_CAMPAIGN_UID,
        authority_identity="completion-receipt-multi",
    )

    assert replay["changed"] is True
    assert replay["residue_count"] == 2
    assert final_replay["changed"] is False
    assert final_replay["residue_count"] == 2
    assert manifest_path.read_bytes() == manifest_bytes


def test_residue_pre_manifest_attempt_replays_safely(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.daemon import ariadne_quarantine

    campaign = tmp_path / "campaign"
    monkeypatch.setattr(
        ariadne_quarantine,
        "quarantine_root",
        lambda _campaign: campaign / ".Q",
    )
    iter_dir, task_map = _write_ariadne_task_map(campaign)
    residue = (
        iter_dir
        / "ariadne"
        / "seeds"
        / ".seed-000001.partial-task-0-pid-12"
    )
    residue.mkdir(parents=True)
    (residue / "partial.json").write_bytes(b"partial\n")
    real_prepare = ariadne_quarantine.prepare_quarantine_manifest

    def interrupt_before_manifest(*args, **kwargs):
        raise OSError("injected pre-manifest interruption")

    monkeypatch.setattr(
        ariadne_quarantine,
        "prepare_quarantine_manifest",
        interrupt_before_manifest,
    )
    with pytest.raises(OSError, match="injected pre-manifest interruption"):
        retain_ariadne_transaction_residue(
            campaign,
            iteration=4,
            task_map=task_map,
            campaign_uid=_CAMPAIGN_UID,
            authority_identity="completion-receipt-pre-manifest",
        )

    attempt = next((campaign / ".Q" / "iteration-000004").iterdir())
    (attempt / ".t-0123456789ab").write_bytes(b"partial manifest")
    monkeypatch.setattr(
        ariadne_quarantine,
        "prepare_quarantine_manifest",
        real_prepare,
    )
    replay = retain_ariadne_transaction_residue(
        campaign,
        iteration=4,
        task_map=task_map,
        campaign_uid=_CAMPAIGN_UID,
        authority_identity="completion-receipt-pre-manifest",
    )

    assert replay["changed"] is True
    assert replay["residue_count"] == 1
    assert not residue.exists()
    assert not (Path(replay["attempt_path"]) / ".t-0123456789ab").exists()


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
