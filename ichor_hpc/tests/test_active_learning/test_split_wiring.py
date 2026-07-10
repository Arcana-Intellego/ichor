"""SPLIT is a read-only projection of the exact pre-QM allocation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
from ichor.hpc.active_learning.daemon.state import fresh_campaign_state
from ichor.hpc.active_learning.point_allocation import (
    allocation_targets,
    create_point_allocation,
    point_allocation_path,
)


def _write_allocation(campaign_dir: Path, config: CampaignConfig, iteration: int):
    iter_dir = (
        campaign_dir / "7_ACTIVE_LEARNING"
        / f"iteration-{iteration:04d}"
    )
    targets = allocation_targets(config, "active")
    create_point_allocation(
        point_allocation_path(
            campaign_dir,
            context="active",
            iteration=iteration,
        ),
        campaign_uid="split-test",
        context="active",
        iteration=iteration,
        targets=targets,
        primary_candidates=[
            {"candidate_id": "candidate-" + str(index), "seed_index": index}
            for index in range(targets["total"])
        ],
        reserve_candidates=[],
    )
    return iter_dir


def test_inline_split_projects_exact_allocation_slots(tmp_path):
    cfg = CampaignConfig()
    cfg.point_allocation.batch_training_size = 2
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.seed_selection.n_seeds_per_iteration = 3
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path / "c", config=cfg)
    state = fresh_campaign_state(max_iterations=1)
    state.iteration = 0
    iter_dir = _write_allocation(tmp_path / "c", cfg, 0)

    ex._inline_split(state)

    sp = iter_dir / "split.json"
    assert sp.is_file()
    payload = json.loads(sp.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["strategy"] == "exact_pre_qm_point_allocation"
    assert payload["iteration"] == 0
    assert payload["targets"] == {
        "train": 2,
        "int_val": 1,
        "ext_val": 0,
        "total": 3,
    }
    assert [slot["slot_id"] for slot in payload["slots"]] == [0, 1, 2]
    assert [slot["split"] for slot in payload["slots"]] == [
        "train",
        "train",
        "int_val",
    ]
    assert all(slot["candidate_id"].startswith("candidate-") for slot in payload["slots"])


def test_inline_split_refuses_missing_allocation_manifest(tmp_path):
    cfg = CampaignConfig()
    cfg.point_allocation.batch_training_size = 3
    cfg.point_allocation.batch_internal_validation_size = 1
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path / "c", config=cfg)
    state = fresh_campaign_state(max_iterations=1)
    state.iteration = 0
    iter_dir = (
        tmp_path / "c"
        / "7_ACTIVE_LEARNING"
        / f"iteration-{0:04d}"
    )
    iter_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="point-allocation manifest missing"):
        ex._inline_split(state)
    assert not (iter_dir / "split.json").exists()
