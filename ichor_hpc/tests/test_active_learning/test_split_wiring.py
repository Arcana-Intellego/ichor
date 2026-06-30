"""Verify _inline_split actually dispatches to the configured strategy.

Before the wiring change _inline_split wrote a fake split.json with
train_indices = list(range(floor)) regardless of what campaign.yaml said.
These tests confirm each of the three available strategies is honoured:
the chosen function is invoked and the resulting indices show up in
split.json on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
from ichor.hpc.active_learning.daemon.state import fresh_campaign_state


def _seed_pool_with_alphas(campaign_dir: Path, iteration: int, alphas):
    """Put a per-seed result.json file into the iteration pool so the
    inline split phase has data to chew on.
    """
    iter_dir = (
        campaign_dir / "7_ACTIVE_LEARNING"
        / f"iteration-{iteration:04d}"
    )
    pool_dir = iter_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    for i, alpha in enumerate(alphas):
        seed_dir = pool_dir / f"seed_{i:04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "result.json").write_text(
            json.dumps({"alpha_final": float(alpha)}),
            encoding="utf-8",
        )
    return iter_dir


@pytest.mark.parametrize(
    "strategy",
    ["stratified_with_holdout", "random_80_20", "pure_top_k"],
)
def test_inline_split_honours_strategy(tmp_path, strategy):
    cfg = CampaignConfig()
    cfg.split.strategy = strategy
    cfg.active_batch.final_batch_size = 2
    cfg.seed_selection.n_seeds_per_iteration = 6
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path / "c", config=cfg)
    state = fresh_campaign_state(max_iterations=1)
    state.iteration = 0
    iter_dir = _seed_pool_with_alphas(
        tmp_path / "c", 0,
        [3.0, 2.0, 5.0, 1.0, 4.0, 0.5],
    )
    ex._inline_split(state)
    sp = iter_dir / "split.json"
    assert sp.is_file()
    payload = json.loads(sp.read_text(encoding="utf-8"))
    # the recorded strategy is the registry key so it round-trips back through
    # get_split_strategy; the actual fraction travels in metadata.
    strategy_label = payload["strategy"]
    if strategy == "stratified_with_holdout":
        assert strategy_label == "stratified_with_holdout"
    elif strategy == "random_80_20":
        assert strategy_label == "random_80_20"
    elif strategy == "pure_top_k":
        assert strategy_label.startswith("pure_top_k")
    assert payload["iteration"] == 0
    # the train and val partitions should be non-empty and disjoint
    train = set(payload["train_indices"])
    val = set(payload["val_indices"])
    assert train and val
    assert not (train & val)


def test_inline_split_empty_pool_falls_back_to_floor(tmp_path):
    """with no seed_*/result.json files, the strategy has nothing to
    sort on. the inline split returns the floor-based stub layout so APPEND
    still has something to do. matches the previous behaviour exactly.
    """
    cfg = CampaignConfig()
    cfg.active_batch.final_batch_size = 4
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path / "c", config=cfg)
    state = fresh_campaign_state(max_iterations=1)
    state.iteration = 0
    iter_dir = (
        tmp_path / "c"
        / "7_ACTIVE_LEARNING"
        / f"iteration-{0:04d}"
    )
    iter_dir.mkdir(parents=True, exist_ok=True)
    ex._inline_split(state)
    payload = json.loads((iter_dir / "split.json").read_text(encoding="utf-8"))
    assert payload["train_indices"] == [0, 1, 2, 3]
    assert payload["val_indices"] == [4]
    assert payload["holdout_indices"] == []
