"""Smoke tests for the ariadne_runner main(argv) composition.

These tests exercise the argument-validation and early-exit branches
of main. The actual ARIADNE descent (which needs the oneAPI .so) is
not invoked here -- those tests live in the cluster-side smoke
documented under examples/csf4_first_live_iter.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


MODULE = "ichor.hpc.active_learning.acquisition.ariadne_runner"


def _run(args, **kw):
    return subprocess.run(
        [sys.executable, "-m", MODULE] + args,
        capture_output=True, text=True, timeout=60, **kw,
    )


def test_main_bad_campaign_dir_exits_2(tmp_path):
    """missing campaign-dir -> exit 2."""
    bad = tmp_path / "no_such_dir"
    result = _run(["--seed-index", "0",
                   "--iteration", "0",
                   "--campaign-dir", str(bad)])
    assert result.returncode == 2
    assert "campaign-dir does not exist" in result.stderr


def test_main_missing_campaign_yaml_exits_2(tmp_path):
    """directory exists but campaign.yaml is missing -> exit 2."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    result = _run(["--seed-index", "0",
                   "--iteration", "0",
                   "--campaign-dir", str(campaign)])
    assert result.returncode == 2
    assert "campaign.yaml not found" in result.stderr


def test_main_missing_state_exits_3(tmp_path):
    """campaign.yaml exists but state.json does not -> exit 3.
    means SEED_SELECT has not run, no iteration to drive ARIADNE in.
    """
    from ichor.hpc.active_learning.config import CampaignConfig
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.to_yaml(campaign / "campaign.yaml")
    result = _run(["--seed-index", "0",
                   "--iteration", "0",
                   "--campaign-dir", str(campaign)])
    assert result.returncode == 3
    # the error message should mention something about the missing
    # state or trajectory pool or models. all three are reasons we
    # might bail with exit 3.
    assert (
        "trajectory pool" in result.stderr
        or "state.json" in result.stderr
        or "models" in result.stderr
    )
