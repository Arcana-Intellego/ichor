"""Smoke tests for the python -m entry points.

These confirm the two modules expose a runnable main(argv) shell so the
daemon-emitted sbatch scripts do not abort with ModuleNotFoundError when
an operator inspects them by hand. The bodies themselves are exercised
by later test files; here we only care that the shell parses --help and
rejects obviously-bad arguments cleanly.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


MODULES = [
    "ichor.hpc.active_learning.acquisition.ariadne_runner",
    "ichor.hpc.active_learning.sampling.diversity",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_python_m_help_exits_zero(module_name):
    """running python -m <module> --help should print usage and exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize("module_name", MODULES)
def test_python_m_no_args_fails_cleanly(module_name):
    """running with no args should error on the missing required flags"""
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "required" in (result.stdout + result.stderr).lower()


def test_ariadne_runner_missing_campaign_dir(tmp_path):
    """if --campaign-dir does not exist, the shell exits 2 cleanly.
    confirms the validation runs before any of the heavy ichor imports.
    """
    bad = tmp_path / "no_such_dir"
    result = subprocess.run(
        [sys.executable, "-m",
         "ichor.hpc.active_learning.acquisition.ariadne_runner",
         "--array-task-id", "0", "--iteration", "1",
         "--campaign-dir", str(bad)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "campaign-dir does not exist" in result.stderr


def test_diversity_runner_missing_campaign_dir(tmp_path):
    """same shape as ariadne -- bad --campaign-dir errors with exit 2."""
    bad = tmp_path / "no_such_dir"
    result = subprocess.run(
        [sys.executable, "-m",
         "ichor.hpc.active_learning.sampling.diversity",
         "--descriptor", "rmsd_massweight",
         "--iteration", "0",
         "--campaign-dir", str(bad)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "campaign-dir does not exist" in result.stderr
