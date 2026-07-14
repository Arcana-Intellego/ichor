"""Cluster-side smoke harness for the first live CSF4 iteration.

These tests document the procedure from examples/csf4_first_live_iter/
as runnable assertions. They are all marked live, which means CI runners
skip them; an operator on CSF4 with sbatch + Gaussian + AIMAll + FEREBUS
+ ARIADNE all available on PATH can run them explicitly with:

    pytest -m live -q ichor_hpc/tests/test_active_learning/test_cluster_smoke_procedure.py

These are not unit tests in the usual sense -- they exercise the actual
cluster pipeline and can take an hour or more to complete. Treat them
as the canonical verification that a fresh CSF4 install is wired up
correctly before launching a real campaign.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


EXAMPLE_DIR = (
    Path(__file__).resolve().parents[3]
    / "examples" / "csf4_first_live_iter"
)


@pytest.mark.live
def test_csf4_example_dir_exists_and_has_required_files():
    """Sanity: the example dir from Phase E.3 is laid out as the
    README assumes. Fast even on a CI runner -- it does not touch the
    cluster -- but live-marked so it stays grouped with the others.
    """
    assert EXAMPLE_DIR.is_dir(), str(EXAMPLE_DIR)
    assert (EXAMPLE_DIR / "README.md").is_file()
    assert (EXAMPLE_DIR / "campaign.yaml").is_file()


@pytest.mark.live
def test_required_cluster_backends_present():
    """Check that every backend the live daemon needs is on PATH.
    Mirrors the preflight check the daemon itself runs at start.
    """
    from ichor.hpc.active_learning.daemon.preflight import check_backends
    avail = check_backends()
    missing = avail.missing
    if missing:
        pytest.skip(
            "missing cluster backends: " + repr(missing)
            + ". install them or load the right module before running this file."
        )


@pytest.mark.live
@pytest.mark.slow
def test_csf4_first_iteration_end_to_end(tmp_path):
    """The real thing. Drives the procedure from the README from a
    fresh tmpdir. Takes about an hour on water-tetramer with the example
    config; budget two hours before assuming it has hung.

    Steps mirrored from the README section 4 onwards:
      1. copy a water-tetramer trajectory into the tmp campaign dir.
      2. ichor-al-daemon init.
      3. copy the example campaign.yaml in.
      4. ichor-al-daemon start --mode live --max-ticks 2000.
      5. assert state.phase == DONE, reference_data_version >= 1.
    """
    # check preflight one more time so we fail fast if the user did not
    # set up the environment properly before invoking pytest -m live.
    from ichor.hpc.active_learning.daemon.preflight import check_backends
    avail = check_backends()
    if not avail.all_present:
        pytest.skip("missing backends: " + repr(avail.missing))

    fixture_pool = (
        Path(__file__).resolve().parent
        / "fixtures" / "water_tetramer.xyz"
    )
    if not fixture_pool.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")

    campaign = tmp_path / "ichor_live_smoke"
    campaign.mkdir()
    shutil.copy(fixture_pool, campaign / "pool.xyz")
    shutil.copy(EXAMPLE_DIR / "campaign.yaml", campaign / "campaign.yaml")

    # init
    rc = subprocess.run(
        ["ichor-al-daemon", "init",
         "--campaign-dir", str(campaign),
         "--source", str(campaign / "pool.xyz")],
        capture_output=True, text=True, timeout=300,
    )
    assert rc.returncode == 0, rc.stderr

    # Start live mode with a two-hour test budget.
    rc = subprocess.run(
        ["ichor-al-daemon", "start", "--mode", "live",
         "--campaign-dir", str(campaign),
         "--max-ticks", "2000"],
        capture_output=True, text=True, timeout=2 * 60 * 60,
    )
    assert rc.returncode == 0, rc.stderr[-2000:]

    # check final state
    from ichor.hpc.active_learning.daemon.state import read_state
    state_path = (
        campaign / ".DATA"
        / "ACTIVE_LEARNING" / "state.json"
    )
    state = read_state(state_path)
    assert str(state.phase) == "CampaignPhase.DONE"
    assert state.reference_data_version >= 1
    assert state.models_version >= 1
    # reference scales should be populated with five real-float keys
    assert state.reference_scales is not None
    expected_keys = {"energy", "force", "omega", "anh", "anh_std"}
    assert set(state.reference_scales.keys()) == expected_keys
