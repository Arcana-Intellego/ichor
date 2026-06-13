"""Tests for ichor.hpc.active_learning.cli."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ichor.hpc.active_learning import cli as cli_mod
from ichor.hpc.active_learning.cli import build_parser, main
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import (
    DAEMON_LOCK_FILENAME,
    DEFAULT_DATA_SUBDIR,
)
from ichor.hpc.active_learning.daemon.journal import append_event
from ichor.hpc.active_learning.daemon.state import (
    DEFAULT_STATE_FILENAME,
    fresh_campaign_state,
    read_state,
    write_state,
)


def _campaign_with_config(tmp_path) -> Path:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    CampaignConfig(max_iterations=2).to_yaml(campaign / "campaign.yaml")
    return campaign


def test_build_parser_has_all_subcommands():
    p = build_parser()
    # Parse a known subcommand to confirm registration.
    args = p.parse_args(["start", "--campaign-dir", "x", "--mock-ariadne"])
    assert args.command == "start"
    assert args.mock_ariadne is True


def test_parser_rejects_missing_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_cli_preflight_prints_structured_backend_status(capsys, monkeypatch):
    from ichor.hpc.active_learning.daemon.preflight import BackendAvailability

    avail = BackendAvailability(
        profile=True,
        sbatch=True,
        sacct=True,
        gaussian=True,
        aimall=True,
        ferebus=True,
        ariadne=True,
        polus_rs=True,
        pyferebus=True,
        bc=True,
        gaussian_binary="jobscript:$g16root/g16/g16",
        sbatch_path="/usr/bin/sbatch",
        sacct_path="/usr/bin/sacct",
        bc_path="/usr/bin/bc",
        aimall_path="/home/user/AIMAll/aimqb.ish",
        ferebus_path="/home/user/.local/bin/ferebus",
        active_profile="csf3",
        profile_error="",
        python_executable="/home/user/.venv/ichor-al-csf3/bin/python",
    )
    monkeypatch.setattr(cli_mod, "check_backends", lambda: avail)

    rc = main(["preflight", "--campaign-dir", "."])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["active_profile"] == "csf3"
    assert payload["python_executable"].endswith("ichor-al-csf3/bin/python")


def test_cli_status_prints_state_json(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    s = fresh_campaign_state(max_iterations=5)
    s.iteration = 3
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, s)
    rc = main(["status", "--campaign-dir", str(campaign)])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["iteration"] == 3
    assert payload["max_iterations"] == 5
    assert payload["state_path"].endswith(DEFAULT_STATE_FILENAME)
    assert payload["lock_file_exists"] is False
    assert payload["lock_held"] is False
    assert "artifact_manifest_status" in payload


def test_cli_status_reports_stale_lock_file_as_not_held(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())
    (data / DAEMON_LOCK_FILENAME).write_text("stale\n", encoding="utf-8")

    rc = main(["status", "--campaign-dir", str(campaign)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lock_file_exists"] is True
    assert payload["lock_held"] is False


def test_cli_status_reports_actually_held_lock(tmp_path, capsys):
    pytest.importorskip("portalocker")
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())
    lock_path = data / DAEMON_LOCK_FILENAME

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import portalocker, sys, time\n"
                "lock = portalocker.Lock(sys.argv[1], mode='a', timeout=0, fail_when_locked=True)\n"
                "lock.acquire()\n"
                "print('ready', flush=True)\n"
                "time.sleep(10)\n"
                "lock.release()\n"
            ),
            str(lock_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        rc = main(["status", "--campaign-dir", str(campaign)])
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=5)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lock_file_exists"] is True
    assert payload["lock_held"] is True


def test_cli_status_surfaces_lock_probe_error(tmp_path, capsys, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())

    monkeypatch.setattr(
        cli_mod,
        "_probe_daemon_lock",
        lambda lock_path: {
            "lock_file_exists": True,
            "lock_held": None,
            "lock_probe_error": "RuntimeError: boom",
        },
    )
    rc = main(["status", "--campaign-dir", str(campaign)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lock_held"] is None
    assert payload["lock_probe_error"] == "RuntimeError: boom"


def test_cli_status_returns_4_when_state_missing(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["status", "--campaign-dir", str(campaign)])
    assert rc == 4


def test_cli_stop_sets_shutdown_flag(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, fresh_campaign_state())
    rc = main(["stop", "--campaign-dir", str(campaign)])
    assert rc == 0
    s = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert s.shutdown_requested is True


def test_cli_resume_explicitly_clears_shutdown_flag(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.shutdown_requested = True
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, state)
    rc = main([
        "resume", "--campaign-dir", str(campaign),
        "--mock-ariadne", "--max-ticks", "0",
    ])
    assert rc == 0
    s = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert s.shutdown_requested is False


def test_cli_stop_when_no_state_returns_4(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["stop", "--campaign-dir", str(campaign)])
    assert rc == 4


def test_cli_journal_prints_filtered_events(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(journal, "alpha", x=1)
    append_event(journal, "beta", x=2)
    append_event(journal, "alpha", x=3)
    rc = main(["journal", "--campaign-dir", str(campaign), "--event-type", "alpha"])
    assert rc == 0
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert len(lines) == 2
    for line in lines:
        payload = json.loads(line)
        assert payload["event"] == "alpha"


def test_cli_journal_returns_4_when_no_journal(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["journal", "--campaign-dir", str(campaign)])
    assert rc == 4


def test_cli_reconcile_writes_proposed_state(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["reconcile", "--campaign-dir", str(campaign)])
    assert rc == 0
    proposed = campaign / DEFAULT_DATA_SUBDIR / (DEFAULT_STATE_FILENAME + ".proposed")
    assert proposed.exists()
    captured = capsys.readouterr()
    assert "Proposed state written" in captured.out


def test_cli_start_with_mock_ariadne_drives_state_machine(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    rc = main([
        "start", "--campaign-dir", str(campaign),
        "--mock-ariadne", "--max-ticks", "5",
        "--poll-interval", "1",
    ])
    assert rc == 0
    # state.json should now exist and be parseable.
    state = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    # Within 5 ticks we should at least have advanced past INIT.
    assert state.phase.value != "INIT"


def test_cli_start_without_mode_refuses(tmp_path, capsys):
    """With no --live / --dry-run / --mock-ariadne flag the CLI must refuse
    rather than silently pick a default. Exit 3 + a message listing the
    three available modes."""
    campaign = _campaign_with_config(tmp_path)
    rc = main(["start", "--campaign-dir", str(campaign)])
    assert rc == 3
    captured = capsys.readouterr()
    assert "no execution mode selected" in captured.err
    # Each available mode is named in the help text.
    assert "--live" in captured.err
    assert "--dry-run" in captured.err
    assert "--mock-ariadne" in captured.err


def test_cli_start_with_mutually_exclusive_flags_refuses(tmp_path, capsys):
    """--live + --dry-run is a mutually-exclusive configuration; exit 2."""
    campaign = _campaign_with_config(tmp_path)
    rc = main(["start", "--campaign-dir", str(campaign), "--live", "--dry-run"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err


def test_cli_start_live_on_windows_refuses_with_exit_12(tmp_path, capsys):
    """When --live is requested but the backends are absent (the off-cluster
    case), the CLI must refuse with exit 12 and a message naming the missing
    binaries -- not silently spin a daemon."""
    campaign = _campaign_with_config(tmp_path)
    rc = main(["start", "--campaign-dir", str(campaign), "--live"])
    # On a CSF4 host with all binaries present this test would skip; in our
    # CI / Windows environment, the backends are absent and exit 12 is the
    # expected refusal code.
    from ichor.hpc.active_learning.daemon.preflight import check_backends
    if check_backends().all_present:
        import pytest
        pytest.skip("all live backends present; refusal path not exercised here")
    assert rc == 12
    captured = capsys.readouterr()
    assert "backends are not available" in captured.err
