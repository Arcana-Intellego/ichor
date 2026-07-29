from __future__ import annotations

import pytest

from ichor.hpc.active_learning import cli


def _write_campaign_yaml(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "campaign.yaml").write_text("schema_version: 14\n", encoding="utf-8")


def _parse(argv):
    parser = cli.build_parser()
    return parser.parse_args(cli.expand_boolean_short_flag_clusters(argv))


def test_resolve_campaign_dir_uses_cwd_when_campaign_yaml_exists(tmp_path, monkeypatch):
    _write_campaign_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert cli.resolve_campaign_dir(None) == tmp_path.resolve()


def test_resolve_campaign_dir_requires_campaign_yaml_for_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(cli.CampaignDirResolutionError) as exc:
        cli.resolve_campaign_dir(None)

    assert "current directory does not contain campaign.yaml" in str(exc.value)


def test_resolve_campaign_dir_explicit_path_overrides_cwd(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    explicit = tmp_path / "explicit"
    cwd.mkdir()
    _write_campaign_yaml(explicit)
    monkeypatch.chdir(cwd)

    assert cli.resolve_campaign_dir(str(explicit)) == explicit.resolve()


def test_status_remains_available_without_campaign_yaml(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert cli.main(["status"]) == 4

    captured = capsys.readouterr()
    assert "campaign file: campaign.yaml is missing" in captured.out
    assert "state.json" in captured.err


def test_status_without_campaign_dir_uses_cwd_campaign(tmp_path, monkeypatch, capsys):
    _write_campaign_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert cli.main(["status"]) == 4
    captured = capsys.readouterr()
    assert ".DATA" in captured.err
    assert "state.json" in captured.err


def test_start_short_flags_match_long_options():
    ns = _parse(
        ["start", "--mode", "dry_run", "-b", "-p", "3", "-t", "9"]
    )

    assert ns.mode == "dry_run"
    assert ns.background is True
    assert ns.poll_interval == 3
    assert ns.max_ticks == 9


def test_removed_execution_mode_short_flags_are_rejected():
    with pytest.raises(SystemExit):
        _parse(["start", "-l"])
    with pytest.raises(SystemExit):
        _parse(["start", "-d"])
    with pytest.raises(SystemExit):
        _parse(["start", "-m"])


def test_start_and_resume_foreground_short_flag_parse():
    assert _parse(["start", "-f"]).foreground is True
    assert _parse(["resume", "-f"]).foreground is True


def test_cluster_expansion_does_not_split_value_taking_options():
    with pytest.raises(cli.ShortFlagClusterError):
        cli.expand_boolean_short_flag_clusters(["start", "-cthing"])
    with pytest.raises(cli.ShortFlagClusterError):
        cli.expand_boolean_short_flag_clusters(["start", "-bt", "10"])


def test_main_reports_ambiguous_short_cluster_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["start", "-bt", "10"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "unsupported short flag cluster '-bt'" in captured.err


def test_init_and_journal_short_flags_parse():
    imp = _parse(["init", "-s", "pool.xyz", "-fy"])
    assert imp.source == "pool.xyz"
    assert imp.force is True
    assert imp.yes is True

    legacy = _parse(["import-pool", "-s", "pool.xyz", "-f"])
    assert legacy.source == "pool.xyz"
    assert legacy.force is True

    journal = _parse(["journal", "-e", "phase_submitted", "-n", "20", "-j"])
    assert journal.event_type == ["phase_submitted"]
    assert journal.last_n == 20
    assert journal.json is True


def test_removed_init_filter_flag_is_rejected():
    with pytest.raises(SystemExit):
        _parse(["init", "-s", "pool.xyz", "-O"])


def test_reconcile_status_stop_and_preflight_short_flags_parse(tmp_path):
    campaign = tmp_path / "campaign"
    _write_campaign_yaml(campaign)

    stop = _parse(["stop", "-c", str(campaign), "-x"])
    assert stop.campaign_dir == str(campaign)
    assert stop.cancel_jobs is True
    assert stop.stop_mode == "immediate"
    assert stop.after_iteration is None

    after_phase = _parse(["stop", "-c", str(campaign), "--after-phase"])
    assert after_phase.stop_mode == "after_phase"
    assert after_phase.after_iteration is None

    after_iteration = _parse(
        ["stop", "-c", str(campaign), "--after-iteration", "7"]
    )
    assert after_iteration.after_iteration == 7

    status = _parse(["status", "-c", str(campaign), "-j", "-v"])
    assert status.json is True
    assert status.verbose is True

    reconcile = _parse(["reconcile", "-c", str(campaign), "-a", "-F"])
    assert reconcile.apply is True
    assert reconcile.allow_fresh_init is True

    preflight = _parse(["preflight", "-c", str(campaign)])
    assert preflight.campaign_dir == str(campaign)


def test_help_mentions_campaign_auto_detection_and_examples(capsys):
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    top = capsys.readouterr().out
    assert "campaign.yaml" in top
    assert "ichor-al-daemon start" in top
    assert "ichor-al-daemon init" in top

    with pytest.raises(SystemExit):
        parser.parse_args(["start", "--help"])
    start = capsys.readouterr().out
    assert "--mode" in start
    assert "dry_run" in start
    assert "live" in start
    assert "-b" in start
    assert "--background" in start
    assert "live mode and in the background by default" in start
    assert "ichor-al-daemon start\n" in start

    with pytest.raises(SystemExit):
        parser.parse_args(["init", "--help"])
    init_help = capsys.readouterr().out
    assert "-s" in init_help
    assert "--source" in init_help
    assert "campaign.yaml" in init_help
