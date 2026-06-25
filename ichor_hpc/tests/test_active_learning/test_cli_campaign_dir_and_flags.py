from __future__ import annotations

import pytest

from ichor.hpc.active_learning import cli


def _write_campaign_yaml(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "campaign.yaml").write_text("schema_version: 3\n", encoding="utf-8")


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


def test_main_reports_missing_campaign_dir_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        cli.main(["status"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "current directory does not contain campaign.yaml" in captured.err


def test_status_without_campaign_dir_uses_cwd_campaign(tmp_path, monkeypatch, capsys):
    _write_campaign_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert cli.main(["status"]) == 4
    captured = capsys.readouterr()
    assert ".DATA" in captured.err
    assert "state.json" in captured.err


def test_start_short_flags_match_long_options():
    ns = _parse(["start", "-l", "-b", "-d", "-m", "-p", "3", "-t", "9"])

    assert ns.live is True
    assert ns.background is True
    assert ns.dry_run is True
    assert ns.mock_ariadne is True
    assert ns.poll_interval == 3
    assert ns.max_ticks == 9


def test_start_clustered_boolean_flags_expand_to_individual_flags():
    expanded = cli.expand_boolean_short_flag_clusters(["start", "-lb", "-t", "10"])
    assert expanded == ["start", "-l", "-b", "-t", "10"]

    ns = _parse(["start", "-lb", "-t", "10"])
    assert ns.live is True
    assert ns.background is True
    assert ns.max_ticks == 10


def test_cluster_expansion_does_not_split_value_taking_options():
    with pytest.raises(cli.ShortFlagClusterError):
        cli.expand_boolean_short_flag_clusters(["start", "-cthing"])
    with pytest.raises(cli.ShortFlagClusterError):
        cli.expand_boolean_short_flag_clusters(["start", "-lt", "10"])


def test_main_reports_ambiguous_short_cluster_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["start", "-lt", "10"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "unsupported short flag cluster '-lt'" in captured.err


def test_init_and_journal_short_flags_parse():
    imp = _parse(["init", "-s", "pool.xyz", "-f", "-O"])
    assert imp.source == "pool.xyz"
    assert imp.force is True
    assert imp.no_outlier_filter is True

    legacy = _parse(["import-pool", "-s", "pool.xyz", "-f", "-O"])
    assert legacy.source == "pool.xyz"
    assert legacy.force is True
    assert legacy.no_outlier_filter is True

    journal = _parse(["journal", "-e", "phase_submitted", "-n", "20", "-j"])
    assert journal.event_type == ["phase_submitted"]
    assert journal.last_n == 20
    assert journal.json is True


def test_reconcile_status_stop_and_preflight_short_flags_parse(tmp_path):
    campaign = tmp_path / "campaign"
    _write_campaign_yaml(campaign)

    stop = _parse(["stop", "-c", str(campaign), "-x"])
    assert stop.campaign_dir == str(campaign)
    assert stop.cancel_jobs is True

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
    assert "ichor-al-daemon start -lb" in top
    assert "ichor-al-daemon init" in top

    with pytest.raises(SystemExit):
        parser.parse_args(["start", "--help"])
    start = capsys.readouterr().out
    assert "-l" in start
    assert "--live" in start
    assert "-b" in start
    assert "--background" in start
    assert "ichor-al-daemon start -lb" in start

    with pytest.raises(SystemExit):
        parser.parse_args(["init", "--help"])
    init_help = capsys.readouterr().out
    assert "-s" in init_help
    assert "--source" in init_help
    assert "campaign.yaml" in init_help
