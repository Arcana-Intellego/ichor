"""Static closure checks for the daemon CLI output audit."""

from __future__ import annotations

import pytest

from ichor.hpc.active_learning import cli
from ichor.hpc.active_learning.daemon.journal import (
    JOURNAL_EVENT_CONTEXTS,
    JOURNAL_EVENT_ITERATION_POLICIES,
    KNOWN_EVENT_TYPES,
)
from ichor.hpc.active_learning.daemon.state import CampaignPhase


EXPECTED_FINDING_IDS = {
    "CLI-W04-P2-001",
    "CLI-W05-P1-001",
    "CLI-W05-P2-002",
    "CLI-W05-P2-003",
    "CLI-W05-P2-004",
    "CLI-W06-P1-001",
    "CLI-W06-P1-002",
    "CLI-W06-P2-003",
    "CLI-W07-P2-001",
    "CLI-W07-P2-002",
    "CLI-W08-P1-001",
    "CLI-W09-P1-001",
    "CLI-W09-P1-002",
    "CLI-W09-P1-003",
    "CLI-W09-P1-004",
    "CLI-W09-P1-005",
    "CLI-W09-P3-006",
    "CLI-W10-P1-001",
    "CLI-W10-P2-002",
    "CLI-W10-P2-003",
    "CLI-W11-P1-001",
    "CLI-W11-P1-002",
    "CLI-W12-P1-001",
    "CLI-W12-P1-002",
    "CLI-W12-P2-003",
    "CLI-W13-P2-001",
    "CLI-W13-P2-002",
    "CLI-W13-P1-003",
    "CLI-W13-P3-004",
    "CLI-W14-P1-001",
    "CLI-W14-P1-002",
    "CLI-W14-P2-003",
    "CLI-W15-P0-001",
    "CLI-W15-P3-002",
    "CLI-W16-P1-001",
    "CLI-W16-P1-002",
    "CLI-W16-P1-003",
    "CLI-W16-P1-004",
    "CLI-W16-P2-005",
    "CLI-W17-P2-001",
    "CLI-W17-P2-002",
    "CLI-W18-P1-001",
    "CLI-W18-P1-002",
    "CLI-W18-P1-003",
    "CLI-W18-P2-004",
}


FINDING_REGRESSION_OWNERS = {
    finding_id: owner
    for owner, finding_ids in {
        "help_init_config": {
            "CLI-W04-P2-001",
            "CLI-W05-P1-001",
            "CLI-W05-P2-002",
            "CLI-W05-P2-003",
            "CLI-W05-P2-004",
        },
        "start_resume_stop": {
            "CLI-W06-P1-001",
            "CLI-W06-P1-002",
            "CLI-W06-P2-003",
            "CLI-W07-P2-001",
            "CLI-W07-P2-002",
        },
        "status": {
            "CLI-W08-P1-001",
            "CLI-W09-P1-001",
            "CLI-W09-P1-002",
            "CLI-W09-P1-003",
            "CLI-W09-P1-004",
            "CLI-W09-P1-005",
            "CLI-W09-P3-006",
            "CLI-W10-P1-001",
            "CLI-W10-P2-002",
            "CLI-W10-P2-003",
            "CLI-W11-P1-001",
            "CLI-W11-P1-002",
        },
        "journal": {
            "CLI-W12-P1-001",
            "CLI-W12-P1-002",
            "CLI-W12-P2-003",
            "CLI-W13-P2-001",
            "CLI-W13-P2-002",
            "CLI-W13-P1-003",
            "CLI-W13-P3-004",
        },
        "reconcile": {
            "CLI-W14-P1-001",
            "CLI-W14-P1-002",
            "CLI-W14-P2-003",
            "CLI-W15-P0-001",
            "CLI-W15-P3-002",
        },
        "preflight": {
            "CLI-W16-P1-001",
            "CLI-W16-P1-002",
            "CLI-W16-P1-003",
            "CLI-W16-P1-004",
            "CLI-W16-P2-005",
        },
        "resources_checkpoints": {
            "CLI-W17-P2-001",
            "CLI-W17-P2-002",
        },
        "menus": {
            "CLI-W18-P1-001",
            "CLI-W18-P1-002",
            "CLI-W18-P1-003",
            "CLI-W18-P2-004",
        },
    }.items()
    for finding_id in finding_ids
}


def test_all_daemon_cli_output_findings_have_one_regression_owner():
    assert len(EXPECTED_FINDING_IDS) == 45
    assert set(FINDING_REGRESSION_OWNERS) == EXPECTED_FINDING_IDS
    assert all(FINDING_REGRESSION_OWNERS.values())


def test_every_known_journal_event_has_one_human_disposition():
    known = set(KNOWN_EVENT_TYPES)
    assert len(KNOWN_EVENT_TYPES) == len(known)
    assert set(cli.JOURNAL_EVENT_LABELS) == known
    groups = [
        cli._JOURNAL_OK_EVENTS,
        cli._JOURNAL_RUN_EVENTS,
        cli._JOURNAL_WAIT_EVENTS,
        cli._JOURNAL_WARN_EVENTS,
        cli._JOURNAL_FAIL_EVENTS,
        cli._JOURNAL_INFO_EVENTS,
        cli._JOURNAL_DYNAMIC_EVENTS,
    ]
    seen = set()
    for group in groups:
        assert not (seen & set(group))
        seen.update(group)
    assert seen == known


def test_every_known_journal_event_has_context_and_iteration_policy():
    known = set(KNOWN_EVENT_TYPES)
    assert set(JOURNAL_EVENT_CONTEXTS) == known
    assert set(JOURNAL_EVENT_ITERATION_POLICIES) == known
    assert all(
        context and context != "UNCLASSIFIED"
        for context in JOURNAL_EVENT_CONTEXTS.values()
    )
    assert set(JOURNAL_EVENT_ITERATION_POLICIES.values()) == {
        "required",
        "state_unavailable_allowed",
    }


def test_every_known_journal_event_renders_a_classified_context():
    for event_type in KNOWN_EVENT_TYPES:
        context = cli._event_context(
            {
                "event": event_type,
                "phase": CampaignPhase.SEED_SELECT.value,
                "iteration": 2,
            }
        )
        assert context not in {"", "-", "UNCLASSIFIED"}, event_type


def test_every_campaign_phase_has_stable_purpose_text():
    phases = {phase.value for phase in CampaignPhase}
    assert set(cli._PHASE_TITLES) == phases
    assert set(cli._PHASE_MEANINGS) == phases
    for purpose in cli._PHASE_MEANINGS.values():
        assert " is next" not in purpose
        assert " is being " not in purpose


def test_additive_output_flags_parse_without_changing_default_commands():
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    commands = {subparsers.dest: set(subparsers.choices)}
    assert commands["command"] == {
        "start",
        "stop",
        "status",
        "resume",
        "reconcile",
        "journal",
        "init",
        "import-pool",
        "preflight",
        "resource-plan",
        "export-batch-geometries",
        "checkpoint",
        "checkpoint-status",
        "verify-checkpoint",
        "restore-checkpoint",
        "config-check",
    }
    assert parser.parse_args(["init", "--verbose"]).verbose is True
    assert parser.parse_args(["stop", "--verbose"]).verbose is True
    assert parser.parse_args(["resource-plan", "--verbose"]).verbose is True
    assert parser.parse_args(
        ["export-batch-geometries", "--iteration", "all"]
    ).iteration == "all"
    assert parser.parse_args(["config-check", "--human"]).human is True
    assert parser.parse_args(["config-check", "--json"]).json is True
    with pytest.raises(SystemExit):
        parser.parse_args(["config-check", "--human", "--json"])


def test_resume_and_force_help_describe_the_real_user_contract():
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    resume_help = subparsers.choices["resume"].format_help()
    init_help = subparsers.choices["init"].format_help()

    assert "Continue an existing campaign after a normal stop" in resume_help
    assert "Equivalent to start" not in resume_help
    assert "uncommitted" in init_help
    assert "committed run data are always refused" in init_help
