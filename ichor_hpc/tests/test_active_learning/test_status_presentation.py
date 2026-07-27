import inspect
import re
from pathlib import Path

import pytest

import ichor.hpc.active_learning.cli as cli
import ichor.hpc.active_learning.daemon.status_recommendations as recommendations
from ichor.hpc.active_learning.daemon import phase_progress

from ichor.hpc.active_learning.daemon.state import CampaignPhase


def _recommendation(code, primary, command):
    return {
        "code": code,
        "severity": "watch",
        "primary": primary,
        "why": "test evidence",
        "command": command,
    }


def _active_ariadne_payload():
    return {
        "phase": CampaignPhase.ARIADNE_ARRAY.value,
        "iteration": 7,
        "max_iterations": 40,
        "reference_data_version": 6,
        "models_version": 6,
        "lock_held": True,
        "pending_jobs": {CampaignPhase.ARIADNE_ARRAY.value: "12345"},
        "active_submission_intents": [
            {
                "phase": CampaignPhase.ARIADNE_ARRAY.value,
                "iteration": 7,
                "status": "SUBMITTED",
                "job_id": "12345",
                "expected_tasks": 150,
            }
        ],
        "artifact_manifest_status": {
            "reference_data": {"ok": True},
            "models": {"ok": True},
        },
        "state_artifact_contract_status": {"ok": True},
        "runtime_progress": {
            "state": "current",
            "age_seconds": 21.0,
            "record": {
                "producer_kind": "scheduler",
                "stage": "scheduler_wait",
                "status": "running",
                "elapsed_seconds": 848.0,
                "counters": {
                    "completed": 96,
                    "total": 150,
                    "unit": "tasks",
                },
                "details": {
                    "running": 18,
                    "pending": 36,
                    "failed": 0,
                    "missing": 0,
                },
            },
        },
        "recommendations": [
            _recommendation(
                "daemon_running",
                "the daemon is running; monitor it instead of starting another",
                "ichor-al-daemon journal --campaign-dir campaign --last-n 40",
            )
        ],
    }


def test_status_running_ariadne_matches_the_agreed_human_contract():
    output = cli._format_status(
        _active_ariadne_payload(),
        verbose=False,
        campaign=Path("campaign"),
        journal_events=[],
    )

    assert output == (
        "Campaign\n"
        "  phase: ARIADNE landing\n"
        "  purpose: generate adversarial geometries from selected seeds\n"
        "  iteration: 7 of 40\n"
        "\n"
        "Current status\n"
        "  overall: running normally\n"
        "  daemon: running\n"
        "  current work: waiting for 1 ARIADNE Slurm array\n"
        "  progress: 96 completed, 18 running, 36 pending\n"
        "  elapsed: 14m 08s\n"
        "  last update: 21s ago\n"
        "\n"
        "Progress so far\n"
        "  QM data: committed through iteration 6\n"
        "  model: FEREBUS model trained through iteration 6\n"
        "  readiness: data and model are up to date for iteration 7\n"
        "\n"
        "What happens next\n"
        "  automatic: after ARIADNE finishes, the daemon will validate the results and begin Phase B diversity selection\n"
        "  you need to do: nothing\n"
        "  follow progress: ichor-al-daemon journal --campaign-dir campaign --last-n 40\n"
    )
    for retired_label in ("Health", "Action", "severity:", "primary:", "why:"):
        assert retired_label not in output


def test_retained_staging_diagnostics_are_verbose_only(tmp_path):
    retained = (
        tmp_path
        / ".DATA"
        / "STAGING_RETIRED"
        / "active-iteration-000008-transaction"
    )
    retained.mkdir(parents=True)
    payload = _active_ariadne_payload()

    concise = cli._format_status(
        payload,
        verbose=False,
        campaign=tmp_path,
        journal_events=[],
    )
    verbose = cli._format_status(
        payload,
        verbose=True,
        campaign=tmp_path,
        journal_events=[],
    )

    assert "Retained Staging Diagnostics" not in concise
    assert "Retained Staging Diagnostics" in verbose
    assert str(retained) in verbose


def test_sampling_policy_details_are_verbose_only():
    payload = _active_ariadne_payload()
    payload["_presentation_sampling_protocol"] = {
        "sampling_aggressiveness": 7,
        "policy_version": 2,
        "target_motion_ratio": 1.55,
        "initial_trust_multiplier": 1.32,
        "movement_trust_multiplier": None,
        "baseline_source": "normalised_ariadne_landing_history",
    }

    concise = cli._format_status(
        payload,
        verbose=False,
        campaign=Path("campaign"),
        journal_events=[],
    )
    verbose = cli._format_status(
        payload,
        verbose=True,
        campaign=Path("campaign"),
        journal_events=[],
    )

    assert "Sampling protocol" not in concise
    assert "Sampling protocol" in verbose
    assert "sampling aggressiveness: 7" in verbose
    assert "preset policy: v2" in verbose
    assert "target movement: 1.55x historical baseline" in verbose
    assert "initial trust radius: 1.32x nominal" in verbose
    assert (
        "movement baseline: accepted ARIADNE movement history, normalised by "
        "producer preset"
    ) in verbose


def test_status_phase_policies_cover_every_campaign_phase():
    expected = {phase.value for phase in CampaignPhase}

    assert set(cli._PHASE_TITLES) == expected
    assert set(cli._PHASE_MEANINGS) == expected
    assert set(cli._PHASE_AUTOMATIC_OUTCOMES) == expected


@pytest.mark.parametrize("daemon_running", [False, True])
@pytest.mark.parametrize("phase", list(CampaignPhase))
def test_every_phase_renders_complete_running_and_stopped_status(
    phase,
    daemon_running,
):
    bootstrap_phase = phase.value in cli._BOOTSTRAP_COLLECTION_PHASES
    iteration = 0 if bootstrap_phase or phase is CampaignPhase.INITIAL_FEREBUS else 2
    reference_version = -1 if bootstrap_phase else iteration - 1
    model_version = -1 if bootstrap_phase or phase is CampaignPhase.INITIAL_FEREBUS else iteration - 1
    if phase is CampaignPhase.INITIAL_FEREBUS:
        reference_version = 0
    if phase is CampaignPhase.REFERENCE_COMMIT:
        reference_version = iteration - 1
        model_version = iteration - 1
    if phase is CampaignPhase.FEREBUS:
        reference_version = iteration
        model_version = iteration - 1
    if phase in {CampaignPhase.STOP_CHECK, CampaignPhase.DONE}:
        reference_version = iteration
        model_version = iteration
    recommendation_code = (
        "campaign_done"
        if phase is CampaignPhase.DONE
        else "halted_unknown"
        if phase is CampaignPhase.HALTED
        else "daemon_running"
        if daemon_running
        else "phase_" + phase.value.lower() + "_ready"
    )
    payload = {
        "phase": phase.value,
        "iteration": iteration,
        "max_iterations": 5,
        "reference_data_version": reference_version,
        "models_version": model_version,
        "lock_held": daemon_running,
        "artifact_manifest_status": {
            "reference_data": {"ok": True},
            "models": {"ok": True},
        },
        "state_artifact_contract_status": {"ok": True},
        "recommendations": [
            _recommendation(
                recommendation_code,
                "review the campaign" if phase is CampaignPhase.DONE else "continue",
                "ichor-al-daemon journal --campaign-dir campaign --last-n 40",
            )
        ],
    }

    output = cli._format_status(
        payload,
        verbose=False,
        campaign=Path("campaign"),
        journal_events=[],
    )

    for heading in (
        "Campaign\n",
        "Current status\n",
        "Progress so far\n",
        "What happens next\n",
    ):
        assert heading in output
    assert "  automatic: " in output
    assert "Health\n" not in output
    assert "Action\n" not in output


def test_status_recommendation_policies_cover_every_constructor():
    source = inspect.getsource(recommendations)
    literal_codes = set(re.findall(r'code="([^"]+)"', source))
    literal_codes.discard("phase_")
    phase_codes = {
        "phase_" + phase.lower() + "_ready"
        for phase in recommendations._PHASE_ACTIONS
    }

    assert cli._STATUS_PRESENTATION_RECOMMENDATIONS == (
        literal_codes | phase_codes | {"recommendation_unavailable"}
    )


def test_stop_and_invalid_artifacts_override_daemon_running_recommendation(tmp_path):
    campaign = tmp_path / "campaign"
    stop_payload = {
        "phase": CampaignPhase.ARIADNE_ARRAY.value,
        "lock_held": True,
        "stop_request": {
            "status": "requested",
            "request_id": "request-1",
            "mode": "after_iteration",
            "target_iteration": 7,
        },
    }
    invalid_payload = {
        "phase": CampaignPhase.STOP_CHECK.value,
        "lock_held": True,
        "state_artifact_contract_status": {
            "ok": False,
            "error": "missing model",
        },
    }

    stop = recommendations.build_status_recommendations(campaign, stop_payload)
    invalid = recommendations.build_status_recommendations(campaign, invalid_payload)

    assert stop[0].code == "user_stop_draining"
    assert invalid[0].code == "stop_check_no_committed_pair"


def test_halted_recovery_precedes_pending_iteration_stop(tmp_path):
    campaign = tmp_path / "campaign"
    result = recommendations.build_status_recommendations(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "latest_halt_event": {
                "reason": (
                    "prior_gaussian_acceptance_manifest_invalid: rejected "
                    "pointdir is not present in POINTS.txt"
                )
            },
            "stop_request": {
                "status": "requested",
                "request_id": "request-14",
                "mode": "after_iteration",
                "target_iteration": 14,
            },
        },
    )

    assert result[0].code.startswith("halted_")
    assert "reconcile" in str(result[0].command)


def test_allocation_environment_retry_precedes_pending_iteration_stop(tmp_path):
    campaign = tmp_path / "campaign"
    payload = {
        "phase": CampaignPhase.ALLOCATION_CHECK.value,
        "iteration": 14,
        "background_startup_state": "failed",
        "background_startup_stage": "environment_transition",
        "_presentation_allocation_check_transition": {
            "safe": True,
            "pending_tasks": 1,
            "replacement_sample_state": "missing_rebuildable",
        },
        "stop_request": {
            "status": "requested",
            "request_id": "request-14",
            "mode": "after_iteration",
            "target_iteration": 14,
        },
    }

    result = recommendations.build_status_recommendations(campaign, payload)
    payload["recommendations"] = [result[0].to_dict()]

    assert result[0].code == "allocation_check_environment_retry"
    assert "rebuild the missing replacement sample" in result[0].why
    assert str(result[0].command).startswith("ichor-al-daemon resume")
    assert cli._status_overall(payload) == "stopped and ready to continue"


def test_conflicting_allocation_evidence_precedes_pending_stop(tmp_path):
    campaign = tmp_path / "campaign"
    payload = {
        "phase": CampaignPhase.ALLOCATION_CHECK.value,
        "iteration": 14,
        "_presentation_allocation_check_transition": {
            "safe": False,
            "reason": "replacement sample manifest conflicts with pending allocation",
        },
        "stop_request": {
            "status": "requested",
            "request_id": "request-14",
            "mode": "after_iteration",
            "target_iteration": 14,
        },
    }

    result = recommendations.build_status_recommendations(campaign, payload)
    payload["recommendations"] = [result[0].to_dict()]

    assert result[0].code == "allocation_check_environment_blocked"
    assert "reconcile" in str(result[0].command)
    assert cli._status_overall(payload) == "needs attention"


def test_aimall_postprocess_recovery_explains_local_reuse_and_pending_stop(
    tmp_path,
):
    campaign = tmp_path / "campaign"
    recovery = {
        "phase": CampaignPhase.AIMALL.value,
        "iteration": 14,
        "logical_total": 149,
        "n_complete": 149,
        "n_reuse": 149,
        "n_retry": 0,
    }
    payload = {
        "phase": CampaignPhase.AIMALL.value,
        "iteration": 14,
        "lock_held": False,
        "_presentation_aimall_postprocess_recovery": recovery,
        "stop_request": {
            "status": "requested",
            "request_id": "request-14",
            "mode": "after_iteration",
            "target_iteration": 14,
        },
    }

    result = recommendations.build_status_recommendations(campaign, payload)

    assert result[0].code == "user_stop_draining"
    assert "149 existing AIMAll outputs" in result[0].primary
    assert "no AIMAll array will be resubmitted" in result[0].primary
    assert "honoured after this iteration genuinely completes" in result[0].why
    assert "149 completed AIMAll outputs" in cli._status_current_activity(payload)


def test_stopped_scheduler_progress_is_explicitly_historical():
    payload = _active_ariadne_payload()
    payload["lock_held"] = False
    payload["background_pid_alive"] = False
    presentation = cli._build_status_presentation(
        payload,
        campaign=Path("campaign"),
        journal_events=[],
    )

    current = dict(presentation.current_status)
    assert current["overall"] == "stopped with unfinished work"
    assert current["daemon"] == "not running"
    assert current["current work"].startswith("The daemon is stopped;")
    assert current["last recorded Slurm progress"] == (
        "96 completed, 18 running, 36 pending"
    )


def test_stopped_local_intent_is_not_described_as_slurm_work():
    payload = {
        "phase": CampaignPhase.FEREBUS.value,
        "iteration": 2,
        "max_iterations": 5,
        "reference_data_version": 2,
        "models_version": 1,
        "lock_held": False,
        "active_submission_intents": [
            {
                "phase": CampaignPhase.FEREBUS.value,
                "iteration": 2,
                "status": "PRE_SUBMIT",
                "job_id": None,
            }
        ],
        "artifact_manifest_status": {
            "reference_data": {"ok": True},
            "models": {"ok": True},
        },
        "state_artifact_contract_status": {"ok": True},
        "recommendations": [
            _recommendation(
                "local_submission_intent",
                "resume the daemon",
                "ichor-al-daemon resume --campaign-dir campaign",
            )
        ],
    }

    next_rows = dict(cli._status_next_rows(payload, campaign=Path("campaign")))

    assert next_rows["automatic"] == (
        "prepared local work will remain paused until the daemon is resumed"
    )
    assert "Slurm" not in next_rows["automatic"]


def test_invalid_done_state_requires_action_instead_of_review_only():
    payload = {
        "phase": CampaignPhase.DONE.value,
        "iteration": 5,
        "max_iterations": 5,
        "reference_data_version": 5,
        "models_version": 5,
        "lock_held": False,
        "artifact_manifest_status": {
            "reference_data": {"ok": True},
            "models": {"ok": False},
        },
        "state_artifact_contract_status": {"ok": False},
        "recommendations": [
            _recommendation(
                "state_artifact_contract_invalid",
                "preview recovery",
                "ichor-al-daemon reconcile --campaign-dir campaign",
            )
        ],
    }

    rows = dict(cli._status_next_rows(payload, campaign=Path("campaign")))

    assert rows["you need to do"] == "preview recovery"
    assert rows["run"] == "ichor-al-daemon reconcile --campaign-dir campaign"
    assert "review" not in rows


def test_status_uses_bounded_journal_evidence_only_as_recorded_activity():
    payload = {
        "phase": CampaignPhase.AIMALL.value,
        "iteration": 2,
        "max_iterations": 5,
        "reference_data_version": 1,
        "models_version": 1,
        "lock_held": False,
        "artifact_manifest_status": {
            "reference_data": {"ok": True},
            "models": {"ok": True},
        },
        "state_artifact_contract_status": {"ok": True},
        "recommendations": [
            _recommendation(
                "phase_aimall_ready",
                "resume the daemon",
                "ichor-al-daemon resume --campaign-dir campaign",
            )
        ],
    }
    events = [
        {
            "event": "phase_activity_progress",
            "phase": CampaignPhase.AIMALL.value,
            "iteration": 2,
            "stage": "scientific_quality",
            "completed": 81,
            "total": 200,
            "unit": "point directories",
            "ts": "2026-07-21T10:00:00+00:00",
        },
        {
            "event": "effective_config_diff",
            "iteration": 2,
            "n_non_default": 3,
            "ts": "2026-07-21T10:01:00+00:00",
        },
    ]

    presentation = cli._build_status_presentation(
        payload,
        campaign=Path("campaign"),
        journal_events=events,
    )

    current = dict(presentation.current_status)
    assert "last recorded activity" in current
    assert "81/200" in current["last recorded activity"]
    assert "progress" not in current


def test_status_unavailable_uses_the_same_four_sections():
    payload = {
        "status_error": "state_unreadable",
        "state_error": "OSError: permission denied",
        "state_path": "campaign/state.json",
        "recommendations": [
            _recommendation(
                "state_unreadable",
                "check that state.json is readable, then preview recovery",
                "ichor-al-daemon reconcile --campaign-dir campaign",
            )
        ],
    }

    default = cli._format_status_unavailable(payload, verbose=False)
    verbose = cli._format_status_unavailable(payload, verbose=True)

    for heading in (
        "Campaign\n",
        "Current status\n",
        "Progress so far\n",
        "What happens next\n",
    ):
        assert heading in default
    assert "OSError" not in default
    assert "State diagnostics\n" in verbose
    assert "OSError: permission denied" in verbose


def test_missing_campaign_files_recommend_initialisation_before_config_repair(tmp_path):
    payload = {
        "status_error": "state_missing",
        "campaign_yaml_exists": False,
        "fresh_init_safe": True,
        "campaign_config_status": {
            "ok": False,
            "error": "FileNotFoundError: campaign.yaml",
        },
    }

    result = recommendations.build_status_recommendations(
        tmp_path / "campaign",
        payload,
    )

    assert result[0].code == "campaign_missing"
    assert "init" in str(result[0].command)


@pytest.mark.parametrize("stage", sorted(phase_progress._STAGE_LABELS))
def test_every_generic_progress_stage_uses_the_shared_status_wording(stage):
    activity = cli._format_generic_progress_activity(
        {"stage": stage, "status": "running"}
    )

    assert activity == phase_progress.format_progress_stage(stage) + "."


@pytest.mark.parametrize(
    "stage",
    [
        "loading",
        "features",
        "reference_neighbours",
        "reference_scales",
        "filtering",
        "random",
        "variance",
        "shortlist",
        "d_optimal",
        "publishing",
    ],
)
def test_every_seed_selection_stage_has_specific_status_wording(stage):
    activity = cli._format_seed_selection_progress(
        {"stage": stage, "completed": 1, "total": 2}
    )

    assert not activity.startswith("Seed selection is running (")
