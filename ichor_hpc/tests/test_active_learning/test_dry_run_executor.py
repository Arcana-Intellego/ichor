"""Unit tests for the DryRunPhaseExecutor and DryRunSacctPoller."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.dry_run_executor import (
    DRYRUN_JOB_PREFIX,
    DryRunPhaseExecutor,
)
from ichor.hpc.active_learning.daemon.dry_run_sacct import (
    DRYRUN_PREFIX,
    DryRunSacctPoller,
)
from ichor.hpc.active_learning.daemon.phase_executor import (
    FailureAction,
    PhaseResult,
)
from ichor.hpc.active_learning.daemon.state import CampaignPhase
from ichor.hpc.active_learning.point_allocation import (
    pending_attempts,
    point_allocation_path,
    read_point_allocation,
    record_quantum_results,
)
from ichor.hpc.active_learning.layout import (
    active_iteration_dir,
    active_phase_b_dir,
    active_seed_selection_dir,
    ariadne_seeds_dir,
)
from ichor.hpc.active_learning.submit.sacct_poll import JobStatus
from ichor.hpc.active_learning.versioning.versioned_directory import VersionedDirectory


def _make_exec(tmp_path: Path) -> DryRunPhaseExecutor:
    cfg = CampaignConfig(max_iterations=2)
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.seed_selection.n_seeds_per_iteration = 2
    return DryRunPhaseExecutor(campaign_dir=tmp_path / "campaign", config=cfg)


def _state(iteration=0, **updates):
    values = {
        "iteration": int(iteration),
        "campaign_uid": "dry-run-test-campaign",
        "replacement_round": 0,
        "reference_data_version": -1,
        "models_version": -1,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _submit_dry_phase(e, state, phase):
    from ichor.hpc.active_learning.daemon.config_lock import (
        canonical_config,
        config_fingerprint,
    )
    from ichor.hpc.active_learning.daemon.submission_intent import (
        update_intent_status,
        write_pre_submit_intent,
    )

    phase_name = phase.value
    write_pre_submit_intent(
        e.campaign_dir,
        campaign_uid=str(state.campaign_uid),
        phase_name=phase_name,
        iteration=int(state.iteration),
        replacement_round=int(getattr(state, "replacement_round", 0)),
        scheduler_identity_kind="synthetic",
        decision_contract={
            "failure_threshold_fraction": float(
                e.config.runtime.failure_threshold_fraction
            ),
            "config_sha256": config_fingerprint(canonical_config(e.config)),
        },
    )
    result = e.submit_or_run(state, phase)
    update_intent_status(
        e.campaign_dir,
        phase_name=phase_name,
        iteration=int(state.iteration),
        status="SUBMITTED",
        job_id=str(result.submitted_job_id),
        expected_tasks=int(result.expected_tasks),
    )
    return result


def _complete_bootstrap(e):
    from ichor.hpc.active_learning.daemon.state import (
        DEFAULT_STATE_FILENAME,
        fresh_campaign_state,
        write_state,
    )

    state = _state()
    state_path = (
        e.campaign_dir / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    )
    if not state_path.is_file():
        write_state(
            state_path,
            fresh_campaign_state(
                max_iterations=e.config.campaign.max_iterations,
                campaign_uid=str(state.campaign_uid),
            ),
        )
    e._ensure_dry_run_trajectory_pool()
    _submit_dry_phase(e, state, CampaignPhase.PHASE_A_DIVERSITY)
    e.postprocess(state, CampaignPhase.PHASE_A_DIVERSITY, observations=[])
    _submit_dry_phase(e, state, CampaignPhase.INITIAL_GAUSSIAN)
    e.postprocess(state, CampaignPhase.INITIAL_GAUSSIAN, observations=[])
    _submit_dry_phase(e, state, CampaignPhase.INITIAL_AIMALL)
    e.postprocess(state, CampaignPhase.INITIAL_AIMALL, observations=[])
    check = e.submit_or_run(state, CampaignPhase.INITIAL_ALLOCATION_CHECK)
    assert check.next_phase_override == CampaignPhase.REFERENCE_COMMIT.value
    commit = e.submit_or_run(state, CampaignPhase.REFERENCE_COMMIT)
    state.reference_data_version = commit.state_updates["reference_data_version"]
    _submit_dry_phase(e, state, CampaignPhase.INITIAL_FEREBUS)
    e.postprocess(state, CampaignPhase.INITIAL_FEREBUS, observations=[])
    return state


def _complete_active_quantum(e, *, iteration=1):
    state = _state(
        iteration,
        reference_data_version=0,
        models_version=0,
    )
    e.submit_or_run(state, CampaignPhase.SEED_SELECT)
    _submit_dry_phase(e, state, CampaignPhase.ARIADNE_ARRAY)
    e.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    _submit_dry_phase(e, state, CampaignPhase.PHASE_B_DIVERSITY)
    e.postprocess(state, CampaignPhase.PHASE_B_DIVERSITY, observations=[])
    _submit_dry_phase(e, state, CampaignPhase.GAUSSIAN)
    e.postprocess(state, CampaignPhase.GAUSSIAN, observations=[])
    _submit_dry_phase(e, state, CampaignPhase.AIMALL)
    e.postprocess(state, CampaignPhase.AIMALL, observations=[])
    check = e.submit_or_run(state, CampaignPhase.ALLOCATION_CHECK)
    assert check.next_phase_override == CampaignPhase.REFERENCE_COMMIT.value
    return state


def test_executor_creates_canonical_subdirs(tmp_path):
    e = _make_exec(tmp_path)
    base = tmp_path / "campaign"
    assert (base / "QM_REFERENCE_DATA").is_dir()
    assert (base / "TRAINED_MODELS").is_dir()
    assert (base / ".DATA" / "BOOTSTRAP").is_dir()
    assert (base / "ACTIVE_LEARNING").is_dir()
    assert (base / ".DATA" / "SCRIPTS").is_dir()


def test_submit_or_run_sbatch_phase_writes_stub_script(tmp_path):
    e = _make_exec(tmp_path)
    e._ensure_dry_run_trajectory_pool()
    state = SimpleNamespace(iteration=0)
    result = e.submit_or_run(state, CampaignPhase.PHASE_A_DIVERSITY)
    assert isinstance(result, PhaseResult)
    assert not result.is_complete
    assert result.submitted_job_id == DRYRUN_JOB_PREFIX + "PHASE_A_DIVERSITY-0"
    scripts = list(
        (
            tmp_path
            / "campaign"
            / ".DATA"
            / "SCRIPTS"
            / "JOBS"
            / "DIVERSITY"
            / "PHASE_A_DIVERSITY"
            / "iteration-000000"
        ).glob("*/job.sh")
    )
    assert len(scripts) == 1
    script = scripts[0]
    content = script.read_text(encoding="utf-8")
    assert "DRYRUN" in content
    assert "iteration=0" in content


def test_submit_or_run_inline_phase_runs_synchronously(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    state = _state(1, reference_data_version=0, models_version=0)
    result = e.submit_or_run(state, CampaignPhase.SEED_SELECT)
    assert result.is_complete is True
    assert result.submitted_job_id is None
    seeds = active_seed_selection_dir(
        active_iteration_dir(tmp_path / "campaign", 1)
    ) / "seeds.xyz"
    assert seeds.exists()


def test_initial_ferebus_postprocess_commits_training_iteration_0(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    v = VersionedDirectory(tmp_path / "campaign" / "QM_REFERENCE_DATA")
    assert 0 in v.list_committed_versions()
    assert v.current_version() == 0


def test_initial_ferebus_postprocess_commits_models_iteration_0(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    v = VersionedDirectory(tmp_path / "campaign" / "TRAINED_MODELS")
    assert 0 in v.list_committed_versions()


def test_ariadne_postprocess_writes_per_seed_results(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    state = _state(1, reference_data_version=0, models_version=0)
    e.submit_or_run(state, CampaignPhase.SEED_SELECT)
    _submit_dry_phase(e, state, CampaignPhase.ARIADNE_ARRAY)
    e.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    seeds_dir = ariadne_seeds_dir(
        active_iteration_dir(tmp_path / "campaign", 1)
    )
    assert seeds_dir.is_dir()
    seed_dirs = sorted(seeds_dir.iterdir())
    assert seed_dirs
    for sd in seed_dirs:
        result_json = sd / "result.json"
        assert result_json.exists()
        payload = json.loads(result_json.read_text())
        assert payload["mock"] is True
        assert "alpha_trajectory" in payload


def test_phase_b_diversity_postprocess_writes_sample_xyz(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    state = _state(1, reference_data_version=0, models_version=0)
    e.submit_or_run(state, CampaignPhase.SEED_SELECT)
    _submit_dry_phase(e, state, CampaignPhase.ARIADNE_ARRAY)
    e.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    _submit_dry_phase(e, state, CampaignPhase.PHASE_B_DIVERSITY)
    e.postprocess(state, CampaignPhase.PHASE_B_DIVERSITY, observations=[])
    sample = active_phase_b_dir(
        active_iteration_dir(tmp_path / "campaign", 1)
    ) / "selected.xyz"
    assert sample.exists()
    manifest = json.loads(
        (sample.parent / "SELECTION.json").read_text(encoding="utf-8")
    )
    assert manifest["descriptor"] == "hybrid_alf_rmsd"


def test_reference_commit_inline_commits_next_training_iteration(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    state = _complete_active_quantum(e)
    e.submit_or_run(state, CampaignPhase.REFERENCE_COMMIT)
    v = VersionedDirectory(tmp_path / "campaign" / "QM_REFERENCE_DATA")
    assert sorted(v.list_committed_versions()) == [0, 1]
    assert v.current_version() == 1


def test_ferebus_postprocess_commits_next_models_iteration(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    state = _complete_active_quantum(e)
    commit = e.submit_or_run(state, CampaignPhase.REFERENCE_COMMIT)
    state.reference_data_version = commit.state_updates["reference_data_version"]
    _submit_dry_phase(e, state, CampaignPhase.FEREBUS)
    e.postprocess(state, CampaignPhase.FEREBUS, observations=[])
    v = VersionedDirectory(tmp_path / "campaign" / "TRAINED_MODELS")
    assert sorted(v.list_committed_versions()) == [0, 1]


def test_active_allocation_replaces_failed_candidate_from_finite_reserve(tmp_path):
    cfg = CampaignConfig(max_iterations=1)
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.seed_selection.n_seeds_per_iteration = 3
    e = DryRunPhaseExecutor(campaign_dir=tmp_path / "campaign", config=cfg)
    _complete_bootstrap(e)
    state = _state(1, reference_data_version=0, models_version=0)
    e.submit_or_run(state, CampaignPhase.SEED_SELECT)
    _submit_dry_phase(e, state, CampaignPhase.ARIADNE_ARRAY)
    e.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    _submit_dry_phase(e, state, CampaignPhase.PHASE_B_DIVERSITY)
    e.postprocess(state, CampaignPhase.PHASE_B_DIVERSITY, observations=[])

    allocation_path = point_allocation_path(
        e.campaign_dir,
        context="active",
        iteration=1,
    )
    allocation = read_point_allocation(allocation_path)
    attempts = pending_attempts(allocation)
    failed = attempts[-1]
    quality_manifest = e.campaign_dir / "synthetic-quantum-quality.json"
    quality_manifest.write_text("{}\n", encoding="utf-8", newline="\n")
    record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": attempt["candidate_id"],
                "accepted": attempt["candidate_id"] != failed["candidate_id"],
                "pointdir": str(
                    e.campaign_dir
                    / ".DATA"
                    / "STAGING"
                    / "iter_1"
                    / ("synthetic-" + str(attempt["slot_id"]) + ".pointdir")
                ),
                "reason": (
                    "synthetic_qm_failure"
                    if attempt["candidate_id"] == failed["candidate_id"]
                    else None
                ),
                "quality_manifest": (
                    None
                    if attempt["candidate_id"] == failed["candidate_id"]
                    else str(quality_manifest.resolve())
                ),
            }
            for attempt in attempts
        ],
    )

    replacement = e.submit_or_run(state, CampaignPhase.ALLOCATION_CHECK)
    assert replacement.next_phase_override == CampaignPhase.REPLACEMENT_GAUSSIAN.value
    assert replacement.state_updates == {"replacement_round": 1}
    state.replacement_round = 1
    pending_before = read_point_allocation(allocation_path)
    round_dir = (
        e.campaign_dir
        / ".DATA"
        / "STAGING"
        / "iter_1"
        / "replacement_round_0001"
    )
    (round_dir / "replacement-SAMPLE.xyz").unlink()
    (round_dir / "REPLACEMENT_SAMPLE.json").unlink()

    replay = e.submit_or_run(state, CampaignPhase.ALLOCATION_CHECK)

    pending_after = read_point_allocation(allocation_path)
    assert replay.next_phase_override == CampaignPhase.REPLACEMENT_GAUSSIAN.value
    assert pending_after == pending_before
    assert (round_dir / "replacement-SAMPLE.xyz").is_file()
    assert (round_dir / "REPLACEMENT_SAMPLE.json").is_file()
    _submit_dry_phase(e, state, CampaignPhase.REPLACEMENT_GAUSSIAN)
    e.postprocess(state, CampaignPhase.REPLACEMENT_GAUSSIAN, observations=[])
    _submit_dry_phase(e, state, CampaignPhase.REPLACEMENT_AIMALL)
    aimall = e.postprocess(state, CampaignPhase.REPLACEMENT_AIMALL, observations=[])
    assert aimall.next_phase_override == CampaignPhase.ALLOCATION_CHECK.value
    complete = e.submit_or_run(state, CampaignPhase.ALLOCATION_CHECK)

    final_allocation = read_point_allocation(allocation_path)
    assert final_allocation["summary"]["complete"] is True
    assert final_allocation["summary"]["accepted"] == {
        "train": 1,
        "int_val": 1,
        "ext_val": 0,
    }
    assert complete.next_phase_override == CampaignPhase.REFERENCE_COMMIT.value
    assert complete.state_updates == {"replacement_round": 0}


def test_handle_failure_returns_scrub_and_continue(tmp_path):
    e = _make_exec(tmp_path)
    action = e.handle_failure(SimpleNamespace(iteration=1), CampaignPhase.GAUSSIAN, observations=[])
    assert action is FailureAction.SCRUB_AND_CONTINUE


# --- DryRunSacctPoller -------------------------------------------------


def test_dry_run_sacct_returns_completed_for_dryrun_prefix():
    poller = DryRunSacctPoller(elapsed_seconds=0)
    obs = poller(DRYRUN_PREFIX + "GAUSSIAN-3")
    assert len(obs) == 1
    assert obs[0].status is JobStatus.COMPLETED
    assert obs[0].exit_code == (0, 0)


def test_dry_run_sacct_returns_every_expected_array_row():
    poller = DryRunSacctPoller(elapsed_seconds=0)

    observations = poller(
        DRYRUN_PREFIX + "AIMALL-0",
        expected_task_count=4,
    )

    assert [row.job_id for row in observations] == [
        DRYRUN_PREFIX + "AIMALL-0_0",
        DRYRUN_PREFIX + "AIMALL-0_1",
        DRYRUN_PREFIX + "AIMALL-0_2",
        DRYRUN_PREFIX + "AIMALL-0_3",
    ]
    assert all(row.status is JobStatus.COMPLETED for row in observations)


def test_dry_run_sacct_rejects_real_slurm_ids():
    poller = DryRunSacctPoller()
    with pytest.raises(ValueError, match="refuses non-synthetic"):
        poller("12345678")


def test_dry_run_sacct_records_invocations():
    poller = DryRunSacctPoller(elapsed_seconds=0)
    poller(DRYRUN_PREFIX + "FEREBUS-0")
    poller(DRYRUN_PREFIX + "AIMALL-1")
    assert poller.invocations == [DRYRUN_PREFIX + "FEREBUS-0", DRYRUN_PREFIX + "AIMALL-1"]
