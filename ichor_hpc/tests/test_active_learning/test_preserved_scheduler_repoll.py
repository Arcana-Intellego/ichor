from pathlib import Path
from types import SimpleNamespace

from ichor.hpc.active_learning import cli as cli_mod
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon import submission_intent
from ichor.hpc.active_learning.daemon.config_lock import write_config_lock
from ichor.hpc.active_learning.daemon.daemon import Daemon
from ichor.hpc.active_learning.daemon.environment_equivalence import (
    _repoll_control_module_digest,
)
from ichor.hpc.active_learning.daemon.preserved_scheduler_repoll import (
    resolve_preserved_scheduler_repoll_authority,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    CampaignState,
    make_lifecycle_context,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.daemon.stop_control import (
    archive_resume_transaction,
    prepare_resume_transaction,
    update_resume_transaction,
)
from ichor.hpc.active_learning.execution_identity import (
    advance_environment_generation,
    ensure_execution_identity,
    read_active_environment_generation,
)
from ichor.hpc.active_learning.daemon.preflight import BackendAvailability


def _campaign_with_restored_repoll(tmp_path: Path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = CampaignConfig(max_iterations=40)
    config.to_yaml(campaign / "campaign.yaml")
    before = CampaignState(
        iteration=29,
        max_iterations=40,
        phase=CampaignPhase.HALTED,
        pending_jobs={CampaignPhase.FEREBUS.value: "13937916"},
    )
    before.lifecycle_context = make_lifecycle_context(
        disposition="halted",
        reason_code="sacct_unknown_timeout",
        message="accounting remained unknown",
        from_phase=CampaignPhase.FEREBUS,
        iteration=29,
        source="daemon",
        job_id="13937916",
        scheduler_uncertain=True,
        recovery_action="resume",
    )
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    write_state(state_path, before)
    write_config_lock(
        campaign,
        config,
        campaign_uid=str(before.campaign_uid),
    )
    ensure_execution_identity(
        campaign,
        campaign_uid=str(before.campaign_uid),
        config=config,
        requested_mode="dry_run",
    )
    generation = read_active_environment_generation(
        campaign,
        expected_campaign_uid=str(before.campaign_uid),
    )["generation"]
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=str(before.campaign_uid),
        phase_name=CampaignPhase.FEREBUS.value,
        iteration=29,
        expected_tasks=4,
        scheduler_identity_kind="slurm",
        environment_generation=int(generation["generation"]),
        environment_generation_digest_sha256=str(generation["digest_sha256"]),
    )
    intent = submission_intent.mark_submitted(
        campaign,
        CampaignPhase.FEREBUS.value,
        29,
        "13937916",
        expected_tasks=4,
    )
    after = CampaignState.from_dict(before.to_dict())
    after.phase = CampaignPhase.FEREBUS
    after.lifecycle_context = None
    after.sacct_empty_streak = {}
    transaction = prepare_resume_transaction(
        campaign,
        before_state=before,
        after_state=after,
        request_id=None,
        operation="resume_scheduler_uncertain_with_pending_stop",
    )
    transaction = update_resume_transaction(
        campaign,
        str(transaction["transaction_id"]),
        status="control_archived",
    )
    write_state(state_path, after)
    transaction = update_resume_transaction(
        campaign,
        str(transaction["transaction_id"]),
        status="state_written",
    )
    archive_resume_transaction(campaign, str(transaction["transaction_id"]))
    return campaign, state_path, intent


def test_restored_scheduler_job_has_exact_repoll_authority(tmp_path):
    campaign, state_path, intent = _campaign_with_restored_repoll(tmp_path)

    authority = resolve_preserved_scheduler_repoll_authority(
        campaign,
        read_state(state_path),
    )

    assert authority is not None
    assert authority.phase == CampaignPhase.FEREBUS.value
    assert authority.iteration == 29
    assert authority.job_id == "13937916"
    assert authority.expected_tasks == 4
    assert authority.submission_identity == intent["submission_identity"]


def test_repoll_authority_survives_accounting_streak_progress(tmp_path):
    campaign, state_path, _intent = _campaign_with_restored_repoll(tmp_path)
    changed = read_state(state_path)
    changed.sacct_empty_streak["13937916:UNKNOWN"] = 1
    write_state(state_path, changed)

    authority = resolve_preserved_scheduler_repoll_authority(campaign, changed)

    assert authority is not None
    assert authority.job_id == "13937916"


def test_repoll_authority_expires_when_scheduler_ownership_changes(tmp_path):
    campaign, state_path, _intent = _campaign_with_restored_repoll(tmp_path)
    changed = read_state(state_path)
    changed.pending_jobs[CampaignPhase.FEREBUS.value] = "13937917"
    write_state(state_path, changed)

    assert resolve_preserved_scheduler_repoll_authority(campaign, changed) is None


def test_active_intent_without_resume_transaction_is_not_authorized(tmp_path):
    campaign, state_path, _intent = _campaign_with_restored_repoll(tmp_path)
    history = campaign / ".DATA" / "ACTIVE_LEARNING" / "resume_transaction_history"
    for path in history.glob("*.json"):
        path.unlink()

    assert (
        resolve_preserved_scheduler_repoll_authority(
            campaign,
            read_state(state_path),
        )
        is None
    )


def test_repoll_authority_accepts_identical_interrupted_transaction_archive(
    tmp_path,
):
    campaign, state_path, _intent = _campaign_with_restored_repoll(tmp_path)
    history = campaign / ".DATA" / "ACTIVE_LEARNING" / "resume_transaction_history"
    archived = next(history.glob("*.json"))
    active = history.parent / "resume_transaction.json"
    active.write_bytes(archived.read_bytes())

    authority = resolve_preserved_scheduler_repoll_authority(
        campaign,
        read_state(state_path),
    )

    assert authority is not None
    assert Path(authority.transaction_path) == archived


def test_repoll_equivalence_ignores_only_declared_daemon_control_methods():
    path = "ichor_hpc/ichor/hpc/active_learning/daemon/daemon.py"
    before = """
class Daemon:
    def _on_pending(self):
        return 'old polling'

    def _liveness_blocks_accounting_timeout(self):
        return True

    def scientific_postprocess(self):
        return 'same science'
"""
    after = """
class Daemon:
    def _on_pending(self):
        return 'new polling'

    def _accounting_liveness_gate(self):
        return 'new helper'

    def scientific_postprocess(self):
        return 'same science'
"""
    changed_science = after.replace("same science", "changed science")

    assert _repoll_control_module_digest(
        path, before
    ) == _repoll_control_module_digest(path, after)
    assert _repoll_control_module_digest(
        path, before
    ) != _repoll_control_module_digest(path, changed_science)


def test_preflight_allows_only_transaction_bound_preserved_repoll(
    tmp_path,
    monkeypatch,
):
    campaign, state_path, _intent = _campaign_with_restored_repoll(tmp_path)
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: {
            "ok": True,
            "pool_n_frames": 10000,
            "required_pool_frames": 4173,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "build_committed_artifact_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    from ichor.hpc.active_learning.daemon import artifact_contracts
    from ichor.hpc.active_learning.daemon import resource_solver

    monkeypatch.setattr(
        artifact_contracts,
        "state_artifact_contract_status",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        resource_solver,
        "validate_partition_supported",
        lambda _partition: None,
    )
    monkeypatch.setattr(
        resource_solver,
        "validate_partition_walltime",
        lambda _partition, _walltime: None,
    )
    availability = BackendAvailability(
        profile=True,
        sbatch=True,
        sacct=True,
        squeue=True,
        gaussian=True,
        gaussian_verified=True,
        aimall=True,
        ferebus=True,
        ariadne=True,
        pyferebus=True,
        bc=True,
        batch_python=True,
        gaussian_binary="jobscript:g16",
        sbatch_path="/usr/bin/sbatch",
        sacct_path="/usr/bin/sacct",
        squeue_path="/usr/bin/squeue",
        bc_path="/usr/bin/bc",
        aimall_path="/opt/aimall",
        ferebus_path="/opt/ferebus",
        active_profile="csf4",
        profile_error="",
        python_executable="/opt/venv/bin/python",
    )

    payload = cli_mod.evaluate_campaign_preflight(
        campaign,
        avail=availability,
    )

    assert payload["ready"] is True
    assert payload["campaign_state"]["issues"] == []
    assert payload["_presentation_scheduler_repoll"]["job_id"] == "13937916"
    assert read_state(state_path).pending_jobs == {
        CampaignPhase.FEREBUS.value: "13937916"
    }


def test_reconcile_preserves_exact_scheduler_repoll_state(
    tmp_path,
    monkeypatch,
):
    campaign, state_path, _intent = _campaign_with_restored_repoll(tmp_path)
    from ichor.hpc.active_learning.daemon import reconcile as reconcile_mod

    intent_inventory = submission_intent.inventory_intents(campaign)
    snapshot = SimpleNamespace(
        verification_level="authority",
        committed_reference_data_versions=[],
        committed_model_versions=[],
        valid_reference_data_versions=[],
        valid_model_versions=[],
        reference_errors={},
        model_errors={},
        reference_view=None,
        model_set=None,
        submission_intents=tuple(intent_inventory["records"]),
        submission_intent_errors=(),
        completion_receipts=(),
        completion_receipt_errors=(),
    )
    monkeypatch.setattr(
        reconcile_mod,
        "build_committed_artifact_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        reconcile_mod,
        "verify_state_referenced_artifacts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        reconcile_mod,
        "_needs_trajectory_pool_check",
        lambda **_kwargs: False,
    )

    report = reconcile_mod.propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.FEREBUS, (
        report.unsafe_reasons,
        report.blocking_artifacts,
        report.notes,
        report.decision,
    )
    assert report.proposed_state.pending_jobs == {
        CampaignPhase.FEREBUS.value: "13937916"
    }
    assert not report.unsafe_reasons
    assert "requires resume re-polling" in report.decision
    assert read_state(state_path).phase is CampaignPhase.FEREBUS


def test_environment_generation_advances_at_preserved_repoll_boundary(
    tmp_path,
    monkeypatch,
):
    campaign, state_path, intent = _campaign_with_restored_repoll(tmp_path)
    from ichor.hpc.active_learning import execution_identity
    from ichor.hpc.active_learning.daemon import artifact_contracts
    from ichor.hpc.active_learning.daemon import artifact_snapshot
    from ichor.hpc.active_learning.daemon import environment_equivalence
    from ichor.hpc.active_learning.daemon import reconcile_transaction

    monkeypatch.setattr(
        execution_identity,
        "_generation_binding_matches",
        lambda _left, _right: False,
    )
    monkeypatch.setattr(
        artifact_contracts,
        "verify_state_referenced_artifacts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        artifact_snapshot,
        "build_committed_artifact_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        reconcile_transaction,
        "inspect_reconcile_transaction_recovery",
        lambda *_args, **_kwargs: {"state": "none"},
    )
    monkeypatch.setattr(
        environment_equivalence,
        "assess_preserved_scheduler_repoll_equivalence",
        lambda _producer, _current: {
            "equivalent": True,
            "checks": {"repoll_control_only": True},
            "changed_runtime_paths": [],
            "reasons": [],
        },
    )

    result = advance_environment_generation(
        campaign,
        config=CampaignConfig.from_yaml(campaign / "campaign.yaml"),
        live_preflight_ok=True,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["transition_kind"] == "preserved_scheduler_repoll"
    assert read_state(state_path).pending_jobs == {
        CampaignPhase.FEREBUS.value: "13937916"
    }
    current_intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.FEREBUS.value,
        29,
    )
    assert current_intent["environment_generation"] == intent["environment_generation"]
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=str(read_state(state_path).campaign_uid),
    )["generation"]
    assert int(active["generation"]) == 1

    authority = resolve_preserved_scheduler_repoll_authority(
        campaign,
        read_state(state_path),
    )
    assert authority is not None
    daemon = Daemon(
        campaign,
        CampaignConfig.from_yaml(campaign / "campaign.yaml"),
    )
    daemon._preserved_scheduler_repoll_authority = authority
    daemon._last_environment_binding = {
        "generation": int(active["generation"]),
        "generation_digest_sha256": str(active["digest_sha256"]),
        "campaign_config_sha256": str(active["campaign_config_sha256"]),
    }

    assert daemon._verify_intent_environment_binding(
        read_state(state_path),
        CampaignPhase.FEREBUS,
        current_intent,
    ) is None
