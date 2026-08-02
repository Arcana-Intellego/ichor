from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import ichor.hpc.active_learning.execution_identity as execution_identity_module
import ichor.hpc.active_learning.daemon.config_lock as config_lock_module
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.execution_identity import (
    ExecutionIdentityError,
    _canonical_digest,
    _environment_fingerprint,
    assert_environment_unchanged,
    environment_status,
    ensure_execution_identity,
    environment_current_path,
    environment_generations_dir,
    execution_identity_path,
    legacy_intent_config_generation_split_is_proven,
    read_active_environment_generation,
    rebind_environment,
)
from ichor.hpc.active_learning.daemon.config_lock import (
    config_fingerprint,
    ensure_config_lock,
    write_config_lock,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.daemon.daemon import Daemon, TickStatus
from ichor.hpc.active_learning.daemon.phase_executor import PhaseResult
from ichor.hpc.active_learning.daemon.submission_intent import (
    ariadne_producer_environment_binding,
    intent_path,
    load_intent,
    mark_failed,
    mark_submitted,
    mark_superseded,
    record_queue_lifecycle,
    resolve_aimall_postprocess_source,
    resolve_ariadne_postprocess_source,
    write_pre_submit_intent,
)
from ichor.hpc.active_learning.cli import build_parser


def _fake_generation(*args, campaign_uid, config, generation=0, **kwargs):
    campaign = Path(args[0] if args else ".").resolve()
    payload = {
        "schema_version": 1,
        "generation": int(generation),
        "campaign_uid": str(campaign_uid),
        "created_at_iso": "2026-01-01T00:00:00+00:00",
        "host": "test-host",
        "operator": "test-operator",
        "python_executable": str((Path.cwd() / "python-test").resolve()),
        "campaign_schema_version": int(config.schema_version),
        "python_version": "3.11.test",
        "ichor_git": {},
        "ichor_package_tree_sha256": "0" * 64,
        "dependencies": [],
        "pyferebus": {},
        "ariadne": {},
        "ferebus_executable": {},
        "machine_profile": {},
        "loaded_modules": [],
        "native_library_paths": {
            "LD_LIBRARY_PATH": "",
            "LIBRARY_PATH": "",
        },
        "campaign_config_sha256": config_fingerprint(config.to_dict()),
        "config_lock_sha256": None,
        "campaign_dir": str(campaign),
    }
    payload["environment_fingerprint_sha256"] = _environment_fingerprint(payload)
    payload["digest_sha256"] = _canonical_digest(payload)
    return payload


def test_first_start_requires_explicit_mode(tmp_path):
    with pytest.raises(ExecutionIdentityError, match="first start requires --mode"):
        ensure_execution_identity(
            tmp_path,
            campaign_uid="uid-1",
            config=CampaignConfig(),
            requested_mode=None,
        )

    assert not execution_identity_path(tmp_path).exists()


def test_package_identity_never_uses_recursive_filesystem_walkers(
    tmp_path,
    monkeypatch,
):
    package_root = tmp_path / "bounded_package"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

    def refuse_recursive_walk(*_args, **_kwargs):
        raise AssertionError("environment capture attempted a recursive walk")

    monkeypatch.setattr(Path, "rglob", refuse_recursive_walk)
    monkeypatch.setattr(os, "walk", refuse_recursive_walk)

    observed = execution_identity_module._tree_hash([package_root])

    assert len(observed) == 64


def test_first_start_binds_mode_seed_and_environment_generation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    config.campaign.reproducibility_seed = 42

    mode, payload = ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-2",
        config=config,
        requested_mode="dry_run",
    )

    assert mode == "dry_run"
    assert payload["campaign_random_seed"] == 42
    assert execution_identity_path(tmp_path).is_file()
    assert (environment_generations_dir(tmp_path) / "generation-000000.json").is_file()
    current = json.loads(environment_current_path(tmp_path).read_text(encoding="utf-8"))
    assert current["generation"] == 0
    generation = json.loads(
        (environment_generations_dir(tmp_path) / "generation-000000.json").read_text(
            encoding="utf-8"
        )
    )
    assert current["generation_digest_sha256"] == generation["digest_sha256"]


def test_bound_mode_cannot_change(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-3",
        config=config,
        requested_mode="live",
    )

    with pytest.raises(ExecutionIdentityError, match="permanently bound to live"):
        ensure_execution_identity(
            tmp_path,
            campaign_uid="uid-3",
            config=config,
            requested_mode="dry_run",
        )


def test_bound_random_seed_cannot_change(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-seed",
        config=config,
        requested_mode="dry_run",
    )
    config.campaign.reproducibility_seed = 7

    with pytest.raises(ExecutionIdentityError, match="reproducibility_seed"):
        ensure_execution_identity(
            tmp_path,
            campaign_uid="uid-seed",
            config=config,
            requested_mode=None,
        )


def test_tampered_execution_identity_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-4",
        config=config,
        requested_mode="dry_run",
    )
    path = execution_identity_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mode"] = "live"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExecutionIdentityError, match="digest mismatch"):
        ensure_execution_identity(
            tmp_path,
            campaign_uid="uid-4",
            config=config,
            requested_mode=None,
        )


def test_environment_status_detects_execution_field_drift(tmp_path, monkeypatch):
    current_python = {"value": "3.11.test"}

    def generation(*args, campaign_uid, config, generation=0, **kwargs):
        payload = _fake_generation(
            campaign_uid=campaign_uid,
            config=config,
            generation=generation,
        )
        payload["python_version"] = current_python["value"]
        payload["environment_fingerprint_sha256"] = _environment_fingerprint(payload)
        payload["digest_sha256"] = _canonical_digest(payload)
        return payload

    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-drift",
        config=config,
        requested_mode="dry_run",
    )

    assert environment_status(
        tmp_path, campaign_uid="uid-drift", config=config
    )["matches"] is True
    current_python["value"] = "3.11.changed"
    status = environment_status(tmp_path, campaign_uid="uid-drift", config=config)

    assert status["matches"] is False
    assert status["changed_fields"] == ["python_version"]
    with pytest.raises(ExecutionIdentityError, match="python_version"):
        assert_environment_unchanged(
            tmp_path,
            campaign_uid="uid-drift",
            config=config,
        )


def test_environment_status_detects_config_only_binding_drift(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-config-drift",
        config=config,
        requested_mode="dry_run",
    )
    changed = CampaignConfig.from_dict(config.to_dict())
    changed.campaign.sampling_aggressiveness += 1

    status = environment_status(
        tmp_path,
        campaign_uid="uid-config-drift",
        config=changed,
    )

    assert status["matches"] is False
    assert status["changed_fields"] == ["campaign_config_sha256"]


def test_ferebus_identity_uses_profile_executable_when_environment_is_unset(
    tmp_path,
    monkeypatch,
):
    executable = tmp_path / "ferebus"
    executable.write_bytes(b"scientific executable")
    monkeypatch.delenv("FEREBUS_PATH", raising=False)
    monkeypatch.setattr(execution_identity_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        execution_identity_module,
        "_profile_ferebus_executable",
        lambda: str(executable),
    )

    identity = execution_identity_module._configured_ferebus_identity()

    assert identity["path"] == str(executable.resolve())
    assert identity["sha256"] == execution_identity_module._sha256_file(executable)


def test_active_environment_pointer_tampering_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-pointer",
        config=config,
        requested_mode="dry_run",
    )
    pointer_path = environment_current_path(tmp_path)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_path"] = "elsewhere.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(ExecutionIdentityError, match="not canonical"):
        read_active_environment_generation(
            tmp_path,
            expected_campaign_uid="uid-pointer",
        )


def test_active_environment_generation_tampering_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-generation",
        config=config,
        requested_mode="dry_run",
    )
    generation_path = (
        environment_generations_dir(tmp_path) / "generation-000000.json"
    )
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation["python_version"] = "tampered"
    generation_path.write_text(json.dumps(generation), encoding="utf-8")

    with pytest.raises(ExecutionIdentityError, match="fingerprint mismatch"):
        read_active_environment_generation(
            tmp_path,
            expected_campaign_uid="uid-generation",
        )


def test_rehashed_but_incomplete_environment_generation_is_rejected(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _fake_generation,
    )
    config = CampaignConfig()
    ensure_execution_identity(
        tmp_path,
        campaign_uid="uid-incomplete-generation",
        config=config,
        requested_mode="dry_run",
    )
    generation_path = environment_generations_dir(tmp_path) / "generation-000000.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation.pop("dependencies")
    generation["environment_fingerprint_sha256"] = _environment_fingerprint(
        generation
    )
    generation["digest_sha256"] = _canonical_digest(generation)
    generation_path.write_text(json.dumps(generation), encoding="utf-8")

    with pytest.raises(ExecutionIdentityError, match="missing dependencies"):
        read_active_environment_generation(
            tmp_path,
            expected_campaign_uid="uid-incomplete-generation",
        )


def _rebind_campaign(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = CampaignConfig()
    config.to_yaml(campaign / "campaign.yaml")
    state = fresh_campaign_state(campaign_uid="uid-rebind")
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = -1
    state.models_version = -1
    state.reference_scales = {
        "energy": 1.0,
        "force": 1.0,
        "omega": 1.0,
        "anh": 1.0,
        "anh_std": 1.0,
    }
    state.reference_scales_iteration = 1
    state.reference_scales_models_version = 0
    state.reference_scales_model_manifest_sha256 = "a" * 64
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    ensure_config_lock(campaign, config, campaign_uid=state.campaign_uid)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        _fake_generation,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.artifact_contracts.verify_state_referenced_artifacts",
        lambda *_args, **_kwargs: None,
    )
    ensure_execution_identity(
        campaign,
        campaign_uid=state.campaign_uid,
        config=config,
        requested_mode="dry_run",
    )
    return campaign, config, state


def test_rebind_rejects_non_idle_or_scheduler_owned_state(tmp_path, monkeypatch):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    state.phase = CampaignPhase.GAUSSIAN
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    with pytest.raises(
        ExecutionIdentityError,
        match="SEED_SELECT, STOP_CHECK or DONE",
    ):
        rebind_environment(campaign, config=config)

    state.phase = CampaignPhase.SEED_SELECT
    state.pending_jobs[CampaignPhase.SEED_SELECT.value] = "dry-1"
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    with pytest.raises(ExecutionIdentityError, match="scheduler ownership"):
        rebind_environment(campaign, config=config)


def _patch_ariadne_retry_transition(
    monkeypatch,
    *,
    logical_total=3,
    n_complete=0,
    retry_task_ids=None,
    contract_error=None,
):
    retry_ids = (
        list(range(int(logical_total)))
        if retry_task_ids is None
        else list(retry_task_ids)
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.recovery_contracts."
        "phase_recovery_contract_error",
        lambda *_args, **_kwargs: contract_error,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.array_recovery.scan_array_tasks",
        lambda _campaign, phase, iteration, force_resubmit=False: {
            "phase": str(getattr(phase, "value", phase)),
            "iteration": int(iteration),
            "logical_total": int(logical_total),
            "n_complete": int(n_complete),
            "n_reuse": int(n_complete),
            "n_retry": len(retry_ids),
            "retry_task_ids": retry_ids,
            "all_complete": bool(logical_total > 0 and not retry_ids),
        },
    )


def _patch_phase_b_transition(
    monkeypatch,
    *,
    accepted_tasks=193,
    rejected_tasks=7,
):
    from ichor.hpc.active_learning.versioning.reference_data import (
        ReferenceDataVersioning,
    )
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelVersioning,
    )

    monkeypatch.setattr(
        ReferenceDataVersioning,
        "current_version",
        lambda _self: 0,
    )
    monkeypatch.setattr(
        TrainedModelVersioning,
        "current_version",
        lambda _self: 0,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.recovery_contracts."
        "phase_recovery_contract_error",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.recovery_contracts."
        "ariadne_results_recovery_summary",
        lambda *_args, **_kwargs: {
            "expected_tasks": 200,
            "accepted_tasks": int(accepted_tasks),
            "rejected_tasks": int(rejected_tasks),
            "missing_rejected_outputs": int(rejected_tasks),
            "tasks_resubmitted": 0,
        },
    )


def _write_terminal_scalar_diversity_intent(
    campaign,
    state,
    *,
    job_id="17997698",
    terminal_status="FAILED",
    scheduler_kind="slurm",
):
    write_pre_submit_intent(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase_name=state.phase.value,
        iteration=int(state.iteration),
        replacement_round=int(state.replacement_round),
        expected_tasks=1,
        scheduler_identity_kind=scheduler_kind,
    )
    mark_submitted(
        campaign,
        state.phase.value,
        int(state.iteration),
        str(job_id),
        expected_tasks=1,
    )
    record_queue_lifecycle(
        campaign,
        state.phase.value,
        int(state.iteration),
        "terminal",
        job_id=str(job_id),
        status=str(terminal_status),
        n_expected=1,
        n_observed=1,
        n_missing=0,
    )
    mark_failed(
        campaign,
        state.phase.value,
        int(state.iteration),
        "test scalar worker failure",
    )
    return load_intent(
        campaign,
        state.phase.value,
        int(state.iteration),
        expected_campaign_uid=str(state.campaign_uid),
    )


def test_rebind_accepts_clean_phase_b_pre_submission_retry(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.PHASE_B_DIVERSITY
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    _patch_phase_b_transition(monkeypatch)

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["transition_kind"] == "phase_b_pre_submission_retry"
    assert result["ariadne_accepted_tasks"] == 193
    assert result["ariadne_rejected_tasks"] == 7
    assert result["ariadne_tasks_resubmitted"] == 0


def test_rebind_accepts_terminal_phase_b_scheduler_retry(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.PHASE_B_DIVERSITY
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    from ichor.hpc.active_learning.daemon.stop_control import (
        build_stop_request,
        install_stop_request,
        stop_request_path,
    )

    install_stop_request(
        campaign,
        build_stop_request(state, mode="after_iteration"),
    )
    stop_bytes = stop_request_path(campaign).read_bytes()
    _write_terminal_scalar_diversity_intent(campaign, state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.cluster_profile.profile_value",
        lambda *_args, **_kwargs: "slurm",
    )
    journal_events = []
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.journal.append_event",
        lambda _path, event, **fields: journal_events.append(
            {"event": event, **fields}
        ),
    )
    _patch_phase_b_transition(
        monkeypatch,
        accepted_tasks=198,
        rejected_tasks=2,
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["transition_kind"] == "phase_b_terminal_scheduler_retry"
    assert result["producer_job_id"] == "17997698"
    assert result["scheduler_terminal_status"] == "FAILED"
    assert result["ariadne_accepted_tasks"] == 198
    assert result["ariadne_rejected_tasks"] == 2
    transition_event = journal_events[-1]
    assert transition_event["event"] == "environment_generation_advanced"
    assert (
        transition_event["transition_kind"]
        == "phase_b_terminal_scheduler_retry"
    )
    assert transition_event["producer_job_id"] == "17997698"
    assert transition_event["scheduler_terminal_status"] == "FAILED"
    assert stop_request_path(campaign).read_bytes() == stop_bytes

    retry = write_pre_submit_intent(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase_name=state.phase.value,
        iteration=int(state.iteration),
        expected_tasks=1,
        scheduler_identity_kind="slurm",
    )
    history = (
        campaign
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "submission_intents"
        / "history"
    )
    archived = list(history.glob("PHASE_B_DIVERSITY-000001-*.json"))
    assert retry["attempt_sequence"] == 2
    assert retry["job_id"] is None
    assert len(archived) == 1
    assert json.loads(archived[0].read_text(encoding="utf-8"))["job_id"] == (
        "17997698"
    )


@pytest.mark.parametrize("scheduler_kind", ["slurm", "sge"])
def test_rebind_accepts_terminal_phase_a_scheduler_retry(
    tmp_path,
    monkeypatch,
    scheduler_kind,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.PHASE_A_DIVERSITY
    state.iteration = 0
    state.reference_data_version = -1
    state.models_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    _write_terminal_scalar_diversity_intent(
        campaign,
        state,
        job_id="17997000",
        scheduler_kind=scheduler_kind,
    )
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.cluster_profile.profile_value",
        lambda *_args, **_kwargs: scheduler_kind,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.recovery_contracts."
        "phase_recovery_contract_error",
        lambda *_args, **_kwargs: None,
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["transition_kind"] == "phase_a_terminal_scheduler_retry"
    assert result["producer_job_id"] == "17997000"
    assert result["scheduler_terminal_status"] == "FAILED"


def test_rebind_rejects_phase_b_partial_output(tmp_path, monkeypatch):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.PHASE_B_DIVERSITY
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    _patch_phase_b_transition(monkeypatch)
    phase_b = campaign / "ACTIVE_LEARNING" / "iteration-000001" / "phase_b"
    phase_b.mkdir(parents=True)
    (phase_b / "selected.xyz").write_text("partial\n", encoding="utf-8")

    with pytest.raises(ExecutionIdentityError, match="partial Phase B output"):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )


def test_phase_b_transition_accepts_jobless_and_terminal_scheduler_retry_intents(
    tmp_path,
    monkeypatch,
):
    campaign, _config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.PHASE_B_DIVERSITY
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    _patch_phase_b_transition(monkeypatch)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.cluster_profile.profile_value",
        lambda *_args, **_kwargs: "slurm",
    )
    intent = {
        "campaign_uid": str(state.campaign_uid),
        "phase": CampaignPhase.PHASE_B_DIVERSITY.value,
        "iteration": 1,
        "replacement_round": 0,
        "scheduler_identity_kind": "slurm",
        "submission_kind": "scalar",
        "expected_tasks": 1,
        "submission_identity": "r0000-a0001-test",
        "status": "SUPERSEDED",
        "reason": "reconcile_apply_retry",
        "job_id": None,
        "job_ids_seen": [],
    }

    result = execution_identity_module._validate_phase_b_transition_boundary(
        campaign,
        state,
        intent_records=[intent],
    )

    assert result["transition_kind"] == "phase_b_pre_submission_retry"
    intent["job_id"] = "12345"
    intent["job_ids_seen"] = ["12345"]
    intent["queue_lifecycle"] = {
        "terminal_status": "FAILED",
        "n_expected": 1,
        "n_observed": 1,
        "n_missing": 0,
    }
    terminal = execution_identity_module._validate_phase_b_transition_boundary(
        campaign,
        state,
        intent_records=[intent],
    )

    assert terminal["transition_kind"] == "phase_b_terminal_scheduler_retry"
    assert terminal["producer_job_id"] == "12345"
    assert terminal["scheduler_terminal_status"] == "FAILED"

    intent["queue_lifecycle"]["n_missing"] = 1
    with pytest.raises(
        ExecutionIdentityError,
        match="exactly one terminal task",
    ):
        execution_identity_module._validate_phase_b_transition_boundary(
            campaign,
            state,
            intent_records=[intent],
        )


def _active_pending_replacement_allocation(campaign, state):
    from ichor.hpc.active_learning.point_allocation import (
        allocate_replacements,
        create_point_allocation,
        pending_attempts,
        point_allocation_path,
        record_quantum_results,
    )

    path = point_allocation_path(
        campaign,
        context="active",
        iteration=int(state.iteration),
    )
    allocation = create_point_allocation(
        path,
        campaign_uid=str(state.campaign_uid),
        context="active",
        iteration=int(state.iteration),
        targets={"train": 1, "int_val": 0, "ext_val": 0, "total": 1},
        primary_candidates=[{"candidate_id": "primary"}],
        reserve_candidates=[
            {"candidate_id": "reserve", "reserve_rank": 0},
        ],
    )
    primary = pending_attempts(allocation)[0]
    rejected = record_quantum_results(
        path,
        [
            {
                "candidate_id": str(primary["candidate_id"]),
                "accepted": False,
                "pointdir": "/synthetic/primary.pointdir",
                "reason": "fixture rejection",
            }
        ],
    )
    return allocate_replacements(
        path,
        replacement_round=1,
        expected_generation=int(rejected["generation"]),
    )


def _patch_active_allocation_transition(monkeypatch, *, sample_state):
    from ichor.hpc.active_learning.versioning.reference_data import (
        ReferenceDataVersioning,
    )
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelVersioning,
    )

    monkeypatch.setattr(
        ReferenceDataVersioning,
        "current_version",
        lambda _self: 0,
    )
    monkeypatch.setattr(
        TrainedModelVersioning,
        "current_version",
        lambda _self: 0,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.replacement_sampling."
        "inspect_replacement_sample_recovery",
        lambda *_args, **_kwargs: {
            "state": sample_state,
            "reason": (
                "fixture conflict" if sample_state == "conflicting" else None
            ),
            "n_candidates": 1,
        },
    )


@pytest.mark.parametrize(
    ("stop_mode", "phase_started"),
    [
        ("immediate", False),
        ("after_phase", False),
        ("after_phase", True),
        ("after_iteration", False),
    ],
)
def test_rebind_accepts_rebuildable_active_allocation_check_and_preserves_stop(
    tmp_path,
    monkeypatch,
    stop_mode,
    phase_started,
):
    from ichor.hpc.active_learning.daemon.stop_control import (
        build_stop_request,
        install_stop_request,
        stop_request_path,
    )

    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ALLOCATION_CHECK
    state.iteration = 1
    state.reference_data_version = 0
    state.validation_set_version = 0
    state.models_version = 0
    state.replacement_round = 1
    _active_pending_replacement_allocation(campaign, state)
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    request, _status = install_stop_request(
        campaign,
        build_stop_request(
            state,
            mode=stop_mode,
            target_iteration=1 if stop_mode == "after_iteration" else None,
            phase_started=phase_started,
        ),
    )
    stop_before = stop_request_path(campaign).read_bytes()
    _patch_active_allocation_transition(
        monkeypatch,
        sample_state="missing_rebuildable",
    )
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["transition_kind"] == "allocation_check_recovery"
    assert result["pending_tasks"] == 1
    assert result["replacement_sample_state"] == "missing_rebuildable"
    rebound = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert rebound.phase is CampaignPhase.ALLOCATION_CHECK
    assert rebound.iteration == 1
    assert rebound.replacement_round == 1
    assert stop_request_path(campaign).read_bytes() == stop_before
    assert request["status"] == "requested"


def test_rebind_rejects_conflicting_allocation_sample_before_generation_write(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ALLOCATION_CHECK
    state.iteration = 1
    state.reference_data_version = 0
    state.validation_set_version = 0
    state.models_version = 0
    state.replacement_round = 1
    _active_pending_replacement_allocation(campaign, state)
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    _patch_active_allocation_transition(
        monkeypatch,
        sample_state="conflicting",
    )
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )

    with pytest.raises(
        ExecutionIdentityError,
        match="replacement sample evidence conflicts",
    ):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )

    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    assert active["generation"]["generation"] == 0


def _write_full_ariadne_producer_intent(
    campaign,
    state,
    *,
    logical_total=3,
    status="FAILED",
):
    task_digest = hashlib.sha256(
        ",".join(str(index) for index in range(int(logical_total))).encode("ascii")
    ).hexdigest()
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )["generation"]
    write_pre_submit_intent(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase_name=CampaignPhase.ARIADNE_ARRAY.value,
        iteration=int(state.iteration),
        expected_tasks=int(logical_total),
        decision_contract={
            "failure_threshold_fraction": 0.25,
            "config_sha256": "c" * 64,
        },
        environment_generation=int(active["generation"]),
        environment_generation_digest_sha256=str(active["digest_sha256"]),
    )
    mark_submitted(
        campaign,
        CampaignPhase.ARIADNE_ARRAY.value,
        int(state.iteration),
        "12345",
        expected_tasks=int(logical_total),
        submission_metadata={
            "array_recovery": {
                "logical_total": int(logical_total),
                "n_complete": 0,
                "n_reuse": 0,
                "n_retry": int(logical_total),
            },
            "logical_task_set_sha256": task_digest,
        },
    )
    mark_failed(
        campaign,
        CampaignPhase.ARIADNE_ARRAY.value,
        int(state.iteration),
        "test postprocess failure",
    )
    if status == "SUPERSEDED":
        mark_superseded(
            campaign,
            CampaignPhase.ARIADNE_ARRAY.value,
            int(state.iteration),
            "reconcile_apply_retry",
        )
    return load_intent(
        campaign,
        CampaignPhase.ARIADNE_ARRAY.value,
        int(state.iteration),
        expected_campaign_uid=str(state.campaign_uid),
    )


def _write_aimall_producer_intent(campaign, state, *, logical_total=2):
    from ichor.hpc.active_learning.daemon import input_staging as staging

    staging_root = (
        campaign
        / ".DATA"
        / "STAGING"
        / ("iter_" + str(int(state.iteration)))
    )
    accepted = []
    for logical_task_id in range(int(logical_total)):
        pointdir = staging_root / (
            "POINT_" + str(logical_task_id).zfill(4) + ".pointdir"
        )
        pointdir.mkdir(parents=True, exist_ok=True)
        accepted.append(pointdir)
    staging.write_quantum_acceptance_manifest(
        staging_root,
        phase_name=CampaignPhase.GAUSSIAN.value,
        iteration=int(state.iteration),
        accepted=accepted,
        rejected=[("POINT_9999.pointdir", "gaussian_failed")],
    )
    staging.write_points_file(staging_root, accepted)
    task_digest = hashlib.sha256(
        ",".join(str(index) for index in range(int(logical_total))).encode(
            "ascii"
        )
    ).hexdigest()
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )["generation"]
    write_pre_submit_intent(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase_name=CampaignPhase.AIMALL.value,
        iteration=int(state.iteration),
        expected_tasks=int(logical_total),
        decision_contract={
            "failure_threshold_fraction": 0.25,
            "config_sha256": "c" * 64,
        },
        environment_generation=int(active["generation"]),
        environment_generation_digest_sha256=str(active["digest_sha256"]),
    )
    mark_submitted(
        campaign,
        CampaignPhase.AIMALL.value,
        int(state.iteration),
        "17888108",
        expected_tasks=int(logical_total),
        submission_metadata={
            "logical_task_set_sha256": task_digest,
        },
    )
    record_queue_lifecycle(
        campaign,
        CampaignPhase.AIMALL.value,
        int(state.iteration),
        "terminal",
        job_id="17888108",
        status="COMPLETED",
        n_expected=int(logical_total),
        n_observed=int(logical_total),
        n_missing=0,
    )
    mark_failed(
        campaign,
        CampaignPhase.AIMALL.value,
        int(state.iteration),
        "prior_gaussian_acceptance_manifest_invalid",
    )
    return load_intent(
        campaign,
        CampaignPhase.AIMALL.value,
        int(state.iteration),
        expected_campaign_uid=str(state.campaign_uid),
    )


def _patch_aimall_transition(monkeypatch, *, current_version=0):
    from ichor.hpc.active_learning.versioning.reference_data import (
        ReferenceDataVersioning,
    )
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelVersioning,
    )

    monkeypatch.setattr(
        ReferenceDataVersioning,
        "current_version",
        lambda _self: int(current_version),
    )
    monkeypatch.setattr(
        TrainedModelVersioning,
        "current_version",
        lambda _self: int(current_version),
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.recovery_contracts."
        "phase_recovery_contract_error",
        lambda *_args, **_kwargs: None,
    )


@pytest.mark.parametrize(
    (
        "phase",
        "iteration",
        "committed_version",
        "current_version",
        "replacement_round",
    ),
    [
        (CampaignPhase.INITIAL_AIMALL, 0, -1, None, 0),
        (CampaignPhase.INITIAL_REPLACEMENT_AIMALL, 0, -1, None, 1),
        (CampaignPhase.AIMALL, 2, 1, 1, 0),
        (CampaignPhase.REPLACEMENT_AIMALL, 2, 1, 1, 1),
    ],
)
def test_aimall_postprocess_environment_boundary_covers_all_phases(
    tmp_path,
    monkeypatch,
    phase,
    iteration,
    committed_version,
    current_version,
    replacement_round,
):
    from ichor.hpc.active_learning.versioning.reference_data import (
        ReferenceDataVersioning,
    )
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelVersioning,
    )

    state = fresh_campaign_state(max_iterations=3)
    state.phase = phase
    state.iteration = iteration
    state.replacement_round = replacement_round
    state.reference_data_version = committed_version
    state.models_version = committed_version
    monkeypatch.setattr(
        ReferenceDataVersioning,
        "current_version",
        lambda _self: current_version,
    )
    monkeypatch.setattr(
        TrainedModelVersioning,
        "current_version",
        lambda _self: current_version,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.recovery_contracts."
        "phase_recovery_contract_error",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.submission_intent."
        "resolve_aimall_postprocess_source",
        lambda *_args, **_kwargs: {
            "logical_total": 2,
            "submission_identity": "r0000-a0001-fixture",
            "job_id": "17888108",
            "environment_generation": 4,
            "environment_generation_digest_sha256": "d" * 64,
        },
    )

    result = (
        execution_identity_module._validate_aimall_postprocess_transition_boundary(
            tmp_path,
            state,
        )
    )

    assert result["transition_kind"] == "aimall_postprocess_only"
    assert result["logical_total"] == 2


@pytest.mark.parametrize(
    (
        "phase",
        "iteration",
        "committed_version",
        "current_version",
        "replacement_round",
    ),
    [
        (CampaignPhase.INITIAL_GAUSSIAN, 0, -1, None, 0),
        (
            CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
            0,
            -1,
            None,
            1,
        ),
        (CampaignPhase.GAUSSIAN, 2, 1, 1, 0),
        (CampaignPhase.REPLACEMENT_GAUSSIAN, 2, 1, 1, 1),
    ],
)
def test_gaussian_postprocess_environment_boundary_covers_all_phases(
    tmp_path,
    monkeypatch,
    phase,
    iteration,
    committed_version,
    current_version,
    replacement_round,
):
    from ichor.hpc.active_learning.versioning.reference_data import (
        ReferenceDataVersioning,
    )
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelVersioning,
    )

    state = fresh_campaign_state(max_iterations=3)
    state.phase = phase
    state.iteration = iteration
    state.replacement_round = replacement_round
    state.reference_data_version = committed_version
    state.models_version = committed_version
    monkeypatch.setattr(
        ReferenceDataVersioning,
        "current_version",
        lambda _self: current_version,
    )
    monkeypatch.setattr(
        TrainedModelVersioning,
        "current_version",
        lambda _self: current_version,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.recovery_contracts."
        "phase_recovery_contract_error",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.submission_intent."
        "resolve_gaussian_postprocess_source",
        lambda *_args, **_kwargs: {
            "logical_total": 2,
            "submission_identity": "r0000-a0001-fixture",
            "job_id": "17888108",
            "environment_generation": 4,
            "environment_generation_digest_sha256": "d" * 64,
        },
    )

    result = (
        execution_identity_module._validate_gaussian_postprocess_transition_boundary(
            tmp_path,
            state,
        )
    )

    assert result["transition_kind"] == "gaussian_postprocess_only"
    assert result["logical_total"] == 2


def test_initial_aimall_postprocess_boundary_rejects_committed_current_pointer(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.versioning.reference_data import (
        ReferenceDataVersioning,
    )
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelVersioning,
    )

    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.INITIAL_AIMALL
    state.iteration = 0
    state.reference_data_version = -1
    state.models_version = -1
    monkeypatch.setattr(
        ReferenceDataVersioning,
        "current_version",
        lambda _self: 0,
    )
    monkeypatch.setattr(
        TrainedModelVersioning,
        "current_version",
        lambda _self: 0,
    )

    with pytest.raises(
        execution_identity_module.ExecutionIdentityError,
        match="requires no committed current",
    ):
        execution_identity_module._validate_aimall_postprocess_transition_boundary(
            tmp_path,
            state,
        )


def test_rebind_accepts_scheduler_complete_aimall_postprocess_boundary(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.AIMALL
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    producer = _write_aimall_producer_intent(campaign, state, logical_total=2)
    mark_superseded(
        campaign,
        CampaignPhase.AIMALL.value,
        1,
        "reconcile_apply_retry",
    )
    _patch_aimall_transition(monkeypatch)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["transition_kind"] == "aimall_postprocess_only"
    assert result["logical_total"] == 2
    assert result["producer_job_id"] == "17888108"
    assert (
        result["postprocess_source"]["submission_identity"]
        == producer["submission_identity"]
    )


def test_aimall_postprocess_source_survives_local_failure_wrapper(
    tmp_path,
    monkeypatch,
):
    campaign, _config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.AIMALL
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    producer = _write_aimall_producer_intent(campaign, state, logical_total=2)
    source = resolve_aimall_postprocess_source(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase_name=CampaignPhase.AIMALL.value,
        iteration=1,
    )
    mark_superseded(
        campaign,
        CampaignPhase.AIMALL.value,
        1,
        "reconcile_apply_retry",
    )
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )["generation"]
    write_pre_submit_intent(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase_name=CampaignPhase.AIMALL.value,
        iteration=1,
        expected_tasks=2,
        decision_contract=dict(source["decision_contract"]),
        postprocess_source=dict(source),
        environment_generation=int(active["generation"]),
        environment_generation_digest_sha256=str(active["digest_sha256"]),
    )
    mark_failed(
        campaign,
        CampaignPhase.AIMALL.value,
        1,
        "second local postprocess failure",
    )

    repeated = resolve_aimall_postprocess_source(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase_name=CampaignPhase.AIMALL.value,
        iteration=1,
    )

    assert repeated == source
    assert repeated["attempt_id"] == producer["attempt_id"]
    assert repeated["job_id"] == "17888108"


def test_aimall_postprocess_source_rejects_incomplete_scheduler_evidence(
    tmp_path,
    monkeypatch,
):
    campaign, _config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.AIMALL
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    producer = _write_aimall_producer_intent(campaign, state, logical_total=2)
    invalid = dict(producer)
    invalid["queue_lifecycle"] = dict(producer["queue_lifecycle"])
    invalid["queue_lifecycle"]["n_missing"] = 1

    with pytest.raises(
        ValueError,
        match="does not prove complete task ownership",
    ):
        resolve_aimall_postprocess_source(
            campaign,
            campaign_uid=str(state.campaign_uid),
            phase_name=CampaignPhase.AIMALL.value,
            iteration=1,
            intent=invalid,
        )


def test_aimall_postprocess_source_rejects_task_set_mismatch(
    tmp_path,
    monkeypatch,
):
    campaign, _config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.AIMALL
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    producer = _write_aimall_producer_intent(campaign, state, logical_total=2)
    invalid = dict(producer)
    invalid["submission_metadata"] = dict(producer["submission_metadata"])
    invalid["submission_metadata"]["logical_task_set_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="task-set digest mismatch"):
        resolve_aimall_postprocess_source(
            campaign,
            campaign_uid=str(state.campaign_uid),
            phase_name=CampaignPhase.AIMALL.value,
            iteration=1,
            intent=invalid,
        )


def test_rebind_accepts_fully_retryable_ariadne_boundary(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    _patch_ariadne_retry_transition(monkeypatch, logical_total=3)

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["generation"] == 1
    rebound = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert rebound.phase is CampaignPhase.ARIADNE_ARRAY
    assert rebound.iteration == 1


def test_rebind_accepts_real_all_retry_ariadne_task_map(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.handoff_manifests import (
        build_seed_selection_manifest,
        seeds_picked_path,
    )
    from ichor.hpc.active_learning.layout import active_iteration_dir
    from ichor.hpc.active_learning.seed_identity import write_ariadne_task_map

    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    iteration_dir = active_iteration_dir(campaign, 1)
    selection = build_seed_selection_manifest(
        campaign_uid=state.campaign_uid,
        campaign_random_seed=config.campaign.reproducibility_seed,
        iteration=1,
        models_version=0,
        model_manifest_sha256="a" * 64,
        model_set_sha256="b" * 64,
        trajectory_sha256="c" * 64,
        selection_strategy="hybrid_variance",
        seed_records=[
            {
                "seed_id": seed_id,
                "frame_id": seed_id - 1,
                "pool_row_index_zero_based": seed_id - 1,
                "selection_origin": "bulk",
                "variance_at_selection": 0.1 * seed_id,
            }
            for seed_id in range(1, 4)
        ],
    )
    selection_path = seeds_picked_path(iteration_dir)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, selection)
    write_ariadne_task_map(iteration_dir, selection)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["generation"] == 1


def test_rebind_rejects_ariadne_boundary_with_reusable_output(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    _patch_ariadne_retry_transition(
        monkeypatch,
        logical_total=3,
        n_complete=1,
        retry_task_ids=[1, 2],
    )

    with pytest.raises(
        ExecutionIdentityError,
        match="requires either every logical task to be retried",
    ):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )


def test_rebind_accepts_single_generation_ariadne_postprocess_only_boundary(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    _patch_ariadne_retry_transition(
        monkeypatch,
        logical_total=3,
        n_complete=3,
        retry_task_ids=[],
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.ariadne_publication."
        "classify_ariadne_publication",
        lambda *_args, **_kwargs: {
            "state": "stale_results_binding",
            "archive_required": True,
            "reason": "test stale publication",
            "files": [],
        },
    )
    producer = _write_full_ariadne_producer_intent(
        campaign,
        state,
        logical_total=3,
        status="SUPERSEDED",
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["transition_kind"] == "ariadne_postprocess_only"
    assert result["producer_environment_generation"] == 0
    assert result["producer_job_id"] == "12345"
    assert result["postprocess_source"]["attempt_id"] == producer["attempt_id"]
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=state.campaign_uid,
    )["generation"]
    postprocess_intent = write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.ARIADNE_ARRAY.value,
        iteration=1,
        expected_tasks=3,
        decision_contract=dict(result["postprocess_source"]["decision_contract"]),
        postprocess_source=dict(result["postprocess_source"]),
        environment_generation=int(active["generation"]),
        environment_generation_digest_sha256=str(active["digest_sha256"]),
    )
    producer_environment = ariadne_producer_environment_binding(
        campaign,
        postprocess_intent,
        expected_campaign_uid=state.campaign_uid,
        expected_iteration=1,
    )
    assert int(active["generation"]) == 1
    assert producer_environment["generation"] == 0


def test_postprocess_source_survives_repeated_local_failure_and_reconcile(
    tmp_path,
    monkeypatch,
):
    campaign, _config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    producer = _write_full_ariadne_producer_intent(campaign, state)
    task_digest = hashlib.sha256(b"0,1,2").hexdigest()
    source = resolve_ariadne_postprocess_source(
        campaign,
        campaign_uid=state.campaign_uid,
        iteration=1,
        logical_total=3,
        logical_task_set_sha256=task_digest,
    )
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=state.campaign_uid,
    )["generation"]
    write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.ARIADNE_ARRAY.value,
        iteration=1,
        expected_tasks=3,
        decision_contract=dict(source["decision_contract"]),
        postprocess_source=source,
        environment_generation=int(active["generation"]),
        environment_generation_digest_sha256=str(active["digest_sha256"]),
    )
    mark_failed(
        campaign,
        CampaignPhase.ARIADNE_ARRAY.value,
        1,
        "second local postprocess failure",
    )
    mark_superseded(
        campaign,
        CampaignPhase.ARIADNE_ARRAY.value,
        1,
        "reconcile_apply_retry",
    )

    repeated = resolve_ariadne_postprocess_source(
        campaign,
        campaign_uid=state.campaign_uid,
        iteration=1,
        logical_total=3,
        logical_task_set_sha256=task_digest,
    )

    assert repeated == source
    assert repeated["attempt_id"] == producer["attempt_id"]
    assert repeated["job_id"] == "12345"


def test_postprocess_source_rejects_arbitrary_supersession(
    tmp_path,
    monkeypatch,
):
    campaign, _config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    _write_full_ariadne_producer_intent(campaign, state)
    mark_superseded(
        campaign,
        CampaignPhase.ARIADNE_ARRAY.value,
        1,
        "unrelated_supersession",
    )

    with pytest.raises(ValueError, match="SUPERSEDED by reconcile_apply_retry"):
        resolve_ariadne_postprocess_source(
            campaign,
            campaign_uid=state.campaign_uid,
            iteration=1,
            logical_total=3,
            logical_task_set_sha256=hashlib.sha256(b"0,1,2").hexdigest(),
        )


def test_rebind_rejects_all_complete_ariadne_without_full_attempt_binding(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    _patch_ariadne_retry_transition(
        monkeypatch,
        logical_total=3,
        n_complete=3,
        retry_task_ids=[],
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.ariadne_publication."
        "classify_ariadne_publication",
        lambda *_args, **_kwargs: {
            "state": "absent",
            "archive_required": False,
            "reason": "absent",
            "files": [],
        },
    )
    _write_full_ariadne_producer_intent(campaign, state, logical_total=3)
    path = intent_path(campaign, CampaignPhase.ARIADNE_ARRAY.value, 1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expected_tasks"] = 1
    payload["retry_expected_tasks"] = 1
    payload["array_recovery"]["n_reuse"] = 2
    payload["array_recovery"]["n_retry"] = 1
    payload["submission_metadata"]["logical_task_set_sha256"] = hashlib.sha256(
        b"2"
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExecutionIdentityError,
        match="not bound to one full-array producer attempt",
    ):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )


@pytest.mark.parametrize(
    ("logical_total", "retry_task_ids", "match"),
    [
        (0, [], "non-empty logical task set"),
        (3, [0, 2], "cover every logical task exactly once"),
    ],
)
def test_rebind_rejects_incomplete_ariadne_retry_identity(
    tmp_path,
    monkeypatch,
    logical_total,
    retry_task_ids,
    match,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    _patch_ariadne_retry_transition(
        monkeypatch,
        logical_total=logical_total,
        retry_task_ids=retry_task_ids,
    )

    with pytest.raises(ExecutionIdentityError, match=match):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )


def test_rebind_rejects_ariadne_boundary_with_invalid_handoff(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    _patch_ariadne_retry_transition(
        monkeypatch,
        contract_error="seed-selection handoff is invalid",
    )

    with pytest.raises(
        ExecutionIdentityError,
        match="failed its recovery contract",
    ):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )


def test_rebind_ariadne_boundary_retains_scheduler_ownership_checks(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    state.pending_jobs[CampaignPhase.ARIADNE_ARRAY.value] = "12345"
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    _patch_ariadne_retry_transition(monkeypatch, logical_total=3)

    with pytest.raises(ExecutionIdentityError, match="pending scheduler ownership"):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )


def test_rebind_accepts_verified_unpublished_reference_commit_recovery(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.REFERENCE_COMMIT
    state.iteration = 0
    state.reference_data_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.reference_commit.classify_reference_commit",
        lambda *_args, **_kwargs: {
            "state": "prepared",
            "ledger": {
                "campaign_uid": state.campaign_uid,
                "iteration": 0,
                "reference_data_version": 0,
                "context": "bootstrap",
            },
        },
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["generation"] == 1
    recovered = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert recovered.phase is CampaignPhase.REFERENCE_COMMIT
    assert recovered.reference_data_version == -1


@pytest.mark.parametrize("transaction_state", ["absent", "invalid", "published"])
def test_rebind_rejects_unverified_or_published_reference_commit_recovery(
    tmp_path,
    monkeypatch,
    transaction_state,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    state.phase = CampaignPhase.REFERENCE_COMMIT
    state.iteration = 0
    state.reference_data_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.reference_commit.classify_reference_commit",
        lambda *_args, **_kwargs: {
            "state": transaction_state,
            "reason": "test transaction is not safely rebindable",
        },
    )

    with pytest.raises(
        ExecutionIdentityError,
        match="valid unpublished recovery transaction",
    ):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )


def test_rebind_rejects_reference_commit_transaction_identity_mismatch(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    state.phase = CampaignPhase.REFERENCE_COMMIT
    state.iteration = 0
    state.reference_data_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.reference_commit.classify_reference_commit",
        lambda *_args, **_kwargs: {
            "state": "prepared",
            "ledger": {
                "campaign_uid": "different-campaign",
                "iteration": 0,
                "reference_data_version": 0,
                "context": "bootstrap",
            },
        },
    )

    with pytest.raises(ExecutionIdentityError, match="inconsistent campaign"):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )


@pytest.mark.parametrize(
    ("phase", "iteration", "reference_version", "model_version"),
    [
        (CampaignPhase.INITIAL_FEREBUS, 0, 0, -1),
        (CampaignPhase.FEREBUS, 2, 2, 1),
    ],
)
def test_rebind_accepts_clean_pre_submission_ferebus_boundary(
    tmp_path,
    monkeypatch,
    phase,
    iteration,
    reference_version,
    model_version,
):
    from ichor.hpc.active_learning.daemon import ferebus_row_cache
    from ichor.hpc.active_learning.versioning import reference_data, trained_models

    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = phase
    state.iteration = iteration
    state.reference_data_version = reference_version
    state.models_version = model_version
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.recovery_contracts.phase_recovery_contract_error",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        reference_data.ReferenceDataVersioning,
        "current_version",
        lambda _self: reference_version,
    )
    monkeypatch.setattr(
        trained_models.TrainedModelVersioning,
        "current_version",
        lambda _self: None if model_version < 0 else model_version,
    )
    monkeypatch.setattr(
        reference_data.ReferenceDataVersioning,
        "resolve",
        lambda *_args, **_kwargs: SimpleNamespace(version=reference_version),
    )
    monkeypatch.setattr(ferebus_row_cache, "clear_row_caches", lambda *_args: False)
    monkeypatch.setattr(
        ferebus_row_cache,
        "ensure_cumulative_row_caches",
        lambda *_args, **_kwargs: None,
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    rebound = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert rebound.phase is phase
    assert rebound.reference_data_version == reference_version
    assert rebound.models_version == model_version


def test_rebind_requires_reconcile_to_archive_failed_ferebus_staging(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    state.phase = CampaignPhase.INITIAL_FEREBUS
    state.iteration = 0
    state.reference_data_version = 0
    state.models_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    (campaign / "TRAINED_MODELS" / "iteration-staging").mkdir(parents=True)

    with pytest.raises(ExecutionIdentityError, match="reconcile to archive"):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )


def test_rebind_rejects_ferebus_current_pointer_mismatch(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.versioning import reference_data, trained_models

    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    state.phase = CampaignPhase.INITIAL_FEREBUS
    state.iteration = 0
    state.reference_data_version = 0
    state.models_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        reference_data.ReferenceDataVersioning,
        "current_version",
        lambda _self: None,
    )
    monkeypatch.setattr(
        trained_models.TrainedModelVersioning,
        "current_version",
        lambda _self: None,
    )

    with pytest.raises(ExecutionIdentityError, match="current pointers"):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )


def test_rebind_requires_conclusive_scheduler_clearance(tmp_path, monkeypatch):
    campaign, config, _state = _rebind_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )

    with pytest.raises(ExecutionIdentityError, match="conclusive scheduler"):
        rebind_environment(campaign, config=config)


def test_rebind_advances_generation_clears_scales_and_is_idempotent(
    tmp_path,
    monkeypatch,
):
    campaign, config, _state = _rebind_campaign(tmp_path, monkeypatch)
    version = {"value": "3.11.changed"}

    def changed_generation(*args, campaign_uid, config, generation=0, **kwargs):
        payload = _fake_generation(
            campaign_uid=campaign_uid,
            config=config,
            generation=generation,
        )
        payload["python_version"] = version["value"]
        payload["environment_fingerprint_sha256"] = _environment_fingerprint(payload)
        payload["digest_sha256"] = _canonical_digest(payload)
        return payload

    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        changed_generation,
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )
    repeated = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["generation"] == 1
    assert repeated["changed"] is False
    assert repeated["generation"] == 1
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid="uid-rebind",
    )
    assert active["generation"]["python_version"] == "3.11.changed"
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.reference_scales is None
    assert state.reference_scales_iteration == -1
    assert state.reference_scales_models_version == -1
    assert state.reference_scales_model_manifest_sha256 is None


def test_config_only_change_advances_generation_but_lock_refresh_does_not(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    original = read_active_environment_generation(
        campaign,
        expected_campaign_uid=state.campaign_uid,
    )["generation"]
    changed = CampaignConfig.from_dict(config.to_dict())
    changed.campaign.sampling_aggressiveness += 1
    write_config_lock(
        campaign,
        changed,
        campaign_uid=state.campaign_uid,
        reason="test_config_only_change",
    )

    result = rebind_environment(
        campaign,
        config=changed,
        scheduler_ownership_clear=True,
    )
    write_config_lock(
        campaign,
        changed,
        campaign_uid=state.campaign_uid,
        reason="timestamp_only_refresh",
    )
    repeated = rebind_environment(
        campaign,
        config=changed,
        scheduler_ownership_clear=True,
    )
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=state.campaign_uid,
    )["generation"]

    assert result["changed"] is True
    assert result["generation"] == 1
    assert repeated["changed"] is False
    assert active["campaign_config_sha256"] == config_fingerprint(
        changed.to_dict()
    )
    assert (
        active["environment_fingerprint_sha256"]
        == original["environment_fingerprint_sha256"]
    )


def test_config_only_generation_collision_allocates_next_free_number(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    collision = _fake_generation(
        campaign,
        campaign_uid=state.campaign_uid,
        config=config,
        generation=1,
    )
    collision_path = environment_generations_dir(campaign) / (
        "generation-000001.json"
    )
    collision_path.write_text(
        json.dumps(collision, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    changed = CampaignConfig.from_dict(config.to_dict())
    changed.campaign.sampling_aggressiveness += 1
    write_config_lock(
        campaign,
        changed,
        campaign_uid=state.campaign_uid,
        reason="test_config_collision",
    )

    result = rebind_environment(
        campaign,
        config=changed,
        scheduler_ownership_clear=True,
    )

    assert result["changed"] is True
    assert result["generation"] == 2


def test_legacy_config_generation_split_requires_authoritative_chronology(
    tmp_path,
    monkeypatch,
):
    clock = {"value": "2026-07-29T10:00:00+00:00"}
    monkeypatch.setattr(
        config_lock_module,
        "_now_iso",
        lambda: clock["value"],
    )
    old_config = CampaignConfig()
    write_config_lock(
        tmp_path,
        old_config,
        campaign_uid="uid-legacy-split",
        reason="initial",
    )
    old_sha256 = config_fingerprint(old_config.to_dict())
    environment = {
        "created_at_iso": "2026-07-29T10:05:00+00:00",
        "campaign_config_sha256": old_sha256,
    }
    new_config = CampaignConfig.from_dict(old_config.to_dict())
    new_config.campaign.sampling_aggressiveness += 1
    new_sha256 = config_fingerprint(new_config.to_dict())
    clock["value"] = "2026-07-29T10:10:00+00:00"
    write_config_lock(
        tmp_path,
        new_config,
        campaign_uid="uid-legacy-split",
        reason="approved_config_change",
    )
    intent = {
        "status": "COMPLETED",
        "job_id": "18116090",
        "created_iso": "2026-07-29T10:15:00+00:00",
    }

    assert legacy_intent_config_generation_split_is_proven(
        tmp_path,
        campaign_uid="uid-legacy-split",
        intent=intent,
        environment=environment,
        decision_config_sha256=new_sha256,
    )
    intent_at_generation_creation = {
        **intent,
        "created_iso": environment["created_at_iso"],
    }
    assert not legacy_intent_config_generation_split_is_proven(
        tmp_path,
        campaign_uid="uid-legacy-split",
        intent=intent_at_generation_creation,
        environment=environment,
        decision_config_sha256=new_sha256,
    )
    intent_before_config_change = {
        **intent,
        "created_iso": "2026-07-29T10:07:00+00:00",
    }
    assert not legacy_intent_config_generation_split_is_proven(
        tmp_path,
        campaign_uid="uid-legacy-split",
        intent=intent_before_config_change,
        environment=environment,
        decision_config_sha256=new_sha256,
    )


def _changed_generation(*args, campaign_uid, config, generation=0, **kwargs):
    payload = _fake_generation(
        campaign_uid=campaign_uid,
        config=config,
        generation=generation,
    )
    payload["python_version"] = "3.11.rebound"
    payload["environment_fingerprint_sha256"] = _environment_fingerprint(payload)
    payload["digest_sha256"] = _canonical_digest(payload)
    return payload


def test_environment_transition_preserves_row_caches_for_lazy_validation(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import ferebus_row_cache

    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.reference_data_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    calls = []
    monkeypatch.setattr(
        ferebus_row_cache,
        "clear_row_caches",
        lambda _campaign: calls.append("clear"),
    )
    monkeypatch.setattr(
        ferebus_row_cache,
        "ensure_cumulative_row_caches",
        lambda _campaign, view: calls.append(("rebuild", view)),
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["generation"] == 1
    assert calls == []


def test_daemon_start_automatically_advances_safe_environment_drift(
    tmp_path,
    monkeypatch,
):
    campaign, config, _state = _rebind_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    daemon = Daemon(
        campaign_dir=campaign,
        config=config,
        executor=_SubmittedExecutor(),
        environment_preflight_ok=True,
    )

    rc = daemon.run(max_ticks=0)

    assert rc == 0
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid="uid-rebind",
    )["generation"]
    assert active["generation"] == 1
    assert active["python_version"] == "3.11.rebound"


def test_daemon_start_advances_fully_retryable_ariadne_drift(
    tmp_path,
    monkeypatch,
):
    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    _patch_ariadne_retry_transition(monkeypatch, logical_total=150)
    daemon = Daemon(
        campaign_dir=campaign,
        config=config,
        executor=_SubmittedExecutor(),
        environment_preflight_ok=True,
    )

    rc = daemon.run(max_ticks=0)

    assert rc == 0
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid="uid-rebind",
    )["generation"]
    assert active["generation"] == 1
    assert active["python_version"] == "3.11.rebound"


def test_rebind_replays_generation_after_state_write_failure(tmp_path, monkeypatch):
    campaign, config, _state = _rebind_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    original_write_state = execution_identity_module.write_state
    calls = {"count": 0}

    def fail_first_state_write(path, state):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("injected state persistence failure")
        return original_write_state(path, state)

    monkeypatch.setattr(
        execution_identity_module,
        "write_state",
        fail_first_state_write,
    )
    with pytest.raises(OSError, match="injected state persistence failure"):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )

    assert json.loads(
        environment_current_path(campaign).read_text(encoding="utf-8")
    )["generation"] == 0
    assert (
        environment_generations_dir(campaign) / "generation-000001.json"
    ).is_file()
    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )
    assert result["generation"] == 1


def test_rebind_replays_generation_after_pointer_write_failure(tmp_path, monkeypatch):
    campaign, config, _state = _rebind_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    original_atomic_write = execution_identity_module.atomic_write_json
    fail_pointer = {"enabled": True}

    def injected_atomic_write(path, payload):
        if (
            fail_pointer["enabled"]
            and Path(path) == environment_current_path(campaign)
            and payload.get("generation") == 1
        ):
            raise OSError("injected pointer persistence failure")
        return original_atomic_write(path, payload)

    monkeypatch.setattr(
        execution_identity_module,
        "atomic_write_json",
        injected_atomic_write,
    )
    with pytest.raises(OSError, match="injected pointer persistence failure"):
        rebind_environment(
            campaign,
            config=config,
            scheduler_ownership_clear=True,
        )

    assert json.loads(
        environment_current_path(campaign).read_text(encoding="utf-8")
    )["generation"] == 0
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.reference_scales is None
    fail_pointer["enabled"] = False
    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )
    assert result["generation"] == 1


def test_rebind_preserves_conflicting_orphan_and_uses_next_generation(
    tmp_path,
    monkeypatch,
):
    campaign, config, _state = _rebind_campaign(tmp_path, monkeypatch)
    orphan = _changed_generation(
        campaign,
        campaign_uid="uid-rebind",
        config=config,
        generation=1,
    )
    orphan_path = environment_generations_dir(campaign) / "generation-000001.json"
    execution_identity_module.atomic_write_json(orphan_path, orphan)

    def later_generation(*args, campaign_uid, config, generation=0, **kwargs):
        payload = _fake_generation(
            campaign_uid=campaign_uid,
            config=config,
            generation=generation,
        )
        payload["python_version"] = "3.11.later"
        payload["environment_fingerprint_sha256"] = _environment_fingerprint(
            payload
        )
        payload["digest_sha256"] = _canonical_digest(payload)
        return payload

    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        later_generation,
    )

    result = rebind_environment(
        campaign,
        config=config,
        scheduler_ownership_clear=True,
    )

    assert result["generation"] == 2
    assert orphan_path.is_file()
    assert (
        environment_generations_dir(campaign) / "generation-000002.json"
    ).is_file()


class _SubmittedExecutor:
    def __init__(self):
        self.submissions = 0
        self.postprocess_calls = 0

    def submit_or_run(self, _state, _phase):
        self.submissions += 1
        return PhaseResult(
            is_complete=False,
            submitted_job_id="dry-environment-1",
            expected_tasks=1,
        )

    def postprocess(self, *_args, **_kwargs):
        self.postprocess_calls += 1
        raise AssertionError("postprocess must not cross environment drift")

    def handle_failure(self, *_args, **_kwargs):
        raise AssertionError


def _daemon_environment_campaign(tmp_path, monkeypatch):
    campaign = tmp_path / "daemon-campaign"
    campaign.mkdir()
    config = CampaignConfig()
    current_version = {"value": "3.11.initial"}

    def generation(*args, campaign_uid, config, generation=0, **kwargs):
        payload = _fake_generation(
            campaign_uid=campaign_uid,
            config=config,
            generation=generation,
        )
        payload["python_version"] = current_version["value"]
        payload["environment_fingerprint_sha256"] = _environment_fingerprint(payload)
        payload["digest_sha256"] = _canonical_digest(payload)
        return payload

    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity.capture_environment_generation",
        generation,
    )
    state = fresh_campaign_state(campaign_uid="uid-daemon-environment")
    state.phase = CampaignPhase.FEREBUS
    state.iteration = 1
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    ensure_execution_identity(
        campaign,
        campaign_uid=state.campaign_uid,
        config=config,
        requested_mode="dry_run",
    )
    ensure_config_lock(
        campaign,
        config,
        campaign_uid=state.campaign_uid,
    )
    return campaign, config, state, current_version


def test_submission_intent_snapshots_active_environment(tmp_path, monkeypatch):
    campaign, config, state, _version = _daemon_environment_campaign(
        tmp_path, monkeypatch
    )
    executor = _SubmittedExecutor()
    daemon = Daemon(campaign_dir=campaign, config=config, executor=executor)

    assert daemon._on_phase_entry(state, state.phase) == TickStatus.SUBMITTED

    intent = load_intent(campaign, CampaignPhase.FEREBUS.value, 1)
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=state.campaign_uid,
    )["generation"]
    assert intent["environment_generation"] == 0
    assert (
        intent["environment_generation_digest_sha256"]
        == active["digest_sha256"]
    )


def test_initial_submission_refuses_stale_environment_config_binding(
    tmp_path,
    monkeypatch,
):
    campaign, config, state, _version = _daemon_environment_campaign(
        tmp_path,
        monkeypatch,
    )
    changed = CampaignConfig.from_dict(config.to_dict())
    changed.campaign.sampling_aggressiveness += 1
    write_config_lock(
        campaign,
        changed,
        campaign_uid=state.campaign_uid,
        reason="approved_config_change",
    )
    executor = _SubmittedExecutor()
    daemon = Daemon(campaign_dir=campaign, config=changed, executor=executor)

    result = daemon._on_phase_entry(state, state.phase)

    assert result == TickStatus.HALTED
    assert executor.submissions == 0
    assert not intent_path(campaign, CampaignPhase.FEREBUS.value, 1).exists()
    halted = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert "stale campaign configuration" in str(
        halted.lifecycle_context.get("message")
    )


def test_submission_refuses_decision_contract_change_during_staging(
    tmp_path,
    monkeypatch,
):
    campaign, config, state, _version = _daemon_environment_campaign(
        tmp_path,
        monkeypatch,
    )

    class _MutatingExecutor(_SubmittedExecutor):
        def submit_or_run(self, bound_state, bound_phase):
            intent = load_intent(
                campaign,
                bound_phase.value,
                bound_state.iteration,
            )
            altered = dict(intent)
            altered_contract = dict(altered["decision_contract"])
            altered_contract["config_sha256"] = "0" * 64
            altered["decision_contract"] = altered_contract
            self._submission_environment_guard(altered)
            self.submissions += 1
            raise AssertionError("scheduler acceptance must not be reached")

    executor = _MutatingExecutor()
    daemon = Daemon(campaign_dir=campaign, config=config, executor=executor)

    result = daemon._on_phase_entry(state, state.phase)

    assert result == TickStatus.HALTED
    assert executor.submissions == 0
    halted = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert "decision contract changed during pre-submit staging" in str(
        halted.lifecycle_context.get("message")
    )


def test_submission_refuses_environment_binding_change_during_staging(
    tmp_path,
    monkeypatch,
):
    campaign, config, state, _version = _daemon_environment_campaign(
        tmp_path,
        monkeypatch,
    )

    class _MutatingExecutor(_SubmittedExecutor):
        def submit_or_run(self, bound_state, bound_phase):
            intent = load_intent(
                campaign,
                bound_phase.value,
                bound_state.iteration,
            )
            altered = dict(intent)
            altered["environment_generation_digest_sha256"] = "0" * 64
            self._submission_environment_guard(altered)
            self.submissions += 1
            raise AssertionError("scheduler acceptance must not be reached")

    executor = _MutatingExecutor()
    daemon = Daemon(campaign_dir=campaign, config=config, executor=executor)

    result = daemon._on_phase_entry(state, state.phase)

    assert result == TickStatus.HALTED
    assert executor.submissions == 0
    halted = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert "environment binding changed during pre-submit staging" in str(
        halted.lifecycle_context.get("message")
    )


def test_daemon_does_not_recapture_environment_on_every_phase_entry(
    tmp_path,
    monkeypatch,
):
    campaign, config, state, version = _daemon_environment_campaign(
        tmp_path, monkeypatch
    )
    executor = _SubmittedExecutor()
    daemon = Daemon(campaign_dir=campaign, config=config, executor=executor)
    version["value"] = "3.11.drifted"

    result = daemon._on_phase_entry(state, state.phase)

    assert result == TickStatus.SUBMITTED
    assert executor.submissions == 1


def test_daemon_does_not_recapture_environment_on_every_postprocess_check(
    tmp_path,
    monkeypatch,
):
    campaign, config, state, version = _daemon_environment_campaign(
        tmp_path, monkeypatch
    )
    executor = _SubmittedExecutor()
    daemon = Daemon(campaign_dir=campaign, config=config, executor=executor)
    version["value"] = "3.11.drifted"

    result = daemon._verify_environment_boundary(
        state,
        state.phase,
        boundary="postprocess",
    )

    assert result is None
    assert executor.postprocess_calls == 0


def test_postprocess_rejects_intent_bound_to_previous_environment_generation(
    tmp_path,
    monkeypatch,
):
    campaign, config, state, version = _daemon_environment_campaign(
        tmp_path,
        monkeypatch,
    )
    executor = _SubmittedExecutor()
    daemon = Daemon(campaign_dir=campaign, config=config, executor=executor)
    assert daemon._on_phase_entry(state, state.phase) == TickStatus.SUBMITTED

    version["value"] = "3.11.rebound-with-active-job"
    generation = execution_identity_module.capture_environment_generation(
        campaign,
        campaign_uid=state.campaign_uid,
        config=config,
        generation=1,
    )
    generation_path = environment_generations_dir(campaign) / "generation-000001.json"
    execution_identity_module.atomic_write_json(generation_path, generation)
    execution_identity_module.atomic_write_json(
        environment_current_path(campaign),
        {
            "schema_version": 1,
            "generation": 1,
            "generation_path": str(generation_path.relative_to(campaign)),
            "generation_digest_sha256": generation["digest_sha256"],
        },
    )

    result = daemon._postprocess(
        state,
        state.phase,
        [],
        SimpleNamespace(parent_job_id="dry-environment-1"),
    )

    assert result == TickStatus.HALTED
    assert executor.postprocess_calls == 0
    halted = read_state(daemon.state_path())
    assert halted.pending_jobs[CampaignPhase.FEREBUS.value] == "dry-environment-1"
    assert "active environment generation changed" in halted.lifecycle_context[
        "message"
    ]


@pytest.mark.parametrize("command", ["environment-status", "rebind-environment"])
def test_environment_commands_are_not_exposed_by_parser(command):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([command, "--campaign-dir", "campaign"])
