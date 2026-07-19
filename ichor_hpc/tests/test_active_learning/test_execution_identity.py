from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import ichor.hpc.active_learning.execution_identity as execution_identity_module
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
    read_active_environment_generation,
    rebind_environment,
)
from ichor.hpc.active_learning.daemon.config_lock import ensure_config_lock
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.daemon.daemon import Daemon, TickStatus
from ichor.hpc.active_learning.daemon.phase_executor import PhaseResult
from ichor.hpc.active_learning.daemon.submission_intent import load_intent
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
        "campaign_config_sha256": "1" * 64,
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
        match="requires every logical task to be retried",
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
