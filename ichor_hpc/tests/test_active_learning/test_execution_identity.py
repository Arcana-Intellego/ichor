from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import argparse

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
from ichor.hpc.active_learning.cli import (
    build_parser,
    cmd_environment_status,
    cmd_rebind_environment,
)


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
    state.phase = CampaignPhase.ARIADNE_ARRAY
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    with pytest.raises(ExecutionIdentityError, match="idle SEED_SELECT or DONE"):
        rebind_environment(campaign, config=config)

    state.phase = CampaignPhase.SEED_SELECT
    state.pending_jobs[CampaignPhase.SEED_SELECT.value] = "dry-1"
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    with pytest.raises(ExecutionIdentityError, match="scheduler ownership"):
        rebind_environment(campaign, config=config)


def test_rebind_requires_conclusive_scheduler_clearance(tmp_path, monkeypatch):
    campaign, config, _state = _rebind_campaign(tmp_path, monkeypatch)

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


def test_rebind_rebuilds_row_caches_before_publishing_generation(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import ferebus_row_cache
    from ichor.hpc.active_learning.versioning import reference_data

    campaign, config, state = _rebind_campaign(tmp_path, monkeypatch)
    state.reference_data_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    monkeypatch.setattr(
        execution_identity_module,
        "capture_environment_generation",
        _changed_generation,
    )
    reference_view = SimpleNamespace(version=0)
    monkeypatch.setattr(
        reference_data.ReferenceDataVersioning,
        "resolve",
        lambda *_args, **_kwargs: reference_view,
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
    assert calls == ["clear", ("rebuild", reference_view)]


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


def test_daemon_halts_before_submission_when_environment_drifted(
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

    assert result == TickStatus.HALTED
    assert executor.submissions == 0
    halted = read_state(daemon.state_path())
    assert halted.phase is CampaignPhase.HALTED
    assert halted.lifecycle_context["reason_code"] == "environment_drift"


def test_daemon_halts_before_postprocess_and_preserves_job_ownership(
    tmp_path,
    monkeypatch,
):
    campaign, config, state, version = _daemon_environment_campaign(
        tmp_path, monkeypatch
    )
    executor = _SubmittedExecutor()
    daemon = Daemon(campaign_dir=campaign, config=config, executor=executor)
    state.pending_jobs[CampaignPhase.FEREBUS.value] = "dry-environment-1"
    write_state(daemon.state_path(), state)
    version["value"] = "3.11.drifted"

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
    assert "submission_intent_environment_binding_invalid" in halted.lifecycle_context[
        "message"
    ]


def test_environment_commands_are_exposed_by_parser():
    parser = build_parser()

    status = parser.parse_args(["environment-status", "--campaign-dir", "campaign"])
    rebind = parser.parse_args(
        ["rebind-environment", "--campaign-dir", "campaign", "--apply"]
    )

    assert status.func is cmd_environment_status
    assert rebind.func is cmd_rebind_environment
    assert rebind.apply is True


def test_environment_status_and_rebind_cli_round_trip(
    tmp_path,
    monkeypatch,
    capsys,
):
    campaign, config, _state = _rebind_campaign(tmp_path, monkeypatch)
    assert cmd_environment_status(
        argparse.Namespace(campaign_dir=str(campaign), json=True)
    ) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["matches"] is True

    version = {"value": "3.11.cli-rebind"}

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
    assert cmd_rebind_environment(
        argparse.Namespace(campaign_dir=str(campaign), apply=True, json=True)
    ) == 0
    rebound = json.loads(capsys.readouterr().out)
    assert rebound["changed"] is True
    assert rebound["generation"] == 1
