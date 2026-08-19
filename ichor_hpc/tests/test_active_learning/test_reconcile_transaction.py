"""Durable reconciliation transaction and pointer rollback contracts."""

from __future__ import annotations

import json
import hashlib
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning import cli
from ichor.hpc.active_learning.daemon.reconcile_transaction import (
    RECONCILE_TRANSACTION_SCHEMA_VERSION,
    apply_reconcile_transaction_recovery,
    begin_reconcile_transaction,
    build_reconcile_commit_plan,
    inspect_reconcile_transaction_recovery,
    inventory_reconcile_transactions,
    publish_reconcile_config_target,
    read_reconcile_transaction,
    restore_version_pointer,
    snapshot_version_pointer,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    CampaignState,
    atomic_write_json,
    make_lifecycle_context,
    write_state,
)
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.config_lock import (
    read_config_lock,
    write_config_lock,
)
from ichor.hpc.active_learning.daemon.reconcile import stateful_campaign_artifacts
from ichor.hpc.active_learning.daemon.submission_intent import (
    ARIADNE_TERMINAL_POSTPROCESS_REASON,
    load_intent,
    mark_failed,
    mark_submitted,
    write_pre_submit_intent,
)
from ichor.hpc.active_learning.daemon.stop_control import (
    build_stop_request,
    complete_stop_request,
    install_stop_request,
    stop_request_path,
)
from ichor.hpc.active_learning.submit import sacct_poll
from ichor.hpc.active_learning.versioning.versioned_directory import VersionedDirectory


def _committed_pointer_fixture(
    campaign: Path,
    *,
    phase: CampaignPhase = CampaignPhase.SEED_SELECT,
    iteration: int = 11,
    version: int = 10,
    symlinks: bool = False,
) -> CampaignState:
    state = CampaignState()
    state.phase = phase
    state.iteration = int(iteration)
    state.max_iterations = 40
    state.reference_data_version = int(version)
    state.validation_set_version = int(version)
    state.models_version = int(version)
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(state_path, state)

    for name in ("QM_REFERENCE_DATA", "TRAINED_MODELS"):
        versions = VersionedDirectory(campaign / name)
        target = versions.iteration_path(version)
        target.mkdir(parents=True)
        if symlinks:
            os.symlink(target.name, versions.current_link_path())
        else:
            versions._pointer_path().write_text(
                target.name + "\n",
                encoding="utf-8",
            )
    return state


def _pointer_commit_plan(campaign: Path, state: CampaignState, *, config=None):
    pointer_anchors = tuple(
        (name, 1, "0" * 64)
        for name in (
            "QM_REFERENCE_DATA/current",
            "QM_REFERENCE_DATA/.current.pointer",
            "TRAINED_MODELS/current",
            "TRAINED_MODELS/.current.pointer",
        )
    )
    return build_reconcile_commit_plan(
        campaign,
        transaction_id=uuid.uuid4().hex,
        proposed_state=state,
        config=config,
        intent_transitions=[],
        artifact_snapshot=SimpleNamespace(anchor_records=pointer_anchors),
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink contract")
def test_commit_plan_accepts_managed_posix_current_symlinks(tmp_path):
    campaign = tmp_path / "campaign"
    state = _committed_pointer_fixture(campaign, symlinks=True)

    plan = _pointer_commit_plan(campaign, state)

    assert plan["stable_authority_records"] == []
    assert [record["before_version"] for record in plan["pointers"]] == [10, 10]
    assert [record["after_version"] for record in plan["pointers"]] == [10, 10]


def test_commit_plan_accepts_managed_pointer_fallback_files(tmp_path):
    campaign = tmp_path / "campaign"
    state = _committed_pointer_fixture(campaign)

    plan = _pointer_commit_plan(campaign, state)

    assert plan["stable_authority_records"] == []
    assert [record["before_version"] for record in plan["pointers"]] == [10, 10]


def test_commit_plan_still_rejects_regular_current_file(tmp_path):
    campaign = tmp_path / "campaign"
    state = _committed_pointer_fixture(campaign)
    pointer = campaign / "QM_REFERENCE_DATA" / "current"
    pointer.write_text("iteration-000010\n", encoding="utf-8")

    with pytest.raises(ValueError, match="current path is not a symlink"):
        _pointer_commit_plan(campaign, state)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink contract")
def test_commit_plan_rejects_noncanonical_current_symlink_target(tmp_path):
    campaign = tmp_path / "campaign"
    state = _committed_pointer_fixture(campaign, symlinks=True)
    pointer = campaign / "QM_REFERENCE_DATA" / "current"
    pointer.unlink()
    os.symlink("../iteration-000010", pointer)

    with pytest.raises(ValueError, match="version basename"):
        _pointer_commit_plan(campaign, state)


def test_pointer_planning_interruption_is_retired_before_retry(tmp_path):
    campaign = tmp_path / "campaign"
    state = _committed_pointer_fixture(campaign)
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase=state.phase.value,
        proposed_iteration=state.iteration,
        planned_operations=["repair_current_pointers", "update_config_lock"],
        intent_transitions=[],
        campaign_uid=state.campaign_uid,
    )
    transaction.set_status("MUTATING")
    transaction.record_interruption(
        "current pointer repair failed: ValueError: path contains a symlink: "
        + str(campaign / "QM_REFERENCE_DATA" / "current")
    )

    inspection = inspect_reconcile_transaction_recovery(campaign)

    assert inspection["action"] == "abandon"
    assert inspection["disposition"] == "completed_safe_cleanup"
    apply_reconcile_transaction_recovery(campaign, inspection)
    assert read_reconcile_transaction(transaction.path)["status"] == "FAILED"
    assert _pointer_commit_plan(campaign, state)["pointers"]


@pytest.mark.parametrize(
    (
        "mode",
        "source_phase",
        "source_iteration",
        "phase_started",
        "current_phase",
        "current_iteration",
    ),
    [
        pytest.param(
            "immediate",
            CampaignPhase.SEED_SELECT,
            11,
            False,
            CampaignPhase.SEED_SELECT,
            11,
            id="immediate",
        ),
        pytest.param(
            "after_phase",
            CampaignPhase.SEED_SELECT,
            11,
            False,
            CampaignPhase.SEED_SELECT,
            11,
            id="after-unstarted-phase",
        ),
        pytest.param(
            "after_phase",
            CampaignPhase.FEREBUS,
            10,
            True,
            CampaignPhase.STOP_CHECK,
            10,
            id="after-completed-phase",
        ),
        pytest.param(
            "after_iteration",
            CampaignPhase.STOP_CHECK,
            10,
            True,
            CampaignPhase.SEED_SELECT,
            11,
            id="after-completed-iteration",
        ),
    ],
)
def test_commit_plan_preserves_completed_stop_while_updating_config(
    tmp_path,
    mode,
    source_phase,
    source_iteration,
    phase_started,
    current_phase,
    current_iteration,
):
    campaign = tmp_path / "campaign"
    state = _committed_pointer_fixture(
        campaign,
        phase=current_phase,
        iteration=current_iteration,
        symlinks=(os.name != "nt"),
    )
    source = CampaignState.from_dict(state.to_dict())
    source.phase = source_phase
    source.iteration = source_iteration
    request, _ = install_stop_request(
        campaign,
        build_stop_request(
            source,
            mode=mode,
            phase_started=phase_started,
        ),
    )
    receipt = (
        {"path": "receipt.json"}
        if mode != "immediate" and phase_started
        else None
    )
    complete_stop_request(
        campaign,
        request["request_id"],
        reason=(
            "immediate"
            if mode == "immediate"
            else (
                ("phase_completed" if phase_started else "phase_not_started")
                if mode == "after_phase"
                else "active_iteration_completed"
            )
        ),
        completion_receipt=receipt,
    )
    state.shutdown_requested = True
    state.lifecycle_context = make_lifecycle_context(
        disposition="stopped",
        reason_code="user_stop_boundary_reached",
        message="test stop boundary reached",
        from_phase=source_phase,
        iteration=current_iteration,
        source="daemon_stop_control",
        recovery_action="use resume to continue",
        details={"mode": mode},
    )
    write_state(
        campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json",
        state,
    )
    original = CampaignConfig()
    write_config_lock(campaign, original, campaign_uid=state.campaign_uid)
    changed = CampaignConfig()
    changed.campaign.sampling_aggressiveness = 7
    stop_before = stop_request_path(campaign).read_bytes()

    plan = _pointer_commit_plan(campaign, state, config=changed)
    publish_reconcile_config_target(campaign, plan["config_lock"])

    assert stop_request_path(campaign).read_bytes() == stop_before
    persisted = CampaignState.from_dict(
        json.loads(
            (
                campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
            ).read_text(encoding="utf-8")
        )
    )
    assert persisted.shutdown_requested is True
    assert persisted.phase is current_phase
    assert persisted.iteration == current_iteration
    lock = read_config_lock(campaign, expected_campaign_uid=state.campaign_uid)
    assert lock["canonical_config"]["campaign"]["sampling_aggressiveness"] == 7
    assert not (
        campaign / ".DATA" / "ACTIVE_LEARNING" / "submission_intents"
    ).exists()


def test_reconcile_transaction_records_lossless_moves_and_status(tmp_path):
    campaign = tmp_path / "campaign"
    evidence = campaign / ".DATA" / "STAGING.before-reconcile" / "failure.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("retained\n", encoding="utf-8")

    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase="FEREBUS",
        proposed_iteration=1,
        planned_operations=["archive_reconcile_evidence"],
        intent_transitions=[],
    )
    transaction.set_status("MUTATING")
    transaction.record_paths("archive_staging", [str(evidence.parent)])
    transaction.set_status("COMMITTED")

    payload = json.loads(transaction.path.read_text(encoding="utf-8"))
    assert payload["status"] == "COMMITTED"
    assert payload["completed_operations"][0]["paths"] == [
        ".DATA/STAGING.before-reconcile"
    ]
    assert read_reconcile_transaction(transaction.path)["status"] == "COMMITTED"


def test_incomplete_reconcile_transaction_blocks_a_second_transaction(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    first = begin_reconcile_transaction(
        campaign,
        proposed_phase="INIT",
        proposed_iteration=0,
        planned_operations=["write state"],
        intent_transitions=[],
    )

    with pytest.raises(RuntimeError, match="prior transaction evidence"):
        begin_reconcile_transaction(
            campaign,
            proposed_phase="INIT",
            proposed_iteration=0,
            planned_operations=["write state"],
            intent_transitions=[],
        )

    assert inventory_reconcile_transactions(campaign)[0]["status"] == "PREPARED"
    assert any(
        "reconcile_transactions" in path
        for path in stateful_campaign_artifacts(campaign)
    )
    first.set_status("FAILED", reason="test repair abandoned")
    second = begin_reconcile_transaction(
        campaign,
        proposed_phase="INIT",
        proposed_iteration=0,
        planned_operations=["write state"],
        intent_transitions=[],
    )
    assert second.path != first.path


def test_reconcile_pointer_snapshot_restores_previous_binding(tmp_path):
    campaign = tmp_path / "campaign"
    versions = VersionedDirectory(campaign / "QM_REFERENCE_DATA")
    versions.iteration_path(0).mkdir(parents=True)
    versions.iteration_path(1).mkdir()
    versions.update_current(0)
    snapshot = snapshot_version_pointer(
        campaign,
        versions,
        label="reference_data",
        requested_version=1,
    )

    versions.update_current(1)
    restore_version_pointer(campaign, snapshot)

    assert versions.current_version() == 0


def _write_v1_transaction(
    campaign,
    *,
    status,
    planned_operations=None,
    proposed_phase="INIT",
    proposed_iteration=0,
    extra=None,
):
    root = campaign / ".DATA" / "ACTIVE_LEARNING" / "reconcile_transactions"
    root.mkdir(parents=True, exist_ok=True)
    transaction_id = uuid.uuid4().hex
    payload = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "status": status,
        "created_at_iso": "2026-01-01T00:00:00+00:00",
        "updated_at_iso": "2026-01-01T00:00:00+00:00",
        "proposed_phase": proposed_phase,
        "proposed_iteration": proposed_iteration,
        "planned_operations": list(planned_operations or []),
        "intent_transitions": [],
        "completed_operations": [],
        "warnings": [],
    }
    payload.update(dict(extra or {}))
    path = root / (transaction_id + ".json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_schema_v1_terminal_transaction_remains_readable_and_unchanged(tmp_path):
    campaign = tmp_path / "campaign"
    path = _write_v1_transaction(campaign, status="COMMITTED")
    before = path.read_bytes()

    payload = read_reconcile_transaction(path)
    inventory = inventory_reconcile_transactions(campaign)

    assert RECONCILE_TRANSACTION_SCHEMA_VERSION == 2
    assert payload["schema_version"] == 1
    assert inventory[0]["status"] == "COMMITTED"
    assert path.read_bytes() == before


def test_schema_v1_prepared_transaction_is_safely_abandoned(tmp_path):
    campaign = tmp_path / "campaign"
    path = _write_v1_transaction(campaign, status="PREPARED")

    inspection = inspect_reconcile_transaction_recovery(campaign)
    assert inspection["recoverable"] is True
    assert inspection["disposition"] == "abandoned_before_mutation"

    apply_reconcile_transaction_recovery(campaign, inspection)

    payload = read_reconcile_transaction(path)
    assert payload["status"] == "FAILED"
    assert payload["resolution"]["disposition"] == "abandoned_before_mutation"


def test_schema_v1_known_mutating_cleanup_is_retired_with_state_unchanged(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    write_state(state_path, CampaignState())
    path = _write_v1_transaction(
        campaign,
        status="MUTATING",
        planned_operations=[
            "archive_reconcile_evidence",
            "repair_current_pointers",
            "write_recovered_state",
            "update_config_lock",
            "publish_intent_transitions",
        ],
    )

    inspection = inspect_reconcile_transaction_recovery(campaign)
    assert inspection["recoverable"] is True
    assert inspection["disposition"] == "completed_safe_cleanup"

    apply_reconcile_transaction_recovery(campaign, inspection)

    assert read_reconcile_transaction(path)["status"] == "FAILED"
    assert CampaignState.from_dict(json.loads(state_path.read_text())).phase.value == "INIT"


def test_schema_v1_untouched_commit_restores_exact_pointer_and_retires(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    write_state(state_path, CampaignState())
    versions = VersionedDirectory(campaign / "QM_REFERENCE_DATA")
    versions.iteration_path(0).mkdir(parents=True)
    pointer_snapshot = snapshot_version_pointer(
        campaign,
        versions,
        label="reference_data",
        requested_version=0,
    )
    versions.update_current(0)
    path = _write_v1_transaction(
        campaign,
        status="COMMITTING",
        proposed_phase="PHASE_A_DIVERSITY",
        extra={"pointer_snapshots": [pointer_snapshot]},
    )

    inspection = inspect_reconcile_transaction_recovery(campaign)

    assert inspection["action"] == "rollback_v1_pointers"
    apply_reconcile_transaction_recovery(campaign, inspection)
    assert versions.current_version() is None
    assert read_reconcile_transaction(path)["status"] == "FAILED"


def test_prepared_atomic_temp_is_promoted_then_abandoned(tmp_path):
    campaign = tmp_path / "campaign"
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase="INIT",
        proposed_iteration=0,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
    )
    temp = transaction.path.parent / ".t-deadbeefcafe"
    os.replace(transaction.path, temp)

    first = inspect_reconcile_transaction_recovery(campaign)
    assert first["action"] == "promote_prepared_orphan"
    apply_reconcile_transaction_recovery(campaign, first)

    second = inspect_reconcile_transaction_recovery(campaign)
    assert second["action"] == "abandon"
    apply_reconcile_transaction_recovery(campaign, second)

    payload = read_reconcile_transaction(transaction.path)
    assert payload["status"] == "FAILED"
    assert not temp.exists()


def test_redundant_atomic_temp_is_archived_before_canonical_recovery(tmp_path):
    campaign = tmp_path / "campaign"
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase="INIT",
        proposed_iteration=0,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
    )
    temp = transaction.path.parent / ".t-feedfacecafe"
    temp.write_bytes(transaction.path.read_bytes())

    first = inspect_reconcile_transaction_recovery(campaign)
    assert first["action"] == "archive_redundant_orphan"
    result = apply_reconcile_transaction_recovery(campaign, first)

    archived = Path(result["archived_path"])
    assert archived.is_file()
    assert archived.name.startswith(
        "o-" + transaction.payload["transaction_id"][:12] + "-"
    )
    assert len(archived.name) < 64
    assert transaction.path.is_file()
    second = inspect_reconcile_transaction_recovery(campaign)
    assert second["action"] == "abandon"


def test_redundant_atomic_temp_recognises_valid_legacy_archive(tmp_path):
    campaign = tmp_path.parent / ("l-" + uuid.uuid4().hex[:4])
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase="INIT",
        proposed_iteration=0,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
    )
    temp = transaction.path.parent / ".t-feedfacecafe"
    content = transaction.path.read_bytes()
    temp.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    archive_root = (
        campaign
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "reconcile_transaction_orphans"
    )
    archive_root.mkdir(parents=True)
    legacy = archive_root / (
        transaction.payload["transaction_id"]
        + "-"
        + digest
        + ".json.part"
    )
    legacy.write_bytes(content)

    inspection = inspect_reconcile_transaction_recovery(campaign)
    result = apply_reconcile_transaction_recovery(campaign, inspection)

    assert Path(result["archived_path"]) == legacy
    assert legacy.read_bytes() == content
    assert not temp.exists()


def test_redundant_atomic_temp_short_archive_collision_fails_closed(tmp_path):
    campaign = tmp_path / "campaign"
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase="INIT",
        proposed_iteration=0,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
    )
    temp = transaction.path.parent / ".t-feedfacecafe"
    content = transaction.path.read_bytes()
    temp.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    archive_root = (
        campaign
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "reconcile_transaction_orphans"
    )
    archive_root.mkdir(parents=True)
    collision = archive_root / (
        "o-"
        + transaction.payload["transaction_id"][:12]
        + "-"
        + digest[:16]
        + ".json.part"
    )
    collision.write_text("conflict", encoding="utf-8")

    inspection = inspect_reconcile_transaction_recovery(campaign)
    with pytest.raises(ValueError, match="archive conflicts"):
        apply_reconcile_transaction_recovery(campaign, inspection)

    assert temp.read_bytes() == content


def test_malformed_atomic_temp_remains_a_manual_review_blocker(tmp_path):
    campaign = tmp_path / "campaign"
    root = campaign / ".DATA" / "ACTIVE_LEARNING" / "reconcile_transactions"
    root.mkdir(parents=True)
    (root / ".t-deadbeefcafe").write_text("not-json", encoding="utf-8")

    inspection = inspect_reconcile_transaction_recovery(campaign)

    assert inspection["state"] == "blocked"
    assert inspection["recoverable"] is False


def test_multiple_nonterminal_transactions_remain_blocked(tmp_path):
    campaign = tmp_path / "campaign"
    _write_v1_transaction(campaign, status="PREPARED")
    _write_v1_transaction(campaign, status="PREPARED")

    inspection = inspect_reconcile_transaction_recovery(campaign)

    assert inspection["state"] == "blocked"
    assert "multiple" in inspection["reason"]


def test_schema_v2_prepared_authority_drift_remains_blocked(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    state = CampaignState()
    write_state(state_path, state)
    begin_reconcile_transaction(
        campaign,
        proposed_phase=state.phase.value,
        proposed_iteration=state.iteration,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
        campaign_uid=state.campaign_uid,
        source_authority_anchor_sha256="0" * 64,
    )

    inspection = inspect_reconcile_transaction_recovery(
        campaign,
        artifact_snapshot=SimpleNamespace(anchor_records=()),
    )

    assert inspection["state"] == "blocked"
    assert "authority changed" in inspection["reason"]


def test_schema_v2_committing_post_state_is_adopted(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    state = CampaignState()
    write_state(state_path, state)
    snapshot = SimpleNamespace(anchor_records=(), anchor_sha256=None)
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase=state.phase.value,
        proposed_iteration=state.iteration,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
        campaign_uid=state.campaign_uid,
    )
    transaction.set_status("MUTATING")
    transaction.prepare_commit(
        build_reconcile_commit_plan(
            campaign,
            transaction_id=transaction.payload["transaction_id"],
            proposed_state=state,
            config=None,
            intent_transitions=[],
            artifact_snapshot=snapshot,
        )
    )

    inspection = inspect_reconcile_transaction_recovery(campaign)
    assert inspection["action"] == "adopt"
    apply_reconcile_transaction_recovery(campaign, inspection)

    payload = read_reconcile_transaction(transaction.path)
    assert payload["status"] == "COMMITTED"
    assert payload["resolution"]["disposition"] == "adopted_committed_state"


def test_schema_v2_committing_untouched_authority_is_abandoned(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    before = CampaignState()
    write_state(state_path, before)
    after = CampaignState.from_dict(before.to_dict())
    after.max_iterations += 1
    snapshot = SimpleNamespace(anchor_records=(), anchor_sha256=None)
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase=after.phase.value,
        proposed_iteration=after.iteration,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
        campaign_uid=after.campaign_uid,
    )
    transaction.set_status("MUTATING")
    transaction.prepare_commit(
        build_reconcile_commit_plan(
            campaign,
            transaction_id=transaction.payload["transaction_id"],
            proposed_state=after,
            config=None,
            intent_transitions=[],
            artifact_snapshot=snapshot,
        )
    )

    inspection = inspect_reconcile_transaction_recovery(campaign)
    assert inspection["action"] == "abandon"
    assert inspection["disposition"] == "rolled_back_exactly"
    apply_reconcile_transaction_recovery(campaign, inspection)
    assert read_reconcile_transaction(transaction.path)["status"] == "FAILED"


def test_schema_v2_committing_unknown_state_remains_blocked(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    before = CampaignState()
    write_state(state_path, before)
    after = CampaignState.from_dict(before.to_dict())
    after.max_iterations += 1
    snapshot = SimpleNamespace(anchor_records=(), anchor_sha256=None)
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase=after.phase.value,
        proposed_iteration=after.iteration,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
        campaign_uid=after.campaign_uid,
    )
    transaction.set_status("MUTATING")
    transaction.prepare_commit(
        build_reconcile_commit_plan(
            campaign,
            transaction_id=transaction.payload["transaction_id"],
            proposed_state=after,
            config=None,
            intent_transitions=[],
            artifact_snapshot=snapshot,
        )
    )
    third = CampaignState.from_dict(before.to_dict())
    third.max_iterations += 2
    write_state(state_path, third)

    inspection = inspect_reconcile_transaction_recovery(campaign)

    assert inspection["state"] == "blocked"
    assert inspection["recoverable"] is False


def test_schema_v2_partial_authority_commit_rolls_forward(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    before = CampaignState()
    write_state(state_path, before)
    after = CampaignState.from_dict(before.to_dict())
    after.max_iterations += 1

    versions = VersionedDirectory(campaign / "QM_REFERENCE_DATA")
    versions.iteration_path(0).mkdir(parents=True)
    versions.iteration_path(1).mkdir()
    versions.update_current(0)

    snapshot = SimpleNamespace(anchor_records=(), anchor_sha256=None)
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase=after.phase.value,
        proposed_iteration=after.iteration,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
        campaign_uid=after.campaign_uid,
    )
    transaction.set_status("MUTATING")
    plan = build_reconcile_commit_plan(
        campaign,
        transaction_id=transaction.payload["transaction_id"],
        proposed_state=after,
        config=None,
        intent_transitions=[],
        artifact_snapshot=snapshot,
    )
    plan["pointers"] = [
        {
            "label": "test_reference_data",
            "parent": "QM_REFERENCE_DATA",
            "prefix": "iteration",
            "name_width": 6,
            "before_version": 0,
            "after_version": 1,
        }
    ]
    transaction.prepare_commit(plan)
    write_state(state_path, after)

    inspection = inspect_reconcile_transaction_recovery(campaign)
    assert inspection["action"] == "roll_forward"
    apply_reconcile_transaction_recovery(campaign, inspection)

    assert versions.current_version() == 1
    assert read_reconcile_transaction(transaction.path)["status"] == "COMMITTED"


def test_schema_v2_roll_forward_repairs_partial_state_backup(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    before = CampaignState()
    write_state(state_path, before)
    after = CampaignState.from_dict(before.to_dict())
    after.max_iterations += 1
    versions = VersionedDirectory(campaign / "QM_REFERENCE_DATA")
    versions.iteration_path(0).mkdir(parents=True)
    versions.iteration_path(1).mkdir()
    versions.update_current(0)
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase=after.phase.value,
        proposed_iteration=after.iteration,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
        campaign_uid=after.campaign_uid,
    )
    transaction.set_status("MUTATING")
    plan = build_reconcile_commit_plan(
        campaign,
        transaction_id=transaction.payload["transaction_id"],
        proposed_state=after,
        config=None,
        intent_transitions=[],
        artifact_snapshot=SimpleNamespace(anchor_records=(), anchor_sha256=None),
    )
    plan["pointers"] = [
        {
            "label": "test_reference_data",
            "parent": "QM_REFERENCE_DATA",
            "prefix": "iteration",
            "name_width": 6,
            "before_version": 0,
            "after_version": 1,
        }
    ]
    transaction.prepare_commit(plan)
    backup = campaign / plan["state"]["backup_path"]
    backup.write_bytes(b'{"partial":')
    versions.update_current(1)

    inspection = inspect_reconcile_transaction_recovery(campaign)

    assert inspection["action"] == "roll_forward"
    assert inspection["state_backup_status"] == "repairable_partial"
    apply_reconcile_transaction_recovery(campaign, inspection)
    assert json.loads(backup.read_text(encoding="utf-8")) == before.to_dict()
    assert CampaignState.from_dict(
        json.loads(state_path.read_text(encoding="utf-8"))
    ).to_dict() == after.to_dict()
    assert read_reconcile_transaction(transaction.path)["status"] == "COMMITTED"


def test_schema_v2_adoption_repairs_partial_state_backup(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    before = CampaignState()
    write_state(state_path, before)
    after = CampaignState.from_dict(before.to_dict())
    after.max_iterations += 1
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase=after.phase.value,
        proposed_iteration=after.iteration,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
        campaign_uid=after.campaign_uid,
    )
    transaction.set_status("MUTATING")
    plan = build_reconcile_commit_plan(
        campaign,
        transaction_id=transaction.payload["transaction_id"],
        proposed_state=after,
        config=None,
        intent_transitions=[],
        artifact_snapshot=SimpleNamespace(anchor_records=(), anchor_sha256=None),
    )
    transaction.prepare_commit(plan)
    backup = campaign / plan["state"]["backup_path"]
    backup.write_bytes(b'{"partial":')
    write_state(state_path, after)

    inspection = inspect_reconcile_transaction_recovery(campaign)

    assert inspection["action"] == "adopt"
    assert inspection["state_backup_status"] == "repairable_partial"
    apply_reconcile_transaction_recovery(campaign, inspection)
    assert json.loads(backup.read_text(encoding="utf-8")) == before.to_dict()
    assert read_reconcile_transaction(transaction.path)["status"] == "COMMITTED"


def test_schema_v2_abandon_removes_interrupted_backup_temporary(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    state = CampaignState()
    write_state(state_path, state)
    proposed = CampaignState.from_dict(state.to_dict())
    proposed.max_iterations += 1
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase=proposed.phase.value,
        proposed_iteration=proposed.iteration,
        planned_operations=["write_recovered_state"],
        intent_transitions=[],
        campaign_uid=proposed.campaign_uid,
    )
    transaction.set_status("MUTATING")
    plan = build_reconcile_commit_plan(
        campaign,
        transaction_id=transaction.payload["transaction_id"],
        proposed_state=proposed,
        config=None,
        intent_transitions=[],
        artifact_snapshot=SimpleNamespace(anchor_records=(), anchor_sha256=None),
    )
    transaction.prepare_commit(plan)
    backup = campaign / plan["state"]["backup_path"]
    temporary = backup.with_name("." + backup.name + ".publishing")
    temporary.write_bytes(b"partial")

    inspection = inspect_reconcile_transaction_recovery(campaign)

    assert inspection["action"] == "abandon"
    apply_reconcile_transaction_recovery(campaign, inspection)
    assert not temporary.exists()
    assert read_reconcile_transaction(transaction.path)["status"] == "FAILED"


def test_schema_v2_partial_config_history_commit_rolls_forward_exact_payloads(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    state = CampaignState()
    write_state(state_path, state)
    original = CampaignConfig()
    write_config_lock(campaign, original, campaign_uid=state.campaign_uid)
    changed = CampaignConfig()
    changed.ferebus.physical_prior_scale = 0.95
    snapshot = SimpleNamespace(anchor_records=(), anchor_sha256=None)
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase=state.phase.value,
        proposed_iteration=state.iteration,
        planned_operations=["update_config_lock"],
        intent_transitions=[],
        campaign_uid=state.campaign_uid,
    )
    transaction.set_status("MUTATING")
    plan = build_reconcile_commit_plan(
        campaign,
        transaction_id=transaction.payload["transaction_id"],
        proposed_state=state,
        config=changed,
        intent_transitions=[],
        artifact_snapshot=snapshot,
    )
    transaction.prepare_commit(plan)
    history_record = plan["config_lock"]["files"][0]
    atomic_write_json(
        campaign / history_record["path"],
        history_record["after_payload"],
    )

    inspection = inspect_reconcile_transaction_recovery(campaign)
    assert inspection["action"] == "roll_forward"
    apply_reconcile_transaction_recovery(campaign, inspection)

    lock = read_config_lock(campaign, expected_campaign_uid=state.campaign_uid)
    assert lock["canonical_config"] == changed.to_dict()
    assert read_reconcile_transaction(transaction.path)["status"] == "COMMITTED"


def test_schema_v2_partial_intent_commit_rolls_forward_exact_payload(tmp_path):
    campaign = tmp_path / "campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    state = CampaignState()
    write_state(state_path, state)
    write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=state.phase.value,
        iteration=state.iteration,
        expected_tasks=1,
        scheduler_identity_kind="synthetic",
    )
    mark_failed(campaign, state.phase.value, state.iteration, "test failure")
    snapshot = SimpleNamespace(anchor_records=(), anchor_sha256=None)
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase=state.phase.value,
        proposed_iteration=state.iteration,
        planned_operations=["publish_intent_transitions"],
        intent_transitions=[],
        campaign_uid=state.campaign_uid,
    )
    transaction.set_status("MUTATING")
    transaction.prepare_commit(
        build_reconcile_commit_plan(
            campaign,
            transaction_id=transaction.payload["transaction_id"],
            proposed_state=state,
            config=None,
            intent_transitions=[],
            artifact_snapshot=snapshot,
        )
    )

    inspection = inspect_reconcile_transaction_recovery(campaign)
    assert inspection["action"] == "roll_forward"
    apply_reconcile_transaction_recovery(campaign, inspection)

    intent = load_intent(campaign, state.phase.value, state.iteration)
    assert intent["status"] == "SUPERSEDED"
    assert intent["reason"] == "reconcile_apply_retry"


def test_commit_plan_preserves_terminal_ariadne_postprocess_reason(tmp_path):
    campaign = tmp_path / "campaign"
    state = CampaignState()
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 10
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True)
    write_state(state_path, state)
    write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=state.phase.value,
        iteration=state.iteration,
        expected_tasks=200,
        scheduler_identity_kind="sge",
    )
    mark_failed(campaign, state.phase.value, state.iteration, "parser-induced halt")

    plan = build_reconcile_commit_plan(
        campaign,
        transaction_id=uuid.uuid4().hex,
        proposed_state=state,
        config=None,
        intent_transitions=[
            {
                "phase": state.phase.value,
                "iteration": state.iteration,
                "target_status": "FAILED",
                "reason": ARIADNE_TERMINAL_POSTPROCESS_REASON,
                "ariadne_terminal_postprocess": True,
            }
        ],
        artifact_snapshot=SimpleNamespace(anchor_records=(), anchor_sha256=None),
    )

    transition = plan["intent_transitions"][0]
    assert transition["target_status"] == "SUPERSEDED"
    assert transition["reason"] == ARIADNE_TERMINAL_POSTPROCESS_REASON
    assert transition["after_payload"]["status"] == "SUPERSEDED"
    assert transition["after_payload"]["reason"] == (
        ARIADNE_TERMINAL_POSTPROCESS_REASON
    )


def test_terminal_intent_classification_is_proposal_only(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    write_pre_submit_intent(
        campaign,
        campaign_uid="reconcile-intent-test",
        phase_name="FEREBUS",
        iteration=0,
        expected_tasks=1,
    )
    mark_submitted(campaign, "FEREBUS", 0, "123", expected_tasks=1)
    active = load_intent(campaign, "FEREBUS", 0)
    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda _job_id: sacct_poll.JobQueueLookup(active=False, rows=[]),
    )
    monkeypatch.setattr(
        sacct_poll,
        "poll_job",
        lambda _job_id: [
            sacct_poll.JobObservation(
                job_id="123",
                status=sacct_poll.JobStatus.CANCELLED,
                exit_code=(1, 0),
                elapsed_seconds=1,
            )
        ],
    )

    resolved, blocking = cli._resolve_terminal_submission_intents_for_apply(
        campaign,
        [active],
    )

    assert blocking == []
    assert resolved[0]["target_status"] == "FAILED"
    assert load_intent(campaign, "FEREBUS", 0)["status"] == "SUBMITTED"
