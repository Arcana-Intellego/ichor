"""Recovery of FEREBUS producer staging displaced by reconcile."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
import ichor.hpc.active_learning.cli as cli_module
from ichor.hpc.active_learning.daemon import ferebus_staging_recovery as recovery
from ichor.hpc.active_learning.daemon import live_executor as live_executor_module
from ichor.hpc.active_learning.daemon.ferebus_staging_recovery import (
    FerebusStagingRecoveryContext,
    archive_partial_ferebus_preparation,
    classify_ferebus_staging_recovery,
    is_redundant_committed_parent_staging,
    restore_archived_ferebus_producer_staging,
)
from ichor.hpc.active_learning.daemon.live_executor import (
    LiveBackendsPhaseExecutor,
)
from ichor.hpc.active_learning.daemon.config_lock import (
    clean_model_iteration_staging_for_reconcile,
    clean_reentry_staging,
)
from ichor.hpc.active_learning.daemon.phase_executor import (
    BackendSubmissionError,
)
from ichor.hpc.active_learning.daemon.reconcile_transaction import (
    begin_reconcile_transaction,
)
from ichor.hpc.active_learning.daemon.state import CampaignPhase
from ichor.hpc.active_learning.ferebus_prior import (
    resolve_ferebus_prior_contract,
)


def _manifest(*, n_tasks: int = 6):
    return {
        "campaign_uid": "campaign-uid",
        "reference_data_version": 8,
        "n_tasks": n_tasks,
    }


def _inspection(
    path: Path,
    disposition: str,
    *,
    bound_task_ids=(),
    task_map_sha256: str = "a" * 64,
    manifest_identity_sha256: str = "b" * 64,
):
    has_map = disposition == "prepared"
    return recovery._TreeInspection(
        disposition=disposition,
        path=Path(path),
        manifest=(None if disposition == "absent" else _manifest()),
        task_map=(
            {"task_map_sha256": task_map_sha256} if has_map else None
        ),
        manifest_sha256=(None if disposition == "absent" else "c" * 64),
        manifest_identity_sha256=(
            None if disposition == "absent" else manifest_identity_sha256
        ),
        producer_bound_task_ids=tuple(bound_task_ids),
        reason=disposition,
    )


def _terminal_sources():
    return {
        "recoveries": (
            {
                "intent": {"submission_identity": "r0000-a0001-source"},
                "receipt": {},
            },
        ),
        "scheduler": "sge",
        "completed": (0, 1, 2, 3, 5),
        "retry": (4,),
        "jobs": ("898506",),
        "identities": ("r0000-a0001-source",),
        "receipt_paths": ("terminal.json",),
    }


@pytest.mark.parametrize(
    "canonical_disposition",
    ["input_only", "partial_preparation"],
)
def test_classifier_finds_unique_archived_terminal_producer(
    tmp_path,
    monkeypatch,
    canonical_disposition,
):
    campaign = tmp_path / "campaign"
    canonical = campaign / "TRAINED_MODELS" / "iteration-staging"
    archived = canonical.with_name(
        "iteration-staging.before-reconcile-" + "1" * 32
    )
    canonical_tree = _inspection(canonical, canonical_disposition)
    producer_tree = _inspection(
        archived,
        "prepared",
        bound_task_ids=(0, 1, 2, 3, 5),
    )
    monkeypatch.setattr(recovery, "_inspect_tree", lambda path: canonical_tree)
    monkeypatch.setattr(
        recovery,
        "_archive_candidates",
        lambda *args, **kwargs: (("1" * 32, archived, producer_tree),),
    )
    monkeypatch.setattr(
        recovery,
        "_terminal_sources",
        lambda *args, **kwargs: _terminal_sources(),
    )
    monkeypatch.setattr(
        recovery,
        "_validate_historical_executable",
        lambda *args, **kwargs: None,
    )

    context = classify_ferebus_staging_recovery(
        campaign,
        campaign_uid="campaign-uid",
        phase="FEREBUS",
        iteration=8,
        reference_data_version=8,
    )

    assert context.disposition == "archived_terminal_producer"
    assert context.producer_path == archived
    assert context.source_transaction_id == "1" * 32
    assert context.completed_logical_task_ids == (0, 1, 2, 3, 5)
    assert context.retry_logical_task_ids == (4,)
    assert context.has_authenticated_producer_task_map is True


def test_new_prepared_map_is_not_mislabelled_as_historical_producer(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    canonical = campaign / "TRAINED_MODELS" / "iteration-staging"
    prepared = _inspection(canonical, "prepared", bound_task_ids=())
    monkeypatch.setattr(recovery, "_inspect_tree", lambda path: prepared)
    monkeypatch.setattr(
        recovery,
        "_archive_candidates",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        recovery,
        "_terminal_sources",
        lambda *args, **kwargs: _terminal_sources(),
    )

    context = classify_ferebus_staging_recovery(
        campaign,
        campaign_uid="campaign-uid",
        phase="FEREBUS",
        iteration=8,
        reference_data_version=8,
    )

    assert context.disposition == "prepared"
    assert context.has_authenticated_task_map is True
    assert context.has_authenticated_producer_task_map is False
    assert "no successful receipt" in context.reason


def test_authenticated_committed_parent_staging_is_not_current_recovery(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    staging = campaign / "TRAINED_MODELS" / "iteration-staging"
    staging.mkdir(parents=True)
    monkeypatch.setattr(
        recovery,
        "read_ferebus_manifest",
        lambda *args, **kwargs: {
            "campaign_uid": "campaign-uid",
            "reference_data_version": 7,
        },
    )
    monkeypatch.setattr(
        recovery,
        "scheduler_terminal_recoveries",
        lambda *args, **kwargs: (),
    )
    parent = SimpleNamespace(
        version=7,
        campaign_uid="campaign-uid",
        reference_data_version=7,
    )
    snapshot = SimpleNamespace(model_set=lambda version: parent)

    assert is_redundant_committed_parent_staging(
        campaign,
        campaign_uid="campaign-uid",
        target_reference_data_version=8,
        parent_model_version=7,
        artifact_snapshot=snapshot,
    )
    assert not is_redundant_committed_parent_staging(
        campaign,
        campaign_uid="campaign-uid",
        target_reference_data_version=9,
        parent_model_version=7,
        artifact_snapshot=snapshot,
    )

    custom_staging = campaign / "CUSTOM_MODELS" / "iteration-staging"
    custom_staging.mkdir(parents=True)
    monkeypatch.setattr(
        recovery,
        "scheduler_terminal_recoveries",
        lambda *args, **kwargs: (),
    )
    assert is_redundant_committed_parent_staging(
        campaign,
        campaign_uid="campaign-uid",
        target_reference_data_version=8,
        parent_model_version=7,
        artifact_snapshot=snapshot,
        models_dir_name="CUSTOM_MODELS",
    )
    monkeypatch.setattr(
        recovery,
        "scheduler_terminal_recoveries",
        lambda *args, **kwargs: ({"receipt": {}},),
    )
    assert not is_redundant_committed_parent_staging(
        campaign,
        campaign_uid="campaign-uid",
        target_reference_data_version=8,
        parent_model_version=7,
        artifact_snapshot=snapshot,
    )


def test_tree_inspection_requires_manifest_bound_job_details(
    tmp_path,
    monkeypatch,
):
    staging = tmp_path / "iteration-staging"
    staging.mkdir()
    (staging / "FEREBUS_TASKS.json").write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        recovery,
        "read_ferebus_manifest",
        lambda *args, **kwargs: {
            **_manifest(),
            "job_details": "job-details",
            "tasks": [],
        },
    )
    monkeypatch.setattr(recovery, "_expected_output_paths", lambda *args: ())

    inspection = recovery._inspect_tree(staging)

    assert inspection.disposition == "contradictory"
    assert "job details" in inspection.reason


def test_classifier_accepts_resource_changes_but_rejects_scientific_config(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    canonical = campaign / "TRAINED_MODELS" / "iteration-staging"
    config = CampaignConfig()
    config.system_name = "WATER"
    config.resources.partition = "multicore"
    manifest = {
        **_manifest(),
        "system": "WATER",
        "properties": ["iqa"],
        "atoms": ["O1"],
        "kernel_contract": {"family": config.ferebus.kernel},
        "prior_mean_contract": resolve_ferebus_prior_contract(
            config,
            atom_labels=["O1"],
        ).to_dict(),
    }
    tree = recovery._TreeInspection(
        disposition="input_only",
        path=canonical,
        manifest=manifest,
        manifest_sha256="c" * 64,
        manifest_identity_sha256="b" * 64,
    )
    monkeypatch.setattr(recovery, "_inspect_tree", lambda path: tree)
    monkeypatch.setattr(
        recovery,
        "_archive_candidates",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        recovery,
        "_terminal_sources",
        lambda *args, **kwargs: _terminal_sources(),
    )

    accepted = classify_ferebus_staging_recovery(
        campaign,
        campaign_uid="campaign-uid",
        phase="FEREBUS",
        iteration=8,
        reference_data_version=8,
        config=config,
    )
    assert accepted.disposition == "input_only"

    config.ferebus.kernel = "rbf"
    rejected = classify_ferebus_staging_recovery(
        campaign,
        campaign_uid="campaign-uid",
        phase="FEREBUS",
        iteration=8,
        reference_data_version=8,
        config=config,
    )
    assert rejected.disposition == "contradictory"
    assert "scientific configuration" in rejected.reason


def test_multiple_archived_terminal_producers_fail_closed(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    canonical = campaign / "TRAINED_MODELS" / "iteration-staging"
    canonical_tree = _inspection(canonical, "input_only")
    first_path = canonical.with_name("iteration-staging.before-reconcile-" + "1" * 32)
    second_path = canonical.with_name("iteration-staging.before-reconcile-" + "2" * 32)
    first = _inspection(first_path, "prepared", bound_task_ids=(0,))
    second = _inspection(second_path, "prepared", bound_task_ids=(1,))
    monkeypatch.setattr(recovery, "_inspect_tree", lambda path: canonical_tree)
    monkeypatch.setattr(
        recovery,
        "_archive_candidates",
        lambda *args, **kwargs: (
            ("1" * 32, first_path, first),
            ("2" * 32, second_path, second),
        ),
    )
    monkeypatch.setattr(
        recovery,
        "_terminal_sources",
        lambda *args, **kwargs: _terminal_sources(),
    )

    context = classify_ferebus_staging_recovery(
        campaign,
        campaign_uid="campaign-uid",
        phase="FEREBUS",
        iteration=8,
        reference_data_version=8,
    )

    assert context.disposition == "contradictory"
    assert "multiple archived" in context.reason


def test_archive_discovery_requires_committed_transaction_operation(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    transaction_id = "5" * 32
    archived = (
        campaign
        / "TRAINED_MODELS"
        / ("iteration-staging.before-reconcile-" + transaction_id)
    )
    archived.mkdir(parents=True)
    transaction = begin_reconcile_transaction(
        campaign,
        proposed_phase="FEREBUS",
        proposed_iteration=8,
        planned_operations=["archive_model_staging"],
        intent_transitions=[],
        campaign_uid="campaign-uid",
        transaction_id=transaction_id,
    )
    transaction.set_status("MUTATING")
    transaction.record_paths("archive_model_staging", [str(archived)])
    transaction.resolve(
        status="COMMITTED",
        disposition="applied",
        reason="test archive",
    )
    inspection = _inspection(
        archived,
        "prepared",
        bound_task_ids=(0,),
    )
    monkeypatch.setattr(recovery, "_inspect_tree", lambda path: inspection)

    candidates = recovery._archive_candidates(
        campaign,
        campaign_uid="campaign-uid",
        iteration=8,
    )

    assert candidates == ((transaction_id, archived, inspection),)


def test_restore_archived_producer_is_atomic_and_replayable(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    canonical = campaign / "TRAINED_MODELS" / "iteration-staging"
    producer = canonical.with_name(
        "iteration-staging.before-reconcile-" + "1" * 32
    )
    canonical.mkdir(parents=True)
    producer.mkdir()
    (canonical / "input.marker").write_text("input", encoding="ascii")
    (producer / "producer.marker").write_text("producer", encoding="ascii")

    def inspect(path):
        candidate = Path(path)
        if not candidate.exists():
            return recovery._TreeInspection("absent", candidate)
        if (candidate / "producer.marker").is_file():
            return _inspection(
                candidate,
                "prepared",
                bound_task_ids=(0, 1, 2, 3, 5),
            )
        if (candidate / "input.marker").is_file():
            return _inspection(candidate, "input_only")
        return recovery._TreeInspection(
            "contradictory",
            candidate,
            reason="unknown test tree",
        )

    monkeypatch.setattr(recovery, "_inspect_tree", inspect)
    context = FerebusStagingRecoveryContext(
        disposition="archived_terminal_producer",
        canonical_path=canonical,
        producer_path=producer,
        source_transaction_id="1" * 32,
        phase="FEREBUS",
        iteration=8,
        n_tasks=6,
        task_map_sha256="a" * 64,
        manifest_identity_sha256="b" * 64,
    )

    restored = restore_archived_ferebus_producer_staging(
        campaign,
        context,
        transaction_id="2" * 32,
    )

    input_archive = canonical.with_name(
        "iteration-staging.before-reconcile-" + "2" * 32
    )
    assert restored["changed"] is True
    assert (canonical / "producer.marker").read_text(encoding="ascii") == "producer"
    assert (input_archive / "input.marker").read_text(encoding="ascii") == "input"
    assert not producer.exists()

    replay = restore_archived_ferebus_producer_staging(
        campaign,
        context,
        transaction_id="2" * 32,
    )
    assert replay["changed"] is False


def test_reconcile_reentry_cleanup_preserves_terminal_producer_staging(
    tmp_path,
):
    campaign = tmp_path / "campaign"
    staging = campaign / "TRAINED_MODELS" / "iteration-staging"
    staging.mkdir(parents=True)
    marker = staging / "producer.marker"
    marker.write_text("producer", encoding="ascii")

    archived = clean_reentry_staging(
        campaign,
        CampaignPhase.FEREBUS,
        archive_identity="4" * 32,
        preserve_ferebus_iteration_staging=True,
    )

    assert archived == []
    assert marker.read_text(encoding="ascii") == "producer"


def test_model_cleanup_archives_only_unrelated_version_staging(
    tmp_path,
):
    campaign = tmp_path / "campaign"
    models = campaign / "TRAINED_MODELS"
    canonical = models / "iteration-staging"
    dangling = models / "iteration-000008.staging"
    canonical.mkdir(parents=True)
    dangling.mkdir()
    marker = canonical / "producer.marker"
    marker.write_text("producer", encoding="ascii")

    archived = clean_model_iteration_staging_for_reconcile(
        campaign,
        SimpleNamespace(phase=CampaignPhase.FEREBUS),
        archive_identity="6" * 32,
        preserve_ferebus_iteration_staging=True,
    )

    expected = dangling.with_name(
        dangling.name + ".before-reconcile-" + "6" * 32
    )
    assert archived == [str(expected)]
    assert expected.is_dir()
    assert marker.read_text(encoding="ascii") == "producer"


def test_restore_replays_after_input_archive_rename(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    canonical = campaign / "TRAINED_MODELS" / "iteration-staging"
    producer = canonical.with_name(
        "iteration-staging.before-reconcile-" + "1" * 32
    )
    input_archive = canonical.with_name(
        "iteration-staging.before-reconcile-" + "2" * 32
    )
    producer.mkdir(parents=True)
    input_archive.mkdir()
    (producer / "producer.marker").write_text("producer", encoding="ascii")
    (input_archive / "input.marker").write_text("input", encoding="ascii")

    def inspect(path):
        candidate = Path(path)
        if not candidate.exists():
            return recovery._TreeInspection("absent", candidate)
        if (candidate / "producer.marker").is_file():
            return _inspection(
                candidate,
                "prepared",
                bound_task_ids=(0, 1, 2, 3, 5),
            )
        if (candidate / "input.marker").is_file():
            return _inspection(candidate, "input_only")
        return recovery._TreeInspection(
            "contradictory",
            candidate,
            reason="unknown test tree",
        )

    monkeypatch.setattr(recovery, "_inspect_tree", inspect)
    context = FerebusStagingRecoveryContext(
        disposition="archived_terminal_producer",
        canonical_path=canonical,
        producer_path=producer,
        source_transaction_id="1" * 32,
        phase="FEREBUS",
        iteration=8,
        n_tasks=6,
        task_map_sha256="a" * 64,
        manifest_identity_sha256="b" * 64,
    )

    restored = restore_archived_ferebus_producer_staging(
        campaign,
        context,
        transaction_id="2" * 32,
    )

    assert restored["changed"] is True
    assert (canonical / "producer.marker").is_file()
    assert (input_archive / "input.marker").is_file()


def test_partial_preparation_archival_is_deterministic_and_replayable(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    canonical = campaign / "TRAINED_MODELS" / "iteration-staging"
    canonical.mkdir(parents=True)
    (canonical / "partial.marker").write_text("partial", encoding="ascii")

    def inspect(path):
        candidate = Path(path)
        if not candidate.exists():
            return recovery._TreeInspection("absent", candidate)
        if (candidate / "partial.marker").is_file():
            return _inspection(candidate, "partial_preparation")
        return recovery._TreeInspection(
            "contradictory",
            candidate,
            reason="unknown test tree",
        )

    monkeypatch.setattr(recovery, "_inspect_tree", inspect)
    context = FerebusStagingRecoveryContext(
        disposition="partial_preparation",
        canonical_path=canonical,
        phase="FEREBUS",
        iteration=8,
        n_tasks=6,
        manifest_identity_sha256="b" * 64,
    )

    archived = archive_partial_ferebus_preparation(
        campaign,
        context,
        attempt_id="3" * 32,
    )

    expected = canonical.with_name(
        "iteration-staging.partial-preparation-" + "3" * 32
    )
    assert archived == {"changed": True, "archived_path": str(expected)}
    assert not canonical.exists()
    assert (expected / "partial.marker").is_file()

    replay = archive_partial_ferebus_preparation(
        campaign,
        context,
        attempt_id="3" * 32,
    )
    assert replay == {"changed": False, "archived_path": str(expected)}


def _terminal_recovery_record():
    identity = "r0000-a0001-source"
    outcomes = [
        {"logical_task_id": task_id}
        for task_id in range(6)
    ]
    return {
        "intent": {
            "submission_identity": identity,
            "attempt_id": "1" * 32,
            "environment_generation": 1,
            "environment_generation_digest_sha256": "d" * 64,
        },
        "receipt": {
            "submission_identity": identity,
            "receipt_sha256": "e" * 64,
            "job_id": "898506",
            "completed_logical_task_ids": [0, 1, 2, 3, 5],
            "outcomes": outcomes,
        },
        "path": "terminal.json",
    }


def _recovery_executor(tmp_path, monkeypatch):
    executor = object.__new__(LiveBackendsPhaseExecutor)
    executor.campaign_dir = tmp_path
    source = _terminal_recovery_record()
    monkeypatch.setattr(
        executor,
        "_scheduler_cancel_recovery_sources",
        lambda state, phase: [source],
    )
    monkeypatch.setattr(
        executor,
        "_scheduler_recovery_environment_assessments",
        lambda recoveries: {
            "r0000-a0001-source": {
                "equivalent": True,
                "reasons": [],
                "path": "equivalence.json",
                "sha256": "f" * 64,
            }
        },
    )
    captured = {}

    def write_ledger(campaign, payload):
        captured.update(payload)
        return dict(payload)

    monkeypatch.setattr(
        live_executor_module,
        "write_phase_recovery_ledger",
        write_ledger,
    )
    return executor, captured


def test_mapless_terminal_recovery_is_one_staging_failure(
    tmp_path,
    monkeypatch,
):
    executor, captured = _recovery_executor(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.ferebus_task_runner.validate_task_receipt",
        lambda *args, **kwargs: pytest.fail(
            "mapless recovery must not validate per-task receipts"
        ),
    )
    context = FerebusStagingRecoveryContext(
        disposition="input_only",
        canonical_path=tmp_path / "TRAINED_MODELS" / "iteration-staging",
        phase="FEREBUS",
        iteration=8,
        n_tasks=6,
        completed_logical_task_ids=(0, 1, 2, 3, 5),
        retry_logical_task_ids=(4,),
        source_job_ids=("898506",),
        source_submission_identities=("r0000-a0001-source",),
    )

    result = executor._prepare_scheduler_cancelled_ferebus_recovery(
        SimpleNamespace(
            campaign_uid="campaign-uid",
            iteration=8,
            replacement_round=0,
        ),
        "FEREBUS",
        context.canonical_path,
        n_tasks=6,
        staging_context=context,
        staging_context_resolver=lambda: context,
    )

    assert result["reusable_logical_task_ids"] == []
    assert result["retry_logical_task_ids"] == [0, 1, 2, 3, 4, 5]
    assert result["staging_has_task_map"] is False
    assert captured["invalid_completed_tasks"] == [
        {
            "scope": "staging",
            "reason": "producer_task_map_unavailable",
            "scheduler_completed_candidates": 5,
        }
    ]


def test_terminal_producer_reuses_only_valid_completed_tasks(
    tmp_path,
    monkeypatch,
):
    executor, captured = _recovery_executor(tmp_path, monkeypatch)
    validated = []
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.ferebus_task_runner.validate_task_receipt",
        lambda staging, task_id: validated.append(task_id),
    )
    context = FerebusStagingRecoveryContext(
        disposition="terminal_producer",
        canonical_path=tmp_path / "TRAINED_MODELS" / "iteration-staging",
        producer_path=tmp_path / "TRAINED_MODELS" / "iteration-staging",
        phase="FEREBUS",
        iteration=8,
        n_tasks=6,
        completed_logical_task_ids=(0, 1, 2, 3, 5),
        retry_logical_task_ids=(4,),
        source_job_ids=("898506",),
        source_submission_identities=("r0000-a0001-source",),
        task_map_sha256="a" * 64,
    )

    result = executor._prepare_scheduler_cancelled_ferebus_recovery(
        SimpleNamespace(
            campaign_uid="campaign-uid",
            iteration=8,
            replacement_round=0,
        ),
        "FEREBUS",
        context.canonical_path,
        n_tasks=6,
        staging_context=context,
        staging_context_resolver=lambda: context,
    )

    assert validated == [0, 1, 2, 3, 5, 0, 1, 2, 3, 5]
    assert result["reusable_logical_task_ids"] == [0, 1, 2, 3, 5]
    assert result["retry_logical_task_ids"] == [4]
    assert result["staging_has_task_map"] is True
    assert captured["invalid_completed_tasks"] == []


def test_recovery_ledger_aborts_when_staging_authority_changes(
    tmp_path,
    monkeypatch,
):
    executor, captured = _recovery_executor(tmp_path, monkeypatch)
    context = FerebusStagingRecoveryContext(
        disposition="input_only",
        canonical_path=tmp_path / "TRAINED_MODELS" / "iteration-staging",
        phase="FEREBUS",
        iteration=8,
        n_tasks=6,
        completed_logical_task_ids=(0, 1, 2, 3, 5),
        retry_logical_task_ids=(4,),
        source_job_ids=("898506",),
        source_submission_identities=("r0000-a0001-source",),
        manifest_identity_sha256="b" * 64,
    )
    changed = FerebusStagingRecoveryContext(
        **{
            **context.__dict__,
            "manifest_identity_sha256": "9" * 64,
        }
    )

    with pytest.raises(BackendSubmissionError, match="changed during"):
        executor._prepare_scheduler_cancelled_ferebus_recovery(
            SimpleNamespace(
                campaign_uid="campaign-uid",
                iteration=8,
                replacement_round=0,
            ),
            "FEREBUS",
            context.canonical_path,
            n_tasks=6,
            staging_context=context,
            staging_context_resolver=lambda: changed,
        )

    assert captured == {}


def test_reconcile_mutation_records_one_producer_restoration(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    context = FerebusStagingRecoveryContext(
        disposition="archived_terminal_producer",
        canonical_path=campaign / "TRAINED_MODELS" / "iteration-staging",
        producer_path=(
            campaign
            / "TRAINED_MODELS"
            / ("iteration-staging.before-reconcile-" + "1" * 32)
        ),
        source_transaction_id="1" * 32,
        phase="FEREBUS",
        iteration=8,
        n_tasks=6,
        completed_logical_task_ids=(0, 1, 2, 3, 5),
        retry_logical_task_ids=(4,),
        source_job_ids=("898506",),
        source_submission_identities=("r0000-a0001-source",),
        task_map_sha256="a" * 64,
        manifest_identity_sha256="b" * 64,
    )
    report = SimpleNamespace(
        proposed_state=SimpleNamespace(
            campaign_uid="campaign-uid",
            phase=CampaignPhase.FEREBUS,
            iteration=8,
            replacement_round=0,
            reference_data_version=8,
        ),
        ferebus_staging_recovery=context.summary(),
        unsafe_reasons=[],
        partial_array_recovery=None,
        ariadne_publication_recovery=None,
        aimall_upstream_gaussian_recovery=None,
        completed_staging_retirement=None,
    )
    reference_view = SimpleNamespace(
        head_manifest_sha256="c" * 64,
        cumulative_view_sha256="d" * 64,
    )
    snapshot = SimpleNamespace(reference_view=lambda version: reference_view)
    config = CampaignConfig()
    captured_config = []
    monkeypatch.setattr(
        cli_module,
        "classify_ferebus_staging_recovery",
        lambda *args, **kwargs: (
            captured_config.append(kwargs.get("config")) or context
        ),
    )
    restored = {
        "changed": True,
        "restored_path": str(context.canonical_path),
        "archived_input_path": str(
            context.canonical_path.with_name(
                "iteration-staging.before-reconcile-" + "2" * 32
            )
        ),
    }
    monkeypatch.setattr(
        cli_module,
        "restore_archived_ferebus_producer_staging",
        lambda *args, **kwargs: dict(restored),
    )
    cleanup_calls = []
    monkeypatch.setattr(
        cli_module,
        "clean_reentry_staging",
        lambda *args, **kwargs: cleanup_calls.append((args, kwargs)) or [],
    )

    class Transaction:
        payload = {"transaction_id": "2" * 32}

        def __init__(self):
            self.operations = []

        def record_paths(self, operation, paths):
            self.operations.append((operation, list(paths)))

    transaction = Transaction()
    result = cli_module._perform_reconcile_apply_mutations(
        campaign,
        report,
        transaction=transaction,
        retrain_ferebus=False,
        force_resubmit_array=False,
        partial_array=None,
        force_array_phase=CampaignPhase.FEREBUS,
        force_array_iteration=8,
        archive_existing_array_outputs=False,
        data_staging_archive_mode=None,
        artifact_snapshot=snapshot,
        campaign_config=config,
    )

    assert result["ferebus_staging_restore"] == restored
    assert captured_config == [config]
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][1]["preserve_ferebus_iteration_staging"] is True
    assert transaction.operations.count(
        (
            "restore_ferebus_producer_staging",
            [restored["restored_path"], restored["archived_input_path"]],
        )
    ) == 1
