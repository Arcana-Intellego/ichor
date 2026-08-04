from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon import ariadne_resource_reuse as reuse
from ichor.hpc.active_learning.daemon import config_lock as config_lock_module
from ichor.hpc.active_learning.daemon import resource_solver
from ichor.hpc.active_learning.daemon.config_lock import (
    canonical_config,
    config_fingerprint,
    read_historical_config_by_fingerprint,
    write_config_lock,
)
from ichor.hpc.active_learning.daemon.resource_records import (
    RESOURCE_FORMULA_VERSION,
    RESOURCE_RESOLUTION_SCHEMA_VERSION,
    resolution_path,
)
from ichor.hpc.active_learning.handoff_manifests import ariadne_task_map_path
from ichor.hpc.active_learning.layout import active_iteration_dir
from ichor.hpc.active_learning.versioning.manifest import sha256_file


def _file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _intent(
    *,
    attempt: str,
    identity: str,
    generation: int,
    generation_digest: str,
) -> dict:
    return {
        "campaign_uid": "campaign-resource-reuse",
        "phase": "ARIADNE_ARRAY",
        "iteration": 15,
        "replacement_round": 0,
        "attempt_id": attempt,
        "submission_identity": identity,
        "scheduler_identity_kind": "slurm",
        "environment_generation": generation,
        "environment_generation_digest_sha256": generation_digest,
    }


def _write_resource_record(
    campaign: Path,
    intent: dict,
    evidence: dict,
) -> dict:
    path = resolution_path(
        campaign,
        "ARIADNE_ARRAY",
        15,
        intent["submission_identity"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RESOURCE_RESOLUTION_SCHEMA_VERSION,
        "formula_version": RESOURCE_FORMULA_VERSION,
        "created_at_iso": "2026-07-28T00:00:00+00:00",
        "campaign_uid": intent["campaign_uid"],
        "phase": intent["phase"],
        "iteration": intent["iteration"],
        "attempt_id": intent["attempt_id"],
        "submission_identity": intent["submission_identity"],
        "resources": {"backend": "ariadne"},
        "evidence": evidence,
        "implementation_identity": {},
        "scratch_path_template": "/scratch/test",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    bound = dict(intent)
    bound["resource_resolution_path"] = str(path.resolve())
    bound["resource_resolution_sha256"] = sha256_file(path)
    return bound


def _resource_fixture(tmp_path: Path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    pool = campaign / "pool.xyz"
    pool.write_text("1\nframe\nH 0 0 0\n", encoding="utf-8")
    iteration_dir = active_iteration_dir(campaign, 15)
    task_map_file = ariadne_task_map_path(iteration_dir)
    task_map_file.parent.mkdir(parents=True, exist_ok=True)
    task_map_file.write_text("{}", encoding="utf-8")
    model_root = campaign / "TRAINED_MODELS" / "iteration-000014"
    model_root.mkdir(parents=True)
    task_map = {
        "n_tasks": 4,
        "trajectory_sha256": sha256_file(pool),
        "models_version": 14,
        "model_manifest_sha256": "a" * 64,
        "model_set_sha256": "b" * 64,
        "tasks": [
            {"logical_task_id": index, "pool_row_index_zero_based": index}
            for index in range(4)
        ],
    }
    evidence = {
        "source": "ariadne_task_map_and_model_set",
        "task_map": _file_record(task_map_file),
        "pool": _file_record(pool),
        "pool_n_frames": 10_000,
        "pool_file_bytes": int(pool.stat().st_size),
        "decoded_coordinate_bytes": 1024,
        "decoded_object_allowance_bytes": 1024,
        "decoded_pool_estimate_bytes": 4096,
        "n_tasks": 4,
        "n_atoms": 6,
        "gradient_dimension": 12,
        "models_version": 14,
        "model_manifest_sha256": "a" * 64,
        "model_set_sha256": "b" * 64,
        "model_bytes": 1024,
        "gradient_dimensions": [2, 5, 7, 12],
        "gradient_dimension_source": "exact_seed_local_subspaces",
    }
    source_config = CampaignConfig()
    source_config.resources.partition = "multicore_small"
    current_config = CampaignConfig()
    current_config.resources.partition = "multicore"
    source_generation = {
        "generation": 8,
        "digest_sha256": "8" * 64,
        "campaign_config_sha256": "c" * 64,
    }
    current_generation = {
        "generation": 9,
        "digest_sha256": "9" * 64,
        "campaign_config_sha256": "d" * 64,
    }
    monkeypatch.setattr(reuse, "read_ariadne_task_map", lambda *_a, **_k: task_map)
    monkeypatch.setattr(
        reuse,
        "resolve_trained_model_set",
        lambda *_a, **_k: SimpleNamespace(
            root=model_root,
            head_manifest_sha256="a" * 64,
            model_set_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        reuse,
        "read_active_environment_generation",
        lambda *_a, **_k: {"generation": current_generation},
    )
    monkeypatch.setattr(
        reuse,
        "read_environment_generation",
        lambda *_a, **_k: source_generation,
    )
    monkeypatch.setattr(
        reuse,
        "read_historical_config_by_fingerprint",
        lambda *_a, **_k: source_config,
    )
    monkeypatch.setattr(
        reuse,
        "assess_resource_evidence_code_equivalence",
        lambda *_a, **_k: {
            "equivalent": True,
            "fingerprint_algorithm": "repository_module_closure_v2",
        },
    )
    return campaign, current_config, evidence


def test_retry_reuses_full_evidence_without_geometry_recomputation(
    tmp_path,
    monkeypatch,
):
    campaign, config, evidence = _resource_fixture(tmp_path, monkeypatch)
    source = _write_resource_record(
        campaign,
        _intent(
            attempt="attempt-1",
            identity="r0000-a0001-source",
            generation=8,
            generation_digest="8" * 64,
        ),
        evidence,
    )
    active = _intent(
        attempt="attempt-2",
        identity="r0000-a0002-retry",
        generation=9,
        generation_digest="9" * 64,
    )
    monkeypatch.setattr(reuse, "intent_attempt_records", lambda *_a, **_k: [source])

    selected = reuse.resolve_reusable_ariadne_resource_evidence(
        campaign,
        config,
        active,
        expected_campaign_uid="campaign-resource-reuse",
        iteration=15,
        replacement_round=0,
        expected_scheduler_kind="slurm",
        expected_models_version=14,
        submitted_task_ids=[1, 3],
    )

    assert selected is not None
    assert selected.source_submission_identity == "r0000-a0001-source"
    assert selected.source_task_count == 4
    assert selected.already_filtered is False
    assert selected.evidence["gradient_dimensions"] == [2, 5, 7, 12]

    monkeypatch.setattr(
        resource_solver,
        "collect_resource_evidence",
        lambda **_kwargs: pytest.fail("fresh ARIADNE evidence was recomputed"),
    )
    monkeypatch.setattr(
        resource_solver,
        "validate_partition_supported",
        lambda _partition: None,
    )
    monkeypatch.setattr(
        resource_solver,
        "partition_core_range",
        lambda _partition: (1, 64),
    )
    monkeypatch.setattr(
        resource_solver,
        "partition_memory_per_core_gb",
        lambda _partition: 8.0,
    )
    progress = []
    resolved = resource_solver.resolve_phase_resources(
        phase_name="ARIADNE_ARRAY",
        config=config,
        partition="multicore",
        campaign_dir=campaign,
        iteration=15,
        array_size=2,
        expected_models_version=14,
        submitted_task_ids=[1, 3],
        evidence_override=selected.evidence,
        progress_callback=lambda stage, **payload: progress.append(
            (stage, payload)
        ),
    )

    assert resolved.extra["evidence"]["gradient_dimensions"] == [5, 12]
    assert resolved.extra["evidence"]["submitted_logical_task_ids"] == [1, 3]
    assert [item[0] for item in progress] == ["resource_rules", "resource_rules"]


def test_fresh_attempt_without_resource_records_uses_legacy_calculation(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    active = _intent(
        attempt="attempt-1",
        identity="r0000-a0001-fresh",
        generation=1,
        generation_digest="1" * 64,
    )
    monkeypatch.setattr(reuse, "intent_attempt_records", lambda *_a, **_k: [])
    monkeypatch.setattr(
        reuse,
        "read_active_environment_generation",
        lambda *_a, **_k: pytest.fail(
            "fresh attempt should not inspect environment history"
        ),
    )

    assert (
        reuse.resolve_reusable_ariadne_resource_evidence(
            campaign,
            CampaignConfig(),
            active,
            expected_campaign_uid="campaign-resource-reuse",
            iteration=15,
            replacement_round=0,
            expected_scheduler_kind="slurm",
            expected_models_version=14,
            submitted_task_ids=[0, 1],
        )
        is None
    )


def test_missing_active_bound_record_fails_closed_before_recomputation(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    active = _intent(
        attempt="attempt-1",
        identity="r0000-a0001-fresh",
        generation=1,
        generation_digest="1" * 64,
    )
    active["resource_resolution_path"] = str(
        resolution_path(
            campaign,
            "ARIADNE_ARRAY",
            15,
            active["submission_identity"],
        )
    )
    active["resource_resolution_sha256"] = "a" * 64
    monkeypatch.setattr(reuse, "intent_attempt_records", lambda *_a, **_k: [])

    with pytest.raises(ValueError, match="active bound resource resolution is missing"):
        reuse.resolve_reusable_ariadne_resource_evidence(
            campaign,
            CampaignConfig(),
            active,
            expected_campaign_uid="campaign-resource-reuse",
            iteration=15,
            replacement_round=0,
            expected_scheduler_kind="slurm",
            expected_models_version=14,
            submitted_task_ids=[0, 1],
        )


def test_fresh_ariadne_evidence_reports_each_dimension(
    tmp_path,
    monkeypatch,
):
    from ichor.core.adversarial import geometry, subspace
    from ichor.hpc.active_learning import handoff_manifests, seed_identity
    from ichor.hpc.active_learning.acquisition import trajectory_pool
    from ichor.hpc.active_learning.versioning import trained_models

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    pool_file = campaign / "pool.xyz"
    pool_file.write_text("1\nframe\nH 0 0 0\n", encoding="utf-8")
    task_map_file = campaign / "TASK_MAP.json"
    task_map_file.write_text("{}", encoding="utf-8")
    model_root = campaign / "TRAINED_MODELS" / "iteration-000014"
    model_root.mkdir(parents=True)

    pool_sha256 = sha256_file(pool_file)
    pool_manifest = campaign / ".DATA" / "TRAJECTORY" / "pool.manifest.json"
    pool_manifest.parent.mkdir(parents=True)
    pool_manifest.write_text(
        json.dumps(
            {
                "source_path": str(pool_file),
                "canonical_path": str(pool_file),
                "sha256": pool_sha256,
                "n_frames": 10_000,
                "natoms": 6,
                "atom_types": ["H"] * 6,
                "masses": [1.0] * 6,
                "imported_iso": "2026-07-28T00:00:00+00:00",
                "schema_version": 2,
            }
        ),
        encoding="utf-8",
    )
    task_map = {
        "trajectory_sha256": pool_sha256,
        "models_version": 14,
        "model_manifest_sha256": "a" * 64,
        "model_set_sha256": "b" * 64,
        "n_tasks": 2,
        "tasks": [
            {"pool_row_index_zero_based": 0},
            {"pool_row_index_zero_based": 1},
        ],
    }
    monkeypatch.setattr(
        trajectory_pool.TrajectoryPool,
        "load",
        lambda _campaign: pytest.fail(
            "ARIADNE resource sizing must not materialise the trajectory pool"
        ),
    )
    monkeypatch.setattr(
        handoff_manifests,
        "ariadne_task_map_path",
        lambda _iteration_dir: task_map_file,
    )
    monkeypatch.setattr(
        seed_identity,
        "read_ariadne_task_map",
        lambda *_args, **_kwargs: task_map,
    )
    monkeypatch.setattr(
        trained_models,
        "resolve_trained_model_set",
        lambda *_args, **_kwargs: SimpleNamespace(
            root=model_root,
            head_manifest_sha256="a" * 64,
            model_set_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        geometry,
        "select_local_neighbours",
        lambda *_args, **_kwargs: pytest.fail(
            "ARIADNE resource sizing must not select neighbours"
        ),
    )
    monkeypatch.setattr(
        subspace,
        "build_local_subspace",
        lambda *_args, **_kwargs: pytest.fail(
            "ARIADNE resource sizing must not build local subspaces"
        ),
    )
    progress = []

    evidence = resource_solver._ariadne_evidence(
        campaign,
        15,
        CampaignConfig(),
        progress_callback=lambda stage, **payload: progress.append(
            (stage, payload)
        ),
    )

    assert evidence["gradient_dimensions"] == [6, 6]
    assert evidence["gradient_dimension"] == 6
    assert evidence["gradient_dimension_source"] == "configured_safe_upper_bound"
    assert [
        (stage, payload["completed"], payload["total"])
        for stage, payload in progress
    ] == [
        ("ariadne_task_map_validation", 1, 1),
        ("ariadne_trajectory_pool_validation", 1, 1),
        ("ariadne_current_model_validation", 1, 1),
        ("ariadne_resource_bound", 0, 2),
        ("ariadne_resource_bound", 2, 2),
    ]


def test_snapshot_resource_context_avoids_full_history_and_reuses_file_hashes(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning import handoff_manifests, seed_identity
    from ichor.hpc.active_learning.versioning import trained_models
    from ichor.hpc.active_learning.versioning.reference_data import (
        ReferenceDataVersioning,
    )

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    iteration_dir = active_iteration_dir(campaign, 15)
    task_map_file = ariadne_task_map_path(iteration_dir)
    task_map_file.parent.mkdir(parents=True)
    task_map_file.write_text("{}", encoding="utf-8")
    selection_file = iteration_dir / "seed_selection" / "SELECTION.json"
    selection_file.parent.mkdir(parents=True)
    selection_file.write_text("{}", encoding="utf-8")
    pool = campaign / "pool.xyz"
    pool.write_text("1\nframe\nH 0 0 0\n", encoding="utf-8")
    pool_sha = sha256_file(pool)
    pool_manifest = campaign / ".DATA" / "TRAJECTORY" / "pool.manifest.json"
    pool_manifest.parent.mkdir(parents=True)
    pool_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_path": str(pool),
                "canonical_path": str(pool),
                "sha256": pool_sha,
                "n_frames": 10_000,
                "natoms": 6,
                "atom_types": ["H"] * 6,
                "masses": [1.0] * 6,
                "imported_iso": "2026-07-28T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    model_root = campaign / "TRAINED_MODELS" / "iteration-000014"
    model_root.mkdir(parents=True)
    model_manifest = model_root / "FEREBUS_TASK_ARTEFACTS.json"
    model_manifest.write_text("{}", encoding="utf-8")
    task_map = {
        "campaign_uid": "campaign-resource-reuse",
        "iteration": 15,
        "trajectory_sha256": pool_sha,
        "models_version": 14,
        "model_manifest_sha256": "a" * 64,
        "model_set_sha256": "b" * 64,
        "selection_manifest": {
            "path": selection_file.relative_to(iteration_dir).as_posix(),
            "size": selection_file.stat().st_size,
            "sha256": sha256_file(selection_file),
        },
        "n_tasks": 2,
        "tasks": [
            {"array_task_id": 0, "pool_row_index_zero_based": 0},
            {"array_task_id": 1, "pool_row_index_zero_based": 1},
        ],
    }
    reference = SimpleNamespace(
        campaign_uid="campaign-resource-reuse",
        head_manifest_sha256="c" * 64,
        cumulative_view_sha256="d" * 64,
    )
    model_set = SimpleNamespace(
        version=14,
        campaign_uid="campaign-resource-reuse",
        reference_data_version=14,
        reference_data_head_manifest_sha256="c" * 64,
        reference_data_view_sha256="d" * 64,
        head_manifest_sha256="a" * 64,
        model_set_sha256="b" * 64,
        root=model_root,
    )

    class Snapshot:
        def __init__(self):
            self.anchor_checks = 0

        def assert_anchors_unchanged(self, _campaign):
            self.anchor_checks += 1

        def reference_view(self, version):
            assert version == 14
            return reference

        def model_set(self, version):
            assert version == 14
            return model_set

    snapshot = Snapshot()
    monkeypatch.setattr(
        handoff_manifests,
        "ariadne_task_map_path",
        lambda _iteration_dir: task_map_file,
    )
    monkeypatch.setattr(
        seed_identity,
        "read_ariadne_task_map",
        lambda *_args, **_kwargs: task_map,
    )
    monkeypatch.setattr(
        trained_models,
        "resolve_trained_model_set",
        lambda *_args, **_kwargs: pytest.fail(
            "snapshot resource authority must not resolve model history"
        ),
    )
    monkeypatch.setattr(
        trained_models,
        "_verify_current_model_payloads",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        trained_models,
        "assert_current_model_payloads_unchanged",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        trained_models.TrainedModelVersioning,
        "current_version",
        lambda _self: 14,
    )
    monkeypatch.setattr(
        ReferenceDataVersioning,
        "current_version",
        lambda _self: 14,
    )
    monkeypatch.setattr(
        resource_solver,
        "_manifest_directory_bytes",
        lambda *_args, **_kwargs: 4096,
    )
    original_sha = resource_solver.sha256_file
    hashed = []

    def counted_sha(path, *args, **kwargs):
        hashed.append(Path(path).resolve())
        return original_sha(path, *args, **kwargs)

    monkeypatch.setattr(resource_solver, "sha256_file", counted_sha)
    context = resource_solver.build_ariadne_resource_authority_context(
        campaign,
        15,
        expected_campaign_uid="campaign-resource-reuse",
        expected_models_version=14,
        artifact_snapshot=snapshot,
    )
    evidence = resource_solver._ariadne_evidence(
        campaign,
        15,
        CampaignConfig(),
        authority_context=context,
    )

    assert evidence["model_authority_source"] == "snapshot_current_delta"
    assert evidence["gradient_dimensions"] == [6, 6]
    assert hashed.count(pool.resolve()) == 1
    assert hashed.count(task_map_file.resolve()) == 1
    assert snapshot.anchor_checks == 0


def test_repeated_retry_prefers_original_full_record_over_subset(
    tmp_path,
    monkeypatch,
):
    campaign, config, evidence = _resource_fixture(tmp_path, monkeypatch)
    full = _write_resource_record(
        campaign,
        _intent(
            attempt="attempt-1",
            identity="r0000-a0001-full",
            generation=8,
            generation_digest="8" * 64,
        ),
        evidence,
    )
    subset_evidence = dict(evidence)
    subset_evidence.update(
        {
            "logical_n_tasks": 4,
            "submitted_logical_task_ids": [1, 3],
            "n_tasks": 2,
            "gradient_dimensions": [5, 12],
            "gradient_dimension": 12,
        }
    )
    subset = _write_resource_record(
        campaign,
        _intent(
            attempt="attempt-2",
            identity="r0000-a0002-subset",
            generation=8,
            generation_digest="8" * 64,
        ),
        subset_evidence,
    )
    active = _intent(
        attempt="attempt-3",
        identity="r0000-a0003-retry",
        generation=9,
        generation_digest="9" * 64,
    )
    monkeypatch.setattr(
        reuse,
        "intent_attempt_records",
        lambda *_a, **_k: [subset, full],
    )

    selected = reuse.resolve_reusable_ariadne_resource_evidence(
        campaign,
        config,
        active,
        expected_campaign_uid="campaign-resource-reuse",
        iteration=15,
        replacement_round=0,
        expected_scheduler_kind="slurm",
        expected_models_version=14,
        submitted_task_ids=[1, 3],
    )

    assert selected is not None
    assert selected.source_submission_identity == "r0000-a0001-full"
    assert selected.source_task_count == 4


def test_explicit_full_retry_map_remains_full_coverage(
    tmp_path,
    monkeypatch,
):
    campaign, config, evidence = _resource_fixture(tmp_path, monkeypatch)
    mapped_full_evidence = dict(evidence)
    mapped_full_evidence.update(
        {
            "logical_n_tasks": 4,
            "submitted_logical_task_ids": [0, 1, 2, 3],
        }
    )
    mapped_full = _write_resource_record(
        campaign,
        _intent(
            attempt="attempt-1",
            identity="r0000-a0001-mapped-full",
            generation=8,
            generation_digest="8" * 64,
        ),
        mapped_full_evidence,
    )
    active = _intent(
        attempt="attempt-2",
        identity="r0000-a0002-retry",
        generation=9,
        generation_digest="9" * 64,
    )
    monkeypatch.setattr(
        reuse,
        "intent_attempt_records",
        lambda *_a, **_k: [mapped_full],
    )

    selected = reuse.resolve_reusable_ariadne_resource_evidence(
        campaign,
        config,
        active,
        expected_campaign_uid="campaign-resource-reuse",
        iteration=15,
        replacement_round=0,
        expected_scheduler_kind="slurm",
        expected_models_version=14,
        submitted_task_ids=[1, 3],
    )

    assert selected is not None
    assert selected.source_submission_identity == "r0000-a0001-mapped-full"
    assert selected.already_filtered is False


def test_full_evidence_reuse_supports_sge_attempts(
    tmp_path,
    monkeypatch,
):
    campaign, config, evidence = _resource_fixture(tmp_path, monkeypatch)
    source_intent = _intent(
        attempt="attempt-1",
        identity="r0000-a0001-source",
        generation=8,
        generation_digest="8" * 64,
    )
    source_intent["scheduler_identity_kind"] = "sge"
    source = _write_resource_record(campaign, source_intent, evidence)
    active = _intent(
        attempt="attempt-2",
        identity="r0000-a0002-retry",
        generation=9,
        generation_digest="9" * 64,
    )
    active["scheduler_identity_kind"] = "sge"
    monkeypatch.setattr(reuse, "intent_attempt_records", lambda *_a, **_k: [source])

    selected = reuse.resolve_reusable_ariadne_resource_evidence(
        campaign,
        config,
        active,
        expected_campaign_uid="campaign-resource-reuse",
        iteration=15,
        replacement_round=0,
        expected_scheduler_kind="sge",
        expected_models_version=14,
        submitted_task_ids=[1, 3],
    )

    assert selected is not None
    assert selected.source_submission_identity == "r0000-a0001-source"


def test_current_attempt_resource_record_is_reused_after_crash(
    tmp_path,
    monkeypatch,
):
    campaign, config, evidence = _resource_fixture(tmp_path, monkeypatch)
    filtered = dict(evidence)
    filtered.update(
        {
            "logical_n_tasks": 4,
            "submitted_logical_task_ids": [1, 3],
            "n_tasks": 2,
            "gradient_dimensions": [5, 12],
            "gradient_dimension": 12,
        }
    )
    active = _write_resource_record(
        campaign,
        _intent(
            attempt="attempt-2",
            identity="r0000-a0002-retry",
            generation=8,
            generation_digest="8" * 64,
        ),
        filtered,
    )
    monkeypatch.setattr(reuse, "intent_attempt_records", lambda *_a, **_k: [])

    selected = reuse.resolve_reusable_ariadne_resource_evidence(
        campaign,
        config,
        active,
        expected_campaign_uid="campaign-resource-reuse",
        iteration=15,
        replacement_round=0,
        expected_scheduler_kind="slurm",
        expected_models_version=14,
        submitted_task_ids=[1, 3],
    )

    assert selected is not None
    assert selected.source_submission_identity == "r0000-a0002-retry"
    assert selected.already_filtered is True


def test_unbound_current_attempt_record_is_adopted_after_write_crash(
    tmp_path,
    monkeypatch,
):
    campaign, config, evidence = _resource_fixture(tmp_path, monkeypatch)
    filtered = dict(evidence)
    filtered.update(
        {
            "logical_n_tasks": 4,
            "submitted_logical_task_ids": [1, 3],
            "n_tasks": 2,
            "gradient_dimensions": [5, 12],
            "gradient_dimension": 12,
        }
    )
    unbound = _intent(
        attempt="attempt-2",
        identity="r0000-a0002-retry",
        generation=8,
        generation_digest="8" * 64,
    )
    _write_resource_record(campaign, unbound, filtered)
    monkeypatch.setattr(reuse, "intent_attempt_records", lambda *_a, **_k: [])

    selected = reuse.resolve_reusable_ariadne_resource_evidence(
        campaign,
        config,
        unbound,
        expected_campaign_uid="campaign-resource-reuse",
        iteration=15,
        replacement_round=0,
        expected_scheduler_kind="slurm",
        expected_models_version=14,
        submitted_task_ids=[1, 3],
    )

    assert selected is not None
    assert selected.source_submission_identity == "r0000-a0002-retry"
    assert selected.already_filtered is True


def test_changed_resource_producer_code_falls_back_to_exact_calculation(
    tmp_path,
    monkeypatch,
):
    campaign, config, evidence = _resource_fixture(tmp_path, monkeypatch)
    source = _write_resource_record(
        campaign,
        _intent(
            attempt="attempt-1",
            identity="r0000-a0001-source",
            generation=8,
            generation_digest="8" * 64,
        ),
        evidence,
    )
    active = _intent(
        attempt="attempt-2",
        identity="r0000-a0002-retry",
        generation=9,
        generation_digest="9" * 64,
    )
    monkeypatch.setattr(reuse, "intent_attempt_records", lambda *_a, **_k: [source])
    monkeypatch.setattr(
        reuse,
        "assess_resource_evidence_code_equivalence",
        lambda *_a, **_k: {"equivalent": False},
    )

    assert (
        reuse.resolve_reusable_ariadne_resource_evidence(
            campaign,
            config,
            active,
            expected_campaign_uid="campaign-resource-reuse",
            iteration=15,
            replacement_round=0,
            expected_scheduler_kind="slurm",
            expected_models_version=14,
            submitted_task_ids=[1, 3],
        )
        is None
    )


def test_changed_subspace_configuration_falls_back_to_exact_calculation(
    tmp_path,
    monkeypatch,
):
    campaign, config, evidence = _resource_fixture(tmp_path, monkeypatch)
    source = _write_resource_record(
        campaign,
        _intent(
            attempt="attempt-1",
            identity="r0000-a0001-source",
            generation=8,
            generation_digest="8" * 64,
        ),
        evidence,
    )
    active = _intent(
        attempt="attempt-2",
        identity="r0000-a0002-retry",
        generation=9,
        generation_digest="9" * 64,
    )
    config.acquisition.subspace.neighbour_count += 1
    monkeypatch.setattr(reuse, "intent_attempt_records", lambda *_a, **_k: [source])

    assert (
        reuse.resolve_reusable_ariadne_resource_evidence(
            campaign,
            config,
            active,
            expected_campaign_uid="campaign-resource-reuse",
            iteration=15,
            replacement_round=0,
            expected_scheduler_kind="slurm",
            expected_models_version=14,
            submitted_task_ids=[1, 3],
        )
        is None
    )


def test_bound_resource_digest_contradiction_fails_closed(
    tmp_path,
    monkeypatch,
):
    campaign, config, evidence = _resource_fixture(tmp_path, monkeypatch)
    source = _write_resource_record(
        campaign,
        _intent(
            attempt="attempt-1",
            identity="r0000-a0001-source",
            generation=8,
            generation_digest="8" * 64,
        ),
        evidence,
    )
    source["resource_resolution_sha256"] = "f" * 64
    active = _intent(
        attempt="attempt-2",
        identity="r0000-a0002-retry",
        generation=9,
        generation_digest="9" * 64,
    )
    monkeypatch.setattr(reuse, "intent_attempt_records", lambda *_a, **_k: [source])

    with pytest.raises(ValueError, match="digest contradicts"):
        reuse.resolve_reusable_ariadne_resource_evidence(
            campaign,
            config,
            active,
            expected_campaign_uid="campaign-resource-reuse",
            iteration=15,
            replacement_round=0,
            expected_scheduler_kind="slurm",
            expected_models_version=14,
            submitted_task_ids=[1, 3],
        )


def test_historical_config_is_resolved_from_authenticated_history(tmp_path):
    original = CampaignConfig()
    original.resources.partition = "multicore_small"
    write_config_lock(
        tmp_path,
        original,
        campaign_uid="campaign-resource-reuse",
    )
    fingerprint = config_fingerprint(canonical_config(original))
    current = CampaignConfig()
    current.resources.partition = "multicore"
    write_config_lock(
        tmp_path,
        current,
        campaign_uid="campaign-resource-reuse",
    )

    restored = read_historical_config_by_fingerprint(
        tmp_path,
        fingerprint,
        expected_campaign_uid="campaign-resource-reuse",
    )

    assert restored.resources.partition == "multicore_small"
    assert config_fingerprint(canonical_config(restored)) == fingerprint


def test_legacy_lock_can_resolve_its_current_config_without_history(
    tmp_path,
    monkeypatch,
):
    config = CampaignConfig()
    canonical = canonical_config(config)
    fingerprint = config_fingerprint(canonical)
    monkeypatch.setattr(
        config_lock_module,
        "read_config_lock",
        lambda *_args, **_kwargs: {"canonical_config": canonical},
    )

    restored = read_historical_config_by_fingerprint(
        tmp_path,
        fingerprint,
        expected_campaign_uid="campaign-resource-reuse",
    )

    assert config_fingerprint(canonical_config(restored)) == fingerprint


def test_live_retry_submission_filters_reused_evidence_before_rendering(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon.live_executor import (
        LiveBackendsPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.resource_records import read_resolution
    from ichor.hpc.active_learning.daemon.submission_intent import (
        load_active_intent,
        write_pre_submit_intent,
    )

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = CampaignConfig()
    config.resources.partition = "multicore"
    retry_ids = list(range(4, 200))
    retry_map = campaign / "retry_tasks.txt"
    retry_map.write_text(
        "".join(str(task_id) + "\n" for task_id in retry_ids),
        encoding="utf-8",
    )
    write_pre_submit_intent(
        campaign,
        campaign_uid="campaign-resource-reuse",
        phase_name="ARIADNE_ARRAY",
        iteration=15,
        expected_tasks=len(retry_ids),
        scheduler_identity_kind="slurm",
    )
    dimensions = [2 + (index % 5) for index in range(200)]
    source = reuse.ReusableAriadneResourceEvidence(
        evidence={
            "source": "ariadne_task_map_and_model_set",
            "n_tasks": 200,
            "n_atoms": 6,
            "gradient_dimension": max(dimensions),
            "gradient_dimensions": dimensions,
            "model_bytes": 1024,
            "models_version": 14,
        },
        source_submission_identity="r0000-a0001-source",
        source_attempt_id="attempt-1",
        source_resolution_path="/campaign/source-resolution.json",
        source_resolution_sha256="a" * 64,
        source_task_count=200,
        already_filtered=False,
    )
    monkeypatch.setattr(
        reuse,
        "resolve_reusable_ariadne_resource_evidence",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        resource_solver,
        "collect_resource_evidence",
        lambda **_kwargs: pytest.fail("fresh ARIADNE evidence was recomputed"),
    )
    monkeypatch.setattr(
        resource_solver,
        "validate_partition_supported",
        lambda _partition: None,
    )
    monkeypatch.setattr(
        resource_solver,
        "partition_core_range",
        lambda _partition: (1, 64),
    )
    monkeypatch.setattr(
        resource_solver,
        "partition_memory_per_core_gb",
        lambda _partition: 8.0,
    )
    executor = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=config,
        backend_check=False,
    )
    progress = []
    journal = []
    monkeypatch.setattr(
        executor,
        "_report_runtime_progress",
        lambda stage, **payload: progress.append((stage, payload)),
    )
    monkeypatch.setattr(
        executor,
        "_journal_event",
        lambda event, **payload: journal.append((event, payload)),
    )

    script = executor._write_real_script(
        "ARIADNE_ARRAY",
        SimpleNamespace(
            iteration=15,
            campaign_uid="campaign-resource-reuse",
            replacement_round=0,
            models_version=14,
        ),
        array_size=len(retry_ids),
        array_task_map=retry_map,
    )

    assert script.is_file()
    intent = load_active_intent(
        campaign,
        "ARIADNE_ARRAY",
        15,
        expected_campaign_uid="campaign-resource-reuse",
    )
    record = read_resolution(intent["resource_resolution_path"])
    selected = record["evidence"]
    assert record["resources"]["partition"] == "multicore"
    assert selected["logical_n_tasks"] == 200
    assert selected["submitted_logical_task_ids"] == retry_ids
    assert selected["gradient_dimensions"] == dimensions[4:]
    assert selected["resource_evidence_reuse"] == {
        "schema_version": 1,
        "fingerprint_algorithm": "repository_module_closure_v2",
        "equivalence_basis": "same_environment_generation",
        "source_submission_identity": "r0000-a0001-source",
        "source_attempt_id": "attempt-1",
        "source_resolution_sha256": "a" * 64,
    }
    assert progress[0] == (
        "ariadne_resource_reuse",
        {
            "completed": 196,
            "total": 196,
            "unit": "retry tasks",
            "source_submission_identity": "r0000-a0001-source",
            "source_attempt_id": "attempt-1",
        },
    )
    event = next(
        payload
        for name, payload in journal
        if name == "resolved_phase_resources"
    )
    assert event["resource_evidence_mode"] == "reused"
    assert event["resource_evidence_source_tasks"] == 200
    assert event["resource_evidence_source_resolution_sha256"] == "a" * 64
    assert (
        event["resource_evidence_fingerprint_algorithm"]
        == "repository_module_closure_v2"
    )
