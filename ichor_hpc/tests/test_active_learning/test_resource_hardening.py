from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon import resource_solver
from ichor.hpc.active_learning.daemon.resource_records import (
    read_resolution,
    resolution_payload,
    verify_resolution,
    write_resolution,
)
from ichor.hpc.active_learning.daemon.resource_usage import (
    collect_usage,
    parse_usage_rows,
    usage_path,
)
from ichor.hpc.active_learning.daemon.scratch import (
    attempt_scratch_root,
    clean_inactive_attempts,
    finish_task_scratch,
    inventory,
    prepare_task_scratch,
)
from ichor.hpc.active_learning.daemon.script_bundles import (
    prepare_attempt_bundle,
    read_array_task_map,
    slurm_log_paths,
    write_attempt_script,
    write_script_binding,
)
from ichor.hpc.active_learning.daemon.state import atomic_write_json
from ichor.hpc.active_learning.sampling.descriptors import (
    MassWeightedRMSDDescriptor,
    build_condensed_distance_store,
)
from ichor.hpc.active_learning.sampling.polus_wrapper import fps_select
from ichor.core.atoms import Atom, Atoms


@pytest.fixture
def resource_profile(monkeypatch):
    monkeypatch.setattr(resource_solver, "validate_partition_supported", lambda _p: None)
    monkeypatch.setattr(resource_solver, "partition_core_range", lambda _p: (1, 64))
    monkeypatch.setattr(
        resource_solver,
        "partition_memory_per_core_gb",
        lambda _p: 8.0,
    )


def _write_pool(source: Path, n_frames: int) -> None:
    lines = []
    for index in range(int(n_frames)):
        lines.extend(
            [
                "2",
                "frame " + str(index),
                "H 0 0 0",
                "H 0 0 " + str(0.7 + 0.01 * index),
            ]
        )
    source.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_empty_committed_bootstrap(campaign: Path) -> None:
    root = campaign / ".DATA" / "ACTIVE_LEARNING" / "bootstrap_inputs"
    root.mkdir(parents=True)
    embedded = {
        "schema_version": 1,
        "confirmed": True,
        "excluded_pool_frame_ids": [],
        "sources": {},
        "model": None,
    }
    atomic_write_json(root / "CUSTOM_BOOTSTRAP.json", embedded)
    pointer = dict(embedded)
    pointer["bootstrap_inputs_root"] = (
        root.resolve().relative_to(campaign.resolve()).as_posix()
    )
    active = campaign / ".DATA" / "ACTIVE_LEARNING"
    atomic_write_json(active / "CUSTOM_BOOTSTRAP.json", pointer)


def test_phase_a_uses_manifest_verified_root_pool_not_data_copy(
    tmp_path,
    resource_profile,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    source = tmp_path / "source.xyz"
    _write_pool(source, 3)
    TrajectoryPool.import_from(source, campaign)
    _write_empty_committed_bootstrap(campaign)
    stale = campaign / ".DATA" / "TRAJECTORY" / "pool.xyz"
    _write_pool(stale, 20)

    resolved = resource_solver.resolve_phase_resources(
        phase_name="PHASE_A_POLUS",
        config=CampaignConfig(),
        partition="multicore",
        campaign_dir=campaign,
        require_evidence=True,
    )

    evidence = resolved.extra["evidence"]
    assert evidence["n_frames"] == 3
    assert resolved.extra["n_pairs"] == 3
    assert Path(evidence["pool"]["path"]) == (campaign / "pool.xyz").resolve()


def test_polus_ten_thousand_frame_formula_resolves_ten_workers(resource_profile):
    evidence = {
        "source": "test_exact_dimensions",
        "n_frames": 10_000,
        "n_atoms": 3,
        "atom_order": ["O", "H", "H"],
    }
    resolved = resource_solver.resolve_phase_resources(
        phase_name="PHASE_A_POLUS",
        config=CampaignConfig(),
        partition="multicore",
        campaign_dir=None,
        require_evidence=False,
        evidence_override=evidence,
    )

    assert resolved.extra["n_pairs"] == 49_995_000
    assert resolved.extra["active_workers"] == 10
    assert resolved.cpus_per_task == 10
    assert resolved.extra["condensed_store_bytes"] == 8 * 49_995_000
    assert resolved.extra["profile_limits"] == {
        "partition_min_cpus": 1,
        "partition_max_cpus": 64,
        "partition_memory_per_core_gb": 8.0,
    }


def test_polus_large_store_falls_back_to_campaign_scratch(
    monkeypatch,
    resource_profile,
):
    monkeypatch.setattr(
        resource_solver.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=10**15),
    )
    resolved = resource_solver.resolve_phase_resources(
        phase_name="PHASE_A_POLUS",
        config=CampaignConfig(),
        partition="multicore",
        campaign_dir=Path.cwd(),
        require_evidence=True,
        evidence_override={
            "source": "test_exact_dimensions",
            "n_frames": 120_000,
            "n_atoms": 3,
            "atom_order": ["O", "H", "H"],
        },
    )
    assert resolved.extra["distance_store_mode"] == "file"
    assert resolved.extra["expected_scratch_bytes"] == resolved.extra[
        "condensed_store_bytes"
    ]
    assert resolved.extra["scratch_requirement_exact"] is True
    assert resolved.extra["campaign_filesystem"][
        "free_bytes_at_resolution"
    ] == 10**15
    assert resolved.extra["scratch_required_bytes"] == pytest.approx(
        1.25 * resolved.extra["condensed_store_bytes"], abs=1
    )


@pytest.mark.parametrize(
    "phase",
    [
        "PHASE_A_POLUS",
        "PHASE_B_POLUS",
        "INITIAL_GAUSSIAN",
        "INITIAL_AIMALL",
        "ARIADNE_ARRAY",
        "INITIAL_FEREBUS",
    ],
)
def test_live_resource_resolution_rejects_missing_backend_evidence(
    tmp_path,
    resource_profile,
    phase,
):
    with pytest.raises(resource_solver.ResourceEvidenceUnavailable):
        resource_solver.resolve_phase_resources(
            phase_name=phase,
            config=CampaignConfig(),
            partition="multicore",
            campaign_dir=tmp_path,
            iteration=1 if phase in {"PHASE_B_POLUS", "ARIADNE_ARRAY"} else 0,
            require_evidence=True,
        )


def test_ariadne_adds_memory_only_cpus_without_more_workers(
    monkeypatch,
    resource_profile,
):
    monkeypatch.setattr(
        resource_solver,
        "partition_memory_per_core_gb",
        lambda _p: 4.0,
    )
    cfg = CampaignConfig()
    cfg.acquisition.gradient.mode = "active_fd"
    cfg.acquisition.subspace.max_subspace_dim = 6
    evidence = {
        "source": "test_exact_dimensions",
        "n_tasks": 2,
        "n_atoms": 6,
        "gradient_dimension": 6,
        "model_bytes": 10 * 1024**3,
    }

    resolved = resource_solver.resolve_phase_resources(
        phase_name="ARIADNE_ARRAY",
        config=cfg,
        partition="multicore",
        campaign_dir=None,
        array_size=2,
        require_evidence=False,
        evidence_override=evidence,
    )

    assert resolved.extra["active_workers"] == 6
    assert resolved.cpus_per_task == 14
    assert resolved.extra["memory_only_cpus"] == 8


def test_ariadne_resource_evidence_is_bound_to_state_model_version(
    resource_profile,
):
    evidence = {
        "source": "test_exact_dimensions",
        "n_tasks": 1,
        "n_atoms": 3,
        "gradient_dimension": 6,
        "model_bytes": 1024,
        "models_version": 4,
    }

    with pytest.raises(
        resource_solver.BackendSubmissionError,
        match="does not match daemon state.models_version 3",
    ):
        resource_solver.resolve_phase_resources(
            phase_name="ARIADNE_ARRAY",
            config=CampaignConfig(),
            partition="multicore",
            campaign_dir=None,
            array_size=1,
            expected_models_version=3,
            require_evidence=False,
            evidence_override=evidence,
        )


def test_partial_array_resources_use_validated_retry_dimensions(
    resource_profile,
):
    cfg = CampaignConfig()
    evidence = {
        "source": "test_exact_dimensions",
        "n_tasks": 4,
        "n_atoms": 3,
        "gradient_dimension": 12,
        "gradient_dimensions": [2, 5, 7, 12],
        "model_bytes": 1024,
        "models_version": 3,
    }

    resolved = resource_solver.resolve_phase_resources(
        phase_name="ARIADNE_ARRAY",
        config=cfg,
        partition="multicore",
        array_size=2,
        expected_models_version=3,
        submitted_task_ids=[1, 2],
        evidence_override=evidence,
    )

    selected = resolved.extra["evidence"]
    assert selected["logical_n_tasks"] == 4
    assert selected["n_tasks"] == 2
    assert selected["submitted_logical_task_ids"] == [1, 2]
    assert selected["gradient_dimensions"] == [5, 7]
    assert selected["gradient_dimension"] == 7

    with pytest.raises(
        resource_solver.BackendSubmissionError,
        match="outside producer evidence",
    ):
        resource_solver.resolve_phase_resources(
            phase_name="ARIADNE_ARRAY",
            config=cfg,
            partition="multicore",
            array_size=1,
            expected_models_version=3,
            submitted_task_ids=[4],
            evidence_override=evidence,
        )


def test_aimall_memory_only_cpus_do_not_increase_snapshotted_naat(
    monkeypatch,
    resource_profile,
):
    monkeypatch.setattr(
        resource_solver,
        "partition_memory_per_core_gb",
        lambda _p: 1.0,
    )
    cfg = CampaignConfig()
    evidence = {
        "source": "test_exact_dimensions",
        "n_tasks": 1,
        "max_n_atoms": 20,
        "max_n_primitives": 1000,
        "atom_orders": [["X"] * 20],
        "primitive_counts": [1000],
    }
    resolved = resource_solver.resolve_phase_resources(
        phase_name="AIMALL",
        config=cfg,
        partition="multicore",
        campaign_dir=None,
        array_size=1,
        require_evidence=False,
        evidence_override=evidence,
    )
    assert resolved.extra["aimall_regime"] == "medium"
    assert resolved.extra["aimall_task_naat"] == [6]
    assert resolved.extra["active_workers"] == 6
    assert resolved.cpus_per_task > 12
    assert resolved.extra["memory_only_cpus"] == resolved.cpus_per_task - 6


def test_aimall_partial_retry_honours_frozen_naat(resource_profile):
    cfg = CampaignConfig()
    cfg.resources.aimall.cpus_per_task = 4
    evidence = {
        "source": "test_exact_dimensions",
        "n_tasks": 2,
        "max_n_atoms": 10,
        "max_n_primitives": 300,
        "atom_orders": [["X"] * 10, ["X"] * 10],
        "primitive_counts": [300, 300],
        "aimall_task_naat": [2, 6],
    }

    with pytest.raises(
        resource_solver.BackendSubmissionError,
        match="frozen naat=6 exceeds",
    ):
        resource_solver.resolve_phase_resources(
            phase_name="AIMALL",
            config=cfg,
            partition="multicore",
            array_size=1,
            submitted_task_ids=[1],
            evidence_override=evidence,
        )


def test_ferebus_formula_uses_exact_split_and_feature_dimensions(resource_profile):
    cfg = CampaignConfig()
    cfg.ferebus.nagents = 4
    evidence = {
        "source": "test_exact_dimensions",
        "n_tasks": 3,
        "max_train_rows": 100,
        "max_internal_rows": 20,
        "max_external_rows": 30,
        "max_total_rows": 150,
        "max_features": 40,
    }
    resolved = resource_solver.resolve_phase_resources(
        phase_name="FEREBUS",
        config=cfg,
        partition="multicore",
        campaign_dir=None,
        require_evidence=False,
        evidence_override=evidence,
    )
    components = resolved.extra["ferebus_memory_components"]
    expected_distance = 8 * 40 * 100**2
    assert components["distance_tensor_bytes"] == expected_distance
    assert components["distance_construction_peak_bytes"] == 2 * expected_distance
    assert resolved.extra["working_peak_bytes"] == max(
        components["distance_construction_peak_bytes"],
        components["distance_tensor_bytes"]
        + components["threaded_estimator_bytes"]
        + components["validation_kernel_bytes"]
        + components["dataset_bytes"],
    )
    assert resolved.extra["active_workers"] == 4


def test_ferebus_uses_one_real_most_demanding_task(resource_profile):
    cfg = CampaignConfig()
    cfg.ferebus.nagents = 4
    evidence = {
        "source": "test_exact_dimensions",
        "n_tasks": 2,
        "max_train_rows": 100,
        "max_internal_rows": 100,
        "max_external_rows": 100,
        "max_total_rows": 280,
        "max_features": 100,
        "task_dimensions": [
            {
                "task_index": 1,
                "property": "iqa",
                "atom": "O1",
                "n_train": 100,
                "n_internal": 0,
                "n_external": 0,
                "n_total": 100,
                "n_features": 1,
            },
            {
                "task_index": 2,
                "property": "iqa",
                "atom": "H2",
                "n_train": 80,
                "n_internal": 100,
                "n_external": 100,
                "n_total": 280,
                "n_features": 100,
            },
        ],
    }

    resolved = resource_solver.resolve_phase_resources(
        phase_name="FEREBUS",
        config=cfg,
        partition="multicore",
        evidence_override=evidence,
    )

    components = resolved.extra["ferebus_memory_components"]
    assert components["distance_tensor_bytes"] == 8 * 100 * 80**2
    assert resolved.extra["working_peak_bytes"] == components["working_peak_bytes"]
    assert resolved.extra["most_demanding_task"]["task_index"] == 2


def test_explicit_gaussian_memory_defines_gauss_mdef_allocation(resource_profile):
    cfg = CampaignConfig()
    cfg.resources.gaussian.cpus_per_task = 2
    cfg.resources.gaussian.mem_per_cpu = "4G"
    evidence = {
        "source": "test_exact_dimensions",
        "n_tasks": 1,
        "max_n_atoms": 3,
        "atom_orders": [["O", "H", "H"]],
    }

    resolved = resource_solver.resolve_phase_resources(
        phase_name="GAUSSIAN",
        config=cfg,
        partition="multicore",
        array_size=1,
        evidence_override=evidence,
    )

    assert resolved.mem_per_cpu == "4G"
    assert resolved.estimated_total_memory_gb == pytest.approx(8.0)
    assert resolved.memory_reason == "gaussian_explicit_allocation_for_gauss_mdef"


def test_gaussian_partial_retry_honours_frozen_link0_cpu_count(resource_profile):
    cfg = CampaignConfig()
    cfg.resources.gaussian.cpus_per_task = 4
    cfg.resources.gaussian.mem_per_cpu = "4G"
    evidence = {
        "source": "test_exact_dimensions",
        "n_tasks": 2,
        "max_n_atoms": 3,
        "atom_orders": [["O", "H", "H"], ["O", "H", "H"]],
        "gaussian_link0_nproc": [8, 4],
    }

    with pytest.raises(
        resource_solver.BackendSubmissionError,
        match="frozen %NProcShared=8 exceeds",
    ):
        resource_solver.resolve_phase_resources(
            phase_name="GAUSSIAN",
            config=cfg,
            partition="multicore",
            array_size=1,
            submitted_task_ids=[0],
            evidence_override=evidence,
        )


def test_attempt_bundle_caps_each_log_directory_and_records_retry_map(tmp_path):
    campaign = tmp_path / "campaign"
    source_map = tmp_path / "retry.txt"
    source_map.write_text("7\n11\n", encoding="utf-8", newline="\n")
    bundle = prepare_attempt_bundle(
        campaign,
        "GAUSSIAN",
        2,
        "r0000-a0001-deadbeef",
        array_size=2,
        max_log_files_per_directory=2,
        source_array_task_map=source_map,
    )
    assert read_array_task_map(bundle.array_task_map) == [7, 11]
    paths = slurm_log_paths(bundle, is_array=True)
    assert paths["output"].endswith("OUTPUTS/%A_%a.o")
    assert paths["error"].endswith("ERRORS/%A_%a.e")
    with pytest.raises(ValueError, match="exceeding"):
        prepare_attempt_bundle(
            campaign,
            "GAUSSIAN",
            2,
            "r0000-a0002-feedface",
            array_size=3,
            max_log_files_per_directory=2,
        )

    accepted_limit = prepare_attempt_bundle(
        campaign,
        "AIMALL",
        2,
        "r0000-a0003-cafebabe",
        array_size=5000,
        max_log_files_per_directory=5000,
    )
    assert accepted_limit.outputs.is_dir()
    assert accepted_limit.errors.is_dir()


def test_resource_resolution_is_immutable_and_digest_verified(tmp_path):
    resolved = SimpleNamespace(
        to_dict=lambda: {
            "backend": "POLUS",
            "cpus_per_task": 2,
            "mem_per_cpu": "4G",
        }
    )
    payload = resolution_payload(
        campaign_uid="uid",
        phase_name="PHASE_A_POLUS",
        iteration=0,
        attempt_id="attempt",
        submission_identity="r0000-a0001-deadbeef",
        resolved=resolved,
        evidence={"source": "fixture", "sha256": "a" * 64},
        scratch_path_template="template",
    )
    binding = write_resolution(tmp_path, payload)
    verified = verify_resolution(binding["path"], binding["sha256"])
    assert verified == read_resolution(binding["path"])
    assert verified["implementation_identity"]["backend"] == "polus"
    changed = dict(payload)
    changed["resources"] = {"backend": "POLUS", "cpus_per_task": 3}
    with pytest.raises(ValueError, match="different content"):
        write_resolution(tmp_path, changed)
    assert read_resolution(binding["path"])["attempt_id"] == "attempt"


def _scratch_resolution(
    campaign,
    *,
    phase,
    iteration,
    attempt_id,
    identity,
):
    payload = resolution_payload(
        campaign_uid="uid",
        phase_name=phase,
        iteration=iteration,
        attempt_id=attempt_id,
        submission_identity=identity,
        resolved=SimpleNamespace(
            to_dict=lambda: {
                "backend": phase.split("_")[0],
                "cpus_per_task": 1,
                "mem_per_cpu": "1G",
            }
        ),
        evidence={"source": "fixture"},
        scratch_path_template="fixture",
    )
    resolution = write_resolution(campaign, payload)
    bundle = prepare_attempt_bundle(
        campaign,
        phase,
        iteration,
        identity,
        array_size=1,
        max_log_files_per_directory=10,
    )
    write_attempt_script(bundle, "#!/bin/bash\ntrue\n")
    resolution["script_binding"] = write_script_binding(bundle)
    return resolution


def test_scratch_failure_retains_then_explicit_cleanup_removes_attempt(tmp_path):
    identity = "r0000-a0001-deadbeef"
    binding = _scratch_resolution(
        tmp_path,
        phase="GAUSSIAN",
        iteration=1,
        attempt_id="attempt",
        identity=identity,
    )
    leaf = prepare_task_scratch(
        tmp_path,
        campaign_uid="uid",
        phase_name="GAUSSIAN",
        iteration=1,
        attempt_id="attempt",
        submission_identity=identity,
        job_id="1234",
        array_task_id=0,
        resource_resolution_path=binding["path"],
        resource_resolution_sha256=binding["sha256"],
        script_binding_path=binding["script_binding"]["path"],
        script_binding_sha256=binding["script_binding"]["sha256"],
    )
    finish_task_scratch(leaf, success=False)
    records = inventory(tmp_path)
    assert records[0]["status"] == "failed_retained"
    assert clean_inactive_attempts(
        tmp_path,
        active_job_ids=[],
        allowed_attempts=["attempt"],
    ) == [str(leaf.parent.parent.resolve())]
    assert not leaf.exists()


def test_scratch_cleanup_never_removes_active_job(tmp_path):
    identity = "r0000-a0001-deadbeef"
    binding = _scratch_resolution(
        tmp_path,
        phase="AIMALL",
        iteration=1,
        attempt_id="attempt",
        identity=identity,
    )
    leaf = prepare_task_scratch(
        tmp_path,
        campaign_uid="uid",
        phase_name="AIMALL",
        iteration=1,
        attempt_id="attempt",
        submission_identity=identity,
        job_id="777",
        array_task_id=2,
        resource_resolution_path=binding["path"],
        resource_resolution_sha256=binding["sha256"],
        script_binding_path=binding["script_binding"]["path"],
        script_binding_sha256=binding["script_binding"]["sha256"],
    )
    assert clean_inactive_attempts(
        tmp_path,
        active_job_ids=["777"],
    ) == []
    assert leaf.is_dir()


def test_successful_scratch_task_self_cleans(tmp_path):
    identity = "r0000-a0001-deadbeef"
    binding = _scratch_resolution(
        tmp_path,
        phase="ARIADNE_ARRAY",
        iteration=1,
        attempt_id="attempt",
        identity=identity,
    )
    leaf = prepare_task_scratch(
        tmp_path,
        campaign_uid="uid",
        phase_name="ARIADNE_ARRAY",
        iteration=1,
        attempt_id="attempt",
        submission_identity=identity,
        job_id="778",
        array_task_id=4,
        resource_resolution_path=binding["path"],
        resource_resolution_sha256=binding["sha256"],
        script_binding_path=binding["script_binding"]["path"],
        script_binding_sha256=binding["script_binding"]["sha256"],
    )
    finish_task_scratch(leaf, success=True)
    assert not leaf.exists()


def test_scratch_inventory_blocks_tampered_resolution_ownership(tmp_path):
    identity = "r0000-a0001-deadbeef"
    binding = _scratch_resolution(
        tmp_path,
        phase="GAUSSIAN",
        iteration=1,
        attempt_id="attempt",
        identity=identity,
    )
    leaf = prepare_task_scratch(
        tmp_path,
        campaign_uid="uid",
        phase_name="GAUSSIAN",
        iteration=1,
        attempt_id="attempt",
        submission_identity=identity,
        job_id="779",
        array_task_id=0,
        resource_resolution_path=binding["path"],
        resource_resolution_sha256=binding["sha256"],
        script_binding_path=binding["script_binding"]["path"],
        script_binding_sha256=binding["script_binding"]["sha256"],
    )
    task_path = leaf / "TASK.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["resource_resolution_sha256"] = "f" * 64
    atomic_write_json(task_path, task)
    records = inventory(tmp_path)
    assert records[0]["status"] == "invalid"
    with pytest.raises(ValueError, match="invalid scratch evidence"):
        clean_inactive_attempts(tmp_path, active_job_ids=[])


def test_scratch_rejects_symlinked_path_component(tmp_path):
    campaign = tmp_path / "campaign"
    real = campaign / ".DATA" / "REAL_SCRATCH"
    real.mkdir(parents=True)
    link = campaign / ".DATA" / "SCRATCH"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this test host")
    with pytest.raises(ValueError, match="unsafe campaign scratch path"):
        attempt_scratch_root(
            campaign,
            "GAUSSIAN",
            1,
            "r0000-a0001-deadbeef",
        )


def test_telemetry_parses_aggregates_is_idempotent_and_bounded(tmp_path):
    provisional_stdout = (
        "100|COMPLETED|0:0|21|4|8Gc|4G|5G|00:00:20|\n"
        "100_0|COMPLETED|0:0|10|2|8Gc|||00:00:09|\n"
        "100_0.batch|COMPLETED|0:0|10|2|8Gc|9G|10G|00:00:09|\n"
        "100_1|FAILED|1:0|20|2|8Gc|3G|4G|00:00:18|\n"
    )
    final_stdout = provisional_stdout + (
        "100_2|COMPLETED|0:0|12|2|8Gc|2G|3G|00:00:11|\n"
    )
    assert len(parse_usage_rows(provisional_stdout)) == 4
    calls = []

    def runner(*_args, **_kwargs):
        calls.append(1)
        stdout = provisional_stdout if len(calls) == 1 else final_stdout
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    intent = {
        "attempt_id": "a1",
        "submission_identity": "r0000-a0001-deadbeef",
        "phase": "GAUSSIAN",
        "iteration": 1,
        "job_id": "100",
        "expected_tasks": 3,
    }
    first = collect_usage(tmp_path, intent=intent, history_limit=1, runner=runner)
    second = collect_usage(tmp_path, intent=intent, history_limit=1, runner=runner)
    third = collect_usage(tmp_path, intent=intent, history_limit=1, runner=runner)
    assert second == third
    assert len(calls) == 2
    assert first["telemetry_status"] == "provisional"
    assert second["telemetry_status"] == "final"
    assert second["collection_sequence"] == 2
    assert first["n_failures"] == 1
    # Slurm places the task's peak RSS on its .batch step.  It must enrich the
    # owning task without increasing the scientific task count.
    assert first["max_rss_mib"] == 9 * 1024
    assert first["n_rows"] == 2
    assert first["n_missing_task_rows"] == 1
    assert second["n_missing_task_rows"] == 0
    payload = json.loads(usage_path(tmp_path).read_text(encoding="utf-8"))
    assert len(payload["attempts"]) == 1


def _frame(distance: float) -> Atoms:
    return Atoms(
        [
            Atom("H", 0.0, 0.0, 0.0),
            Atom("H", 0.0, 0.0, float(distance)),
        ]
    )


def test_condensed_polus_matches_dense_fps_without_dense_builder(monkeypatch):
    frames = [_frame(0.7), _frame(0.9), _frame(1.2), _frame(1.6)]
    descriptor = MassWeightedRMSDDescriptor()
    dense = descriptor.pairwise_distance_matrix(frames)
    expected = fps_select(dense, 3, descriptor_name=descriptor.name)
    monkeypatch.setattr(
        descriptor,
        "pairwise_distance_matrix",
        lambda _frames: (_ for _ in ()).throw(AssertionError("dense path used")),
    )
    condensed = build_condensed_distance_store(descriptor, frames, workers=1)
    observed = fps_select(condensed, 3, descriptor_name=descriptor.name)
    assert np.allclose(condensed.to_square(), dense)
    assert observed.indices == expected.indices
    assert observed.diversities == pytest.approx(expected.diversities)
