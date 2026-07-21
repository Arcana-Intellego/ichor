"""Tests for ichor.hpc.active_learning.daemon.reconcile."""
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import ichor.hpc.active_learning.daemon.reconcile as reconcile_mod
import ichor.hpc.active_learning.daemon.recovery_contracts as recovery_contracts_mod
from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.daemon.journal import append_event
from ichor.hpc.active_learning.daemon.reconcile import (
    RECONCILE_SUFFIX,
    ReconciliationReport,
    propose_recovery,
    stateful_campaign_artifacts,
    write_proposed_state,
)
from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.daemon.recovery_contracts import (
    active_iteration_handoff_decisions,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    DEFAULT_STATE_FILENAME,
    atomic_write_json,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.handoff_manifests import (
    ARIADNE_RESULTS_SCHEMA_VERSION,
    SEED_SELECTION_SCHEMA_VERSION,
    seeds_picked_path,
    write_ariadne_results_manifest,
)
from ichor.hpc.active_learning.layout import (
    active_allocation_dir,
    active_ariadne_dir,
    active_iteration_dir,
    ariadne_seed_dir,
    bootstrap_selection_dir,
)
from ichor.hpc.active_learning.point_allocation import (
    create_point_allocation,
    pending_attempts,
    point_allocation_path,
    read_point_allocation,
    record_quantum_results,
)
from ichor.hpc.active_learning.replacement_sampling import (
    prepare_replacement_round,
    replacement_round_dir,
)
from ichor.hpc.active_learning.versioning.provenance import (
    enrich_with_point_allocation,
    write_seed_provenance,
)
from ichor.hpc.active_learning.versioning.versioned_directory import VersionedDirectory


_FIXTURE_CAMPAIGN_UID = "reconcile-test"


def _trust_marker_models(monkeypatch):
    """Adapt legacy marker-only model fixtures to the snapshot test seam."""
    original_builder = reconcile_mod.build_committed_artifact_snapshot

    def build(campaign, **kwargs):
        snapshot = original_builder(campaign, **kwargs)
        uid = (
            snapshot.reference_views[-1].campaign_uid
            if snapshot.reference_views
            else _FIXTURE_CAMPAIGN_UID
        )
        models = tuple(
            SimpleNamespace(
                version=int(version),
                campaign_uid=str(uid),
                root=(
                    Path(campaign)
                    / "TRAINED_MODELS"
                    / ("iteration-" + f"{int(version):06d}")
                ),
            )
            for version in snapshot.committed_model_versions
        )
        return replace(snapshot, model_sets=models, model_errors={})

    monkeypatch.setattr(
        reconcile_mod,
        "build_committed_artifact_snapshot",
        build,
    )


def _campaign_dirs(tmp_path):
    campaign = tmp_path / "campaign"
    data = campaign / ".DATA" / "ACTIVE_LEARNING"
    training = campaign / "QM_REFERENCE_DATA"
    models = campaign / "TRAINED_MODELS"
    data.mkdir(parents=True, exist_ok=True)
    training.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return campaign, data, training, models


def _write_pool(campaign):
    src = campaign / "pool_source.xyz"
    src.write_text(
        "1\n"
        "frame 0\n"
        "H 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    TrajectoryPool.import_from(
        src,
        campaign,
        overwrite=True,
    )


def _write_pending_allocation(campaign, *, context, iteration, n):
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    campaign_uid = (
        read_state(state_path).campaign_uid
        if state_path.is_file()
        else _FIXTURE_CAMPAIGN_UID
    )
    targets = {"train": int(n), "int_val": 0, "ext_val": 0, "total": int(n)}
    allocation_path = point_allocation_path(
        campaign,
        context=context,
        iteration=iteration,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid=campaign_uid,
        context=context,
        iteration=iteration,
        targets=targets,
        primary_candidates=[
            {
                "candidate_id": (
                    "candidate-"
                    + context
                    + "-"
                    + str(iteration)
                    + "-"
                    + str(i)
                ),
                "frame_id": i,
            }
            for i in range(int(n))
        ],
        reserve_candidates=[],
    )
    records = []
    for slot in allocation["slots"]:
        attempt = slot["attempts"][0]
        records.append({
            **attempt,
            "slot_id": int(slot["slot_id"]),
            "split": str(slot["split"]),
        })
    return allocation_path, allocation, records


def _complete_handoff_allocation(campaign, *, context, iteration, pointdirs):
    allocation_path, allocation, records = _write_pending_allocation(
        campaign,
        context=context,
        iteration=iteration,
        n=len(pointdirs),
    )
    results = []
    campaign_uid = str(allocation["campaign_uid"])
    for pointdir, record in zip(pointdirs, records):
        write_seed_provenance(
            pointdir,
            campaign_uid=campaign_uid,
            iteration=iteration,
            trajectory_sha256="0" * 64,
            seed_frame_id=record.get("frame_id"),
            seed_id=(
                int(record["slot_id"]) + 1
                if str(context) == "active"
                else None
            ),
            seed_uid=(
                format(int(record["slot_id"]) + 1, "064x")
                if str(context) == "active"
                else None
            ),
            array_task_id_zero_based=(
                int(record["slot_id"])
                if str(context) == "active"
                else None
            ),
            seed_selection_origin="reconcile_fixture",
            seed_variance_at_selection=None,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
        )
        enrich_with_point_allocation(
            pointdir,
            candidate_id=record["candidate_id"],
            context=context,
            slot_id=record["slot_id"],
            split=record["split"],
            allocation_slot_assignment_sha256=str(
                allocation["slot_assignment_sha256"]
            ),
        )
        results.append({
            "candidate_id": record["candidate_id"],
            "accepted": True,
            "pointdir": str(pointdir.resolve()),
        })
    from ichor_hpc.tests.quantum_test_support import (
        attach_synthetic_quantum_batch,
    )

    attach_synthetic_quantum_batch(
        campaign,
        results,
        phase_name=(
            CampaignPhase.INITIAL_AIMALL.value
            if str(context) == "bootstrap"
            else CampaignPhase.AIMALL.value
        ),
        iteration=int(iteration),
    )
    return record_quantum_results(allocation_path, results)


def _write_valid_initial_aimall_handoff(campaign, *, iteration=0):
    initial = campaign / ".DATA" / "STAGING" / "initial"
    pointdir = initial / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    stg.write_points_file(initial, [pointdir])
    stg.write_quantum_acceptance_manifest(
        initial,
        phase_name=CampaignPhase.INITIAL_AIMALL.value,
        iteration=iteration,
        accepted=[pointdir],
        rejected=[],
    )
    _complete_handoff_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        pointdirs=[pointdir],
    )
    return initial


def _write_bootstrap_handoff(campaign, *, phase, iteration=0, archived=False, suffix="20260627-201927"):
    root = (
        campaign / ".DATA" / ("STAGING.archived-" + suffix)
        if archived
        else campaign / ".DATA" / "STAGING"
    )
    initial = root / "initial"
    pointdir = initial / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    (pointdir / "input.wfn").write_text("wfn\n", encoding="utf-8")
    stg.write_points_file(initial, [pointdir])
    stg.write_quantum_acceptance_manifest(
        initial,
        phase_name=phase,
        iteration=iteration,
        accepted=[pointdir],
        rejected=[],
    )
    if phase == CampaignPhase.INITIAL_AIMALL.value:
        _complete_handoff_allocation(
            campaign,
            context="bootstrap",
            iteration=0,
            pointdirs=[pointdir],
        )
    else:
        _write_pending_allocation(
            campaign,
            context="bootstrap",
            iteration=0,
            n=1,
        )
    return initial


def _iter_dir(campaign, iteration):
    path = active_iteration_dir(campaign, int(iteration))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_seeds_picked(campaign, iteration, *, n=1):
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.handoff_manifests import (
        build_seed_selection_manifest,
    )
    from ichor.hpc.active_learning.seed_identity import write_ariadne_task_map

    iter_dir = _iter_dir(campaign, iteration)
    frame_ids = list(range(int(n)))
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    state = read_state(state_path) if state_path.is_file() else fresh_campaign_state()
    campaign_uid = str(state.campaign_uid or "reconcile-test")
    model_version = max(0, int(state.models_version))
    model_manifest_sha256 = "c" * 64
    trajectory_sha256 = "0" * 64
    payload = build_seed_selection_manifest(
        campaign_uid=campaign_uid,
        campaign_random_seed=0,
        iteration=int(iteration),
        models_version=model_version,
        model_manifest_sha256=model_manifest_sha256,
        model_set_sha256=model_manifest_sha256,
        trajectory_sha256=trajectory_sha256,
        selection_strategy="hybrid_variance",
        seed_records=[
            {
                "seed_id": int(i + 1),
                "frame_id": int(i),
                "pool_row_index_zero_based": int(i),
                "selection_origin": "bulk",
                "variance_at_selection": 0.0,
            }
            for i in frame_ids
        ],
    )
    selection_path = seeds_picked_path(iter_dir)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, payload)
    write_ariadne_task_map(iter_dir, payload)
    return iter_dir


def _write_ariadne_handoff(
    campaign,
    iteration,
    *,
    n=2,
    include_safety=True,
    n_rejected=0,
):
    from ichor.hpc.active_learning.ariadne_outputs import (
        write_optimisation_trajectory,
        write_seed_output_manifest,
    )
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.seed_identity import read_ariadne_task_map
    from ichor.hpc.active_learning.versioning.manifest import sha256_file
    from ichor.hpc.active_learning.versioning.provenance import PROVENANCE_FILENAME

    iter_dir = _write_seeds_picked(campaign, iteration, n=n)
    task_map = read_ariadne_task_map(
        iter_dir,
        expected_iteration=int(iteration),
    )
    ariadne_root = active_ariadne_dir(iter_dir)
    accepted = []
    rejected = []
    accepted_limit = int(n) - int(n_rejected)
    for task in task_map["tasks"]:
        seed_id = int(task["seed_id"])
        array_task_id = int(task["array_task_id"])
        seed_uid = str(task["seed_uid"])
        if array_task_id >= accepted_limit:
            rejected.append({
                "seed_id": seed_id,
                "seed_uid": seed_uid,
                "array_task_id": array_task_id,
                "seed_dir": "seeds/seed-" + str(seed_id).zfill(6),
                "reason": (
                    "seed_output_invalid: ValueError: resource implementation "
                    "ICHOR package tree has drifted"
                ),
            })
            continue
        seed_dir = ariadne_seed_dir(iter_dir, seed_id)
        seed_dir.mkdir(parents=True, exist_ok=False)
        result = {
            "iteration": int(iteration),
            "seed_id": seed_id,
            "seed_uid": seed_uid,
            "array_task_id": array_task_id,
            "seed_frame_id": array_task_id,
            "trajectory_sha256": "0" * 64,
            "atom_types": ["O", "H", "H"],
            "final_coordinates": [
                [0.0, 0.0, 0.0],
                [0.96 + 0.40 * (array_task_id + 1), 0.0, 0.0],
                [-0.24, 0.93 + 0.30 * (array_task_id + 1), 0.0],
            ],
            "alpha_trajectory": [0.0, 1.0],
            "alpha_initial": 0.0,
            "alpha_final": 1.0,
            "n_evaluations": 2,
            "return_code": 0,
            "wall_seconds": 1.0,
            "fell_back_to_ds": False,
            "whitened_distance_final": 0.5,
            "task_success": True,
        }
        if include_safety:
            result["landing_safety"] = {
                "accepted": True,
                "policy": "raw_final",
                "selected_origin": "raw_final",
                "reasons": [],
            }
        result_path = seed_dir / "result.json"
        atomic_write_json(result_path, result)
        write_optimisation_trajectory(
            seed_dir,
            atom_types=result["atom_types"],
            coordinate_frames=[result["final_coordinates"]],
            alpha_values=[result["alpha_final"]],
            gradient_norms=[0.0],
            origins=["raw_final"],
        )
        output_manifest = write_seed_output_manifest(
            seed_dir,
            campaign_uid=str(task_map["campaign_uid"]),
            iteration=int(iteration),
            seed_id=seed_id,
            seed_uid=seed_uid,
            array_task_id=array_task_id,
            task_success=True,
            task_exit_code=0,
        )
        provenance_path = write_seed_provenance(
            seed_dir,
            campaign_uid=str(task_map["campaign_uid"]),
            iteration=int(iteration),
            trajectory_sha256="0" * 64,
            seed_frame_id=array_task_id,
            seed_id=seed_id,
            seed_uid=seed_uid,
            array_task_id_zero_based=array_task_id,
            seed_selection_origin="bulk",
            seed_variance_at_selection=0.0,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
        )
        record = {
            "seed_id": seed_id,
            "seed_uid": seed_uid,
            "array_task_id": array_task_id,
            "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
            "result_json": result_path.relative_to(ariadne_root).as_posix(),
            "provenance_json": provenance_path.relative_to(ariadne_root).as_posix(),
            "output_manifest": output_manifest.relative_to(ariadne_root).as_posix(),
            "seed_frame_id": array_task_id,
            "pool_row_index_zero_based": array_task_id,
            "selection_origin": "bulk",
            "variance_at_selection": 0.0,
            "alpha_initial": 0.0,
            "alpha_final": 1.0,
            "whitened_distance_final": 0.5,
            "return_code": 0,
            "result_sha256": sha256_file(result_path),
            "provenance_sha256": sha256_file(
                seed_dir / PROVENANCE_FILENAME
            ),
            "output_manifest_sha256": sha256_file(output_manifest),
        }
        if include_safety:
            record["landing_safety"] = dict(result["landing_safety"])
        accepted.append(record)
    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "campaign_uid": str(task_map["campaign_uid"]),
        "iteration": int(iteration),
        "trajectory_sha256": "0" * 64,
        "task_map": {
            "path": "TASK_MAP.json",
            "sha256": sha256_file(ariadne_root / "TASK_MAP.json"),
        },
        "expected_n": int(n),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
    })
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.config_lock import (
        canonical_config,
        config_fingerprint,
    )
    from ichor.hpc.active_learning.handoff_manifests import (
        write_ariadne_batch_decision,
    )

    config_path = campaign / "campaign.yaml"
    config = CampaignConfig.from_yaml(config_path) if config_path.is_file() else CampaignConfig()
    write_ariadne_batch_decision(
        iter_dir,
        campaign_uid=str(task_map["campaign_uid"]),
        iteration=int(iteration),
        config_sha256=config_fingerprint(canonical_config(config)),
        failure_threshold_fraction=float(config.runtime.failure_threshold_fraction),
        expected_n=int(n),
        n_accepted=len(accepted),
        n_rejected=len(rejected),
        accepted=True,
        reasons=[],
    )
    return iter_dir


def _write_phase_b_handoff(campaign, iteration, *, n=2):
    from types import SimpleNamespace

    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.sampling.diversity import _run_phase_b

    iter_dir = _write_ariadne_handoff(campaign, iteration, n=n)
    config = CampaignConfig()
    config.phase_b.descriptor = "rmsd_massweight"
    config.point_allocation.batch_training_size = 1
    config.point_allocation.batch_internal_validation_size = int(n) - 1
    config.to_yaml(Path(campaign) / "campaign.yaml")
    from ichor.hpc.active_learning.daemon.config_lock import (
        canonical_config,
        config_fingerprint,
    )
    from ichor.hpc.active_learning.handoff_manifests import (
        write_ariadne_batch_decision,
    )

    write_ariadne_batch_decision(
        iter_dir,
        campaign_uid="reconcile-test",
        iteration=int(iteration),
        config_sha256=config_fingerprint(canonical_config(config)),
        failure_threshold_fraction=float(config.runtime.failure_threshold_fraction),
        expected_n=int(n),
        n_accepted=int(n),
        n_rejected=0,
        accepted=True,
        reasons=[],
    )
    from ichor.hpc.active_learning.sampling_protocol import (
        resolve_sampling_protocol,
    )

    resolve_sampling_protocol(
        campaign,
        config,
        iteration=int(iteration),
    )
    assert _run_phase_b(
        SimpleNamespace(iteration=int(iteration)),
        Path(campaign),
        config,
    ) == 0
    return iter_dir


def _write_split(campaign, iteration):
    from ichor.hpc.active_learning.daemon.state import atomic_write_json

    iter_dir = _iter_dir(campaign, iteration)
    allocation_path = point_allocation_path(
        campaign,
        context="active",
        iteration=iteration,
    )
    if not allocation_path.is_file():
        _write_pending_allocation(
            campaign,
            context="active",
            iteration=iteration,
            n=1,
        )
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    split_path = active_allocation_dir(iter_dir) / "SPLIT_RECEIPT.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(split_path, {
        "schema_version": 3,
        "iteration": int(iteration),
        "strategy": "exact_pre_qm_point_allocation",
        "point_allocation_manifest": str(allocation_path.resolve()),
        "slot_assignment_sha256": str(allocation["slot_assignment_sha256"]),
        "targets": dict(allocation["targets"]),
        "slots": [
            {
                "slot_id": int(slot["slot_id"]),
                "split": str(slot["split"]),
                "candidate_id": str(slot["attempts"][0]["candidate_id"]),
            }
            for slot in allocation["slots"]
        ],
    })
    return split_path


def _commit_reference_versions(campaign, versions):
    for version in versions:
        context = "bootstrap" if int(version) == 0 else "active"
        iteration = 0 if context == "bootstrap" else int(version)
        staging_name = "initial" if context == "bootstrap" else "iter_" + str(iteration)
        pointdir = (
            Path(campaign)
            / ".DATA"
            / "STAGING"
            / staging_name
            / "POINT_0000.pointdir"
        )
        pointdir.mkdir(parents=True, exist_ok=True)
        (pointdir / "input.gjf").write_text(
            "#p hf/sto-3g\n\nreconcile fixture\n\n0 1\n"
            "O 0.000000 0.000000 0.000000\n"
            "H 0.960000 0.000000 0.000000\n"
            "H -0.240000 0.930000 0.000000\n\n",
            encoding="utf-8",
        )
        (pointdir / "geometry.xyz").write_text(
            "3\nreconcile fixture\n"
            "O 0.000000 0.000000 0.000000\n"
            "H 0.960000 0.000000 0.000000\n"
            "H -0.240000 0.930000 0.000000\n",
            encoding="utf-8",
        )
        _complete_handoff_allocation(
            campaign,
            context=context,
            iteration=iteration,
            pointdirs=[pointdir],
        )
        stg.commit_reference_data_delta(
            campaign,
            reference_data_version=int(version),
            context=context,
            iteration=iteration,
        )
        shutil.rmtree(pointdir.parent, ignore_errors=True)


def _commit_training_and_model_versions(training, models, versions):
    _commit_reference_versions(Path(training).parent, versions)
    mv = VersionedDirectory(models)
    for version in versions:
        staged = mv.stage(None, int(version))
        (staged / "marker.txt").write_text("model " + str(version), encoding="utf-8")
        mv.commit(int(version))


def test_propose_recovery_on_empty_campaign_returns_init(tmp_path):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    report = propose_recovery(campaign)
    assert report.proposed_state.phase is CampaignPhase.INIT
    assert report.proposed_state.reference_data_version == -1
    assert report.proposed_state.models_version == -1
    assert report.committed_reference_data_versions == []
    assert report.committed_model_versions == []
    assert report.existing_state_loaded is False


def test_propose_recovery_treats_completed_staging_as_routine_cleanup(
    tmp_path,
    monkeypatch,
):
    _trust_marker_models(monkeypatch)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=3)
    state.campaign_uid = _FIXTURE_CAMPAIGN_UID
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    historical = campaign / ".DATA" / "STAGING" / "initial"
    historical.mkdir(parents=True)

    report = propose_recovery(campaign)

    retirement = report.completed_staging_retirement
    assert [record["iteration"] for record in retirement["eligible"]] == [0]
    assert retirement["eligible"][0]["action"] == "delete"
    assert ".DATA/STAGING is non-empty" not in report.unsafe_reasons
    assert ".DATA/STAGING" not in report.blocking_artifacts


def test_script_inventory_accepts_submission_intent_records(tmp_path):
    campaign = tmp_path / "campaign"
    identity = "r0000-a0001-test"
    script = (
        campaign
        / ".DATA"
        / "SCRIPTS"
        / "JOBS"
        / "GAUSSIAN"
        / "INITIAL_GAUSSIAN"
        / "iteration-000000"
        / identity
        / "job.sh"
    )
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n", encoding="utf-8", newline="\n")

    inventory = reconcile_mod._scripts_inventory(
        campaign,
        intent_records=[
            {
                "submission_identity": identity,
                "submitted_script_path": str(script),
                "status": "COMPLETED",
            }
        ],
    )

    assert inventory["error"] is None
    assert inventory["attempt_bundle_count"] == 1
    assert inventory["attempt_bundles"][0]["submission_identity"] == identity


def test_active_replacement_recovery_advances_only_with_durable_handoffs(tmp_path):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    iter_dir = _write_ariadne_handoff(campaign, 1, n=2)
    from ichor.hpc.active_learning.handoff_manifests import (
        read_ariadne_results_manifest,
    )
    accepted = read_ariadne_results_manifest(
        iter_dir,
        expected_iteration=1,
    )["accepted"]

    def allocation_candidate(record, candidate_id):
        return {
            "candidate_id": candidate_id,
            "seed_id": int(record["seed_id"]),
            "seed_uid": str(record["seed_uid"]),
            "seed_frame_id": int(record["seed_frame_id"]),
            "result_json": str(record["result_json"]),
            "result_sha256": str(record["result_sha256"]),
            "landing_safety": dict(record["landing_safety"]),
        }

    allocation_path = point_allocation_path(
        campaign,
        context="active",
        iteration=1,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid=_FIXTURE_CAMPAIGN_UID,
        context="active",
        iteration=1,
        targets={"train": 1, "int_val": 0, "ext_val": 0, "total": 1},
        primary_candidates=[
            allocation_candidate(accepted[0], "candidate-primary")
        ],
        reserve_candidates=[
            {
                **allocation_candidate(accepted[1], "candidate-reserve"),
                "reserve_rank": 0,
            }
        ],
    )
    initial_attempt = pending_attempts(allocation)[0]
    record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": str(initial_attempt["candidate_id"]),
                "accepted": False,
                "pointdir": "/synthetic/failed.pointdir",
                "reason": "synthetic failure",
            }
        ],
    )
    state = fresh_campaign_state(max_iterations=3)
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0

    decisions = active_iteration_handoff_decisions(campaign, state)
    assert [decision.phase for decision in decisions] == [CampaignPhase.ALLOCATION_CHECK]

    prepare_replacement_round(
        campaign,
        context="active",
        iteration=1,
        replacement_round=1,
    )
    decisions = active_iteration_handoff_decisions(campaign, state)
    assert [decision.phase for decision in decisions] == [
        CampaignPhase.REPLACEMENT_GAUSSIAN
    ]
    assert decisions[0].replacement_round == 1

    round_dir = replacement_round_dir(
        campaign,
        context="active",
        iteration=1,
        replacement_round=1,
    )
    pointdir = round_dir / "POINT_0001.pointdir"
    pointdir.mkdir()
    stg.write_points_file(round_dir, [pointdir])
    stg.write_quantum_acceptance_manifest(
        round_dir,
        phase_name=CampaignPhase.REPLACEMENT_GAUSSIAN.value,
        iteration=1,
        accepted=[pointdir],
        rejected=[],
    )
    quality_manifest = round_dir / "quantum_quality.json"
    atomic_write_json(quality_manifest, {})
    decisions = active_iteration_handoff_decisions(campaign, state)
    assert [decision.phase for decision in decisions] == [
        CampaignPhase.REPLACEMENT_AIMALL
    ]

    replacement = read_point_allocation(allocation_path)
    replacement_attempt = pending_attempts(replacement)[0]
    record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": str(replacement_attempt["candidate_id"]),
                "accepted": True,
                "pointdir": str(pointdir.resolve()),
                "quality_manifest": str(quality_manifest.resolve()),
            }
        ],
        expected_generation=int(replacement["generation"]),
    )
    decisions = active_iteration_handoff_decisions(campaign, state)
    assert [decision.phase for decision in decisions] == [
        CampaignPhase.REFERENCE_COMMIT
    ]


def test_recovery_contract_status_marks_halted_state_not_runnable(tmp_path):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED

    status = recovery_contracts_mod.recovery_contract_status(campaign, state)

    assert status["selected_phase"] == CampaignPhase.HALTED.value
    assert status["contract_ok"] is False
    assert status["required_inputs"] == []
    assert status["trusted_inputs"] == []
    assert status["missing_or_invalid_inputs"] == [
        "phase HALTED is not a runnable recovery phase"
    ]


def test_recovery_contract_status_reports_seed_handoff_contract(tmp_path):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    _write_seeds_picked(campaign, 2, n=3)
    state = fresh_campaign_state(max_iterations=5)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 2

    status = recovery_contracts_mod.recovery_contract_status(campaign, state)

    assert status["selected_phase"] == CampaignPhase.ARIADNE_ARRAY.value
    assert status["iteration"] == 2
    assert status["contract_ok"] is True
    assert status["required_inputs"] == ["seed_selection/SELECTION.json"]
    assert status["trusted_inputs"] == ["seed_selection/SELECTION.json"]
    assert status["missing_or_invalid_inputs"] == []


def test_propose_recovery_phase_a_sample_reenters_initial_gaussian(tmp_path):
    from ichor.hpc.active_learning.handoff_manifests import write_phase_a_sample_manifest
    from ichor.hpc.active_learning.sampling.diversity_contract import (
        diversity_selector_contract,
    )

    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    state = fresh_campaign_state()
    state.campaign_uid = "reconcile-test"
    write_state(data / DEFAULT_STATE_FILENAME, state)
    from ichor.hpc.active_learning.layout import bootstrap_selection_dir

    initial = bootstrap_selection_dir(campaign)
    initial.mkdir(parents=True)
    sample = initial / "selected.xyz"
    index = initial / "selected_indices.dat"
    sample.write_text("1\nframe 0\nH 0.0 0.0 0.0\n", encoding="utf-8")
    index.write_text("0\n", encoding="utf-8")
    allocation_path, allocation, allocation_records = _write_pending_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        n=1,
    )
    write_phase_a_sample_manifest(initial, {
        "phase": "PHASE_A_DIVERSITY",
        "iteration": 0,
        "sample_xyz": str(sample.resolve()),
        "index_path": str(index.resolve()),
        "n_select": 1,
        "n_frames": 1,
        "selected_indices": [0],
        "descriptor": "mass_weighted_rmsd",
        "selector": diversity_selector_contract(),
        "n_pool_frames": 1,
        "point_allocation": {
            "manifest": str(allocation_path.resolve()),
            "targets": dict(allocation["targets"]),
            "primary": allocation_records,
            "reserve_frame_ids": [],
            "reserve_count": 0,
        },
        "source_pool_manifest": ".DATA/TRAJECTORY/pool.manifest.json",
    })

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.INITIAL_GAUSSIAN
    assert report.proposed_state.reference_data_version == -1
    assert report.proposed_state.models_version == -1
    assert report.phase_a_handoff is not None
    assert "valid Phase A sample" in report.decision


def test_propose_recovery_never_trusts_stop_check_without_committed_versions(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    state = fresh_campaign_state(max_iterations=50)
    state.phase = CampaignPhase.STOP_CHECK
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)

    report = propose_recovery(campaign)

    assert report.committed_reference_data_versions == []
    assert report.committed_model_versions == []
    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert report.proposed_state.reference_data_version == -1
    assert report.proposed_state.models_version == -1
    assert "latest coherent committed reference-data/model pair is trusted" not in report.decision
    assert "no committed versions" in report.decision


def test_propose_recovery_halted_phase_a_failure_retries_phase_a(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.iteration = 0
    state.reference_data_version = 0
    state.validation_set_version = 0
    state.models_version = 0
    state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] = None
    write_state(data / DEFAULT_STATE_FILENAME, state)
    append_event(
        data / "journal.ndjson",
        "halt",
        from_phase=CampaignPhase.PHASE_A_DIVERSITY.value,
        iteration=0,
        reason="backend_submission_failed: partition 'multicore_small' is not present",
    )

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.PHASE_A_DIVERSITY
    assert report.proposed_state.iteration == 0
    assert report.proposed_state.reference_data_version == -1
    assert report.proposed_state.validation_set_version == -1
    assert report.proposed_state.models_version == -1
    assert report.proposed_state.pending_jobs == {}
    assert "PHASE_A_DIVERSITY" in report.decision
    assert any(
        c.get("phase") == CampaignPhase.PHASE_A_DIVERSITY.value
        for c in report.recovery_candidates
    )


def test_propose_recovery_halted_phase_a_failure_requires_pool(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.iteration = 0
    state.reference_data_version = 0
    state.validation_set_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    append_event(
        data / "journal.ndjson",
        "halt",
        from_phase=CampaignPhase.PHASE_A_DIVERSITY.value,
        iteration=0,
        reason="backend_submission_failed: partition 'multicore_small' is not present",
    )

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert not any(
        c.get("phase") == CampaignPhase.PHASE_A_DIVERSITY.value
        for c in report.recovery_candidates
    )
    assert any("trajectory pool" in reason for reason in report.unsafe_reasons)


def test_propose_recovery_blocks_real_pending_job_without_intent(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.iteration = 0
    state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] = "123456"
    write_state(data / DEFAULT_STATE_FILENAME, state)
    append_event(
        data / "journal.ndjson",
        "halt",
        from_phase=CampaignPhase.PHASE_A_DIVERSITY.value,
        iteration=0,
        reason="scheduler status uncertain",
    )

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert any("pending job(s) without active submission intent" in r for r in report.unsafe_reasons)
    assert "pending_jobs" in report.blocking_artifacts


def test_propose_recovery_initial_aimall_handoff_reenters_reference_commit(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    append_event(
        data / "journal.ndjson",
        "phase_transition",
        from_phase=CampaignPhase.INITIAL_GAUSSIAN.value,
        to_phase=CampaignPhase.INITIAL_AIMALL.value,
        iteration=0,
    )
    state = fresh_campaign_state(max_iterations=50)
    state.phase = CampaignPhase.STOP_CHECK
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    _write_valid_initial_aimall_handoff(campaign)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.REFERENCE_COMMIT
    assert report.proposed_state.reference_data_version == -1
    assert report.proposed_state.models_version == -1
    assert "exact point allocation is complete" in report.decision
    assert any(
        "protected active staging handoff for REFERENCE_COMMIT" in artefact
        for artefact in report.trusted_artifacts
    )


def test_propose_recovery_reference_commit_journal_handoff_reenters_commit(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    append_event(
        data / "journal.ndjson",
        "phase_transition",
        from_phase=CampaignPhase.INITIAL_AIMALL.value,
        to_phase=CampaignPhase.REFERENCE_COMMIT.value,
        iteration=0,
    )
    state = fresh_campaign_state(max_iterations=50)
    state.phase = CampaignPhase.STOP_CHECK
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    _write_valid_initial_aimall_handoff(campaign)

    report = propose_recovery(campaign)

    assert report.last_phase_in_journal == CampaignPhase.REFERENCE_COMMIT.value
    assert report.proposed_state.phase is CampaignPhase.REFERENCE_COMMIT
    assert report.proposed_state.reference_data_version == -1
    assert report.proposed_state.models_version == -1


def test_propose_recovery_initial_aimall_missing_handoff_halts(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    append_event(
        data / "journal.ndjson",
        "phase_transition",
        from_phase=CampaignPhase.INITIAL_GAUSSIAN.value,
        to_phase=CampaignPhase.INITIAL_AIMALL.value,
        iteration=0,
    )
    state = fresh_campaign_state(max_iterations=50)
    state.phase = CampaignPhase.STOP_CHECK
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert report.proposed_state.reference_data_version == -1
    assert report.proposed_state.models_version == -1
    assert "INITIAL_AIMALL completed" in report.decision
    assert any("initial AIMAll handoff invalid or missing" in r for r in report.unsafe_reasons)
    assert ".DATA/STAGING/initial" in report.blocking_artifacts


def test_propose_recovery_archived_initial_gaussian_handoff_reenters_initial_aimall(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    append_event(
        data / "journal.ndjson",
        "reconcile_applied",
        phase=CampaignPhase.STOP_CHECK.value,
        iteration=0,
    )
    state = fresh_campaign_state(max_iterations=50)
    state.phase = CampaignPhase.STOP_CHECK
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    archived = _write_bootstrap_handoff(
        campaign,
        phase=CampaignPhase.INITIAL_GAUSSIAN.value,
        archived=True,
    )

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.INITIAL_AIMALL
    assert report.proposed_state.reference_data_version == -1
    assert report.proposed_state.models_version == -1
    assert report.bootstrap_handoff is not None
    assert report.bootstrap_handoff["archived"] is True
    assert report.bootstrap_handoff["path"] == str(archived)
    assert "valid initial Gaussian handoff" in report.decision


def test_propose_recovery_archived_initial_aimall_handoff_reenters_reference_commit(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    state = fresh_campaign_state(max_iterations=50)
    state.phase = CampaignPhase.STOP_CHECK
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    archived = _write_bootstrap_handoff(
        campaign,
        phase=CampaignPhase.INITIAL_AIMALL.value,
        archived=True,
    )

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.REFERENCE_COMMIT
    assert report.bootstrap_handoff is not None
    assert report.bootstrap_handoff["path"] == str(archived)
    assert "exact point allocation is complete" in report.decision


def test_propose_recovery_bootstrap_training_only_reenters_initial_ferebus(
    tmp_path,
):
    campaign, data, training, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    pointdir = campaign / ".DATA" / "STAGING" / "initial" / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    _complete_handoff_allocation(
        campaign,
        context="bootstrap",
        iteration=0,
        pointdirs=[pointdir],
    )
    stg.commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    state = fresh_campaign_state(
        max_iterations=50,
        campaign_uid="reconcile-test",
    )
    state.phase = CampaignPhase.HALTED
    state.iteration = 0
    state.reference_data_version = 0
    state.models_version = -1
    write_state(data / DEFAULT_STATE_FILENAME, state)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.INITIAL_FEREBUS
    assert report.proposed_state.iteration == 0
    assert "exact point allocation is complete" in report.decision


def test_propose_recovery_reports_unique_measurement_only_ferebus_candidate(
    tmp_path,
):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_reference_versions(campaign, (0,))
    state = fresh_campaign_state(
        max_iterations=50,
        campaign_uid="reconcile-test",
    )
    state.phase = CampaignPhase.HALTED
    state.reference_data_version = 0
    state.models_version = -1
    write_state(data / DEFAULT_STATE_FILENAME, state)
    candidate = (
        campaign
        / "TRAINED_MODELS"
        / "rejected-candidates"
        / "reference-000000"
        / ("a" * 64)
    )
    candidate.mkdir(parents=True)
    error = "ferebus_quality_metric_failed:ValueError:legacy metric"
    atomic_write_json(
        candidate / "FEREBUS_QUALITY.json",
        {
            "schema_version": 4,
            "campaign_uid": "reconcile-test",
            "reference_data_version": 0,
            "measurement_complete": False,
            "measurement_errors": [error],
        },
    )
    from ichor.hpc.active_learning.daemon.completion_receipts import (
        canonical_sha256,
    )
    from ichor.hpc.active_learning.versioning.manifest import sha256_file

    quality_path = candidate / "FEREBUS_QUALITY.json"
    evaluation = {
        "accepted": False,
        "reasons": ["ferebus_aggregate_metric_missing", error],
    }
    digest = canonical_sha256(evaluation)
    evaluation["evaluation_sha256"] = digest
    atomic_write_json(
        candidate / "FEREBUS_QUALITY_DECISION.json",
        {
            "schema_version": 2,
            "campaign_uid": "reconcile-test",
            "reference_data_version": 0,
            "quality": {
                "path": "FEREBUS_QUALITY.json",
                "size": quality_path.stat().st_size,
                "sha256": sha256_file(quality_path),
            },
            "evaluations": [evaluation],
            "current_evaluation_sha256": digest,
        },
    )
    atomic_write_json(candidate / "FEREBUS_TASKS.json", {})
    atomic_write_json(candidate / "FEREBUS_TASK_MAP.json", {})

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.INITIAL_FEREBUS
    assert report.ferebus_candidate_recovery is not None
    assert report.ferebus_candidate_recovery["candidate_id"] == "a" * 64
    assert any(
        "without another scheduler submission" in note
        for note in report.notes
    )


def test_propose_recovery_missing_state_nonempty_staging_halts(tmp_path):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    staging = campaign / ".DATA" / "STAGING" / "iter_1"
    staging.mkdir(parents=True)
    (staging / "POINTS.txt").write_text("", encoding="utf-8")
    report = propose_recovery(campaign)
    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert any("non-empty campaign" in n for n in report.notes)
    assert report.unsafe_reasons


def test_stateful_campaign_artifacts_include_config_lock(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    (data / "config_lock.json").write_text("{}", encoding="utf-8")

    findings = stateful_campaign_artifacts(campaign)

    assert ".DATA/ACTIVE_LEARNING/config_lock.json" in findings


def test_stateful_campaign_artifacts_include_phase_a_outputs(tmp_path):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    phase_a = bootstrap_selection_dir(campaign)
    phase_a.mkdir(parents=True)
    (phase_a / "SELECTION.json").write_text("{}", encoding="utf-8")
    (phase_a / "selected.xyz").write_text("sample\n", encoding="utf-8")
    (phase_a / "selected_indices.dat").write_text("0\n", encoding="utf-8")

    findings = stateful_campaign_artifacts(campaign)

    normalised = {Path(value).as_posix() for value in findings}
    assert ".DATA/BOOTSTRAP/selection/SELECTION.json" in normalised
    assert ".DATA/BOOTSTRAP/selection/selected.xyz" in normalised
    assert ".DATA/BOOTSTRAP/selection/selected_indices.dat" in normalised


def test_propose_recovery_active_submission_intent_is_adoption_ready(tmp_path):
    from ichor.hpc.active_learning.daemon.submission_intent import (
        mark_submitted,
        write_pre_submit_intent,
    )

    campaign, _, _, _ = _campaign_dirs(tmp_path)
    write_pre_submit_intent(
        campaign,
        campaign_uid="uid",
        phase_name="FEREBUS",
        iteration=3,
    )
    mark_submitted(campaign, "FEREBUS", 3, "123456")
    report = propose_recovery(campaign)
    assert report.proposed_state.phase is CampaignPhase.FEREBUS
    assert report.proposed_state.iteration == 3
    assert report.active_submission_intents
    reason = "\n".join(report.unsafe_reasons)
    assert "job_id=123456" in reason
    assert "expected_job_name=uid-FEREBUS-3" in reason


def test_propose_recovery_excludes_receipt_backed_jobless_ferebus_intent(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon.completion_receipts import (
        write_completion_receipt,
    )
    from ichor.hpc.active_learning.daemon.submission_intent import (
        write_pre_submit_intent,
    )

    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, (0,))
    _trust_marker_models(monkeypatch)

    before = fresh_campaign_state(
        max_iterations=2,
        campaign_uid=_FIXTURE_CAMPAIGN_UID,
    )
    before.phase = CampaignPhase.INITIAL_FEREBUS
    before.reference_data_version = 0
    before.validation_set_version = 0
    intent = write_pre_submit_intent(
        campaign,
        campaign_uid=before.campaign_uid,
        phase_name=before.phase.value,
        iteration=0,
        expected_tasks=1,
    )
    after = fresh_campaign_state(
        max_iterations=2,
        campaign_uid=_FIXTURE_CAMPAIGN_UID,
    )
    after.phase = CampaignPhase.SEED_SELECT
    after.iteration = 1
    after.reference_data_version = 0
    after.validation_set_version = 0
    after.models_version = 0
    write_completion_receipt(
        campaign,
        campaign_uid=before.campaign_uid,
        phase=before.phase.value,
        iteration=0,
        replacement_round=0,
        config_sha256="c" * 64,
        state_before=before,
        state_after=after,
        next_phase=after.phase.value,
        next_iteration=after.iteration,
        state_updates={"models_version": 0},
        evidence=[],
        job_id=None,
        expected_tasks=1,
        submission_identity=str(intent["submission_identity"]),
    )
    halted = fresh_campaign_state(
        max_iterations=2,
        campaign_uid=_FIXTURE_CAMPAIGN_UID,
    )
    halted.phase = CampaignPhase.HALTED
    halted.iteration = 1
    halted.reference_data_version = 0
    halted.validation_set_version = 0
    halted.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, halted)

    report = propose_recovery(campaign)

    assert report.active_submission_intents == []
    assert len(report.receipt_backed_intent_repairs) == 1
    repair = report.receipt_backed_intent_repairs[0]
    assert repair["phase"] == CampaignPhase.INITIAL_FEREBUS.value
    assert repair["completion_receipt"]["receipt_id"]
    assert not any(
        "active submission intent" in reason
        or "prepared scratch" in reason
        for reason in report.unsafe_reasons
    )


def test_propose_recovery_force_cannot_mint_uid_for_nonempty_campaign(tmp_path):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    staging = campaign / ".DATA" / "STAGING" / "iter_1"
    staging.mkdir(parents=True)
    (staging / "POINTS.txt").write_text("", encoding="utf-8")
    report = propose_recovery(campaign, allow_fresh_init_on_nonempty=True)
    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert any(
        "no recoverable trusted campaign_uid" in reason
        for reason in report.unsafe_reasons
    )


def test_propose_recovery_finds_committed_reference_data_versions(tmp_path):
    campaign, _, training, _ = _campaign_dirs(tmp_path)
    _commit_reference_versions(campaign, (0, 1, 2))
    report = propose_recovery(campaign)
    assert report.committed_reference_data_versions == [0, 1, 2]
    assert report.proposed_state.reference_data_version == 2
    assert report.proposed_state.models_version == -1
    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert report.deep_verification_required is True
    assert any("trajectory pool" in r for r in report.unsafe_reasons)


def test_authority_recovery_never_uses_recursive_filesystem_walkers(
    tmp_path,
    monkeypatch,
):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_reference_versions(campaign, (0,))

    def refuse_recursive_walk(*_args, **_kwargs):
        raise AssertionError("authority recovery attempted a recursive walk")

    monkeypatch.setattr(Path, "rglob", refuse_recursive_walk)
    monkeypatch.setattr(os, "walk", refuse_recursive_walk)

    report = propose_recovery(campaign, verification_level="authority")

    assert report.committed_reference_data_versions == [0]
    assert report.valid_reference_data_versions == [0]
    assert report.proposed_state.phase is CampaignPhase.INITIAL_FEREBUS
    assert report.deep_verification_required is True


def test_authority_recovery_reuses_control_inventory_from_snapshot(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import (
        reconcile_transaction,
        reference_commit,
        submission_intent,
    )
    from ichor.hpc.active_learning.daemon.artifact_snapshot import (
        build_committed_artifact_snapshot,
    )

    campaign, _, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_reference_versions(campaign, (0,))
    snapshot = build_committed_artifact_snapshot(
        campaign,
        verification_level="authority",
    )

    def refuse_second_inventory(*_args, **_kwargs):
        raise AssertionError("recovery re-read authority control records")

    monkeypatch.setattr(submission_intent, "inventory_intents", refuse_second_inventory)
    monkeypatch.setattr(
        reconcile_transaction,
        "inventory_reconcile_transactions",
        refuse_second_inventory,
    )
    monkeypatch.setattr(
        reference_commit,
        "inventory_reference_commits",
        refuse_second_inventory,
    )

    report = propose_recovery(
        campaign,
        artifact_snapshot=snapshot,
        verification_level="authority",
    )

    assert report.valid_reference_data_versions == [0]
    assert report.artifact_snapshot is snapshot


def test_propose_recovery_reports_decision_and_trusted_versions(tmp_path):
    campaign, _, training, models = _campaign_dirs(tmp_path)
    mv = VersionedDirectory(models)
    _commit_reference_versions(campaign, (0,))
    s = mv.stage(None, 0)
    (s / "marker.txt").write_text("model", encoding="utf-8")
    mv.commit(0)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert "committed model artefacts are present but invalid" in report.decision
    assert "reference-data version 0" in report.trusted_artifacts
    assert "model version 0" in report.blocking_artifacts
    assert "trajectory pool" in report.blocking_artifacts


def test_propose_recovery_sets_iteration_from_active_version_mapping(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, _, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    mv = VersionedDirectory(models)
    _commit_reference_versions(campaign, range(3))
    for version in range(3):
        s = mv.stage(None, version)
        (s / "marker.txt").write_text("model " + str(version), encoding="utf-8")
        mv.commit(version)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.STOP_CHECK, report.unsafe_reasons
    assert report.proposed_state.reference_data_version == 2
    assert report.proposed_state.models_version == 2
    assert report.proposed_state.iteration == 2


def test_propose_recovery_training_one_ahead_reenters_ferebus(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, _, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    mv = VersionedDirectory(models)
    _commit_reference_versions(campaign, range(3))
    for version in range(2):
        s = mv.stage(None, version)
        (s / "marker.txt").write_text("model " + str(version), encoding="utf-8")
        mv.commit(version)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.FEREBUS
    assert report.proposed_state.reference_data_version == 2
    assert report.proposed_state.models_version == 1
    assert report.proposed_state.iteration == 2
    assert not any("newer committed reference-data version" in r for r in report.unsafe_reasons)


def test_propose_recovery_preserves_existing_seed_select_cursor(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(recovery_contracts_mod, "verify_committed_model_version", lambda *a, **k: None)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    mv = VersionedDirectory(models)
    _commit_reference_versions(campaign, (0,))
    s = mv.stage(None, 0)
    (s / "marker.txt").write_text("model", encoding="utf-8")
    mv.commit(0)
    state = fresh_campaign_state(max_iterations=3)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.SEED_SELECT
    assert report.proposed_state.iteration == 1
    assert report.decision == (
        "SEED_SELECT: bootstrap reference data and models are committed"
    )


def test_propose_recovery_prefers_seeds_over_stale_seed_select(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=3)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    _write_seeds_picked(campaign, 1)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.ARIADNE_ARRAY
    assert report.proposed_state.iteration == 1
    assert "valid seed-selection handoff" in report.decision


def test_recovery_rejects_ariadne_results_without_landing_safety(
    tmp_path,
    monkeypatch,
):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=3)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    _write_ariadne_handoff(campaign, 1, n=1, include_safety=False)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.ARIADNE_ARRAY
    assert report.proposed_state.iteration == 1
    assert "valid seed-selection handoff" in report.decision
    assert "valid ARIADNE results handoff" not in report.decision


def test_recovery_cannot_advance_from_rejected_ariadne_batch(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.config_lock import (
        canonical_config,
        config_fingerprint,
    )
    from ichor.hpc.active_learning.handoff_manifests import (
        ariadne_batch_decision_path,
        ariadne_results_path,
        write_ariadne_batch_decision,
    )

    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(recovery_contracts_mod, "verify_committed_model_version", lambda *a, **k: None)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=3, campaign_uid="reconcile-test")
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    iter_dir = _write_ariadne_handoff(campaign, 1, n=2)
    results_path = ariadne_results_path(iter_dir)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    rejected_record = results["accepted"].pop()
    results["rejected"] = [{
        "seed_id": int(rejected_record["seed_id"]),
        "seed_uid": str(rejected_record["seed_uid"]),
        "array_task_id": int(rejected_record["array_task_id"]),
        "reason": "synthetic task rejection",
    }]
    results["n_accepted"] = 1
    results["n_rejected"] = 1
    write_ariadne_results_manifest(iter_dir, results)
    ariadne_batch_decision_path(iter_dir).unlink()
    config = CampaignConfig()
    config.runtime.failure_threshold_fraction = 0.0
    config.to_yaml(campaign / "campaign.yaml")
    write_ariadne_batch_decision(
        iter_dir,
        campaign_uid="reconcile-test",
        iteration=1,
        config_sha256=config_fingerprint(canonical_config(config)),
        failure_threshold_fraction=0.0,
        expected_n=2,
        n_accepted=1,
        n_rejected=1,
        accepted=False,
        reasons=["too_many_failures: 1/2"],
    )

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.ARIADNE_ARRAY
    assert "valid ARIADNE results handoff" not in report.decision


def test_reconcile_reuses_accepted_ariadne_results_when_rejected_outputs_absent(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.config import CampaignConfig

    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(
        recovery_contracts_mod,
        "verify_committed_model_version",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        reconcile_mod,
        "_validate_recovered_state_contract",
        lambda *a, **k: None,
    )
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    CampaignConfig().to_yaml(campaign / "campaign.yaml")
    state = fresh_campaign_state(max_iterations=3, campaign_uid="reconcile-test")
    state.phase = CampaignPhase.HALTED
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    append_event(
        data / "journal.ndjson",
        "halt",
        from_phase=CampaignPhase.PHASE_B_DIVERSITY.value,
        iteration=1,
        reason=(
            "backend_submission_failed: resource evidence not yet produced for "
            "PHASE_B_DIVERSITY"
        ),
    )
    _write_ariadne_handoff(campaign, 1, n=10, n_rejected=1)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.PHASE_B_DIVERSITY
    assert "valid ARIADNE results handoff" in report.decision
    assert report.ariadne_results_recovery == {
        "expected_tasks": 10,
        "accepted_tasks": 9,
        "rejected_tasks": 1,
        "missing_rejected_outputs": 1,
        "tasks_resubmitted": 0,
    }
    assert not report.unsafe_reasons


def test_reconcile_classifies_stale_ariadne_decision_as_cleanable(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(
        reconcile_mod,
        "_validate_recovered_state_contract",
        lambda *args, **kwargs: None,
    )
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=3, campaign_uid="reconcile-test")
    state.phase = CampaignPhase.HALTED
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    append_event(
        data / "journal.ndjson",
        "halt",
        from_phase=CampaignPhase.ARIADNE_ARRAY.value,
        iteration=1,
        reason="postprocess_exception: stale batch decision",
    )
    iteration_dir = _write_ariadne_handoff(campaign, 1, n=2)
    results_path = active_ariadne_dir(iteration_dir) / "RESULTS.json"
    results_path.write_text(
        results_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
        newline="\n",
    )

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert report.partial_array_recovery["n_reuse"] == 2
    assert report.partial_array_recovery["n_retry"] == 0
    assert report.ariadne_publication_recovery["state"] == "stale_results_binding"
    assert "stale ARIADNE publication" in report.unsafe_reasons


def test_propose_recovery_does_not_preserve_existing_phase_for_committed_iteration(
    tmp_path,
    monkeypatch,
):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0, 1])
    state = fresh_campaign_state(max_iterations=3)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = 1
    state.models_version = 1
    write_state(data / DEFAULT_STATE_FILENAME, state)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.STOP_CHECK
    assert report.proposed_state.iteration == 1
    assert "active iteration is fully committed" in report.decision


def test_propose_recovery_prefers_phase_b_over_stale_seed_select(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=3)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    _write_phase_b_handoff(campaign, 1, n=2)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.SPLIT
    assert report.proposed_state.iteration == 1
    assert "valid Phase B handoff" in report.decision


def test_phase_b_geometry_drift_requires_deep_reconcile_verification(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.handoff_manifests import phase_b_selection_path
    from ichor.hpc.active_learning.versioning.manifest import sha256_file

    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(recovery_contracts_mod, "verify_committed_model_version", lambda *a, **k: None)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    CampaignConfig().to_yaml(campaign / "campaign.yaml")
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=3, campaign_uid="reconcile-test")
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    iter_dir = _write_phase_b_handoff(campaign, 1, n=2)
    manifest_path = phase_b_selection_path(iter_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = (iter_dir / manifest["selected_xyz"]["path"]).resolve()
    lines = selected.read_text(encoding="utf-8").splitlines()
    atom = lines[2].split()
    atom[1] = str(float(atom[1]) + 0.25)
    lines[2] = " ".join(atom)
    selected.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest["selected_xyz"]["size"] = int(selected.stat().st_size)
    manifest["selected_xyz"]["sha256"] = sha256_file(selected)
    atomic_write_json(manifest_path, manifest)

    authority_report = propose_recovery(campaign)
    deep_report = propose_recovery(campaign, verification_level="deep")

    assert authority_report.proposed_state.phase is CampaignPhase.SPLIT
    assert "valid Phase B handoff" in authority_report.decision
    assert deep_report.proposed_state.phase is CampaignPhase.PHASE_B_DIVERSITY
    assert "valid Phase B handoff" not in deep_report.decision
    assert "valid ARIADNE results handoff" in deep_report.decision


def test_propose_recovery_prefers_split_over_stale_phase_b(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=3)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.PHASE_B_DIVERSITY
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    _write_phase_b_handoff(campaign, 1, n=2)
    _write_split(campaign, 1)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.GAUSSIAN
    assert report.proposed_state.iteration == 1
    assert "valid split handoff" in report.decision


def test_propose_recovery_invalid_split_reenters_split(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=3)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.SPLIT
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    _write_phase_b_handoff(campaign, 1, n=2)
    split_path = _write_split(campaign, 1)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    split_payload["slots"][0]["candidate_id"] = "wrong-candidate"
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.SPLIT
    assert report.proposed_state.iteration == 1
    assert "valid Phase B handoff" in report.decision


def test_propose_recovery_cross_iteration_partial_handoff_beats_stop_check(
    tmp_path,
    monkeypatch,
):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0, 1, 2])
    state = fresh_campaign_state(max_iterations=5)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.STOP_CHECK
    state.iteration = 7
    state.reference_data_version = 2
    state.models_version = 2
    write_state(data / DEFAULT_STATE_FILENAME, state)
    _write_seeds_picked(campaign, 3)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.ARIADNE_ARRAY
    assert report.proposed_state.iteration == 3
    assert "valid seed-selection handoff" in report.decision


def test_propose_recovery_protects_active_gaussian_handoff(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    mv = VersionedDirectory(models)
    _commit_reference_versions(campaign, (0,))
    s = mv.stage(None, 0)
    (s / "marker.txt").write_text("model", encoding="utf-8")
    mv.commit(0)
    state = fresh_campaign_state(max_iterations=3)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.HALTED
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    staging = campaign / ".DATA" / "STAGING" / "iter_1"
    pointdir = staging / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    stg.write_points_file(staging, [pointdir])
    stg.write_quantum_acceptance_manifest(
        staging,
        phase_name=CampaignPhase.GAUSSIAN.value,
        iteration=1,
        accepted=[pointdir],
        rejected=[],
    )
    _write_pending_allocation(
        campaign,
        context="active",
        iteration=1,
        n=1,
    )

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.AIMALL
    assert ".DATA/STAGING is non-empty" not in report.unsafe_reasons
    assert any("protected active staging handoff" in item for item in report.trusted_artifacts)


def test_propose_recovery_finds_staging_handoff_in_later_iteration(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0, 1])
    state = fresh_campaign_state(max_iterations=4)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.HALTED
    state.iteration = 2
    state.reference_data_version = 1
    state.models_version = 1
    write_state(data / DEFAULT_STATE_FILENAME, state)
    staging = campaign / ".DATA" / "STAGING" / "iter_2"
    pointdir = staging / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    stg.write_points_file(staging, [pointdir])
    stg.write_quantum_acceptance_manifest(
        staging,
        phase_name=CampaignPhase.AIMALL.value,
        iteration=2,
        accepted=[pointdir],
        rejected=[],
    )
    _complete_handoff_allocation(
        campaign,
        context="active",
        iteration=2,
        pointdirs=[pointdir],
    )

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.REFERENCE_COMMIT
    assert report.proposed_state.iteration == 2
    assert ".DATA/STAGING is non-empty" not in report.unsafe_reasons
    assert any("protected active staging handoff" in item for item in report.trusted_artifacts)


def test_propose_recovery_halts_on_multiple_valid_staging_handoffs(tmp_path, monkeypatch):
    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(reconcile_mod, "_validate_recovered_state_contract", lambda *a, **k: None)
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=4)
    state.campaign_uid = "reconcile-test"
    state.phase = CampaignPhase.HALTED
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    for iteration in (1, 2):
        staging = campaign / ".DATA" / "STAGING" / ("iter_" + str(iteration))
        pointdir = staging / "POINT_0000.pointdir"
        pointdir.mkdir(parents=True, exist_ok=True)
        stg.write_points_file(staging, [pointdir])
        stg.write_quantum_acceptance_manifest(
            staging,
            phase_name=CampaignPhase.GAUSSIAN.value,
            iteration=iteration,
            accepted=[pointdir],
            rejected=[],
        )

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert any("multiple valid staging handoffs" in r for r in report.unsafe_reasons)
    assert ".DATA/STAGING is non-empty" not in report.unsafe_reasons


def test_authority_recovery_defers_trajectory_pool_payload_drift_to_deep(tmp_path):
    campaign, _, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    mv = VersionedDirectory(models)
    _commit_reference_versions(campaign, (0,))
    s = mv.stage(None, 0)
    (s / "marker.txt").write_text("model", encoding="utf-8")
    mv.commit(0)

    pool_xyz = campaign / "pool.xyz"
    with pool_xyz.open("a", encoding="utf-8") as f:
        f.write("# drift\n")

    authority_report = propose_recovery(campaign)
    deep_report = propose_recovery(campaign, verification_level="deep")

    assert not any(
        "trajectory pool SHA mismatch" in reason
        for reason in authority_report.unsafe_reasons
    )
    assert deep_report.proposed_state.phase is CampaignPhase.HALTED
    assert any(
        "trajectory pool SHA mismatch" in reason
        for reason in deep_report.unsafe_reasons
    )
    assert "trajectory pool" in deep_report.blocking_artifacts


def test_deep_recovery_reports_unmanifested_committed_pointdir(tmp_path):
    campaign, _, training, _ = _campaign_dirs(tmp_path)
    _commit_reference_versions(campaign, (0,))
    v = VersionedDirectory(training)
    rogue = v.iteration_path(0) / "POINT_9999.pointdir"
    rogue.mkdir()
    (rogue / "input.gjf").write_text("%chk=x\n", encoding="utf-8")
    authority_report = propose_recovery(campaign)
    report = propose_recovery(campaign, verification_level="deep")
    assert authority_report.valid_reference_data_versions == [0]
    assert report.committed_reference_data_versions == [0]
    assert report.valid_reference_data_versions == []
    assert any("committed reference-data version 0" in r for r in report.unsafe_reasons)
    assert report.proposed_state.phase is CampaignPhase.HALTED


def test_propose_recovery_preserves_existing_campaign_uid(tmp_path):
    campaign, data, training, _ = _campaign_dirs(tmp_path)
    v = VersionedDirectory(training)
    s = v.stage(None, 0); (s / "x.txt").write_text("hi"); v.commit(0)
    existing = fresh_campaign_state(max_iterations=42)
    existing.iteration = 5
    existing.phase = CampaignPhase.STOP_CHECK
    existing.last_acquisition_alpha0 = 0.41
    write_state(data / DEFAULT_STATE_FILENAME, existing)
    report = propose_recovery(campaign)
    assert report.proposed_state.campaign_uid == existing.campaign_uid
    assert report.proposed_state.max_iterations == 42
    assert report.proposed_state.last_acquisition_alpha0 == pytest.approx(0.41)
    assert report.existing_state_loaded is True


def test_propose_recovery_clears_pending_jobs(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    existing = fresh_campaign_state()
    existing.pending_jobs = {"FEREBUS": "9999"}
    write_state(data / DEFAULT_STATE_FILENAME, existing)
    report = propose_recovery(campaign)
    assert report.proposed_state.pending_jobs == {}


def test_propose_recovery_clears_shutdown_request(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    existing = fresh_campaign_state()
    existing.shutdown_requested = True
    write_state(data / DEFAULT_STATE_FILENAME, existing)
    report = propose_recovery(campaign)
    assert report.proposed_state.shutdown_requested is False


def test_propose_recovery_reads_last_journal_transition(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    journal = data / "journal.ndjson"
    append_event(journal, "phase_transition", from_phase="INIT", to_phase="PHASE_A_DIVERSITY", iteration=0)
    append_event(journal, "phase_transition", from_phase="GAUSSIAN", to_phase="AIMALL", iteration=3)
    report = propose_recovery(campaign)
    assert report.last_phase_in_journal == "AIMALL"
    assert report.last_iteration_in_journal == 3
    assert report.last_phase_event_in_journal == "phase_transition"
    assert report.last_phase_retryable is False


def test_propose_recovery_marks_halt_journal_phase_retryable(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    journal = data / "journal.ndjson"
    append_event(journal, "phase_succeeded", phase="FEREBUS", iteration=2)
    append_event(journal, "halt", from_phase="PHASE_B_DIVERSITY", iteration=3)

    report = propose_recovery(campaign)

    assert report.last_phase_in_journal == "PHASE_B_DIVERSITY"
    assert report.last_iteration_in_journal == 3
    assert report.last_phase_event_in_journal == "halt"
    assert report.last_phase_retryable is True


def test_propose_recovery_corrupt_state_does_not_crash(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    (data / DEFAULT_STATE_FILENAME).write_text("{not valid json")
    report = propose_recovery(campaign)
    # Falls through to a fresh state since existing did not load.
    assert report.existing_state_loaded is False
    assert any("failed validation" in n or "unreadable" in n for n in report.notes)


def test_propose_recovery_recovers_uid_from_valid_committed_data_after_non_json_state(
    tmp_path,
):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_reference_versions(campaign, (0,))
    (data / DEFAULT_STATE_FILENAME).write_text("{not valid json", encoding="utf-8")

    report = propose_recovery(campaign)

    assert report.proposed_state.campaign_uid == "reconcile-test"
    assert report.proposed_state.phase is CampaignPhase.INITIAL_FEREBUS
    assert not any(
        "no recoverable trusted campaign_uid" in reason
        for reason in report.unsafe_reasons
    )


def test_propose_recovery_refuses_state_uid_disagreement_with_committed_data(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_reference_versions(campaign, (0,))
    state = fresh_campaign_state(campaign_uid="different-campaign")
    state.phase = CampaignPhase.HALTED
    state.reference_data_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert any(
        "state campaign_uid disagrees with trusted artefacts" in reason
        for reason in report.unsafe_reasons
    )


def test_reconcile_preserves_completed_lifecycle_until_explicit_reopen(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon.state import make_lifecycle_context

    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(
        reconcile_mod,
        "verify_state_referenced_artifacts",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        reconcile_mod,
        "_validate_recovered_state_contract",
        lambda *a, **k: None,
    )
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=1, campaign_uid="reconcile-test")
    state.phase = CampaignPhase.DONE
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    state.lifecycle_context = make_lifecycle_context(
        disposition="completed",
        reason_code="max_iterations_reached",
        message="campaign complete",
        from_phase=CampaignPhase.STOP_CHECK,
        iteration=1,
        source="daemon",
    )
    write_state(data / DEFAULT_STATE_FILENAME, state)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.DONE
    assert report.proposed_state.lifecycle_context["disposition"] == "completed"
    assert "only resume --reopen-converged" in " ".join(report.notes)


def test_reconcile_preserves_user_stop_until_resume(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.daemon.state import make_lifecycle_context

    _trust_marker_models(monkeypatch)
    monkeypatch.setattr(
        reconcile_mod,
        "verify_state_referenced_artifacts",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        reconcile_mod,
        "_validate_recovered_state_contract",
        lambda *a, **k: None,
    )
    campaign, data, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    _commit_training_and_model_versions(training, models, [0])
    state = fresh_campaign_state(max_iterations=3, campaign_uid="reconcile-test")
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    state.shutdown_requested = True
    state.lifecycle_context = make_lifecycle_context(
        disposition="stopped",
        reason_code="user_stop_request",
        message="operator requested stop",
        from_phase=CampaignPhase.SEED_SELECT,
        iteration=1,
        source="operator",
    )
    write_state(data / DEFAULT_STATE_FILENAME, state)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.SEED_SELECT
    assert report.proposed_state.shutdown_requested is True
    assert report.proposed_state.lifecycle_context["disposition"] == "stopped"
    assert "only resume may clear it" in " ".join(report.notes)


def test_stop_control_makes_missing_state_a_recovery_case(tmp_path):
    from ichor.hpc.active_learning.daemon.reconcile import stateful_campaign_artifacts
    from ichor.hpc.active_learning.daemon.stop_control import (
        build_stop_request,
        install_stop_request,
    )

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    state = fresh_campaign_state(max_iterations=2, campaign_uid="stop-control-test")
    install_stop_request(campaign, build_stop_request(state, mode="immediate"))

    assert str(Path(".DATA") / "ACTIVE_LEARNING" / "stop_request.json") in (
        stateful_campaign_artifacts(campaign)
    )


def test_write_proposed_state_creates_proposed_file(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    report = propose_recovery(campaign)
    target = write_proposed_state(campaign, report)
    assert target.exists()
    assert target.name == DEFAULT_STATE_FILENAME + RECONCILE_SUFFIX
    payload = read_state(target)
    assert payload.phase == report.proposed_state.phase
    # Does NOT clobber the live state.json.
    assert not (data / DEFAULT_STATE_FILENAME).exists()


# --- M15 F16: propose_recovery preserves M13/M14/M15 fields ------------


def test_propose_recovery_preserves_reference_scales_cache(tmp_path):
    """M13's reference_scales + reference_scales_iteration must survive
    a reconcile pass; otherwise STOP_CHECK alpha-trend gets reset and the
    cached scales get recomputed regardless of refresh policy."""
    from ichor.hpc.active_learning.daemon.reconcile import propose_recovery
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, CampaignState, fresh_campaign_state, write_state,
    )
    cd = tmp_path / "campaign"
    (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    state = fresh_campaign_state(max_iterations=20)
    state.iteration = 7
    state.phase = CampaignPhase.STOP_CHECK
    state.reference_scales = {"energy": 1.0e-3, "force": 1.0e-2, "omega": 1.0, "anh": 1.0, "anh_std": 1.0}
    state.reference_scales_iteration = 7
    state.reference_scales_models_version = 6
    state.reference_scales_model_manifest_sha256 = "a" * 64
    write_state(cd / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    report = propose_recovery(cd)
    proposal = report.proposed_state
    assert proposal.reference_scales == {"energy": 1.0e-3, "force": 1.0e-2, "omega": 1.0, "anh": 1.0, "anh_std": 1.0}
    assert proposal.reference_scales_iteration == 7
    assert proposal.reference_scales_models_version == 6
    assert proposal.reference_scales_model_manifest_sha256 == "a" * 64


def test_propose_recovery_preserves_alpha_history(tmp_path):
    """M14 alpha-trend history must survive reconcile."""
    from ichor.hpc.active_learning.daemon.reconcile import propose_recovery
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, fresh_campaign_state, write_state,
    )
    cd = tmp_path / "campaign"
    (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    state = fresh_campaign_state(max_iterations=20)
    state.iteration = 12
    state.phase = CampaignPhase.STOP_CHECK
    state.alpha_history = [0.1, 0.08, 0.05, 0.03, 0.02]
    write_state(cd / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    report = propose_recovery(cd)
    proposal = report.proposed_state
    assert proposal.alpha_history == [0.1, 0.08, 0.05, 0.03, 0.02]


def test_propose_recovery_preserves_M15_diagnostic_fields(tmp_path):
    """M15 F3 last_n_anti_overlap_flagged + F6 sacct_empty_streak must
    survive reconcile too -- otherwise the daemon would lose the in-flight
    sacct timeout state on every recovery."""
    from ichor.hpc.active_learning.daemon.reconcile import propose_recovery
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, fresh_campaign_state, write_state,
    )
    cd = tmp_path / "campaign"
    (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    state = fresh_campaign_state(max_iterations=20)
    state.iteration = 3
    state.phase = CampaignPhase.SEED_SELECT
    state.last_n_anti_overlap_flagged = 7
    state.sacct_empty_streak = {"99999": 4, "11111": 1}
    write_state(cd / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    report = propose_recovery(cd)
    proposal = report.proposed_state
    assert proposal.last_n_anti_overlap_flagged == 7
    assert proposal.sacct_empty_streak == {"99999": 4, "11111": 1}
