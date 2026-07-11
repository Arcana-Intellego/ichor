"""Tests for the Phase-B anti-overlap consumer (case d).

We exercise the full integration: polus_wrapper.main runs Phase B,
invokes filter_candidates_against_training with the geometry novelty-derived
minimum separation,
and writes both raw + filtered samples plus the DedupReport.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    DEFAULT_STATE_FILENAME,
    fresh_campaign_state,
    write_state,
)


MODULE = "ichor.hpc.active_learning.sampling.polus_wrapper"


def _make_seed_result(seed_dir, atom_types, final_coords):
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_id = int(seed_dir.name.split("-")[-1])
    payload = {
        "atom_types": atom_types,
        "final_coordinates": [list(c) for c in final_coords],
        "alpha_trajectory": [0.0, 1.0],
        "alpha_initial": 0.0,
        "alpha_final": 1.0,
        "n_evaluations": 1,
        "return_code": 0,
        "wall_seconds": 1.0,
        "fell_back_to_ds": False,
        "whitened_distance_final": 0.5,
        "iteration": 1,
        "seed_id": seed_id,
        "seed_frame_id": seed_id - 1,
        "landing_safety": {
            "accepted": True,
            "policy": "raw_final",
            "selected_origin": "raw_final",
            "selected_candidate_index": 0,
            "reasons": [],
            "record_only_reasons": [],
            "metrics": {"whitened_distance": 0.5},
            "raw_final": {},
            "n_candidates_evaluated": 1,
            "n_safe_candidates": 1,
        },
    }
    (seed_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_ariadne_manifest(iter_dir):
    from ichor.hpc.active_learning.ariadne_outputs import (
        write_optimisation_trajectory,
        write_seed_output_manifest,
    )
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.handoff_manifests import (
        ARIADNE_RESULTS_SCHEMA_VERSION,
        seeds_picked_path,
        write_ariadne_results_manifest,
    )
    from ichor.hpc.active_learning.layout import active_ariadne_dir
    from ichor.hpc.active_learning.seed_identity import (
        deterministic_seed_uid,
        selection_fingerprint_sha256,
        write_ariadne_task_map,
    )
    from ichor.hpc.active_learning.versioning.manifest import sha256_file
    from ichor.hpc.active_learning.versioning.provenance import (
        PROVENANCE_FILENAME,
        write_seed_provenance,
    )

    ariadne_root = active_ariadne_dir(iter_dir)
    seed_dirs = sorted((ariadne_root / "seeds").glob("seed-*"))
    selection = {
        "schema_version": 2,
        "campaign_uid": "test",
        "iteration": 1,
        "models_version": 0,
        "model_manifest_sha256": "c" * 64,
        "trajectory_sha256": "0" * 64,
        "selection_strategy": "hybrid_variance",
        "n_picked": len(seed_dirs),
        "seed_records": [
            {
                "seed_id": seed_id,
                "frame_id": seed_id - 1,
                "pool_row_index_zero_based": seed_id - 1,
                "selection_origin": "bulk",
                "variance_at_selection": 0.0,
            }
            for seed_id in range(1, len(seed_dirs) + 1)
        ],
    }
    fingerprint = selection_fingerprint_sha256(selection)
    selection["selection_fingerprint_sha256"] = fingerprint
    for record in selection["seed_records"]:
        record["seed_uid"] = deterministic_seed_uid(
            campaign_uid="test",
            iteration=1,
            seed_id=int(record["seed_id"]),
            frame_id=int(record["frame_id"]),
            models_version=0,
            model_manifest_sha256="c" * 64,
            selection_fingerprint_sha256_value=fingerprint,
        )
    selection_path = seeds_picked_path(iter_dir)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, selection)
    task_map_path = write_ariadne_task_map(iter_dir, selection)

    accepted = []
    for array_task_id, seed_dir in enumerate(seed_dirs):
        seed_id = array_task_id + 1
        seed_uid = str(selection["seed_records"][array_task_id]["seed_uid"])
        result_path = seed_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.update({
            "iteration": 1,
            "seed_id": seed_id,
            "seed_uid": seed_uid,
            "array_task_id": array_task_id,
            "seed_frame_id": array_task_id,
            "trajectory_sha256": "0" * 64,
        })
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
            campaign_uid="test",
            iteration=1,
            seed_id=seed_id,
            seed_uid=seed_uid,
            array_task_id=array_task_id,
            task_success=True,
            task_exit_code=0,
        )
        prov_path = seed_dir / PROVENANCE_FILENAME
        write_seed_provenance(
            seed_dir,
            campaign_uid="test",
            iteration=1,
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
            mode_weighting_policy="variance",
        )
        accepted.append({
            "seed_id": seed_id,
            "seed_uid": seed_uid,
            "array_task_id": array_task_id,
            "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
            "result_json": result_path.relative_to(ariadne_root).as_posix(),
            "provenance_json": prov_path.relative_to(ariadne_root).as_posix(),
            "output_manifest": output_manifest.relative_to(ariadne_root).as_posix(),
            "seed_frame_id": array_task_id,
            "pool_row_index_zero_based": array_task_id,
            "selection_origin": "bulk",
            "variance_at_selection": 0.0,
            "alpha_initial": float(result["alpha_initial"]),
            "alpha_final": float(result["alpha_final"]),
            "whitened_distance_final": float(result["whitened_distance_final"]),
            "return_code": 0,
            "landing_safety": dict(result["landing_safety"]),
            "result_sha256": sha256_file(result_path),
            "provenance_sha256": sha256_file(prov_path),
            "output_manifest_sha256": sha256_file(output_manifest),
        })
    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "campaign_uid": "test",
        "iteration": 1,
        "trajectory_sha256": "0" * 64,
        "task_map": {
            "path": task_map_path.relative_to(ariadne_root).as_posix(),
            "sha256": sha256_file(task_map_path),
        },
        "expected_n": len(accepted),
        "n_accepted": len(accepted),
        "n_rejected": 0,
        "accepted": accepted,
        "rejected": [],
    })
    from ichor.hpc.active_learning.daemon.config_lock import (
        canonical_config,
        config_fingerprint,
    )
    from ichor.hpc.active_learning.handoff_manifests import (
        write_ariadne_batch_decision,
    )

    campaign = iter_dir.parent.parent
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    write_ariadne_batch_decision(
        iter_dir,
        campaign_uid="test",
        iteration=1,
        config_sha256=config_fingerprint(canonical_config(config)),
        failure_threshold_fraction=float(config.runtime.failure_threshold_fraction),
        expected_n=len(accepted),
        n_accepted=len(accepted),
        n_rejected=0,
        accepted=True,
        reasons=[],
    )
    from ichor.hpc.active_learning.sampling_protocol import (
        resolve_sampling_protocol,
    )

    resolve_sampling_protocol(
        campaign,
        CampaignConfig.from_yaml(campaign / "campaign.yaml"),
        iteration=1,
    )


def _write_pointdir(path, atom_types, coords):
    """Write a minimal-but-readable .pointdir directory."""
    path.mkdir(parents=True, exist_ok=True)
    natoms = len(atom_types)
    xyz_lines = [str(natoms), "training point"]
    for sym, c in zip(atom_types, coords):
        xyz_lines.append(
            "{0} {1:.6f} {2:.6f} {3:.6f}".format(
                sym, float(c[0]), float(c[1]), float(c[2]),
            )
        )
    (path / (path.name.replace(".pointdir", "") + ".xyz")).write_text(
        chr(10).join(xyz_lines) + chr(10), encoding="utf-8",
    )


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", MODULE] + args,
        capture_output=True, text=True, timeout=60,
    )


def _write_campaign_state(campaign):
    state = fresh_campaign_state()
    state.campaign_uid = "test"
    state.iteration = 1
    state.phase = CampaignPhase.PHASE_B_POLUS
    state.reference_data_version = 0
    state.models_version = 0
    data_dir = campaign / ".DATA" / "ACTIVE_LEARNING"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_state(data_dir / DEFAULT_STATE_FILENAME, state)


def _commit_reference_point(campaign, atom_types, coords):
    from ichor.hpc.active_learning.daemon.input_staging import (
        commit_reference_data_delta,
    )
    from ichor.hpc.active_learning.point_allocation import (
        create_point_allocation,
        pending_attempts,
        point_allocation_path,
        record_quantum_results,
    )
    from ichor.hpc.active_learning.versioning.provenance import (
        enrich_with_point_allocation,
        write_seed_provenance,
    )

    allocation_path = point_allocation_path(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="test",
        context="bootstrap",
        iteration=0,
        targets={"train": 1, "int_val": 0, "ext_val": 0, "total": 1},
        primary_candidates=[
            {
                "candidate_id": "reference-candidate-0",
                "frame_id": 0,
                "pointdir_name": "POINT_0000.pointdir",
            }
        ],
        reserve_candidates=[],
    )
    attempt = pending_attempts(allocation)[0]
    pointdir = campaign / ".DATA" / "STAGING" / "initial" / "POINT_0000.pointdir"
    _write_pointdir(pointdir, atom_types, coords)
    write_seed_provenance(
        pointdir,
        campaign_uid="test",
        iteration=0,
        trajectory_sha256="0" * 64,
        seed_frame_id=0,
        seed_selection_origin="bootstrap",
        seed_variance_at_selection=None,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
    )
    enrich_with_point_allocation(
        pointdir,
        candidate_id=str(attempt["candidate_id"]),
        context="bootstrap",
        slot_id=int(attempt["slot_id"]),
        split=str(attempt["split"]),
        replacement_round=int(attempt.get("round", 0)),
        allocation_slot_assignment_sha256=str(
            allocation["slot_assignment_sha256"]
        ),
    )
    record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": str(attempt["candidate_id"]),
                "accepted": True,
                "pointdir": str(pointdir),
            }
        ],
    )
    commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )


def test_min_separation_zero_drops_nothing(tmp_path):
    """Default min_separation=0 means the filter passes everything."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.point_allocation.batch_training_size = 2
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")
    _write_campaign_state(campaign)
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    pool_dir = iter_dir / "ariadne" / "seeds"
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    geometries = [
        base,
        [(0.0, 0.0, 0.0), (1.20, 0.0, 0.0), (-0.24, 0.93, 0.0)],
        [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.55, 1.20, 0.0)],
    ]
    for seed_id, coords in enumerate(geometries, start=1):
        _make_seed_result(pool_dir / f"seed-{seed_id:06d}", atom_types, coords)
    _write_ariadne_manifest(iter_dir)
    rc = _run(["--descriptor", "rmsd_massweight",
               "--iteration", "1",
               "--campaign-dir", str(campaign)])
    assert rc.returncode == 0, rc.stderr
    d = json.loads(
        (iter_dir / "phase_b" / "SELECTION.json").read_text(encoding="utf-8")
    )["dedup"]
    assert d["n_kept"] == 3
    assert d["n_dropped"] == 0


def test_min_separation_underfills_after_dropping_close_candidates(tmp_path):
    """With a training point coincident with one of the candidates
    and a positive derived minimum separation, Phase B must fail if the remaining
    candidates cannot still fill the configured active batch."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.point_allocation.batch_training_size = 2
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.geometry_novelty.fallback_scale_angstrom = 0.2
    cfg.to_yaml(campaign / "campaign.yaml")
    _write_campaign_state(campaign)

    # Commit one reference point coincident with seed-000001.
    _commit_reference_point(
        campaign,
        ["O", "H", "H"],
        [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )

    # Three candidates: seed-000001 is coincident with training.
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    pool_dir = iter_dir / "ariadne" / "seeds"
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    _make_seed_result(pool_dir / "seed-000001", atom_types, base)
    _make_seed_result(
        pool_dir / "seed-000002", atom_types,
        [(0.0, 0.0, 0.0), (1.60, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )
    _make_seed_result(
        pool_dir / "seed-000003", atom_types,
        [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.70, 1.35, 0.0)],
    )
    _write_ariadne_manifest(iter_dir)
    rc = _run(["--descriptor", "rmsd_massweight",
               "--iteration", "1",
               "--campaign-dir", str(campaign)])
    assert rc.returncode == 3
    assert "phase_b_point_allocation_underfilled_after_anti_overlap" in rc.stderr
    failure = json.loads(
        (iter_dir / "phase_b" / "SELECTION.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["failure_reason"] == "point_allocation_underfilled_after_anti_overlap"
    assert failure["dedup"]["n_dropped"] >= 1
    assert 0.0 < float(failure["dedup"]["min_separation"]) <= 0.2


def test_min_separation_all_candidates_removed_fails_at_phase_b(tmp_path):
    """If anti-overlap removes every selected candidate, Phase B must not
    publish a successful empty selection manifest for the next phase to reject.
    """
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.point_allocation.batch_training_size = 2
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.geometry_novelty.fallback_scale_angstrom = 0.2
    cfg.to_yaml(campaign / "campaign.yaml")
    _write_campaign_state(campaign)

    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    atom_types = ["O", "H", "H"]
    _commit_reference_point(campaign, atom_types, base)

    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    pool_dir = iter_dir / "ariadne" / "seeds"
    for seed_id in range(1, 4):
        # Pure translations are removed by the aligned RMSD metric, so all
        # three candidates overlap the committed training point.
        coords = [(c[0] + float(seed_id - 1), c[1], c[2]) for c in base]
        _make_seed_result(pool_dir / f"seed-{seed_id:06d}", atom_types, coords)
    _write_ariadne_manifest(iter_dir)

    rc = _run(["--descriptor", "rmsd_massweight",
               "--iteration", "1",
               "--campaign-dir", str(campaign)])

    assert rc.returncode == 3
    assert "phase_b_geometry_novelty_no_non_duplicate_candidate" in rc.stderr
    failure = json.loads(
        (iter_dir / "phase_b" / "SELECTION.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["failure_reason"] == "no_non_duplicate_candidate"
    assert failure["dedup"]["n_kept"] == 0


def test_scaled_min_separation_rescues_farthest_non_duplicate(tmp_path):
    """Scaled mode should keep one farthest non-duplicate candidate instead
    of failing the campaign when the estimated threshold is too strict.
    """
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 0
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.geometry_novelty.fallback_scale_angstrom = 10.0

    cfg.to_yaml(campaign / "campaign.yaml")

    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    atom_types = ["O", "H", "H"]
    _commit_reference_point(campaign, atom_types, base)

    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    pool_dir = iter_dir / "ariadne" / "seeds"
    _make_seed_result(
        pool_dir / "seed-000001",
        atom_types,
        [(0.0, 0.0, 0.0), (0.98, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )
    _make_seed_result(
        pool_dir / "seed-000002",
        atom_types,
        [(0.0, 0.0, 0.0), (1.05, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )
    _make_seed_result(
        pool_dir / "seed-000003",
        atom_types,
        [(0.0, 0.0, 0.0), (1.00, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )
    _write_ariadne_manifest(iter_dir)
    _write_campaign_state(campaign)

    rc = _run(["--descriptor", "rmsd_massweight",
               "--iteration", "1",
               "--campaign-dir", str(campaign)])

    assert rc.returncode == 0, rc.stderr
    dedup = json.loads(
        (iter_dir / "phase_b" / "SELECTION.json").read_text(encoding="utf-8")
    )["dedup"]
    assert dedup["threshold_mode"] == "scaled"
    assert dedup["relaxation"]["applied"] is True
    assert dedup["n_kept"] == 1
    assert dedup["n_dropped"] == 2
    assert (iter_dir / "protocol" / "GEOMETRY_NOVELTY_SCALE.json").is_file()
    journal = campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    events = [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event.get("event") == "phase_b_geometry_novelty_relaxed"
        for event in events
    )
