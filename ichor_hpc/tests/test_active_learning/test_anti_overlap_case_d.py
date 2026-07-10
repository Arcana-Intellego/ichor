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
    DEFAULT_STATE_FILENAME,
    fresh_campaign_state,
    write_state,
)


MODULE = "ichor.hpc.active_learning.sampling.polus_wrapper"


def _make_seed_result(seed_dir, atom_types, final_coords):
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_index = int(seed_dir.name.split("_")[-1])
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
        "iteration": 0,
        "seed_index": seed_index,
        "seed_frame_id": seed_index,
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
    from ichor.hpc.active_learning.handoff_manifests import (
        ARIADNE_RESULTS_SCHEMA_VERSION,
        write_ariadne_results_manifest,
    )
    from ichor.hpc.active_learning.versioning.provenance import (
        PROVENANCE_FILENAME,
        write_seed_provenance,
    )

    accepted = []
    for seed_dir in sorted((iter_dir / "pool").glob("seed_*")):
        seed_index = int(seed_dir.name.split("_")[-1])
        result_path = seed_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        prov_path = seed_dir / PROVENANCE_FILENAME
        write_seed_provenance(
            seed_dir,
            campaign_uid="test",
            iteration=0,
            trajectory_sha256="0" * 64,
            seed_frame_id=seed_index,
            seed_selection_origin="bulk",
            seed_variance_at_selection=0.0,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
            mode_weighting_policy="variance",
        )
        accepted.append({
            "seed_index": seed_index,
            "seed_dir": str(seed_dir.resolve()),
            "result_json": str(result_path.resolve()),
            "provenance_json": str(prov_path.resolve()),
            "seed_frame_id": seed_index,
            "selection_index": seed_index,
            "selection_origin": "bulk",
            "variance_at_selection": 0.0,
            "alpha_initial": float(result["alpha_initial"]),
            "alpha_final": float(result["alpha_final"]),
            "whitened_distance_final": float(result["whitened_distance_final"]),
            "return_code": 0,
            "landing_safety": dict(result["landing_safety"]),
        })
    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "iteration": 0,
        "trajectory_sha256": "0" * 64,
        "expected_n": len(accepted),
        "n_accepted": len(accepted),
        "n_rejected": 0,
        "accepted": accepted,
        "rejected": [],
    })


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
    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    pool_dir = iter_dir / "pool"
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    geometries = [
        base,
        [(0.0, 0.0, 0.0), (1.20, 0.0, 0.0), (-0.24, 0.93, 0.0)],
        [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.55, 1.20, 0.0)],
    ]
    for i, coords in enumerate(geometries):
        _make_seed_result(pool_dir / f"seed_{i:04d}", atom_types, coords)
    _write_ariadne_manifest(iter_dir)
    rc = _run(["--descriptor", "rmsd_massweight",
               "--iteration", "0",
               "--campaign-dir", str(campaign)])
    assert rc.returncode == 0, rc.stderr
    dedup = iter_dir / "phase_b_dedup.json"
    d = json.loads(dedup.read_text(encoding="utf-8"))
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

    # Commit one reference point coincident with seed_0000.
    _commit_reference_point(
        campaign,
        ["O", "H", "H"],
        [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )

    # 3 candidates: seed_0000 coincident with training, others spread apart.
    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    pool_dir = iter_dir / "pool"
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    _make_seed_result(pool_dir / "seed_0000", atom_types, base)
    _make_seed_result(
        pool_dir / "seed_0001", atom_types,
        [(0.0, 0.0, 0.0), (1.60, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )
    _make_seed_result(
        pool_dir / "seed_0002", atom_types,
        [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.70, 1.35, 0.0)],
    )
    _write_ariadne_manifest(iter_dir)
    rc = _run(["--descriptor", "rmsd_massweight",
               "--iteration", "0",
               "--campaign-dir", str(campaign)])
    assert rc.returncode == 3
    assert "phase_b_point_allocation_underfilled_after_anti_overlap" in rc.stderr
    dedup = iter_dir / "phase_b_dedup.json"
    d = json.loads(dedup.read_text(encoding="utf-8"))
    # at least one candidate should have been dropped (the duplicate).
    assert d["n_dropped"] >= 1
    assert 0.0 < float(d["min_separation"]) <= 0.2
    assert not (iter_dir / "PHASE_B_SELECTION.json").exists()


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

    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    pool_dir = iter_dir / "pool"
    for i in range(3):
        # Pure translations are removed by the aligned RMSD metric, so all
        # three candidates overlap the committed training point.
        coords = [(c[0] + float(i), c[1], c[2]) for c in base]
        _make_seed_result(pool_dir / f"seed_{i:04d}", atom_types, coords)
    _write_ariadne_manifest(iter_dir)

    rc = _run(["--descriptor", "rmsd_massweight",
               "--iteration", "0",
               "--campaign-dir", str(campaign)])

    assert rc.returncode == 3
    assert "phase_b_geometry_novelty_no_non_duplicate_candidate" in rc.stderr
    assert (iter_dir / "phase_b_dedup.json").is_file()
    assert not (iter_dir / "PHASE_B_SELECTION.json").exists()


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

    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    pool_dir = iter_dir / "pool"
    _make_seed_result(
        pool_dir / "seed_0000",
        atom_types,
        [(0.0, 0.0, 0.0), (0.98, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )
    _make_seed_result(
        pool_dir / "seed_0001",
        atom_types,
        [(0.0, 0.0, 0.0), (1.05, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )
    _make_seed_result(
        pool_dir / "seed_0002",
        atom_types,
        [(0.0, 0.0, 0.0), (1.00, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )
    _write_ariadne_manifest(iter_dir)
    _write_campaign_state(campaign)

    rc = _run(["--descriptor", "rmsd_massweight",
               "--iteration", "0",
               "--campaign-dir", str(campaign)])

    assert rc.returncode == 0, rc.stderr
    dedup = json.loads((iter_dir / "phase_b_dedup.json").read_text(encoding="utf-8"))
    assert dedup["threshold_mode"] == "scaled"
    assert dedup["relaxation"]["applied"] is True
    assert dedup["n_kept"] == 1
    assert dedup["n_dropped"] == 2
    assert (iter_dir / "GEOMETRY_NOVELTY_SCALE.json").is_file()
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
