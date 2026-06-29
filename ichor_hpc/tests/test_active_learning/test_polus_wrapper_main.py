"""Tests for the polus_wrapper main(argv) Phase A + Phase B bodies.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.handoff_manifests import PHASE_A_SAMPLE_FILENAME


MODULE = "ichor.hpc.active_learning.sampling.polus_wrapper"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "water_tetramer.xyz"


def _import_pool(campaign, source):
    """Drop a trajectory into the campaign-canonical location."""
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
    TrajectoryPool.import_from(source, campaign, overwrite=True, outlier_filter_enabled=False)


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", MODULE] + args,
        capture_output=True, text=True, timeout=120,
    )


def test_phase_a_writes_sample_and_index(tmp_path):
    """Phase A reads the trajectory pool and writes the canonical
    initial-SAMPLE-N.xyz + initial-INDEX-N.dat files."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.initial_train_size = 5
    cfg.initial_val_size = 2
    cfg.to_yaml(campaign / "campaign.yaml")
    _import_pool(campaign, FIXTURE)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "-1",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 0, result.stderr

    outdir = campaign / "3_DIVERSITY_SAMPLING" / "initial"
    samples = list(outdir.glob("initial-SAMPLE-*.xyz"))
    indices = list(outdir.glob("initial-INDEX-*.dat"))
    assert len(samples) == 1
    assert len(indices) == 1
    # n_select = 5 + 2 = 7; we expect 7 entries in INDEX.dat
    idx_lines = indices[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(idx_lines) == 7
    for ln in idx_lines:
        assert ln.strip().isdigit()
    manifest = json.loads((outdir / PHASE_A_SAMPLE_FILENAME).read_text(encoding="utf-8"))
    assert manifest["phase"] == "PHASE_A_POLUS"
    assert manifest["iteration"] == -1
    assert manifest["n_select"] == 7
    assert manifest["n_frames"] == 7
    assert manifest["sample_xyz"].endswith(samples[0].name)
    assert manifest["index_path"].endswith(indices[0].name)
    assert len(manifest["selected_indices"]) == 7
    assert manifest["trajectory_sha256"]


def test_phase_a_caps_at_pool_size(tmp_path):
    """If initial_train + initial_val exceed the pool, we cap at the
    pool size rather than failing."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.initial_train_size = 100
    cfg.initial_val_size = 100
    cfg.to_yaml(campaign / "campaign.yaml")
    _import_pool(campaign, FIXTURE)
    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "-1",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 0, result.stderr
    # fixture water_tetramer.xyz has 20 frames -- we cap at that.
    outdir = campaign / "3_DIVERSITY_SAMPLING" / "initial"
    sample_files = list(outdir.glob("initial-SAMPLE-*.xyz"))
    assert len(sample_files) == 1
    assert "-SAMPLE-20." in sample_files[0].name
    manifest = json.loads((outdir / PHASE_A_SAMPLE_FILENAME).read_text(encoding="utf-8"))
    assert manifest["n_select"] == 20
    assert len(manifest["selected_indices"]) == 20


def _make_seed_result(seed_dir, atom_types, final_coords, alpha_final=1.0):
    """Drop a synthetic result.json into a per-seed pool dir."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_index = int(seed_dir.name.split("_")[-1])
    payload = {
        "atom_types": atom_types,
        "final_coordinates": [list(c) for c in final_coords],
        "alpha_trajectory": [0.0, float(alpha_final)],
        "alpha_initial": 0.0,
        "alpha_final": float(alpha_final),
        "n_evaluations": 1,
        "return_code": 0,
        "wall_seconds": 1.0,
        "fell_back_to_ds": False,
        "whitened_distance_final": 0.5,
        "iteration": 0,
        "seed_index": seed_index,
        "seed_frame_id": seed_index,
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
    pool_dir = iter_dir / "pool"
    for seed_dir in sorted(pool_dir.glob("seed_*")):
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
        rec = {
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
        }
        if isinstance(result.get("landing_safety"), dict):
            rec["landing_safety"] = dict(result["landing_safety"])
        accepted.append(rec)
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


def _set_landing_safety(seed_dir, accepted, reasons=None, policy="raw_final"):
    result_path = seed_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["landing_safety"] = {
        "accepted": bool(accepted),
        "policy": str(policy),
        "selected_origin": str(policy),
        "selected_candidate_index": 0,
        "reasons": [str(r) for r in (reasons or [])],
        "record_only_reasons": [],
        "metrics": {"whitened_distance": 0.5},
        "raw_final": {},
        "n_candidates_evaluated": 1,
        "n_safe_candidates": 1 if accepted else 0,
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase_b_writes_sample_and_dedup(tmp_path):
    """Phase B reads the per-seed result.json files, runs FPS, then
    runs the (optional) anti-overlap pass and writes both raw + final"""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.batch_sizing.floor = 2
    cfg.batch_sizing.cap = 4
    # synthetic geometries lack ALF assignment so default hybrid_alf_rmsd
    # would crash on the feature extractor. mass-weighted RMSD is the
    # safe choice for unit tests.
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")
    # build 5 candidate seeds with distinct geometries.
    iter_dir = (
        campaign / "7_ACTIVE_LEARNING"
        / "iteration-0000"
    )
    pool_dir = iter_dir / "pool"
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    for i in range(5):
        # spread the seeds out in x
        coords = [(c[0] + 0.1 * i, c[1], c[2]) for c in base]
        seed_dir = pool_dir / f"seed_{i:04d}"
        _make_seed_result(seed_dir, atom_types, coords)
        _set_landing_safety(seed_dir, accepted=True)
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 0, result.stderr
    raw = iter_dir / "phase_b_SAMPLE_raw.xyz"
    final = iter_dir / "phase_b_SAMPLE.xyz"
    dedup = iter_dir / "phase_b_dedup.json"
    assert raw.is_file()
    assert final.is_file()
    assert dedup.is_file()
    d = json.loads(dedup.read_text(encoding="utf-8"))
    # fresh campaign, no committed training set to dedup against, so case (d) drops nothing
    # regardless of the (now 0.05 by default) min_separation -- see A43.
    assert d["n_dropped"] == 0
    assert d["min_separation"] == 0.05


def test_phase_b_rejects_unsafe_accepted_landing_before_fps(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.batch_sizing.floor = 3
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")

    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    pool_dir = iter_dir / "pool"
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    for i in range(3):
        coords = [(c[0] + 0.4 * i, c[1], c[2]) for c in base]
        seed_dir = pool_dir / f"seed_{i:04d}"
        _make_seed_result(seed_dir, atom_types, coords)
        _set_landing_safety(seed_dir, accepted=(i != 1),
                            reasons=["no_safe_non_seed_landing"] if i == 1 else [])
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 3
    assert "no_safe_non_seed_landing" in result.stderr
    assert not (iter_dir / "phase_b_SAMPLE_raw.xyz").exists()


def test_phase_b_rejects_partial_missing_landing_safety(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")

    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    pool_dir = iter_dir / "pool"
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    _make_seed_result(pool_dir / "seed_0000", atom_types, base)
    _set_landing_safety(pool_dir / "seed_0000", accepted=True)
    _make_seed_result(
        pool_dir / "seed_0001", atom_types,
        [(c[0] + 0.4, c[1], c[2]) for c in base],
    )
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 3
    assert "missing_landing_safety" in result.stderr


def test_phase_b_rejects_all_missing_landing_safety_by_default(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")

    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    pool_dir = iter_dir / "pool"
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    for i in range(2):
        _make_seed_result(
            pool_dir / f"seed_{i:04d}",
            atom_types,
            [(c[0] + 0.4 * i, c[1], c[2]) for c in base],
        )
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])

    assert result.returncode == 3
    assert "missing_landing_safety" in result.stderr
    assert not (iter_dir / "phase_b_SAMPLE_raw.xyz").exists()


def test_phase_b_accepts_all_missing_landing_safety_with_legacy_override(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.adversarial_safety.accept_legacy_missing_landing_safety = True
    cfg.to_yaml(campaign / "campaign.yaml")

    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    pool_dir = iter_dir / "pool"
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    for i in range(2):
        _make_seed_result(
            pool_dir / f"seed_{i:04d}",
            atom_types,
            [(c[0] + 0.4 * i, c[1], c[2]) for c in base],
        )
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (iter_dir / "PHASE_B_SELECTION.json").read_text(encoding="utf-8")
    )
    assert manifest["safety_filter"]["legacy_missing_safety"] is True


def test_phase_b_all_unsafe_candidates_halts_before_gaussian_handoff(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")

    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    pool_dir = iter_dir / "pool"
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    for i in range(2):
        seed_dir = pool_dir / f"seed_{i:04d}"
        _make_seed_result(
            seed_dir,
            atom_types,
            [(c[0] + 0.4 * i, c[1], c[2]) for c in base],
        )
        _set_landing_safety(
            seed_dir,
            accepted=False,
            reasons=["no_safe_non_seed_landing"],
            policy="rejected",
        )
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])

    assert result.returncode == 3
    assert "no_safe_non_seed_landing" in result.stderr
    assert not (iter_dir / "phase_b_SAMPLE_raw.xyz").exists()


def test_phase_b_no_seeds_returns_3(tmp_path):
    """Empty pool dir -> exit 3."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.to_yaml(campaign / "campaign.yaml")
    iter_dir = (
        campaign / "7_ACTIVE_LEARNING"
        / "iteration-0000"
    )
    (iter_dir / "pool").mkdir(parents=True)
    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 3
    assert "ARIADNE results manifest" in result.stderr


def test_acquisition_weighted_descriptor_refuses(tmp_path):
    """acquisition_weighted needs a posterior plumbed in. for now
    we refuse with a clear message rather than crash."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "acquisition_weighted"
    cfg.to_yaml(campaign / "campaign.yaml")
    iter_dir = (
        campaign / "7_ACTIVE_LEARNING"
        / "iteration-0000"
    )
    pool_dir = iter_dir / "pool"
    _make_seed_result(
        pool_dir / "seed_0000",
        ["O", "H", "H"],
        [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )
    _set_landing_safety(pool_dir / "seed_0000", accepted=True)
    _write_ariadne_manifest(iter_dir)
    result = _run([
        "--descriptor", "acquisition_weighted",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 3
    assert "acquisition_weighted" in result.stderr
