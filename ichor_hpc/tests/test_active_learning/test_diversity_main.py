"""Tests for the diversity main(argv) Phase A + Phase B bodies.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.handoff_manifests import (
    HandoffManifestError,
    PHASE_A_SAMPLE_FILENAME,
    read_phase_a_sample_manifest,
)
from ichor.hpc.active_learning.layout import (
    active_iteration_dir,
    active_phase_b_dir,
    ariadne_seed_dir,
    ariadne_seeds_dir,
    bootstrap_selection_dir,
)


MODULE = "ichor.hpc.active_learning.sampling.diversity"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "water_tetramer.xyz"


def _active_iter(campaign):
    return active_iteration_dir(campaign, 1)


def _import_pool(campaign, source):
    """Drop a trajectory into the campaign-canonical location."""
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
    from ichor.hpc.active_learning.custom_bootstrap import (
        commit_bootstrap_plan,
        inspect_bootstrap_inputs,
    )

    pool = TrajectoryPool.import_from(source, campaign, overwrite=True)
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    commit_bootstrap_plan(
        inspect_bootstrap_inputs(
            campaign,
            config,
            pool.to_atoms_list(),
            pool_sha256=pool.sha256,
        )
    )


def _write_custom_training_from_fixture(campaign, n_frames=1):
    from ichor.core.files.xyz import Trajectory
    from ichor.hpc.active_learning.sampling.diversity import _write_xyz_file

    traj = Trajectory(FIXTURE)
    traj.read()
    frames = []
    base = [atoms.copy() for atoms in traj][0]
    for idx in range(int(n_frames)):
        frame = base.copy()
        if idx:
            frame[1].coordinates = [
                frame[1].x + 0.1 * idx,
                frame[1].y,
                frame[1].z,
            ]
        frames.append(frame)
    source = campaign / "bootstrap" / "training_set_bootstrap.xyz"
    source.parent.mkdir(parents=True, exist_ok=True)
    _write_xyz_file(frames, source)
    return frames


def _run(args):
    if "--campaign-dir" in args:
        from ichor.hpc.active_learning.daemon.state import (
            DEFAULT_STATE_FILENAME,
            fresh_campaign_state,
            write_state,
        )

        campaign = Path(args[args.index("--campaign-dir") + 1])
        state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
        if not state_path.is_file():
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state = fresh_campaign_state()
            iteration = int(args[args.index("--iteration") + 1])
            if iteration >= 1:
                from ichor.hpc.active_learning.daemon.state import CampaignPhase

                state.campaign_uid = "test"
                state.phase = CampaignPhase.PHASE_B_DIVERSITY
                state.iteration = iteration
                state.reference_data_version = 0
                state.models_version = 0
            write_state(state_path, state)
    return subprocess.run(
        [sys.executable, "-m", MODULE] + args,
        capture_output=True, text=True, timeout=120,
    )


def test_phase_a_writes_sample_and_index(tmp_path):
    """Phase A reads the trajectory pool and writes the canonical
    .DATA/BOOTSTRAP/selection sample, index, and manifest files."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.point_allocation.bootstrap_training_size = 4
    cfg.point_allocation.bootstrap_internal_validation_size = 2
    cfg.point_allocation.bootstrap_external_validation_size = 2
    cfg.max_iterations = 1
    cfg.seed_selection.n_seeds_per_iteration = 4
    cfg.to_yaml(campaign / "campaign.yaml")
    _import_pool(campaign, FIXTURE)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 0, result.stderr

    outdir = bootstrap_selection_dir(campaign)
    samples = list(outdir.glob("selected.xyz"))
    indices = list(outdir.glob("selected_indices.dat"))
    assert len(samples) == 1
    assert len(indices) == 1
    # n_select is the sum of the exact bootstrap slot counts (8).
    idx_lines = indices[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(idx_lines) == 8
    for ln in idx_lines:
        assert ln.strip().isdigit()
    manifest = json.loads((outdir / PHASE_A_SAMPLE_FILENAME).read_text(encoding="utf-8"))
    assert manifest["phase"] == "PHASE_A_DIVERSITY"
    assert manifest["iteration"] == 0
    assert manifest["n_select"] == 8
    assert manifest["n_frames"] == 8
    assert manifest["sample_xyz"].endswith(samples[0].name)
    assert manifest["index_path"].endswith(indices[0].name)
    assert len(manifest["selected_indices"]) == 8
    assert manifest["trajectory_sha256"]
    assert len(manifest["sample_xyz_sha256"]) == 64
    assert len(manifest["index_sha256"]) == 64
    assert len(manifest["point_allocation"]["slot_assignment_sha256"]) == 64
    read_phase_a_sample_manifest(outdir)


def test_phase_a_manifest_rejects_sample_drift(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.point_allocation.bootstrap_training_size = 2
    cfg.point_allocation.bootstrap_internal_validation_size = 2
    cfg.point_allocation.bootstrap_external_validation_size = 2
    cfg.max_iterations = 1
    cfg.seed_selection.n_seeds_per_iteration = 4
    cfg.to_yaml(campaign / "campaign.yaml")
    _import_pool(campaign, FIXTURE)
    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 0, result.stderr
    outdir = bootstrap_selection_dir(campaign)
    sample = outdir / "selected.xyz"
    sample.write_text(
        sample.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HandoffManifestError, match="sample XYZ (size|hash) mismatch"):
        read_phase_a_sample_manifest(outdir)


def test_phase_a_preserves_custom_training_geometry_and_fills_remainder_from_pool(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.point_allocation.bootstrap_training_size = 4
    cfg.point_allocation.bootstrap_internal_validation_size = 2
    cfg.point_allocation.bootstrap_external_validation_size = 2
    cfg.campaign.custom_bootstrap = True
    cfg.max_iterations = 1
    cfg.seed_selection.n_seeds_per_iteration = 4
    cfg.to_yaml(campaign / "campaign.yaml")
    _write_custom_training_from_fixture(campaign, n_frames=1)
    _import_pool(campaign, FIXTURE)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "0",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 0, result.stderr

    outdir = bootstrap_selection_dir(campaign)
    manifest = json.loads((outdir / PHASE_A_SAMPLE_FILENAME).read_text(encoding="utf-8"))
    assert manifest["n_select"] == 8
    assert manifest["custom_bootstrap"] is True
    assert manifest["custom_bootstrap_counts"]["train"] == 1
    assert manifest["bootstrap_pool_frame_count"] == 7
    assert manifest["selected_indices"][0] is None
    assert len(manifest["selected_pool_indices"]) == 7
    assert len(manifest["selected_indices"]) == 8
    assert 0 not in manifest["selected_pool_indices"]
    assert manifest["excluded_pool_frame_ids"] == [0]
    custom_manifest = (
        campaign
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "CUSTOM_BOOTSTRAP.json"
    )
    assert custom_manifest.is_file()
    idx_lines = (outdir / "selected_indices.dat").read_text(encoding="utf-8").splitlines()
    assert idx_lines[0] == "custom:0"
    assert all(line.strip().isdigit() for line in idx_lines[1:])

    read_phase_a_sample_manifest(outdir)
    custom_manifest.write_text(
        custom_manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        HandoffManifestError,
        match="custom-bootstrap manifest (size|hash) mismatch",
    ):
        read_phase_a_sample_manifest(outdir)


def test_bootstrap_discovery_rejects_more_custom_training_rows_than_target(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.point_allocation.bootstrap_training_size = 4
    cfg.point_allocation.bootstrap_internal_validation_size = 2
    cfg.point_allocation.bootstrap_external_validation_size = 2
    cfg.campaign.custom_bootstrap = True
    cfg.max_iterations = 1
    cfg.seed_selection.n_seeds_per_iteration = 4
    cfg.to_yaml(campaign / "campaign.yaml")
    _write_custom_training_from_fixture(campaign, n_frames=5)

    with pytest.raises(ValueError, match="permits at most 4"):
        _import_pool(campaign, FIXTURE)


def test_bootstrap_inspection_fails_when_target_exceeds_pool_size(tmp_path):
    """If the bootstrap target exceeds the pool, initialisation fails fast."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.point_allocation.bootstrap_training_size = 96
    cfg.point_allocation.bootstrap_internal_validation_size = 2
    cfg.point_allocation.bootstrap_external_validation_size = 2
    cfg.max_iterations = 1
    cfg.seed_selection.n_seeds_per_iteration = 4
    cfg.to_yaml(campaign / "campaign.yaml")
    with pytest.raises(ValueError, match="need 100, available 20"):
        _import_pool(campaign, FIXTURE)
    outdir = bootstrap_selection_dir(campaign)
    assert not (outdir / "selected.xyz").is_file()


def _make_seed_result(seed_dir, atom_types, final_coords, alpha_final=1.0):
    """Drop a synthetic result into a canonical one-based seed directory."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_id = int(seed_dir.name.split("-")[-1])
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
        "iteration": 1,
        "seed_id": seed_id,
        "seed_frame_id": seed_id - 1,
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
        build_seed_selection_manifest,
        seeds_picked_path,
        write_ariadne_results_manifest,
    )
    from ichor.hpc.active_learning.layout import active_ariadne_dir
    from ichor.hpc.active_learning.seed_identity import write_ariadne_task_map
    from ichor.hpc.active_learning.versioning.manifest import sha256_file
    from ichor.hpc.active_learning.versioning.provenance import (
        PROVENANCE_FILENAME,
        write_seed_provenance,
    )

    ariadne_root = active_ariadne_dir(iter_dir)
    seed_dirs = sorted((ariadne_root / "seeds").glob("seed-*"))
    frame_ids = list(range(len(seed_dirs)))
    selection = build_seed_selection_manifest(
        campaign_uid="test",
        campaign_random_seed=0,
        iteration=1,
        models_version=0,
        model_manifest_sha256="c" * 64,
        model_set_sha256="d" * 64,
        trajectory_sha256="0" * 64,
        selection_strategy="hybrid_variance",
        seed_records=[
            {
                "seed_id": seed_id,
                "frame_id": seed_id - 1,
                "pool_row_index_zero_based": seed_id - 1,
                "selection_origin": "bulk",
                "variance_at_selection": 0.0,
            }
            for seed_id in range(1, len(seed_dirs) + 1)
        ],
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
        rec = {
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
            "result_sha256": sha256_file(result_path),
            "provenance_sha256": sha256_file(prov_path),
            "output_manifest_sha256": sha256_file(output_manifest),
        }
        if isinstance(result.get("landing_safety"), dict):
            rec["landing_safety"] = dict(result["landing_safety"])
        accepted.append(rec)
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
    from ichor.hpc.active_learning.sampling_protocol import (
        resolve_sampling_protocol,
    )

    campaign = iter_dir.parent.parent
    config_path = campaign / "campaign.yaml"
    config = (
        CampaignConfig.from_yaml(config_path)
        if config_path.is_file()
        else CampaignConfig()
    )
    from ichor.hpc.active_learning.daemon.config_lock import (
        canonical_config,
        config_fingerprint,
    )
    from ichor.hpc.active_learning.handoff_manifests import (
        write_ariadne_batch_decision,
    )

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
    resolve_sampling_protocol(
        campaign,
        config,
        iteration=1,
    )


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
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 1
    # synthetic geometries lack ALF assignment so default hybrid_alf_rmsd
    # would crash on the feature extractor. mass-weighted RMSD is the
    # safe choice for unit tests.
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")
    # build 5 candidate seeds with distinct geometries.
    iter_dir = _active_iter(campaign)
    pool_dir = ariadne_seeds_dir(iter_dir)
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    for i in range(5):
        # distort the shape rather than only translating it; Phase B anti-overlap
        # aligns geometries and now also de-duplicates within the selected batch.
        coords = [
            (0.0, 0.0, 0.0),
            (0.96 + 0.20 * i, 0.0, 0.0),
            (-0.24, 0.93 + 0.15 * i, 0.0),
        ]
        seed_dir = ariadne_seed_dir(iter_dir, i + 1)
        _make_seed_result(seed_dir, atom_types, coords)
        _set_landing_safety(seed_dir, accepted=True)
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "1",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 0, result.stderr
    phase_b_dir = active_phase_b_dir(iter_dir)
    raw = phase_b_dir / "considered_candidates.xyz"
    final = phase_b_dir / "selected.xyz"
    dedup = phase_b_dir / "SELECTION.json"
    assert raw.is_file()
    assert final.is_file()
    assert dedup.is_file()
    manifest = json.loads(dedup.read_text(encoding="utf-8"))
    d = manifest["dedup"]
    # fresh campaign, no committed QM reference data to dedup against, so case (d) drops nothing
    # regardless of the geometry novelty-derived default minimum separation.
    assert d["n_dropped"] == 0
    assert d["min_separation"] == 0.025
    assert "distance_to_nearest_angstrom" in manifest["considered"][0]
    assert "scaled_distance_to_nearest" in manifest["considered"][0]
    assert "novelty_score" in manifest["considered"][0]


def test_phase_b_rejects_unsafe_accepted_landing_before_fps(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.point_allocation.batch_training_size = 2
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")

    iter_dir = _active_iter(campaign)
    pool_dir = ariadne_seeds_dir(iter_dir)
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    for i in range(3):
        coords = [(c[0] + 0.4 * i, c[1], c[2]) for c in base]
        seed_dir = ariadne_seed_dir(iter_dir, i + 1)
        _make_seed_result(seed_dir, atom_types, coords)
        _set_landing_safety(seed_dir, accepted=(i != 1),
                            reasons=["no_safe_non_seed_landing"] if i == 1 else [])
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "1",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 3
    assert "no_safe_non_seed_landing" in result.stderr
    assert not (active_phase_b_dir(iter_dir) / "selected_raw.xyz").exists()


def test_phase_b_rejects_partial_missing_landing_safety(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")

    iter_dir = _active_iter(campaign)
    pool_dir = ariadne_seeds_dir(iter_dir)
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    _make_seed_result(ariadne_seed_dir(iter_dir, 1), atom_types, base)
    _set_landing_safety(ariadne_seed_dir(iter_dir, 1), accepted=True)
    _make_seed_result(
        ariadne_seed_dir(iter_dir, 2), atom_types,
        [(c[0] + 0.4, c[1], c[2]) for c in base],
    )
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "1",
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

    iter_dir = _active_iter(campaign)
    pool_dir = ariadne_seeds_dir(iter_dir)
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    for i in range(2):
        _make_seed_result(
            ariadne_seed_dir(iter_dir, i + 1),
            atom_types,
            [
                (0.0, 0.0, 0.0),
                (0.96 + 0.30 * i, 0.0, 0.0),
                (-0.24, 0.93 + 0.20 * i, 0.0),
            ],
        )
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "1",
        "--campaign-dir", str(campaign),
    ])

    assert result.returncode == 3
    assert "missing_landing_safety" in result.stderr
    assert not (active_phase_b_dir(iter_dir) / "selected_raw.xyz").exists()


def test_phase_b_rejects_missing_landing_safety_even_with_stale_legacy_flag(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.to_yaml(campaign / "campaign.yaml")

    iter_dir = _active_iter(campaign)
    pool_dir = ariadne_seeds_dir(iter_dir)
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    for i in range(2):
        _make_seed_result(
            ariadne_seed_dir(iter_dir, i + 1),
            atom_types,
            [
                (0.0, 0.0, 0.0),
                (0.96 + 0.30 * i, 0.0, 0.0),
                (-0.24, 0.93 + 0.20 * i, 0.0),
            ],
        )
    _write_ariadne_manifest(iter_dir)

    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "1",
        "--campaign-dir", str(campaign),
    ])

    assert result.returncode == 3
    assert "missing_landing_safety" in result.stderr


def test_phase_b_all_unsafe_candidates_halts_before_gaussian_handoff(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")

    iter_dir = _active_iter(campaign)
    pool_dir = ariadne_seeds_dir(iter_dir)
    atom_types = ["O", "H", "H"]
    base = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    for i in range(2):
        seed_dir = ariadne_seed_dir(iter_dir, i + 1)
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
        "--iteration", "1",
        "--campaign-dir", str(campaign),
    ])

    assert result.returncode == 3
    assert "no_safe_non_seed_landing" in result.stderr
    assert not (active_phase_b_dir(iter_dir) / "selected_raw.xyz").exists()


def test_phase_b_no_seeds_returns_3(tmp_path):
    """Empty pool dir -> exit 3."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.to_yaml(campaign / "campaign.yaml")
    iter_dir = _active_iter(campaign)
    ariadne_seeds_dir(iter_dir).mkdir(parents=True)
    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "1",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 3
    assert "ARIADNE results manifest" in result.stderr


def test_phase_b_uses_public_configured_descriptor_single_candidate(tmp_path):
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "rmsd_massweight"
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 0
    cfg.to_yaml(campaign / "campaign.yaml")
    iter_dir = _active_iter(campaign)
    pool_dir = ariadne_seeds_dir(iter_dir)
    _make_seed_result(
        ariadne_seed_dir(iter_dir, 1),
        ["O", "H", "H"],
        [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)],
    )
    _set_landing_safety(ariadne_seed_dir(iter_dir, 1), accepted=True)
    _write_ariadne_manifest(iter_dir)
    result = _run([
        "--descriptor", "rmsd_massweight",
        "--iteration", "1",
        "--campaign-dir", str(campaign),
    ])
    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (active_phase_b_dir(iter_dir) / "SELECTION.json").read_text(encoding="utf-8")
    )
    assert manifest["descriptor"] == "rmsd_massweight"
    assert manifest["n_kept"] == 1
