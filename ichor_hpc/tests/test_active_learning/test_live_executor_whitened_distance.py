"""Live executor _parse_ariadne_array_postprocess prefers the JSON
whitened_distance_final field when present, falls back to the synthetic
alpha-delta proxy when it is missing.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.live_executor import LiveBackendsPhaseExecutor
from ichor.hpc.active_learning.daemon.state import CampaignPhase
from ichor.hpc.active_learning.layout import active_iteration_dir, ariadne_seed_dir
from ichor.hpc.active_learning.versioning.provenance import (
    PROVENANCE_FILENAME, write_seed_provenance,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live_outputs"


def _seed_iter_pool(campaign_dir, iteration, results, *, config=None):
    """Publish synthetic results through the canonical ARIADNE handoff."""
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
    from ichor.hpc.active_learning.ariadne_outputs import (
        write_optimisation_trajectory,
        write_seed_output_manifest,
    )
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.daemon.config_lock import (
        canonical_config,
        config_fingerprint,
    )
    from ichor.hpc.active_learning.daemon.submission_intent import (
        write_pre_submit_intent,
    )
    from ichor.hpc.active_learning.handoff_manifests import (
        ARIADNE_RESULTS_SCHEMA_VERSION,
        build_seed_selection_manifest,
        seeds_picked_path,
        write_ariadne_results_manifest,
    )
    from ichor.hpc.active_learning.layout import (
        active_ariadne_dir,
        active_iteration_dir,
        ariadne_seed_dir,
    )
    from ichor.hpc.active_learning.seed_identity import write_ariadne_task_map
    from ichor.hpc.active_learning.sampling_protocol import resolve_sampling_protocol
    from ichor.hpc.active_learning.versioning.manifest import sha256_file

    campaign_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = CampaignConfig() if config is None else config
    write_pre_submit_intent(
        campaign_dir,
        campaign_uid="u",
        phase_name=CampaignPhase.ARIADNE_ARRAY.value,
        iteration=int(iteration),
        decision_contract={
            "failure_threshold_fraction": float(
                resolved_config.runtime.failure_threshold_fraction
            ),
            "config_sha256": config_fingerprint(
                canonical_config(resolved_config)
            ),
        },
    )
    pool_source = campaign_dir / "whitened-distance-pool.xyz"
    pool_lines = []
    for frame_id in range(len(results)):
        offset = 0.01 * float(frame_id)
        pool_lines.extend([
            "3",
            "frame " + str(frame_id),
            "O " + str(offset) + " 0.0 0.0",
            "H " + str(0.96 + offset) + " 0.0 0.0",
            "H " + str(-0.24 + offset) + " 0.93 0.0",
        ])
    pool_source.write_text("\n".join(pool_lines) + "\n", encoding="utf-8")
    trajectory_pool = TrajectoryPool.import_from(
        pool_source,
        campaign_dir,
        overwrite=True,
    )
    iter_dir = active_iteration_dir(campaign_dir, int(iteration))
    ariadne_root = active_ariadne_dir(iter_dir)
    frame_ids = []
    seed_records = []
    for array_task_id, _payload in enumerate(results):
        seed_id = array_task_id + 1
        frame_ids.append(array_task_id)
        seed_records.append({
            "seed_id": seed_id,
            "frame_id": array_task_id,
            "pool_row_index_zero_based": array_task_id,
            "selection_origin": "bulk",
            "variance_at_selection": 0.0,
        })
    selection = build_seed_selection_manifest(
        campaign_uid="u",
        campaign_random_seed=0,
        iteration=int(iteration),
        models_version=0,
        model_manifest_sha256="c" * 64,
        model_set_sha256="d" * 64,
        trajectory_sha256=str(trajectory_pool.sha256),
        selection_strategy="hybrid_variance",
        seed_records=seed_records,
    )
    selection_path = seeds_picked_path(iter_dir)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, selection)
    task_map_path = write_ariadne_task_map(iter_dir, selection)
    selected_records = selection["seed_records"]
    resolved_protocol = resolve_sampling_protocol(
        campaign_dir,
        resolved_config,
        iteration=int(iteration),
    )

    def protocol_binding(path):
        resolved = path.resolve()
        return (
            resolved.relative_to(campaign_dir.resolve()).as_posix(),
            sha256_file(resolved),
        )

    resolved_manifest, resolved_manifest_sha256 = protocol_binding(
        resolved_protocol.manifest_path
    )
    scale_manifest, scale_manifest_sha256 = protocol_binding(
        resolved_protocol.scale_model_path
    )
    audit_manifest, audit_manifest_sha256 = protocol_binding(
        resolved_protocol.audit_manifest_path
    )

    accepted = []
    for array_task_id, payload in enumerate(results):
        seed_id = array_task_id + 1
        seed_uid = str(selected_records[array_task_id]["seed_uid"])
        sd = ariadne_seed_dir(iter_dir, seed_id)
        sd.mkdir(parents=True, exist_ok=False)
        payload = dict(payload)
        payload.setdefault("atom_types", ["O", "H", "H"])
        payload.setdefault(
            "final_coordinates",
            [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)],
        )
        payload.setdefault("alpha_trajectory", [payload["alpha_initial"], payload["alpha_final"]])
        payload["iteration"] = int(iteration)
        payload["seed_id"] = seed_id
        payload["seed_uid"] = seed_uid
        payload["array_task_id"] = array_task_id
        payload["seed_frame_id"] = array_task_id
        payload.setdefault("trajectory_sha256", str(trajectory_pool.sha256))
        seed_atoms = trajectory_pool.frame(array_task_id)
        initial_coordinates = [
            [float(atom.x), float(atom.y), float(atom.z)]
            for atom in seed_atoms
        ]
        payload["initial_coordinates"] = [list(row) for row in initial_coordinates]
        payload["seed_coordinates"] = [list(row) for row in initial_coordinates]
        payload.setdefault("task_success", True)
        payload.setdefault(
            "landing_safety",
            {
                "accepted": True,
                "policy": "raw_final",
                "selected_origin": "raw_final",
                "reasons": [],
                "record_only_reasons": [],
                "metrics": {
                    "max_displacement_ang": 0.01,
                    "min_pair_distance_ang": 0.90,
                },
            },
        )
        payload["sampling_protocol"] = {
            "sampling_aggressiveness": int(
                resolved_protocol.sampling_aggressiveness
            ),
            "resolved_manifest": resolved_manifest,
            "resolved_manifest_sha256": resolved_manifest_sha256,
            "scale_model_manifest": scale_manifest,
            "scale_model_manifest_sha256": scale_manifest_sha256,
            "audit_manifest": audit_manifest,
            "audit_manifest_sha256": audit_manifest_sha256,
        }
        result_path = sd / "result.json"
        atomic_write_json(result_path, payload)
        write_optimisation_trajectory(
            sd,
            atom_types=payload["atom_types"],
            coordinate_frames=[payload["final_coordinates"]],
            alpha_values=[payload["alpha_final"]],
            gradient_norms=[0.0],
            origins=["raw_final"],
        )
        output_manifest = write_seed_output_manifest(
            sd,
            campaign_uid="u",
            iteration=int(iteration),
            seed_id=seed_id,
            seed_uid=seed_uid,
            array_task_id=array_task_id,
            task_success=True,
            task_exit_code=0,
        )
        write_seed_provenance(
            sd, campaign_uid="u", iteration=iteration,
            trajectory_sha256=str(trajectory_pool.sha256),
            seed_frame_id=array_task_id,
            seed_id=seed_id,
            seed_uid=seed_uid,
            array_task_id_zero_based=array_task_id,
            seed_selection_origin="test",
            seed_variance_at_selection=0.0,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
        )
        provenance_path = sd / PROVENANCE_FILENAME
        accepted.append({
            "seed_id": seed_id,
            "seed_uid": seed_uid,
            "array_task_id": array_task_id,
            "seed_dir": sd.relative_to(ariadne_root).as_posix(),
            "result_json": result_path.relative_to(ariadne_root).as_posix(),
            "provenance_json": provenance_path.relative_to(ariadne_root).as_posix(),
            "output_manifest": output_manifest.relative_to(ariadne_root).as_posix(),
            "seed_frame_id": array_task_id,
            "pool_row_index_zero_based": array_task_id,
            "selection_origin": "bulk",
            "variance_at_selection": 0.0,
            "alpha_initial": float(payload["alpha_initial"]),
            "alpha_final": float(payload["alpha_final"]),
            "whitened_distance_final": payload.get("whitened_distance_final"),
            "return_code": int(payload["return_code"]),
            "landing_safety": payload.get("landing_safety"),
            "result_sha256": sha256_file(result_path),
            "provenance_sha256": sha256_file(provenance_path),
            "output_manifest_sha256": sha256_file(output_manifest),
        })
    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "campaign_uid": "u",
        "iteration": int(iteration),
        "trajectory_sha256": str(trajectory_pool.sha256),
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
    return iter_dir


def _make_executor(tmp_path):
    return LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=CampaignConfig(),
        backend_check=False,
    )


def test_postprocess_prefers_json_whitened_distance(tmp_path):
    ex = _make_executor(tmp_path)
    # the value 0.42 lies in the safe band, so the seed will NOT be
    # flagged for anti-overlap. that is fine for this test -- we just
    # want to confirm the JSON value gets read.
    _seed_iter_pool(
        tmp_path / "campaign", 1,
        [{"alpha_initial": 0.5, "alpha_final": 1.0,
          "whitened_distance_final": 0.42,
          "n_evaluations": 1, "return_code": 0,
          "wall_seconds": 1.0, "fell_back_to_ds": False}],
    )
    state = SimpleNamespace(iteration=1, campaign_uid="u", models_version=0)
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.is_complete is True
    # the seed provenance should now have anti_overlap with d ~= 0.42
    from ichor.hpc.active_learning.versioning.provenance import read_provenance
    seed_dir = ariadne_seed_dir(active_iteration_dir(tmp_path / "campaign", 1), 1)
    prov = read_provenance(seed_dir)
    assert prov["anti_overlap"] is not None
    d = prov["anti_overlap"]["min_whitened_distance_to_training"]
    assert d == pytest.approx(0.42, abs=1.0e-6)


def test_postprocess_ignores_hidden_anti_overlap_bounds_and_records_override(
    tmp_path,
):
    cfg = CampaignConfig()
    cfg.anti_overlap.min_post_ariadne_whitened_distance = 0.5
    cfg.anti_overlap.max_post_ariadne_whitened_distance = 2.0
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )
    _seed_iter_pool(
        tmp_path / "campaign", 1,
        [{"alpha_initial": 0.5, "alpha_final": 1.0,
          "whitened_distance_final": 0.42,
          "n_evaluations": 1, "return_code": 0,
          "wall_seconds": 1.0, "fell_back_to_ds": False}],
        config=cfg,
    )
    result = ex._parse_ariadne_array_postprocess(
        SimpleNamespace(iteration=1, campaign_uid="u", models_version=0),
        CampaignPhase("ARIADNE_ARRAY"),
        observations=[],
    )
    assert result.is_complete is True
    from ichor.hpc.active_learning.versioning.provenance import read_provenance
    prov = read_provenance(
        ariadne_seed_dir(active_iteration_dir(tmp_path / "campaign", 1), 1)
    )
    assert prov["anti_overlap"]["flag"] is None
    resolved = json.loads(
        (
            active_iteration_dir(tmp_path / "campaign", 1)
            / "protocol"
            / "SAMPLING_PROTOCOL_RESOLVED.json"
        ).read_text(encoding="utf-8")
    )
    assert any(
        str(record.get("path", "")).startswith("anti_overlap.")
        for record in resolved["hidden_overrides_detected"]
    )


def test_postprocess_falls_back_to_synthetic_when_field_missing(tmp_path):
    ex = _make_executor(tmp_path)
    # NO whitened_distance_final key -- the parser should fall back to
    # the synthetic |alpha_final - alpha_initial| = 0.5 proxy.
    _seed_iter_pool(
        tmp_path / "campaign", 1,
        [{"alpha_initial": 0.5, "alpha_final": 1.0,
          "n_evaluations": 1, "return_code": 0,
          "wall_seconds": 1.0, "fell_back_to_ds": False}],
    )
    state = SimpleNamespace(iteration=1, campaign_uid="u", models_version=0)
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.is_complete is True
    from ichor.hpc.active_learning.versioning.provenance import read_provenance
    seed_dir = ariadne_seed_dir(active_iteration_dir(tmp_path / "campaign", 1), 1)
    prov = read_provenance(seed_dir)
    d = prov["anti_overlap"]["min_whitened_distance_to_training"]
    assert d == pytest.approx(0.5, abs=1.0e-6)


def test_postprocess_rejects_explicit_unsafe_landing(tmp_path):
    ex = _make_executor(tmp_path)
    iter_dir = _seed_iter_pool(
        tmp_path / "campaign", 1,
        [{
            "alpha_initial": 0.5,
            "alpha_final": 1.0,
            "whitened_distance_final": 0.42,
            "n_evaluations": 1,
            "return_code": 0,
            "wall_seconds": 1.0,
            "fell_back_to_ds": False,
            "landing_safety": {
                "accepted": False,
                "policy": "unsafe_raw_final",
                "reasons": ["no_safe_non_seed_landing"],
                "record_only_reasons": [],
                "metrics": {},
            },
        }],
    )
    state = SimpleNamespace(iteration=1, campaign_uid="u", models_version=0)
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.is_complete is True
    assert "ariadne_no_usable_seed_results" in result.failure_reason
    from ichor.hpc.active_learning.handoff_manifests import (
        read_ariadne_landing_audit,
        read_ariadne_results_manifest,
    )
    audit = read_ariadne_landing_audit(iter_dir, expected_iteration=1)
    assert audit["summary"]["rejected"] == 1
    manifest = read_ariadne_results_manifest(
        iter_dir, expected_iteration=1, require_nonempty=False,
    )
    assert manifest["accepted"] == []
    assert manifest["rejected"][0]["reason"] == (
        "ariadne_unusable:no_safe_non_seed_landing"
    )
