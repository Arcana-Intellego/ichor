import json
from pathlib import Path

import pytest

from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.geometry_protocol import PHASE_B_MIN_SEPARATION_SCALE
from ichor.hpc.active_learning.handoff_manifests import (
    ariadne_landing_audit_path,
    ariadne_results_path,
    seeds_picked_path,
)
from ichor.hpc.active_learning.layout import active_iteration_dir
from ichor.hpc.active_learning.seed_identity import (
    deterministic_seed_uid,
    selection_fingerprint_sha256,
)
from ichor.hpc.active_learning.daemon.state import atomic_write_json
from ichor.hpc.active_learning.geometry_novelty import (
    apply_geometry_novelty_to_acquisition_config,
    compute_geometry_novelty_scale,
    effective_phase_b_min_separation,
    ensure_geometry_novelty_scale,
    geometry_novelty_input_fingerprint,
    geometry_novelty_scale_path,
    novelty_score,
    read_geometry_novelty_scale,
    resolve_geometry_novelty_consumers,
    scaled_distances,
    write_geometry_novelty_scale,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "water_tetramer.xyz"
)


def _iter_dir(campaign, iteration=1):
    path = active_iteration_dir(campaign, iteration)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_seed_selection(iter_dir, frame_ids, *, trajectory_sha256="a" * 64, neighbours=None):
    iteration = int(iter_dir.name.split("-")[-1])
    records = []
    for seed_id, frame_id in enumerate(frame_ids, start=1):
        record = {
            "seed_id": seed_id,
            "frame_id": int(frame_id),
            "pool_row_index_zero_based": int(frame_id),
            "selection_origin": "bulk",
            "variance_at_selection": 0.0,
        }
        if neighbours and int(frame_id) in neighbours:
            record["subspace_neighbour_frame_ids"] = list(neighbours[int(frame_id)])
        records.append(record)
    payload = {
        "schema_version": 2,
        "campaign_uid": "geometry-novelty-test",
        "iteration": iteration,
        "models_version": 0,
        "model_manifest_sha256": "c" * 64,
        "trajectory_sha256": str(trajectory_sha256),
        "n_picked": len(records),
        "seed_records": records,
    }
    fingerprint = selection_fingerprint_sha256(payload)
    payload["selection_fingerprint_sha256"] = fingerprint
    for record in records:
        record["seed_uid"] = deterministic_seed_uid(
            campaign_uid="geometry-novelty-test",
            iteration=iteration,
            seed_id=int(record["seed_id"]),
            frame_id=int(record["frame_id"]),
            models_version=0,
            model_manifest_sha256="c" * 64,
            selection_fingerprint_sha256_value=fingerprint,
        )
    path = seeds_picked_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def test_geometry_novelty_falls_back_when_pool_history_missing(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.fallback_scale_angstrom = 0.04
    iter_dir = _iter_dir(tmp_path)

    payload = compute_geometry_novelty_scale(tmp_path, cfg, iteration=1)
    path = write_geometry_novelty_scale(iter_dir, payload)
    loaded = read_geometry_novelty_scale(iter_dir)

    assert path == geometry_novelty_scale_path(iter_dir)
    assert loaded["schema_version"] == 1
    assert loaded["scale_angstrom"] == pytest.approx(0.04)
    assert loaded["fallback_used"] is True
    assert "no_motion_values" in loaded["reasons"]


def test_geometry_novelty_uses_seed_neighbour_motion(tmp_path):
    cfg = CampaignConfig()
    TrajectoryPool.import_from(FIXTURE, tmp_path)
    iter_dir = _iter_dir(tmp_path)
    _write_seed_selection(iter_dir, [0], neighbours={0: [1, 2]})

    payload = compute_geometry_novelty_scale(tmp_path, cfg, iteration=1)

    assert payload["fallback_used"] is False
    assert payload["n_values"] == 2
    assert payload["scale_angstrom"] > cfg.geometry_novelty.scale_floor_angstrom
    assert payload["local_motion_summary_angstrom"]["n"] == 2
    assert payload["diagnostics"]["local_motion"]["neighbour_sources"] == {
        "manifest_ids": 1
    }


def test_geometry_novelty_nearest_neighbour_fallback_is_not_trajectory_adjacent(tmp_path):
    cfg = CampaignConfig()
    cfg.acquisition.subspace.neighbour_count = 1
    source = tmp_path / "pool_source.xyz"
    source.write_text(
        "\n".join(
            [
                "3",
                "frame 0",
                "O 0.0 0.0 0.0",
                "H 1.0 0.0 0.0",
                "H 0.0 1.0 0.0",
                "3",
                "frame 1",
                "O 0.0 0.0 0.0",
                "H 5.0 0.0 0.0",
                "H 0.0 5.0 0.0",
                "3",
                "frame 2",
                "O 0.0 0.0 0.0",
                "H 1.02 0.0 0.0",
                "H 0.0 1.0 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    TrajectoryPool.import_from(source, tmp_path)
    iter_dir = _iter_dir(tmp_path)
    _write_seed_selection(iter_dir, [0])

    payload = compute_geometry_novelty_scale(tmp_path, cfg, iteration=1)

    assert payload["fallback_used"] is False
    assert payload["n_values"] == 1
    assert payload["scale_angstrom"] < 0.1
    local_diag = payload["diagnostics"]["local_motion"]
    assert local_diag["neighbour_sources"] == {"nearest_pool_rmsd": 1}
    assert local_diag["nearest_pool_rmsd"]["nearest_k"] == 1


def test_geometry_novelty_movement_history_reads_ariadne_landing_audit(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.scale_source = "movement_history"
    iter1 = _iter_dir(tmp_path, iteration=1)
    ariadne_landing_audit_path(iter1).parent.mkdir(parents=True, exist_ok=True)
    ariadne_landing_audit_path(iter1).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 1,
                "summary": {},
                "seeds": [
                    {
                        "seed_id": 1,
                        "landing_safety": {
                            "accepted": True,
                            "metrics": {"movement_rmsd_ang": 0.07},
                        },
                    },
                    {
                        "seed_id": 2,
                        "landing_safety": {
                            "accepted": False,
                            "metrics": {"movement_rmsd_ang": 9.0},
                        },
                    },
                    {
                        "seed_id": 3,
                        "handoff_accepted": False,
                        "landing_safety": {
                            "accepted": True,
                            "metrics": {"movement_rmsd_ang": 8.0},
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = compute_geometry_novelty_scale(tmp_path, cfg, iteration=2)

    assert payload["fallback_used"] is False
    assert payload["scale_angstrom"] == pytest.approx(0.07)
    history_diag = payload["diagnostics"]["movement_history"]
    assert history_diag["source"] == "ariadne_landing_audit"
    assert history_diag["n_accepted_movements"] == 1
    assert history_diag["n_skipped_handoff_rejected"] == 1
    assert history_diag["n_skipped_rejected"] == 1


def test_geometry_novelty_movement_history_falls_back_to_ariadne_results(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.scale_source = "movement_history"
    iter1 = _iter_dir(tmp_path, iteration=1)
    ariadne_results_path(iter1).parent.mkdir(parents=True, exist_ok=True)
    ariadne_results_path(iter1).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 1,
                "accepted": [
                    {
                        "seed_id": 1,
                        "landing_safety": {
                            "accepted": True,
                            "metrics": {"aligned_mass_weighted_rmsd_ang": 0.09},
                        },
                    }
                ],
                "rejected": [],
            }
        ),
        encoding="utf-8",
    )

    payload = compute_geometry_novelty_scale(tmp_path, cfg, iteration=2)

    assert payload["fallback_used"] is False
    assert payload["scale_angstrom"] == pytest.approx(0.09)
    assert payload["diagnostics"]["movement_history"]["source"] == "ariadne_results"


def test_scaled_threshold_and_scores_are_dimensionless():
    cfg = CampaignConfig()
    payload = {"scale_angstrom": 0.04}

    threshold, mode = effective_phase_b_min_separation(cfg, payload)

    assert mode == "scaled"
    assert threshold == pytest.approx(0.02)
    assert scaled_distances([0.01, 0.04], 0.04) == pytest.approx([0.25, 1.0])
    assert novelty_score(0.02, 0.04, "linear_cap") == pytest.approx(0.5)
    assert novelty_score(10.0, 0.04, "linear_cap") == pytest.approx(1.0)
    assert novelty_score(0.04, 0.04, "exponential") == pytest.approx(1.0 - 2.718281828459045 ** -1)


def test_geometry_novelty_resolves_all_current_consumers():
    cfg = CampaignConfig()
    payload = {"scale_angstrom": 0.04}

    resolved = resolve_geometry_novelty_consumers(cfg, payload)

    assert resolved["threshold_mode"] == "scaled"
    assert resolved["scale_resolution_mode"] == "computed"
    assert resolved["phase_b"]["scaled_threshold"] == pytest.approx(PHASE_B_MIN_SEPARATION_SCALE)
    assert resolved["phase_b"]["min_separation_scaled"] == pytest.approx(PHASE_B_MIN_SEPARATION_SCALE)
    assert resolved["phase_b"]["effective_min_separation_angstrom"] == pytest.approx(0.02)
    assert resolved["movement_band"]["target_peak_angstrom"] == pytest.approx(0.016)
    assert resolved["movement_utility"]["low_softness_angstrom"] == pytest.approx(0.004)
    assert resolved["movement_utility"]["high_softness_angstrom"] == pytest.approx(0.016)
    assert resolved["fullspace_confinement"]["rmsd_scale_angstrom"] == pytest.approx(0.4)


def test_geometry_novelty_disabled_reports_fallback_protocol_semantics():
    cfg = CampaignConfig()
    cfg.geometry_novelty.enabled = False
    cfg.geometry_novelty.fallback_scale_angstrom = 0.06

    resolved = resolve_geometry_novelty_consumers(cfg, {"scale_angstrom": 0.04})
    threshold, mode = effective_phase_b_min_separation(cfg, {"scale_angstrom": 0.04})

    assert mode == "absolute"
    assert resolved["scale_resolution_mode"] == "fallback_protocol"
    assert resolved["phase_b"]["scale_resolution_mode"] == "fallback_protocol"
    assert threshold == pytest.approx(0.03)


def test_ensure_geometry_novelty_scale_recomputes_sidecar_without_fingerprint(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.fallback_scale_angstrom = 0.04
    iter_dir = _iter_dir(tmp_path)
    write_geometry_novelty_scale(
        iter_dir,
        {
            "schema_version": 1,
            "iteration": 1,
            "scale_angstrom": 0.04,
            "fallback_used": True,
            "n_values": 0,
        },
    )

    payload = ensure_geometry_novelty_scale(tmp_path, cfg, iteration=1)
    loaded = read_geometry_novelty_scale(iter_dir)

    assert payload["scale_angstrom"] == pytest.approx(0.04)
    assert "missing_input_fingerprint_recomputed" in payload["reasons"]
    assert payload["resolved_consumers"]["phase_b"]["threshold_mode"] == "scaled"
    assert loaded["resolved_consumers"]["movement_band"]["target_peak_angstrom"] == pytest.approx(0.016)


def test_ensure_geometry_novelty_scale_backfills_current_sidecar_resolved_consumers(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.fallback_scale_angstrom = 0.04
    iter_dir = _iter_dir(tmp_path)
    payload = compute_geometry_novelty_scale(tmp_path, cfg, iteration=1)
    write_geometry_novelty_scale(iter_dir, payload)

    loaded = ensure_geometry_novelty_scale(tmp_path, cfg, iteration=1)

    assert "input_fingerprint_mismatch_recomputed" not in loaded["reasons"]
    assert "missing_input_fingerprint_recomputed" not in loaded["reasons"]
    assert "resolved_consumers_backfilled" in loaded["reasons"]
    assert loaded["resolved_consumers"]["phase_b"]["threshold_mode"] == "scaled"


def test_ensure_geometry_novelty_scale_reuses_current_sidecar_without_rewrite(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.fallback_scale_angstrom = 0.04
    iter_dir = _iter_dir(tmp_path)

    first = ensure_geometry_novelty_scale(tmp_path, cfg, iteration=1)
    before = geometry_novelty_scale_path(iter_dir).read_text(encoding="utf-8")
    second = ensure_geometry_novelty_scale(tmp_path, cfg, iteration=1)
    after = geometry_novelty_scale_path(iter_dir).read_text(encoding="utf-8")

    assert second == first
    assert after == before


def test_geometry_novelty_recomputes_when_seed_selection_changes(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.fallback_scale_angstrom = 0.04
    iter_dir = _iter_dir(tmp_path)
    seed_path = _write_seed_selection(
        iter_dir,
        [0],
        trajectory_sha256="a" * 64,
    )
    first = ensure_geometry_novelty_scale(tmp_path, cfg, iteration=1)

    _write_seed_selection(
        iter_dir,
        [0],
        trajectory_sha256="b" * 64,
    )
    second = ensure_geometry_novelty_scale(tmp_path, cfg, iteration=1)

    assert second["input_fingerprint"]["trajectory_sha256"] == "b" * 64
    assert second["input_fingerprint"] != first["input_fingerprint"]
    assert "input_fingerprint_mismatch_recomputed" in second["reasons"]
    mismatch = second["diagnostics"]["input_fingerprint_mismatch"]["keys"]
    assert "seed_selection_sha256" in mismatch
    assert "trajectory_sha256" in mismatch


def test_geometry_novelty_recomputes_when_config_changes(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.fallback_scale_angstrom = 0.04
    ensure_geometry_novelty_scale(tmp_path, cfg, iteration=1)

    cfg.geometry_novelty.statistic = "p75"
    payload = ensure_geometry_novelty_scale(tmp_path, cfg, iteration=1)

    assert payload["statistic"] == "p75"
    assert "input_fingerprint_mismatch_recomputed" in payload["reasons"]
    mismatch = payload["diagnostics"]["input_fingerprint_mismatch"]["keys"]
    assert "geometry_novelty_config" in mismatch


def test_geometry_novelty_fingerprint_omits_history_for_local_motion(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.scale_source = "local_motion"
    iter1 = _iter_dir(tmp_path, iteration=1)
    ariadne_landing_audit_path(iter1).parent.mkdir(parents=True, exist_ok=True)
    ariadne_landing_audit_path(iter1).write_text(
        json.dumps({"schema_version": 1, "seeds": []}),
        encoding="utf-8",
    )

    fingerprint = geometry_novelty_input_fingerprint(tmp_path, cfg, iteration=2)

    assert fingerprint["history_source_files"] == []


def test_geometry_novelty_recomputes_when_history_file_changes(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.scale_source = "movement_history"
    iter1 = _iter_dir(tmp_path, iteration=1)
    ariadne_landing_audit_path(iter1).parent.mkdir(parents=True, exist_ok=True)
    audit = ariadne_landing_audit_path(iter1)
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 1,
                "seeds": [
                    {
                        "seed_id": 1,
                        "landing_safety": {
                            "accepted": True,
                            "metrics": {"movement_rmsd_ang": 0.07},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    first = ensure_geometry_novelty_scale(tmp_path, cfg, iteration=2)
    assert first["scale_angstrom"] == pytest.approx(0.07)

    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 1,
                "seeds": [
                    {
                        "seed_id": 1,
                        "landing_safety": {
                            "accepted": True,
                            "metrics": {"movement_rmsd_ang": 0.11},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    second = ensure_geometry_novelty_scale(tmp_path, cfg, iteration=2)

    assert second["scale_angstrom"] == pytest.approx(0.11)
    assert "input_fingerprint_mismatch_recomputed" in second["reasons"]
    mismatch = second["diagnostics"]["input_fingerprint_mismatch"]["keys"]
    assert "history_source_files" in mismatch


def test_geometry_novelty_sidecar_records_provenance_and_checks_iteration(tmp_path):
    cfg = CampaignConfig()
    iter_dir = _iter_dir(tmp_path)
    _write_seed_selection(iter_dir, [0], trajectory_sha256="a" * 64)

    payload = compute_geometry_novelty_scale(tmp_path, cfg, iteration=1)
    write_geometry_novelty_scale(iter_dir, payload)
    loaded = read_geometry_novelty_scale(iter_dir, expected_iteration=1)

    assert loaded["provenance"]["trajectory_sha256"] == "a" * 64
    selection_path = Path(loaded["provenance"]["seed_selection_manifest"])
    assert selection_path.parts[-2:] == ("seed_selection", "SELECTION.json")
    assert len(loaded["provenance"]["seed_selection_sha256"]) == 64
    with pytest.raises(ValueError, match="iteration mismatch"):
        read_geometry_novelty_scale(iter_dir, expected_iteration=2)


def test_geometry_novelty_applies_to_core_acquisition_config():
    cfg = CampaignConfig()
    acq_cfg = cfg.to_acquisition_config()
    payload = {"scale_angstrom": 0.04}

    patched = apply_geometry_novelty_to_acquisition_config(acq_cfg, cfg, payload)

    assert patched is not acq_cfg
    assert patched.movement_band.geometry_novelty_scale_angstrom == pytest.approx(0.04)
    assert patched.movement_band.target_peak_floor_ang == pytest.approx(0.016)
    assert patched.movement_utility.low_softness_ang == pytest.approx(0.004)
    assert patched.movement_utility.high_softness_ang == pytest.approx(0.016)
    assert patched.fullspace_confinement.rmsd_scale_ang == pytest.approx(0.4)


def test_geometry_novelty_disabled_preserves_absolute_core_acquisition_config():
    cfg = CampaignConfig()
    cfg.geometry_novelty.enabled = False
    acq_cfg = cfg.to_acquisition_config()

    patched = apply_geometry_novelty_to_acquisition_config(
        acq_cfg,
        cfg,
        {"scale_angstrom": 0.04},
    )

    assert patched is acq_cfg
