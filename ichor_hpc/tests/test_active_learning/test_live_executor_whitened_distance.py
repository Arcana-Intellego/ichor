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
from ichor.hpc.active_learning.versioning.provenance import (
    PROVENANCE_FILENAME, write_seed_provenance,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live_outputs"


def _seed_iter_pool(campaign_dir, iteration, results):
    """Drop synthetic per-seed result.json files into the iteration pool."""
    iter_dir = (
        campaign_dir / "7_ACTIVE_LEARNING"
        / f"iteration-{iteration:04d}"
    )
    pool_dir = iter_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = []
    seed_records = []
    for idx, payload in enumerate(results):
        sd = pool_dir / f"seed_{idx:04d}"
        sd.mkdir()
        payload = dict(payload)
        payload.setdefault("atom_types", ["O", "H", "H"])
        payload.setdefault(
            "final_coordinates",
            [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)],
        )
        payload.setdefault("alpha_trajectory", [payload["alpha_initial"], payload["alpha_final"]])
        payload["iteration"] = int(iteration)
        payload["seed_index"] = int(idx)
        payload["seed_frame_id"] = int(idx)
        (sd / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        # write the provenance sidecar that enrich_with_anti_overlap needs
        write_seed_provenance(
            sd, campaign_uid="u", iteration=iteration,
            trajectory_sha256="0" * 64,
            seed_frame_id=idx,
            seed_selection_origin="test",
            seed_variance_at_selection=0.0,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
        )
        frame_ids.append(int(idx))
        seed_records.append({
            "seed_index": int(idx),
            "frame_id": int(idx),
            "selection_index": int(idx),
            "selection_origin": "bulk",
            "variance_at_selection": 0.0,
        })
    (iter_dir / "seeds_picked.json").write_text(
        json.dumps({
            "schema_version": 1,
            "iteration": int(iteration),
            "n_picked": len(results),
            "frame_ids": frame_ids,
            "indices": frame_ids,
            "bulk_indices": frame_ids,
            "variance_indices": [],
            "variances": [0.0 for _ in results],
            "seed_records": seed_records,
            "trajectory_sha256": "0" * 64,
        }),
        encoding="utf-8",
    )
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
        tmp_path / "campaign", 0,
        [{"alpha_initial": 0.5, "alpha_final": 1.0,
          "whitened_distance_final": 0.42,
          "n_evaluations": 1, "return_code": 0,
          "wall_seconds": 1.0, "fell_back_to_ds": False}],
    )
    state = SimpleNamespace(iteration=0, campaign_uid="u")
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.is_complete is True
    # the seed provenance should now have anti_overlap with d ~= 0.42
    from ichor.hpc.active_learning.versioning.provenance import read_provenance
    pool_dir = (
        tmp_path / "campaign"
        / "7_ACTIVE_LEARNING" / "iteration-0000" / "pool"
    )
    prov = read_provenance(pool_dir / "seed_0000")
    assert prov["anti_overlap"] is not None
    d = prov["anti_overlap"]["min_whitened_distance_to_training"]
    assert d == pytest.approx(0.42, abs=1.0e-6)


def test_postprocess_uses_campaign_anti_overlap_bounds(tmp_path):
    cfg = CampaignConfig()
    cfg.anti_overlap.min_post_ariadne_whitened_distance = 0.5
    cfg.anti_overlap.max_post_ariadne_whitened_distance = 2.0
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )
    _seed_iter_pool(
        tmp_path / "campaign", 0,
        [{"alpha_initial": 0.5, "alpha_final": 1.0,
          "whitened_distance_final": 0.42,
          "n_evaluations": 1, "return_code": 0,
          "wall_seconds": 1.0, "fell_back_to_ds": False}],
    )
    result = ex._parse_ariadne_array_postprocess(
        SimpleNamespace(iteration=0, campaign_uid="u"),
        CampaignPhase("ARIADNE_ARRAY"),
        observations=[],
    )
    assert result.is_complete is True
    from ichor.hpc.active_learning.versioning.provenance import read_provenance
    prov = read_provenance(
        tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0000" / "pool" / "seed_0000"
    )
    assert prov["anti_overlap"]["flag"] == "moved_too_little"


def test_postprocess_falls_back_to_synthetic_when_field_missing(tmp_path):
    ex = _make_executor(tmp_path)
    # NO whitened_distance_final key -- the parser should fall back to
    # the synthetic |alpha_final - alpha_initial| = 0.5 proxy.
    _seed_iter_pool(
        tmp_path / "campaign", 0,
        [{"alpha_initial": 0.5, "alpha_final": 1.0,
          "n_evaluations": 1, "return_code": 0,
          "wall_seconds": 1.0, "fell_back_to_ds": False}],
    )
    state = SimpleNamespace(iteration=0, campaign_uid="u")
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.is_complete is True
    from ichor.hpc.active_learning.versioning.provenance import read_provenance
    pool_dir = (
        tmp_path / "campaign"
        / "7_ACTIVE_LEARNING" / "iteration-0000" / "pool"
    )
    prov = read_provenance(pool_dir / "seed_0000")
    d = prov["anti_overlap"]["min_whitened_distance_to_training"]
    assert d == pytest.approx(0.5, abs=1.0e-6)


def test_postprocess_rejects_explicit_unsafe_landing(tmp_path):
    ex = _make_executor(tmp_path)
    iter_dir = _seed_iter_pool(
        tmp_path / "campaign", 0,
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
    state = SimpleNamespace(iteration=0, campaign_uid="u")
    result = ex._parse_ariadne_array_postprocess(
        state, CampaignPhase("ARIADNE_ARRAY"), observations=[],
    )
    assert result.is_complete is True
    assert "ariadne_no_seed_results_parsed" in result.failure_reason
    from ichor.hpc.active_learning.handoff_manifests import (
        read_ariadne_landing_audit,
        read_ariadne_results_manifest,
    )
    audit = read_ariadne_landing_audit(iter_dir, expected_iteration=0)
    assert audit["summary"]["rejected"] == 1
    manifest = read_ariadne_results_manifest(
        iter_dir, expected_iteration=0, require_nonempty=False,
    )
    assert manifest["accepted"] == []
    assert manifest["rejected"][0]["reason"] == "no_safe_non_seed_landing"
