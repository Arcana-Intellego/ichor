import json

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.error_calibration import (
    ERROR_CALIBRATION_AUDIT_FILENAME,
    append_records,
    build_calibration_model,
    load_calibration_model_for_acquisition,
    load_records,
    lookup_calibrated_abs_error,
    synthetic_dry_records,
    update_from_aimall_acceptance,
    write_calibration_model,
)
from ichor.hpc.active_learning.versioning.provenance import (
    enrich_with_error_calibration_input,
    write_seed_provenance,
)


def _record(i, *, atom_type="C", raw=None, err=None):
    raw = float(i + 1) if raw is None else float(raw)
    err = float(i + 1) * 1.0e-4 if err is None else float(err)
    return {
        "schema_version": 1,
        "iteration": 1,
        "model_version": 0,
        "pointdir": "POINT_" + str(i).zfill(4) + ".pointdir",
        "seed_index": i,
        "atom": "C1",
        "atom_type": atom_type,
        "property": "iqa",
        "predicted_iqa_ha": -1.0,
        "true_iqa_ha": -1.0 - err,
        "abs_error_ha": err,
        "raw_uncertainty": raw,
        "raw_atom_variance": raw,
        "raw_total_energy_variance": raw,
        "raw_total_score": 1.0,
        "landing_policy": "raw_final",
        "safety_metrics": {},
        "provenance": {},
    }


def _atom_record(point, atom, *, model_version=0, raw_atom=1.0, raw_total=10.0, pred=-1.0, truth=-1.1):
    return {
        "schema_version": 1,
        "iteration": 1,
        "model_version": int(model_version),
        "pointdir": "POINT_" + str(point).zfill(4) + ".pointdir",
        "seed_index": int(point),
        "atom": atom,
        "atom_type": atom[0],
        "property": "iqa",
        "predicted_iqa_ha": float(pred),
        "true_iqa_ha": float(truth),
        "abs_error_ha": float(abs(pred - truth)),
        "raw_uncertainty": float(raw_atom),
        "raw_atom_variance": float(raw_atom),
        "raw_total_energy_variance": float(raw_total),
        "raw_total_score": 1.0,
        "landing_policy": "raw_final",
        "safety_metrics": {},
        "provenance": {},
    }


def test_append_records_is_idempotent(tmp_path):
    records = [_record(0), _record(1)]
    merged, added, skipped = append_records(tmp_path, records)
    assert added == 2
    assert skipped == 0
    assert len(merged) == 2

    merged, added, skipped = append_records(tmp_path, records)
    assert added == 0
    assert skipped == 2
    assert load_records(tmp_path) == merged


def test_calibration_model_builds_monotone_bins_and_sparse_group_fallback():
    cfg = CampaignConfig()
    cfg.error_calibration.n_bins = 3
    cfg.error_calibration.min_bin_records = 2
    records = [
        _record(0, atom_type="C", raw=1.0, err=0.003),
        _record(1, atom_type="C", raw=2.0, err=0.001),
        _record(2, atom_type="C", raw=3.0, err=0.002),
        _record(3, atom_type="H", raw=4.0, err=0.006),
    ]

    model = build_calibration_model(records, cfg, iteration=2)

    assert model["schema_version"] == 1
    global_bins = model["tables"]["global"]["bins"]
    calibrated = [b["calibrated_abs_error_ha"] for b in global_bins]
    assert calibrated == sorted(calibrated)
    assert "global_total" in model["tables"]
    assert "atom_type:C" in model["tables"]
    assert "atom_type:H" not in model["tables"]


def test_calibration_global_total_table_uses_total_variance_and_total_error():
    cfg = CampaignConfig()
    cfg.error_calibration.n_bins = 2
    cfg.error_calibration.min_bin_records = 1
    records = [
        _atom_record(0, "C1", raw_atom=1.0, raw_total=20.0, pred=-1.0, truth=-1.5),
        _atom_record(0, "H2", raw_atom=1.0, raw_total=20.0, pred=-1.0, truth=-0.8),
        _atom_record(1, "C1", raw_atom=2.0, raw_total=40.0, pred=-1.0, truth=-1.1),
        _atom_record(1, "H2", raw_atom=2.0, raw_total=40.0, pred=-1.0, truth=-1.2),
    ]

    model = build_calibration_model(records, cfg, iteration=1, current_model_version=0)

    total_bins = model["tables"]["global_total"]["bins"]
    assert total_bins[0]["raw_uncertainty_max"] == 20.0
    assert total_bins[0]["median_abs_error_ha"] == pytest.approx(0.3)
    assert lookup_calibrated_abs_error(model, 20.0) == pytest.approx(
        total_bins[0]["calibrated_abs_error_ha"]
    )


def test_calibration_model_defaults_to_current_model_version():
    cfg = CampaignConfig()
    cfg.error_calibration.min_bin_records = 1
    old_records = [
        _atom_record(0, "C1", model_version=0, raw_atom=1.0, raw_total=1.0, pred=-1.0, truth=-3.0),
    ]
    current_records = [
        _atom_record(1, "C1", model_version=2, raw_atom=2.0, raw_total=2.0, pred=-1.0, truth=-1.1),
    ]

    model = build_calibration_model(
        old_records + current_records,
        cfg,
        iteration=3,
        current_model_version=2,
    )

    assert model["model_version_policy"] == "current"
    assert model["current_model_version"] == 2
    assert model["n_records"] == 1
    assert model["reference_error_ha"] == pytest.approx(0.1)


def test_load_calibration_model_for_acquisition_requires_apply_mode_and_records(tmp_path):
    cfg = CampaignConfig()
    cfg.error_calibration.mode = "record_only"
    model = build_calibration_model(synthetic_dry_records(iteration=1, models_version=0, n_points=4), cfg, iteration=1)
    write_calibration_model(tmp_path, model)
    loaded, reason = load_calibration_model_for_acquisition(tmp_path, cfg)
    assert loaded is None
    assert reason == "record_only"

    cfg.error_calibration.mode = "apply_to_acquisition"
    cfg.error_calibration.apply_strength = 0.5
    cfg.error_calibration.min_records_to_apply = 2
    cfg.error_calibration.min_bin_records = 2
    model = build_calibration_model(synthetic_dry_records(iteration=1, models_version=0, n_points=4), cfg, iteration=1)
    write_calibration_model(tmp_path, model)
    loaded, reason = load_calibration_model_for_acquisition(tmp_path, cfg)
    assert reason == "loaded"
    assert loaded["usable_for_acquisition"] is True


def test_update_from_aimall_acceptance_joins_provenance_and_quality(tmp_path):
    campaign = tmp_path
    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0001"
    pointdir = campaign / ".DATA" / "STAGING" / "iter_1" / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True)
    write_seed_provenance(
        pointdir,
        campaign_uid="uid",
        iteration=1,
        trajectory_sha256="sha",
        seed_frame_id=5,
        seed_selection_origin="variance",
        seed_variance_at_selection=2.0,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
    )
    enrich_with_error_calibration_input(
        pointdir,
        {
            "schema_version": 1,
            "property": "iqa",
            "model_version": 0,
            "seed_index": 3,
            "result_json": str(iter_dir / "pool" / "seed_0003" / "result.json"),
            "total_energy_variance": 2.0,
            "raw_total_score": 1.5,
            "landing_policy": "raw_final",
            "safety_metrics": {"whitened_distance": 0.2},
            "per_atom": [
                {
                    "atom": "C1",
                    "atom_type": "C",
                    "property": "iqa",
                    "predicted_iqa_ha": -1.0,
                    "raw_variance": 2.0,
                }
            ],
        },
    )
    quality = [
        {
            "pointdir": "POINT_0000.pointdir",
            "accepted": True,
            "per_atom": [{"atom": "C1", "iqa_ha": -1.125}],
        }
    ]
    cfg = CampaignConfig()

    audit = update_from_aimall_acceptance(
        campaign_dir=campaign,
        iter_dir=iter_dir,
        config=cfg,
        iteration=1,
        models_version=0,
        accepted_pointdirs=[pointdir],
        quality_records=quality,
    )

    assert audit["n_added_records"] == 1
    records = load_records(campaign)
    assert records[0]["abs_error_ha"] == 0.125
    assert records[0]["raw_uncertainty"] == 2.0
    assert (iter_dir / ERROR_CALIBRATION_AUDIT_FILENAME).is_file()
    audit_payload = json.loads((iter_dir / ERROR_CALIBRATION_AUDIT_FILENAME).read_text())
    assert audit_payload["n_total_records"] == 1
