import hashlib
import json
import math

import pytest

from ichor.core.adversarial.error_calibration import lookup_calibrated_error
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.calibration_model import (
    make_calibration_table,
)
from ichor.hpc.active_learning.daemon.error_calibration import (
    ErrorCalibrationError,
    append_records,
    build_calibration_model,
    load_calibration_model_for_acquisition,
    load_records,
    lookup_calibrated_abs_error,
    mark_calibration_model_stale,
    model_path,
    records_path,
    synthetic_dry_records,
    update_from_aimall_acceptance,
    write_calibration_model,
)
from ichor.hpc.active_learning.daemon.error_calibration_contract import (
    CALIBRATION_ESTIMATOR,
    CALIBRATION_MODEL_SCHEMA_VERSION,
    CALIBRATION_OUTPUT_UNITS,
    CALIBRATION_RECORDS_SCHEMA_VERSION,
    UNBOUND_ENVIRONMENT_DIGEST,
    calibration_context_sha256,
    canonical_sha256,
    record_key,
    validate_model,
)
from ichor.hpc.active_learning.ferebus_prior import (
    resolve_ferebus_prior_contract,
)
from ichor.hpc.active_learning.versioning.provenance import (
    enrich_with_error_calibration_input,
    write_seed_provenance,
)


def _applicable_config() -> CampaignConfig:
    config = CampaignConfig()
    config.error_calibration.mode = "apply_to_acquisition"
    config.error_calibration.apply_strength = 1.0
    config.error_calibration.min_records_to_apply = 1
    config.error_calibration.min_model_versions_to_apply = 1
    config.error_calibration.min_bin_records = 1
    return config


def _record(
    point: int,
    atom: str = "C1",
    *,
    config: CampaignConfig | None = None,
    frame_atoms=None,
    iteration: int = 1,
    model_version: int = 0,
    raw_atom: float = 1.0,
    raw_total: float = 10.0,
    predicted: float = -1.0,
    truth: float = -1.1,
    environment_generation: int = 0,
    environment_digest: str = UNBOUND_ENVIRONMENT_DIGEST,
    model_set_digest: str | None = None,
):
    config = config or CampaignConfig()
    frame_atoms = tuple(frame_atoms or (atom,))
    prior_digest = resolve_ferebus_prior_contract(config).contract_sha256
    context_digest = calibration_context_sha256(
        config,
        prior_mean_contract_sha256=prior_digest,
        environment_generation_digest_sha256=environment_digest,
    )
    pointdir = "POINT_" + str(point).zfill(4) + ".pointdir"
    atom_identity = json.dumps(list(frame_atoms), separators=(",", ":")).encode()
    error = abs(float(predicted) - float(truth))
    record = {
        "schema_version": CALIBRATION_RECORDS_SCHEMA_VERSION,
        "iteration": int(iteration),
        "model_version": int(model_version),
        "model_set_sha256": model_set_digest
        or canonical_sha256({"model_version": int(model_version)}),
        "prior_mean_contract_sha256": prior_digest,
        "environment_generation": int(environment_generation),
        "environment_generation_digest_sha256": environment_digest,
        "calibration_context_sha256": context_digest,
        "pointdir": pointdir,
        "seed_id": int(point) + 1,
        "seed_uid": format(int(point) + 1, "064x"),
        "sampling_aggressiveness": int(config.campaign.sampling_aggressiveness),
        "frame_atom_count": len(frame_atoms),
        "frame_atom_identity_sha256": hashlib.sha256(atom_identity).hexdigest(),
        "atom": atom,
        "atom_type": atom.rstrip("0123456789"),
        "property": "iqa",
        "predicted_iqa_ha": float(predicted),
        "true_iqa_ha": float(truth),
        "abs_error_ha": float(error),
        "abs_error_ha_per_sqrt_atom": float(error),
        "raw_uncertainty": float(raw_atom),
        "raw_atom_variance": float(raw_atom),
        "raw_total_energy_variance": float(raw_total),
        "raw_total_score": 1.0,
        "landing_policy": "raw_final",
        "safety_metrics": {},
        "source_digests": {
            "provenance_json_sha256": canonical_sha256(
                {"point": point, "source": "provenance"}
            ),
            "result_json_sha256": canonical_sha256(
                {"point": point, "source": "result"}
            ),
            "sampling_protocol_sha256": canonical_sha256(
                {"iteration": iteration, "source": "protocol"}
            ),
        },
        "provenance": {
            "pointdir": pointdir,
            "provenance_json": pointdir + "/.provenance.json",
            "result_json": "result-" + str(point) + ".json",
        },
    }
    record["record_id"] = record_key(record)
    return record


def _frame_records(
    point: int,
    atoms=("C1", "H2"),
    **kwargs,
):
    return [
        _record(point, atom, frame_atoms=atoms, **kwargs)
        for atom in atoms
    ]


def test_append_records_is_idempotent_and_detects_conflicting_replay(tmp_path):
    records = [_record(0), _record(1)]
    merged, added, skipped = append_records(tmp_path, records)
    assert (added, skipped, len(merged)) == (2, 0, 2)

    replayed, added, skipped = append_records(tmp_path, records)
    assert (added, skipped) == (0, 2)
    assert replayed == merged

    conflict = dict(records[0])
    conflict["raw_total_score"] = 2.0
    with pytest.raises(ErrorCalibrationError, match="replay conflict"):
        append_records(tmp_path, [conflict])


def test_append_records_quarantines_malformed_file(tmp_path):
    path = records_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    merged, added, skipped = append_records(tmp_path, [_record(0)])
    assert (added, skipped, len(merged)) == (1, 0, 1)
    assert list(path.parent.glob(path.name + ".corrupt.*"))
    assert load_records(tmp_path) == merged


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(schema_version=True),
        lambda payload: payload.update(n_records=None),
        lambda payload: payload.update(n_records=2),
        lambda payload: payload.update(records=[None]),
        lambda payload: payload.update(records=[payload["records"][0]] * 2, n_records=2),
    ],
)
def test_records_reader_rejects_malformed_authoritative_shapes(tmp_path, mutation):
    append_records(tmp_path, [_record(0)])
    path = records_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ErrorCalibrationError):
        load_records(tmp_path)


def test_retention_never_splits_a_molecular_frame(tmp_path):
    records = [
        record
        for point in (0, 1)
        for record in _frame_records(point)
    ]
    retained, added, skipped = append_records(tmp_path, records, max_records=3)
    assert (added, skipped) == (4, 0)
    assert len(retained) == 2
    assert {record["pointdir"] for record in retained} == {
        "POINT_0001.pointdir"
    }


def test_tied_predictors_are_never_split_between_bins():
    rows = [
        {
            "raw_uncertainty": raw,
            "abs_error_ha_per_sqrt_atom": error,
        }
        for raw, error in ((1.0, 0.1), (1.0, 0.2), (1.0, 0.3), (2.0, 0.4))
    ]
    first = make_calibration_table(
        rows, n_bins=4, min_bin_records=1, quantile=0.75
    )
    second = make_calibration_table(
        list(reversed(rows)), n_bins=4, min_bin_records=1, quantile=0.75
    )
    assert first == second
    assert first["bins"][0]["n"] == 3
    assert first["bins"][0]["raw_uncertainty_min"] == 1.0
    assert first["bins"][0]["raw_uncertainty_max"] == 1.0


def test_isotonic_pooling_recomputes_the_requested_quantile():
    rows = [
        {"raw_uncertainty": 1.0, "abs_error_ha_per_sqrt_atom": 100.0},
        {"raw_uncertainty": 2.0, "abs_error_ha_per_sqrt_atom": 1.0},
        {"raw_uncertainty": 2.0, "abs_error_ha_per_sqrt_atom": 1.0},
        {"raw_uncertainty": 2.0, "abs_error_ha_per_sqrt_atom": 1.0},
    ]
    table = make_calibration_table(
        rows, n_bins=2, min_bin_records=1, quantile=0.75
    )
    fitted = [
        entry["calibrated_abs_error_ha_per_sqrt_atom"]
        for entry in table["bins"]
    ]
    assert fitted == pytest.approx([25.75, 25.75])
    assert fitted != pytest.approx([50.5, 50.5])


def test_total_error_is_normalised_by_sqrt_atom_count():
    config = _applicable_config()
    records = _frame_records(
        0,
        atoms=("C1", "H2", "H3", "H4"),
        config=config,
        predicted=-1.0,
        truth=-1.1,
        raw_total=8.0,
    )
    model = build_calibration_model(records, config, iteration=1)
    total_bin = model["tables"]["global_total"]["bins"][0]
    assert total_bin["calibrated_abs_error_ha_per_sqrt_atom"] == pytest.approx(
        0.2
    )
    assert model["reference_error_ha_per_sqrt_atom"] == pytest.approx(0.2)
    assert model["output_units"] == CALIBRATION_OUTPUT_UNITS


def test_incomplete_molecular_frames_do_not_enter_any_fitted_table():
    config = _applicable_config()
    incomplete = _record(
        0,
        "C1",
        config=config,
        frame_atoms=("C1", "H2"),
    )
    model = build_calibration_model([incomplete], config, iteration=1)
    assert model["n_records"] == 0
    assert model["n_total_error_records"] == 0
    assert not model["tables"]["global_total"]["usable"]


def test_rolling_model_normalises_each_model_uncertainty_axis():
    config = _applicable_config()
    config.error_calibration.min_records_to_apply = 2
    config.error_calibration.min_model_versions_to_apply = 2
    records = [
        _record(
            0,
            config=config,
            model_version=0,
            iteration=1,
            raw_total=100.0,
            predicted=-1.0,
            truth=-1.2,
        ),
        _record(
            1,
            config=config,
            model_version=1,
            iteration=2,
            raw_total=2.0,
            predicted=-1.0,
            truth=-1.2,
        ),
    ]
    model = build_calibration_model(
        records, config, iteration=2, current_model_version=1
    )
    assert model["model_policy"] == "rolling_normalised"
    assert model["model_uncertainty_normalisation"]["0"][
        "raw_uncertainty_median"
    ] == 100.0
    assert model["model_uncertainty_normalisation"]["1"][
        "raw_uncertainty_median"
    ] == 2.0
    assert model["contributing_model_versions"] == [0, 1]
    assert model["usable_for_acquisition"]


def test_legitimate_zero_error_is_preserved_by_both_lookup_paths():
    config = _applicable_config()
    record = _record(
        0,
        config=config,
        predicted=-1.0,
        truth=-1.0,
        raw_total=2.0,
    )
    model = build_calibration_model([record], config, iteration=1)
    assert model["reference_error_ha_per_sqrt_atom"] == 0.0
    assert lookup_calibrated_abs_error(
        model, 2.0, application_uncertainty_scale=2.0
    ) == 0.0
    assert lookup_calibrated_error(
        model, 2.0, application_uncertainty_scale=2.0
    ) == 0.0


def test_model_is_bound_to_estimator_settings_and_environment(tmp_path):
    config = _applicable_config()
    records = [_record(0, config=config)]
    model = build_calibration_model(records, config, iteration=1)
    write_calibration_model(tmp_path, model)

    loaded, reason = load_calibration_model_for_acquisition(
        tmp_path, config, current_model_version=0, current_iteration=1
    )
    assert reason == "loaded"
    assert loaded == model

    config.error_calibration.quantile = 0.5
    loaded, reason = load_calibration_model_for_acquisition(
        tmp_path, config, current_model_version=0, current_iteration=1
    )
    assert loaded is None
    assert reason == "estimator_settings_changed"


def test_environment_mismatch_rejects_model(tmp_path, monkeypatch):
    config = _applicable_config()
    model = build_calibration_model(
        [_record(0, config=config)], config, iteration=1
    )
    write_calibration_model(tmp_path, model)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.error_calibration.active_environment_binding",
        lambda _campaign: {
            "generation": 1,
            "generation_digest_sha256": "f" * 64,
            "bound": True,
        },
    )
    loaded, reason = load_calibration_model_for_acquisition(
        tmp_path, config, current_model_version=0, current_iteration=1
    )
    assert loaded is None
    assert reason == "environment_generation_mismatch"


def test_model_age_is_checked_at_application_time(tmp_path):
    config = _applicable_config()
    config.error_calibration.max_model_age_iterations = 1
    model = build_calibration_model(
        [_record(0, config=config)], config, iteration=1
    )
    write_calibration_model(tmp_path, model)
    loaded, reason = load_calibration_model_for_acquisition(
        tmp_path, config, current_model_version=0, current_iteration=3
    )
    assert loaded is None
    assert reason == "calibration_model_too_old"


def test_record_only_mode_is_behaviourally_inactive(tmp_path):
    config = _applicable_config()
    model = build_calibration_model(
        [_record(0, config=config)], config, iteration=1
    )
    write_calibration_model(tmp_path, model)
    config.error_calibration.mode = "record_only"
    loaded, reason = load_calibration_model_for_acquisition(
        tmp_path, config, current_model_version=0
    )
    assert loaded is None
    assert reason == "record_only"


def test_malformed_model_is_quarantined_before_acquisition(tmp_path):
    config = _applicable_config()
    path = model_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version": true}', encoding="utf-8")
    loaded, reason = load_calibration_model_for_acquisition(tmp_path, config)
    assert loaded is None
    assert reason == "malformed_model"
    assert not path.exists()
    assert list(path.parent.glob(path.name + ".corrupt.*"))


def test_stale_model_is_valid_but_never_loaded(tmp_path):
    config = _applicable_config()
    mark_calibration_model_stale(tmp_path, reason="build failed", iteration=3)
    loaded, reason = load_calibration_model_for_acquisition(tmp_path, config)
    assert loaded is None
    assert reason == "stale_model"


def test_model_validator_rejects_overlapping_bins():
    config = _applicable_config()
    records = [
        _record(0, config=config, raw_total=1.0),
        _record(1, config=config, raw_total=2.0),
    ]
    model = build_calibration_model(records, config, iteration=1)
    broken = json.loads(json.dumps(model))
    bins = broken["tables"]["global_total"]["bins"]
    if len(bins) == 1:
        duplicate = dict(bins[0])
        broken["tables"]["global_total"]["bins"].append(duplicate)
        broken["tables"]["global_total"]["n_records"] *= 2
    else:
        bins[1]["raw_uncertainty_min"] = bins[0]["raw_uncertainty_max"]
    with pytest.raises(Exception, match="overlap|split tied"):
        validate_model(broken)


def test_synthetic_records_bind_to_the_supplied_config():
    config = _applicable_config()
    config.campaign.sampling_aggressiveness = 9
    records = synthetic_dry_records(
        iteration=3,
        models_version=2,
        n_points=2,
        sampling_aggressiveness=9,
        config=config,
    )
    model = build_calibration_model(records, config, iteration=3)
    assert {row["sampling_aggressiveness"] for row in records} == {9}
    assert model["n_records"] == 2


def test_update_from_aimall_acceptance_joins_content_bound_sources(tmp_path):
    from ichor.hpc.active_learning.layout import active_iteration_dir

    campaign = tmp_path
    iteration_dir = active_iteration_dir(campaign, 1)
    pointdir = campaign / ".DATA" / "STAGING" / "iter_1" / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True)
    result_path = iteration_dir / "ariadne" / "seeds" / "seed-000003" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text('{"accepted": true}\n', encoding="utf-8")
    config = CampaignConfig()
    prior_digest = resolve_ferebus_prior_contract(config).contract_sha256
    context_digest = calibration_context_sha256(
        config,
        prior_mean_contract_sha256=prior_digest,
        environment_generation_digest_sha256=UNBOUND_ENVIRONMENT_DIGEST,
    )
    protocol_digest = "d" * 64
    write_seed_provenance(
        pointdir,
        campaign_uid="uid",
        iteration=1,
        trajectory_sha256="a" * 64,
        seed_frame_id=5,
        seed_id=3,
        seed_uid="c" * 64,
        array_task_id_zero_based=2,
        seed_selection_origin="variance",
        seed_variance_at_selection=2.0,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
    )
    enrich_with_error_calibration_input(
        pointdir,
        {
            "property": "iqa",
            "model_version": 0,
            "model_set_sha256": "b" * 64,
            "prior_mean_contract_sha256": prior_digest,
            "environment_generation": 0,
            "environment_generation_digest_sha256": UNBOUND_ENVIRONMENT_DIGEST,
            "calibration_context_sha256": context_digest,
            "sampling_protocol_sha256": protocol_digest,
            "seed_id": 3,
            "seed_uid": "c" * 64,
            "result_json": result_path.relative_to(campaign).as_posix(),
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
            "pointdir": pointdir.name,
            "accepted": True,
            "per_atom": [{"atom": "C1", "iqa_ha": -1.125}],
        }
    ]
    audit = update_from_aimall_acceptance(
        campaign_dir=campaign,
        iter_dir=iteration_dir,
        config=config,
        iteration=1,
        models_version=0,
        accepted_pointdirs=[pointdir],
        quality_records=quality,
    )
    assert audit["n_added_records"] == 1
    record = load_records(campaign)[0]
    assert record["abs_error_ha"] == 0.125
    assert record["abs_error_ha_per_sqrt_atom"] == 0.125
    assert record["source_digests"]["sampling_protocol_sha256"] == protocol_digest
    assert math.isfinite(record["raw_uncertainty"])


def test_update_from_aimall_acceptance_rejects_external_pointdir(tmp_path):
    from ichor.hpc.active_learning.layout import active_iteration_dir

    campaign = tmp_path / "campaign"
    external_pointdir = tmp_path / "external" / "POINT_0000.pointdir"
    external_pointdir.mkdir(parents=True)

    audit = update_from_aimall_acceptance(
        campaign_dir=campaign,
        iter_dir=active_iteration_dir(campaign, 1),
        config=CampaignConfig(),
        iteration=1,
        models_version=0,
        accepted_pointdirs=[external_pointdir],
        quality_records=[
            {
                "pointdir": external_pointdir.name,
                "accepted": True,
                "per_atom": [{"atom": "C1", "iqa_ha": -1.0}],
            }
        ],
    )

    assert audit["n_new_records"] == 0
    assert audit["skipped"] == {"pointdir_path_escapes_campaign": 1}


def test_model_schema_and_estimator_are_fixed_internal_contracts():
    config = _applicable_config()
    model = build_calibration_model(
        [_record(0, config=config)], config, iteration=1
    )
    assert model["schema_version"] == CALIBRATION_MODEL_SCHEMA_VERSION
    assert model["estimator"] == CALIBRATION_ESTIMATOR
    assert model["model_policy"] == "rolling_normalised"
    assert "monotone_estimator" not in model
    assert "model_version_policy" not in model
