"""FEREBUS quality sidecar tests."""
import json
import hashlib
import numpy as np
import pytest

from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.daemon.ferebus_quality import (
    FEREBUS_QUALITY_DECISION_MANIFEST,
    FEREBUS_QUALITY_DECISION_POLICY,
    FEREBUS_QUALITY_MANIFEST,
    FerebusQualityMeasurementIncomplete,
    _parse_perf,
    evaluate_ferebus_quality,
    enrich_task_receipt_with_quality,
    validate_ferebus_task_measurement,
    evaluate_ferebus_quality_decision,
    read_ferebus_quality_decision,
    _stream_model_metrics,
    validate_ferebus_quality_evidence,
    write_ferebus_quality_decision,
    write_ferebus_quality_manifest,
)
from ichor.hpc.active_learning.daemon.ferebus_model_factors import (
    FerebusModelFactorError,
    publish_task_factor,
    read_task_factor,
)
from ichor.hpc.active_learning.daemon.ferebus_model_admission import (
    FerebusModelAdmissionError,
    build_ferebus_model_admission_context,
    enrich_task_receipt_with_model_admission,
    validate_ferebus_task_model_admission,
)
import ichor.hpc.active_learning.daemon.ferebus_quality as ferebus_quality_module
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.ferebus_prior import (
    backend_kernel_token,
    resolve_ferebus_prior_contract,
    validate_ferebus_config_contract,
)
from ichor.hpc.active_learning.daemon.ferebus_task_runner import (
    write_preexisting_model_receipts,
)
from ichor.hpc.active_learning.daemon.state import atomic_write_json
from ichor.hpc.active_learning.submit.pyferebus_wrap import (
    _write_structured_task_map,
)
from ichor.hpc.active_learning.versioning.manifest import sha256_file
from ichor.hpc.active_learning.versioning.reference_data import canonical_json_sha256


PRIOR = resolve_ferebus_prior_contract(CampaignConfig())
OXYGEN_PRIOR = PRIOR.expected_mean_ha("iqa", "O1")


def _gates(**overrides):
    gates = CampaignConfig().quality_gates
    for name, value in overrides.items():
        setattr(gates, name, value)
    return gates


def _write_zero_model(path, *, atom="O1", ntrain=3, nfeats=3):
    rows = [
        [0.1 * (j + 1) + i * 0.1 for j in range(nfeats)]
        for i in range(ntrain)
    ]
    lines = [
        "# jitter 1.0e-6",
        "# likelihood -1.0",
        "",
        "[system]",
        "name WATER",
        "atom " + atom,
        "property iqa",
        "ALF 1 2 3",
        "",
        "[dimensions]",
        "number_of_atoms 3",
        "number_of_features " + str(nfeats),
        "number_of_training_points " + str(ntrain),
        "",
        "[mean]",
        "type constant",
        "value " + repr(OXYGEN_PRIOR),
        "",
        "[kernels]",
        "number_of_kernels 1",
        "composition k1",
        "prefactor 1.0",
        "",
        "[kernel.k1]",
        "type rbf",
        "number_of_dimensions " + str(nfeats),
        "active_dimensions " + " ".join(str(i + 1) for i in range(nfeats)),
        "thetas " + " ".join("1.0" for _ in range(nfeats)),
        "",
        "[training_data]",
        "units.x " + " ".join("bohr" for _ in range(nfeats)),
        "units.y Ha",
        "",
        "[training_data.x]",
    ]
    lines += [" ".join(str(v) for v in row) for row in rows]
    lines += ["", "[training_data.y]"]
    lines += [repr(OXYGEN_PRIOR) for _ in range(ntrain)]
    lines += ["", "[weights]"]
    lines += ["0.0" for _ in range(ntrain)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dataset(path, targets):
    lines = ["f1,f2,f3,iqa"]
    for i, target in enumerate(targets):
        lines.append(f"{0.1+i*0.1},{0.2+i*0.1},{0.3+i*0.1},{target}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _dataset_identity(path, rows, root):
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": int(rows),
    }


def _seed_quality_staging(
    tmp_path,
    *,
    ext_targets=None,
    campaign_layout=False,
):
    if ext_targets is None:
        ext_targets = (OXYGEN_PRIOR, OXYGEN_PRIOR)
    staging = (
        tmp_path / "campaign" / "TRAINED_MODELS" / "iteration-staging"
        if campaign_layout
        else tmp_path / "iteration-staging"
    )
    task_dir = staging / "iqa" / "O1"
    datasets_dir = task_dir / "datasets"
    datasets_dir.mkdir(parents=True)
    model = task_dir / "WATER_iqa_O1.model"
    _write_zero_model(model)
    train_csv = datasets_dir / "WATER_O1_TRAINING_SET.csv"
    int_csv = datasets_dir / "WATER_O1_INT_VALIDATION_SET.csv"
    ext_csv = datasets_dir / "WATER_O1_EXT_VALIDATION_SET.csv"
    _write_dataset(train_csv, (OXYGEN_PRIOR,) * 3)
    _write_dataset(int_csv, (OXYGEN_PRIOR,) * 2)
    _write_dataset(ext_csv, ext_targets)
    config_path = task_dir / "ferebus.config"
    config_path.write_text(
        "mean_type = 21\n"
        'level_of_theory = "b3lyp/aug-cc-pvtz"\n'
        "iqaDeviationFactor = 1.0\n"
        "scaling = 1\n"
        "scale_feats = 1\n"
        "scale_prop = 0\n",
        encoding="utf-8",
        newline="\n",
    )
    parsed_contract = validate_ferebus_config_contract(config_path, PRIOR)
    row_order = [
        "POINT_" + str(index).zfill(6) + ".pointdir"
        for index in range(7)
    ]
    split_rows = {
        "train": [0, 1, 2],
        "int_val": [3, 4],
        "ext_val": [5, 6],
    }
    source_rows = [
        {
            "source_row_index": index,
            "pointdir_name": name,
            "introduced_in_version": 0,
            "split": next(
                split for split, indexes in split_rows.items() if index in indexes
            ),
            "provenance_sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
        }
        for index, name in enumerate(row_order)
    ]
    split_identities = {
        split: [source_rows[index] for index in indexes]
        for split, indexes in split_rows.items()
    }
    row_identity_payload = {
        "schema_version": stg.FEREBUS_ROW_IDENTITIES_SCHEMA_VERSION,
        "campaign_uid": "quality-test",
        "reference_data_version": 0,
        "reference_data_view_sha256": "b" * 64,
        "source_rows": source_rows,
        "source_rows_sha256": canonical_json_sha256(source_rows),
        "splits": {
            split: {
                "rows": rows,
                "n_rows": len(rows),
                "row_identity_sha256": canonical_json_sha256(rows),
            }
            for split, rows in split_identities.items()
        },
    }
    row_identity_path = staging / stg.FEREBUS_ROW_IDENTITIES
    atomic_write_json(row_identity_path, row_identity_payload)
    split_payload = {
        "schema_version": 7,
        "allocation_policy": "exact_per_reference_data_version",
        "historical_training_rows": 0,
        "assignments": {
            name: {
                "split": source_rows[index]["split"],
                "first_seen_reference_data_version": 0,
                "assignment_version": 4,
                "allocation_manifest_sha256": "c" * 64,
                "provenance_sha256": source_rows[index]["provenance_sha256"],
            }
            for index, name in enumerate(row_order)
        },
        "version_allocations": {},
    }
    split_path = staging / stg.FEREBUS_SPLIT_SNAPSHOT
    atomic_write_json(split_path, split_payload)
    datasets = {
        "train": _dataset_identity(train_csv, 3, staging),
        "int_val": _dataset_identity(int_csv, 2, staging),
        "ext_val": _dataset_identity(ext_csv, 2, staging),
    }
    for split, record in datasets.items():
        record["row_identity_sha256"] = row_identity_payload["splits"][split][
            "row_identity_sha256"
        ]
        record["row_identity_count"] = record["rows"]
    manifest = {
            "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
            "campaign_uid": "quality-test",
            "system": "WATER",
            "reference_data_version": 0,
            "reference_data_head_manifest_sha256": "a" * 64,
            "reference_data_view_sha256": "b" * 64,
            "n_reference_points": 7,
            "pointdir_row_order": row_order,
            "properties": ["iqa"],
            "atoms": ["O1"],
            "n_atoms": 1,
            "n_tasks": 1,
            "prior_mean_contract": PRIOR.to_dict(),
            "kernel_contract": {
                "family": "rbf",
                "backend_token": backend_kernel_token("rbf"),
                "loss": "huber",
                "constant_noise": True,
                "full_ard": True,
                "feature_scaling": True,
                "property_scaling": False,
                "kernel_prefactor_mode": 2,
            },
            "row_identity_snapshot": {
                "path": stg.FEREBUS_ROW_IDENTITIES,
                "size": row_identity_path.stat().st_size,
                "sha256": sha256_file(row_identity_path),
                "source_rows_sha256": row_identity_payload["source_rows_sha256"],
            },
            "split_ledger": {
                "path": stg.FEREBUS_SPLIT_SNAPSHOT,
                "size": split_path.stat().st_size,
                "sha256": sha256_file(split_path),
                "counts": {"train": 3, "int_val": 2, "ext_val": 2},
                "version_allocation": {},
                "allocation_policy": "exact_per_reference_data_version",
                "allocation_manifest": "test",
                "allocation_manifest_sha256": "c" * 64,
                "forced_splits": {
                    name: source_rows[index]["split"]
                    for index, name in enumerate(row_order)
                },
            },
            "degenerate_property_stats": [],
            "tasks": [{
                "task_index": 1,
                "property": "iqa",
                "atom": "O1",
                "prior_mean": PRIOR.task_payload(
                    "iqa",
                    "O1",
                    training_values=[OXYGEN_PRIOR] * 3,
                    training_dataset_sha256=sha256_file(train_csv),
                ),
                "alf_1_indexed": [1, 2, 3],
                "alf_cli": "1_2_3",
                "property_dir": "iqa",
                "output_dir": "iqa/O1",
                "input_dir": "iqa/O1/datasets",
                "config_path": "iqa/O1/ferebus.config",
                "expected_model_path": "iqa/O1/WATER_iqa_O1.model",
                "training_csv": "iqa/O1/datasets/WATER_O1_TRAINING_SET.csv",
                "int_validation_csv": "iqa/O1/datasets/WATER_O1_INT_VALIDATION_SET.csv",
                "ext_validation_csv": "iqa/O1/datasets/WATER_O1_EXT_VALIDATION_SET.csv",
                "command_args": [
                    "-c", "iqa/O1/ferebus.config",
                    "-I", "iqa/O1/datasets",
                    "-O", "iqa/O1",
                    "-P", "iqa",
                    "-A", "O1",
                    "-ALF", "1_2_3",
                ],
                "row_counts": {"train": 3, "int_val": 2, "ext_val": 2},
                "historical_training_rows": 0,
                "historical_training_row_ids": [],
                "row_ids": split_rows,
                "datasets": datasets,
                "generated_config": {
                    "path": "iqa/O1/ferebus.config",
                    "size": config_path.stat().st_size,
                    "sha256": sha256_file(config_path),
                    "parsed_contract": parsed_contract,
                    "prior_mean_contract_sha256": PRIOR.contract_sha256,
                },
                "degenerate_property_stats": False,
            }],
        }
    atomic_write_json(staging / stg.FEREBUS_TASK_MANIFEST, manifest)
    model.with_suffix(".perf").write_text(
        "RMSE 0.0\nMAE 0.0\ncovariance_condition_number 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_structured_task_map(
        staging,
        executable="ferebus",
        execution_kind="synthetic_dry_run",
    )
    write_preexisting_model_receipts(
        staging,
        execution_kind="synthetic_dry_run",
    )
    return staging


def test_task_model_admission_is_inline_and_stat_bound(tmp_path):
    staging = _seed_quality_staging(tmp_path, campaign_layout=True)
    enrich_task_receipt_with_quality(staging, 0)

    evidence = enrich_task_receipt_with_model_admission(staging, 0)
    context = build_ferebus_model_admission_context(staging)

    assert evidence["semantic"]["training_data_binding_complete"] is True
    assert context.sources == ("task",)
    assert context.statistics == {"task": 1, "cache": 0, "local": 0, "total": 1}

    dataset = staging / "iqa/O1/datasets/WATER_O1_EXT_VALIDATION_SET.csv"
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FerebusModelAdmissionError, match="changed after task admission"):
        validate_ferebus_task_model_admission(staging, 0, evidence)


def test_legacy_model_admission_falls_back_once_then_uses_cache(
    tmp_path,
    monkeypatch,
):
    import ichor.hpc.active_learning.daemon.ferebus_model_admission as admission

    staging = _seed_quality_staging(tmp_path, campaign_layout=True)
    enrich_task_receipt_with_quality(staging, 0)
    calls = []

    def inline_worker(observed_staging, logical_task_id):
        calls.append(int(logical_task_id))
        admission._cache_worker(observed_staging, logical_task_id)

    monkeypatch.setattr(admission, "_run_cache_subprocess", inline_worker)

    first = build_ferebus_model_admission_context(staging)
    second = build_ferebus_model_admission_context(staging)

    assert first.sources == ("local",)
    assert second.sources == ("cache",)
    assert calls == [0]


def test_ferebus_quality_computes_metrics_and_condition_number(tmp_path):
    staging = _seed_quality_staging(tmp_path)

    payload = evaluate_ferebus_quality(staging)

    assert payload["measurement_complete"] is True
    assert payload["summary"]["n_tasks"] == 1
    assert payload["summary"]["mean_ext_rmse"] == 0.0
    assert payload["summary"]["min_ext_r2"] == 1.0
    assert payload["summary"]["max_condition_number"] > 0.0
    record = payload["records"][0]
    assert record["row_counts"] == {"train": 3, "int_val": 2, "ext_val": 2}
    assert record["metrics"]["ext_val"] == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}


def test_task_factor_cache_publishes_and_validates_without_receipt_change(tmp_path):
    from ichor.core.models import Model

    staging = _seed_quality_staging(tmp_path, campaign_layout=True)
    receipt = staging / "iqa" / "O1" / "FEREBUS_TASK_RECEIPT.json"
    before = receipt.read_bytes()
    model = Model(staging / "iqa" / "O1" / "WATER_iqa_O1.model")

    manifest = publish_task_factor(staging, 0, model=model)
    factor = read_task_factor(staging, 0, model=model)

    assert manifest.is_file()
    assert factor.shape == (model.ntrain, model.ntrain)
    assert receipt.read_bytes() == before


def test_task_factor_cache_rebuilds_corrupt_optional_evidence(tmp_path):
    from ichor.core.models import Model

    staging = _seed_quality_staging(tmp_path, campaign_layout=True)
    model = Model(staging / "iqa" / "O1" / "WATER_iqa_O1.model")
    manifest = publish_task_factor(staging, 0, model=model)
    factor_path = manifest.parent / "factor.npy"
    with factor_path.open("r+b") as handle:
        handle.seek(-8, 2)
        handle.write(b"\xff" * 8)

    with pytest.raises(FerebusModelFactorError):
        read_task_factor(staging, 0, model=model)
    publish_task_factor(staging, 0, model=model)
    assert read_task_factor(staging, 0, model=model).shape == (
        model.ntrain,
        model.ntrain,
    )


def test_task_quality_enrichment_is_reused_without_local_prediction(
    tmp_path,
    monkeypatch,
):
    staging = _seed_quality_staging(tmp_path)
    measurement = enrich_task_receipt_with_quality(staging, 0)
    validated = validate_ferebus_task_measurement(staging, 0, measurement)
    stats = {}

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("inline task measurement must avoid local prediction")

    monkeypatch.setattr(
        ferebus_quality_module,
        "_run_measurement_subprocess",
        forbidden_subprocess,
    )
    quality = evaluate_ferebus_quality(staging, measurement_stats=stats)

    assert quality["measurement_complete"] is True
    assert quality["records"][0]["metrics"] == validated["metrics"]
    assert stats == {
        "inline": 1,
        "cached": 0,
        "local": 0,
        "incumbent_reused": 0,
        "incumbent_local": 0,
    }


def test_invalid_optional_task_quality_falls_back_and_warms_cache(
    tmp_path,
    monkeypatch,
):
    staging = _seed_quality_staging(tmp_path)
    receipt_path = staging / "iqa" / "O1" / "FEREBUS_TASK_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["quality_measurement"] = {"kind": "tampered"}
    atomic_write_json(receipt_path, receipt)

    monkeypatch.setattr(
        ferebus_quality_module,
        "_run_measurement_subprocess",
        lambda root, task_id: ferebus_quality_module._measure_task_cache_worker(
            root, task_id
        ),
    )
    first_stats = {}
    first = evaluate_ferebus_quality(staging, measurement_stats=first_stats)
    assert first["measurement_complete"] is True
    assert first_stats["local"] == 1

    monkeypatch.setattr(
        ferebus_quality_module,
        "_run_measurement_subprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("warm cache must avoid recomputation")
        ),
    )
    second_stats = {}
    second = evaluate_ferebus_quality(staging, measurement_stats=second_stats)
    assert second["records"] == first["records"]
    assert second_stats["cached"] == 1


def test_ferebus_quality_streams_large_csv_in_bounded_chunks(tmp_path):
    path = tmp_path / "large.csv"
    rows = ["f1,f2,f3,iqa"]
    for index in range(1201):
        target = float(index) / 100.0
        rows.append(
            ",".join(
                (
                    repr(target - 1.0),
                    "0.0",
                    "0.0",
                    repr(target),
                )
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")

    class RecordingModel:
        x = np.zeros((1, 3), dtype=float)
        y = np.zeros((1, 1), dtype=float)

        def __init__(self):
            self.largest_batch = 0

        def predict(self, features):
            self.largest_batch = max(self.largest_batch, int(features.shape[0]))
            return np.asarray(features[:, 0], dtype=float)

    model = RecordingModel()
    metrics, incumbent, count = _stream_model_metrics(path, "iqa", model)

    assert count == 1201
    assert model.largest_batch <= 512
    assert incumbent is None
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(1.0)


def test_ferebus_quality_optional_thresholds_are_enforced(tmp_path):
    staging = _seed_quality_staging(tmp_path, ext_targets=(1.0, 1.0))

    payload = evaluate_ferebus_quality(staging)
    decision = evaluate_ferebus_quality_decision(
        payload,
        _gates(ferebus_max_ext_rmse_ha=0.5),
    )

    assert decision["accepted"] is False
    assert "ferebus_ext_rmse_threshold_exceeded" in decision["reasons"]


def test_ferebus_promotion_warns_for_aggregate_regression_beyond_five_percent():
    quality = {
        "measurement_complete": True,
        "records": [
            {
                "property": "iqa",
                "atom": "O1",
                "row_counts": {"ext_val": 10},
                "condition_number": 1.0,
                "metrics": {"ext_val": {"rmse": 1.06, "mae": 1.0, "r2": 0.0}},
                "incumbent_ext_metrics": {"rmse": 1.0, "mae": 1.0, "r2": 0.0},
                "incumbent_binding": {"bound": True},
            }
        ],
        "summary": {
            "aggregate_iqa_ext_rmse": 1.06,
            "incumbent_aggregate_iqa_ext_rmse": 1.0,
        },
    }

    decision = evaluate_ferebus_quality_decision(quality, _gates())

    assert decision["accepted"] is True
    assert decision["reasons"] == []
    assert "ferebus_aggregate_ext_rmse_regressed" in decision["warnings"]
    assert decision["decision_policy"] == FEREBUS_QUALITY_DECISION_POLICY
    assert decision["n_warnings"] == 1
    assert decision["promotion"]["aggregate_rmse_limit"] == pytest.approx(1.05)


def test_ferebus_promotion_warns_for_single_task_regression_beyond_twenty_percent():
    quality = {
        "measurement_complete": True,
        "records": [
            {
                "property": "iqa",
                "atom": "O1",
                "row_counts": {"ext_val": 10},
                "condition_number": 1.0,
                "metrics": {"ext_val": {"rmse": 0.121, "mae": 0.1, "r2": 0.0}},
                "incumbent_ext_metrics": {"rmse": 0.1, "mae": 0.1, "r2": 0.0},
                "incumbent_binding": {"bound": True},
            }
        ],
        "summary": {
            "aggregate_iqa_ext_rmse": 0.121,
            "incumbent_aggregate_iqa_ext_rmse": 0.1,
        },
    }
    gates = _gates(ferebus_max_aggregate_ext_rmse_increase_fraction=1.0)

    decision = evaluate_ferebus_quality_decision(quality, gates)

    assert decision["accepted"] is True
    assert decision["reasons"] == []
    assert "ferebus_task_ext_rmse_regressed" in decision["warnings"]
    assert decision["tasks"][0]["accepted"] is True
    assert decision["tasks"][0]["reasons"] == []
    assert decision["tasks"][0]["warnings"] == [
        "ferebus_task_ext_rmse_regressed"
    ]
    assert decision["n_warned"] == 1
    assert decision["tasks"][0]["relative_rmse_limit"] == pytest.approx(0.12)


def test_ferebus_absolute_failure_remains_hard_alongside_relative_warning():
    quality = {
        "measurement_complete": True,
        "records": [
            {
                "property": "iqa",
                "atom": "O1",
                "row_counts": {"ext_val": 10},
                "condition_number": 1.0,
                "metrics": {
                    "ext_val": {"rmse": 1.06, "mae": 1.0, "r2": 0.0}
                },
                "incumbent_ext_metrics": {
                    "rmse": 1.0,
                    "mae": 1.0,
                    "r2": 0.0,
                },
                "incumbent_binding": {"bound": True},
            }
        ],
        "summary": {
            "aggregate_iqa_ext_rmse": 1.06,
            "incumbent_aggregate_iqa_ext_rmse": 1.0,
        },
    }

    decision = evaluate_ferebus_quality_decision(
        quality,
        _gates(ferebus_max_ext_rmse_ha=1.01),
    )

    assert decision["accepted"] is False
    assert "ferebus_ext_rmse_threshold_exceeded" in decision["reasons"]
    assert "ferebus_aggregate_ext_rmse_regressed" in decision["warnings"]


def test_ferebus_promotion_rejects_non_numeric_thresholds():
    with pytest.raises(ValueError, match="must be numeric"):
        evaluate_ferebus_quality_decision(
            {"measurement_complete": True, "records": [], "summary": {}},
            _gates(ferebus_max_task_ext_rmse_increase_fraction=True),
        )


def test_hartree_rmse_threshold_is_not_applied_to_multipoles(tmp_path):
    staging = _seed_quality_staging(tmp_path, ext_targets=(1.0, 1.0))
    quality = evaluate_ferebus_quality(staging)
    quality["records"][0]["property"] = "q00"

    decision = evaluate_ferebus_quality_decision(
        quality,
        _gates(ferebus_max_ext_rmse_ha=0.01),
    )

    assert decision["accepted"] is True


def test_ferebus_manifest_rejects_dataset_hash_drift(tmp_path):
    staging = _seed_quality_staging(tmp_path)
    dataset = staging / "iqa" / "O1" / "datasets" / "WATER_O1_TRAINING_SET.csv"
    dataset.write_text(
        dataset.read_text(encoding="utf-8") + "0.9,0.9,0.9,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset (size|SHA-256) mismatch"):
        stg.read_ferebus_manifest(staging)


def test_write_ferebus_quality_manifest(tmp_path):
    payload = {
        "schema_version": 1,
        "accepted": True,
        "measurement_complete": True,
        "summary": {"n_tasks": 0},
    }

    path = write_ferebus_quality_manifest(tmp_path, payload)

    assert path.name == FEREBUS_QUALITY_MANIFEST
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_legacy_ferebus_performance_aliases_are_canonicalised(tmp_path):
    path = tmp_path / "legacy.perf"
    path.write_text(
        "RMSE 1.0\n"
        "MAE 0.5\n"
        "weights_l2_nor 2.0\n"
        "covariance_con 3.0\n",
        encoding="utf-8",
        newline="\n",
    )

    parsed = _parse_perf(path)

    assert parsed["weights_l2_norm"] == 2.0
    assert parsed["covariance_condition_number"] == 3.0
    assert "weights_l2_nor" not in parsed
    assert "covariance_con" not in parsed


def test_legacy_and_canonical_performance_names_are_ambiguous(tmp_path):
    path = tmp_path / "ambiguous.perf"
    path.write_text(
        "covariance_con 3.0\n"
        "covariance_condition_number 3.0\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="duplicate or ambiguous"):
        _parse_perf(path)


def test_incomplete_quality_cannot_become_a_decision_or_manifest(tmp_path):
    quality = {
        "measurement_complete": False,
        "measurement_errors": ["ferebus_quality_metric_failed:ValueError:test"],
        "records": [],
        "summary": {},
    }

    with pytest.raises(FerebusQualityMeasurementIncomplete):
        write_ferebus_quality_manifest(tmp_path, quality)
    with pytest.raises(FerebusQualityMeasurementIncomplete):
        evaluate_ferebus_quality_decision(quality, _gates())


def test_quality_validator_recomputes_summary_from_task_records(tmp_path):
    staging = _seed_quality_staging(tmp_path)
    quality = evaluate_ferebus_quality(staging)
    quality["summary"]["n_measured"] = 0
    write_ferebus_quality_manifest(staging, quality)

    with pytest.raises(ValueError, match="summary n_measured mismatch"):
        validate_ferebus_quality_evidence(staging)


def test_quality_validator_binds_condition_to_native_performance(tmp_path):
    staging = _seed_quality_staging(tmp_path)
    quality = evaluate_ferebus_quality(staging)
    quality["records"][0]["condition_number"] = 2.0
    quality["summary"]["max_condition_number"] = 2.0
    write_ferebus_quality_manifest(staging, quality)

    with pytest.raises(ValueError, match="condition number disagrees"):
        validate_ferebus_quality_evidence(staging)


def test_quality_decision_reuses_immutable_metrics_for_threshold_change(tmp_path):
    staging = _seed_quality_staging(tmp_path, ext_targets=(1.0, 1.0))
    strict = _gates(ferebus_max_ext_rmse_ha=0.5)
    quality = evaluate_ferebus_quality(staging, gates=strict)
    write_ferebus_quality_manifest(staging, quality)
    model = staging / "iqa" / "O1" / "WATER_iqa_O1.model"
    model_bytes = model.read_bytes()

    decision_path = write_ferebus_quality_decision(
        staging,
        config_sha256="strict-config",
        gates=strict,
    )
    rejected = read_ferebus_quality_decision(
        staging,
        expected_config_sha256="strict-config",
        require_accepted=False,
    )
    assert rejected["current_evaluation"]["accepted"] is False

    write_ferebus_quality_decision(
        staging,
        config_sha256="relaxed-config",
        gates=_gates(),
    )
    accepted = read_ferebus_quality_decision(
        staging,
        expected_config_sha256="relaxed-config",
        require_accepted=True,
    )
    assert accepted["current_evaluation"]["accepted"] is True
    assert len(accepted["evaluations"]) == 2
    assert model.read_bytes() == model_bytes
    assert decision_path.name == FEREBUS_QUALITY_DECISION_MANIFEST


def test_quality_decision_rejects_model_drift(tmp_path):
    staging = _seed_quality_staging(tmp_path)
    write_ferebus_quality_manifest(staging, evaluate_ferebus_quality(staging))
    write_ferebus_quality_decision(
        staging,
        config_sha256="config-a",
        gates=_gates(),
    )
    model = staging / "iqa" / "O1" / "WATER_iqa_O1.model"
    model.write_bytes(model.read_bytes() + b"\n# tampered\n")

    with pytest.raises(ValueError, match="model (size|SHA-256|hash) mismatch"):
        read_ferebus_quality_decision(
            staging,
            expected_config_sha256="config-a",
        )
