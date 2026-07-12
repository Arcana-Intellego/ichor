"""FEREBUS quality sidecar tests."""
import json
import hashlib
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.daemon.ferebus_quality import (
    FEREBUS_QUALITY_DECISION_MANIFEST,
    FEREBUS_QUALITY_MANIFEST,
    evaluate_ferebus_quality,
    evaluate_ferebus_quality_decision,
    read_ferebus_quality_decision,
    write_ferebus_quality_decision,
    write_ferebus_quality_manifest,
)
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.ferebus_prior import (
    resolve_ferebus_prior_contract,
)


PRIOR = resolve_ferebus_prior_contract(CampaignConfig())
OXYGEN_PRIOR = PRIOR.expected_mean_ha("iqa", "O1")


def _write_zero_model(path, *, atom="O1", ntrain=3, nfeats=3):
    rows = [
        [0.1 + i * 0.1 + j * 0.01 for j in range(nfeats)]
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
    lines += ["0.0" for _ in range(ntrain)]
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


def _seed_quality_staging(tmp_path, *, ext_targets=None):
    if ext_targets is None:
        ext_targets = (OXYGEN_PRIOR, OXYGEN_PRIOR)
    staging = tmp_path / "iteration-staging"
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
    (task_dir / "ferebus.config").write_text("config\n", encoding="utf-8")
    (staging / stg.FEREBUS_TASK_MANIFEST).write_text(
        json.dumps({
            "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
            "campaign_uid": "quality-test",
            "system": "WATER",
            "reference_data_version": 4,
            "reference_data_head_manifest_sha256": "a" * 64,
            "reference_data_view_sha256": "b" * 64,
            "n_reference_points": 7,
            "pointdir_row_order": [
                "POINT_" + str(index).zfill(6) + ".pointdir"
                for index in range(7)
            ],
            "properties": ["iqa"],
            "atoms": ["O1"],
            "n_atoms": 1,
            "n_tasks": 1,
            "prior_mean_contract": PRIOR.to_dict(),
            "tasks": [{
                "task_index": 1,
                "property": "iqa",
                "atom": "O1",
                "prior_mean": PRIOR.task_payload("iqa", "O1"),
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
                "row_ids": {
                    "train": [0, 1, 2],
                    "int_val": [3, 4],
                    "ext_val": [5, 6],
                },
                "datasets": {
                    "train": _dataset_identity(
                        train_csv, 3, staging
                    ),
                    "int_val": _dataset_identity(
                        int_csv, 2, staging
                    ),
                    "ext_val": _dataset_identity(
                        ext_csv, 2, staging
                    ),
                },
            }],
        }),
        encoding="utf-8",
    )
    return staging


def test_ferebus_quality_computes_metrics_and_condition_number(tmp_path):
    staging = _seed_quality_staging(tmp_path)

    payload = evaluate_ferebus_quality(staging)

    assert payload["accepted"] is True
    assert payload["summary"]["n_tasks"] == 1
    assert payload["summary"]["mean_ext_rmse"] == 0.0
    assert payload["summary"]["min_ext_r2"] == 1.0
    assert payload["summary"]["max_condition_number"] > 0.0
    record = payload["records"][0]
    assert record["row_counts"] == {"train": 3, "int_val": 2, "ext_val": 2}
    assert record["metrics"]["ext_val"] == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}


def test_ferebus_quality_optional_thresholds_are_enforced(tmp_path):
    staging = _seed_quality_staging(tmp_path, ext_targets=(1.0, 1.0))

    payload = evaluate_ferebus_quality(
        staging,
        gates=SimpleNamespace(ferebus_max_ext_rmse_ha=0.5),
    )

    assert payload["accepted"] is False
    assert "ferebus_ext_rmse_threshold_exceeded" in payload["reasons"]


def test_hartree_rmse_threshold_is_not_applied_to_multipoles(tmp_path):
    staging = _seed_quality_staging(tmp_path, ext_targets=(1.0, 1.0))
    quality = evaluate_ferebus_quality(staging)
    quality["records"][0]["property"] = "q00"

    decision = evaluate_ferebus_quality_decision(
        quality,
        SimpleNamespace(ferebus_max_ext_rmse_ha=0.01),
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
    payload = {"schema_version": 1, "accepted": True, "summary": {"n_tasks": 0}}

    path = write_ferebus_quality_manifest(tmp_path, payload)

    assert path.name == FEREBUS_QUALITY_MANIFEST
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_quality_decision_reuses_immutable_metrics_for_threshold_change(tmp_path):
    staging = _seed_quality_staging(tmp_path, ext_targets=(1.0, 1.0))
    strict = SimpleNamespace(ferebus_max_ext_rmse_ha=0.5)
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
        gates=SimpleNamespace(),
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
        gates=SimpleNamespace(),
    )
    model = staging / "iqa" / "O1" / "WATER_iqa_O1.model"
    model.write_bytes(model.read_bytes() + b"\n# tampered\n")

    with pytest.raises(ValueError, match="model hash mismatch"):
        read_ferebus_quality_decision(
            staging,
            expected_config_sha256="config-a",
        )
