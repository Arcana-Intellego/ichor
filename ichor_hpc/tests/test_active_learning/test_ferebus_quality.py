"""FEREBUS quality sidecar tests."""
import json
from types import SimpleNamespace

from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.daemon.ferebus_quality import (
    FEREBUS_QUALITY_MANIFEST,
    evaluate_ferebus_quality,
    write_ferebus_quality_manifest,
)


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
        "type zero",
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


def _seed_quality_staging(tmp_path, *, ext_targets=(0.0, 0.0)):
    staging = tmp_path / "iteration-staging"
    task_dir = staging / "iqa" / "O1"
    task_dir.mkdir(parents=True)
    model = task_dir / "WATER_iqa_O1.model"
    _write_zero_model(model)
    train_csv = task_dir / "WATER_O1_TRAINING_SET.csv"
    int_csv = task_dir / "WATER_O1_INT_VALIDATION_SET.csv"
    ext_csv = task_dir / "WATER_O1_EXT_VALIDATION_SET.csv"
    _write_dataset(train_csv, (0.0, 0.0, 0.0))
    _write_dataset(int_csv, (0.0, 0.0))
    _write_dataset(ext_csv, ext_targets)
    (task_dir / "ferebus.config").write_text("config\n", encoding="utf-8")
    (staging / stg.FEREBUS_TASK_MANIFEST).write_text(
        json.dumps({
            "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
            "system": "WATER",
            "reference_data_version": 4,
            "reference_data_head_manifest_sha256": "a" * 64,
            "reference_data_view_sha256": "b" * 64,
            "n_reference_points": 7,
            "pointdir_row_order": [
                "POINT_" + str(index).zfill(6) + ".pointdir"
                for index in range(7)
            ],
            "tasks": [{
                "task_index": 1,
                "property": "iqa",
                "atom": "O1",
                "alf_1_indexed": [1, 2, 3],
                "config_path": str(task_dir / "ferebus.config"),
                "expected_model_path": str(model),
                "training_csv": str(train_csv),
                "int_validation_csv": str(int_csv),
                "ext_validation_csv": str(ext_csv),
                "row_counts": {"train": 3, "int_val": 2, "ext_val": 2},
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


def test_write_ferebus_quality_manifest(tmp_path):
    payload = {"schema_version": 1, "accepted": True, "summary": {"n_tasks": 0}}

    path = write_ferebus_quality_manifest(tmp_path, payload)

    assert path.name == FEREBUS_QUALITY_MANIFEST
    assert json.loads(path.read_text(encoding="utf-8")) == payload
