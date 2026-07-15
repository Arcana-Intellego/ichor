"""The hardened FEREBUS + AIMAll output validators (A36, A37).

These are the daemon-side defence for the model-corruption cluster: A52 (a partial model set
loaded as if complete) and A56 (a truncated .model read into uninitialised np.empty) both live in
ichor_core/models which we are not allowed to touch, so the validator has to catch them here.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from ichor.hpc.active_learning.daemon.live_executor import (
    validate_aimall_completed,
    validate_ferebus_completed,
)
from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.ferebus_prior import (
    backend_kernel_token,
    resolve_ferebus_prior_contract,
    validate_ferebus_config_contract,
)
from ichor.hpc.active_learning.versioning.reference_data import (
    canonical_json_sha256,
)


PRIOR = resolve_ferebus_prior_contract(CampaignConfig())


def _write_model(path, ntrain, nrows, nfeats=3):
    """a minimal FEREBUS-style .model: a number_of_training_points header and an [training_data.x]
    block of nrows feature rows. nrows < ntrain is the truncated case (A56)."""
    lines = [
        "number_of_training_points " + str(ntrain),
        "number_of_features " + str(nfeats),
        "[training_data.x]",
    ]
    lines += [" ".join(["0.1"] * nfeats) for _ in range(nrows)]
    lines.append("")  # blank line terminates the x block
    lines.append("[training_data.y]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_loadable_model(
    path,
    *,
    atom="O1",
    prop="iqa",
    system="WATER",
    alf=(1, 2, 3),
    ntrain=5,
    nfeats=3,
):
    prior_mean = PRIOR.expected_mean_ha(prop, atom)
    rows = [
        [0.1 + i * 0.1 + j * 0.01 for j in range(nfeats)]
        for i in range(ntrain)
    ]
    lines = [
        "# jitter 1.0e-6",
        "# likelihood -1.0",
        "",
        "[system]",
        "name " + system,
        "atom " + atom,
        "property " + prop,
        "ALF " + " ".join(str(int(x)) for x in alf),
        "",
        "[dimensions]",
        "number_of_atoms 3",
        "number_of_features " + str(nfeats),
        "number_of_training_points " + str(ntrain),
        "",
        "[mean]",
        "type constant",
        "value " + repr(prior_mean),
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
    lines += [
        "",
        "[training_data.y]",
    ]
    lines += [str(-75.0 - i * 0.01) for i in range(ntrain)]
    lines += [
        "",
        "[weights]",
    ]
    lines += ["0.0" for _ in range(ntrain)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_root(root, atom="O1", prop="iqa"):
    d = root / prop / atom
    d.mkdir(parents=True, exist_ok=True)
    (d / "datasets").mkdir(exist_ok=True)
    (d / "ferebus.config").write_text(
        "mean_type = 21\n"
        + 'level_of_theory = "' + PRIOR.level_of_theory + '"\n'
        + "iqaDeviationFactor = 1.0\nscaling = 1\n"
        + "scale_feats = 1\nscale_prop = 0\n",
        encoding="utf-8",
    )
    return d


def _write_manifest(root, atoms):
    row_order = [
        "POINT_" + str(index).zfill(6) + ".pointdir"
        for index in range(9)
    ]
    row_ids = {
        "train": [0, 1, 2, 3, 4],
        "int_val": [5, 6],
        "ext_val": [7, 8],
    }
    source_rows = [
        {"pointdir_name": pointdir_name}
        for pointdir_name in row_order
    ]
    split_rows = {
        split: [source_rows[index] for index in indexes]
        for split, indexes in row_ids.items()
    }
    row_identity_payload = {
        "schema_version": stg.FEREBUS_ROW_IDENTITIES_SCHEMA_VERSION,
        "campaign_uid": "validator-test",
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
            for split, rows in split_rows.items()
        },
    }
    row_identity_path = root / stg.FEREBUS_ROW_IDENTITIES
    row_identity_path.write_text(
        json.dumps(row_identity_payload),
        encoding="utf-8",
    )
    split_payload = {
        "schema_version": 7,
        "assignments": {
            row_order[index]: {"split": split}
            for split, indexes in row_ids.items()
            for index in indexes
        },
    }
    split_path = root / stg.FEREBUS_SPLIT_SNAPSHOT
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")
    tasks = []
    for i, atom in enumerate(atoms, start=1):
        d = root / "iqa" / atom
        task_dir = "iqa/" + atom
        input_dir = task_dir + "/datasets"
        dataset_records = {}
        for split, suffix, rows in (
            ("train", "TRAINING_SET", 5),
            ("int_val", "INT_VALIDATION_SET", 2),
            ("ext_val", "EXT_VALIDATION_SET", 2),
        ):
            path = d / "datasets" / ("WATER_" + atom + "_" + suffix + ".csv")
            csv_rows = ["f1,f2,f3,iqa"]
            for row_index in range(rows):
                features = [
                    0.1 + row_index * 0.1 + feature_index * 0.01
                    for feature_index in range(3)
                ]
                csv_rows.append(
                    ",".join(str(value) for value in features)
                    + ","
                    + str(-75.0 - row_index * 0.01)
                )
            path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
            dataset_records[split] = {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": rows,
                "row_identity_sha256": row_identity_payload["splits"][split][
                    "row_identity_sha256"
                ],
                "row_identity_count": rows,
            }
        tasks.append({
            "task_index": i,
            "property": "iqa",
            "atom": atom,
            "prior_mean": PRIOR.task_payload(
                "iqa",
                atom,
                training_values=[0.0] * 5,
                training_dataset_sha256=dataset_records["train"]["sha256"],
            ),
            "alf_1_indexed": [1, 2, 3],
            "alf_cli": "1_2_3",
            "property_dir": "iqa",
            "output_dir": task_dir,
            "input_dir": input_dir,
            "config_path": task_dir + "/ferebus.config",
            "expected_model_path": task_dir + f"/WATER_iqa_{atom}.model",
            "training_csv": input_dir + f"/WATER_{atom}_TRAINING_SET.csv",
            "int_validation_csv": input_dir + f"/WATER_{atom}_INT_VALIDATION_SET.csv",
            "ext_validation_csv": input_dir + f"/WATER_{atom}_EXT_VALIDATION_SET.csv",
            "command_args": [
                "-c", task_dir + "/ferebus.config",
                "-I", input_dir,
                "-O", task_dir,
                "-P", "iqa",
                "-A", atom,
                "-ALF", "1_2_3",
            ],
            "row_counts": {"train": 5, "int_val": 2, "ext_val": 2},
            "row_ids": dict(row_ids),
            "datasets": dataset_records,
            "generated_config": {
                "path": task_dir + "/ferebus.config",
                "size": (d / "ferebus.config").stat().st_size,
                "sha256": hashlib.sha256(
                    (d / "ferebus.config").read_bytes()
                ).hexdigest(),
                "parsed_contract": validate_ferebus_config_contract(
                    d / "ferebus.config", PRIOR
                ),
                "prior_mean_contract_sha256": PRIOR.contract_sha256,
            },
        })
    (root / stg.FEREBUS_TASK_MANIFEST).write_text(
        json.dumps({
            "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
            "campaign_uid": "validator-test",
            "system": "WATER",
            "reference_data_version": 0,
            "reference_data_head_manifest_sha256": "a" * 64,
            "reference_data_view_sha256": "b" * 64,
            "n_reference_points": 9,
            "pointdir_row_order": row_order,
            "properties": ["iqa"],
            "atoms": list(atoms),
            "n_atoms": len(atoms),
            "n_tasks": len(tasks),
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
                "sha256": hashlib.sha256(row_identity_path.read_bytes()).hexdigest(),
                "source_rows_sha256": row_identity_payload[
                    "source_rows_sha256"
                ],
            },
            "split_ledger": {
                "path": stg.FEREBUS_SPLIT_SNAPSHOT,
                "size": split_path.stat().st_size,
                "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            },
            "tasks": tasks,
        }),
        encoding="utf-8",
    )


def _write_execution_evidence(root):
    from ichor.hpc.active_learning.daemon.ferebus_task_runner import (
        write_preexisting_model_receipts,
    )
    from ichor.hpc.active_learning.submit.pyferebus_wrap import (
        _write_structured_task_map,
    )

    task_map_path = _write_structured_task_map(
        root,
        executable="ferebus",
        execution_kind="synthetic_dry_run",
    )
    task_map = json.loads(task_map_path.read_text(encoding="utf-8"))
    for task in task_map["tasks"]:
        performance_path = root.joinpath(
            *str(task["expected_performance_path"]).split("/")
        )
        performance_path.write_text(
            "RMSE 0.0\nMAE 0.0\ncovariance_condition_number 1.0\n",
            encoding="utf-8",
        )
    write_preexisting_model_receipts(
        root,
        execution_kind="synthetic_dry_run",
    )


# --- A37: FEREBUS validator ------------------------------------------------


def test_ferebus_rejects_truncated_model(tmp_path):
    # header says 10 training points but only 7 rows are written -- the core reader would fill the
    # missing 3 with np.empty garbage (A56). we must reject it.
    d = _task_root(tmp_path, "O1")
    _write_manifest(tmp_path, ["O1"])
    _write_model(d / "WATER_iqa_O1.model", ntrain=10, nrows=7)
    ok, reason = validate_ferebus_completed(tmp_path)
    assert not ok and reason == "model_truncated"


def test_ferebus_rejects_partial_atom_set(tmp_path):
    # ATOMS.txt asked for 3 atoms but the array only produced 2 models (A52 -- a partial set must
    # not commit as a complete models version).
    for atom in ("O1", "H2", "H3"):
        _task_root(tmp_path, atom)
    _write_manifest(tmp_path, ["O1", "H2", "H3"])
    _write_model(tmp_path / "iqa" / "O1" / "WATER_iqa_O1.model", ntrain=5, nrows=5)
    _write_model(tmp_path / "iqa" / "H2" / "WATER_iqa_H2.model", ntrain=5, nrows=5)
    ok, reason = validate_ferebus_completed(tmp_path)
    assert not ok and reason.startswith("expected_model_missing")


def test_ferebus_accepts_complete_untruncated_set(tmp_path):
    for a in ("O1", "H2", "H3"):
        _task_root(tmp_path, a)
        _write_loadable_model(
            tmp_path / "iqa" / a / f"WATER_iqa_{a}.model",
            atom=a,
            alf=(1, 2, 3),
        )
    _write_manifest(tmp_path, ["O1", "H2", "H3"])
    _write_execution_evidence(tmp_path)
    ok, reason = validate_ferebus_completed(tmp_path)
    assert ok, reason


def test_ferebus_rejects_structurally_incomplete_model(tmp_path):
    d = _task_root(tmp_path, "O1")
    _write_manifest(tmp_path, ["O1"])
    _write_model(d / "WATER_iqa_O1.model", ntrain=5, nrows=5)
    ok, reason = validate_ferebus_completed(tmp_path)
    assert not ok
    assert (
        "model_section_missing" in reason
        or "model_parse_failed" in reason
        or reason.startswith("model_truncated")
    )


def test_ferebus_rejects_wrong_model_property(tmp_path):
    d = _task_root(tmp_path, "O1")
    _write_manifest(tmp_path, ["O1"])
    _write_loadable_model(d / "WATER_iqa_O1.model", atom="O1", prop="q00")
    ok, reason = validate_ferebus_completed(tmp_path)
    assert not ok
    assert reason == "model_metadata_mismatch:property"


def test_ferebus_rejects_empty_staging(tmp_path):
    ok, reason = validate_ferebus_completed(tmp_path)
    assert not ok and reason.startswith("ferebus_manifest_invalid")


# --- A36: AIMAll validator -------------------------------------------------


class _Int:
    atom_name = "O1"
    net_charge = 0.0
    iqa = -75.0
    integration_error = 1.0e-6


def test_aimall_rejects_partial_int_set(tmp_path):
    # 9 atoms in the geometry but only 3 .int files parsed -> a partial AIMAll that the old
    # n_int>=1 check waved through, then blew up the FEREBUS feature export later.
    ints = SimpleNamespace(path=str(tmp_path), ints=[_Int(), _Int(), _Int()])
    pdir = SimpleNamespace(ints=ints, atoms=[0] * 9)
    ok, reason = validate_aimall_completed(pdir)
    assert not ok and reason == "aimall_partial_3_of_9_int"


def test_aimall_accepts_one_int_per_atom(tmp_path):
    from ichor.core.files.point_directory import PointDirectory

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "live_outputs"
        / "initial_quantum"
        / "POINT_0000.pointdir"
    )
    pdir = PointDirectory(fixture)
    ok, reason = validate_aimall_completed(pdir)
    assert ok, reason
