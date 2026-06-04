"""The hardened FEREBUS + AIMAll output validators (A36, A37).

These are the daemon-side defence for the model-corruption cluster: A52 (a partial model set
loaded as if complete) and A56 (a truncated .model read into uninitialised np.empty) both live in
ichor_core/models which we are not allowed to touch, so the validator has to catch them here.
"""
from types import SimpleNamespace

from ichor.hpc.active_learning.daemon.live_executor import (
    validate_aimall_completed,
    validate_ferebus_completed,
)
from ichor.hpc.active_learning.daemon import input_staging as stg


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
    (d / "ferebus.config").write_text("config\n", encoding="utf-8")
    return d


def _write_manifest(root, atoms):
    tasks = []
    for i, atom in enumerate(atoms, start=1):
        d = root / "iqa" / atom
        tasks.append({
            "task_index": i,
            "property": "iqa",
            "atom": atom,
            "alf_1_indexed": [1, 2, 3],
            "config_path": str(d / "ferebus.config"),
            "expected_model_path": str(d / f"WATER_iqa_{atom}.model"),
            "row_counts": {"train": 5, "int_val": 2, "ext_val": 2},
        })
    (root / stg.FEREBUS_TASK_MANIFEST).write_text(
        __import__("json").dumps({
            "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
            "system": "WATER",
            "tasks": tasks,
        }),
        encoding="utf-8",
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
    ints = SimpleNamespace(path=str(tmp_path), ints=[_Int(), _Int(), _Int()])
    pdir = SimpleNamespace(ints=ints, atoms=[0, 0, 0])
    ok, reason = validate_aimall_completed(pdir)
    assert ok, reason
