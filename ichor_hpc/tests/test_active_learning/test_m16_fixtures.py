"""M16 Day 1 smoke tests: fixture pack loads cleanly via PointDirectory,
the parser helper walks the staging tree, the validators return the
expected ok/reject decisions on each fixture subdirectory.
"""
import shutil
from pathlib import Path

import pytest

from ichor.core.files.point_directory import PointDirectory
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.live_executor import (
    LiveBackendsPhaseExecutor,
    validate_aimall_completed,
    validate_ferebus_completed,
    validate_gaussian_completed,
)
from ichor.hpc.active_learning.daemon import input_staging as stg


FIXTURES = (
    Path(__file__).resolve().parent / "fixtures" / "live_outputs"
)


# --- Fixture-pack presence ---------------------------------------------


def test_fixture_root_exists():
    assert FIXTURES.is_dir(), (
        "M16 fixture pack missing at " + str(FIXTURES)
        + " -- rebuild via the M16-P1 build script."
    )


def test_every_canonical_subdir_present():
    expected = {
        "initial_quantum",
        "iter_quantum",
        "iter_quantum_scf_failure",
        "ferebus_staging",
        "ferebus_staging_empty",
        "ariadne_pool",
        "polus_phase_a",
        "polus_phase_b",
    }
    actual = {p.name for p in FIXTURES.iterdir() if p.is_dir()}
    assert expected.issubset(actual), "missing: " + repr(expected - actual)


def test_initial_quantum_has_four_pointdirs():
    pdirs = sorted(
        p for p in (FIXTURES / "initial_quantum").iterdir()
        if PointDirectory.check_path(p)
    )
    assert len(pdirs) == 4


def test_iter_quantum_has_four_pointdirs():
    pdirs = sorted(
        p for p in (FIXTURES / "iter_quantum").iterdir()
        if PointDirectory.check_path(p)
    )
    assert len(pdirs) == 4


# --- PointDirectory discovers the real fixture content -----------------


def test_point_directory_discovers_gaussian_wfn_ints():
    """Every pointdir in initial_quantum/ must expose the four file types
    the parser depends on: gjf, gaussian_output, wfn, ints."""
    pdirs = sorted(
        p for p in (FIXTURES / "initial_quantum").iterdir()
        if PointDirectory.check_path(p)
    )
    assert pdirs, "no pointdirs found"
    for pdir_path in pdirs:
        pdir = PointDirectory(pdir_path)
        assert pdir.gjf is not None, str(pdir_path)
        assert pdir.gaussian_output is not None, str(pdir_path)
        assert pdir.wfn is not None, str(pdir_path)
        assert pdir.ints is not None, str(pdir_path)
        assert pdir.wfn.total_energy < 0.0, "WFN energy should be negative"


# --- Validators on happy-path fixtures ---------------------------------


def test_gaussian_validator_accepts_clean_fixture():
    pdir = PointDirectory(FIXTURES / "initial_quantum" / "POINT_0000.pointdir")
    ok, reason = validate_gaussian_completed(pdir)
    assert ok, "expected clean fixture to pass; got reason=" + reason


def test_aimall_validator_accepts_clean_fixture():
    pdir = PointDirectory(FIXTURES / "initial_quantum" / "POINT_0000.pointdir")
    ok, reason = validate_aimall_completed(pdir)
    assert ok, "expected clean fixture to pass; got reason=" + reason


def test_ferebus_validator_accepts_clean_fixture(tmp_path):
    root = tmp_path / "ferebus_staging"
    model_dir = root / "iqa" / "O1"
    model_dir.mkdir(parents=True)
    (model_dir / "ferebus.config").write_text("config\n", encoding="utf-8")
    model = model_dir / "WATER_iqa_O1.model"
    rows = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    model.write_text(
        "\n".join([
            "# jitter 1.0e-6",
            "# likelihood -1.0",
            "",
            "[system]",
            "name WATER",
            "atom O1",
            "property iqa",
            "ALF 1 2 3",
            "",
            "[dimensions]",
            "number_of_atoms 3",
            "number_of_training_points 2",
            "number_of_features 3",
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
            "number_of_dimensions 3",
            "active_dimensions 1 2 3",
            "thetas 1.0 1.0 1.0",
            "",
            "[training_data]",
            "units.x bohr bohr radians",
            "units.y Ha",
            "",
            "[training_data.x]",
            " ".join(str(v) for v in rows[0]),
            " ".join(str(v) for v in rows[1]),
            "",
            "[training_data.y]",
            "-75.0",
            "-75.1",
            "",
            "[weights]",
            "0.0",
            "0.0",
        ]) + "\n",
        encoding="utf-8",
    )
    (root / stg.FEREBUS_TASK_MANIFEST).write_text(
        __import__("json").dumps({
            "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
            "system": "WATER",
            "tasks": [{
                "property": "iqa",
                "atom": "O1",
                "alf_1_indexed": [1, 2, 3],
                "config_path": str(model_dir / "ferebus.config"),
                "expected_model_path": str(model),
                "row_counts": {"train": 2},
            }],
        }),
        encoding="utf-8",
    )
    ok, reason = validate_ferebus_completed(root)
    assert ok, "expected ferebus_staging to pass; got reason=" + reason


# --- Validators on failure-path fixtures -------------------------------


def test_gaussian_validator_rejects_scf_failure_fixture():
    pdir = PointDirectory(
        FIXTURES / "iter_quantum_scf_failure" / "POINT_0000.pointdir"
    )
    ok, reason = validate_gaussian_completed(pdir)
    assert not ok
    assert reason == "scf_nonconvergence_or_crash"


def test_gaussian_validator_rejects_wfn_gjf_geometry_mismatch(tmp_path):
    src = FIXTURES / "initial_quantum" / "POINT_0000.pointdir"
    dst = tmp_path / "POINT_0000.pointdir"
    shutil.copytree(src, dst)
    gjf = next(dst.glob("*.gjf"))
    text = gjf.read_text(encoding="utf-8")
    text = text.replace(
        "O  -0.03348733  -0.46689766  -0.00424905",
        "O   9.00000000  -0.46689766  -0.00424905",
    )
    gjf.write_text(text, encoding="utf-8", newline="\n")

    ok, reason = validate_gaussian_completed(PointDirectory(dst))

    assert not ok
    assert reason == "wfn_gjf_geometry_mismatch"


def test_ferebus_validator_rejects_empty_staging():
    ok, reason = validate_ferebus_completed(FIXTURES / "ferebus_staging_empty")
    assert not ok
    assert reason.startswith("ferebus_manifest_invalid")


def test_ferebus_validator_rejects_missing_staging(tmp_path):
    ok, reason = validate_ferebus_completed(tmp_path / "does_not_exist")
    assert not ok
    assert reason == "ferebus_staging_missing"


# --- _parse_staged_pointdirs helper -------------------------------------


def _make_executor(tmp_path):
    return LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=CampaignConfig(),
        backend_check=False,
    )


def test_parse_staged_pointdirs_all_pass(tmp_path):
    """initial_quantum has 4 clean pointdirs; both Gaussian and AIMAll
    validators should accept them all."""
    ex = _make_executor(tmp_path)
    kept, rejected = ex._parse_staged_pointdirs(
        FIXTURES / "initial_quantum",
        validators=(validate_gaussian_completed, validate_aimall_completed),
    )
    assert len(kept) == 4
    assert len(rejected) == 0


def test_parse_staged_pointdirs_partial_failure(tmp_path):
    """iter_quantum_scf_failure has 2 pointdirs; POINT_0000 is the corrupted
    one. The helper should return 1 kept + 1 rejected."""
    ex = _make_executor(tmp_path)
    kept, rejected = ex._parse_staged_pointdirs(
        FIXTURES / "iter_quantum_scf_failure",
        validators=(validate_gaussian_completed,),
    )
    assert len(kept) == 1
    assert len(rejected) == 1
    rejected_name, reason = rejected[0]
    assert rejected_name == "POINT_0000.pointdir"
    assert reason == "scf_nonconvergence_or_crash"


def test_parse_staged_pointdirs_missing_root_returns_empty(tmp_path):
    """If the staging root does not exist, helper returns empty lists rather
    than raising."""
    ex = _make_executor(tmp_path)
    kept, rejected = ex._parse_staged_pointdirs(
        tmp_path / "no_such_dir",
        validators=(validate_gaussian_completed,),
    )
    assert kept == []
    assert rejected == []


def test_parse_staged_pointdirs_ignores_non_pointdir_children(tmp_path):
    """Non-pointdir children of the staging root must be silently
    ignored (the sbatch jobs may drop sentinel files there)."""
    ex = _make_executor(tmp_path)
    staging = tmp_path / "noisy_staging"
    staging.mkdir()
    real = staging / "POINT_0000.pointdir"
    real.mkdir()
    (staging / "POINTS.txt").write_text("some content", encoding="utf-8")
    (staging / "junk_dir").mkdir()
    kept, rejected = ex._parse_staged_pointdirs(
        staging,
        validators=(),
    )
    assert len(kept) == 1
    assert len(rejected) == 0


# --- ARIADNE per-seed fixture loads as JSON ----------------------------


def test_ariadne_pool_result_json_loadable():
    import json
    seeds = sorted((FIXTURES / "ariadne_pool").iterdir())
    assert len(seeds) == 3
    for seed_dir in seeds:
        result_path = seed_dir / "result.json"
        assert result_path.is_file(), str(result_path)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for key in ("alpha_trajectory", "alpha_initial", "alpha_final",
                    "return_code", "fell_back_to_ds", "mock"):
            assert key in payload, "missing key in result.json: " + key


# --- POLUS fixtures -----------------------------------------------------


def test_polus_phase_a_sample_xyz_loadable():
    from ichor.core.files.xyz import Trajectory
    sample_path = FIXTURES / "polus_phase_a" / "initial-SAMPLE-2.xyz"
    assert sample_path.is_file()
    traj = Trajectory(sample_path)
    traj.read()
    frames = list(traj)
    assert len(frames) == 2


def test_polus_phase_b_sample_xyz_loadable():
    from ichor.core.files.xyz import Trajectory
    sample_path = FIXTURES / "polus_phase_b" / "phase_b_SAMPLE.xyz"
    assert sample_path.is_file()
    traj = Trajectory(sample_path)
    traj.read()
    frames = list(traj)
    assert len(frames) == 2
