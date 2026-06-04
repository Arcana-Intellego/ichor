"""Opt-in pyferebus/FEREBUS smoke tests for CSF4 live-contract hardening.

These tests are marked live and skipped unless explicitly enabled. The first
smoke uses real pyferebus generation with fake sbatch, so it proves the Python
staging/generation contract without spending cluster time. The second smoke
submits a tiny real FEREBUS job and should only be run deliberately on CSF4.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.daemon.preflight import check_backends
from ichor.hpc.active_learning.submit.pyferebus_wrap import submit_ferebus


pytestmark = pytest.mark.live


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _write_csv(path: Path, offset: float) -> None:
    path.write_text(
        "f1,f2,f3,iqa\n"
        + "\n".join(
            f"{offset + i + 0.1},{offset + i + 0.2},{offset + i + 0.3},{-75.0 - offset - i * 0.01}"
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_smoke_staging(tmp_path: Path) -> Path:
    staging = tmp_path / "ferebus_smoke"
    prop = staging / "iqa"
    prop.mkdir(parents=True)
    for kind, offset in (
        ("TRAINING", 0.0),
        ("INT_VALIDATION", 10.0),
        ("EXT_VALIDATION", 20.0),
    ):
        _write_csv(prop / f"WATER_O1_{kind}_SET.csv", offset)
    (staging / "job-details").write_text(
        "system_name WATER\n"
        "natoms 1\n"
        "atoms O1\n"
        "props iqa\n"
        "O1 1 2 3\n"
        "iqa-O1 -75.01 -75.00 0.01 -75.005 -75.005 0.005 0.00006666\n",
        encoding="utf-8",
        newline="\n",
    )
    return staging


class _FakeSbatch:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="987654\n", stderr="")


def test_real_pyferebus_generation_smoke_with_fake_sbatch(tmp_path):
    if not _enabled("ICHOR_RUN_REAL_PYFEREBUS_SMOKE"):
        pytest.skip("set ICHOR_RUN_REAL_PYFEREBUS_SMOKE=1 to run real pyferebus generation")
    trainer = pytest.importorskip("pyferebus.executors.trainer")

    staging = _write_smoke_staging(tmp_path)
    fake_sbatch = _FakeSbatch()
    submission = submit_ferebus(
        staging / "job-details",
        staging,
        model_class=trainer.MODEL,
        submit_runner=fake_sbatch,
        walltime_hours=1,
        ncores=1,
        nagents=4,
        maxiter=1,
        overwrite_workdir=False,
        move_dataset_files=True,
    )

    assert submission.job_id == "987654"
    assert fake_sbatch.calls[0][0] == ["sbatch", "--parsable", str(staging / "runFerebus.sh")]
    assert (staging / "runFerebus.sh").is_file()
    assert (staging / "commands").is_file()
    assert (staging / "list.txt").is_file()
    assert (staging / "iqa" / "O1" / "datasets" / "WATER_O1_TRAINING_SET.csv").is_file()
    assert (staging / "iqa" / "O1" / "ferebus.config").is_file()
    assert (staging / "ferebus-stdout").is_dir()
    assert (staging / "ferebus-stderr").is_dir()

    command = (staging / "commands").read_text(encoding="utf-8").strip()
    for flag in ("-c", "-I", "-O", "-P", "-A", "-ALF"):
        assert flag in command.split()
    assert "-P iqa" in command
    assert "-A O1" in command
    assert "-ALF 1_2_3" in command
    assert (staging / "list.txt").read_text(encoding="utf-8").strip().endswith("iqa/O1")
    assert "#SBATCH -a 1-1" in (staging / "runFerebus.sh").read_text(encoding="utf-8")


def test_tiny_live_ferebus_submission_smoke(tmp_path):
    if not _enabled("ICHOR_RUN_LIVE_FEREBUS_SMOKE"):
        pytest.skip("set ICHOR_RUN_LIVE_FEREBUS_SMOKE=1 to submit a tiny real FEREBUS job")
    trainer = pytest.importorskip("pyferebus.executors.trainer")
    avail = check_backends()
    required = {"sbatch", "sacct", "ferebus", "bc", "pyferebus"}
    missing = sorted(required & set(avail.missing))
    if missing:
        pytest.skip("missing required live FEREBUS smoke backends: " + repr(missing))

    staging = _write_smoke_staging(tmp_path)
    submission = submit_ferebus(
        staging / "job-details",
        staging,
        model_class=trainer.MODEL,
        walltime_hours=1,
        ncores=1,
        nagents=4,
        maxiter=1,
        overwrite_workdir=False,
        move_dataset_files=True,
        path_to_executable=avail.ferebus_path or None,
    )

    from ichor.hpc.active_learning.submit.sacct_poll import aggregate_states, poll_job

    deadline = time.time() + 20 * 60
    last_summary = None
    while time.time() < deadline:
        observations = poll_job(submission.job_id)
        last_summary = aggregate_states(submission.job_id, observations)
        if last_summary.is_terminal and last_summary.n_tasks > 0:
            break
        time.sleep(10.0)
    else:
        pytest.fail("FEREBUS smoke job did not reach terminal state in 20 minutes")

    assert last_summary is not None
    assert last_summary.n_failed == 0, repr(last_summary)
    model = staging / "iqa" / "O1" / "WATER_iqa_O1.model"
    assert model.is_file(), str(model)
    assert model.stat().st_size > 0
    from ichor.hpc.active_learning.daemon.live_executor import _ferebus_model_data_rows_ok
    ok, reason = _ferebus_model_data_rows_ok(model)
    assert ok, reason
