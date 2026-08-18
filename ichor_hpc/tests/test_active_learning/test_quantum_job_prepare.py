"""Quantum job preparation cannot reuse stale task products."""

import json
from pathlib import Path

import pytest

from ichor.hpc.active_learning.daemon.quantum_job_prepare import (
    prepare_quantum_task,
)


def _task(tmp_path):
    campaign = tmp_path / "campaign"
    pointdir = campaign / ".DATA" / "STAGING" / "initial" / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True)
    (pointdir / "input.gjf").write_text("input\n", encoding="utf-8")
    (pointdir / "provenance.json").write_text("{}\n", encoding="utf-8")
    return campaign, pointdir


def test_gaussian_preparation_removes_all_downstream_products(tmp_path):
    campaign, pointdir = _task(tmp_path)
    for name in (
        "input.gau",
        "input.wfn",
        "GAUSSIAN_TASK_RECEIPT.json",
        "WFN_METHOD_RECEIPT.json",
        "AIMALL_TASK.json",
        "AIMALL_COMPLETION_RECEIPT.json",
        "QUANTUM_ACCEPTANCE_RECEIPT.json",
    ):
        (pointdir / name).write_text("stale\n", encoding="utf-8")
    atomic = pointdir / "input_atomicfiles"
    atomic.mkdir()
    (atomic / "h1.int").write_text("stale\n", encoding="utf-8")
    for name in (
        "input.aim",
        "input.agp",
        "input.agpviz",
        "input.extout",
        "input.int",
        "input.mgp",
        "input.mgpviz",
        "input.sum",
        "input.sumviz",
    ):
        (pointdir / name).write_text("stale\n", encoding="utf-8")

    prepare_quantum_task(campaign, pointdir, backend="gaussian")

    assert (pointdir / "input.gjf").is_file()
    assert (pointdir / "provenance.json").is_file()
    assert sorted(path.name for path in pointdir.iterdir()) == [
        "input.gjf",
        "provenance.json",
    ]


def test_aimall_preparation_preserves_bound_inputs(tmp_path):
    campaign, pointdir = _task(tmp_path)
    for name in (
        "input.wfn",
        "GAUSSIAN_TASK_RECEIPT.json",
        "WFN_METHOD_RECEIPT.json",
        "AIMALL_COMPLETION_RECEIPT.json",
        "QUANTUM_ACCEPTANCE_RECEIPT.json",
    ):
        (pointdir / name).write_text("evidence\n", encoding="utf-8")
    (pointdir / "AIMALL_TASK.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "ferebus_row_shard": {
                    "directory": (
                        ".DATA/CACHE/FEREBUS_ROW_SHARDS/fixture/POINT_0000"
                    )
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    atomic = pointdir / "input_atomicfiles"
    atomic.mkdir()
    (atomic / "h1.int").write_text("stale\n", encoding="utf-8")

    prepare_quantum_task(campaign, pointdir, backend="aimall")

    for name in (
        "input.gjf",
        "input.wfn",
        "GAUSSIAN_TASK_RECEIPT.json",
        "WFN_METHOD_RECEIPT.json",
        "AIMALL_TASK.json",
    ):
        assert (pointdir / name).is_file()
    assert not atomic.exists()
    assert not (pointdir / "AIMALL_COMPLETION_RECEIPT.json").exists()
    assert not (pointdir / "QUANTUM_ACCEPTANCE_RECEIPT.json").exists()
    assert not [
        path
        for pattern in (
            "*.aim",
            "*.agp",
            "*.agpviz",
            "*.extout",
            "*.int",
            "*.mgp",
            "*.mgpviz",
            "*.sum",
            "*.sumviz",
        )
        for path in pointdir.glob(pattern)
    ]


def test_quantum_preparation_rejects_symlinked_children(tmp_path):
    campaign, pointdir = _task(tmp_path)
    target = tmp_path / "outside.txt"
    target.write_text("outside\n", encoding="utf-8")
    link = pointdir / "input.wfn"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="contains a symlink"):
        prepare_quantum_task(campaign, pointdir, backend="gaussian")
    assert target.read_text(encoding="utf-8") == "outside\n"
