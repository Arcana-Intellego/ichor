"""Tests for ichor.hpc.active_learning.submit.pyferebus_wrap.

We do not invoke a real cluster. The pyferebus MODEL is stubbed with a class
that records its kwargs and writes a fake runFerebus.sh; subprocess.run is
stubbed with a callable that records its argv and returns a configurable
CompletedProcess-like object.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from ichor.hpc.active_learning.daemon.ferebus_task_runner import (
    FerebusTaskRunnerError,
    _canonical_sha256,
    validate_task_receipts,
)
from ichor.hpc.active_learning.submit.pyferebus_wrap import (
    FerebusSubmission,
    FerebusSubmissionError,
    parse_sbatch_parsable_output,
    submit_ferebus,
)


@dataclass
class _StubCompletedProcess:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _StubModel:
    kwargs: Dict[str, Any]
    workdir: Path
    write_script: bool = True
    write_commands: bool = True
    write_list: bool = True
    script_text: str = "#!/bin/sh\n            ferebus ${line}\n"
    command_lines: Optional[List[str]] = None
    list_lines: Optional[List[str]] = None
    run_calls: int = 0

    def run(self):
        from ichor.hpc.active_learning.config import CampaignConfig
        from ichor.hpc.active_learning.daemon import input_staging as stg
        from ichor.hpc.active_learning.ferebus_prior import (
            backend_kernel_token,
            resolve_ferebus_prior_contract,
        )
        from ichor.hpc.active_learning.versioning.manifest import sha256_file
        from ichor.hpc.active_learning.versioning.reference_data import (
            canonical_json_sha256,
        )

        self.run_calls += 1
        prior = resolve_ferebus_prior_contract(CampaignConfig())
        task_dir = self.workdir / "iqa" / "O1"
        task_dir.mkdir(parents=True, exist_ok=True)
        datasets_dir = task_dir / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        dataset_records = {}
        split_rows = {
            "train": [0, 1],
            "int_val": [2, 3],
            "ext_val": [4, 5],
        }
        source_rows = [
            {
                "source_row_index": index,
                "pointdir_name": "POINT_" + str(index).zfill(6) + ".pointdir",
                "introduced_in_version": 0,
                "split": split,
                "provenance_sha256": format(index + 1, "064x"),
            }
            for index, split in enumerate(
                ("train", "train", "int_val", "int_val", "ext_val", "ext_val")
            )
        ]
        row_identity_payload = {
            "schema_version": stg.FEREBUS_ROW_IDENTITIES_SCHEMA_VERSION,
            "campaign_uid": "pyferebus-test",
            "reference_data_version": 0,
            "reference_data_view_sha256": "b" * 64,
            "source_rows": source_rows,
            "source_rows_sha256": canonical_json_sha256(source_rows),
            "splits": {
                split: {
                    "rows": [source_rows[index] for index in indexes],
                    "n_rows": len(indexes),
                    "row_identity_sha256": canonical_json_sha256(
                        [source_rows[index] for index in indexes]
                    ),
                }
                for split, indexes in split_rows.items()
            },
        }
        row_identity_path = self.workdir / stg.FEREBUS_ROW_IDENTITIES
        row_identity_path.write_text(
            json.dumps(row_identity_payload, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        split_payload = {
            "schema_version": 7,
            "allocation_policy": "exact_per_reference_data_version",
            "historical_training_rows": 0,
            "assignments": {
                row["pointdir_name"]: {
                    "split": row["split"],
                    "first_seen_reference_data_version": 0,
                    "assignment_version": 4,
                    "allocation_manifest_sha256": "c" * 64,
                    "provenance_sha256": row["provenance_sha256"],
                }
                for row in source_rows
            },
            "version_allocations": {},
        }
        split_path = self.workdir / stg.FEREBUS_SPLIT_SNAPSHOT
        split_path.write_text(
            json.dumps(split_payload, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for split, filename in (
            ("train", "WATER_O1_TRAINING_SET.csv"),
            ("int_val", "WATER_O1_INT_VALIDATION_SET.csv"),
            ("ext_val", "WATER_O1_EXT_VALIDATION_SET.csv"),
        ):
            dataset = datasets_dir / filename
            dataset.write_text(
                "point_name,f1,f2,f3,iqa\n"
                "P1,0.0,0.0,0.0,-75.0\n"
                "P2,0.1,0.1,0.1,-75.0\n",
                encoding="utf-8",
                newline="\n",
            )
            dataset_records[split] = {
                "path": dataset.relative_to(self.workdir).as_posix(),
                "size": dataset.stat().st_size,
                "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "rows": 2,
                "row_identity_sha256": row_identity_payload["splits"][split][
                    "row_identity_sha256"
                ],
                "row_identity_count": 2,
            }
        (task_dir / "ferebus.config").write_text(
            "mean_type = 15\n"
            'level_of_theory = "b3lyp/6-31+g(d,p)"\n'
            "iqaDeviationFactor = 0.975\n"
            "scaling = 1\n"
            "scale_feats = 1\n"
            "scale_prop = 1\n",
            encoding="utf-8",
            newline="\n",
        )
        command_args = [
            "-c",
            "iqa/O1/ferebus.config",
            "-I",
            "iqa/O1/datasets",
            "-O",
            "iqa/O1",
            "-P",
            "iqa",
            "-A",
            "O1",
            "-ALF",
            "1_2_3",
        ]
        kernel_family = (
            "periodic_rbf"
            if self.kwargs.get("kernel") == "rbfc_per"
            else "rbf"
        )
        (self.workdir / "FEREBUS_TASKS.json").write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "campaign_uid": "pyferebus-test",
                    "system": "WATER",
                    "reference_data_version": 0,
                    "reference_data_head_manifest_sha256": "a" * 64,
                    "reference_data_view_sha256": "b" * 64,
                    "n_reference_points": 6,
                    "pointdir_row_order": [
                        row["pointdir_name"] for row in source_rows
                    ],
                    "properties": ["iqa"],
                    "atoms": ["O1"],
                    "n_atoms": 1,
                    "n_tasks": 1,
                    "prior_mean_contract": prior.to_dict(),
                    "kernel_contract": {
                        "family": kernel_family,
                        "backend_token": backend_kernel_token(kernel_family),
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
                        "source_rows_sha256": row_identity_payload[
                            "source_rows_sha256"
                        ],
                    },
                    "split_ledger": {
                        "path": stg.FEREBUS_SPLIT_SNAPSHOT,
                        "size": split_path.stat().st_size,
                        "sha256": sha256_file(split_path),
                        "counts": {"train": 2, "int_val": 2, "ext_val": 2},
                        "version_allocation": {},
                        "allocation_policy": "exact_per_reference_data_version",
                        "allocation_manifest": "test",
                        "allocation_manifest_sha256": "c" * 64,
                        "forced_splits": {
                            row["pointdir_name"]: row["split"]
                            for row in source_rows
                        },
                    },
                    "degenerate_property_stats": [],
                    "tasks": [
                        {
                            "task_index": 1,
                            "property": "iqa",
                            "atom": "O1",
                            "prior_mean": prior.task_payload(
                                "iqa",
                                "O1",
                                training_values=[-75.0, -75.0],
                                training_dataset_sha256=dataset_records["train"][
                                    "sha256"
                                ],
                            ),
                            "alf_1_indexed": [1, 2, 3],
                            "alf_cli": "1_2_3",
                            "property_dir": "iqa",
                            "output_dir": "iqa/O1",
                            "input_dir": "iqa/O1/datasets",
                            "config_path": "iqa/O1/ferebus.config",
                            "command_args": command_args,
                            "training_csv": dataset_records["train"]["path"],
                            "int_validation_csv": dataset_records["int_val"]["path"],
                            "ext_validation_csv": dataset_records["ext_val"]["path"],
                            "row_counts": {
                                "train": 2,
                                "int_val": 2,
                                "ext_val": 2,
                            },
                            "historical_training_rows": 0,
                            "historical_training_row_ids": [],
                            "row_ids": split_rows,
                            "datasets": dataset_records,
                            "expected_model_path": "iqa/O1/WATER_iqa_O1.model",
                            "degenerate_property_stats": False,
                        }
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if self.write_script:
            (self.workdir / "runFerebus.sh").write_text(
                self.script_text,
                encoding="utf-8",
            )
        if self.write_commands:
            command_lines = self.command_lines if self.command_lines is not None else [
                "-c "
                + str(task_dir / "ferebus.config")
                + "  -I "
                + str(task_dir / "datasets")
                + " -O "
                + str(task_dir)
                + " -P iqa -A O1 -ALF 1_2_3"
            ]
            (self.workdir / "commands").write_text(
                "\n".join(command_lines) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        if self.write_list:
            list_lines = self.list_lines if self.list_lines is not None else [str(task_dir)]
            (self.workdir / "list.txt").write_text(
                "\n".join(list_lines) + "\n",
                encoding="utf-8",
                newline="\n",
            )


@dataclass
class _StubRunner:
    calls: List[List[str]] = field(default_factory=list)
    kwargs: List[Dict[str, Any]] = field(default_factory=list)
    result: _StubCompletedProcess = field(default_factory=lambda: _StubCompletedProcess(stdout="12345678\n"))

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        self.kwargs.append(dict(kwargs))
        return self.result


def _make_model_class(captured: List[_StubModel], **model_options):
    def _factory(**kwargs):
        workdir = Path(kwargs["workingDirectory"])
        m = _StubModel(kwargs=dict(kwargs), workdir=workdir, **model_options)
        captured.append(m)
        return m
    return _factory


def test_parse_sbatch_parsable_bare_job_id():
    job_id, cluster = parse_sbatch_parsable_output("12345678\n")
    assert job_id == "12345678"
    assert cluster is None


def test_parse_sbatch_parsable_with_cluster():
    job_id, cluster = parse_sbatch_parsable_output("12345678;csf4\n")
    assert job_id == "12345678"
    assert cluster == "csf4"


def test_parse_sbatch_parsable_strips_whitespace():
    job_id, cluster = parse_sbatch_parsable_output("  \n  12345678;csf4   \n")
    assert job_id == "12345678"
    assert cluster == "csf4"


def test_parse_sbatch_parsable_empty_raises():
    with pytest.raises(FerebusSubmissionError):
        parse_sbatch_parsable_output("")


def test_parse_sbatch_parsable_non_numeric_raises():
    with pytest.raises(FerebusSubmissionError):
        parse_sbatch_parsable_output("OOPS\n")


@pytest.mark.parametrize(
    "output",
    [
        "12345\nwarning\n",
        "12345;cluster;extra\n",
        "12345;bad cluster\n",
        "0\n",
        "+12345\n",
    ],
)
def test_parse_sbatch_parsable_rejects_noncanonical_output(output):
    with pytest.raises(FerebusSubmissionError):
        parse_sbatch_parsable_output(output)


def test_submit_ferebus_happy_path(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured)
    runner = _StubRunner()
    jd = tmp_path / "job.json"
    jd.write_text("{}")

    result = submit_ferebus(
        jd,
        tmp_path,
        platform="CSF4",
        walltime_hours=12,
        ncores=8,
        kernel="rbf",
        loss="huber",
        is_constant_noise=True,
        transfer_learning=False,
        nagents=10,
        maxiter=50,
        full_ARD=True,
        feature_scaling=True,
        model_class=model_class,
        submit_runner=runner,
    )

    assert isinstance(result, FerebusSubmission)
    assert result.job_id == "12345678"
    assert result.cluster is None
    assert result.submission_script == tmp_path / "runFerebus.sh"
    assert result.transfer_learning is False
    # MODEL was called with submitToComputeNode=False (forced).
    assert captured[0].kwargs["submitToComputeNode"] is False
    assert captured[0].kwargs["overwriteWD"] is False
    assert captured[0].kwargs["moveDatasetFiles"] is True
    # Surfaced kwargs were threaded through.
    assert captured[0].kwargs["wallTime"] == 12
    assert captured[0].kwargs["ncores"] == 8
    assert captured[0].kwargs["kernel"] == "rbf"
    assert captured[0].kwargs["loss"] == "huber"
    assert captured[0].kwargs["is_constant_noise"] is True
    assert captured[0].kwargs["transfer_learning"] is False
    assert captured[0].kwargs["nagents"] == 10
    assert captured[0].kwargs["maxiter"] == 50
    assert captured[0].kwargs["full_ARD"] is True
    assert captured[0].kwargs["scaling"] is True
    assert captured[0].kwargs["scale_feats"] is True
    assert captured[0].kwargs["scale_prop"] is False
    assert captured[0].kwargs["meanType"] == 21
    assert captured[0].kwargs["level_of_theory"] == "b3lyp/aug-cc-pvtz"
    config_text = (tmp_path / "iqa" / "O1" / "ferebus.config").read_text(
        encoding="utf-8"
    )
    assert "mean_type = 21" in config_text
    assert "scale_prop = 0" in config_text
    script = (tmp_path / "runFerebus.sh").read_text(encoding="utf-8")
    assert "#SBATCH --time=12:00:00" in script
    assert "set -eo pipefail" in script
    assert "set -euo pipefail" not in script
    assert "export LC_ALL=C" in script
    assert "export LC_NUMERIC=C" in script
    assert "#SBATCH --array=0-0" in script
    assert "ferebus ${line}" not in script
    assert "-m ichor.hpc.active_learning.daemon.ferebus_task_runner" in script
    task_map = json.loads((tmp_path / "FEREBUS_TASK_MAP.json").read_text(encoding="utf-8"))
    assert task_map["execution_kind"] == "native_ferebus"
    assert task_map["n_tasks"] == 1
    assert task_map["tasks"][0]["argv"][0] == "ferebus"
    assert task_map["tasks"][0]["argv"][1:] == [
        "-c", "iqa/O1/ferebus.config",
        "-I", "iqa/O1/datasets",
        "-O", "iqa/O1",
        "-P", "iqa",
        "-A", "O1",
        "-ALF", "1_2_3",
    ]
    # sbatch is driven from the staging directory because task-map paths are relative.
    assert len(runner.calls) == 1
    assert runner.calls[0][:2] == ["sbatch", "--parsable"]
    assert runner.calls[0][-1] == "runFerebus.sh"
    assert runner.calls[0][2].startswith(
        "--export=ALL,ICHOR_SCRIPT_BINDING_SHA256="
    )
    assert runner.kwargs[0]["cwd"] == str(tmp_path)
    assert runner.kwargs[0]["check"] is False
    assert runner.kwargs[0]["capture_output"] is True
    assert runner.kwargs[0]["text"] is True


def test_submit_ferebus_keeps_sbatch_directives_before_shell_commands(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(
        captured,
        script_text=(
            "#!/bin/bash --login\n"
            "set -eo pipefail\n"
            "export LC_ALL=C\n"
            "export LC_NUMERIC=C\n"
            "#SBATCH --partition multicore\n"
            "#SBATCH -n 2\n"
            "#SBATCH -t 12-0\n"
            "#SBATCH --job-name=ferebus-light\n"
            "module load compilers/gcc/13.3.0\n"
            "ferebus ${line}\n"
        ),
    )
    jd = tmp_path / "job.json"
    jd.write_text("{}")

    submit_ferebus(
        jd,
        tmp_path,
        platform="CSF3",
        model_class=model_class,
        submit_runner=_StubRunner(),
    )

    lines = (tmp_path / "runFerebus.sh").read_text(encoding="utf-8").splitlines()
    assert lines[:9] == [
        "#!/bin/bash --login",
        "#SBATCH --partition multicore",
        "#SBATCH -n 2",
        "#SBATCH --job-name=ferebus-light",
        "#SBATCH --time=1-00:00:00",
        "#SBATCH --array=0-0",
        "set -eo pipefail",
        "export LC_ALL=C",
        "export LC_NUMERIC=C",
    ]
    assert not any(line.strip() == "#SBATCH -t 12-0" for line in lines)
    last_sbatch = max(i for i, line in enumerate(lines) if line.startswith("#SBATCH"))
    first_shell_command = min(
        i
        for i, line in enumerate(lines)
        if line
        and not line.startswith("#!")
        and not line.startswith("#SBATCH")
        and not line.startswith("#")
    )
    assert first_shell_command > last_sbatch


def test_submit_ferebus_replaces_pyferebus_static_job_name(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(
        captured,
        script_text=(
            "#!/bin/bash --login\n"
            "#SBATCH --job-name=ferebus-light\n"
            "#SBATCH -J another-stale-name\n"
            "ferebus ${line}\n"
        ),
    )
    jd = tmp_path / "job.json"
    jd.write_text("{}")

    submit_ferebus(
        jd,
        tmp_path,
        expected_job_name="abc123-FEREBUS-4",
        model_class=model_class,
        submit_runner=_StubRunner(),
    )

    script = (tmp_path / "runFerebus.sh").read_text(encoding="utf-8")
    assert script.count("#SBATCH --job-name=abc123-FEREBUS-4") == 1
    assert "ferebus-light" not in script
    assert "another-stale-name" not in script


def test_submit_ferebus_rejects_control_character_in_expected_job_name(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured)
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FerebusSubmissionError, match="control character"):
        submit_ferebus(
            jd,
            tmp_path,
            expected_job_name="bad\nname",
            model_class=model_class,
            submit_runner=_StubRunner(),
        )


def test_submit_ferebus_rewrites_pyferebus_day_walltime_to_hours(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(
        captured,
        script_text=(
            "#!/bin/bash --login\n"
            "#SBATCH --partition multicore\n"
            "#SBATCH -t 2-0\n"
            "ferebus ${line}\n"
        ),
    )
    jd = tmp_path / "job.json"
    jd.write_text("{}")

    submit_ferebus(
        jd,
        tmp_path,
        walltime_hours=2,
        model_class=model_class,
        submit_runner=_StubRunner(),
    )

    script = (tmp_path / "runFerebus.sh").read_text(encoding="utf-8")
    assert "#SBATCH --time=02:00:00" in script
    assert "#SBATCH -t 2-0" not in script


def test_submit_ferebus_replaces_all_memory_directives(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(
        captured,
        script_text=(
            "#!/bin/bash --login\n"
            "#SBATCH --mem=32G\n"
            "#SBATCH --mem-per-cpu=2G\n"
            "#SBATCH --mem-bind=local\n"
            "#SBATCH --job-name=ferebus-light\n"
            "ferebus ${line}\n"
        ),
    )
    jd = tmp_path / "job.json"
    jd.write_text("{}")

    submit_ferebus(
        jd,
        tmp_path,
        mem_per_cpu="6G",
        model_class=model_class,
        submit_runner=_StubRunner(),
    )

    lines = (tmp_path / "runFerebus.sh").read_text(encoding="utf-8").splitlines()
    assert "#SBATCH --mem-per-cpu=6G" in lines
    assert "#SBATCH --mem=32G" not in lines
    assert "#SBATCH --mem-per-cpu=2G" not in lines
    assert "#SBATCH --mem-bind=local" in lines


def test_submit_ferebus_formats_long_walltime_as_days(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(
        captured,
        script_text=(
            "#!/bin/bash --login\n"
            "#SBATCH --time=7-0\n"
            "ferebus ${line}\n"
        ),
    )
    jd = tmp_path / "job.json"
    jd.write_text("{}")

    submit_ferebus(
        jd,
        tmp_path,
        walltime_hours=26,
        model_class=model_class,
        submit_runner=_StubRunner(),
    )

    script = (tmp_path / "runFerebus.sh").read_text(encoding="utf-8")
    assert "#SBATCH --time=1-02:00:00" in script
    assert "#SBATCH --time=7-0" not in script


def test_submit_ferebus_patches_configured_executable(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured)
    jd = tmp_path / "job"
    jd.write_text("system_name WATER\n")
    exe = tmp_path / "bin with space" / "ferebus"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(exe, 0o755)
    submit_ferebus(
        jd,
        tmp_path,
        path_to_executable=exe,
        model_class=model_class,
        submit_runner=_StubRunner(),
    )
    script = (tmp_path / "runFerebus.sh").read_text(encoding="utf-8")
    assert str(exe) not in script
    assert "ferebus ${line}" not in script
    task_map = json.loads((tmp_path / "FEREBUS_TASK_MAP.json").read_text(encoding="utf-8"))
    assert task_map["executable"]["path"] == str(exe.resolve())
    assert task_map["tasks"][0]["argv"][0] == str(exe.resolve())


def test_submit_ferebus_rejects_control_char_in_configured_executable(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured)
    jd = tmp_path / "job"
    jd.write_text("system_name WATER\n")
    with pytest.raises(FerebusSubmissionError, match="control character"):
        submit_ferebus(
            jd,
            tmp_path,
            path_to_executable="ferebus\n",
            model_class=model_class,
            submit_runner=_StubRunner(),
        )


def test_submit_ferebus_captures_cluster_field(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured)
    runner = _StubRunner(result=_StubCompletedProcess(stdout="42;csf4-test\n"))
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    result = submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=runner)
    assert result.job_id == "42"
    assert result.cluster == "csf4-test"


def test_submit_ferebus_raises_when_workdir_missing(tmp_path):
    missing = tmp_path / "no_such_dir"
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FileNotFoundError):
        submit_ferebus(jd, missing, model_class=_make_model_class([]), submit_runner=_StubRunner())


def test_submit_ferebus_raises_when_script_not_produced(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured, write_script=False)
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FerebusSubmissionError, match="runFerebus.sh"):
        submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())


def test_submit_ferebus_ignores_missing_legacy_commands_file(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured, write_commands=False)
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())
    assert (tmp_path / "FEREBUS_TASK_MAP.json").is_file()


def test_submit_ferebus_ignores_empty_legacy_commands_file(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured, command_lines=[])
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())
    assert (tmp_path / "FEREBUS_TASK_MAP.json").is_file()


def test_submit_ferebus_ignores_missing_legacy_list_file(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured, write_list=False)
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())
    assert (tmp_path / "FEREBUS_TASK_MAP.json").is_file()


def test_submit_ferebus_ignores_legacy_command_list_count_mismatch(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(
        captured,
        command_lines=[
            "-c a -I b -O c -P iqa -A O1 -ALF 1_2_3",
            "-c a -I b -O c -P iqa -A H2 -ALF 1_2_3",
        ],
    )
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())
    task_map = json.loads((tmp_path / "FEREBUS_TASK_MAP.json").read_text(encoding="utf-8"))
    assert task_map["n_tasks"] == 1


def test_task_receipt_validation_rejects_authenticated_cardinality_mismatch(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured)
    job_details = tmp_path / "job.json"
    job_details.write_text("{}")
    submit_ferebus(
        job_details,
        tmp_path,
        model_class=model_class,
        submit_runner=_StubRunner(),
    )
    task_map_path = tmp_path / "FEREBUS_TASK_MAP.json"
    task_map = json.loads(task_map_path.read_text(encoding="utf-8"))
    task_map["n_tasks"] = 2
    material = dict(task_map)
    material.pop("task_map_sha256")
    task_map["task_map_sha256"] = _canonical_sha256(material)
    task_map_path.write_text(
        json.dumps(task_map, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(FerebusTaskRunnerError, match="cardinality mismatch"):
        validate_task_receipts(tmp_path)


def test_submit_ferebus_raises_when_expected_task_count_mismatch(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured)
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FerebusSubmissionError, match="n_tasks=2"):
        submit_ferebus(
            jd,
            tmp_path,
            expected_tasks=2,
            model_class=model_class,
            submit_runner=_StubRunner(),
        )


def test_submit_ferebus_ignores_untrusted_legacy_list_entry(tmp_path):
    captured: List[_StubModel] = []
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    outside.mkdir()
    model_class = _make_model_class(captured, list_lines=[str(outside)])
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())
    assert (tmp_path / "FEREBUS_TASK_MAP.json").is_file()


def test_submit_ferebus_ignores_untrusted_legacy_command_content(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(
        captured,
        command_lines=["-c a -I b -O c -P iqa -A O1"],
    )
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())
    task_map = json.loads((tmp_path / "FEREBUS_TASK_MAP.json").read_text(encoding="utf-8"))
    assert task_map["tasks"][0]["argv"][-2:] == ["-ALF", "1_2_3"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable bit is not meaningful on Windows")
def test_submit_ferebus_rejects_missing_configured_executable_on_posix(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured)
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FerebusSubmissionError, match="does not exist"):
        submit_ferebus(
            jd,
            tmp_path,
            path_to_executable=tmp_path / "missing-ferebus",
            model_class=model_class,
            submit_runner=_StubRunner(),
        )


def test_submit_ferebus_replaces_untrusted_generated_script_body(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(
        captured,
        script_text="#!/bin/sh\n            mpirun ${line}\n",
    )
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    exe = tmp_path / "ferebus"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(exe, 0o755)
    submit_ferebus(
        jd,
        tmp_path,
        path_to_executable=exe,
        model_class=model_class,
        submit_runner=_StubRunner(),
    )
    script = (tmp_path / "runFerebus.sh").read_text(encoding="utf-8")
    assert "mpirun" not in script
    assert "ferebus_task_runner" in script


def test_submit_ferebus_raises_on_sbatch_nonzero(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured)
    runner = _StubRunner(result=_StubCompletedProcess(returncode=1, stderr="Permission denied"))
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FerebusSubmissionError, match="sbatch exited with code 1"):
        submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=runner)


def test_submit_ferebus_extra_passes_through_known_kwargs(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured)
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    submit_ferebus(
        jd, tmp_path,
        extra={"huber_delta": 0.05, "stagnationCheckPoint": 0.10},
        model_class=model_class,
        submit_runner=_StubRunner(),
    )
    assert captured[0].kwargs["huber_delta"] == 0.05
    assert captured[0].kwargs["stagnationCheckPoint"] == 0.10


def test_submit_ferebus_extra_rejects_unknown(tmp_path):
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(ValueError, match="unknown extra arguments"):
        submit_ferebus(
            jd, tmp_path,
            extra={"made_up_param": 99},
            model_class=_make_model_class([]),
            submit_runner=_StubRunner(),
        )


def test_submit_ferebus_extra_rejects_managed_kwargs(tmp_path):
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(ValueError, match="managed by the wrapper"):
        submit_ferebus(
            jd,
            tmp_path,
            extra={"overwriteWD": True},
            model_class=_make_model_class([]),
            submit_runner=_StubRunner(),
        )


def test_submit_ferebus_extra_cannot_override_physical_prior(tmp_path):
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(ValueError, match="managed by the wrapper"):
        submit_ferebus(
            jd,
            tmp_path,
            extra={"meanType": 15},
            model_class=_make_model_class([]),
            submit_runner=_StubRunner(),
        )
