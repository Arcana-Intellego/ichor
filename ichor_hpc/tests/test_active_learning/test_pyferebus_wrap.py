"""Tests for ichor.hpc.active_learning.submit.pyferebus_wrap.

We do not invoke a real cluster. The pyferebus MODEL is stubbed with a class
that records its kwargs and writes a fake runFerebus.sh; subprocess.run is
stubbed with a callable that records its argv and returns a configurable
CompletedProcess-like object.
"""
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

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
        self.run_calls += 1
        task_dir = self.workdir / "iqa" / "O1"
        task_dir.mkdir(parents=True, exist_ok=True)
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
        kernel="rbf_per",
        loss="mse",
        is_constant_noise=False,
        transfer_learning=True,
        nagents=10,
        maxiter=50,
        full_ARD=False,
        feature_scaling=False,
        model_class=model_class,
        submit_runner=runner,
    )

    assert isinstance(result, FerebusSubmission)
    assert result.job_id == "12345678"
    assert result.cluster is None
    assert result.submission_script == tmp_path / "runFerebus.sh"
    assert result.transfer_learning is True
    # MODEL was called with submitToComputeNode=False (forced).
    assert captured[0].kwargs["submitToComputeNode"] is False
    assert captured[0].kwargs["overwriteWD"] is False
    assert captured[0].kwargs["moveDatasetFiles"] is True
    # Surfaced kwargs were threaded through.
    assert captured[0].kwargs["wallTime"] == 12
    assert captured[0].kwargs["ncores"] == 8
    assert captured[0].kwargs["kernel"] == "rbf_per"
    assert captured[0].kwargs["loss"] == "mse"
    assert captured[0].kwargs["is_constant_noise"] is False
    assert captured[0].kwargs["transfer_learning"] is True
    assert captured[0].kwargs["nagents"] == 10
    assert captured[0].kwargs["maxiter"] == 50
    assert captured[0].kwargs["full_ARD"] is False
    assert captured[0].kwargs["scaling"] is False
    assert captured[0].kwargs["scale_feats"] is False
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
    # sbatch is driven from the staging directory because the generated script reads
    # sibling commands/list.txt files with relative paths.
    assert runner.calls == [["sbatch", "--parsable", "runFerebus.sh"]]
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
    assert lines[:8] == [
        "#!/bin/bash --login",
        "#SBATCH --partition multicore",
        "#SBATCH -n 2",
        "#SBATCH --job-name=ferebus-light",
        "#SBATCH --time=1-00:00:00",
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
    assert shlex.quote(str(exe)) + " ${line}" in script
    assert "            ferebus ${line}" not in script


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


def test_submit_ferebus_raises_when_commands_missing(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured, write_commands=False)
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FerebusSubmissionError, match="commands"):
        submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())


def test_submit_ferebus_raises_when_commands_empty(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured, command_lines=[])
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FerebusSubmissionError, match="commands"):
        submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())


def test_submit_ferebus_raises_when_list_missing(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(captured, write_list=False)
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FerebusSubmissionError, match="list.txt"):
        submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())


def test_submit_ferebus_raises_when_command_list_counts_mismatch(tmp_path):
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
    with pytest.raises(FerebusSubmissionError, match="count mismatch"):
        submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())


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


def test_submit_ferebus_rejects_list_entry_outside_workdir(tmp_path):
    captured: List[_StubModel] = []
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    outside.mkdir()
    model_class = _make_model_class(captured, list_lines=[str(outside)])
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FerebusSubmissionError, match="escapes the working directory"):
        submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())


def test_submit_ferebus_raises_when_command_missing_required_flags(tmp_path):
    captured: List[_StubModel] = []
    model_class = _make_model_class(
        captured,
        command_lines=["-c a -I b -O c -P iqa -A O1"],
    )
    jd = tmp_path / "job.json"
    jd.write_text("{}")
    with pytest.raises(FerebusSubmissionError, match="-ALF"):
        submit_ferebus(jd, tmp_path, model_class=model_class, submit_runner=_StubRunner())


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


def test_submit_ferebus_rejects_unpatchable_generated_script(tmp_path):
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
    with pytest.raises(FerebusSubmissionError, match="could not patch"):
        submit_ferebus(
            jd,
            tmp_path,
            path_to_executable=exe,
            model_class=model_class,
            submit_runner=_StubRunner(),
        )


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
