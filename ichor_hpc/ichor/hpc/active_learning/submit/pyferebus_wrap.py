"""pyferebus wrapper that captures the SLURM JobID.

The raw pyferebus.MODEL.run() shells out via os.system at
pyferebus/src/pyferebus/executors/trainer.py:97. That call:

  - discards the JobID (daemon cannot poll for completion);
  - discards the exit code (submission failure is silent);
  - the exception handler does f.write(e) where e is an Exception,
    which itself raises TypeError and masks the original error.

submit_ferebus(...) bypasses that path by constructing MODEL with
submitToComputeNode=False, so pyferebus only writes runFerebus.sh.
We then drive sbatch --parsable via subprocess.run, capturing JobID,
optional cluster, and stdout/stderr. The result is a FerebusSubmission
dataclass ready for sacct polling.

Both pyferebus.MODEL and subprocess.run are injectable via the optional
model_class and submit_runner parameters; tests pass stubs so no real
cluster is touched.
"""
from __future__ import annotations

import subprocess
import os
import shlex
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


__all__ = [
    "FerebusSubmission",
    "FerebusSubmissionError",
    "submit_ferebus",
    "parse_sbatch_parsable_output",
]


_DEFAULT_MODEL_KWARGS: Dict[str, Any] = {
    "moveDatasetFiles": True,
    "overwriteWD": True,
    "ncores": 16,
    "ntasks": None,
    "rerun": False,
    "wallTime": 24,
    "kernel": "rbfc_per",
    "prefactor": -1,
    "KPM": 2,
    "populationProps": False,
    "maxiter": 200,
    "nagents": 20,
    "nluckyagents": 5,
    "updatefreq": 20,
    "maxdamping": 2.0,
    "mindamping": 0.1,
    "mintheta": 0.0,
    "maxtheta": 0.1,
    "adecayfactor": 10.0,
    "loss": "huber",
    "huber_delta": 0.01,
    "stagnationCheckPoint": 0.20,
    "overwritestdout": True,
    "pathToExecutable": None,
    "training": 1,
    "validation": 1,
    "minWN": 1.0e-10,
    "maxWN": 1.0e-3,
    "nilTest": 0,
    "cmeanRangeFactor": 5.0,
    "meanType": 15,
    "is_constant_noise": True,
    "regNoise": 1.0e-4,
    "max_reg_weight": 10.0,
    "batch_size": 100,
    "weights_lambda": 0.001,
    "gwo_cycles": 1,
    "elitism": True,
    "transfer_learning": False,
    "split_ratio": 0.25,
    "split_method": "random",
    "relax_weight": 0.25,
    "multisource": False,
    "full_seeding": True,
    "nsources": 4,
    "seedRNG": False,
    "level_of_theory": "b3lyp/6-31+g(d,p)",
    "iqa_deviation_factor": 1.0,
    "scaling": True,
    "full_ARD": True,
}


class FerebusSubmissionError(RuntimeError):
    """Raised when sbatch submission fails or JobID cannot be parsed."""


@dataclass(frozen=True)
class FerebusSubmission:
    job_id: str
    cluster: Optional[str]
    submission_script: Path
    working_dir: Path
    transfer_learning: bool
    sbatch_stdout: str = ""
    sbatch_stderr: str = ""
    pyferebus_kwargs: Mapping[str, Any] = field(default_factory=dict)


_REQUIRED_COMMAND_FLAGS = ("-c", "-I", "-O", "-P", "-A", "-ALF")


def parse_sbatch_parsable_output(stdout: str) -> Tuple[str, Optional[str]]:
    """Parse "sbatch --parsable" stdout into (job_id, cluster).

    SLURM emits one line of the form "<JOBID>" or "<JOBID>;<CLUSTER>".
    We are lenient about surrounding whitespace and ignore empty trailing
    lines. Raises FerebusSubmissionError on empty / non-numeric output.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise FerebusSubmissionError(
            "sbatch produced no parsable stdout; got: " + repr(stdout)
        )
    head = lines[0]
    parts = head.split(";")
    job_id = parts[0].strip()
    if not job_id or not job_id[0].isdigit():
        raise FerebusSubmissionError(
            "sbatch --parsable did not yield a JobID; first token: " + repr(head)
        )
    cluster = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    return job_id, cluster


def _ensure_path(value: Union[str, Path]) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("path must be str or Path, got " + type(value).__name__)
    return Path(value).resolve()


def _reject_control_chars(label: str, value: str) -> None:
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise FerebusSubmissionError(label + " contains a control character")


def _read_required_nonempty_lines(path: Path, label: str) -> List[str]:
    if not path.is_file():
        raise FerebusSubmissionError(
            "pyferebus did not produce required " + label + ": " + str(path)
        )
    if path.stat().st_size <= 0:
        raise FerebusSubmissionError(
            "pyferebus produced empty " + label + ": " + str(path)
        )
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not lines:
        raise FerebusSubmissionError(
            "pyferebus produced blank " + label + ": " + str(path)
        )
    return lines


def _validate_generated_pyferebus_artifacts(working_dir: Path) -> Path:
    script = working_dir / "runFerebus.sh"
    if not script.is_file():
        raise FerebusSubmissionError(
            "pyferebus did not produce " + script.name + " in " + str(working_dir)
            + "; MODEL.run() may have failed silently."
        )
    if script.stat().st_size <= 0:
        raise FerebusSubmissionError(
            "pyferebus produced empty runFerebus.sh in " + str(working_dir)
        )

    command_lines = _read_required_nonempty_lines(working_dir / "commands", "commands")
    list_lines = _read_required_nonempty_lines(working_dir / "list.txt", "list.txt")
    if len(command_lines) != len(list_lines):
        raise FerebusSubmissionError(
            "pyferebus commands/list.txt task count mismatch: commands="
            + str(len(command_lines))
            + " list.txt="
            + str(len(list_lines))
        )

    for i, folder in enumerate(list_lines, start=1):
        if not Path(folder).is_dir():
            raise FerebusSubmissionError(
                "pyferebus list.txt entry "
                + str(i)
                + " does not point to an existing task directory: "
                + folder
            )

    for i, line in enumerate(command_lines, start=1):
        tokens = line.split()
        missing = [flag for flag in _REQUIRED_COMMAND_FLAGS if flag not in tokens]
        if missing:
            raise FerebusSubmissionError(
                "pyferebus command "
                + str(i)
                + " missing required flags "
                + repr(missing)
                + ": "
                + line
            )
    return script


def _harden_generated_script(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    if "set -euo pipefail" in text:
        text = text.replace("set -euo pipefail", "set -eo pipefail")
        script.write_text(text, encoding="utf-8", newline="\n")
    lines = text.splitlines()
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    additions = []
    if "set -eo pipefail" not in text and "set -euo pipefail" not in text:
        additions.append("set -eo pipefail")
    if "export LC_ALL=C" not in text:
        additions.append("export LC_ALL=C")
    if "export LC_NUMERIC=C" not in text:
        additions.append("export LC_NUMERIC=C")
    if additions:
        lines[insert_at:insert_at] = additions
        script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _validate_configured_executable(path_to_executable: Union[str, Path]) -> str:
    exe = str(path_to_executable)
    _reject_control_chars("configured FEREBUS executable", exe)
    if not exe or exe == "ferebus":
        return exe
    if os.name != "nt":
        p = Path(exe).expanduser()
        if not p.is_file():
            raise FerebusSubmissionError(
                "configured FEREBUS executable does not exist: " + str(p)
            )
        if not os.access(str(p), os.X_OK):
            raise FerebusSubmissionError(
                "configured FEREBUS executable is not executable: " + str(p)
            )
    return exe


def _patch_generated_executable(script: Path, path_to_executable: Union[str, Path]) -> None:
    exe = _validate_configured_executable(path_to_executable)
    if not exe or exe == "ferebus":
        return
    text = script.read_text(encoding="utf-8")
    executable_call = shlex.quote(exe) + " ${line}"
    patched_text = re.sub(
        r"(?m)^([ \t]*)ferebus[ \t]+\$\{line\}[ \t]*$",
        lambda match: match.group(1) + executable_call,
        text,
    )
    if patched_text != text:
        text = patched_text
        script.write_text(text, encoding="utf-8", newline="\n")
    elif executable_call not in text:
        raise FerebusSubmissionError(
            "could not patch FEREBUS executable path into " + str(script)
        )
    patched = script.read_text(encoding="utf-8")
    if re.search(r"(?m)^[ \t]*ferebus[ \t]+\$\{line\}[ \t]*$", patched) or executable_call not in patched:
        raise FerebusSubmissionError(
            "FEREBUS executable patch validation failed for " + str(script)
        )


def _resolve_model_kwargs(
    transfer_learning: bool,
    walltime_hours: int,
    ncores: int,
    kernel: str,
    loss: str,
    is_constant_noise: bool,
    nagents: int,
    maxiter: int,
    full_ARD: bool,
    scaling: bool,
    overwrite_workdir: bool,
    move_dataset_files: bool,
    extra: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(_DEFAULT_MODEL_KWARGS)
    merged["transfer_learning"] = bool(transfer_learning)
    merged["wallTime"] = int(walltime_hours)
    merged["ncores"] = int(ncores)
    merged["kernel"] = str(kernel)
    merged["loss"] = str(loss)
    merged["is_constant_noise"] = bool(is_constant_noise)
    merged["nagents"] = int(nagents)
    merged["maxiter"] = int(maxiter)
    merged["full_ARD"] = bool(full_ARD)
    merged["scaling"] = bool(scaling)
    merged["overwriteWD"] = bool(overwrite_workdir)
    merged["moveDatasetFiles"] = bool(move_dataset_files)
    if extra:
        # these four are set by the wrapper itself further down, so accepting
        # them here would just collide at the model_class(...) call. reject
        # them outright rather than letting them slip through as "known".
        reserved = {
            "submitToComputeNode", "platform", "workingDirectory", "jdFile",
            "overwriteWD", "moveDatasetFiles", "pathToExecutable",
        }
        clash = set(extra) & reserved
        if clash:
            raise ValueError(
                "these kwargs are managed by the wrapper, not extra: "
                + repr(sorted(clash))
            )
        unknown = set(extra) - set(_DEFAULT_MODEL_KWARGS)
        if unknown:
            raise ValueError(
                "unknown extra arguments: " + repr(sorted(unknown))
            )
        merged.update(extra)
    return merged


def submit_ferebus(
    jd_file: Union[str, Path],
    working_directory: Union[str, Path],
    *,
    platform: str = "CSF4",
    walltime_hours: int = 24,
    ncores: int = 16,
    kernel: str = "rbfc_per",
    loss: str = "huber",
    is_constant_noise: bool = True,
    transfer_learning: bool = False,
    nagents: int = 20,
    maxiter: int = 200,
    full_ARD: bool = True,
    scaling: bool = True,
    overwrite_workdir: bool = False,
    move_dataset_files: bool = True,
    path_to_executable: Optional[Union[str, Path]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    model_class: Optional[Any] = None,
    submit_runner: Optional[Any] = None,
) -> FerebusSubmission:
    """Generate the FEREBUS submission script via pyferebus, then submit it
    ourselves through sbatch --parsable so we capture the JobID.

    Parameters mirror the most-used pyferebus.MODEL kwargs. Any remaining
    upstream kwargs can be passed via "extra={...}" using exact upstream names
    (e.g. {"huber_delta": 0.02, "stagnationCheckPoint": 0.15}).

    "model_class" and "submit_runner" are test seams. By default they bind to
    "pyferebus.executors.trainer.MODEL" and "subprocess.run" respectively.

    Returns
    -------
    FerebusSubmission
        Captured JobID and metadata.

    Raises
    ------
    FileNotFoundError
        If "working_directory" does not exist before submission.
    FerebusSubmissionError
        If sbatch exits non-zero or its stdout cannot be parsed.
    """
    jd_file_path = _ensure_path(jd_file)
    working_dir = _ensure_path(working_directory)
    if not working_dir.exists():
        raise FileNotFoundError(
            "working_directory does not exist: " + str(working_dir)
        )

    if model_class is None:
        from pyferebus.executors.trainer import MODEL as _MODEL
        model_class = _MODEL
    if submit_runner is None:
        submit_runner = subprocess.run

    model_kwargs = _resolve_model_kwargs(
        transfer_learning=transfer_learning,
        walltime_hours=walltime_hours,
        ncores=ncores,
        kernel=kernel,
        loss=loss,
        is_constant_noise=is_constant_noise,
        nagents=nagents,
        maxiter=maxiter,
        full_ARD=full_ARD,
        scaling=scaling,
        overwrite_workdir=overwrite_workdir,
        move_dataset_files=move_dataset_files,
        extra=extra,
    )
    #Force submitToComputeNode=False so pyferebus only writes the script;
    #we drive sbatch ourselves to capture the JobID and exit code.
    model_kwargs["submitToComputeNode"] = False

    cwd = os.getcwd()
    try:
        # pyferebus' SLURM writer enumerates property directories relative to cwd even when
        # workingDirectory is absolute, so run its generation step from the staged workdir.
        os.chdir(str(working_dir))
        model = model_class(
            jdFile=str(jd_file_path),
            workingDirectory=str(working_dir),
            platform=platform,
            **model_kwargs,
        )
        model.run()
    finally:
        os.chdir(cwd)

    script = _validate_generated_pyferebus_artifacts(working_dir)
    _harden_generated_script(script)
    if path_to_executable:
        _patch_generated_executable(script, path_to_executable)

    completed = submit_runner(
        ["sbatch", "--parsable", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = getattr(completed, "stdout", "") or ""
    stderr = getattr(completed, "stderr", "") or ""
    return_code = int(getattr(completed, "returncode", 1))
    if return_code != 0:
        raise FerebusSubmissionError(
            "sbatch exited with code " + str(return_code)
            + ". stdout: " + repr(stdout) + " stderr: " + repr(stderr)
        )

    job_id, cluster = parse_sbatch_parsable_output(stdout)
    return FerebusSubmission(
        job_id=job_id,
        cluster=cluster,
        submission_script=script,
        working_dir=working_dir,
        transfer_learning=bool(transfer_learning),
        sbatch_stdout=stdout,
        sbatch_stderr=stderr,
        pyferebus_kwargs=dict(model_kwargs),
    )
