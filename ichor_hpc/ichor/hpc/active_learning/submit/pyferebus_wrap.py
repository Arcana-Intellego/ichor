"""pyferebus wrapper that captures the native scheduler JobID.

The raw pyferebus.MODEL.run() shells out via os.system at
pyferebus/src/pyferebus/executors/trainer.py:97. That call:

  - discards the JobID (daemon cannot poll for completion);
  - discards the exit code (submission failure is silent);
  - the exception handler does f.write(e) where e is an Exception,
    which itself raises TypeError and masks the original error.

submit_ferebus(...) bypasses that path by constructing MODEL with
submitToComputeNode=False, so pyferebus only writes runFerebus.sh.
ICHOR replaces the generated scheduler wrapper and submits it through the
selected Slurm or SGE adapter, capturing the JobID and command output.

Both pyferebus.MODEL and subprocess.run are injectable via the optional
model_class and submit_runner parameters; tests pass stubs so no real
cluster is touched.
"""
from __future__ import annotations

import math
import subprocess
import os
import shlex
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .slurm_contracts import (
    parse_sbatch_parsable_output as _parse_sbatch_parsable_output,
)
from .scheduler_backend import get_scheduler_backend


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
    "meanType": 21,
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
    "level_of_theory": "b3lyp/aug-cc-pvtz",
    "iqa_deviation_factor": 1.0,
    "scaling": True,
    "scale_feats": True,
    "scale_prop": False,
    "full_ARD": True,
}


class FerebusSubmissionError(RuntimeError):
    """Raised when scheduler submission fails or its JobID is invalid."""


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
    generated_configs: Tuple[Mapping[str, Any], ...] = ()
    script_binding: Mapping[str, Any] = field(default_factory=dict)


def parse_sbatch_parsable_output(stdout: str) -> Tuple[str, Optional[str]]:
    """Parse "sbatch --parsable" stdout into (job_id, cluster).

    SLURM emits one line of the form "<JOBID>" or "<JOBID>;<CLUSTER>".
    Surrounding whitespace is accepted, but all other output must match the
    documented one-line Slurm contract exactly.
    """
    try:
        return _parse_sbatch_parsable_output(stdout)
    except ValueError as exc:
        raise FerebusSubmissionError(
            "sbatch --parsable output is invalid: " + str(exc)
        ) from exc


def _ensure_path(value: Union[str, Path]) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("path must be str or Path, got " + type(value).__name__)
    return Path(value).resolve()


def _reject_control_chars(label: str, value: str) -> None:
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise FerebusSubmissionError(label + " contains a control character")


def _validate_generated_pyferebus_artifacts(
    working_dir: Path,
    *,
    expected_tasks: Optional[int] = None,
) -> Tuple[Path, int]:
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

    from ..strict_json import strict_json as json
    from ..daemon.input_staging import FEREBUS_TASK_SCHEMA_VERSION

    manifest_path = working_dir / "FEREBUS_TASKS.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FerebusSubmissionError(
            "daemon FEREBUS_TASKS.json is missing or unreadable"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != FEREBUS_TASK_SCHEMA_VERSION
        or not isinstance(manifest.get("tasks"), list)
        or not manifest["tasks"]
    ):
        raise FerebusSubmissionError("daemon FEREBUS task manifest is invalid")
    declared_count = manifest.get("n_tasks")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(manifest["tasks"])
    ):
        raise FerebusSubmissionError(
            "daemon FEREBUS task manifest cardinality is invalid"
        )
    if expected_tasks is not None and (
        isinstance(expected_tasks, bool) or not isinstance(expected_tasks, int)
    ):
        raise FerebusSubmissionError(
            "expected FEREBUS task count must be an exact integer"
        )
    if expected_tasks is not None and declared_count != expected_tasks:
        raise FerebusSubmissionError(
            "daemon FEREBUS task count "
            + str(declared_count)
            + " does not match daemon FEREBUS_TASKS.json n_tasks="
            + str(expected_tasks)
        )
    return script, declared_count


def _format_slurm_walltime_hours(walltime_hours) -> str:
    try:
        total_seconds = int(math.ceil(float(walltime_hours) * 3600.0))
    except (TypeError, ValueError) as exc:
        raise FerebusSubmissionError("FEREBUS walltime_hours must be a positive number") from exc
    if total_seconds <= 0:
        raise FerebusSubmissionError("FEREBUS walltime_hours must be > 0")
    days, rem = divmod(total_seconds, 24 * 3600)
    hh, rem = divmod(rem, 3600)
    mm, ss = divmod(rem, 60)
    clock = f"{hh:02d}:{mm:02d}:{ss:02d}"
    return str(days) + "-" + clock if days else clock


def _harden_generated_script(
    script: Path,
    *,
    walltime_hours,
    partition: Optional[str],
    mem_per_cpu: Optional[str],
    cpus_per_task: Optional[int],
    ntasks: Optional[int],
    expected_job_name: Optional[str] = None,
    output_path: Optional[str] = None,
    error_path: Optional[str] = None,
    runtime_preamble: Optional[Sequence[str]] = None,
    expected_tasks: Optional[int] = None,
    array_concurrency_limit: Optional[int] = None,
    scheduler_kind: str = "slurm",
    scheduler_queue: Optional[str] = None,
    parallel_environment: Optional[str] = None,
) -> None:
    text = script.read_text(encoding="utf-8")
    scheduler = str(scheduler_kind).strip().lower()
    if scheduler not in {"slurm", "sge"}:
        raise FerebusSubmissionError(
            "unsupported FEREBUS scheduler " + repr(scheduler)
        )
    if expected_job_name is not None:
        _reject_control_chars("expected FEREBUS scheduler job name", str(expected_job_name))
    hardening_lines = {
        "set -eo pipefail",
        "set -euo pipefail",
        "export LC_ALL=C",
        "export LC_NUMERIC=C",
    }
    def _drop_scheduler_directive(line: str) -> bool:
        stripped = line.strip()
        if stripped.startswith("#$"):
            return True
        if scheduler == "sge":
            return stripped.startswith("#SBATCH")
        if re.match(r"^#SBATCH\s+(?:-t\b|--time(?:=|\b))", stripped):
            return True
        if partition is not None and re.match(
            r"^#SBATCH\s+(?:-p\b|--partition(?:=|\b))", stripped
        ):
            return True
        if mem_per_cpu is not None and re.match(
            r"^#SBATCH\s+(?:--mem(?:=|\s|$)|--mem-per-cpu(?:=|\s|$))",
            stripped,
        ):
            return True
        if cpus_per_task is not None and re.match(
            r"^#SBATCH\s+(?:-c\b|--cpus-per-task(?:=|\b))", stripped
        ):
            return True
        if ntasks is not None and re.match(
            r"^#SBATCH\s+(?:-n\b|--ntasks(?:=|\b))", stripped
        ):
            return True
        if expected_job_name is not None and re.match(
            r"^#SBATCH\s+(?:-J\b|--job-name(?:=|\b))", stripped
        ):
            return True
        if output_path is not None and re.match(
            r"^#SBATCH\s+(?:-o\b|--output(?:=|\b))", stripped
        ):
            return True
        if error_path is not None and re.match(
            r"^#SBATCH\s+(?:-e\b|--error(?:=|\b))", stripped
        ):
            return True
        if expected_tasks is not None and re.match(
            r"^#SBATCH\s+(?:-a\b|--array(?:=|\b))", stripped
        ):
            return True
        return False

    lines = [
        line
        for line in text.splitlines()
        if line.strip() not in hardening_lines
        and not _drop_scheduler_directive(line)
    ]
    if lines and lines[0].startswith("#!"):
        lines[0] = "#!/bin/bash" if scheduler == "sge" else lines[0]
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    while insert_at < len(lines):
        stripped = lines[insert_at].lstrip()
        if stripped.startswith("#SBATCH") or stripped.startswith("#$"):
            insert_at += 1
            continue
        break
    directives: List[str] = []
    environment_aliases: List[str]
    if scheduler == "slurm":
        if expected_job_name is not None:
            directives.append("#SBATCH --job-name=" + str(expected_job_name))
        directives.append("#SBATCH --time=" + _format_slurm_walltime_hours(walltime_hours))
        if partition is not None:
            directives.append("#SBATCH --partition=" + str(partition))
        if mem_per_cpu is not None:
            directives.append("#SBATCH --mem-per-cpu=" + str(mem_per_cpu))
        if cpus_per_task is not None:
            directives.append("#SBATCH --cpus-per-task=" + str(int(cpus_per_task)))
        if ntasks is not None:
            directives.append("#SBATCH --ntasks=" + str(int(ntasks)))
        if output_path is not None:
            directives.append("#SBATCH --output=" + str(output_path))
        if error_path is not None:
            directives.append("#SBATCH --error=" + str(error_path))
        environment_aliases = [
            'export ICHOR_SCHEDULER_JOB_ID="${SLURM_JOB_ID:?missing SLURM_JOB_ID}"',
            'export ICHOR_SCHEDULER_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"',
            'export ICHOR_SCHEDULER_CPUS="${SLURM_CPUS_PER_TASK:-1}"',
        ]
    else:
        directives.extend(["#$ -S /bin/bash", "#$ -V"])
        if expected_job_name is not None:
            directives.append("#$ -N " + str(expected_job_name))
        total_seconds = int(math.ceil(float(walltime_hours) * 3600.0))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        directives.append(
            "#$ -l h_rt="
            + str(hours).zfill(2)
            + ":"
            + str(minutes).zfill(2)
            + ":"
            + str(seconds).zfill(2)
        )
        if scheduler_queue:
            directives.append("#$ -q " + str(scheduler_queue))
        cpus = int(cpus_per_task or 1)
        if cpus > 1:
            if not parallel_environment:
                raise FerebusSubmissionError(
                    "SGE FEREBUS multicore work requires a parallel environment"
                )
            directives.append(
                "#$ -pe " + str(parallel_environment) + " " + str(cpus)
            )
        if ntasks is not None and int(ntasks) != 1:
            raise FerebusSubmissionError(
                "SGE FEREBUS requires one shared-memory task"
            )
        if mem_per_cpu is not None:
            match = re.fullmatch(
                r"([1-9][0-9]*)([KMGT]?)",
                str(mem_per_cpu).strip().upper(),
            )
            if match is None:
                raise FerebusSubmissionError(
                    "unsupported SGE FEREBUS memory syntax: "
                    + repr(mem_per_cpu)
                )
            scale = {
                "": 1,
                "K": 1.0 / 1024.0,
                "M": 1,
                "G": 1024,
                "T": 1024 * 1024,
            }[match.group(2)]
            total_memory_mib = int(
                math.ceil(int(match.group(1)) * scale * cpus)
            )
            directives.append("#$ -l h_vmem=" + str(total_memory_mib) + "M")
        if output_path is not None:
            directives.append("#$ -o " + str(output_path))
        if error_path is not None:
            directives.append("#$ -e " + str(error_path))
        environment_aliases = [
            'export ICHOR_SCHEDULER_JOB_ID="${JOB_ID:?missing JOB_ID}"',
            'export ICHOR_SCHEDULER_ARRAY_TASK_ID="$(( ${SGE_TASK_ID:?missing SGE_TASK_ID} - 1 ))"',
            'export ICHOR_SCHEDULER_CPUS="${NSLOTS:-1}"',
        ]
    if expected_tasks is not None:
        if isinstance(expected_tasks, bool) or not isinstance(expected_tasks, int):
            raise FerebusSubmissionError("expected FEREBUS task count must be an integer")
        if expected_tasks <= 0:
            raise FerebusSubmissionError("expected FEREBUS task count must be > 0")
        array_value = (
            "0-" + str(expected_tasks - 1)
            if scheduler == "slurm"
            else "1-" + str(expected_tasks) + ":1"
        )
        if array_concurrency_limit is not None:
            if (
                isinstance(array_concurrency_limit, bool)
                or not isinstance(array_concurrency_limit, int)
                or array_concurrency_limit <= 0
            ):
                raise FerebusSubmissionError(
                    "FEREBUS array concurrency limit must be a positive integer"
                )
            if scheduler == "slurm":
                array_value += "%" + str(min(expected_tasks, array_concurrency_limit))
        directives.append(
            "#SBATCH --array=" + array_value
            if scheduler == "slurm"
            else "#$ -t " + array_value
        )
        if scheduler == "sge" and array_concurrency_limit is not None:
            directives.append(
                "#$ -tc " + str(min(expected_tasks, array_concurrency_limit))
            )
    lines[insert_at:insert_at] = directives + [
        "set -eo pipefail",
        "export LC_ALL=C",
        "export LC_NUMERIC=C",
        *environment_aliases,
    ] + list(runtime_preamble or [])
    script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    if expected_job_name is not None:
        patched = script.read_text(encoding="utf-8")
        matches = re.findall(
            (
                r"(?m)^#SBATCH\s+(?:-J\s+|--job-name(?:=|\s+))(.+?)\s*$"
                if scheduler == "slurm"
                else r"(?m)^#\$\s+-N\s+(.+?)\s*$"
            ),
            patched,
        )
        if matches != [str(expected_job_name)]:
            raise FerebusSubmissionError(
                "FEREBUS job-name patch validation failed for "
                + str(script)
                + ": "
                + repr(matches)
            )
    if expected_tasks is not None:
        patched = script.read_text(encoding="utf-8")
        array_matches = re.findall(
            (
                r"(?m)^#SBATCH\s+(?:-a\s+|--array(?:=|\s+))(.+?)\s*$"
                if scheduler == "slurm"
                else r"(?m)^#\$\s+-t\s+(.+?)\s*$"
            ),
            patched,
        )
        expected_array = (
            "0-" + str(expected_tasks - 1)
            if scheduler == "slurm"
            else "1-" + str(expected_tasks) + ":1"
        )
        if scheduler == "slurm" and array_concurrency_limit is not None:
            expected_array += "%" + str(
                min(expected_tasks, int(array_concurrency_limit))
            )
        if array_matches != [expected_array]:
            raise FerebusSubmissionError(
                "FEREBUS array patch validation failed: " + repr(array_matches)
            )


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


def _validate_python_executable(python_executable: Union[str, Path]) -> str:
    """Return the lexical interpreter path used by the submitted script."""
    executable = str(python_executable)
    _reject_control_chars("configured Python executable", executable)
    if not executable.strip():
        raise FerebusSubmissionError(
            "configured Python executable must be a non-empty path"
        )
    path = Path(executable).expanduser()
    if not path.is_absolute():
        raise FerebusSubmissionError(
            "configured Python executable must be absolute: " + executable
        )
    if os.name != "nt":
        if path.is_symlink() and not path.exists():
            raise FerebusSubmissionError(
                "configured Python executable is a dangling symlink: "
                + executable
            )
        if not path.is_file():
            raise FerebusSubmissionError(
                "configured Python executable does not exist: " + executable
            )
        if not os.access(str(path), os.X_OK):
            raise FerebusSubmissionError(
                "configured Python executable is not executable: " + executable
            )
    return executable


def _resolve_model_kwargs(
    transfer_learning: bool,
    walltime_hours,
    ncores: int,
    kernel: str,
    loss: str,
    is_constant_noise: bool,
    nagents: int,
    maxiter: int,
    full_ARD: bool,
    prior_mean_type: int,
    prior_mean_level_of_theory: str,
    prior_mean_iqa_deviation_factor: float,
    feature_scaling: bool,
    property_scaling: bool,
    overwrite_workdir: bool,
    move_dataset_files: bool,
    extra: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(_DEFAULT_MODEL_KWARGS)
    merged["transfer_learning"] = bool(transfer_learning)
    merged["wallTime"] = max(1, int(math.ceil(float(walltime_hours))))
    merged["ncores"] = int(ncores)
    merged["kernel"] = str(kernel)
    merged["loss"] = str(loss)
    merged["is_constant_noise"] = bool(is_constant_noise)
    merged["nagents"] = int(nagents)
    merged["maxiter"] = int(maxiter)
    merged["full_ARD"] = bool(full_ARD)
    merged["meanType"] = int(prior_mean_type)
    merged["level_of_theory"] = str(prior_mean_level_of_theory)
    merged["iqa_deviation_factor"] = float(prior_mean_iqa_deviation_factor)
    merged["scaling"] = bool(feature_scaling or property_scaling)
    merged["scale_feats"] = bool(feature_scaling)
    merged["scale_prop"] = bool(property_scaling)
    merged["overwriteWD"] = bool(overwrite_workdir)
    merged["moveDatasetFiles"] = bool(move_dataset_files)
    if extra:
        # these four are set by the wrapper itself further down, so accepting
        # them here would just collide at the model_class(...) call. reject
        # them outright rather than letting them slip through as "known".
        reserved = {
            "submitToComputeNode", "platform", "workingDirectory", "jdFile",
            "overwriteWD", "moveDatasetFiles", "pathToExecutable",
            "meanType", "level_of_theory", "iqa_deviation_factor",
            "scaling", "scale_feats", "scale_prop",
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


def _validate_active_learning_training_contract(
    *,
    kernel: Any,
    loss: Any,
    is_constant_noise: Any,
    transfer_learning: Any,
    full_ARD: Any,
    feature_scaling: Any,
    property_scaling: Any,
    nagents: Any,
    maxiter: Any,
) -> None:
    """Reject backend options that are not supported by the active-learning contract."""
    if kernel not in {"rbf", "rbfc_per"}:
        raise FerebusSubmissionError(
            "active-learning FEREBUS kernel must be 'rbf' or 'rbfc_per'"
        )
    if loss != "huber":
        raise FerebusSubmissionError(
            "active-learning FEREBUS requires Huber loss"
        )
    fixed_booleans = {
        "is_constant_noise": (is_constant_noise, True),
        "transfer_learning": (transfer_learning, False),
        "full_ARD": (full_ARD, True),
        "feature_scaling": (feature_scaling, True),
        "property_scaling": (property_scaling, False),
    }
    for label, (observed, required) in fixed_booleans.items():
        if not isinstance(observed, bool) or observed is not required:
            raise FerebusSubmissionError(
                "active-learning FEREBUS requires "
                + label
                + "="
                + str(required)
            )
    for label, value in (("nagents", nagents), ("maxiter", maxiter)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FerebusSubmissionError(
                "active-learning FEREBUS " + label + " must be a positive integer"
            )


_PRIOR_CONFIG_FIELDS = {
    "mean_type": ("mean_type", None),
    "level_of_theory": ("level_of_theory", None),
    "iqadeviationfactor": ("iqaDeviationFactor", None),
    "scaling": ("scaling", None),
    "scale_feats": ("scale_feats", None),
    "scale_prop": ("scale_prop", None),
}


def _patch_one_generated_config(script_path: Path, contract: Any) -> Dict[str, Any]:
    """Patch one pyferebus config and prove its physical-prior contract."""
    from ..daemon.state import atomic_write_text
    from ..ferebus_prior import validate_ferebus_config_contract
    from ..versioning.manifest import sha256_file

    path = Path(script_path)
    if path.is_symlink() or not path.is_file():
        raise FerebusSubmissionError(
            "pyferebus did not produce a regular ferebus.config: " + str(path)
        )
    replacements = {
        "mean_type": str(int(contract.mean_type)),
        "level_of_theory": '"'
        + str(contract.level_of_theory or "not_applicable")
        + '"',
        "iqadeviationfactor": repr(float(contract.iqa_deviation_factor)),
        "scaling": "1" if (contract.feature_scaling or contract.property_scaling) else "0",
        "scale_feats": "1" if contract.feature_scaling else "0",
        "scale_prop": "1" if contract.property_scaling else "0",
    }
    lines = path.read_text(encoding="utf-8").splitlines()
    seen = set()
    patched_lines: List[str] = []
    for raw in lines:
        match = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=).*$", raw)
        if match is None:
            patched_lines.append(raw)
            continue
        folded = match.group(2).casefold()
        if folded not in replacements:
            patched_lines.append(raw)
            continue
        if folded in seen:
            raise FerebusSubmissionError(
                "pyferebus config contains duplicate managed field "
                + repr(match.group(2))
                + ": "
                + str(path)
            )
        seen.add(folded)
        canonical_name = _PRIOR_CONFIG_FIELDS[folded][0]
        patched_lines.append(
            match.group(1) + canonical_name + " = " + replacements[folded]
        )
    for folded, (canonical_name, _unused) in _PRIOR_CONFIG_FIELDS.items():
        if folded not in seen:
            patched_lines.append(canonical_name + " = " + replacements[folded])
    atomic_write_text(
        path,
        "\n".join(patched_lines) + "\n",
    )
    try:
        parsed = validate_ferebus_config_contract(path, contract)
    except Exception as exc:
        raise FerebusSubmissionError(
            "generated FEREBUS config violates the physical-prior contract: "
            + str(path)
            + ": "
            + str(exc)
        ) from exc
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "parsed_contract": parsed,
        "prior_mean_contract_sha256": contract.contract_sha256,
    }


def _patch_generated_configs(working_dir: Path, contract: Any) -> List[Dict[str, Any]]:
    from ..strict_json import strict_json as json

    manifest_path = working_dir / "FEREBUS_TASKS.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FerebusSubmissionError(
            "cannot enumerate generated FEREBUS configs"
        ) from exc
    tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise FerebusSubmissionError("FEREBUS task manifest has no tasks")
    root = working_dir.resolve()
    records: List[Dict[str, Any]] = []
    for expected_index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict) or task.get("task_index") != expected_index:
            raise FerebusSubmissionError("FEREBUS task ordering is invalid")
        relative = task.get("config_path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).is_absolute()
        ):
            raise FerebusSubmissionError("FEREBUS config path is invalid")
        config_path = working_dir.joinpath(*relative.split("/"))
        try:
            config_path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise FerebusSubmissionError(
                "FEREBUS config path escapes its working directory"
            ) from exc
        current = working_dir
        for part in Path(*relative.split("/")).parts:
            current = current / part
            if current.is_symlink():
                raise FerebusSubmissionError("FEREBUS config path contains a symlink")
        records.append(_patch_one_generated_config(config_path, contract))
    return records


def _bind_generated_configs_to_task_manifest(
    working_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Bind generated config bytes into FEREBUS_TASKS before scheduler submission."""
    from ..strict_json import strict_json as json

    from ..daemon.state import atomic_write_json

    manifest_path = working_dir / "FEREBUS_TASKS.json"
    if not manifest_path.is_file():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FerebusSubmissionError(
            "cannot bind generated configs to FEREBUS_TASKS.json"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise FerebusSubmissionError("FEREBUS_TASKS.json is invalid during config binding")
    by_relative: Dict[str, Dict[str, Any]] = {}
    for record in records:
        path = Path(str(record["path"]))
        try:
            relative = path.resolve().relative_to(working_dir.resolve()).as_posix()
        except ValueError as exc:
            raise FerebusSubmissionError(
                "generated FEREBUS config escapes its working directory"
            ) from exc
        bound = dict(record)
        bound["path"] = relative
        by_relative[relative] = bound
    for task in payload["tasks"]:
        if not isinstance(task, dict):
            raise FerebusSubmissionError("FEREBUS_TASKS.json contains an invalid task")
        relative = str(task.get("config_path") or "")
        record = by_relative.get(relative)
        if record is None:
            raise FerebusSubmissionError(
                "generated FEREBUS config does not match task " + repr(relative)
            )
        task["generated_config"] = record
    if len(by_relative) != len(payload["tasks"]):
        raise FerebusSubmissionError("generated FEREBUS config/task coverage mismatch")
    atomic_write_json(manifest_path, payload)


def _read_bound_generated_configs(
    working_dir: Path,
) -> Tuple[Mapping[str, Any], ...]:
    """Authenticate already generated configs without rewriting their manifest."""
    from ..strict_json import strict_json as json
    from ..versioning.manifest import sha256_file

    root = Path(working_dir).resolve()
    manifest_path = root / "FEREBUS_TASKS.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FerebusSubmissionError(
            "cannot read prepared FEREBUS task manifest"
        ) from exc
    tasks = payload.get("tasks") if isinstance(payload, Mapping) else None
    if not isinstance(tasks, list) or not tasks:
        raise FerebusSubmissionError(
            "prepared FEREBUS task manifest has no tasks"
        )
    records: List[Mapping[str, Any]] = []
    seen = set()
    for task in tasks:
        binding = (
            task.get("generated_config")
            if isinstance(task, Mapping)
            else None
        )
        if not isinstance(binding, Mapping):
            raise FerebusSubmissionError(
                "prepared FEREBUS task has no generated-config binding"
            )
        relative = str(binding.get("path") or "")
        if (
            not relative
            or Path(relative).is_absolute()
            or "\\" in relative
            or relative in seen
        ):
            raise FerebusSubmissionError(
                "prepared FEREBUS generated-config path is invalid"
            )
        seen.add(relative)
        parts = tuple(relative.split("/"))
        if any(part in {"", ".", ".."} for part in parts):
            raise FerebusSubmissionError(
                "prepared FEREBUS generated-config path is not canonical"
            )
        candidate = root.joinpath(*parts)
        try:
            candidate.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise FerebusSubmissionError(
                "prepared FEREBUS generated config escapes staging"
            ) from exc
        current = root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise FerebusSubmissionError(
                    "prepared FEREBUS generated config contains a symlink"
                )
        if candidate.is_symlink() or not candidate.is_file():
            raise FerebusSubmissionError(
                "prepared FEREBUS generated config is missing"
            )
        size = binding.get("size")
        digest = str(binding.get("sha256") or "")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or int(candidate.stat().st_size) != int(size)
            or sha256_file(candidate) != digest
        ):
            raise FerebusSubmissionError(
                "prepared FEREBUS generated config binding is invalid"
            )
        records.append(
            {
                "path": str(candidate),
                "size": int(size),
                "sha256": digest,
            }
        )
    return tuple(records)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    import hashlib
    from ..strict_json import strict_json as json

    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_structured_task_map(
    working_dir: Path,
    *,
    executable: Union[str, Path],
    execution_kind: str = "native_ferebus",
    performance_required: bool = True,
    require_existing_match: bool = False,
) -> Path:
    """Bind pyferebus-generated configs to shell-free daemon task records."""
    from ..daemon.ferebus_task_runner import (
        FEREBUS_TASK_MAP_FILENAME,
        FEREBUS_TASK_MAP_SCHEMA_VERSION,
        FEREBUS_TASK_RECEIPT_FILENAME,
    )
    from ..strict_json import strict_json as json
    from ..daemon.state import atomic_write_json
    from ..versioning.manifest import sha256_file

    if execution_kind not in {
        "native_ferebus",
        "imported_model_bootstrap",
        "synthetic_dry_run",
    }:
        raise FerebusSubmissionError(
            "unsupported FEREBUS execution kind " + repr(execution_kind)
        )
    if not isinstance(performance_required, bool):
        raise FerebusSubmissionError("performance_required must be a boolean")
    if performance_required != (execution_kind != "imported_model_bootstrap"):
        raise FerebusSubmissionError(
            "FEREBUS execution kind and performance requirement disagree"
        )
    manifest_path = working_dir / "FEREBUS_TASKS.json"
    try:
        from ..daemon.input_staging import read_ferebus_manifest

        manifest = read_ferebus_manifest(
            working_dir,
            verify_dataset_files=True,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise FerebusSubmissionError(
            "cannot build FEREBUS task map from authenticated inputs: " + str(exc)
        ) from exc
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise FerebusSubmissionError("FEREBUS task manifest has no tasks")
    declared_n_tasks = manifest.get("n_tasks")
    if (
        isinstance(declared_n_tasks, bool)
        or not isinstance(declared_n_tasks, int)
        or declared_n_tasks != len(tasks)
    ):
        raise FerebusSubmissionError("FEREBUS task manifest cardinality is invalid")
    raw_executable = _validate_configured_executable(executable)
    resolved_executable = raw_executable
    if raw_executable and raw_executable != "ferebus":
        resolved_executable = str(Path(raw_executable).expanduser().resolve())
    elif raw_executable == "ferebus":
        resolved_executable = shutil.which("ferebus") or "ferebus"
    executable_path = Path(resolved_executable)
    executable_record = {
        "path": resolved_executable,
        "sha256": (
            sha256_file(executable_path)
            if executable_path.is_file() and not executable_path.is_symlink()
            else None
        ),
    }
    mapped_tasks: List[Dict[str, Any]] = []
    for expected_index, task in enumerate(tasks, start=1):
        if (
            not isinstance(task, dict)
            or isinstance(task.get("task_index"), bool)
            or task.get("task_index") != expected_index
        ):
            raise FerebusSubmissionError("FEREBUS task ordering is invalid")
        generated_config = task.get("generated_config")
        datasets = task.get("datasets")
        if not isinstance(generated_config, dict) or not isinstance(datasets, dict):
            raise FerebusSubmissionError("FEREBUS task inputs are not fully bound")
        command_args = task.get("command_args")
        if (
            not isinstance(command_args, list)
            or not command_args
            or any(
                not isinstance(value, str)
                or not value
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
                for value in command_args
            )
        ):
            raise FerebusSubmissionError("FEREBUS task argv is invalid")
        argv = [resolved_executable, *command_args]
        mapped_tasks.append(
            {
                "task_index": expected_index,
                "property": task.get("property"),
                "atom": task.get("atom"),
                "argv": argv,
                "config": {
                    key: generated_config[key]
                    for key in ("path", "size", "sha256")
                },
                "datasets": {
                    split: {
                        key: datasets[split][key]
                        for key in ("path", "size", "sha256")
                    }
                    for split in ("train", "int_val", "ext_val")
                },
                "expected_model_path": task.get("expected_model_path"),
                "expected_performance_path": Path(
                    str(task.get("expected_model_path"))
                ).with_suffix(".perf").as_posix(),
                "receipt_path": (
                    str(task.get("output_dir"))
                    + "/"
                    + FEREBUS_TASK_RECEIPT_FILENAME
                ),
            }
        )
    payload: Dict[str, Any] = {
        "schema_version": FEREBUS_TASK_MAP_SCHEMA_VERSION,
        "task_manifest_path": manifest_path.name,
        "task_manifest_sha256": sha256_file(manifest_path),
        "executable": executable_record,
        "execution_kind": execution_kind,
        "performance_required": performance_required,
        "n_tasks": len(mapped_tasks),
        "tasks": mapped_tasks,
    }
    payload["task_map_sha256"] = _canonical_sha256(payload)
    path = working_dir / FEREBUS_TASK_MAP_FILENAME
    if require_existing_match:
        if path.is_symlink() or not path.is_file():
            raise FerebusSubmissionError(
                "prepared FEREBUS task map is missing"
            )
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FerebusSubmissionError(
                "prepared FEREBUS task map is unreadable"
            ) from exc
        if existing != payload:
            raise FerebusSubmissionError(
                "prepared FEREBUS task map does not match current immutable inputs"
            )
        return path
    atomic_write_json(path, payload)
    return path


def _replace_with_structured_task_script(
    script: Path,
    *,
    task_map: Path,
    scheduler_kind: str,
    python_executable: Union[str, Path],
    scheduler_task_map: Optional[Path] = None,
) -> None:
    """Discard backend-owned shell commands and invoke the structured runner."""
    from ..daemon.state import atomic_write_text

    scheduler = str(scheduler_kind).strip().lower()
    if scheduler not in {"slurm", "sge"}:
        raise FerebusSubmissionError(
            "unsupported FEREBUS scheduler " + repr(scheduler)
        )
    shebang = "#!/bin/bash" if scheduler == "sge" else "#!/bin/bash --login"
    scheduler_directives: List[str] = []
    if scheduler == "slurm":
        try:
            original_lines = script.read_text(encoding="utf-8").splitlines()
            if original_lines and original_lines[0].startswith("#!"):
                shebang = original_lines[0]
            scheduler_directives = [
                line
                for line in original_lines
                if line.lstrip().startswith("#SBATCH")
            ]
        except OSError:
            pass
    command = (
        shlex.quote(_validate_python_executable(python_executable))
        + " -m ichor.hpc.active_learning.daemon.ferebus_task_runner"
        + " --task-map "
        + shlex.quote(str(task_map.resolve()))
    )
    if scheduler_task_map is not None:
        from ..daemon.script_bundles import read_array_task_map

        retry_map = Path(scheduler_task_map)
        logical_ids = list(read_array_task_map(retry_map))
        if not logical_ids:
            raise FerebusSubmissionError(
                "FEREBUS scheduler task map contains no retry tasks"
            )
        command += (
            " --scheduler-task-map "
            + shlex.quote(str(retry_map.resolve()))
        )
    command += ' --task-index "${ICHOR_SCHEDULER_ARRAY_TASK_ID}"'
    body = [shebang, *scheduler_directives, command]
    atomic_write_text(script, "\n".join(body) + "\n")


def submit_ferebus(
    jd_file: Union[str, Path],
    working_directory: Union[str, Path],
    *,
    platform: str = "CSF4",
    walltime_hours=24,
    ncores: int = 16,
    partition: Optional[str] = None,
    mem_per_cpu: Optional[str] = None,
    cpus_per_task: Optional[int] = None,
    ntasks: Optional[int] = None,
    kernel: str = "rbfc_per",
    loss: str = "huber",
    is_constant_noise: bool = True,
    transfer_learning: bool = False,
    nagents: int = 20,
    maxiter: int = 200,
    full_ARD: bool = True,
    prior_mean_type: int = 21,
    prior_mean_level_of_theory: str = "b3lyp/aug-cc-pvtz",
    prior_mean_iqa_deviation_factor: float = 1.0,
    feature_scaling: bool = True,
    property_scaling: bool = False,
    overwrite_workdir: bool = False,
    move_dataset_files: bool = True,
    path_to_executable: Optional[Union[str, Path]] = None,
    python_executable: Optional[Union[str, Path]] = None,
    expected_tasks: Optional[int] = None,
    submitted_tasks: Optional[int] = None,
    scheduler_task_map: Optional[Union[str, Path]] = None,
    reuse_prepared_inputs: bool = False,
    require_existing_task_map_match: Optional[bool] = None,
    expected_job_name: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    model_class: Optional[Any] = None,
    submit_runner: Optional[Any] = None,
    submission_script_path: Optional[Union[str, Path]] = None,
    output_path: Optional[str] = None,
    error_path: Optional[str] = None,
    runtime_preamble: Optional[Sequence[str]] = None,
    scheduler_timeout_seconds: int = 60,
    array_concurrency_limit: Optional[int] = None,
    scheduler_kind: str = "slurm",
    scheduler_queue: Optional[str] = None,
    parallel_environment: Optional[str] = None,
    prepared_callback: Optional[
        Callable[[Path, Path, Sequence[Mapping[str, Any]]], Mapping[str, Any]]
    ] = None,
    pre_submit_hook: Optional[
        Callable[[Path, Mapping[str, Any]], None]
    ] = None,
) -> FerebusSubmission:
    """Generate FEREBUS inputs via pyferebus and submit through the adapter.

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
        If scheduler submission fails or its output cannot be parsed.
    """
    jd_file_path = _ensure_path(jd_file)
    working_dir = _ensure_path(working_directory)
    if not working_dir.exists():
        raise FileNotFoundError(
            "working_directory does not exist: " + str(working_dir)
        )

    _validate_active_learning_training_contract(
        kernel=kernel,
        loss=loss,
        is_constant_noise=is_constant_noise,
        transfer_learning=transfer_learning,
        full_ARD=full_ARD,
        feature_scaling=feature_scaling,
        property_scaling=property_scaling,
        nagents=nagents,
        maxiter=maxiter,
    )

    if not isinstance(reuse_prepared_inputs, bool):
        raise FerebusSubmissionError(
            "reuse_prepared_inputs must be a boolean"
        )
    if require_existing_task_map_match is None:
        require_existing_task_map_match = bool(reuse_prepared_inputs)
    if not isinstance(require_existing_task_map_match, bool):
        raise FerebusSubmissionError(
            "require_existing_task_map_match must be a boolean or null"
        )
    if require_existing_task_map_match and not reuse_prepared_inputs:
        raise FerebusSubmissionError(
            "existing FEREBUS task-map matching requires prepared-input reuse"
        )
    if model_class is None and not reuse_prepared_inputs:
        from pyferebus.executors.trainer import MODEL as _MODEL
        model_class = _MODEL
    if submit_runner is None:
        submit_runner = subprocess.run

    from ..ferebus_prior import (
        FerebusPriorContract,
        PRIOR_MEAN_TYPES,
        contract_from_payload,
    )

    if isinstance(prior_mean_type, bool) or not isinstance(prior_mean_type, int):
        raise FerebusSubmissionError(
            "active-learning FEREBUS mean type must be an exact integer"
        )
    strategies = {
        mean_type: strategy for strategy, mean_type in PRIOR_MEAN_TYPES.items()
    }
    strategy = strategies.get(prior_mean_type)
    if strategy is None:
        raise FerebusSubmissionError(
            "unsupported active-learning FEREBUS mean type: "
            + str(prior_mean_type)
        )

    prior_contract = FerebusPriorContract(
        strategy=strategy,
        mean_type=int(prior_mean_type),
        level_of_theory=(
            str(prior_mean_level_of_theory)
            if strategy == "physical_atomic_iqa"
            else None
        ),
        physical_prior_scale=float(prior_mean_iqa_deviation_factor),
    )
    try:
        prior_contract = contract_from_payload(prior_contract.to_dict())
    except Exception as exc:
        raise FerebusSubmissionError(
            "invalid active-learning FEREBUS prior contract: " + str(exc)
        ) from exc
    if not bool(feature_scaling) or bool(property_scaling):
        raise FerebusSubmissionError(
            "active-learning FEREBUS requires feature scaling and disables property scaling"
        )

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
        prior_mean_type=prior_contract.mean_type,
        prior_mean_level_of_theory=prior_contract.level_of_theory,
        prior_mean_iqa_deviation_factor=prior_contract.iqa_deviation_factor,
        feature_scaling=prior_contract.feature_scaling,
        property_scaling=prior_contract.property_scaling,
        overwrite_workdir=overwrite_workdir,
        move_dataset_files=move_dataset_files,
        extra=extra,
    )
    # Force pyferebus to write inputs without submitting its generated script.
    model_kwargs["submitToComputeNode"] = False

    if not reuse_prepared_inputs:
        cwd = os.getcwd()
        try:
            # pyferebus' writer enumerates property directories relative to
            # cwd even when workingDirectory is absolute.
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

    script, generated_task_count = _validate_generated_pyferebus_artifacts(
        working_dir,
        expected_tasks=expected_tasks,
    )
    if expected_tasks is None:
        expected_tasks = generated_task_count
    submitted_task_count = (
        int(expected_tasks)
        if submitted_tasks is None
        else int(submitted_tasks)
    )
    if submitted_task_count <= 0 or submitted_task_count > int(expected_tasks):
        raise FerebusSubmissionError(
            "FEREBUS submitted task count must be between one and the "
            "generated task count"
        )
    scheduler_task_map_path = (
        None if scheduler_task_map is None else Path(scheduler_task_map)
    )
    if scheduler_task_map_path is None and submitted_task_count != int(
        expected_tasks
    ):
        raise FerebusSubmissionError(
            "a partial FEREBUS submission requires a scheduler task map"
        )
    if reuse_prepared_inputs:
        generated_configs = _read_bound_generated_configs(working_dir)
    else:
        generated_configs = _patch_generated_configs(
            working_dir,
            prior_contract,
        )
        _bind_generated_configs_to_task_manifest(
            working_dir,
            generated_configs,
        )
    if prepared_callback is not None:
        overrides = prepared_callback(working_dir, script, generated_configs)
        if not isinstance(overrides, Mapping):
            raise FerebusSubmissionError(
                "FEREBUS prepared callback must return a mapping"
            )
        allowed = {
            "partition",
            "mem_per_cpu",
            "cpus_per_task",
            "ntasks",
            "output_path",
            "error_path",
            "runtime_preamble",
            "submission_script_path",
            "path_to_executable",
            "array_concurrency_limit",
            "scheduler_queue",
            "parallel_environment",
        }
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise FerebusSubmissionError(
                "FEREBUS prepared callback returned unknown fields: "
                + repr(unknown)
            )
        partition = overrides.get("partition", partition)
        mem_per_cpu = overrides.get("mem_per_cpu", mem_per_cpu)
        cpus_per_task = overrides.get("cpus_per_task", cpus_per_task)
        ntasks = overrides.get("ntasks", ntasks)
        output_path = overrides.get("output_path", output_path)
        error_path = overrides.get("error_path", error_path)
        runtime_preamble = overrides.get("runtime_preamble", runtime_preamble)
        submission_script_path = overrides.get(
            "submission_script_path", submission_script_path
        )
        path_to_executable = overrides.get(
            "path_to_executable", path_to_executable
        )
        array_concurrency_limit = overrides.get(
            "array_concurrency_limit", array_concurrency_limit
        )
        scheduler_queue = overrides.get("scheduler_queue", scheduler_queue)
        parallel_environment = overrides.get(
            "parallel_environment",
            parallel_environment,
        )
    task_map = _write_structured_task_map(
        working_dir,
        executable=path_to_executable or "ferebus",
        require_existing_match=bool(require_existing_task_map_match),
    )
    _replace_with_structured_task_script(
        script,
        task_map=task_map,
        scheduler_kind=scheduler_kind,
        python_executable=(
            sys.executable if python_executable is None else python_executable
        ),
        scheduler_task_map=scheduler_task_map_path,
    )
    _harden_generated_script(
        script,
        walltime_hours=walltime_hours,
        partition=partition,
        mem_per_cpu=mem_per_cpu,
        cpus_per_task=cpus_per_task,
        ntasks=ntasks,
        expected_job_name=expected_job_name,
        output_path=output_path,
        error_path=error_path,
        runtime_preamble=runtime_preamble,
        expected_tasks=submitted_task_count,
        array_concurrency_limit=array_concurrency_limit,
        scheduler_kind=scheduler_kind,
        scheduler_queue=scheduler_queue,
        parallel_environment=parallel_environment,
    )
    if submission_script_path is not None:
        from ..daemon.state import atomic_write_text

        submitted_script = Path(submission_script_path)
        submitted_body = script.read_text(encoding="utf-8")
        if submitted_script.exists() or submitted_script.is_symlink():
            if submitted_script.is_symlink() or not submitted_script.is_file():
                raise FerebusSubmissionError(
                    "FEREBUS attempt script is not a regular file: "
                    + str(submitted_script)
                )
            if submitted_script.read_text(encoding="utf-8") != submitted_body:
                raise FerebusSubmissionError(
                    "FEREBUS attempt script already exists with different content"
                )
        else:
            atomic_write_text(submitted_script, submitted_body)
        try:
            submitted_script.chmod(0o700)
        except OSError:
            pass
        script = submitted_script

    from ..daemon.script_bundles import AttemptBundle, write_script_binding

    binding_bundle = AttemptBundle(
        root=script.parent,
        script=script,
        outputs=script.parent / "OUTPUTS",
        errors=script.parent / "ERRORS",
    )
    script_binding = write_script_binding(binding_bundle)
    if pre_submit_hook is not None:
        pre_submit_hook(script, script_binding)

    submitted_argument = (
        str(script) if submission_script_path is not None else script.name
    )
    scheduler = get_scheduler_backend(scheduler_kind)
    try:
        scheduler_submission = scheduler.submit(
            submitted_argument,
            binding_sha256=str(script_binding["sha256"]),
            runner=submit_runner,
            timeout_seconds=int(scheduler_timeout_seconds),
            cwd=str(working_dir),
        )
    except Exception as exc:
        raise FerebusSubmissionError(
            scheduler.submit_command + " submission failed: " + str(exc)
        ) from exc
    stdout = scheduler_submission.stdout
    stderr = scheduler_submission.stderr
    job_id = scheduler_submission.job_id
    cluster: Optional[str] = None
    if str(scheduler_kind).strip().lower() == "slurm":
        # The adapter validates the same parsable Slurm contract. Preserve the
        # historical optional-cluster field for callers and stored metadata.
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
        generated_configs=tuple(generated_configs),
        script_binding=dict(script_binding),
    )
