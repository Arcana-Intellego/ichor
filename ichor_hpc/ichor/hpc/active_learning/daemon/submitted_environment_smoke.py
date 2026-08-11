"""Opt-in scheduler smoke for the configured daemon runtime environment.

The smoke submits one short, single-core job.  It imports the Python modules
used by daemon jobs and verifies configured executable paths, but performs no
scientific calculation.  Ordinary preflight and daemon start never submit it.
"""
from __future__ import annotations

import math
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from .cluster_profile import profile_value, require_cluster_profile
from .resource_solver import (
    parallel_environment_for_partition,
    partition_memory_per_core_gb,
    scheduler_queue_for_partition,
    slurm_memory_mib,
    validate_partition_supported,
)
from .runtime_environment import (
    SUBMITTED_PYTHON_IMPORTS,
    ariadne_runtime_command_prefix,
    configured_python_isolation_lines,
    module_initialisation_lines,
    native_runtime_setup_lines,
    normalise_module_list,
    python_library_path_export_lines,
)
from .state import atomic_write_json, atomic_write_text
from ..submit.slurm_contracts import parse_sbatch_parsable_output
from ..submit.scheduler_backend import get_scheduler_backend
from ..submit.sacct_poll import aggregate_states


SMOKE_SUCCESS_MARKER = "ICHOR_SUBMITTED_ENVIRONMENT_SMOKE_OK"
_SAFE_SHEBANG_RE = re.compile(r"^#![A-Za-z0-9_./+-]+(?: [A-Za-z0-9_./+-]+)*$")
_SAFE_SLURM_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]*$")


class SubmittedEnvironmentSmokeError(RuntimeError):
    """Raised when the commissioning smoke cannot be rendered or submitted."""


def _safe_jobscript_shebang() -> str:
    raw = profile_value("hpc", "jobscript_shebang", default="#!/bin/bash --login")
    value = str(raw).strip()
    if not _SAFE_SHEBANG_RE.fullmatch(value):
        raise SubmittedEnvironmentSmokeError(
            "configured hpc.jobscript_shebang is unsafe: " + repr(value)
        )
    return value


def _safe_slurm_token(label: str, value: Any) -> str:
    token = str(value).strip()
    if not _SAFE_SLURM_TOKEN_RE.fullmatch(token):
        raise SubmittedEnvironmentSmokeError(
            label + " contains characters that cannot be rendered safely: " + repr(token)
        )
    return token


def _safe_directive_path(label: str, value: Path) -> str:
    text = str(Path(value).resolve())
    if any(character in text for character in "\r\n\x00") or " " in text:
        raise SubmittedEnvironmentSmokeError(
            label + " must not contain spaces or control characters: " + repr(text)
        )
    return text


def _smoke_mem_per_cpu(config: Any, partition: str) -> str:
    raw = str(config.resources.defaults.mem_per_cpu).strip()
    profile_gb = float(partition_memory_per_core_gb(partition))
    if raw.lower() == "auto":
        return str(max(1, int(math.floor(profile_gb)))) + "G"
    requested_gb = slurm_memory_mib(raw) / 1024.0
    if requested_gb > profile_gb + 1.0e-9:
        raise SubmittedEnvironmentSmokeError(
            "resources.defaults.mem_per_cpu="
            + repr(raw)
            + " exceeds the active profile cap of "
            + str(profile_gb)
            + " GB/core for partition "
            + repr(partition)
        )
    return raw


def _smoke_partition(config: Any, scheduler: str) -> str:
    configured = str(config.resources.defaults.partition)
    if scheduler != "sge":
        return configured
    partitions = profile_value("hpc", "partitions", default={})
    if isinstance(partitions, dict):
        serial = partitions.get("serial")
        if (
            isinstance(serial, dict)
            and bool(serial.get("daemon_supported", False))
            and int(serial.get("min_cpus", 0)) <= 1
            and int(serial.get("max_cpus", 0)) >= 1
        ):
            return "serial"
    return configured


def _required_path(label: str, value: Any) -> str:
    path = str(value or "").strip()
    if not path:
        raise SubmittedEnvironmentSmokeError(label + " is not available from preflight")
    if any(character in path for character in "\r\n\x00"):
        raise SubmittedEnvironmentSmokeError(label + " path contains control characters")
    return path


def render_submitted_environment_smoke_script(
    *,
    config: Any,
    availability: Any,
    output_path: Path,
) -> str:
    """Render the exact module/import/executable checks for one scheduler job."""
    require_cluster_profile()
    scheduler = str(profile_value("hpc", "scheduler", default="")).strip().lower()
    if scheduler not in {"slurm", "sge"}:
        raise SubmittedEnvironmentSmokeError(
            "submitted environment smoke requires hpc.scheduler: slurm or sge"
        )
    partition = _safe_slurm_token(
        "resources.defaults.partition",
        _smoke_partition(config, scheduler),
    )
    validate_partition_supported(partition)
    mem_per_cpu = _smoke_mem_per_cpu(config, partition)
    output_text = _safe_directive_path("smoke output path", output_path)
    runtime_modules = normalise_module_list(
        list(getattr(availability, "batch_runtime_modules", ()) or ()),
        label="daemon runtime",
    )
    gaussian_modules = normalise_module_list(
        profile_value("software", "gaussian", "modules", default=[]),
        label="gaussian",
    )
    aimall_modules = normalise_module_list(
        profile_value("software", "aimall", "modules", default=[]),
        label="aimall",
    )
    python_executable = _required_path(
        "configured submitted Python", availability.python_executable
    )
    gaussian_path = _required_path("Gaussian", availability.gaussian_binary)
    aimall_path = _required_path("AIMAll", availability.aimall_path)
    ferebus_path = _required_path("FEREBUS", availability.ferebus_path)
    bc_path = _required_path("bc", availability.bc_path)

    probe = "\n".join(
        [
            "import importlib",
            "modules = " + repr(SUBMITTED_PYTHON_IMPORTS),
            "for names in modules.values():",
            "    for name in names:",
            "        importlib.import_module(name)",
            "ariadne = importlib.import_module('ariadne')",
            (
                "runtime = importlib.import_module("
                "'ichor.hpc.active_learning.acquisition.ariadne_local_runner')"
            ),
            "runtime.probe_ariadne_runtime(ariadne)",
        ]
    )
    lines = [_safe_jobscript_shebang()]
    if scheduler == "slurm":
        lines += [
            "#SBATCH --job-name=ichor-env-smoke",
            "#SBATCH --partition=" + partition,
            "#SBATCH --time=00:05:00",
            "#SBATCH --mem-per-cpu=" + mem_per_cpu,
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --output=" + output_text,
        ]
    else:
        queue = _safe_slurm_token(
            "SGE queue",
            scheduler_queue_for_partition(partition),
        )
        pe = parallel_environment_for_partition(partition)
        lines += [
            "#$ -S /bin/bash",
            "#$ -V",
            "#$ -N ichor-env-smoke",
            "#$ -q " + queue,
            "#$ -l h_rt=00:05:00",
            "#$ -l h_vmem="
            + str(int(math.ceil(slurm_memory_mib(mem_per_cpu))))
            + "M",
            "#$ -o " + output_text,
        ]
        if pe:
            lines.append("#$ -pe " + _safe_slurm_token("SGE PE", pe) + " 1")
    lines += [
        "set -euo pipefail",
        "export LC_ALL=C",
        "export LC_NUMERIC=C",
        *(
            module_initialisation_lines()
            if scheduler == "sge"
            else []
        ),
        "module purge",
    ]
    lines.extend("module load " + module for module in runtime_modules)
    lines.extend(native_runtime_setup_lines())
    lines.extend(configured_python_isolation_lines())
    lines.extend(
        python_library_path_export_lines(
            list(getattr(availability, "batch_python_library_paths", ()) or ())
        )
    )
    lines.append(
        ariadne_runtime_command_prefix()
        + shlex.quote(python_executable)
        + " -c "
        + shlex.quote(probe)
    )
    for module in gaussian_modules:
        if module not in runtime_modules:
            lines.append("module load " + module)
    for module in aimall_modules:
        if module not in runtime_modules and module not in gaussian_modules:
            lines.append("module load " + module)
    for label, executable in (
        ("Gaussian", gaussian_path),
        ("AIMAll", aimall_path),
        ("FEREBUS", ferebus_path),
        ("bc", bc_path),
    ):
        lines.append("test -x " + shlex.quote(executable))
        lines.append("printf '%s: %s\\n' " + shlex.quote(label) + " " + shlex.quote(executable))
    lines.append("printf '%s\\n' " + shlex.quote(SMOKE_SUCCESS_MARKER))
    return "\n".join(lines) + "\n"


def _parse_job_id(stdout: str) -> str:
    try:
        job_id, _cluster = parse_sbatch_parsable_output(stdout)
    except ValueError as exc:
        raise SubmittedEnvironmentSmokeError(
            "sbatch --parsable --wait returned invalid output: " + str(exc)
        ) from exc
    return job_id


def run_submitted_environment_smoke(
    *,
    campaign_dir: Path,
    config: Any,
    availability: Any,
    runner: Callable[..., Any] = subprocess.run,
    settle_attempts: int = 120,
    settle_seconds: float = 2.0,
) -> Dict[str, Any]:
    """Submit the commissioning smoke and return a machine-readable result."""
    campaign = Path(campaign_dir).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    smoke_dir = campaign / ".DATA" / "ACTIVE_LEARNING" / "environment_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    script_path = smoke_dir / ("submitted-environment-" + stamp + ".sh")
    output_path = smoke_dir / ("submitted-environment-" + stamp + ".out")
    result_path = smoke_dir / ("submitted-environment-" + stamp + ".json")
    result: Dict[str, Any] = {
        "schema_version": 1,
        "submitted": False,
        "ok": False,
        "active_profile": str(getattr(availability, "active_profile", "") or ""),
        "job_id": "",
        "script_path": str(script_path),
        "output_path": str(output_path),
        "result_path": str(result_path),
        "error": "",
    }
    try:
        if not bool(getattr(availability, "all_present", False)):
            raise SubmittedEnvironmentSmokeError(
                "ordinary campaign preflight must pass before the submitted smoke"
            )
        script = render_submitted_environment_smoke_script(
            config=config,
            availability=availability,
            output_path=output_path,
        )
        atomic_write_text(script_path, script)
        scheduler = str(
            getattr(availability, "scheduler_kind", "")
            or profile_value("hpc", "scheduler", default="slurm")
        ).strip().lower()
        if scheduler == "slurm":
            completed = runner(
                [
                    str(availability.sbatch_path),
                    "--parsable",
                    "--wait",
                    str(script_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            result["submitted"] = True
            if int(getattr(completed, "returncode", 1)) != 0:
                detail = str(
                    getattr(completed, "stderr", "")
                    or getattr(completed, "stdout", "")
                    or "submitted smoke failed"
                ).strip()
                raise SubmittedEnvironmentSmokeError(
                    "submitted smoke job returned non-zero status: " + detail[:1000]
                )
            result["job_id"] = _parse_job_id(getattr(completed, "stdout", ""))
        else:
            from ..versioning.manifest import sha256_file

            backend = get_scheduler_backend(scheduler)
            submission = backend.submit(
                script_path,
                binding_sha256=sha256_file(script_path),
                runner=runner,
                timeout_seconds=60,
            )
            result["submitted"] = True
            result["job_id"] = submission.job_id
            terminal = None
            for attempt in range(max(1, int(settle_attempts))):
                observations = backend.poll_job(
                    submission.job_id,
                    accounting_runner=runner,
                    queue_runner=runner,
                    timeout_seconds=60,
                )
                summary = aggregate_states(
                    submission.job_id,
                    observations,
                    expected_task_count=1,
                    submission_kind="scalar",
                    strict_parent_job_id=False,
                )
                if summary.is_terminal:
                    terminal = summary
                    break
                if attempt + 1 < max(1, int(settle_attempts)) and settle_seconds > 0:
                    time.sleep(float(settle_seconds))
            if terminal is None:
                raise SubmittedEnvironmentSmokeError(
                    "Sun Grid Engine did not produce terminal smoke accounting"
                )
            if not terminal.is_fully_successful:
                raise SubmittedEnvironmentSmokeError(
                    "submitted smoke job completed unsuccessfully"
                )
        output = ""
        for attempt in range(max(1, int(settle_attempts))):
            if output_path.is_file():
                output = output_path.read_text(encoding="utf-8", errors="replace")
                if SMOKE_SUCCESS_MARKER in output:
                    break
            if attempt + 1 < max(1, int(settle_attempts)) and settle_seconds > 0:
                time.sleep(float(settle_seconds))
        if SMOKE_SUCCESS_MARKER not in output:
            raise SubmittedEnvironmentSmokeError(
                "submitted smoke completed without its success marker; inspect "
                + str(output_path)
            )
        result["ok"] = True
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    atomic_write_json(result_path, result)
    return result


__all__ = [
    "SMOKE_SUCCESS_MARKER",
    "SubmittedEnvironmentSmokeError",
    "render_submitted_environment_smoke_script",
    "run_submitted_environment_smoke",
]
