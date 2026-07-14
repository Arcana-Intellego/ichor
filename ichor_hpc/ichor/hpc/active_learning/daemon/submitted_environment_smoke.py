"""Opt-in Slurm smoke for the configured daemon runtime environment.

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
    partition_memory_per_core_gb,
    slurm_memory_mib,
    validate_partition_supported,
)
from .runtime_environment import (
    SUBMITTED_PYTHON_IMPORTS,
    normalise_module_list,
)
from .state import atomic_write_json, atomic_write_text
from ..submit.slurm_contracts import parse_sbatch_parsable_output


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
    """Render the exact module/import/executable checks for one Slurm job."""
    require_cluster_profile()
    scheduler = str(profile_value("hpc", "scheduler", default="")).strip().lower()
    if scheduler != "slurm":
        raise SubmittedEnvironmentSmokeError(
            "submitted environment smoke requires hpc.scheduler: slurm"
        )
    partition = _safe_slurm_token(
        "resources.defaults.partition",
        config.resources.defaults.partition,
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
        ]
    )
    lines = [
        _safe_jobscript_shebang(),
        "#SBATCH --job-name=ichor-env-smoke",
        "#SBATCH --partition=" + partition,
        "#SBATCH --time=00:05:00",
        "#SBATCH --mem-per-cpu=" + mem_per_cpu,
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --output=" + output_text,
        "set -euo pipefail",
        "export LC_ALL=C",
        "export LC_NUMERIC=C",
        "module purge",
    ]
    lines.extend("module load " + module for module in runtime_modules)
    lines.append(
        shlex.quote(python_executable) + " -c " + shlex.quote(probe)
    )
    for module in gaussian_modules:
        if module not in runtime_modules:
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
    settle_attempts: int = 5,
    settle_seconds: float = 1.0,
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
