"""Backend availability detector.

Probes the host for the binaries / Python modules that the
LiveBackendsPhaseExecutor requires before it can drive a real campaign:

    * sbatch     -- SLURM submit
    * sacct      -- SLURM accounting (also needed by the daemon poll loop)
    * g16        -- Gaussian; required by the generated SLURM scripts
    * aimqb.ish  -- AIMAll wrapper script (per pyferebus convention)
    * FEREBUS    -- Fortran kriging executable
    * ariadne    -- importable Python module from the oneAPI build

Pytest's "live" marker decorates tests that need any of these and skips
them when the host is missing the relevant tool. The CLI uses the same
detector to refuse live mode with a clear error rather than running
a daemon that cannot submit anything.
"""
from __future__ import annotations

from ..strict_json import strict_json as json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

from .cluster_profile import (
    ClusterProfileError,
    active_machine,
    expanded_profile_value,
    profile_value,
    require_cluster_profile,
)
from .runtime_environment import (
    SUBMITTED_PYTHON_IMPORTS,
    configured_daemon_runtime_modules,
    configured_python_library_paths,
    normalise_module_list,
    python_library_path_export_lines,
)

__all__ = [
    "BackendAvailability",
    "check_backends",
    "missing_backend_message",
]


@dataclass(frozen=True)
class BackendAvailability:
    profile: bool
    sbatch: bool
    sacct: bool
    gaussian: bool
    aimall: bool
    ferebus: bool
    ariadne: bool
    pyferebus: bool
    bc: bool
    gaussian_binary: str
    sbatch_path: str
    sacct_path: str
    bc_path: str
    aimall_path: str
    ferebus_path: str
    active_profile: str
    profile_error: str
    python_executable: str
    squeue: bool = False
    squeue_path: str = ""
    batch_python: bool = False
    batch_python_version: str = ""
    batch_python_error: str = ""
    batch_runtime_modules: tuple = ()
    batch_python_library_paths: tuple = ()
    gaussian_verified: bool = False
    gaussian_probe_error: str = ""
    ariadne_probe_error: str = ""
    ariadne_abi_probe: Optional[Dict[str, object]] = None
    ariadne_runtime_probe: Optional[Dict[str, object]] = None
    pyferebus_probe_error: str = ""

    @property
    def all_present(self) -> bool:
        return (
            self.profile and self.sbatch and self.sacct and self.squeue and
            self.batch_python and self.gaussian and
            self.aimall and self.ferebus and self.ariadne and
            self.pyferebus and self.bc
        )

    @property
    def missing(self) -> List[str]:
        out: List[str] = []
        for attr in (
            "profile", "sbatch", "sacct", "squeue", "batch_python",
            "gaussian", "aimall", "ferebus",
            "ariadne", "pyferebus", "bc",
        ):
            if not getattr(self, attr):
                out.append(attr)
        return out


def _which(name: str) -> str:
    return shutil.which(name) or ""


def _from_config_or_path(backend_name: str, *path_lookup_names: str) -> str:
    """Resolve a backend executable to an actual filesystem path.

    Priority:
      1. Look up ~/ichor_config.yaml -> MACHINE -> software -> backend
         -> executable_path; expand $HOME / $VAR / ~; if the result
         is an executable file, return it.
      2. Otherwise fall back to shutil.which over the supplied
         path_lookup_names (preserves the historical behaviour for
         pre-config installs and for unit tests).

    Returns the empty string if no usable executable is found.
    """
    import os
    raw = expanded_profile_value(
        "software", backend_name, "executable_path", default=None
    )
    if raw:
        expanded = str(raw)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded
    # PATH-based fallback
    for name in path_lookup_names:
        found = shutil.which(name)
        if found:
            return found
    return ""


def _gaussian_binary() -> str:
    """Return login-node g16, or a jobscript-resolved Gaussian config marker.

    Some Gaussian modules are jobscript-only and unavailable on login nodes.
    That is still a valid live configuration as long as ichor_config.yaml
    declares both the Gaussian module and executable path for generated SLURM
    scripts.
    """
    found = _which("g16")
    if found:
        return found

    modules = profile_value("software", "gaussian", "modules", default=None)
    executable = profile_value(
        "software", "gaussian", "executable_path", default=None
    )

    if executable and modules:
        return "jobscript:" + str(executable)

    return ""


def _run_login_shell(script: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
    bash = shutil.which("bash")
    if not bash:
        raise FileNotFoundError("bash is unavailable for submitted-environment probe")
    return subprocess.run(
        [bash, "--login", "-c", str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=int(timeout),
    )


def _probe_configured_python_details(
    python_executable: str,
    modules: Optional[List[str]] = None,
    library_paths: Optional[List[str]] = None,
) -> tuple[bool, str, str, Dict[str, Dict[str, object]]]:
    """Probe each required import under the exact submitted Python environment."""
    import_status = {
        label: {"ok": False, "error": "probe did not run"}
        for label in SUBMITTED_PYTHON_IMPORTS
    }
    executable = str(python_executable or "").strip()
    if not executable:
        return (
            False,
            "",
            "software.python.python_path is not configured",
            import_status,
        )
    if not os.path.isfile(executable):
        return (
            False,
            "",
            "configured Python is not a file: " + executable,
            import_status,
        )
    if not os.access(executable, os.X_OK):
        return (
            False,
            "",
            "configured Python is not executable: " + executable,
            import_status,
        )
    probe = "\n".join(
        [
            "import importlib, json, sys",
            "modules = " + repr(SUBMITTED_PYTHON_IMPORTS),
            "results = {}",
            "for label, names in modules.items():",
            "    try:",
            "        for name in names:",
            "            importlib.import_module(name)",
            "    except Exception as exc:",
            "        results[label] = {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)[:300]}",
            "    else:",
            "        results[label] = {'ok': True, 'error': ''}",
            "if results.get('ariadne', {}).get('ok'):",
            "    try:",
            "        ariadne = importlib.import_module('ariadne')",
            "        probe_module = importlib.import_module('ichor.hpc.active_learning.acquisition.ariadne_abi')",
            "        runtime_module = importlib.import_module('ichor.hpc.active_learning.acquisition.ariadne_local_runner')",
            "        abi = probe_module.probe_ariadne_module(ariadne)",
            "        runtime_smoke = runtime_module.probe_ariadne_runtime(ariadne)",
            "        results['ariadne']['abi'] = abi",
            "        results['ariadne']['runtime_smoke'] = runtime_smoke",
            "    except Exception as exc:",
            "        results['ariadne'] = {'ok': False, 'error': 'ABI probe failed: ' + type(exc).__name__ + ': ' + str(exc)[:300]}",
            "print(json.dumps({'executable': sys.executable, 'version': list(sys.version_info[:3]), 'modules': results}, sort_keys=True))",
        ]
    )
    try:
        module_lines = ["module load " + module for module in (modules or [])]
        script = "\n".join(
            [
                "set -euo pipefail",
                "module purge",
                *module_lines,
                *python_library_path_export_lines(list(library_paths or [])),
            ]
            + ["exec " + shlex.quote(executable) + " -c " + shlex.quote(probe)]
        )
        completed = _run_login_shell(script, timeout=30)
    except Exception as exc:
        return (
            False,
            "",
            type(exc).__name__ + ": " + str(exc),
            import_status,
        )
    if int(completed.returncode) != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return (
            False,
            "",
            "configured Python import probe failed: " + detail[:500],
            import_status,
        )
    try:
        payload = json.loads((completed.stdout or "").strip().splitlines()[-1])
        version_values = tuple(int(value) for value in payload["version"])
        raw_modules = payload["modules"]
        if not isinstance(raw_modules, dict):
            raise TypeError("modules is not an object")
        parsed_status: Dict[str, Dict[str, object]] = {}
        for label in SUBMITTED_PYTHON_IMPORTS:
            raw_status = raw_modules.get(label)
            if not isinstance(raw_status, dict) or not isinstance(
                raw_status.get("ok"), bool
            ):
                raise TypeError("module status is invalid for " + label)
            parsed_status[label] = {
                "ok": bool(raw_status["ok"]),
                "error": str(raw_status.get("error") or ""),
            }
            if label == "ariadne" and isinstance(raw_status.get("abi"), dict):
                parsed_status[label]["abi"] = dict(raw_status["abi"])
            if label == "ariadne" and isinstance(
                raw_status.get("runtime_smoke"), dict
            ):
                parsed_status[label]["runtime_smoke"] = dict(
                    raw_status["runtime_smoke"]
                )
    except Exception as exc:
        return (
            False,
            "",
            "configured Python returned malformed probe output: " + str(exc),
            import_status,
        )
    version = ".".join(str(value) for value in version_values)
    if version_values[:2] < (3, 11):
        return (
            False,
            version,
            "configured batch Python must be Python 3.11 or newer",
            parsed_status,
        )
    return True, version, "", parsed_status


def _probe_configured_python(
    python_executable: str,
    modules: Optional[List[str]] = None,
) -> tuple[bool, str, str]:
    """Compatibility wrapper requiring Python 3.11 and every runtime import."""
    try:
        library_paths = configured_python_library_paths()
    except Exception as exc:
        return False, "", "configured Python library path is invalid: " + str(exc)
    interpreter_ok, version, error, import_status = (
        _probe_configured_python_details(
            python_executable,
            modules,
            library_paths,
        )
    )
    missing = [
        label
        for label, status in import_status.items()
        if not bool(status.get("ok", False))
    ]
    if interpreter_ok and missing:
        details = "; ".join(
            label + ": " + str(import_status[label].get("error") or "import failed")
            for label in missing
        )
        return False, version, "configured Python import probe failed: " + details
    return interpreter_ok, version, error


def _probe_gaussian_environment() -> tuple[bool, str, str]:
    """Resolve Gaussian under the same module block used by its jobs."""
    raw_modules = profile_value("software", "gaussian", "modules", default=None)
    try:
        gaussian_modules = normalise_module_list(raw_modules, label="gaussian")
        runtime_modules = configured_daemon_runtime_modules()
    except (ValueError, ClusterProfileError) as exc:
        return False, "", str(exc)
    modules = list(runtime_modules)
    for module in gaussian_modules:
        if module not in modules:
            modules.append(module)
    executable = str(
        profile_value("software", "gaussian", "executable_path", default="") or ""
    ).strip()
    if not modules and not executable:
        found = _which("g16")
        return bool(found), found, "" if found else "Gaussian is not configured"
    if executable and not re.fullmatch(r"[A-Za-z0-9_./${}:+~-]+", executable):
        return False, "", "configured Gaussian executable path is unsafe"
    module_lines = ["module load " + module for module in modules]
    if executable:
        command = (
            "candidate="
            + shlex.quote(executable)
            + "; candidate=$(eval \"printf '%s' \\\"$candidate\\\"\"); "
            + "test -x \"$candidate\"; printf '%s\\n' \"$candidate\""
        )
    else:
        command = "command -v g16"
    try:
        completed = _run_login_shell(
            "\n".join(
                ["set -euo pipefail", "module purge", *module_lines, command]
            ),
            timeout=30,
        )
    except Exception as exc:
        return False, "", type(exc).__name__ + ": " + str(exc)
    resolved = (completed.stdout or "").strip().splitlines()
    resolved_path = resolved[-1] if resolved else ""
    if int(completed.returncode) != 0 or not resolved_path:
        detail = (completed.stderr or completed.stdout or "").strip()
        return False, "", "Gaussian module/path probe failed: " + detail[:500]
    return True, resolved_path, ""


def check_backends() -> BackendAvailability:
    sbatch = _which("sbatch")
    sacct = _which("sacct")
    squeue = _which("squeue")
    bc = _which("bc")
    profile_error = ""
    try:
        profile = require_cluster_profile()
        machine = profile.machine
        profile_ok = True
    except ClusterProfileError as exc:
        machine = active_machine() or ""
        profile_ok = False
        profile_error = str(exc)
    try:
        gauss_ok, gauss, gaussian_probe_error = _probe_gaussian_environment()
        aim = _from_config_or_path("aimall", "aimqb.ish", "aimqb")
        fer = _from_config_or_path("ferebus", "FEREBUS", "ferebus")
        python_executable = expanded_profile_value(
            "software", "python", "python_path", default=""
        ) or ""
    except Exception as exc:
        gauss_ok, gauss = False, ""
        gaussian_probe_error = type(exc).__name__ + ": " + str(exc)
        aim = ""
        fer = ""
        python_executable = ""
    try:
        runtime_modules = configured_daemon_runtime_modules()
        python_library_paths = configured_python_library_paths()
        missing_library_paths = [
            path for path in python_library_paths if not os.path.isdir(path)
        ]
        if missing_library_paths:
            raise ValueError(
                "configured Python library paths are not directories: "
                + repr(missing_library_paths)
            )
    except Exception as exc:
        runtime_modules = []
        python_library_paths = []
        batch_python = False
        batch_python_version = ""
        batch_python_error = "runtime module configuration invalid: " + str(exc)
        submitted_imports = {
            label: {"ok": False, "error": batch_python_error}
            for label in SUBMITTED_PYTHON_IMPORTS
        }
    else:
        (
            batch_python,
            batch_python_version,
            batch_python_error,
            submitted_imports,
        ) = _probe_configured_python_details(
            str(python_executable),
            runtime_modules,
            python_library_paths,
        )
    ariadne_status = submitted_imports["ariadne"]
    pyferebus_status = submitted_imports["pyferebus"]
    return BackendAvailability(
        profile=profile_ok,
        sbatch=bool(sbatch),
        sacct=bool(sacct),
        squeue=bool(squeue),
        gaussian=bool(gauss_ok),
        aimall=bool(aim),
        ferebus=bool(fer),
        ariadne=bool(batch_python and ariadne_status["ok"]),
        pyferebus=bool(batch_python and pyferebus_status["ok"]),
        bc=bool(bc),
        gaussian_binary=gauss,
        sbatch_path=sbatch,
        sacct_path=sacct,
        squeue_path=squeue,
        bc_path=bc,
        aimall_path=aim,
        ferebus_path=fer,
        active_profile=machine,
        profile_error=profile_error,
        python_executable=str(python_executable),
        batch_python=batch_python,
        batch_python_version=batch_python_version,
        batch_python_error=batch_python_error,
        batch_runtime_modules=tuple(runtime_modules),
        batch_python_library_paths=tuple(python_library_paths),
        gaussian_verified=bool(gauss_ok),
        gaussian_probe_error=str(gaussian_probe_error),
        ariadne_probe_error=str(ariadne_status.get("error") or ""),
        ariadne_abi_probe=(
            dict(ariadne_status["abi"])
            if isinstance(ariadne_status.get("abi"), dict)
            else None
        ),
        ariadne_runtime_probe=(
            dict(ariadne_status["runtime_smoke"])
            if isinstance(ariadne_status.get("runtime_smoke"), dict)
            else None
        ),
        pyferebus_probe_error=str(pyferebus_status.get("error") or ""),
    )


def missing_backend_message(avail: BackendAvailability) -> str:
    if avail.all_present:
        return ""
    profile = avail.active_profile or "<unresolved>"
    lines = [
        "The following configured Slurm backends are not available on PATH / PYTHONPATH:",
        "Active ICHOR profile: " + profile,
    ]
    if not avail.profile:
        lines.append(
            "  - profile. "
            + (
                avail.profile_error
                if avail.profile_error
                else "Set ICHOR_MACHINE to a top-level key in ~/ichor_config.yaml "
                "(for example ICHOR_MACHINE=csf3)."
            )
        )
    if not avail.sbatch:
        lines.append("  - sbatch (SLURM submit). Are you on a Slurm login node?")
    if not avail.sacct:
        lines.append("  - sacct (SLURM accounting).")
    if not avail.squeue:
        lines.append("  - squeue (SLURM liveness and throttled-array visibility).")
    if not avail.batch_python:
        lines.append(
            "  - configured batch Python (batch_python). "
            + (avail.batch_python_error or "The configured interpreter probe failed.")
        )
    if not avail.gaussian:
        lines.append(
            "  - Gaussian. Either g16 must be on PATH, or ~/ichor_config.yaml "
            "must declare <MACHINE>.software.gaussian.modules and "
            "executable_path. The module name is cluster-specific; do not load Gaussian on "
            "the login node; it is jobscript-only."
        )
    if not avail.aimall:
        lines.append(
            "  - aimqb.ish (AIMAll). Declare the path in "
            "~/ichor_config.yaml under <MACHINE>.software.aimall."
            "executable_path (default ~/AIMAll/aimqb.ish), "
            "or put aimqb.ish on PATH."
        )
    if not avail.ferebus:
        lines.append(
            "  - FEREBUS Fortran binary. Declare the path in "
            "~/ichor_config.yaml under <MACHINE>.software.ferebus."
            "executable_path (suggested $HOME/.local/bin/ferebus), "
            "or put a ferebus binary on PATH."
        )
    if not avail.ariadne:
        lines.append(
            "  - ariadne in the submitted Python environment. "
            + (
                avail.ariadne_probe_error
                or "Install ARIADNE into the configured batch venv."
            )
        )
    if not avail.pyferebus:
        lines.append(
            "  - pyferebus in the submitted Python environment. "
            + (
                avail.pyferebus_probe_error
                or "Install pyferebus into the configured batch venv."
            )
        )
    if not avail.bc:
        lines.append(
            "  - bc. pyferebus runFerebus.sh uses bc for the per-array random sleep."
        )
    lines.append("")
    lines.append(
        "Live mode requires all of the above. On Windows or off-cluster, "
        "create a separate campaign with `--mode dry_run` for the full "
        "file-system flow with stubbed backends."
    )
    
    return "\n".join(lines)




