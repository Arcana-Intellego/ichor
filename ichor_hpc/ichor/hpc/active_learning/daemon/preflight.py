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
detector to refuse the "--live" mode with a clear error rather than running
a daemon that cannot submit anything.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List

from .import_utils import quiet_import_module
from .cluster_profile import (
    ClusterProfileError,
    active_machine,
    expanded_profile_value,
    profile_value,
    require_cluster_profile,
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
    polus_rs: bool
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

    @property
    def all_present(self) -> bool:
        return (
            self.profile and self.sbatch and self.sacct and self.squeue and
            self.batch_python and self.gaussian and
            self.aimall and self.ferebus and self.ariadne and
            self.polus_rs and self.pyferebus and self.bc
        )

    @property
    def missing(self) -> List[str]:
        out: List[str] = []
        for attr in (
            "profile", "sbatch", "sacct", "squeue", "batch_python",
            "gaussian", "aimall", "ferebus",
            "ariadne", "polus_rs", "pyferebus", "bc",
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


def _ariadne_importable() -> bool:
    try:
        importlib.import_module("ariadne")
    except Exception:
        return False
    return True


def _polus_rs_importable() -> bool:
    # the FEREBUS-dataset split only needs the dependency-light RS sampler
    # subtree, not the full polus package (managers/descriptors drag in
    # sklearn/torch). probe exactly what we import.
    try:
        quiet_import_module("polus.samplers.RS.randomSampling")
    except Exception:
        return False
    return True


def _pyferebus_importable() -> bool:
    try:
        importlib.import_module("pyferebus.executors.trainer")
        importlib.import_module("pyferebus.writers.config_file")
        importlib.import_module("pyferebus.writers.commands_file")
        importlib.import_module("pyferebus.writers.submission_script")
    except Exception:
        return False
    return True


def _probe_configured_python(python_executable: str) -> tuple[bool, str, str]:
    """Run the submitted-job import contract under the configured interpreter."""
    executable = str(python_executable or "").strip()
    if not executable:
        return False, "", "software.python.python_path is not configured"
    if not os.path.isfile(executable):
        return False, "", "configured Python is not a file: " + executable
    if not os.access(executable, os.X_OK):
        return False, "", "configured Python is not executable: " + executable
    probe = (
        "import importlib,json,sys;"
        "mods=['ichor.core','ichor.hpc','ariadne',"
        "'polus.samplers.RS.randomSampling',"
        "'pyferebus.executors.trainer'];"
        "[importlib.import_module(name) for name in mods];"
        "print(json.dumps({'executable':sys.executable,"
        "'version':list(sys.version_info[:3])},sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return False, "", type(exc).__name__ + ": " + str(exc)
    if int(completed.returncode) != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return False, "", "configured Python import probe failed: " + detail[:500]
    try:
        payload = json.loads((completed.stdout or "").strip().splitlines()[-1])
        version_values = tuple(int(value) for value in payload["version"])
    except Exception as exc:
        return False, "", "configured Python returned malformed probe output: " + str(exc)
    version = ".".join(str(value) for value in version_values)
    if version_values[:2] != (3, 11):
        return False, version, "configured batch Python must be Python 3.11"
    return True, version, ""


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
    gauss = _gaussian_binary()
    aim = _from_config_or_path("aimall", "aimqb.ish", "aimqb")
    fer = _from_config_or_path("ferebus", "FEREBUS", "ferebus")
    python_executable = expanded_profile_value(
        "software", "python", "python_path", default=""
    ) or ""
    batch_python, batch_python_version, batch_python_error = (
        _probe_configured_python(str(python_executable))
    )
    return BackendAvailability(
        profile=profile_ok,
        sbatch=bool(sbatch),
        sacct=bool(sacct),
        squeue=bool(squeue),
        gaussian=bool(gauss),
        aimall=bool(aim),
        ferebus=bool(fer),
        ariadne=_ariadne_importable(),
        polus_rs=_polus_rs_importable(),
        pyferebus=_pyferebus_importable(),
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
            "  - ariadne (Python). pip install ariadne into the active venv."
        )
    if not avail.polus_rs:
        lines.append(
            "  - polus_rs: polus.samplers.RS.randomSampling (Python). Install POLUS in the active venv."
        )
    if not avail.pyferebus:
        lines.append(
            "  - pyferebus (Python). Install the local pyferebus package in the active venv."
        )
    if not avail.bc:
        lines.append(
            "  - bc. pyferebus runFerebus.sh uses bc for the per-array random sleep."
        )
    lines.append("")
    lines.append(
        "Live mode requires all of the above. On Windows / off-cluster, use "
        "`--dry-run` (full file-system flow with stubbed backends) or "
        "`--mock-ariadne` (pure state-machine progression)."
    )
    
    return "\n".join(lines)




