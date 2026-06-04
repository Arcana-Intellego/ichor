"""Backend availability detector.

Probes the host for the binaries / Python modules that the
LiveBackendsPhaseExecutor requires before it can drive a real campaign:

    * sbatch     -- SLURM submit
    * sacct      -- SLURM accounting (also needed by the daemon poll loop)
    * g16        -- Gaussian; required by the generated CSF4 sbatch scripts
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
import shutil
from dataclasses import dataclass
from typing import List


__all__ = [
    "BackendAvailability",
    "check_backends",
    "missing_backend_message",
]


@dataclass(frozen=True)
class BackendAvailability:
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

    @property
    def all_present(self) -> bool:
        return (
            self.sbatch and self.sacct and self.gaussian and
            self.aimall and self.ferebus and self.ariadne and
            self.polus_rs and self.pyferebus and self.bc
        )

    @property
    def missing(self) -> List[str]:
        out: List[str] = []
        for attr in (
            "sbatch", "sacct", "gaussian", "aimall", "ferebus",
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
    try:
        from ichor.hpc.global_variables import ICHOR_CONFIG, MACHINE, get_param_from_config
    except Exception:
        ICHOR_CONFIG = None
        MACHINE = None
        get_param_from_config = None
    machine = MACHINE
    if ICHOR_CONFIG and not machine and "csf4" in ICHOR_CONFIG:
        machine = "csf4"
    if ICHOR_CONFIG and machine and get_param_from_config:
        raw = get_param_from_config(
            ICHOR_CONFIG, machine, "software", backend_name,
            "executable_path", default=None,
        )
        if raw:
            expanded = os.path.expanduser(os.path.expandvars(str(raw)))
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

    On CSF4 the Gaussian module is jobscript-only and refuses to load on login
    nodes. That is still a valid live configuration as long as ichor_config.yaml
    declares both the Gaussian module and executable path for the generated
    SLURM scripts.
    """
    found = _which("g16")
    if found:
        return found

    try:
        from ichor.hpc.global_variables import ICHOR_CONFIG, MACHINE, get_param_from_config
    except Exception:
        return ""

    if not ICHOR_CONFIG:
        return ""

    machine = MACHINE
    if not machine and "csf4" in ICHOR_CONFIG:
        machine = "csf4"
    if not machine:
        return ""

    modules = get_param_from_config(
        ICHOR_CONFIG, machine, "software", "gaussian", "modules", default=None
    )
    executable = get_param_from_config(
        ICHOR_CONFIG, machine, "software", "gaussian", "executable_path",
        default=None,
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
        importlib.import_module("polus.samplers.RS.randomSampling")
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


def check_backends() -> BackendAvailability:
    sbatch = _which("sbatch")
    sacct = _which("sacct")
    bc = _which("bc")
    gauss = _gaussian_binary()
    aim = _from_config_or_path("aimall", "aimqb.ish", "aimqb")
    fer = _from_config_or_path("ferebus", "FEREBUS", "ferebus")
    return BackendAvailability(
        sbatch=bool(sbatch),
        sacct=bool(sacct),
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
        bc_path=bc,
        aimall_path=aim,
        ferebus_path=fer,
    )


def missing_backend_message(avail: BackendAvailability) -> str:
    if avail.all_present:
        return ""
    lines = ["The following backends are not available on PATH / PYTHONPATH:"]
    if not avail.sbatch:
        lines.append("  - sbatch (SLURM submit). Are you on a CSF4 login node?")
    if not avail.sacct:
        lines.append("  - sacct (SLURM accounting).")
    if not avail.gaussian:
        lines.append(
            "  - Gaussian. Either g16 must be on PATH, or ~/ichor_config.yaml "
            "must declare <MACHINE>.software.gaussian.modules and "
            "executable_path. On CSF4 this is usually "
            "gaussian/g16c01_em64t_detectcpu, but do not load Gaussian on "
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




