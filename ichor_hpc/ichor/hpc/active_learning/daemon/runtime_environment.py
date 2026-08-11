"""Shared live-job module environment contract."""
from __future__ import annotations

import os
import re
import shlex
from typing import Any, List

from .cluster_profile import active_machine, profile_value


DEFAULT_DAEMON_PYTHON_MODULES: List[str] = [
    "python/3.11.3-gcccore-12.3.0",
]
DEFAULT_DAEMON_ARIADNE_RUNTIME_MODULES: List[str] = [
    "compilers/oneapi/2024.2.0",
    "compiler-rt tbb compiler",
    "mkl/2024.2",
]
DEFAULT_DAEMON_RUNTIME_MODULES: List[str] = (
    DEFAULT_DAEMON_PYTHON_MODULES + DEFAULT_DAEMON_ARIADNE_RUNTIME_MODULES
)

SUBMITTED_PYTHON_IMPORTS = {
    "ichor_core": ("ichor.core", "scipy"),
    "ichor_hpc": ("ichor.hpc",),
    "ariadne": ("ariadne",),
    "pyferebus": (
        "typing_extensions",
        "ichor.hpc.active_learning.daemon.ferebus_task_runner",
        "ichor.hpc.active_learning.daemon.ferebus_quality",
        "ichor.hpc.active_learning.daemon.ferebus_model_admission",
        "ichor.hpc.active_learning.daemon.ferebus_model_factors",
        "pyferebus.executors.trainer",
        "pyferebus.writers.config_file",
        "pyferebus.writers.commands_file",
        "pyferebus.writers.submission_script",
    ),
}

_MODULE_TOKEN_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]*(?: [A-Za-z0-9][A-Za-z0-9_.:/+-]*)*$"
)


def normalise_module_list(raw: Any, *, label: str) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError as exc:
            raise ValueError(
                "configured " + label + " modules must be a string or list"
            ) from exc
    modules: List[str] = []
    for value in values:
        module = str(value).strip()
        if not module:
            continue
        if any(character in module for character in "\r\n\x00"):
            raise ValueError(
                "configured " + label + " module contains control characters"
            )
        if not _MODULE_TOKEN_RE.fullmatch(module):
            raise ValueError(
                "configured "
                + label
                + " module contains unsafe characters: "
                + repr(module)
            )
        modules.append(module)
    return modules


def configured_daemon_runtime_modules() -> List[str]:
    python_modules = profile_value(
        "software", "python", "modules", default=None
    )
    ariadne_modules = profile_value(
        "software", "ariadne_runtime", "modules", default=None
    )
    machine = (active_machine() or "").lower()
    legacy_defaults = (not machine) or machine == "csf4"
    default_python = list(DEFAULT_DAEMON_PYTHON_MODULES) if legacy_defaults else []
    default_ariadne = (
        list(DEFAULT_DAEMON_ARIADNE_RUNTIME_MODULES) if legacy_defaults else []
    )
    return (
        normalise_module_list(python_modules, label="python")
        if python_modules is not None
        else default_python
    ) + (
        normalise_module_list(ariadne_modules, label="ariadne_runtime")
        if ariadne_modules is not None
        else default_ariadne
    )


def configured_python_library_paths() -> List[str]:
    """Return absolute loader paths required by the submitted interpreter."""
    raw = profile_value("software", "python", "library_path", default=None)
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError as exc:
            raise ValueError(
                "configured python library_path must be a string or list"
            ) from exc
    paths: List[str] = []
    for value in values:
        expanded = os.path.abspath(
            os.path.expanduser(os.path.expandvars(str(value).strip()))
        )
        if not str(value).strip():
            continue
        if any(character in expanded for character in "\r\n\x00"):
            raise ValueError(
                "configured python library_path contains control characters"
            )
        if not os.path.isabs(expanded):
            raise ValueError("configured python library_path must be absolute")
        if expanded not in paths:
            paths.append(expanded)
    return paths


def configured_python_isolation_lines() -> List[str]:
    """Keep CSF4 EasyBuild package paths from shadowing the configured venv."""
    if (active_machine() or "").lower() != "csf4":
        return []
    return [
        "unset PYTHONPATH PYTHONHOME",
        "export PYTHONNOUSERSITE=1",
    ]


def python_library_path_export_lines(paths: List[str]) -> List[str]:
    """Render deterministic loader-path exports before the first Python call."""
    if not paths:
        return []
    prefix = ":".join(paths)
    return [
        "export LD_LIBRARY_PATH="
        + shlex.quote(prefix)
        + '${LD_LIBRARY_PATH:+:"$LD_LIBRARY_PATH"}'
    ]


def ariadne_runtime_command_prefix() -> str:
    """Scope the ffluxlab MKL preload to processes which import ARIADNE."""
    if (active_machine() or "").lower() != "ffluxlab":
        return ""
    return (
        'env LD_PRELOAD="$ICHOR_ARIADNE_LD_PRELOAD'
        '${LD_PRELOAD:+:$LD_PRELOAD}" '
    )


def native_runtime_setup_lines() -> List[str]:
    """Render machine-specific native runtime discovery for submitted jobs."""
    if (active_machine() or "").lower() != "ffluxlab":
        return []
    return [
        'ICHOR_GCC_ROOT="$(readlink -f /home/modules/compilers/gcc/11.1.0)"',
        'ICHOR_GCC_LIBSTDCXX="$ICHOR_GCC_ROOT/lib64/libstdc++.so.6"',
        (
            '[ -e "$ICHOR_GCC_LIBSTDCXX" ] || '
            "{ echo 'GCC 11 libstdc++ runtime is unavailable' >&2; exit 1; }"
        ),
        'ICHOR_INTEL_ROOT="$(readlink -f /home/modules/compilers/intel/21.0.3)"',
        (
            'ICHOR_INTEL_LIBIMF="$(find "$ICHOR_INTEL_ROOT" '
            "\\( -type f -o -type l \\) "
            "-path '*/intel64_lin/libimf.so' -print -quit)\""
        ),
        (
            '[ -n "$ICHOR_INTEL_LIBIMF" ] || '
            "{ echo '64-bit Intel runtime is unavailable' >&2; exit 1; }"
        ),
        (
            'ICHOR_MKL_LP64="${MKLROOT:+$MKLROOT/lib/intel64/'
            'libmkl_intel_lp64.so.1}"'
        ),
        'if [ -z "$ICHOR_MKL_LP64" ] || [ ! -e "$ICHOR_MKL_LP64" ]; then',
        (
            '  ICHOR_MKL_LP64="$(find "$ICHOR_INTEL_ROOT" '
            "\\( -type f -o -type l \\) "
            "-path '*/mkl/*/lib/intel64/libmkl_intel_lp64.so.1' "
            "-print -quit)\""
        ),
        "fi",
        (
            '[ -n "$ICHOR_MKL_LP64" ] || '
            "{ echo '64-bit Intel MKL runtime is unavailable' >&2; exit 1; }"
        ),
        'ICHOR_GCC_RUNTIME_DIR="$(dirname "$ICHOR_GCC_LIBSTDCXX")"',
        'ICHOR_INTEL_RUNTIME_DIR="$(dirname "$ICHOR_INTEL_LIBIMF")"',
        'ICHOR_MKL_RUNTIME_DIR="$(dirname "$ICHOR_MKL_LP64")"',
        'ICHOR_MKL_SEQUENTIAL="$ICHOR_MKL_RUNTIME_DIR/libmkl_sequential.so.1"',
        'ICHOR_MKL_CORE="$ICHOR_MKL_RUNTIME_DIR/libmkl_core.so.1"',
        (
            '[ -e "$ICHOR_MKL_SEQUENTIAL" ] || '
            "{ echo 'Intel MKL sequential runtime is unavailable' >&2; exit 1; }"
        ),
        (
            '[ -e "$ICHOR_MKL_CORE" ] || '
            "{ echo 'Intel MKL core runtime is unavailable' >&2; exit 1; }"
        ),
        (
            'export ICHOR_ARIADNE_LD_PRELOAD="$ICHOR_MKL_LP64'
            ':$ICHOR_MKL_SEQUENTIAL:$ICHOR_MKL_CORE"'
        ),
        (
            'export LD_LIBRARY_PATH="$ICHOR_GCC_RUNTIME_DIR'
            ':$ICHOR_INTEL_RUNTIME_DIR'
            ':$ICHOR_MKL_RUNTIME_DIR'
            '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
        ),
        (
            'export LIBRARY_PATH="$ICHOR_GCC_RUNTIME_DIR'
            ':$ICHOR_INTEL_RUNTIME_DIR'
            ':$ICHOR_MKL_RUNTIME_DIR'
            '${LIBRARY_PATH:+:$LIBRARY_PATH}"'
        ),
    ]


def module_initialisation_lines() -> List[str]:
    """Make the Environment Modules shell function available in batch jobs."""
    return [
        "if ! type module >/dev/null 2>&1; then",
        "  for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do",
        '    if [ -r "$init" ]; then . "$init"; break; fi',
        "  done",
        "fi",
        "type module >/dev/null 2>&1 || { echo 'environment modules are unavailable' >&2; exit 1; }",
    ]


def module_load_lines(modules: List[str]) -> List[str]:
    return ["module load " + module for module in modules]


__all__ = [
    "DEFAULT_DAEMON_ARIADNE_RUNTIME_MODULES",
    "DEFAULT_DAEMON_PYTHON_MODULES",
    "DEFAULT_DAEMON_RUNTIME_MODULES",
    "SUBMITTED_PYTHON_IMPORTS",
    "ariadne_runtime_command_prefix",
    "configured_daemon_runtime_modules",
    "configured_python_isolation_lines",
    "configured_python_library_paths",
    "module_initialisation_lines",
    "module_load_lines",
    "native_runtime_setup_lines",
    "normalise_module_list",
    "python_library_path_export_lines",
]
