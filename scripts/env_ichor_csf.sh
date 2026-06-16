#!/usr/bin/env bash
#
# Source this helper to enter the ICHOR CSF runtime environment.
#
# It intentionally does not install packages or edit configuration. It only
# prepares the current shell with the module stack, venv, and runtime library
# paths needed by ichor-cli, ichor-al-daemon, ARIADNE, and PLUMED.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cat >&2 <<'EOF'
This script must be sourced, not executed:

  source scripts/env_ichor_csf.sh csf3
  source scripts/env_ichor_csf.sh csf4
EOF
    exit 1
fi

_ichor_env_usage() {
    cat <<'EOF'
Usage:
  source scripts/env_ichor_csf.sh csf3|csf4 [options]

Options:
  --venv PATH             Override venv path
  --python-prefix PATH    CSF3 private Python prefix, default ~/opt/python-3.11.15
  --plumed-prefix PATH    PLUMED prefix, default ~/opt/plumed-2.10.0
  --no-purge              Do not module purge before loading runtime modules
  --smoke                 Run fuller xTB/RDKit/daemon import checks
  --quiet                 Print only errors
  -h, --help              Show this help
EOF
}

_ichor_env_error() {
    echo "ERROR: $*" >&2
}

_ichor_env_note() {
    if [[ "${ICHOR_ENV_QUIET:-0}" -eq 0 ]]; then
        echo "$*"
    fi
}

_ichor_env_expand_path() {
    local value="$1"
    value="${value/#\~/${HOME}}"
    printf '%s\n' "${value}"
}

_ichor_env_initialise_modules() {
    if command -v module >/dev/null 2>&1; then
        return 0
    fi
    # shellcheck disable=SC1091
    [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true
    # shellcheck disable=SC1091
    [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash || true
}

_ichor_env_module() {
    _ichor_env_initialise_modules
    if ! command -v module >/dev/null 2>&1; then
        _ichor_env_error "module command is unavailable in this shell"
        return 1
    fi
    module "$@"
}

_ichor_env_import_check() {
    local label="$1"
    local code="$2"
    if python -c "${code}" >/tmp/ichor_env_check.$$ 2>&1; then
        _ichor_env_note "${label} OK"
        rm -f /tmp/ichor_env_check.$$
        return 0
    fi
    cat /tmp/ichor_env_check.$$ >&2
    rm -f /tmp/ichor_env_check.$$
    _ichor_env_error "${label} check failed"
    return 1
}

_ichor_env_main() {
    local machine="${1:-}"
    if [[ -z "${machine}" || "${machine}" == "-h" || "${machine}" == "--help" ]]; then
        _ichor_env_usage
        return 0
    fi
    shift || true

    local do_purge=1
    local do_smoke=0
    local python_prefix="${HOME}/opt/python-3.11.15"
    local plumed_prefix="${HOME}/opt/plumed-2.10.0"
    local venv=""
    ICHOR_ENV_QUIET=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --venv)
                venv="${2:?missing value for --venv}"
                shift 2
                ;;
            --python-prefix)
                python_prefix="${2:?missing value for --python-prefix}"
                shift 2
                ;;
            --plumed-prefix)
                plumed_prefix="${2:?missing value for --plumed-prefix}"
                shift 2
                ;;
            --no-purge)
                do_purge=0
                shift
                ;;
            --smoke)
                do_smoke=1
                shift
                ;;
            --quiet)
                ICHOR_ENV_QUIET=1
                shift
                ;;
            -h|--help)
                _ichor_env_usage
                return 0
                ;;
            *)
                _ichor_env_error "unrecognised option: $1"
                _ichor_env_usage >&2
                return 2
                ;;
        esac
    done

    case "${machine}" in
        csf3|csf4) ;;
        *)
            _ichor_env_error "machine must be csf3 or csf4"
            _ichor_env_usage >&2
            return 2
            ;;
    esac

    python_prefix="$(_ichor_env_expand_path "${python_prefix}")"
    plumed_prefix="$(_ichor_env_expand_path "${plumed_prefix}")"
    if [[ -z "${venv}" ]]; then
        venv="${HOME}/.venv/ichor-${machine}"
    fi
    venv="$(_ichor_env_expand_path "${venv}")"

    if [[ "${do_purge}" -eq 1 ]]; then
        _ichor_env_module purge || return 1
    fi

    if [[ "${machine}" == "csf3" ]]; then
        _ichor_env_module load compilers/intel/oneapi/2025.0.1 || return 1
        _ichor_env_module load umf compiler-rt tbb compiler || return 1
        _ichor_env_module load mkl/2025.0 || return 1
    else
        _ichor_env_module load python/3.11.3-gcccore-12.3.0 || return 1
        _ichor_env_module load python-bundle-pypi/2023.06-gcccore-12.3.0 || return 1
        _ichor_env_module load compilers/oneapi/2024.2.0 || return 1
        _ichor_env_module load compiler-rt tbb compiler || return 1
        _ichor_env_module load mkl/2024.2 || return 1
    fi

    unset CC CXX FC F77 F90
    export ICHOR_MACHINE="${machine}"
    export PLUMED_KERNEL="${plumed_prefix}/lib/libplumedKernel.so"

    if [[ "${machine}" == "csf3" ]]; then
        export LD_LIBRARY_PATH="${python_prefix}/lib:${plumed_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    else
        export LD_LIBRARY_PATH="${plumed_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi

    if [[ ! -f "${venv}/bin/activate" ]]; then
        _ichor_env_error "venv activation script not found: ${venv}/bin/activate"
        return 1
    fi
    # shellcheck disable=SC1091
    source "${venv}/bin/activate" || return 1
    hash -r 2>/dev/null || true

    _ichor_env_note "ICHOR runtime environment ready for ${machine}"
    if [[ "${ICHOR_ENV_QUIET}" -eq 0 ]]; then
        which python
        which ichor-cli || true
        which ichor-al-daemon || true
        python -V
        echo "ICHOR_MACHINE=${ICHOR_MACHINE}"
        echo "PLUMED_KERNEL=${PLUMED_KERNEL}"
    fi

    _ichor_env_import_check "ARIADNE" "import ariadne" || return 1
    _ichor_env_import_check "PLUMED" "import os, plumed; p=plumed.Plumed(kernel=os.environ['PLUMED_KERNEL']); p.finalize()" || return 1

    if [[ "${do_smoke}" -eq 1 ]]; then
        _ichor_env_import_check "ICHOR packages" "import ichor.core, ichor.hpc, ichor.cli" || return 1
        _ichor_env_import_check "POLUS RS" "import polus.samplers.RS.randomSampling" || return 1
        _ichor_env_import_check "pyferebus" "import pyferebus.executors.trainer" || return 1
        _ichor_env_import_check "RDKit" "from rdkit import Chem" || return 1
        _ichor_env_import_check "xTB" "from xtb.ase.calculator import XTB" || return 1
        _ichor_env_import_check "xTB runtime smoke" "from ichor.hpc.runtime_preflight import ensure_xtb_ase_available; ensure_xtb_ase_available(run_energy=True)" || return 1
        _ichor_env_import_check "PLUMED runtime smoke" "from ichor.hpc.runtime_preflight import ensure_plumed_available; ensure_plumed_available(run_ase_smoke=True)" || return 1
    fi
}

_ichor_env_main "$@"
_ichor_env_rc=$?
unset -f _ichor_env_usage _ichor_env_error _ichor_env_note _ichor_env_expand_path
unset -f _ichor_env_initialise_modules _ichor_env_module _ichor_env_import_check
unset -f _ichor_env_main
unset ICHOR_ENV_QUIET
return "${_ichor_env_rc}"
