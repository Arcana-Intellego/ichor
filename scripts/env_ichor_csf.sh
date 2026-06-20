#!/usr/bin/env bash
#
# Source this helper to enter the ICHOR CSF runtime environment.
#
# It does not install packages or edit configuration. It prepares the current
# shell with the module stack, venv, and runtime library paths needed by
# ichor-cli, ichor-al-daemon, ARIADNE, and PLUMED. Import checks are opt-in via
# --smoke.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cat >&2 <<'EOF'
This script must be sourced, not executed:

  source scripts/env_ichor_csf.sh csf3
  source scripts/env_ichor_csf.sh csf4
  source scripts/env_ichor_csf.sh auto
  source scripts/env_ichor_csf.sh --machine csf3
EOF
    exit 1
fi

_ichor_env_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_ichor_env_script_dir}/lib_ichor_csf.sh"

_ichor_env_usage() {
    cat <<'EOF'
Usage:
  source scripts/env_ichor_csf.sh auto|csf3|csf4 [options]
  source scripts/env_ichor_csf.sh --machine auto|csf3|csf4 [options]

Options:
  --machine auto|csf3|csf4  Select machine explicitly
  --venv PATH               Override venv path
  --python-prefix PATH      CSF3 private Python prefix, default ~/opt/python-3.11.15
  --plumed-prefix PATH      PLUMED prefix, default ~/opt/plumed-2.10.0
  --no-purge                Do not module purge before loading runtime modules
  --env-check               Module/venv/path checks only; no scientific imports
  --smoke                   Run import/path smoke checks
  --smoke-heavy             Also run xTB energy and ASE+PLUMED smokes
  --print-env               Print resolved runtime environment details
  --debug, --trace          Enable shell tracing while this helper runs
  --quiet                   Print only errors
  -h, --help                Show this help
EOF
}

_ichor_env_error() {
    ichor_csf_error "$@"
}

_ichor_env_note() {
    if [[ "${ICHOR_ENV_QUIET:-0}" -eq 0 ]]; then
        echo "$*"
    fi
}

_ichor_env_module() {
    if ! ichor_csf_module "$@"; then
        return 1
    fi
}

_ichor_env_import_check() {
    local label="$1"
    local code="$2"
    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/ichor_env_check.XXXXXX")" || return 1
    if python -c "${code}" >"${tmp}" 2>&1; then
        _ichor_env_note "${label} OK"
        rm -f "${tmp}"
        return 0
    fi
    cat "${tmp}" >&2
    rm -f "${tmp}"
    _ichor_env_error "${label} check failed"
    return 1
}

_ichor_env_print_env() {
    echo "ICHOR_MACHINE=${ICHOR_MACHINE:-}"
    echo "VIRTUAL_ENV=${VIRTUAL_ENV:-}"
    echo "python=$(command -v python || true)"
    echo "ichor-cli=$(command -v ichor-cli || true)"
    echo "ichor-al-daemon=$(command -v ichor-al-daemon || true)"
    echo "PLUMED_KERNEL=${PLUMED_KERNEL:-}"
    echo "PLUMED_LIBRARY_PATH=${PLUMED_LIBRARY_PATH:-}"
    echo "module type=$(type -t module 2>/dev/null || echo unavailable)"
    echo "module init=${ICHOR_CSF_MODULE_INIT:-unresolved}"
    echo "PATH:"
    local i=0
    local part
    local -a path_parts
    IFS=':' read -r -a path_parts <<< "${PATH:-}"
    for part in "${path_parts[@]}"; do
        echo "  ${part}"
        i=$((i + 1))
        if [[ "${i}" -ge 12 ]]; then
            break
        fi
    done
    echo "LD_LIBRARY_PATH:"
    IFS=':' read -r -a path_parts <<< "${LD_LIBRARY_PATH:-}"
    for part in "${path_parts[@]}"; do
        if [[ -n "${part}" ]]; then
            echo "  ${part}"
        fi
    done
}

_ichor_env_load_runtime_modules() {
    local machine="$1"
    local do_purge="$2"
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
}

_ichor_env_check_venv_ownership() {
    local venv="$1"
    local cmd resolved
    for cmd in python ichor-cli ichor-al-daemon; do
        resolved="$(command -v "${cmd}" 2>/dev/null || true)"
        if [[ -z "${resolved}" ]]; then
            _ichor_env_error "${cmd} is not on PATH after activating ${venv}"
            return 1
        fi
        if ! ichor_csf_path_inside "${resolved}" "${venv}"; then
            ichor_csf_warn "${cmd} resolves outside target venv: ${resolved}"
        fi
    done
}

_ichor_env_main() {
    local machine=""
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        _ichor_env_usage
        return 0
    fi
    if [[ "${1:-}" == "--machine" ]]; then
        machine="${2:?missing value for --machine}"
        shift 2
    elif [[ -n "${1:-}" && "${1:-}" != --* ]]; then
        machine="$1"
        shift
    else
        machine="auto"
    fi

    local do_purge=1
    local do_env_check=0
    local do_smoke=0
    local do_smoke_heavy=0
    local do_print_env=0
    local debug_trace=0
    local python_prefix="${HOME}/opt/python-3.11.15"
    local plumed_prefix="${HOME}/opt/plumed-2.10.0"
    local venv=""
    ICHOR_ENV_QUIET=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --machine)
                machine="${2:?missing value for --machine}"
                shift 2
                ;;
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
            --env-check)
                do_env_check=1
                shift
                ;;
            --smoke)
                do_smoke=1
                shift
                ;;
            --smoke-heavy)
                do_smoke=1
                do_smoke_heavy=1
                shift
                ;;
            --print-env)
                do_print_env=1
                shift
                ;;
            --debug|--trace)
                debug_trace=1
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
        auto|csf3|csf4) ;;
        *)
            _ichor_env_error "machine must be auto, csf3, or csf4"
            _ichor_env_usage >&2
            return 2
            ;;
    esac
    machine="$(ichor_csf_detect_machine "${machine}")" || return 2

    if [[ "${debug_trace}" -eq 1 ]]; then
        export PS4='+ env_ichor_csf.sh:${LINENO}:${FUNCNAME[0]:-main}: '
        set -x
    fi

    python_prefix="$(ichor_csf_expand_path "${python_prefix}")"
    plumed_prefix="$(ichor_csf_expand_path "${plumed_prefix}")"
    if [[ -z "${venv}" ]]; then
        venv="${HOME}/.venv/ichor-${machine}"
    fi
    venv="$(ichor_csf_expand_path "${venv}")"

    ichor_csf_deactivate_existing_venv "loading CSF modules" || return 1
    _ichor_env_load_runtime_modules "${machine}" "${do_purge}" || return 1

    unset CC CXX FC F77 F90
    export ICHOR_MACHINE="${machine}"
    export PLUMED_KERNEL="${plumed_prefix}/lib/libplumedKernel.so"
    export PLUMED_LIBRARY_PATH="${plumed_prefix}/lib"

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

    ichor_csf_warn_path_hazards "${venv}"
    _ichor_env_check_venv_ownership "${venv}" || return 1

    _ichor_env_note "ICHOR runtime environment ready for ${machine}"
    if [[ "${ICHOR_ENV_QUIET}" -eq 0 ]]; then
        which python
        which ichor-cli || true
        which ichor-al-daemon || true
        python -V
        echo "ICHOR_MACHINE=${ICHOR_MACHINE}"
        echo "PLUMED_KERNEL=${PLUMED_KERNEL}"
    fi

    if [[ "${do_print_env}" -eq 1 || "${do_env_check}" -eq 1 ]]; then
        _ichor_env_print_env
    fi

    if [[ "${do_smoke}" -eq 1 ]]; then
        _ichor_env_import_check "ARIADNE" "import ariadne; assert hasattr(ariadne, 'Geometric_Trqn') or hasattr(ariadne, 'Ds_Optimiser')" || return 1
        _ichor_env_import_check "PLUMED" "import os, plumed; p=plumed.Plumed(kernel=os.environ['PLUMED_KERNEL']); p.finalize()" || return 1
        _ichor_env_import_check "ICHOR packages" "import ichor.core, ichor.hpc, ichor.cli" || return 1
        _ichor_env_import_check "POLUS RS" "import polus.samplers.RS.randomSampling" || return 1
        _ichor_env_import_check "pyferebus" "import pyferebus.executors.trainer" || return 1
        _ichor_env_import_check "RDKit" "from rdkit import Chem" || return 1
        _ichor_env_import_check "xTB" "from xtb.ase.calculator import XTB" || return 1
    fi

    if [[ "${do_smoke_heavy}" -eq 1 ]]; then
        _ichor_env_import_check "xTB runtime smoke" "from ichor.hpc.runtime_preflight import ensure_xtb_ase_available; ensure_xtb_ase_available(run_energy=True)" || return 1
        _ichor_env_import_check "PLUMED runtime smoke" "from ichor.hpc.runtime_preflight import ensure_plumed_available; ensure_plumed_available(run_ase_smoke=True)" || return 1
    fi
}

_ichor_env_main "$@"
_ichor_env_rc=$?
unset -f _ichor_env_usage _ichor_env_error _ichor_env_note _ichor_env_module
unset -f _ichor_env_import_check _ichor_env_print_env _ichor_env_load_runtime_modules
unset -f _ichor_env_check_venv_ownership _ichor_env_main
unset ICHOR_ENV_QUIET _ichor_env_script_dir
return "${_ichor_env_rc}"
