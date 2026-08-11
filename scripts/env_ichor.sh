#!/usr/bin/env bash
#
# Source this helper to enter the ICHOR runtime environment.
#
# It does not install packages or edit configuration. It prepares the current
# shell with the module stack, venv, and runtime library paths needed by
# ichor-cli, ichor-al-daemon, ARIADNE, and PLUMED. Import checks are opt-in via
# --smoke.
#
#   source scripts/env_ichor.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' \
        "This script must be sourced, not executed:" \
        "" \
        "  source scripts/env_ichor.sh" \
        "  source scripts/env_ichor.sh --machine csf3" \
        "  source scripts/env_ichor.sh --machine csf4" \
        "  source scripts/env_ichor.sh --machine ffluxlab" >&2
    exit 1
fi

_ichor_env_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_ichor_env_script_dir}/lib_ichor.sh"

_ichor_env_usage() {
    cat <<'EOF'
Usage:
  source scripts/env_ichor.sh [options]
  source scripts/env_ichor.sh --machine csf3|csf4|ffluxlab [options]
  source scripts/env_ichor.sh auto|csf3|csf4|ffluxlab [options]

Options:
  --machine csf3|csf4|ffluxlab
                             Override automatic machine detection
  --venv PATH               Override venv path
  --python-prefix PATH      CSF3/ffluxlab private Python prefix,
                             default ~/opt/python-3.11.15
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


_ichor_env_prepend_ld_library_paths_once() {
    local -a prefixes=("$@")
    local -a existing_parts
    local prefix part skip
    local result=""

    IFS=':' read -r -a existing_parts <<< "${LD_LIBRARY_PATH:-}"
    for prefix in "${prefixes[@]}"; do
        if [[ -z "${prefix}" ]]; then
            continue
        fi
        result="${result:+${result}:}${prefix}"
    done
    for part in "${existing_parts[@]}"; do
        if [[ -z "${part}" ]]; then
            continue
        fi
        skip=0
        for prefix in "${prefixes[@]}"; do
            if [[ "${part}" == "${prefix}" ]]; then
                skip=1
                break
            fi
        done
        if [[ "${skip}" -eq 0 ]]; then
            result="${result:+${result}:}${part}"
        fi
    done
    export LD_LIBRARY_PATH="${result}"
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


_ichor_env_ariadne_check() {
    local label="$1"
    local code="$2"
    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/ichor_env_ariadne_check.XXXXXX")" || return 1
    if [[ "${ICHOR_MACHINE:-}" == "ffluxlab" ]]; then
        if [[ -z "${ICHOR_ARIADNE_LD_PRELOAD:-}" ]]; then
            rm -f "${tmp}"
            _ichor_env_error "ARIADNE MKL preload contract is unavailable"
            return 1
        fi
        if env \
            LD_PRELOAD="${ICHOR_ARIADNE_LD_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}" \
            python -c "${code}" >"${tmp}" 2>&1; then
            _ichor_env_note "${label} OK"
            rm -f "${tmp}"
            return 0
        fi
    elif python -c "${code}" >"${tmp}" 2>&1; then
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
    echo "PYTHONPATH=${PYTHONPATH:-}"
    echo "PYTHONHOME=${PYTHONHOME:-}"
    echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-}"
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
    elif [[ "${machine}" == "csf4" ]]; then
        _ichor_env_module load python/3.11.3-gcccore-12.3.0 || return 1
        _ichor_env_module load compilers/oneapi/2024.2.0 || return 1
        _ichor_env_module load compiler-rt tbb compiler || return 1
        _ichor_env_module load mkl/2024.2 || return 1
    else
        _ichor_env_module load compilers/intel/21.0.3 || return 1
    fi
}

_ichor_env_ffluxlab_intel_runtime() {
    local configured_root="/home/modules/compilers/intel/21.0.3"
    local intel_root libimf libmkl libmkl_sequential libmkl_core
    local runtime_dir mkl_runtime_dir
    intel_root="$(readlink -f "${configured_root}" 2>/dev/null || true)"
    if [[ -z "${intel_root}" || ! -d "${intel_root}" ]]; then
        _ichor_env_error "ffluxlab Intel module root is unavailable: ${configured_root}"
        return 1
    fi
    libimf="$(
        find "${intel_root}" \
            \( -type f -o -type l \) \
            -path '*/intel64_lin/libimf.so' -print -quit
    )"
    if [[ -z "${libimf}" ]]; then
        _ichor_env_error "64-bit libimf.so is absent beneath ${intel_root}"
        return 1
    fi
    libmkl="${MKLROOT:+${MKLROOT}/lib/intel64/libmkl_intel_lp64.so.1}"
    if [[ -z "${libmkl}" || ! -e "${libmkl}" ]]; then
        libmkl="$(
            find "${intel_root}" \
                \( -type f -o -type l \) \
                -path '*/mkl/*/lib/intel64/libmkl_intel_lp64.so.1' \
                -print -quit
        )"
    fi
    if [[ -z "${libmkl}" ]]; then
        _ichor_env_error \
            "64-bit libmkl_intel_lp64.so.1 is absent beneath ${intel_root}"
        return 1
    fi
    runtime_dir="$(dirname "${libimf}")"
    mkl_runtime_dir="$(dirname "${libmkl}")"
    libmkl_sequential="${mkl_runtime_dir}/libmkl_sequential.so.1"
    libmkl_core="${mkl_runtime_dir}/libmkl_core.so.1"
    if [[ ! -e "${libmkl_sequential}" || ! -e "${libmkl_core}" ]]; then
        _ichor_env_error \
            "complete Intel MKL runtime is unavailable beneath ${mkl_runtime_dir}"
        return 1
    fi
    _ichor_env_prepend_ld_library_paths_once \
        "${runtime_dir}" \
        "${mkl_runtime_dir}"
    export LIBRARY_PATH="${runtime_dir}:${mkl_runtime_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export ICHOR_INTEL_RUNTIME_DIR="${runtime_dir}"
    export ICHOR_MKL_RUNTIME_DIR="${mkl_runtime_dir}"
    export ICHOR_ARIADNE_LD_PRELOAD="${libmkl}:${libmkl_sequential}:${libmkl_core}"
}

_ichor_env_ffluxlab_gcc_runtime() {
    local configured_root="/home/modules/compilers/gcc/11.1.0"
    local gcc_root libstdcxx runtime_dir
    gcc_root="$(readlink -f "${configured_root}" 2>/dev/null || true)"
    if [[ -z "${gcc_root}" || ! -d "${gcc_root}" ]]; then
        _ichor_env_error "ffluxlab GCC module root is unavailable: ${configured_root}"
        return 1
    fi
    libstdcxx="${gcc_root}/lib64/libstdc++.so.6"
    if [[ ! -e "${libstdcxx}" ]]; then
        libstdcxx="$(
            find "${gcc_root}" \
                \( -type f -o -type l \) \
                -path '*/lib64/libstdc++.so.6' \
                -print -quit
        )"
    fi
    if [[ -z "${libstdcxx}" || ! -e "${libstdcxx}" ]]; then
        _ichor_env_error \
            "GCC 11 libstdc++.so.6 is absent beneath ${gcc_root}"
        return 1
    fi
    runtime_dir="$(dirname "${libstdcxx}")"
    _ichor_env_prepend_ld_library_paths_once "${runtime_dir}"
    export LIBRARY_PATH="${runtime_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export ICHOR_GCC_RUNTIME_DIR="${runtime_dir}"
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
        auto|csf3|csf4|ffluxlab) ;;
        *)
            _ichor_env_error "machine must be auto, csf3, csf4, or ffluxlab"
            _ichor_env_usage >&2
            return 2
            ;;
    esac
    machine="$(ichor_detect_machine "${machine}")" || return 2

    if [[ "${do_smoke_heavy}" -eq 1 && ! -f "${HOME}/ichor_config.yaml" ]]; then
        _ichor_env_error \
            "ICHOR configuration not found: ${HOME}/ichor_config.yaml. Complete the full installer or run install_ichor.sh --only config before --smoke-heavy."
        return 1
    fi

    if [[ "${debug_trace}" -eq 1 ]]; then
        export PS4='+ env_ichor.sh:${LINENO}:${FUNCNAME[0]:-main}: '
        set -x
    fi

    python_prefix="$(ichor_csf_expand_path "${python_prefix}")"
    plumed_prefix="$(ichor_csf_expand_path "${plumed_prefix}")"
    if [[ -z "${venv}" ]]; then
        venv="${HOME}/.venv/ichor-${machine}"
    fi
    venv="$(ichor_csf_expand_path "${venv}")"

    ichor_csf_deactivate_existing_venv "loading cluster modules" || return 1
    _ichor_env_load_runtime_modules "${machine}" "${do_purge}" || return 1
    if [[ "${machine}" == "csf4" ]]; then
        ichor_csf_isolate_python_environment
    fi
    if [[ "${machine}" == "ffluxlab" ]]; then
        _ichor_env_ffluxlab_intel_runtime || return 1
        _ichor_env_ffluxlab_gcc_runtime || return 1
    fi

    unset CC CXX FC F77 F90
    export ICHOR_MACHINE="${machine}"
    export PLUMED_KERNEL="${plumed_prefix}/lib/libplumedKernel.so"
    export PLUMED_LIBRARY_PATH="${plumed_prefix}/lib"

    if [[ "${machine}" == "csf3" || "${machine}" == "ffluxlab" ]]; then
        _ichor_env_prepend_ld_library_paths_once \
            "${python_prefix}/lib" \
            "${ICHOR_GCC_RUNTIME_DIR:-}" \
            "${ICHOR_INTEL_RUNTIME_DIR:-}" \
            "${ICHOR_MKL_RUNTIME_DIR:-}" \
            "${plumed_prefix}/lib"
    else
        _ichor_env_prepend_ld_library_paths_once "${plumed_prefix}/lib"
    fi

    if [[ ! -f "${venv}/bin/activate" ]]; then
        _ichor_env_error "venv activation script not found: ${venv}/bin/activate"
        return 1
    fi
    # shellcheck disable=SC1091
    source "${venv}/bin/activate" || return 1
    if [[ "${machine}" == "csf4" ]]; then
        ichor_csf_isolate_python_environment
    fi
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
        _ichor_env_ariadne_check "ARIADNE" "import ariadne; assert hasattr(ariadne, 'Geometric_Trqn') or hasattr(ariadne, 'Ds_Optimiser')" || return 1
        _ichor_env_import_check "PLUMED" "import os, plumed; p=plumed.Plumed(kernel=os.environ['PLUMED_KERNEL']); p.finalize()" || return 1
        _ichor_env_import_check "ICHOR packages" "import ichor.core, ichor.hpc, ichor.cli" || return 1
        _ichor_env_import_check "pyferebus" "import pyferebus.executors.trainer" || return 1
        _ichor_env_import_check "RDKit" "from rdkit import Chem" || return 1
        _ichor_env_import_check "xTB" "from xtb.ase.calculator import XTB" || return 1
    fi

    if [[ "${do_smoke_heavy}" -eq 1 ]]; then
        _ichor_env_ariadne_check "ARIADNE runtime smoke" "import ariadne; from ichor.hpc.active_learning.acquisition.ariadne_local_runner import probe_ariadne_runtime; probe_ariadne_runtime(ariadne)" || return 1
        _ichor_env_import_check "xTB runtime smoke" "from ichor.hpc.runtime_preflight import ensure_xtb_ase_available; ensure_xtb_ase_available(run_energy=True)" || return 1
        _ichor_env_import_check "PLUMED runtime smoke" "from ichor.hpc.runtime_preflight import ensure_plumed_available; ensure_plumed_available(run_ase_smoke=True)" || return 1
    fi
}

_ichor_env_main "$@"
_ichor_env_rc=$?
unset -f _ichor_env_usage _ichor_env_error _ichor_env_note _ichor_env_module
unset -f _ichor_env_prepend_ld_library_paths_once
unset -f _ichor_env_import_check _ichor_env_ariadne_check
unset -f _ichor_env_print_env _ichor_env_load_runtime_modules
unset -f _ichor_env_ffluxlab_intel_runtime
unset -f _ichor_env_ffluxlab_gcc_runtime
unset -f _ichor_env_check_venv_ownership _ichor_env_main
unset ICHOR_ENV_QUIET _ichor_env_script_dir
return "${_ichor_env_rc}"
