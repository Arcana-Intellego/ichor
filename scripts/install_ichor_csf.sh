#!/usr/bin/env bash
#
# Install ICHOR active-learning dependencies on Manchester CSF3/CSF4.
#
# The script is intentionally conservative:
#   * downloads are opt-in via --allow-download;
#   * Gaussian and AIMAll are verified, not installed;
#   * ~/ichor_config.yaml is backed up before the active CSF profile is upserted;
#   * Intel compiler variables used for ARIADNE are explicitly cleared before
#     the PLUMED GCC build.

set -euo pipefail

PYTHON_VERSION="3.11.15"
PLUMED_VERSION="2.10.0"
PYTHON_TARBALL="Python-${PYTHON_VERSION}.tgz"
PLUMED_TARBALL="plumed-${PLUMED_VERSION}.tgz"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_TARBALL}"
PLUMED_URL="https://github.com/plumed/plumed2/releases/download/v${PLUMED_VERSION}/${PLUMED_TARBALL}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_ichor_csf.sh"

MACHINE="${ICHOR_MACHINE:-auto}"
PROJECTS_DIR="${ICHOR_PROJECTS_DIR:-${HOME}/projects}"
REPO_ROOT="${ICHOR_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
VENV="${ICHOR_VENV:-}"
PYTHON_PREFIX="${PYTHON_PREFIX:-${HOME}/opt/python-${PYTHON_VERSION}}"
AIMALL_PATH="${AIMALL_PATH:-${HOME}/AIMAll/aimqb.ish}"
FEREBUS_PATH="${FEREBUS_PATH:-${HOME}/.local/bin/ferebus}"
INSTALL_JOBS="${ICHOR_INSTALL_JOBS:-4}"
ONLY_STAGE="all"
ALLOW_DOWNLOAD=0
SKIP_PLUMED=0
SKIP_ARIADNE_BUILD=0
SKIP_FEREBUS_BUILD=0
ASSUME_YES=0
DRY_RUN=0
DEBUG_TRACE=0
CURRENT_STAGE="startup"
LOG_FILE=""
ARIADNE_CC=""
ARIADNE_CXX=""
ARIADNE_FC=""
ARIADNE_CC_METHOD=""
ARIADNE_CXX_METHOD=""
ARIADNE_FC_METHOD=""

usage() {
    cat <<'EOF'
Usage: scripts/install_ichor_csf.sh [options]

Options:
  --machine auto|csf3|csf4        default: auto
  --repo-root PATH                default: script-derived repo root
  --projects-dir PATH             default: ~/projects
  --venv PATH                     default: ~/.venv/ichor-csf3 or ~/.venv/ichor-csf4
  --python-prefix PATH            CSF3 default: ~/opt/python-3.11.15
  --only all|python|packages|ariadne|plumed|ferebus|config|verify|doctor
                                  default: all. A component stage repairs or
                                  reinstalls that component.
  --allow-download                permit Python/PLUMED/OpenBLAS/source downloads
  --skip-plumed                   skip native PLUMED build and smoke
  --skip-ariadne-build            only verify import ariadne
  --skip-ferebus-build            only verify configured ferebus executable
  --aimall-path PATH              default: ~/AIMAll/aimqb.ish
  --ferebus-path PATH             default: ~/.local/bin/ferebus
  --jobs N                        default: 4
  --yes                           non-interactive mode
  --dry-run                       print planned actions without installing
  --debug, --trace                print shell trace in the install log
  -h, --help                      show this help

Environment equivalents:
  ICHOR_MACHINE, ICHOR_PROJECTS_DIR, ICHOR_REPO_ROOT, ICHOR_VENV,
  ICHOR_INSTALL_JOBS, AIMALL_PATH, FEREBUS_PATH, PLUMED_KERNEL,
  PLUMED_LIBRARY_PATH, PIP_FIND_LINKS.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

note() {
    echo ""
    echo "==> $*"
}

warn() {
    echo "WARNING: $*" >&2
}

on_error() {
    local status=$?
    local line="${BASH_LINENO[0]:-unknown}"
    local command="${BASH_COMMAND:-unknown}"
    {
        echo ""
        echo "ERROR: install stage '${CURRENT_STAGE}' failed"
        echo "  line: ${line}"
        echo "  command: ${command}"
        [[ -n "${LOG_FILE}" ]] && echo "  log: ${LOG_FILE}"
        if [[ "${CURRENT_STAGE}" == "ariadne" ]]; then
            print_ariadne_manual_recovery_block || true
        fi
    } >&2
    exit "${status}"
}

trap on_error ERR

while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine) MACHINE="${2:?missing value for --machine}"; shift 2 ;;
        --repo-root) REPO_ROOT="${2:?missing value for --repo-root}"; shift 2 ;;
        --projects-dir) PROJECTS_DIR="${2:?missing value for --projects-dir}"; shift 2 ;;
        --venv) VENV="${2:?missing value for --venv}"; shift 2 ;;
        --python-prefix) PYTHON_PREFIX="${2:?missing value for --python-prefix}"; shift 2 ;;
        --only) ONLY_STAGE="${2:?missing value for --only}"; shift 2 ;;
        --allow-download) ALLOW_DOWNLOAD=1; shift ;;
        --skip-plumed) SKIP_PLUMED=1; shift ;;
        --skip-ariadne-build) SKIP_ARIADNE_BUILD=1; shift ;;
        --skip-ferebus-build) SKIP_FEREBUS_BUILD=1; shift ;;
        --aimall-path) AIMALL_PATH="${2:?missing value for --aimall-path}"; shift 2 ;;
        --ferebus-path) FEREBUS_PATH="${2:?missing value for --ferebus-path}"; shift 2 ;;
        --jobs) INSTALL_JOBS="${2:?missing value for --jobs}"; shift 2 ;;
        --yes) ASSUME_YES=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --debug|--trace) DEBUG_TRACE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unrecognised option: $1" ;;
    esac
done

case "${MACHINE}" in
    auto|csf3|csf4) ;;
    *) die "--machine must be auto, csf3, or csf4" ;;
esac
case "${ONLY_STAGE}" in
    all|python|packages|ariadne|plumed|ferebus|config|verify|doctor) ;;
    *) die "--only must be one of all, python, packages, ariadne, plumed, ferebus, config, verify, doctor" ;;
esac
[[ "${INSTALL_JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"

expand_path() {
    ichor_csf_expand_path "$1"
}

PROJECTS_DIR="$(expand_path "${PROJECTS_DIR}")"
REPO_ROOT="$(expand_path "${REPO_ROOT}")"
PYTHON_PREFIX="$(expand_path "${PYTHON_PREFIX}")"
AIMALL_PATH="$(expand_path "${AIMALL_PATH}")"
FEREBUS_PATH="$(expand_path "${FEREBUS_PATH}")"

if [[ -z "${VENV}" ]]; then
    # Filled after machine detection.
    VENV=""
else
    VENV="$(expand_path "${VENV}")"
fi

run_cmd() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

run_shell() {
    local command="$1"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ ${command}"
    else
        bash -c "${command}"
    fi
}

run_in_dir() {
    local dir="$1"
    shift
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '+ cd %q &&' "${dir}"
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    (cd "${dir}" && "$@")
}

run_shell_in_dir() {
    local dir="$1"
    local command="$2"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ cd $(printf '%q' "${dir}") && ${command}"
        return 0
    fi
    (cd "${dir}" && bash -c "${command}")
}

deactivate_existing_venv() {
    ichor_csf_deactivate_existing_venv "$@"
}

backup_existing_path() {
    local path="$1"
    local label="$2"
    if [[ ! -e "${path}" ]]; then
        return 0
    fi
    local backup="${path}.bak.$(date +%Y%m%d-%H%M%S)"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ move existing ${label} ${path} -> ${backup}"
    else
        run_cmd mv "${path}" "${backup}"
    fi
}

require_dir() {
    local path="$1"
    local label="$2"
    if [[ ! -d "${path}" ]]; then
        die "${label} not found: ${path}"
    fi
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        [[ -f "${path}" ]] || warn "dry-run: ${label} not found yet: ${path}"
        return 0
    fi
    if [[ ! -f "${path}" ]]; then
        die "${label} not found: ${path}"
    fi
}

require_cmd() {
    local cmd="$1"
    local hint="${2:-}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ require command ${cmd}"
        return 0
    fi
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        if [[ -n "${hint}" ]]; then
            die "required command '${cmd}' is not on PATH. ${hint}"
        fi
        die "required command '${cmd}' is not on PATH"
    fi
}

resolve_required_cmd_into() {
    local target_var="$1"
    local cmd="$2"
    local hint="${3:-}"
    local resolved
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ resolve command ${cmd}" >&2
        printf -v "${target_var}" '<resolved:%s>' "${cmd}"
        return 0
    fi

    resolved="$(command -v "${cmd}" || true)"
    if [[ -z "${resolved}" ]]; then
        if [[ "${cmd}" == "icx" || "${cmd}" == "icpx" || "${cmd}" == "ifx" || "${hint}" == ARIADNE* ]]; then
            module_debug "compiler resolution failed for ${cmd}"
        fi
        if [[ -n "${hint}" ]]; then
            die "required command '${cmd}' is not on PATH. ${hint}"
        fi
        die "required command '${cmd}' is not on PATH"
    fi
    printf -v "${target_var}" '%s' "${resolved}"
}

module_is_current_shell_function() {
    ichor_csf_module_is_shell_function
}

initialise_modules() {
    ichor_csf_initialise_modules
}

module_debug() {
    ichor_csf_module_debug "$@"
}

find_ariadne_compiler_path() {
    local exe="$1"
    local result
    result="$(ichor_csf_find_ariadne_compiler_path "${exe}" || true)"
    [[ -n "${result}" ]] || return 1
    printf '%s\n' "${result%%|*}"
}

ensure_ariadne_compilers_on_path() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        return 0
    fi
    local compiler result resolved method bin_dir missing=0
    for compiler in icx icpx ifx; do
        result="$(ichor_csf_find_ariadne_compiler_path "${compiler}" || true)"
        resolved="${result%%|*}"
        method="${result#*|}"
        if [[ -z "${resolved}" ]]; then
            warn "ARIADNE compiler '${compiler}' was not found after loading ARIADNE modules"
            missing=1
            continue
        fi
        bin_dir="$(dirname "${resolved}")"
        ichor_csf_prepend_path_once "${bin_dir}"
        case "${compiler}" in
            icx) ARIADNE_CC="${resolved}"; ARIADNE_CC_METHOD="${method}" ;;
            icpx) ARIADNE_CXX="${resolved}"; ARIADNE_CXX_METHOD="${method}" ;;
            ifx) ARIADNE_FC="${resolved}"; ARIADNE_FC_METHOD="${method}" ;;
        esac
    done
    hash -r 2>/dev/null || true
    if [[ "${missing}" -ne 0 ]]; then
        module_debug "ARIADNE compiler check after module load"
        die "ARIADNE compiler modules did not expose icx/icpx/ifx"
    fi
}

module_cmd() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '+ module'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    ichor_csf_module "$@"
}

detect_machine() {
    ichor_csf_detect_machine "${MACHINE}" || die "could not auto-detect CSF3/CSF4. Pass --machine csf3 or --machine csf4."
}

url_available() {
    local url="$1"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        return 1
    fi
    if command -v curl >/dev/null 2>&1; then
        curl -fsSI --max-time 8 "${url}" >/dev/null 2>&1
    elif command -v wget >/dev/null 2>&1; then
        wget --spider -q --timeout=8 "${url}" >/dev/null 2>&1
    else
        return 1
    fi
}

download_to() {
    local url="$1"
    local dest="$2"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ download or stage ${url} -> ${dest}" >&2
        return 0
    fi
    [[ "${ALLOW_DOWNLOAD}" -eq 1 ]] || die "download disabled for ${url}; rerun with --allow-download or stage ${dest}"
    mkdir -p "$(dirname "${dest}")"
    if command -v curl >/dev/null 2>&1; then
        run_cmd curl -fL "${url}" -o "${dest}"
    elif command -v wget >/dev/null 2>&1; then
        run_cmd wget -O "${dest}" "${url}"
    else
        die "neither curl nor wget is available for download: ${url}"
    fi
}

print_download_readiness() {
    note "Checking download readiness"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        warn "dry-run: network checks skipped"
    else
        for label_url in \
            "PyPI|https://pypi.org/simple/" \
            "Python|${PYTHON_URL}" \
            "PLUMED|${PLUMED_URL}" \
            "GitHub/OpenBLAS|https://github.com/"; do
            local label="${label_url%%|*}"
            local url="${label_url#*|}"
            if url_available "${url}"; then
                echo "  ${label}: reachable"
            else
                warn "${label} source is not reachable from this node"
            fi
        done
    fi
    cat <<EOF

If online downloads are unavailable and a required source is missing:
  * Python: place ${PYTHON_TARBALL} in ${PROJECTS_DIR}/_sources/
  * PLUMED: place ${PLUMED_TARBALL} in ${PROJECTS_DIR}/_sources/
            or extract it as ${PROJECTS_DIR}/plumed-${PLUMED_VERSION}
  * OpenBLAS/FEREBUS: ensure ${PROJECTS_DIR}/FEREBUS_CPU/libs/openblas/lib64/libopenblas.a
            or ${PROJECTS_DIR}/FEREBUS_CPU/libs/openblas/lib/libopenblas.a exists,
            or run 'cd ${PROJECTS_DIR}/FEREBUS_CPU/libs && ./fetchOpenBlas.sh'
            on an internet-capable node.
  * Python wheels: set PIP_FIND_LINKS=/path/to/wheelhouse before running this script.
EOF
}

python_import_ok() {
    local module_name="$1"
    "${PYTHON}" -c "import ${module_name}" >/dev/null 2>&1
}

config_path_for_yaml() {
    local path="$1"
    if [[ "${path}" == "${HOME}"/* ]]; then
        printf '$HOME/%s\n' "${path#"${HOME}/"}"
    else
        printf '%s\n' "${path}"
    fi
}

ensure_repo_root() {
    if [[ -d "${REPO_ROOT}/ichor_core" && -d "${REPO_ROOT}/ichor_hpc" && -d "${REPO_ROOT}/ichor_cli" ]]; then
        return 0
    fi
    for candidate in "${PROJECTS_DIR}/ichor-active-learning" "${PROJECTS_DIR}/ichor"; do
        if [[ -d "${candidate}/ichor_core" && -d "${candidate}/ichor_hpc" && -d "${candidate}/ichor_cli" ]]; then
            REPO_ROOT="${candidate}"
            return 0
        fi
    done
    die "ICHOR repo root not found. Pass --repo-root /path/to/ichor-active-learning or /path/to/ichor."
}

load_python_stack() {
    if [[ "${MACHINE}" == "csf4" ]]; then
        module_cmd purge
        module_cmd load python/3.11.3-gcccore-12.3.0
        module_cmd load python-bundle-pypi/2023.06-gcccore-12.3.0
    fi
}

load_ariadne_modules() {
    module_cmd purge
    if [[ "${MACHINE}" == "csf3" ]]; then
        module_cmd load compilers/intel/oneapi/2025.0.1
        module_cmd load umf compiler-rt tbb compiler
        module_cmd load mkl/2025.0
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    else
        module_cmd load python/3.11.3-gcccore-12.3.0
        module_cmd load python-bundle-pypi/2023.06-gcccore-12.3.0
        module_cmd load compilers/oneapi/2024.2.0
        module_cmd load compiler-rt tbb compiler
        module_cmd load mkl/2024.2
    fi
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        ensure_ariadne_compilers_on_path
    fi
}

resolve_ariadne_compilers() {
    ensure_ariadne_compilers_on_path
    resolve_required_cmd_into ARIADNE_CC icx "ARIADNE requires the Intel oneAPI C compiler. Run 'module list' and check the oneAPI compiler module."
    resolve_required_cmd_into ARIADNE_CXX icpx "ARIADNE requires the Intel oneAPI C++ compiler. Run 'module list' and check the oneAPI compiler module."
    resolve_required_cmd_into ARIADNE_FC ifx "ARIADNE requires the Intel oneAPI Fortran compiler. Run 'module list' and check the oneAPI compiler module."
}

load_gcc_build_modules() {
    module_cmd purge
    if [[ "${MACHINE}" == "csf3" ]]; then
        module_cmd load compilers/gcc/13.3.0
        module_cmd load tools/gcc/cmake/3.31.6
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    else
        module_cmd load python/3.11.3-gcccore-12.3.0
        module_cmd load python-bundle-pypi/2023.06-gcccore-12.3.0
        module_cmd load cmake/3.23.1-gcccore-11.3.0
    fi
}

load_csf3_python_build_modules() {
    load_gcc_build_modules
    module_cmd load libs/gcc/openssl/1.1.1w
}

resolve_openssl_prefix() {
    local prefix
    for var_name in EBROOTOPENSSL OPENSSL_ROOT_DIR; do
        prefix="${!var_name:-}"
        if [[ -n "${prefix}" && -d "${prefix}" ]]; then
            printf '%s\n' "${prefix}"
            return 0
        fi
    done

    local openssl_bin
    openssl_bin="$(command -v openssl || true)"
    if [[ -n "${openssl_bin}" ]]; then
        prefix="$(cd "$(dirname "${openssl_bin}")/.." && pwd)"
        if [[ -d "${prefix}/include/openssl" ]]; then
            printf '%s\n' "${prefix}"
            return 0
        fi
    fi

    die "OpenSSL prefix could not be resolved. On CSF3 load libs/gcc/openssl/1.1.1w before building private Python."
}

verify_python_ssl() {
    local python_exe="$1"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ ${python_exe} -c 'import ssl; print(ssl.OPENSSL_VERSION)'"
        return 0
    fi
    if ! "${python_exe}" -c "import ssl; print(ssl.OPENSSL_VERSION)" >/dev/null 2>&1; then
        die "Private Python at ${python_exe} cannot import ssl. Move aside ${PYTHON_PREFIX} and rebuild with the CSF3 OpenSSL module loaded."
    fi
}

ensure_csf3_python() {
    local rebuild="${1:-0}"
    if [[ "${MACHINE}" != "csf3" ]]; then
        return 0
    fi
    load_csf3_python_build_modules
    local py="${PYTHON_PREFIX}/bin/python3.11"
    if [[ "${rebuild}" -eq 1 && -x "${py}" ]]; then
        note "Rebuilding private CPython ${PYTHON_VERSION}"
        backup_existing_path "${PYTHON_PREFIX}" "private Python prefix"
    fi
    if [[ -x "${py}" ]]; then
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        verify_python_ssl "${py}"
        return 0
    fi

    note "Private CPython ${PYTHON_VERSION} is missing; preparing build"
    require_cmd gcc "Load a GCC compiler module first."
    require_cmd make "Load a build tool module first."
    require_cmd openssl "Load libs/gcc/openssl/1.1.1w first."
    local openssl_prefix
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        openssl_prefix="\${OPENSSL_PREFIX_FROM_MODULE}"
    else
        openssl_prefix="$(resolve_openssl_prefix)"
    fi
    local sources_dir="${PROJECTS_DIR}/_sources"
    local tarball="${sources_dir}/${PYTHON_TARBALL}"
    if [[ ! -f "${tarball}" ]]; then
        download_to "${PYTHON_URL}" "${tarball}"
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ build private CPython ${PYTHON_VERSION} from ${tarball} with --with-openssl=${openssl_prefix}"
        verify_python_ssl "${py}"
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        return 0
    fi
    require_file "${tarball}" "Python source tarball"
    local build_parent="${HOME}/src"
    local src_dir="${build_parent}/Python-${PYTHON_VERSION}"
    run_cmd mkdir -p "${build_parent}" "$(dirname "${PYTHON_PREFIX}")"
    run_cmd tar -xzf "${tarball}" -C "${build_parent}"
    run_in_dir "${src_dir}" ./configure "--prefix=${PYTHON_PREFIX}" --enable-shared --with-ensurepip=install "--with-openssl=${openssl_prefix}" --with-openssl-rpath=auto
    run_in_dir "${src_dir}" make -j "${INSTALL_JOBS}"
    run_in_dir "${src_dir}" make install
    export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    verify_python_ssl "${py}"
}

create_or_activate_venv() {
    local recreate="${1:-0}"
    local base_python
    if [[ "${MACHINE}" == "csf3" ]]; then
        base_python="${PYTHON_PREFIX}/bin/python3.11"
    else
        base_python="python"
    fi
    if [[ "${recreate}" -eq 1 && -d "${VENV}" ]]; then
        backup_existing_path "${VENV}" "venv"
    fi
    if [[ ! -x "${VENV}/bin/python" ]]; then
        note "Creating venv at ${VENV}"
        run_cmd "${base_python}" -m venv "${VENV}"
    fi
    deactivate_existing_venv "activating target venv"
    # shellcheck disable=SC1091
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        source "${VENV}/bin/activate"
    else
        echo "+ source $(printf '%q' "${VENV}/bin/activate")"
    fi
    PYTHON="${VENV}/bin/python"
    PIP="${PYTHON} -m pip"
    if [[ "${MACHINE}" == "csf3" ]]; then
        verify_python_ssl "${PYTHON}"
    fi
}

activate_existing_venv() {
    if [[ "${DRY_RUN}" -eq 0 && ! -f "${VENV}/bin/activate" ]]; then
        die "venv activation script not found: ${VENV}/bin/activate. Run --only python first."
    fi
    deactivate_existing_venv "activating target venv"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        # shellcheck disable=SC1091
        source "${VENV}/bin/activate"
    else
        echo "+ source $(printf '%q' "${VENV}/bin/activate")"
    fi
    PYTHON="${VENV}/bin/python"
    PIP="${PYTHON} -m pip"
    if [[ "${MACHINE}" == "csf3" ]]; then
        verify_python_ssl "${PYTHON}"
    fi
}

prepare_python_and_venv() {
    local rebuild_python="${1:-0}"
    local recreate_venv="${2:-0}"
    deactivate_existing_venv "Python module setup"
    load_python_stack
    ensure_csf3_python "${rebuild_python}"
    create_or_activate_venv "${recreate_venv}"
}

prepare_runtime_environment() {
    note "Preparing runtime environment"
    deactivate_existing_venv "runtime module setup"
    load_ariadne_modules
    activate_existing_venv
    unset CC CXX FC F77 F90
    export ICHOR_MACHINE="${MACHINE}"
    if [[ "${SKIP_PLUMED}" -eq 0 ]]; then
        export PLUMED_KERNEL="${PLUMED_KERNEL:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib/libplumedKernel.so}"
        export PLUMED_LIBRARY_PATH="${PLUMED_LIBRARY_PATH:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib}"
        export LD_LIBRARY_PATH="${PLUMED_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

pip_install() {
    # shellcheck disable=SC2086
    run_cmd "${PYTHON}" -m pip install "$@"
}

install_python_packages() {
    note "Installing ICHOR, POLUS, and pyferebus Python packages"
    require_dir "${REPO_ROOT}/ichor_core" "ichor_core package"
    require_dir "${REPO_ROOT}/ichor_hpc" "ichor_hpc package"
    require_dir "${REPO_ROOT}/ichor_cli" "ichor_cli package"
    require_dir "${PROJECTS_DIR}/POLUS/polus_core_subpackage" "POLUS core package"
    require_dir "${PROJECTS_DIR}/FEREBUS_CPU/pyferebus" "pyferebus package"

    pip_install --upgrade pip setuptools wheel
    pip_install pytest
    pip_install -e "${REPO_ROOT}/ichor_core"
    pip_install -e "${REPO_ROOT}/ichor_hpc"
    pip_install -e "${REPO_ROOT}/ichor_cli"
    pip_install -e "${PROJECTS_DIR}/POLUS/polus_core_subpackage" --no-deps
    pip_install -e "${PROJECTS_DIR}/FEREBUS_CPU/pyferebus" --no-deps
}

verify_ariadne_api() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ ${PYTHON} -c 'import ariadne; check optimiser API'"
        return 0
    fi
    "${PYTHON}" - <<'PY'
import ariadne

if not (hasattr(ariadne, "Geometric_Trqn") or hasattr(ariadne, "Ds_Optimiser")):
    raise SystemExit("ariadne imported but no documented optimiser API was found")
print("ARIADNE OK")
PY
}

print_ariadne_import_info() {
    local label="${1:-ARIADNE import}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ ${PYTHON:-python} -c 'print ${label} ariadne path and mtime'"
        return 0
    fi
    if [[ -z "${PYTHON:-}" || ! -x "${PYTHON}" ]]; then
        warn "${label}: target Python is not ready"
        return 0
    fi
    "${PYTHON}" - "${label}" <<'PY' || true
import os
import sys
import time

label = sys.argv[1]
try:
    import ariadne
except Exception as exc:
    print(f"{label}: ariadne not importable ({exc})")
    raise SystemExit(0)

path = getattr(ariadne, "__file__", "<unknown>")
try:
    mtime = time.ctime(os.path.getmtime(path))
except OSError:
    mtime = "<unavailable>"
print(f"{label}: {path}")
print(f"{label} mtime: {mtime}")
print(
    f"{label} API: Geometric_Trqn={hasattr(ariadne, 'Geometric_Trqn')} "
    f"Ds_Optimiser={hasattr(ariadne, 'Ds_Optimiser')}"
)
PY
}

assert_ariadne_inside_venv() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ ${PYTHON} -c 'assert ariadne imported from target venv'"
        return 0
    fi
    "${PYTHON}" - "${VENV}" <<'PY'
from pathlib import Path
import sys

import ariadne

venv = Path(sys.argv[1]).resolve()
path = Path(ariadne.__file__).resolve()
if venv not in path.parents:
    raise SystemExit(f"ariadne imported outside target venv: {path} (venv: {venv})")
print(f"ARIADNE import path is inside target venv: {path}")
PY
}

print_ariadne_compiler_paths() {
    echo "ARIADNE_CC=${ARIADNE_CC:-<unresolved>} (${ARIADNE_CC_METHOD:-unknown})"
    echo "ARIADNE_CXX=${ARIADNE_CXX:-<unresolved>} (${ARIADNE_CXX_METHOD:-unknown})"
    echo "ARIADNE_FC=${ARIADNE_FC:-<unresolved>} (${ARIADNE_FC_METHOD:-unknown})"
}

ariadne_receipt_path() {
    printf '%s\n' "${HOME}/.cache/ichor-al-install/ariadne-${MACHINE}-last-install.json"
}

write_ariadne_receipt() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ write ARIADNE install receipt $(ariadne_receipt_path)"
        return 0
    fi
    mkdir -p "${HOME}/.cache/ichor-al-install"
    local ariadne_root="${PROJECTS_DIR}/ARIADNE"
    local repo_commit=""
    repo_commit="$(git -C "${ariadne_root}" rev-parse --short HEAD 2>/dev/null || true)"
    "${PYTHON}" - "$(ariadne_receipt_path)" "${MACHINE}" "${ariadne_root}" "${repo_commit}" "${PYTHON}" "${VENV}" "${ARIADNE_CC:-}" "${ARIADNE_CXX:-}" "${ARIADNE_FC:-}" <<'PY'
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ariadne

receipt, machine, repo_root, repo_commit, python, venv, cc, cxx, fc = sys.argv[1:10]
ariadne_file = Path(ariadne.__file__).resolve()
payload = {
    "schema_version": 1,
    "machine": machine,
    "timestamp_iso": datetime.now(timezone.utc).isoformat(),
    "repo_root": repo_root,
    "repo_commit": repo_commit,
    "python": python,
    "venv": venv,
    "ariadne_file": str(ariadne_file),
    "ariadne_mtime": time.ctime(os.path.getmtime(ariadne_file)),
    "ariadne_api": {
        "Geometric_Trqn": hasattr(ariadne, "Geometric_Trqn"),
        "Ds_Optimiser": hasattr(ariadne, "Ds_Optimiser"),
    },
    "compilers": {"CC": cc, "CXX": cxx, "FC": fc},
    "cmake_settings": {
        "ARIADNE_SAFE_IFX_FLAGS": machine == "csf3",
    },
}
path = Path(receipt)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote ARIADNE install receipt: {path}")
PY
}

doctor_ariadne_receipt() {
    local receipt
    receipt="$(ariadne_receipt_path)"
    if [[ ! -f "${receipt}" ]]; then
        echo "ARIADNE receipt: missing (${receipt})"
        return 0
    fi
    echo "ARIADNE receipt: ${receipt}"
    if [[ "${DRY_RUN}" -eq 1 || -z "${PYTHON:-}" || ! -x "${PYTHON}" ]]; then
        return 0
    fi
    "${PYTHON}" - "${receipt}" "${PROJECTS_DIR}/ARIADNE" <<'PY' || true
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

receipt = Path(sys.argv[1])
ariadne_repo = Path(sys.argv[2])
data = json.loads(receipt.read_text(encoding="utf-8"))
try:
    import ariadne
except Exception as exc:
    print(f"ARIADNE receipt compare: current import failed ({exc})")
    raise SystemExit(0)

current = str(Path(ariadne.__file__).resolve())
recorded = str(Path(data.get("ariadne_file", "")).resolve())
if current != recorded:
    print(f"WARNING: ARIADNE import differs from receipt: current={current} receipt={recorded}")
try:
    current_mtime = os.path.getmtime(current)
except OSError:
    current_mtime = 0
if data.get("python") != sys.executable:
    print(f"WARNING: ARIADNE receipt Python differs: current={sys.executable} receipt={data.get('python')}")
try:
    commit = subprocess.check_output(["git", "-C", str(ariadne_repo), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = ""
if commit and data.get("repo_commit") and commit != data.get("repo_commit"):
    print(f"WARNING: ARIADNE repo changed since last receipt: current={commit} receipt={data.get('repo_commit')}")
print(f"ARIADNE receipt compare complete: current_mtime={current_mtime}")
PY
}

print_ariadne_manual_recovery_block() {
    local ariadne_root="${PROJECTS_DIR:-${HOME}/projects}/ARIADNE"
    local python_exe="${PYTHON:-${VENV:-${HOME}/.venv/ichor-${MACHINE:-csf3}}/bin/python}"
    local safe_flag=""
    if [[ "${MACHINE:-}" == "csf3" ]]; then
        safe_flag=" --config-settings=cmake.define.ARIADNE_SAFE_IFX_FLAGS=ON"
    fi
    cat >&2 <<EOF

Manual ARIADNE recovery block:
cd ${ariadne_root}
module purge
EOF
    if [[ "${MACHINE:-}" == "csf3" ]]; then
        cat >&2 <<'EOF'
module load compilers/intel/oneapi/2025.0.1
module load umf compiler-rt tbb compiler
module load mkl/2025.0
EOF
        echo "source ${VENV:-${HOME}/.venv/ichor-csf3}/bin/activate" >&2
        echo "export LD_LIBRARY_PATH=${PYTHON_PREFIX:-${HOME}/opt/python-3.11.15}/lib:\${LD_LIBRARY_PATH}" >&2
    else
        cat >&2 <<'EOF'
module load python/3.11.3-gcccore-12.3.0
module load python-bundle-pypi/2023.06-gcccore-12.3.0
module load compilers/oneapi/2024.2.0
module load compiler-rt tbb compiler
module load mkl/2024.2
EOF
        echo "source ${VENV:-${HOME}/.venv/ichor-csf4}/bin/activate" >&2
    fi
    cat >&2 <<EOF
export CC=${ARIADNE_CC:-/resolved/path/icx}
export CXX=${ARIADNE_CXX:-/resolved/path/icpx}
export FC=${ARIADNE_FC:-/resolved/path/ifx}
export CMAKE_BUILD_PARALLEL_LEVEL=${INSTALL_JOBS:-4}
export MAKEFLAGS=-j${INSTALL_JOBS:-4}
${python_exe} -m pip install . --no-build-isolation -v --force-reinstall --no-deps${safe_flag}
EOF
}

install_ariadne_if_needed() {
    local reinstall="${1:-0}"
    CURRENT_STAGE="ariadne"
    note "Checking ARIADNE"
    deactivate_existing_venv "ARIADNE module setup"
    load_ariadne_modules
    # shellcheck disable=SC1091
    create_or_activate_venv 0
    resolve_ariadne_compilers
    print_ariadne_compiler_paths
    print_ariadne_import_info "ARIADNE before install"
    if [[ "${reinstall}" -eq 0 && "${DRY_RUN}" -eq 0 ]] && python_import_ok ariadne; then
        echo "ARIADNE already importable"
        verify_ariadne_api
        assert_ariadne_inside_venv
        return 0
    fi
    if [[ "${SKIP_ARIADNE_BUILD}" -eq 1 ]]; then
        [[ "${DRY_RUN}" -eq 1 ]] && echo "dry-run: would verify import ariadne"
        [[ "${DRY_RUN}" -eq 1 ]] || die "ARIADNE is not importable and --skip-ariadne-build was set"
        return 0
    fi
    local ariadne_root="${PROJECTS_DIR}/ARIADNE"
    require_dir "${ariadne_root}" "ARIADNE source tree"
    pip_install -r "${ariadne_root}/requirements-build.txt"
    [[ -n "${ARIADNE_CC:-}" && -n "${ARIADNE_CXX:-}" && -n "${ARIADNE_FC:-}" ]] || die "ARIADNE compiler paths were not resolved after loading oneAPI modules."
    [[ "${DRY_RUN}" -eq 1 || -x "${ARIADNE_CC}" ]] || die "resolved ARIADNE C compiler is not executable: ${ARIADNE_CC}"
    [[ "${DRY_RUN}" -eq 1 || -x "${ARIADNE_CXX}" ]] || die "resolved ARIADNE C++ compiler is not executable: ${ARIADNE_CXX}"
    [[ "${DRY_RUN}" -eq 1 || -x "${ARIADNE_FC}" ]] || die "resolved ARIADNE Fortran compiler is not executable: ${ARIADNE_FC}"
    export CC="${ARIADNE_CC}"
    export CXX="${ARIADNE_CXX}"
    export FC="${ARIADNE_FC}"
    export CMAKE_BUILD_PARALLEL_LEVEL="${INSTALL_JOBS}"
    export MAKEFLAGS="-j${INSTALL_JOBS}"
    local ariadne_pip_config=""
    if [[ "${MACHINE}" == "csf3" ]]; then
        ariadne_pip_config=" --config-settings=cmake.define.ARIADNE_SAFE_IFX_FLAGS=ON"
    fi
    [[ "${DRY_RUN}" -eq 1 ]] && echo "+ export CC=${CC} CXX=${CXX} FC=${FC} CMAKE_BUILD_PARALLEL_LEVEL=${INSTALL_JOBS} MAKEFLAGS=-j${INSTALL_JOBS}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ clean ARIADNE-local build artefacts in ${ariadne_root}"
    else
        rm -rf "${ariadne_root}/_skbuild" "${ariadne_root}/build"
    fi
    # shellcheck disable=SC2086
    if ! run_in_dir "${ariadne_root}" "${PYTHON}" -m pip install . --no-build-isolation -v --force-reinstall --no-deps ${ariadne_pip_config}; then
        print_ariadne_manual_recovery_block
        die "ARIADNE install failed"
    fi
    unset CC CXX FC F77 F90
    unset MAKEFLAGS CMAKE_BUILD_PARALLEL_LEVEL
    print_ariadne_import_info "ARIADNE after install"
    verify_ariadne_api
    assert_ariadne_inside_venv
    write_ariadne_receipt
}

plumed_source_dir() {
    local extracted="${PROJECTS_DIR}/plumed-${PLUMED_VERSION}"
    local sources_dir="${PROJECTS_DIR}/_sources"
    local tarball="${sources_dir}/${PLUMED_TARBALL}"
    if [[ -d "${extracted}" ]]; then
        printf '%s\n' "${extracted}"
        return 0
    fi
    if [[ ! -f "${tarball}" ]]; then
        download_to "${PLUMED_URL}" "${tarball}"
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '%s\n' "${extracted}"
        return 0
    fi
    require_file "${tarball}" "PLUMED source tarball"
    run_cmd mkdir -p "${PROJECTS_DIR}"
    run_cmd tar -xzf "${tarball}" -C "${PROJECTS_DIR}"
    printf '%s\n' "${extracted}"
}

install_plumed_if_needed() {
    local reinstall="${1:-0}"
    if [[ "${SKIP_PLUMED}" -eq 1 ]]; then
        warn "Skipping PLUMED install by request"
        return 0
    fi
    note "Checking PLUMED"
    local default_lib="${HOME}/opt/plumed-${PLUMED_VERSION}/lib"
    local kernel="${PLUMED_KERNEL:-${default_lib}/libplumedKernel.so}"
    local library_path="${PLUMED_LIBRARY_PATH:-${default_lib}}"
    export PLUMED_KERNEL="${kernel}"
    export LD_LIBRARY_PATH="${library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    if [[ "${reinstall}" -eq 0 && "${DRY_RUN}" -eq 0 ]] && [[ -r "${kernel}" ]] && "${PYTHON}" -c "import os, plumed; p=plumed.Plumed(kernel=os.environ['PLUMED_KERNEL']); p.finalize()" >/dev/null 2>&1; then
        echo "PLUMED already usable"
    else
        deactivate_existing_venv "PLUMED GCC module setup"
        load_gcc_build_modules
        # shellcheck disable=SC1091
        create_or_activate_venv 0
        unset CC CXX FC F77 F90
        export CC=gcc
        export CXX=g++
        require_cmd gcc "Load a GCC compiler module first."
        require_cmd g++ "Load a GCC compiler module first."
        require_cmd make "Load a compiler/build module first."
        local src
        src="$(plumed_source_dir)"
        if [[ "${reinstall}" -eq 1 ]]; then
            run_shell_in_dir "${src}" "make clean || true"
        fi
        run_in_dir "${src}" ./configure "--prefix=${HOME}/opt/plumed-${PLUMED_VERSION}" --disable-external-blas --disable-external-lapack --disable-mpi
        run_in_dir "${src}" make -j "${INSTALL_JOBS}"
        run_in_dir "${src}" make install
        pip_install --force-reinstall "plumed==${PLUMED_VERSION}"
        unset CC CXX FC F77 F90
    fi
    export PLUMED_KERNEL="${kernel}"
    export LD_LIBRARY_PATH="${library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${PYTHON}" -c "import os, plumed; p=plumed.Plumed(kernel=os.environ['PLUMED_KERNEL']); p.finalize(); print('PLUMED OK')"
    else
        echo "+ export PLUMED_KERNEL=$(printf '%q' "${kernel}")"
    fi
}

install_ferebus_if_needed() {
    local reinstall="${1:-0}"
    note "Checking FEREBUS"
    if [[ "${reinstall}" -eq 0 && -x "${FEREBUS_PATH}" ]]; then
        echo "FEREBUS executable already present: ${FEREBUS_PATH}"
        return 0
    fi
    if [[ "${SKIP_FEREBUS_BUILD}" -eq 1 ]]; then
        die "FEREBUS executable missing at ${FEREBUS_PATH} and --skip-ferebus-build was set"
    fi
    local root="${PROJECTS_DIR}/FEREBUS_CPU"
    require_dir "${root}" "FEREBUS_CPU source tree"
    local openblas_a=""
    for candidate in "${root}/libs/openblas/lib64/libopenblas.a" "${root}/libs/openblas/lib/libopenblas.a"; do
        if [[ -f "${candidate}" ]]; then
            openblas_a="${candidate}"
            break
        fi
    done
    if [[ -z "${openblas_a}" ]]; then
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            warn "dry-run: FEREBUS static OpenBLAS is not present; install will require staged OpenBLAS or --allow-download"
        elif [[ "${ALLOW_DOWNLOAD}" -eq 1 ]]; then
            run_in_dir "${root}/libs" ./fetchOpenBlas.sh
        else
            die "FEREBUS static OpenBLAS is missing. Stage it under ${root}/libs/openblas or rerun with --allow-download."
        fi
    fi
    if [[ -z "${openblas_a}" && "${DRY_RUN}" -eq 0 ]]; then
        for candidate in "${root}/libs/openblas/lib64/libopenblas.a" "${root}/libs/openblas/lib/libopenblas.a"; do
            if [[ -f "${candidate}" ]]; then
                openblas_a="${candidate}"
                break
            fi
        done
    fi
    if [[ -z "${openblas_a}" && "${DRY_RUN}" -eq 0 ]]; then
        die "FEREBUS static OpenBLAS is still missing after setup"
    fi
    load_gcc_build_modules
    require_cmd cmake "Load the CSF CMake module first."
    require_cmd gfortran "Load a GCC compiler module first."
    local build_dir="${root}/build-ichor-install"
    case "${build_dir}" in
        "${root}/build-ichor-install") ;;
        *) die "refusing to clean unexpected FEREBUS build directory: ${build_dir}" ;;
    esac
    if [[ "${reinstall}" -eq 1 ]]; then
        run_cmd rm -rf "${build_dir}"
        backup_existing_path "${FEREBUS_PATH}" "FEREBUS executable"
    fi
    run_cmd mkdir -p "${build_dir}" "$(dirname "${FEREBUS_PATH}")"
    run_cmd cmake -S "${root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
    run_cmd cmake --build "${build_dir}" -j "${INSTALL_JOBS}"
    if [[ -f "${build_dir}/ferebus" ]]; then
        run_cmd cp "${build_dir}/ferebus" "${FEREBUS_PATH}"
        run_cmd chmod 755 "${FEREBUS_PATH}"
    else
        run_cmd cmake --install "${build_dir}"
    fi
    [[ "${DRY_RUN}" -eq 1 || -x "${FEREBUS_PATH}" ]] || die "FEREBUS build did not create executable: ${FEREBUS_PATH}"
}

upsert_ichor_config() {
    note "Updating ~/ichor_config.yaml"
    local venv_config
    local aimall_config
    local ferebus_config
    local plumed_kernel_config
    local plumed_lib_config
    venv_config="$(config_path_for_yaml "${VENV}/bin/python")"
    aimall_config="$(config_path_for_yaml "${AIMALL_PATH}")"
    ferebus_config="$(config_path_for_yaml "${FEREBUS_PATH}")"
    plumed_kernel_config="$(config_path_for_yaml "${PLUMED_KERNEL:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib/libplumedKernel.so}")"
    plumed_lib_config="$(config_path_for_yaml "${PLUMED_LIBRARY_PATH:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib}")"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ upsert ${MACHINE} profile in ~/ichor_config.yaml"
        return 0
    fi
    "${PYTHON}" - "${MACHINE}" "${venv_config}" "${aimall_config}" "${ferebus_config}" "${plumed_kernel_config}" "${plumed_lib_config}" <<'PY'
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import yaml

machine, python_path, aimall_path, ferebus_path, plumed_kernel, plumed_lib = sys.argv[1:7]
path = Path.home() / "ichor_config.yaml"
data = {}
if path.exists():
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise SystemExit("~/ichor_config.yaml must contain a YAML mapping")
    data = loaded
    backup = path.with_name(path.name + ".bak." + str(int(time.time())))
    shutil.copy2(path, backup)
    print(f"Backed up existing config to {backup}")

def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value

if machine == "csf3":
    profile = {
        "hpc": {
            "scheduler": "slurm",
            "jobscript_shebang": "#!/bin/bash --login",
            "max_array_task_id": 25000,
            "memory_per_core_gb": 8,
            "memory_per_core_gb_by_partition": {
                "multicore": 8,
                "interactive": 8,
                "serial": 5,
                "multicore_small": 5,
                "himem": 32,
            },
            "parallel_environments": {
                "serial": [1, 1],
                "multicore": [2, 168],
                "interactive": [1, 168],
                "multicore_small": [2, 32],
                "himem": [1, 32],
            },
            "partitions": {
                "multicore": {
                    "min_cpus": 2,
                    "max_cpus": 168,
                    "memory_per_core_gb": 8,
                    "max_walltime_hours": 168,
                    "daemon_supported": True,
                },
                "interactive": {
                    "min_cpus": 1,
                    "max_cpus": 168,
                    "memory_per_core_gb": 8,
                    "max_walltime_hours": 24,
                    "daemon_supported": True,
                },
                "multicore_small": {
                    "min_cpus": 2,
                    "max_cpus": 32,
                    "memory_per_core_gb": 5,
                    "max_walltime_hours": 168,
                    "daemon_supported": True,
                },
                "serial": {
                    "min_cpus": 1,
                    "max_cpus": 1,
                    "memory_per_core_gb": 5,
                    "max_walltime_hours": 168,
                    "daemon_supported": True,
                },
                "himem": {
                    "min_cpus": 1,
                    "max_cpus": 32,
                    "memory_per_core_gb": 32,
                    "max_walltime_hours": 168,
                    "daemon_supported": True,
                },
            },
        },
        "software": {
            "python": {"env_name": "ichor-csf3", "python_path": python_path, "modules": []},
            "gaussian": {
                "executable_path": "$g09root/g09/g09",
                "modules": ["apps/binapps/gaussian/g09d01_em64t"],
            },
            "aimall": {"executable_path": aimall_path},
            "ferebus": {"executable_path": ferebus_path, "pyferebus_platform": "CSF3"},
            "ariadne_runtime": {
                "modules": [
                    "compilers/intel/oneapi/2025.0.1",
                    "umf compiler-rt tbb compiler",
                    "mkl/2025.0",
                ]
            },
            "plumed": {"kernel_path": plumed_kernel, "library_path": plumed_lib, "modules": []},
        },
    }
else:
    profile = {
        "hpc": {
            "scheduler": "slurm",
            "jobscript_shebang": "#!/bin/bash",
            "max_array_task_id": 25000,
            "memory_per_core_gb": 4,
            "memory_per_core_gb_by_partition": {
                "serial": 4,
                "multicore": 4,
                "multinode": 4,
            },
            "parallel_environments": {"serial": [1, 1], "multicore": [2, 40]},
            "partitions": {
                "serial": {
                    "min_cpus": 1,
                    "max_cpus": 1,
                    "memory_per_core_gb": 4,
                    "max_walltime_hours": 168,
                    "daemon_supported": True,
                },
                "multicore": {
                    "min_cpus": 2,
                    "max_cpus": 40,
                    "memory_per_core_gb": 4,
                    "max_walltime_hours": 168,
                    "daemon_supported": True,
                },
                "multinode": {
                    "min_cpus": 2,
                    "max_cpus": 10000,
                    "memory_per_core_gb": 4,
                    "max_walltime_hours": 168,
                    "daemon_supported": False,
                },
            },
        },
        "software": {
            "python": {
                "env_name": "ichor-csf4",
                "python_path": python_path,
                "modules": ["python/3.11.3-gcccore-12.3.0"],
            },
            "gaussian": {
                "executable_path": "$g16root/g16/g16",
                "modules": ["gaussian/g16c01_em64t_detectcpu"],
            },
            "aimall": {"executable_path": aimall_path},
            "ferebus": {"executable_path": ferebus_path, "pyferebus_platform": "CSF4"},
            "ariadne_runtime": {
                "modules": [
                    "compilers/oneapi/2024.2.0",
                    "compiler-rt tbb compiler",
                    "mkl/2024.2",
                ]
            },
            "plumed": {"kernel_path": plumed_kernel, "library_path": plumed_lib, "modules": []},
        },
    }

current = data.setdefault(machine, {})
deep_update(current, profile)
path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
print(f"Updated {path} profile {machine}")
PY
}

verify_operator_backends() {
    note "Verifying Gaussian/AIMAll operator backends"
    if [[ ! -x "${AIMALL_PATH}" ]]; then
        die "AIMAll executable is missing or not executable: ${AIMALL_PATH}. Re-run with --aimall-path PATH after installing AIMAll."
    fi
    if [[ ! -x "${FEREBUS_PATH}" ]]; then
        die "FEREBUS executable is missing or not executable: ${FEREBUS_PATH}"
    fi
    echo "AIMAll: ${AIMALL_PATH}"
    echo "FEREBUS: ${FEREBUS_PATH}"
}

require_yaml_available() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ ${PYTHON} -c 'import yaml'"
        return 0
    fi
    "${PYTHON}" -c "import yaml" >/dev/null 2>&1 || die "PyYAML is not installed in ${VENV}. Run --only packages before --only config."
}

final_checks() {
    local label="${1:-install}"
    note "Running final verification"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ final import checks and ichor-al-daemon preflight"
        return 0
    fi
    export ICHOR_MACHINE="${MACHINE}"
    "${PYTHON}" -c "import ichor.core, ichor.hpc, ichor.cli; print('ICHOR packages OK')"
    "${PYTHON}" -c "import polus.samplers.RS.randomSampling; print('POLUS RS OK')"
    "${PYTHON}" -c "import pyferebus.executors.trainer; print('pyferebus OK')"
    verify_ariadne_api
    "${PYTHON}" -c "import ase, rdkit, tqdm, portalocker; print('ASE/RDKit/tqdm/portalocker OK')"
    "${PYTHON}" -c "from xtb.ase.calculator import XTB; print('xTB OK')"
    "${PYTHON}" -c "from ichor.hpc.runtime_preflight import ensure_xtb_ase_available; ensure_xtb_ase_available(run_energy=True); print('xTB smoke OK')"
    if [[ "${SKIP_PLUMED}" -eq 0 ]]; then
        "${PYTHON}" -c "from ichor.hpc.runtime_preflight import ensure_plumed_available; ensure_plumed_available(run_ase_smoke=True); print('PLUMED smoke OK')"
    fi
    verify_operator_backends
    local smoke_dir
    smoke_dir="$(mktemp -d)"
    cat > "${smoke_dir}/campaign.yaml" <<'EOF'
schema_version: 3
max_iterations: 1
EOF
    "${VENV}/bin/ichor-al-daemon" preflight --campaign-dir "${smoke_dir}"
    echo ""
    if [[ "${label}" == "verify" ]]; then
        echo "Verification complete."
    else
        echo "Install complete."
    fi
    echo "Enter the runtime environment with:"
    echo "  source ${REPO_ROOT}/scripts/env_ichor_csf.sh ${MACHINE} --smoke"
}

verify_entrypoints() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ command -v ichor-cli"
        echo "+ command -v ichor-al-daemon"
        return 0
    fi
    command -v ichor-cli >/dev/null 2>&1 || die "ichor-cli is not on PATH after activating ${VENV}"
    command -v ichor-al-daemon >/dev/null 2>&1 || die "ichor-al-daemon is not on PATH after activating ${VENV}"
}

path_report_command() {
    local name="$1"
    local resolved
    resolved="$(ichor_csf_command_path "${name}")"
    if [[ -z "${resolved}" ]]; then
        echo "${name}: missing"
        return 0
    fi
    if ichor_csf_path_inside "${resolved}" "${VENV}"; then
        echo "${name}: ${resolved} (target venv)"
    else
        echo "${name}: ${resolved} (outside target venv)"
    fi
}

doctor_python_imports() {
    if [[ ! -x "${VENV}/bin/python" ]]; then
        echo "Python import checks: skipped; target venv Python missing"
        return 0
    fi
    PYTHON="${VENV}/bin/python"
    print_ariadne_import_info "ARIADNE doctor"
    doctor_ariadne_receipt
    if [[ -n "${PLUMED_KERNEL:-}" ]]; then
        "${PYTHON}" - <<'PY' || true
import os
try:
    import plumed
    p = plumed.Plumed(kernel=os.environ["PLUMED_KERNEL"])
    p.finalize()
    print("PLUMED wrapper/kernel: OK")
except Exception as exc:
    print(f"PLUMED wrapper/kernel: failed ({exc})")
PY
    fi
}

doctor_load_ariadne_modules() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ module purge"
        if [[ "${MACHINE}" == "csf3" ]]; then
            echo "+ module load compilers/intel/oneapi/2025.0.1"
            echo "+ module load umf compiler-rt tbb compiler"
            echo "+ module load mkl/2025.0"
        else
            echo "+ module load python/3.11.3-gcccore-12.3.0"
            echo "+ module load python-bundle-pypi/2023.06-gcccore-12.3.0"
            echo "+ module load compilers/oneapi/2024.2.0"
            echo "+ module load compiler-rt tbb compiler"
            echo "+ module load mkl/2024.2"
        fi
        return 0
    fi
    if ! ichor_csf_module purge; then
        echo "module purge: failed"
        return 1
    fi
    if [[ "${MACHINE}" == "csf3" ]]; then
        ichor_csf_module load compilers/intel/oneapi/2025.0.1 || return 1
        ichor_csf_module load umf compiler-rt tbb compiler || return 1
        ichor_csf_module load mkl/2025.0 || return 1
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    else
        ichor_csf_module load python/3.11.3-gcccore-12.3.0 || return 1
        ichor_csf_module load python-bundle-pypi/2023.06-gcccore-12.3.0 || return 1
        ichor_csf_module load compilers/oneapi/2024.2.0 || return 1
        ichor_csf_module load compiler-rt tbb compiler || return 1
        ichor_csf_module load mkl/2024.2 || return 1
    fi
}

doctor_print_ariadne_compiler_discovery() {
    local compiler result resolved method
    for compiler in icx icpx ifx; do
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            echo "${compiler}: <dry-run>"
            continue
        fi
        result="$(ichor_csf_find_ariadne_compiler_path "${compiler}" || true)"
        resolved="${result%%|*}"
        method="${result#*|}"
        if [[ -n "${resolved}" ]]; then
            echo "${compiler}: ${resolved} (${method})"
        else
            echo "${compiler}: unresolved"
        fi
    done
}

stage_doctor() {
    CURRENT_STAGE="doctor"
    note "Running CSF doctor diagnostics"
    echo "machine: ${MACHINE}"
    echo "repo root: ${REPO_ROOT}"
    echo "projects dir: ${PROJECTS_DIR}"
    echo "venv: ${VENV}"
    echo "python prefix: ${PYTHON_PREFIX}"
    echo "module shell function before init: $(type -t module 2>/dev/null || echo unavailable)"
    initialise_modules || true
    echo "module shell function after init: $(type -t module 2>/dev/null || echo unavailable)"
    echo "module init: ${ICHOR_CSF_MODULE_INIT:-unresolved}"
    if module_is_current_shell_function; then
        module list || true
    fi

    echo ""
    echo "ARIADNE compiler discovery:"
    if doctor_load_ariadne_modules; then
        doctor_print_ariadne_compiler_discovery
    else
        echo "ARIADNE module load failed; compiler discovery may be incomplete"
    fi

    echo ""
    echo "Target venv paths:"
    if [[ -f "${VENV}/bin/activate" ]]; then
        activate_existing_venv || true
        ichor_csf_warn_path_hazards "${VENV}"
        path_report_command python
        path_report_command pip
        path_report_command ichor-cli
        path_report_command ichor-al-daemon
    else
        echo "target venv activation script missing: ${VENV}/bin/activate"
    fi

    echo ""
    echo "Backend paths:"
    echo "AIMAll: ${AIMALL_PATH} $([[ -x "${AIMALL_PATH}" ]] && echo executable || echo missing-or-not-executable)"
    echo "FEREBUS: ${FEREBUS_PATH} $([[ -x "${FEREBUS_PATH}" ]] && echo executable || echo missing-or-not-executable)"
    if [[ "${MACHINE}" == "csf3" ]]; then
        echo "Gaussian expected: apps/binapps/gaussian/g09d01_em64t / \$g09root/g09/g09"
    else
        echo "Gaussian expected: gaussian/g16c01_em64t_detectcpu / \$g16root/g16/g16"
    fi
    export PLUMED_KERNEL="${PLUMED_KERNEL:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib/libplumedKernel.so}"
    export PLUMED_LIBRARY_PATH="${PLUMED_LIBRARY_PATH:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib}"
    echo "PLUMED_KERNEL: ${PLUMED_KERNEL} $([[ -r "${PLUMED_KERNEL}" ]] && echo readable || echo missing-or-unreadable)"
    echo "PLUMED_LIBRARY_PATH: ${PLUMED_LIBRARY_PATH}"
    doctor_python_imports
}

require_all_sibling_repos() {
    require_dir "${PROJECTS_DIR}/POLUS" "POLUS sibling repo"
    require_dir "${PROJECTS_DIR}/FEREBUS_CPU" "FEREBUS_CPU sibling repo"
    require_dir "${PROJECTS_DIR}/ARIADNE" "ARIADNE sibling repo"
}

stage_python() {
    CURRENT_STAGE="python"
    prepare_python_and_venv 1 1
    pip_install --upgrade pip setuptools wheel
}

stage_packages() {
    CURRENT_STAGE="packages"
    ensure_repo_root
    require_dir "${PROJECTS_DIR}/POLUS" "POLUS sibling repo"
    require_dir "${PROJECTS_DIR}/FEREBUS_CPU" "FEREBUS_CPU sibling repo"
    prepare_python_and_venv 0 0
    install_python_packages
    verify_entrypoints
}

stage_ariadne() {
    CURRENT_STAGE="ariadne"
    require_dir "${PROJECTS_DIR}/ARIADNE" "ARIADNE sibling repo"
    prepare_python_and_venv 0 0
    install_ariadne_if_needed 1
}

stage_plumed() {
    CURRENT_STAGE="plumed"
    prepare_python_and_venv 0 0
    install_plumed_if_needed 1
}

stage_ferebus() {
    CURRENT_STAGE="ferebus"
    require_dir "${PROJECTS_DIR}/FEREBUS_CPU" "FEREBUS_CPU sibling repo"
    install_ferebus_if_needed 1
}

stage_config() {
    CURRENT_STAGE="config"
    deactivate_existing_venv "config module setup"
    load_python_stack
    if [[ "${MACHINE}" == "csf3" ]]; then
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    activate_existing_venv
    require_yaml_available
    if [[ "${SKIP_PLUMED}" -eq 0 ]]; then
        export PLUMED_KERNEL="${PLUMED_KERNEL:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib/libplumedKernel.so}"
        export PLUMED_LIBRARY_PATH="${PLUMED_LIBRARY_PATH:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib}"
    fi
    upsert_ichor_config
}

stage_verify() {
    CURRENT_STAGE="verify"
    prepare_runtime_environment
    final_checks verify
}

stage_all() {
    CURRENT_STAGE="all"
    ensure_repo_root
    require_all_sibling_repos
    prepare_python_and_venv 0 0
    install_python_packages
    install_ariadne_if_needed 0
    install_plumed_if_needed 0
    install_ferebus_if_needed 0
    prepare_runtime_environment
    require_yaml_available
    upsert_ichor_config
    final_checks install
}

main() {
    MACHINE="$(detect_machine)"
    if [[ -z "${VENV}" ]]; then
        VENV="${HOME}/.venv/ichor-${MACHINE}"
    fi
    VENV="$(expand_path "${VENV}")"
    local log_dir="${HOME}/.cache/ichor-al-install"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        mkdir -p "${log_dir}"
        LOG_FILE="${log_dir}/install-${MACHINE}-$(date +%Y%m%d-%H%M%S).log"
        exec > >(tee -a "${LOG_FILE}") 2>&1
        echo "Writing install log to ${LOG_FILE}"
        if [[ "${DEBUG_TRACE}" -eq 1 ]]; then
            export PS4='+ ${BASH_SOURCE##*/}:${LINENO}:${FUNCNAME[0]:-main}: '
            set -x
        fi
    fi

    note "Install settings"
    cat <<EOF
machine       = ${MACHINE}
repo root     = ${REPO_ROOT}
projects dir  = ${PROJECTS_DIR}
venv          = ${VENV}
jobs          = ${INSTALL_JOBS}
only          = ${ONLY_STAGE}
downloads     = ${ALLOW_DOWNLOAD}
dry run       = ${DRY_RUN}
EOF

    if [[ "${ONLY_STAGE}" != "verify" && "${ONLY_STAGE}" != "config" && "${ONLY_STAGE}" != "doctor" ]]; then
        print_download_readiness
    fi

    case "${ONLY_STAGE}" in
        all) stage_all ;;
        python) stage_python ;;
        packages) stage_packages ;;
        ariadne) stage_ariadne ;;
        plumed) stage_plumed ;;
        ferebus) stage_ferebus ;;
        config) stage_config ;;
        verify) stage_verify ;;
        doctor) stage_doctor ;;
    esac
}

main "$@"
