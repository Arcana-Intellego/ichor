#!/usr/bin/env bash
#
# Install ICHOR active-learning dependencies on supported cluster platforms.
#
# The script is intentionally conservative:
#   * downloads are opt-in via --allow-download;
#   * Gaussian and AIMAll are verified, not installed;
#   * ~/ichor_config.yaml is backed up before the active profile is upserted;
#   * Intel compiler variables used for ARIADNE are explicitly cleared before
#     the PLUMED GCC build.

set -euo pipefail

PYTHON_VERSION="3.11.15"
PLUMED_VERSION="2.10.0"
BINUTILS_VERSION="2.42"
OPENSSL_VERSION="1.1.1w"
READLINE_VERSION="8.2"
SQLITE_VERSION="3450300"
CMAKE_VERSION="3.31.6"
PYTHON_TARBALL="Python-${PYTHON_VERSION}.tgz"
PLUMED_TARBALL="plumed-${PLUMED_VERSION}.tgz"
BINUTILS_TARBALL="binutils-${BINUTILS_VERSION}.tar.xz"
OPENSSL_TARBALL="openssl-${OPENSSL_VERSION}.tar.gz"
READLINE_TARBALL="readline-${READLINE_VERSION}.tar.gz"
SQLITE_TARBALL="sqlite-autoconf-${SQLITE_VERSION}.tar.gz"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_TARBALL}"
PLUMED_URL="https://github.com/plumed/plumed2/releases/download/v${PLUMED_VERSION}/${PLUMED_TARBALL}"
BINUTILS_URL="https://ftp.gnu.org/gnu/binutils/${BINUTILS_TARBALL}"
OPENSSL_URL="https://github.com/openssl/openssl/releases/download/OpenSSL_1_1_1w/${OPENSSL_TARBALL}"
READLINE_URL="https://ftp.gnu.org/gnu/readline/${READLINE_TARBALL}"
SQLITE_URL="https://sqlite.org/2024/${SQLITE_TARBALL}"
ARIADNE_URL="https://github.com/Arcana-Intellego/ARIADNE.git"
FEREBUS_URL="https://github.com/Arcana-Intellego/FEREBUS_CPU.git"
PYTHON_SHA256="f4de1b10bd6c70cbb9fa1cd71fc5038b832747a74ee59d599c69ce4846defb50"
PLUMED_SHA256="5aaf718ac530a1c8df6e0644c22acc84ad4202778106a1d584477057775f2995"
BINUTILS_SHA256="f6e4d41fd5fc778b06b7891457b3620da5ecea1006c6a4a41ae998109f85a800"
OPENSSL_SHA256="cf3098950cb4d853ad95c0841f1f9c6d3dc102dccfcacd521d93925208b76ac8"
READLINE_SHA256="3feb7171f16a84ee82ca18a36d7b9be109a52c04f492a053331d7d1095007c35"
SQLITE_SHA256="b2809ca53124c19c60f42bf627736eae011afdcc205bb48270a5ee9a38191531"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FFLUXLAB_PYTHON_CONSTRAINTS="${SCRIPT_DIR}/constraints/ffluxlab-python311.txt"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_ichor.sh"

MACHINE="${ICHOR_MACHINE:-auto}"
PROJECTS_DIR="${ICHOR_PROJECTS_DIR:-${HOME}/projects}"
REPO_ROOT="${ICHOR_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
VENV="${ICHOR_VENV:-}"
PYTHON_PREFIX="${PYTHON_PREFIX:-${HOME}/opt/python-${PYTHON_VERSION}}"
PYTHON_DEPS_PREFIX="${ICHOR_PYTHON_DEPS_PREFIX:-${HOME}/opt/ichor-python-deps-${PYTHON_VERSION}}"
OPENSSL_PREFIX="${ICHOR_OPENSSL_PREFIX:-${HOME}/opt/openssl-${OPENSSL_VERSION}}"
BINUTILS_PREFIX="${ICHOR_BINUTILS_PREFIX:-${HOME}/opt/binutils-${BINUTILS_VERSION}}"
AIMALL_PATH="${AIMALL_PATH:-}"
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
Usage: scripts/install_ichor.sh [options]

Options:
  --machine auto|csf3|csf4|ffluxlab
                                  default: auto
  --repo-root PATH                default: script-derived repo root
  --projects-dir PATH             default: ~/projects
  --venv PATH                     default: ~/.venv/ichor-MACHINE
  --python-prefix PATH            CSF3/ffluxlab default: ~/opt/python-3.11.15
  --only all|python|packages|ariadne|plumed|ferebus|config|verify|doctor
                                  default: all. A component stage repairs or
                                  reinstalls that component.
  --allow-download                permit pinned dependency and missing source downloads
  --skip-plumed                   skip native PLUMED build and smoke
  --skip-ariadne-build            only verify import ariadne
  --skip-ferebus-build            only verify configured ferebus executable
  --aimall-path PATH              default: aimall on ffluxlab,
                                           ~/AIMAll/aimqb.ish on CSF
  --ferebus-path PATH             default: ~/.local/bin/ferebus
  --jobs N                        default: 4
  --yes                           non-interactive mode
  --dry-run                       print planned actions without installing
  --debug, --trace                print shell trace in the install log
  -h, --help                      show this help

Environment equivalents:
  ICHOR_MACHINE, ICHOR_PROJECTS_DIR, ICHOR_REPO_ROOT, ICHOR_VENV,
  ICHOR_INSTALL_JOBS, ICHOR_BINUTILS_PREFIX, AIMALL_PATH, FEREBUS_PATH,
  PLUMED_KERNEL, PLUMED_LIBRARY_PATH, PIP_FIND_LINKS.
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
    auto|csf3|csf4|ffluxlab) ;;
    *) die "--machine must be auto, csf3, csf4, or ffluxlab" ;;
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
PYTHON_DEPS_PREFIX="$(expand_path "${PYTHON_DEPS_PREFIX}")"
OPENSSL_PREFIX="$(expand_path "${OPENSSL_PREFIX}")"
BINUTILS_PREFIX="$(expand_path "${BINUTILS_PREFIX}")"
if [[ -n "${AIMALL_PATH}" ]]; then
    AIMALL_PATH="$(expand_path "${AIMALL_PATH}")"
fi
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

isolate_target_python_environment() {
    [[ "${MACHINE}" == "csf4" ]] || return 0
    ichor_csf_isolate_python_environment
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
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        [[ -d "${path}" ]] || warn "dry-run: ${label} not found yet: ${path}"
        return 0
    fi
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
        if [[ "${cmd}" == "icx" || "${cmd}" == "icpx" || "${cmd}" == "ifx" \
            || "${cmd}" == "icc" || "${cmd}" == "icpc" \
            || "${cmd}" == "ifort" || "${hint}" == ARIADNE* ]]; then
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

ariadne_fortran_compiler_name() {
    if [[ "${MACHINE}" == "ffluxlab" ]]; then
        printf 'ifort\n'
    else
        printf 'ifx\n'
    fi
}

ariadne_c_compiler_name() {
    if [[ "${MACHINE}" == "ffluxlab" ]]; then
        printf 'icc\n'
    else
        printf 'icx\n'
    fi
}

ariadne_cxx_compiler_name() {
    if [[ "${MACHINE}" == "ffluxlab" ]]; then
        printf 'icpc\n'
    else
        printf 'icpx\n'
    fi
}

ensure_ariadne_compilers_on_path() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        return 0
    fi
    local compiler result resolved method bin_dir missing=0
    local c_compiler cxx_compiler fortran_compiler
    c_compiler="$(ariadne_c_compiler_name)"
    cxx_compiler="$(ariadne_cxx_compiler_name)"
    fortran_compiler="$(ariadne_fortran_compiler_name)"
    for compiler in "${c_compiler}" "${cxx_compiler}" "${fortran_compiler}"; do
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
            icx|icc) ARIADNE_CC="${resolved}"; ARIADNE_CC_METHOD="${method}" ;;
            icpx|icpc) ARIADNE_CXX="${resolved}"; ARIADNE_CXX_METHOD="${method}" ;;
            ifx|ifort) ARIADNE_FC="${resolved}"; ARIADNE_FC_METHOD="${method}" ;;
        esac
    done
    hash -r 2>/dev/null || true
    if [[ "${missing}" -ne 0 ]]; then
        module_debug "ARIADNE compiler check after module load"
        die "ARIADNE compiler modules did not expose ${c_compiler}/${cxx_compiler}/${fortran_compiler}"
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
    ichor_detect_machine "${MACHINE}" \
        || die "could not auto-detect the platform. Pass --machine csf3, csf4, or ffluxlab."
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

file_sha256() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{print tolower($1)}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${path}" | awk '{print tolower($1)}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "${path}" | awk '{print tolower($NF)}'
    else
        die "sha256sum, shasum, or openssl is required to verify source archives"
    fi
}

verify_sha256() {
    local path="$1"
    local expected="${2,,}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ verify SHA-256 ${expected} ${path}" >&2
        return 0
    fi
    require_file "${path}" "source archive"
    local observed
    observed="$(file_sha256 "${path}")"
    [[ "${observed}" == "${expected}" ]] || die "SHA-256 mismatch for ${path}: expected ${expected}, observed ${observed}"
}

download_to() {
    local url="$1"
    local dest="$2"
    local expected_sha256="$3"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ download ${url} -> ${dest}, then verify SHA-256 ${expected_sha256}" >&2
        return 0
    fi
    [[ "${ALLOW_DOWNLOAD}" -eq 1 ]] || die "download disabled for ${url}; rerun with --allow-download or stage ${dest}"
    mkdir -p "$(dirname "${dest}")"
    local partial="${dest}.partial.$$"
    rm -f "${partial}"
    if command -v curl >/dev/null 2>&1; then
        run_cmd curl -fL "${url}" -o "${partial}"
    elif command -v wget >/dev/null 2>&1; then
        run_cmd wget -O "${partial}" "${url}"
    else
        die "neither curl nor wget is available for download: ${url}"
    fi
    verify_sha256 "${partial}" "${expected_sha256}"
    run_cmd mv "${partial}" "${dest}"
}

ensure_public_checkout() {
    local path="$1"
    local url="$2"
    local branch="$3"
    local label="$4"
    if [[ -e "${path}" || -L "${path}" ]]; then
        [[ -d "${path}" && ! -L "${path}" ]] \
            || die "${label} source path exists but is not a regular directory: ${path}"
        echo "${label} source already present; leaving it unchanged: ${path}"
        return 0
    fi
    [[ "${ALLOW_DOWNLOAD}" -eq 1 ]] \
        || die "${label} source is missing at ${path}; rerun with --allow-download to clone ${url}"
    require_cmd git "Git is required to obtain missing source repositories."
    run_cmd mkdir -p "$(dirname "${path}")"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ GIT_TERMINAL_PROMPT=0 git clone --branch ${branch} --single-branch ${url} ${path}"
        return 0
    fi
    GIT_TERMINAL_PROMPT=0 git clone \
        --branch "${branch}" \
        --single-branch \
        "${url}" \
        "${path}"
}

ensure_required_source_checkouts() {
    ensure_public_checkout \
        "${PROJECTS_DIR}/ARIADNE" \
        "${ARIADNE_URL}" \
        main \
        ARIADNE
    ensure_public_checkout \
        "${PROJECTS_DIR}/FEREBUS_CPU" \
        "${FEREBUS_URL}" \
        restore-gradient-refinement \
        FEREBUS_CPU
}

print_download_readiness() {
    note "Checking download readiness"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        warn "dry-run: network checks skipped"
    else
        local -a sources=(
            "PyPI|https://pypi.org/simple/" \
            "Python|${PYTHON_URL}" \
            "PLUMED|${PLUMED_URL}" \
            "OpenSSL|${OPENSSL_URL}" \
            "Readline|${READLINE_URL}" \
            "SQLite|${SQLITE_URL}" \
            "ARIADNE|${ARIADNE_URL}" \
            "FEREBUS_CPU|${FEREBUS_URL}"
        )
        if [[ "${MACHINE}" == "ffluxlab" ]]; then
            sources+=("GNU Binutils|${BINUTILS_URL}")
        fi
        for label_url in "${sources[@]}"; do
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
            (the installer verifies it before extraction)
  * FEREBUS: retain the bundled libs/openblas tree from the FEREBUS_CPU
            checkout; the installer never replaces or rebuilds it.
  * ffluxlab linker: place ${BINUTILS_TARBALL} in ${PROJECTS_DIR}/_sources/;
            the installer verifies it before building private linker tools.
  * ffluxlab Python dependencies: place ${OPENSSL_TARBALL},
            ${READLINE_TARBALL}, and ${SQLITE_TARBALL} in
            ${PROJECTS_DIR}/_sources/.
  * ARIADNE/FEREBUS_CPU: clone the public HTTPS repositories into
            ${PROJECTS_DIR}/ARIADNE and ${PROJECTS_DIR}/FEREBUS_CPU.
  * Python wheels: set PIP_FIND_LINKS=/path/to/wheelhouse before running this script.
EOF
}

python_import_ok() {
    local module_name="$1"
    isolate_target_python_environment
    "${PYTHON}" -c "import ${module_name}" >/dev/null 2>&1
}

run_ariadne_python() {
    if [[ "${MACHINE}" == "ffluxlab" ]]; then
        [[ -n "${ICHOR_ARIADNE_LD_PRELOAD:-}" ]] \
            || die "ffluxlab ARIADNE MKL preload contract is unavailable"
        env \
            LD_PRELOAD="${ICHOR_ARIADNE_LD_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}" \
            "${PYTHON}" "$@"
    elif [[ "${MACHINE}" == "csf4" ]]; then
        ichor_csf_run_isolated_python "${PYTHON}" "$@"
    else
        "${PYTHON}" "$@"
    fi
}

ariadne_import_ok() {
    run_ariadne_python -c "import ariadne" >/dev/null 2>&1
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
        isolate_target_python_environment
    elif [[ "${MACHINE}" == "ffluxlab" ]]; then
        module_cmd purge
        module_cmd load compilers/gcc/11.1.0
    fi
}

resolve_ffluxlab_intel_runtime() {
    [[ "${MACHINE}" == "ffluxlab" ]] || return 0
    local configured_root="/home/modules/compilers/intel/21.0.3"
    local intel_root libimf libmkl libmkl_sequential libmkl_core
    local runtime_dir mkl_runtime_dir
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ resolve 64-bit Intel and MKL runtimes beneath ${configured_root}"
        export ICHOR_INTEL_RUNTIME_DIR="<resolved-intel64-runtime>"
        export ICHOR_MKL_RUNTIME_DIR="<resolved-mkl-intel64-runtime>"
        return 0
    fi
    intel_root="$(readlink -f "${configured_root}" 2>/dev/null || true)"
    [[ -n "${intel_root}" && -d "${intel_root}" ]] \
        || die "ffluxlab Intel module root is unavailable: ${configured_root}"
    libimf="$(
        find "${intel_root}" \
            \( -type f -o -type l \) \
            -path '*/intel64_lin/libimf.so' -print -quit
    )"
    [[ -n "${libimf}" ]] \
        || die "64-bit libimf.so is absent beneath ${intel_root}"
    libmkl="${MKLROOT:+${MKLROOT}/lib/intel64/libmkl_intel_lp64.so.1}"
    if [[ -z "${libmkl}" || ! -e "${libmkl}" ]]; then
        libmkl="$(
            find "${intel_root}" \
                \( -type f -o -type l \) \
                -path '*/mkl/*/lib/intel64/libmkl_intel_lp64.so.1' \
                -print -quit
        )"
    fi
    [[ -n "${libmkl}" ]] \
        || die "64-bit libmkl_intel_lp64.so.1 is absent beneath ${intel_root}"
    runtime_dir="$(dirname "${libimf}")"
    mkl_runtime_dir="$(dirname "${libmkl}")"
    libmkl_sequential="${mkl_runtime_dir}/libmkl_sequential.so.1"
    libmkl_core="${mkl_runtime_dir}/libmkl_core.so.1"
    [[ -e "${libmkl_sequential}" && -e "${libmkl_core}" ]] \
        || die "complete Intel MKL runtime is unavailable beneath ${mkl_runtime_dir}"
    export LD_LIBRARY_PATH="${runtime_dir}:${mkl_runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LIBRARY_PATH="${runtime_dir}:${mkl_runtime_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export ICHOR_INTEL_RUNTIME_DIR="${runtime_dir}"
    export ICHOR_MKL_RUNTIME_DIR="${mkl_runtime_dir}"
    export ICHOR_ARIADNE_LD_PRELOAD="${libmkl}:${libmkl_sequential}:${libmkl_core}"
}

resolve_ffluxlab_gcc_runtime() {
    [[ "${MACHINE}" == "ffluxlab" ]] || return 0
    local configured_root="/home/modules/compilers/gcc/11.1.0"
    local gcc_root libstdcxx runtime_dir
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ resolve GCC 11 libstdc++ runtime beneath ${configured_root}"
        export ICHOR_GCC_RUNTIME_DIR="<resolved-gcc11-runtime>"
        return 0
    fi
    gcc_root="$(readlink -f "${configured_root}" 2>/dev/null || true)"
    [[ -n "${gcc_root}" && -d "${gcc_root}" ]] \
        || die "ffluxlab GCC module root is unavailable: ${configured_root}"
    libstdcxx="${gcc_root}/lib64/libstdc++.so.6"
    if [[ ! -e "${libstdcxx}" ]]; then
        libstdcxx="$(
            find "${gcc_root}" \
                \( -type f -o -type l \) \
                -path '*/lib64/libstdc++.so.6' \
                -print -quit
        )"
    fi
    [[ -n "${libstdcxx}" && -e "${libstdcxx}" ]] \
        || die "GCC 11 libstdc++.so.6 is absent beneath ${gcc_root}"
    runtime_dir="$(dirname "${libstdcxx}")"
    export LD_LIBRARY_PATH="${runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LIBRARY_PATH="${runtime_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export ICHOR_GCC_RUNTIME_DIR="${runtime_dir}"
}

load_ariadne_modules() {
    module_cmd purge
    if [[ "${MACHINE}" == "csf3" ]]; then
        module_cmd load compilers/intel/oneapi/2025.0.1
        module_cmd load umf compiler-rt tbb compiler
        module_cmd load mkl/2025.0
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    elif [[ "${MACHINE}" == "csf4" ]]; then
        module_cmd load python/3.11.3-gcccore-12.3.0
        module_cmd load compilers/oneapi/2024.2.0
        module_cmd load compiler-rt tbb compiler
        module_cmd load mkl/2024.2
        isolate_target_python_environment
    else
        module_cmd load compilers/intel/21.0.3
        resolve_ffluxlab_intel_runtime
        resolve_ffluxlab_gcc_runtime
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        ensure_ariadne_compilers_on_path
    fi
}

resolve_ariadne_compilers() {
    local c_compiler cxx_compiler fortran_compiler
    c_compiler="$(ariadne_c_compiler_name)"
    cxx_compiler="$(ariadne_cxx_compiler_name)"
    fortran_compiler="$(ariadne_fortran_compiler_name)"
    ensure_ariadne_compilers_on_path
    resolve_required_cmd_into ARIADNE_CC "${c_compiler}" "ARIADNE requires a supported Intel oneAPI C compiler. Run 'module list' and check the oneAPI compiler module."
    resolve_required_cmd_into ARIADNE_CXX "${cxx_compiler}" "ARIADNE requires a supported Intel oneAPI C++ compiler. Run 'module list' and check the oneAPI compiler module."
    resolve_required_cmd_into ARIADNE_FC "${fortran_compiler}" "ARIADNE requires a supported Intel oneAPI Fortran compiler. Run 'module list' and check the oneAPI compiler module."
}

load_gcc_build_modules() {
    module_cmd purge
    if [[ "${MACHINE}" == "csf3" ]]; then
        module_cmd load compilers/gcc/13.3.0
        module_cmd load tools/gcc/cmake/3.31.6
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    elif [[ "${MACHINE}" == "csf4" ]]; then
        module_cmd load python/3.11.3-gcccore-12.3.0
        module_cmd load cmake/3.23.1-gcccore-11.3.0
        isolate_target_python_environment
    else
        module_cmd load compilers/gcc/11.1.0
        if [[ -d "${VENV:-}/bin" ]]; then
            export PATH="${VENV}/bin${PATH:+:${PATH}}"
        fi
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

load_csf3_python_build_modules() {
    load_gcc_build_modules
    module_cmd load libs/gcc/openssl/1.1.1w
}

load_ffluxlab_python_build_modules() {
    module_cmd purge
    module_cmd load compilers/gcc/11.1.0
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

build_ffluxlab_openssl() {
    if [[ -f "${OPENSSL_PREFIX}/include/openssl/ssl.h" ]] \
        && [[ -f "${OPENSSL_PREFIX}/lib/libssl.so" \
            || -f "${OPENSSL_PREFIX}/lib64/libssl.so" ]]; then
        return 0
    fi
    note "Building pinned OpenSSL ${OPENSSL_VERSION} for private Python"
    local tarball="${PROJECTS_DIR}/_sources/${OPENSSL_TARBALL}"
    local build_parent="${PROJECTS_DIR}/_sources/build"
    local source_dir="${build_parent}/openssl-${OPENSSL_VERSION}"
    [[ -f "${tarball}" ]] \
        || download_to "${OPENSSL_URL}" "${tarball}" "${OPENSSL_SHA256}"
    verify_sha256 "${tarball}" "${OPENSSL_SHA256}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        mkdir -p "${build_parent}"
        case "${source_dir}" in
            "${PROJECTS_DIR}/_sources/build/openssl-${OPENSSL_VERSION}") ;;
            *) die "refusing to replace unexpected OpenSSL source directory: ${source_dir}" ;;
        esac
        rm -rf "${source_dir}"
        tar -xzf "${tarball}" -C "${build_parent}"
    else
        echo "+ extract verified ${tarball} -> ${source_dir}"
    fi
    run_in_dir "${source_dir}" ./config \
        "--prefix=${OPENSSL_PREFIX}" \
        "--openssldir=${OPENSSL_PREFIX}/ssl" \
        shared
    run_in_dir "${source_dir}" make -j "${INSTALL_JOBS}"
    run_in_dir "${source_dir}" make install_sw
}

build_ffluxlab_readline_if_missing() {
    local system_readline=""
    system_readline="$(gcc -print-file-name=libreadline.so 2>/dev/null || true)"
    if [[ -f /usr/include/readline/readline.h ]] \
        && [[ -n "${system_readline}" && "${system_readline}" != libreadline.so \
            && -f "${system_readline}" ]]; then
        return 0
    fi
    if [[ -f "${PYTHON_DEPS_PREFIX}/include/readline/readline.h" ]] \
        && [[ -f "${PYTHON_DEPS_PREFIX}/lib/libreadline.so" \
            || -f "${PYTHON_DEPS_PREFIX}/lib64/libreadline.so" ]]; then
        return 0
    fi
    note "Building pinned Readline ${READLINE_VERSION} for private Python"
    local tarball="${PROJECTS_DIR}/_sources/${READLINE_TARBALL}"
    local build_parent="${PROJECTS_DIR}/_sources/build"
    local source_dir="${build_parent}/readline-${READLINE_VERSION}"
    [[ -f "${tarball}" ]] \
        || download_to "${READLINE_URL}" "${tarball}" "${READLINE_SHA256}"
    verify_sha256 "${tarball}" "${READLINE_SHA256}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        mkdir -p "${build_parent}"
        case "${source_dir}" in
            "${PROJECTS_DIR}/_sources/build/readline-${READLINE_VERSION}") ;;
            *) die "refusing to replace unexpected Readline source directory: ${source_dir}" ;;
        esac
        rm -rf "${source_dir}"
        tar -xzf "${tarball}" -C "${build_parent}"
    else
        echo "+ extract verified ${tarball} -> ${source_dir}"
    fi
    run_in_dir "${source_dir}" ./configure \
        "--prefix=${PYTHON_DEPS_PREFIX}" \
        --with-curses
    run_in_dir "${source_dir}" make -j "${INSTALL_JOBS}"
    run_in_dir "${source_dir}" make install
}

build_ffluxlab_sqlite_if_missing() {
    local system_sqlite=""
    system_sqlite="$(gcc -print-file-name=libsqlite3.so 2>/dev/null || true)"
    if [[ -f /usr/include/sqlite3.h ]] \
        && [[ -n "${system_sqlite}" && "${system_sqlite}" != libsqlite3.so \
            && -f "${system_sqlite}" ]]; then
        return 0
    fi
    if [[ -f "${PYTHON_DEPS_PREFIX}/include/sqlite3.h" ]] \
        && [[ -f "${PYTHON_DEPS_PREFIX}/lib/libsqlite3.so" \
            || -f "${PYTHON_DEPS_PREFIX}/lib64/libsqlite3.so" ]]; then
        return 0
    fi
    note "Building pinned SQLite 3.45.3 for private Python"
    local tarball="${PROJECTS_DIR}/_sources/${SQLITE_TARBALL}"
    local build_parent="${PROJECTS_DIR}/_sources/build"
    local source_dir="${build_parent}/sqlite-autoconf-${SQLITE_VERSION}"
    [[ -f "${tarball}" ]] \
        || download_to "${SQLITE_URL}" "${tarball}" "${SQLITE_SHA256}"
    verify_sha256 "${tarball}" "${SQLITE_SHA256}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        mkdir -p "${build_parent}"
        case "${source_dir}" in
            "${PROJECTS_DIR}/_sources/build/sqlite-autoconf-${SQLITE_VERSION}") ;;
            *) die "refusing to replace unexpected SQLite source directory: ${source_dir}" ;;
        esac
        rm -rf "${source_dir}"
        tar -xzf "${tarball}" -C "${build_parent}"
    else
        echo "+ extract verified ${tarball} -> ${source_dir}"
    fi
    run_in_dir "${source_dir}" ./configure \
        "--prefix=${PYTHON_DEPS_PREFIX}" \
        --enable-shared \
        --disable-static
    run_in_dir "${source_dir}" make -j "${INSTALL_JOBS}"
    run_in_dir "${source_dir}" make install
}

prepare_ffluxlab_python_dependencies() {
    [[ "${MACHINE}" == "ffluxlab" ]] || return 0
    build_ffluxlab_openssl
    build_ffluxlab_readline_if_missing
    build_ffluxlab_sqlite_if_missing
    export CPPFLAGS="-I${PYTHON_DEPS_PREFIX}/include${CPPFLAGS:+ ${CPPFLAGS}}"
    export LDFLAGS="-L${PYTHON_DEPS_PREFIX}/lib -Wl,-rpath,${PYTHON_DEPS_PREFIX}/lib${LDFLAGS:+ ${LDFLAGS}}"
    export PKG_CONFIG_PATH="${PYTHON_DEPS_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
    export LD_LIBRARY_PATH="${OPENSSL_PREFIX}/lib:${OPENSSL_PREFIX}/lib64:${PYTHON_DEPS_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

verify_python_ssl() {
    local python_exe="$1"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ ${python_exe} -c 'import ssl; print(ssl.OPENSSL_VERSION)'"
        return 0
    fi
    if ! "${python_exe}" -c "import ssl; print(ssl.OPENSSL_VERSION)" >/dev/null 2>&1; then
        die "Private Python at ${python_exe} cannot import ssl. Move aside ${PYTHON_PREFIX} and rebuild its pinned OpenSSL environment."
    fi
}

verify_private_python_stdlib() {
    local python_exe="$1"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ ${python_exe} -c 'import ssl, sqlite3, readline'"
        return 0
    fi
    "${python_exe}" -c "import ssl, sqlite3, readline" \
        || die "Private Python is missing ssl, sqlite3, or readline support: ${python_exe}"
}

ensure_private_python() {
    local rebuild="${1:-0}"
    if [[ "${MACHINE}" != "csf3" && "${MACHINE}" != "ffluxlab" ]]; then
        return 0
    fi
    if [[ "${MACHINE}" == "csf3" ]]; then
        load_csf3_python_build_modules
    else
        load_ffluxlab_python_build_modules
        prepare_ffluxlab_python_dependencies
    fi
    local py="${PYTHON_PREFIX}/bin/python3.11"
    if [[ "${rebuild}" -eq 1 && -x "${py}" ]]; then
        note "Rebuilding private CPython ${PYTHON_VERSION}"
        backup_existing_path "${PYTHON_PREFIX}" "private Python prefix"
    fi
    if [[ -x "${py}" ]]; then
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        verify_python_ssl "${py}"
        verify_private_python_stdlib "${py}"
        return 0
    fi

    note "Private CPython ${PYTHON_VERSION} is missing; preparing build"
    require_cmd gcc "Load a GCC compiler module first."
    require_cmd make "Load a build tool module first."
    local openssl_prefix
    if [[ "${MACHINE}" == "ffluxlab" ]]; then
        openssl_prefix="${OPENSSL_PREFIX}"
    elif [[ "${DRY_RUN}" -eq 1 ]]; then
        openssl_prefix="\${OPENSSL_PREFIX_FROM_MODULE}"
    else
        require_cmd openssl "Load libs/gcc/openssl/1.1.1w first."
        openssl_prefix="$(resolve_openssl_prefix)"
    fi
    local sources_dir="${PROJECTS_DIR}/_sources"
    local tarball="${sources_dir}/${PYTHON_TARBALL}"
    if [[ ! -f "${tarball}" ]]; then
        download_to "${PYTHON_URL}" "${tarball}" "${PYTHON_SHA256}"
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ build private CPython ${PYTHON_VERSION} from ${tarball} with --with-openssl=${openssl_prefix}"
        verify_python_ssl "${py}"
        export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        return 0
    fi
    require_file "${tarball}" "Python source tarball"
    verify_sha256 "${tarball}" "${PYTHON_SHA256}"
    local build_parent="${HOME}/src"
    local src_dir="${build_parent}/Python-${PYTHON_VERSION}"
    run_cmd mkdir -p "${build_parent}" "$(dirname "${PYTHON_PREFIX}")"
    run_cmd rm -rf "${src_dir}"
    run_cmd tar -xzf "${tarball}" -C "${build_parent}"
    run_in_dir "${src_dir}" ./configure "--prefix=${PYTHON_PREFIX}" --enable-shared --with-ensurepip=install "--with-openssl=${openssl_prefix}" --with-openssl-rpath=auto
    run_in_dir "${src_dir}" make -j "${INSTALL_JOBS}"
    run_in_dir "${src_dir}" make install
    export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    verify_python_ssl "${py}"
    verify_private_python_stdlib "${py}"
}

ensure_csf3_python() {
    ensure_private_python "$@"
}

validate_csf4_venv_contract() {
    [[ "${MACHINE}" == "csf4" ]] || return 0
    isolate_target_python_environment
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ validate isolated CSF4 venv contract at ${VENV}"
        return 0
    fi
    "${PYTHON}" - "${VENV}" <<'PY'
from pathlib import Path
import os
import site
import sys

expected = Path(sys.argv[1]).resolve()
observed = Path(sys.prefix).resolve()
if observed != expected or sys.prefix == sys.base_prefix:
    raise SystemExit(
        f"configured Python is not isolated in the target venv: "
        f"expected={expected} observed={observed}"
    )

config = expected / "pyvenv.cfg"
settings = {}
for line in config.read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        settings[key.strip().lower()] = value.strip().lower()
if settings.get("include-system-site-packages") != "false":
    raise SystemExit(
        "target venv must set include-system-site-packages = false"
    )
if site.ENABLE_USER_SITE is not False:
    raise SystemExit("target venv unexpectedly enables user site-packages")
if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
    raise SystemExit("CSF4 Python isolation variables were not cleared")
if os.environ.get("PYTHONNOUSERSITE") != "1":
    raise SystemExit("PYTHONNOUSERSITE=1 is required on CSF4")
PY
}

verify_csf4_python_package_origins() {
    [[ "${MACHINE}" == "csf4" ]] || return 0
    local include_ariadne="${1:-0}"
    isolate_target_python_environment
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ verify CSF4 Python package origins are inside ${VENV}"
        return 0
    fi
    "${PYTHON}" - \
        "${VENV}" \
        "${include_ariadne}" \
        "${REPO_ROOT}" \
        "${PROJECTS_DIR}/FEREBUS_CPU/pyferebus" <<'PY'
from importlib import import_module
from pathlib import Path
import sys

venv = Path(sys.argv[1]).resolve()
modules = [
    "packaging",
    "numpy",
    "scipy",
    "pandas",
    "yaml",
    "cffi",
    "ase",
    "xtb",
    "plumed",
    "rdkit",
    "tqdm",
    "portalocker",
    "typing_extensions",
]
if sys.argv[2] == "1":
    modules.append("ariadne")

failures = []
for name in modules:
    module = import_module(name)
    raw_origin = getattr(module, "__file__", None)
    if not raw_origin:
        failures.append(f"{name}: import has no file origin")
        continue
    origin = Path(raw_origin).resolve()
    try:
        origin.relative_to(venv)
    except ValueError:
        failures.append(f"{name}: {origin} is outside {venv}")
if failures:
    raise SystemExit(
        "CSF4 venv package isolation failed:\n  " + "\n  ".join(failures)
    )

editable_origins = {
    "ichor.core": Path(sys.argv[3]).resolve() / "ichor_core",
    "ichor.hpc": Path(sys.argv[3]).resolve() / "ichor_hpc",
    "ichor.cli": Path(sys.argv[3]).resolve() / "ichor_cli",
    "pyferebus.executors.trainer": Path(sys.argv[4]).resolve(),
}
for name, expected_root in editable_origins.items():
    module = import_module(name)
    origin = Path(module.__file__).resolve()
    try:
        origin.relative_to(expected_root)
    except ValueError as exc:
        raise SystemExit(
            f"editable package {name} is outside {expected_root}: {origin}"
        ) from exc
print("CSF4 venv package origins OK")
PY
}

verify_ariadne_packaging_api() {
    [[ "${MACHINE}" == "csf4" ]] || return 0
    isolate_target_python_environment
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ verify isolated packaging API required by ARIADNE"
        return 0
    fi
    "${PYTHON}" - "${VENV}" <<'PY'
from pathlib import Path
import inspect
import sys

import packaging
import packaging.utils

venv = Path(sys.argv[1]).resolve()
origin = Path(packaging.__file__).resolve()
try:
    origin.relative_to(venv)
except ValueError as exc:
    raise SystemExit(
        f"ARIADNE build packaging is outside the target venv: {origin}"
    ) from exc
parameters = inspect.signature(packaging.utils.canonicalize_name).parameters
if "validate" not in parameters or not hasattr(packaging.utils, "InvalidName"):
    raise SystemExit(
        "ARIADNE requires a modern isolated packaging API; rerun --only python "
        "or --only packages"
    )
print(f"ARIADNE packaging API OK: {origin}")
PY
}

create_or_activate_venv() {
    local recreate="${1:-0}"
    local base_python
    if [[ "${MACHINE}" == "csf3" || "${MACHINE}" == "ffluxlab" ]]; then
        base_python="${PYTHON_PREFIX}/bin/python3.11"
    else
        base_python="python"
    fi
    if [[ "${recreate}" -eq 1 && -d "${VENV}" ]]; then
        backup_existing_path "${VENV}" "venv"
    fi
    isolate_target_python_environment
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
    isolate_target_python_environment
    validate_csf4_venv_contract
    if [[ "${MACHINE}" == "csf3" || "${MACHINE}" == "ffluxlab" ]]; then
        verify_python_ssl "${PYTHON}"
        verify_private_python_stdlib "${PYTHON}"
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
    isolate_target_python_environment
    validate_csf4_venv_contract
    if [[ "${MACHINE}" == "csf3" || "${MACHINE}" == "ffluxlab" ]]; then
        verify_python_ssl "${PYTHON}"
        verify_private_python_stdlib "${PYTHON}"
    fi
}

prepare_python_and_venv() {
    local rebuild_python="${1:-0}"
    local recreate_venv="${2:-0}"
    deactivate_existing_venv "Python module setup"
    load_python_stack
    ensure_private_python "${rebuild_python}"
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
    local -a constraint_args=()
    if [[ "${MACHINE}" == "ffluxlab" ]]; then
        require_file \
            "${FFLUXLAB_PYTHON_CONSTRAINTS}" \
            "ffluxlab Python constraints"
        constraint_args=(--constraint "${FFLUXLAB_PYTHON_CONSTRAINTS}")
    fi
    isolate_target_python_environment
    run_cmd "${PYTHON}" -m pip install "${constraint_args[@]}" "$@"
}

ensure_private_cmake() {
    [[ "${MACHINE}" == "ffluxlab" ]] || return 0
    if [[ "${DRY_RUN}" -eq 0 ]] \
        && "${PYTHON}" - "${CMAKE_VERSION}" <<'PY' >/dev/null 2>&1
import subprocess
import sys

minimum = tuple(int(part) for part in sys.argv[1].split("."))
output = subprocess.check_output(["cmake", "--version"], text=True)
observed = tuple(int(part) for part in output.splitlines()[0].split()[-1].split("."))
raise SystemExit(0 if observed >= minimum else 1)
PY
    then
        return 0
    fi
    note "Installing private CMake ${CMAKE_VERSION}"
    pip_install "cmake==${CMAKE_VERSION}"
    export PATH="${VENV}/bin${PATH:+:${PATH}}"
}

install_python_packages() {
    note "Installing ICHOR and pyferebus Python packages"
    require_dir "${REPO_ROOT}/ichor_core" "ichor_core package"
    require_dir "${REPO_ROOT}/ichor_hpc" "ichor_hpc package"
    require_dir "${REPO_ROOT}/ichor_cli" "ichor_cli package"
    require_dir "${PROJECTS_DIR}/FEREBUS_CPU/pyferebus" "pyferebus package"

    pip_install --upgrade pip setuptools wheel
    ensure_private_cmake
    if [[ "${MACHINE}" == "ffluxlab" ]]; then
        note "Installing the ffluxlab-compatible compiled Python stack"
        pip_install \
            --only-binary=:all: \
            matplotlib \
            numpy \
            pandas \
            pyarrow \
            rdkit \
            scipy \
            xtb
    fi
    pip_install cffi pytest
    pip_install -e "${REPO_ROOT}/ichor_core"
    pip_install -e "${REPO_ROOT}/ichor_hpc"
    pip_install -e "${REPO_ROOT}/ichor_cli"
    pip_install -e "${PROJECTS_DIR}/FEREBUS_CPU/pyferebus" --no-deps
    verify_csf4_python_package_origins 0
}

verify_ariadne_api() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ scoped ARIADNE Python: import module and run ABI/runtime probes"
        return 0
    fi
    run_ariadne_python - <<'PY'
import ariadne
from ichor.hpc.active_learning.acquisition.ariadne_abi import probe_ariadne_module
from ichor.hpc.active_learning.acquisition.ariadne_local_runner import (
    probe_ariadne_runtime,
)

receipt = probe_ariadne_module(ariadne)
print("ARIADNE ABI OK:", receipt["contract_version"])
runtime = probe_ariadne_runtime(ariadne)
print(
    "ARIADNE runtime OK:",
    runtime["trqn"]["status_length"],
    runtime["ds"]["status_length"],
)
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
    run_ariadne_python - "${label}" <<'PY' || true
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
    run_ariadne_python - "${VENV}" <<'PY'
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
    run_ariadne_python - "$(ariadne_receipt_path)" "${MACHINE}" "${ariadne_root}" "${repo_commit}" "${PYTHON}" "${VENV}" "${ARIADNE_CC:-}" "${ARIADNE_CXX:-}" "${ARIADNE_FC:-}" <<'PY'
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
        "ARIADNE_SAFE_IFX_FLAGS": machine in {"csf3", "ffluxlab"},
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
    run_ariadne_python - "${receipt}" "${PROJECTS_DIR}/ARIADNE" <<'PY' || true
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
    if [[ "${MACHINE:-}" == "csf3" || "${MACHINE:-}" == "ffluxlab" ]]; then
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
    elif [[ "${MACHINE:-}" == "csf4" ]]; then
        cat >&2 <<'EOF'
module load python/3.11.3-gcccore-12.3.0
module load compilers/oneapi/2024.2.0
module load compiler-rt tbb compiler
module load mkl/2024.2
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
EOF
        echo "source ${VENV:-${HOME}/.venv/ichor-csf4}/bin/activate" >&2
    else
        cat >&2 <<'EOF'
module load compilers/intel/21.0.3
intel_root="$(readlink -f /home/modules/compilers/intel/21.0.3)"
intel_runtime="$(find "$intel_root" \( -type f -o -type l \) -path '*/intel64_lin/libimf.so' -print -quit)"
intel_mkl="${MKLROOT:+$MKLROOT/lib/intel64/libmkl_intel_lp64.so.1}"
if [[ -z "$intel_mkl" || ! -e "$intel_mkl" ]]; then
    intel_mkl="$(find "$intel_root" \( -type f -o -type l \) -path '*/mkl/*/lib/intel64/libmkl_intel_lp64.so.1' -print -quit)"
fi
[[ -n "$intel_runtime" && -n "$intel_mkl" ]] || { echo "Intel runtime discovery failed" >&2; exit 1; }
intel_runtime="$(dirname "$intel_runtime")"
intel_mkl="$(dirname "$intel_mkl")"
gcc_root="$(readlink -f /home/modules/compilers/gcc/11.1.0)"
gcc_runtime="$gcc_root/lib64"
[[ -e "$gcc_runtime/libstdc++.so.6" ]] || { echo "GCC 11 runtime discovery failed" >&2; exit 1; }
export LD_LIBRARY_PATH="$gcc_runtime:$intel_runtime:$intel_mkl:$HOME/opt/python-3.11.15/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$gcc_runtime:$intel_runtime:$intel_mkl${LIBRARY_PATH:+:$LIBRARY_PATH}"
EOF
        echo "source ${VENV:-${HOME}/.venv/ichor-ffluxlab}/bin/activate" >&2
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
    if [[ "${reinstall}" -eq 0 && "${DRY_RUN}" -eq 0 ]] && ariadne_import_ok; then
        echo "ARIADNE already importable"
        verify_ariadne_api
        assert_ariadne_inside_venv
        write_ariadne_receipt
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
    verify_ariadne_packaging_api
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
    if [[ "${MACHINE}" == "csf3" || "${MACHINE}" == "ffluxlab" ]]; then
        ariadne_pip_config=" --config-settings=cmake.define.ARIADNE_SAFE_IFX_FLAGS=ON"
    fi
    [[ "${DRY_RUN}" -eq 1 ]] && echo "+ export CC=${CC} CXX=${CXX} FC=${FC} CMAKE_BUILD_PARALLEL_LEVEL=${INSTALL_JOBS} MAKEFLAGS=-j${INSTALL_JOBS}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ clean ARIADNE-local build artefacts in ${ariadne_root}"
    else
        rm -rf "${ariadne_root}/_skbuild" "${ariadne_root}/build"
    fi
    isolate_target_python_environment
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
    verify_csf4_python_package_origins 1
    write_ariadne_receipt
}

plumed_source_dir() {
    local extracted="${PROJECTS_DIR}/plumed-${PLUMED_VERSION}"
    local sources_dir="${PROJECTS_DIR}/_sources"
    local tarball="${sources_dir}/${PLUMED_TARBALL}"
    if [[ -d "${extracted}" ]]; then
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            require_file "${extracted}/.ichor-source.sha256" "verified PLUMED source receipt"
            [[ "$(tr -d '[:space:]' < "${extracted}/.ichor-source.sha256")" == "${PLUMED_SHA256}" ]] || die "existing PLUMED source was not extracted from the pinned archive"
        fi
        printf '%s\n' "${extracted}"
        return 0
    fi
    if [[ ! -f "${tarball}" ]]; then
        download_to "${PLUMED_URL}" "${tarball}" "${PLUMED_SHA256}"
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '%s\n' "${extracted}"
        return 0
    fi
    require_file "${tarball}" "PLUMED source tarball"
    verify_sha256 "${tarball}" "${PLUMED_SHA256}"
    run_cmd mkdir -p "${PROJECTS_DIR}"
    run_cmd tar -xzf "${tarball}" -C "${PROJECTS_DIR}"
    printf '%s\n' "${PLUMED_SHA256}" > "${extracted}/.ichor-source.sha256"
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

private_binutils_usable() {
    local linker="${BINUTILS_PREFIX}/bin/ld"
    [[ -x "${linker}" ]] || return 1
    local version_line
    version_line="$("${linker}" --version 2>/dev/null | head -n 1 || true)"
    [[ "${version_line}" == *" ${BINUTILS_VERSION}"* ]]
}

build_private_binutils() {
    local tarball="${PROJECTS_DIR}/_sources/${BINUTILS_TARBALL}"
    local build_parent="${PROJECTS_DIR}/_sources/build"
    local source_dir="${build_parent}/binutils-${BINUTILS_VERSION}"
    local build_dir="${build_parent}/binutils-${BINUTILS_VERSION}-build"

    note "Building private GNU Binutils ${BINUTILS_VERSION} for FEREBUS"
    require_cmd gcc "Load a GCC compiler module before building GNU Binutils."
    require_cmd make "Load a GCC build environment before building GNU Binutils."
    require_cmd tar "GNU tar is required to unpack GNU Binutils."
    if [[ ! -f "${tarball}" ]]; then
        download_to "${BINUTILS_URL}" "${tarball}" "${BINUTILS_SHA256}"
    fi
    verify_sha256 "${tarball}" "${BINUTILS_SHA256}"

    case "${source_dir}" in
        "${PROJECTS_DIR}/_sources/build/binutils-${BINUTILS_VERSION}") ;;
        *) die "refusing to replace unexpected Binutils source directory: ${source_dir}" ;;
    esac
    case "${build_dir}" in
        "${PROJECTS_DIR}/_sources/build/binutils-${BINUTILS_VERSION}-build") ;;
        *) die "refusing to replace unexpected Binutils build directory: ${build_dir}" ;;
    esac
    run_cmd mkdir -p "${build_parent}"
    run_cmd rm -rf "${source_dir}" "${build_dir}"
    run_cmd tar -xf "${tarball}" -C "${build_parent}"
    run_cmd mkdir -p "${build_dir}"
    run_in_dir "${build_dir}" "${source_dir}/configure" \
        "--prefix=${BINUTILS_PREFIX}" \
        --disable-gdb \
        --disable-gdbserver \
        --disable-gold \
        --disable-gprofng \
        --disable-libdecnumber \
        --disable-nls \
        --disable-readline \
        --disable-shared \
        --disable-sim \
        --disable-werror \
        --enable-static
    # ffluxlab does not provide Texinfo, and documentation is not part of the
    # private linker contract used to consume FEREBUS's bundled OpenBLAS.
    run_in_dir "${build_dir}" make -j "${INSTALL_JOBS}" \
        MAKEINFO=true all-binutils all-ld
    backup_existing_path "${BINUTILS_PREFIX}" "private Binutils installation"
    run_in_dir "${build_dir}" make MAKEINFO=true \
        install-binutils install-ld

    if [[ "${DRY_RUN}" -eq 0 ]]; then
        private_binutils_usable \
            || die "private GNU Binutils ${BINUTILS_VERSION} installation is unusable at ${BINUTILS_PREFIX}"
        echo "FEREBUS linker: $("${BINUTILS_PREFIX}/bin/ld" --version | head -n 1)"
    fi
}

ensure_ffluxlab_binutils() {
    [[ "${MACHINE}" == "ffluxlab" ]] || return 0
    [[ ! -L "${BINUTILS_PREFIX}" ]] \
        || die "private Binutils prefix must not be a symlink: ${BINUTILS_PREFIX}"
    if private_binutils_usable; then
        echo "Private FEREBUS linker already usable: ${BINUTILS_PREFIX}/bin/ld"
        return 0
    fi
    build_private_binutils
}

verify_ffluxlab_gfortran_linker() {
    [[ "${MACHINE}" == "ffluxlab" ]] || return 0
    local linker_flag="-B${BINUTILS_PREFIX}/bin/"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ verify gfortran ${linker_flag} selects ${BINUTILS_PREFIX}/bin/ld"
        return 0
    fi
    require_cmd readlink "GNU readlink is required to validate the FEREBUS linker."
    local expected_linker
    local selected_linker
    expected_linker="$(readlink -f "${BINUTILS_PREFIX}/bin/ld")"
    selected_linker="$(gfortran "${linker_flag}" -print-prog-name=ld)"
    selected_linker="$(readlink -f "${selected_linker}")"
    [[ "${selected_linker}" == "${expected_linker}" ]] \
        || die "gfortran did not select the private FEREBUS linker: expected ${expected_linker}, observed ${selected_linker}"
    echo "gfortran FEREBUS linker: ${selected_linker}"
}

resolve_bundled_ferebus_openblas() {
    local root="$1"
    local candidate
    for candidate in \
        "${root}/libs/openblas/lib64/libopenblas.a" \
        "${root}/libs/openblas/lib/libopenblas.a"; do
        if [[ -L "${candidate}" ]]; then
            die "bundled FEREBUS OpenBLAS archive must not be a symlink: ${candidate}"
        fi
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '%s\n' "${root}/libs/openblas/lib64/libopenblas.a"
        return 0
    fi
    die "bundled FEREBUS OpenBLAS archive is missing. The installer will not rebuild it. Restore the tracked bundle with: git -C ${root} restore --source=HEAD -- libs/openblas"
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
    load_gcc_build_modules
    require_cmd make "Load a GCC build environment first."
    local openblas_a
    openblas_a="$(resolve_bundled_ferebus_openblas "${root}")"
    local openblas_sha256=""
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        openblas_sha256="$(file_sha256 "${openblas_a}")"
        echo "Bundled FEREBUS OpenBLAS archive: ${openblas_a}"
    else
        echo "+ preserve bundled FEREBUS OpenBLAS archive ${openblas_a}"
    fi
    ensure_ffluxlab_binutils
    require_cmd cmake "Load the CSF CMake module first."
    require_cmd gfortran "Load a GCC compiler module first."
    verify_ffluxlab_gfortran_linker
    local build_dir="${root}/build-ichor-install"
    case "${build_dir}" in
        "${root}/build-ichor-install") ;;
        *) die "refusing to clean unexpected FEREBUS build directory: ${build_dir}" ;;
    esac
    if [[ "${reinstall}" -eq 1 || "${MACHINE}" == "ffluxlab" ]]; then
        run_cmd rm -rf "${build_dir}"
    fi
    if [[ "${reinstall}" -eq 1 ]]; then
        backup_existing_path "${FEREBUS_PATH}" "FEREBUS executable"
    fi
    run_cmd mkdir -p "${build_dir}" "$(dirname "${FEREBUS_PATH}")"
    local -a cmake_args=(
        -S "${root}"
        -B "${build_dir}"
        -DCMAKE_BUILD_TYPE=Release
    )
    if [[ "${MACHINE}" == "ffluxlab" ]]; then
        cmake_args+=(
            "-DCMAKE_EXE_LINKER_FLAGS=-B${BINUTILS_PREFIX}/bin/"
        )
    fi
    run_cmd cmake "${cmake_args[@]}"
    run_cmd cmake --build "${build_dir}" -j "${INSTALL_JOBS}"
    if [[ -f "${build_dir}/ferebus" ]]; then
        run_cmd cp "${build_dir}/ferebus" "${FEREBUS_PATH}"
        run_cmd chmod 755 "${FEREBUS_PATH}"
    else
        run_cmd cmake --install "${build_dir}"
    fi
    [[ "${DRY_RUN}" -eq 1 || -x "${FEREBUS_PATH}" ]] || die "FEREBUS build did not create executable: ${FEREBUS_PATH}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        local observed_openblas_sha256
        observed_openblas_sha256="$(file_sha256 "${openblas_a}")"
        [[ "${observed_openblas_sha256}" == "${openblas_sha256}" ]] \
            || die "FEREBUS build modified its bundled OpenBLAS archive: ${openblas_a}"
    fi
}

upsert_ichor_config() {
    note "Updating ~/ichor_config.yaml"
    isolate_target_python_environment
    local venv_config
    local aimall_config
    local ferebus_config
    local plumed_kernel_config
    local plumed_lib_config
    local python_library_config
    venv_config="$(config_path_for_yaml "${VENV}/bin/python")"
    aimall_config="$(config_path_for_yaml "${AIMALL_PATH}")"
    ferebus_config="$(config_path_for_yaml "${FEREBUS_PATH}")"
    plumed_kernel_config="$(config_path_for_yaml "${PLUMED_KERNEL:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib/libplumedKernel.so}")"
    plumed_lib_config="$(config_path_for_yaml "${PLUMED_LIBRARY_PATH:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib}")"
    python_library_config="$(config_path_for_yaml "${PYTHON_PREFIX}/lib")"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ initialise/update ~/ichor_config.yaml from repo template and upsert ${MACHINE} profile"
        return 0
    fi
    "${PYTHON}" "${REPO_ROOT}/scripts/upsert_ichor_config.py" \
        --destination "${HOME}/ichor_config.yaml" \
        --canonical-config "${REPO_ROOT}/ichor_config.yaml" \
        --machine "${MACHINE}" \
        --python-path "${venv_config}" \
        --python-library-path "${python_library_config}" \
        --aimall-path "${aimall_config}" \
        --ferebus-path "${ferebus_config}" \
        --plumed-kernel "${plumed_kernel_config}" \
        --plumed-library-path "${plumed_lib_config}"
}

verify_operator_backends() {
    note "Verifying Gaussian/AIMAll user-provided backends"
    local resolved_aimall=""
    if [[ "${MACHINE}" == "ffluxlab" ]]; then
        module_cmd load apps/aimall/19.02.13
    fi
    if [[ "${AIMALL_PATH}" == */* ]]; then
        [[ -x "${AIMALL_PATH}" ]] && resolved_aimall="${AIMALL_PATH}"
    else
        resolved_aimall="$(command -v "${AIMALL_PATH}" 2>/dev/null || true)"
    fi
    if [[ -z "${resolved_aimall}" ]]; then
        die "AIMAll executable is missing or not executable: ${AIMALL_PATH}. Re-run with --aimall-path PATH after installing AIMAll."
    fi
    if [[ ! -x "${FEREBUS_PATH}" ]]; then
        die "FEREBUS executable is missing or not executable: ${FEREBUS_PATH}"
    fi
    echo "AIMAll: ${resolved_aimall}"
    echo "FEREBUS: ${FEREBUS_PATH}"
}

require_yaml_available() {
    isolate_target_python_environment
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
    [[ -f "${HOME}/ichor_config.yaml" ]] \
        || die "ICHOR configuration not found: ${HOME}/ichor_config.yaml. Complete the full installer or run --only config before --only verify."
    export ICHOR_MACHINE="${MACHINE}"
    isolate_target_python_environment
    validate_csf4_venv_contract
    verify_csf4_python_package_origins 1
    "${PYTHON}" -c "import ichor.core, ichor.hpc, ichor.cli; print('ICHOR packages OK')"
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
schema_version: 14
campaign:
  system_name: INSTALLER_SMOKE
  max_iterations: 1
  sampling_aggressiveness: 5
  custom_bootstrap: false
point_allocation:
  bootstrap_training_size: 8
  bootstrap_internal_validation_size: 2
  bootstrap_external_validation_size: 2
  batch_training_size: 3
  batch_internal_validation_size: 1
seed_selection:
  n_seeds_per_iteration: 8
  bulk_fraction: 0.2
  strategy: d_optimal
  d_optimal_degenerate_policy: score_backfill
ferebus:
  prior_mean_strategy: physical_atomic_iqa
  prior_mean_level_of_theory: auto
  physical_prior_scale: 1.0
EOF
    : > "${smoke_dir}/pool.xyz"
    for _ichor_i in $(seq 1 20); do
        cat >> "${smoke_dir}/pool.xyz" <<'EOF'
3
installer smoke water frame
O 0.000000 0.000000 0.000000
H 0.957200 0.000000 0.000000
H -0.239987 0.927297 0.000000
EOF
    done
    "${VENV}/bin/ichor-al-daemon" init --campaign-dir "${smoke_dir}" --yes
    "${VENV}/bin/ichor-al-daemon" preflight --campaign-dir "${smoke_dir}"
    echo ""
    if [[ "${label}" == "verify" ]]; then
        echo "Verification complete."
    else
        echo "Install complete."
    fi
    echo "Enter the runtime environment with:"
    echo "  source ${REPO_ROOT}/scripts/env_ichor.sh --smoke"
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
        elif [[ "${MACHINE}" == "csf4" ]]; then
            echo "+ module load python/3.11.3-gcccore-12.3.0"
            echo "+ module load compilers/oneapi/2024.2.0"
            echo "+ module load compiler-rt tbb compiler"
            echo "+ module load mkl/2024.2"
            echo "+ unset PYTHONPATH PYTHONHOME"
            echo "+ export PYTHONNOUSERSITE=1"
        else
            echo "+ module load compilers/intel/21.0.3"
            echo "+ resolve 64-bit Intel and MKL runtimes beneath /home/modules/compilers/intel/21.0.3"
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
    elif [[ "${MACHINE}" == "csf4" ]]; then
        ichor_csf_module load python/3.11.3-gcccore-12.3.0 || return 1
        ichor_csf_module load compilers/oneapi/2024.2.0 || return 1
        ichor_csf_module load compiler-rt tbb compiler || return 1
        ichor_csf_module load mkl/2024.2 || return 1
        ichor_csf_isolate_python_environment
    else
        ichor_csf_module load compilers/intel/21.0.3 || return 1
        resolve_ffluxlab_intel_runtime || return 1
        resolve_ffluxlab_gcc_runtime || return 1
    fi
}

doctor_print_ariadne_compiler_discovery() {
    local compiler result resolved method
    local c_compiler cxx_compiler fortran_compiler
    c_compiler="$(ariadne_c_compiler_name)"
    cxx_compiler="$(ariadne_cxx_compiler_name)"
    fortran_compiler="$(ariadne_fortran_compiler_name)"
    for compiler in "${c_compiler}" "${cxx_compiler}" "${fortran_compiler}"; do
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
    note "Running ICHOR platform doctor diagnostics"
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
    local doctor_aimall=""
    if [[ "${AIMALL_PATH}" == */* ]]; then
        [[ -x "${AIMALL_PATH}" ]] && doctor_aimall="${AIMALL_PATH}"
    else
        doctor_aimall="$(command -v "${AIMALL_PATH}" 2>/dev/null || true)"
        if [[ -z "${doctor_aimall}" && "${MACHINE}" == "ffluxlab" ]]; then
            ichor_csf_module load apps/aimall/19.02.13 >/dev/null 2>&1 || true
            doctor_aimall="$(command -v "${AIMALL_PATH}" 2>/dev/null || true)"
        fi
    fi
    echo "AIMAll: ${AIMALL_PATH} $([[ -n "${doctor_aimall}" ]] && echo "${doctor_aimall}" || echo missing-or-not-executable)"
    echo "FEREBUS: ${FEREBUS_PATH} $([[ -x "${FEREBUS_PATH}" ]] && echo executable || echo missing-or-not-executable)"
    if [[ "${MACHINE}" == "csf3" ]]; then
        echo "Gaussian expected: apps/binapps/gaussian/g09d01_em64t / \$g09root/g09/g09"
    elif [[ "${MACHINE}" == "csf4" ]]; then
        echo "Gaussian expected: gaussian/g16c01_em64t_detectcpu / \$g16root/g16/g16"
    else
        echo "Gaussian expected: apps/gaussian/g09 / g09"
        echo "SGE tools expected: qsub qstat qacct qdel"
    fi
    export PLUMED_KERNEL="${PLUMED_KERNEL:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib/libplumedKernel.so}"
    export PLUMED_LIBRARY_PATH="${PLUMED_LIBRARY_PATH:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib}"
    echo "PLUMED_KERNEL: ${PLUMED_KERNEL} $([[ -r "${PLUMED_KERNEL}" ]] && echo readable || echo missing-or-unreadable)"
    echo "PLUMED_LIBRARY_PATH: ${PLUMED_LIBRARY_PATH}"
    doctor_python_imports
}

require_all_sibling_repos() {
    ensure_required_source_checkouts
    require_dir "${PROJECTS_DIR}/FEREBUS_CPU" "FEREBUS_CPU sibling repo"
    require_dir "${PROJECTS_DIR}/ARIADNE" "ARIADNE sibling repo"
}

stage_python() {
    CURRENT_STAGE="python"
    prepare_python_and_venv 1 1
    pip_install --upgrade pip setuptools wheel
    ensure_private_cmake
}

stage_packages() {
    CURRENT_STAGE="packages"
    ensure_repo_root
    ensure_public_checkout \
        "${PROJECTS_DIR}/FEREBUS_CPU" \
        "${FEREBUS_URL}" \
        restore-gradient-refinement \
        FEREBUS_CPU
    require_dir "${PROJECTS_DIR}/FEREBUS_CPU" "FEREBUS_CPU sibling repo"
    prepare_python_and_venv 0 0
    install_python_packages
    verify_entrypoints
}

stage_ariadne() {
    CURRENT_STAGE="ariadne"
    ensure_public_checkout \
        "${PROJECTS_DIR}/ARIADNE" \
        "${ARIADNE_URL}" \
        main \
        ARIADNE
    require_dir "${PROJECTS_DIR}/ARIADNE" "ARIADNE sibling repo"
    prepare_python_and_venv 0 0
    ensure_private_cmake
    install_ariadne_if_needed 1
}

stage_plumed() {
    CURRENT_STAGE="plumed"
    prepare_python_and_venv 0 0
    install_plumed_if_needed 1
}

stage_ferebus() {
    CURRENT_STAGE="ferebus"
    ensure_public_checkout \
        "${PROJECTS_DIR}/FEREBUS_CPU" \
        "${FEREBUS_URL}" \
        restore-gradient-refinement \
        FEREBUS_CPU
    require_dir "${PROJECTS_DIR}/FEREBUS_CPU" "FEREBUS_CPU sibling repo"
    prepare_python_and_venv 0 0
    ensure_private_cmake
    install_ferebus_if_needed 1
}

stage_config() {
    CURRENT_STAGE="config"
    deactivate_existing_venv "config module setup"
    load_python_stack
    if [[ "${MACHINE}" == "csf3" || "${MACHINE}" == "ffluxlab" ]]; then
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
    if [[ -z "${AIMALL_PATH}" ]]; then
        if [[ "${MACHINE}" == "ffluxlab" ]]; then
            AIMALL_PATH="aimall"
        else
            AIMALL_PATH="${HOME}/AIMAll/aimqb.ish"
        fi
    fi
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
