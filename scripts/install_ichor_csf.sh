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

MACHINE="${ICHOR_MACHINE:-auto}"
PROJECTS_DIR="${ICHOR_PROJECTS_DIR:-${HOME}/projects}"
REPO_ROOT="${ICHOR_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
VENV="${ICHOR_VENV:-}"
PYTHON_PREFIX="${PYTHON_PREFIX:-${HOME}/opt/python-${PYTHON_VERSION}}"
AIMALL_PATH="${AIMALL_PATH:-${HOME}/AIMAll/aimqb.ish}"
FEREBUS_PATH="${FEREBUS_PATH:-${HOME}/.local/bin/ferebus}"
INSTALL_JOBS="${ICHOR_INSTALL_JOBS:-4}"
ALLOW_DOWNLOAD=0
SKIP_PLUMED=0
SKIP_ARIADNE_BUILD=0
SKIP_FEREBUS_BUILD=0
ASSUME_YES=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: scripts/install_ichor_csf.sh [options]

Options:
  --machine auto|csf3|csf4        default: auto
  --repo-root PATH                default: script-derived repo root
  --projects-dir PATH             default: ~/projects
  --venv PATH                     default: ~/.venv/ichor-csf3 or ~/.venv/ichor-csf4
  --python-prefix PATH            CSF3 default: ~/opt/python-3.11.15
  --allow-download                permit Python/PLUMED/OpenBLAS/source downloads
  --skip-plumed                   skip native PLUMED build and smoke
  --skip-ariadne-build            only verify import ariadne
  --skip-ferebus-build            only verify configured ferebus executable
  --aimall-path PATH              default: ~/AIMAll/aimqb.ish
  --ferebus-path PATH             default: ~/.local/bin/ferebus
  --jobs N                        default: 4
  --yes                           non-interactive mode
  --dry-run                       print planned actions without installing
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine) MACHINE="${2:?missing value for --machine}"; shift 2 ;;
        --repo-root) REPO_ROOT="${2:?missing value for --repo-root}"; shift 2 ;;
        --projects-dir) PROJECTS_DIR="${2:?missing value for --projects-dir}"; shift 2 ;;
        --venv) VENV="${2:?missing value for --venv}"; shift 2 ;;
        --python-prefix) PYTHON_PREFIX="${2:?missing value for --python-prefix}"; shift 2 ;;
        --allow-download) ALLOW_DOWNLOAD=1; shift ;;
        --skip-plumed) SKIP_PLUMED=1; shift ;;
        --skip-ariadne-build) SKIP_ARIADNE_BUILD=1; shift ;;
        --skip-ferebus-build) SKIP_FEREBUS_BUILD=1; shift ;;
        --aimall-path) AIMALL_PATH="${2:?missing value for --aimall-path}"; shift 2 ;;
        --ferebus-path) FEREBUS_PATH="${2:?missing value for --ferebus-path}"; shift 2 ;;
        --jobs) INSTALL_JOBS="${2:?missing value for --jobs}"; shift 2 ;;
        --yes) ASSUME_YES=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unrecognised option: $1" ;;
    esac
done

case "${MACHINE}" in
    auto|csf3|csf4) ;;
    *) die "--machine must be auto, csf3, or csf4" ;;
esac
[[ "${INSTALL_JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"

expand_path() {
    local value="$1"
    value="${value/#\~/${HOME}}"
    printf '%s\n' "${value}"
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
        bash -lc "${command}"
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
        if [[ -n "${hint}" ]]; then
            die "required command '${cmd}' is not on PATH. ${hint}"
        fi
        die "required command '${cmd}' is not on PATH"
    fi
    printf -v "${target_var}" '%s' "${resolved}"
}

initialise_modules() {
    if command -v module >/dev/null 2>&1; then
        return 0
    fi
    # Common Environment Modules/Lmod initialisation points.
    # shellcheck disable=SC1091
    [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true
    # shellcheck disable=SC1091
    [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash || true
}

module_cmd() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '+ module'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    initialise_modules
    command -v module >/dev/null 2>&1 || die "module command is unavailable on this shell"
    module "$@"
}

detect_machine() {
    if [[ "${MACHINE}" != "auto" ]]; then
        printf '%s\n' "${MACHINE}"
        return 0
    fi
    local host
    host="$(hostname -f 2>/dev/null || hostname 2>/dev/null || true)"
    host="${host,,}"
    if [[ "${host}" == *csf3* || "${host}" == *login3* ]]; then
        printf 'csf3\n'
    elif [[ "${host}" == *csf4* || "${host}" == *login0* ]]; then
        printf 'csf4\n'
    else
        die "could not auto-detect CSF3/CSF4 from hostname '${host}'. Pass --machine csf3 or --machine csf4."
    fi
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
    if [[ "${MACHINE}" != "csf3" ]]; then
        return 0
    fi
    load_csf3_python_build_modules
    local py="${PYTHON_PREFIX}/bin/python3.11"
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
    run_shell "cd $(printf '%q' "${src_dir}") && ./configure --prefix=$(printf '%q' "${PYTHON_PREFIX}") --enable-shared --with-ensurepip=install --with-openssl=$(printf '%q' "${openssl_prefix}") --with-openssl-rpath=auto"
    run_shell "cd $(printf '%q' "${src_dir}") && make -j $(printf '%q' "${INSTALL_JOBS}")"
    run_shell "cd $(printf '%q' "${src_dir}") && make install"
    export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    verify_python_ssl "${py}"
}

create_or_activate_venv() {
    local base_python
    if [[ "${MACHINE}" == "csf3" ]]; then
        base_python="${PYTHON_PREFIX}/bin/python3.11"
    else
        base_python="python"
    fi
    if [[ ! -x "${VENV}/bin/python" ]]; then
        note "Creating venv at ${VENV}"
        run_cmd "${base_python}" -m venv "${VENV}"
    fi
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
    pip_install -e "${REPO_ROOT}/ichor_core"
    pip_install -e "${REPO_ROOT}/ichor_hpc"
    pip_install -e "${REPO_ROOT}/ichor_cli"
    pip_install -e "${PROJECTS_DIR}/POLUS/polus_core_subpackage" --no-deps
    pip_install -e "${PROJECTS_DIR}/FEREBUS_CPU/pyferebus" --no-deps
}

install_ariadne_if_needed() {
    note "Checking ARIADNE"
    load_ariadne_modules
    # shellcheck disable=SC1091
    [[ "${DRY_RUN}" -eq 0 ]] && source "${VENV}/bin/activate" || echo "+ source $(printf '%q' "${VENV}/bin/activate")"
    if [[ "${DRY_RUN}" -eq 0 ]] && python_import_ok ariadne; then
        echo "ARIADNE already importable"
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
    [[ "${DRY_RUN}" -eq 1 ]] && echo "+ export CC=${CC} CXX=${CXX} FC=${FC} CMAKE_BUILD_PARALLEL_LEVEL=${INSTALL_JOBS} MAKEFLAGS=-j${INSTALL_JOBS}"
    run_shell "cd $(printf '%q' "${ariadne_root}") && $(printf '%q' "${PYTHON}") -m pip install . --no-build-isolation -v"
    unset CC CXX FC F77 F90
    unset MAKEFLAGS CMAKE_BUILD_PARALLEL_LEVEL
    [[ "${DRY_RUN}" -eq 1 ]] || python_import_ok ariadne || die "ARIADNE install completed but import ariadne still fails"
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

    if [[ "${DRY_RUN}" -eq 0 ]] && [[ -r "${kernel}" ]] && "${PYTHON}" -c "import os, plumed; p=plumed.Plumed(kernel=os.environ['PLUMED_KERNEL']); p.finalize()" >/dev/null 2>&1; then
        echo "PLUMED already usable"
    else
        load_gcc_build_modules
        # shellcheck disable=SC1091
        [[ "${DRY_RUN}" -eq 0 ]] && source "${VENV}/bin/activate" || echo "+ source $(printf '%q' "${VENV}/bin/activate")"
        unset CC CXX FC F77 F90
        export CC=gcc
        export CXX=g++
        require_cmd gcc "Load a GCC compiler module first."
        require_cmd g++ "Load a GCC compiler module first."
        require_cmd make "Load a compiler/build module first."
        local src
        src="$(plumed_source_dir)"
        run_shell "cd $(printf '%q' "${src}") && ./configure --prefix=$(printf '%q' "${HOME}/opt/plumed-${PLUMED_VERSION}") --disable-external-blas --disable-external-lapack --disable-mpi"
        run_shell "cd $(printf '%q' "${src}") && make -j $(printf '%q' "${INSTALL_JOBS}")"
        run_shell "cd $(printf '%q' "${src}") && make install"
        pip_install "plumed==${PLUMED_VERSION}"
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
    note "Checking FEREBUS"
    if [[ -x "${FEREBUS_PATH}" ]]; then
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
            run_shell "cd $(printf '%q' "${root}/libs") && ./fetchOpenBlas.sh"
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
    run_cmd mkdir -p "${build_dir}" "$(dirname "${FEREBUS_PATH}")"
    run_shell "cmake -S $(printf '%q' "${root}") -B $(printf '%q' "${build_dir}") -DCMAKE_BUILD_TYPE=Release"
    run_shell "cmake --build $(printf '%q' "${build_dir}") -j $(printf '%q' "${INSTALL_JOBS}")"
    if [[ -f "${build_dir}/ferebus" ]]; then
        run_cmd cp "${build_dir}/ferebus" "${FEREBUS_PATH}"
        run_cmd chmod 755 "${FEREBUS_PATH}"
    else
        run_shell "cmake --install $(printf '%q' "${build_dir}")"
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
            "parallel_environments": {"serial": [1, 1], "multicore": [2, 168]},
        },
        "software": {
            "python": {"env_name": "ichor-csf3", "python_path": python_path, "modules": []},
            "gaussian": {
                "executable_path": "$g16root/g16/g16",
                "modules": ["apps/binapps/gaussian/g16c01_em64t_detectcpu"],
                "scratch_root": "/scratch/$USER",
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
            "parallel_environments": {"serial": [1, 1], "multicore": [2, 32]},
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
                "scratch_root": "/scratch/$USER",
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

final_checks() {
    note "Running final verification"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ final import checks and ichor-al-daemon preflight"
        return 0
    fi
    export ICHOR_MACHINE="${MACHINE}"
    "${PYTHON}" -c "import ichor.core, ichor.hpc, ichor.cli; print('ICHOR packages OK')"
    "${PYTHON}" -c "import polus.samplers.RS.randomSampling; print('POLUS RS OK')"
    "${PYTHON}" -c "import pyferebus.executors.trainer; print('pyferebus OK')"
    "${PYTHON}" -c "import ariadne; print('ARIADNE OK')"
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
schema_version: 2
max_iterations: 1
EOF
    "${VENV}/bin/ichor-al-daemon" preflight --campaign-dir "${smoke_dir}"
    echo ""
    echo "Install complete."
    echo "Activate with:"
    echo "  export ICHOR_MACHINE=${MACHINE}"
    if [[ "${MACHINE}" == "csf3" ]]; then
        echo "  export LD_LIBRARY_PATH=${PYTHON_PREFIX}/lib:\$LD_LIBRARY_PATH"
    fi
    echo "  source ${VENV}/bin/activate"
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
        local log_file="${log_dir}/install-${MACHINE}-$(date +%Y%m%d-%H%M%S).log"
        exec > >(tee -a "${log_file}") 2>&1
        echo "Writing install log to ${log_file}"
    fi

    note "Install settings"
    cat <<EOF
machine       = ${MACHINE}
repo root     = ${REPO_ROOT}
projects dir  = ${PROJECTS_DIR}
venv          = ${VENV}
jobs          = ${INSTALL_JOBS}
downloads     = ${ALLOW_DOWNLOAD}
dry run       = ${DRY_RUN}
EOF

    print_download_readiness
    ensure_repo_root
    require_dir "${PROJECTS_DIR}/POLUS" "POLUS sibling repo"
    require_dir "${PROJECTS_DIR}/FEREBUS_CPU" "FEREBUS_CPU sibling repo"
    require_dir "${PROJECTS_DIR}/ARIADNE" "ARIADNE sibling repo"

    load_python_stack
    ensure_csf3_python
    create_or_activate_venv
    install_python_packages
    install_ariadne_if_needed
    install_plumed_if_needed
    install_ferebus_if_needed
    load_ariadne_modules
    # shellcheck disable=SC1091
    [[ "${DRY_RUN}" -eq 0 ]] && source "${VENV}/bin/activate" || echo "+ source $(printf '%q' "${VENV}/bin/activate")"
    if [[ "${SKIP_PLUMED}" -eq 0 ]]; then
        export PLUMED_KERNEL="${PLUMED_KERNEL:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib/libplumedKernel.so}"
        export LD_LIBRARY_PATH="${PLUMED_LIBRARY_PATH:-${HOME}/opt/plumed-${PLUMED_VERSION}/lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    upsert_ichor_config
    final_checks
}

main "$@"
