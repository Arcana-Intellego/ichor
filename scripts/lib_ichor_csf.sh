#!/usr/bin/env bash
#
# Shared shell helpers for ICHOR CSF3/CSF4 installer and runtime scripts.
# This file is meant to be sourced, not executed.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: scripts/lib_ichor_csf.sh must be sourced, not executed" >&2
    exit 1
fi

ichor_csf_note() {
    echo ""
    echo "==> $*"
}

ichor_csf_warn() {
    echo "WARNING: $*" >&2
}

ichor_csf_error() {
    echo "ERROR: $*" >&2
}

ichor_csf_expand_path() {
    local value="$1"
    value="${value/#\~/${HOME}}"
    printf '%s\n' "${value}"
}

ichor_csf_detect_machine() {
    local requested="${1:-auto}"
    if [[ "${requested}" != "auto" ]]; then
        printf '%s\n' "${requested}"
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
        ichor_csf_error "could not auto-detect CSF3/CSF4 from hostname '${host}'"
        return 1
    fi
}

ichor_csf_module_is_shell_function() {
    [[ "$(type -t module 2>/dev/null || true)" == "function" ]]
}

ichor_csf_initialise_modules() {
    if ichor_csf_module_is_shell_function; then
        return 0
    fi

    local init_file
    for init_file in \
        /etc/profile.d/modules.sh \
        /usr/share/Modules/init/bash \
        /usr/share/lmod/lmod/init/bash \
        /opt/apps/etc/profile.d/modules.sh \
        /opt/apps/Modules/init/bash \
        /opt/apps/modules/init/bash \
        /opt/apps/lmod/lmod/init/bash; do
        if [[ -f "${init_file}" ]]; then
            # shellcheck disable=SC1090
            source "${init_file}" || true
        fi
        if ichor_csf_module_is_shell_function; then
            ICHOR_CSF_MODULE_INIT="${init_file}"
            return 0
        fi
    done

    if command -v modulecmd >/dev/null 2>&1; then
        module() { eval "$(modulecmd bash "$@")"; }
        if ichor_csf_module_is_shell_function; then
            ICHOR_CSF_MODULE_INIT="modulecmd"
            return 0
        fi
    fi

    if command -v lmod >/dev/null 2>&1; then
        module() { eval "$(lmod bash "$@")"; }
        if ichor_csf_module_is_shell_function; then
            ICHOR_CSF_MODULE_INIT="lmod"
            return 0
        fi
    fi

    return 1
}

ichor_csf_module_debug() {
    local context="${1:-module environment diagnostics}"
    {
        echo "Module diagnostics (${context}):"
        echo "  module type: $(type -t module 2>/dev/null || echo unavailable)"
        echo "  module init: ${ICHOR_CSF_MODULE_INIT:-unresolved}"
        echo "  module path: $(command -v module 2>/dev/null || echo unavailable)"
        echo "  PATH=${PATH:-}"
        echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        echo "  LOADEDMODULES=${LOADEDMODULES:-}"
        if ichor_csf_module_is_shell_function; then
            module list || true
        fi
    } >&2
}

ichor_csf_module() {
    if ! ichor_csf_initialise_modules; then
        ichor_csf_module_debug "module initialisation failed"
        ichor_csf_error "module command is unavailable as a shell function on this shell"
        return 1
    fi
    if ! module "$@"; then
        ichor_csf_module_debug "module $* failed"
        ichor_csf_error "module command failed: module $*"
        return 1
    fi
    hash -r 2>/dev/null || true
}

ichor_csf_deactivate_existing_venv() {
    local reason="${1:-environment setup}"
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        return 0
    fi
    ichor_csf_warn "Deactivating active venv ${VIRTUAL_ENV} before ${reason}"
    local active_bin="${VIRTUAL_ENV}/bin"
    local new_path=""
    local part
    local -a path_parts
    IFS=':' read -r -a path_parts <<< "${PATH:-}"
    for part in "${path_parts[@]}"; do
        if [[ -z "${part}" || "${part}" == "${active_bin}" ]]; then
            continue
        fi
        if [[ -z "${new_path}" ]]; then
            new_path="${part}"
        else
            new_path="${new_path}:${part}"
        fi
    done
    export PATH="${new_path}"
    unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
    hash -r 2>/dev/null || true
}

ichor_csf_find_oneapi_setvars() {
    local candidate
    for candidate in \
        "${ONEAPIDIR:-}/setvars.sh" \
        "${ONEAPI_ROOT:-}/setvars.sh" \
        /opt/apps/compilers/intel/oneapi/2025.0.1/setvars.sh \
        /opt/apps/compilers/oneapi/2024.2.0/setvars.sh; do
        if [[ -n "${candidate}" && "${candidate}" != "/setvars.sh" && -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

ichor_csf_source_oneapi_setvars() {
    local setvars
    setvars="$(ichor_csf_find_oneapi_setvars || true)"
    if [[ -z "${setvars}" ]]; then
        return 1
    fi
    # shellcheck disable=SC1090
    if source "${setvars}" >/dev/null 2>&1; then
        hash -r 2>/dev/null || true
        printf '%s\n' "${setvars}"
        return 0
    fi
    return 1
}

ichor_csf_find_ariadne_compiler_path() {
    local exe="$1"
    local resolved
    resolved="$(command -v "${exe}" 2>/dev/null || true)"
    if [[ -n "${resolved}" ]]; then
        printf '%s|PATH\n' "${resolved}"
        return 0
    fi

    local root candidate path_part compiler_root
    for root in \
        "${ONEAPIDIR:-}" \
        "${ONEAPI_ROOT:-}" \
        /opt/apps/compilers/intel/oneapi/2025.0.1 \
        /opt/apps/compilers/oneapi/2024.2.0; do
        if [[ -z "${root}" || ! -d "${root}" ]]; then
            continue
        fi
        for candidate in \
            "${root}"/compiler/*/bin/"${exe}" \
            "${root}"/compiler/latest/bin/"${exe}"; do
            if [[ -x "${candidate}" ]]; then
                printf '%s|known-root\n' "${candidate}"
                return 0
            fi
        done
    done

    for candidate in \
        /opt/apps/compilers/intel/oneapi/2025.0.1/compiler/2025.0/bin/"${exe}" \
        /opt/apps/compilers/oneapi/2024.2.0/compiler/latest/bin/"${exe}"; do
        if [[ -x "${candidate}" ]]; then
            printf '%s|known-bin\n' "${candidate}"
            return 0
        fi
    done

    local -a library_parts
    IFS=':' read -r -a library_parts <<< "${LD_LIBRARY_PATH:-}"
    for path_part in "${library_parts[@]}"; do
        case "${path_part}" in
            */compiler/*/lib)
                compiler_root="$(dirname "${path_part}")"
                candidate="${compiler_root}/bin/${exe}"
                ;;
            */compiler/*/opt/compiler/lib)
                compiler_root="$(cd "${path_part}/../../.." 2>/dev/null && pwd || true)"
                candidate="${compiler_root}/bin/${exe}"
                ;;
            *)
                candidate=""
                ;;
        esac
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            printf '%s|ld-library-path\n' "${candidate}"
            return 0
        fi
    done

    if ichor_csf_source_oneapi_setvars >/dev/null; then
        resolved="$(command -v "${exe}" 2>/dev/null || true)"
        if [[ -n "${resolved}" ]]; then
            printf '%s|setvars\n' "${resolved}"
            return 0
        fi
    fi

    return 1
}

ichor_csf_prepend_path_once() {
    local path="$1"
    if [[ -z "${path}" ]]; then
        return 0
    fi
    case ":${PATH:-}:" in
        *":${path}:"*) ;;
        *) export PATH="${path}${PATH:+:${PATH}}" ;;
    esac
}

ichor_csf_path_inside() {
    local child="$1"
    local parent="$2"
    case "${child}" in
        "${parent}"/*) return 0 ;;
        *) return 1 ;;
    esac
}

ichor_csf_warn_path_hazards() {
    local venv="$1"
    local venv_bin="${venv}/bin"
    local seen_venv=0
    local part
    local -a path_parts
    IFS=':' read -r -a path_parts <<< "${PATH:-}"
    for part in "${path_parts[@]}"; do
        if [[ "${part}" == "${venv_bin}" ]]; then
            seen_venv=1
        fi
        if [[ "${part}" == "${HOME}/.local/bin" && "${seen_venv}" -eq 0 ]]; then
            ichor_csf_warn "${HOME}/.local/bin appears before target venv bin in PATH"
            return 0
        fi
    done
}

ichor_csf_command_path() {
    command -v "$1" 2>/dev/null || true
}
