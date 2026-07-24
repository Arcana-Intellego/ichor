#!/usr/bin/env bash
#
# Shared shell helpers for ICHOR installer and runtime scripts.
# This file is meant to be sourced, not executed.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: scripts/lib_ichor.sh must be sourced, not executed" >&2
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

ichor_csf_detect_machine_from_evidence() {
    local csf3_module_root_present="${1:-0}"
    local csf4_module_root_present="${2:-0}"
    shift 2

    local evidence lower short_name
    local evidence_csf3=0
    local evidence_csf4=0
    for evidence in "$@"; do
        lower="${evidence,,}"
        lower="${lower%%[[:space:]]*}"
        [[ -n "${lower}" ]] || continue
        short_name="${lower%%.*}"

        if [[ "${lower}" == *csf3* ]]; then
            evidence_csf3=1
        fi
        if [[ "${lower}" == *csf4* ]]; then
            evidence_csf4=1
        fi

        # CSF3 login nodes use names such as login1; CSF4 uses zero-padded
        # names such as login02. Match complete short hostnames only.
        if [[ "${short_name}" =~ ^login[1-9][0-9]*$ ]]; then
            evidence_csf3=1
        elif [[ "${short_name}" =~ ^login0[0-9]+$ ]]; then
            evidence_csf4=1
        fi
    done

    if [[ "${evidence_csf3}" -eq 1 && "${evidence_csf4}" -eq 1 ]]; then
        ichor_csf_error "conflicting CSF3/CSF4 hostname evidence; pass --machine csf3 or --machine csf4"
        return 1
    fi
    if [[ "${evidence_csf3}" -eq 1 ]]; then
        printf 'csf3\n'
        return 0
    fi
    if [[ "${evidence_csf4}" -eq 1 ]]; then
        printf 'csf4\n'
        return 0
    fi

    if [[ "${csf3_module_root_present}" -eq 1 && "${csf4_module_root_present}" -eq 1 ]]; then
        ichor_csf_error "both CSF3 and CSF4 module roots are visible; pass --machine csf3 or --machine csf4"
        return 1
    fi
    if [[ "${csf3_module_root_present}" -eq 1 ]]; then
        printf 'csf3\n'
        return 0
    fi
    if [[ "${csf4_module_root_present}" -eq 1 ]]; then
        printf 'csf4\n'
        return 0
    fi

    ichor_csf_error "could not auto-detect CSF3/CSF4; pass --machine csf3 or --machine csf4"
    return 1
}

ichor_detect_machine_from_evidence() {
    local csf3_module_root_present="${1:-0}"
    local csf4_module_root_present="${2:-0}"
    local ffluxlab_scheduler_present="${3:-0}"
    shift 3

    local evidence lower short_name
    local evidence_csf3=0
    local evidence_csf4=0
    local evidence_ffluxlab=0
    for evidence in "$@"; do
        lower="${evidence,,}"
        lower="${lower%%[[:space:]]*}"
        [[ -n "${lower}" ]] || continue
        short_name="${lower%%.*}"
        [[ "${lower}" == *csf3* ]] && evidence_csf3=1
        [[ "${lower}" == *csf4* ]] && evidence_csf4=1
        [[ "${lower}" == *ffluxlab* ]] && evidence_ffluxlab=1
        if [[ "${short_name}" =~ ^login[1-9][0-9]*$ ]]; then
            evidence_csf3=1
        elif [[ "${short_name}" =~ ^login0[0-9]+$ ]]; then
            evidence_csf4=1
        fi
    done
    if [[ "${ffluxlab_scheduler_present}" -eq 1 ]]; then
        evidence_ffluxlab=1
    fi

    if [[ "${ffluxlab_scheduler_present}" -eq 1 ]] \
        && [[ "${csf3_module_root_present}" -eq 1 \
            || "${csf4_module_root_present}" -eq 1 ]]; then
        ichor_csf_error \
            "conflicting scheduler and module-root evidence; pass --machine explicitly"
        return 1
    fi

    local detected=$((evidence_csf3 + evidence_csf4 + evidence_ffluxlab))
    if [[ "${detected}" -gt 1 ]]; then
        ichor_csf_error \
            "conflicting machine evidence; pass --machine csf3, csf4, or ffluxlab"
        return 1
    fi
    if [[ "${evidence_csf3}" -eq 1 ]]; then
        printf 'csf3\n'
        return 0
    fi
    if [[ "${evidence_csf4}" -eq 1 ]]; then
        printf 'csf4\n'
        return 0
    fi
    if [[ "${evidence_ffluxlab}" -eq 1 ]]; then
        printf 'ffluxlab\n'
        return 0
    fi

    if [[ "${csf3_module_root_present}" -eq 1 && "${csf4_module_root_present}" -eq 1 ]]; then
        ichor_csf_error \
            "both CSF3 and CSF4 module roots are visible; pass --machine explicitly"
        return 1
    fi
    if [[ "${csf3_module_root_present}" -eq 1 ]]; then
        printf 'csf3\n'
        return 0
    fi
    if [[ "${csf4_module_root_present}" -eq 1 ]]; then
        printf 'csf4\n'
        return 0
    fi
    ichor_csf_error \
        "could not auto-detect the platform; pass --machine csf3, csf4, or ffluxlab"
    return 1
}

ichor_detect_machine() {
    local requested="${1:-auto}"
    case "${requested}" in
        csf3|csf4|ffluxlab)
            printf '%s\n' "${requested}"
            return 0
            ;;
        auto) ;;
        *)
            ichor_csf_error "machine must be auto, csf3, csf4, or ffluxlab"
            return 1
            ;;
    esac

    local host_fqdn host_short host_plain host_environment
    host_fqdn="$(hostname -f 2>/dev/null || true)"
    host_short="$(hostname -s 2>/dev/null || true)"
    host_plain="$(hostname 2>/dev/null || true)"
    host_environment="${HOSTNAME:-}"

    local csf3_module_root_present=0
    local csf4_module_root_present=0
    local ffluxlab_scheduler_present=0
    [[ -d /opt/apps/etc/modulefiles/interpreters ]] && csf3_module_root_present=1
    [[ -d /opt/software/RI/apps ]] && csf4_module_root_present=1
    if [[ -d /home/modules ]] \
        && command -v qsub >/dev/null 2>&1 \
        && command -v qstat >/dev/null 2>&1; then
        ffluxlab_scheduler_present=1
    fi

    ichor_detect_machine_from_evidence \
        "${csf3_module_root_present}" \
        "${csf4_module_root_present}" \
        "${ffluxlab_scheduler_present}" \
        "${host_fqdn}" \
        "${host_short}" \
        "${host_plain}" \
        "${host_environment}"
}

ichor_csf_detect_machine() {
    ichor_detect_machine "$@"
}

ichor_csf_module_is_shell_function() {
    [[ "$(type -t module 2>/dev/null || true)" == "function" ]]
}

ichor_csf_initialise_modules() {
    if ichor_csf_module_is_shell_function; then
        return 0
    fi

    # An external module wrapper cannot mutate this script's PATH after load.
    # Resolve a shell function before accepting the module environment.
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
        /home/modules/compilers/intel/21.0.3 \
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
        /home/modules/compilers/intel/21.0.3/compiler/*/linux/bin/intel64/"${exe}" \
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
