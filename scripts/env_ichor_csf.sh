#!/usr/bin/env bash
# Compatibility entry point. New workflows should source env_ichor.sh.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced, not executed:" >&2
    echo "  source scripts/env_ichor_csf.sh" >&2
    exit 1
fi

_ichor_compat_source="${BASH_SOURCE[0]}"
_ichor_compat_parent="${_ichor_compat_source%/*}"
[[ "${_ichor_compat_parent}" != "${_ichor_compat_source}" ]] \
    || _ichor_compat_parent="."
_ichor_compat_script_dir="$(cd "${_ichor_compat_parent}" && pwd)"
# shellcheck disable=SC1091
source "${_ichor_compat_script_dir}/env_ichor.sh" "$@"
unset _ichor_compat_source _ichor_compat_parent _ichor_compat_script_dir
