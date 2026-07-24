#!/usr/bin/env bash
# Compatibility entry point. New scripts should source lib_ichor.sh.

_ichor_compat_source="${BASH_SOURCE[0]}"
_ichor_compat_parent="${_ichor_compat_source%/*}"
[[ "${_ichor_compat_parent}" != "${_ichor_compat_source}" ]] \
    || _ichor_compat_parent="."
_ichor_compat_script_dir="$(cd "${_ichor_compat_parent}" && pwd)"
# shellcheck disable=SC1091
source "${_ichor_compat_script_dir}/lib_ichor.sh"
unset _ichor_compat_source _ichor_compat_parent _ichor_compat_script_dir
