#!/usr/bin/env bash
# Compatibility entry point. New workflows should run install_ichor.sh.

set -euo pipefail
_ichor_compat_source="${BASH_SOURCE[0]}"
_ichor_compat_parent="${_ichor_compat_source%/*}"
[[ "${_ichor_compat_parent}" != "${_ichor_compat_source}" ]] \
    || _ichor_compat_parent="."
SCRIPT_DIR="$(cd "${_ichor_compat_parent}" && pwd)"
exec bash "${SCRIPT_DIR}/install_ichor.sh" "$@"
