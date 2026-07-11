"""Smoke tests for the sourced CSF runtime helper."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "env_ichor_csf.sh"
LIB = REPO_ROOT / "scripts" / "lib_ichor_csf.sh"


def test_env_script_is_present_and_documents_sourcing():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.R_OK)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "This script must be sourced" in text
    assert "source scripts/env_ichor_csf.sh csf3" in text
    assert "source scripts/env_ichor_csf.sh csf4" in text
    assert "source scripts/env_ichor_csf.sh auto" in text
    assert "source scripts/env_ichor_csf.sh --machine csf3" in text


def test_env_script_contains_required_runtime_contracts():
    text = SCRIPT.read_text(encoding="utf-8")
    lib_text = LIB.read_text(encoding="utf-8")
    assert 'source "${_ichor_env_script_dir}/lib_ichor_csf.sh"' in text
    assert "compilers/intel/oneapi/2025.0.1" in text
    assert "umf compiler-rt tbb compiler" in text
    assert "mkl/2025.0" in text
    assert "python/3.11.3-gcccore-12.3.0" in text
    assert "compilers/oneapi/2024.2.0" in text
    assert "mkl/2024.2" in text
    assert "unset CC CXX FC F77 F90" in text
    assert 'export ICHOR_MACHINE="${machine}"' in text
    assert "PLUMED_KERNEL" in text
    assert 'venv="${HOME}/.venv/ichor-${machine}"' in text
    assert "ichor_csf_deactivate_existing_venv" in text
    assert "ichor_csf_warn_path_hazards" in text
    assert "ichor_csf_path_inside" in text
    assert "--machine auto|csf3|csf4" in text
    assert "--env-check" in text
    assert "--debug, --trace" in text
    assert "mktemp" in text
    assert "import ariadne; assert hasattr" in text
    assert "plumed.Plumed" in text
    assert "ensure_xtb_ase_available" in text
    assert "ensure_plumed_available" in text
    assert "--smoke-heavy" in text
    assert "--print-env" in text
    assert "_ichor_env_print_env" in text
    assert "/opt/apps/etc/profile.d/modules.sh" in lib_text
    assert "/opt/apps/lmod/lmod/init/bash" in lib_text
    assert "ichor_csf_module_is_shell_function" in lib_text


def test_env_script_default_setup_does_not_smoke_imports():
    text = SCRIPT.read_text(encoding="utf-8")
    marker = 'if [[ "${do_smoke}" -eq 1 ]]; then'
    assert marker in text
    before_smoke = text[: text.index(marker)]
    assert "import ariadne; assert hasattr" not in before_smoke
    assert "plumed.Plumed" not in before_smoke


def test_env_script_bash_syntax():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available on this host")
    subprocess.run([bash, "-n", str(LIB)], check=True)
    subprocess.run([bash, "-n", str(SCRIPT)], check=True)


def test_env_script_refuses_direct_execution():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available on this host")
    result = subprocess.run(
        [bash, str(SCRIPT), "csf3"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must be sourced" in result.stderr
