"""Smoke tests for the sourced CSF runtime helper."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "env_ichor_csf.sh"


def test_env_script_is_present_and_documents_sourcing():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.R_OK)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "This script must be sourced" in text
    assert "source scripts/env_ichor_csf.sh csf3" in text
    assert "source scripts/env_ichor_csf.sh csf4" in text
    assert "source scripts/env_ichor_csf.sh auto" in text


def test_env_script_contains_required_runtime_contracts():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "compilers/intel/oneapi/2025.0.1" in text
    assert "umf compiler-rt tbb compiler" in text
    assert "mkl/2025.0" in text
    assert "python/3.11.3-gcccore-12.3.0" in text
    assert "compilers/oneapi/2024.2.0" in text
    assert "mkl/2024.2" in text
    assert "unset CC CXX FC F77 F90" in text
    assert 'export ICHOR_MACHINE="${machine}"' in text
    assert "PLUMED_KERNEL" in text
    assert "ichor-csf3" in text
    assert "ichor-csf4" in text
    assert "_ichor_env_deactivate_existing_venv" in text
    assert "import ariadne; assert hasattr" in text
    assert "plumed.Plumed" in text
    assert "ensure_xtb_ase_available" in text
    assert "ensure_plumed_available" in text
    assert "--smoke-heavy" in text
    assert "--print-env" in text
    assert "_ichor_env_print_env" in text


def test_env_script_bash_syntax():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available on this host")
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
