"""Smoke tests for the unified CSF install script."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "install_ichor_csf.sh"


def _make_fake_projects(tmp_path: Path) -> Path:
    projects = tmp_path / "projects"
    (projects / "POLUS" / "polus_core_subpackage").mkdir(parents=True)
    (projects / "FEREBUS_CPU" / "pyferebus").mkdir(parents=True)
    (projects / "FEREBUS_CPU" / "libs").mkdir(parents=True)
    (projects / "ARIADNE").mkdir(parents=True)
    return projects


def _run_dry(machine: str, tmp_path: Path) -> subprocess.CompletedProcess:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available on this host")
    projects = _make_fake_projects(tmp_path)
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    (tmp_path / "home").mkdir()
    return subprocess.run(
        [
            bash,
            str(SCRIPT),
            "--dry-run",
            "--machine",
            machine,
            "--projects-dir",
            str(projects),
            "--repo-root",
            str(REPO_ROOT),
            "--yes",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_install_script_is_present():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.R_OK)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--machine auto|csf3|csf4" in text
    assert "PLUMED" in text
    assert "ARIADNE" in text
    assert "FEREBUS_CPU" in text
    assert "from xtb.ase.calculator import XTB" in text
    assert "pyferebus_platform\": \"CSF3\"" in text
    assert "pyferebus_platform\": \"CSF4\"" in text
    assert "libs/gcc/openssl/1.1.1w" in text
    assert "--with-openssl=" in text
    assert "--with-openssl-rpath=auto" in text
    assert "import ssl; print(ssl.OPENSSL_VERSION)" in text
    assert "command -v \"${cmd}\"" in text
    assert "resolve_required_cmd icx" in text
    assert "resolve_required_cmd icpx" in text
    assert "resolve_required_cmd ifx" in text
    assert "export CC=icx" not in text
    assert "unset CC CXX FC F77 F90" in text
    assert "export CC=gcc" in text
    assert "export CXX=g++" in text


def test_install_script_bash_syntax():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available on this host")
    subprocess.run([bash, "-n", str(SCRIPT)], check=True)


@pytest.mark.parametrize(
    ("machine", "venv_name"),
    [("csf3", "ichor-csf3"), ("csf4", "ichor-csf4")],
)
def test_install_script_dry_run_renders_cluster_defaults(machine, venv_name, tmp_path):
    result = _run_dry(machine, tmp_path)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"machine       = {machine}" in output
    assert venv_name in output
    assert "POLUS" in output
    assert "FEREBUS_CPU" in output
    assert "ARIADNE" in output
    assert "PLUMED" in output
    assert "source " in output
    assert f".venv/{venv_name}/bin/activate" in output.replace("\\", "/")
    assert "upsert " + machine + " profile in ~/ichor_config.yaml" in output


def test_install_script_dry_run_uses_parallel_build_flags(tmp_path):
    result = _run_dry("csf4", tmp_path)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "MAKEFLAGS=-j4" in output
    assert "make -j 4" in output
    assert "cmake --build" in output
    assert "-j 4" in output
