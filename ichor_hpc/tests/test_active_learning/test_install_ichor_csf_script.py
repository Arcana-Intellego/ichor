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


def _run_dry(
    machine: str,
    tmp_path: Path,
    *extra_args: str,
) -> subprocess.CompletedProcess:
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
            *extra_args,
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
    assert "--only all|python|packages|ariadne|plumed|ferebus|config|verify" in text
    assert "PLUMED" in text
    assert "ARIADNE" in text
    assert "FEREBUS_CPU" in text
    assert "from xtb.ase.calculator import XTB" in text
    assert "pyferebus_platform\": \"CSF3\"" in text
    assert "pyferebus_platform\": \"CSF4\"" in text
    assert "libs/gcc/openssl/1.1.1w" in text
    assert "--with-openssl=" in text
    assert "--with-openssl-rpath=auto" in text
    assert "module_is_current_shell_function()" in text
    assert "module_debug()" in text
    assert "/opt/apps/etc/profile.d/modules.sh" in text
    assert "/opt/apps/lmod/lmod/init/bash" in text
    assert '[[ "$(type -t module' in text
    assert "cannot mutate this script's PATH" in text
    assert "import ssl; print(ssl.OPENSSL_VERSION)" in text
    assert "command -v \"${cmd}\"" in text
    assert "resolve_ariadne_compilers()" in text
    assert "resolve_required_cmd_into ARIADNE_CC icx" in text
    assert "resolve_required_cmd_into ARIADNE_CXX icpx" in text
    assert "resolve_required_cmd_into ARIADNE_FC ifx" in text
    assert "deactivate_existing_venv" in text
    assert "export CC=\"${ARIADNE_CC}\"" in text
    assert "export CXX=\"${ARIADNE_CXX}\"" in text
    assert "export FC=\"${ARIADNE_FC}\"" in text
    assert "export CC=\"$(resolve_required_cmd" not in text
    assert "export CC=icx" not in text
    assert "ARIADNE_SAFE_IFX_FLAGS=ON" in text
    assert "--force-reinstall --no-deps" in text
    assert 'if [[ "${MACHINE}" == "csf3" ]]' in text
    assert "unset CC CXX FC F77 F90" in text
    assert "export CC=gcc" in text
    assert "export CXX=g++" in text
    assert "apps/binapps/gaussian/g09d01_em64t" in text
    assert "$g09root/g09/g09" in text
    assert "gaussian/g16c01_em64t_detectcpu" in text
    assert "$g16root/g16/g16" in text


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
    assert "only          = all" in output
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


@pytest.mark.parametrize(
    "stage",
    ["python", "packages", "ariadne", "plumed", "ferebus", "config", "verify"],
)
def test_install_script_dry_run_accepts_only_stages(stage, tmp_path):
    result = _run_dry("csf3", tmp_path, "--only", stage)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"only          = {stage}" in output


def test_install_script_dry_run_ariadne_stage_reinstalls_only_ariadne(tmp_path):
    result = _run_dry("csf3", tmp_path, "--only", "ariadne")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "resolve command icx" in output
    assert "--force-reinstall --no-deps" in output
    assert "ARIADNE_SAFE_IFX_FLAGS=ON" in output
    assert "module load compilers/intel/oneapi/2025.0.1" in output
    assert "module load umf compiler-rt tbb compiler" in output
    assert "module load mkl/2025.0" in output
    assert "cmake --build" not in output


def test_install_script_verifies_ariadne_compilers_after_module_load():
    text = SCRIPT.read_text(encoding="utf-8")
    marker = "load_ariadne_modules()"
    start = text.index(marker)
    body = text[start:text.index("\n}\n\nresolve_ariadne_compilers", start)]
    assert "for compiler in icx icpx ifx" in body
    assert "ARIADNE compiler check after module load" in body
    assert "ARIADNE compiler modules did not expose icx/icpx/ifx" in body
    assert "module_debug" in body


def test_install_script_dry_run_config_stage_preserves_gaussian_profiles(tmp_path):
    csf3 = _run_dry("csf3", tmp_path, "--only", "config")
    csf4 = _run_dry("csf4", tmp_path, "--only", "config")

    assert csf3.returncode == 0, csf3.stdout + csf3.stderr
    assert csf4.returncode == 0, csf4.stdout + csf4.stderr

    text = SCRIPT.read_text(encoding="utf-8")
    assert "apps/binapps/gaussian/g09d01_em64t" in text
    assert "$g09root/g09/g09" in text
    assert "gaussian/g16c01_em64t_detectcpu" in text
    assert "$g16root/g16/g16" in text
