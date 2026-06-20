"""Smoke tests for the unified CSF install script."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "install_ichor_csf.sh"
LIB = REPO_ROOT / "scripts" / "lib_ichor_csf.sh"


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
    lib_text = LIB.read_text(encoding="utf-8")
    assert "--machine auto|csf3|csf4" in text
    assert "--only all|python|packages|ariadne|plumed|ferebus|config|verify|doctor" in text
    assert "--debug, --trace" in text
    assert "trap on_error ERR" in text
    assert "stage_doctor()" in text
    assert "ariadne-csf3-last-install.json" in text or "ariadne-${MACHINE}-last-install.json" in text
    assert "Manual ARIADNE recovery block" in text
    assert "PLUMED" in text
    assert "ARIADNE" in text
    assert "FEREBUS_CPU" in text
    assert "from xtb.ase.calculator import XTB" in text
    assert "pyferebus_platform\": \"CSF3\"" in text
    assert "pyferebus_platform\": \"CSF4\"" in text
    assert "libs/gcc/openssl/1.1.1w" in text
    assert "--with-openssl=" in text
    assert "--with-openssl-rpath=auto" in text
    assert "source \"${SCRIPT_DIR}/lib_ichor_csf.sh\"" in text
    assert "ichor_csf_module_is_shell_function()" in lib_text
    assert "ichor_csf_module_debug()" in lib_text
    assert "ichor_csf_find_ariadne_compiler_path()" in lib_text
    assert "setvars.sh" not in lib_text
    assert "source_oneapi_setvars_if_available" not in text
    assert "/opt/apps/compilers/intel/oneapi/2025.0.1/compiler/2025.0/bin" in lib_text
    assert '"/compiler/*/bin/"${exe}"' in lib_text
    assert "*/compiler/*/opt/compiler/lib" in lib_text
    assert "/opt/apps/etc/profile.d/modules.sh" in lib_text
    assert "/opt/apps/lmod/lmod/init/bash" in lib_text
    assert '[[ "$(type -t module' in lib_text
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
    assert "assert_ariadne_inside_venv" in text
    assert "write_ariadne_receipt" in text
    assert "print_ariadne_import_info \"ARIADNE before install\"" in text
    assert "print_ariadne_import_info \"ARIADNE after install\"" in text
    assert "run_in_dir \"${ariadne_root}\"" in text
    assert "bash -lc" not in text
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
    subprocess.run([bash, "-n", str(LIB)], check=True)
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
    ["python", "packages", "ariadne", "plumed", "ferebus", "config", "verify", "doctor"],
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
    assert "clean ARIADNE-local build artefacts" in output
    assert "write ARIADNE install receipt" in output
    assert "module load compilers/intel/oneapi/2025.0.1" in output
    assert "module load umf compiler-rt tbb compiler" in output
    assert "module load mkl/2025.0" in output
    assert "cmake --build" not in output


def test_install_script_verifies_ariadne_compilers_after_module_load():
    text = SCRIPT.read_text(encoding="utf-8")
    lib_text = LIB.read_text(encoding="utf-8")
    marker = "load_ariadne_modules()"
    start = text.index(marker)
    body = text[start:text.index("\n}\n\nresolve_ariadne_compilers", start)]
    assert "ensure_ariadne_compilers_on_path" in body

    helper_start = text.index("ensure_ariadne_compilers_on_path()")
    helper_body = text[helper_start:text.index("\n}\n\nmodule_cmd", helper_start)]
    assert "for compiler in icx icpx ifx" in helper_body
    assert "ichor_csf_find_ariadne_compiler_path" in helper_body
    assert "ichor_csf_prepend_path_once" in helper_body
    assert "ARIADNE compiler check after module load" in helper_body
    assert "ARIADNE compiler modules did not expose icx/icpx/ifx" in helper_body
    assert "module_debug" in helper_body
    assert "known-bin" in lib_text
    assert "setvars" not in lib_text


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
