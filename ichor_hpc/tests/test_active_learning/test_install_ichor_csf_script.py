"""Smoke tests for the unified CSF install script."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "install_ichor_csf.sh"
LIB = REPO_ROOT / "scripts" / "lib_ichor_csf.sh"
CANONICAL_PROFILE = REPO_ROOT / "ichor_config.yaml"
UPSERT = REPO_ROOT / "scripts" / "upsert_ichor_config.py"


def _make_fake_projects(tmp_path: Path) -> Path:
    projects = tmp_path / "projects"
    (projects / "FEREBUS_CPU" / "pyferebus").mkdir(
        parents=True,
        exist_ok=True,
    )
    (projects / "FEREBUS_CPU" / "libs").mkdir(parents=True, exist_ok=True)
    (projects / "ARIADNE").mkdir(parents=True, exist_ok=True)
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
    (tmp_path / "home").mkdir(exist_ok=True)
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
    profile_text = CANONICAL_PROFILE.read_text(encoding="utf-8")
    upsert_text = UPSERT.read_text(encoding="utf-8")
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
    assert 'init --campaign-dir "${smoke_dir}" --yes' in text
    assert ': > "${smoke_dir}/pool.xyz"' in text
    assert 'pyferebus_platform: "CSF3"' in profile_text
    assert 'pyferebus_platform: "CSF4"' in profile_text
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
    assert "cannot mutate this script's PATH" in lib_text
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
    assert "apps/binapps/gaussian/g09d01_em64t" in profile_text
    assert "$g09root/g09/g09" in profile_text
    assert "gaussian/g16c01_em64t_detectcpu" in profile_text
    assert "$g16root/g16/g16" in profile_text
    assert "OPENBLAS_SHA256=" in text
    assert 'verify_sha256 "${tarball}" "${OPENBLAS_SHA256}"' in text
    assert "./fetchOpenBlas.sh" not in text
    assert 'NO_SHARED=1 NUM_THREADS=64' in text
    assert 'scripts/upsert_ichor_config.py' in text
    assert 'canonical = _load_mapping(canonical_config' in upsert_text
    assert 'data[str(machine)] = profile' in upsert_text
    assert 'os.replace(temporary, path)' in upsert_text
    assert '_fsync_parent(path)' in upsert_text


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
    assert "POLUS" not in output
    assert "FEREBUS_CPU" in output
    assert "ARIADNE" in output
    assert "PLUMED" in output
    assert "source " in output
    assert f".venv/{venv_name}/bin/activate" in output.replace("\\", "/")
    assert (
        "initialise/update ~/ichor_config.yaml from repo template and upsert "
        + machine
        + " profile"
    ) in output


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


def test_install_script_config_uses_canonical_gaussian_profiles(tmp_path):
    csf3 = _run_dry("csf3", tmp_path, "--only", "config")
    csf4 = _run_dry("csf4", tmp_path, "--only", "config")

    assert csf3.returncode == 0, csf3.stdout + csf3.stderr
    assert csf4.returncode == 0, csf4.stdout + csf4.stderr

    text = SCRIPT.read_text(encoding="utf-8")
    upsert_text = UPSERT.read_text(encoding="utf-8")
    profile_text = CANONICAL_PROFILE.read_text(encoding="utf-8")
    assert 'profile = copy.deepcopy(raw_profile)' in upsert_text
    assert "apps/binapps/gaussian/g09d01_em64t" in profile_text
    assert "$g09root/g09/g09" in profile_text
    assert "gaussian/g16c01_em64t_detectcpu" in profile_text
    assert "$g16root/g16/g16" in profile_text
    assert '"multinode"' not in text


def test_profile_upsert_is_atomic_and_preserves_unrelated_profiles(tmp_path):
    destination = tmp_path / "ichor_config.yaml"
    original = {
        "private-cluster": {"operator_value": 17},
        "csf4": {"hpc": {"partitions": {"multinode": {}}}},
    }
    destination.write_text(
        yaml.safe_dump(original, sort_keys=False),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(UPSERT),
            "--destination",
            str(destination),
            "--canonical-config",
            str(CANONICAL_PROFILE),
            "--machine",
            "csf4",
            "--python-path",
            "$HOME/.venv/ichor-csf4/bin/python",
            "--python-library-path",
            "$HOME/opt/python-3.11.15/lib",
            "--aimall-path",
            "$HOME/AIMAll/aimqb.ish",
            "--ferebus-path",
            "$HOME/.local/bin/ferebus",
            "--plumed-kernel",
            "$HOME/opt/plumed-2.10.0/lib/libplumedKernel.so",
            "--plumed-library-path",
            "$HOME/opt/plumed-2.10.0/lib",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    installed = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert installed["private-cluster"] == {"operator_value": 17}
    assert "multinode" not in installed["csf4"]["hpc"]["partitions"]
    assert installed["csf4"]["software"]["python"]["python_path"].startswith(
        "$HOME/"
    )
    backups = list(tmp_path.glob("ichor_config.yaml.bak.*"))
    assert len(backups) == 1
    assert yaml.safe_load(backups[0].read_text(encoding="utf-8")) == original
    assert not list(tmp_path.glob("ichor_config.yaml.tmp.*"))


def test_canonical_csf_profiles_match_documented_limits(monkeypatch, tmp_path):
    from ichor.hpc.active_learning.daemon.cluster_profile import (
        ClusterProfile,
        validate_cluster_profile,
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    profiles = yaml.safe_load(CANONICAL_PROFILE.read_text(encoding="utf-8"))
    csf3 = profiles["csf3"]
    csf4 = profiles["csf4"]

    assert csf3["hpc"]["jobscript_shebang"] == "#!/bin/bash --login"
    assert csf3["hpc"]["partitions"]["serial"]["memory_per_core_gb"] == 4
    assert csf3["hpc"]["partitions"]["interactive"]["max_walltime_hours"] == 6
    assert csf3["software"]["python"]["library_path"].endswith(
        "/python-3.11.15/lib"
    )
    assert csf4["hpc"]["jobscript_shebang"] == "#!/bin/bash --login"
    assert csf4["hpc"]["partitions"]["multicore"]["max_cpus"] == 40
    assert "multinode" not in csf4["hpc"]["partitions"]
    validate_cluster_profile(ClusterProfile(machine="csf3", config=profiles))
    validate_cluster_profile(ClusterProfile(machine="csf4", config=profiles))


def test_cluster_profile_rejects_relative_submitted_python_path(monkeypatch, tmp_path):
    from ichor.hpc.active_learning.daemon.cluster_profile import (
        ClusterProfile,
        ClusterProfileError,
        validate_cluster_profile,
    )

    profiles = yaml.safe_load(CANONICAL_PROFILE.read_text(encoding="utf-8"))
    profiles["csf3"]["software"]["python"]["python_path"] = "venv/bin/python"
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ClusterProfileError, match="must resolve absolutely"):
        validate_cluster_profile(ClusterProfile(machine="csf3", config=profiles))


def test_cluster_guides_do_not_duplicate_machine_profile_yaml():
    for relative in (
        "examples/csf3_first_live_iter/README.md",
        "examples/csf4_first_live_iter/README.md",
    ):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "```yaml\ncsf3:" not in text
        assert "```yaml\ncsf4:" not in text
        assert "single canonical" in text
