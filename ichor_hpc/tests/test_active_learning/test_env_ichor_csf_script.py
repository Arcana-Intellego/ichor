"""Smoke tests for the sourced CSF runtime helper."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "env_ichor_csf.sh"
LIB = REPO_ROOT / "scripts" / "lib_ichor_csf.sh"


def _bash() -> str:
    bash = shutil.which("bash")
    if bash:
        return bash
    windows_roots = (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"))
    for root in filter(None, windows_roots):
        candidate = Path(root) / "Git" / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    pytest.skip("bash is not available on this host")


def _run_bash(
    source: str,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bash(), "-c", source],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _make_fake_venv(tmp_path: Path, machine: str) -> tuple[Path, dict[str, str]]:
    home = tmp_path / "home"
    venv = home / ".venv" / f"ichor-{machine}"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    activate = bin_dir / "activate"
    activate.write_text(
        "\n".join(
            (
                f'export VIRTUAL_ENV={shlex.quote(venv.as_posix())}',
                f'export PATH={shlex.quote(bin_dir.as_posix())}:"$PATH"',
                "",
            )
        ),
        encoding="utf-8",
    )
    for command in ("python", "ichor-cli", "ichor-al-daemon"):
        executable = bin_dir / command
        executable.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"${1:-}\" == \"-V\" ]]; then echo 'Python 3.11 test'; fi\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment["HOME"] = home.as_posix()
    environment["HOSTNAME"] = ""
    return venv, environment


def test_env_script_is_present_and_documents_sourcing():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.R_OK)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "This script must be sourced" in text
    assert "source scripts/env_ichor_csf.sh\n" in text
    assert "source scripts/env_ichor_csf.sh --machine csf3" in text
    assert "source scripts/env_ichor_csf.sh --machine csf4" in text


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
    assert "--machine csf3|csf4" in text
    assert "ichor_csf_detect_machine_from_evidence" in lib_text
    assert "hostname -f" in lib_text
    assert "hostname -s" in lib_text
    assert "pass --machine csf3 or --machine csf4" in lib_text
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


def test_machine_detection_precedes_runtime_environment_mutation():
    text = SCRIPT.read_text(encoding="utf-8")
    detection = 'machine="$(ichor_csf_detect_machine "${machine}")"'
    assert text.index(detection) < text.index("ichor_csf_deactivate_existing_venv")
    module_load = '_ichor_env_load_runtime_modules "${machine}" "${do_purge}"'
    assert text.index(detection) < text.index(module_load)


def test_env_script_bash_syntax():
    bash = _bash()
    subprocess.run([bash, "-n", str(LIB)], check=True)
    subprocess.run([bash, "-n", str(SCRIPT)], check=True)


def test_env_script_refuses_direct_execution():
    bash = _bash()
    result = subprocess.run(
        [bash, str(SCRIPT), "csf3"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must be sourced" in result.stderr


@pytest.mark.parametrize(
    ("machine", "fqdn", "short_name"),
    (
        ("csf3", "login1.csf3.itservices.manchester.ac.uk", "login1"),
        ("csf4", "login02.csf4.itservices.manchester.ac.uk", "login02"),
    ),
)
def test_env_script_sources_without_machine_argument(
    tmp_path: Path,
    machine: str,
    fqdn: str,
    short_name: str,
):
    venv, environment = _make_fake_venv(tmp_path, machine)
    source = f"""
module() {{ return 0; }}
hostname() {{
    case "${{1:-}}" in
        -f) printf '%s\\n' {shlex.quote(fqdn)} ;;
        -s|'') printf '%s\\n' {shlex.quote(short_name)} ;;
        *) return 1 ;;
    esac
}}
source {shlex.quote(SCRIPT.as_posix())}
printf 'machine=%s\\nvenv=%s\\n' "$ICHOR_MACHINE" "$VIRTUAL_ENV"
"""
    result = _run_bash(source, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"machine={machine}" in result.stdout
    assert f"venv={venv.as_posix()}" in result.stdout


def test_env_script_accepts_options_first_with_automatic_detection(tmp_path: Path):
    _, environment = _make_fake_venv(tmp_path, "csf3")
    source = f"""
module() {{ return 0; }}
hostname() {{ printf 'login1\\n'; }}
source {shlex.quote(SCRIPT.as_posix())} --quiet --no-purge
printf 'machine=%s\\n' "$ICHOR_MACHINE"
"""
    result = _run_bash(source, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "machine=csf3"


def test_env_script_is_idempotent_for_native_library_paths(tmp_path: Path):
    _, environment = _make_fake_venv(tmp_path, "csf3")
    environment["LD_LIBRARY_PATH"] = "/module/runtime/lib"
    source = f"""
module() {{ return 0; }}
hostname() {{ printf 'login1\n'; }}
source {shlex.quote(SCRIPT.as_posix())} --quiet --no-purge
first="$LD_LIBRARY_PATH"
source {shlex.quote(SCRIPT.as_posix())} --quiet --no-purge
second="$LD_LIBRARY_PATH"
printf 'first=%s\nsecond=%s\n' "$first" "$second"
"""

    result = _run_bash(source, env=environment)

    assert result.returncode == 0, result.stdout + result.stderr
    values = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith(("first=", "second="))
    )
    assert values["first"] == values["second"]
    parts = values["second"].split(":")
    assert sum(
        part.endswith("/opt/python-3.11.15/lib") for part in parts
    ) == 1
    assert sum(
        part.endswith("/opt/plumed-2.10.0/lib") for part in parts
    ) == 1


@pytest.mark.parametrize(
    ("csf3_root", "csf4_root", "evidence", "message"),
    (
        (0, 0, ("login1", "login02"), "conflicting CSF3/CSF4 hostname evidence"),
        (1, 1, ("unknown-host",), "both CSF3 and CSF4 module roots are visible"),
        (0, 0, ("unknown-host",), "could not auto-detect CSF3/CSF4"),
    ),
)
def test_machine_evidence_fails_closed(
    csf3_root: int,
    csf4_root: int,
    evidence: tuple[str, ...],
    message: str,
):
    arguments = " ".join(shlex.quote(value) for value in evidence)
    source = (
        f"source {shlex.quote(LIB.as_posix())}; "
        f"ichor_csf_detect_machine_from_evidence {csf3_root} {csf4_root} {arguments}"
    )
    result = _run_bash(source)
    assert result.returncode != 0
    assert message in result.stderr
    assert "--machine csf3 or --machine csf4" in result.stderr


@pytest.mark.parametrize(
    ("csf3_root", "csf4_root", "evidence", "expected"),
    (
        (0, 0, ("login1",), "csf3"),
        (0, 0, ("login02",), "csf4"),
        (0, 0, ("worker.csf3.itservices.manchester.ac.uk",), "csf3"),
        (0, 0, ("worker.csf4.itservices.manchester.ac.uk",), "csf4"),
        (1, 0, ("unknown-host",), "csf3"),
        (0, 1, ("unknown-host",), "csf4"),
    ),
)
def test_machine_evidence_resolves_supported_sources(
    csf3_root: int,
    csf4_root: int,
    evidence: tuple[str, ...],
    expected: str,
):
    arguments = " ".join(shlex.quote(value) for value in evidence)
    source = (
        f"source {shlex.quote(LIB.as_posix())}; "
        f"ichor_csf_detect_machine_from_evidence {csf3_root} {csf4_root} {arguments}"
    )
    result = _run_bash(source)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    ("fqdn", "short_name", "plain", "environment_host", "expected"),
    (
        ("node.csf3.example", "unknown", "unknown", "unknown", "csf3"),
        ("unknown", "login02", "unknown", "unknown", "csf4"),
        ("unknown", "unknown", "login1", "unknown", "csf3"),
        ("unknown", "unknown", "unknown", "node.csf4.example", "csf4"),
    ),
)
def test_machine_detector_consults_every_hostname_source(
    fqdn: str,
    short_name: str,
    plain: str,
    environment_host: str,
    expected: str,
):
    source = f"""
source {shlex.quote(LIB.as_posix())}
hostname() {{
    case "${{1:-}}" in
        -f) printf '%s\\n' {shlex.quote(fqdn)} ;;
        -s) printf '%s\\n' {shlex.quote(short_name)} ;;
        '') printf '%s\\n' {shlex.quote(plain)} ;;
        *) return 1 ;;
    esac
}}
HOSTNAME={shlex.quote(environment_host)}
ichor_csf_detect_machine auto
"""
    result = _run_bash(source)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("machine", ("csf3", "csf4"))
def test_explicit_machine_override_remains_available(machine: str):
    source = (
        f"source {shlex.quote(LIB.as_posix())}; "
        f"ichor_csf_detect_machine {shlex.quote(machine)}"
    )
    result = _run_bash(source)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == machine
