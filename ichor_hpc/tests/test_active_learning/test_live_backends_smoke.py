"""Live-backend smoke tests.

Every test in this file is decorated with one or more skip-if-absent
guards (sbatch, sacct, Gaussian, AIMAll, FEREBUS, ariadne). On Windows /
off-cluster, all tests SKIP cleanly with a clear reason; on a CSF4 login
node (or any host where the binary is present) they run for real.

These tests do NOT consume real core-hours: they invoke each binary with
its trivial smoke argument (typically --version) and parse the result.
The most expensive test submits a one-second sleep via sbatch and checks
the JobID round-trips through sacct -- core-hour cost ~0.

Run only the live tests when on the cluster:

    pytest -m live ichor_hpc/tests/test_active_learning/test_live_backends_smoke.py
"""
from __future__ import annotations

import json
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon import live_executor as live_executor_mod
from ichor.hpc.active_learning.daemon.live_executor import (
    LiveBackendNotAvailableError,
    LiveBackendsPhaseExecutor,
    build_sbatch_script,
)
from ichor.hpc.active_learning.daemon.phase_executor import BackendSubmissionError
from ichor.hpc.active_learning.daemon.preflight import (
    BackendAvailability,
    check_backends,
    missing_backend_message,
)
from ichor.hpc.active_learning.daemon.submitted_environment_smoke import (
    SMOKE_SUCCESS_MARKER,
    render_submitted_environment_smoke_script,
    run_submitted_environment_smoke,
)
from ichor.hpc.active_learning.daemon.runtime_environment import (
    SUBMITTED_PYTHON_IMPORTS,
)


# --- skip-if-absent helpers ------------------------------------------------


def _which(name: str) -> str:
    return shutil.which(name) or ""


requires_sbatch = pytest.mark.skipif(
    not _which("sbatch"), reason="sbatch not on PATH"
)
requires_sacct = pytest.mark.skipif(
    not _which("sacct"), reason="sacct not on PATH"
)
requires_gaussian = pytest.mark.skipif(
    not _which("g16"),
    reason="g16 not on PATH",
)
requires_aimall = pytest.mark.skipif(
    not (_which("aimqb.ish") or _which("aimqb")),
    reason="aimqb.ish / aimqb not on PATH",
)
requires_ferebus = pytest.mark.skipif(
    not (_which("FEREBUS") or _which("ferebus")),
    reason="FEREBUS not on PATH",
)

try:
    import ariadne as _ariadne_module  # type: ignore[import]
    HAS_ARIADNE = True
except Exception:
    HAS_ARIADNE = False

requires_ariadne = pytest.mark.skipif(
    not HAS_ARIADNE, reason="ariadne not importable"
)


def _install_fake_global_variables(monkeypatch, config, machine):
    fake_global_variables = ModuleType("ichor.hpc.global_variables")
    fake_global_variables.ICHOR_CONFIG = config
    fake_global_variables.MACHINE = machine

    def fake_get_param_from_config(config_obj, *keys, default=None):
        value = config_obj
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    fake_global_variables.get_param_from_config = fake_get_param_from_config
    monkeypatch.setitem(
        sys.modules,
        "ichor.hpc.global_variables",
        fake_global_variables,
    )


def _install_fake_script_render_profile(monkeypatch):
    python_path = "/opt/ichor-test/bin/python"
    gaussian_module = "apps/binapps/gaussian/g16c01_em64t_detectcpu"
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "scheduler": "slurm",
                    "jobscript_shebang": "#!/bin/bash --login",
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                    "parallel_environments": {"multicore": [1, 168]},
                },
                "software": {
                    "python": {
                        "modules": [],
                        "python_path": python_path,
                    },
                    "gaussian": {
                        "modules": [gaussian_module],
                        "executable_path": "g16",
                        "scratch_root": "/scratch/$USER",
                    },
                    "ariadne_runtime": {"modules": []},
                },
            },
        },
        "csf3",
    )
    return python_path, gaussian_module


# --- preflight introspection (always run) ----------------------------------


def test_ichor_machine_env_override_selects_profile(monkeypatch):
    from ichor.hpc.useful_functions.get_machine import init_machine

    monkeypatch.setenv("ICHOR_MACHINE", "csf3")
    assert init_machine("login3", {"csf3": {}, "csf4": {}}) == "csf3"


def test_ichor_machine_env_override_rejects_unknown_profile(monkeypatch):
    from ichor.hpc.useful_functions.get_machine import init_machine

    monkeypatch.setenv("ICHOR_MACHINE", "missing")
    with pytest.raises(ValueError, match="ICHOR_MACHINE"):
        init_machine("login3", {"csf3": {}, "csf4": {}})


def test_check_backends_returns_structured_result():
    a = check_backends()
    assert isinstance(a, BackendAvailability)
    assert isinstance(a.all_present, bool)
    # Each component flag is a bool, mirrored to its absolute path string.
    assert isinstance(a.sbatch, bool)
    assert isinstance(a.sacct, bool)
    assert isinstance(a.gaussian, bool)
    assert isinstance(a.aimall, bool)
    assert isinstance(a.ferebus, bool)
    assert isinstance(a.ariadne, bool)


def test_quiet_import_module_suppresses_import_time_banner(monkeypatch, capsys):
    from ichor.hpc.active_learning.daemon import import_utils

    def noisy_import(module_name):
        print("backend banner on stdout")
        print("backend banner on stderr", file=sys.stderr)
        return SimpleNamespace(module_name=module_name)

    monkeypatch.setattr(import_utils.importlib, "import_module", noisy_import)

    module = import_utils.quiet_import_module("example.noisy_backend")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert module.module_name == "example.noisy_backend"


def test_default_profile_is_not_live_active_learning_profile(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {"_default": {"hpc": {"scheduler": "slurm"}}},
        "_default",
    )

    a = check_backends()

    assert a.profile is False
    assert "_default" in a.profile_error
    with pytest.raises(BackendSubmissionError, match="_default"):
        build_sbatch_script(
            phase_name="PHASE_A_DIVERSITY",
            iteration=0,
            campaign_dir=Path("/scratch/campaign"),
            config=CampaignConfig(),
        )


def test_missing_backend_message_lists_each_missing():
    a = check_backends()
    msg = missing_backend_message(a)
    if a.all_present:
        assert msg == ""
    else:
        # Every missing component must be named in the message.
        for component in a.missing:
            assert component in msg.lower() or component.upper() in msg


def test_missing_backend_message_names_rendered_gaussian_module():
    a = BackendAvailability(
        profile=True,
        sbatch=True,
        sacct=True,
        gaussian=False,
        aimall=True,
        ferebus=True,
        ariadne=True,
        pyferebus=True,
        bc=True,
        gaussian_binary="",
        sbatch_path="/usr/bin/sbatch",
        sacct_path="/usr/bin/sacct",
        bc_path="/usr/bin/bc",
        aimall_path="/opt/AIMAll/aimqb.ish",
        ferebus_path="/usr/local/bin/ferebus",
        active_profile="csf3",
        profile_error="",
        python_executable="/home/user/.venv/ichor-al-csf3/bin/python",
    )
    msg = missing_backend_message(a)
    assert "Active ICHOR profile: csf3" in msg
    assert "cluster-specific" in msg
    assert "gaussian/g16`" not in msg


def test_submitted_environment_smoke_renders_exact_runtime_contract(
    tmp_path, monkeypatch,
):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "scheduler": "slurm",
                    "jobscript_shebang": "#!/bin/bash --login",
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                },
                "software": {
                    "gaussian": {"modules": ["gaussian/test"]},
                },
            }
        },
        "csf3",
    )
    availability = SimpleNamespace(
        active_profile="csf3",
        batch_runtime_modules=("python/test", "mkl/test"),
        python_executable="/home/user/.venv/ichor-csf3/bin/python",
        gaussian_binary="/opt/gaussian/g16",
        aimall_path="/home/user/AIMAll/aimqb.ish",
        ferebus_path="/home/user/.local/bin/ferebus",
        bc_path="/usr/bin/bc",
    )

    body = render_submitted_environment_smoke_script(
        config=CampaignConfig(),
        availability=availability,
        output_path=tmp_path / "smoke.out",
    )
    assert "module purge" in body

    assert body.startswith("#!/bin/bash --login\n")
    assert "#SBATCH --partition=multicore" in body
    assert "#SBATCH --mem-per-cpu=8G" in body
    assert body.index("module load python/test") < body.index("module load gaussian/test")
    assert "pyferebus.executors.trainer" in body
    assert "test -x /opt/gaussian/g16" in body
    assert "test -x /home/user/AIMAll/aimqb.ish" in body
    assert SMOKE_SUCCESS_MARKER in body


def test_submitted_environment_smoke_records_success(tmp_path, monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf4": {
                "hpc": {
                    "scheduler": "slurm",
                    "jobscript_shebang": "#!/bin/bash --login",
                    "memory_per_core_gb_by_partition": {"multicore": 4},
                },
                "software": {"gaussian": {"modules": ["gaussian/test"]}},
            }
        },
        "csf4",
    )
    availability = SimpleNamespace(
        all_present=True,
        active_profile="csf4",
        sbatch_path="/usr/bin/sbatch",
        batch_runtime_modules=("python/test", "mkl/test"),
        python_executable="/home/user/.venv/ichor-csf4/bin/python",
        gaussian_binary="/opt/gaussian/g16",
        aimall_path="/home/user/AIMAll/aimqb.ish",
        ferebus_path="/home/user/.local/bin/ferebus",
        bc_path="/usr/bin/bc",
    )

    def fake_runner(argv, **kwargs):
        script_path = Path(argv[-1])
        output_line = next(
            line
            for line in script_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("#SBATCH --output=")
        )
        Path(output_line.split("=", 1)[1]).write_text(
            SMOKE_SUCCESS_MARKER + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    result = run_submitted_environment_smoke(
        campaign_dir=tmp_path,
        config=CampaignConfig(),
        availability=availability,
        runner=fake_runner,
        settle_seconds=0,
    )

    assert result["ok"] is True
    assert result["submitted"] is True
    assert result["job_id"] == "12345"
    assert Path(result["script_path"]).is_file()
    assert json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))["ok"] is True


def test_live_executor_refuses_when_sbatch_absent_on_windows():
    """On any host without sbatch, the constructor must refuse."""
    a = check_backends()
    if a.sbatch:
        pytest.skip("sbatch is on PATH; this test is for off-cluster hosts")
    cfg = CampaignConfig()
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(LiveBackendNotAvailableError):
            LiveBackendsPhaseExecutor(campaign_dir=Path(td), config=cfg)


# --- sbatch script body smoke (no backends needed) ------------------------


def test_build_sbatch_script_renders_gaussian_block(monkeypatch):
    _, gaussian_module = _install_fake_script_render_profile(monkeypatch)
    body = build_sbatch_script(
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
    )
    assert "#SBATCH --job-name=INITIAL_GAUSSIAN-0" in body
    assert "module load " + gaussian_module in body
    assert "g16 < input.gjf" in body
    assert "> input.gau" in body
    assert "> output.log" not in body
    assert "|| true" not in body
    assert "set -euo pipefail" in body
    assert "export LC_ALL=C" in body
    assert "export LC_NUMERIC=C" in body
    assert "export GAUSS_MDEF=" in body
    preparation = "ichor.hpc.active_learning.daemon.quantum_job_prepare"
    assert preparation in body
    assert body.index(preparation) < body.index("g16 < input.gjf")


def test_build_sbatch_script_renders_aimall_directives():
    cfg = CampaignConfig()
    cfg.aimall.boaq = "auto_gs2"
    cfg.aimall.iasmesh = "medium"
    cfg.resources.aimall_cpus_per_task = 8
    body = build_sbatch_script(
        phase_name="INITIAL_AIMALL",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
        array_size=1,
    )

    assert "#SBATCH --cpus-per-task=8" in body
    assert "AIMALL_TASK.json" in body
    assert '-nproc="${SLURM_CPUS_PER_TASK:-1}"' in body
    assert '-naat="$AIMALL_NAAT"' in body
    assert "-encomp=3" in body
    assert "-boaq=auto_gs2" in body
    assert "-iasmesh=medium" in body
    command_line = next(line for line in body.splitlines() if "aimqb.ish" in line)
    assert command_line.endswith(" input.wfn")
    preparation = "ichor.hpc.active_learning.daemon.quantum_job_prepare"
    assert preparation in body
    assert body.index(preparation) < body.index(command_line)


def test_gaussian_memory_uses_environment_contract(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf4": {
                "hpc": {
                    "scheduler": "slurm",
                    "memory_per_core_gb_by_partition": {"multicore": 4},
                }
            }
        },
        "csf4",
    )
    cfg = CampaignConfig()
    cfg.resources.gaussian_mem_per_cpu = "auto"

    body = build_sbatch_script(
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )

    memory_lines = [
        line for line in body.splitlines()
        if "GAUSS_" in line or "--mem" in line
    ]
    assert "#SBATCH --mem-per-cpu=4G" in body
    assert "export GAUSS_MDEF=13GB" in body, memory_lines
    assert "%mem" not in body.lower()


def test_build_sbatch_script_uses_strict_daemon_module_loads(monkeypatch):
    monkeypatch.setattr(
        live_executor_mod,
        "_configured_daemon_runtime_modules",
        lambda: list(live_executor_mod.DEFAULT_DAEMON_RUNTIME_MODULES),
    )
    body = build_sbatch_script(
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
    )
    assert "module load python/3.11.3-gcccore-12.3.0" in body
    assert "module load compilers/oneapi/2024.2.0" in body
    assert "module load compiler-rt tbb compiler" in body
    assert "module load mkl/2024.2" in body
    assert "anaconda" not in body.lower()
    assert "|| true" not in body


def test_build_sbatch_script_uses_configured_runtime_modules(monkeypatch):
    fake_global_variables = ModuleType("ichor.hpc.global_variables")
    fake_global_variables.ICHOR_CONFIG = {
        "csf4": {
            "hpc": {
                "memory_per_core_gb_by_partition": {"multicore": 4},
            },
            "software": {
                "python": {"modules": ["python/custom"]},
                "ariadne_runtime": {"modules": ["oneapi/custom", "mkl/custom"]},
            }
        }
    }
    fake_global_variables.MACHINE = "csf4"

    def fake_get_param_from_config(config, *keys, default=None):
        value = config
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    fake_global_variables.get_param_from_config = fake_get_param_from_config
    monkeypatch.setitem(
        sys.modules,
        "ichor.hpc.global_variables",
        fake_global_variables,
    )

    body = build_sbatch_script(
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
    )
    assert "module load python/custom" in body
    assert "module load oneapi/custom" in body
    assert "module load mkl/custom" in body
    assert "module load python/3.11.3-gcccore-12.3.0" not in body


def test_build_sbatch_script_rejects_unsafe_configured_module(monkeypatch):
    fake_global_variables = ModuleType("ichor.hpc.global_variables")
    fake_global_variables.ICHOR_CONFIG = {
        "csf4": {
            "hpc": {
                "memory_per_core_gb_by_partition": {"multicore": 4},
            },
            "software": {"python": {"modules": ["python,custom"]}},
        }
    }
    fake_global_variables.MACHINE = "csf4"

    def fake_get_param_from_config(config, *keys, default=None):
        value = config
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    fake_global_variables.get_param_from_config = fake_get_param_from_config
    monkeypatch.setitem(sys.modules, "ichor.hpc.global_variables", fake_global_variables)

    with pytest.raises(BackendSubmissionError, match="unsafe characters"):
        build_sbatch_script(
            phase_name="PHASE_A_DIVERSITY",
            iteration=0,
            campaign_dir=Path("/scratch/campaign"),
            config=CampaignConfig(),
        )


def test_runtime_modules_keep_ariadne_defaults_when_only_python_configured(monkeypatch):
    fake_global_variables = ModuleType("ichor.hpc.global_variables")
    fake_global_variables.ICHOR_CONFIG = {
        "csf4": {
            "software": {
                "python": {"modules": ["python/custom"]},
            }
        }
    }
    fake_global_variables.MACHINE = "csf4"

    def fake_get_param_from_config(config, *keys, default=None):
        value = config
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    fake_global_variables.get_param_from_config = fake_get_param_from_config
    monkeypatch.setitem(
        sys.modules,
        "ichor.hpc.global_variables",
        fake_global_variables,
    )

    assert live_executor_mod._configured_daemon_runtime_modules() == [
        "python/custom",
        *live_executor_mod.DEFAULT_DAEMON_ARIADNE_RUNTIME_MODULES,
    ]


def test_csf3_profile_accepts_empty_python_modules_and_uses_runtime_modules(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "jobscript_shebang": "#!/bin/bash --login",
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                },
                "software": {
                    "python": {
                        "modules": [],
                        "python_path": "$HOME/.venv/ichor-al-csf3/bin/python",
                    },
                    "ariadne_runtime": {
                        "modules": [
                            "compilers/intel/oneapi/2025.0.1",
                            "umf compiler-rt tbb compiler",
                            "mkl/2025.0",
                        ],
                    },
                },
            }
        },
        "csf3",
    )

    body = build_sbatch_script(
        phase_name="ARIADNE_ARRAY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
        array_size=2,
    )
    assert body.startswith("#!/bin/bash --login")
    assert "module load python/" not in body
    assert "module load compilers/intel/oneapi/2025.0.1" in body
    assert "module load umf compiler-rt tbb compiler" in body
    assert "module load mkl/2025.0" in body
    assert "$HOME/.venv/ichor-al-csf3/bin/python" in body


def test_csf3_gaussian_block_uses_configured_module_path_and_scratch(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "jobscript_shebang": "#!/bin/bash --login",
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                },
                "software": {
                    "gaussian": {
                        "modules": ["apps/binapps/gaussian/g16c01_em64t_detectcpu"],
                        "executable_path": "$g16root/g16/g16",
                    }
                },
            }
        },
        "csf3",
    )
    body = build_sbatch_script(
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
        array_size=1,
    )
    assert body.startswith("#!/bin/bash --login")
    assert "module load apps/binapps/gaussian/g16c01_em64t_detectcpu" in body
    assert "$g16root/g16/g16 < input.gjf > input.gau" in body
    expected_campaign = shlex.quote(str(Path("/scratch/campaign").resolve()))
    assert "export ICHOR_CAMPAIGN_DIR=" + expected_campaign in body
    assert "export ICHOR_GAUSSIAN_PHASE=INITIAL_GAUSSIAN" in body
    assert 'export GAUSS_SCRDIR="$ICHOR_JOB_SCRATCH/gaussian"' in body
    assert 'rm -rf -- "$GAUSS_SCRDIR"' not in body
    assert "ichor_gaussian_${SLURM_JOB_ID}" not in body
    assert 'export GAUSS_PDEF="${SLURM_CPUS_PER_TASK:-1}"' in body
    assert "export GAUSS_MDEF=27GB" in body


def test_gaussian_block_ignores_configured_scratch_root(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "jobscript_shebang": "#!/bin/bash --login",
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                },
                "software": {
                    "gaussian": {
                        "modules": ["apps/binapps/gaussian/g09d01_em64t"],
                        "executable_path": "$g09root/g09/g09",
                        "scratch_root": "/scratch/$USER",
                    }
                },
            }
        },
        "csf3",
    )

    body = build_sbatch_script(
        phase_name="GAUSSIAN",
        iteration=2,
        campaign_dir=Path("/net/scratch/q/campaign"),
        config=CampaignConfig(),
        array_size=1,
        campaign_uid="campaign:with unsafe/chars",
    )

    assert "export ICHOR_CAMPAIGN_UID=campaign_with_unsafe_chars" in body
    assert 'export GAUSS_SCRDIR="$ICHOR_JOB_SCRATCH/gaussian"' in body
    assert 'rm -rf -- "$GAUSS_SCRDIR"' not in body
    assert "ICHOR_GAUSSIAN_SCRATCH_ROOT" not in body
    assert "ichor-gaussian" not in body
    assert "/scratch/$USER" not in body


def test_default_trqn_scale_mode_is_not_warned():
    warnings = live_executor_mod._ariadne_optional_diagnostic_warnings({
        "optimiser_diagnostics": {
            "trqn_scale_mode": "adaptive_initial_gradient_rms",
        }
    })

    assert "trqn_scale_mode_unknown" not in warnings


def test_ariadne_seed_provenance_is_created_with_canonical_identity(tmp_path):
    from ichor.hpc.active_learning.versioning.provenance import (
        PROVENANCE_FILENAME,
        validate_provenance,
    )

    campaign = tmp_path / "campaign"
    ex = LiveBackendsPhaseExecutor.__new__(LiveBackendsPhaseExecutor)
    ex.campaign_dir = campaign
    ex.config = CampaignConfig()
    ex.artefact_log = []
    state = SimpleNamespace(iteration=1, campaign_uid="test-campaign")
    seed_record = {
        "seed_id": 1,
        "seed_uid": "b" * 64,
        "frame_id": 7,
        "selection_origin": "d_optimal",
        "variance_at_selection": 1.0,
        "subspace_neighbour_frame_ids": [1, 2],
        "subspace_dimension": 2,
        "subspace_eigenvalues": [1.0, 0.5],
    }
    picked = {"trajectory_sha256": "a" * 64}
    seed_dir = ex._seed_dir_for_record(1, seed_record)
    seed_dir.mkdir(parents=True)
    prov_path = seed_dir / PROVENANCE_FILENAME

    created_path, created = ex._ensure_ariadne_seed_provenance(
        state,
        picked,
        seed_record,
    )

    assert created_path == prov_path
    assert created is True
    validate_provenance(
        seed_dir,
        campaign_uid="test-campaign",
        iteration=1,
        trajectory_sha256="a" * 64,
        seed_frame_id=7,
        seed_id=1,
        seed_uid="b" * 64,
        array_task_id_zero_based=0,
    )


def test_ariadne_seed_provenance_identity_mismatch_still_fails(tmp_path):
    from ichor.hpc.active_learning.versioning.provenance import write_seed_provenance

    campaign = tmp_path / "campaign"
    ex = LiveBackendsPhaseExecutor.__new__(LiveBackendsPhaseExecutor)
    ex.campaign_dir = campaign
    ex.config = CampaignConfig()
    ex.artefact_log = []
    state = SimpleNamespace(iteration=1, campaign_uid="test-campaign")
    seed_record = {
        "seed_id": 1,
        "seed_uid": "b" * 64,
        "frame_id": 7,
        "selection_origin": "bulk",
        "variance_at_selection": 1.0,
        "subspace_neighbour_frame_ids": [],
        "subspace_dimension": 0,
        "subspace_eigenvalues": [],
    }
    seed_dir = ex._seed_dir_for_record(1, seed_record)
    seed_dir.mkdir(parents=True)
    write_seed_provenance(
        seed_dir,
        campaign_uid="wrong",
        iteration=1,
        trajectory_sha256="a" * 64,
        seed_frame_id=7,
        seed_id=1,
        seed_uid="b" * 64,
        array_task_id_zero_based=0,
        seed_selection_origin="bulk",
        seed_variance_at_selection=1.0,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
    )

    with pytest.raises(BackendSubmissionError, match="campaign_uid mismatch"):
        ex._ensure_ariadne_seed_provenance(
            state,
            {"trajectory_sha256": "a" * 64},
            seed_record,
        )


def test_profile_memory_auto_resolves_csf4_partition_cap(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf4": {
                "hpc": {
                    "memory_per_core_gb_by_partition": {"multicore": 4},
                }
            }
        },
        "csf4",
    )
    cfg = CampaignConfig()
    cfg.resources.gaussian_cpus_per_task = 2
    body = build_sbatch_script(
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
        array_size=1,
    )
    assert "#SBATCH --mem-per-cpu=4G" in body
    assert "#SBATCH --cpus-per-task=2" in body
    assert "export GAUSS_MDEF=6GB" in body


@pytest.mark.parametrize(("machine", "max_cores"), [("csf3", 168), ("csf4", 40)])
def test_multicore_one_core_request_fails_before_sbatch(monkeypatch, machine, max_cores):
    _install_fake_global_variables(
        monkeypatch,
        {
            machine: {
                "hpc": {
                    "scheduler": "slurm",
                    "parallel_environments": {"multicore": [2, max_cores]},
                    "memory_per_core_gb_by_partition": {
                        "multicore": 8 if machine == "csf3" else 4,
                    },
                }
            }
        },
        machine,
    )
    cfg = CampaignConfig()
    cfg.resources.partition = "multicore"
    cfg.resources.diversity_cpus_per_task = 1

    with pytest.raises(BackendSubmissionError, match="configured range is \\[2,"):
        build_sbatch_script(
            phase_name="PHASE_A_DIVERSITY",
            iteration=0,
            campaign_dir=Path("/scratch/campaign"),
            config=cfg,
        )


def test_multicore_two_core_request_passes_profile_range_check(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "scheduler": "slurm",
                    "parallel_environments": {"multicore": [2, 168]},
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                }
            }
        },
        "csf3",
    )
    cfg = CampaignConfig()
    cfg.resources.partition = "multicore"
    cfg.resources.diversity_cpus_per_task = 2

    body = build_sbatch_script(
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )

    assert "#SBATCH --partition=multicore" in body
    assert "#SBATCH --cpus-per-task=2" in body


def test_serial_one_core_request_passes_profile_range_check(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "scheduler": "slurm",
                    "parallel_environments": {"serial": [1, 1]},
                    "memory_per_core_gb_by_partition": {"serial": 5},
                }
            }
        },
        "csf3",
    )
    cfg = CampaignConfig()
    cfg.resources.partition = "serial"
    cfg.resources.diversity_cpus_per_task = 1

    body = build_sbatch_script(
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )

    assert "#SBATCH --partition=serial" in body
    assert "#SBATCH --cpus-per-task=1" in body


def test_backend_partition_override_uses_grouped_resource_partition(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "scheduler": "slurm",
                    "partitions": {
                        "multicore": {
                            "min_cpus": 2,
                            "max_cpus": 168,
                            "memory_per_core_gb": 8,
                            "max_walltime_hours": 168,
                            "daemon_supported": True,
                        },
                        "interactive": {
                            "min_cpus": 1,
                            "max_cpus": 168,
                            "memory_per_core_gb": 8,
                            "max_walltime_hours": 24,
                            "daemon_supported": True,
                        },
                    },
                }
            }
        },
        "csf3",
    )
    cfg = CampaignConfig()
    cfg.resources.defaults.partition = "multicore"
    cfg.resources.diversity.partition = "interactive"
    cfg.resources.diversity.cpus_per_task = 1
    body = build_sbatch_script(
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )
    assert "#SBATCH --partition=interactive" in body


def test_unknown_partition_fails_against_active_profile(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "scheduler": "slurm",
                    "partitions": {
                        "multicore": {
                            "min_cpus": 2,
                            "max_cpus": 168,
                            "memory_per_core_gb": 8,
                            "max_walltime_hours": 168,
                            "daemon_supported": True,
                        },
                    },
                }
            }
        },
        "csf3",
    )
    cfg = CampaignConfig()
    cfg.resources.diversity.partition = "multinode"
    with pytest.raises(BackendSubmissionError, match="not present"):
        build_sbatch_script(
            phase_name="PHASE_A_DIVERSITY",
            iteration=0,
            campaign_dir=Path("/scratch/campaign"),
            config=cfg,
        )


def test_unsupported_profile_partition_fails_before_sbatch(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf4": {
                "hpc": {
                    "scheduler": "slurm",
                    "partitions": {
                        "multinode": {
                            "min_cpus": 2,
                            "max_cpus": 10000,
                            "memory_per_core_gb": 4,
                            "max_walltime_hours": 168,
                            "daemon_supported": False,
                        },
                    },
                }
            }
        },
        "csf4",
    )
    cfg = CampaignConfig()
    cfg.resources.defaults.partition = "multinode"
    with pytest.raises(BackendSubmissionError, match="not supported"):
        build_sbatch_script(
            phase_name="PHASE_A_DIVERSITY",
            iteration=0,
            campaign_dir=Path("/scratch/campaign"),
            config=cfg,
        )


def test_gaussian_cpus_must_fit_live_profile_range(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "scheduler": "slurm",
                    "parallel_environments": {"multicore": [2, 4]},
                }
            }
        },
        "csf3",
    )
    cfg = CampaignConfig()
    cfg.resources.partition = "multicore"
    cfg.resources.gaussian_cpus_per_task = 8

    with pytest.raises(BackendSubmissionError, match="gaussian.cpus_per_task"):
        build_sbatch_script(
            phase_name="INITIAL_GAUSSIAN",
            iteration=0,
            campaign_dir=Path("/scratch/campaign"),
            config=cfg,
            array_size=1,
        )


def test_explicit_memory_above_profile_cap_fails_before_sbatch(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf4": {
                "hpc": {
                    "memory_per_core_gb_by_partition": {"multicore": 4},
                }
            }
        },
        "csf4",
    )
    cfg = CampaignConfig()
    cfg.resources.diversity_mem_per_cpu = "8G"
    with pytest.raises(BackendSubmissionError, match="exceeds configured profile memory cap"):
        build_sbatch_script(
            phase_name="PHASE_A_DIVERSITY",
            iteration=0,
            campaign_dir=Path("/scratch/campaign"),
            config=cfg,
        )


def test_array_concurrency_limit_renders_slurm_percent_throttle():
    cfg = CampaignConfig()
    cfg.resources.array_concurrency_limit = 7
    body = build_sbatch_script(
        phase_name="ARIADNE_ARRAY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
        array_size=20,
    )
    assert "#SBATCH --array=0-19%7" in body


def test_configured_array_task_limit_rejects_too_large_array(monkeypatch):
    _install_fake_global_variables(
        monkeypatch,
        {"csf3": {"hpc": {"max_array_task_id": 25000}}},
        "csf3",
    )
    with pytest.raises(BackendSubmissionError, match="max_array_task_id"):
        build_sbatch_script(
            phase_name="ARIADNE_ARRAY",
            iteration=0,
            campaign_dir=Path("/scratch/campaign"),
            config=CampaignConfig(),
            array_size=25002,
        )


def test_build_sbatch_script_uses_yaml_scheduler_resources_by_default():
    cfg = CampaignConfig()
    cfg.resources.partition = "csf4-debug"
    cfg.resources.diversity_walltime_hours = 7
    body = build_sbatch_script(
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )
    assert "#SBATCH --partition=csf4-debug" in body
    assert "#SBATCH --time=07:00:00" in body


def test_build_sbatch_script_explicit_scheduler_overrides_win():
    cfg = CampaignConfig()
    cfg.resources.partition = "yaml-partition"
    cfg.resources.default_walltime_hours = 7
    body = build_sbatch_script(
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
        partition="explicit-partition",
        walltime_hours=11,
    )
    assert "#SBATCH --partition=explicit-partition" in body
    assert "#SBATCH --time=11:00:00" in body
    assert "yaml-partition" not in body
    assert "#SBATCH --time=07:00:00" not in body


def test_build_sbatch_script_uses_phase_walltime_override():
    cfg = CampaignConfig()
    cfg.resources.default_walltime_hours = 12
    cfg.resources.gaussian_walltime_hours = 3
    cfg.resources.aimall_walltime_hours = 4
    cfg.resources.ariadne_walltime_hours = 5
    cfg.resources.diversity_walltime_hours = 6
    assert "#SBATCH --time=03:00:00" in build_sbatch_script(
        phase_name="GAUSSIAN",
        iteration=1,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
        array_size=1,
    )
    assert "#SBATCH --time=04:00:00" in build_sbatch_script(
        phase_name="AIMALL",
        iteration=1,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
        array_size=1,
    )
    assert "#SBATCH --time=05:00:00" in build_sbatch_script(
        phase_name="ARIADNE_ARRAY",
        iteration=1,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
        array_size=1,
    )
    assert "#SBATCH --time=06:00:00" in build_sbatch_script(
        phase_name="PHASE_B_DIVERSITY",
        iteration=1,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )


def test_fractional_walltime_renders_minutes():
    cfg = CampaignConfig()
    cfg.resources.diversity.walltime_hours = 0.25
    body = build_sbatch_script(
        phase_name="PHASE_B_DIVERSITY",
        iteration=1,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )
    assert "#SBATCH --time=00:15:00" in body


def test_ariadne_auto_cpus_match_active_fd_worker_count(monkeypatch):
    from ichor.hpc.active_learning.daemon.resource_solver import resolve_phase_resources

    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "parallel_environments": {"multicore": [1, 64]},
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                }
            }
        },
        "csf3",
    )
    cfg = CampaignConfig()
    cfg.resources.partition = "multicore"
    cfg.resources.ariadne_cpus_per_task = "auto"
    cfg.acquisition.gradient.mode = "active_fd"
    cfg.acquisition.subspace.max_subspace_dim = 6

    resolved = resolve_phase_resources(
        phase_name="ARIADNE_ARRAY",
        config=cfg,
        partition="multicore",
        campaign_dir=None,
        iteration=1,
        require_evidence=False,
    )

    assert resolved.cpus_per_task == 6
    assert resolved.cpu_reason == "ariadne_active_fd_direction_workers"


def test_ariadne_auto_cpus_match_cartesian_fd_component_count(monkeypatch, tmp_path):
    from ichor.hpc.active_learning.daemon.resource_solver import resolve_phase_resources

    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "parallel_environments": {"multicore": [1, 64]},
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                }
            }
        },
        "csf3",
    )
    staging = tmp_path / ".DATA" / "STAGING" / "iter_1"
    pointdir = staging / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True)
    (pointdir / "input.gjf").write_text(
        "\n".join([
            "# test",
            "",
            "title",
            "",
            "0 1",
            "O 0.0 0.0 0.0",
            "H 0.0 0.0 1.0",
            "H 0.0 1.0 0.0",
            "H 1.0 0.0 0.0",
            "",
        ]),
        encoding="utf-8",
    )
    (staging / "POINTS.txt").write_text(str(pointdir) + "\n", encoding="utf-8")
    cfg = CampaignConfig()
    cfg.resources.partition = "multicore"
    cfg.resources.ariadne_cpus_per_task = "auto"
    cfg.acquisition.gradient.mode = "cartesian_fd"

    resolved = resolve_phase_resources(
        phase_name="ARIADNE_ARRAY",
        config=cfg,
        partition="multicore",
        campaign_dir=None,
        iteration=1,
        n_atoms_override=4,
        require_evidence=False,
    )

    assert resolved.cpus_per_task == 12
    assert resolved.cpu_reason == "ariadne_cartesian_fd_component_workers"


def test_only_resource_resolving_array_staging_receives_partition_override(
    monkeypatch, tmp_path
):
    from ichor.hpc.active_learning.daemon import input_staging as stg

    cfg = CampaignConfig()
    cfg.resources.partition = "yaml-partition"
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    sample = campaign / "sample.xyz"
    sample.write_text("0\n\n", encoding="utf-8")
    calls = {}

    def fake_stage_gaussian(
        campaign_dir,
        config,
        phase_name,
        iteration,
        sample_xyz,
        *,
        partition_override=None,
    ):
        calls["gaussian"] = {
            "campaign_dir": Path(campaign_dir),
            "phase_name": phase_name,
            "iteration": iteration,
            "sample_xyz": Path(sample_xyz),
            "partition_override": partition_override,
        }
        return campaign / "gaussian-stage", 3

    def fake_stage_aimall(
        campaign_dir,
        config,
        phase_name,
        iteration,
        *,
        partition_override=None,
        staging_override=None,
        **_kwargs,
    ):
        calls["aimall"] = {
            "campaign_dir": Path(campaign_dir),
            "phase_name": phase_name,
            "iteration": iteration,
            "partition_override": partition_override,
            "staging_override": staging_override,
        }
        return campaign / "aimall-stage", 2

    monkeypatch.setattr(stg, "stage_gaussian_inputs", fake_stage_gaussian)
    monkeypatch.setattr(stg, "stage_aimall_inputs", fake_stage_aimall)
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=cfg,
        sbatch_runner=object(),
        backend_check=False,
        partition="override-partition",
    )
    monkeypatch.setattr(
        ex,
        "_locate_sample_xyz",
        lambda phase_name, iteration, **_kwargs: sample,
    )

    assert ex._array_size_after_staging("GAUSSIAN", SimpleNamespace(iteration=4)) == 3
    assert ex._array_size_after_staging("AIMALL", SimpleNamespace(iteration=4)) == 2

    assert calls["gaussian"]["partition_override"] is None
    assert calls["gaussian"]["sample_xyz"] == sample
    assert calls["aimall"]["partition_override"] == "override-partition"


def _write_minimal_ariadne_task_map(iter_dir):
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.handoff_manifests import (
        build_seed_selection_manifest,
        seeds_picked_path,
    )
    from ichor.hpc.active_learning.seed_identity import write_ariadne_task_map

    payload = build_seed_selection_manifest(
        campaign_uid="uid",
        campaign_random_seed=0,
        iteration=1,
        models_version=0,
        model_manifest_sha256="c" * 64,
        trajectory_sha256="d" * 64,
        selection_strategy="hybrid_variance",
        seed_records=[{
            "seed_id": 1,
            "frame_id": 0,
            "pool_row_index_zero_based": 0,
            "selection_origin": "bulk",
            "variance_at_selection": 0.0,
        }],
    )
    selection_path = seeds_picked_path(iter_dir)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, payload)
    write_ariadne_task_map(iter_dir, payload)


def test_ariadne_array_staging_precomputes_geometry_novelty_scale(tmp_path, monkeypatch):
    from ichor.hpc.active_learning import geometry_novelty

    cfg = CampaignConfig()
    campaign = tmp_path / "campaign"
    from ichor.hpc.active_learning.layout import active_iteration_dir

    iter_dir = active_iteration_dir(campaign, 1)
    _write_minimal_ariadne_task_map(iter_dir)
    calls = []

    def fake_ensure(campaign_dir, config, *, iteration):
        calls.append((Path(campaign_dir), config, int(iteration)))
        return {
            "scale_angstrom": 0.04,
            "scale_resolution_mode": "fallback_protocol",
            "n_values": 0,
        }

    monkeypatch.setattr(
        geometry_novelty,
        "ensure_geometry_novelty_scale",
        fake_ensure,
    )
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=cfg,
        sbatch_runner=object(),
        backend_check=False,
    )

    n = ex._array_size_after_staging(
        "ARIADNE_ARRAY",
        SimpleNamespace(iteration=1, campaign_uid="uid"),
    )

    assert n == 1
    assert calls == [(campaign, cfg, 1)]


def test_ariadne_array_staging_fails_before_submit_when_scale_precompute_fails(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning import geometry_novelty

    cfg = CampaignConfig()
    campaign = tmp_path / "campaign"
    from ichor.hpc.active_learning.layout import active_iteration_dir

    iter_dir = active_iteration_dir(campaign, 1)
    _write_minimal_ariadne_task_map(iter_dir)

    def fake_ensure(campaign_dir, config, *, iteration):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        geometry_novelty,
        "ensure_geometry_novelty_scale",
        fake_ensure,
    )
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=cfg,
        sbatch_runner=object(),
        backend_check=False,
    )

    with pytest.raises(
        BackendSubmissionError,
        match="sampling protocol resolution failed",
    ):
        ex._array_size_after_staging(
            "ARIADNE_ARRAY",
            SimpleNamespace(iteration=1, campaign_uid="uid"),
        )


def test_build_sbatch_script_renders_ferebus_block():
    body = build_sbatch_script(
        phase_name="INITIAL_FEREBUS",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
    )
    # Live FEREBUS uses pyferebus_wrap.submit_ferebus(), not the generic renderer.
    assert "pyferebus wrapper" in body
    assert 'ferebus_${ATOM}.toml' not in body
    assert "runferebus.py" not in body.lower()


def test_live_ferebus_submit_uses_pyferebus_wrapper(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.daemon import input_staging as stg
    from ichor.hpc.active_learning.submit import pyferebus_wrap
    from ichor.hpc.active_learning.submit.pyferebus_wrap import FerebusSubmission
    from ichor.hpc.active_learning.ferebus_prior import (
        resolve_ferebus_prior_contract,
    )

    cfg = CampaignConfig()
    prior = resolve_ferebus_prior_contract(cfg)
    cfg.resources.default_walltime_hours = 9
    cfg.resources.ferebus_walltime_hours = 2
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    for version in range(5):
        (
            campaign
            / "QM_REFERENCE_DATA"
            / ("iteration-" + str(version).zfill(6))
        ).mkdir(parents=True)
    staging = campaign / "TRAINED_MODELS" / "iteration-staging"
    staging.mkdir(parents=True)
    (staging / stg.FEREBUS_JOB_DETAILS).write_text(
        "system_name WATER\n", encoding="utf-8",
    )
    dataset_identities = {}
    for atom in ("O1", "H2", "H3"):
        datasets_dir = staging / "iqa" / atom / "datasets"
        datasets_dir.mkdir(parents=True)
        records = {}
        for split, suffix, rows in (
            ("train", "TRAINING_SET", 3),
            ("int_val", "INT_VALIDATION_SET", 1),
            ("ext_val", "EXT_VALIDATION_SET", 1),
        ):
            dataset = datasets_dir / ("WATER_" + atom + "_" + suffix + ".csv")
            dataset.write_text(
                "f1,f2,f3,iqa\n" + "0.1,0.2,0.3,0.0\n" * rows,
                encoding="utf-8",
            )
            records[split] = {
                "path": dataset.relative_to(staging).as_posix(),
                "size": dataset.stat().st_size,
                "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "rows": rows,
            }
        dataset_identities[atom] = records
    (staging / stg.FEREBUS_TASK_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
                "campaign_uid": "backend-smoke",
                "system": "WATER",
                "reference_data_version": 4,
                "reference_data_head_manifest_sha256": "a" * 64,
                "reference_data_view_sha256": "b" * 64,
                "n_reference_points": 5,
                "pointdir_row_order": [
                    "POINT_" + str(index).zfill(6) + ".pointdir"
                    for index in range(5)
                ],
                "properties": ["iqa"],
                "atoms": ["O1", "H2", "H3"],
                "n_atoms": 3,
                "n_tasks": 3,
                "prior_mean_contract": prior.to_dict(),
                "tasks": [
                    {
                        "task_index": index,
                        "property": "iqa",
                        "atom": atom,
                        "prior_mean": prior.task_payload("iqa", atom),
                        "alf_1_indexed": list(alf),
                        "alf_cli": "_".join(
                            str(value) for value in alf
                        ),
                        "property_dir": "iqa",
                        "output_dir": "iqa/" + atom,
                        "input_dir": "iqa/" + atom + "/datasets",
                        "config_path": "iqa/" + atom + "/ferebus.config",
                        "training_csv": "iqa/" + atom + "/datasets/WATER_" + atom + "_TRAINING_SET.csv",
                        "int_validation_csv": "iqa/" + atom + "/datasets/WATER_" + atom + "_INT_VALIDATION_SET.csv",
                        "ext_validation_csv": "iqa/" + atom + "/datasets/WATER_" + atom + "_EXT_VALIDATION_SET.csv",
                        "expected_model_path": "iqa/" + atom + "/WATER_iqa_" + atom + ".model",
                        "command_args": [
                            "-c", "iqa/" + atom + "/ferebus.config",
                            "-I", "iqa/" + atom + "/datasets",
                            "-O", "iqa/" + atom,
                            "-P", "iqa",
                            "-A", atom,
                            "-ALF", "_".join(
                                str(value) for value in alf
                            ),
                        ],
                        "row_counts": {
                            "train": 3,
                            "int_val": 1,
                            "ext_val": 1,
                        },
                        "row_ids": {
                            "train": [0, 1, 2],
                            "int_val": [3],
                            "ext_val": [4],
                        },
                        "datasets": dataset_identities[atom],
                    }
                    for index, (atom, alf) in enumerate(
                        (
                            ("O1", (1, 2, 3)),
                            ("H2", (2, 1, 3)),
                            ("H3", (3, 1, 2)),
                        ),
                        start=1,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    calls = {}

    def fake_stage(campaign_dir, config, reference_data_version, *, is_initial=False):
        calls["stage"] = {
            "campaign_dir": Path(campaign_dir),
            "reference_data_version": reference_data_version,
            "is_initial": is_initial,
        }
        return staging, 3

    def fake_submit(jd_file, working_directory, **kwargs):
        working = Path(working_directory)
        script = working / "runFerebus.sh"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        manifest_path = working / stg.FEREBUS_TASK_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generated = []
        for task in manifest["tasks"]:
            config_path = working / task["config_path"]
            config_path.write_text(
                "mean_type = 21\n"
                'level_of_theory = "b3lyp/aug-cc-pvtz"\n'
                "iqaDeviationFactor = 1.0\n"
                "scaling = 1\n"
                "scale_feats = 1\n"
                "scale_prop = 0\n",
                encoding="utf-8",
            )
            record = {
                "path": task["config_path"],
                "size": config_path.stat().st_size,
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "parsed_contract": {
                    "mean_type": 21,
                    "level_of_theory": "b3lyp/aug-cc-pvtz",
                    "iqa_deviation_factor": 1.0,
                    "scaling": True,
                    "scale_feats": True,
                    "scale_prop": False,
                },
                "prior_mean_contract_sha256": prior.contract_sha256,
            }
            task["generated_config"] = record
            generated.append(record)
        manifest_path.write_text(
            json.dumps(manifest), encoding="utf-8", newline="\n"
        )
        overrides = kwargs["prepared_callback"](working, script, generated)
        submitted_script = Path(overrides["submission_script_path"])
        submitted_script.write_text("#!/bin/sh\n", encoding="utf-8")
        from ichor.hpc.active_learning.daemon.script_bundles import (
            AttemptBundle,
            write_script_binding,
        )

        script_binding = write_script_binding(
            AttemptBundle(
                root=submitted_script.parent,
                script=submitted_script,
                outputs=submitted_script.parent / "OUTPUTS",
                errors=submitted_script.parent / "ERRORS",
            )
        )
        kwargs["pre_submit_hook"](submitted_script, script_binding)
        calls["submit"] = {
            "jd_file": Path(jd_file),
            "working_directory": working,
            "kwargs": dict(kwargs),
            "overrides": dict(overrides),
        }
        return FerebusSubmission(
            job_id="4242",
            cluster=None,
            submission_script=submitted_script,
            working_dir=working,
            transfer_learning=False,
            generated_configs=tuple(generated),
            script_binding=script_binding,
        )

    monkeypatch.setattr(stg, "stage_ferebus_inputs", fake_stage)
    monkeypatch.setattr(pyferebus_wrap, "submit_ferebus", fake_submit)
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                },
                "software": {
                    "ferebus": {
                        "executable_path": "$HOME/.local/bin/ferebus",
                        "pyferebus_platform": "CSF3",
                    }
                }
            }
        },
        "csf3",
    )

    runner = object()
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=cfg,
        sbatch_runner=runner,
        backend_check=False,
    )
    from ichor.hpc.active_learning.daemon.submission_intent import (
        write_pre_submit_intent,
    )
    from ichor.hpc.active_learning.daemon.scheduler_contracts import (
        infer_expected_tasks_from_artifacts,
    )

    assert infer_expected_tasks_from_artifacts(
        campaign,
        phase="FEREBUS",
        iteration=0,
    ) == 3

    write_pre_submit_intent(
        campaign,
        campaign_uid="campaign-uid",
        phase_name="FEREBUS",
        iteration=0,
        expected_tasks=3,
    )
    result = ex.submit_or_run(
        SimpleNamespace(
            iteration=0,
            reference_data_version=4,
            campaign_uid="campaign-uid",
        ),
        "FEREBUS",
    )

    assert result.submitted_job_id == "4242"
    assert result.expected_tasks == 3
    assert calls["stage"]["reference_data_version"] == 4
    assert calls["stage"]["is_initial"] is False
    assert calls["submit"]["jd_file"] == staging / stg.FEREBUS_JOB_DETAILS
    assert calls["submit"]["working_directory"] == staging
    assert calls["submit"]["kwargs"]["overwrite_workdir"] is False
    assert calls["submit"]["kwargs"]["move_dataset_files"] is True
    assert calls["submit"]["kwargs"]["submit_runner"] is runner
    assert calls["submit"]["kwargs"]["walltime_hours"] == cfg.resources.ferebus_walltime_hours
    assert calls["submit"]["kwargs"]["platform"] == "CSF3"
    assert calls["submit"]["kwargs"]["partition"] == "multicore"
    assert calls["submit"]["kwargs"]["ncores"] == cfg.ferebus.nagents
    assert calls["submit"]["kwargs"]["expected_tasks"] == 3
    assert calls["submit"]["overrides"]["cpus_per_task"] >= cfg.ferebus.nagents
    assert calls["submit"]["overrides"]["ntasks"] == 1
    assert calls["submit"]["overrides"]["mem_per_cpu"].endswith("G")


def test_build_sbatch_script_renders_aimall_block(monkeypatch):
    _install_fake_script_render_profile(monkeypatch)
    monkeypatch.setattr(
        live_executor_mod,
        "_configured_backend_path",
        lambda backend_name, fallback: "/opt/AIM All/aimqb.ish",
    )
    cfg = CampaignConfig()
    cfg.resources.aimall_cpus_per_task = 8
    body = build_sbatch_script(
        phase_name="AIMALL",
        iteration=3,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )
    command_line = next(line for line in body.splitlines() if "aimqb.ish" in line)
    assert command_line.startswith(shlex.quote("/opt/AIM All/aimqb.ish") + " -nogui")
    assert '-nproc="${SLURM_CPUS_PER_TASK:-1}"' in command_line
    assert '-naat="$AIMALL_NAAT"' in command_line
    assert "-encomp=3" in command_line
    assert command_line.endswith(" input.wfn")
    assert "#SBATCH --cpus-per-task=8" in body
    assert "AIMALL-3" in body


def test_build_sbatch_script_renders_ariadne_block(monkeypatch):
    python_path, _ = _install_fake_script_render_profile(monkeypatch)
    body = build_sbatch_script(
        phase_name="ARIADNE_ARRAY",
        iteration=2,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
    )
    assert "ariadne_runner" in body
    assert shlex.quote(python_path) + " -m ichor.hpc.active_learning.acquisition.ariadne_runner" in body
    assert "\npython -m ichor.hpc.active_learning.acquisition.ariadne_runner" not in body
    assert "--array-task-id $ICHOR_LOGICAL_ARRAY_TASK_ID" in body
    assert "--iteration 2" in body


def test_build_sbatch_script_renders_diversity_block_with_configured_descriptor(monkeypatch):
    python_path, _ = _install_fake_script_render_profile(monkeypatch)
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "hybrid_alf_rmsd"
    body = build_sbatch_script(
        phase_name="PHASE_B_DIVERSITY",
        iteration=4,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )
    assert "sampling.diversity" in body
    assert shlex.quote(python_path) + " -m ichor.hpc.active_learning.sampling.diversity" in body
    assert "\npython -m ichor.hpc.active_learning.sampling.diversity" not in body
    assert "--descriptor hybrid_alf_rmsd" in body
    assert "--iteration 4" in body


def test_build_sbatch_script_renders_phase_a_diversity_as_bootstrap_iteration_zero():
    body = build_sbatch_script(
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
    )
    assert "sampling.diversity" in body
    assert "--descriptor rmsd_massweight" in body
    assert "--iteration 0" in body
    assert "OPENBLAS_NUM_THREADS=1" in body


@pytest.mark.parametrize("bucket", ["initial", "iter_1"])
@pytest.mark.parametrize("required_filename", ["input.gjf", "input.wfn"])
def test_replacement_pointdir_guard_is_rooted_at_exact_round(
    tmp_path,
    bucket,
    required_filename,
):
    round_dir = (
        tmp_path
        / ".DATA"
        / "STAGING"
        / bucket
        / "replacement_round_0002"
    )
    points = round_dir / "POINTS.txt"
    lines = live_executor_mod._pointdir_selection_lines(
        str(points),
        required_filename=required_filename,
    )
    rendered = "\n".join(lines)

    assert "export ICHOR_STAGING_ROOT=" + shlex.quote(str(round_dir.resolve())) in rendered
    assert "if [ ! -f " + shlex.quote(str(points.resolve())) + " ]; then" in rendered
    assert "POINT_DIR escapes exact campaign staging round" in rendered


@pytest.mark.parametrize(
    "phase_name,required_filename",
    [
        ("REPLACEMENT_GAUSSIAN", "input.gjf"),
        ("REPLACEMENT_AIMALL", "input.wfn"),
        ("INITIAL_REPLACEMENT_GAUSSIAN", "input.gjf"),
        ("INITIAL_REPLACEMENT_AIMALL", "input.wfn"),
    ],
)
def test_replacement_sbatch_uses_exact_nested_points_file(
    tmp_path,
    monkeypatch,
    phase_name,
    required_filename,
):
    campaign = tmp_path / "campaign"
    iteration = 0 if phase_name.startswith("INITIAL_") else 1
    bucket = "initial" if iteration == 0 else "iter_1"
    round_dir = (
        campaign
        / ".DATA"
        / "STAGING"
        / bucket
        / "replacement_round_0002"
    )
    pointdir = round_dir / "POINT_0012.pointdir"
    pointdir.mkdir(parents=True)
    (pointdir / "input.gjf").write_text(
        "# test\n\ntitle\n\n0 1\nH 0.0 0.0 0.0\n\n",
        encoding="utf-8",
        newline="\n",
    )
    (pointdir / "input.wfn").write_text("fixture\n", encoding="utf-8")
    (round_dir / "POINTS.txt").write_text(
        str(pointdir.resolve()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (pointdir / "input.wfn").write_text(
        "TEST WAVEFUNCTION 18 PRIMITIVES\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(live_executor_mod, "_configured_scheduler", lambda: "slurm")
    monkeypatch.setattr(
        live_executor_mod,
        "_configured_daemon_runtime_modules",
        lambda: [],
    )
    monkeypatch.setattr(
        live_executor_mod,
        "_configured_backend_modules",
        lambda backend, fallback: [],
    )

    body = build_sbatch_script(
        phase_name=phase_name,
        iteration=iteration,
        campaign_dir=campaign,
        config=CampaignConfig(),
        array_size=1,
        replacement_round=2,
    )

    assert str((round_dir / "POINTS.txt").resolve()) in body
    assert "export ICHOR_STAGING_ROOT=" + shlex.quote(str(round_dir.resolve())) in body
    assert required_filename + ' missing in $POINT_DIR' in body


@pytest.mark.skipif(not _which("bash"), reason="bash is required for shell guard execution")
@pytest.mark.parametrize("required_filename", ["input.gjf", "input.wfn"])
def test_replacement_pointdir_guard_executes_valid_round_and_rejects_sibling(
    tmp_path,
    required_filename,
):
    round_dir = (
        tmp_path
        / ".DATA"
        / "STAGING"
        / "iter_1"
        / "replacement_round_0002"
    )
    pointdir = round_dir / "POINT_0004.pointdir"
    pointdir.mkdir(parents=True)
    (pointdir / required_filename).write_text("fixture\n", encoding="utf-8")
    points = round_dir / "POINTS.txt"
    points.write_text(str(pointdir.resolve()) + "\n", encoding="utf-8", newline="\n")
    fragment = "\n".join(
        [
            "set -eu",
            "ICHOR_LOGICAL_ARRAY_TASK_ID=0",
            *live_executor_mod._pointdir_selection_lines(
                str(points),
                required_filename=required_filename,
            ),
            'printf "%s\\n" "$POINT_DIR"',
        ]
    )

    accepted = subprocess.run(
        ["bash", "-c", fragment],
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == str(pointdir.resolve())

    sibling = round_dir.parent / "replacement_round_0001" / pointdir.name
    sibling.mkdir(parents=True)
    (sibling / required_filename).write_text("fixture\n", encoding="utf-8")
    points.write_text(str(sibling.resolve()) + "\n", encoding="utf-8", newline="\n")
    rejected = subprocess.run(
        ["bash", "-c", fragment],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "escapes exact campaign staging round" in rejected.stderr


def test_configured_batch_python_probe_uses_exact_interpreter(monkeypatch):
    from ichor.hpc.active_learning.daemon import preflight

    executable = "/home/user/.venv/ichor-csf3/bin/python"
    calls = []
    monkeypatch.setattr(preflight.os.path, "isfile", lambda value: value == executable)
    monkeypatch.setattr(preflight.os, "access", lambda value, mode: value == executable)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/bin/bash" if name == "bash" else None)

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "executable": executable,
                    "version": [3, 11, 15],
                    "modules": {
                        label: {"ok": True, "error": ""}
                        for label in SUBMITTED_PYTHON_IMPORTS
                    },
                }
            ) + "\n",
            stderr="",
        )

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    ok, version, error = preflight._probe_configured_python(
        executable,
        ["python/3.11", "mkl/2024.2"],
    )

    assert ok is True
    assert version == "3.11.15"
    assert error == ""
    assert calls[0][0][:3] == ["/bin/bash", "--login", "-c"]
    script = calls[0][0][3]
    assert "module load python/3.11" in script
    assert "module load mkl/2024.2" in script
    assert executable in script
    assert "ariadne" in script
    assert "ichor.hpc" in script
    assert "polus.samplers.RS.randomSampling" not in script
    assert "pyferebus.executors.trainer" in script


def test_configured_batch_python_probe_rejects_wrong_version(monkeypatch):
    from ichor.hpc.active_learning.daemon import preflight

    executable = "/home/user/.venv/wrong/bin/python"
    monkeypatch.setattr(preflight.os.path, "isfile", lambda value: value == executable)
    monkeypatch.setattr(preflight.os, "access", lambda value, mode: value == executable)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/bin/bash" if name == "bash" else None)
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "executable": executable,
                    "version": [3, 13, 1],
                    "modules": {
                        label: {"ok": True, "error": ""}
                        for label in SUBMITTED_PYTHON_IMPORTS
                    },
                }
            ) + "\n",
            stderr="",
        ),
    )

    ok, version, error = preflight._probe_configured_python(executable)

    assert ok is False
    assert version == "3.13.1"
    assert "Python 3.11" in error


def test_configured_batch_python_probe_reports_submitted_import_failure(monkeypatch):
    from ichor.hpc.active_learning.daemon import preflight

    executable = "/home/user/.venv/ichor-csf3/bin/python"
    monkeypatch.setattr(preflight.os.path, "isfile", lambda value: value == executable)
    monkeypatch.setattr(preflight.os, "access", lambda value, mode: value == executable)
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: "/bin/bash" if name == "bash" else None,
    )
    statuses = {
        label: {"ok": True, "error": ""}
        for label in SUBMITTED_PYTHON_IMPORTS
    }
    statuses["ariadne"] = {
        "ok": False,
        "error": "ImportError: libmkl_rt.so not found",
    }
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "executable": executable,
                    "version": [3, 11, 15],
                    "modules": statuses,
                }
            )
            + "\n",
            stderr="",
        ),
    )

    interpreter_ok, version, error, imports = (
        preflight._probe_configured_python_details(executable, ["mkl/2024.2"])
    )
    contract_ok, _, contract_error = preflight._probe_configured_python(
        executable,
        ["mkl/2024.2"],
    )

    assert interpreter_ok is True
    assert version == "3.11.15"
    assert error == ""
    assert imports["ariadne"]["ok"] is False
    assert contract_ok is False
    assert "libmkl_rt.so" in contract_error


def test_gaussian_probe_uses_combined_submitted_runtime_module_stack(monkeypatch):
    from ichor.hpc.active_learning.daemon import preflight

    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {"scheduler": "slurm"},
                "software": {
                    "python": {"modules": ["python/3.11"]},
                    "ariadne_runtime": {"modules": ["mkl/2025.0"]},
                    "gaussian": {
                        "modules": ["apps/gaussian/g16"],
                        "executable_path": "$g16root/g16/g16",
                    },
                },
            }
        },
        "csf3",
    )
    scripts = []

    def fake_login_shell(script, *, timeout=30):
        scripts.append(script)
        return SimpleNamespace(
            returncode=0,
            stdout="/opt/gaussian/g16\n",
            stderr="",
        )

    monkeypatch.setattr(preflight, "_run_login_shell", fake_login_shell)

    ok, resolved, error = preflight._probe_gaussian_environment()

    assert ok is True
    assert resolved == "/opt/gaussian/g16"
    assert error == ""
    assert scripts[0].index("module load python/3.11") < scripts[0].index(
        "module load apps/gaussian/g16"
    )
    assert "module load mkl/2025.0" in scripts[0]


def test_replacement_resource_solver_uses_nested_round_atom_count(
    monkeypatch,
    tmp_path,
):
    from ichor.hpc.active_learning.daemon.resource_solver import resolve_phase_resources

    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "parallel_environments": {"multicore": [1, 64]},
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                }
            }
        },
        "csf3",
    )
    root = tmp_path / ".DATA" / "STAGING" / "iter_1"
    round_dir = root / "replacement_round_0002"
    pointdir = round_dir / "POINT_0012.pointdir"
    pointdir.mkdir(parents=True)
    geometry = ["C " + str(index) + ".0 0.0 0.0" for index in range(18)]
    (pointdir / "input.gjf").write_text(
        "\n".join(["# test", "", "title", "", "0 1", *geometry, ""]),
        encoding="utf-8",
        newline="\n",
    )
    (round_dir / "POINTS.txt").write_text(
        str(pointdir.resolve()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (pointdir / "input.wfn").write_text(
        "TEST WAVEFUNCTION 18 PRIMITIVES\n",
        encoding="utf-8",
        newline="\n",
    )
    cfg = CampaignConfig()

    resolved = resolve_phase_resources(
        phase_name="REPLACEMENT_AIMALL",
        config=cfg,
        partition="multicore",
        campaign_dir=tmp_path,
        iteration=1,
        replacement_round=2,
    )

    assert resolved.extra["n_atoms"] == 18


def test_replacement_resource_solver_refuses_missing_round_evidence(
    monkeypatch,
    tmp_path,
):
    from ichor.hpc.active_learning.daemon.resource_solver import resolve_phase_resources

    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
                "hpc": {
                    "parallel_environments": {"multicore": [1, 64]},
                    "memory_per_core_gb_by_partition": {"multicore": 8},
                }
            }
        },
        "csf3",
    )

    with pytest.raises(
        BackendSubmissionError,
        match="resource evidence not yet produced",
    ):
        resolve_phase_resources(
            phase_name="REPLACEMENT_GAUSSIAN",
            config=CampaignConfig(),
            partition="multicore",
            campaign_dir=tmp_path,
            iteration=1,
            replacement_round=1,
        )


def test_write_real_script_creates_sbatch_log_dirs(tmp_path):
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
    from ichor.hpc.active_learning.daemon.submission_intent import (
        write_pre_submit_intent,
    )

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    source = tmp_path / "pool-source.xyz"
    source.write_text(
        "1\nframe 0\nH 0 0 0\n1\nframe 1\nH 0 0 0.1\n",
        encoding="utf-8",
        newline="\n",
    )
    TrajectoryPool.import_from(source, campaign)
    from ichor.hpc.active_learning.custom_bootstrap import (
        commit_bootstrap_plan,
        inspect_bootstrap_inputs,
    )

    config = CampaignConfig()
    config.point_allocation.bootstrap_training_size = 1
    config.point_allocation.bootstrap_internal_validation_size = 1
    config.point_allocation.bootstrap_external_validation_size = 0
    pool = TrajectoryPool.load(campaign)
    commit_bootstrap_plan(
        inspect_bootstrap_inputs(
            campaign,
            config,
            pool.to_atoms_list(),
            pool_sha256=pool.sha256,
        )
    )
    write_pre_submit_intent(
        campaign,
        campaign_uid="uid",
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        expected_tasks=1,
    )
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=config,
        backend_check=False,
    )
    script = ex._write_real_script(
        "PHASE_A_DIVERSITY",
        SimpleNamespace(iteration=0, campaign_uid="uid"),
    )
    assert script.name == "job.sh"
    assert script.parent.parent.name == "iteration-000000"
    assert (script.parent / "OUTPUTS").is_dir()
    assert (script.parent / "ERRORS").is_dir()


# --- live binaries (skipped off-cluster) ----------------------------------


@pytest.mark.live
@requires_sbatch
def test_sbatch_help_runs():
    """sbatch --help exits 0 on a healthy SLURM host."""
    result = subprocess.run(
        ["sbatch", "--help"], capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0
    assert "sbatch" in result.stdout.lower() or "sbatch" in result.stderr.lower()


@pytest.mark.live
@requires_sacct
def test_sacct_help_runs():
    result = subprocess.run(
        ["sacct", "--help"], capture_output=True, text=True, timeout=20,
    )
    # sacct prints help to stderr on some versions; accept either stream.
    assert result.returncode == 0
    text = (result.stdout + result.stderr).lower()
    assert "sacct" in text


@pytest.mark.live
@requires_sbatch
@requires_sacct
def test_sbatch_one_shot_roundtrip(tmp_path):
    """Submit a 1-second sleep via sbatch --parsable; poll sacct; assert
    the JobID round-trips and reaches a terminal state."""
    script = tmp_path / "smoke.sh"
    script.write_text(
        "#!/bin/bash\n#SBATCH --time=00:01:00\nsleep 1\n",
        encoding="utf-8",
    )
    os.chmod(script, 0o755)
    submit = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert submit.returncode == 0, submit.stderr
    job_id = submit.stdout.strip().split(";")[0]
    assert job_id and job_id[0].isdigit()

    from ichor.hpc.active_learning.submit.sacct_poll import (
        aggregate_states,
        poll_job,
    )

    deadline = time.time() + 180.0
    last_obs = []
    while time.time() < deadline:
        try:
            last_obs = poll_job(job_id)
        except RuntimeError:
            time.sleep(5.0)
            continue
        summary = aggregate_states(job_id, last_obs)
        if summary.is_terminal and summary.n_tasks > 0:
            assert summary.n_completed >= 1 or summary.n_failed >= 1
            return
        time.sleep(5.0)
    pytest.fail("smoke job did not reach a terminal sacct state in 180s")


@pytest.mark.live
@requires_gaussian
def test_gaussian_binary_runs_version():
    binary = _which("g16") or _which("g09")
    result = subprocess.run(
        [binary, "--help"], capture_output=True, text=True, timeout=20, check=False,
    )
    # Gaussian exits non-zero on --help but writes its banner. Accept any output.
    text = (result.stdout + result.stderr).lower()
    assert "gaussian" in text or "g16" in text or "g09" in text


@pytest.mark.live
@requires_aimall
def test_aimall_binary_exists_and_is_executable():
    aim = _which("aimqb.ish") or _which("aimqb")
    assert aim
    assert os.access(aim, os.X_OK)


@pytest.mark.live
@requires_ferebus
def test_ferebus_binary_exists_and_is_executable():
    fer = _which("FEREBUS") or _which("ferebus")
    assert fer
    assert os.access(fer, os.X_OK)


@pytest.mark.live
@requires_ariadne
def test_ariadne_optimiser_class_present():
    import ariadne  # type: ignore[import]
    assert hasattr(ariadne, "Geometric_Trqn") or hasattr(ariadne, "Ds_Optimiser"), (
        "ariadne imported but no documented optimiser class found"
    )


@pytest.mark.live
@requires_sbatch
@requires_gaussian
@requires_aimall
@requires_ferebus
@requires_ariadne
def test_full_backend_set_available():
    """All-in-one preflight: only passes on a CSF4-equivalent host where
    every backend the daemon needs is present. This is the test that flips
    from skip to pass when you move from Windows to the cluster."""
    a = check_backends()
    assert a.all_present, missing_backend_message(a)
