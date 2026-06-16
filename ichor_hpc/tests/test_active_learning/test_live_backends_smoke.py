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
            phase_name="PHASE_A_POLUS",
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
        polus_rs=True,
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


def test_build_sbatch_script_renders_gaussian_block():
    body = build_sbatch_script(
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
    )
    assert "#SBATCH --job-name=INITIAL_GAUSSIAN-0" in body
    assert "module load gaussian/g16c01_em64t_detectcpu" in body
    assert "g16 < input.gjf" in body
    assert "> input.gau" in body
    assert "> output.log" not in body
    assert "|| true" not in body
    assert "set -euo pipefail" in body
    assert "export LC_ALL=C" in body
    assert "export LC_NUMERIC=C" in body
    assert "export GAUSS_MDEF=" in body


def test_link0_gaussian_memory_validates_after_auto_resolution(monkeypatch):
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
    cfg.gaussian.memory_mode = "link0"
    cfg.resources.mem_per_cpu = "auto"
    cfg.resources.partition = "multicore"
    cfg.gaussian.nproc = 1
    cfg.gaussian.mem = "8GB"

    with pytest.raises(BackendSubmissionError, match="gaussian.mem"):
        build_sbatch_script(
            phase_name="INITIAL_GAUSSIAN",
            iteration=0,
            campaign_dir=Path("/scratch/campaign"),
            config=cfg,
        )


def test_slurm_env_gaussian_memory_still_uses_environment_contract(monkeypatch):
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
    cfg.gaussian.memory_mode = "slurm_env"
    cfg.resources.mem_per_cpu = "auto"
    cfg.gaussian.mem = "500GB"

    body = build_sbatch_script(
        phase_name="INITIAL_GAUSSIAN",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )

    assert "#SBATCH --mem-per-cpu=4G" in body
    assert "export GAUSS_MDEF=3GB" in body
    assert "%mem" not in body.lower()


def test_build_sbatch_script_uses_strict_daemon_module_loads(monkeypatch):
    monkeypatch.setattr(
        live_executor_mod,
        "_configured_daemon_runtime_modules",
        lambda: list(live_executor_mod.DEFAULT_DAEMON_RUNTIME_MODULES),
    )
    body = build_sbatch_script(
        phase_name="PHASE_A_POLUS",
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
        phase_name="PHASE_A_POLUS",
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
        "csf4": {"software": {"python": {"modules": ["python,custom"]}}}
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
            phase_name="PHASE_A_POLUS",
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
                        "scratch_root": "/scratch/$USER",
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
    assert "export GAUSS_SCRATCH_ROOT=/scratch/$USER" in body
    assert 'export GAUSS_SCRDIR="${GAUSS_SCRATCH_ROOT%/}/ichor_gaussian_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"' in body
    assert 'export GAUSS_PDEF="${SLURM_CPUS_PER_TASK:-1}"' in body
    assert "export GAUSS_MDEF=6GB" in body


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
    cfg.gaussian.nproc = 2
    cfg.resources.cpus_per_task = 2
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


@pytest.mark.parametrize(("machine", "max_cores"), [("csf3", 168), ("csf4", 32)])
def test_multicore_one_core_request_fails_before_sbatch(monkeypatch, machine, max_cores):
    _install_fake_global_variables(
        monkeypatch,
        {
            machine: {
                "hpc": {
                    "scheduler": "slurm",
                    "parallel_environments": {"multicore": [2, max_cores]},
                }
            }
        },
        machine,
    )
    cfg = CampaignConfig()
    cfg.resources.partition = "multicore"
    cfg.resources.cpus_per_task = 1

    with pytest.raises(BackendSubmissionError, match="configured range is \\[2,"):
        build_sbatch_script(
            phase_name="PHASE_A_POLUS",
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
                }
            }
        },
        "csf3",
    )
    cfg = CampaignConfig()
    cfg.resources.partition = "multicore"
    cfg.resources.cpus_per_task = 2

    body = build_sbatch_script(
        phase_name="PHASE_A_POLUS",
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
                }
            }
        },
        "csf3",
    )
    cfg = CampaignConfig()
    cfg.resources.partition = "serial"
    cfg.resources.cpus_per_task = 1

    body = build_sbatch_script(
        phase_name="PHASE_A_POLUS",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )

    assert "#SBATCH --partition=serial" in body
    assert "#SBATCH --cpus-per-task=1" in body


def test_gaussian_nproc_must_not_exceed_live_resource_cpus():
    cfg = CampaignConfig()
    cfg.resources.cpus_per_task = 2
    cfg.gaussian.nproc = 3

    with pytest.raises(BackendSubmissionError, match="gaussian.nproc"):
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
    cfg.resources.mem_per_cpu = "8G"
    with pytest.raises(BackendSubmissionError, match="exceeds configured profile memory cap"):
        build_sbatch_script(
            phase_name="PHASE_A_POLUS",
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
    cfg.resources.walltime_hours = 7
    body = build_sbatch_script(
        phase_name="PHASE_A_POLUS",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )
    assert "#SBATCH --partition=csf4-debug" in body
    assert "#SBATCH --time=7:00:00" in body


def test_build_sbatch_script_explicit_scheduler_overrides_win():
    cfg = CampaignConfig()
    cfg.resources.partition = "yaml-partition"
    cfg.resources.walltime_hours = 7
    body = build_sbatch_script(
        phase_name="PHASE_A_POLUS",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
        partition="explicit-partition",
        walltime_hours=11,
    )
    assert "#SBATCH --partition=explicit-partition" in body
    assert "#SBATCH --time=11:00:00" in body
    assert "yaml-partition" not in body
    assert "#SBATCH --time=7:00:00" not in body


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

    cfg = CampaignConfig()
    cfg.resources.walltime_hours = 9
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    staging = campaign / "6_TRAINED_MODELS" / "iteration-staging"
    staging.mkdir(parents=True)
    (staging / stg.FEREBUS_JOB_DETAILS).write_text(
        "system_name WATER\n", encoding="utf-8",
    )

    calls = {}

    def fake_stage(campaign_dir, config, training_version, *, is_initial=False):
        calls["stage"] = {
            "campaign_dir": Path(campaign_dir),
            "training_version": training_version,
            "is_initial": is_initial,
        }
        return staging, 3

    def fake_submit(jd_file, working_directory, **kwargs):
        script = Path(working_directory) / "runFerebus.sh"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        calls["submit"] = {
            "jd_file": Path(jd_file),
            "working_directory": Path(working_directory),
            "kwargs": dict(kwargs),
        }
        return FerebusSubmission(
            job_id="4242",
            cluster=None,
            submission_script=script,
            working_dir=Path(working_directory),
            transfer_learning=False,
        )

    monkeypatch.setattr(stg, "stage_ferebus_inputs", fake_stage)
    monkeypatch.setattr(pyferebus_wrap, "submit_ferebus", fake_submit)
    _install_fake_global_variables(
        monkeypatch,
        {
            "csf3": {
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
    result = ex.submit_or_run(
        SimpleNamespace(iteration=0, training_set_version=4),
        "FEREBUS",
    )

    assert result.submitted_job_id == "4242"
    assert calls["stage"]["training_version"] == 4
    assert calls["stage"]["is_initial"] is False
    assert calls["submit"]["jd_file"] == staging / stg.FEREBUS_JOB_DETAILS
    assert calls["submit"]["working_directory"] == staging
    assert calls["submit"]["kwargs"]["overwrite_workdir"] is False
    assert calls["submit"]["kwargs"]["move_dataset_files"] is True
    assert calls["submit"]["kwargs"]["submit_runner"] is runner
    assert calls["submit"]["kwargs"]["walltime_hours"] == cfg.resources.walltime_hours
    assert calls["submit"]["kwargs"]["platform"] == "CSF3"


def test_build_sbatch_script_renders_aimall_block(monkeypatch):
    monkeypatch.setattr(
        live_executor_mod,
        "_configured_backend_path",
        lambda backend_name, fallback: "/opt/AIM All/aimqb.ish",
    )
    body = build_sbatch_script(
        phase_name="AIMALL",
        iteration=3,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
    )
    assert shlex.quote("/opt/AIM All/aimqb.ish") + " -nogui -encomp=3 input.wfn" in body
    assert "#SBATCH --cpus-per-task=8" in body
    assert "AIMALL-3" in body


def test_build_sbatch_script_renders_ariadne_block():
    body = build_sbatch_script(
        phase_name="ARIADNE_ARRAY",
        iteration=2,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
    )
    assert "ariadne_runner" in body
    assert shlex.quote(sys.executable) + " -m ichor.hpc.active_learning.acquisition.ariadne_runner" in body
    assert "\npython -m ichor.hpc.active_learning.acquisition.ariadne_runner" not in body
    assert "--seed-index $SLURM_ARRAY_TASK_ID" in body
    assert "--iteration 2" in body


def test_build_sbatch_script_renders_polus_block_with_configured_descriptor():
    cfg = CampaignConfig()
    cfg.phase_b.descriptor = "acquisition_weighted"
    body = build_sbatch_script(
        phase_name="PHASE_B_POLUS",
        iteration=4,
        campaign_dir=Path("/scratch/campaign"),
        config=cfg,
    )
    assert "polus_wrapper" in body
    assert shlex.quote(sys.executable) + " -m ichor.hpc.active_learning.sampling.polus_wrapper" in body
    assert "\npython -m ichor.hpc.active_learning.sampling.polus_wrapper" not in body
    assert "--descriptor acquisition_weighted" in body
    assert "--iteration 4" in body


def test_build_sbatch_script_renders_phase_a_polus_as_negative_iteration():
    body = build_sbatch_script(
        phase_name="PHASE_A_POLUS",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=CampaignConfig(),
    )
    assert "polus_wrapper" in body
    assert "--descriptor rmsd_massweight" in body
    assert "--iteration -1" in body


def test_write_real_script_creates_sbatch_log_dirs(tmp_path):
    ex = LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=CampaignConfig(),
        backend_check=False,
    )
    ex._write_real_script(
        "PHASE_A_POLUS",
        SimpleNamespace(iteration=0, campaign_uid="uid"),
    )
    assert (tmp_path / "campaign" / ".DATA" / "SCRIPTS" / "OUTPUTS").is_dir()
    assert (tmp_path / "campaign" / ".DATA" / "SCRIPTS" / "ERRORS").is_dir()


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
