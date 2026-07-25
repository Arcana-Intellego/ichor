from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.live_executor import (
    build_scheduler_script,
)
from ichor.hpc.active_learning.daemon.preflight import BackendAvailability
from ichor.hpc.active_learning.daemon.resource_usage import (
    collect_usage,
    parse_sge_usage_records,
)
from ichor.hpc.active_learning.daemon.submitted_environment_smoke import (
    render_submitted_environment_smoke_script,
)
from ichor.hpc.active_learning.daemon.submission_intent import expected_job_name
from ichor.hpc.active_learning.submit.sacct_poll import JobStatus, aggregate_states
from ichor.hpc.active_learning.submit.scheduler_backend import SgeScheduler
from ichor.hpc.active_learning.submit.sge import (
    expand_sge_tasks,
    find_accounted_job_by_name_detailed,
    find_active_job_by_name_detailed,
    parse_qacct_output,
    parse_qstat_xml,
    parse_sge_duration_seconds,
    parse_qsub_terse_output,
    poll_job,
    qacct_observations,
    qstat_observations,
)


QSTAT_XML = """<?xml version='1.0'?>
<job_info>
  <queue_info>
    <Queue-List>
      <name>all.q@compute-0-1.local</name>
      <job_list state="running">
        <JB_job_number>898176</JB_job_number>
        <JB_name>ichor-campaign-ariadne</JB_name>
        <JB_owner>q81036tb</JB_owner>
        <state>r</state>
        <slots>4</slots>
        <tasks>2</tasks>
      </job_list>
    </Queue-List>
  </queue_info>
  <job_info>
    <job_list state="pending">
      <JB_job_number>898176</JB_job_number>
      <JB_name>ichor-campaign-ariadne</JB_name>
      <JB_owner>q81036tb</JB_owner>
      <state>qw</state>
      <slots>4</slots>
      <tasks>3-5:1</tasks>
    </job_list>
  </job_info>
</job_info>
"""


QACCT = """==============================================================
qname        all.q
hostname     compute-0-1.local
owner        q81036tb
jobname      ichor-campaign-ariadne
jobnumber    898176
taskid       1
slots        4
failed       0
exit_status  0
ru_wallclock 26s
ru_maxrss    18.023KB
maxvmem      14.570MB
==============================================================
"""


def _completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _install_ffluxlab_profile(monkeypatch):
    profile = {
        "ffluxlab": {
            "hpc": {
                "scheduler": "sge",
                "jobscript_shebang": "#!/bin/bash",
                "max_array_tasks": 75000,
                "max_job_log_files_per_directory": 5000,
                "memory_per_core_gb": 4,
                "memory_per_core_gb_by_partition": {
                    "serial": 4,
                    "multicore": 4,
                    "himem": 16,
                },
                "parallel_environments": {
                    "serial": [1, 1],
                    "multicore": [2, 44],
                    "himem": [1, 44],
                },
                "partitions": {
                    "serial": {
                        "min_cpus": 1,
                        "max_cpus": 1,
                        "memory_per_core_gb": 4,
                        "max_walltime_hours": 168,
                        "daemon_supported": True,
                        "scheduler_queue": "all.q",
                        "parallel_environment": None,
                    },
                    "multicore": {
                        "min_cpus": 2,
                        "max_cpus": 44,
                        "memory_per_core_gb": 4,
                        "max_walltime_hours": 168,
                        "daemon_supported": True,
                        "scheduler_queue": "all.q",
                        "parallel_environment": "smp",
                    },
                    "himem": {
                        "min_cpus": 1,
                        "max_cpus": 44,
                        "memory_per_core_gb": 16,
                        "max_walltime_hours": 168,
                        "daemon_supported": True,
                        "scheduler_queue": "mem.q",
                        "parallel_environment": "smp",
                    },
                },
            },
            "software": {
                "python": {
                    "python_path": "/home/user/.venv/ichor-ffluxlab/bin/python",
                    "library_path": ["/home/user/opt/python-3.11.15/lib"],
                    "modules": [],
                },
                "ariadne_runtime": {
                    "modules": ["compilers/intel/21.0.3"],
                },
                "gaussian": {
                    "executable_path": "g09",
                    "modules": ["apps/gaussian/g09"],
                },
                "aimall": {
                    "executable_path": "aimall",
                    "modules": ["apps/aimall/19.02.13"],
                },
                "ferebus": {
                    "executable_path": "/home/user/.local/bin/ferebus",
                    "pyferebus_platform": "CSF3",
                },
            },
        }
    }
    fake = ModuleType("ichor.hpc.global_variables")
    fake.ICHOR_CONFIG = profile
    fake.MACHINE = "ffluxlab"

    def get_param(config, *keys, default=None):
        value = config
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    fake.get_param_from_config = get_param
    monkeypatch.setitem(sys.modules, "ichor.hpc.global_variables", fake)


def _explicit_sge_config():
    config = CampaignConfig()
    config.resources.defaults.partition = "multicore"
    config.resources.defaults.cpus_per_task = 4
    config.resources.defaults.mem_per_cpu = "4G"
    return config


def test_parse_qsub_terse_scalar_and_array():
    assert parse_qsub_terse_output("898176\n") == "898176"
    assert parse_qsub_terse_output("898176.1-200:1\n") == "898176"
    with pytest.raises(ValueError):
        parse_qsub_terse_output("Your job 898176 was submitted")


def test_expand_sge_tasks_is_inclusive_and_rejects_duplicates():
    assert expand_sge_tasks("1,3-7:2") == [1, 3, 5, 7]
    with pytest.raises(ValueError):
        expand_sge_tasks("1,1")


def test_qstat_xml_converts_native_tasks_to_zero_based_logical_ids():
    rows = parse_qstat_xml(QSTAT_XML)
    assert [row.logical_job_id for row in rows] == [
        "898176_1",
        "898176_2",
        "898176_3",
        "898176_4",
    ]
    assert [row.state for row in rows] == ["r", "qw", "qw", "qw"]


def test_qstat_xml_accepts_captured_ffluxlab_comma_task_expression():
    captured = QSTAT_XML.replace("<tasks>3-5:1</tasks>", "<tasks>1,2</tasks>")
    rows = parse_qstat_xml(captured)
    pending = [row for row in rows if row.state == "qw"]
    assert [row.task_index for row in pending] == [0, 1]


def test_qacct_success_and_failure_contracts():
    records = parse_qacct_output(QACCT)
    observations = qacct_observations(records)
    assert observations[0].job_id == "898176_0"
    assert observations[0].status is JobStatus.COMPLETED
    assert observations[0].elapsed_seconds == 26

    failed = dict(records[0], exit_status="7")
    failed_observation = qacct_observations([failed])[0]
    assert failed_observation.status is JobStatus.FAILED
    assert failed_observation.exit_code == (7, 0)

    cancelled = dict(records[0], failed="100", exit_status="137")
    assert qacct_observations(
        [cancelled], cancellation_requested=True
    )[0].status is JobStatus.CANCELLED
    assert qacct_observations([cancelled])[0].status is JobStatus.FAILED


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("26s", 26),
        ("12.6", 13),
        ("2m", 120),
        ("1:02:03", 3723),
        ("1:02:03:04", 93784),
    ],
)
def test_sge_duration_parser_accepts_observed_and_accounting_forms(
    value, expected
):
    assert parse_sge_duration_seconds(value) == expected


def test_sge_resource_usage_parses_ffluxlab_units():
    rows = parse_sge_usage_records(
        parse_qacct_output(QACCT),
        job_id="898176",
    )
    assert rows == [
        {
            "job_id": "898176_0",
            "job_id_raw": "898176.1",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "elapsed_seconds": 26,
            "allocated_cpus": 4,
            "requested_memory": "",
            "max_rss_mib": pytest.approx(18.023 / 1024.0),
            "max_vm_size_mib": pytest.approx(14.570),
            "total_cpu": "",
        }
    ]


def test_sge_resource_usage_queries_through_recorded_scheduler(tmp_path):
    calls = []

    def runner(command, **kwargs):
        calls.append(list(command))
        return _completed(QACCT)

    summary = collect_usage(
        tmp_path,
        intent={
            "attempt_id": "a1",
            "submission_identity": "r0000-a0001-deadbeef",
            "phase": "ARIADNE_ARRAY",
            "iteration": 1,
            "job_id": "898176",
            "expected_tasks": 1,
            "submission_kind": "array",
            "scheduler_identity_kind": "sge",
        },
        history_limit=10,
        runner=runner,
    )

    assert calls == [["qacct", "-j", "898176"]]
    assert summary["telemetry_status"] == "final"
    assert summary["n_rows"] == 1
    assert summary["n_failures"] == 0
    assert summary["query_command"] == ["qacct", "-j", "898176"]


def test_poll_merges_finished_accounting_with_live_queue_rows():
    def qstat_runner(command, **kwargs):
        assert command[:2] == ["qstat", "-xml"]
        return _completed(QSTAT_XML)

    def qacct_runner(command, **kwargs):
        assert command == ["qacct", "-j", "898176"]
        return _completed(QACCT)

    observations = poll_job(
        "898176",
        qstat_runner=qstat_runner,
        qacct_runner=qacct_runner,
    )
    summary = aggregate_states(
        "898176",
        observations,
        expected_task_count=5,
        submission_kind="array",
        strict_parent_job_id=False,
    )
    assert summary.n_completed == 1
    assert summary.n_pending_or_running == 4
    assert summary.n_missing == 0


def test_qstat_held_and_error_states_remain_fail_closed():
    held = QSTAT_XML.replace("<state>r</state>", "<state>hqw</state>")
    observations = qstat_observations(parse_qstat_xml(held))
    assert observations[0].status is JobStatus.PENDING

    error = QSTAT_XML.replace("<state>r</state>", "<state>Eqw</state>")
    observations = qstat_observations(parse_qstat_xml(error))
    assert observations[0].status is JobStatus.UNKNOWN
    assert observations[0].parse_error


def test_delayed_qacct_remains_live_while_qstat_owns_array():
    def qstat_runner(command, **kwargs):
        return _completed(QSTAT_XML)

    def qacct_runner(command, **kwargs):
        return _completed(
            "",
            "error: job id 898176 not found",
            returncode=1,
        )

    observations = poll_job(
        "898176",
        qstat_runner=qstat_runner,
        qacct_runner=qacct_runner,
    )
    summary = aggregate_states(
        "898176",
        observations,
        expected_task_count=5,
        submission_kind="array",
        strict_parent_job_id=False,
    )
    assert summary.n_observed == 4
    assert summary.n_missing == 1
    assert summary.n_pending_or_running == 5
    assert not summary.is_terminal


def test_name_adoption_requires_unique_current_owner(monkeypatch):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.submit.sge.getpass.getuser",
        lambda: "q81036tb",
    )

    def qstat_runner(command, **kwargs):
        return _completed(QSTAT_XML)

    found = find_active_job_by_name_detailed(
        "ichor-campaign-ariadne",
        qstat_runner=qstat_runner,
    )
    assert found.job_id == "898176"
    assert not found.inconclusive

    foreign = QSTAT_XML.replace("q81036tb", "someone-else")

    def foreign_runner(command, **kwargs):
        return _completed(foreign)

    found = find_active_job_by_name_detailed(
        "ichor-campaign-ariadne",
        qstat_runner=foreign_runner,
    )
    assert found.job_id is None


def test_accounted_name_adoption_waits_for_complete_array(monkeypatch):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.submit.sge.getpass.getuser",
        lambda: "q81036tb",
    )

    def qacct_runner(command, **kwargs):
        return _completed(QACCT)

    def empty_qstat_runner(command, **kwargs):
        return _completed("<job_info><queue_info/><job_info/></job_info>")

    lookup = find_accounted_job_by_name_detailed(
        "ichor-campaign-ariadne",
        expected_task_count=5,
        submission_kind="array",
        qacct_runner=qacct_runner,
        qstat_runner=empty_qstat_runner,
    )
    assert lookup.job_id == "898176"
    assert not lookup.terminal


def test_sge_submit_exports_binding_and_parses_parent_id():
    seen = {}

    def runner(command, **kwargs):
        seen["command"] = command
        return _completed("898176.1-5:1\n")

    result = SgeScheduler().submit(
        "/tmp/job.sh",
        binding_sha256="a" * 64,
        runner=runner,
    )
    assert result.job_id == "898176"
    assert seen["command"] == [
        "qsub",
        "-terse",
        "-v",
        "ICHOR_SCRIPT_BINDING_SHA256=" + "a" * 64,
        "/tmp/job.sh",
    ]


@pytest.mark.parametrize(
    ("phase", "array_size", "marker"),
    [
        ("INITIAL_GAUSSIAN", 2, "g09 < input.gjf"),
        ("INITIAL_AIMALL", 2, "aimall -nogui"),
        ("ARIADNE_ARRAY", 2, "ariadne_runner"),
        ("PHASE_A_DIVERSITY", None, "diversity"),
        ("INITIAL_FEREBUS", 2, "ferebus"),
    ],
)
def test_sge_scripts_use_native_directives_and_generic_task_identity(
    monkeypatch, phase, array_size, marker
):
    _install_ffluxlab_profile(monkeypatch)
    config = _explicit_sge_config()
    expected_cpus = 4
    if phase in {"INITIAL_FEREBUS", "FEREBUS"}:
        config.resources.ferebus.cpus_per_task = 20
        expected_cpus = 20
    body = build_scheduler_script(
        phase_name=phase,
        iteration=1,
        campaign_dir=Path("/scratch/campaign"),
        config=config,
        array_size=array_size,
        scheduler_kind="sge",
    )
    assert body.startswith("#!/bin/bash\n")
    assert "#$ -S /bin/bash" in body
    assert "#$ -q all.q" in body
    assert "#$ -pe smp " + str(expected_cpus) in body
    assert "#$ -l h_vmem=" + str(expected_cpus * 4096) + "M" in body
    assert "#SBATCH" not in body
    assert (
        'export ICHOR_SCHEDULER_JOB_ID="${JOB_ID:?missing JOB_ID}"'
        in body
    )
    assert 'export ICHOR_SCHEDULER_CPUS="${NSLOTS:-1}"' in body
    if array_size is not None:
        assert "#$ -t 1-2" in body
        assert 'ICHOR_SCHEDULER_ARRAY_TASK_ID="$((SGE_TASK_ID - 1))"' in body
    else:
        assert "#$ -t " not in body
        assert "export ICHOR_SCHEDULER_ARRAY_TASK_ID=0" in body
    assert "module load compilers/intel/21.0.3" in body
    assert "libmkl_intel_lp64.so.1" in body
    assert 'ICHOR_MKL_RUNTIME_DIR="$(dirname "$ICHOR_MKL_LP64")"' in body
    assert "$ICHOR_INTEL_RUNTIME_DIR:$ICHOR_MKL_RUNTIME_DIR" in body
    if phase == "INITIAL_GAUSSIAN":
        assert "module load apps/gaussian/g09" in body
    if phase == "INITIAL_AIMALL":
        assert "module load apps/aimall/19.02.13" in body
    assert marker in body


def test_sge_script_native_runtime_setup_is_valid_bash(tmp_path, monkeypatch):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available on this host")
    _install_ffluxlab_profile(monkeypatch)
    body = build_scheduler_script(
        phase_name="ARIADNE_ARRAY",
        iteration=1,
        campaign_dir=Path("/scratch/campaign"),
        config=_explicit_sge_config(),
        array_size=2,
        scheduler_kind="sge",
    )
    script = tmp_path / "job.sh"
    script.write_text(body, encoding="utf-8")

    result = subprocess.run(
        [bash, "-n", str(script)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_sge_serial_resource_translation_uses_no_parallel_environment(monkeypatch):
    _install_ffluxlab_profile(monkeypatch)
    config = _explicit_sge_config()
    config.resources.defaults.partition = "serial"
    config.resources.defaults.cpus_per_task = 1
    body = build_scheduler_script(
        phase_name="PHASE_A_DIVERSITY",
        iteration=0,
        campaign_dir=Path("/scratch/campaign"),
        config=config,
        array_size=None,
        scheduler_kind="sge",
    )
    assert "#$ -q all.q" in body
    assert "#$ -l h_vmem=4096M" in body
    assert "#$ -pe " not in body


def test_sge_submitted_environment_smoke_uses_single_core_serial_queue(
    monkeypatch,
):
    _install_ffluxlab_profile(monkeypatch)
    config = _explicit_sge_config()
    availability = SimpleNamespace(
        batch_runtime_modules=("compilers/intel/21.0.3",),
        batch_python_library_paths=("/home/user/opt/python-3.11.15/lib",),
        python_executable="/home/user/.venv/ichor-ffluxlab/bin/python",
        gaussian_binary="/home/modules/apps/gaussian/g09/g09",
        aimall_path="/home/modules/apps/aimall/19.02.13/AIMAll/aimall",
        ferebus_path="/home/user/.local/bin/ferebus",
        bc_path="/usr/bin/bc",
    )
    body = render_submitted_environment_smoke_script(
        config=config,
        availability=availability,
        output_path=Path("/scratch/campaign/smoke.out"),
    )
    assert "#$ -q all.q" in body
    assert "#$ -l h_vmem=4096M" in body
    assert "#$ -pe " not in body
    assert "module load apps/gaussian/g09" in body
    assert "module load apps/aimall/19.02.13" in body
    assert "libmkl_intel_lp64.so.1" in body
    assert "$ICHOR_INTEL_RUNTIME_DIR:$ICHOR_MKL_RUNTIME_DIR" in body
    assert "#SBATCH" not in body


def test_sge_preflight_requires_only_sge_scheduler_commands():
    availability = BackendAvailability(
        profile=True,
        sbatch=False,
        sacct=False,
        squeue=False,
        gaussian=True,
        aimall=True,
        ferebus=True,
        ariadne=True,
        pyferebus=True,
        bc=True,
        gaussian_binary="/opt/g09",
        sbatch_path="",
        sacct_path="",
        bc_path="/usr/bin/bc",
        aimall_path="/opt/aimall",
        ferebus_path="/opt/ferebus",
        active_profile="ffluxlab",
        profile_error="",
        python_executable="/home/user/.venv/ichor-ffluxlab/bin/python",
        batch_python=True,
        scheduler_kind="sge",
        qsub=True,
        qacct=True,
        qstat=True,
        qdel=True,
    )
    assert availability.all_present
    assert not {"sbatch", "sacct", "squeue"}.intersection(availability.missing)


def test_sge_array_throttle_is_emitted_only_when_configured(monkeypatch):
    _install_ffluxlab_profile(monkeypatch)
    config = _explicit_sge_config()
    body = build_scheduler_script(
        phase_name="ARIADNE_ARRAY",
        iteration=1,
        campaign_dir=Path("/scratch/campaign"),
        config=config,
        array_size=5,
        scheduler_kind="sge",
    )
    assert "#$ -tc " not in body

    config.resources.array_concurrency_limit = 3
    throttled = build_scheduler_script(
        phase_name="ARIADNE_ARRAY",
        iteration=1,
        campaign_dir=Path("/scratch/campaign"),
        config=config,
        array_size=5,
        scheduler_kind="sge",
    )
    assert "#$ -tc 3" in throttled


def test_sge_submission_intent_name_is_deterministic_and_safe():
    name = expected_job_name(
        "2717bdc2-c174-4e80-9748-6612967bd50b",
        "INITIAL_REPLACEMENT_GAUSSIAN",
        123,
        replacement_round=4,
        attempt_sequence=5,
        attempt_id="a" * 32,
        scheduler_identity_kind="sge",
    )
    assert name.startswith("ichor-")
    assert len(name) <= 128
    assert " " not in name


def test_sge_cancellation_lookup_preserves_owner_and_uses_parent_job_id():
    commands = []

    def qstat_runner(command, **kwargs):
        commands.append(command)
        return _completed(QSTAT_XML)

    backend = SgeScheduler()
    lookup = backend.cancellation_lookup(
        "898176",
        queue_runner=qstat_runner,
    )
    assert lookup["active"] is True
    assert lookup["inconclusive"] is False
    assert {row["owner"] for row in lookup["rows"]} == {"q81036tb"}

    def qdel_runner(command, **kwargs):
        commands.append(command)
        return _completed()

    assert backend.cancel("898176", runner=qdel_runner) == (True, "")
    assert commands[-1] == ["qdel", "898176"]


def test_sge_accounted_name_adoption_accepts_exact_complete_array(monkeypatch):
    monkeypatch.setattr(
        "ichor.hpc.active_learning.submit.sge.getpass.getuser",
        lambda: "q81036tb",
    )
    base = parse_qacct_output(QACCT)[0]
    records = [
        dict(base, taskid=str(task_id))
        for task_id in range(1, 6)
    ]
    output = "\n".join(
        "==============================================================\n"
        + "\n".join(key + " " + str(value) for key, value in record.items())
        for record in records
    )

    def qacct_runner(command, **kwargs):
        return _completed(output)

    def qstat_runner(command, **kwargs):
        pytest.fail("complete accounting must not need qstat")

    lookup = find_accounted_job_by_name_detailed(
        "ichor-campaign-ariadne",
        expected_task_count=5,
        submission_kind="array",
        qacct_runner=qacct_runner,
        qstat_runner=qstat_runner,
    )
    assert lookup.job_id == "898176"
    assert lookup.terminal is True
    assert lookup.successful is True
    assert lookup.failed is False


def test_stop_job_collection_uses_profile_scheduler_for_state_only_job(
    tmp_path, monkeypatch
):
    from ichor.hpc.active_learning import cli

    _install_ffluxlab_profile(monkeypatch)
    state = SimpleNamespace(
        campaign_uid="2717bdc2-c174-4e80-9748-6612967bd50b",
        iteration=3,
        pending_jobs={"ARIADNE_ARRAY": "898176"},
    )
    jobs = cli._collect_stop_cancel_jobs(tmp_path, state)
    assert jobs["898176"]["scheduler_identity_kinds"] == {"sge"}
    assert next(iter(jobs["898176"]["expected_job_names"])).startswith(
        "ichor-"
    )
