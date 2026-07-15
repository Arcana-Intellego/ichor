"""Crash-recovery + state-integrity guards (P3): A24/A25 adopt-or-submit, the sacct-by-name
lookup, the manifest robustness (A4/A55), the current_version dir guard (A54), and the
reference_scales validation (A31)."""
import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import Daemon, TickStatus
from ichor.hpc.active_learning.daemon.phase_executor import PhaseResult
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    CampaignState,
    StateSchemaError,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.versioning.manifest import write_manifest


# --- A24/A25: adopt an orphaned in-flight job rather than double-submit --------------------


class _ExplodingExecutor:
    """submit_or_run must NOT be called when there is a job to adopt."""
    def submit_or_run(self, state, phase):
        raise AssertionError("should have adopted the in-flight job, not submitted a duplicate")
    def postprocess(self, *a, **k):
        raise AssertionError
    def handle_failure(self, *a, **k):
        raise AssertionError


class _SubmittingExecutor:
    def submit_or_run(self, state, phase):
        return PhaseResult(is_complete=False, submitted_job_id="dry-111")
    def postprocess(self, *a, **k):
        raise AssertionError
    def handle_failure(self, *a, **k):
        raise AssertionError


class _IntentCheckingExecutor:
    def __init__(self, campaign_dir):
        self.campaign_dir = campaign_dir

    def submit_or_run(self, state, phase):
        from ichor.hpc.active_learning.daemon.submission_intent import load_active_intent

        intent = load_active_intent(
            self.campaign_dir, phase.value, int(state.iteration),
        )
        assert intent is not None
        assert intent["status"] == "PRE_SUBMIT"
        return PhaseResult(is_complete=False, submitted_job_id="dry-222")

    def postprocess(self, *a, **k):
        raise AssertionError

    def handle_failure(self, *a, **k):
        raise AssertionError


class _InlineExecutor:
    def submit_or_run(self, state, phase):
        return PhaseResult(is_complete=True, state_updates={})
    def postprocess(self, *a, **k):
        raise AssertionError
    def handle_failure(self, *a, **k):
        raise AssertionError


class _StrictExplodingExecutor(_ExplodingExecutor):
    strict_committed_artifact_verification = True


def _active_state(phase):
    state = fresh_campaign_state()
    state.phase = phase
    state.iteration = 1
    return state


def test_daemon_adopts_inflight_sbatch_job(tmp_path):
    d = Daemon(
        campaign_dir=tmp_path, config=CampaignConfig(),
        executor=_ExplodingExecutor(),
        job_finder=lambda state, phase: "987654",  # pretend this phase already has a running job
    )
    d.state_path().parent.mkdir(parents=True, exist_ok=True)  # the daemon makes this at campaign start
    state = _active_state(CampaignPhase.FEREBUS)
    status = d._on_phase_entry(state, state.phase)
    assert status == TickStatus.SUBMITTED
    assert state.pending_jobs["FEREBUS"] == "987654"  # adopted, not resubmitted


def test_strict_daemon_halts_on_unmanifested_committed_training_pointdir(tmp_path):
    d = Daemon(
        campaign_dir=tmp_path,
        config=CampaignConfig(),
        executor=_StrictExplodingExecutor(),
    )
    d.state_path().parent.mkdir(parents=True, exist_ok=True)
    training = tmp_path / "QM_REFERENCE_DATA" / "iteration-000000"
    rogue = training / "POINT_9999.pointdir"
    rogue.mkdir(parents=True)
    (rogue / "input.gjf").write_text("%chk=x\n", encoding="utf-8")
    write_manifest(training, {})
    state = _active_state(CampaignPhase.SEED_SELECT)
    state.reference_data_version = 0
    state.models_version = -1
    write_state(d.state_path(), state)

    assert d.tick() == TickStatus.HALTED
    halted = read_state(d.state_path())
    assert halted.phase is CampaignPhase.HALTED


def test_strict_daemon_halts_on_invalid_committed_model_version(tmp_path):
    d = Daemon(
        campaign_dir=tmp_path,
        config=CampaignConfig(),
        executor=_StrictExplodingExecutor(),
    )
    d.state_path().parent.mkdir(parents=True, exist_ok=True)
    training = tmp_path / "QM_REFERENCE_DATA" / "iteration-000000"
    training.mkdir(parents=True)
    write_manifest(training, {})
    models = tmp_path / "TRAINED_MODELS" / "iteration-000000"
    models.mkdir(parents=True)
    write_manifest(models, {})
    state = _active_state(CampaignPhase.SEED_SELECT)
    state.reference_data_version = 0
    state.models_version = 0
    write_state(d.state_path(), state)

    assert d.tick() == TickStatus.HALTED
    halted = read_state(d.state_path())
    assert halted.phase is CampaignPhase.HALTED


def test_daemon_submits_when_no_inflight_job(tmp_path):
    d = Daemon(
        campaign_dir=tmp_path, config=CampaignConfig(),
        executor=_SubmittingExecutor(),
        job_finder=lambda state, phase: None,  # nothing running -> submit as normal
    )
    d.state_path().parent.mkdir(parents=True, exist_ok=True)  # the daemon makes this at campaign start
    state = _active_state(CampaignPhase.FEREBUS)
    status = d._on_phase_entry(state, state.phase)
    assert status == TickStatus.SUBMITTED
    assert state.pending_jobs["FEREBUS"] == "dry-111"  # the freshly-submitted id
    from ichor.hpc.active_learning.daemon.submission_intent import load_intent
    intent = load_intent(tmp_path, "FEREBUS", 1)
    assert intent["status"] == "SUBMITTED"
    assert intent["job_id"] == "dry-111"


def test_submission_intent_exists_before_executor_calls_sbatch(tmp_path):
    d = Daemon(
        campaign_dir=tmp_path, config=CampaignConfig(),
        executor=_IntentCheckingExecutor(tmp_path),
        job_finder=lambda state, phase: None,
    )
    d.state_path().parent.mkdir(parents=True, exist_ok=True)
    state = _active_state(CampaignPhase.FEREBUS)
    status = d._on_phase_entry(state, state.phase)
    assert status == TickStatus.SUBMITTED


def test_active_submission_intent_requires_successful_adoption_check(tmp_path):
    from ichor.hpc.active_learning.daemon.submission_intent import (
        load_intent,
        write_pre_submit_intent,
    )

    d = Daemon(
        campaign_dir=tmp_path, config=CampaignConfig(),
        executor=_SubmittingExecutor(),
        job_finder=lambda state, phase: (_ for _ in ()).throw(RuntimeError("sacct down")),
    )
    d.state_path().parent.mkdir(parents=True, exist_ok=True)
    state = _active_state(CampaignPhase.FEREBUS)
    write_pre_submit_intent(
        tmp_path,
        campaign_uid=state.campaign_uid,
        phase_name="FEREBUS",
        iteration=1,
    )
    status = d._on_phase_entry(state, state.phase)
    assert status == TickStatus.HALTED
    assert read_state(d.state_path()).phase is CampaignPhase.HALTED
    intent = load_intent(tmp_path, "FEREBUS", 1)
    assert intent["status"] == "PRE_SUBMIT"


def test_active_submission_intent_terminal_job_is_adopted_for_postprocess(tmp_path):
    from ichor.hpc.active_learning.daemon.submission_intent import (
        load_intent,
        mark_submitted,
        write_pre_submit_intent,
    )
    from ichor.hpc.active_learning.submit.sacct_poll import JobObservation, JobStatus

    d = Daemon(
        campaign_dir=tmp_path,
        config=CampaignConfig(),
        executor=_ExplodingExecutor(),
        job_finder=lambda state, phase: None,
        job_liveness_checker=lambda job_id: type(
            "Lookup",
            (),
            {"active": False, "inconclusive": False, "rows": []},
        )(),
        sacct_poller=lambda job_id: [
            JobObservation(
                job_id=str(job_id),
                status=JobStatus.COMPLETED,
                exit_code=(0, 0),
                elapsed_seconds=5,
            )
        ],
    )
    d.state_path().parent.mkdir(parents=True, exist_ok=True)
    state = _active_state(CampaignPhase.FEREBUS)
    write_pre_submit_intent(
        tmp_path,
        campaign_uid=state.campaign_uid,
        phase_name="FEREBUS",
        iteration=1,
    )
    mark_submitted(tmp_path, "FEREBUS", 1, "333", expected_tasks=1)

    status = d._on_phase_entry(state, state.phase)

    assert status == TickStatus.SUBMITTED
    assert state.pending_jobs["FEREBUS"] == "333"
    intent = load_intent(tmp_path, "FEREBUS", 1)
    assert intent["status"] == "ADOPTED"
    assert intent["job_id"] == "333"


def test_active_submission_intent_with_no_accounting_rows_halts_before_resubmit(tmp_path):
    from ichor.hpc.active_learning.daemon.submission_intent import (
        load_intent,
        mark_submitted,
        write_pre_submit_intent,
    )

    d = Daemon(
        campaign_dir=tmp_path,
        config=CampaignConfig(),
        executor=_SubmittingExecutor(),
        job_finder=lambda state, phase: None,
        job_liveness_checker=lambda job_id: type(
            "Lookup",
            (),
            {"active": False, "inconclusive": False, "rows": []},
        )(),
        sacct_poller=lambda job_id: [],
    )
    d.state_path().parent.mkdir(parents=True, exist_ok=True)
    state = _active_state(CampaignPhase.FEREBUS)
    write_pre_submit_intent(
        tmp_path,
        campaign_uid=state.campaign_uid,
        phase_name="FEREBUS",
        iteration=1,
    )
    mark_submitted(tmp_path, "FEREBUS", 1, "333", expected_tasks=1)

    status = d._on_phase_entry(state, state.phase)

    assert status == TickStatus.HALTED
    intent = load_intent(tmp_path, "FEREBUS", 1)
    assert intent["status"] == "SUBMITTED"
    assert "FEREBUS" not in state.pending_jobs


def test_submission_intent_reader_rejects_phase_iteration_mismatch(tmp_path):
    from ichor.hpc.active_learning.daemon.submission_intent import (
        intent_path,
        load_intent,
        write_pre_submit_intent,
    )
    import json

    state = fresh_campaign_state()
    write_pre_submit_intent(
        tmp_path,
        campaign_uid=state.campaign_uid,
        phase_name="FEREBUS",
        iteration=0,
    )
    path = intent_path(tmp_path, "FEREBUS", 0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phase"] = "GAUSSIAN"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="phase mismatch"):
        load_intent(tmp_path, "FEREBUS", 0)

    payload["phase"] = "FEREBUS"
    payload["iteration"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="iteration mismatch"):
        load_intent(tmp_path, "FEREBUS", 0)


def test_submission_intent_records_job_id_even_if_state_persist_fails(monkeypatch, tmp_path):
    d = Daemon(
        campaign_dir=tmp_path, config=CampaignConfig(),
        executor=_SubmittingExecutor(),
        job_finder=lambda state, phase: None,
    )
    d.state_path().parent.mkdir(parents=True, exist_ok=True)
    state = _active_state(CampaignPhase.FEREBUS)

    def failing_persist(_state):
        raise OSError("simulated persist failure")

    monkeypatch.setattr(d, "_persist", failing_persist)
    with pytest.raises(OSError, match="simulated persist failure"):
        d._on_phase_entry(state, state.phase)

    from ichor.hpc.active_learning.daemon.submission_intent import load_intent
    intent = load_intent(tmp_path, "FEREBUS", 1)
    assert intent["status"] == "SUBMITTED"
    assert intent["job_id"] == "dry-111"


def test_daemon_does_not_adopt_for_inline_phase(tmp_path):
    # inline phases (SEED_SELECT etc.) have no sbatch job -- the adopt check must be skipped, the
    # finder never even consulted.
    seen = {"n": 0}

    def finder(state, phase):
        seen["n"] += 1
        return "999"

    d = Daemon(
        campaign_dir=tmp_path, config=CampaignConfig(),
        executor=_InlineExecutor(), job_finder=finder,
    )
    d.state_path().parent.mkdir(parents=True, exist_ok=True)  # the daemon makes this at campaign start
    state = _active_state(CampaignPhase.SEED_SELECT)
    d._on_phase_entry(state, state.phase)
    assert seen["n"] == 0
    assert state.pending_jobs.get("SEED_SELECT") != "999"


# --- find_running_job_by_name --------------------------------------------------------------


def _stub_sacct(stdout, returncode=0):
    class _CP:
        pass
    cp = _CP()
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = ""
    return lambda *a, **k: cp


def test_find_running_job_returns_base_id_of_nonterminal():
    from ichor.hpc.active_learning.submit.sacct_poll import find_running_job_by_name
    runner = _stub_sacct("555_0|RUNNING\n555_1|PENDING\n")
    assert find_running_job_by_name("camp-FEREBUS-1", sacct_runner=runner) == "555"


def test_find_running_job_ignores_terminal_states():
    from ichor.hpc.active_learning.submit.sacct_poll import find_running_job_by_name
    runner = _stub_sacct("555_0|COMPLETED\n555_1|FAILED\n")
    assert find_running_job_by_name("x", sacct_runner=runner) is None


def test_find_running_job_does_not_adopt_unknown_state():
    from ichor.hpc.active_learning.submit.sacct_poll import find_running_job_by_name
    runner = _stub_sacct("555_0|WEIRD_NEW_STATE\n")
    assert find_running_job_by_name("x", sacct_runner=runner) is None


def test_find_running_job_handles_sacct_error():
    from ichor.hpc.active_learning.submit.sacct_poll import find_running_job_by_name
    runner = _stub_sacct("", returncode=1)
    assert find_running_job_by_name("x", sacct_runner=runner) is None


def test_live_job_finder_checks_new_and_legacy_uid_prefixes():
    from types import SimpleNamespace

    from ichor.hpc.active_learning.daemon.live_executor import make_live_job_finder

    seen = []

    def runner(cmd, **kwargs):
        name = cmd[cmd.index("--name") + 1]
        seen.append(name)
        if name.startswith("abcdefghijkl-"):
            return _stub_sacct("")()
        return _stub_sacct("555_0|RUNNING\n")()

    finder = make_live_job_finder(sacct_runner=runner)
    lookup = finder(
        SimpleNamespace(campaign_uid="abcdefghijklmnop", iteration=3),
        CampaignPhase.FEREBUS,
    )

    assert lookup.job_id == "555"
    assert seen == ["abcdefghijkl-FEREBUS-3", "abcdefgh-FEREBUS-3"]


def test_live_job_finder_uses_squeue_fallback_when_sacct_has_no_rows():
    from types import SimpleNamespace

    from ichor.hpc.active_learning.daemon.live_executor import make_live_job_finder

    sacct_seen = []
    squeue_seen = []

    def sacct_runner(cmd, **kwargs):
        sacct_seen.append(cmd[cmd.index("--name") + 1])
        return _stub_sacct("")()

    def squeue_runner(cmd, **kwargs):
        squeue_seen.append(cmd[cmd.index("--name") + 1])
        return _stub_sacct("999_[0-4%2]|PENDING|abcdefghijkl-GAUSSIAN-2\n")()

    finder = make_live_job_finder(
        sacct_runner=sacct_runner,
        squeue_runner=squeue_runner,
    )
    lookup = finder(
        SimpleNamespace(campaign_uid="abcdefghijklmnop", iteration=2),
        CampaignPhase.GAUSSIAN,
    )

    assert lookup.job_id == "999"
    assert sacct_seen == ["abcdefghijkl-GAUSSIAN-2"]
    assert squeue_seen == ["abcdefghijkl-GAUSSIAN-2"]


# --- A4 / A55: manifest robustness ---------------------------------------------------------


def test_read_manifest_wraps_corrupt_json(tmp_path):
    from ichor.hpc.active_learning.versioning.manifest import (
        MANIFEST_FILENAME, ManifestMismatchError, read_manifest,
    )
    (tmp_path / MANIFEST_FILENAME).write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(ManifestMismatchError):
        read_manifest(tmp_path)


def test_manifest_skips_transient_files(tmp_path):
    from ichor.hpc.active_learning.versioning.manifest import compute_directory_manifest
    (tmp_path / "real.txt").write_text("data", encoding="utf-8")
    (tmp_path / ".nfs01234567").write_text("nfs silly-rename", encoding="utf-8")
    (tmp_path / "buf.swp").write_text("editor swap", encoding="utf-8")
    (tmp_path / "stale.lock").write_text("lock", encoding="utf-8")
    m = compute_directory_manifest(tmp_path)
    assert "real.txt" in m
    assert ".nfs01234567" not in m and "buf.swp" not in m and "stale.lock" not in m


# --- A54: current_version must not return a version whose dir is gone ----------------------


def test_current_version_none_for_dangling_pointer(tmp_path):
    from ichor.hpc.active_learning.versioning.versioned_directory import VersionedDirectory
    v = VersionedDirectory(tmp_path)
    # pointer (windows-fallback form) names a version whose directory does not exist
    (tmp_path / ".current.pointer").write_text("iteration-000007\n", encoding="utf-8")
    assert v.current_version() is None


# --- A31: reference_scales validation in from_dict -----------------------------------------


def test_from_dict_rejects_malformed_reference_scales():
    base = CampaignState().to_dict()
    for bad in (["not", "a", "dict"], "a string", {"energy": "notnumber"}, {"energy": True}):
        payload = dict(base)
        payload["reference_scales"] = bad
        with pytest.raises(StateSchemaError):
            CampaignState.from_dict(payload)


def test_from_dict_accepts_valid_reference_scales():
    base = CampaignState().to_dict()
    complete = {
        "energy": 1.0,
        "force": 2.0,
        "omega": 3.0,
        "anh": 4.0,
        "anh_std": 5.0,
    }
    for good in (None, complete):
        payload = dict(base)
        payload["reference_scales"] = good
        CampaignState.from_dict(payload)  # must not raise


def test_from_dict_rejects_nonfinite_state_values():
    base = CampaignState().to_dict()
    for key, value in (
        ("last_acquisition_alpha0", float("nan")),
        ("last_acquisition_alpha0", float("inf")),
        ("reference_scales", {"energy": float("nan")}),
        ("alpha_history", [0.1, float("inf")]),
    ):
        payload = dict(base)
        payload[key] = value
        with pytest.raises(StateSchemaError):
            CampaignState.from_dict(payload)


def test_write_state_rejects_nonfinite_values(tmp_path):
    state = CampaignState()
    state.last_acquisition_alpha0 = float("nan")
    with pytest.raises(StateSchemaError):
        write_state(tmp_path / "state.json", state)


def test_daemon_persist_preserves_external_shutdown_request(tmp_path):
    d = Daemon(campaign_dir=tmp_path, config=CampaignConfig(), executor=_InlineExecutor())
    d.state_path().parent.mkdir(parents=True, exist_ok=True)
    disk_state = fresh_campaign_state()
    write_state(d.state_path(), disk_state)

    stale_state = read_state(d.state_path())
    disk_state.shutdown_requested = True
    write_state(d.state_path(), disk_state)

    stale_state.phase = CampaignPhase.PHASE_A_DIVERSITY
    d._persist(stale_state)
    assert read_state(d.state_path()).shutdown_requested is True


def test_submission_attempt_identity_changes_across_replacement_rounds(tmp_path):
    from ichor.hpc.active_learning.daemon.submission_intent import (
        INTENT_HISTORY_DIR_NAME,
        intent_dir,
        load_intent,
        write_pre_submit_intent,
    )

    first = write_pre_submit_intent(
        tmp_path,
        campaign_uid="campaign-uid",
        phase_name="REPLACEMENT_GAUSSIAN",
        iteration=3,
        replacement_round=1,
    )
    from ichor.hpc.active_learning.daemon.submission_intent import mark_failed

    mark_failed(
        tmp_path,
        "REPLACEMENT_GAUSSIAN",
        3,
        "first replacement attempt completed unsuccessfully",
    )
    second = write_pre_submit_intent(
        tmp_path,
        campaign_uid="campaign-uid",
        phase_name="REPLACEMENT_GAUSSIAN",
        iteration=3,
        replacement_round=2,
    )

    assert first["attempt_sequence"] == 1
    assert second["attempt_sequence"] == 2
    assert first["attempt_id"] != second["attempt_id"]
    assert first["submission_identity"] != second["submission_identity"]
    assert "-r0001-" in first["expected_job_name"]
    assert "-r0002-" in second["expected_job_name"]
    assert load_intent(tmp_path, "REPLACEMENT_GAUSSIAN", 3) == second
    history = intent_dir(tmp_path) / INTENT_HISTORY_DIR_NAME
    archived = list(history.glob("REPLACEMENT_GAUSSIAN-000003-*.json"))
    assert len(archived) == 1
