"""M15 F2 tests: persist-before-journal ordering + idempotent _inline_append
under simulated _persist failures.

The contract enforced here:
  * If _persist raises mid-tick, the journal MUST NOT record a transition
    that hasn't actually landed in state.json.
  * A re-run after such a failure MUST NOT duplicate-commit the same
    training-set version.
"""
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
from ichor.hpc.active_learning.daemon.phase_executor import BackendSubmissionError
from ichor.hpc.active_learning.daemon.state import CampaignPhase
from ichor.hpc.active_learning.versioning.versioned_directory import VersionedDirectory


def _make_executor(tmp_path):
    cfg = CampaignConfig(max_iterations=3)
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.seed_selection.n_seeds_per_iteration = 2
    return DryRunPhaseExecutor(campaign_dir=tmp_path / "campaign", config=cfg)


def _prepare_bootstrap(ex, state):
    ex.postprocess(state, CampaignPhase.PHASE_A_POLUS, observations=[])
    ex.postprocess(state, CampaignPhase.INITIAL_GAUSSIAN, observations=[])
    ex.postprocess(state, CampaignPhase.INITIAL_AIMALL, observations=[])
    result = ex.submit_or_run(state, CampaignPhase.INITIAL_ALLOCATION_CHECK)
    assert result.next_phase_override == CampaignPhase.INITIAL_FEREBUS.value
    ex.postprocess(state, CampaignPhase.INITIAL_FEREBUS, observations=[])


def _prepare_active_quantum_allocation(ex, state):
    ex.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    ex.postprocess(state, CampaignPhase.PHASE_B_POLUS, observations=[])
    ex.submit_or_run(state, CampaignPhase.SPLIT)
    ex.postprocess(state, CampaignPhase.GAUSSIAN, observations=[])
    ex.postprocess(state, CampaignPhase.AIMALL, observations=[])
    result = ex.submit_or_run(state, CampaignPhase.ALLOCATION_CHECK)
    assert result.next_phase_override == CampaignPhase.APPEND.value


def test_inline_append_idempotent_when_next_version_already_committed(tmp_path):
    """If state.reference_data_version + 1 is already a committed iteration
    (e.g. we crashed between commit and _persist on the previous run), the
    re-run of APPEND must be a no-op rather than producing a duplicate
    iteration directory."""
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=0, campaign_uid="uid", reference_data_version=0)
    _prepare_bootstrap(ex, state)
    # First APPEND: commits version 1.
    _prepare_active_quantum_allocation(ex, state)
    result1 = ex.submit_or_run(state, CampaignPhase.APPEND)
    assert result1.state_updates["reference_data_version"] == 1

    v = VersionedDirectory(tmp_path / "campaign" / "QM_REFERENCE_DATA")
    assert sorted(v.list_committed_versions()) == [0, 1]

    # Simulate "crash after commit, before _persist updated state". The
    # state still reports reference_data_version=0; the same APPEND call
    # must NOT produce iteration-0002 because version 1 already exists.
    result2 = ex.submit_or_run(state, CampaignPhase.APPEND)
    assert result2.state_updates["reference_data_version"] == 1
    assert sorted(v.list_committed_versions()) == [0, 1]
    assert v.current_version() == 1


def test_inline_append_idempotent_skip_journals_clearly(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=0, campaign_uid="uid", reference_data_version=0)
    _prepare_bootstrap(ex, state)
    _prepare_active_quantum_allocation(ex, state)
    ex.submit_or_run(state, CampaignPhase.APPEND)
    # Second call should land an idempotent_skip=True in the journal.
    ex.submit_or_run(state, CampaignPhase.APPEND)
    journal_path = tmp_path / "campaign" / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    commits = [e for e in events if e.get("event") == "reference_data_committed"]
    assert any(e.get("idempotent_skip") is True for e in commits)


def test_inline_append_idempotent_skip_repairs_current_pointer(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=0, campaign_uid="uid", reference_data_version=0)
    _prepare_bootstrap(ex, state)
    _prepare_active_quantum_allocation(ex, state)
    ex.submit_or_run(state, CampaignPhase.APPEND)

    v = VersionedDirectory(tmp_path / "campaign" / "QM_REFERENCE_DATA")
    v.update_current(0)
    assert v.current_version() == 0

    ex.submit_or_run(state, CampaignPhase.APPEND)
    assert v.current_version() == 1


def test_inline_append_idempotent_skip_rejects_mismatched_allocation(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=0, campaign_uid="uid", reference_data_version=0)
    _prepare_bootstrap(ex, state)
    _prepare_active_quantum_allocation(ex, state)
    ex.submit_or_run(state, CampaignPhase.APPEND)

    snapshot = (
        tmp_path
        / "campaign"
        / "QM_REFERENCE_DATA"
        / "iteration-000001"
        / "POINT_ALLOCATION.version-000001.json"
    )
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["iteration"] = 999
    os.chmod(snapshot, stat.S_IMODE(snapshot.stat().st_mode) | stat.S_IWUSR)
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        BackendSubmissionError,
        match="reference-data APPEND transaction failed",
    ):
        ex.submit_or_run(state, CampaignPhase.APPEND)


def test_initial_training_idempotent_skip_repairs_current_pointer(tmp_path):
    from ichor.hpc.active_learning.daemon.input_staging import commit_initial_reference_data

    ex = _make_executor(tmp_path)
    campaign = ex.campaign_dir
    state = SimpleNamespace(iteration=0, campaign_uid="uid", reference_data_version=-1)
    ex.postprocess(state, CampaignPhase.PHASE_A_POLUS, observations=[])
    ex.postprocess(state, CampaignPhase.INITIAL_GAUSSIAN, observations=[])
    ex.postprocess(state, CampaignPhase.INITIAL_AIMALL, observations=[])
    v = VersionedDirectory(campaign / "QM_REFERENCE_DATA")
    assert commit_initial_reference_data(campaign) is True
    for pointer in (v.current_link_path(), v._pointer_path()):
        if pointer.is_symlink() or pointer.is_file():
            pointer.unlink()
    assert v.current_version() is None

    assert commit_initial_reference_data(campaign) is False
    assert v.current_version() == 0


def test_dry_ferebus_idempotent_skip_repairs_model_current_pointer(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(
        iteration=1,
        campaign_uid="uid",
        reference_data_version=0,
        models_version=0,
    )
    _prepare_bootstrap(
        ex,
        SimpleNamespace(iteration=0, campaign_uid="uid", reference_data_version=-1),
    )
    _prepare_active_quantum_allocation(ex, state)
    appended = ex.submit_or_run(state, CampaignPhase.APPEND)
    state.reference_data_version = int(appended.state_updates["reference_data_version"])

    first = ex.postprocess(state, CampaignPhase.FEREBUS, observations=[])
    assert first.state_updates["models_version"] == 1
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelVersioning,
    )

    v = TrainedModelVersioning(tmp_path / "campaign" / "TRAINED_MODELS")
    for pointer in (v.current_link_path(), v._pointer_path()):
        if pointer.is_symlink() or pointer.is_file():
            pointer.unlink()
    v._pointer_path().write_text("iteration-000000\n", encoding="utf-8")
    assert v.current_version() == 0

    second = ex.postprocess(state, CampaignPhase.FEREBUS, observations=[])
    assert second.state_updates["models_version"] == 1
    assert sorted(v.list_committed_versions()) == [0, 1]
    assert v.current_version() == 1


def test_persist_before_journal_in_advance(monkeypatch, tmp_path):
    """When _persist raises mid-_advance, the phase_transition journal
    entry MUST NOT be written -- otherwise restart would re-run a phase
    state.json never confirms."""
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.phase_executor import MockPhaseExecutor
    from ichor.hpc.active_learning.daemon.state import (
        CampaignState,
        CampaignPhase as CP,
        fresh_campaign_state,
        write_state,
    )

    campaign = tmp_path / "campaign"
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    state0 = fresh_campaign_state(max_iterations=10)
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state0)

    daemon = Daemon(
        campaign_dir=campaign,
        config=CampaignConfig(),
        executor=MockPhaseExecutor(),
        sleep_fn=lambda s: None,
    )

    # Make EVERY _persist call raise so the first inline _advance fails.
    def failing_persist(state):
        raise OSError("simulated disk-full")

    monkeypatch.setattr(daemon, "_persist", failing_persist)

    # Capture journal events emitted during the tick.
    emitted = []
    real_journal = daemon._journal

    def recording_journal(event_type, **payload):
        emitted.append((event_type, payload))
        return real_journal(event_type, **payload)

    monkeypatch.setattr(daemon, "_journal", recording_journal)

    # Drive one tick; the failure happens during _advance after the inline
    # phase (INIT -> PHASE_A_POLUS).
    with pytest.raises(OSError, match="simulated disk-full"):
        daemon.tick()

    # The persist-before-journal ordering must have prevented the
    # phase_transition event from being emitted when _persist failed.
    transition_events = [e for e in emitted if e[0] == "phase_transition"]
    assert len(transition_events) == 0, (
        "phase_transition event was journaled despite _persist failure: "
        + str(transition_events)
    )
