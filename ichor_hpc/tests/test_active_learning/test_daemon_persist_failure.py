"""M15 F2 tests: persist-before-journal ordering + idempotent _inline_append
under simulated _persist failures.

The contract enforced here:
  * If _persist raises mid-tick, the journal MUST NOT record a transition
    that hasn't actually landed in state.json.
  * A re-run after such a failure MUST NOT duplicate-commit the same
    training-set version.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
from ichor.hpc.active_learning.daemon.state import CampaignPhase
from ichor.hpc.active_learning.versioning.training_set import TrainingSetVersioning


def _make_executor(tmp_path):
    cfg = CampaignConfig(max_iterations=3)
    cfg.batch_sizing.floor = 2
    cfg.seed_selection.n_seeds_per_iteration = 2
    return DryRunPhaseExecutor(campaign_dir=tmp_path / "campaign", config=cfg)


def test_inline_append_idempotent_when_next_version_already_committed(tmp_path):
    """If state.training_set_version + 1 is already a committed iteration
    (e.g. we crashed between commit and _persist on the previous run), the
    re-run of APPEND must be a no-op rather than producing a duplicate
    iteration directory."""
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=0, campaign_uid="uid", training_set_version=0)
    # Bootstrap iteration 0 via INITIAL_FEREBUS
    ex.postprocess(state, CampaignPhase.INITIAL_FEREBUS, observations=[])
    # First APPEND: commits version 1.
    ex.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    result1 = ex.submit_or_run(state, CampaignPhase.APPEND)
    assert result1.state_updates["training_set_version"] == 1

    v = TrainingSetVersioning(tmp_path / "campaign" / "5_TRAINING")
    assert sorted(v.list_committed_versions()) == [0, 1]

    # Simulate "crash after commit, before _persist updated state". The
    # state still reports training_set_version=0; the same APPEND call
    # must NOT produce iteration-0002 because version 1 already exists.
    result2 = ex.submit_or_run(state, CampaignPhase.APPEND)
    assert result2.state_updates["training_set_version"] == 1
    assert sorted(v.list_committed_versions()) == [0, 1]
    assert v.current_version() == 1


def test_inline_append_idempotent_skip_journals_clearly(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=0, campaign_uid="uid", training_set_version=0)
    ex.postprocess(state, CampaignPhase.INITIAL_FEREBUS, observations=[])
    ex.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    ex.submit_or_run(state, CampaignPhase.APPEND)
    # Second call should land an idempotent_skip=True in the journal.
    ex.submit_or_run(state, CampaignPhase.APPEND)
    journal_path = tmp_path / "campaign" / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    commits = [e for e in events if e.get("event") == "training_set_committed"]
    assert any(e.get("idempotent_skip") is True for e in commits)


def test_inline_append_idempotent_skip_repairs_current_pointer(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=0, campaign_uid="uid", training_set_version=0)
    ex.postprocess(state, CampaignPhase.INITIAL_FEREBUS, observations=[])
    ex.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    ex.submit_or_run(state, CampaignPhase.APPEND)

    v = TrainingSetVersioning(tmp_path / "campaign" / "5_TRAINING")
    v.update_current(0)
    assert v.current_version() == 0

    ex.submit_or_run(state, CampaignPhase.APPEND)
    assert v.current_version() == 1


def test_initial_training_idempotent_skip_repairs_current_pointer(tmp_path):
    from ichor.hpc.active_learning.daemon.input_staging import commit_initial_training_set

    campaign = tmp_path / "campaign"
    v = TrainingSetVersioning(campaign / "5_TRAINING")
    staging = v.stage(None, 0)
    (staging / "marker.txt").write_text("initial", encoding="utf-8")
    v.commit(0)
    assert v.current_version() is None

    assert commit_initial_training_set(campaign) is False
    assert v.current_version() == 0


def test_dry_ferebus_idempotent_skip_repairs_model_current_pointer(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=1, campaign_uid="uid", models_version=0)
    ex.postprocess(SimpleNamespace(iteration=0, campaign_uid="uid"), CampaignPhase.INITIAL_FEREBUS, observations=[])

    first = ex.postprocess(state, CampaignPhase.FEREBUS, observations=[])
    assert first.state_updates["models_version"] == 1
    v = TrainingSetVersioning(tmp_path / "campaign" / "6_TRAINED_MODELS")
    v.update_current(0)
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
