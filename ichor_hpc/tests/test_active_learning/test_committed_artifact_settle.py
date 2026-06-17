import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import Daemon, TickStatus
from ichor.hpc.active_learning.daemon.dry_run_sacct import DryRunSacctPoller
from ichor.hpc.active_learning.daemon.journal import iter_events
from ichor.hpc.active_learning.daemon.phase_executor import MockPhaseExecutor
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state,
    read_state,
    write_state,
)


class _StrictExecutor(MockPhaseExecutor):
    strict_committed_artifact_verification = True


def _daemon(tmp_path, *, attempts=2, settle_seconds=7, sleep_calls=None):
    cfg = CampaignConfig(max_iterations=2)
    cfg.runtime.postprocess_settle_attempts = attempts
    cfg.runtime.postprocess_settle_seconds = settle_seconds
    sleep_calls = sleep_calls if sleep_calls is not None else []
    return Daemon(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        executor=_StrictExecutor(treat_as_sbatch=set()),
        sacct_poller=DryRunSacctPoller().poll,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )


def _write_seed_select_state(daemon):
    daemon.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=2)
    state.phase = CampaignPhase.SEED_SELECT
    state.training_set_version = 0
    state.models_version = 0
    write_state(daemon.state_path(), state)
    return state


def test_committed_artifact_verification_retries_transient_missing_then_succeeds(
    tmp_path,
    monkeypatch,
):
    calls = {"n": 0}
    sleep_calls = []
    daemon = _daemon(tmp_path, attempts=3, settle_seconds=5, sleep_calls=sleep_calls)
    _write_seed_select_state(daemon)

    def fake_verify(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError("missing committed model")

    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.artifact_contracts.verify_state_referenced_artifacts",
        fake_verify,
    )

    status = daemon.tick()

    assert status == TickStatus.ADVANCED
    assert calls["n"] == 2
    assert sleep_calls == [5.0]
    events = list(iter_events(daemon.journal_path()))
    retries = [e for e in events if e.get("event") == "committed_artifact_settle_retry"]
    assert len(retries) == 1
    assert retries[0]["phase"] == "SEED_SELECT"


def test_committed_artifact_verification_persistent_missing_halts_after_retries(
    tmp_path,
    monkeypatch,
):
    calls = {"n": 0}
    sleep_calls = []
    daemon = _daemon(tmp_path, attempts=2, settle_seconds=3, sleep_calls=sleep_calls)
    _write_seed_select_state(daemon)

    def fake_verify(*args, **kwargs):
        calls["n"] += 1
        raise FileNotFoundError("missing committed model")

    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.artifact_contracts.verify_state_referenced_artifacts",
        fake_verify,
    )

    status = daemon.tick()

    assert status == TickStatus.HALTED
    assert calls["n"] == 2
    assert sleep_calls == [3.0]
    state = read_state(daemon.state_path())
    assert state.phase is CampaignPhase.HALTED
    events = list(iter_events(daemon.journal_path()))
    retries = [e for e in events if e.get("event") == "committed_artifact_settle_retry"]
    assert len(retries) == 1
    halt = [e for e in events if e.get("event") == "halt"][-1]
    assert "committed_artifact_contract_invalid" in halt["reason"]


def test_committed_artifact_verification_non_settle_error_halts_immediately(
    tmp_path,
    monkeypatch,
):
    calls = {"n": 0}
    sleep_calls = []
    daemon = _daemon(tmp_path, attempts=3, settle_seconds=9, sleep_calls=sleep_calls)
    _write_seed_select_state(daemon)

    def fake_verify(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("state training/model version skew")

    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.artifact_contracts.verify_state_referenced_artifacts",
        fake_verify,
    )

    status = daemon.tick()

    assert status == TickStatus.HALTED
    assert calls["n"] == 1
    assert sleep_calls == []
    events = list(iter_events(daemon.journal_path()))
    assert not [e for e in events if e.get("event") == "committed_artifact_settle_retry"]
    halt = [e for e in events if e.get("event") == "halt"][-1]
    assert "state training/model version skew" in halt["reason"]
