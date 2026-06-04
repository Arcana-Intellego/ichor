"""M15 F1 tests: LiveBackendsPhaseExecutor refuses to silently fall through
to dry-run stub postprocess for SBATCH phases."""
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.live_executor import (
    LIVE_POSTPROCESS_IMPLEMENTED,
    LiveBackendsPhaseExecutor,
)
from ichor.hpc.active_learning.daemon.phase_executor import (
    INLINE_PHASES,
    SBATCH_PHASES,
)
from ichor.hpc.active_learning.daemon.state import CampaignPhase


def _make_executor(tmp_path):
    cfg = CampaignConfig()
    return LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )


def test_live_postprocess_implemented_covers_quantum_phases():
    """M16 Day 2: the four quantum phases (Gaussian + AIMAll, INITIAL +
    iter) are registered. FEREBUS / ARIADNE_ARRAY / POLUS land on Day 3."""
    assert {"INITIAL_GAUSSIAN", "GAUSSIAN", "INITIAL_AIMALL", "AIMALL"}.issubset(
        LIVE_POSTPROCESS_IMPLEMENTED
    )


def test_postprocess_refuses_for_every_unimplemented_sbatch_phase(tmp_path):
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=0, campaign_uid="uid")
    for phase_name in sorted(SBATCH_PHASES):
        if phase_name in LIVE_POSTPROCESS_IMPLEMENTED:
            continue  # implemented phases land in their own per-phase tests
        phase = CampaignPhase(phase_name)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            ex.postprocess(state, phase, observations=[])


def test_refusal_emits_journal_event(tmp_path, monkeypatch):
    import json
    from ichor.hpc.active_learning.daemon import live_executor as le
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=3, campaign_uid="uid")
    # M16 Day 3: all nine SBATCH phases are now implemented (see
    # LIVE_POSTPROCESS_IMPLEMENTED). To still exercise the F1 refusal
    # path -- which protects future-added phases until their parser
    # is registered -- monkeypatch the implemented set to remove FEREBUS
    # for the duration of this test, then verify refusal + journal event.
    monkeypatch.setattr(
        le, "LIVE_POSTPROCESS_IMPLEMENTED",
        frozenset(le.LIVE_POSTPROCESS_IMPLEMENTED - {"FEREBUS"}),
    )
    # Also remove the FEREBUS handler from the dict so postprocess does
    # not dispatch to it directly (the F1 guard fires when not handled).
    original_handlers = ex._live_postprocess_handlers
    monkeypatch.setattr(
        ex, "_live_postprocess_handlers",
        lambda: {k: v for k, v in original_handlers().items() if k != "FEREBUS"},
    )
    phase = CampaignPhase("FEREBUS")
    with pytest.raises(NotImplementedError):
        ex.postprocess(state, phase, observations=[])
    journal_path = (
        tmp_path / "campaign" / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    )
    assert journal_path.is_file()
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    refused = [e for e in events if e.get("event") == "live_postprocess_refused"]
    assert refused, "live_postprocess_refused event missing"
    assert refused[-1]["phase"] == "FEREBUS"
    assert refused[-1]["iteration"] == 3


def test_inline_phases_still_delegate_to_dry_run(tmp_path):
    """Inline phases (SEED_SELECT, SPLIT, APPEND, STOP_CHECK) must NOT
    be refused -- they have no backend artefacts to mangle."""
    ex = _make_executor(tmp_path)
    state = SimpleNamespace(iteration=0, campaign_uid="uid")
    # Inline-phase names that we expect to pass through to super().postprocess
    # without raising. The dry-run executor returns a PhaseResult or {} from
    # these. We just need them to NOT raise NotImplementedError.
    for phase_name in ("SEED_SELECT", "SPLIT", "APPEND", "STOP_CHECK"):
        phase = CampaignPhase(phase_name)
        # super().postprocess returns a PhaseResult; we just want no refusal
        ex.postprocess(state, phase, observations=[])


def test_registering_a_phase_disables_refusal(tmp_path, monkeypatch):
    """When a phase is registered in LIVE_POSTPROCESS_IMPLEMENTED AND a
    handler is added, refusal must NOT fire."""
    from ichor.hpc.active_learning.daemon import live_executor as le

    handled = []

    def fake_handler(state, phase, observations):
        # M16-P4: handlers now take (state, phase, observations) so the
        # shared quantum parser can dispatch by phase.
        handled.append(state.iteration)
        from ichor.hpc.active_learning.daemon.phase_executor import PhaseResult
        return PhaseResult(is_complete=True)

    monkeypatch.setattr(le, "LIVE_POSTPROCESS_IMPLEMENTED", frozenset({"GAUSSIAN"}))
    ex = _make_executor(tmp_path)
    monkeypatch.setattr(
        ex, "_live_postprocess_handlers", lambda: {"GAUSSIAN": fake_handler},
    )
    state = SimpleNamespace(iteration=7, campaign_uid="uid")
    ex.postprocess(state, CampaignPhase("GAUSSIAN"), observations=[])
    assert handled == [7]


def test_implementing_set_must_be_subset_of_sbatch_phases():
    """Sanity check on the guard: any phase in LIVE_POSTPROCESS_IMPLEMENTED
    must also be in SBATCH_PHASES (otherwise the guard is shaped wrong)."""
    for phase_name in LIVE_POSTPROCESS_IMPLEMENTED:
        assert phase_name in SBATCH_PHASES, (
            f"{phase_name!r} in LIVE_POSTPROCESS_IMPLEMENTED but not in SBATCH_PHASES"
        )
