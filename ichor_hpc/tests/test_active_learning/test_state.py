"""Tests for ichor.hpc.active_learning.daemon.state."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    CampaignState,
    SCHEMA_VERSION,
    StateSchemaError,
    atomic_write_json,
    atomic_write_text,
    fresh_campaign_state,
    read_state,
    write_state,
)


def test_fresh_state_has_defaults():
    s = fresh_campaign_state(max_iterations=42)
    assert s.iteration == 0
    assert s.phase is CampaignPhase.INIT
    assert s.max_iterations == 42
    assert s.schema_version == SCHEMA_VERSION
    assert s.campaign_uid
    assert s.campaign_started_iso


def test_state_roundtrip(tmp_path):
    s = fresh_campaign_state(max_iterations=10)
    s.iteration = 7
    s.phase = CampaignPhase.ARIADNE_ARRAY
    s.pending_jobs = {"ARIADNE_ARRAY": "12345", "FEREBUS": None}
    s.training_set_version = 7
    s.last_acquisition_alpha0 = 0.832
    s.stop_streak = 2
    target = tmp_path / "state.json"
    write_state(target, s)
    loaded = read_state(target)
    assert loaded.iteration == 7
    assert loaded.phase is CampaignPhase.ARIADNE_ARRAY
    assert loaded.pending_jobs == {"ARIADNE_ARRAY": "12345", "FEREBUS": None}
    assert loaded.last_acquisition_alpha0 == pytest.approx(0.832)
    assert loaded.stop_streak == 2


def test_atomic_write_text_does_not_leave_tmp_on_success(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_text(target, '{"hello": "world"}\n')
    # No stray *.tmp files left next to target.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
    assert target.read_text() == '{"hello": "world"}\n'


def test_atomic_write_json_preserves_payload(tmp_path):
    target = tmp_path / "out.json"
    payload = {"a": 1, "b": [2, 3], "c": {"nested": True}}
    atomic_write_json(target, payload)
    assert json.loads(target.read_text()) == payload


def test_atomic_write_text_simulated_kill_during_write_keeps_old_content(
    tmp_path, monkeypatch
):
    """Simulate kill -9 during the tempfile write (before rename). The
    target should be untouched (either pre-existing content or absent)."""
    target = tmp_path / "state.json"
    target.write_text('{"version": 1}\n')

    real_replace = pytest.MonkeyPatch().setattr  # type: ignore[unused-variable]

    import os
    def boom_replace(src, dst):
        raise RuntimeError("simulated kill before rename completes")
    monkeypatch.setattr(os, "replace", boom_replace)

    with pytest.raises(RuntimeError):
        atomic_write_text(target, '{"version": 2}\n')

    # Target still has the old contents (rename did not happen).
    assert target.read_text() == '{"version": 1}\n'


def test_atomic_write_text_raises_when_parent_missing(tmp_path):
    bad = tmp_path / "missing_dir" / "out.json"
    with pytest.raises(FileNotFoundError):
        atomic_write_text(bad, "data\n")


def test_read_state_rejects_wrong_schema_version(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"schema_version": 999, "phase": "INIT"}))
    with pytest.raises(StateSchemaError, match="schema_version"):
        read_state(p)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema_version", None, "schema_version"),
        ("reference_scales_iteration", None, "reference_scales_iteration"),
        ("last_n_anti_overlap_flagged", "abc", "last_n_anti_overlap_flagged"),
    ],
)
def test_read_state_wraps_malformed_integer_fields(field, value, match):
    payload = fresh_campaign_state().to_dict()
    payload[field] = value
    with pytest.raises(StateSchemaError, match=match):
        CampaignState.from_dict(payload)


def test_read_state_wraps_malformed_sacct_empty_streak_value():
    payload = fresh_campaign_state().to_dict()
    payload["sacct_empty_streak"] = {"123": "abc"}
    with pytest.raises(StateSchemaError, match="sacct_empty_streak"):
        CampaignState.from_dict(payload)


def test_read_state_rejects_unknown_phase(tmp_path):
    p = tmp_path / "state.json"
    bad = fresh_campaign_state().to_dict()
    bad["phase"] = "INVALID_PHASE"
    p.write_text(json.dumps(bad))
    with pytest.raises(StateSchemaError, match="phase"):
        read_state(p)


def test_read_state_rejects_missing_required_field(tmp_path):
    p = tmp_path / "state.json"
    bad = fresh_campaign_state().to_dict()
    bad.pop("campaign_uid")
    p.write_text(json.dumps(bad))
    with pytest.raises(StateSchemaError):
        read_state(p)


def test_read_state_rejects_non_string_pending_job_value(tmp_path):
    p = tmp_path / "state.json"
    bad = fresh_campaign_state().to_dict()
    bad["pending_jobs"] = {"X": 12345}    # must be str or null
    p.write_text(json.dumps(bad))
    with pytest.raises(StateSchemaError):
        read_state(p)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("iteration", -1, "iteration"),
        ("max_iterations", 0, "max_iterations"),
        ("training_set_version", -2, "training_set_version"),
        ("validation_set_version", -2, "validation_set_version"),
        ("models_version", -2, "models_version"),
        ("stop_streak", -1, "stop_streak"),
        ("reference_scales_iteration", -2, "reference_scales_iteration"),
        ("last_n_anti_overlap_flagged", -1, "last_n_anti_overlap_flagged"),
    ],
)
def test_read_state_rejects_invalid_integer_ranges(field, value, match):
    payload = fresh_campaign_state().to_dict()
    payload[field] = value
    with pytest.raises(StateSchemaError, match=match):
        CampaignState.from_dict(payload)


def test_read_state_rejects_negative_sacct_empty_streak_value():
    payload = fresh_campaign_state().to_dict()
    payload["sacct_empty_streak"] = {"123": -1}
    with pytest.raises(StateSchemaError, match="sacct_empty_streak"):
        CampaignState.from_dict(payload)


def test_read_state_rejects_unknown_pending_job_phase():
    payload = fresh_campaign_state().to_dict()
    payload["pending_jobs"] = {"NOT_A_PHASE": "123"}
    with pytest.raises(StateSchemaError, match="pending_jobs key"):
        CampaignState.from_dict(payload)


def test_read_state_rejects_empty_pending_job_id():
    payload = fresh_campaign_state().to_dict()
    payload["pending_jobs"] = {CampaignPhase.FEREBUS.value: ""}
    with pytest.raises(StateSchemaError, match="pending_jobs value"):
        CampaignState.from_dict(payload)


def test_state_is_terminal_flag():
    s = fresh_campaign_state()
    assert not s.is_terminal
    s.phase = CampaignPhase.DONE
    assert s.is_terminal
    s.phase = CampaignPhase.HALTED
    assert s.is_terminal


# --- M15 F3: schema bump + new state fields ----------------------------


def test_schema_version_is_three():
    from ichor.hpc.active_learning.daemon.state import SCHEMA_VERSION
    assert SCHEMA_VERSION == 3


def test_last_n_anti_overlap_flagged_default_and_roundtrip():
    from ichor.hpc.active_learning.daemon.state import CampaignState
    st = CampaignState()
    assert st.last_n_anti_overlap_flagged == 0
    st.last_n_anti_overlap_flagged = 7
    st2 = CampaignState.from_dict(st.to_dict())
    assert st2.last_n_anti_overlap_flagged == 7


def test_sacct_empty_streak_default_and_roundtrip():
    from ichor.hpc.active_learning.daemon.state import CampaignState
    st = CampaignState()
    assert st.sacct_empty_streak == {}
    st.sacct_empty_streak = {"12345": 3, "67890": 1}
    st2 = CampaignState.from_dict(st.to_dict())
    assert st2.sacct_empty_streak == {"12345": 3, "67890": 1}


def test_schema_v2_payload_rejected():
    """Pre-M15 state.json (schema 2) MUST be rejected -- system never deployed
    so this is a clean break, but the rejection message should be clear."""
    from ichor.hpc.active_learning.daemon.state import (
        CampaignState,
        StateSchemaError,
    )
    import pytest
    payload = CampaignState().to_dict()
    payload["schema_version"] = 2
    with pytest.raises(StateSchemaError, match="schema_version"):
        CampaignState.from_dict(payload)
