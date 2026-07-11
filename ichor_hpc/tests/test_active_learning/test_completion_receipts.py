import json

import pytest

from ichor.hpc.active_learning.daemon.completion_receipts import (
    CompletionReceiptError,
    evidence_records,
    replayable_completion_receipts,
    receipt_reference,
    validate_completion_reference,
    write_completion_receipt,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state,
    read_state,
    write_state,
)


def test_completion_receipt_roundtrip_binds_state_and_evidence(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    evidence = campaign / "handoff.json"
    evidence.write_text('{"ok": true}\n', encoding="utf-8")
    before = fresh_campaign_state(campaign_uid="uid")
    after = fresh_campaign_state(campaign_uid="uid")
    after.phase = CampaignPhase.PHASE_A_POLUS

    path = write_completion_receipt(
        campaign,
        campaign_uid="uid",
        phase="INIT",
        iteration=0,
        replacement_round=0,
        config_sha256="a" * 64,
        state_before=before,
        state_after=after,
        next_phase="PHASE_A_POLUS",
        next_iteration=0,
        state_updates={},
        evidence=evidence_records(campaign, [evidence]),
    )
    reference = receipt_reference(campaign, path)

    payload = validate_completion_reference(
        campaign,
        reference,
        expected_campaign_uid="uid",
    )

    assert payload["phase"] == "INIT"
    assert payload["next_phase"] == "PHASE_A_POLUS"
    assert payload["evidence"][0]["path"] == "handoff.json"


def test_completion_receipt_detects_evidence_drift(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    evidence = campaign / "handoff.json"
    evidence.write_text('{"ok": true}\n', encoding="utf-8")
    before = fresh_campaign_state(campaign_uid="uid")
    after = fresh_campaign_state(campaign_uid="uid")
    after.phase = CampaignPhase.PHASE_A_POLUS
    path = write_completion_receipt(
        campaign,
        campaign_uid="uid",
        phase="INIT",
        iteration=0,
        replacement_round=0,
        config_sha256="a" * 64,
        state_before=before,
        state_after=after,
        next_phase="PHASE_A_POLUS",
        next_iteration=0,
        state_updates={},
        evidence=evidence_records(campaign, [evidence]),
    )
    reference = receipt_reference(campaign, path)
    evidence.write_text('{"ok": false}\n', encoding="utf-8")

    with pytest.raises(CompletionReceiptError, match="evidence .* mismatch"):
        validate_completion_reference(campaign, reference)


def test_completion_receipt_rejects_path_escape(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"x": 1}), encoding="utf-8")

    with pytest.raises(CompletionReceiptError, match="outside campaign"):
        evidence_records(campaign, [outside])


def test_completion_receipt_refuses_missing_required_evidence(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()

    with pytest.raises(CompletionReceiptError, match="evidence is missing"):
        evidence_records(campaign, [campaign / "missing-handoff.json"])


def test_completion_receipt_exposes_replayable_post_state(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    before = fresh_campaign_state(campaign_uid="uid")
    after = fresh_campaign_state(campaign_uid="uid")
    after.phase = CampaignPhase.PHASE_A_POLUS
    write_completion_receipt(
        campaign,
        campaign_uid="uid",
        phase="INIT",
        iteration=0,
        replacement_round=0,
        config_sha256="b" * 64,
        state_before=before,
        state_after=after,
        next_phase="PHASE_A_POLUS",
        next_iteration=0,
        state_updates={},
        evidence=[],
    )

    matches = replayable_completion_receipts(
        campaign,
        before,
        expected_config_sha256="b" * 64,
    )

    assert len(matches) == 1
    assert matches[0]["payload"]["state_after"]["phase"] == "PHASE_A_POLUS"


def test_daemon_replays_receipt_after_state_persist_crash(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon, TickStatus
    from ichor.hpc.active_learning.daemon.phase_executor import MockPhaseExecutor

    campaign = tmp_path / "campaign"
    daemon = Daemon(
        campaign_dir=campaign,
        config=CampaignConfig(),
        executor=MockPhaseExecutor(),
    )
    daemon.data_dir().mkdir(parents=True)
    before = fresh_campaign_state(campaign_uid="uid")
    write_state(daemon.state_path(), before)
    real_persist = daemon._persist

    def fail_after_receipt(_state):
        raise OSError("simulated state persistence failure")

    monkeypatch.setattr(daemon, "_persist", fail_after_receipt)
    with pytest.raises(OSError, match="simulated state persistence failure"):
        daemon._advance(before, CampaignPhase.INIT, {})
    monkeypatch.setattr(daemon, "_persist", real_persist)

    assert read_state(daemon.state_path()).phase is CampaignPhase.INIT
    assert daemon.tick() == TickStatus.ADVANCED
    recovered = read_state(daemon.state_path())
    assert recovered.phase is CampaignPhase.PHASE_A_POLUS
    assert recovered.last_completion_receipt is not None
