from __future__ import annotations

import pytest

import ichor.hpc.active_learning.daemon.ariadne_decision_authority as authority
import ichor.hpc.active_learning.daemon.scheduler_recovery as scheduler_recovery
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.config_lock import (
    canonical_config,
    config_fingerprint,
)


CAMPAIGN_UID = "ariadne-decision-authority-test"
FROZEN_CONFIG_SHA = "a" * 64
CURRENT_CONFIG_SHA = "b" * 64
CONTRACT = {
    "failure_threshold_fraction": 0.05,
    "config_sha256": FROZEN_CONFIG_SHA,
}


def _intent(
    *,
    submission_identity: str,
    attempt_sequence: int,
    generation: int,
    decision_contract=None,
):
    return {
        "campaign_uid": CAMPAIGN_UID,
        "phase": "ARIADNE_ARRAY",
        "iteration": 15,
        "replacement_round": 0,
        "attempt_id": str(attempt_sequence) * 32,
        "attempt_sequence": int(attempt_sequence),
        "submission_identity": str(submission_identity),
        "scheduler_identity_kind": "slurm",
        "logical_expected_tasks": 200,
        "environment_generation": int(generation),
        "environment_generation_digest_sha256": str(generation) * 64,
        "decision_contract": dict(
            CONTRACT if decision_contract is None else decision_contract
        ),
    }


def _install_common_fakes(monkeypatch, tmp_path, *, records, current, configs):
    monkeypatch.setattr(
        authority,
        "intent_attempt_records",
        lambda *_args, **_kwargs: tuple(dict(record) for record in records),
    )
    monkeypatch.setattr(
        authority,
        "load_intent",
        lambda *_args, **_kwargs: dict(current),
    )

    def read_generation(_campaign, *, generation, expected_campaign_uid):
        assert expected_campaign_uid == CAMPAIGN_UID
        return {
            "generation": int(generation),
            "campaign_uid": CAMPAIGN_UID,
            "digest_sha256": str(generation) * 64,
            "campaign_config_sha256": str(configs[int(generation)]),
        }

    monkeypatch.setattr(authority, "read_environment_generation", read_generation)
    monkeypatch.setattr(
        authority,
        "read_historical_config_by_fingerprint",
        lambda _campaign, fingerprint, **_kwargs: (
            object()
            if fingerprint == FROZEN_CONFIG_SHA
            else (_ for _ in ()).throw(FileNotFoundError(fingerprint))
        ),
    )
    monkeypatch.setattr(
        scheduler_recovery,
        "phase_recovery_ledger_path",
        lambda *_args, **_kwargs: tmp_path / "missing-ledger.json",
    )


def test_resolves_decision_contract_from_single_producer(
    tmp_path,
    monkeypatch,
):
    producer = _intent(
        submission_identity="r0000-a0001-source",
        attempt_sequence=1,
        generation=1,
    )
    _install_common_fakes(
        monkeypatch,
        tmp_path,
        records=[producer],
        current=producer,
        configs={1: FROZEN_CONFIG_SHA},
    )

    resolved = authority.resolve_ariadne_handoff_decision_contract(
        tmp_path,
        campaign_uid=CAMPAIGN_UID,
        iteration=15,
        logical_total=200,
    )

    assert resolved["decision_contract"] == CONTRACT
    assert resolved["producer_submission_identities"] == [
        "r0000-a0001-source"
    ]


def test_legacy_unbound_producer_requires_authenticated_config_history(
    tmp_path,
    monkeypatch,
):
    producer = _intent(
        submission_identity="r0000-a0001-source",
        attempt_sequence=1,
        generation=1,
    )
    producer.pop("environment_generation")
    producer.pop("environment_generation_digest_sha256")
    _install_common_fakes(
        monkeypatch,
        tmp_path,
        records=[producer],
        current=producer,
        configs={},
    )

    resolved = authority.resolve_ariadne_handoff_decision_contract(
        tmp_path,
        campaign_uid=CAMPAIGN_UID,
        iteration=15,
        logical_total=200,
    )

    assert resolved["decision_contract"] == CONTRACT
    assert resolved["authority_kind"] == "submission_intent_legacy_unbound"


def test_unbound_synthetic_producer_can_use_matching_campaign_yaml(
    tmp_path,
    monkeypatch,
):
    config = CampaignConfig()
    config.to_yaml(tmp_path / "campaign.yaml")
    config_sha256 = config_fingerprint(canonical_config(config))
    producer = _intent(
        submission_identity="r0000-a0001-source",
        attempt_sequence=1,
        generation=1,
        decision_contract={
            **CONTRACT,
            "config_sha256": config_sha256,
        },
    )
    producer["scheduler_identity_kind"] = "synthetic"
    producer.pop("environment_generation")
    producer.pop("environment_generation_digest_sha256")
    _install_common_fakes(
        monkeypatch,
        tmp_path,
        records=[producer],
        current=producer,
        configs={},
    )
    monkeypatch.setattr(
        authority,
        "read_historical_config_by_fingerprint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("config lock is absent")
        ),
    )

    resolved = authority.resolve_ariadne_handoff_decision_contract(
        tmp_path,
        campaign_uid=CAMPAIGN_UID,
        iteration=15,
        logical_total=200,
    )

    assert resolved["decision_contract"]["config_sha256"] == config_sha256
    assert resolved["authority_kind"] == "submission_intent_legacy_unbound"


def test_mixed_attempt_recovery_uses_original_frozen_contract(
    tmp_path,
    monkeypatch,
):
    source = _intent(
        submission_identity="r0000-a0001-source",
        attempt_sequence=1,
        generation=1,
    )
    retry = _intent(
        submission_identity="r0000-a0002-retry",
        attempt_sequence=2,
        generation=2,
    )
    retry["array_recovery"] = {
        "logical_total": 200,
        "n_reuse": 4,
        "n_retry": 196,
    }
    _install_common_fakes(
        monkeypatch,
        tmp_path,
        records=[source, retry],
        current=retry,
        configs={1: FROZEN_CONFIG_SHA, 2: CURRENT_CONFIG_SHA},
    )
    ledger_path = tmp_path / "phase-recovery-ledger.json"
    ledger_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        scheduler_recovery,
        "phase_recovery_ledger_path",
        lambda *_args, **_kwargs: ledger_path,
    )
    monkeypatch.setattr(
        scheduler_recovery,
        "read_phase_recovery_ledger",
        lambda _path: {
            "schema_version": 1,
            "campaign_uid": CAMPAIGN_UID,
            "phase": "ARIADNE_ARRAY",
            "iteration": 15,
            "replacement_round": 0,
            "reusable_logical_task_ids": list(range(4)),
            "retry_logical_task_ids": list(range(4, 200)),
            "source_terminal_receipts": [
                {"submission_identity": "r0000-a0001-source"}
            ],
        },
    )

    resolved = authority.resolve_ariadne_handoff_decision_contract(
        tmp_path,
        campaign_uid=CAMPAIGN_UID,
        iteration=15,
        logical_total=200,
    )

    assert resolved["decision_contract"] == CONTRACT
    assert resolved["producer_submission_identities"] == [
        "r0000-a0001-source",
        "r0000-a0002-retry",
    ]


def test_mixed_attempt_recovery_rejects_contradictory_contracts(
    tmp_path,
    monkeypatch,
):
    source = _intent(
        submission_identity="r0000-a0001-source",
        attempt_sequence=1,
        generation=1,
        decision_contract={
            **CONTRACT,
            "failure_threshold_fraction": 0.10,
        },
    )
    retry = _intent(
        submission_identity="r0000-a0002-retry",
        attempt_sequence=2,
        generation=2,
    )
    _install_common_fakes(
        monkeypatch,
        tmp_path,
        records=[source, retry],
        current=retry,
        configs={1: FROZEN_CONFIG_SHA, 2: CURRENT_CONFIG_SHA},
    )
    ledger_path = tmp_path / "phase-recovery-ledger.json"
    ledger_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        scheduler_recovery,
        "phase_recovery_ledger_path",
        lambda *_args, **_kwargs: ledger_path,
    )
    monkeypatch.setattr(
        scheduler_recovery,
        "read_phase_recovery_ledger",
        lambda _path: {
            "schema_version": 1,
            "campaign_uid": CAMPAIGN_UID,
            "phase": "ARIADNE_ARRAY",
            "iteration": 15,
            "replacement_round": 0,
            "reusable_logical_task_ids": list(range(200)),
            "retry_logical_task_ids": [],
            "source_terminal_receipts": [
                {"submission_identity": "r0000-a0001-source"}
            ],
        },
    )

    with pytest.raises(ValueError, match="contradictory decision contracts"):
        authority.resolve_ariadne_handoff_decision_contract(
            tmp_path,
            campaign_uid=CAMPAIGN_UID,
            iteration=15,
            logical_total=200,
        )


def test_decision_contract_must_match_a_producer_environment(
    tmp_path,
    monkeypatch,
):
    producer = _intent(
        submission_identity="r0000-a0001-source",
        attempt_sequence=1,
        generation=1,
    )
    _install_common_fakes(
        monkeypatch,
        tmp_path,
        records=[producer],
        current=producer,
        configs={1: CURRENT_CONFIG_SHA},
    )

    with pytest.raises(ValueError, match="not bound to a producer environment"):
        authority.resolve_ariadne_handoff_decision_contract(
            tmp_path,
            campaign_uid=CAMPAIGN_UID,
            iteration=15,
            logical_total=200,
        )


def test_proven_legacy_config_generation_split_is_accepted(
    tmp_path,
    monkeypatch,
):
    producer = _intent(
        submission_identity="r0000-a0001-source",
        attempt_sequence=1,
        generation=1,
    )
    _install_common_fakes(
        monkeypatch,
        tmp_path,
        records=[producer],
        current=producer,
        configs={1: CURRENT_CONFIG_SHA},
    )
    monkeypatch.setattr(
        authority,
        "legacy_intent_config_generation_split_is_proven",
        lambda *_args, **_kwargs: True,
    )

    resolved = authority.resolve_ariadne_handoff_decision_contract(
        tmp_path,
        campaign_uid=CAMPAIGN_UID,
        iteration=15,
        logical_total=200,
    )

    assert resolved["decision_contract"] == CONTRACT
    assert (
        resolved["authority_kind"]
        == "submission_intent_legacy_config_generation_split"
    )
