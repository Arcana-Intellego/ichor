"""Authority resolution for a published ARIADNE batch decision."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from ..execution_identity import read_environment_generation
from .config_lock import (
    canonical_config,
    config_fingerprint,
    config_lock_path,
    read_historical_config_by_fingerprint,
)
from .submission_intent import intent_attempt_records, load_intent


def _intent_logical_total(intent: Mapping[str, Any]) -> Optional[int]:
    source = intent.get("postprocess_source")
    if isinstance(source, Mapping):
        value = source.get("logical_total")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
    recovery = intent.get("array_recovery")
    if isinstance(recovery, Mapping):
        value = recovery.get("logical_total")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
    value = intent.get("logical_expected_tasks")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return int(value)
    value = intent.get("expected_tasks")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return int(value)
    return None


def _environment_for_intent(
    campaign: Path,
    intent: Mapping[str, Any],
    *,
    campaign_uid: str,
) -> Optional[Dict[str, Any]]:
    generation = intent.get("environment_generation")
    digest = intent.get("environment_generation_digest_sha256")
    if generation is None and digest is None:
        return None
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise ValueError(
            "ARIADNE decision producer has no valid environment binding"
        )
    payload = read_environment_generation(
        campaign,
        generation=int(generation),
        expected_campaign_uid=str(campaign_uid),
    )
    if str(payload.get("digest_sha256") or "") != digest:
        raise ValueError(
            "ARIADNE decision producer environment digest mismatch"
        )
    return payload


def _decision_contract(intent: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    contract = intent.get("decision_contract")
    if not isinstance(contract, Mapping):
        return None
    try:
        threshold = float(contract["failure_threshold_fraction"])
        config_sha256 = str(contract["config_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ARIADNE decision contract is malformed") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("ARIADNE decision failure threshold is invalid")
    if (
        len(config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in config_sha256)
    ):
        raise ValueError("ARIADNE decision config digest is invalid")
    return {
        "failure_threshold_fraction": threshold,
        "config_sha256": config_sha256,
    }


def _recovery_source_identities(
    campaign: Path,
    *,
    campaign_uid: str,
    iteration: int,
    replacement_round: int,
    logical_total: int,
) -> Sequence[str]:
    from .scheduler_recovery import (
        phase_recovery_ledger_path,
        read_phase_recovery_ledger,
    )

    path = phase_recovery_ledger_path(
        campaign,
        phase="ARIADNE_ARRAY",
        iteration=int(iteration),
        replacement_round=int(replacement_round),
    )
    if not path.exists() and not path.is_symlink():
        return ()
    ledger = read_phase_recovery_ledger(path)
    identity = (
        str(ledger.get("campaign_uid") or ""),
        str(ledger.get("phase") or ""),
        int(ledger.get("iteration", -1)),
        int(ledger.get("replacement_round", -1)),
    )
    expected_identity = (
        str(campaign_uid),
        "ARIADNE_ARRAY",
        int(iteration),
        int(replacement_round),
    )
    if identity != expected_identity:
        raise ValueError("ARIADNE recovery-ledger identity mismatch")
    reusable = list(ledger.get("reusable_logical_task_ids") or [])
    retry = list(ledger.get("retry_logical_task_ids") or [])
    if sorted(int(value) for value in reusable + retry) != list(
        range(int(logical_total))
    ):
        raise ValueError(
            "ARIADNE recovery ledger does not cover the full logical task set"
        )
    identities = [
        str(record.get("submission_identity") or "")
        for record in list(ledger.get("source_terminal_receipts") or [])
        if isinstance(record, Mapping)
    ]
    if not identities or any(not value for value in identities):
        raise ValueError(
            "ARIADNE recovery ledger has no valid source attempt identities"
        )
    return tuple(identities)


def _validate_contract_config_history(
    campaign: Path,
    *,
    campaign_uid: str,
    config_sha256: str,
    scheduler_kinds: set[str],
    allow_unbound_synthetic: bool,
) -> None:
    try:
        read_historical_config_by_fingerprint(
            campaign,
            str(config_sha256),
            expected_campaign_uid=str(campaign_uid),
        )
        return
    except (FileNotFoundError, ValueError):
        lock_path = config_lock_path(campaign)
        if (
            lock_path.exists()
            or lock_path.is_symlink()
            or scheduler_kinds != {"synthetic"}
            or not allow_unbound_synthetic
        ):
            raise
    from ..config import CampaignConfig

    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    if config_fingerprint(canonical_config(config)) != str(config_sha256):
        raise ValueError(
            "synthetic ARIADNE decision contract does not match campaign.yaml"
        )


def resolve_ariadne_handoff_decision_contract(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    iteration: int,
    logical_total: int,
    replacement_round: int = 0,
) -> Optional[Dict[str, Any]]:
    """Resolve the frozen decision controls for one published ARIADNE batch."""
    campaign = Path(campaign_dir).expanduser().resolve()
    expected_total = int(logical_total)
    if expected_total <= 0:
        raise ValueError("ARIADNE handoff logical task total must be positive")
    records = list(
        intent_attempt_records(
            campaign,
            "ARIADNE_ARRAY",
            int(iteration),
            expected_campaign_uid=str(campaign_uid),
        )
    )
    current = load_intent(
        campaign,
        "ARIADNE_ARRAY",
        int(iteration),
        expected_campaign_uid=str(campaign_uid),
    )
    if current is None and not records:
        return None
    by_identity = {
        str(record.get("submission_identity") or ""): dict(record)
        for record in records
        if str(record.get("submission_identity") or "")
    }
    selected = dict(current) if isinstance(current, Mapping) else None
    if selected is None and records:
        selected = dict(
            max(
                records,
                key=lambda record: (
                    int(record.get("attempt_sequence", 0)),
                    str(record.get("attempt_id") or ""),
                ),
            )
        )
    if selected is None:  # pragma: no cover - guarded above
        return None
    selected_contract = _decision_contract(selected)
    if selected_contract is None:
        if any(_decision_contract(record) is not None for record in records):
            raise ValueError(
                "current ARIADNE intent lost its historical decision contract"
            )
        return None
    selected_total = _intent_logical_total(selected)
    if selected_total is not None and selected_total != expected_total:
        raise ValueError(
            "ARIADNE decision intent logical task count mismatch"
        )

    recovery_source_identities = set(
        _recovery_source_identities(
            campaign,
            campaign_uid=str(campaign_uid),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            logical_total=expected_total,
        )
    )
    source_identities = set(recovery_source_identities)
    selected_identity = str(selected.get("submission_identity") or "")
    if selected_identity:
        source_identities.add(selected_identity)
    nested_source = selected.get("postprocess_source")
    producer_records = []
    if isinstance(nested_source, Mapping):
        producer_records.append(dict(nested_source))
    for identity in sorted(source_identities):
        record = by_identity.get(identity)
        if record is None:
            if identity == selected_identity:
                record = selected
            else:
                raise ValueError(
                    "ARIADNE recovery source intent is unavailable: " + identity
                )
        producer_records.append(dict(record))
    if not producer_records:
        producer_records.append(selected)

    contract_bound_to_environment = False
    authenticated_unbound_contract = False
    scheduler_kinds = set()
    producer_attempt_ids = []
    producer_submission_identities = []
    for producer in producer_records:
        contract = _decision_contract(producer)
        if contract != selected_contract:
            raise ValueError(
                "ARIADNE producer attempts have contradictory decision contracts"
            )
        environment = _environment_for_intent(
            campaign,
            producer,
            campaign_uid=str(campaign_uid),
        )
        if environment is None:
            producer_identity = str(
                producer.get("submission_identity") or ""
            )
            if (
                producer_identity in recovery_source_identities
                or (
                    not recovery_source_identities
                    and producer_identity == selected_identity
                )
            ):
                authenticated_unbound_contract = True
        elif str(environment.get("campaign_config_sha256") or "") == str(
            selected_contract["config_sha256"]
        ):
            contract_bound_to_environment = True
        scheduler_kind = producer.get("scheduler_identity_kind")
        if scheduler_kind is not None:
            scheduler_kinds.add(str(scheduler_kind))
        attempt_id = str(producer.get("attempt_id") or "")
        submission_identity = str(producer.get("submission_identity") or "")
        if attempt_id:
            producer_attempt_ids.append(attempt_id)
        if submission_identity:
            producer_submission_identities.append(submission_identity)
    if len(scheduler_kinds) > 1:
        raise ValueError("ARIADNE producer scheduler identity changed")
    _validate_contract_config_history(
        campaign,
        campaign_uid=str(campaign_uid),
        config_sha256=str(selected_contract["config_sha256"]),
        scheduler_kinds=scheduler_kinds,
        allow_unbound_synthetic=authenticated_unbound_contract,
    )
    if (
        not contract_bound_to_environment
        and not authenticated_unbound_contract
    ):
        raise ValueError(
            "ARIADNE decision contract is not bound to a producer environment"
        )

    return {
        "decision_contract": dict(selected_contract),
        "logical_total": expected_total,
        "producer_attempt_ids": sorted(set(producer_attempt_ids)),
        "producer_submission_identities": sorted(
            set(producer_submission_identities)
        ),
        "authority_kind": (
            "submission_intent"
            if contract_bound_to_environment
            else "submission_intent_legacy_unbound"
        ),
    }


__all__ = ["resolve_ariadne_handoff_decision_contract"]
