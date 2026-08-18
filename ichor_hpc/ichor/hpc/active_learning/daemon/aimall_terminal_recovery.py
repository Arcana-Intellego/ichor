"""Structural output classification and bounded AIMAll retry provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..strict_json import strict_json as json

from .aimall_output_validation import (
    AIMALL_AUTHORITY_INVALID,
    AIMALL_OUTPUT_INVALID,
    AIMALL_STRUCTURAL_INVALID_EXIT_CODE,
    assess_aimall_output,
)


AIMALL_STRUCTURAL_RETRY_KIND = "aimall_structural_retry_v1"
AIMALL_STRUCTURAL_RETRY_METADATA_KEY = "aimall_structural_retry"
AIMALL_STRUCTURAL_VALIDATOR_KIND = "aimall_structural_validator_v1"
AIMALL_STRUCTURAL_VALIDATOR_METADATA_KEY = "aimall_structural_validator"
AIMALL_NATIVE_EXIT_CODE_REMAP = 85


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(label + " is not a SHA-256 digest")
    return text


def _task_ids(values: Any, label: str) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(label + " must be a list")
    parsed = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(label + " contains an invalid logical task ID")
        parsed.append(int(value))
    if parsed != sorted(set(parsed)):
        raise ValueError(label + " must be unique and ordered")
    return parsed


def build_aimall_structural_validator_metadata(
    submitted_script_sha256: str,
) -> dict[str, Any]:
    """Bind the reserved exit code to one immutable submitted script."""
    return {
        "kind": AIMALL_STRUCTURAL_VALIDATOR_KIND,
        "invalid_exit_code": AIMALL_STRUCTURAL_INVALID_EXIT_CODE,
        "native_exit_code_remap": AIMALL_NATIVE_EXIT_CODE_REMAP,
        "submitted_script_sha256": _sha256_text(
            submitted_script_sha256,
            "AIMAll structural validator script digest",
        ),
    }


def validate_aimall_structural_validator_metadata(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("AIMAll structural validator metadata must be an object")
    payload = dict(value)
    if set(payload) != {
        "kind",
        "invalid_exit_code",
        "native_exit_code_remap",
        "submitted_script_sha256",
    }:
        raise ValueError("AIMAll structural validator metadata keys are invalid")
    if payload.get("kind") != AIMALL_STRUCTURAL_VALIDATOR_KIND:
        raise ValueError("AIMAll structural validator metadata kind is invalid")
    if payload.get("invalid_exit_code") != AIMALL_STRUCTURAL_INVALID_EXIT_CODE:
        raise ValueError("AIMAll structural validator exit code is invalid")
    if payload.get("native_exit_code_remap") != AIMALL_NATIVE_EXIT_CODE_REMAP:
        raise ValueError("AIMAll native exit-code remap is invalid")
    payload["submitted_script_sha256"] = _sha256_text(
        payload.get("submitted_script_sha256"),
        "AIMAll structural validator script digest",
    )
    return payload


def aimall_intent_has_structural_validator_contract(
    campaign_dir: Path,
    intent: Mapping[str, Any],
) -> bool:
    """Prove that exit 86 came from the new structural validator."""
    submission_metadata = intent.get("submission_metadata")
    if not isinstance(submission_metadata, Mapping):
        return False
    value = submission_metadata.get(AIMALL_STRUCTURAL_VALIDATOR_METADATA_KEY)
    if value is None:
        return False
    metadata = validate_aimall_structural_validator_metadata(value)
    submitted_sha256 = _sha256_text(
        intent.get("submitted_script_sha256"),
        "AIMAll intent submitted-script digest",
    )
    if metadata["submitted_script_sha256"] != submitted_sha256:
        raise ValueError(
            "AIMAll structural validator metadata and submitted script differ"
        )
    binding_path = intent.get("script_binding_path")
    binding_sha256 = intent.get("script_binding_sha256")
    if not isinstance(binding_path, str) or not binding_path:
        raise ValueError("AIMAll structural validator has no script binding")
    binding_sha256 = _sha256_text(
        binding_sha256,
        "AIMAll intent script-binding digest",
    )
    from .filesystem import campaign_owned_path
    from .script_bundles import verify_script_binding

    binding = campaign_owned_path(Path(campaign_dir), Path(binding_path))
    script_binding = verify_script_binding(binding, binding_sha256)
    if str(script_binding.get("script_sha256") or "") != submitted_sha256:
        raise ValueError(
            "AIMAll structural validator script binding has the wrong digest"
        )
    script_path = campaign_owned_path(
        Path(campaign_dir),
        Path(str(script_binding.get("script_path") or "")),
    )
    submitted_path = intent.get("submitted_script_path")
    if not isinstance(submitted_path, str) or not submitted_path:
        raise ValueError("AIMAll structural validator has no submitted script path")
    if script_path != campaign_owned_path(
        Path(campaign_dir),
        Path(submitted_path),
    ):
        raise ValueError(
            "AIMAll structural validator submitted script path mismatch"
        )
    try:
        script_text = script_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            "AIMAll structural validator submitted script is unreadable"
        ) from exc
    validator_command = (
        "-m ichor.hpc.active_learning.daemon.aimall_output_validation"
    )
    native_remap = (
        'if [[ "$ICHOR_AIMALL_BACKEND_STATUS" -eq '
        + str(AIMALL_STRUCTURAL_INVALID_EXIT_CODE)
        + " ]]; then exit "
        + str(AIMALL_NATIVE_EXIT_CODE_REMAP)
        + "; fi"
    )
    if validator_command not in script_text or native_remap not in script_text:
        raise ValueError(
            "AIMAll submitted script does not implement its structural "
            "validator contract"
        )
    return True


def build_aimall_structural_retry_metadata(
    *,
    campaign_uid: str,
    phase_name: str,
    iteration: int,
    replacement_round: int,
    task_ids: Sequence[int],
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build self-authenticating metadata bound before retry submission."""
    parsed_ids = sorted(set(int(value) for value in task_ids))
    if not parsed_ids:
        raise ValueError("AIMAll structural retry metadata has no tasks")
    sources = []
    for record in source_records:
        source = {
            "kind": str(record.get("kind") or ""),
            "sha256": _sha256_text(
                record.get("sha256"),
                "AIMAll structural retry source digest",
            ),
            "submission_identity": str(
                record.get("submission_identity") or ""
            ),
            "job_id": str(record.get("job_id") or ""),
        }
        if source["kind"] not in {
            "postprocess_source",
            "scheduler_terminal_receipt",
            "structural_retry_provenance",
        }:
            raise ValueError("AIMAll structural retry source kind is invalid")
        if not source["submission_identity"]:
            raise ValueError(
                "AIMAll structural retry source submission identity is empty"
            )
        sources.append(source)
    if not sources:
        raise ValueError("AIMAll structural retry metadata has no source evidence")
    payload = {
        "kind": AIMALL_STRUCTURAL_RETRY_KIND,
        "campaign_uid": str(campaign_uid),
        "phase": str(phase_name),
        "iteration": int(iteration),
        "replacement_round": int(replacement_round),
        "task_ids": parsed_ids,
        "sources": sources,
    }
    payload["metadata_sha256"] = _sha256_payload(payload)
    return validate_aimall_structural_retry_metadata(
        payload,
        expected_campaign_uid=str(campaign_uid),
        expected_phase=str(phase_name),
        expected_iteration=int(iteration),
        expected_replacement_round=int(replacement_round),
    )


def validate_aimall_structural_retry_metadata(
    value: Any,
    *,
    expected_campaign_uid: str,
    expected_phase: str,
    expected_iteration: int,
    expected_replacement_round: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("AIMAll structural retry metadata must be an object")
    payload = dict(value)
    required = {
        "kind",
        "campaign_uid",
        "phase",
        "iteration",
        "replacement_round",
        "task_ids",
        "sources",
        "metadata_sha256",
    }
    if set(payload) != required:
        raise ValueError("AIMAll structural retry metadata keys are invalid")
    identity = (
        str(payload.get("campaign_uid") or ""),
        str(payload.get("phase") or ""),
        payload.get("iteration"),
        payload.get("replacement_round"),
    )
    expected = (
        str(expected_campaign_uid),
        str(expected_phase),
        int(expected_iteration),
        int(expected_replacement_round),
    )
    if identity != expected or payload.get("kind") != AIMALL_STRUCTURAL_RETRY_KIND:
        raise ValueError("AIMAll structural retry metadata identity mismatch")
    payload["task_ids"] = _task_ids(
        payload.get("task_ids"),
        "AIMAll structural retry task IDs",
    )
    if not payload["task_ids"]:
        raise ValueError("AIMAll structural retry metadata has no tasks")
    source_values = payload.get("sources")
    if not isinstance(source_values, list) or not source_values:
        raise ValueError("AIMAll structural retry sources are invalid")
    sources = []
    for value in source_values:
        if not isinstance(value, Mapping) or set(value) != {
            "kind",
            "sha256",
            "submission_identity",
            "job_id",
        }:
            raise ValueError("AIMAll structural retry source is invalid")
        source = dict(value)
        if source.get("kind") not in {
            "postprocess_source",
            "scheduler_terminal_receipt",
            "structural_retry_provenance",
        }:
            raise ValueError("AIMAll structural retry source kind is invalid")
        source["sha256"] = _sha256_text(
            source.get("sha256"),
            "AIMAll structural retry source digest",
        )
        if not isinstance(source.get("submission_identity"), str) or not source[
            "submission_identity"
        ]:
            raise ValueError(
                "AIMAll structural retry source submission identity is invalid"
            )
        if not isinstance(source.get("job_id"), str):
            raise ValueError("AIMAll structural retry source JobID is invalid")
        sources.append(source)
    payload["sources"] = sources
    digest = _sha256_text(
        payload.get("metadata_sha256"),
        "AIMAll structural retry metadata digest",
    )
    unsigned = dict(payload)
    unsigned.pop("metadata_sha256", None)
    if _sha256_payload(unsigned) != digest:
        raise ValueError("AIMAll structural retry metadata digest mismatch")
    payload["metadata_sha256"] = digest
    return payload


def _retry_metadata_for_intent(
    intent: Mapping[str, Any],
    *,
    campaign_uid: str,
    phase_name: str,
    iteration: int,
    replacement_round: int,
) -> dict[str, Any] | None:
    submission_metadata = intent.get("submission_metadata")
    if not isinstance(submission_metadata, Mapping):
        return None
    value = submission_metadata.get(AIMALL_STRUCTURAL_RETRY_METADATA_KEY)
    if value is None:
        return None
    return validate_aimall_structural_retry_metadata(
        value,
        expected_campaign_uid=str(campaign_uid),
        expected_phase=str(phase_name),
        expected_iteration=int(iteration),
        expected_replacement_round=int(replacement_round),
    )


def _outcome_is_exit_86(
    outcome: Mapping[str, Any],
    *,
    validator_contract: bool,
) -> bool:
    return (
        bool(validator_contract)
        and str(outcome.get("status") or "") != "COMPLETED"
        and outcome.get("exit_code")
        == [AIMALL_STRUCTURAL_INVALID_EXIT_CODE, 0]
    )


def resolve_aimall_structural_recovery_policy(
    campaign_dir: Path,
    *,
    campaign_uid: str,
    phase_name: str,
    iteration: int,
    replacement_round: int,
    recoveries: Sequence[Mapping[str, Any]] = (),
    local_classification: Mapping[str, Any] | None = None,
    invalid_completed_tasks: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Derive first retries and exhausted structural failures from authority."""
    from .submission_intent import intent_attempt_records

    intent_records = intent_attempt_records(
        campaign_dir,
        str(phase_name),
        int(iteration),
        expected_campaign_uid=str(campaign_uid),
    )
    retry_metadata_by_identity: dict[str, dict[str, Any]] = {}
    source_records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for intent in intent_records:
        if int(intent.get("replacement_round", -1)) != int(
            replacement_round
        ):
            continue
        metadata = _retry_metadata_for_intent(
            intent,
            campaign_uid=str(campaign_uid),
            phase_name=str(phase_name),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
        )
        if metadata is None:
            continue
        identity = str(intent.get("submission_identity") or "")
        if not identity or identity in retry_metadata_by_identity:
            raise ValueError(
                "AIMAll structural retry intent history is ambiguous"
            )
        retry_metadata_by_identity[identity] = metadata
        source_records_by_key[("structural_retry_provenance", identity)] = {
            "kind": "structural_retry_provenance",
            "sha256": str(metadata["metadata_sha256"]),
            "submission_identity": identity,
            "job_id": str(intent.get("job_id") or ""),
        }

    exit_86_counts: dict[int, int] = {}
    latest_by_task: dict[int, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    validator_contract_by_identity: dict[str, bool] = {}
    for recovery in recoveries:
        intent = recovery.get("intent")
        receipt = recovery.get("receipt")
        if not isinstance(intent, Mapping) or not isinstance(receipt, Mapping):
            raise ValueError("AIMAll scheduler recovery record is invalid")
        metadata = _retry_metadata_for_intent(
            intent,
            campaign_uid=str(campaign_uid),
            phase_name=str(phase_name),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
        )
        retry_origin_ids = set(
            [] if metadata is None else metadata["task_ids"]
        )
        identity = str(intent.get("submission_identity") or "")
        has_exit_86 = any(
            isinstance(outcome, Mapping)
            and str(outcome.get("status") or "") != "COMPLETED"
            and outcome.get("exit_code")
            == [AIMALL_STRUCTURAL_INVALID_EXIT_CODE, 0]
            for outcome in receipt.get("outcomes", [])
        )
        if has_exit_86:
            validator_contract_by_identity[identity] = (
                aimall_intent_has_structural_validator_contract(
                    Path(campaign_dir),
                    intent,
                )
            )
        source_records_by_key[("scheduler_terminal_receipt", identity)] = {
            "kind": "scheduler_terminal_receipt",
            "sha256": str(receipt.get("receipt_sha256") or ""),
            "submission_identity": identity,
            "job_id": str(intent.get("job_id") or ""),
        }
        for outcome in receipt.get("outcomes", []):
            if not isinstance(outcome, Mapping):
                raise ValueError("AIMAll scheduler outcome is invalid")
            task_id = int(outcome["logical_task_id"])
            latest_by_task[task_id] = (outcome, intent)
            if _outcome_is_exit_86(
                outcome,
                validator_contract=validator_contract_by_identity.get(
                    identity,
                    False,
                ),
            ):
                exit_86_counts[task_id] = exit_86_counts.get(task_id, 0) + 1
            if task_id in retry_origin_ids:
                latest_by_task[task_id] = (outcome, intent)

    local_invalid: dict[int, Mapping[str, Any]] = {}
    if isinstance(local_classification, Mapping):
        for record in local_classification.get("invalid_completed_tasks", []):
            if not isinstance(record, Mapping):
                raise ValueError("AIMAll local output classification is invalid")
            local_invalid[int(record["task_id"])] = record
        if local_invalid:
            identity = str(
                local_classification.get("producer_submission_identity")
                or ""
            )
            source_records_by_key[("postprocess_source", identity)] = {
                "kind": "postprocess_source",
                "sha256": str(
                    local_classification.get("source_sha256") or ""
                ),
                "submission_identity": identity,
                "job_id": str(
                    local_classification.get("producer_job_id") or ""
                ),
            }
    invalid_by_task = dict(local_invalid)
    for record in invalid_completed_tasks:
        if not isinstance(record, Mapping):
            raise ValueError("AIMAll invalid-completed record is malformed")
        if str(record.get("classification") or "structural") != "structural":
            continue
        invalid_by_task[int(record["task_id"])] = record

    structural_retry = set(invalid_by_task)
    for metadata in retry_metadata_by_identity.values():
        structural_retry.update(int(value) for value in metadata["task_ids"])
    terminal_rejection = set()
    terminal_reasons: dict[int, str] = {}
    for task_id, (outcome, intent) in latest_by_task.items():
        metadata = _retry_metadata_for_intent(
            intent,
            campaign_uid=str(campaign_uid),
            phase_name=str(phase_name),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
        )
        was_structural_retry = bool(
            metadata is not None and task_id in set(metadata["task_ids"])
        )
        identity = str(intent.get("submission_identity") or "")
        if _outcome_is_exit_86(
            outcome,
            validator_contract=validator_contract_by_identity.get(
                identity,
                False,
            ),
        ):
            if was_structural_retry or exit_86_counts.get(task_id, 0) >= 2:
                terminal_rejection.add(task_id)
                terminal_reasons[task_id] = "aimall_structural_retry_exhausted"
                structural_retry.discard(task_id)
            else:
                structural_retry.add(task_id)
        elif was_structural_retry and not (
            str(outcome.get("status") or "") == "COMPLETED"
            and outcome.get("exit_code") == [0, 0]
        ):
            # Infrastructure failure did not consume the one scientific retry.
            structural_retry.add(task_id)
        elif was_structural_retry:
            structural_retry.discard(task_id)

    for task_id, record in invalid_by_task.items():
        latest = latest_by_task.get(task_id)
        was_structural_retry = False
        if latest is not None:
            metadata = _retry_metadata_for_intent(
                latest[1],
                campaign_uid=str(campaign_uid),
                phase_name=str(phase_name),
                iteration=int(iteration),
                replacement_round=int(replacement_round),
            )
            was_structural_retry = bool(
                metadata is not None and task_id in set(metadata["task_ids"])
            )
        if not was_structural_retry and isinstance(
            local_classification,
            Mapping,
        ):
            producer_identity = str(
                local_classification.get("producer_submission_identity")
                or ""
            )
            producer_metadata = retry_metadata_by_identity.get(
                producer_identity
            )
            was_structural_retry = bool(
                producer_metadata is not None
                and task_id in set(producer_metadata["task_ids"])
            )
        if was_structural_retry:
            terminal_rejection.add(task_id)
            terminal_reasons[task_id] = str(
                record.get("reason") or "aimall_structural_retry_exhausted"
            )
            structural_retry.discard(task_id)

    from .quantum_task_contracts import quantum_task_contract

    contract = quantum_task_contract(
        campaign_dir,
        str(phase_name),
        int(iteration),
        replacement_round=int(replacement_round),
        expected_campaign_uid=str(campaign_uid),
        validate_points_file=False,
    )
    task_by_id = {
        int(task.logical_task_id): task for task in contract.tasks
    }
    if len(task_by_id) != len(contract.tasks):
        raise ValueError("AIMAll task contract has duplicate logical IDs")
    known_ids = set(task_by_id)
    if not (structural_retry | terminal_rejection).issubset(known_ids):
        raise ValueError("AIMAll structural recovery references an unknown task")
    for task_id in sorted(terminal_rejection):
        task = task_by_id[task_id]
        assessment = assess_aimall_output(Path(task.pointdir))
        if assessment.category == AIMALL_AUTHORITY_INVALID:
            raise ValueError(
                "AIMAll terminal rejection has invalid inherited authority for "
                "logical task "
                + str(task_id)
                + ": "
                + assessment.reason
            )
        if assessment.category != AIMALL_OUTPUT_INVALID:
            raise ValueError(
                "AIMAll terminal structural outcome contradicts a valid current "
                "output for logical task "
                + str(task_id)
            )
        terminal_reasons[task_id] = str(
            terminal_reasons.get(task_id) or assessment.reason
        )
    return {
        "structural_retry_task_ids": sorted(structural_retry),
        "terminal_rejection_task_ids": sorted(terminal_rejection),
        "terminal_rejection_reasons": {
            str(task_id): terminal_reasons[task_id]
            for task_id in sorted(terminal_rejection)
        },
        "source_records": [
            source_records_by_key[key]
            for key in sorted(source_records_by_key)
        ],
    }


def classify_aimall_postprocess_source_outputs(
    campaign_dir: Path,
    *,
    campaign_uid: str,
    phase_name: str,
    iteration: int,
    replacement_round: int,
    source: Mapping[str, Any],
    expected_method: str,
    publish_valid_receipts: bool,
) -> dict[str, Any]:
    """Validate source candidates and optionally bind valid historical work."""
    from .quantum_task_contracts import quantum_task_contract
    from .quantum_task_receipts import (
        AIMALL_TASK_RECEIPT,
        write_quantum_task_receipt_from_postprocess_source,
    )

    contract = quantum_task_contract(
        campaign_dir,
        str(phase_name),
        int(iteration),
        replacement_round=int(replacement_round),
        expected_campaign_uid=str(campaign_uid),
        validate_points_file=False,
    )
    if int(source.get("logical_total", -1)) != int(contract.logical_total):
        raise ValueError(
            "AIMAll postprocess source and logical task contract counts differ"
        )
    from .input_staging import validate_existing_aimall_task_authorities

    validate_existing_aimall_task_authorities(
        campaign_dir,
        phase_name=str(phase_name),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        expected_campaign_uid=str(campaign_uid),
        expected_method=str(expected_method),
        task_contract=contract,
    )
    reusable = []
    invalid = []
    fingerprints = {}
    for task in contract.tasks:
        logical_id = int(task.logical_task_id)
        assessment = assess_aimall_output(Path(task.pointdir))
        fingerprints[logical_id] = str(assessment.fingerprint_sha256)
        receipt_path = Path(task.pointdir) / AIMALL_TASK_RECEIPT
        if assessment.category == AIMALL_AUTHORITY_INVALID:
            raise ValueError(
                "AIMAll task authority is invalid for logical task "
                + str(logical_id)
                + ": "
                + str(assessment.reason)
            )
        if not assessment.valid:
            if receipt_path.exists() or receipt_path.is_symlink():
                raise ValueError(
                    "AIMAll output contradicts its published task receipt for "
                    "logical task "
                    + str(logical_id)
                    + ": "
                    + str(assessment.reason)
                )
            invalid.append(
                {
                    "task_id": logical_id,
                    "reason": str(assessment.reason),
                    "fingerprint_sha256": str(
                        assessment.fingerprint_sha256
                    ),
                }
            )
            continue
        if bool(publish_valid_receipts):
            write_quantum_task_receipt_from_postprocess_source(
                Path(task.pointdir),
                phase_name=str(phase_name),
                iteration=int(iteration),
                logical_task_id=logical_id,
                source=source,
            )
        reusable.append(logical_id)
    return {
        "logical_total": int(contract.logical_total),
        "scheduler_completed_candidates": int(contract.logical_total),
        "validated_reusable_task_ids": reusable,
        "invalid_completed_tasks": invalid,
        "output_fingerprints": fingerprints,
        "producer_job_id": str(source.get("job_id") or ""),
        "producer_submission_identity": str(
            source.get("submission_identity") or ""
        ),
        "source_sha256": str(source.get("source_sha256") or ""),
    }


__all__ = [
    "AIMALL_NATIVE_EXIT_CODE_REMAP",
    "AIMALL_STRUCTURAL_RETRY_KIND",
    "AIMALL_STRUCTURAL_RETRY_METADATA_KEY",
    "AIMALL_STRUCTURAL_VALIDATOR_KIND",
    "AIMALL_STRUCTURAL_VALIDATOR_METADATA_KEY",
    "aimall_intent_has_structural_validator_contract",
    "build_aimall_structural_validator_metadata",
    "build_aimall_structural_retry_metadata",
    "classify_aimall_postprocess_source_outputs",
    "resolve_aimall_structural_recovery_policy",
    "validate_aimall_structural_retry_metadata",
    "validate_aimall_structural_validator_metadata",
]
