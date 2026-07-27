"""Bounded recovery for AIMAll rejections caused solely by INT parsing."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..config import CampaignConfig
from ..layout import active_allocation_dir, active_iteration_dir, staging_phase_dir
from ..point_allocation import (
    allocation_manifest_sha256,
    point_allocation_path,
    read_point_allocation,
    revalidate_rejected_quantum_results,
)
from ..strict_json import load_path, strict_json as json
from ..versioning.manifest import sha256_file
from ..versioning.provenance import validate_provenance
from .array_recovery import read_array_ledger
from .completion_receipts import (
    inventory_completion_receipts,
    validate_completion_reference,
)
from .config_lock import read_config_lock
from .filesystem import campaign_owned_path
from .quantum_acceptance_receipts import (
    QUANTUM_ACCEPTANCE_RECEIPT,
    read_quantum_acceptance_receipt,
    write_quantum_acceptance_receipt,
)
from .quantum_quality import (
    evaluate_aimall_pointdir,
    read_quantum_quality_manifest,
    write_quantum_quality_manifest,
)
from .quantum_task_receipts import (
    AIMALL_TASK_RECEIPT,
    read_quantum_task_receipt,
    write_quantum_task_receipt_from_terminal_intent,
)
from .reference_commit import inventory_reference_commits, prepare_reference_data_delta
from .state import CampaignPhase, CampaignState, atomic_write_json, read_state
from .submission_intent import ACTIVE_STATUSES, inventory_intents, load_intent


AIMALL_QUALITY_REVALIDATION_SCHEMA_VERSION = 1
AIMALL_QUALITY_REVALIDATION_FILENAME = "AIMALL_QUALITY_REVALIDATION.json"
AIMALL_QUALITY_REVALIDATION_REASON = "dft_model_missing_or_unsupported"
AIMALL_QUALITY_REVALIDATION_CALIBRATION_AUDIT = (
    "ERROR_CALIBRATION_REVALIDATION_AUDIT.json"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _revalidation_root(campaign: Path, iteration: int) -> Path:
    return (
        active_allocation_dir(active_iteration_dir(campaign, int(iteration)))
        / "quality_revalidations"
    )


def _ledger_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "campaign_uid": str(payload.get("campaign_uid") or ""),
        "iteration": payload.get("iteration"),
        "phase": str(payload.get("phase") or ""),
        "allocation_path": str(payload.get("allocation_path") or ""),
        "source_allocation_generation": payload.get(
            "source_allocation_generation"
        ),
        "source_allocation_sha256": str(
            payload.get("source_allocation_sha256") or ""
        ),
        "config_lock_fingerprint_sha256": str(
            payload.get("config_lock_fingerprint_sha256") or ""
        ),
        "producer": dict(payload.get("producer") or {}),
        "source_quality_manifest": dict(
            payload.get("source_quality_manifest") or {}
        ),
        "candidates": [
            {
                key: record.get(key)
                for key in (
                    "candidate_id",
                    "slot_id",
                    "split",
                    "round",
                    "pointdir",
                    "logical_task_id",
                    "prior_reason",
                    "source_quality_record_sha256",
                    "quality_record_sha256",
                )
            }
            for record in list(payload.get("candidates") or [])
        ],
    }


def _validate_digest(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(label + " is not a lowercase SHA-256 digest")
    return text


def _validate_ledger(payload: Any, path: Path) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("AIMAll quality-revalidation ledger must be an object")
    data = dict(payload)
    if data.get("schema_version") != AIMALL_QUALITY_REVALIDATION_SCHEMA_VERSION:
        raise ValueError("unsupported AIMAll quality-revalidation schema")
    if data.get("status") not in {
        "prepared",
        "evidence_published",
        "allocation_updated",
        "calibration_updated",
        "complete",
    }:
        raise ValueError("AIMAll quality-revalidation status is invalid")
    if (
        isinstance(data.get("iteration"), bool)
        or not isinstance(data.get("iteration"), int)
        or int(data["iteration"]) < 1
    ):
        raise ValueError("AIMAll quality-revalidation iteration is invalid")
    if (
        isinstance(data.get("models_version"), bool)
        or not isinstance(data.get("models_version"), int)
        or int(data["models_version"]) < 0
    ):
        raise ValueError("AIMAll quality-revalidation model version is invalid")
    for key in ("campaign_uid", "phase", "allocation_path"):
        if not isinstance(data.get(key), str) or not str(data[key]).strip():
            raise ValueError("AIMAll quality-revalidation " + key + " is empty")
    if data["phase"] != CampaignPhase.AIMALL.value:
        raise ValueError("AIMAll quality-revalidation phase is invalid")
    for key in ("source_allocation_generation",):
        if (
            isinstance(data.get(key), bool)
            or not isinstance(data.get(key), int)
            or int(data[key]) < 0
        ):
            raise ValueError("AIMAll quality-revalidation " + key + " is invalid")
    producer = data.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("AIMAll quality-revalidation producer is missing")
    for key in ("attempt_id", "submission_identity", "job_id", "array_ledger_path"):
        if not isinstance(producer.get(key), str) or not str(producer[key]).strip():
            raise ValueError(
                "AIMAll quality-revalidation producer " + key + " is empty"
            )
    if (
        isinstance(producer.get("expected_tasks"), bool)
        or not isinstance(producer.get("expected_tasks"), int)
        or int(producer["expected_tasks"]) < 1
    ):
        raise ValueError("AIMAll quality-revalidation producer task count is invalid")
    _validate_digest(
        producer.get("array_ledger_sha256"),
        "AIMAll quality-revalidation array-ledger digest",
    )
    if not isinstance(producer.get("completion_receipt"), dict):
        raise ValueError("AIMAll quality-revalidation completion receipt is missing")
    source_quality = data.get("source_quality_manifest")
    if (
        not isinstance(source_quality, dict)
        or not isinstance(source_quality.get("path"), str)
        or not str(source_quality["path"]).strip()
    ):
        raise ValueError("AIMAll source quality-manifest binding is invalid")
    _validate_digest(
        source_quality.get("sha256"),
        "AIMAll source quality-manifest digest",
    )
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("AIMAll quality-revalidation candidates are missing")
    candidate_ids = []
    slot_ids = []
    logical_task_ids = []
    for record in candidates:
        if not isinstance(record, dict):
            raise ValueError("AIMAll quality-revalidation candidate is invalid")
        candidate_id = str(record.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("AIMAll quality-revalidation candidate ID is empty")
        candidate_ids.append(candidate_id)
        if not isinstance(record.get("split"), str) or not str(
            record["split"]
        ).strip():
            raise ValueError("AIMAll quality-revalidation candidate split is empty")
        for key in ("slot_id", "round", "logical_task_id"):
            if (
                isinstance(record.get(key), bool)
                or not isinstance(record.get(key), int)
                or int(record[key]) < 0
            ):
                raise ValueError(
                    "AIMAll quality-revalidation candidate " + key + " is invalid"
                )
        slot_ids.append(int(record["slot_id"]))
        logical_task_ids.append(int(record["logical_task_id"]))
        if not isinstance(record.get("pointdir"), str) or not str(
            record["pointdir"]
        ).strip():
            raise ValueError("AIMAll quality-revalidation pointdir is empty")
        if str(record.get("prior_reason") or "") != (
            AIMALL_QUALITY_REVALIDATION_REASON
        ):
            raise ValueError("AIMAll quality-revalidation reason is ineligible")
        _validate_digest(
            record.get("source_quality_record_sha256"),
            "original AIMAll quality-record digest",
        )
        corrected_digest = _validate_digest(
            record.get("quality_record_sha256"),
            "AIMAll quality-revalidation record digest",
        )
        quality_record = record.get("quality_record")
        if not isinstance(quality_record, dict) or _canonical_sha256(
            quality_record
        ) != corrected_digest:
            raise ValueError(
                "AIMAll quality-revalidation record content conflicts with its digest"
            )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("AIMAll quality-revalidation candidates are duplicated")
    if len(slot_ids) != len(set(slot_ids)) or len(logical_task_ids) != len(
        set(logical_task_ids)
    ):
        raise ValueError("AIMAll quality-revalidation task identities are duplicated")
    _validate_digest(
        data.get("source_allocation_sha256"),
        "AIMAll quality-revalidation allocation digest",
    )
    _validate_digest(
        data.get("config_lock_fingerprint_sha256"),
        "AIMAll quality-revalidation config digest",
    )
    transaction_id = _validate_digest(
        data.get("transaction_id"),
        "AIMAll quality-revalidation transaction ID",
    )
    if transaction_id != _canonical_sha256(_ledger_identity(data)):
        raise ValueError("AIMAll quality-revalidation identity mismatch")
    status_order = {
        "prepared": 0,
        "evidence_published": 1,
        "allocation_updated": 2,
        "calibration_updated": 3,
        "complete": 4,
    }
    status_rank = status_order[str(data["status"])]
    if status_rank >= 1:
        corrected = data.get("corrected_quality_manifest")
        if (
            not isinstance(corrected, dict)
            or not isinstance(corrected.get("path"), str)
            or not str(corrected["path"]).strip()
        ):
            raise ValueError("corrected AIMAll quality-manifest binding is missing")
        _validate_digest(
            corrected.get("sha256"),
            "corrected AIMAll quality-manifest digest",
        )
        corrections = data.get("corrections")
        if not isinstance(corrections, list) or len(corrections) != len(candidates):
            raise ValueError("AIMAll quality-revalidation corrections are incomplete")
    if status_rank >= 2:
        if (
            isinstance(data.get("allocation_generation"), bool)
            or not isinstance(data.get("allocation_generation"), int)
            or int(data["allocation_generation"])
            <= int(data["source_allocation_generation"])
        ):
            raise ValueError("revalidated allocation generation is invalid")
        _validate_digest(
            data.get("allocation_sha256"),
            "revalidated point-allocation digest",
        )
    if status_rank >= 3:
        calibration = data.get("calibration")
        if not isinstance(calibration, dict) or calibration.get("status") not in {
            "updated",
            "failed_nonfatal",
        }:
            raise ValueError("AIMAll quality-revalidation calibration result is invalid")
    if status_rank >= 4:
        reference_ledger = data.get("reference_commit_ledger")
        reference_path = Path(str(reference_ledger or ""))
        if (
            not isinstance(reference_ledger, str)
            or not reference_ledger
            or reference_path.is_absolute()
            or any(part in {".", ".."} for part in reference_path.parts)
        ):
            raise ValueError("AIMAll reference-commit ledger binding is invalid")
    return data


def read_aimall_quality_revalidation(path: Path) -> Dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError("AIMAll quality-revalidation ledger is not a regular file")
    return _validate_ledger(load_path(target), target)


def inventory_aimall_quality_revalidations(
    campaign_dir: Path,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir).resolve()
    records = []
    errors = []
    active_root = campaign / "ACTIVE_LEARNING"
    if not active_root.is_dir() or active_root.is_symlink():
        return {"records": records, "errors": errors}
    pattern = (
        "iteration-*/allocation/quality_revalidations/*/"
        + AIMALL_QUALITY_REVALIDATION_FILENAME
    )
    for path in sorted(active_root.glob(pattern)):
        try:
            resolved = campaign_owned_path(campaign, path)
            payload = read_aimall_quality_revalidation(resolved)
            records.append(
                {
                    "path": resolved.relative_to(campaign).as_posix(),
                    "payload": payload,
                    "sha256": sha256_file(resolved),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "path": str(path),
                    "error": type(exc).__name__ + ": " + str(exc),
                }
            )
    return {"records": records, "errors": errors}


def _iteration_ledgers(campaign: Path, iteration: int) -> list[Dict[str, Any]]:
    inventory = inventory_aimall_quality_revalidations(campaign)
    if inventory["errors"]:
        raise ValueError(
            "invalid AIMAll quality-revalidation evidence: "
            + str(inventory["errors"][0]["error"])
        )
    return [
        record
        for record in inventory["records"]
        if int(record["payload"]["iteration"]) == int(iteration)
    ]


def _vacant_parser_rejections(allocation: Mapping[str, Any]) -> list[Dict[str, Any]]:
    summary = dict(allocation.get("summary") or {})
    if (
        bool(summary.get("complete", False))
        or int(summary.get("pending_slots", -1)) != 0
        or int(summary.get("deficit_total", 0)) <= 0
    ):
        return []
    candidates: list[Dict[str, Any]] = []
    for slot in list(allocation.get("slots") or []):
        if slot.get("accepted_attempt") is not None:
            continue
        attempts = list(slot.get("attempts") or [])
        if not attempts:
            return []
        attempt = dict(attempts[-1])
        if (
            str(attempt.get("status") or "") != "rejected"
            or str(attempt.get("reason") or "")
            != AIMALL_QUALITY_REVALIDATION_REASON
        ):
            return []
        candidates.append(
            {
                **attempt,
                "slot_id": int(slot["slot_id"]),
                "split": str(slot["split"]),
            }
        )
    if len(candidates) != int(summary.get("deficit_total", -1)):
        return []
    return candidates


def has_aimall_quality_revalidation_candidates(campaign_dir: Path) -> bool:
    """Return whether the halted allocation has only exact parser rejections."""
    campaign = Path(campaign_dir)
    try:
        state = read_state(
            campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
        )
        if state.phase is not CampaignPhase.HALTED or int(state.iteration) < 1:
            return False
        context = state.lifecycle_context or {}
        if (
            context.get("reason_code") != "replacement_reserve_exhausted"
            or context.get("from_phase") != CampaignPhase.ALLOCATION_CHECK.value
        ):
            return False
        allocation = read_point_allocation(
            point_allocation_path(
                campaign,
                context="active",
                iteration=int(state.iteration),
            ),
            expected_campaign_uid=str(state.campaign_uid),
            expected_context="active",
            expected_iteration=int(state.iteration),
        )
        return bool(_vacant_parser_rejections(allocation))
    except Exception:
        return False


def _points_membership(staging: Path) -> list[str]:
    points = staging / "POINTS.txt"
    if points.is_symlink() or not points.is_file():
        raise ValueError("AIMAll POINTS.txt is missing or symlinked")
    names = []
    for line_number, raw in enumerate(
        points.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = raw.strip()
        if not text:
            continue
        path = Path(text)
        expected = staging / path.name
        if path.resolve(strict=False) != expected.resolve(strict=False):
            raise ValueError(
                "AIMAll POINTS.txt entry is not canonical at line "
                + str(line_number)
            )
        names.append(path.name)
    if not names or len(names) != len(set(names)):
        raise ValueError("AIMAll POINTS.txt membership is empty or duplicated")
    return names


def _matching_completion_receipt(
    campaign: Path,
    *,
    state: CampaignState,
    intent: Mapping[str, Any],
    expected_tasks: int,
) -> Dict[str, Any]:
    inventory = inventory_completion_receipts(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    if inventory["errors"]:
        raise ValueError("AIMAll completion-receipt inventory is invalid")
    matches = []
    for record in inventory["records"]:
        payload = record.get("payload") or {}
        if (
            payload.get("phase") == CampaignPhase.AIMALL.value
            and int(payload.get("iteration", -1)) == int(state.iteration)
            and payload.get("next_phase") == CampaignPhase.ALLOCATION_CHECK.value
            and str(payload.get("job_id") or "") == str(intent.get("job_id") or "")
            and str(payload.get("submission_identity") or "")
            == str(intent.get("submission_identity") or "")
            and int(payload.get("expected_tasks", -1)) == int(expected_tasks)
        ):
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            "AIMAll revalidation requires exactly one matching completion receipt"
        )
    validate_completion_reference(
        campaign,
        matches[0]["reference"],
        expected_campaign_uid=str(state.campaign_uid),
    )
    return dict(matches[0])


def inspect_aimall_quality_revalidation(
    campaign_dir: Path,
    *,
    config: Optional[CampaignConfig] = None,
) -> Dict[str, Any]:
    """Inspect only the bounded rejected point set and return a write-free verdict."""
    campaign = Path(campaign_dir).resolve()
    result: Dict[str, Any] = {
        "state": "not_applicable",
        "eligible": False,
        "iteration": None,
        "candidate_count": 0,
        "old_reason": AIMALL_QUALITY_REVALIDATION_REASON,
        "all_accepted_on_revalidation": False,
        "no_slurm_submission": True,
        "reason": "campaign is not at the parser-revalidation boundary",
    }
    try:
        state = read_state(
            campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
        )
        result["iteration"] = int(state.iteration)
        iteration_ledgers = _iteration_ledgers(campaign, int(state.iteration))
        complete_matches = [
            record
            for record in iteration_ledgers
            if str(record["payload"]["status"]) == "complete"
        ]
        if len(complete_matches) > 1:
            raise ValueError("multiple completed AIMAll quality revalidations exist")
        complete = complete_matches[0] if complete_matches else None
        if complete is not None:
            result.update(
                {
                    "state": "complete",
                    "eligible": False,
                    "candidate_count": len(
                        complete["payload"].get("candidates") or []
                    ),
                    "all_accepted_on_revalidation": True,
                    "ledger_path": str(complete["path"]),
                    "reason": "AIMAll quality revalidation is already complete",
                }
            )
            return result
        incomplete = [
            record
            for record in iteration_ledgers
            if str(record["payload"]["status"]) != "complete"
        ]
        if len(incomplete) > 1:
            raise ValueError("multiple AIMAll quality revalidations are incomplete")
        if incomplete:
            payload = incomplete[0]["payload"]
            result.update(
                {
                    "state": "resumable",
                    "eligible": True,
                    "candidate_count": len(payload.get("candidates") or []),
                    "all_accepted_on_revalidation": True,
                    "ledger_path": str(incomplete[0]["path"]),
                    "transaction_id": str(payload["transaction_id"]),
                    "allocation_generation": payload.get(
                        "allocation_generation"
                    ),
                    "reason": "an interrupted AIMAll quality revalidation can resume",
                }
            )
            return result
        if state.phase is not CampaignPhase.HALTED or int(state.iteration) < 1:
            return result
        lifecycle = state.lifecycle_context or {}
        if (
            lifecycle.get("reason_code") != "replacement_reserve_exhausted"
            or lifecycle.get("from_phase") != CampaignPhase.ALLOCATION_CHECK.value
        ):
            return result
        if any(bool(value) for value in state.pending_jobs.values()):
            raise ValueError("campaign state still records scheduler ownership")
        intent_inventory = inventory_intents(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        if intent_inventory["errors"]:
            raise ValueError("submission-intent inventory is invalid")
        active = [
            record
            for record in intent_inventory["records"]
            if str(record.get("status") or "") in ACTIVE_STATUSES
        ]
        if active:
            raise ValueError("an active submission intent still exists")
        reference_transactions = inventory_reference_commits(
            campaign,
            verification="authority",
        )
        incomplete_reference_transactions = [
            record
            for record in reference_transactions
            if str(record.get("state") or "") != "complete"
        ]
        if incomplete_reference_transactions:
            raise ValueError(
                "a reference-data publication transaction already exists"
            )

        allocation_path = point_allocation_path(
            campaign,
            context="active",
            iteration=int(state.iteration),
        )
        allocation = read_point_allocation(
            allocation_path,
            expected_campaign_uid=str(state.campaign_uid),
            expected_context="active",
            expected_iteration=int(state.iteration),
        )
        candidates = _vacant_parser_rejections(allocation)
        if not candidates:
            return result
        if any(int(record.get("round", -1)) != 0 for record in candidates):
            raise ValueError(
                "AIMAll parser revalidation is limited to one original AIMAll batch"
            )
        config = config or CampaignConfig.from_yaml(campaign / "campaign.yaml")
        lock = read_config_lock(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        locked_config = dict(lock["canonical_config"])
        current_config = config.to_dict()
        if locked_config.get("quality_gates") != current_config.get("quality_gates"):
            raise ValueError("AIMAll quality gates differ from the locked campaign")
        if (locked_config.get("gaussian") or {}).get("method") != (
            current_config.get("gaussian") or {}
        ).get("method"):
            raise ValueError("Gaussian method differs from the locked campaign")

        staging = campaign_owned_path(
            campaign,
            staging_phase_dir(campaign, CampaignPhase.AIMALL.value, state.iteration),
        )
        point_names = _points_membership(staging)
        intent = load_intent(
            campaign,
            CampaignPhase.AIMALL.value,
            int(state.iteration),
            expected_campaign_uid=str(state.campaign_uid),
        )
        if (
            not isinstance(intent, dict)
            or intent.get("status") != "COMPLETED"
            or not intent.get("job_id")
            or int(intent.get("expected_tasks", -1)) != len(point_names)
        ):
            raise ValueError("terminal AIMAll producer intent is not authoritative")
        completion = _matching_completion_receipt(
            campaign,
            state=state,
            intent=intent,
            expected_tasks=len(point_names),
        )
        ledger = read_array_ledger(
            campaign,
            CampaignPhase.AIMALL.value,
            int(state.iteration),
        )
        if ledger is None or int(ledger.get("logical_total", -1)) != len(point_names):
            raise ValueError("AIMAll array ledger does not match POINTS.txt")
        ledger_tasks = {
            int(record["task_id"]): record for record in list(ledger.get("tasks") or [])
        }
        if set(ledger_tasks) != set(range(len(point_names))):
            raise ValueError("AIMAll array ledger task membership is incomplete")

        old_quality_paths = {
            str(record.get("quality_manifest") or "") for record in candidates
        }
        if len(old_quality_paths) != 1:
            raise ValueError("parser rejections do not share one quality manifest")
        old_quality_path = campaign_owned_path(
            campaign,
            campaign / Path(next(iter(old_quality_paths))),
        )
        old_quality = read_quantum_quality_manifest(
            old_quality_path.parent,
            expected_phase=CampaignPhase.AIMALL.value,
            expected_iteration=int(state.iteration),
            expected_method=str(config.gaussian.method),
            manifest_path=old_quality_path,
        )
        old_by_name = {
            str(record["pointdir"]): record for record in old_quality["records"]
        }

        from ichor.core.files import PointDirectory

        inspected = []
        for candidate in sorted(candidates, key=lambda item: int(item["slot_id"])):
            pointdir = campaign_owned_path(campaign, Path(str(candidate["pointdir"])))
            if pointdir.parent != staging or pointdir.name not in point_names:
                raise ValueError("rejected pointdir is outside canonical AIMAll staging")
            logical_task_id = point_names.index(pointdir.name)
            task_metadata = load_path(pointdir / "AIMALL_TASK.json")
            if (
                not isinstance(task_metadata, dict)
                or task_metadata.get("pointdir") != pointdir.name
                or int(task_metadata.get("task_index", -1)) != logical_task_id
            ):
                raise ValueError("AIMAll task metadata does not match POINTS.txt")
            ledger_output = Path(
                str(ledger_tasks[logical_task_id].get("output_path") or "")
            )
            if ledger_output.resolve(strict=False) != pointdir.resolve(strict=False):
                raise ValueError("AIMAll array ledger output path mismatch")
            validate_provenance(
                pointdir,
                campaign_uid=str(state.campaign_uid),
                iteration=int(state.iteration),
                allocation_candidate_id=str(candidate["candidate_id"]),
                allocation_context="active",
                allocation_slot_id=int(candidate["slot_id"]),
                allocation_split=str(candidate["split"]),
                allocation_slot_assignment_sha256=str(
                    allocation["slot_assignment_sha256"]
                ),
            )
            old_record = old_by_name.get(pointdir.name)
            if (
                not isinstance(old_record, dict)
                or old_record.get("accepted") is not False
                or list(old_record.get("reasons") or [])
                != [AIMALL_QUALITY_REVALIDATION_REASON]
            ):
                raise ValueError(
                    "original AIMAll quality evidence is not the exact parser rejection"
                )
            quality_record = evaluate_aimall_pointdir(
                PointDirectory(pointdir),
                config.quality_gates,
                expected_method=str(config.gaussian.method),
            )
            inspected.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "slot_id": int(candidate["slot_id"]),
                    "split": str(candidate["split"]),
                    "round": int(candidate.get("round", 0)),
                    "pointdir": pointdir.relative_to(campaign).as_posix(),
                    "logical_task_id": int(logical_task_id),
                    "prior_reason": AIMALL_QUALITY_REVALIDATION_REASON,
                    "source_quality_record_sha256": _canonical_sha256(
                        old_record
                    ),
                    "quality_record": quality_record,
                    "quality_record_sha256": _canonical_sha256(quality_record),
                }
            )
        if not all(bool(record["quality_record"].get("accepted")) for record in inspected):
            raise ValueError("one or more rejected points still fail scientific quality gates")
        if len(inspected) != int(allocation["summary"]["deficit_total"]):
            raise ValueError("revalidation would not close the complete allocation deficit")

        source_quality = {
            "path": old_quality_path.relative_to(campaign).as_posix(),
            "sha256": sha256_file(old_quality_path),
        }
        producer = {
            "phase": CampaignPhase.AIMALL.value,
            "iteration": int(state.iteration),
            "attempt_id": str(intent["attempt_id"]),
            "submission_identity": str(intent["submission_identity"]),
            "job_id": str(intent["job_id"]),
            "expected_tasks": int(intent["expected_tasks"]),
            "completion_receipt": dict(completion["reference"]),
            "array_ledger_path": Path(str(ledger["path"])).resolve().relative_to(
                campaign
            ).as_posix(),
            "array_ledger_sha256": sha256_file(Path(str(ledger["path"]))),
        }
        identity_payload = {
            "campaign_uid": str(state.campaign_uid),
            "iteration": int(state.iteration),
            "phase": CampaignPhase.AIMALL.value,
            "allocation_path": allocation_path.relative_to(campaign).as_posix(),
            "source_allocation_generation": int(allocation["generation"]),
            "source_allocation_sha256": allocation_manifest_sha256(allocation_path),
            "config_lock_fingerprint_sha256": str(lock["fingerprint_sha256"]),
            "producer": producer,
            "source_quality_manifest": source_quality,
            "candidates": inspected,
        }
        transaction_id = _canonical_sha256(_ledger_identity(identity_payload))
        ledger_path = (
            _revalidation_root(campaign, int(state.iteration))
            / transaction_id[:24]
            / AIMALL_QUALITY_REVALIDATION_FILENAME
        )
        result.update(
            {
                "state": "eligible",
                "eligible": True,
                "candidate_count": len(inspected),
                "all_accepted_on_revalidation": True,
                "reason": "all parser-rejected AIMAll points pass current locked gates",
                "allocation_generation": int(allocation["generation"]),
                "transaction_id": transaction_id,
                "ledger_path": ledger_path.relative_to(campaign).as_posix(),
                "inspection": identity_payload,
                "models_version": int(state.models_version),
            }
        )
        return result
    except Exception as exc:
        result.update(
            {
                "state": "ineligible",
                "eligible": False,
                "reason": type(exc).__name__ + ": " + str(exc),
            }
        )
        return result


def _write_ledger(path: Path, payload: Dict[str, Any], status: str) -> None:
    payload["status"] = str(status)
    payload["updated_at_iso"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def apply_aimall_quality_revalidation(
    campaign_dir: Path,
    *,
    config: CampaignConfig,
) -> Dict[str, Any]:
    """Apply or resume the bounded parser correction transaction."""
    campaign = Path(campaign_dir).resolve()
    existing_ledgers = inventory_aimall_quality_revalidations(campaign)
    if existing_ledgers["errors"]:
        raise ValueError("existing AIMAll quality-revalidation evidence is invalid")
    incomplete = [
        record
        for record in existing_ledgers["records"]
        if str(record["payload"].get("status")) != "complete"
    ]
    if len(incomplete) > 1:
        raise ValueError("another AIMAll quality-revalidation transaction is incomplete")
    if incomplete:
        ledger_path = campaign / str(incomplete[0]["path"])
        ledger = dict(incomplete[0]["payload"])
        transaction_id = str(ledger["transaction_id"])
        inspection = {
            "state": "resumable",
            "eligible": True,
            "transaction_id": transaction_id,
            "models_version": int(ledger["models_version"]),
        }
    else:
        inspection = inspect_aimall_quality_revalidation(campaign, config=config)
        if inspection.get("state") == "complete":
            return inspection
        if not bool(inspection.get("eligible", False)):
            raise ValueError(
                str(inspection.get("reason") or "revalidation is ineligible")
            )
        source = dict(inspection["inspection"])
        transaction_id = str(inspection["transaction_id"])
        ledger_path = campaign / str(inspection["ledger_path"])
    if ledger_path.is_file():
        ledger = read_aimall_quality_revalidation(ledger_path)
        if str(ledger["transaction_id"]) != transaction_id:
            raise ValueError("AIMAll quality-revalidation transaction conflicts")
    else:
        ledger = {
            "schema_version": AIMALL_QUALITY_REVALIDATION_SCHEMA_VERSION,
            "transaction_id": transaction_id,
            **source,
            "models_version": int(inspection["models_version"]),
            "status": "prepared",
            "created_at_iso": _now_iso(),
            "updated_at_iso": _now_iso(),
            "corrected_quality_manifest": None,
            "allocation_generation": None,
            "calibration": None,
            "reference_commit_ledger": None,
        }
        _write_ledger(ledger_path, ledger, "prepared")

    lock = read_config_lock(
        campaign,
        expected_campaign_uid=str(ledger["campaign_uid"]),
    )
    if str(lock["fingerprint_sha256"]) != str(
        ledger["config_lock_fingerprint_sha256"]
    ):
        raise ValueError("campaign config lock changed during AIMAll revalidation")
    current_config = config.to_dict()
    locked_config = dict(lock["canonical_config"])
    if locked_config.get("quality_gates") != current_config.get("quality_gates"):
        raise ValueError("AIMAll quality gates differ from the revalidation ledger")
    if (locked_config.get("gaussian") or {}).get("method") != (
        current_config.get("gaussian") or {}
    ).get("method"):
        raise ValueError("Gaussian method differs from the revalidation ledger")
    source_quality_path = campaign_owned_path(
        campaign,
        campaign / str(ledger["source_quality_manifest"]["path"]),
    )
    if sha256_file(source_quality_path) != str(
        ledger["source_quality_manifest"]["sha256"]
    ):
        raise ValueError("original AIMAll quality manifest changed during revalidation")
    validate_completion_reference(
        campaign,
        ledger["producer"]["completion_receipt"],
        expected_campaign_uid=str(ledger["campaign_uid"]),
    )
    array_ledger_path = campaign_owned_path(
        campaign,
        campaign / str(ledger["producer"]["array_ledger_path"]),
    )
    if sha256_file(array_ledger_path) != str(
        ledger["producer"]["array_ledger_sha256"]
    ):
        raise ValueError("AIMAll array ledger changed during revalidation")

    records = [dict(record["quality_record"]) for record in ledger["candidates"]]
    corrected_quality_path = ledger_path.parent / "quantum_quality.json"
    if not corrected_quality_path.is_file():
        write_quantum_quality_manifest(
            corrected_quality_path.parent,
            phase_name=str(ledger["phase"]),
            iteration=int(ledger["iteration"]),
            records=records,
            gates=config.quality_gates,
            manifest_path=corrected_quality_path,
        )
    corrected_quality = read_quantum_quality_manifest(
        corrected_quality_path.parent,
        expected_phase=str(ledger["phase"]),
        expected_iteration=int(ledger["iteration"]),
        expected_pointdirs=[str(record["pointdir"]) for record in records],
        expected_method=str(config.gaussian.method),
        manifest_path=corrected_quality_path,
    )
    if [_canonical_sha256(record) for record in corrected_quality["records"]] != [
        str(record["quality_record_sha256"]) for record in ledger["candidates"]
    ]:
        raise ValueError("corrected AIMAll quality manifest conflicts with inspection")

    intent = load_intent(
        campaign,
        str(ledger["phase"]),
        int(ledger["iteration"]),
        expected_campaign_uid=str(ledger["campaign_uid"]),
    )
    if not isinstance(intent, dict):
        raise ValueError("terminal AIMAll producer intent disappeared")
    producer = dict(ledger["producer"])
    if (
        str(intent.get("status") or "") != "COMPLETED"
        or str(intent.get("attempt_id") or "") != str(producer["attempt_id"])
        or str(intent.get("submission_identity") or "")
        != str(producer["submission_identity"])
        or str(intent.get("job_id") or "") != str(producer["job_id"])
        or int(intent.get("expected_tasks", -1)) != int(producer["expected_tasks"])
    ):
        raise ValueError("terminal AIMAll producer intent changed during revalidation")
    allocation_path = campaign_owned_path(
        campaign,
        campaign / str(ledger["allocation_path"]),
    )
    current_allocation = read_point_allocation(
        allocation_path,
        expected_campaign_uid=str(ledger["campaign_uid"]),
        expected_context="active",
        expected_iteration=int(ledger["iteration"]),
    )
    matching_source_batches = [
        record
        for record in list(current_allocation.get("applied_quantum_batches") or [])
        if str(record.get("batch_identity") or "") == transaction_id
    ]
    if not matching_source_batches:
        if (
            int(current_allocation["generation"])
            != int(ledger["source_allocation_generation"])
            or allocation_manifest_sha256(allocation_path)
            != str(ledger["source_allocation_sha256"])
        ):
            raise ValueError("point allocation changed before AIMAll revalidation")
    elif (
        len(matching_source_batches) != 1
        or str(matching_source_batches[0].get("kind") or "")
        != "quality_revalidation"
    ):
        raise ValueError("point allocation has conflicting revalidation evidence")
    if ledger.get("allocation_sha256") is not None and (
        allocation_manifest_sha256(allocation_path)
        != str(ledger.get("allocation_sha256") or "")
    ):
        raise ValueError("revalidated point allocation changed after publication")
    assignment_sha256 = str(current_allocation["slot_assignment_sha256"])
    staging = campaign_owned_path(
        campaign,
        staging_phase_dir(
            campaign,
            CampaignPhase.AIMALL.value,
            int(ledger["iteration"]),
        ),
    )
    point_names = _points_membership(staging)
    corrections = []
    pointdirs = []
    from ichor.core.files import PointDirectory

    for candidate, quality_record in zip(ledger["candidates"], records):
        pointdir = campaign_owned_path(campaign, campaign / str(candidate["pointdir"]))
        if (
            pointdir.parent != staging
            or pointdir.name not in point_names
            or point_names.index(pointdir.name)
            != int(candidate["logical_task_id"])
        ):
            raise ValueError("AIMAll point membership changed during revalidation")
        validate_provenance(
            pointdir,
            campaign_uid=str(ledger["campaign_uid"]),
            iteration=int(ledger["iteration"]),
            allocation_candidate_id=str(candidate["candidate_id"]),
            allocation_context="active",
            allocation_slot_id=int(candidate["slot_id"]),
            allocation_split=str(candidate["split"]),
            allocation_slot_assignment_sha256=assignment_sha256,
        )
        observed_quality = evaluate_aimall_pointdir(
            PointDirectory(pointdir),
            config.quality_gates,
            expected_method=str(config.gaussian.method),
        )
        if _canonical_sha256(observed_quality) != str(
            candidate["quality_record_sha256"]
        ):
            raise ValueError(
                "AIMAll point quality changed after revalidation inspection: "
                + pointdir.name
            )
        pointdirs.append(pointdir)
        task_receipt = pointdir / AIMALL_TASK_RECEIPT
        if task_receipt.is_file():
            task_payload = read_quantum_task_receipt(
                pointdir,
                phase_name=str(ledger["phase"]),
                iteration=int(ledger["iteration"]),
                logical_task_id=int(candidate["logical_task_id"]),
            )
            if (
                str(task_payload["attempt_id"]) != str(intent["attempt_id"])
                or str(task_payload["submission_identity"])
                != str(intent["submission_identity"])
                or str(task_payload["job_id"]) != str(intent["job_id"])
            ):
                raise ValueError("existing AIMAll task receipt has wrong producer identity")
        else:
            write_quantum_task_receipt_from_terminal_intent(
                pointdir,
                phase_name=str(ledger["phase"]),
                iteration=int(ledger["iteration"]),
                logical_task_id=int(candidate["logical_task_id"]),
                intent=intent,
            )
        acceptance_path = pointdir / QUANTUM_ACCEPTANCE_RECEIPT
        if acceptance_path.is_file():
            acceptance = read_quantum_acceptance_receipt(
                campaign,
                pointdir,
                expected_campaign_uid=str(ledger["campaign_uid"]),
                expected_phase=str(ledger["phase"]),
                expected_iteration=int(ledger["iteration"]),
                expected_candidate_id=str(candidate["candidate_id"]),
                expected_assignment_sha256=str(
                    assignment_sha256
                ),
            )
        else:
            write_quantum_acceptance_receipt(
                campaign,
                pointdir,
                phase_name=str(ledger["phase"]),
                iteration=int(ledger["iteration"]),
                quality_manifest=corrected_quality_path,
                quality_record=quality_record,
            )
            acceptance = read_quantum_acceptance_receipt(
                campaign,
                pointdir,
                expected_campaign_uid=str(ledger["campaign_uid"]),
                expected_phase=str(ledger["phase"]),
                expected_iteration=int(ledger["iteration"]),
                expected_candidate_id=str(candidate["candidate_id"]),
                expected_assignment_sha256=assignment_sha256,
            )
        quality_binding = dict(acceptance.get("quality_manifest") or {})
        if (
            str(quality_binding.get("path") or "")
            != corrected_quality_path.relative_to(campaign).as_posix()
            or str(quality_binding.get("sha256") or "")
            != sha256_file(corrected_quality_path)
            or str(acceptance.get("quality_record_sha256") or "")
            != str(candidate["quality_record_sha256"])
        ):
            raise ValueError(
                "AIMAll acceptance receipt does not bind corrected quality evidence"
            )
        corrections.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "accepted": True,
                "prior_reason": AIMALL_QUALITY_REVALIDATION_REASON,
                "pointdir": str(pointdir),
                "quality_manifest": corrected_quality_path.relative_to(
                    campaign
                ).as_posix(),
                "quantum_acceptance_receipt": acceptance_path.relative_to(
                    campaign
                ).as_posix(),
                "quantum_acceptance_receipt_sha256": sha256_file(acceptance_path),
                "accepted_pointdir_content_sha256": str(
                    acceptance["content_sha256"]
                ),
            }
        )
    ledger["corrected_quality_manifest"] = {
        "path": corrected_quality_path.relative_to(campaign).as_posix(),
        "sha256": sha256_file(corrected_quality_path),
    }
    ledger["corrections"] = corrections
    _write_ledger(ledger_path, ledger, "evidence_published")

    result_fingerprint = _canonical_sha256(corrections)
    current = read_point_allocation(
        allocation_path,
        expected_campaign_uid=str(ledger["campaign_uid"]),
        expected_context="active",
        expected_iteration=int(ledger["iteration"]),
    )
    matching_batches = [
        record
        for record in list(current.get("applied_quantum_batches") or [])
        if str(record.get("batch_identity") or "") == transaction_id
    ]
    if matching_batches:
        if (
            len(matching_batches) != 1
            or str(matching_batches[0].get("result_fingerprint") or "")
            != result_fingerprint
            or str(matching_batches[0].get("kind") or "")
            != "quality_revalidation"
        ):
            raise ValueError("applied AIMAll quality revalidation conflicts")
        updated = current
    else:
        if int(current["generation"]) != int(ledger["source_allocation_generation"]):
            raise ValueError("point allocation changed before quality revalidation")
        updated = revalidate_rejected_quantum_results(
            allocation_path,
            corrections,
            expected_generation=int(ledger["source_allocation_generation"]),
            batch_identity=transaction_id,
            result_fingerprint=result_fingerprint,
        )
    if not bool(updated["summary"]["complete"]):
        raise ValueError("AIMAll quality revalidation did not complete the allocation")
    ledger["allocation_generation"] = int(updated["generation"])
    ledger["allocation_sha256"] = allocation_manifest_sha256(allocation_path)
    _write_ledger(ledger_path, ledger, "allocation_updated")

    try:
        from .error_calibration import update_from_aimall_acceptance

        calibration = update_from_aimall_acceptance(
            campaign_dir=campaign,
            iter_dir=active_iteration_dir(campaign, int(ledger["iteration"])),
            config=config,
            iteration=int(ledger["iteration"]),
            models_version=int(inspection["models_version"]),
            accepted_pointdirs=pointdirs,
            quality_records=records,
            audit_filename=AIMALL_QUALITY_REVALIDATION_CALIBRATION_AUDIT,
        )
        ledger["calibration"] = {"status": "updated", "audit": calibration}
    except Exception as exc:
        try:
            from .error_calibration import mark_calibration_model_stale

            mark_calibration_model_stale(
                campaign,
                reason=type(exc).__name__ + ": " + str(exc)[:240],
                iteration=int(ledger["iteration"]),
            )
        except Exception:
            pass
        ledger["calibration"] = {
            "status": "failed_nonfatal",
            "error": type(exc).__name__ + ": " + str(exc)[:240],
        }
    _write_ledger(ledger_path, ledger, "calibration_updated")

    reference_ledger = prepare_reference_data_delta(
        campaign,
        reference_data_version=int(ledger["iteration"]),
        context="active",
        iteration=int(ledger["iteration"]),
        expected_campaign_uid=str(ledger["campaign_uid"]),
    )
    ledger["reference_commit_ledger"] = reference_ledger.relative_to(
        campaign
    ).as_posix()
    _write_ledger(ledger_path, ledger, "complete")
    return {
        "state": "complete",
        "eligible": False,
        "iteration": int(ledger["iteration"]),
        "candidate_count": len(ledger["candidates"]),
        "old_reason": AIMALL_QUALITY_REVALIDATION_REASON,
        "all_accepted_on_revalidation": True,
        "no_slurm_submission": True,
        "allocation_generation": int(ledger["allocation_generation"]),
        "ledger_path": ledger_path.relative_to(campaign).as_posix(),
        "reference_commit_ledger": ledger["reference_commit_ledger"],
        "reason": "AIMAll quality revalidation completed without scheduler work",
    }


__all__ = [
    "AIMALL_QUALITY_REVALIDATION_FILENAME",
    "AIMALL_QUALITY_REVALIDATION_REASON",
    "AIMALL_QUALITY_REVALIDATION_SCHEMA_VERSION",
    "apply_aimall_quality_revalidation",
    "has_aimall_quality_revalidation_candidates",
    "inspect_aimall_quality_revalidation",
    "inventory_aimall_quality_revalidations",
    "read_aimall_quality_revalidation",
]
