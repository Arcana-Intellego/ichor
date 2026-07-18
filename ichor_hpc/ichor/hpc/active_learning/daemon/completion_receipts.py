"""Durable write-ahead receipts for daemon phase completion."""
from __future__ import annotations

import hashlib
from ..strict_json import strict_json as json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from ..versioning.manifest import sha256_file
from .state import CampaignState, atomic_write_json
from .filesystem import operational_path


COMPLETION_RECEIPT_SCHEMA_VERSION = 2
COMPLETION_RECEIPT_DIRNAME = "phase_completions"


class CompletionReceiptError(ValueError):
    """Raised when a phase-completion receipt cannot be trusted."""


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompletionReceiptError(label + " must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise CompletionReceiptError(label + " must be >= " + str(minimum))
    return parsed


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CompletionReceiptError(label + " must be a lowercase SHA-256 digest")
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_iso_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompletionReceiptError(label + " must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CompletionReceiptError(label + " must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CompletionReceiptError(label + " must include a timezone")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def state_projection(value: Union[CampaignState, Mapping[str, Any]]) -> Dict[str, Any]:
    payload = value.to_dict() if isinstance(value, CampaignState) else dict(value)
    payload.pop("last_completion_receipt", None)
    return payload


def receipt_dir(campaign_dir: Union[str, Path]) -> Path:
    return operational_path(campaign_dir, COMPLETION_RECEIPT_DIRNAME)


def _inside_campaign(campaign: Path, path: Path) -> Path:
    root = campaign.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise CompletionReceiptError("completion evidence is outside campaign: " + str(path))
    return resolved


def evidence_records(
    campaign_dir: Union[str, Path],
    paths: Iterable[Union[str, Path]],
) -> list[Dict[str, Any]]:
    campaign = Path(campaign_dir)
    root = campaign.resolve()
    records: list[Dict[str, Any]] = []
    seen = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise CompletionReceiptError(
                "required completion evidence is missing: " + str(path)
            )
        if path.is_symlink() or not path.is_file():
            raise CompletionReceiptError("completion evidence is not a regular file: " + str(path))
        resolved = _inside_campaign(campaign, path)
        relative = resolved.relative_to(root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        records.append(
            {
                "path": relative,
                "size": int(resolved.stat().st_size),
                "sha256": sha256_file(resolved),
            }
        )
    records.sort(key=lambda item: str(item["path"]))
    return records


def write_completion_receipt(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    phase: str,
    iteration: int,
    replacement_round: int,
    config_sha256: str,
    state_before: Union[CampaignState, Mapping[str, Any]],
    state_after: Union[CampaignState, Mapping[str, Any]],
    next_phase: str,
    next_iteration: int,
    state_updates: Mapping[str, Any],
    evidence: Iterable[Mapping[str, Any]],
    job_id: Optional[str] = None,
    expected_tasks: Optional[int] = None,
    submission_identity: Optional[str] = None,
) -> Path:
    if not isinstance(campaign_uid, str) or not campaign_uid:
        raise CompletionReceiptError("completion receipt campaign_uid is empty")
    if not isinstance(phase, str) or not phase:
        raise CompletionReceiptError("completion receipt phase is empty")
    if not isinstance(next_phase, str) or not next_phase:
        raise CompletionReceiptError("completion receipt next_phase is empty")
    iteration_value = _exact_int(iteration, "completion receipt iteration")
    replacement_value = _exact_int(
        replacement_round,
        "completion receipt replacement_round",
    )
    next_iteration_value = _exact_int(
        next_iteration,
        "completion receipt next_iteration",
    )
    expected_tasks_value = None
    if expected_tasks is not None:
        expected_tasks_value = _exact_int(
            expected_tasks,
            "completion receipt expected_tasks",
            minimum=1,
        )
    _sha256(config_sha256, "completion receipt config_sha256")
    before_state = state_projection(state_before)
    after_state = state_projection(state_after)
    CampaignState.from_dict(after_state)
    before_digest = canonical_sha256(before_state)
    after_digest = canonical_sha256(after_state)
    identity = {
        "campaign_uid": campaign_uid,
        "phase": phase,
        "iteration": iteration_value,
        "replacement_round": replacement_value,
        "job_id": None if job_id is None else str(job_id),
        "submission_identity": (
            None if submission_identity is None else str(submission_identity)
        ),
        "config_sha256": config_sha256,
        "state_before_sha256": before_digest,
        "state_after_sha256": after_digest,
    }
    receipt_id = canonical_sha256(identity)
    payload: Dict[str, Any] = {
        "schema_version": COMPLETION_RECEIPT_SCHEMA_VERSION,
        "receipt_id": receipt_id,
        **identity,
        "expected_tasks": expected_tasks_value,
        "next_phase": next_phase,
        "next_iteration": next_iteration_value,
        "state_updates": dict(state_updates),
        "state_after": after_state,
        "evidence": [dict(record) for record in evidence],
        "created_at_iso": _now_iso(),
    }
    root = receipt_dir(campaign_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (receipt_id + ".json")
    if path.exists():
        existing = read_completion_receipt(path)
        comparable_existing = dict(existing)
        comparable_payload = dict(payload)
        comparable_existing.pop("created_at_iso", None)
        comparable_payload.pop("created_at_iso", None)
        if canonical_sha256(comparable_existing) != canonical_sha256(comparable_payload):
            raise CompletionReceiptError(
                "completion receipt ID exists with different content: " + str(path)
            )
        return path
    atomic_write_json(path, payload)
    return path


def read_completion_receipt(path: Union[str, Path]) -> Dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise CompletionReceiptError("completion receipt is not a regular file: " + str(target))
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompletionReceiptError("completion receipt is unreadable: " + str(target)) from exc
    if not isinstance(payload, dict):
        raise CompletionReceiptError("completion receipt must be a JSON object")
    if _exact_int(payload.get("schema_version"), "completion receipt schema_version") != COMPLETION_RECEIPT_SCHEMA_VERSION:
        raise CompletionReceiptError("unsupported completion receipt schema")
    if not isinstance(payload.get("campaign_uid"), str) or not payload["campaign_uid"]:
        raise CompletionReceiptError("completion receipt campaign_uid is empty")
    if not isinstance(payload.get("phase"), str) or not payload["phase"]:
        raise CompletionReceiptError("completion receipt phase is empty")
    _exact_int(payload.get("iteration"), "completion receipt iteration")
    _exact_int(payload.get("replacement_round"), "completion receipt replacement_round")
    for label in ("config_sha256", "state_before_sha256", "state_after_sha256"):
        _sha256(payload.get(label), "completion receipt " + label)
    expected_tasks = payload.get("expected_tasks")
    if expected_tasks is not None:
        _exact_int(expected_tasks, "completion receipt expected_tasks", minimum=1)
    _exact_int(payload.get("next_iteration"), "completion receipt next_iteration")
    if not isinstance(payload.get("next_phase"), str) or not payload["next_phase"]:
        raise CompletionReceiptError("completion receipt next_phase is empty")
    _require_iso_timestamp(
        payload.get("created_at_iso"),
        "completion receipt created_at_iso",
    )
    if not isinstance(payload.get("state_updates"), dict):
        raise CompletionReceiptError("completion receipt state_updates must be an object")
    identity = {
        key: payload.get(key)
        for key in (
            "campaign_uid",
            "phase",
            "iteration",
            "replacement_round",
            "job_id",
            "submission_identity",
            "config_sha256",
            "state_before_sha256",
            "state_after_sha256",
        )
    }
    if str(payload.get("receipt_id") or "") != canonical_sha256(identity):
        raise CompletionReceiptError("completion receipt identity digest mismatch")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        raise CompletionReceiptError("completion receipt evidence must be a list")
    after_state = payload.get("state_after")
    if not isinstance(after_state, dict):
        raise CompletionReceiptError("completion receipt state_after must be an object")
    if canonical_sha256(state_projection(after_state)) != str(
        payload.get("state_after_sha256") or ""
    ):
        raise CompletionReceiptError("completion receipt state_after digest mismatch")
    if str(after_state.get("campaign_uid") or "") != str(
        payload.get("campaign_uid") or ""
    ):
        raise CompletionReceiptError("completion receipt state_after campaign UID mismatch")
    if str(after_state.get("phase") or "") != str(payload.get("next_phase") or ""):
        raise CompletionReceiptError("completion receipt next phase mismatch")
    if _exact_int(
        after_state.get("iteration"),
        "completion receipt state_after iteration",
    ) != _exact_int(
        payload.get("next_iteration"),
        "completion receipt next_iteration",
    ):
        raise CompletionReceiptError("completion receipt next iteration mismatch")
    seen_evidence = set()
    for record in evidence:
        if not isinstance(record, dict):
            raise CompletionReceiptError("completion evidence record is invalid")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise CompletionReceiptError("completion evidence path is invalid")
        path_value = Path(relative)
        if path_value.is_absolute() or ".." in path_value.parts:
            raise CompletionReceiptError("completion evidence path escapes campaign")
        canonical = path_value.as_posix()
        if canonical != relative or canonical in seen_evidence:
            raise CompletionReceiptError(
                "completion evidence paths must be unique and canonical"
            )
        seen_evidence.add(canonical)
        _exact_int(record.get("size"), "completion evidence size")
        _sha256(record.get("sha256"), "completion evidence sha256")
    return payload


def replayable_completion_receipts(
    campaign_dir: Union[str, Path],
    state: Union[CampaignState, Mapping[str, Any]],
    *,
    expected_config_sha256: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return validated receipts whose pre-state exactly matches ``state``."""
    campaign = Path(campaign_dir)
    root = receipt_dir(campaign)
    if not root.is_dir():
        return []
    before_sha = canonical_sha256(state_projection(state))
    campaign_uid = str(state_projection(state).get("campaign_uid") or "")
    matches: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = read_completion_receipt(path)
        except CompletionReceiptError:
            continue
        if str(payload.get("campaign_uid") or "") != campaign_uid:
            continue
        if expected_config_sha256 is not None and str(
            payload.get("config_sha256") or ""
        ) != str(expected_config_sha256):
            continue
        if str(payload.get("state_before_sha256") or "") != before_sha:
            continue
        reference = receipt_reference(campaign, path)
        validate_completion_reference(
            campaign,
            reference,
            expected_campaign_uid=campaign_uid,
        )
        matches.append({"path": path, "reference": reference, "payload": payload})
    return matches


def inventory_completion_receipts(
    campaign_dir: Union[str, Path],
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Read completion receipts without traversing their scientific evidence."""
    campaign = Path(campaign_dir).resolve()
    root = receipt_dir(campaign)
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    if not root.is_dir() or root.is_symlink():
        return {"records": records, "errors": errors}
    for path in sorted(root.glob("*.json")):
        try:
            payload = read_completion_receipt(path)
            if expected_campaign_uid is not None and str(
                payload.get("campaign_uid") or ""
            ) != str(expected_campaign_uid):
                raise CompletionReceiptError(
                    "completion receipt campaign UID mismatch"
                )
            relative = path.resolve().relative_to(campaign).as_posix()
            records.append(
                {
                    "path": relative,
                    "payload": dict(payload),
                    "reference": {
                        "path": relative,
                        "sha256": sha256_file(path),
                        "receipt_id": str(payload["receipt_id"]),
                    },
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


def receipt_reference(campaign_dir: Union[str, Path], path: Union[str, Path]) -> Dict[str, str]:
    campaign = Path(campaign_dir).resolve()
    resolved = _inside_campaign(Path(campaign_dir), Path(path))
    payload = read_completion_receipt(resolved)
    return {
        "path": resolved.relative_to(campaign).as_posix(),
        "sha256": sha256_file(resolved),
        "receipt_id": str(payload["receipt_id"]),
    }


def validate_completion_reference(
    campaign_dir: Union[str, Path],
    reference: Mapping[str, Any],
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir)
    relative = Path(str(reference.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise CompletionReceiptError("completion receipt reference escapes campaign")
    path = _inside_campaign(campaign, campaign / relative)
    if path.is_symlink() or not path.is_file():
        raise CompletionReceiptError("completion receipt file is missing")
    if sha256_file(path) != str(reference.get("sha256") or ""):
        raise CompletionReceiptError("completion receipt file hash mismatch")
    payload = read_completion_receipt(path)
    if str(payload["receipt_id"]) != str(reference.get("receipt_id") or ""):
        raise CompletionReceiptError("completion receipt reference identity mismatch")
    if expected_campaign_uid is not None and str(payload["campaign_uid"]) != str(
        expected_campaign_uid
    ):
        raise CompletionReceiptError("completion receipt campaign UID mismatch")
    for record in payload["evidence"]:
        if not isinstance(record, dict):
            raise CompletionReceiptError("completion evidence record is invalid")
        evidence_path = Path(str(record.get("path") or ""))
        if evidence_path.is_absolute() or ".." in evidence_path.parts:
            raise CompletionReceiptError("completion evidence path escapes campaign")
        resolved = _inside_campaign(campaign, campaign / evidence_path)
        if resolved.is_symlink() or not resolved.is_file():
            raise CompletionReceiptError("completion evidence file is missing")
        if _exact_int(record.get("size"), "completion evidence size") != int(
            resolved.stat().st_size
        ):
            raise CompletionReceiptError("completion evidence size mismatch")
        if str(record.get("sha256") or "") != sha256_file(resolved):
            raise CompletionReceiptError("completion evidence hash mismatch")
    return payload


__all__ = [
    "COMPLETION_RECEIPT_SCHEMA_VERSION",
    "CompletionReceiptError",
    "canonical_sha256",
    "evidence_records",
    "inventory_completion_receipts",
    "read_completion_receipt",
    "replayable_completion_receipts",
    "receipt_dir",
    "receipt_reference",
    "state_projection",
    "validate_completion_reference",
    "write_completion_receipt",
]
