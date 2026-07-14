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


COMPLETION_RECEIPT_SCHEMA_VERSION = 1
COMPLETION_RECEIPT_DIRNAME = "phase_completions"


class CompletionReceiptError(ValueError):
    """Raised when a phase-completion receipt cannot be trusted."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    before_state = state_projection(state_before)
    after_state = state_projection(state_after)
    before_digest = canonical_sha256(before_state)
    after_digest = canonical_sha256(after_state)
    identity = {
        "campaign_uid": str(campaign_uid),
        "phase": str(phase),
        "iteration": int(iteration),
        "replacement_round": int(replacement_round),
        "job_id": None if job_id is None else str(job_id),
        "submission_identity": (
            None if submission_identity is None else str(submission_identity)
        ),
        "config_sha256": str(config_sha256),
        "state_before_sha256": before_digest,
        "state_after_sha256": after_digest,
    }
    receipt_id = canonical_sha256(identity)
    payload: Dict[str, Any] = {
        "schema_version": COMPLETION_RECEIPT_SCHEMA_VERSION,
        "receipt_id": receipt_id,
        **identity,
        "expected_tasks": None if expected_tasks is None else int(expected_tasks),
        "next_phase": str(next_phase),
        "next_iteration": int(next_iteration),
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
    if int(payload.get("schema_version", -1)) != COMPLETION_RECEIPT_SCHEMA_VERSION:
        raise CompletionReceiptError("unsupported completion receipt schema")
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
    try:
        if int(after_state.get("iteration")) != int(payload.get("next_iteration")):
            raise CompletionReceiptError("completion receipt next iteration mismatch")
    except (TypeError, ValueError) as exc:
        raise CompletionReceiptError("completion receipt next iteration is invalid") from exc
    return payload


def replayable_completion_receipts(
    campaign_dir: Union[str, Path],
    state: Union[CampaignState, Mapping[str, Any]],
    *,
    expected_config_sha256: str,
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
        if str(payload.get("config_sha256") or "") != str(expected_config_sha256):
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
        if int(record.get("size", -1)) != int(resolved.stat().st_size):
            raise CompletionReceiptError("completion evidence size mismatch")
        if str(record.get("sha256") or "") != sha256_file(resolved):
            raise CompletionReceiptError("completion evidence hash mismatch")
    return payload


__all__ = [
    "COMPLETION_RECEIPT_SCHEMA_VERSION",
    "CompletionReceiptError",
    "canonical_sha256",
    "evidence_records",
    "read_completion_receipt",
    "replayable_completion_receipts",
    "receipt_dir",
    "receipt_reference",
    "state_projection",
    "validate_completion_reference",
    "write_completion_receipt",
]
