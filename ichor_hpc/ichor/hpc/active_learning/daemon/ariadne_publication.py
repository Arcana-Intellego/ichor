from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..handoff_manifests import (
    ACQUISITION_MATURITY_AUDIT_FILENAME,
    ARIADNE_BATCH_DECISION_FILENAME,
    ARIADNE_BATCH_DECISION_SCHEMA_VERSION,
    ARIADNE_RESULTS_FILENAME,
    ARIADNE_RESULTS_SCHEMA_VERSION,
    read_ariadne_batch_decision,
)
from ..layout import active_ariadne_dir, active_iteration_dir, active_iteration_name
from ..strict_json import strict_json as json
from ..versioning.manifest import sha256_file
from .filesystem import campaign_owned_path, operational_path
from .state import _fsync_parent_dir, atomic_write_json


ARIADNE_PUBLICATION_ARCHIVE_SCHEMA_VERSION = 1
ARIADNE_PUBLICATION_ARCHIVE_FILENAME = "ARCHIVE.json"
ARIADNE_PUBLICATION_FILENAMES = (
    ARIADNE_RESULTS_FILENAME,
    ARIADNE_BATCH_DECISION_FILENAME,
    ACQUISITION_MATURITY_AUDIT_FILENAME,
)
_ARCHIVE_STATUSES = {"prepared", "moving", "complete"}
_ENTRY_STATES = {"source", "archived"}


class AriadnePublicationError(RuntimeError):
    """Raised when derived ARIADNE batch evidence cannot be classified safely."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AriadnePublicationError(
            label + " must be an exact integer >= " + str(minimum)
        )
    return int(value)


def _sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise AriadnePublicationError(label + " must be a SHA-256 digest")
    return text


def _publication_paths(campaign_dir: Path, iteration: int) -> Dict[str, Path]:
    campaign = Path(campaign_dir).resolve()
    root = campaign_owned_path(
        campaign,
        active_ariadne_dir(active_iteration_dir(campaign, int(iteration))),
    )
    return {
        name: campaign_owned_path(campaign, root / name)
        for name in ARIADNE_PUBLICATION_FILENAMES
    }


def _file_record(campaign: Path, path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AriadnePublicationError(
            "ARIADNE publication entry is not a regular file: " + str(path)
        )
    return {
        "path": path.relative_to(campaign).as_posix(),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def classify_ariadne_publication(
    campaign_dir: Path,
    iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify the small, derived ARIADNE batch publication.

    Per-seed directories are deliberately outside this contract.  They are the
    producer outputs from which this publication can be rebuilt.
    """
    campaign = Path(campaign_dir).resolve()
    pending = _incomplete_archive(campaign, int(iteration))
    if pending is not None:
        manifest_path, payload = pending
        return {
            "state": "archive_incomplete",
            "iteration": int(iteration),
            "archive_required": True,
            "reason": "ARIADNE publication archive transaction is incomplete",
            "files": [],
            "archive_manifest": str(manifest_path),
            "archive_id": str(payload["archive_id"]),
        }
    paths = _publication_paths(campaign, int(iteration))
    existing: Dict[str, Dict[str, Any]] = {}
    for name, path in paths.items():
        if path.is_symlink():
            return {
                "state": "invalid",
                "iteration": int(iteration),
                "archive_required": False,
                "reason": "ARIADNE publication contains a symlink: " + str(path),
                "files": [],
            }
        if not path.exists():
            continue
        if not path.is_file():
            return {
                "state": "invalid",
                "iteration": int(iteration),
                "archive_required": False,
                "reason": "ARIADNE publication entry is not a file: " + str(path),
                "files": [],
            }
        existing[name] = _file_record(campaign, path)

    files = [existing[name] for name in ARIADNE_PUBLICATION_FILENAMES if name in existing]
    if not files:
        return {
            "state": "absent",
            "iteration": int(iteration),
            "archive_required": False,
            "reason": "no derived ARIADNE batch publication exists",
            "files": [],
        }

    results_path = paths[ARIADNE_RESULTS_FILENAME]
    decision_path = paths[ARIADNE_BATCH_DECISION_FILENAME]
    if decision_path.exists() and not results_path.exists():
        return {
            "state": "invalid",
            "iteration": int(iteration),
            "archive_required": False,
            "reason": "ARIADNE batch decision exists without RESULTS.json",
            "files": files,
        }
    if not decision_path.exists():
        return {
            "state": "incomplete",
            "iteration": int(iteration),
            "archive_required": True,
            "reason": "ARIADNE derived publication has no batch decision",
            "files": files,
        }

    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(decision, Mapping) or not isinstance(results, Mapping):
            raise AriadnePublicationError(
                "ARIADNE decision and results must be JSON objects"
            )
        if _exact_int(
            decision.get("schema_version"),
            "ARIADNE batch decision schema_version",
        ) != ARIADNE_BATCH_DECISION_SCHEMA_VERSION:
            raise AriadnePublicationError("unsupported ARIADNE batch decision schema")
        if _exact_int(
            results.get("schema_version"),
            "ARIADNE results schema_version",
        ) != ARIADNE_RESULTS_SCHEMA_VERSION:
            raise AriadnePublicationError("unsupported ARIADNE results schema")
        if _exact_int(decision.get("iteration"), "ARIADNE decision iteration") != int(iteration):
            raise AriadnePublicationError("ARIADNE decision iteration mismatch")
        if _exact_int(results.get("iteration"), "ARIADNE results iteration") != int(iteration):
            raise AriadnePublicationError("ARIADNE results iteration mismatch")
        decision_uid = str(decision.get("campaign_uid") or "")
        results_uid = str(results.get("campaign_uid") or "")
        if not decision_uid or decision_uid != results_uid:
            raise AriadnePublicationError("ARIADNE decision/results campaign UID mismatch")
        if expected_campaign_uid is not None and decision_uid != str(expected_campaign_uid):
            raise AriadnePublicationError("ARIADNE publication campaign UID mismatch")
        binding = decision.get("results")
        if not isinstance(binding, Mapping):
            raise AriadnePublicationError("ARIADNE decision results binding is missing")
        if str(binding.get("path") or "") != ARIADNE_RESULTS_FILENAME:
            raise AriadnePublicationError("ARIADNE decision results path is noncanonical")
        expected_size = _exact_int(
            binding.get("size"),
            "ARIADNE decision results size",
        )
        expected_sha256 = _sha256(
            binding.get("sha256"),
            "ARIADNE decision results digest",
        )
    except (OSError, ValueError, AriadnePublicationError) as exc:
        return {
            "state": "invalid",
            "iteration": int(iteration),
            "archive_required": False,
            "reason": type(exc).__name__ + ": " + str(exc),
            "files": files,
        }

    observed = existing[ARIADNE_RESULTS_FILENAME]
    if (
        int(observed["size"]) != int(expected_size)
        or str(observed["sha256"]) != str(expected_sha256)
    ):
        return {
            "state": "stale_results_binding",
            "iteration": int(iteration),
            "archive_required": True,
            "reason": "ARIADNE batch decision binds an earlier RESULTS.json",
            "files": files,
            "expected_results": {
                "size": int(expected_size),
                "sha256": str(expected_sha256),
            },
            "observed_results": {
                "size": int(observed["size"]),
                "sha256": str(observed["sha256"]),
            },
        }

    try:
        validated_decision = read_ariadne_batch_decision(
            active_iteration_dir(campaign, int(iteration)),
            expected_iteration=int(iteration),
            expected_campaign_uid=expected_campaign_uid,
            require_accepted=False,
            verify_current_config=False,
        )
    except Exception as exc:
        return {
            "state": "invalid",
            "iteration": int(iteration),
            "archive_required": False,
            "reason": type(exc).__name__ + ": " + str(exc),
            "files": files,
        }
    return {
        "state": "complete",
        "iteration": int(iteration),
        "archive_required": False,
        "reason": "ARIADNE derived publication is internally consistent",
        "files": files,
        "accepted": bool(
            validated_decision["current_evaluation"].get("accepted", False)
        ),
        "current_evaluation_sha256": str(
            validated_decision.get("current_evaluation_sha256") or ""
        ),
    }


def ariadne_publication_archive_root(campaign_dir: Path, iteration: int) -> Path:
    campaign = Path(campaign_dir).resolve()
    return operational_path(
        campaign,
        "ariadne_publication_archives",
        active_iteration_name(int(iteration)),
    )


def _read_archive_manifest(path: Path, campaign: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AriadnePublicationError(
            "ARIADNE publication archive manifest is not a regular file: " + str(path)
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AriadnePublicationError(
            "ARIADNE publication archive manifest is unreadable: " + str(path)
        ) from exc
    if not isinstance(payload, dict):
        raise AriadnePublicationError("ARIADNE publication archive must be an object")
    if _exact_int(payload.get("schema_version"), "archive schema_version") != ARIADNE_PUBLICATION_ARCHIVE_SCHEMA_VERSION:
        raise AriadnePublicationError("unsupported ARIADNE publication archive schema")
    archive_id = str(payload.get("archive_id") or "")
    if not archive_id or path.parent.name != archive_id:
        raise AriadnePublicationError("ARIADNE publication archive identity mismatch")
    status = str(payload.get("status") or "")
    if status not in _ARCHIVE_STATUSES:
        raise AriadnePublicationError("ARIADNE publication archive status is invalid")
    _exact_int(payload.get("iteration"), "archive iteration")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise AriadnePublicationError("ARIADNE publication archive entries are empty")
    for entry in entries:
        if not isinstance(entry, dict):
            raise AriadnePublicationError("ARIADNE publication archive entry is invalid")
        if str(entry.get("state") or "") not in _ENTRY_STATES:
            raise AriadnePublicationError("ARIADNE publication archive entry state is invalid")
        _exact_int(entry.get("size"), "archive entry size")
        _sha256(entry.get("sha256"), "archive entry digest")
        campaign_owned_path(campaign, str(entry.get("source_path") or ""))
        campaign_owned_path(campaign, str(entry.get("archived_path") or ""))
    return payload


def _write_archive_manifest(path: Path, payload: Dict[str, Any]) -> None:
    payload["updated_at_iso"] = _now_iso()
    atomic_write_json(path, payload)


def _incomplete_archive(campaign: Path, iteration: int) -> Optional[tuple[Path, Dict[str, Any]]]:
    root = ariadne_publication_archive_root(campaign, int(iteration))
    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise AriadnePublicationError(
            "ARIADNE publication archive root is unsafe: " + str(root)
        )
    incomplete = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_symlink() or not child.is_dir():
            raise AriadnePublicationError(
                "unexpected ARIADNE publication archive entry: " + str(child)
            )
        manifest_path = child / ARIADNE_PUBLICATION_ARCHIVE_FILENAME
        payload = _read_archive_manifest(manifest_path, campaign)
        if str(payload["status"]) != "complete":
            incomplete.append((manifest_path, payload))
    if len(incomplete) > 1:
        raise AriadnePublicationError(
            "multiple incomplete ARIADNE publication archives exist"
        )
    return incomplete[0] if incomplete else None


def archive_ariadne_publication(
    campaign_dir: Path,
    iteration: int,
    *,
    reason: str,
    campaign_uid: str,
    submission_identity: Optional[str] = None,
    classification: Optional[Mapping[str, Any]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Move one stale derived publication into a resumable evidence bundle."""
    campaign = Path(campaign_dir).resolve()
    pending = _incomplete_archive(campaign, int(iteration))
    if pending is not None:
        manifest_path, payload = pending
        if int(payload["iteration"]) != int(iteration):
            raise AriadnePublicationError("incomplete archive iteration mismatch")
        if str(payload.get("campaign_uid") or "") != str(campaign_uid):
            raise AriadnePublicationError("incomplete archive campaign UID mismatch")
    else:
        observed = dict(
            classification
            or classify_ariadne_publication(
                campaign,
                int(iteration),
                expected_campaign_uid=str(campaign_uid),
            )
        )
        if str(observed.get("state") or "") == "invalid":
            raise AriadnePublicationError(
                "refusing to archive invalid ARIADNE publication: "
                + str(observed.get("reason") or "unknown error")
            )
        should_archive = bool(observed.get("archive_required", False)) or (
            bool(force) and bool(observed.get("files"))
        )
        if not should_archive:
            return {
                "changed": False,
                "classification": observed,
                "archived_paths": [],
            }
        files = list(observed.get("files") or [])
        if not files:
            raise AriadnePublicationError(
                "ARIADNE publication archival has no source files"
            )
        root = ariadne_publication_archive_root(campaign, int(iteration))
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise AriadnePublicationError(
                "ARIADNE publication archive root is unsafe: " + str(root)
            )
        archive_id = (
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        archive_dir = campaign_owned_path(campaign, root / archive_id)
        archive_dir.mkdir()
        try:
            os.chmod(archive_dir, 0o700)
        except OSError as exc:
            raise AriadnePublicationError(
                "could not secure ARIADNE publication archive: " + str(archive_dir)
            ) from exc
        entries = []
        for record in files:
            source = campaign_owned_path(campaign, str(record["path"]))
            if source.name not in ARIADNE_PUBLICATION_FILENAMES:
                raise AriadnePublicationError(
                    "ARIADNE publication archive source is noncanonical"
                )
            target = campaign_owned_path(campaign, archive_dir / source.name)
            entries.append(
                {
                    "source_path": source.relative_to(campaign).as_posix(),
                    "archived_path": target.relative_to(campaign).as_posix(),
                    "size": int(record["size"]),
                    "sha256": str(record["sha256"]),
                    "state": "source",
                }
            )
        payload = {
            "schema_version": ARIADNE_PUBLICATION_ARCHIVE_SCHEMA_VERSION,
            "archive_id": archive_id,
            "campaign_uid": str(campaign_uid),
            "iteration": int(iteration),
            "status": "prepared",
            "reason": str(reason),
            "submission_identity": (
                None if submission_identity is None else str(submission_identity)
            ),
            "created_at_iso": _now_iso(),
            "updated_at_iso": _now_iso(),
            "source_classification": str(observed.get("state") or ""),
            "entries": entries,
        }
        manifest_path = archive_dir / ARIADNE_PUBLICATION_ARCHIVE_FILENAME
        _write_archive_manifest(manifest_path, payload)

    payload["status"] = "moving"
    _write_archive_manifest(manifest_path, payload)
    archive_dir = manifest_path.parent
    expected_sources = _publication_paths(campaign, int(iteration))
    for entry in payload["entries"]:
        source = campaign_owned_path(campaign, str(entry["source_path"]))
        target = campaign_owned_path(campaign, str(entry["archived_path"]))
        if source != expected_sources.get(source.name):
            raise AriadnePublicationError(
                "ARIADNE publication archive source is noncanonical"
            )
        if target.parent != archive_dir or target.name != source.name:
            raise AriadnePublicationError(
                "ARIADNE publication archive target is noncanonical"
            )
        source_exists = source.exists() or source.is_symlink()
        target_exists = target.exists() or target.is_symlink()
        if source_exists and target_exists:
            raise AriadnePublicationError(
                "ARIADNE publication exists at both source and archive target"
            )
        if source_exists:
            observed = _file_record(campaign, source)
            if (
                int(observed["size"]) != int(entry["size"])
                or str(observed["sha256"]) != str(entry["sha256"])
            ):
                raise AriadnePublicationError(
                    "ARIADNE publication changed after archive preparation"
                )
            os.replace(source, target)
            _fsync_parent_dir(source)
            _fsync_parent_dir(target)
        elif not target_exists:
            raise AriadnePublicationError(
                "ARIADNE publication archive entry is missing at source and target"
            )
        observed_target = _file_record(campaign, target)
        if (
            int(observed_target["size"]) != int(entry["size"])
            or str(observed_target["sha256"]) != str(entry["sha256"])
        ):
            raise AriadnePublicationError(
                "archived ARIADNE publication digest mismatch"
            )
        entry["state"] = "archived"
        _write_archive_manifest(manifest_path, payload)
    payload["status"] = "complete"
    _write_archive_manifest(manifest_path, payload)
    return {
        "changed": True,
        "archive_id": str(payload["archive_id"]),
        "archive_dir": str(archive_dir),
        "manifest_path": str(manifest_path),
        "archived_paths": [str(entry["archived_path"]) for entry in payload["entries"]],
        "source_classification": str(payload.get("source_classification") or ""),
    }


__all__ = [
    "ARIADNE_PUBLICATION_ARCHIVE_FILENAME",
    "ARIADNE_PUBLICATION_ARCHIVE_SCHEMA_VERSION",
    "ARIADNE_PUBLICATION_FILENAMES",
    "AriadnePublicationError",
    "archive_ariadne_publication",
    "ariadne_publication_archive_root",
    "classify_ariadne_publication",
]
