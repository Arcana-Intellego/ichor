"""Exact, durable point-allocation contracts for bootstrap and active batches."""
from __future__ import annotations

import copy
import hashlib
from .strict_json import strict_json as json
import math
import re
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .daemon.state import atomic_write_json
from .layout import (
    active_allocation_dir,
    active_iteration_dir,
    bootstrap_allocation_dir,
)


POINT_ALLOCATION_SCHEMA_VERSION = 4
POINT_ALLOCATION_FILENAME = "POINT_ALLOCATION.json"
POINT_ALLOCATION_LOCK_FILENAME = "POINT_ALLOCATION.lock"
VALID_CONTEXTS = frozenset({"bootstrap", "active"})
VALID_SPLITS = frozenset({"train", "int_val", "ext_val"})
MAX_MANIFEST_INTEGER = (1 << 63) - 1
CANDIDATE_INTEGER_FIELDS = frozenset({
    "frame_id",
    "custom_index",
    "source_index",
    "seed_id",
    "seed_frame_id",
    "pool_row_index_zero_based",
    "candidate_pool_index_zero_based",
    "considered_rank",
    "final_rank",
    "reserve_rank",
    "diversity_rank",
    "seed_index",
    "replacement_round",
})


def _required_integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(label + " must be an exact JSON integer")
    result = int(value)
    if result < minimum or result > MAX_MANIFEST_INTEGER:
        raise ValueError(
            label
            + " must be between "
            + str(minimum)
            + " and "
            + str(MAX_MANIFEST_INTEGER)
        )
    return result


def _optional_integer(value: Any, label: str, *, minimum: int = 0) -> Optional[int]:
    if value is None:
        return None
    return _required_integer(value, label, minimum=minimum)


def _validate_candidate_integer_fields(record: Dict[str, Any], label: str) -> None:
    for key in CANDIDATE_INTEGER_FIELDS:
        if key in record and record[key] is not None:
            record[key] = _required_integer(record[key], label + " " + key)


def _validate_terminal_attempt(attempt: Dict[str, Any], label: str) -> None:
    status = str(attempt.get("status") or "")
    if status == "pending":
        for field_name in (
            "pointdir",
            "reason",
            "quality_manifest",
            "quantum_acceptance_receipt",
            "quantum_acceptance_receipt_sha256",
            "accepted_pointdir_content_sha256",
        ):
            if attempt.get(field_name) not in (None, ""):
                raise ValueError(label + " pending attempt contains terminal evidence")
        return
    pointdir_text = str(attempt.get("pointdir") or "").strip()
    pointdir_path = Path(pointdir_text)
    if (
        not pointdir_text
        or any(character in pointdir_text for character in "\r\n\x00")
        or any(part in {".", ".."} for part in pointdir_path.parts)
        or Path(pointdir_text).name != pointdir_text.replace("\\", "/").split("/")[-1]
        or not pointdir_path.name.endswith(".pointdir")
    ):
        raise ValueError(label + " terminal attempt has an invalid pointdir")
    reason = attempt.get("reason")
    if status == "accepted":
        if reason not in (None, ""):
            raise ValueError(label + " accepted attempt cannot have a rejection reason")
        quality = str(attempt.get("quality_manifest") or "").strip()
        if (
            not quality
            or any(character in quality for character in "\r\n\x00")
            or any(part in {".", ".."} for part in Path(quality).parts)
            or Path(quality).suffix.lower() != ".json"
        ):
            raise ValueError(label + " accepted attempt lacks quality-manifest evidence")
        receipt = str(attempt.get("quantum_acceptance_receipt") or "").strip()
        receipt_sha = str(
            attempt.get("quantum_acceptance_receipt_sha256") or ""
        ).strip()
        content_sha = str(
            attempt.get("accepted_pointdir_content_sha256") or ""
        ).strip()
        if any((receipt, receipt_sha, content_sha)):
            if (
                not receipt
                or Path(receipt).is_absolute()
                or any(part in {".", ".."} for part in Path(receipt).parts)
                or not re.fullmatch(r"[0-9a-f]{64}", receipt_sha)
                or not re.fullmatch(r"[0-9a-f]{64}", content_sha)
            ):
                raise ValueError(label + " quantum acceptance evidence is invalid")
    elif not isinstance(reason, str) or not reason.strip():
        raise ValueError(label + " rejected attempt must have a non-empty reason")


class PointAllocationLockError(RuntimeError):
    """Raised when allocation ownership cannot be acquired in time."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _generation_sha256(payload: Mapping[str, Any]) -> str:
    canonical = copy.deepcopy(dict(payload))
    canonical.pop("summary", None)
    return _sha256_json(canonical)


def slot_assignment_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the immutable initial candidate-to-split slot assignment."""
    slots = payload.get("slots")
    if not isinstance(slots, list):
        raise ValueError("point-allocation slots must be a list")
    assignment = []
    for slot in slots:
        attempts = slot.get("attempts") if isinstance(slot, Mapping) else None
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("point-allocation slot lacks an initial attempt")
        assignment.append(
            {
                "slot_id": _required_integer(
                    slot.get("slot_id"), "point-allocation slot_id"
                ),
                "split": str(slot["split"]),
                "candidate_id": str(attempts[0]["candidate_id"]),
            }
        )
    return _sha256_json(assignment)


def point_allocation_path(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
) -> Path:
    campaign = Path(campaign_dir)
    resolved_iteration = _required_integer(iteration, "point-allocation iteration")
    if str(context) == "bootstrap":
        if resolved_iteration != 0:
            raise ValueError("bootstrap point allocation requires iteration 0")
        return bootstrap_allocation_dir(campaign) / POINT_ALLOCATION_FILENAME
    if str(context) == "active":
        return active_allocation_dir(
            active_iteration_dir(campaign, resolved_iteration)
        ) / POINT_ALLOCATION_FILENAME
    raise ValueError("point-allocation context must be bootstrap or active")


@contextmanager
def _allocation_lock(path: Path):
    import portalocker

    lock_path = Path(path).with_name(POINT_ALLOCATION_LOCK_FILENAME)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(
            str(lock_path),
            mode="a",
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
            timeout=30.0,
        ):
            yield
    except (portalocker.LockException, portalocker.AlreadyLocked) as exc:
        raise PointAllocationLockError(
            "could not acquire point-allocation ownership within 30 seconds"
        ) from exc


def allocation_targets(config: Any, context: str) -> Dict[str, int]:
    block = config.point_allocation
    if str(context) == "bootstrap":
        counts = {
            "train": _required_integer(
                block.bootstrap_training_size, "bootstrap training target"
            ),
            "int_val": _required_integer(
                block.bootstrap_internal_validation_size,
                "bootstrap internal-validation target",
            ),
            "ext_val": _required_integer(
                block.bootstrap_external_validation_size,
                "bootstrap external-validation target",
            ),
        }
    elif str(context) == "active":
        counts = {
            "train": _required_integer(
                block.batch_training_size, "active training target"
            ),
            "int_val": _required_integer(
                block.batch_internal_validation_size,
                "active internal-validation target",
            ),
            "ext_val": 0,
        }
    else:
        raise ValueError("point-allocation context must be bootstrap or active")
    if (
        counts["train"] <= 0
        or counts["int_val"] < 0
        or counts["ext_val"] < 0
        or (str(context) == "bootstrap" and counts["int_val"] <= 0)
    ):
        raise ValueError("invalid point-allocation target counts: " + repr(counts))
    counts["total"] = counts["train"] + counts["int_val"] + counts["ext_val"]
    return counts


def stable_candidate_id(
    *,
    campaign_uid: str,
    context: str,
    iteration: int,
    source_identity: Any,
    provenance_sha256: Optional[str] = None,
) -> str:
    resolved_iteration = _required_integer(iteration, "candidate iteration")
    payload = {
        "campaign_uid": str(campaign_uid),
        "context": str(context),
        "iteration": resolved_iteration,
        "source_identity": source_identity,
        "provenance_sha256": str(provenance_sha256 or ""),
    }
    return "candidate-" + _sha256_json(payload)[:24]


def _normalise_candidate(record: Mapping[str, Any]) -> Dict[str, Any]:
    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, bool):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    "point-allocation candidate contains a non-finite number"
                )
            return numeric
        return value

    candidate = json_safe(dict(record))
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        raise ValueError("point-allocation candidate is missing candidate_id")
    candidate["candidate_id"] = candidate_id
    _validate_candidate_integer_fields(candidate, "point-allocation candidate")
    return candidate


def _immutable_candidate_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the candidate fields that must survive an idempotent retry.

    Allocation status, round bookkeeping, and quantum outcomes are mutable.
    Source identity, paths, provenance hashes, ranks, and safety diagnostics are
    not: accepting changed candidate evidence under the same ID would make a
    retry silently reinterpret an existing allocation.
    """
    payload = copy.deepcopy(dict(record))
    for key in (
        "status",
        "round",
        "consumed_round",
        "pointdir",
        "reason",
        "quality_manifest",
        "quantum_acceptance_receipt",
        "quantum_acceptance_receipt_sha256",
        "accepted_pointdir_content_sha256",
    ):
        payload.pop(key, None)
    return payload


def _ordered_slot_splits(targets: Mapping[str, int]) -> List[str]:
    return (
        ["train"] * int(targets["train"])
        + ["int_val"] * int(targets["int_val"])
        + ["ext_val"] * int(targets["ext_val"])
    )


def _assignment_order(
    candidate_ids: Iterable[str],
    *,
    campaign_uid: str,
    context: str,
    iteration: int,
) -> List[str]:
    salt = str(campaign_uid) + ":" + str(context) + ":" + str(int(iteration)) + ":"
    return sorted(
        (str(candidate_id) for candidate_id in candidate_ids),
        key=lambda candidate_id: (
            hashlib.sha256((salt + candidate_id).encode("utf-8")).hexdigest(),
            candidate_id,
        ),
    )


def _refresh_summary(payload: Dict[str, Any]) -> None:
    targets = dict(payload["targets"])
    accepted = {split: 0 for split in VALID_SPLITS}
    pending = 0
    vacant = 0
    for slot in payload["slots"]:
        split = str(slot["split"])
        selected = slot.get("accepted_attempt")
        if selected is not None:
            accepted[split] += 1
            continue
        attempts = list(slot.get("attempts") or [])
        if attempts and str(attempts[-1].get("status")) == "pending":
            pending += 1
        else:
            vacant += 1
    deficits = {
        split: int(targets[split]) - int(accepted[split])
        for split in ("train", "int_val", "ext_val")
    }
    reserve = list(payload.get("reserve") or [])
    reserve_available = sum(
        1 for record in reserve if str(record.get("status", "available")) == "available"
    )
    complete = all(value == 0 for value in deficits.values()) and pending == 0
    payload["summary"] = {
        "target_total": int(targets["total"]),
        "accepted": accepted,
        "accepted_total": int(sum(accepted.values())),
        "deficits": deficits,
        "deficit_total": int(sum(deficits.values())),
        "pending_slots": int(pending),
        "vacant_slots": int(vacant),
        "reserve_total": int(len(reserve)),
        "reserve_available": int(reserve_available),
        "reserve_consumed": int(len(reserve) - reserve_available),
        "complete": bool(complete),
        "status": "complete" if complete else ("pending" if pending else "underfilled"),
    }


def _validate_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("point-allocation manifest must be a JSON object")
    data = copy.deepcopy(dict(payload))
    if _required_integer(
        data.get("schema_version"), "point-allocation schema_version"
    ) != POINT_ALLOCATION_SCHEMA_VERSION:
        raise ValueError("unsupported point-allocation manifest schema")
    campaign_uid = str(data.get("campaign_uid") or "").strip()
    if not campaign_uid:
        raise ValueError("point-allocation campaign_uid must be non-empty")
    data["campaign_uid"] = campaign_uid
    context = str(data.get("context") or "")
    if context not in VALID_CONTEXTS:
        raise ValueError("point-allocation manifest context is invalid")
    iteration = _required_integer(data.get("iteration"), "point-allocation iteration")
    if (
        iteration < 0
        or (context == "bootstrap" and iteration != 0)
        or (context == "active" and iteration < 1)
    ):
        raise ValueError("point-allocation manifest iteration is invalid")
    data["iteration"] = iteration
    generation = _required_integer(data.get("generation"), "point-allocation generation")
    data["generation"] = generation
    previous_hash = data.get("previous_generation_sha256")
    if generation == 0:
        if previous_hash is not None:
            raise ValueError("generation zero cannot reference a predecessor")
    else:
        previous_hash = str(previous_hash or "")
        if (
            len(previous_hash) != 64
            or any(character not in "0123456789abcdef" for character in previous_hash)
        ):
            raise ValueError("point-allocation predecessor hash is invalid")
        data["previous_generation_sha256"] = previous_hash
    targets = data.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("point-allocation manifest targets must be an object")
    for split in ("train", "int_val", "ext_val"):
        value = _required_integer(
            targets.get(split), "point-allocation target " + split
        )
        targets[split] = value
    targets["total"] = _required_integer(
        targets.get("total"), "point-allocation target total"
    )
    if targets["total"] != sum(targets[split] for split in VALID_SPLITS):
        raise ValueError("point-allocation target total is inconsistent")
    slots = data.get("slots")
    reserve = data.get("reserve")
    if not isinstance(slots, list) or len(slots) != int(targets["total"]):
        raise ValueError("point-allocation slot count does not match target")
    if not isinstance(reserve, list):
        raise ValueError("point-allocation reserve must be a list")
    applied_batches = data.get("applied_quantum_batches", [])
    if not isinstance(applied_batches, list):
        raise ValueError("point-allocation applied quantum batches must be a list")
    batch_ids: set[str] = set()
    for record in applied_batches:
        if not isinstance(record, dict):
            raise ValueError("point-allocation quantum batch record must be an object")
        batch_id = str(record.get("batch_identity") or "")
        fingerprint = str(record.get("result_fingerprint") or "")
        if len(batch_id) != 64 or len(fingerprint) != 64 or batch_id in batch_ids:
            raise ValueError("point-allocation quantum batch identity is invalid")
        batch_ids.add(batch_id)
        record["source_generation"] = _required_integer(
            record.get("source_generation"),
            "point-allocation quantum batch source generation",
        )
        candidate_ids = record.get("candidate_ids")
        if not isinstance(candidate_ids, list) or not candidate_ids:
            raise ValueError("point-allocation quantum batch candidate IDs are invalid")
    attempt_candidate_ids: set[str] = set()
    attempt_locations: Dict[str, Tuple[int, str, Dict[str, Any]]] = {}
    split_counts = {split: 0 for split in VALID_SPLITS}
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict) or _required_integer(
            slot.get("slot_id"), "point-allocation slot_id"
        ) != index:
            raise ValueError("point-allocation slots must be ordered by slot_id")
        split = str(slot.get("split") or "")
        if split not in VALID_SPLITS:
            raise ValueError("point-allocation slot split is invalid")
        split_counts[split] += 1
        attempts = slot.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("point-allocation slot must contain at least one attempt")
        accepted_indexes = []
        pending_indexes = []
        previous_round = -1
        for attempt_index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                raise ValueError("point-allocation attempt must be an object")
            _validate_candidate_integer_fields(
                attempt,
                "point-allocation attempt " + str(attempt_index),
            )
            candidate_id = str(attempt.get("candidate_id") or "")
            if not candidate_id or candidate_id in attempt_candidate_ids:
                raise ValueError("point-allocation candidate IDs must be non-empty and unique")
            attempt_candidate_ids.add(candidate_id)
            status = str(attempt.get("status") or "")
            if status not in {"pending", "accepted", "rejected"}:
                raise ValueError("point-allocation attempt status is invalid")
            if status == "accepted":
                accepted_indexes.append(attempt_index)
            elif status == "pending":
                pending_indexes.append(attempt_index)
            attempt_round = _required_integer(
                attempt.get("round"), "point-allocation attempt round"
            )
            attempt["round"] = attempt_round
            mandatory_custom = attempt.get("mandatory_custom", False)
            if not isinstance(mandatory_custom, bool):
                raise ValueError(
                    "point-allocation mandatory_custom must be a JSON Boolean"
                )
            if attempt_index == 0 and attempt_round != 0:
                raise ValueError("point-allocation primary attempt must be round zero")
            if attempt_round <= previous_round:
                raise ValueError(
                    "point-allocation attempt rounds must increase strictly per slot"
                )
            if mandatory_custom and (
                attempt_index != 0 or attempt_round != 0
            ):
                raise ValueError(
                    "mandatory custom attempts must be round-zero candidates"
                )
            previous_round = attempt_round
            _validate_terminal_attempt(
                attempt,
                "point-allocation attempt " + str(attempt_index),
            )
            attempt_locations[candidate_id] = (index, split, attempt)
        if len(pending_indexes) > 1 or (
            pending_indexes and pending_indexes[0] != len(attempts) - 1
        ):
            raise ValueError(
                "only the latest point-allocation attempt may be pending"
            )
        selected = _optional_integer(
            slot.get("accepted_attempt"),
            "point-allocation accepted_attempt",
        )
        slot["accepted_attempt"] = selected
        if selected is None:
            if accepted_indexes:
                raise ValueError("accepted point-allocation attempt is not selected")
        elif accepted_indexes != [selected]:
            raise ValueError("point-allocation accepted_attempt is inconsistent")
        if selected is not None and selected != len(attempts) - 1:
            raise ValueError("accepted point-allocation attempt must be final")
    if split_counts != {split: int(targets[split]) for split in VALID_SPLITS}:
        raise ValueError("point-allocation slot split counts do not match targets")
    consumed_ids: set[str] = set()
    for record in reserve:
        if not isinstance(record, dict):
            raise ValueError("point-allocation reserve record must be an object")
        _validate_candidate_integer_fields(record, "point-allocation reserve")
        candidate_id = str(record.get("candidate_id") or "")
        status = str(record.get("status", "available"))
        if not candidate_id:
            raise ValueError("point-allocation reserve candidate ID must be non-empty")
        if status == "available" and candidate_id in attempt_candidate_ids:
            raise ValueError("available reserve candidate is already assigned to a slot")
        if status == "consumed" and candidate_id not in attempt_candidate_ids:
            raise ValueError("consumed reserve candidate is not assigned to a slot")
        if status not in {"available", "consumed"}:
            raise ValueError("point-allocation reserve status is invalid")
        if status == "consumed":
            consumed_round = _required_integer(
                record.get("consumed_round"),
                "point-allocation consumed reserve round",
                minimum=1,
            )
            record["consumed_round"] = consumed_round
            slot_id, split, attempt = attempt_locations[candidate_id]
            if _required_integer(
                attempt.get("round"), "point-allocation replacement attempt round"
            ) != consumed_round:
                raise ValueError("consumed reserve round does not match its exact attempt")
            if _immutable_candidate_payload(record) != _immutable_candidate_payload(attempt):
                raise ValueError("consumed reserve payload does not match its exact attempt")
            if int(slot_id) < 0 or split not in VALID_SPLITS:
                raise ValueError("consumed reserve attempt has an invalid slot assignment")
            consumed_ids.add(candidate_id)
    reserve_ids = [str(record.get("candidate_id")) for record in reserve]
    if len(set(reserve_ids)) != len(reserve_ids):
        raise ValueError("point-allocation reserve candidate IDs must be unique")
    replacement_attempt_ids = {
        candidate_id
        for candidate_id, (_slot_id, _split, attempt) in attempt_locations.items()
        if _required_integer(
            attempt.get("round"), "point-allocation replacement attempt round"
        ) > 0
    }
    if replacement_attempt_ids != consumed_ids:
        raise ValueError(
            "point-allocation replacement attempts do not exactly match consumed reserve"
        )
    mandatory_failures = [
        attempt
        for _candidate_id, (_slot_id, _split, attempt) in attempt_locations.items()
        if bool(attempt.get("mandatory_custom", False))
        and str(attempt.get("status")) == "rejected"
    ]
    mandatory_flag = data.get("mandatory_custom_failed", False)
    if not isinstance(mandatory_flag, bool):
        raise ValueError("point-allocation mandatory_custom_failed must be a boolean")
    if bool(mandatory_flag) != bool(mandatory_failures):
        raise ValueError("point-allocation mandatory-custom failure flag is inconsistent")
    assignment_sha = str(data.get("slot_assignment_sha256") or "")
    expected_assignment_sha = slot_assignment_sha256(data)
    if assignment_sha != expected_assignment_sha:
        raise ValueError("point-allocation slot assignment SHA mismatch")
    provided_summary = copy.deepcopy(data.get("summary"))
    _refresh_summary(data)
    if provided_summary is not None and provided_summary != data["summary"]:
        raise ValueError("point-allocation summary is inconsistent with allocation records")
    return data


def _read_payload_file(path: Path) -> Dict[str, Any]:
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError("point-allocation manifest missing: " + str(manifest))
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("point-allocation manifest unreadable: " + str(manifest)) from exc
    return _validate_payload(raw)


def _verify_history_chain(
    path: Path,
    current: Mapping[str, Any],
    *,
    history_dir: Optional[Path] = None,
) -> None:
    generation = int(current["generation"])
    history_dir = path.parent / "history" if history_dir is None else Path(history_dir)
    history_files = {
        int(candidate.stem.split("-")[-1]): candidate
        for candidate in history_dir.glob("generation-*.json")
        if candidate.is_file()
        and candidate.stem.split("-")[-1].isdigit()
    } if history_dir.is_dir() else {}
    missing = [value for value in range(generation) if value not in history_files]
    if missing:
        raise ValueError(
            "point-allocation history is missing generation(s): " + repr(missing[:8])
        )
    unexpected = sorted(value for value in history_files if value > generation)
    if unexpected:
        raise ValueError(
            "point-allocation history contains future generation(s): "
            + repr(unexpected[:8])
        )
    previous: Optional[Dict[str, Any]] = None
    for value in range(generation):
        archived = _read_payload_file(history_files[value])
        if int(archived["generation"]) != value:
            raise ValueError(
                "point-allocation history filename/generation mismatch at "
                + str(history_files[value])
            )
        expected_hash = None if previous is None else _generation_sha256(previous)
        if archived.get("previous_generation_sha256") != expected_hash:
            raise ValueError(
                "point-allocation history predecessor mismatch at generation "
                + str(value)
            )
        previous = archived
    expected_current_hash = None if previous is None else _generation_sha256(previous)
    if current.get("previous_generation_sha256") != expected_current_hash:
        raise ValueError("point-allocation current predecessor hash is invalid")
    crash_archive = history_files.get(generation)
    if crash_archive is not None:
        archived_current = _read_payload_file(crash_archive)
        if _generation_sha256(archived_current) != _generation_sha256(current):
            raise ValueError(
                "point-allocation crash-window history conflicts with current generation"
            )


def read_point_allocation(
    path: str | Path,
    *,
    history_dir: Optional[str | Path] = None,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = Path(path)
    payload = _read_payload_file(manifest)
    _verify_history_chain(
        manifest,
        payload,
        history_dir=None if history_dir is None else Path(history_dir),
    )
    if expected_campaign_uid is not None and str(payload["campaign_uid"]) != str(
        expected_campaign_uid
    ):
        raise ValueError("point-allocation campaign UID mismatch")
    return payload


def create_point_allocation(
    path: str | Path,
    *,
    campaign_uid: str,
    context: str,
    iteration: int,
    targets: Mapping[str, int],
    primary_candidates: Sequence[Mapping[str, Any]],
    reserve_candidates: Sequence[Mapping[str, Any]],
    forced_candidate_splits: Optional[Mapping[str, str]] = None,
    mandatory_candidate_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    manifest = Path(path)
    resolved_iteration = _required_integer(iteration, "point-allocation iteration")
    normalised_targets = {
        split: _required_integer(
            targets.get(split), "point-allocation target " + split
        )
        for split in ("train", "int_val", "ext_val")
    }
    normalised_targets["total"] = _required_integer(
        targets.get("total"), "point-allocation target total"
    )
    if normalised_targets["total"] != sum(
        normalised_targets[split] for split in VALID_SPLITS
    ):
        raise ValueError("point-allocation target total is inconsistent")
    targets = normalised_targets
    primary = [_normalise_candidate(record) for record in primary_candidates]
    reserve = [_normalise_candidate(record) for record in reserve_candidates]
    expected_total = int(targets["total"])
    if len(primary) != expected_total:
        raise ValueError(
            "primary candidate count " + str(len(primary))
            + " does not match point-allocation target " + str(expected_total)
        )
    all_ids = [record["candidate_id"] for record in primary + reserve]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("point-allocation candidate IDs contain duplicates")
    forced = {
        str(candidate_id): str(split)
        for candidate_id, split in (forced_candidate_splits or {}).items()
    }
    mandatory = [str(value) for value in mandatory_candidate_ids]
    primary_by_id = {record["candidate_id"]: record for record in primary}
    if any(candidate_id not in primary_by_id for candidate_id in forced):
        raise ValueError("forced candidate IDs must be primary candidates")
    if any(candidate_id not in primary_by_id for candidate_id in mandatory):
        raise ValueError("mandatory candidate IDs must be primary candidates")
    if any(candidate_id not in forced for candidate_id in mandatory):
        raise ValueError("mandatory candidates must have an authoritative split")
    forced_counts = {split: 0 for split in VALID_SPLITS}
    for split in forced.values():
        if split not in VALID_SPLITS:
            raise ValueError("forced candidate split is invalid: " + repr(split))
        forced_counts[split] += 1
    for split in VALID_SPLITS:
        if forced_counts[split] > int(targets[split]):
            raise ValueError("forced candidates exceed " + split + " allocation")

    slots = [
        {"slot_id": index, "split": split, "attempts": [], "accepted_attempt": None}
        for index, split in enumerate(_ordered_slot_splits(targets))
    ]
    mandatory_set = set(mandatory)
    for split in VALID_SPLITS:
        split_slots = [slot for slot in slots if slot["split"] == split]
        split_ids = [
            record["candidate_id"] for record in primary
            if forced.get(record["candidate_id"]) == split
        ]
        for candidate_id, slot in zip(split_ids, split_slots):
            candidate = primary_by_id.pop(candidate_id)
            attempt = {**candidate, "round": 0, "status": "pending"}
            if candidate_id in mandatory_set:
                attempt["mandatory_custom"] = True
            slot["attempts"].append(attempt)
    available_slots = [slot for slot in slots if not slot["attempts"]]
    ordered_ids = _assignment_order(
        primary_by_id,
        campaign_uid=campaign_uid,
        context=context,
        iteration=resolved_iteration,
    )
    for candidate_id, slot in zip(ordered_ids, available_slots):
        slot["attempts"].append(
            {**primary_by_id[candidate_id], "round": 0, "status": "pending"}
        )
    payload: Dict[str, Any] = {
        "schema_version": POINT_ALLOCATION_SCHEMA_VERSION,
        "generation": 0,
        "previous_generation_sha256": None,
        "campaign_uid": str(campaign_uid),
        "context": str(context),
        "iteration": resolved_iteration,
        "targets": {
            "train": int(targets["train"]),
            "int_val": int(targets["int_val"]),
            "ext_val": int(targets["ext_val"]),
            "total": int(targets["total"]),
        },
        "slots": slots,
        "reserve": [{**record, "status": "available"} for record in reserve],
    }
    payload["slot_assignment_sha256"] = slot_assignment_sha256(payload)
    payload = _validate_payload(payload)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with _allocation_lock(manifest):
        if manifest.exists():
            existing = read_point_allocation(manifest)
            immutable_match = (
                str(existing.get("campaign_uid")) == str(payload.get("campaign_uid"))
                and str(existing.get("context")) == str(payload.get("context"))
                and int(existing.get("iteration", -1)) == int(payload.get("iteration", -2))
                and dict(existing.get("targets") or {}) == dict(payload.get("targets") or {})
            )
            existing_round_zero = {
                str(attempt["candidate_id"]): (
                    int(slot["slot_id"]),
                    str(slot["split"]),
                    bool(attempt.get("mandatory_custom", False)),
                )
                for slot in existing["slots"]
                for attempt in list(slot.get("attempts") or [])
                if int(attempt.get("round", -1)) == 0
            }
            requested_round_zero = {
                str(attempt["candidate_id"]): (
                    int(slot["slot_id"]),
                    str(slot["split"]),
                    bool(attempt.get("mandatory_custom", False)),
                )
                for slot in payload["slots"]
                for attempt in list(slot.get("attempts") or [])
            }
            existing_primary_payloads = {
                str(attempt["candidate_id"]): _immutable_candidate_payload(attempt)
                for slot in existing["slots"]
                for attempt in list(slot.get("attempts") or [])
                if int(attempt.get("round", -1)) == 0
            }
            requested_primary_payloads = {
                str(attempt["candidate_id"]): _immutable_candidate_payload(attempt)
                for slot in payload["slots"]
                for attempt in list(slot.get("attempts") or [])
            }
            existing_reserve_payloads = {
                str(record["candidate_id"]): _immutable_candidate_payload(record)
                for record in list(existing.get("reserve") or [])
            }
            requested_reserve_payloads = {
                str(record["candidate_id"]): _immutable_candidate_payload(record)
                for record in list(payload.get("reserve") or [])
            }
            existing_universe = {
                str(attempt["candidate_id"])
                for slot in existing["slots"]
                for attempt in list(slot.get("attempts") or [])
            } | {
                str(record["candidate_id"])
                for record in list(existing.get("reserve") or [])
            }
            if (
                not immutable_match
                or existing_round_zero != requested_round_zero
                or existing_primary_payloads != requested_primary_payloads
                or existing_reserve_payloads != requested_reserve_payloads
                or existing_universe != set(all_ids)
            ):
                raise ValueError("refusing to replace an existing point-allocation manifest")
            return existing
        atomic_write_json(manifest, payload)
    return payload


def _mutate_manifest(path: Path, mutator, *, expected_generation: Optional[int] = None):
    with _allocation_lock(path):
        payload = read_point_allocation(path)
        generation = int(payload.get("generation", 0))
        resolved_expected_generation = _optional_integer(
            expected_generation,
            "expected point-allocation generation",
        )
        if resolved_expected_generation is not None and generation != resolved_expected_generation:
            raise ValueError(
                "stale point-allocation generation: expected "
                + str(resolved_expected_generation) + ", found " + str(generation)
            )
        previous = dict(payload)
        previous.pop("summary", None)
        updated = mutator(copy.deepcopy(payload))
        if updated == payload:
            return payload
        # The mutator changes authoritative records. Recompute their derived
        # summary during validation instead of comparing against the stale copy.
        updated.pop("summary", None)
        updated["previous_generation_sha256"] = _generation_sha256(previous)
        updated["generation"] = generation + 1
        updated = _validate_payload(updated)
        history_dir = path.parent / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / (
            "generation-" + str(generation).zfill(6) + ".json"
        )
        if history_path.exists():
            archived = _read_payload_file(history_path)
            if _sha256_json(archived) != _sha256_json(payload):
                raise ValueError(
                    "point-allocation history conflicts with current generation"
                )
        else:
            atomic_write_json(history_path, payload)
        atomic_write_json(path, updated)
        return updated


def record_quantum_results(
    path: str | Path,
    results: Sequence[Mapping[str, Any]],
    *,
    expected_generation: Optional[int] = None,
    batch_identity: Optional[str] = None,
    result_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = Path(path)
    normalised: Dict[str, Dict[str, Any]] = {}
    for record in results:
        if not isinstance(record, Mapping):
            raise ValueError("quantum result must be a JSON object")
        candidate_id = str(record.get("candidate_id") or "").strip()
        accepted = record.get("accepted")
        if accepted is not True and accepted is not False:
            raise ValueError(
                "quantum result accepted must be an exact JSON Boolean for "
                + repr(candidate_id)
            )
        result = dict(record)
        result["candidate_id"] = candidate_id
        result["accepted"] = accepted
        terminal_probe = {
            "status": "accepted" if accepted else "rejected",
            "pointdir": result.get("pointdir"),
            "reason": None if accepted else result.get("reason"),
            "quality_manifest": result.get("quality_manifest"),
        }
        _validate_terminal_attempt(
            terminal_probe,
            "quantum result " + repr(candidate_id),
        )
        if candidate_id in normalised:
            raise ValueError("quantum result candidate IDs must be unique and non-empty")
        normalised[candidate_id] = result
    if "" in normalised or len(normalised) != len(results):
        raise ValueError("quantum result candidate IDs must be unique and non-empty")

    def mutate(payload):
        applied_batches = payload.setdefault("applied_quantum_batches", [])
        if not isinstance(applied_batches, list):
            raise ValueError("point-allocation applied quantum batches must be a list")
        if batch_identity is not None:
            matching = [
                record for record in applied_batches
                if isinstance(record, Mapping)
                and str(record.get("batch_identity") or "") == str(batch_identity)
            ]
            if matching:
                if len(matching) != 1 or str(
                    matching[0].get("result_fingerprint") or ""
                ) != str(result_fingerprint or ""):
                    raise ValueError("quantum result batch identity conflicts with allocation history")
                return payload
        pending_by_id: Dict[str, Tuple[Dict[str, Any], int, Dict[str, Any]]] = {}
        terminal_by_id: Dict[str, Dict[str, Any]] = {}
        for slot in payload["slots"]:
            for attempt_index, attempt in enumerate(slot["attempts"]):
                if str(attempt.get("status")) == "pending":
                    pending_by_id[str(attempt["candidate_id"])] = (slot, attempt_index, attempt)
                else:
                    terminal_by_id[str(attempt["candidate_id"])] = attempt
        if not pending_by_id and set(normalised).issubset(terminal_by_id):
            for candidate_id, result in normalised.items():
                attempt = terminal_by_id[candidate_id]
                expected_status = "accepted" if result["accepted"] is True else "rejected"
                if str(attempt.get("status")) != expected_status:
                    raise ValueError(
                        "quantum retry conflicts with recorded allocation result for "
                        + candidate_id
                    )
                if str(attempt.get("pointdir") or "") != str(result.get("pointdir") or ""):
                    raise ValueError(
                        "quantum retry pointdir conflicts with recorded allocation result for "
                        + candidate_id
                    )
            return payload
        if set(normalised) != set(pending_by_id):
            raise ValueError(
                "quantum results do not cover exactly the pending allocation candidates; "
                "expected=" + repr(sorted(pending_by_id))
                + " got=" + repr(sorted(normalised))
            )
        for candidate_id, result in normalised.items():
            slot, attempt_index, attempt = pending_by_id[candidate_id]
            accepted = result["accepted"] is True
            pointdir = str(result.get("pointdir") or "").strip()
            if not pointdir:
                raise ValueError("quantum allocation result is missing pointdir")
            attempt["status"] = "accepted" if accepted else "rejected"
            attempt["pointdir"] = pointdir
            attempt["reason"] = None if accepted else str(result.get("reason") or "rejected")
            if result.get("quality_manifest") is not None:
                attempt["quality_manifest"] = str(result.get("quality_manifest"))
            for evidence_key in (
                "quantum_acceptance_receipt",
                "quantum_acceptance_receipt_sha256",
                "accepted_pointdir_content_sha256",
            ):
                if result.get(evidence_key) is not None:
                    attempt[evidence_key] = str(result[evidence_key])
            if accepted:
                if slot.get("accepted_attempt") is not None:
                    raise ValueError("point-allocation slot already has an accepted attempt")
                slot["accepted_attempt"] = int(attempt_index)
            elif bool(attempt.get("mandatory_custom", False)):
                payload["mandatory_custom_failed"] = True
        if batch_identity is not None:
            applied_batches.append(
                {
                    "batch_identity": str(batch_identity),
                    "result_fingerprint": str(result_fingerprint or ""),
                    "source_generation": int(payload.get("generation", 0)),
                    "candidate_ids": sorted(normalised),
                }
            )
        return payload

    return _mutate_manifest(
        manifest,
        mutate,
        expected_generation=expected_generation,
    )


def revalidate_rejected_quantum_results(
    path: str | Path,
    results: Sequence[Mapping[str, Any]],
    *,
    expected_generation: int,
    batch_identity: str,
    result_fingerprint: str,
) -> Dict[str, Any]:
    """Promote an exact set of terminal parser rejections to accepted results."""
    manifest = Path(path)
    normalised: Dict[str, Dict[str, Any]] = {}
    for raw in results:
        if not isinstance(raw, Mapping):
            raise ValueError("quantum revalidation result must be a JSON object")
        result = dict(raw)
        candidate_id = str(result.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in normalised:
            raise ValueError(
                "quantum revalidation candidate IDs must be unique and non-empty"
            )
        if result.get("accepted") is not True:
            raise ValueError("quantum revalidation results must all be accepted")
        if str(result.get("prior_reason") or "") != (
            "dft_model_missing_or_unsupported"
        ):
            raise ValueError("quantum revalidation prior reason is ineligible")
        terminal_probe = {
            "status": "accepted",
            "pointdir": result.get("pointdir"),
            "reason": None,
            "quality_manifest": result.get("quality_manifest"),
            "quantum_acceptance_receipt": result.get(
                "quantum_acceptance_receipt"
            ),
            "quantum_acceptance_receipt_sha256": result.get(
                "quantum_acceptance_receipt_sha256"
            ),
            "accepted_pointdir_content_sha256": result.get(
                "accepted_pointdir_content_sha256"
            ),
        }
        _validate_terminal_attempt(
            terminal_probe,
            "quantum revalidation result " + repr(candidate_id),
        )
        normalised[candidate_id] = result
    if not normalised:
        raise ValueError("quantum revalidation result set is empty")
    for digest, label in (
        (batch_identity, "quantum revalidation batch identity"),
        (result_fingerprint, "quantum revalidation result fingerprint"),
    ):
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(label + " is invalid")

    def mutate(payload):
        applied_batches = payload.setdefault("applied_quantum_batches", [])
        if not isinstance(applied_batches, list):
            raise ValueError(
                "point-allocation applied quantum batches must be a list"
            )
        matching_batches = [
            record
            for record in applied_batches
            if isinstance(record, Mapping)
            and str(record.get("batch_identity") or "") == str(batch_identity)
        ]
        if matching_batches:
            if (
                len(matching_batches) != 1
                or str(matching_batches[0].get("result_fingerprint") or "")
                != str(result_fingerprint)
                or str(matching_batches[0].get("kind") or "")
                != "quality_revalidation"
                or list(matching_batches[0].get("candidate_ids") or [])
                != sorted(normalised)
            ):
                raise ValueError("quantum revalidation batch identity conflicts")
            return payload
        if int(payload.get("generation", -1)) != int(expected_generation):
            raise ValueError(
                "stale point-allocation generation: expected "
                + str(int(expected_generation))
                + ", found "
                + str(int(payload.get("generation", -1)))
            )
        eligible_candidate_ids = set()
        matching_attempts: Dict[
            str, Tuple[Dict[str, Any], int, Dict[str, Any]]
        ] = {}
        for slot in payload["slots"]:
            if slot.get("accepted_attempt") is not None:
                continue
            attempts = list(slot.get("attempts") or [])
            if not attempts:
                continue
            attempt_index = len(attempts) - 1
            attempt = attempts[attempt_index]
            candidate_id = str(attempt.get("candidate_id") or "")
            if (
                str(attempt.get("status") or "") == "rejected"
                and str(attempt.get("reason") or "")
                == "dft_model_missing_or_unsupported"
            ):
                eligible_candidate_ids.add(candidate_id)
            if candidate_id in normalised:
                matching_attempts[candidate_id] = (slot, attempt_index, attempt)
        if (
            set(matching_attempts) != set(normalised)
            or eligible_candidate_ids != set(normalised)
        ):
            raise ValueError(
                "quantum revalidation does not cover exactly the eligible vacant slots"
            )
        for candidate_id, result in normalised.items():
            slot, attempt_index, attempt = matching_attempts[candidate_id]
            if (
                str(attempt.get("status") or "") != "rejected"
                or str(attempt.get("reason") or "") != str(result["prior_reason"])
                or str(attempt.get("pointdir") or "")
                != str(result.get("pointdir") or "")
            ):
                raise ValueError(
                    "quantum revalidation conflicts with recorded rejection for "
                    + candidate_id
                )
            attempt["status"] = "accepted"
            attempt["reason"] = None
            attempt["quality_manifest"] = str(result["quality_manifest"])
            for evidence_key in (
                "quantum_acceptance_receipt",
                "quantum_acceptance_receipt_sha256",
                "accepted_pointdir_content_sha256",
            ):
                attempt[evidence_key] = str(result[evidence_key])
            slot["accepted_attempt"] = int(attempt_index)
        applied_batches.append(
            {
                "batch_identity": str(batch_identity),
                "result_fingerprint": str(result_fingerprint),
                "source_generation": int(payload.get("generation", 0)),
                "candidate_ids": sorted(normalised),
                "kind": "quality_revalidation",
            }
        )
        return payload

    return _mutate_manifest(
        manifest,
        mutate,
        expected_generation=None,
    )


def allocate_replacements(
    path: str | Path,
    *,
    replacement_round: int,
    expected_generation: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_round = _required_integer(
        replacement_round,
        "replacement_round",
        minimum=1,
    )
    manifest = Path(path)

    def mutate(payload):
        if bool(payload.get("mandatory_custom_failed", False)):
            raise ValueError("mandatory custom bootstrap geometry failed; replacement is forbidden")
        vacant = []
        for slot in payload["slots"]:
            if slot.get("accepted_attempt") is not None:
                continue
            attempts = list(slot.get("attempts") or [])
            if attempts and str(attempts[-1].get("status")) == "pending":
                continue
            vacant.append(slot)
        available = [
            record for record in payload["reserve"]
            if str(record.get("status", "available")) == "available"
        ]
        if len(available) < len(vacant):
            raise ValueError(
                "point-allocation reserve exhausted: need " + str(len(vacant))
                + ", available " + str(len(available))
            )
        for slot, reserve_record in zip(vacant, available):
            reserve_record["status"] = "consumed"
            reserve_record["consumed_round"] = resolved_round
            candidate = {
                key: copy.deepcopy(value)
                for key, value in reserve_record.items()
                if key not in {"status", "consumed_round"}
            }
            slot["attempts"].append(
                {**candidate, "round": resolved_round, "status": "pending"}
            )
        return payload

    return _mutate_manifest(
        manifest,
        mutate,
        expected_generation=expected_generation,
    )


def pending_attempts(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    data = _validate_payload(payload)
    pending: List[Dict[str, Any]] = []
    for slot in data["slots"]:
        for attempt in slot["attempts"]:
            if str(attempt.get("status")) == "pending":
                pending.append({**attempt, "slot_id": int(slot["slot_id"]), "split": slot["split"]})
    return pending


def accepted_attempts(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    data = _validate_payload(payload)
    accepted: List[Dict[str, Any]] = []
    for slot in data["slots"]:
        selected = slot.get("accepted_attempt")
        if selected is None:
            continue
        attempt = dict(slot["attempts"][int(selected)])
        accepted.append({**attempt, "slot_id": int(slot["slot_id"]), "split": slot["split"]})
    return accepted


def allocation_manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "POINT_ALLOCATION_SCHEMA_VERSION",
    "POINT_ALLOCATION_FILENAME",
    "allocation_targets",
    "slot_assignment_sha256",
    "stable_candidate_id",
    "point_allocation_path",
    "create_point_allocation",
    "read_point_allocation",
    "record_quantum_results",
    "revalidate_rejected_quantum_results",
    "allocate_replacements",
    "pending_attempts",
    "accepted_attempts",
    "allocation_manifest_sha256",
]
