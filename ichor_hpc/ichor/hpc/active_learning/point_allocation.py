"""Exact, durable point-allocation contracts for bootstrap and active batches."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .daemon.state import atomic_write_json


POINT_ALLOCATION_SCHEMA_VERSION = 1
POINT_ALLOCATION_FILENAME = "POINT_ALLOCATION.json"
POINT_ALLOCATION_LOCK_FILENAME = "POINT_ALLOCATION.lock"
VALID_CONTEXTS = frozenset({"bootstrap", "active"})
VALID_SPLITS = frozenset({"train", "int_val", "ext_val"})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _generation_sha256(payload: Mapping[str, Any]) -> str:
    canonical = copy.deepcopy(dict(payload))
    canonical.pop("summary", None)
    return _sha256_json(canonical)


def point_allocation_path(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
) -> Path:
    campaign = Path(campaign_dir)
    if str(context) == "bootstrap":
        return campaign / "3_DIVERSITY_SAMPLING" / "initial" / POINT_ALLOCATION_FILENAME
    if str(context) == "active":
        return (
            campaign
            / "7_ACTIVE_LEARNING"
            / ("iteration-" + str(int(iteration)).zfill(4))
            / POINT_ALLOCATION_FILENAME
        )
    raise ValueError("point-allocation context must be bootstrap or active")


@contextmanager
def _allocation_lock(path: Path):
    import portalocker

    lock_path = Path(path).with_name(POINT_ALLOCATION_LOCK_FILENAME)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(
        str(lock_path),
        mode="a",
        flags=portalocker.LOCK_EX,
        timeout=30.0,
    ):
        yield


def allocation_targets(config: Any, context: str) -> Dict[str, int]:
    block = config.point_allocation
    if str(context) == "bootstrap":
        counts = {
            "train": int(block.bootstrap_training_size),
            "int_val": int(block.bootstrap_internal_validation_size),
            "ext_val": int(block.bootstrap_external_validation_size),
        }
    elif str(context) == "active":
        counts = {
            "train": int(block.batch_training_size),
            "int_val": int(block.batch_internal_validation_size),
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
    payload = {
        "campaign_uid": str(campaign_uid),
        "context": str(context),
        "iteration": int(iteration),
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
            return numeric if math.isfinite(numeric) else None
        return value

    candidate = json_safe(dict(record))
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        raise ValueError("point-allocation candidate is missing candidate_id")
    candidate["candidate_id"] = candidate_id
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
    if int(data.get("schema_version", -1)) != POINT_ALLOCATION_SCHEMA_VERSION:
        raise ValueError("unsupported point-allocation manifest schema")
    campaign_uid = str(data.get("campaign_uid") or "").strip()
    if not campaign_uid:
        raise ValueError("point-allocation campaign_uid must be non-empty")
    data["campaign_uid"] = campaign_uid
    context = str(data.get("context") or "")
    if context not in VALID_CONTEXTS:
        raise ValueError("point-allocation manifest context is invalid")
    iteration = int(data.get("iteration", -1))
    if iteration < 0 or (context == "bootstrap" and iteration != 0):
        raise ValueError("point-allocation manifest iteration is invalid")
    data["iteration"] = iteration
    generation = int(data.get("generation", -1))
    if generation < 0:
        raise ValueError("point-allocation generation must be >= 0")
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
        value = int(targets.get(split, -1))
        if value < 0:
            raise ValueError("point-allocation target is invalid for " + split)
        targets[split] = value
    targets["total"] = int(targets.get("total", -1))
    if targets["total"] != sum(targets[split] for split in VALID_SPLITS):
        raise ValueError("point-allocation target total is inconsistent")
    slots = data.get("slots")
    reserve = data.get("reserve")
    if not isinstance(slots, list) or len(slots) != int(targets["total"]):
        raise ValueError("point-allocation slot count does not match target")
    if not isinstance(reserve, list):
        raise ValueError("point-allocation reserve must be a list")
    attempt_candidate_ids: set[str] = set()
    split_counts = {split: 0 for split in VALID_SPLITS}
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict) or int(slot.get("slot_id", -1)) != index:
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
        for attempt_index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                raise ValueError("point-allocation attempt must be an object")
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
            if int(attempt.get("round", -1)) < 0:
                raise ValueError("point-allocation attempt round must be >= 0")
        if len(pending_indexes) > 1 or (
            pending_indexes and pending_indexes[0] != len(attempts) - 1
        ):
            raise ValueError(
                "only the latest point-allocation attempt may be pending"
            )
        selected = slot.get("accepted_attempt")
        if selected is None:
            if accepted_indexes:
                raise ValueError("accepted point-allocation attempt is not selected")
        elif accepted_indexes != [int(selected)]:
            raise ValueError("point-allocation accepted_attempt is inconsistent")
        if selected is not None and int(selected) != len(attempts) - 1:
            raise ValueError("accepted point-allocation attempt must be final")
    if split_counts != {split: int(targets[split]) for split in VALID_SPLITS}:
        raise ValueError("point-allocation slot split counts do not match targets")
    for record in reserve:
        if not isinstance(record, dict):
            raise ValueError("point-allocation reserve record must be an object")
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
    reserve_ids = [str(record.get("candidate_id")) for record in reserve]
    if len(set(reserve_ids)) != len(reserve_ids):
        raise ValueError("point-allocation reserve candidate IDs must be unique")
    _refresh_summary(data)
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


def _verify_history_chain(path: Path, current: Mapping[str, Any]) -> None:
    generation = int(current["generation"])
    history_dir = path.parent / ".point_allocation_history"
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


def read_point_allocation(path: str | Path) -> Dict[str, Any]:
    manifest = Path(path)
    payload = _read_payload_file(manifest)
    _verify_history_chain(manifest, payload)
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
    anchor_candidate_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    manifest = Path(path)
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
    anchors = [str(value) for value in anchor_candidate_ids]
    primary_by_id = {record["candidate_id"]: record for record in primary}
    if any(candidate_id not in primary_by_id for candidate_id in anchors):
        raise ValueError("anchor candidate IDs must be primary candidates")
    if len(anchors) > int(targets["train"]):
        raise ValueError("anchor candidates exceed bootstrap training allocation")

    slots = [
        {"slot_id": index, "split": split, "attempts": [], "accepted_attempt": None}
        for index, split in enumerate(_ordered_slot_splits(targets))
    ]
    free_train = [slot for slot in slots if slot["split"] == "train"]
    for candidate_id, slot in zip(anchors, free_train):
        candidate = primary_by_id.pop(candidate_id)
        slot["attempts"].append(
            {**candidate, "round": 0, "status": "pending", "mandatory_anchor": True}
        )
    available_slots = [slot for slot in slots if not slot["attempts"]]
    ordered_ids = _assignment_order(
        primary_by_id,
        campaign_uid=campaign_uid,
        context=context,
        iteration=iteration,
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
        "iteration": int(iteration),
        "targets": {
            "train": int(targets["train"]),
            "int_val": int(targets["int_val"]),
            "ext_val": int(targets["ext_val"]),
            "total": int(targets["total"]),
        },
        "slots": slots,
        "reserve": [{**record, "status": "available"} for record in reserve],
    }
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
                    bool(attempt.get("mandatory_anchor", False)),
                )
                for slot in existing["slots"]
                for attempt in list(slot.get("attempts") or [])
                if int(attempt.get("round", -1)) == 0
            }
            requested_round_zero = {
                str(attempt["candidate_id"]): (
                    int(slot["slot_id"]),
                    str(slot["split"]),
                    bool(attempt.get("mandatory_anchor", False)),
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
        if expected_generation is not None and generation != int(expected_generation):
            raise ValueError(
                "stale point-allocation generation: expected "
                + str(int(expected_generation)) + ", found " + str(generation)
            )
        previous = dict(payload)
        previous.pop("summary", None)
        updated = mutator(copy.deepcopy(payload))
        if updated == payload:
            return payload
        updated["previous_generation_sha256"] = _generation_sha256(previous)
        updated["generation"] = generation + 1
        updated = _validate_payload(updated)
        history_dir = path.parent / ".point_allocation_history"
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
) -> Dict[str, Any]:
    manifest = Path(path)
    normalised = {str(record.get("candidate_id") or ""): dict(record) for record in results}
    if "" in normalised or len(normalised) != len(results):
        raise ValueError("quantum result candidate IDs must be unique and non-empty")

    def mutate(payload):
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
                expected_status = "accepted" if bool(result.get("accepted", False)) else "rejected"
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
            accepted = bool(result.get("accepted", False))
            pointdir = str(result.get("pointdir") or "").strip()
            if not pointdir:
                raise ValueError("quantum allocation result is missing pointdir")
            attempt["status"] = "accepted" if accepted else "rejected"
            attempt["pointdir"] = pointdir
            attempt["reason"] = None if accepted else str(result.get("reason") or "rejected")
            if result.get("quality_manifest") is not None:
                attempt["quality_manifest"] = str(result.get("quality_manifest"))
            if accepted:
                if slot.get("accepted_attempt") is not None:
                    raise ValueError("point-allocation slot already has an accepted attempt")
                slot["accepted_attempt"] = int(attempt_index)
            elif bool(attempt.get("mandatory_anchor", False)):
                payload["mandatory_anchor_failed"] = True
        return payload

    return _mutate_manifest(
        manifest,
        mutate,
        expected_generation=expected_generation,
    )


def allocate_replacements(
    path: str | Path,
    *,
    replacement_round: int,
    expected_generation: Optional[int] = None,
) -> Dict[str, Any]:
    if int(replacement_round) <= 0:
        raise ValueError("replacement_round must be > 0")
    manifest = Path(path)

    def mutate(payload):
        if bool(payload.get("mandatory_anchor_failed", False)):
            raise ValueError("mandatory bootstrap anchor failed; replacement is forbidden")
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
            reserve_record["consumed_round"] = int(replacement_round)
            candidate = {
                key: copy.deepcopy(value)
                for key, value in reserve_record.items()
                if key not in {"status", "consumed_round"}
            }
            slot["attempts"].append(
                {**candidate, "round": int(replacement_round), "status": "pending"}
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
    "stable_candidate_id",
    "point_allocation_path",
    "create_point_allocation",
    "read_point_allocation",
    "record_quantum_results",
    "allocate_replacements",
    "pending_attempts",
    "accepted_attempts",
    "allocation_manifest_sha256",
]
