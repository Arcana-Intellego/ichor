"""Persistent pointdir-level FEREBUS split assignments."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .state import atomic_write_json


FEREBUS_SPLIT_LEDGER_FILENAME = "ferebus_split_assignments.json"
BOOTSTRAP_EXTERNAL_VALIDATION_FILENAME = "bootstrap_external_validation.json"
FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION = 3
_LOCK_FILENAME = "ferebus_split_assignments.lock"
_SPLITS = ("train", "int_val", "ext_val")


@contextmanager
def _ledger_lock(campaign_dir: Path):
    import portalocker

    data = Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING"
    data.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(
        str(data / _LOCK_FILENAME),
        mode="a",
        flags=portalocker.LOCK_EX,
        timeout=30.0,
    ):
        yield


def ledger_path(campaign_dir: Path) -> Path:
    return Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / FEREBUS_SPLIT_LEDGER_FILENAME


def bootstrap_external_validation_path(campaign_dir: Path) -> Path:
    return (
        Path(campaign_dir)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / BOOTSTRAP_EXTERNAL_VALIDATION_FILENAME
    )


def _empty_payload() -> Dict[str, Any]:
    return {
        "schema_version": FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION,
        "assignments": {},
    }


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return _empty_payload()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("FEREBUS split ledger unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("FEREBUS split ledger must be a JSON object: " + str(path))
    if int(data.get("schema_version", -1)) != FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION:
        raise ValueError("unsupported FEREBUS split ledger schema: " + str(path))
    assignments = data.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("FEREBUS split ledger assignments must be an object")
    return data


def _counts(assignments: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    out = {name: 0 for name in _SPLITS}
    for record in assignments.values():
        split = str(record.get("split"))
        if split in out:
            out[split] += 1
    return out


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(int(lower), min(int(upper), int(value)))


def _plan_train_internal(n_rows: int, fractions: Sequence[float]) -> Dict[str, int]:
    n = int(n_rows)
    if n <= 0:
        return {"train": 0, "int_val": 0}
    train_fraction, internal_fraction = (float(fractions[0]), float(fractions[1]))
    n_train = int(round(float(n) * train_fraction))
    n_train = _clamp_int(n_train, 1, n)
    n_int = n - n_train
    if n >= 2 and internal_fraction > 0.0 and n_int < 1:
        n_int = 1
        n_train = n - 1
    return {"train": int(n_train), "int_val": int(n_int)}


def _initial_target_counts(
    n_rows: int,
    train_internal_fractions: Sequence[float],
    external_validation_size: int,
) -> Dict[str, int]:
    n = int(n_rows)
    if n <= 0:
        return {"train": 0, "int_val": 0, "ext_val": 0}
    if isinstance(external_validation_size, bool) or not isinstance(
        external_validation_size,
        int,
    ):
        raise ValueError("external_validation_size must be an integer")
    n_ext = int(external_validation_size)
    if n_ext < 0:
        raise ValueError("external_validation_size must be >= 0")
    if n_ext >= n:
        raise ValueError(
            "external_validation_size must be smaller than initial labelled size"
        )
    internal = _plan_train_internal(n - n_ext, train_internal_fractions)
    return {
        "train": internal["train"],
        "int_val": internal["int_val"],
        "ext_val": n_ext,
    }


def plan_initial_split_counts(
    n_rows: int,
    train_internal_fractions: Sequence[float],
    external_validation_size: int,
) -> Dict[str, int]:
    """Public wrapper for the bootstrap split-size contract.

    Anchor planning uses the same arithmetic as the ledger so the Phase-A
    capacity check cannot drift from the actual model-0 train/internal/external
    row assignment.
    """
    return dict(
        _initial_target_counts(
            int(n_rows),
            train_internal_fractions,
            external_validation_size,
        )
    )


def _target_counts_for_incremental(
    total_n: int,
    assignments: Mapping[str, Mapping[str, Any]],
    train_internal_fractions: Sequence[float],
) -> Dict[str, int]:
    counts = _counts(assignments)
    non_external_n = max(0, int(total_n) - int(counts["ext_val"]))
    internal = _plan_train_internal(non_external_n, train_internal_fractions)
    return {
        "train": internal["train"],
        "int_val": internal["int_val"],
        "ext_val": counts["ext_val"],
    }


def _choose_split(
    assignments: Mapping[str, Mapping[str, Any]],
    *,
    n_after: int,
    train_internal_fractions: Sequence[float],
) -> str:
    counts = _counts(assignments)
    targets = _target_counts_for_incremental(
        n_after,
        assignments,
        train_internal_fractions,
    )
    candidate_splits = ("train", "int_val")
    deficits = {split: targets[split] - counts[split] for split in candidate_splits}
    return max(candidate_splits, key=lambda split: (deficits[split], split == "train"))


def ensure_split_assignments(
    campaign_dir: Path,
    pointdir_names: Sequence[str],
    *,
    training_version: int,
    train_internal_fractions: Sequence[float],
    external_validation_size: int,
    pointdir_identity: Optional[Mapping[str, str]] = None,
    forced_splits: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Assign pointdirs to train/internal/external without moving old rows."""
    names = [str(n) for n in pointdir_names]
    if len(set(names)) != len(names):
        raise ValueError("duplicate pointdir names passed to FEREBUS split ledger")
    name_set = set(names)
    forced_map = {str(k): str(v) for k, v in (forced_splits or {}).items()}
    unknown_forced = sorted(name for name in forced_map if name not in name_set)
    if unknown_forced:
        raise ValueError(
            "forced FEREBUS split references unknown pointdirs: "
            + repr(unknown_forced)
        )
    for name, split in sorted(forced_map.items()):
        if split not in _SPLITS:
            raise ValueError(
                "forced FEREBUS split for "
                + name
                + " must be one of "
                + repr(_SPLITS)
            )
    train_internal_tuple = tuple(float(x) for x in train_internal_fractions)
    if isinstance(external_validation_size, bool) or not isinstance(
        external_validation_size,
        int,
    ):
        raise ValueError("external_validation_size must be an integer")
    external_size = int(external_validation_size)
    path = ledger_path(Path(campaign_dir))
    with _ledger_lock(Path(campaign_dir)):
        payload = _load(path)
        assignments: Dict[str, Dict[str, Any]] = {
            str(k): dict(v) for k, v in payload.get("assignments", {}).items()
        }
        identities = {str(k): str(v) for k, v in (pointdir_identity or {}).items()}
        for name in names:
            if name not in assignments:
                continue
            current_identity = identities.get(name)
            if not current_identity:
                continue
            recorded_identity = assignments[name].get("provenance_sha256")
            if recorded_identity is None:
                assignments[name]["provenance_sha256"] = current_identity
            elif str(recorded_identity) != current_identity:
                raise ValueError(
                    "FEREBUS split ledger pointdir identity mismatch for "
                    + name
                )
        for name, split in sorted(forced_map.items()):
            if name in assignments and str(assignments[name].get("split")) != split:
                raise ValueError(
                    "forced FEREBUS split for "
                    + name
                    + " conflicts with existing ledger split "
                    + repr(assignments[name].get("split"))
                )
        new_names = [name for name in sorted(names) if name not in assignments]
        if not assignments and new_names:
            sizes = _initial_target_counts(
                len(new_names),
                train_internal_tuple,
                external_size,
            )
            forced_counts = {split: 0 for split in _SPLITS}
            for name in new_names:
                if name in forced_map:
                    forced_counts[forced_map[name]] += 1
            for split in _SPLITS:
                if forced_counts[split] > sizes[split]:
                    raise ValueError(
                        "forced FEREBUS "
                        + split
                        + " rows exceed planned initial split size: "
                        + str(forced_counts[split])
                        + " > "
                        + str(sizes[split])
                    )
            remaining_splits: List[str] = (
                ["train"] * (sizes["train"] - forced_counts["train"])
                + ["int_val"] * (sizes["int_val"] - forced_counts["int_val"])
                + ["ext_val"] * (sizes["ext_val"] - forced_counts["ext_val"])
            )
            remaining_iter = iter(remaining_splits)
            for name in new_names:
                forced_split = forced_map.get(name)
                split = forced_split if forced_split is not None else next(remaining_iter)
                assignments[name] = {
                    "split": split,
                    "first_seen_training_version": int(training_version),
                    "assignment_version": 1,
                    "train_internal_fractions_at_assignment": list(train_internal_tuple),
                    "external_validation_size_at_assignment": external_size,
                    "provenance_sha256": identities.get(name),
                }
                if forced_split is not None:
                    assignments[name]["forced_split"] = str(forced_split)
                    assignments[name]["forced_split_reason"] = "bootstrap_anchor"
        else:
            for name in new_names:
                forced_split = forced_map.get(name)
                if forced_split is None:
                    split = _choose_split(
                        assignments,
                        n_after=len(assignments) + 1,
                        train_internal_fractions=train_internal_tuple,
                    )
                else:
                    split = forced_split
                assignments[name] = {
                    "split": split,
                    "first_seen_training_version": int(training_version),
                    "assignment_version": 1,
                    "train_internal_fractions_at_assignment": list(train_internal_tuple),
                    "external_validation_size_at_assignment": external_size,
                    "provenance_sha256": identities.get(name),
                }
                if forced_split is not None:
                    assignments[name]["forced_split"] = str(forced_split)
                    assignments[name]["forced_split_reason"] = "bootstrap_anchor"
        payload = {
            "schema_version": FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION,
            "split_policy": {
                "bootstrap_external_validation_size": external_size,
                "ferebus_train_fraction": train_internal_tuple[0],
                "ferebus_internal_validation_fraction": train_internal_tuple[1],
                "external_validation_applies_to_bootstrap_only": True,
                "forced_split_count": int(len(forced_map)),
            },
            "assignments": assignments,
        }
        atomic_write_json(path, payload)
        external_names = [
            name
            for name, record in sorted(assignments.items())
            if str(record.get("split")) == "ext_val"
        ]
        atomic_write_json(
            bootstrap_external_validation_path(Path(campaign_dir)),
            {
                "schema_version": 1,
                "ledger_schema_version": FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION,
                "external_validation_size": external_size,
                "n_external": len(external_names),
                "pointdirs": external_names,
                "applies_to_bootstrap_only": True,
            },
        )

    row_ids = {split: [] for split in _SPLITS}
    missing = []
    for idx, name in enumerate(names):
        record = assignments.get(name)
        if not record:
            missing.append(name)
            continue
        split = str(record.get("split"))
        if split not in row_ids:
            raise ValueError("invalid FEREBUS split in ledger for " + name + ": " + split)
        row_ids[split].append(int(idx))
    if missing:
        raise ValueError("FEREBUS split ledger did not assign: " + repr(missing))
    return {
        "path": str(path),
        "assignments": assignments,
        "row_ids": row_ids,
        "counts": {split: len(row_ids[split]) for split in _SPLITS},
        "split_policy": dict(payload.get("split_policy") or {}),
    }
