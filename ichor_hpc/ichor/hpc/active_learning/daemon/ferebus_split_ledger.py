"""Persistent pointdir-level FEREBUS split assignments."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .ferebus_dataset import plan_sizes
from .state import atomic_write_json


FEREBUS_SPLIT_LEDGER_FILENAME = "ferebus_split_assignments.json"
FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION = 1
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


def _target_counts(n_rows: int, fractions: Sequence[float]) -> Dict[str, int]:
    n_tr, n_iv, n_ev = plan_sizes(n_rows, fractions)
    return {"train": n_tr, "int_val": n_iv, "ext_val": n_ev}


def _choose_split(
    assignments: Mapping[str, Mapping[str, Any]],
    *,
    n_after: int,
    fractions: Sequence[float],
) -> str:
    counts = _counts(assignments)
    targets = _target_counts(n_after, fractions)
    deficits = {split: targets[split] - counts[split] for split in _SPLITS}
    return max(_SPLITS, key=lambda split: (deficits[split], split == "train"))


def ensure_split_assignments(
    campaign_dir: Path,
    pointdir_names: Sequence[str],
    *,
    training_version: int,
    fractions: Sequence[float],
    pointdir_identity: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Assign pointdirs to train/internal/external without moving old rows."""
    names = [str(n) for n in pointdir_names]
    if len(set(names)) != len(names):
        raise ValueError("duplicate pointdir names passed to FEREBUS split ledger")
    fractions_tuple = tuple(float(x) for x in fractions)
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
        new_names = [name for name in sorted(names) if name not in assignments]
        if not assignments and new_names:
            sizes = _target_counts(len(new_names), fractions_tuple)
            ordered_splits: List[str] = (
                ["train"] * sizes["train"]
                + ["int_val"] * sizes["int_val"]
                + ["ext_val"] * sizes["ext_val"]
            )
            for name, split in zip(new_names, ordered_splits):
                assignments[name] = {
                    "split": split,
                    "first_seen_training_version": int(training_version),
                    "assignment_version": 1,
                    "fractions_at_assignment": list(fractions_tuple),
                    "provenance_sha256": identities.get(name),
                }
        else:
            for name in new_names:
                split = _choose_split(
                    assignments,
                    n_after=len(assignments) + 1,
                    fractions=fractions_tuple,
                )
                assignments[name] = {
                    "split": split,
                    "first_seen_training_version": int(training_version),
                    "assignment_version": 1,
                    "fractions_at_assignment": list(fractions_tuple),
                    "provenance_sha256": identities.get(name),
                }
        payload = {
            "schema_version": FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION,
            "assignments": assignments,
        }
        atomic_write_json(path, payload)

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
    }
