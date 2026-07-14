"""Persistent exact pointdir-level FEREBUS split assignments."""
from __future__ import annotations

from ..strict_json import strict_json as json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .state import atomic_write_json


FEREBUS_SPLIT_LEDGER_FILENAME = "ferebus_split_assignments.json"
BOOTSTRAP_EXTERNAL_VALIDATION_FILENAME = "bootstrap_external_validation.json"
FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION = 6
_LOCK_FILENAME = "ferebus_split_assignments.lock"
_SPLITS = ("train", "int_val", "ext_val")


@contextmanager
def _ledger_lock(campaign_dir: Path):
    import portalocker
    from .filesystem import operational_data_dir

    data = operational_data_dir(campaign_dir)
    data.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(
        str(data / _LOCK_FILENAME),
        mode="a",
        flags=portalocker.LOCK_EX,
        timeout=30.0,
    ):
        yield


def ledger_path(campaign_dir: Path) -> Path:
    from .filesystem import operational_path

    return operational_path(campaign_dir, FEREBUS_SPLIT_LEDGER_FILENAME)


def bootstrap_external_validation_path(campaign_dir: Path) -> Path:
    from .filesystem import operational_path

    return operational_path(campaign_dir, BOOTSTRAP_EXTERNAL_VALIDATION_FILENAME)


def _empty_payload() -> Dict[str, Any]:
    return {
        "schema_version": FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION,
        "allocation_policy": "exact_per_reference_data_version",
        "historical_training_rows": 0,
        "assignments": {},
        "version_allocations": {},
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
        raise ValueError(
            "unsupported FEREBUS split ledger schema; schema 6 is required: "
            + str(path)
        )
    if str(data.get("allocation_policy")) != "exact_per_reference_data_version":
        raise ValueError("FEREBUS split ledger allocation policy is invalid")
    if not isinstance(data.get("assignments"), dict):
        raise ValueError("FEREBUS split ledger assignments must be an object")
    if not isinstance(data.get("version_allocations"), dict):
        raise ValueError("FEREBUS split ledger version_allocations must be an object")
    return data


def _counts(assignments: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    counts = {split: 0 for split in _SPLITS}
    for record in assignments.values():
        split = str(record.get("split"))
        if split not in counts:
            raise ValueError("invalid split in FEREBUS ledger: " + repr(split))
        counts[split] += 1
    return counts


def _normalise_expected(expected_counts: Mapping[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in _SPLITS:
        raw = expected_counts.get(split, 0)
        if isinstance(raw, bool):
            raise ValueError("expected FEREBUS split counts must be integers")
        value = int(raw)
        if value < 0:
            raise ValueError("expected FEREBUS split count must be >= 0 for " + split)
        counts[split] = value
    return counts


def ensure_split_assignments(
    campaign_dir: Path,
    pointdir_names: Sequence[str],
    *,
    reference_data_version: int,
    reference_data_view_sha256: str,
    expected_new_counts: Mapping[str, int],
    pointdir_identity: Optional[Mapping[str, str]] = None,
    forced_splits: Optional[Mapping[str, str]] = None,
    allocation_manifest_sha256: Optional[str] = None,
    historical_training_rows: int = 0,
) -> Dict[str, Any]:
    """Assign only the new pointdirs for one version using exact split quotas."""
    campaign = Path(campaign_dir)
    version = int(reference_data_version)
    if version < 0:
        raise ValueError("reference_data_version must be >= 0")
    view_sha = str(reference_data_view_sha256 or "")
    if len(view_sha) != 64 or any(character not in "0123456789abcdef" for character in view_sha):
        raise ValueError("reference_data_view_sha256 must be a lowercase SHA-256")
    names = [str(name) for name in pointdir_names]
    if len(set(names)) != len(names):
        raise ValueError("duplicate pointdir names passed to FEREBUS split ledger")
    identities = {str(key): str(value) for key, value in (pointdir_identity or {}).items()}
    missing_identities = sorted(set(names) - set(identities))
    if missing_identities:
        raise ValueError(
            "FEREBUS pointdir identities are missing: "
            + repr(missing_identities[:8])
        )
    for name, identity in identities.items():
        if len(identity) != 64 or any(
            character not in "0123456789abcdef" for character in identity
        ):
            raise ValueError("FEREBUS pointdir identity is not a SHA-256 for " + name)
    allocation_sha = str(allocation_manifest_sha256 or "")
    if len(allocation_sha) != 64 or any(
        character not in "0123456789abcdef" for character in allocation_sha
    ):
        raise ValueError("allocation_manifest_sha256 must be a lowercase SHA-256")
    forced = {str(key): str(value) for key, value in (forced_splits or {}).items()}
    expected = _normalise_expected(expected_new_counts)
    if isinstance(historical_training_rows, bool):
        raise ValueError("historical_training_rows must be an integer")
    historical_rows = int(historical_training_rows)
    if historical_rows < 0:
        raise ValueError("historical_training_rows must be >= 0")
    if expected["train"] <= 0 and not (version == 0 and historical_rows > 0):
        raise ValueError(
            "every FEREBUS version must add training rows unless version 0 "
            "has an imported model-backed training baseline"
        )
    if version > 0 and expected["ext_val"] != 0:
        raise ValueError("active FEREBUS versions cannot add external-validation rows")
    if version == 0 and expected["int_val"] <= 0:
        raise ValueError("bootstrap FEREBUS allocation must include internal-validation rows")
    for name, split in forced.items():
        if name not in names:
            raise ValueError("forced FEREBUS split references unknown pointdir: " + name)
        if split not in _SPLITS:
            raise ValueError("invalid forced FEREBUS split for " + name + ": " + split)

    path = ledger_path(campaign)
    with _ledger_lock(campaign):
        payload = _load(path)
        recorded_historical = int(payload.get("historical_training_rows", 0))
        if payload.get("assignments") and recorded_historical != historical_rows:
            raise ValueError("FEREBUS historical training-row count is immutable")
        assignments = {
            str(name): dict(record)
            for name, record in payload["assignments"].items()
        }
        version_allocations = {
            str(key): dict(value)
            for key, value in payload["version_allocations"].items()
        }
        unknown_existing = sorted(set(assignments) - set(names))
        if unknown_existing:
            raise ValueError(
                "committed reference-data view no longer contains ledger pointdirs: "
                + repr(unknown_existing[:8])
            )
        for name in names:
            if name not in assignments:
                continue
            identity = identities.get(name)
            recorded = assignments[name].get("provenance_sha256")
            if identity and recorded and str(identity) != str(recorded):
                raise ValueError("FEREBUS split ledger pointdir identity mismatch for " + name)
            if identity and recorded is None:
                assignments[name]["provenance_sha256"] = identity
            if name in forced and str(assignments[name].get("split")) != forced[name]:
                raise ValueError("committed FEREBUS split conflicts with allocation provenance for " + name)

        new_names = [name for name in names if name not in assignments]
        version_key = str(version)
        existing_version_record = version_allocations.get(version_key)
        if not new_names and existing_version_record is not None:
            if dict(existing_version_record.get("expected_new_counts") or {}) != expected:
                raise ValueError(
                    "FEREBUS retry allocation counts changed for version " + str(version)
                )
            if int(existing_version_record.get("historical_training_rows", 0)) != historical_rows:
                raise ValueError(
                    "FEREBUS retry historical training-row count changed for version "
                    + str(version)
                )
            recorded_hash = str(existing_version_record.get("allocation_manifest_sha256") or "")
            if allocation_sha != recorded_hash:
                raise ValueError(
                    "FEREBUS retry allocation manifest hash changed for version "
                    + str(version)
                )
            if str(existing_version_record.get("reference_data_view_sha256") or "") != view_sha:
                raise ValueError(
                    "FEREBUS retry reference-data view hash changed for version "
                    + str(version)
                )
        elif len(new_names) != sum(expected.values()):
            raise ValueError(
                "new pointdir count does not match exact allocation for reference-data version "
                + str(version) + ": expected " + str(sum(expected.values()))
                + ", found " + str(len(new_names))
            )
        missing_forced = sorted(name for name in new_names if name not in forced)
        if missing_forced:
            raise ValueError(
                "new pointdirs are missing authoritative allocation splits: "
                + repr(missing_forced[:8])
            )
        actual_new = {split: 0 for split in _SPLITS}
        for name in new_names:
            split = forced[name]
            actual_new[split] += 1
        if not new_names and existing_version_record is not None:
            actual_new = {
                split: int((existing_version_record.get("actual_new_counts") or {}).get(split, 0))
                for split in _SPLITS
            }
        if actual_new != expected:
            raise ValueError(
                "new FEREBUS split counts do not match point allocation: expected "
                + repr(expected) + ", found " + repr(actual_new)
            )

        version_record = {
            "reference_data_version": version,
            "reference_data_view_sha256": view_sha,
            "expected_new_counts": dict(expected),
            "actual_new_counts": dict(actual_new),
            "pointdirs": list(new_names),
            "allocation_manifest_sha256": allocation_sha,
            "historical_training_rows": int(historical_rows),
        }
        if existing_version_record is not None and new_names and existing_version_record != version_record:
            raise ValueError("FEREBUS version allocation is immutable for version " + str(version))
        for name in new_names:
            assignments[name] = {
                "split": forced[name],
                "first_seen_reference_data_version": version,
                "assignment_version": 3,
                "allocation_manifest_sha256": allocation_sha,
                "provenance_sha256": identities.get(name),
            }
        if existing_version_record is None:
            version_allocations[version_key] = version_record
        payload = {
            "schema_version": FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION,
            "allocation_policy": "exact_per_reference_data_version",
            "historical_training_rows": int(historical_rows),
            "assignments": assignments,
            "version_allocations": version_allocations,
        }
        atomic_write_json(path, payload)

        external_names = sorted(
            name for name, record in assignments.items()
            if str(record.get("split")) == "ext_val"
        )
        atomic_write_json(
            bootstrap_external_validation_path(campaign),
            {
                "schema_version": 2,
                "ledger_schema_version": FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION,
                "bootstrap_external_validation_size": len(external_names),
                "n_external": len(external_names),
                "pointdirs": external_names,
                "applies_to_bootstrap_only": True,
            },
        )

    row_ids = {split: [] for split in _SPLITS}
    for row_id, name in enumerate(names):
        split = str(assignments[name]["split"])
        row_ids[split].append(int(row_id))
    return {
        "path": str(path),
        "assignments": assignments,
        "row_ids": row_ids,
        "counts": _counts(assignments),
        "effective_counts": {
            **_counts(assignments),
            "train": int(_counts(assignments)["train"]) + int(historical_rows),
        },
        "historical_training_rows": int(historical_rows),
        "version_allocation": dict(version_allocations[str(version)]),
        "allocation_policy": "exact_per_reference_data_version",
    }


__all__ = [
    "FEREBUS_SPLIT_LEDGER_FILENAME",
    "FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION",
    "ledger_path",
    "ensure_split_assignments",
]
