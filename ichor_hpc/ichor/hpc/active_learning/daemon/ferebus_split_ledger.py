"""Persistent exact pointdir-level FEREBUS split assignments."""
from __future__ import annotations

from ..strict_json import strict_json as json
from contextlib import contextmanager
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from .state import atomic_write_json


FEREBUS_SPLIT_LEDGER_FILENAME = "ferebus_split_assignments.json"
BOOTSTRAP_EXTERNAL_VALIDATION_FILENAME = "bootstrap_external_validation.json"
FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION = 7
_LOCK_FILENAME = "ferebus_split_assignments.lock"
_SPLITS = ("train", "int_val", "ext_val")
_SAFE_POINTDIR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class FerebusSplitLedgerLockError(RuntimeError):
    """Raised when split-ledger ownership cannot be acquired in time."""


@contextmanager
def _ledger_lock(campaign_dir: Path):
    import portalocker
    from .filesystem import operational_data_dir

    data = operational_data_dir(campaign_dir)
    data.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(
            str(data / _LOCK_FILENAME),
            mode="a",
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
            timeout=30.0,
        ):
            yield
    except (portalocker.LockException, portalocker.AlreadyLocked) as exc:
        raise FerebusSplitLedgerLockError(
            "could not acquire FEREBUS split-ledger ownership within 30 seconds"
        ) from exc


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


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(label + " must be an exact JSON integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(label + " must be >= " + str(minimum))
    return parsed


def _sha256(value: Any, label: str) -> str:
    text = value if isinstance(value, str) else ""
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(label + " must be a lowercase SHA-256")
    return text


def _pointdir_name(value: Any, label: str) -> str:
    text = value if isinstance(value, str) else ""
    if not _SAFE_POINTDIR_RE.fullmatch(text):
        raise ValueError(label + " must be a safe point-directory name")
    return text


def _validate_payload(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the complete immutable split history and its cross-links."""
    if _exact_int(data.get("schema_version"), "FEREBUS split schema") != FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION:
        raise ValueError("unsupported FEREBUS split ledger schema; schema 7 is required")
    if data.get("allocation_policy") != "exact_per_reference_data_version":
        raise ValueError("FEREBUS split ledger allocation policy is invalid")
    historical_rows = _exact_int(
        data.get("historical_training_rows"),
        "FEREBUS historical_training_rows",
    )
    raw_assignments = data.get("assignments")
    raw_versions = data.get("version_allocations")
    if not isinstance(raw_assignments, dict):
        raise ValueError("FEREBUS split ledger assignments must be an object")
    if not isinstance(raw_versions, dict):
        raise ValueError("FEREBUS split ledger version_allocations must be an object")

    assignments: Dict[str, Dict[str, Any]] = {}
    for raw_name, raw_record in raw_assignments.items():
        name = _pointdir_name(raw_name, "FEREBUS assignment key")
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "split",
            "first_seen_reference_data_version",
            "assignment_version",
            "allocation_manifest_sha256",
            "provenance_sha256",
        }:
            raise ValueError("FEREBUS split assignment has invalid fields for " + name)
        split = raw_record.get("split")
        if split not in _SPLITS:
            raise ValueError("invalid split in FEREBUS ledger for " + name)
        first_seen = _exact_int(
            raw_record.get("first_seen_reference_data_version"),
            "FEREBUS assignment first-seen version",
        )
        if _exact_int(
            raw_record.get("assignment_version"),
            "FEREBUS assignment version",
            minimum=1,
        ) != 4:
            raise ValueError("unsupported FEREBUS assignment record version")
        assignments[name] = {
            "split": split,
            "first_seen_reference_data_version": first_seen,
            "assignment_version": 4,
            "allocation_manifest_sha256": _sha256(
                raw_record.get("allocation_manifest_sha256"),
                "FEREBUS assignment allocation digest",
            ),
            "provenance_sha256": _sha256(
                raw_record.get("provenance_sha256"),
                "FEREBUS assignment provenance digest",
            ),
        }

    versions: Dict[str, Dict[str, Any]] = {}
    assigned_by_versions = set()
    numeric_versions = []
    for raw_key, raw_record in raw_versions.items():
        if not isinstance(raw_key, str) or not raw_key.isdigit():
            raise ValueError("FEREBUS version-allocation key must be a decimal integer")
        version = int(raw_key)
        if raw_key != str(version):
            raise ValueError("FEREBUS version-allocation key is not canonical")
        numeric_versions.append(version)
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "reference_data_version",
            "reference_data_view_sha256",
            "expected_new_counts",
            "actual_new_counts",
            "pointdirs",
            "allocation_manifest_sha256",
            "historical_training_rows",
            "reference_data_manifest_sha256",
        }:
            raise ValueError(
                "FEREBUS version allocation has invalid fields for version "
                + raw_key
            )
        if _exact_int(
            raw_record.get("reference_data_version"),
            "FEREBUS allocation reference-data version",
        ) != version:
            raise ValueError("FEREBUS version allocation identity mismatch")
        expected = _normalise_expected(raw_record.get("expected_new_counts") or {})
        actual = _normalise_expected(raw_record.get("actual_new_counts") or {})
        if expected != actual:
            raise ValueError("FEREBUS version allocation expected/actual counts differ")
        pointdirs_raw = raw_record.get("pointdirs")
        if not isinstance(pointdirs_raw, list):
            raise ValueError("FEREBUS version pointdirs must be a list")
        pointdirs = [
            _pointdir_name(value, "FEREBUS version pointdir")
            for value in pointdirs_raw
        ]
        if len(pointdirs) != len(set(pointdirs)):
            raise ValueError("FEREBUS version allocation contains duplicate pointdirs")
        if len(pointdirs) != sum(actual.values()):
            raise ValueError("FEREBUS version allocation cardinality mismatch")
        allocation_sha = _sha256(
            raw_record.get("allocation_manifest_sha256"),
            "FEREBUS version allocation digest",
        )
        observed_counts = {split: 0 for split in _SPLITS}
        for name in pointdirs:
            if name in assigned_by_versions:
                raise ValueError("FEREBUS pointdir appears in more than one version")
            assigned_by_versions.add(name)
            assignment = assignments.get(name)
            if assignment is None:
                raise ValueError("FEREBUS version references an unknown assignment")
            if assignment["first_seen_reference_data_version"] != version:
                raise ValueError("FEREBUS assignment first-seen version mismatch")
            if assignment["allocation_manifest_sha256"] != allocation_sha:
                raise ValueError("FEREBUS assignment allocation digest mismatch")
            observed_counts[assignment["split"]] += 1
        if observed_counts != actual:
            raise ValueError("FEREBUS version split counts do not match assignments")
        if _exact_int(
            raw_record.get("historical_training_rows"),
            "FEREBUS version historical_training_rows",
        ) != historical_rows:
            raise ValueError("FEREBUS historical training-row count is inconsistent")
        versions[raw_key] = {
            "reference_data_version": version,
            "reference_data_view_sha256": _sha256(
                raw_record.get("reference_data_view_sha256"),
                "FEREBUS reference-data view digest",
            ),
            "expected_new_counts": expected,
            "actual_new_counts": actual,
            "pointdirs": pointdirs,
            "allocation_manifest_sha256": allocation_sha,
            "historical_training_rows": historical_rows,
            "reference_data_manifest_sha256": _sha256(
                raw_record.get("reference_data_manifest_sha256"),
                "FEREBUS reference-data manifest digest",
            ),
        }
    if numeric_versions and sorted(numeric_versions) != list(
        range(max(numeric_versions) + 1)
    ):
        raise ValueError("FEREBUS version allocations are not contiguous from zero")
    if assigned_by_versions != set(assignments):
        raise ValueError("FEREBUS assignments are not fully covered by version history")
    return {
        "schema_version": FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION,
        "allocation_policy": "exact_per_reference_data_version",
        "historical_training_rows": historical_rows,
        "assignments": assignments,
        "version_allocations": versions,
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
    try:
        return _validate_payload(data)
    except ValueError as exc:
        raise ValueError(
            "FEREBUS split ledger is invalid: " + str(path) + ": " + str(exc)
        ) from exc


def _verify_reference_manifest_bindings(
    campaign: Path,
    payload: Mapping[str, Any],
) -> None:
    """Bind every ledger allocation to its immutable reference manifest bytes."""
    from ..versioning.manifest import sha256_file
    from ..versioning.reference_data import (
        REFERENCE_DATA_VERSION_FILENAME,
        REFERENCE_DATA_VERSION_SCHEMA_VERSION,
    )

    versions = payload.get("version_allocations")
    if not isinstance(versions, Mapping):
        raise ValueError("FEREBUS split ledger version history is invalid")
    for raw_version, record in versions.items():
        version = int(raw_version)
        if not isinstance(record, Mapping):
            raise ValueError("FEREBUS reference binding record is invalid")
        path = (
            campaign
            / "QM_REFERENCE_DATA"
            / ("iteration-" + f"{version:06d}")
            / REFERENCE_DATA_VERSION_FILENAME
        )
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "FEREBUS split ledger reference manifest is missing: " + str(path)
            )
        if sha256_file(path) != str(record.get("reference_data_manifest_sha256") or ""):
            raise ValueError(
                "FEREBUS split ledger reference manifest SHA-256 mismatch for version "
                + str(version)
            )
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                "FEREBUS split ledger reference manifest is unreadable"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise ValueError("FEREBUS reference manifest must be an object")
        if _exact_int(
            manifest.get("schema_version"),
            "FEREBUS reference manifest schema",
        ) != REFERENCE_DATA_VERSION_SCHEMA_VERSION:
            raise ValueError("unsupported FEREBUS reference manifest schema")
        if _exact_int(
            manifest.get("reference_data_version"),
            "FEREBUS bound reference-data version",
        ) != version:
            raise ValueError("FEREBUS bound reference-data version mismatch")
        if _sha256(
            manifest.get("point_allocation_sha256"),
            "FEREBUS bound point-allocation digest",
        ) != str(record.get("allocation_manifest_sha256") or ""):
            raise ValueError(
                "FEREBUS split ledger allocation/reference digest mismatch"
            )


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
        counts[split] = _exact_int(
            expected_counts.get(split, 0),
            "expected FEREBUS split count for " + split,
        )
    return counts


def ensure_split_assignments(
    campaign_dir: Path,
    pointdir_names: Sequence[str],
    *,
    reference_data_version: int,
    reference_data_view_sha256: str,
    expected_new_counts: Mapping[str, int],
    pointdir_identity: Optional[Mapping[str, str]] = None,
    pointdir_versions: Optional[Mapping[str, int]] = None,
    forced_splits: Optional[Mapping[str, str]] = None,
    allocation_manifest_sha256: Optional[str] = None,
    historical_training_rows: int = 0,
) -> Dict[str, Any]:
    """Assign only the new pointdirs for one version using exact split quotas."""
    campaign = Path(campaign_dir)
    version = _exact_int(reference_data_version, "reference_data_version")
    view_sha = str(reference_data_view_sha256 or "")
    if len(view_sha) != 64 or any(character not in "0123456789abcdef" for character in view_sha):
        raise ValueError("reference_data_view_sha256 must be a lowercase SHA-256")
    names = [str(name) for name in pointdir_names]
    if len(set(names)) != len(names):
        raise ValueError("duplicate pointdir names passed to FEREBUS split ledger")
    for name in names:
        _pointdir_name(name, "FEREBUS pointdir")
    identities = {str(key): str(value) for key, value in (pointdir_identity or {}).items()}
    versions = {
        str(key): _exact_int(value, "pointdir introduced version")
        for key, value in (pointdir_versions or {}).items()
    }
    missing_identities = sorted(set(names) - set(identities))
    if missing_identities:
        raise ValueError(
            "FEREBUS pointdir identities are missing: "
            + repr(missing_identities[:8])
        )
    missing_versions = sorted(set(names) - set(versions))
    if missing_versions:
        raise ValueError(
            "FEREBUS pointdir introduced versions are missing: "
            + repr(missing_versions[:8])
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
    historical_rows = _exact_int(
        historical_training_rows,
        "historical_training_rows",
    )
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
        if payload.get("assignments"):
            _verify_reference_manifest_bindings(campaign, payload)
        recorded_historical = _exact_int(
            payload.get("historical_training_rows", 0),
            "recorded historical_training_rows",
        )
        if not payload.get("assignments") and version > 0:
            from ..versioning.reference_data import ReferenceDataVersioning
            from ..versioning.manifest import sha256_file

            versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
            reconstructed_assignments: Dict[str, Dict[str, Any]] = {}
            reconstructed_versions: Dict[str, Dict[str, Any]] = {}
            for historical_version in range(version + 1):
                historical_view = versioning.resolve(
                    historical_version,
                    verification="deep",
                )
                names_for_version = [
                    name for name in names if versions[name] == historical_version
                ]
                expected_cumulative_names = [
                    name
                    for name in names
                    if versions[name] <= historical_version
                ]
                observed_cumulative_names = [
                    str(entry.pointdir_name) for entry in historical_view.entries
                ]
                if observed_cumulative_names != expected_cumulative_names:
                    raise ValueError(
                        "cannot reconstruct FEREBUS ledger: reference-data row order "
                        "or introduced-version history disagrees"
                    )
                observed_identities = {
                    str(entry.pointdir_name): str(entry.provenance_sha256)
                    for entry in historical_view.entries
                }
                if any(
                    observed_identities.get(name) != identities[name]
                    for name in expected_cumulative_names
                ):
                    raise ValueError(
                        "cannot reconstruct FEREBUS ledger: reference-data identity "
                        "history disagrees"
                    )
                counts_for_version = {split: 0 for split in _SPLITS}
                for name in names_for_version:
                    split = forced.get(name)
                    if split not in _SPLITS:
                        raise ValueError(
                            "cannot reconstruct FEREBUS split for " + name
                        )
                    counts_for_version[split] += 1
                version_manifest = (
                    campaign
                    / "QM_REFERENCE_DATA"
                    / ("iteration-" + f"{historical_version:06d}")
                    / "REFERENCE_DATA_VERSION.json"
                )
                if not version_manifest.is_file() or version_manifest.is_symlink():
                    raise ValueError(
                        "cannot reconstruct FEREBUS ledger without immutable reference manifest: "
                        + str(version_manifest)
                    )
                reference_payload = json.loads(
                    version_manifest.read_text(encoding="utf-8")
                )
                historical_allocation_sha = _sha256(
                    reference_payload.get("point_allocation_sha256"),
                    "historical reference-data allocation digest",
                )
                reconstructed_versions[str(historical_version)] = {
                    "reference_data_version": historical_version,
                    "reference_data_view_sha256": historical_view.cumulative_view_sha256,
                    "expected_new_counts": dict(counts_for_version),
                    "actual_new_counts": dict(counts_for_version),
                    "pointdirs": list(names_for_version),
                    "allocation_manifest_sha256": historical_allocation_sha,
                    "historical_training_rows": int(historical_rows),
                    "reference_data_manifest_sha256": sha256_file(version_manifest),
                }
                for name in names_for_version:
                    reconstructed_assignments[name] = {
                        "split": forced[name],
                        "first_seen_reference_data_version": historical_version,
                        "assignment_version": 4,
                        "allocation_manifest_sha256": historical_allocation_sha,
                        "provenance_sha256": identities[name],
                    }
            payload = {
                "schema_version": FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION,
                "allocation_policy": "exact_per_reference_data_version",
                "historical_training_rows": int(historical_rows),
                "assignments": reconstructed_assignments,
                "version_allocations": reconstructed_versions,
            }
            recorded_historical = int(historical_rows)
            payload = _validate_payload(payload)
            _verify_reference_manifest_bindings(campaign, payload)
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
            if _exact_int(
                existing_version_record.get("historical_training_rows", 0),
                "version historical_training_rows",
            ) != historical_rows:
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
                split: _exact_int(
                    (existing_version_record.get("actual_new_counts") or {}).get(split, 0),
                    "recorded actual count for " + split,
                )
                for split in _SPLITS
            }
        if actual_new != expected:
            raise ValueError(
                "new FEREBUS split counts do not match point allocation: expected "
                + repr(expected) + ", found " + repr(actual_new)
            )

        current_reference_manifest = (
            campaign
            / "QM_REFERENCE_DATA"
            / ("iteration-" + f"{version:06d}")
            / "REFERENCE_DATA_VERSION.json"
        )
        if not current_reference_manifest.is_file() or current_reference_manifest.is_symlink():
            raise ValueError(
                "FEREBUS split ledger requires the immutable reference-data manifest"
            )
        from ..versioning.manifest import sha256_file

        version_record = {
            "reference_data_version": version,
            "reference_data_view_sha256": view_sha,
            "expected_new_counts": dict(expected),
            "actual_new_counts": dict(actual_new),
            "pointdirs": list(new_names),
            "allocation_manifest_sha256": allocation_sha,
            "historical_training_rows": int(historical_rows),
            "reference_data_manifest_sha256": sha256_file(
                current_reference_manifest
            ),
        }
        if existing_version_record is not None and new_names and existing_version_record != version_record:
            raise ValueError("FEREBUS version allocation is immutable for version " + str(version))
        for name in new_names:
            assignments[name] = {
                "split": forced[name],
                "first_seen_reference_data_version": version,
                "assignment_version": 4,
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
        payload = _validate_payload(payload)
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
