"""Persistent exact pointdir-level FEREBUS split ledger tests."""
import hashlib
import json

import pytest

from ichor.hpc.active_learning.daemon.ferebus_split_ledger import (
    FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION,
    bootstrap_external_validation_path,
    ensure_split_assignments,
    ledger_path,
)


def _names(n):
    return ["POINT_" + str(i).zfill(6) + ".pointdir" for i in range(n)]


def _forced(names, counts):
    out = {}
    offset = 0
    for split in ("train", "int_val", "ext_val"):
        for name in names[offset : offset + int(counts[split])]:
            out[name] = split
        offset += int(counts[split])
    return out


def _ensure(tmp_path, names, *, version, counts, identities=None, digest="allocation"):
    resolved_identities = identities or {
        name: format(index + 1, "064x") for index, name in enumerate(names)
    }
    return ensure_split_assignments(
        tmp_path,
        names,
        reference_data_version=int(version),
        reference_data_view_sha256=format(int(version) + 1, "064x"),
        expected_new_counts=counts,
        forced_splits=_forced(
            names if version == 0 else names[-sum(counts.values()) :],
            counts,
        ),
        pointdir_identity=resolved_identities,
        allocation_manifest_sha256=hashlib.sha256(digest.encode("utf-8")).hexdigest(),
    )


def test_initial_split_ledger_uses_exact_allocation_counts(tmp_path):
    names = _names(10)
    counts = {"train": 6, "int_val": 2, "ext_val": 2}
    result = _ensure(tmp_path, names, version=0, counts=counts)

    assert result["counts"] == counts
    assert result["row_ids"]["train"] == list(range(6))
    assert result["row_ids"]["int_val"] == [6, 7]
    assert result["row_ids"]["ext_val"] == [8, 9]
    bootstrap = json.loads(
        bootstrap_external_validation_path(tmp_path).read_text(encoding="utf-8")
    )
    assert bootstrap["bootstrap_external_validation_size"] == 2
    assert bootstrap["pointdirs"] == names[8:10]


def test_existing_assignments_never_change_and_new_version_is_exact(tmp_path):
    first_names = _names(10)
    first_counts = {"train": 6, "int_val": 2, "ext_val": 2}
    first = _ensure(tmp_path, first_names, version=0, counts=first_counts)
    first_assignments = dict(first["assignments"])

    grown_names = _names(14)
    second_counts = {"train": 3, "int_val": 1, "ext_val": 0}
    second = _ensure(
        tmp_path,
        grown_names,
        version=1,
        counts=second_counts,
        digest="allocation-v1",
    )

    for name, record in first_assignments.items():
        assert second["assignments"][name]["split"] == record["split"]
    assert second["counts"] == {"train": 9, "int_val": 3, "ext_val": 2}
    assert second["version_allocation"]["actual_new_counts"] == second_counts
    payload = json.loads(ledger_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION
    assert len(payload["assignments"]) == 14


def test_retry_is_idempotent(tmp_path):
    names = _names(4)
    counts = {"train": 3, "int_val": 1, "ext_val": 0}
    first = _ensure(tmp_path, names, version=0, counts=counts)
    second = _ensure(tmp_path, names, version=0, counts=counts)

    assert second["assignments"] == first["assignments"]
    assert second["version_allocation"] == first["version_allocation"]


def test_active_version_cannot_add_external_validation_rows(tmp_path):
    first = _names(3)
    _ensure(
        tmp_path,
        first,
        version=0,
        counts={"train": 1, "int_val": 1, "ext_val": 1},
    )
    with pytest.raises(ValueError, match="cannot add external-validation"):
        _ensure(
            tmp_path,
            _names(6),
            version=1,
            counts={"train": 2, "int_val": 0, "ext_val": 1},
            digest="allocation-v1",
        )


def test_split_ledger_rejects_duplicate_pointdir_names(tmp_path):
    names = ["POINT_000000.pointdir", "POINT_000000.pointdir"]
    with pytest.raises(ValueError, match="duplicate"):
        ensure_split_assignments(
            tmp_path,
            names,
            reference_data_version=0,
            reference_data_view_sha256="1" * 64,
            expected_new_counts={"train": 1, "int_val": 1, "ext_val": 0},
            pointdir_identity={"POINT_000000.pointdir": "1" * 64},
            forced_splits={"POINT_000000.pointdir": "train"},
            allocation_manifest_sha256="a" * 64,
        )


def test_split_ledger_rejects_pointdir_identity_mismatch(tmp_path):
    names = _names(2)
    counts = {"train": 1, "int_val": 1, "ext_val": 0}
    _ensure(
        tmp_path,
        names,
        version=0,
        counts=counts,
        identities={names[0]: "a" * 64, names[1]: "b" * 64},
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        _ensure(
            tmp_path,
            names,
            version=0,
            counts=counts,
            identities={names[0]: "c" * 64, names[1]: "b" * 64},
        )


def test_split_ledger_requires_authoritative_split_for_every_new_point(tmp_path):
    names = _names(2)
    with pytest.raises(ValueError, match="missing authoritative"):
        ensure_split_assignments(
            tmp_path,
            names,
            reference_data_version=0,
            reference_data_view_sha256="1" * 64,
            expected_new_counts={"train": 1, "int_val": 1, "ext_val": 0},
            pointdir_identity={
                names[0]: "a" * 64,
                names[1]: "b" * 64,
            },
            forced_splits={names[0]: "train"},
            allocation_manifest_sha256="c" * 64,
        )
