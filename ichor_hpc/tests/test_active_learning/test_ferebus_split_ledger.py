"""Persistent pointdir-level FEREBUS split ledger tests."""
import json

import pytest

from ichor.hpc.active_learning.daemon.ferebus_split_ledger import (
    bootstrap_external_validation_path,
    ensure_split_assignments,
    ledger_path,
)


def _names(n):
    return ["POINT_" + str(i).zfill(4) + ".pointdir" for i in range(n)]


def test_initial_split_ledger_assigns_exact_planned_counts(tmp_path):
    names = _names(10)
    result = ensure_split_assignments(
        tmp_path,
        names,
        training_version=0,
        train_internal_fractions=(0.75, 0.25),
        external_validation_size=2,
    )

    assert result["counts"] == {
        "train": 6,
        "int_val": 2,
        "ext_val": 2,
    }
    assert result["row_ids"]["train"] == [0, 1, 2, 3, 4, 5]
    assert result["row_ids"]["int_val"] == [6, 7]
    assert result["row_ids"]["ext_val"] == [8, 9]
    bootstrap = json.loads(
        bootstrap_external_validation_path(tmp_path).read_text(encoding="utf-8")
    )
    assert bootstrap["external_validation_size"] == 2
    assert bootstrap["pointdirs"] == names[8:10]


def test_existing_split_ledger_assignments_never_change(tmp_path):
    first_names = _names(10)
    first = ensure_split_assignments(
        tmp_path,
        first_names,
        training_version=0,
        train_internal_fractions=(0.75, 0.25),
        external_validation_size=2,
    )
    first_assignments = dict(first["assignments"])

    grown_names = _names(20)
    second = ensure_split_assignments(
        tmp_path,
        grown_names,
        training_version=1,
        train_internal_fractions=(0.75, 0.25),
        external_validation_size=2,
    )

    for name, record in first_assignments.items():
        assert second["assignments"][name]["split"] == record["split"]
    assert second["counts"] == {"train": 14, "int_val": 4, "ext_val": 2}
    assert second["row_ids"]["train"][:6] == [0, 1, 2, 3, 4, 5]
    payload = json.loads(ledger_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert len(payload["assignments"]) == 20


def test_split_ledger_rejects_duplicate_pointdir_names(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        ensure_split_assignments(
            tmp_path,
            ["POINT_0000.pointdir", "POINT_0000.pointdir"],
            training_version=0,
            train_internal_fractions=(0.75, 0.25),
            external_validation_size=2,
        )


def test_split_ledger_rejects_pointdir_identity_mismatch(tmp_path):
    names = _names(3)
    ensure_split_assignments(
        tmp_path,
        names,
        training_version=0,
        train_internal_fractions=(0.75, 0.25),
        external_validation_size=1,
        pointdir_identity={names[0]: "sha-a"},
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        ensure_split_assignments(
            tmp_path,
            names,
            training_version=1,
            train_internal_fractions=(0.75, 0.25),
            external_validation_size=1,
            pointdir_identity={names[0]: "sha-b"},
        )


def test_split_ledger_backfills_missing_pointdir_identity(tmp_path):
    names = _names(2)
    ensure_split_assignments(
        tmp_path,
        names,
        training_version=0,
        train_internal_fractions=(0.5, 0.5),
        external_validation_size=0,
    )
    result = ensure_split_assignments(
        tmp_path,
        names,
        training_version=1,
        train_internal_fractions=(0.5, 0.5),
        external_validation_size=0,
        pointdir_identity={names[1]: "sha-later"},
    )

    assert result["assignments"][names[1]]["provenance_sha256"] == "sha-later"
