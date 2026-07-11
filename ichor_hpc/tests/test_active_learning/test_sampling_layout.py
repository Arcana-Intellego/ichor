from pathlib import Path

import pytest

from ichor.hpc.active_learning.layout import (
    ACTIVE_LEARNING_DIRNAME,
    LEGACY_ACTIVE_LEARNING_DIRNAME,
    LEGACY_BOOTSTRAP_DIRNAME,
    active_iteration_name,
    parse_active_iteration_name,
    parse_seed_directory_name,
    reject_legacy_sampling_layout,
    seed_directory_name,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "iteration-000001"),
        (42, "iteration-000042"),
        (1_000_000, "iteration-1000000"),
    ],
)
def test_active_iteration_names_are_one_based_and_canonical(value, expected):
    assert active_iteration_name(value) == expected
    assert parse_active_iteration_name(expected) == value


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.0],
)
def test_active_iteration_name_rejects_non_positive_or_ambiguous_ids(value):
    with pytest.raises(ValueError):
        active_iteration_name(value)


@pytest.mark.parametrize(
    "name",
    [
        "iteration-0000",
        "iteration-000000",
        "iteration-0000001",
        "iteration-000001-extra",
    ],
)
def test_active_iteration_parser_rejects_noncanonical_names(name):
    with pytest.raises(ValueError):
        parse_active_iteration_name(name)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "seed-000001"),
        (42, "seed-000042"),
        (1_000_000, "seed-1000000"),
    ],
)
def test_seed_directory_names_are_one_based_and_canonical(value, expected):
    assert seed_directory_name(value) == expected
    assert parse_seed_directory_name(expected) == value


@pytest.mark.parametrize(
    "name",
    [
        "seed_000001",
        "seed-0000",
        "seed-000000",
        "seed-0000001",
        "seed-000001-extra",
    ],
)
def test_seed_directory_parser_rejects_noncanonical_names(name):
    with pytest.raises(ValueError):
        parse_seed_directory_name(name)


@pytest.mark.parametrize(
    "legacy_root",
    [LEGACY_BOOTSTRAP_DIRNAME, LEGACY_ACTIVE_LEARNING_DIRNAME],
)
def test_legacy_sampling_roots_are_rejected(tmp_path: Path, legacy_root: str):
    (tmp_path / legacy_root).mkdir()

    with pytest.raises(RuntimeError, match="unsupported legacy sampling layout"):
        reject_legacy_sampling_layout(tmp_path)


def test_noncanonical_active_iteration_entry_is_rejected(tmp_path: Path):
    active_root = tmp_path / ACTIVE_LEARNING_DIRNAME
    active_root.mkdir()
    (active_root / "iteration-0001").mkdir()

    with pytest.raises(RuntimeError, match="invalid active-iteration directory name"):
        reject_legacy_sampling_layout(tmp_path)
