"""Deterministic campaign random-seed derivation tests."""

import numpy as np
import pytest

from ichor.hpc.active_learning.randomness import derive_rng_seed


_BASE = {
    "campaign_uid": "campaign-123",
    "campaign_random_seed": 42,
    "iteration": 7,
    "phase": "SEED_SELECT",
    "logical_task_id": "seed-batch",
    "random_purpose": "bulk-seed-selection",
}


def test_rng_derivation_has_stable_golden_vector():
    derived = derive_rng_seed(**_BASE)

    assert derived.digest_sha256 == (
        "253cb9ad3bff7f61748083cc6ac52e11"
        "b47e49bf16dfac9f5509c08098e63531"
    )
    assert derived.derived_seed_128 == 49496739626371260683644968643204558353
    assert derived.rng_algorithm == "numpy.random.PCG64"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("campaign_uid", "campaign-124"),
        ("campaign_random_seed", 43),
        ("iteration", 8),
        ("phase", "ARIADNE_ARRAY"),
        ("logical_task_id", "seed-000001"),
        ("random_purpose", "ariadne-seed-optimisation"),
    ],
)
def test_rng_derivation_is_domain_separated(field, replacement):
    changed = dict(_BASE)
    changed[field] = replacement

    assert derive_rng_seed(**changed).digest_sha256 != derive_rng_seed(**_BASE).digest_sha256


def test_rng_derivation_length_framing_prevents_concatenation_ambiguity():
    first = derive_rng_seed(**{**_BASE, "phase": "ab", "logical_task_id": "c"})
    second = derive_rng_seed(**{**_BASE, "phase": "a", "logical_task_id": "bc"})

    assert first.digest_sha256 != second.digest_sha256


def test_rng_derivation_replays_pcg64_sequence():
    derived = derive_rng_seed(**_BASE)
    first = np.random.Generator(np.random.PCG64(derived.derived_seed_128)).integers(
        0, 2**31, size=10
    )
    second = np.random.Generator(np.random.PCG64(derived.derived_seed_128)).integers(
        0, 2**31, size=10
    )

    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("campaign_random_seed", True),
        ("campaign_random_seed", -1),
        ("iteration", 1.5),
        ("phase", "bad\nphase"),
        ("logical_task_id", ""),
    ],
)
def test_rng_derivation_rejects_noncanonical_inputs(field, value):
    payload = dict(_BASE)
    payload[field] = value

    with pytest.raises(ValueError):
        derive_rng_seed(**payload)
