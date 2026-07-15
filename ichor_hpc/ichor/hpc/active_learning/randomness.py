"""Deterministic, domain-separated random-seed derivation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Dict


RNG_DERIVATION_SCHEMA_VERSION = 1
RNG_DERIVATION_VERSION = "ichor-sha256-pcg64-v1"
RNG_ALGORITHM = "numpy.random.PCG64"
MAX_UNSIGNED_128 = (1 << 128) - 1


@dataclass(frozen=True)
class RngDerivation:
    schema_version: int
    derivation_version: str
    rng_algorithm: str
    campaign_uid: str
    campaign_random_seed: int
    iteration: int
    phase: str
    logical_task_id: str
    random_purpose: str
    digest_sha256: str
    derived_seed_128: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(label + " must be a non-empty string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(label + " contains a control character")
    encoded = value.encode("utf-8")
    if len(encoded) > (1 << 64) - 1:
        raise ValueError(label + " is too long")
    return value


def _required_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(label + " must be an exact integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(label + " must be non-negative")
    return parsed


def _framed_text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(8, byteorder="big", signed=False) + encoded


def _framed_integer(value: int) -> bytes:
    width = max(1, (int(value).bit_length() + 7) // 8)
    encoded = int(value).to_bytes(width, byteorder="big", signed=False)
    return len(encoded).to_bytes(8, byteorder="big", signed=False) + encoded


def derive_rng_seed(
    *,
    campaign_uid: str,
    campaign_random_seed: int,
    iteration: int,
    phase: str,
    logical_task_id: str,
    random_purpose: str,
) -> RngDerivation:
    """Derive the first 128 SHA-256 bits as NumPy PCG64 entropy.

    Text fields are UTF-8 with unsigned 64-bit big-endian length prefixes.
    Integer fields are minimal unsigned big-endian values with unsigned
    64-bit big-endian length prefixes. The fixed domain prefix prevents this
    digest from being confused with any other ICHOR hash.
    """
    uid = _required_text(campaign_uid, "campaign_uid")
    random_seed = _required_nonnegative_integer(
        campaign_random_seed, "campaign_random_seed"
    )
    iteration_value = _required_nonnegative_integer(iteration, "iteration")
    phase_value = _required_text(phase, "phase")
    task_value = _required_text(logical_task_id, "logical_task_id")
    purpose_value = _required_text(random_purpose, "random_purpose")
    payload = b"ICHOR-RNG\x00v1\x00" + b"".join(
        (
            _framed_text(uid),
            _framed_integer(random_seed),
            _framed_integer(iteration_value),
            _framed_text(phase_value),
            _framed_text(task_value),
            _framed_text(purpose_value),
        )
    )
    digest = hashlib.sha256(payload).digest()
    entropy = int.from_bytes(digest[:16], byteorder="big", signed=False)
    if entropy < 0 or entropy > MAX_UNSIGNED_128:
        raise RuntimeError("derived PCG64 entropy is outside unsigned 128-bit range")
    return RngDerivation(
        schema_version=RNG_DERIVATION_SCHEMA_VERSION,
        derivation_version=RNG_DERIVATION_VERSION,
        rng_algorithm=RNG_ALGORITHM,
        campaign_uid=uid,
        campaign_random_seed=random_seed,
        iteration=iteration_value,
        phase=phase_value,
        logical_task_id=task_value,
        random_purpose=purpose_value,
        digest_sha256=digest.hex(),
        derived_seed_128=entropy,
    )


def validate_rng_derivation(
    value: Any,
    *,
    expected_campaign_uid: str | None = None,
    expected_iteration: int | None = None,
    expected_phase: str | None = None,
    expected_logical_task_id: str | None = None,
    expected_random_purpose: str | None = None,
) -> RngDerivation:
    """Validate and reproduce a persisted RNG derivation record.

    Re-deriving the digest is important: merely checking that the stored digest
    has 64 hexadecimal characters would allow a modified seed or purpose to
    silently change stochastic campaign choices after a restart.
    """
    if not isinstance(value, dict):
        raise ValueError("randomness record must be a JSON object")
    schema = _required_nonnegative_integer(
        value.get("schema_version"), "randomness schema_version"
    )
    if schema != RNG_DERIVATION_SCHEMA_VERSION:
        raise ValueError("unsupported randomness schema_version")
    if value.get("derivation_version") != RNG_DERIVATION_VERSION:
        raise ValueError("unsupported randomness derivation_version")
    if value.get("rng_algorithm") != RNG_ALGORITHM:
        raise ValueError("unsupported randomness rng_algorithm")

    observed = derive_rng_seed(
        campaign_uid=_required_text(value.get("campaign_uid"), "campaign_uid"),
        campaign_random_seed=_required_nonnegative_integer(
            value.get("campaign_random_seed"), "campaign_random_seed"
        ),
        iteration=_required_nonnegative_integer(value.get("iteration"), "iteration"),
        phase=_required_text(value.get("phase"), "phase"),
        logical_task_id=_required_text(
            value.get("logical_task_id"), "logical_task_id"
        ),
        random_purpose=_required_text(
            value.get("random_purpose"), "random_purpose"
        ),
    )
    digest = value.get("digest_sha256")
    if not isinstance(digest, str) or digest != observed.digest_sha256:
        raise ValueError("randomness digest_sha256 does not match its inputs")
    stored_entropy = _required_nonnegative_integer(
        value.get("derived_seed_128"), "derived_seed_128"
    )
    if stored_entropy > MAX_UNSIGNED_128:
        raise ValueError("derived_seed_128 exceeds unsigned 128-bit range")
    if stored_entropy != observed.derived_seed_128:
        raise ValueError("derived_seed_128 does not match randomness digest")

    expected = (
        ("campaign_uid", expected_campaign_uid, observed.campaign_uid),
        ("iteration", expected_iteration, observed.iteration),
        ("phase", expected_phase, observed.phase),
        ("logical_task_id", expected_logical_task_id, observed.logical_task_id),
        ("random_purpose", expected_random_purpose, observed.random_purpose),
    )
    for label, wanted, actual in expected:
        if wanted is not None and actual != wanted:
            raise ValueError("randomness " + label + " mismatch")
    return observed


__all__ = [
    "RNG_ALGORITHM",
    "RNG_DERIVATION_SCHEMA_VERSION",
    "RNG_DERIVATION_VERSION",
    "RngDerivation",
    "derive_rng_seed",
    "validate_rng_derivation",
]
