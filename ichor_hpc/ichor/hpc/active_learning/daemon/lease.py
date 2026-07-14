"""Strict daemon lease heartbeat schema and liveness evaluation."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional


LEASE_HEARTBEAT_SCHEMA_VERSION = 2


class LeaseHeartbeatError(ValueError):
    """Raised when daemon lease evidence is malformed or inconsistent."""


@dataclass(frozen=True)
class LeaseLiveness:
    disposition: str
    age_seconds: Optional[float]
    error: Optional[str] = None

    @property
    def fresh(self) -> bool:
        return self.disposition == "fresh"

    @property
    def stale(self) -> bool:
        return self.disposition == "stale"


def _exact_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LeaseHeartbeatError(label + " must be an exact positive integer")
    return value


def validate_lease_heartbeat(
    payload: Any,
    *,
    expected_owner_token: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise LeaseHeartbeatError("daemon lease heartbeat must be a JSON object")
    data = dict(payload)
    if data.get("schema_version") != LEASE_HEARTBEAT_SCHEMA_VERSION:
        raise LeaseHeartbeatError("unsupported daemon lease heartbeat schema")
    owner_token = data.get("owner_token")
    if (
        not isinstance(owner_token, str)
        or len(owner_token) != 32
        or any(character not in "0123456789abcdef" for character in owner_token)
    ):
        raise LeaseHeartbeatError("daemon lease owner_token is invalid")
    if expected_owner_token is not None and owner_token != str(expected_owner_token):
        raise LeaseHeartbeatError("daemon lease owner token no longer matches")
    timestamp = data.get("time")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise LeaseHeartbeatError("daemon lease time must be numeric")
    timestamp_value = float(timestamp)
    if not math.isfinite(timestamp_value) or timestamp_value <= 0.0:
        raise LeaseHeartbeatError("daemon lease time must be finite and positive")
    data["time"] = timestamp_value
    data["pid"] = _exact_positive_int(data.get("pid"), "daemon lease pid")
    host = data.get("host")
    if not isinstance(host, str) or not host.strip():
        raise LeaseHeartbeatError("daemon lease host must be a non-empty string")
    data["host"] = host.strip()
    phase = data.get("phase")
    if phase is not None and (not isinstance(phase, str) or not phase):
        raise LeaseHeartbeatError("daemon lease phase must be a string or null")
    iteration = data.get("iteration")
    if iteration is not None and (
        isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0
    ):
        raise LeaseHeartbeatError(
            "daemon lease iteration must be an exact non-negative integer or null"
        )
    campaign_uid = data.get("campaign_uid")
    if campaign_uid is not None and (
        not isinstance(campaign_uid, str) or not campaign_uid
    ):
        raise LeaseHeartbeatError("daemon lease campaign_uid must be a string or null")
    written_at = data.get("written_at_iso")
    if written_at is not None:
        if not isinstance(written_at, str) or not written_at:
            raise LeaseHeartbeatError("daemon lease written_at_iso is invalid")
        try:
            parsed = datetime.fromisoformat(written_at)
        except ValueError as exc:
            raise LeaseHeartbeatError("daemon lease written_at_iso is invalid") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise LeaseHeartbeatError("daemon lease written_at_iso requires a timezone")
    return data


def evaluate_lease_liveness(
    payload: Any,
    *,
    stale_seconds: int,
    clock_skew_tolerance_seconds: int,
    now: Optional[float] = None,
) -> LeaseLiveness:
    if (
        isinstance(stale_seconds, bool)
        or not isinstance(stale_seconds, int)
        or stale_seconds <= 0
    ):
        raise ValueError("lease stale time must be a positive integer")
    if (
        isinstance(clock_skew_tolerance_seconds, bool)
        or not isinstance(clock_skew_tolerance_seconds, int)
        or clock_skew_tolerance_seconds < 0
    ):
        raise ValueError("clock-skew tolerance must be a non-negative integer")
    try:
        data = validate_lease_heartbeat(payload)
    except LeaseHeartbeatError as exc:
        return LeaseLiveness("invalid", None, str(exc))
    current = time.time() if now is None else float(now)
    age = current - float(data["time"])
    if age < -float(clock_skew_tolerance_seconds):
        return LeaseLiveness(
            "clock_skew",
            age,
            "daemon lease heartbeat is too far in the future",
        )
    bounded_age = max(0.0, age)
    if bounded_age < float(stale_seconds):
        return LeaseLiveness("fresh", bounded_age)
    return LeaseLiveness("stale", bounded_age)


__all__ = [
    "LEASE_HEARTBEAT_SCHEMA_VERSION",
    "LeaseHeartbeatError",
    "LeaseLiveness",
    "evaluate_lease_liveness",
    "validate_lease_heartbeat",
]
