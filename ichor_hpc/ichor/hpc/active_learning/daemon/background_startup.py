"""Durable status records for detached daemon startup.

``daemon.pid`` remains owned by the daemon process.  This module records the
launcher/child handshake separately so a foreground launcher can stop waiting
without terminating a healthy child whose startup work is still in progress.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from ..strict_json import strict_json as json
from .state import atomic_write_json


BACKGROUND_CHILD_ENV = "ICHOR_DAEMON_BACKGROUND_CHILD"
BACKGROUND_READINESS_ENV = "ICHOR_DAEMON_READINESS_PATH"
BACKGROUND_STARTUP_PATH_ENV = "ICHOR_DAEMON_STARTUP_PATH"
BACKGROUND_LAUNCH_ID_ENV = "ICHOR_DAEMON_LAUNCH_ID"
BACKGROUND_STARTUP_FILENAME = "daemon.startup.json"
BACKGROUND_STARTUP_SCHEMA_VERSION = 1

BACKGROUND_STARTUP_ACTIVE_STATES = frozenset(
    {"prepared", "spawned", "starting", "ownership_acquired"}
)
BACKGROUND_STARTUP_ACKNOWLEDGED_STATES = frozenset(
    {"ownership_acquired", "ready"}
)
BACKGROUND_STARTUP_TERMINAL_STATES = frozenset({"failed", "stopped"})

_STATE_RANK = {
    "prepared": 0,
    "spawned": 1,
    "starting": 2,
    "ownership_acquired": 3,
    "ready": 4,
    "failed": 5,
    "stopped": 5,
}


class BackgroundStartupError(RuntimeError):
    """Raised when a startup record cannot be updated safely."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def read_background_startup(path: Path) -> Dict[str, Any]:
    """Read one startup record without mutating it."""
    path = Path(path)
    if not path.exists():
        return {}
    if path.is_symlink():
        return {
            "state": "invalid",
            "failure": "background startup record must not be a symbolic link",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        return {
            "state": "invalid",
            "failure": type(exc).__name__ + ": " + str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "state": "invalid",
            "failure": "background startup record must contain a JSON object",
        }
    return dict(payload)


def _normalise_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    result["schema_version"] = BACKGROUND_STARTUP_SCHEMA_VERSION
    state = str(result.get("state") or "prepared")
    if state not in _STATE_RANK:
        raise BackgroundStartupError("unknown background startup state " + repr(state))
    result["state"] = state
    launch_id = str(result.get("launch_id") or "").strip()
    if not launch_id:
        raise BackgroundStartupError("background startup launch_id is required")
    result["launch_id"] = launch_id
    now_iso = _utc_now_iso()
    now_unix = float(time.time())
    result.setdefault("started_at_iso", now_iso)
    result.setdefault("started_at_unix", now_unix)
    result["updated_at_iso"] = now_iso
    result["updated_at_unix"] = now_unix
    result.setdefault("stage_history", [])
    return result


def initialise_background_startup(
    path: Path,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Atomically publish a new launch record, replacing stale history."""
    import portalocker

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise BackgroundStartupError(
            "background startup record must not be a symbolic link: " + str(path)
        )
    with portalocker.Lock(str(_lock_path(path)), mode="a+", timeout=30):
        result = _normalise_payload(payload)
        _append_stage_history(result)
        atomic_write_json(path, result)
    return result


def _append_stage_history(payload: Dict[str, Any]) -> None:
    history = payload.get("stage_history")
    if not isinstance(history, list):
        history = []
    state = str(payload.get("state") or "")
    stage = str(payload.get("stage") or "")
    if history:
        latest = history[-1]
        if isinstance(latest, dict) and (
            str(latest.get("state") or "") == state
            and str(latest.get("stage") or "") == stage
        ):
            payload["stage_history"] = history[-32:]
            return
    started = float(payload.get("started_at_unix") or time.time())
    now = float(payload.get("updated_at_unix") or time.time())
    history.append(
        {
            "state": state,
            "stage": stage,
            "at_iso": str(payload.get("updated_at_iso") or _utc_now_iso()),
            "elapsed_seconds": max(0.0, now - started),
        }
    )
    payload["stage_history"] = history[-32:]


def update_background_startup(
    path: Path,
    launch_id: str,
    **updates: Any,
) -> Dict[str, Any]:
    """Update one launch record without allowing stale writers to regress it."""
    import portalocker

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise BackgroundStartupError(
            "background startup record must not be a symbolic link: " + str(path)
        )
    with portalocker.Lock(str(_lock_path(path)), mode="a+", timeout=30):
        current = read_background_startup(path)
        if not current:
            raise BackgroundStartupError(
                "background startup record does not exist: " + str(path)
            )
        observed_launch_id = str(current.get("launch_id") or "")
        if observed_launch_id != str(launch_id):
            raise BackgroundStartupError(
                "background startup launch changed while updating "
                + str(path)
                + ": expected "
                + repr(str(launch_id))
                + ", observed "
                + repr(observed_launch_id)
            )

        requested_state = str(updates.get("state") or current.get("state") or "")
        if requested_state not in _STATE_RANK:
            raise BackgroundStartupError(
                "unknown background startup state " + repr(requested_state)
            )
        current_state = str(current.get("state") or "prepared")
        if current_state not in _STATE_RANK:
            current_state = "prepared"

        # The launcher can publish ``spawned`` after a very fast child has
        # already acquired ownership.  Merge non-state fields in that race,
        # but never move the state or stage backwards.
        if _STATE_RANK[requested_state] < _STATE_RANK[current_state]:
            updates.pop("state", None)
            updates.pop("stage", None)
            requested_state = current_state

        result = dict(current)
        result.update(updates)
        result["schema_version"] = BACKGROUND_STARTUP_SCHEMA_VERSION
        result["launch_id"] = str(launch_id)
        now_iso = _utc_now_iso()
        now_unix = float(time.time())
        result["updated_at_iso"] = now_iso
        result["updated_at_unix"] = now_unix
        result.setdefault("started_at_iso", now_iso)
        result.setdefault("started_at_unix", now_unix)

        state = str(result.get("state") or requested_state)
        timestamp_fields = {
            "spawned": "spawned_at_iso",
            "ownership_acquired": "ownership_acquired_at_iso",
            "ready": "ready_at_iso",
            "failed": "failed_at_iso",
            "stopped": "stopped_at_iso",
        }
        timestamp_field = timestamp_fields.get(state)
        if timestamp_field:
            result.setdefault(timestamp_field, now_iso)
        if result.get("failure") is not None:
            result["failure"] = str(result["failure"])[:1000]
        _append_stage_history(result)
        atomic_write_json(path, result)
    return result


def startup_path_from_environment() -> Path | None:
    raw = (
        str(os.environ.get(BACKGROUND_STARTUP_PATH_ENV) or "").strip()
        or str(os.environ.get(BACKGROUND_READINESS_ENV) or "").strip()
    )
    return Path(raw) if raw else None


def launch_id_from_environment() -> str:
    return str(os.environ.get(BACKGROUND_LAUNCH_ID_ENV) or "").strip()
