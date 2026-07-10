"""Daemon campaign state primitives.

Source of truth for the long-running daemon: a single JSON file at
"<campaign-dir>/.DATA/ACTIVE_LEARNING/state.json".

Current write protocol:

    1.open tempfile next to the target on the same filesystem,
    2.write the payload, flush, fsync the file descriptor,
    3.os.replace(tmp, target)  -- POSIX rename on the same fs - 
      simple write() may crash here. We essentially prevent torn reads here;
      no half-written files visible.
    4.fsync the parent directory  -- mandatory on Lustre / NFS for the
      rename to be visible across clients within close-to-open consistency.
      This takes care of the stale visibility issues, i.e.,
      no out-of-date views from other cluster nodes.

"""
from __future__ import annotations

import json
import math
import os
import platform
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


__all__ = [
    "CampaignPhase",
    "CampaignState",
    "StateSchemaError",
    "SCHEMA_VERSION",
    "DEFAULT_STATE_FILENAME",
    "atomic_write_json",
    "atomic_write_text",
    "read_state",
    "write_state",
    "fresh_campaign_state",
]


SCHEMA_VERSION = 4
DEFAULT_STATE_FILENAME = "state.json"


class CampaignPhase(str, Enum):
    INIT = "INIT"
    PHASE_A_POLUS = "PHASE_A_POLUS"
    INITIAL_GAUSSIAN = "INITIAL_GAUSSIAN"
    INITIAL_AIMALL = "INITIAL_AIMALL"
    INITIAL_ALLOCATION_CHECK = "INITIAL_ALLOCATION_CHECK"
    INITIAL_REPLACEMENT_GAUSSIAN = "INITIAL_REPLACEMENT_GAUSSIAN"
    INITIAL_REPLACEMENT_AIMALL = "INITIAL_REPLACEMENT_AIMALL"
    INITIAL_FEREBUS = "INITIAL_FEREBUS"
    SEED_SELECT = "SEED_SELECT"
    ARIADNE_ARRAY = "ARIADNE_ARRAY"
    PHASE_B_POLUS = "PHASE_B_POLUS"
    SPLIT = "SPLIT"
    GAUSSIAN = "GAUSSIAN"
    AIMALL = "AIMALL"
    ALLOCATION_CHECK = "ALLOCATION_CHECK"
    REPLACEMENT_GAUSSIAN = "REPLACEMENT_GAUSSIAN"
    REPLACEMENT_AIMALL = "REPLACEMENT_AIMALL"
    APPEND = "APPEND"
    FEREBUS = "FEREBUS"
    STOP_CHECK = "STOP_CHECK"
    DONE = "DONE"
    HALTED = "HALTED"


class StateSchemaError(ValueError):
    """Raised when state.json fails to validate."""


def _coerce_alpha_history(payload):
    # alpha_history drives the stop check, so a bad value is a schema problem.
    # raise StateSchemaError (not a bare ValueError) so the run loop routes it
    # to the graceful reconcile hint rather than a raw traceback.
    raw = payload.get("alpha_history", []) or []
    try:
        values = [float(x) for x in raw]
    except (TypeError, ValueError) as exc:
        raise StateSchemaError(
            "alpha_history must be a list of numbers, got " + repr(raw)[:120]
        ) from exc
    for value in values:
        if not math.isfinite(value):
            raise StateSchemaError(
                "alpha_history must contain only finite numbers"
            )
    return values


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _coerce_state_int(payload: Dict[str, Any], key: str, default: Any = None) -> int:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        raise StateSchemaError(key + " must be an integer")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise StateSchemaError(key + " must be an integer") from exc


def _coerce_sacct_empty_streak(payload: Dict[str, Any]) -> Dict[str, int]:
    raw = payload.get("sacct_empty_streak", {}) or {}
    if not isinstance(raw, dict):
        raise StateSchemaError("sacct_empty_streak must be an object")
    parsed: Dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            raise StateSchemaError("sacct_empty_streak values must be integers")
        try:
            parsed[str(key)] = int(value)
        except (TypeError, ValueError) as exc:
            raise StateSchemaError("sacct_empty_streak values must be integers") from exc
        if parsed[str(key)] < 0:
            raise StateSchemaError("sacct_empty_streak values must be >= 0")
    return parsed


@dataclass
class CampaignState:
    """One snapshot of the daemon's campaign state.

    The "pending_jobs" mapping
    holds SLURM JobIDs (string or None) keyed by phase name; 
    the daemon polls
    sacct against these between transitions.
    """

    iteration: int = 0
    max_iterations: int = 50
    phase: CampaignPhase = CampaignPhase.INIT
    pending_jobs: Dict[str, Optional[str]] = field(default_factory=dict)
    training_set_version: int = 0
    validation_set_version: int = 0
    models_version: int = 0
    replacement_round: int = 0
    last_acquisition_alpha0: Optional[float] = None
    stop_streak: int = 0
    shutdown_requested: bool = False
    #Cached per-iteration reference scales.
    #The daemon recomputes them before SEED_SELECT according to
    #acquisition.references.refresh_policy and passes the cached dict
    #to every SeedLocalAdversarialAcquisition built in that iteration.
    reference_scales: Optional[Dict[str, float]] = None
    reference_scales_iteration: int = -1
    #Bounded by stop block window in the
    #config. Appended on every STOP_CHECK inline call from state.last_acquisition_alpha0.
    alpha_history: List[float] = field(default_factory=list)
    #Promoted from executor return dict; the count of seeds whose
    #post-ARIADNE whitened distance tripped the anti-overlap thresholds in
    #the most recent ARIADNE_ARRAY phase. Future STOP_CHECK rules may use
    #this as a convergence signal.
    last_n_anti_overlap_flagged: int = 0
    #Per-job sacct empty-result streak. Once a JobID's streak
    #exceeds config.poll_sacct_empty_max_ticks the daemon treats it as
    #UNKNOWN and proceeds to failure handling. Reset to 0 on any non-empty
    #sacct response.
    sacct_empty_streak: Dict[str, int] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    campaign_uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    campaign_started_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CampaignState":
        if not isinstance(payload, dict):
            raise StateSchemaError("state.json must contain a JSON object")
        schema = _coerce_state_int(payload, "schema_version", -1)
        if schema != SCHEMA_VERSION:
            raise StateSchemaError(
                "state.json schema_version " + str(schema)
                + " != " + str(SCHEMA_VERSION) + " (expected)"
            )
        try:
            phase = CampaignPhase(payload["phase"])
        except (KeyError, ValueError) as exc:
            raise StateSchemaError("invalid or missing phase: " + repr(payload.get("phase"))) from exc

        required_str = ("campaign_uid", "campaign_started_iso")
        required_int = ("iteration", "max_iterations",
                         "training_set_version", "validation_set_version",
                         "models_version", "replacement_round", "stop_streak")
        for key in required_str:
            if not isinstance(payload.get(key), str) or not payload[key]:
                raise StateSchemaError("missing or non-string field: " + key)
        for key in required_int:
            if not isinstance(payload.get(key), int) or isinstance(payload.get(key), bool):
                raise StateSchemaError("missing or non-int field: " + key)
        iteration = int(payload["iteration"])
        max_iterations = int(payload["max_iterations"])
        training_set_version = int(payload["training_set_version"])
        validation_set_version = int(payload["validation_set_version"])
        models_version = int(payload["models_version"])
        replacement_round = int(payload["replacement_round"])
        stop_streak = int(payload["stop_streak"])
        if iteration < 0:
            raise StateSchemaError("iteration must be >= 0")
        if max_iterations < 1:
            raise StateSchemaError("max_iterations must be >= 1")
        if training_set_version < -1:
            raise StateSchemaError("training_set_version must be >= -1")
        if validation_set_version < -1:
            raise StateSchemaError("validation_set_version must be >= -1")
        if models_version < -1:
            raise StateSchemaError("models_version must be >= -1")
        if replacement_round < 0:
            raise StateSchemaError("replacement_round must be >= 0")
        if stop_streak < 0:
            raise StateSchemaError("stop_streak must be >= 0")

        pending = payload.get("pending_jobs", {})
        if not isinstance(pending, dict):
            raise StateSchemaError("pending_jobs must be an object")
        known_phase_values = {phase.value for phase in CampaignPhase}
        for k, v in pending.items():
            if not isinstance(k, str):
                raise StateSchemaError("pending_jobs key must be string")
            if k not in known_phase_values:
                raise StateSchemaError("pending_jobs key is not a known phase: " + repr(k))
            if v is not None and not isinstance(v, str):
                raise StateSchemaError("pending_jobs value must be string or null")
            if isinstance(v, str) and not v:
                raise StateSchemaError("pending_jobs value must be non-empty string or null")

        alpha0 = payload.get("last_acquisition_alpha0")
        if alpha0 is not None and not _is_finite_number(alpha0):
            raise StateSchemaError("last_acquisition_alpha0 must be finite number or null")

        # reference_scales was the ONE state field that skipped validation -- it was passed through
        # raw. a malformed one (a list, a string, a dict with non-number values) round-tripped
        # happily and only blew up far later, deep in ARIADNE on a compute node, with an error that
        # never routed to the graceful reconcile path. validate it here: null, or {str -> number}.
        # (booleans are int subclasses in python, so reject those too.) (A31)
        ref_scales = payload.get("reference_scales")
        if ref_scales is not None:
            if not isinstance(ref_scales, dict) or not all(
                isinstance(k, str)
                and _is_finite_number(v)
                for k, v in ref_scales.items()
            ):
                raise StateSchemaError(
                    "reference_scales must be null or an object of string -> finite number"
                )

        reference_scales_iteration = _coerce_state_int(
            payload,
            "reference_scales_iteration",
            -1,
        )
        if reference_scales_iteration < -1:
            raise StateSchemaError("reference_scales_iteration must be >= -1")
        last_n_anti_overlap_flagged = _coerce_state_int(
            payload,
            "last_n_anti_overlap_flagged",
            0,
        )
        if last_n_anti_overlap_flagged < 0:
            raise StateSchemaError("last_n_anti_overlap_flagged must be >= 0")
        sacct_empty_streak = _coerce_sacct_empty_streak(payload)

        return cls(
            iteration=iteration,
            max_iterations=max_iterations,
            phase=phase,
            pending_jobs=dict(pending),
            training_set_version=training_set_version,
            validation_set_version=validation_set_version,
            models_version=models_version,
            replacement_round=replacement_round,
            last_acquisition_alpha0=None if alpha0 is None else float(alpha0),
            stop_streak=stop_streak,
            shutdown_requested=bool(payload.get("shutdown_requested", False)),
            reference_scales=payload.get("reference_scales"),
            reference_scales_iteration=reference_scales_iteration,
            alpha_history=_coerce_alpha_history(payload),
            last_n_anti_overlap_flagged=last_n_anti_overlap_flagged,
            sacct_empty_streak=sacct_empty_streak,
            schema_version=schema,
            campaign_uid=str(payload["campaign_uid"]),
            campaign_started_iso=str(payload["campaign_started_iso"]),
        )

    @property
    def is_terminal(self) -> bool:
        return self.phase in (CampaignPhase.DONE, CampaignPhase.HALTED)


def fresh_campaign_state(
    *,
    max_iterations: int = 50,
    campaign_uid: Optional[str] = None,
    started_iso: Optional[str] = None,
) -> CampaignState:
    return CampaignState(
        max_iterations=int(max_iterations),
        campaign_uid=campaign_uid or str(uuid.uuid4()),
        campaign_started_iso=started_iso or datetime.now(timezone.utc).isoformat(),
    )


def _fsync_parent_dir(target: Path) -> None:
    """Best-effort fsync of the parent directory. Required on POSIX systems
    like CSF4 (i.e. Lustre) for the rename to be durable across crashes and
    consistent across NFS clients."""
    if platform.system() == "Windows":
        return
    parent = target.parent
    try:
        fd = os.open(str(parent), os.O_RDONLY)
    except (OSError, NotImplementedError):
        return
    try:
        os.fsync(fd)
    except (OSError, NotImplementedError):
        pass
    finally:
        os.close(fd)


def atomic_write_text(target: Union[str, Path], text: str) -> None:
    """Write "text" to "target" atomically.

    Sequence:
        tmp = target.<pid>.<uuid>.tmp
        open(tmp, "w"); write; flush; fsync(fd); close
        os.replace(tmp, target)            -- POSIX atomic rename
        fsync(parent_dir)                  -- Lustre / NFS durability
    """
    target = Path(target)
    if not target.parent.exists():
        raise FileNotFoundError("parent directory does not exist: " + str(target.parent))
    tmp = target.with_name(
        target.name + "." + str(os.getpid()) + "." + uuid.uuid4().hex + ".tmp"
    )
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, NotImplementedError):
            #Just to be sure - fsync may be unsupported on some filesystems (tmpfs/test envs).
            pass
    os.replace(str(tmp), str(target))
    _fsync_parent_dir(target)


def atomic_write_json(target: Union[str, Path], payload: Any) -> None:
    """Serialise "payload" as JSON (indent=2, sort_keys=True) and write it
    atomically via :func:"atomic_write_text"."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(target, text)



def read_state(path: Union[str, Path]) -> CampaignState:
    """Load a CampaignState from disk and validate it.

    Raises FileNotFoundError if the file does not exist; StateSchemaError on
    schema mismatch or corrupted/malformed contents. 
    JSON parse errors propagate as
    json.JSONDecodeError so callers can distinguish "corrupt file" from
    "schema drift".
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return CampaignState.from_dict(payload)



def write_state(path: Union[str, Path], state: CampaignState) -> None:
    """Persist a CampaignState atomically (see :func:`atomic_write_json`)."""
    payload = state.to_dict()
    CampaignState.from_dict(payload)
    atomic_write_json(path, payload)
