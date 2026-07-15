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

from ..strict_json import strict_json as json
import errno
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
    "make_lifecycle_context",
]


SCHEMA_VERSION = 8
READABLE_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})
DEFAULT_STATE_FILENAME = "state.json"


class CampaignPhase(str, Enum):
    INIT = "INIT"
    PHASE_A_DIVERSITY = "PHASE_A_DIVERSITY"
    INITIAL_GAUSSIAN = "INITIAL_GAUSSIAN"
    INITIAL_AIMALL = "INITIAL_AIMALL"
    INITIAL_ALLOCATION_CHECK = "INITIAL_ALLOCATION_CHECK"
    INITIAL_REPLACEMENT_GAUSSIAN = "INITIAL_REPLACEMENT_GAUSSIAN"
    INITIAL_REPLACEMENT_AIMALL = "INITIAL_REPLACEMENT_AIMALL"
    INITIAL_FEREBUS = "INITIAL_FEREBUS"
    SEED_SELECT = "SEED_SELECT"
    ARIADNE_ARRAY = "ARIADNE_ARRAY"
    PHASE_B_DIVERSITY = "PHASE_B_DIVERSITY"
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


def make_lifecycle_context(
    *,
    disposition: str,
    reason_code: str,
    message: str,
    from_phase: Union[CampaignPhase, str],
    iteration: int,
    source: str,
    job_id: Optional[str] = None,
    scheduler_uncertain: bool = False,
    recovery_action: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    phase_name = (
        from_phase.value if isinstance(from_phase, CampaignPhase) else str(from_phase)
    )
    payload: Dict[str, Any] = {
        "disposition": str(disposition),
        "reason_code": str(reason_code),
        "message": str(message),
        "from_phase": phase_name,
        "iteration": int(iteration),
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "scheduler_uncertain": bool(scheduler_uncertain),
    }
    if job_id is not None:
        payload["job_id"] = str(job_id)
    if recovery_action is not None:
        payload["recovery_action"] = str(recovery_action)
    if details is not None:
        payload["details"] = dict(details)
    return payload


def _coerce_alpha_history(payload):
    # alpha_history drives the stop check, so a bad value is a schema problem.
    # raise StateSchemaError (not a bare ValueError) so the run loop routes it
    # to the graceful reconcile hint rather than a raw traceback.
    raw = payload.get("alpha_history", [])
    if not isinstance(raw, list):
        raise StateSchemaError("alpha_history must be a JSON list")
    values = []
    for item in raw:
        if not _is_finite_number(item):
            raise StateSchemaError("alpha_history must contain only finite numbers")
        values.append(float(item))
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
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise StateSchemaError(key + " must be an integer")
    return raw


def _coerce_sacct_empty_streak(payload: Dict[str, Any]) -> Dict[str, int]:
    raw = payload.get("sacct_empty_streak", {})
    if not isinstance(raw, dict):
        raise StateSchemaError("sacct_empty_streak must be an object")
    parsed: Dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise StateSchemaError("sacct_empty_streak keys must be non-empty strings")
        if not isinstance(value, int) or isinstance(value, bool):
            raise StateSchemaError("sacct_empty_streak values must be integers")
        parsed[key] = value
        if value < 0:
            raise StateSchemaError("sacct_empty_streak values must be >= 0")
    return parsed


def _coerce_lifecycle_context(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = payload.get("lifecycle_context")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise StateSchemaError("lifecycle_context must be an object or null")
    required = ("disposition", "reason_code", "message", "from_phase", "iteration", "timestamp_iso")
    parsed = dict(raw)
    for key in required:
        if key == "iteration":
            continue
        if not isinstance(parsed.get(key), str) or not parsed[key]:
            raise StateSchemaError("lifecycle_context." + key + " must be a non-empty string")
    if parsed["disposition"] not in {"halted", "stopped", "completed"}:
        raise StateSchemaError(
            "lifecycle_context.disposition must be halted, stopped, or completed"
        )
    if parsed["from_phase"] not in {phase.value for phase in CampaignPhase}:
        raise StateSchemaError("lifecycle_context.from_phase is not a known phase")
    if not isinstance(parsed.get("iteration"), int) or isinstance(
        parsed.get("iteration"), bool
    ):
        raise StateSchemaError("lifecycle_context.iteration must be an integer")
    if parsed["iteration"] < 0:
        raise StateSchemaError("lifecycle_context.iteration must be >= 0")
    try:
        timestamp = datetime.fromisoformat(parsed["timestamp_iso"])
    except ValueError as exc:
        raise StateSchemaError(
            "lifecycle_context.timestamp_iso must be ISO-8601"
        ) from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise StateSchemaError(
            "lifecycle_context.timestamp_iso must include a timezone"
        )
    for key in ("job_id", "exception_type", "source", "recovery_action"):
        value = parsed.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise StateSchemaError("lifecycle_context." + key + " must be a string or null")
    scheduler_uncertain = parsed.get("scheduler_uncertain")
    if scheduler_uncertain is not None and not isinstance(scheduler_uncertain, bool):
        raise StateSchemaError(
            "lifecycle_context.scheduler_uncertain must be a boolean or null"
        )
    details = parsed.get("details")
    if details is not None and not isinstance(details, dict):
        raise StateSchemaError("lifecycle_context.details must be an object or null")
    return parsed


def _coerce_completion_reference(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    raw = payload.get("last_completion_receipt")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise StateSchemaError("last_completion_receipt must be an object or null")
    path = raw.get("path")
    digest = raw.get("sha256")
    receipt_id = raw.get("receipt_id")
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise StateSchemaError("last_completion_receipt.path must be a relative path")
    if ".." in Path(path).parts:
        raise StateSchemaError("last_completion_receipt.path must stay inside the campaign")
    for key, value in (("sha256", digest), ("receipt_id", receipt_id)):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value.lower())
        ):
            raise StateSchemaError("last_completion_receipt." + key + " must be SHA256")
    return {"path": path, "sha256": str(digest).lower(), "receipt_id": str(receipt_id).lower()}


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
    reference_data_version: int = -1
    validation_set_version: int = -1
    models_version: int = -1
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
    lifecycle_context: Optional[Dict[str, Any]] = None
    last_completion_receipt: Optional[Dict[str, str]] = None
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
        if schema not in READABLE_SCHEMA_VERSIONS:
            raise StateSchemaError(
                "unsupported state.json schema_version " + str(schema)
                + "; readable versions are " + repr(sorted(READABLE_SCHEMA_VERSIONS))
            )
        try:
            phase = CampaignPhase(payload["phase"])
        except (KeyError, ValueError) as exc:
            raise StateSchemaError("invalid or missing phase: " + repr(payload.get("phase"))) from exc

        required_str = ("campaign_uid", "campaign_started_iso")
        required_int = ("iteration", "max_iterations",
                         "reference_data_version", "validation_set_version",
                         "models_version", "replacement_round", "stop_streak")
        for key in required_str:
            if not isinstance(payload.get(key), str) or not payload[key]:
                raise StateSchemaError("missing or non-string field: " + key)
        try:
            parsed_started = datetime.fromisoformat(payload["campaign_started_iso"])
        except ValueError as exc:
            raise StateSchemaError("campaign_started_iso must be ISO-8601") from exc
        if parsed_started.tzinfo is None or parsed_started.utcoffset() is None:
            raise StateSchemaError("campaign_started_iso must include a timezone")
        campaign_uid = payload["campaign_uid"]
        if (
            len(campaign_uid) > 128
            or any(
                character
                not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
                for character in campaign_uid
            )
        ):
            raise StateSchemaError(
                "campaign_uid must be a non-empty safe identity token"
            )
        for key in required_int:
            if not isinstance(payload.get(key), int) or isinstance(payload.get(key), bool):
                raise StateSchemaError("missing or non-int field: " + key)
        iteration = int(payload["iteration"])
        max_iterations = int(payload["max_iterations"])
        reference_data_version = int(payload["reference_data_version"])
        validation_set_version = int(payload["validation_set_version"])
        models_version = int(payload["models_version"])
        replacement_round = int(payload["replacement_round"])
        stop_streak = int(payload["stop_streak"])
        if iteration < 0:
            raise StateSchemaError("iteration must be >= 0")
        bootstrap_phases = {
            CampaignPhase.INIT,
            CampaignPhase.PHASE_A_DIVERSITY,
            CampaignPhase.INITIAL_GAUSSIAN,
            CampaignPhase.INITIAL_AIMALL,
            CampaignPhase.INITIAL_ALLOCATION_CHECK,
            CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
            CampaignPhase.INITIAL_REPLACEMENT_AIMALL,
            CampaignPhase.INITIAL_FEREBUS,
        }
        active_phases = {
            CampaignPhase.SEED_SELECT,
            CampaignPhase.ARIADNE_ARRAY,
            CampaignPhase.PHASE_B_DIVERSITY,
            CampaignPhase.SPLIT,
            CampaignPhase.GAUSSIAN,
            CampaignPhase.AIMALL,
            CampaignPhase.ALLOCATION_CHECK,
            CampaignPhase.REPLACEMENT_GAUSSIAN,
            CampaignPhase.REPLACEMENT_AIMALL,
            CampaignPhase.APPEND,
            CampaignPhase.FEREBUS,
            CampaignPhase.STOP_CHECK,
        }
        if phase in bootstrap_phases and iteration != 0:
            raise StateSchemaError("bootstrap phase requires iteration 0")
        if phase in active_phases and iteration < 1:
            raise StateSchemaError("active-learning phase requires iteration >= 1")
        if max_iterations < 1:
            raise StateSchemaError("max_iterations must be >= 1")
        if reference_data_version < -1:
            raise StateSchemaError("reference_data_version must be >= -1")
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
            try:
                from .model_contract import validate_reference_scales

                ref_scales = validate_reference_scales(ref_scales)
            except Exception as exc:
                raise StateSchemaError("reference_scales are invalid: " + str(exc)) from exc

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
        shutdown_requested = payload.get("shutdown_requested", False)
        if not isinstance(shutdown_requested, bool):
            raise StateSchemaError("shutdown_requested must be a JSON boolean")
        lifecycle_context = _coerce_lifecycle_context(payload)
        completion_reference = _coerce_completion_reference(payload)
        if lifecycle_context is not None:
            if int(lifecycle_context["iteration"]) != iteration:
                raise StateSchemaError(
                    "lifecycle_context.iteration must match state iteration"
                )
            disposition = str(lifecycle_context["disposition"])
            if disposition == "halted" and phase is not CampaignPhase.HALTED:
                raise StateSchemaError("halted lifecycle_context requires phase HALTED")
            if disposition == "completed" and phase is not CampaignPhase.DONE:
                raise StateSchemaError("completed lifecycle_context requires phase DONE")
            if disposition == "stopped" and not shutdown_requested:
                raise StateSchemaError("stopped lifecycle_context requires shutdown_requested=true")

        return cls(
            iteration=iteration,
            max_iterations=max_iterations,
            phase=phase,
            pending_jobs=dict(pending),
            reference_data_version=reference_data_version,
            validation_set_version=validation_set_version,
            models_version=models_version,
            replacement_round=replacement_round,
            last_acquisition_alpha0=None if alpha0 is None else float(alpha0),
            stop_streak=stop_streak,
            shutdown_requested=shutdown_requested,
            reference_scales=ref_scales,
            reference_scales_iteration=reference_scales_iteration,
            alpha_history=_coerce_alpha_history(payload),
            last_n_anti_overlap_flagged=last_n_anti_overlap_flagged,
            sacct_empty_streak=sacct_empty_streak,
            lifecycle_context=lifecycle_context,
            last_completion_receipt=completion_reference,
            schema_version=SCHEMA_VERSION,
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


_UNSUPPORTED_FSYNC_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "EINVAL", None),
        getattr(errno, "ENOSYS", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)


def _fsync_file_descriptor(fd: int) -> None:
    """Synchronise a descriptor, ignoring only unsupported-operation errors."""
    try:
        os.fsync(fd)
    except NotImplementedError:
        return
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_FSYNC_ERRNOS:
            return
        raise


def _fsync_parent_dir(target: Path) -> None:
    """Synchronise rename metadata for a durable POSIX publication."""
    if platform.system() == "Windows":
        return
    parent = target.parent
    try:
        fd = os.open(str(parent), os.O_RDONLY)
    except NotImplementedError:
        return
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_FSYNC_ERRNOS:
            return
        raise
    try:
        _fsync_file_descriptor(fd)
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
    tmp = target.parent / (".tmp-" + uuid.uuid4().hex)
    try:
        with open(tmp, "x", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            _fsync_file_descriptor(f.fileno())
        os.replace(str(tmp), str(target))
    finally:
        if tmp.exists():
            tmp.unlink()
    _fsync_parent_dir(target)


def atomic_write_json(target: Union[str, Path], payload: Any) -> None:
    """Serialise "payload" as JSON (indent=2, sort_keys=True) and write it
    atomically via :func:"atomic_write_text"."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(target, text)


def _validated_state_path(path: Union[str, Path]) -> Path:
    """Reject symlinked daemon state paths before reading or replacing them."""
    p = Path(path)
    if (
        p.name == DEFAULT_STATE_FILENAME
        and p.parent.name == "ACTIVE_LEARNING"
        and p.parent.parent.name == ".DATA"
    ):
        from .filesystem import campaign_owned_path

        return campaign_owned_path(p.parent.parent.parent, p)
    if p.is_symlink():
        raise ValueError("state path is a symlink: " + str(p))
    return p



def read_state(path: Union[str, Path]) -> CampaignState:
    """Load a CampaignState from disk and validate it.

    Raises FileNotFoundError if the file does not exist; StateSchemaError on
    schema mismatch or corrupted/malformed contents. 
    JSON parse errors propagate as
    json.JSONDecodeError so callers can distinguish "corrupt file" from
    "schema drift".
    """
    p = _validated_state_path(path)
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return CampaignState.from_dict(payload)



def write_state(path: Union[str, Path], state: CampaignState) -> None:
    """Persist a CampaignState atomically (see :func:`atomic_write_json`)."""
    payload = state.to_dict()
    CampaignState.from_dict(payload)
    atomic_write_json(_validated_state_path(path), payload)
