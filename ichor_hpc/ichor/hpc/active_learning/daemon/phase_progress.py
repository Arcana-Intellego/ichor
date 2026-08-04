"""Best-effort runtime progress for long daemon phase activities."""
from __future__ import annotations

from ..strict_json import strict_json as json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from .filesystem import operational_data_dir
from .state import atomic_write_json


PHASE_PROGRESS_SCHEMA_VERSION = 1
PHASE_PROGRESS_WRITE_INTERVAL_SECONDS = 2.0
PHASE_PROGRESS_JOURNAL_INTERVAL_SECONDS = 30.0
SCHEDULER_PROGRESS_JOURNAL_INTERVAL_SECONDS = 60.0

PRODUCER_KINDS = frozenset({"local", "scheduler", "worker", "checkpoint"})


_STAGE_LABELS = {
    "campaign_validation": "Validating campaign",
    "lock_acquisition": "Acquiring daemon ownership",
    "lease_acquisition": "Acquiring daemon lease",
    "state_loading": "Loading campaign state",
    "environment_transition": "Checking execution environment",
    "ready": "Daemon ready",
    "shutdown": "Stopping daemon",
    "phase_entry": "Entering phase",
    "handoff_validation": "Validating phase inputs",
    "resource_resolution": "Resolving resources",
    "ariadne_resource_validation": "Validating ARIADNE resource inputs",
    "ariadne_resource_dimensions": "Computing ARIADNE local subspace dimensions",
    "ariadne_resource_bound": "Applying configured ARIADNE resource bound",
    "ariadne_resource_reuse": "Reusing validated ARIADNE resource evidence",
    "resource_rules": "Applying scheduler resource rules",
    "input_staging": "Staging inputs",
    "gaussian_input_staging": "Staging Gaussian inputs",
    "aimall_input_validation": "Validating Gaussian outputs for AIMAll",
    "aimall_task_staging": "Staging AIMAll tasks",
    "script_rendering": "Writing submission scripts",
    "slurm_submission": "Submitting Slurm work",
    "sge_submission": "Submitting Sun Grid Engine work",
    "scheduler_wait": "Waiting for Slurm work",
    "sge_scheduler_wait": "Waiting for Sun Grid Engine work",
    "output_visibility": "Checking output visibility",
    "structural_parsing": "Parsing completed outputs",
    "scientific_quality": "Evaluating scientific quality",
    "receipt_publication": "Publishing task receipts",
    "acceptance_publication": "Publishing accepted outputs",
    "allocation_join": "Updating point allocation",
    "calibration": "Updating calibration evidence",
    "descriptor_construction": "Building diversity descriptors",
    "medoid_selection": "Selecting diversity medoid",
    "farthest_point_sampling": "Selecting diverse geometries",
    "safety_filter": "Filtering unsafe geometries",
    "novelty_filter": "Applying geometry novelty",
    "result_parsing": "Parsing ARIADNE results",
    "landing_classification": "Classifying ARIADNE landings",
    "audit_publication": "Publishing ARIADNE audits",
    "batch_publication": "Publishing ARIADNE batch",
    "row_cache_validation": "Validating FEREBUS row cache",
    "csv_construction": "Building FEREBUS training data",
    "split_ledger": "Applying FEREBUS data split",
    "task_staging": "Staging FEREBUS tasks",
    "model_parsing": "Parsing trained models",
    "quality_metrics": "Evaluating FEREBUS model quality",
    "incumbent_comparison": "Comparing candidate and current models",
    "ferebus_authority": "Binding FEREBUS model authority",
    "ferebus_task_quality": "Validating task-computed FEREBUS quality",
    "ferebus_local_quality": "Computing missing FEREBUS quality locally",
    "ferebus_incumbent_quality": "Reusing incumbent external metrics",
    "ferebus_quality_validation": "Validating FEREBUS quality evidence",
    "promotion_decision": "Recording FEREBUS promotion decision",
    "model_snapshot_build": "Building FEREBUS model snapshot",
    "model_snapshot_validation": "Validating staged FEREBUS model snapshot",
    "model_commit": "Committing FEREBUS models",
    "model_factor_validation": "Validating FEREBUS model factors",
    "model_factor_adoption": "Adopting FEREBUS model factors",
    "model_factor_fallback": "Computing missing FEREBUS model factors locally",
    "allocation_loading": "Loading point allocation",
    "completeness_check": "Checking point allocation",
    "replacement_selection": "Selecting replacement points",
    "split_publication": "Publishing data split",
    "iteration_finalisation": "Finalising active-learning iteration",
    "iteration_authority": "Validating iteration authority",
    "iteration_inventory": "Hashing iteration inventory",
    "iteration_manifest_publication": "Publishing iteration manifest",
    "staging_retirement": "Retiring completed staging",
    "stopping_criteria": "Checking stopping criteria",
    "checkpoint_copy": "Copying checkpoint objects",
    "checkpoint_verification": "Verifying checkpoint",
    "next_iteration": "Preparing next iteration",
    "reference_commit_move": "Publishing accepted QM point directories",
    "reference_commit_cache": "Building committed FEREBUS row cache",
    "reference_commit_publication": "Publishing QM reference metadata",
    "reference_commit_pointer": "Advancing QM reference pointer",
    "reference_commit_complete": "QM reference commit complete",
    "reference_commit_validation": "Checking QM reference transaction",
}


def format_progress_stage(stage: Any) -> str:
    """Return one shared human description for journal and status output."""
    text = str(stage or "").strip()
    if not text:
        return "Working"
    return _STAGE_LABELS.get(text, text.replace("_", " ").strip().capitalize())


def progress_root(campaign_dir: Path) -> Path:
    return operational_data_dir(Path(campaign_dir)) / "runtime_progress"


def progress_path(campaign_dir: Path, phase: str, producer_kind: str) -> Path:
    kind = str(producer_kind).strip().lower()
    if kind not in PRODUCER_KINDS:
        raise ValueError("unsupported phase progress producer kind: " + kind)
    return progress_root(campaign_dir) / (str(phase) + "." + kind + ".json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_optional(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(label + " must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def validate_phase_progress(
    payload: Any,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("phase progress must be a JSON object")
    if payload.get("schema_version") != PHASE_PROGRESS_SCHEMA_VERSION:
        raise ValueError("unsupported phase progress schema")
    out = dict(payload)
    for key in ("campaign_uid", "phase", "producer_kind", "stage", "status"):
        if not isinstance(out.get(key), str) or not str(out[key]).strip():
            raise ValueError("phase progress " + key + " must be a non-empty string")
    if expected_campaign_uid is not None and out["campaign_uid"] != str(
        expected_campaign_uid
    ):
        raise ValueError("phase progress campaign UID mismatch")
    if out["producer_kind"] not in PRODUCER_KINDS:
        raise ValueError("phase progress producer kind is invalid")
    for key in ("iteration", "replacement_round"):
        value = out.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("phase progress " + key + " must be non-negative")
    counters = out.get("counters", {})
    if not isinstance(counters, dict):
        raise ValueError("phase progress counters must be an object")
    for key in ("completed", "total"):
        value = counters.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError("phase progress counter " + key + " is invalid")
    completed = counters.get("completed")
    total = counters.get("total")
    if completed is not None and total is not None and completed > total:
        raise ValueError("phase progress completed count exceeds total")
    _finite_optional(out.get("elapsed_seconds"), "phase progress elapsed_seconds")
    _finite_optional(out.get("throughput"), "phase progress throughput")
    for key in ("started_at_iso", "updated_at_iso"):
        value = out.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError("phase progress " + key + " is missing")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("phase progress " + key + " requires a timezone")
    return out


def read_phase_progress_records(
    campaign_dir: Path,
    *,
    phase: str,
    expected_campaign_uid: Optional[str] = None,
    errors: Optional[list[str]] = None,
) -> list[Dict[str, Any]]:
    """Read bounded sidecars for one phase without traversing campaign data."""
    paths = [
        progress_path(campaign_dir, phase, kind)
        for kind in ("local", "scheduler", "worker", "checkpoint")
    ]
    if str(phase) == "SEED_SELECT":
        paths.append(progress_root(campaign_dir) / "SEED_SELECT.json")
    records: list[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = validate_phase_progress(
                payload,
                expected_campaign_uid=expected_campaign_uid,
            )
            record["_path"] = str(path)
            records.append(record)
        except (OSError, ValueError) as exc:
            if errors is not None and path.name != "SEED_SELECT.json":
                errors.append(
                    path.name + ": " + type(exc).__name__ + ": " + str(exc)
                )
            continue
    records.sort(key=lambda record: str(record.get("updated_at_iso") or ""))
    return records


class PhaseProgressReporter:
    """Publish compact phase progress without participating in phase outcomes."""

    def __init__(
        self,
        campaign_dir: Path,
        *,
        campaign_uid: str,
        phase: str,
        iteration: int,
        replacement_round: int = 0,
        producer_kind: str = "local",
        identity: Optional[Mapping[str, Any]] = None,
        journal_callback: Optional[
            Callable[[str, Mapping[str, Any]], None]
        ] = None,
        write_interval_seconds: float = PHASE_PROGRESS_WRITE_INTERVAL_SECONDS,
        journal_interval_seconds: Optional[float] = None,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.campaign_uid = str(campaign_uid)
        self.phase = str(phase)
        self.iteration = int(iteration)
        self.replacement_round = int(replacement_round)
        self.producer_kind = str(producer_kind).lower()
        self.identity = {
            str(key): value
            for key, value in dict(identity or {}).items()
            if key in {"daemon_pid", "daemon_start_id", "job_id", "attempt_id"}
            and value is not None
        }
        if self.producer_kind not in PRODUCER_KINDS:
            raise ValueError("unsupported phase progress producer kind")
        if self.iteration < 0 or self.replacement_round < 0:
            raise ValueError("phase progress iteration values must be non-negative")
        self.journal_callback = journal_callback
        self.write_interval_seconds = max(0.05, float(write_interval_seconds))
        if journal_interval_seconds is None:
            journal_interval_seconds = (
                SCHEDULER_PROGRESS_JOURNAL_INTERVAL_SECONDS
                if self.producer_kind == "scheduler"
                else PHASE_PROGRESS_JOURNAL_INTERVAL_SECONDS
            )
        self.journal_interval_seconds = max(0.05, float(journal_interval_seconds))
        self.monotonic_fn = monotonic_fn
        self.path = progress_path(
            self.campaign_dir,
            self.phase,
            self.producer_kind,
        )
        self._started_monotonic = float(monotonic_fn())
        self._started_at_iso = _now_iso()
        self._last_write_monotonic = float("-inf")
        self._last_journal_monotonic = float("-inf")
        self._snapshot: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._timing_lock = threading.Lock()
        self._timing_stage = ""
        self._stage_started_monotonic = self._started_monotonic
        self._stage_initial_completed = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._terminal = False

    def __enter__(self) -> "PhaseProgressReporter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is not None:
            self.fail(type(exc).__name__ + ": " + str(exc))
        else:
            self.close()

    def start(
        self,
        stage: str,
        *,
        completed: Optional[int] = None,
        total: Optional[int] = None,
        unit: Optional[str] = None,
        **details: Any,
    ) -> None:
        self._publish(
            stage=stage,
            status="running",
            completed=completed,
            total=total,
            unit=unit,
            force_sidecar=True,
            force_journal=True,
            journal_event=(
                "checkpoint_progress"
                if self.producer_kind == "checkpoint"
                else "phase_activity_started"
            ),
            details=details,
        )
        if self._thread is None:
            try:
                thread = threading.Thread(
                    target=self._heartbeat_worker,
                    name="ichor-phase-progress-" + self.phase,
                    daemon=True,
                )
                thread.start()
                self._thread = thread
            except Exception:
                self._thread = None

    def update(
        self,
        *,
        stage: Optional[str] = None,
        completed: Optional[int] = None,
        total: Optional[int] = None,
        unit: Optional[str] = None,
        throughput: Optional[float] = None,
        **details: Any,
    ) -> None:
        with self._lock:
            prior_stage = str(self._snapshot.get("stage") or "")
        resolved_stage = str(stage or prior_stage or "phase_entry")
        stage_changed = bool(prior_stage and resolved_stage != prior_stage)
        self._publish(
            stage=resolved_stage,
            status="running",
            completed=completed,
            total=total,
            unit=unit,
            throughput=throughput,
            force_sidecar=stage_changed,
            force_journal=stage_changed,
            details=details,
        )

    def complete(self, *, stage: Optional[str] = None, **details: Any) -> None:
        self._stop_thread()
        with self._lock:
            prior_stage = str(self._snapshot.get("stage") or "phase_entry")
            counters = dict(self._snapshot.get("counters") or {})
        self._publish(
            stage=str(stage or prior_stage),
            status="completed",
            completed=counters.get("total", counters.get("completed")),
            total=counters.get("total"),
            unit=counters.get("unit"),
            force_sidecar=True,
            force_journal=True,
            journal_event=(
                "checkpoint_progress"
                if self.producer_kind == "checkpoint"
                else "phase_activity_completed"
            ),
            details=details,
            terminal=True,
        )

    def fail(self, error: str, *, stage: Optional[str] = None) -> None:
        self._stop_thread()
        with self._lock:
            prior_stage = str(self._snapshot.get("stage") or "phase_entry")
            counters = dict(self._snapshot.get("counters") or {})
        self._publish(
            stage=str(stage or prior_stage),
            status="failed",
            completed=counters.get("completed"),
            total=counters.get("total"),
            unit=counters.get("unit"),
            force_sidecar=True,
            force_journal=True,
            journal_event=(
                "checkpoint_progress"
                if self.producer_kind == "checkpoint"
                else "phase_activity_failed"
            ),
            details={"error": str(error)[:180]},
            terminal=True,
        )

    def close(self) -> None:
        self._stop_thread()

    @property
    def thread_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def _stop_thread(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=max(1.0, self.write_interval_seconds + 1.0))
            except Exception:
                pass
        self._thread = None

    def _heartbeat_worker(self) -> None:
        while not self._stop.wait(self.write_interval_seconds):
            try:
                with self._lock:
                    snapshot = dict(self._snapshot)
                if not snapshot or self._terminal:
                    continue
                counters = dict(snapshot.get("counters") or {})
                refreshed = self._record(
                    stage=str(snapshot.get("stage") or "phase_entry"),
                    status=str(snapshot.get("status") or "running"),
                    completed=counters.get("completed"),
                    total=counters.get("total"),
                    unit=counters.get("unit"),
                    throughput=None,
                    details=dict(snapshot.get("details") or {}),
                )
                with self._lock:
                    self._snapshot = refreshed
                self._write_snapshot(refreshed, force=True)
                self._emit_journal(refreshed, force=False)
            except Exception:
                continue

    def _record(
        self,
        *,
        stage: str,
        status: str,
        completed: Optional[int],
        total: Optional[int],
        unit: Optional[str],
        throughput: Optional[float],
        details: Mapping[str, Any],
    ) -> Dict[str, Any]:
        now = float(self.monotonic_fn())
        resolved_throughput = _finite_optional(
            throughput, "progress throughput"
        )
        with self._timing_lock:
            if str(stage) != self._timing_stage:
                self._timing_stage = str(stage)
                self._stage_started_monotonic = now
                self._stage_initial_completed = int(completed or 0)
            stage_elapsed = max(0.0, now - self._stage_started_monotonic)
            if (
                resolved_throughput is None
                and completed is not None
                and stage_elapsed >= 1.0
            ):
                completed_delta = max(
                    0, int(completed) - self._stage_initial_completed
                )
                if completed_delta > 0:
                    resolved_throughput = completed_delta / stage_elapsed
        counters: Dict[str, Any] = {}
        if completed is not None:
            counters["completed"] = int(completed)
        if total is not None:
            counters["total"] = int(total)
        if unit:
            counters["unit"] = str(unit)[:40]
        record: Dict[str, Any] = {
            "schema_version": PHASE_PROGRESS_SCHEMA_VERSION,
            "campaign_uid": self.campaign_uid,
            "phase": self.phase,
            "iteration": self.iteration,
            "replacement_round": self.replacement_round,
            "producer_kind": self.producer_kind,
            "stage": str(stage),
            "status": str(status),
            "counters": counters,
            "elapsed_seconds": max(0.0, now - self._started_monotonic),
            "throughput": resolved_throughput,
            "started_at_iso": self._started_at_iso,
            "updated_at_iso": _now_iso(),
        }
        record.update(self.identity)
        compact_details: Dict[str, Any] = {}
        for key, value in list(details.items())[:12]:
            if value is None or isinstance(value, (str, int, float, bool)):
                compact_details[str(key)] = (
                    value[:160] if isinstance(value, str) else value
                )
        if compact_details:
            record["details"] = compact_details
        validate_phase_progress(record, expected_campaign_uid=self.campaign_uid)
        return record

    def _publish(
        self,
        *,
        stage: str,
        status: str,
        completed: Optional[int] = None,
        total: Optional[int] = None,
        unit: Optional[str] = None,
        throughput: Optional[float] = None,
        force_sidecar: bool,
        force_journal: bool,
        journal_event: Optional[str] = None,
        details: Mapping[str, Any],
        terminal: bool = False,
    ) -> None:
        if self._terminal:
            return
        try:
            record = self._record(
                stage=stage,
                status=status,
                completed=completed,
                total=total,
                unit=unit,
                throughput=throughput,
                details=details,
            )
            with self._lock:
                self._snapshot = record
            self._write_snapshot(record, force=force_sidecar)
            self._emit_journal(
                record,
                force=force_journal,
                event_type=journal_event,
            )
        except Exception:
            pass
        finally:
            if terminal:
                self._terminal = True

    def _write_snapshot(self, record: Mapping[str, Any], *, force: bool) -> None:
        now = float(self.monotonic_fn())
        if not force and now - self._last_write_monotonic < self.write_interval_seconds:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.path, dict(record))
            self._last_write_monotonic = now
        except Exception:
            return

    def _emit_journal(
        self,
        record: Mapping[str, Any],
        *,
        force: bool,
        event_type: Optional[str] = None,
    ) -> None:
        callback = self.journal_callback
        if callback is None:
            return
        now = float(self.monotonic_fn())
        if not force and now - self._last_journal_monotonic < self.journal_interval_seconds:
            return
        if event_type is None:
            event_type = (
                "scheduler_progress"
                if self.producer_kind == "scheduler"
                else "checkpoint_progress"
                if self.producer_kind == "checkpoint"
                else "phase_activity_progress"
            )
        counters = dict(record.get("counters") or {})
        payload: Dict[str, Any] = {
            "phase": self.phase,
            "iteration": self.iteration,
            "replacement_round": self.replacement_round,
            "producer_kind": self.producer_kind,
            "stage": str(record.get("stage") or ""),
            "status": str(record.get("status") or ""),
            "elapsed_seconds": float(record.get("elapsed_seconds") or 0.0),
        }
        payload.update(self.identity)
        for key in ("completed", "total", "unit"):
            if counters.get(key) is not None:
                payload[key] = counters[key]
        details = record.get("details")
        if isinstance(details, Mapping):
            for key in (
                "running",
                "pending",
                "failed",
                "missing",
                "accepted",
                "rejected",
            ):
                value = details.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    payload[key] = value
        if record.get("throughput") is not None:
            payload["throughput"] = record["throughput"]
        try:
            callback(str(event_type), payload)
            self._last_journal_monotonic = now
        except Exception:
            return


def newest_matching_progress(
    records: Iterable[Mapping[str, Any]],
    *,
    campaign_uid: str,
    phase: str,
    iteration: int,
    replacement_round: int,
    daemon_pid: Optional[int] = None,
    daemon_start_id: Optional[str] = None,
    job_ids: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    allowed_jobs = {str(job_id) for job_id in (job_ids or ()) if str(job_id)}
    matches: list[Dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        if (
            record.get("campaign_uid") != str(campaign_uid)
            or record.get("phase") != str(phase)
            or record.get("iteration") != int(iteration)
            or record.get("replacement_round") != int(replacement_round)
        ):
            continue
        if record.get("producer_kind") == "local" and record.get("daemon_pid"):
            if daemon_pid is not None and int(record["daemon_pid"]) != int(daemon_pid):
                continue
        if (
            record.get("producer_kind") == "local"
            and record.get("daemon_start_id")
            and daemon_start_id
            and str(record["daemon_start_id"]) != str(daemon_start_id)
        ):
            continue
        if record.get("producer_kind") == "scheduler" and record.get("job_id"):
            if allowed_jobs and str(record["job_id"]) not in allowed_jobs:
                continue
        matches.append(record)
    if not matches:
        return None
    matches.sort(key=lambda item: str(item.get("updated_at_iso") or ""))
    return matches[-1]


__all__ = [
    "PHASE_PROGRESS_SCHEMA_VERSION",
    "PHASE_PROGRESS_WRITE_INTERVAL_SECONDS",
    "PHASE_PROGRESS_JOURNAL_INTERVAL_SECONDS",
    "SCHEDULER_PROGRESS_JOURNAL_INTERVAL_SECONDS",
    "PhaseProgressReporter",
    "format_progress_stage",
    "newest_matching_progress",
    "progress_path",
    "progress_root",
    "read_phase_progress_records",
    "validate_phase_progress",
]
