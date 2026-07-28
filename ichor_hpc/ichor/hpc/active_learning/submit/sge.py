"""Sun Grid Engine contracts for the active-learning daemon.

The older :mod:`ichor.hpc.batch_system.sge` integration is a generic script
helper.  The daemon needs stricter evidence: XML queue observations, delayed
accounting, exact array identities and fail-closed name adoption.
"""
from __future__ import annotations

import getpass
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .sacct_poll import (
    ArrayJobSummary,
    JobNameAccountingLookup,
    JobNameLookup,
    JobObservation,
    JobQueueLookup,
    JobStatus,
    aggregate_states,
)
from .slurm_contracts import run_scheduler_command


_PARENT_JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")
_TERSE_ARRAY_RE = re.compile(
    r"^([1-9][0-9]*)\.([1-9][0-9]*)-([1-9][0-9]*):([1-9][0-9]*)$"
)
_TASK_TOKEN_RE = re.compile(
    r"^[1-9][0-9]*(?:-[1-9][0-9]*(?::[1-9][0-9]*)?)?$"
)
_SGE_SAFE_JOB_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_DURATION_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([smhd]?)$", re.IGNORECASE)


def validate_sge_parent_job_id(value: Any) -> str:
    job_id = str(value).strip()
    if not _PARENT_JOB_ID_RE.fullmatch(job_id):
        raise ValueError("SGE parent JobID must be a positive decimal integer")
    return job_id


def parse_qsub_terse_output(stdout: Any) -> str:
    """Return the parent JobID from one exact ``qsub -terse`` result."""
    lines = [line.strip() for line in str(stdout or "").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(
            "qsub -terse must emit exactly one non-empty line; got "
            + str(len(lines))
        )
    value = lines[0]
    if _PARENT_JOB_ID_RE.fullmatch(value):
        return value
    array = _TERSE_ARRAY_RE.fullmatch(value)
    if array is None:
        raise ValueError("qsub -terse returned an invalid job identity: " + repr(value))
    first, last, step = (int(array.group(index)) for index in (2, 3, 4))
    if last < first or step <= 0:
        raise ValueError("qsub -terse returned an invalid array range")
    return array.group(1)


def sge_safe_job_name(value: Any) -> str:
    """Return a deterministic SGE-safe job name without losing its digest tail."""
    raw = str(value).strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    candidate = cleaned if cleaned.startswith("ichor-") else "ichor-" + cleaned
    if len(candidate) > 128:
        suffix = candidate[-24:]
        candidate = candidate[: 128 - len(suffix) - 1].rstrip(".-") + "-" + suffix
    if not _SGE_SAFE_JOB_NAME_RE.fullmatch(candidate):
        raise ValueError("could not construct an SGE-safe job name")
    return candidate


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str:
    for child in list(element):
        if _local_name(child.tag) == name:
            return str(child.text or "").strip()
    return ""


def expand_sge_tasks(value: Any, *, maximum_tasks: int = 75000) -> List[int]:
    """Expand an SGE task expression into native one-based task IDs."""
    text = str(value or "").strip()
    if not text:
        return []
    result: List[int] = []
    seen = set()
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not _TASK_TOKEN_RE.fullmatch(token):
            raise ValueError("malformed SGE task expression: " + repr(text))
        if "-" not in token:
            values: Iterable[int] = (int(token),)
        else:
            bounds, *raw_step = token.split(":")
            first_text, last_text = bounds.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            step = int(raw_step[0]) if raw_step else 1
            if last < first:
                raise ValueError("descending SGE task range is invalid")
            values = range(first, last + 1, step)
        for native_task_id in values:
            if native_task_id in seen:
                raise ValueError("SGE task expression contains duplicate task IDs")
            seen.add(native_task_id)
            result.append(native_task_id)
            if len(result) > int(maximum_tasks):
                raise ValueError("SGE task expression exceeds the configured task limit")
    return sorted(result)


@dataclass(frozen=True)
class SgeQueueRow:
    parent_job_id: str
    task_index: Optional[int]
    native_task_id: Optional[int]
    state: str
    job_name: str
    owner: str
    slots: Optional[int] = None
    queue: str = ""

    @property
    def logical_job_id(self) -> str:
        if self.task_index is None:
            return self.parent_job_id
        return self.parent_job_id + "_" + str(int(self.task_index))


def parse_qstat_xml(stdout: str, *, maximum_tasks: int = 75000) -> List[SgeQueueRow]:
    """Parse ``qstat -xml`` without depending on fixed-width text output."""
    text = str(stdout or "").strip()
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError("qstat XML is malformed: " + str(exc)) from exc
    rows: List[SgeQueueRow] = []
    for element in root.iter():
        if _local_name(element.tag) != "job_list":
            continue
        parent = validate_sge_parent_job_id(_child_text(element, "JB_job_number"))
        name = _child_text(element, "JB_name")
        owner = _child_text(element, "JB_owner")
        state = _child_text(element, "state") or str(element.attrib.get("state") or "")
        tasks = expand_sge_tasks(
            _child_text(element, "tasks"),
            maximum_tasks=int(maximum_tasks),
        )
        slots_text = _child_text(element, "slots")
        try:
            slots = int(slots_text) if slots_text else None
        except ValueError as exc:
            raise ValueError("qstat XML contains a malformed slot count") from exc
        queue = _child_text(element, "queue_name")
        if not queue:
            parent_element = element
            # Queue-List/name is not a child of job_list. ElementTree has no
            # parent pointer, so leave this optional field empty.
            del parent_element
        if tasks:
            for native_task_id in tasks:
                rows.append(
                    SgeQueueRow(
                        parent_job_id=parent,
                        task_index=int(native_task_id) - 1,
                        native_task_id=int(native_task_id),
                        state=state,
                        job_name=name,
                        owner=owner,
                        slots=slots,
                        queue=queue,
                    )
                )
        else:
            rows.append(
                SgeQueueRow(
                    parent_job_id=parent,
                    task_index=None,
                    native_task_id=None,
                    state=state,
                    job_name=name,
                    owner=owner,
                    slots=slots,
                    queue=queue,
                )
            )
    return rows


def _qstat_status(raw: str) -> JobStatus:
    state = str(raw or "").strip()
    if state in {"r", "t", "Rr", "Rt"}:
        return JobStatus.RUNNING
    if state in {"qw", "hqw", "hRqw", "s", "S", "T", "ts", "tsS", "tT"}:
        return JobStatus.PENDING
    if state.startswith("d"):
        return JobStatus.COMPLETING
    return JobStatus.UNKNOWN


def qstat_observations(rows: Sequence[SgeQueueRow]) -> List[JobObservation]:
    return [
        JobObservation(
            job_id=row.logical_job_id,
            status=_qstat_status(row.state),
            exit_code=None,
            elapsed_seconds=None,
            raw_status=row.state,
            parse_error=(
                None
                if _qstat_status(row.state) is not JobStatus.UNKNOWN
                else "unrecognised SGE queue state " + repr(row.state)
            ),
            job_id_raw=(
                row.parent_job_id
                if row.native_task_id is None
                else row.parent_job_id + "." + str(row.native_task_id)
            ),
        )
        for row in rows
    ]


def parse_qacct_output(stdout: str) -> List[Dict[str, str]]:
    """Parse one or more key/value records emitted by ``qacct``."""
    records: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for raw_line in str(stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if set(line) == {"="}:
            if current:
                records.append(current)
                current = {}
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts
        if key in current:
            raise ValueError("qacct record contains duplicate field " + repr(key))
        current[key] = value.strip()
    if current:
        records.append(current)
    return records


def _exact_int(record: Dict[str, str], key: str, *, minimum: int = 0) -> int:
    value = str(record.get(key, "")).strip()
    if not re.fullmatch(r"-?[0-9]+", value):
        raise ValueError("qacct field " + key + " is not an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError("qacct field " + key + " is below its minimum")
    return parsed


def parse_sge_duration_seconds(value: Any) -> int:
    """Parse the duration forms emitted by ffluxlab's Grid Engine."""
    text = str(value or "").strip()
    match = _DURATION_RE.fullmatch(text)
    if match is not None:
        amount = float(match.group(1))
        scale = {
            "": 1.0,
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
            "d": 86400.0,
        }[match.group(2).lower()]
        seconds = amount * scale
    elif ":" in text:
        parts = text.split(":")
        if len(parts) not in {3, 4} or any(
            not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", part)
            for part in parts
        ):
            raise ValueError("SGE duration is malformed: " + repr(text))
        values = [float(part) for part in parts]
        if len(values) == 3:
            hours, minutes, seconds_part = values
            days = 0.0
        else:
            days, hours, minutes, seconds_part = values
        if minutes >= 60.0 or seconds_part >= 60.0 or (
            len(values) == 4 and hours >= 24.0
        ):
            raise ValueError("SGE duration has an out-of-range component")
        seconds = (
            days * 86400.0
            + hours * 3600.0
            + minutes * 60.0
            + seconds_part
        )
    else:
        raise ValueError("SGE duration is malformed: " + repr(text))
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("SGE duration must be finite and non-negative")
    return int(round(seconds))


def _optional_duration_seconds(record: Dict[str, str], key: str) -> Optional[int]:
    value = str(record.get(key, "")).strip()
    if not value:
        return None
    try:
        return parse_sge_duration_seconds(value)
    except ValueError as exc:
        raise ValueError("qacct field " + key + " has an invalid duration") from exc


def qacct_observations(
    records: Sequence[Dict[str, str]],
    *,
    cancellation_requested: bool = False,
) -> List[JobObservation]:
    observations: List[JobObservation] = []
    for record in records:
        parent = validate_sge_parent_job_id(record.get("jobnumber", ""))
        task_text = str(record.get("taskid", "")).strip()
        if task_text in {"", "undefined", "NONE"}:
            job_id = parent
            raw_id = parent
        else:
            native_task_id = _exact_int(record, "taskid", minimum=1)
            job_id = parent + "_" + str(native_task_id - 1)
            raw_id = parent + "." + str(native_task_id)
        failed = _exact_int(record, "failed")
        exit_status = _exact_int(record, "exit_status")
        if failed == 0 and exit_status == 0:
            status = JobStatus.COMPLETED
        elif cancellation_requested and failed == 100 and exit_status == 137:
            status = JobStatus.CANCELLED
        else:
            status = JobStatus.FAILED
        observations.append(
            JobObservation(
                job_id=job_id,
                status=status,
                exit_code=(exit_status, 0),
                elapsed_seconds=_optional_duration_seconds(record, "ru_wallclock"),
                raw_status="failed=" + str(failed) + " exit_status=" + str(exit_status),
                job_id_raw=raw_id,
            )
        )
    return observations


def _run(
    runner: Callable[..., Any],
    command: Sequence[str],
    *,
    timeout_seconds: int,
) -> Any:
    return run_scheduler_command(
        runner,
        list(command),
        timeout_seconds=int(timeout_seconds),
        check=False,
        capture_output=True,
        text=True,
    )


def _qacct_missing(stderr: str, stdout: str = "") -> bool:
    text = (str(stderr or "") + "\n" + str(stdout or "")).lower()
    return (
        "not found" in text
        or "error: job id" in text
        or "no jobs" in text
        or "not in accounting" in text
    )


def query_qstat(
    *,
    runner: Callable[..., Any] = subprocess.run,
    user: Optional[str] = None,
    timeout_seconds: int = 60,
) -> List[SgeQueueRow]:
    completed = _run(
        runner,
        ["qstat", "-xml", "-u", str(user or getpass.getuser())],
        timeout_seconds=int(timeout_seconds),
    )
    if int(getattr(completed, "returncode", 1)) != 0:
        raise RuntimeError(
            "qstat exited with code "
            + str(int(getattr(completed, "returncode", 1)))
            + ": "
            + repr(getattr(completed, "stderr", "") or "")
        )
    return parse_qstat_xml(getattr(completed, "stdout", "") or "")


def query_qacct(
    selector: str,
    *,
    runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: int = 60,
) -> List[Dict[str, str]]:
    completed = _run(
        runner,
        ["qacct", "-j", str(selector)],
        timeout_seconds=int(timeout_seconds),
    )
    stdout = getattr(completed, "stdout", "") or ""
    stderr = getattr(completed, "stderr", "") or ""
    if int(getattr(completed, "returncode", 1)) != 0:
        if _qacct_missing(stderr, stdout):
            return []
        raise RuntimeError(
            "qacct exited with code "
            + str(int(getattr(completed, "returncode", 1)))
            + ": "
            + repr(stderr)
        )
    return parse_qacct_output(stdout)


def poll_job(
    job_id: str,
    *,
    qstat_runner: Callable[..., Any] = subprocess.run,
    qacct_runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: int = 60,
    cancellation_requested: bool = False,
) -> List[JobObservation]:
    parent = validate_sge_parent_job_id(job_id)
    queue_rows = [
        row
        for row in query_qstat(
            runner=qstat_runner,
            timeout_seconds=int(timeout_seconds),
        )
        if row.parent_job_id == parent
    ]
    accounting = [
        record
        for record in query_qacct(
            parent,
            runner=qacct_runner,
            timeout_seconds=int(timeout_seconds),
        )
        if str(record.get("jobnumber") or "") == parent
    ]
    merged: Dict[str, JobObservation] = {}
    for observation in qacct_observations(
        accounting,
        cancellation_requested=bool(cancellation_requested),
    ):
        if observation.job_id in merged:
            raise ValueError(
                "qacct returned duplicate task records for "
                + str(observation.job_id)
            )
        merged[observation.job_id] = observation
    # Live ownership wins over accounting during the brief SGE reporting race.
    for observation in qstat_observations(queue_rows):
        merged[observation.job_id] = observation
    return [merged[key] for key in sorted(merged)]


def find_active_job_by_id_detailed(
    job_id: str,
    *,
    qstat_runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: int = 60,
) -> JobQueueLookup:
    try:
        parent = validate_sge_parent_job_id(job_id)
        rows = [
            row
            for row in query_qstat(
                runner=qstat_runner,
                timeout_seconds=int(timeout_seconds),
            )
            if row.parent_job_id == parent
        ]
    except Exception as exc:
        return JobQueueLookup(
            active=False,
            inconclusive=True,
            error=type(exc).__name__ + ": " + str(exc),
        )
    return JobQueueLookup(
        active=bool(rows),
        rows=[(row.logical_job_id, row.state) for row in rows],
    )


def find_active_job_by_name_detailed(
    name: str,
    *,
    qstat_runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: int = 60,
) -> JobNameLookup:
    owner = str(getpass.getuser())
    try:
        rows = [
            row
            for row in query_qstat(
                runner=qstat_runner,
                timeout_seconds=int(timeout_seconds),
            )
            if row.job_name == str(name) and row.owner == owner
        ]
    except Exception as exc:
        return JobNameLookup(
            None,
            inconclusive=True,
            error=type(exc).__name__ + ": " + str(exc),
        )
    parent_ids = sorted({row.parent_job_id for row in rows})
    compact = [(row.logical_job_id, row.state) for row in rows]
    if not parent_ids:
        return JobNameLookup(None, rows=compact)
    if len(parent_ids) != 1:
        return JobNameLookup(
            None,
            inconclusive=True,
            rows=compact,
            error="multiple active SGE jobs share expected name: " + repr(parent_ids),
        )
    return JobNameLookup(parent_ids[0], rows=compact)


def find_accounted_job_by_name_detailed(
    name: str,
    *,
    expected_task_count: Optional[int] = None,
    qacct_runner: Callable[..., Any] = subprocess.run,
    qstat_runner: Callable[..., Any] = subprocess.run,
    use_qstat_fallback: bool = True,
    submission_kind: Optional[str] = None,
    timeout_seconds: int = 60,
) -> JobNameAccountingLookup:
    owner = str(getpass.getuser())
    try:
        queried_records = query_qacct(
            str(name),
            runner=qacct_runner,
            timeout_seconds=int(timeout_seconds),
        )
    except Exception as exc:
        return JobNameAccountingLookup(
            None,
            inconclusive=True,
            error=type(exc).__name__ + ": " + str(exc),
        )
    foreign = [
        record
        for record in queried_records
        if str(record.get("jobname") or "") == str(name)
        and str(record.get("owner") or "") != owner
    ]
    if foreign:
        return JobNameAccountingLookup(
            None,
            inconclusive=True,
            error="qacct returned a matching job name owned by another user",
        )
    records = [
        record
        for record in queried_records
        if str(record.get("jobname") or "") == str(name)
        and str(record.get("owner") or "") == owner
    ]
    parent_ids = sorted(
        {
            validate_sge_parent_job_id(record.get("jobnumber", ""))
            for record in records
        }
    )
    if not parent_ids:
        if use_qstat_fallback:
            active = find_active_job_by_name_detailed(
                name,
                qstat_runner=qstat_runner,
                timeout_seconds=int(timeout_seconds),
            )
            return JobNameAccountingLookup(
                active.job_id,
                inconclusive=active.inconclusive,
                rows=list(active.rows),
                error=active.error,
            )
        return JobNameAccountingLookup(None)
    if len(parent_ids) != 1:
        return JobNameAccountingLookup(
            None,
            inconclusive=True,
            error="multiple SGE jobs share expected name: " + repr(parent_ids),
        )
    parent = parent_ids[0]
    observations = qacct_observations(records)
    summary = aggregate_states(
        parent,
        observations,
        expected_task_count=expected_task_count,
        submission_kind=submission_kind,
        strict_parent_job_id=False,
    )
    if use_qstat_fallback and int(summary.n_missing) > 0:
        active = find_active_job_by_name_detailed(
            name,
            qstat_runner=qstat_runner,
            timeout_seconds=int(timeout_seconds),
        )
        if active.job_id or active.inconclusive:
            return JobNameAccountingLookup(
                active.job_id,
                inconclusive=active.inconclusive,
                rows=list(active.rows),
                error=active.error,
            )
    rows = [(item.job_id, item.status.value) for item in observations]
    if summary.n_unknown:
        return JobNameAccountingLookup(
            None,
            inconclusive=True,
            rows=rows,
            error="qacct returned inconclusive task evidence",
        )
    if summary.is_fully_successful:
        return JobNameAccountingLookup(
            parent,
            terminal=True,
            successful=True,
            rows=rows,
        )
    if summary.is_terminal:
        return JobNameAccountingLookup(
            parent,
            terminal=True,
            failed=True,
            rows=rows,
        )
    return JobNameAccountingLookup(parent, rows=rows)


def find_running_job_by_name_detailed(
    name: str,
    *,
    qstat_runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: int = 60,
    **_: Any,
) -> JobNameLookup:
    return find_active_job_by_name_detailed(
        name,
        qstat_runner=qstat_runner,
        timeout_seconds=int(timeout_seconds),
    )


__all__ = [
    "SgeQueueRow",
    "expand_sge_tasks",
    "find_accounted_job_by_name_detailed",
    "find_active_job_by_id_detailed",
    "find_active_job_by_name_detailed",
    "find_running_job_by_name_detailed",
    "parse_qacct_output",
    "parse_qstat_xml",
    "parse_qsub_terse_output",
    "parse_sge_duration_seconds",
    "poll_job",
    "qacct_observations",
    "qstat_observations",
    "query_qacct",
    "query_qstat",
    "sge_safe_job_name",
    "validate_sge_parent_job_id",
]
