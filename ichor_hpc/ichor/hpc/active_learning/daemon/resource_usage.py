"""Advisory scheduler resource-use telemetry for completed attempts."""
from __future__ import annotations

import math
import re
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .state import atomic_write_json
from ..strict_json import strict_json as json
from ..submit import sge
from ..submit.scheduler_backend import get_scheduler_backend


USAGE_SCHEMA_VERSION = 2
USAGE_FILENAME = "resource_usage_records.json"
_MEMORY_RE = re.compile(
    r"^([0-9]+(?:\.[0-9]+)?)([KMGT]?)(?:[cn])?$",
    re.IGNORECASE,
)


def usage_path(campaign_dir: Union[str, Path]) -> Path:
    from .filesystem import operational_path

    return operational_path(campaign_dir, USAGE_FILENAME)


def read_usage_records(campaign_dir: Union[str, Path]) -> Dict[str, Any]:
    """Read the telemetry ledger without hiding corruption as absence."""
    path = usage_path(campaign_dir)
    if path.is_symlink():
        raise ValueError("resource usage records must not be a symlink: " + str(path))
    if not path.exists():
        return {"schema_version": USAGE_SCHEMA_VERSION, "attempts": []}
    if not path.is_file():
        raise ValueError("resource usage records are not a regular file: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("resource usage records are unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise ValueError("resource usage records must contain a JSON object")
    if payload.get("schema_version") != USAGE_SCHEMA_VERSION:
        raise ValueError("resource usage records have an unsupported schema")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or any(
        not isinstance(item, dict) for item in attempts
    ):
        raise ValueError("resource usage records attempts must be a list of objects")
    identities = []
    for item in attempts:
        status = item.get("telemetry_status")
        if status not in {"provisional", "final"}:
            raise ValueError("resource usage attempt has an invalid telemetry status")
        attempt_id = item.get("attempt_id")
        job_id = item.get("job_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("resource usage attempt has no attempt_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("resource usage attempt has no job_id")
        identities.append((attempt_id, job_id))
    if len(identities) != len(set(identities)):
        raise ValueError("resource usage records contain duplicate attempt identities")
    return payload


def _memory_mib(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text or text in {"Unknown", "N/A", "-"}:
        return None
    match = _MEMORY_RE.match(text)
    if not match:
        return None
    amount = float(match.group(1))
    scale = {"": 1.0 / (1024.0 * 1024.0), "K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}[match.group(2).upper()]
    return amount * scale


def parse_usage_rows(stdout: str) -> List[Dict[str, Any]]:
    """Parse logical and physical identities from one ``sacct`` response.

    Production rows contain ``JobID`` followed by ``JobIDRaw``.  The older
    nine-column injected-test seam remains valid and uses its single identity
    for both fields.
    """
    rows: List[Dict[str, Any]] = []
    for raw in str(stdout).splitlines():
        if not raw.strip():
            continue
        parts = raw.rstrip("\n").split("|")
        if len(parts) >= 10:
            job_id, job_id_raw = parts[0], parts[1]
            values = parts[2:10]
        elif len(parts) >= 9:
            job_id = parts[0]
            job_id_raw = job_id
            values = parts[1:9]
        else:
            raise ValueError("sacct usage row has fewer than nine columns")
        state, exit_code, elapsed, alloc_cpus, req_mem, max_rss, max_vm, total_cpu = values
        job_id = str(job_id).strip()
        job_id_raw = str(job_id_raw).strip()
        if not job_id:
            raise ValueError("sacct usage row has an empty logical JobID")
        if not job_id_raw:
            raise ValueError("sacct usage row has an empty JobIDRaw")
        try:
            elapsed_seconds = int(elapsed or 0)
            allocated_cpus = int(alloc_cpus or 0)
        except ValueError as exc:
            raise ValueError("sacct usage row has malformed numeric fields") from exc
        rows.append({
            "job_id": job_id,
            "job_id_raw": job_id_raw,
            "state": state,
            "exit_code": exit_code,
            "elapsed_seconds": elapsed_seconds,
            "allocated_cpus": allocated_cpus,
            "requested_memory": req_mem,
            "max_rss_mib": _memory_mib(max_rss),
            "max_vm_size_mib": _memory_mib(max_vm),
            "total_cpu": total_cpu,
        })
    return rows


def _sge_memory_mib(value: Any, *, bare_unit: str) -> Optional[float]:
    text = str(value or "").strip()
    if not text or text in {"Unknown", "N/A", "-"}:
        return None
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)(?:I?B)?",
        text,
        re.IGNORECASE,
    )
    if match is not None and match.group(2):
        amount = float(match.group(1))
        scale = {
            "K": 1.0 / 1024.0,
            "M": 1.0,
            "G": 1024.0,
            "T": 1024.0 * 1024.0,
        }[match.group(2).upper()]
        return amount * scale
    try:
        amount = float(text)
    except ValueError:
        return None
    if amount < 0 or not math.isfinite(amount):
        return None
    if bare_unit == "kib":
        return amount / 1024.0
    if bare_unit == "bytes":
        return amount / (1024.0 * 1024.0)
    raise ValueError("unsupported bare SGE memory unit")


def parse_sge_usage_records(
    records: Sequence[Mapping[str, Any]],
    *,
    job_id: str,
) -> List[Dict[str, Any]]:
    """Normalise qacct records into the existing telemetry row contract."""
    parent = sge.validate_sge_parent_job_id(job_id)
    rows: List[Dict[str, Any]] = []
    for record in records:
        if str(record.get("jobnumber") or "").strip() != parent:
            continue
        task_text = str(record.get("taskid") or "").strip()
        if task_text in {"", "undefined", "NONE"}:
            logical_job_id = parent
            raw_job_id = parent
        else:
            if not re.fullmatch(r"[1-9][0-9]*", task_text):
                raise ValueError("qacct telemetry contains a malformed task ID")
            native_task_id = int(task_text)
            logical_job_id = parent + "_" + str(native_task_id - 1)
            raw_job_id = parent + "." + str(native_task_id)
        try:
            failed = int(str(record.get("failed") or ""))
            exit_status = int(str(record.get("exit_status") or ""))
            elapsed_seconds = sge.parse_sge_duration_seconds(
                record.get("ru_wallclock") or "0"
            )
            allocated_cpus = int(str(record.get("slots") or "1"))
        except (TypeError, ValueError) as exc:
            raise ValueError("qacct telemetry contains malformed numeric fields") from exc
        if min(failed, exit_status, elapsed_seconds) < 0 or allocated_cpus <= 0:
            raise ValueError("qacct telemetry contains out-of-range numeric fields")
        rows.append(
            {
                "job_id": logical_job_id,
                "job_id_raw": raw_job_id,
                "state": (
                    "COMPLETED"
                    if failed == 0 and exit_status == 0
                    else "FAILED"
                ),
                "exit_code": str(exit_status) + ":0",
                "elapsed_seconds": elapsed_seconds,
                "allocated_cpus": allocated_cpus,
                "requested_memory": "",
                # SGE reports ru_maxrss as KiB when no suffix is present.
                "max_rss_mib": _sge_memory_mib(
                    record.get("ru_maxrss"),
                    bare_unit="kib",
                ),
                # maxvmem is normally suffixed; a bare value is bytes.
                "max_vm_size_mib": _sge_memory_mib(
                    record.get("maxvmem"),
                    bare_unit="bytes",
                ),
                "total_cpu": str(record.get("cpu") or ""),
            }
        )
    return rows


def _quantile(values: Sequence[float], fraction: float) -> Optional[float]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    index = (len(finite) - 1) * float(fraction)
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return finite[lo]
    return finite[lo] * (hi - index) + finite[hi] * (index - lo)


def summarise_usage(
    *,
    attempt_id: str,
    submission_identity: str,
    phase_name: str,
    iteration: int,
    job_id: str,
    rows: Sequence[Mapping[str, Any]],
    expected_tasks: Optional[int] = None,
    query_command: Optional[Sequence[str]] = None,
    query_stdout: str = "",
    collection_sequence: int = 1,
) -> Dict[str, Any]:
    rss = [float(row["max_rss_mib"]) for row in rows if row.get("max_rss_mib") is not None]
    vm = [float(row["max_vm_size_mib"]) for row in rows if row.get("max_vm_size_mib") is not None]
    elapsed = [float(row.get("elapsed_seconds", 0)) for row in rows]
    p95_rss = _quantile(rss, 0.95)
    p95_elapsed = _quantile(elapsed, 0.95)
    failures = [row for row in rows if not str(row.get("state", "")).startswith("COMPLETED")]
    outliers = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            float(row.get("max_rss_mib") or -1.0),
            float(row.get("elapsed_seconds") or 0.0),
        ),
        reverse=True,
    )[:10]
    missing_rows = (
        None
        if expected_tasks is None
        else max(0, int(expected_tasks) - len(rows))
    )
    complete_metrics = bool(rows) and all(
        row.get("max_rss_mib") is not None
        and int(row.get("elapsed_seconds", -1)) >= 0
        for row in rows
    )
    final = (missing_rows in {None, 0}) and complete_metrics
    return {
        "attempt_id": str(attempt_id),
        "submission_identity": str(submission_identity),
        "phase": str(phase_name),
        "iteration": int(iteration),
        "job_id": str(job_id),
        "collected_at_iso": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "n_expected_tasks": (
            None if expected_tasks is None else int(expected_tasks)
        ),
        "n_missing_task_rows": missing_rows,
        "telemetry_status": "final" if final else "provisional",
        "collection_complete": bool(final),
        "collection_sequence": int(collection_sequence),
        "query_command": list(query_command or []),
        "query_stdout_sha256": hashlib.sha256(
            str(query_stdout).encode("utf-8")
        ).hexdigest(),
        "n_failures": len(failures),
        "max_rss_mib": max(rss) if rss else None,
        "p50_rss_mib": _quantile(rss, 0.50),
        "p95_rss_mib": p95_rss,
        "max_vm_size_mib": max(vm) if vm else None,
        "p50_vm_size_mib": _quantile(vm, 0.50),
        "p95_vm_size_mib": _quantile(vm, 0.95),
        "max_elapsed_seconds": max(elapsed) if elapsed else None,
        "p50_elapsed_seconds": _quantile(elapsed, 0.50),
        "p95_elapsed_seconds": p95_elapsed,
        "recommended_memory_mib": None if p95_rss is None else p95_rss * 1.25,
        "recommended_walltime_seconds": None if p95_elapsed is None else p95_elapsed * 1.5,
        "max_allocated_cpus": max(
            (int(row.get("allocated_cpus", 0)) for row in rows),
            default=0,
        ),
        "requested_memory_values": sorted(
            {str(row.get("requested_memory")) for row in rows if row.get("requested_memory")}
        ),
        "state_counts": {
            state: sum(1 for row in rows if str(row.get("state") or "") == state)
            for state in sorted({str(row.get("state") or "") for row in rows})
        },
        "failure_sample": [dict(row) for row in failures[:10]],
        "outlier_sample": outliers,
    }


def _scientific_task_rows(
    rows: Sequence[Mapping[str, Any]],
    job_id: str,
) -> List[Dict[str, Any]]:
    """Return one accounting row per task, enriched with its Slurm steps.

    Slurm normally reports allocation state and elapsed time on the task row,
    but reports ``MaxRSS`` and ``MaxVMSize`` on the corresponding ``.batch``
    step.  Treating step rows as independent tasks inflates task counts, while
    dropping them loses the memory measurement.  Merge their maxima into the
    owning task instead.
    """
    def logical_id(row: Mapping[str, Any]) -> str:
        return str(row.get("job_id") or row.get("job_id_raw") or "")

    direct = [
        dict(row)
        for row in rows
        if "." not in logical_id(row)
    ]
    array_rows = [
        row
        for row in direct
        if logical_id(row).startswith(str(job_id) + "_")
    ]
    task_rows = array_rows or [
        row
        for row in direct
        if logical_id(row) == str(job_id)
    ]
    if not task_rows:
        return []

    enriched: List[Dict[str, Any]] = []
    for task in task_rows:
        task_id = logical_id(task)
        steps = [
            row
            for row in rows
            if logical_id(row).startswith(task_id + ".")
            and not logical_id(row).endswith(".extern")
        ]
        merged = dict(task)
        for field in ("max_rss_mib", "max_vm_size_mib"):
            values = [
                float(row[field])
                for row in [task, *steps]
                if row.get(field) is not None
                and math.isfinite(float(row[field]))
            ]
            merged[field] = max(values) if values else None
        if not str(merged.get("total_cpu") or "").strip():
            for step in steps:
                value = str(step.get("total_cpu") or "").strip()
                if value:
                    merged["total_cpu"] = value
                    break
        enriched.append(merged)
    return enriched


def append_usage_summary(
    campaign_dir: Union[str, Path],
    summary: Mapping[str, Any],
    *,
    history_limit: int,
) -> Dict[str, Any]:
    path = usage_path(campaign_dir)
    if int(history_limit) <= 0:
        raise ValueError("resource usage history limit must be > 0")
    payload = read_usage_records(campaign_dir)
    attempts = list(payload["attempts"])
    identity = (str(summary.get("attempt_id")), str(summary.get("job_id")))
    existing_index = next(
        (
            index
            for index, item in enumerate(attempts)
            if (str(item.get("attempt_id")), str(item.get("job_id"))) == identity
        ),
        None,
    )
    if existing_index is None:
        attempts.append(dict(summary))
    else:
        existing = attempts[existing_index]
        if str(existing.get("telemetry_status")) == "final":
            if dict(existing) != dict(summary):
                raise ValueError("final resource usage telemetry is immutable")
        else:
            old_sequence = int(existing.get("collection_sequence", 0))
            new_sequence = int(summary.get("collection_sequence", 0))
            old_rows = int(existing.get("n_rows", 0))
            new_rows = int(summary.get("n_rows", 0))
            if new_sequence <= old_sequence or new_rows < old_rows:
                raise ValueError(
                    "provisional telemetry replacement is not monotonic"
                )
            attempts[existing_index] = dict(summary)
    payload["attempts"] = attempts[-int(history_limit):]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return payload


def collect_usage(
    campaign_dir: Union[str, Path],
    *,
    intent: Mapping[str, Any],
    history_limit: int,
    runner: Any = subprocess.run,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    job_id = str(intent.get("job_id") or "")
    if not job_id:
        raise ValueError("submission intent has no JobID for telemetry")
    path = usage_path(campaign_dir)
    existing = read_usage_records(campaign_dir)
    previous = next(
        (
            dict(item)
            for item in existing["attempts"]
            if str(item.get("attempt_id")) == str(intent.get("attempt_id"))
            and str(item.get("job_id")) == job_id
        ),
        None,
    )
    if previous is not None and str(previous.get("telemetry_status")) == "final":
        return previous
    scheduler_kind = str(
        intent.get("scheduler_identity_kind") or "slurm"
    ).strip().lower()
    backend = get_scheduler_backend(scheduler_kind)
    evidence = backend.collect_usage_evidence(
        job_id,
        runner=runner,
        timeout_seconds=int(timeout_seconds),
    )
    query_command = list(evidence.command)
    if scheduler_kind == "slurm":
        query_stdout = evidence.stdout
        parsed_rows = parse_usage_rows(query_stdout)
        if not parsed_rows:
            raise RuntimeError("sacct telemetry returned no rows")
        rows = _scientific_task_rows(parsed_rows, job_id)
        if not rows:
            raise RuntimeError("sacct telemetry returned no scientific task rows")
    elif scheduler_kind == "sge":
        records = list(evidence.records or ())
        query_stdout = json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
        )
        rows = parse_sge_usage_records(records, job_id=job_id)
        if not rows:
            raise RuntimeError("qacct telemetry returned no scientific task rows")
    else:
        raise AssertionError("validated scheduler backend was not handled")
    expected_raw = intent.get("expected_tasks")
    expected_tasks = None
    if expected_raw is not None:
        if isinstance(expected_raw, bool) or not isinstance(expected_raw, int):
            raise ValueError("submission intent expected_tasks is malformed")
        expected_tasks = expected_raw
        if expected_tasks <= 0:
            raise ValueError("submission intent expected_tasks must be > 0")
    submission_kind = str(intent.get("submission_kind") or "")
    if submission_kind == "array" and expected_tasks is not None:
        expected_ids = {job_id + "_" + str(index) for index in range(expected_tasks)}
        observed_ids = {
            str(row.get("job_id") or row.get("job_id_raw") or "")
            for row in rows
        }
        unexpected = sorted(observed_ids - expected_ids)
        if unexpected:
            raise ValueError(
                "scheduler telemetry returned unexpected array task IDs: "
                + repr(unexpected[:10])
            )
    summary = summarise_usage(
        attempt_id=str(intent.get("attempt_id") or ""),
        submission_identity=str(intent.get("submission_identity") or ""),
        phase_name=str(intent.get("phase") or ""),
        iteration=int(intent.get("iteration", 0)),
        job_id=job_id,
        rows=rows,
        expected_tasks=expected_tasks,
        query_command=query_command,
        query_stdout=query_stdout,
        collection_sequence=(
            1 if previous is None else int(previous.get("collection_sequence", 0)) + 1
        ),
    )
    append_usage_summary(campaign_dir, summary, history_limit=int(history_limit))
    return summary
