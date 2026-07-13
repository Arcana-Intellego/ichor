"""Advisory Slurm resource-use telemetry for completed attempts."""
from __future__ import annotations

import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .state import atomic_write_json


USAGE_SCHEMA_VERSION = 1
USAGE_FILENAME = "resource_usage_records.json"
_MEMORY_RE = re.compile(
    r"^([0-9]+(?:\.[0-9]+)?)([KMGT]?)(?:[cn])?$",
    re.IGNORECASE,
)


def usage_path(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / USAGE_FILENAME


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
    rows: List[Dict[str, Any]] = []
    for raw in str(stdout).splitlines():
        if not raw.strip():
            continue
        parts = raw.rstrip("\n").split("|")
        if len(parts) < 9:
            raise ValueError("sacct usage row has fewer than nine columns")
        job_id, state, exit_code, elapsed, alloc_cpus, req_mem, max_rss, max_vm, total_cpu = parts[:9]
        try:
            elapsed_seconds = int(elapsed or 0)
            allocated_cpus = int(alloc_cpus or 0)
        except ValueError as exc:
            raise ValueError("sacct usage row has malformed numeric fields") from exc
        rows.append({
            "job_id_raw": job_id,
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
        "n_missing_task_rows": (
            None
            if expected_tasks is None
            else max(0, int(expected_tasks) - len(rows))
        ),
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
    direct = [
        dict(row)
        for row in rows
        if "." not in str(row.get("job_id_raw") or "")
    ]
    array_rows = [
        row
        for row in direct
        if str(row.get("job_id_raw") or "").startswith(str(job_id) + "_")
    ]
    task_rows = array_rows or [
        row
        for row in direct
        if str(row.get("job_id_raw") or "") == str(job_id)
    ]
    if not task_rows:
        return []

    enriched: List[Dict[str, Any]] = []
    for task in task_rows:
        task_id = str(task.get("job_id_raw") or "")
        steps = [
            row
            for row in rows
            if str(row.get("job_id_raw") or "").startswith(task_id + ".")
            and not str(row.get("job_id_raw") or "").endswith(".extern")
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
    import json

    path = usage_path(campaign_dir)
    if int(history_limit) <= 0:
        raise ValueError("resource usage history limit must be > 0")
    if path.is_symlink():
        raise ValueError("resource usage records must not be a symlink: " + str(path))
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("resource usage records are unreadable: " + str(path)) from exc
    else:
        payload = {"schema_version": USAGE_SCHEMA_VERSION, "attempts": []}
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != USAGE_SCHEMA_VERSION:
        raise ValueError("resource usage records have an unsupported schema")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("resource usage records attempts must be a list")
    identity = (str(summary.get("attempt_id")), str(summary.get("job_id")))
    if not any((str(item.get("attempt_id")), str(item.get("job_id"))) == identity for item in attempts if isinstance(item, dict)):
        attempts.append(dict(summary))
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
) -> Dict[str, Any]:
    job_id = str(intent.get("job_id") or "")
    if not job_id:
        raise ValueError("submission intent has no JobID for telemetry")
    path = usage_path(campaign_dir)
    if path.is_symlink():
        raise ValueError("resource usage records must not be a symlink: " + str(path))
    if path.is_file() and not path.is_symlink():
        import json

        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict) and isinstance(existing.get("attempts"), list):
            for item in existing["attempts"]:
                if (
                    isinstance(item, dict)
                    and str(item.get("attempt_id")) == str(intent.get("attempt_id"))
                    and str(item.get("job_id")) == job_id
                ):
                    return dict(item)
    completed = runner(
        [
            "sacct", "-n", "-P", "-j", job_id,
            "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,MaxVMSize,TotalCPU",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if int(getattr(completed, "returncode", 1)) != 0:
        raise RuntimeError("sacct telemetry failed: " + str(getattr(completed, "stderr", "")))
    parsed_rows = parse_usage_rows(getattr(completed, "stdout", "") or "")
    if not parsed_rows:
        raise RuntimeError("sacct telemetry returned no rows")
    rows = _scientific_task_rows(parsed_rows, job_id)
    if not rows:
        raise RuntimeError("sacct telemetry returned no scientific task rows")
    expected_raw = intent.get("expected_tasks")
    expected_tasks = None
    if expected_raw is not None:
        try:
            expected_tasks = int(expected_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("submission intent expected_tasks is malformed") from exc
        if expected_tasks <= 0:
            raise ValueError("submission intent expected_tasks must be > 0")
    summary = summarise_usage(
        attempt_id=str(intent.get("attempt_id") or ""),
        submission_identity=str(intent.get("submission_identity") or ""),
        phase_name=str(intent.get("phase") or ""),
        iteration=int(intent.get("iteration", 0)),
        job_id=job_id,
        rows=rows,
        expected_tasks=expected_tasks,
    )
    append_usage_summary(campaign_dir, summary, history_limit=int(history_limit))
    return summary
