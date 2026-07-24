"""Read-only resource planning for current, submitted, and future phases."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .phase_executor import INLINE_PHASES, SBATCH_PHASES
from .resource_records import (
    read_resolution,
    resolution_path as canonical_resolution_path,
    verify_resolution,
)
from .resource_solver import (
    ResourceEvidenceUnavailable,
    ResourceEvidenceInvalid,
    collect_resource_evidence,
    resolve_phase_resources,
)
from .submission_intent import load_intent
from .resource_usage import read_usage_records, usage_path


RESOURCE_PLAN_SCHEMA_VERSION = 2


BOOTSTRAP_PHASES = (
    "INIT",
    "PHASE_A_DIVERSITY",
    "INITIAL_GAUSSIAN",
    "INITIAL_AIMALL",
    "INITIAL_ALLOCATION_CHECK",
    "INITIAL_REPLACEMENT_GAUSSIAN",
    "INITIAL_REPLACEMENT_AIMALL",
    "REFERENCE_COMMIT",
    "INITIAL_FEREBUS",
)
ACTIVE_PHASES = (
    "SEED_SELECT",
    "ARIADNE_ARRAY",
    "PHASE_B_DIVERSITY",
    "SPLIT",
    "GAUSSIAN",
    "AIMALL",
    "ALLOCATION_CHECK",
    "REPLACEMENT_GAUSSIAN",
    "REPLACEMENT_AIMALL",
    "REFERENCE_COMMIT",
    "FEREBUS",
    "STOP_CHECK",
)


def _orphaned_resolutions(
    campaign_dir: Path,
    phase_name: str,
    iteration: int,
) -> List[str]:
    root = (
        campaign_dir
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "resource_resolutions"
        / str(phase_name)
        / ("iteration-" + str(int(iteration)).zfill(6))
    )
    if not root.is_dir() or root.is_symlink():
        return []
    files = [path for path in root.glob("*.json") if path.is_file() and not path.is_symlink()]
    if not files:
        return []
    return [str(path.resolve()) for path in sorted(files, key=lambda path: path.name)]


def _usage_for_identity(campaign_dir: Path, identity: str) -> Dict[str, Any]:
    path = usage_path(campaign_dir)
    try:
        payload = read_usage_records(campaign_dir)
    except ValueError as exc:
        return {
            "status": "invalid",
            "path": str(path),
            "error": type(exc).__name__ + ": " + str(exc),
            "summary": None,
        }
    attempts = payload["attempts"]
    matches = [
        dict(item)
        for item in attempts
        if isinstance(item, dict)
        and str(item.get("submission_identity") or "") == str(identity)
    ]
    if not matches:
        return {
            "status": "absent",
            "path": str(path),
            "error": None,
            "summary": None,
        }
    summary = matches[-1]
    return {
        "status": str(summary.get("telemetry_status") or "invalid"),
        "path": str(path),
        "error": None,
        "summary": summary,
    }


def _absent_usage(campaign_dir: Path) -> Dict[str, Any]:
    return {
        "status": "absent",
        "path": str(usage_path(campaign_dir)),
        "error": None,
        "summary": None,
    }


def _submitted_plan(
    campaign_dir: Path,
    phase_name: str,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    intent = load_intent(campaign_dir, phase_name, int(iteration))
    if not isinstance(intent, dict):
        return None
    path = None
    if intent.get("resource_resolution_path"):
        path = Path(str(intent["resource_resolution_path"]))
    elif str(intent.get("status") or "") == "PRE_SUBMIT":
        # A new attempt has an identity but has not yet snapshotted resources.
        # Do not mislabel an older retry's resolution as the current attempt.
        return None
    if path is None:
        return None
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            "submission intent resource resolution is not a regular file: "
            + str(path)
        )
    digest = intent.get("resource_resolution_sha256")
    payload = (
        verify_resolution(
            path,
            str(digest),
            campaign_dir=campaign_dir,
            verify_bound_inputs=False,
        )
        if isinstance(digest, str) and digest
        else read_resolution(path)
    )
    expected_path = canonical_resolution_path(
        campaign_dir,
        str(payload.get("phase") or ""),
        int(payload.get("iteration", -1)),
        str(payload.get("submission_identity") or ""),
    )
    if path.resolve() != expected_path.resolve():
        raise ValueError(
            "submitted resource resolution is outside its canonical store"
        )
    if str(payload.get("phase")) != str(phase_name):
        raise ValueError("submitted resource resolution phase mismatch")
    if int(payload.get("iteration", -1)) != int(iteration):
        raise ValueError("submitted resource resolution iteration mismatch")
    for payload_key, intent_key in (
        ("campaign_uid", "campaign_uid"),
        ("attempt_id", "attempt_id"),
        ("submission_identity", "submission_identity"),
    ):
        expected = intent.get(intent_key)
        if expected not in (None, "") and str(payload.get(payload_key)) != str(
            expected
        ):
            raise ValueError(
                "submitted resource resolution " + payload_key + " mismatch"
            )
    intent_status = str(intent.get("status") or "")
    status = {
        "PRE_SUBMIT": "prepared",
        "SUBMITTED": "submitted",
        "ADOPTED": "submitted",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "SUPERSEDED": "superseded",
    }.get(intent_status)
    if status is None:
        raise ValueError("submission intent has an unknown resource-plan status")
    if status == "submitted" and not str(intent.get("job_id") or ""):
        raise ValueError("submitted resource plan has no scheduler JobID")
    identity = str(payload.get("submission_identity") or "")
    return {
        "phase": str(phase_name),
        "iteration": int(iteration),
        "status": status,
        "attempt_status": intent_status,
        "job_id": intent.get("job_id"),
        "attempt_reason": intent.get("reason"),
        "submission_identity": identity,
        "resource_resolution_path": str(path.resolve()),
        "formula_version": payload.get("formula_version"),
        "resources": payload["resources"],
        "evidence": payload["evidence"],
        "scratch_path_template": payload.get("scratch_path_template"),
        "telemetry": _usage_for_identity(campaign_dir, identity),
    }


def plan_phase(
    campaign_dir: Union[str, Path],
    config: Any,
    phase_name: str,
    iteration: int,
    *,
    replacement_round: int = 0,
    expected_models_version: Optional[int] = None,
    expected_reference_data_version: Optional[int] = None,
    preview_allowed: bool = True,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir).resolve()
    phase = str(phase_name)
    if phase in INLINE_PHASES:
        return {
            "phase": phase,
            "iteration": int(iteration),
            "status": "local",
            "message": (
                "phase runs in the daemon process and requests no scheduler "
                "resources"
            ),
        }
    if phase not in SBATCH_PHASES:
        return {
            "phase": phase,
            "iteration": int(iteration),
            "status": "local",
            "message": "terminal phase requests no scheduler resources",
        }
    existing = _submitted_plan(campaign, phase, int(iteration))
    if existing is not None:
        return existing
    orphaned = _orphaned_resolutions(campaign, phase, int(iteration))
    if not bool(preview_allowed):
        return {
            "phase": phase,
            "iteration": int(iteration),
            "status": "evidence_not_yet_produced",
            "message": (
                "no intent-bound immutable resolution exists for this "
                "historical iteration; current campaign heads will not be "
                "substituted"
            ),
            "orphaned_resolutions": orphaned,
        }
    try:
        evidence = collect_resource_evidence(
            phase_name=phase,
            config=config,
            campaign_dir=campaign,
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            require_evidence=True,
        )
        array_size = (
            int(evidence["n_tasks"])
            if phase in {"ARIADNE_ARRAY", "INITIAL_FEREBUS", "FEREBUS"}
            or "GAUSSIAN" in phase
            or "AIMALL" in phase
            else None
        )
        resolved = resolve_phase_resources(
            phase_name=phase,
            config=config,
            campaign_dir=campaign,
            iteration=int(iteration),
            array_size=array_size,
            replacement_round=int(replacement_round),
            expected_models_version=(
                expected_models_version if phase == "ARIADNE_ARRAY" else None
            ),
            expected_reference_data_version=(
                0
                if phase == "INITIAL_FEREBUS"
                else (
                    expected_reference_data_version
                    if phase == "FEREBUS"
                    else None
                )
            ),
            require_evidence=True,
            evidence_override=evidence,
        )
    except ResourceEvidenceUnavailable as exc:
        return {
            "phase": phase,
            "iteration": int(iteration),
            "status": "evidence_not_yet_produced",
            "message": str(exc),
            "orphaned_resolutions": orphaned,
        }
    except ResourceEvidenceInvalid as exc:
        return {
            "phase": phase,
            "iteration": int(iteration),
            "status": "evidence_invalid",
            "message": str(exc),
            "orphaned_resolutions": orphaned,
        }
    return {
        "phase": phase,
        "iteration": int(iteration),
        "status": "ready",
        "resources": resolved.to_dict(),
        "evidence": evidence,
        "telemetry": _absent_usage(campaign),
        "orphaned_resolutions": orphaned,
    }


def build_resource_plan(
    campaign_dir: Union[str, Path],
    config: Any,
    *,
    current_phase: str,
    current_iteration: int,
    phase_name: Optional[str] = None,
    iteration: Optional[int] = None,
    all_phases: bool = False,
    replacement_round: int = 0,
    current_models_version: Optional[int] = None,
    current_reference_data_version: Optional[int] = None,
) -> Dict[str, Any]:
    selected_iteration = int(
        current_iteration if iteration is None else iteration
    )
    if selected_iteration < 0:
        raise ValueError("resource-plan iteration must be >= 0")
    if all_phases:
        phases: Sequence[str] = (
            BOOTSTRAP_PHASES if selected_iteration == 0 else ACTIVE_PHASES
        )
    else:
        phases = (str(phase_name or current_phase),)
    return {
        "schema_version": RESOURCE_PLAN_SCHEMA_VERSION,
        "campaign_dir": str(Path(campaign_dir).resolve()),
        "selected_iteration": selected_iteration,
        "current_phase": str(current_phase),
        "current_iteration": int(current_iteration),
        "all": bool(all_phases),
        "replacement_round": int(replacement_round),
        "plans": [
            plan_phase(
                campaign_dir,
                config,
                phase,
                selected_iteration,
                replacement_round=int(replacement_round),
                expected_models_version=current_models_version,
                expected_reference_data_version=current_reference_data_version,
                preview_allowed=(selected_iteration == int(current_iteration)),
            )
            for phase in phases
        ],
    }


def _scheduler_name(resources: Any) -> str:
    if not isinstance(resources, dict):
        return "scheduler"
    extra = resources.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    return (
        "Sun Grid Engine"
        if str(extra.get("scheduler") or "slurm").strip().lower() == "sge"
        else "Slurm"
    )


def _human_status(value: Any, *, scheduler_name: str = "scheduler") -> str:
    return {
        "prepared": "prepared for submission",
        "submitted": "submitted to " + scheduler_name,
        "completed": "completed",
        "failed": "failed",
        "superseded": "replaced by a later attempt",
        "ready": "ready for submission",
        "local": "runs locally without scheduler submission",
        "evidence_not_yet_produced": "waiting for data from an earlier phase",
        "evidence_invalid": "resource evidence is invalid",
    }.get(str(value), str(value).replace("_", " "))


def _gibibytes(value: Any) -> str:
    try:
        return format(float(value) / float(1024 ** 3), ".2f") + " GiB"
    except (TypeError, ValueError):
        return "not required"


def format_resource_plan(
    payload: Dict[str, Any],
    *,
    verbose: bool = False,
) -> str:
    if not verbose:
        lines = [
            "Resource plan",
            "Campaign: " + str(payload["campaign_dir"]),
            "Iteration: "
            + str(payload.get("selected_iteration", payload.get("current_iteration"))),
        ]
        for plan in payload["plans"]:
            lines.append("")
            lines.append(str(plan["phase"]))
            resources = plan.get("resources")
            scheduler_name = _scheduler_name(resources)
            lines.append(
                "  Status: "
                + _human_status(
                    plan.get("status"),
                    scheduler_name=scheduler_name,
                )
            )
            if isinstance(resources, dict):
                extra = resources.get("extra")
                extra = extra if isinstance(extra, dict) else {}
                if scheduler_name == "Sun Grid Engine":
                    lines.append(
                        "  Runs on: Sun Grid Engine queue "
                        + str(extra.get("scheduler_queue") or resources.get("partition"))
                    )
                    parallel_environment = extra.get("parallel_environment")
                    if parallel_environment:
                        lines.append(
                            "  Parallel environment: "
                            + str(parallel_environment)
                        )
                else:
                    lines.append(
                        "  Runs on: Slurm partition "
                        + str(resources.get("partition"))
                    )
                array_size = extra.get("array_size")
                if array_size is not None:
                    lines.append("  Array tasks: " + str(array_size))
                    concurrency = extra.get("array_concurrency")
                    if concurrency is not None:
                        lines.append("  Maximum concurrent tasks: " + str(concurrency))
                lines.append("  CPUs per task: " + str(resources.get("cpus_per_task")))
                per_task = extra.get("per_task_allocation_gb")
                if per_task is not None:
                    lines.append("  Memory per task: " + str(per_task) + " GiB")
                else:
                    lines.append(
                        "  Memory: " + str(resources.get("mem_per_cpu")) + " per CPU"
                    )
                scratch_bytes = extra.get("expected_scratch_bytes")
                if scratch_bytes:
                    lines.append("  Expected scratch space: " + _gibibytes(scratch_bytes))
                for warning in resources.get("warnings") or []:
                    lines.append("  Warning: " + str(warning))
            elif plan.get("message"):
                lines.append("  " + str(plan.get("message")))
        return "\n".join(lines) + "\n"

    lines = [
        "Resource plan",
        "Campaign: " + str(payload["campaign_dir"]),
        "Current: "
        + str(payload["current_phase"])
        + " iteration "
        + str(payload["current_iteration"]),
    ]
    for plan in payload["plans"]:
        lines.append("")
        lines.append(
            str(plan["phase"])
            + " (iteration "
            + str(plan["iteration"])
            + "): "
            + str(plan["status"])
        )
        if plan.get("message"):
            lines.append("  " + str(plan["message"]))
        resources = plan.get("resources")
        if isinstance(resources, dict):
            lines.append(
                "  partition="
                + str(resources.get("partition"))
                + " cpus_per_task="
                + str(resources.get("cpus_per_task"))
                + " mem_per_cpu="
                + str(resources.get("mem_per_cpu"))
            )
            lines.append(
                "  cpu_formula="
                + str(resources.get("cpu_reason"))
                + " memory_formula="
                + str(resources.get("memory_reason"))
                + " estimated_total_gb="
                + str(resources.get("estimated_total_memory_gb"))
            )
            extra = resources.get("extra")
            if isinstance(extra, dict):
                lines.append(
                    "  active_workers="
                    + str(extra.get("active_workers"))
                    + " memory_only_cpus="
                    + str(extra.get("memory_only_cpus"))
                    + " peak_allocation_gb="
                    + str(extra.get("peak_allocation_gb"))
                )
                lines.append(
                    "  array_size="
                    + str(extra.get("array_size"))
                    + " concurrency="
                    + str(extra.get("array_concurrency"))
                    + " per_task_allocation_gb="
                    + str(extra.get("per_task_allocation_gb"))
                    + " safety_factor="
                    + str(extra.get("memory_estimate_safety_factor"))
                )
                if extra.get("distance_store_mode"):
                    lines.append(
                        "  diversity distance_store="
                        + str(extra.get("distance_store_mode"))
                        + " bytes="
                        + str(extra.get("condensed_store_bytes"))
                    )
                lines.append(
                    "  scratch_mode="
                    + str(extra.get("scratch_mode"))
                    + " expected_scratch_bytes="
                    + str(extra.get("expected_scratch_bytes"))
                    + " exact="
                    + str(extra.get("scratch_requirement_exact"))
                )
                if extra.get("scratch_required_bytes"):
                    lines.append(
                        "  scratch_required_bytes="
                        + str(extra.get("scratch_required_bytes"))
                        + " free_at_resolution="
                        + str(extra.get("scratch_free_bytes_at_resolution"))
                    )
                profile_limits = extra.get("profile_limits")
                if isinstance(profile_limits, dict):
                    lines.append(
                        "  profile_limits min_cpus="
                        + str(profile_limits.get("partition_min_cpus"))
                        + " max_cpus="
                        + str(profile_limits.get("partition_max_cpus"))
                        + " memory_per_core_gb="
                        + str(
                            profile_limits.get(
                                "partition_memory_per_core_gb"
                            )
                        )
                    )
                filesystem = extra.get("campaign_filesystem")
                if isinstance(filesystem, dict):
                    lines.append(
                        "  campaign_filesystem="
                        + str(filesystem.get("path"))
                        + " free_bytes="
                        + str(filesystem.get("free_bytes_at_resolution"))
                    )
            warnings = resources.get("warnings")
            if isinstance(warnings, list):
                for warning in warnings:
                    lines.append("  warning: " + str(warning))
        evidence = plan.get("evidence")
        if isinstance(evidence, dict):
            lines.append("  evidence_source=" + str(evidence.get("source")))
            hashed = []

            def collect_hashes(value: Any) -> None:
                if isinstance(value, dict):
                    if isinstance(value.get("sha256"), str):
                        hashed.append(
                            (str(value.get("path") or "<unlabelled>"), str(value["sha256"]))
                        )
                    for child in value.values():
                        collect_hashes(child)
                elif isinstance(value, list):
                    for child in value:
                        collect_hashes(child)

            collect_hashes(evidence)
            for path, digest in hashed[:5]:
                lines.append("  evidence_sha256 " + path + " " + digest)
            if len(hashed) > 5:
                lines.append(
                    "  evidence_sha256 ... "
                    + str(len(hashed) - 5)
                    + " additional file(s)"
                )
        if plan.get("scratch_path_template"):
            lines.append(
                "  scratch_template=" + str(plan.get("scratch_path_template"))
            )
        telemetry = plan.get("telemetry")
        if isinstance(telemetry, dict):
            telemetry_status = str(telemetry.get("status") or "invalid")
            lines.append("  telemetry_status=" + telemetry_status)
            if telemetry.get("error"):
                lines.append("  telemetry_error=" + str(telemetry["error"]))
            summary = telemetry.get("summary")
            if isinstance(summary, dict):
                lines.append(
                    "  observed p95_rss_mib="
                    + str(summary.get("p95_rss_mib"))
                    + " p95_elapsed_seconds="
                    + str(summary.get("p95_elapsed_seconds"))
                )
                lines.append(
                    "  advisory memory_mib="
                    + str(summary.get("recommended_memory_mib"))
                    + " walltime_seconds="
                    + str(summary.get("recommended_walltime_seconds"))
                    + " missing_task_rows="
                    + str(summary.get("n_missing_task_rows"))
                )
        orphaned = plan.get("orphaned_resolutions")
        if isinstance(orphaned, list) and orphaned:
            lines.append(
                "  warning: "
                + str(len(orphaned))
                + " unbound resource resolution artefact(s) were ignored"
            )
    return "\n".join(lines) + "\n"
