"""Read-only resource planning for current, submitted, and future phases."""
from __future__ import annotations

from ..strict_json import strict_json as json
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
    collect_resource_evidence,
    resolve_phase_resources,
)
from .submission_intent import load_intent


RESOURCE_PLAN_SCHEMA_VERSION = 1


BOOTSTRAP_PHASES = (
    "INIT",
    "PHASE_A_POLUS",
    "INITIAL_GAUSSIAN",
    "INITIAL_AIMALL",
    "INITIAL_ALLOCATION_CHECK",
    "INITIAL_REPLACEMENT_GAUSSIAN",
    "INITIAL_REPLACEMENT_AIMALL",
    "INITIAL_FEREBUS",
)
ACTIVE_PHASES = (
    "SEED_SELECT",
    "ARIADNE_ARRAY",
    "PHASE_B_POLUS",
    "SPLIT",
    "GAUSSIAN",
    "AIMALL",
    "ALLOCATION_CHECK",
    "REPLACEMENT_GAUSSIAN",
    "REPLACEMENT_AIMALL",
    "APPEND",
    "FEREBUS",
    "STOP_CHECK",
)


def _latest_resolution(
    campaign_dir: Path,
    phase_name: str,
    iteration: int,
) -> Optional[Path]:
    root = (
        campaign_dir
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "resource_resolutions"
        / str(phase_name)
        / ("iteration-" + str(int(iteration)).zfill(6))
    )
    if not root.is_dir() or root.is_symlink():
        return None
    files = [path for path in root.glob("*.json") if path.is_file() and not path.is_symlink()]
    if not files:
        return None
    return max(files, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _usage_for_identity(campaign_dir: Path, identity: str) -> Optional[Dict[str, Any]]:
    path = campaign_dir / ".DATA" / "ACTIVE_LEARNING" / "resource_usage_records.json"
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    attempts = payload.get("attempts") if isinstance(payload, dict) else None
    if not isinstance(attempts, list):
        return None
    matches = [
        dict(item)
        for item in attempts
        if isinstance(item, dict)
        and str(item.get("submission_identity") or "") == str(identity)
    ]
    return matches[-1] if matches else None


def _submitted_plan(
    campaign_dir: Path,
    phase_name: str,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    intent = load_intent(campaign_dir, phase_name, int(iteration))
    path = None
    if isinstance(intent, dict) and intent.get("resource_resolution_path"):
        path = Path(str(intent["resource_resolution_path"]))
    elif isinstance(intent, dict) and str(intent.get("status") or "") == "PRE_SUBMIT":
        # A new attempt has an identity but has not yet snapshotted resources.
        # Do not mislabel an older retry's resolution as the current attempt.
        return None
    if path is None or not path.is_file():
        path = _latest_resolution(campaign_dir, phase_name, int(iteration))
    if path is None:
        return None
    if isinstance(intent, dict) and intent.get("resource_resolution_path"):
        digest = intent.get("resource_resolution_sha256")
        payload = (
            verify_resolution(path, str(digest))
            if isinstance(digest, str) and digest
            else read_resolution(path)
        )
    else:
        payload = read_resolution(path)
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
    if isinstance(intent, dict):
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
                    "submitted resource resolution "
                    + payload_key
                    + " mismatch"
                )
    status = "submitted"
    intent_status = str((intent or {}).get("status") or "")
    if intent_status in {"COMPLETED", "FAILED", "SUPERSEDED"}:
        status = "completed"
    identity = str(payload.get("submission_identity") or "")
    return {
        "phase": str(phase_name),
        "iteration": int(iteration),
        "status": status,
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
) -> Dict[str, Any]:
    campaign = Path(campaign_dir).resolve()
    phase = str(phase_name)
    if phase in INLINE_PHASES:
        return {
            "phase": phase,
            "iteration": int(iteration),
            "status": "local",
            "message": "phase runs in the daemon process and requests no Slurm resources",
        }
    if phase not in SBATCH_PHASES:
        return {
            "phase": phase,
            "iteration": int(iteration),
            "status": "local",
            "message": "terminal phase requests no Slurm resources",
        }
    existing = _submitted_plan(campaign, phase, int(iteration))
    if existing is not None:
        return existing
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
            if phase == "ARIADNE_ARRAY" or "GAUSSIAN" in phase or "AIMALL" in phase
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
        }
    return {
        "phase": phase,
        "iteration": int(iteration),
        "status": "ready",
        "resources": resolved.to_dict(),
        "evidence": evidence,
        "telemetry": None,
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
            )
            for phase in phases
        ],
    }


def format_resource_plan(payload: Dict[str, Any]) -> str:
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
                        "  POLUS distance_store="
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
            lines.append(
                "  observed p95_rss_mib="
                + str(telemetry.get("p95_rss_mib"))
                + " p95_elapsed_seconds="
                + str(telemetry.get("p95_elapsed_seconds"))
            )
            lines.append(
                "  advisory memory_mib="
                + str(telemetry.get("recommended_memory_mib"))
                + " walltime_seconds="
                + str(telemetry.get("recommended_walltime_seconds"))
                + " missing_task_rows="
                + str(telemetry.get("n_missing_task_rows"))
            )
    return "\n".join(lines) + "\n"
