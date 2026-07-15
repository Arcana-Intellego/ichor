"""Recovery for a corrupt or missing state.json with user in the loop.

The daemon's design treats 'state.json' as the only authoritative checkpoint.
If it goes missing or fails schema validation, the daemon refuses to auto-
recover; a silent reconstruction is exactly the bug class we want to avoid.

"reconcile" inspects the on-disk artefacts that DO exist (QM_REFERENCE_DATA/
committed iterations, TRAINED_MODELS/, journal entries) and proposes a
"CampaignState" it believes is consistent with them. The proposal is
written to "<state_path>.proposed" and the operator must explicitly
promote it ("mv state.json.proposed state.json") before restarting the
daemon.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from ..strict_json import strict_json as json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union
import uuid

from ..acquisition.trajectory_pool import TrajectoryPool
from ..versioning.reference_data import ReferenceDataVersioning
from ..versioning.trained_models import TrainedModelVersioning
from ..layout import (
    ACTIVE_LEARNING_DIRNAME,
    BOOTSTRAP_DIRNAME,
    QM_REFERENCE_DATA_DIRNAME,
    TRAINED_MODELS_DIRNAME,
    reject_legacy_campaign_layout,
)
from .artifact_contracts import (
    verify_committed_model_version,
    verify_committed_reference_data_version,
    verify_state_referenced_artifacts,
)
from .journal import iter_events
from .filesystem import campaign_owned_path
from .recovery_contracts import (
    RecoveryDecision,
    active_iteration_handoff_decisions,
    select_recovery_phase,
    staging_handoff_decisions,
    validate_phase_recovery_contract,
)
from . import submission_intent as _submission_intent
from .array_recovery import (
    compact_array_recovery_summary,
    discover_partial_array_recovery,
    refresh_array_ledger,
    supports_partial_array_recovery,
)
from .state import (
    CampaignPhase,
    CampaignState,
    DEFAULT_STATE_FILENAME,
    StateSchemaError,
    fresh_campaign_state,
    make_lifecycle_context,
    read_state,
    atomic_write_text,
    write_state,
)


__all__ = [
    "ReconciliationReport",
    "data_staging_inventory",
    "propose_recovery",
    "restore_archived_bootstrap_handoff",
    "stateful_campaign_artifacts",
    "write_proposed_state",
    "RECONCILE_SUFFIX",
]


RECONCILE_SUFFIX = ".proposed"

_RECOVERY_PHASE_PROGRESS = {
    phase: rank
    for rank, phase in enumerate((
        CampaignPhase.INIT,
        CampaignPhase.PHASE_A_DIVERSITY,
        CampaignPhase.INITIAL_GAUSSIAN,
        CampaignPhase.INITIAL_AIMALL,
        CampaignPhase.INITIAL_ALLOCATION_CHECK,
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
        CampaignPhase.INITIAL_REPLACEMENT_AIMALL,
        CampaignPhase.INITIAL_FEREBUS,
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
    ))
}


def _iso_from_timestamp(value: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()


def data_staging_inventory(campaign_dir: Union[str, Path]) -> Dict[str, Any]:
    """Return a read-only summary of ``.DATA/STAGING``.

    The inventory is intentionally conservative: it never follows symlinks,
    and failures are returned as diagnostics so recovery dashboards remain
    usable on partially damaged campaign trees.
    """
    campaign = Path(campaign_dir)
    staging = campaign / ".DATA" / "STAGING"
    payload: Dict[str, Any] = {
        "path": str(staging),
        "exists": staging.exists(),
        "is_dir": staging.is_dir(),
        "is_symlink": staging.is_symlink(),
        "top_level_count": 0,
        "top_level_entries": [],
        "total_entries": 0,
        "total_bytes": 0,
        "has_symlink": False,
        "symlink_entries": [],
        "oldest_mtime_iso": None,
        "newest_mtime_iso": None,
        "error": None,
    }
    if not staging.exists():
        return payload
    if staging.is_symlink():
        payload["has_symlink"] = True
        payload["symlink_entries"] = ["."]
        return payload
    if not staging.is_dir():
        payload["error"] = ".DATA/STAGING exists but is not a directory"
        return payload
    try:
        children = sorted(
            [p for p in staging.iterdir() if p.name not in (".", "..")],
            key=lambda p: p.name,
        )
        payload["top_level_count"] = len(children)
        payload["top_level_entries"] = [p.name for p in children[:10]]
        mtimes: List[float] = []
        for path in [staging] + list(staging.rglob("*")):
            try:
                st = path.lstat()
            except OSError:
                continue
            payload["total_entries"] = int(payload["total_entries"]) + 1
            mtimes.append(float(st.st_mtime))
            if path.is_symlink():
                payload["has_symlink"] = True
                try:
                    rel = str(path.relative_to(staging))
                except ValueError:
                    rel = str(path)
                payload["symlink_entries"].append(rel)
                continue
            if path.is_file():
                payload["total_bytes"] = int(payload["total_bytes"]) + int(st.st_size)
        if mtimes:
            payload["oldest_mtime_iso"] = _iso_from_timestamp(min(mtimes))
            payload["newest_mtime_iso"] = _iso_from_timestamp(max(mtimes))
    except Exception as exc:
        payload["error"] = type(exc).__name__ + ": " + str(exc)[:180]
    return payload


def stateful_campaign_artifacts(campaign_dir: Union[str, Path]) -> List[str]:
    """Return recovery-relevant artefacts that make fresh state unsafe.

    A first-run campaign may contain ``campaign.yaml`` and an imported
    trajectory pool before ``state.json`` exists. Once any daemon-owned
    stateful artefact appears, a missing ``state.json`` is a recovery
    condition and must go through reconcile rather than fresh initialisation.
    """
    campaign = Path(campaign_dir)
    findings: List[str] = []
    stateful_journal_events = {
        "campaign_started",
        "daemon_started",
        "phase_transition",
        "sbatch",
        "phase_succeeded",
        "phase_succeeded_live",
        "halt",
        "reconcile_applied",
        "adopted_inflight_job",
        "tick_exception_halted",
        "operator_stop_requested",
        "operator_stop_boundary_reached",
    }

    def add_matches(pattern: str) -> None:
        for path in sorted(campaign.glob(pattern)):
            findings.append(str(path.relative_to(campaign)))

    add_matches("QM_REFERENCE_DATA/iteration-*")
    add_matches("TRAINED_MODELS/iteration-*")
    add_matches(ACTIVE_LEARNING_DIRNAME + "/iteration-*")
    add_matches(".DATA/" + BOOTSTRAP_DIRNAME + "/selection/SELECTION.json")
    add_matches(".DATA/" + BOOTSTRAP_DIRNAME + "/selection/selected.xyz")
    add_matches(".DATA/" + BOOTSTRAP_DIRNAME + "/selection/selected_indices.dat")
    add_matches(".DATA/" + BOOTSTRAP_DIRNAME + "/allocation/POINT_ALLOCATION.json")
    add_matches(".DATA/ACTIVE_LEARNING/stop_request.json")
    add_matches(".DATA/ACTIVE_LEARNING/stop_request_history/*.json")
    add_matches(".DATA/ACTIVE_LEARNING/reconcile_transactions/*.json")
    config_lock = campaign / ".DATA" / "ACTIVE_LEARNING" / "config_lock.json"
    pool_manifest = campaign / ".DATA" / "TRAJECTORY" / "pool.manifest.json"
    if config_lock.is_file() and not pool_manifest.is_file():
        findings.append(".DATA/ACTIVE_LEARNING/config_lock.json")
    journal_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    if journal_path.is_file():
        try:
            for event in iter_events(journal_path):
                if str(event.get("event", "")) in stateful_journal_events:
                    findings.append(".DATA/ACTIVE_LEARNING/journal.ndjson")
                    break
        except Exception:
            findings.append(".DATA/ACTIVE_LEARNING/journal.ndjson")
    proposed_state = (
        campaign
        / ".DATA"
        / "ACTIVE_LEARNING"
        / (DEFAULT_STATE_FILENAME + RECONCILE_SUFFIX)
    )
    if proposed_state.exists():
        findings.append(
            ".DATA/ACTIVE_LEARNING/"
            + DEFAULT_STATE_FILENAME
            + RECONCILE_SUFFIX
        )
    intents = campaign / ".DATA" / "ACTIVE_LEARNING" / "submission_intents"
    if intents.is_dir():
        for path in sorted(intents.glob("*.json")):
            findings.append(str(path.relative_to(campaign)))
    staging = campaign / ".DATA" / "STAGING"
    if staging.is_dir():
        for path in sorted(staging.iterdir()):
            findings.append(str(path.relative_to(campaign)))
    scripts = campaign / ".DATA" / "SCRIPTS"
    if scripts.is_dir():
        for path in sorted(scripts.glob("*.sh")):
            findings.append(str(path.relative_to(campaign)))
        jobs = scripts / "JOBS"
        if jobs.is_dir():
            for path in sorted(jobs.rglob("job.sh")):
                findings.append(str(path.relative_to(campaign)))
    scratch = campaign / ".DATA" / "SCRATCH"
    if scratch.is_dir():
        scratch_tasks = sorted(scratch.rglob("TASK.json"))
        if scratch_tasks:
            for path in scratch_tasks:
                findings.append(str(path.relative_to(campaign)))
        elif any(scratch.iterdir()):
            findings.append(".DATA/SCRATCH")
    return findings


@dataclass
class ReconciliationReport:
    """Diagnostic bundle returned by :func: propose_recovery."""

    proposed_state: CampaignState
    committed_reference_data_versions: List[int] = field(default_factory=list)
    committed_model_versions: List[int] = field(default_factory=list)
    valid_reference_data_versions: List[int] = field(default_factory=list)
    valid_model_versions: List[int] = field(default_factory=list)
    last_phase_in_journal: Optional[str] = None
    last_iteration_in_journal: Optional[int] = None
    last_phase_event_in_journal: Optional[str] = None
    last_phase_retryable: bool = False
    last_halt_event: Optional[Dict[str, Any]] = None
    script_inventory: Dict[str, Any] = field(default_factory=dict)
    scratch_inventory: List[Dict[str, Any]] = field(default_factory=list)
    reconcile_transactions: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    existing_state_loaded: bool = False
    unsafe_reasons: List[str] = field(default_factory=list)
    active_submission_intents: List[Dict[str, Any]] = field(default_factory=list)
    decision: str = ""
    trusted_artifacts: List[str] = field(default_factory=list)
    blocking_artifacts: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    recovery_candidates: List[Dict[str, Any]] = field(default_factory=list)
    partial_array_recovery: Optional[Dict[str, Any]] = None
    bootstrap_handoff: Optional[Dict[str, Any]] = None
    phase_a_handoff: Optional[Dict[str, Any]] = None


def _candidate_payload(decision: RecoveryDecision) -> Dict[str, Any]:
    return {
        "phase": decision.phase.value,
        "iteration": int(decision.iteration),
        "path": str(decision.trusted_artifact or ""),
        "reason": str(decision.reason),
        "replacement_round": int(getattr(decision, "replacement_round", 0)),
    }


def _append_recovery_candidate(
    candidates: List[Dict[str, Any]],
    decision: Optional[RecoveryDecision],
) -> None:
    if decision is None:
        return
    payload = _candidate_payload(decision)
    key = (
        str(payload.get("phase")),
        int(payload.get("iteration", 0)),
        str(payload.get("path") or ""),
    )
    for existing in candidates:
        existing_key = (
            str(existing.get("phase")),
            int(existing.get("iteration", 0)),
            str(existing.get("path") or ""),
        )
        if existing_key == key:
            return
    candidates.append(payload)


def _needs_trajectory_pool_check(
    *,
    existing_loaded: bool,
    reference_data_versions: List[int],
    model_versions: List[int],
    active_intents: List[Dict[str, Any]],
    last_phase: Optional[str],
    staging_children: List[Path],
    script_files: List[Path],
    dangling_reference_data: List[Path],
    dangling_models: List[Path],
    has_model_iteration_staging: bool,
) -> bool:
    return any(
        (
            existing_loaded,
            bool(reference_data_versions),
            bool(model_versions),
            bool(active_intents),
            bool(last_phase),
            bool(staging_children),
            bool(script_files),
            bool(dangling_reference_data),
            bool(dangling_models),
            bool(has_model_iteration_staging),
        )
    )


def _trajectory_pool_unsafe_reason(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, FileNotFoundError):
        if "pool manifest" in msg:
            return "trajectory pool manifest missing"
        if "pool xyz" in msg:
            return "trajectory pool missing"
        return "trajectory pool missing"
    if isinstance(exc, RuntimeError) and "pool drift detected" in msg:
        return "trajectory pool SHA mismatch"
    if isinstance(exc, ValueError):
        if "natoms" in msg or "atom_types" in msg or "masses" in msg:
            return "trajectory pool atom count invalid"
        return "trajectory pool unreadable"
    return "trajectory pool unreadable"


def _scripts_inventory(campaign: Path) -> Dict[str, Any]:
    scripts = campaign / ".DATA" / "SCRIPTS"
    payload: Dict[str, Any] = {
        "path": str(scripts),
        "exists": scripts.exists(),
        "is_dir": scripts.is_dir(),
        "count": 0,
        "sample": [],
        "legacy_flat_script_count": 0,
        "attempt_bundle_count": 0,
        "attempt_bundles": [],
        "has_symlink": False,
        "symlink_entries": [],
        "invalid_bundle_entries": [],
        "error": None,
    }
    if not scripts.exists():
        return payload
    if not scripts.is_dir():
        payload["error"] = ".DATA/SCRIPTS exists but is not a directory"
        return payload
    try:
        symlinks = sorted(
            [path for path in scripts.rglob("*") if path.is_symlink()],
            key=lambda path: str(path.relative_to(scripts)),
        )
        payload["has_symlink"] = bool(symlinks)
        payload["symlink_entries"] = [
            str(path.relative_to(campaign)) for path in symlinks[:20]
        ]
        files = sorted(
            [p for p in scripts.rglob("*") if p.is_file() and not p.is_symlink()],
            key=lambda p: str(p.relative_to(scripts)),
        )
        payload["count"] = len(files)
        payload["sample"] = [
            str(p.relative_to(campaign))
            for p in files[:5]
        ]
        payload["legacy_flat_script_count"] = len(
            [path for path in scripts.glob("*.sh") if path.is_file()]
        )
        intent_status_by_identity: Dict[str, str] = {}
        intents_root = campaign / ".DATA" / "ACTIVE_LEARNING" / "submission_intents"
        if intents_root.is_dir() and not intents_root.is_symlink():
            for intent_path in sorted(intents_root.glob("*.json")):
                try:
                    intent_payload = json.loads(
                        intent_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    continue
                if not isinstance(intent_payload, dict):
                    continue
                identity = str(intent_payload.get("submission_identity") or "")
                if identity:
                    intent_status_by_identity[identity] = str(
                        intent_payload.get("status") or ""
                    )
        bundles = []
        jobs_root = scripts / "JOBS"
        if jobs_root.is_dir() and not jobs_root.is_symlink():
            for job_script in sorted(jobs_root.rglob("job.sh")):
                if job_script.is_symlink() or not job_script.is_file():
                    continue
                try:
                    backend, phase, iteration_token, identity, filename = (
                        job_script.relative_to(jobs_root).parts
                    )
                except ValueError:
                    payload["invalid_bundle_entries"].append(
                        str(job_script.relative_to(campaign))
                    )
                    continue
                if filename != "job.sh":
                    continue
                bundle = job_script.parent
                outputs = bundle / "OUTPUTS"
                errors = bundle / "ERRORS"
                if (
                    outputs.is_symlink()
                    or errors.is_symlink()
                    or not outputs.is_dir()
                    or not errors.is_dir()
                ):
                    payload["invalid_bundle_entries"].append(
                        str(bundle.relative_to(campaign))
                    )
                    continue
                output_count = (
                    len([path for path in outputs.iterdir() if path.is_file()])
                    if outputs.is_dir() and not outputs.is_symlink()
                    else 0
                )
                error_count = (
                    len([path for path in errors.iterdir() if path.is_file()])
                    if errors.is_dir() and not errors.is_symlink()
                    else 0
                )
                bundles.append(
                    {
                        "backend": backend,
                        "phase": phase,
                        "iteration": iteration_token,
                        "submission_identity": identity,
                        "path": str(bundle.relative_to(campaign)),
                        "intent_status": intent_status_by_identity.get(identity),
                        "output_log_count": output_count,
                        "error_log_count": error_count,
                        "has_array_task_map": (bundle / "array_task_map.json").is_file(),
                    }
                )
        payload["attempt_bundle_count"] = len(bundles)
        payload["attempt_bundles"] = bundles
    except Exception as exc:
        payload["error"] = type(exc).__name__ + ": " + str(exc)[:180]
    return payload


def _initial_aimall_handoff_indicated(
    *,
    existing: Optional[CampaignState],
    last_phase: Optional[str],
) -> bool:
    if str(last_phase or "") in {
        CampaignPhase.INITIAL_AIMALL.value,
        CampaignPhase.INITIAL_FEREBUS.value,
    }:
        return True
    if existing is None:
        return False
    try:
        phase = CampaignPhase(existing.phase)
    except Exception:
        return False
    return phase in {
        CampaignPhase.INITIAL_AIMALL,
        CampaignPhase.INITIAL_FEREBUS,
    }


def _journal_phase_hint(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = str(event.get("event", ""))
    if name == "phase_transition":
        phase = event.get("to_phase")
        retryable = False
    if name in {
        "sbatch",
        "phase_succeeded",
        "phase_succeeded_live",
        "reconcile_applied",
    }:
        phase = event.get("phase")
        retryable = False
    if name in {"halt", "tick_exception_halted"}:
        phase = event.get("from_phase")
        retryable = True
    if name not in {
        "phase_transition",
        "sbatch",
        "phase_succeeded",
        "phase_succeeded_live",
        "reconcile_applied",
        "halt",
        "tick_exception_halted",
    }:
        return None
    if phase is None:
        return None
    return {
        "phase": phase,
        "event": name,
        "retryable": bool(retryable),
    }


def _validate_initial_ferebus_bootstrap(
    campaign_dir: Union[str, Path],
    *,
    iteration: int,
) -> None:
    from .input_staging import read_quantum_acceptance_manifest

    read_quantum_acceptance_manifest(
        Path(campaign_dir) / ".DATA" / "STAGING" / "initial",
        expected_phase=CampaignPhase.INITIAL_AIMALL.value,
        expected_iteration=int(iteration),
        require_nonempty=True,
        require_points_file_membership=True,
    )


def _read_bootstrap_handoff_at(
    initial_dir: Path,
    *,
    expected_iteration: int,
    archived: bool,
) -> Optional[Dict[str, Any]]:
    from .input_staging import (
        quantum_acceptance_manifest_path,
        read_quantum_acceptance_manifest,
    )
    from ..strict_json import strict_json as _json

    initial = Path(initial_dir)
    manifest_path = quantum_acceptance_manifest_path(initial)
    if not manifest_path.is_file():
        return None
    try:
        raw = _json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    phase = str(raw.get("phase") or "")
    if phase not in {
        CampaignPhase.INITIAL_GAUSSIAN.value,
        CampaignPhase.INITIAL_AIMALL.value,
    }:
        return None
    try:
        pointdirs, manifest = read_quantum_acceptance_manifest(
            initial,
            expected_phase=phase,
            expected_iteration=int(expected_iteration),
            require_nonempty=True,
            require_points_file_membership=not archived,
        )
    except Exception:
        return None
    return {
        "path": str(initial),
        "manifest_path": str(manifest_path),
        "phase": phase,
        "iteration": int(manifest.get("iteration", expected_iteration)),
        "n_total": int(manifest.get("n_total", len(pointdirs))),
        "accepted_count": len(pointdirs),
        "archived": bool(archived),
        "campaign_uid": str(manifest.get("campaign_uid") or ""),
    }


def _find_bootstrap_handoff(
    campaign_dir: Union[str, Path],
    *,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    campaign = Path(campaign_dir)
    live = _read_bootstrap_handoff_at(
        campaign / ".DATA" / "STAGING" / "initial",
        expected_iteration=int(iteration),
        archived=False,
    )
    if live is not None:
        return live
    data = campaign / ".DATA"
    archives = sorted(
        [
            path
            for path in data.glob("STAGING.archived-*")
            if path.is_dir() and not path.is_symlink()
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    for archive in archives:
        handoff = _read_bootstrap_handoff_at(
            archive / "initial",
            expected_iteration=int(iteration),
            archived=True,
        )
        if handoff is not None:
            return handoff
    return None


def _find_phase_a_handoff(campaign_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    from ..handoff_manifests import (
        phase_a_sample_manifest_path,
        read_phase_a_sample_manifest,
    )
    from ..layout import bootstrap_selection_dir

    initial = bootstrap_selection_dir(Path(campaign_dir))
    try:
        manifest = read_phase_a_sample_manifest(initial, require_nonempty=True)
    except Exception:
        return None
    return {
        "path": str(initial),
        "manifest_path": str(phase_a_sample_manifest_path(initial)),
        "phase": CampaignPhase.PHASE_A_DIVERSITY.value,
        "iteration": 0,
        "n_select": int(manifest.get("n_select", 0)),
        "sample_xyz": str(manifest.get("sample_xyz", "")),
        "index_path": str(manifest.get("index_path", "")),
        "campaign_uid": str(manifest.get("campaign_uid") or ""),
    }


def restore_archived_bootstrap_handoff(
    campaign_dir: Union[str, Path],
    report: ReconciliationReport,
) -> List[str]:
    handoff = report.bootstrap_handoff
    if not isinstance(handoff, dict) or not bool(handoff.get("archived")):
        return []
    campaign = Path(campaign_dir)
    source = campaign_owned_path(
        campaign,
        Path(str(handoff.get("path") or "")),
    )
    target = campaign_owned_path(
        campaign,
        campaign / ".DATA" / "STAGING" / "initial",
    )
    phase = str(handoff.get("phase") or "")
    iteration = int(handoff.get("iteration", 0))
    if not source.is_dir():
        raise FileNotFoundError("archived bootstrap handoff is missing: " + str(source))
    if source.is_symlink():
        raise ValueError("refusing to restore symlinked bootstrap handoff: " + str(source))
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                "refusing to restore bootstrap handoff containing symlink: "
                + str(path)
            )
    from .input_staging import read_quantum_acceptance_manifest

    source_pointdirs, _manifest = read_quantum_acceptance_manifest(
        source,
        expected_phase=phase,
        expected_iteration=iteration,
        require_nonempty=True,
        require_points_file_membership=False,
    )
    pointdir_names = [path.name for path in source_pointdirs]
    if target.exists():
        if target.is_symlink():
            raise ValueError("refusing to restore over symlinked .DATA/STAGING/initial")
        if not target.is_dir():
            raise ValueError(".DATA/STAGING/initial exists but is not a directory")
        existing = [p for p in target.iterdir() if p.name not in (".", "..")]
        if existing:
            raise ValueError(
                "refusing to restore archived bootstrap handoff because "
                ".DATA/STAGING/initial is not empty"
            )
        target.rmdir()
    campaign_owned_path(campaign, target.parent).mkdir(parents=True, exist_ok=True)
    temporary = campaign_owned_path(
        campaign,
        target.with_name(target.name + ".restore-" + uuid.uuid4().hex),
    )
    created_target = False
    try:
        shutil.copytree(str(source), str(temporary), symlinks=False)
        points_body = "\n".join(
            str((target / name).resolve(strict=False))
            for name in pointdir_names
        )
        atomic_write_text(
            temporary / "POINTS.txt",
            points_body + ("\n" if points_body else ""),
        )
        temporary.replace(target)
        created_target = True
        read_quantum_acceptance_manifest(
            target,
            expected_phase=phase,
            expected_iteration=iteration,
            require_nonempty=True,
            require_points_file_membership=True,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(str(temporary), ignore_errors=False)
        if created_target and target.exists():
            shutil.rmtree(str(target), ignore_errors=False)
        raise
    return [str(target)]


def _trusted_campaign_uid_sources(
    campaign: Path,
    *,
    reference_data_dir_name: str,
    models_dir_name: str,
    valid_reference_data_versions: List[int],
    valid_model_versions: List[int],
    bootstrap_handoff: Optional[Dict[str, Any]],
    phase_a_handoff: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Collect campaign identities only from independently validated artefacts."""
    from .completion_receipts import (
        receipt_dir,
        receipt_reference,
        validate_completion_reference,
    )

    sources: List[Dict[str, str]] = []

    def add(label: str, value: Any) -> None:
        uid = str(value or "")
        if uid:
            sources.append({"source": str(label), "campaign_uid": uid})

    reference_versions = ReferenceDataVersioning(
        campaign / reference_data_dir_name
    )
    for version in valid_reference_data_versions:
        try:
            view = reference_versions.resolve(int(version), verification="deep")
        except Exception:
            continue
        add("reference-data version " + str(int(version)), view.campaign_uid)
    model_versions = TrainedModelVersioning(campaign / models_dir_name)
    for version in valid_model_versions:
        try:
            model_set = model_versions.resolve(int(version), verification="deep")
        except Exception:
            continue
        add("model version " + str(int(version)), model_set.campaign_uid)
    for label, handoff in (
        ("bootstrap handoff", bootstrap_handoff),
        ("Phase A handoff", phase_a_handoff),
    ):
        if isinstance(handoff, dict):
            add(label, handoff.get("campaign_uid"))

    intent_root = _submission_intent.intent_dir(campaign)
    if intent_root.is_dir():
        for path in sorted(intent_root.glob("*.json")):
            try:
                from ..strict_json import strict_json as _json

                raw = _json.loads(path.read_text(encoding="utf-8"))
                intent = _submission_intent.load_intent(
                    campaign,
                    str(raw.get("phase") or ""),
                    int(raw.get("iteration")),
                )
            except Exception:
                continue
            if intent is not None:
                add("submission intent " + path.name, intent.get("campaign_uid"))

    receipts = receipt_dir(campaign)
    if receipts.is_dir():
        for path in sorted(receipts.glob("*.json")):
            try:
                reference = receipt_reference(campaign, path)
                receipt = validate_completion_reference(campaign, reference)
            except Exception:
                continue
            add("phase completion " + path.name, receipt.get("campaign_uid"))
    return sources


def _validate_recovered_state_contract(
    campaign_dir: Union[str, Path],
    state: CampaignState,
    *,
    bootstrap_handoff: Optional[Dict[str, Any]] = None,
    phase_a_handoff: Optional[Dict[str, Any]] = None,
) -> None:
    phase = CampaignPhase(state.phase)
    if phase in {CampaignPhase.INIT, CampaignPhase.DONE, CampaignPhase.HALTED}:
        return
    reference_data_version = int(getattr(state, "reference_data_version", -1))
    model_version = int(getattr(state, "models_version", -1))
    if (
        phase is CampaignPhase.INITIAL_GAUSSIAN
        and reference_data_version < 0
        and model_version < 0
    ):
        if isinstance(phase_a_handoff, dict):
            return
    if (
        phase is CampaignPhase.INITIAL_AIMALL
        and reference_data_version < 0
        and model_version < 0
        and isinstance(bootstrap_handoff, dict)
        and bootstrap_handoff.get("phase")
        == CampaignPhase.INITIAL_GAUSSIAN.value
        and int(bootstrap_handoff.get("iteration", -1))
        == int(getattr(state, "iteration", 0))
    ):
        return
    if (
        phase is CampaignPhase.INITIAL_FEREBUS
        and reference_data_version < 0
        and model_version < 0
    ):
        if (
            isinstance(bootstrap_handoff, dict)
            and bootstrap_handoff.get("phase") == CampaignPhase.INITIAL_AIMALL.value
            and int(bootstrap_handoff.get("iteration", -1)) == int(getattr(state, "iteration", 0))
        ):
            return
        _validate_initial_ferebus_bootstrap(
            campaign_dir,
            iteration=int(getattr(state, "iteration", 0)),
        )
        return
    validate_phase_recovery_contract(campaign_dir, state)
    verify_state_referenced_artifacts(campaign_dir, state)


def propose_recovery(
    campaign_dir: Union[str, Path],
    *,
    reference_data_dir_name: str = QM_REFERENCE_DATA_DIRNAME,
    models_dir_name: str = TRAINED_MODELS_DIRNAME,
    data_subdir: Union[str, Path] = Path(".DATA") / "ACTIVE_LEARNING",
    iteration_prefix: str = "iteration",
    allow_fresh_init_on_nonempty: bool = False,
    _active_reconcile_transaction_id: Optional[str] = None,
) -> ReconciliationReport:
    """Inspect the campaign tree and propose a recovered CampaignState.

    The recovered state is conservative: it always positions the daemon at
    a stable "safe" entry point (STOP_CHECK for completed iterations or
    INIT when nothing is committed yet) and clears any pending_jobs so the
    next run re-submits rather than blindly polls unknown JobIDs.

    Heuristics:
        - committed reference-data versions in "QM_REFERENCE_DATA/" define the maximum
          completed iteration; the next iteration to plan from is one past
          that.
        - committed model versions in "TRAINED_MODELS/" likewise.
        - journal entries inform "last_phase_in_journal" for the report only.
        - if a prior state.json exists and parses, its "max_iterations",
          "campaign_uid", and "campaign_started_iso" are preserved so the
          recovery does not destroy provenance.
    """
    campaign = Path(campaign_dir)
    reject_legacy_campaign_layout(campaign)
    data = campaign / data_subdir
    state_path = data / DEFAULT_STATE_FILENAME

    notes: List[str] = []
    unsafe_reasons: List[str] = []
    trusted_artifacts: List[str] = []
    blocking_artifacts: List[str] = []
    recommended_actions: List[str] = []
    recovery_candidates: List[Dict[str, Any]] = []

    existing: Optional[CampaignState] = None
    existing_loaded = False
    salvaged_uid = None
    salvaged_started = None
    if state_path.exists():
        try:
            existing = read_state(state_path)
            existing_loaded = True
            notes.append("existing state.json loaded; preserved campaign_uid and max_iterations")
        except (StateSchemaError, ValueError) as exc:
            notes.append("existing state.json failed validation: " + str(exc)[:200])
            # present but schema-invalid (a bad alpha_history, say). salvage the
            # campaign identity from the raw json so recovery does not silently
            # mint a brand-new campaign_uid and bin the provenance.
            try:
                from ..strict_json import strict_json as _json
                raw = _json.loads(state_path.read_text(encoding="utf-8"))
                salvaged_uid = raw.get("campaign_uid")
                salvaged_started = raw.get("campaign_started_iso")
                if salvaged_uid:
                    notes.append("salvaged campaign_uid from the unparsable state.json")
                else:
                    notes.append(
                        "no campaign_uid could be salvaged; trusted artefacts "
                        "must provide a unanimous identity"
                    )
            except Exception:
                notes.append(
                    "state.json is not JSON; trusted artefacts must provide a "
                    "unanimous campaign identity"
                )
        except Exception as exc:
            notes.append("existing state.json unreadable: " + type(exc).__name__)
            notes.append(
                "restore state.json or recover a unanimous campaign_uid from "
                "trusted artefacts"
            )
    else:
        notes.append("no existing state.json")

    if existing is not None:
        try:
            from .completion_receipts import (
                replayable_completion_receipts,
                validate_completion_reference,
            )
            from .config_lock import config_lock_path

            if isinstance(existing.last_completion_receipt, dict):
                validate_completion_reference(
                    campaign,
                    existing.last_completion_receipt,
                    expected_campaign_uid=str(existing.campaign_uid),
                )
                trusted_artifacts.append("state-referenced phase completion receipt")
            lock_payload = json.loads(
                config_lock_path(campaign).read_text(encoding="utf-8")
            )
            locked_fingerprint = str(lock_payload.get("fingerprint_sha256") or "")
            if locked_fingerprint:
                replayable = replayable_completion_receipts(
                    campaign,
                    existing,
                    expected_config_sha256=None,
                )
                if len(replayable) > 1:
                    unsafe_reasons.append(
                        "multiple phase-completion receipts match state.json"
                    )
                    blocking_artifacts.append("phase completion receipts")
                elif len(replayable) == 1:
                    match = replayable[0]
                    existing = CampaignState.from_dict(
                        dict(match["payload"]["state_after"])
                    )
                    existing.last_completion_receipt = dict(match["reference"])
                    notes.append(
                        "replayed phase-completion receipt in recovery proposal: "
                        + str(match["payload"].get("phase"))
                        + "@"
                        + str(match["payload"].get("iteration"))
                    )
                    trusted_artifacts.append("replayable phase completion receipt")
        except FileNotFoundError:
            pass
        except Exception as exc:
            unsafe_reasons.append(
                "phase-completion receipt validation failed: "
                + type(exc).__name__
                + ": "
                + str(exc)[:180]
            )
            blocking_artifacts.append("phase completion receipt")

    #committed reference-data versions
    training_dir = campaign / reference_data_dir_name
    if training_dir.is_dir():
        tv = ReferenceDataVersioning(
            training_dir,
            prefix=iteration_prefix,
        ).list_committed_versions()
    else:
        tv = []
        notes.append("training dir " + reference_data_dir_name + " missing")
    #committed model versions
    models_dir = campaign / models_dir_name
    if models_dir.is_dir():
        mv = TrainedModelVersioning(
            models_dir,
            prefix=iteration_prefix,
        ).list_committed_versions()
    else:
        mv = []
        notes.append("models dir " + models_dir_name + " missing")

    #last phase observed in journal (informational)
    last_phase = None
    last_iter = None
    last_phase_event = None
    last_phase_retryable = False
    last_halt_event = None
    journal_path = data / "journal.ndjson"
    if journal_path.exists():
        try:
            for event in iter_events(journal_path):
                phase_hint = _journal_phase_hint(event)
                if phase_hint:
                    last_phase = str(phase_hint["phase"])
                    last_phase_event = str(phase_hint.get("event") or "")
                    last_phase_retryable = bool(phase_hint.get("retryable", False))
                    last_iter = event.get("iteration", last_iter)
                if str(event.get("event") or "") == "halt":
                    last_halt_event = dict(event)
        except Exception as exc:
            journal_problem = (
                "journal is corrupt or unreadable: "
                + type(exc).__name__
                + ": "
                + str(exc)
            )
            notes.append(journal_problem)
            unsafe_reasons.append(journal_problem)
    # Bootstrap is the only sampling transaction that uses iteration zero.
    # A stale active-loop state must never redirect recovery away from it.
    bootstrap_iteration = 0
    bootstrap_handoff = _find_bootstrap_handoff(
        campaign,
        iteration=bootstrap_iteration,
    )
    phase_a_handoff = _find_phase_a_handoff(campaign)
    initial_handoff_indicated = _initial_aimall_handoff_indicated(
        existing=existing,
        last_phase=last_phase,
    )
    if bootstrap_handoff is not None:
        initial_handoff_indicated = True

    active_intents: List[Dict[str, Any]] = []
    intent_root = _submission_intent.intent_dir(campaign)
    if intent_root.is_dir():
        for p in sorted(intent_root.glob("*.json")):
            try:
                stem = p.stem
                matches = [
                    phase.value
                    for phase in CampaignPhase
                    if stem.startswith(phase.value + "-")
                ]
                if len(matches) != 1:
                    raise ValueError("intent filename has no unique phase identity")
                phase_name = matches[0]
                iteration_text = stem[len(phase_name) + 1 :]
                if len(iteration_text) != 6 or not iteration_text.isdigit():
                    raise ValueError("intent filename iteration is invalid")
                payload = _submission_intent.load_intent(
                    campaign,
                    phase_name,
                    int(iteration_text),
                )
                if payload is None:
                    raise ValueError("intent disappeared during inventory")
            except Exception as exc:
                unsafe_reasons.append(
                    "malformed submission intent: "
                    + str(p)
                    + ": "
                    + type(exc).__name__
                    + ": "
                    + str(exc)[:160]
                )
                blocking_artifacts.append("malformed submission intent")
                continue
            if str(payload.get("status")) in _submission_intent.ACTIVE_STATUSES:
                active_intents.append(payload)
    active_intents.sort(
        key=lambda item: (
            str(item.get("updated_at_iso") or item.get("updated_iso") or ""),
            str(item.get("phase") or ""),
            int(item.get("iteration") or 0),
        )
    )
    from .scratch import inventory as scratch_inventory_records
    from .reconcile_transaction import inventory_reconcile_transactions

    scratch_inventory = scratch_inventory_records(campaign)
    reconcile_transactions = inventory_reconcile_transactions(campaign)
    blocking_reconcile_transactions = [
        record
        for record in reconcile_transactions
        if record.get("status") not in {"COMMITTED", "FAILED"}
        and str(record.get("transaction_id") or "")
        != str(_active_reconcile_transaction_id or "")
    ]
    if blocking_reconcile_transactions:
        unsafe_reasons.append(
            "incomplete or invalid reconcile transaction evidence: "
            + "; ".join(
                str(record.get("path"))
                + " ("
                + str(record.get("status"))
                + ")"
                for record in blocking_reconcile_transactions[:5]
            )
        )
        blocking_artifacts.append("reconcile transaction evidence")
        recommended_actions.append(
            "Inspect and resolve the recorded reconcile transaction before "
            "applying another repair."
        )
    invalid_scratch = [
        record for record in scratch_inventory if record.get("status") == "invalid"
    ]
    prepared_scratch = [
        record for record in scratch_inventory if record.get("status") == "prepared"
    ]
    retained_scratch = [
        record
        for record in scratch_inventory
        if record.get("status") in {"failed_retained", "completed"}
    ]
    if invalid_scratch:
        unsafe_reasons.append(
            "invalid campaign scratch evidence: "
            + "; ".join(
                str(record.get("path")) + " (" + str(record.get("reason")) + ")"
                for record in invalid_scratch[:5]
            )
        )
        blocking_artifacts.append("invalid scratch evidence")
    if prepared_scratch:
        unsafe_reasons.append(
            "scheduler-inconclusive prepared scratch task(s) preserve job ownership: "
            + ", ".join(
                str(record.get("job_id")) + "@" + str(record.get("phase"))
                for record in prepared_scratch[:8]
            )
        )
        blocking_artifacts.append("prepared scratch ownership")
        recommended_actions.append(
            "Inspect the recorded scratch JobIDs with sacct/squeue before recovery."
        )
    if retained_scratch:
        unsafe_reasons.append(
            "retained inactive scratch requires explicit reconcile clean-up"
        )
        blocking_artifacts.append("retained scratch evidence")

    staging_root = campaign / ".DATA" / "STAGING"
    staging_children = [
        p for p in (staging_root.iterdir() if staging_root.is_dir() else [])
        if p.name not in (".", "..")
    ]
    unexpected_staging_children = list(staging_children)
    valid_live_bootstrap_handoff = (
        isinstance(bootstrap_handoff, dict)
        and not bool(bootstrap_handoff.get("archived"))
    )
    if initial_handoff_indicated and not tv and not mv and valid_live_bootstrap_handoff:
        unexpected_staging_children = [
            p for p in staging_children
            if p.name != "initial"
        ]
    scripts_root = campaign / ".DATA" / "SCRIPTS"
    script_inventory = _scripts_inventory(campaign)
    # Persistent attempt bundles under SCRIPTS/JOBS are immutable operational
    # evidence, not stale re-entry artefacts. Only legacy flat scripts retain
    # the old reconcile-cleanable meaning.
    script_files = [
        p for p in (scripts_root.glob("*.sh") if scripts_root.is_dir() else [])
        if p.is_file()
    ]
    if script_inventory.get("error"):
        unsafe_reasons.append(
            "submission script inventory failed: "
            + str(script_inventory.get("error"))
        )
        blocking_artifacts.append("submission script inventory")
    if bool(script_inventory.get("has_symlink")):
        unsafe_reasons.append(
            "submission script hierarchy contains symlinks: "
            + ", ".join(script_inventory.get("symlink_entries") or [])
        )
        blocking_artifacts.append("symlinked submission script evidence")
    if script_inventory.get("invalid_bundle_entries"):
        unsafe_reasons.append(
            "submission attempt bundle hierarchy is malformed: "
            + ", ".join(script_inventory.get("invalid_bundle_entries") or [])
        )
        blocking_artifacts.append("malformed submission attempt bundle")
    dangling_reference_data = (
        ReferenceDataVersioning(
            training_dir,
            prefix=iteration_prefix,
        ).list_dangling_staging()
        if training_dir.is_dir() else []
    )
    dangling_models = (
        TrainedModelVersioning(
            models_dir,
            prefix=iteration_prefix,
        ).list_dangling_staging()
        if models_dir.is_dir() else []
    )
    model_iteration_staging = models_dir / "iteration-staging"
    has_model_iteration_staging = model_iteration_staging.is_dir()
    recoverable_ferebus_staging = False
    recoverable_ferebus_reason = ""
    if has_model_iteration_staging and not model_iteration_staging.is_symlink():
        try:
            from .ferebus_quality import validate_ferebus_quality_evidence
            from .live_executor import validate_ferebus_completed

            staging_ok, staging_reason = validate_ferebus_completed(
                model_iteration_staging
            )
            if not staging_ok:
                raise ValueError(staging_reason)
            quality = validate_ferebus_quality_evidence(model_iteration_staging)
            recoverable_ferebus_staging = True
            recoverable_ferebus_reason = (
                "complete FEREBUS staging for reference-data version "
                + str(int(quality.get("reference_data_version", -1)))
            )
        except Exception as exc:
            recoverable_ferebus_reason = type(exc).__name__ + ": " + str(exc)[:180]

    if active_intents:
        unsafe_reasons.append(
            "active submission intent(s) present: "
            + ", ".join(
                str(i.get("phase")) + "@" + str(i.get("iteration"))
                + " job_id=" + str(i.get("job_id"))
                + " expected_job_name=" + str(i.get("expected_job_name"))
                for i in active_intents
            )
        )
        blocking_artifacts.append("active submission intent(s)")
        recommended_actions.append(
            "If these jobs should be cancelled, run: ichor-al-daemon stop --campaign-dir "
            + str(campaign)
            + " --cancel-jobs"
        )
    if existing is not None:
        covered_pending = {
            (
                str(intent.get("phase") or ""),
                str(intent.get("job_id") or ""),
            )
            for intent in active_intents
        }
        uncovered_pending = []
        for phase_name, job_id in existing.pending_jobs.items():
            if job_id is None:
                continue
            job_text = str(job_id)
            if not job_text:
                continue
            key = (str(phase_name), job_text)
            if key in covered_pending:
                continue
            uncovered_pending.append(str(phase_name) + " job_id=" + job_text)
        if uncovered_pending:
            unsafe_reasons.append(
                "pending job(s) without active submission intent: "
                + ", ".join(uncovered_pending)
            )
            blocking_artifacts.append("pending_jobs")
            recommended_actions.append(
                "Inspect pending_jobs in state.json and Slurm before applying recovery."
            )
    if script_files:
        unsafe_reasons.append(".DATA/SCRIPTS contains sbatch scripts")
        trusted_artifacts.append(".DATA/SCRIPTS can be archived by reconcile --apply")
    if dangling_reference_data:
        unsafe_reasons.append("dangling reference-data staging directories exist")
        blocking_artifacts.append("dangling reference-data staging")
    if recoverable_ferebus_staging:
        trusted_artifacts.append(recoverable_ferebus_reason)
        notes.append(
            "complete FEREBUS staging is protected for quality-policy "
            "reevaluation or idempotent commit"
        )
    if dangling_models or (has_model_iteration_staging and not recoverable_ferebus_staging):
        unsafe_reasons.append("dangling model staging directories exist")
        blocking_artifacts.append("dangling model staging")
        if has_model_iteration_staging and recoverable_ferebus_reason:
            notes.append(
                "FEREBUS iteration-staging is not recoverable: "
                + recoverable_ferebus_reason
            )

    valid_reference_data_versions: List[int] = []
    for version in tv:
        try:
            verify_committed_reference_data_version(
                campaign,
                int(version),
                reference_data_dir_name=reference_data_dir_name,
            )
            valid_reference_data_versions.append(int(version))
            trusted_artifacts.append("reference-data version " + str(version))
        except Exception as exc:
            unsafe_reasons.append(
                "committed reference-data version "
                + str(version)
                + " manifest invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)[:160]
            )
            blocking_artifacts.append("reference-data version " + str(version))
    valid_model_versions: List[int] = []
    for version in mv:
        try:
            verify_committed_model_version(
                campaign,
                int(version),
                models_dir_name=models_dir_name,
            )
            valid_model_versions.append(int(version))
            trusted_artifacts.append("model version " + str(version))
        except Exception as exc:
            unsafe_reasons.append(
                "committed model version "
                + str(version)
                + " manifest invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)[:160]
            )
            blocking_artifacts.append("model version " + str(version))
    if tv != valid_reference_data_versions:
        notes.append(
            "valid reference-data versions differ from discovered committed versions"
        )
    if mv != valid_model_versions:
        notes.append(
            "valid model versions differ from discovered committed versions"
        )

    trusted_uid_sources: List[Dict[str, str]] = []
    try:
        trusted_uid_sources = _trusted_campaign_uid_sources(
            campaign,
            reference_data_dir_name=reference_data_dir_name,
            models_dir_name=models_dir_name,
            valid_reference_data_versions=valid_reference_data_versions,
            valid_model_versions=valid_model_versions,
            bootstrap_handoff=bootstrap_handoff,
            phase_a_handoff=phase_a_handoff,
        )
    except Exception as exc:
        unsafe_reasons.append(
            "campaign identity inventory failed: "
            + type(exc).__name__
            + ": "
            + str(exc)[:180]
        )
        blocking_artifacts.append("campaign identity")
    trusted_uids = sorted(
        {str(item["campaign_uid"]) for item in trusted_uid_sources}
    )
    state_uid = (
        str(existing.campaign_uid)
        if existing is not None
        else (str(salvaged_uid) if salvaged_uid else "")
    )
    recovered_uid: Optional[str] = None
    if len(trusted_uids) > 1:
        unsafe_reasons.append(
            "trusted campaign identity disagreement: "
            + ", ".join(trusted_uids)
        )
        blocking_artifacts.append("campaign identity disagreement")
    elif trusted_uids:
        recovered_uid = trusted_uids[0]
        if state_uid and state_uid != recovered_uid:
            unsafe_reasons.append(
                "state campaign_uid disagrees with trusted artefacts: state="
                + state_uid
                + " trusted="
                + recovered_uid
            )
            blocking_artifacts.append("campaign identity disagreement")
        else:
            notes.append(
                "campaign_uid recovered from "
                + str(len(trusted_uid_sources))
                + " agreeing trusted artefact(s)"
            )
            trusted_artifacts.extend(
                str(item["source"])
                + " campaign_uid="
                + str(item["campaign_uid"])
                for item in trusted_uid_sources
            )
    elif state_uid:
        recovered_uid = state_uid
        notes.append("campaign_uid retained from state because no artefact identity was available")
    elif stateful_campaign_artifacts(campaign):
        unsafe_reasons.append(
            "non-empty campaign has no recoverable trusted campaign_uid; refusing "
            "to mint a replacement identity"
        )
        blocking_artifacts.append("campaign identity unavailable")

    if _needs_trajectory_pool_check(
        existing_loaded=existing_loaded,
        reference_data_versions=tv,
        model_versions=mv,
        active_intents=active_intents,
        last_phase=last_phase or (
            str(bootstrap_handoff.get("phase"))
            if isinstance(bootstrap_handoff, dict)
            else (
                str(phase_a_handoff.get("phase"))
                if isinstance(phase_a_handoff, dict)
                else None
            )
        ),
        staging_children=staging_children,
        script_files=script_files,
        dangling_reference_data=dangling_reference_data,
        dangling_models=dangling_models,
        has_model_iteration_staging=has_model_iteration_staging,
    ):
        try:
            pool = TrajectoryPool.load(campaign)
            if pool.manifest.natoms <= 0:
                raise ValueError("pool natoms must be positive")
            trusted_artifacts.append(
                "trajectory pool SHA " + str(pool.sha256)[:12]
            )
        except Exception as exc:
            reason = _trajectory_pool_unsafe_reason(exc)
            unsafe_reasons.append(reason + ": " + str(exc)[:180])
            blocking_artifacts.append("trajectory pool")

    #build the recovered state
    if existing is not None:
        #Remeber: In v1, these
        #were silently dropped on reconcile, wiping the alpha-trend
        # history (forcing STOP_CHECK to rebuild) and the cached
       # reference scales (forcing recompute regardless of refresh policy).
        recovered = CampaignState(
            iteration=existing.iteration,
            max_iterations=existing.max_iterations,
            phase=existing.phase,
            pending_jobs={},
            reference_data_version=existing.reference_data_version,
            validation_set_version=existing.validation_set_version,
            models_version=existing.models_version,
            last_acquisition_alpha0=existing.last_acquisition_alpha0,
            stop_streak=existing.stop_streak,
            shutdown_requested=False,
            # cached GP reference scales (so we don't recompute on resume):
            reference_scales=existing.reference_scales,
            reference_scales_iteration=existing.reference_scales_iteration,
            # alpha trend across iterations (drives the stop check):
            alpha_history=list(existing.alpha_history),
            # anti-overlap diagnostic + sacct stale-job streak counters:
            last_n_anti_overlap_flagged=existing.last_n_anti_overlap_flagged,
            sacct_empty_streak=dict(existing.sacct_empty_streak),
            lifecycle_context=(
                copy.deepcopy(existing.lifecycle_context)
                if isinstance(existing.lifecycle_context, dict)
                else None
            ),
            last_completion_receipt=(
                dict(existing.last_completion_receipt)
                if isinstance(existing.last_completion_receipt, dict)
                else None
            ),
            campaign_uid=existing.campaign_uid,
            campaign_started_iso=existing.campaign_started_iso,
        )
    else:
        recovered = fresh_campaign_state()
        # a corrupt-but-readable state.json still carries its identity; keep it
        # so recovery does not silently start a brand-new campaign.
        if salvaged_uid:
            recovered.campaign_uid = str(salvaged_uid)
            if salvaged_started:
                recovered.campaign_started_iso = str(salvaged_started)
    if recovered_uid is not None:
        recovered.campaign_uid = str(recovered_uid)

    coherent_pairs = sorted(set(valid_reference_data_versions).intersection(valid_model_versions))
    latest_reference_data_only = max(valid_reference_data_versions) if valid_reference_data_versions else None
    latest_model_only = max(valid_model_versions) if valid_model_versions else None
    reference_data_model_skew_reentry = False
    no_coherent_pair = False
    initial_handoff_error: Optional[str] = None
    initial_handoff_valid = False
    if not tv and not mv and initial_handoff_indicated:
        if bootstrap_handoff is not None:
            initial_handoff_valid = True
            source = "archived" if bootstrap_handoff.get("archived") else "live"
            trusted_artifacts.append(
                source
                + " bootstrap handoff "
                + str(bootstrap_handoff.get("phase"))
                + " at "
                + str(bootstrap_handoff.get("path"))
            )
            if str(bootstrap_handoff.get("phase")) == CampaignPhase.INITIAL_GAUSSIAN.value:
                trusted_artifacts.append("initial Gaussian acceptance manifest")
            elif str(bootstrap_handoff.get("phase")) == CampaignPhase.INITIAL_AIMALL.value:
                trusted_artifacts.append("initial AIMAll acceptance manifest")
            if bool(bootstrap_handoff.get("archived")):
                notes.append(
                    "archived bootstrap handoff found at "
                    + str(bootstrap_handoff.get("path"))
                )
        else:
            try:
                _validate_initial_ferebus_bootstrap(
                    campaign,
                    iteration=int(getattr(recovered, "iteration", 0)),
                )
                initial_handoff_valid = True
                trusted_artifacts.append("initial AIMAll acceptance manifest")
            except Exception as exc:
                initial_handoff_error = type(exc).__name__ + ": " + str(exc)[:180]
    phase_a_handoff_valid = not tv and not mv and isinstance(phase_a_handoff, dict)
    if phase_a_handoff_valid:
        trusted_artifacts.append(
            "Phase A sample handoff at " + str(phase_a_handoff.get("path"))
        )

    if coherent_pairs:
        coherent = int(coherent_pairs[-1])
        if (
            latest_reference_data_only is not None
            and latest_model_only is not None
            and int(latest_reference_data_only) == coherent + 1
            and int(latest_model_only) == coherent
        ):
            reference_data_model_skew_reentry = True
        target_reference_data = int(latest_reference_data_only) if reference_data_model_skew_reentry else coherent
        target_model = int(latest_model_only) if reference_data_model_skew_reentry else coherent
        if recovered.reference_data_version != target_reference_data:
            notes.append(
                "reference_data_version set to recovered committed version "
                + str(target_reference_data)
            )
        if recovered.models_version != target_model:
            notes.append(
                "models_version set to recovered committed version "
                + str(target_model)
            )
        recovered.reference_data_version = target_reference_data
        recovered.models_version = target_model
        if not existing_loaded:
            target_iteration = int(target_reference_data)
            try:
                current_iteration = int(recovered.iteration)
            except Exception:
                current_iteration = -1
            if current_iteration != target_iteration:
                notes.append(
                    "iteration set from committed active-version mapping: "
                    + str(target_iteration)
                )
                recovered.iteration = int(target_iteration)
        if reference_data_model_skew_reentry:
            notes.append(
                "valid committed reference-data version "
                + str(target_reference_data)
                + " is one ahead of committed model version "
                + str(target_model)
                + "; re-entry can train FEREBUS"
            )
        if (
            latest_reference_data_only is not None
            and latest_reference_data_only > coherent
            and not reference_data_model_skew_reentry
        ):
            unsafe_reasons.append(
                "newer committed reference-data version has no matching model: "
                + str(latest_reference_data_only)
            )
        if (
            latest_model_only is not None
            and latest_model_only > coherent
            and not reference_data_model_skew_reentry
        ):
            unsafe_reasons.append(
                "newer committed model version has no matching QM reference data: "
                + str(latest_model_only)
            )
    elif valid_reference_data_versions and not valid_model_versions:
        recovered.reference_data_version = int(latest_reference_data_only)
        recovered.models_version = -1
        notes.append(
            "valid training exists without any valid model; re-entry must train FEREBUS"
        )
    elif valid_reference_data_versions or valid_model_versions:
        no_coherent_pair = True
        unsafe_reasons.append(
            "no coherent committed reference-data/model version pair exists "
            + "(reference_data="
            + repr(valid_reference_data_versions)
            + ", models="
            + repr(valid_model_versions)
            + ")"
        )
    else:
        if recovered.reference_data_version != -1 or recovered.models_version != -1:
            notes.append(
                "stale state version pointers reset because no committed "
                "training or model versions were verified"
            )
        recovered.reference_data_version = -1
        recovered.models_version = -1

    try:
        protected_staging_handoffs = staging_handoff_decisions(campaign, recovered)
    except Exception as exc:
        protected_staging_handoffs = []
        unsafe_reasons.append(
            "staging handoff inventory failed: "
            + type(exc).__name__
            + ": "
            + str(exc)[:180]
        )
    protected_staging_paths = set()
    for decision in protected_staging_handoffs:
        _append_recovery_candidate(recovery_candidates, decision)
        if decision.trusted_artifact:
            protected_path = (campaign / decision.trusted_artifact).resolve(strict=False)
            protected_staging_paths.add(str(protected_path))
            trusted_artifacts.append(
                "protected active staging handoff for "
                + decision.phase.value
                + " at "
                + str(decision.trusted_artifact)
            )
    if protected_staging_paths:
        unexpected_staging_children = [
            p for p in unexpected_staging_children
            if str(p.resolve(strict=False)) not in protected_staging_paths
        ]
    if len(protected_staging_handoffs) > 1:
        unsafe_reasons.append(
            "multiple valid staging handoffs need operator review: "
            + ", ".join(
                str(d.phase.value)
                + "@"
                + str(int(d.iteration))
                + " "
                + str(d.trusted_artifact)
                for d in protected_staging_handoffs
            )
        )
        blocking_artifacts.append(".DATA/STAGING")
    try:
        partial_iteration_handoffs = active_iteration_handoff_decisions(campaign, recovered)
    except Exception as exc:
        partial_iteration_handoffs = []
        unsafe_reasons.append(
            "active-iteration handoff inventory failed: "
            + type(exc).__name__
            + ": "
            + str(exc)[:180]
        )
    if len(partial_iteration_handoffs) > 1:
        unsafe_reasons.append(
            "multiple valid active-iteration handoffs need operator review: "
            + ", ".join(
                str(d.phase.value)
                + "@"
                + str(int(d.iteration))
                + " "
                + str(d.trusted_artifact)
                for d in partial_iteration_handoffs
            )
        )
        blocking_artifacts.append(ACTIVE_LEARNING_DIRNAME)
    for decision in partial_iteration_handoffs:
        _append_recovery_candidate(recovery_candidates, decision)
    combined_handoffs = list(protected_staging_handoffs) + list(partial_iteration_handoffs)
    combined_iterations = {int(d.iteration) for d in combined_handoffs}
    if len(combined_iterations) > 1:
        unsafe_reasons.append(
            "valid handoffs exist in multiple iterations: "
            + ", ".join(
                str(d.phase.value)
                + "@"
                + str(int(d.iteration))
                + " "
                + str(d.trusted_artifact)
                for d in combined_handoffs
            )
        )
        blocking_artifacts.append("active-learning handoff inventory")

    partial_array_recovery: Optional[Dict[str, Any]] = None
    partial_array_decision: Optional[RecoveryDecision] = None
    preferred_phase = last_phase
    preferred_iteration = last_iter
    if existing is not None and supports_partial_array_recovery(existing.phase):
        preferred_phase = existing.phase.value
        preferred_iteration = int(existing.iteration)
    if preferred_phase is not None and supports_partial_array_recovery(preferred_phase):
        try:
            partial_array_recovery = discover_partial_array_recovery(
                campaign,
                preferred_phase=preferred_phase,
                preferred_iteration=(
                    int(preferred_iteration)
                    if preferred_iteration is not None
                    else int(getattr(recovered, "iteration", 0))
                ),
            )
        except Exception as exc:
            notes.append(
                "partial array recovery scan failed: "
                + type(exc).__name__
                + ": "
                + str(exc)[:180]
            )
            partial_array_recovery = None
    if isinstance(partial_array_recovery, dict):
        try:
            partial_phase = CampaignPhase(str(partial_array_recovery["phase"]))
            partial_iteration = int(partial_array_recovery["iteration"])
            partial_array_decision = RecoveryDecision(
                phase=partial_phase,
                iteration=partial_iteration,
                reason=(
                    "partial array recovery available: "
                    + str(int(partial_array_recovery.get("n_complete") or 0))
                    + "/"
                    + str(int(partial_array_recovery.get("logical_total") or 0))
                    + " logical tasks already complete"
                ),
                trusted_artifact=str(partial_array_recovery.get("path") or ""),
            )
            _append_recovery_candidate(recovery_candidates, partial_array_decision)
            trusted_artifacts.append(
                "partial array recovery ledger for "
                + partial_phase.value
                + "@"
                + str(partial_iteration)
            )
            bucket = "initial" if partial_phase.value.startswith("INITIAL_") else (
                "iter_" + str(partial_iteration)
            )
            partial_staging = (campaign / ".DATA" / "STAGING" / bucket).resolve(strict=False)
            unexpected_staging_children = [
                p
                for p in unexpected_staging_children
                if str(p.resolve(strict=False)) != str(partial_staging)
            ]
        except Exception as exc:
            notes.append(
                "partial array recovery candidate was ignored: "
                + type(exc).__name__
                + ": "
                + str(exc)[:180]
            )
            partial_array_recovery = None
            partial_array_decision = None
    if unexpected_staging_children:
        unsafe_reasons.append(".DATA/STAGING is non-empty")
        blocking_artifacts.append(".DATA/STAGING")

    cleanup_only_blockers = {
        ".DATA/STAGING",
        "dangling reference-data staging",
        "dangling model staging",
    }
    existing_contract_valid = False
    if existing is not None and (
        existing.phase is CampaignPhase.DONE or existing.shutdown_requested
    ):
        try:
            verify_state_referenced_artifacts(campaign, existing)
        except Exception as exc:
            reason = (
                "existing state artefact contract invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)[:180]
            )
            if reason not in unsafe_reasons:
                unsafe_reasons.append(reason)
            if "existing state artefact contract" not in blocking_artifacts:
                blocking_artifacts.append("existing state artefact contract")
        else:
            existing_contract_valid = True
    non_cleanup_blockers = [
        blocker
        for blocker in blocking_artifacts
        if blocker not in cleanup_only_blockers
    ]
    preserve_completed_state = bool(
        existing is not None
        and existing.phase is CampaignPhase.DONE
        and not existing.shutdown_requested
        and existing_contract_valid
        and not non_cleanup_blockers
        and not active_intents
    )
    preserve_stopped_state = bool(
        existing is not None
        and existing.shutdown_requested
        and existing_contract_valid
        and not non_cleanup_blockers
        and not active_intents
    )
    identity_recovery_blocked = any(
        str(blocker).startswith("campaign identity")
        for blocker in blocking_artifacts
    )

    phase_recovery = None
    if not active_intents and not unsafe_reasons:
        handoff_recovery = None
        if combined_handoffs:
            handoff_recovery = max(
                combined_handoffs,
                key=lambda decision: _RECOVERY_PHASE_PROGRESS.get(
                    decision.phase,
                    -1,
                ),
            )
        if (
            partial_array_decision is not None
            and (
                handoff_recovery is None
                or _RECOVERY_PHASE_PROGRESS.get(partial_array_decision.phase, -1)
                >= _RECOVERY_PHASE_PROGRESS.get(handoff_recovery.phase, -1)
            )
        ):
            phase_recovery = partial_array_decision
        elif handoff_recovery is not None:
            phase_recovery = handoff_recovery
        else:
            phase_recovery = select_recovery_phase(
                campaign,
                recovered,
                valid_reference_data_versions=valid_reference_data_versions,
                valid_model_versions=valid_model_versions,
                existing_loaded=existing_loaded,
                last_phase=last_phase,
                last_iteration=last_iter,
                last_phase_retryable=last_phase_retryable,
            )
        _append_recovery_candidate(recovery_candidates, phase_recovery)

    # choose a safe re-entry phase. If we have NOTHING committed, start at
    #  INIT; otherwise rewind to STOP_CHECK so the next tick decides whether
    # to loop or terminate.
    if preserve_completed_state:
        recovered.phase = CampaignPhase.DONE
        recovered.iteration = int(existing.iteration)
        decision = "DONE: existing completed lifecycle and artefact chain are trusted"
        notes.append(
            "preserved completed campaign; only resume --reopen-converged may reopen it"
        )
    elif preserve_stopped_state:
        recovered.phase = CampaignPhase(existing.phase)
        recovered.iteration = int(existing.iteration)
        decision = "STOPPED: existing operator stop request remains authoritative"
        notes.append(
            "preserved operator stop request; only resume may clear it"
        )
    elif identity_recovery_blocked:
        recovered.phase = CampaignPhase.HALTED
        decision = "HALTED: campaign identity cannot be recovered unambiguously"
        notes.append(
            "re-entry HALTED because the non-empty campaign has identity "
            "evidence that is missing, conflicting, or unreadable"
        )
    elif not tv and not mv and not existing_loaded and unsafe_reasons and not allow_fresh_init_on_nonempty:
        recovered.phase = CampaignPhase.HALTED
        decision = "HALTED: non-empty campaign has no valid state or committed versions"
        notes.append(
            "non-empty campaign with no valid state.json; proposed HALTED instead of fresh INIT"
        )
        for reason in unsafe_reasons:
            notes.append("unsafe recovery reason: " + reason)
        if active_intents:
            latest = active_intents[-1]
            try:
                recovered.phase = CampaignPhase(str(latest.get("phase")))
                recovered.iteration = int(latest.get("iteration"))
                notes.append(
                    "active submission intent selected re-entry phase "
                    + recovered.phase.value
                    + " iteration "
                    + str(recovered.iteration)
                )
                decision = (
                    recovered.phase.value
                    + ": selected from active submission intent for operator review"
                )
            except Exception:
                recovered.phase = CampaignPhase.HALTED
    elif phase_recovery is not None:
        recovered.phase = phase_recovery.phase
        recovered.iteration = int(phase_recovery.iteration)
        recovered.replacement_round = int(
            getattr(phase_recovery, "replacement_round", 0)
        )
        decision = phase_recovery.reason
        notes.append("phase-aware recovery selected " + recovered.phase.value)
        if phase_recovery.trusted_artifact:
            trusted_artifacts.append(str(phase_recovery.trusted_artifact))
        if not valid_reference_data_versions and not valid_model_versions:
            recovered.reference_data_version = -1
            recovered.validation_set_version = -1
            recovered.models_version = -1
    elif not tv and not mv and initial_handoff_indicated and initial_handoff_valid and not unsafe_reasons:
        handoff_phase = (
            str(bootstrap_handoff.get("phase"))
            if isinstance(bootstrap_handoff, dict)
            else CampaignPhase.INITIAL_AIMALL.value
        )
        if handoff_phase == CampaignPhase.INITIAL_GAUSSIAN.value:
            recovered.phase = CampaignPhase.INITIAL_AIMALL
            recovered.iteration = 0
            decision = "INITIAL_AIMALL: valid initial Gaussian handoff exists without committed models"
            notes.append(
                "re-entry at INITIAL_AIMALL to process the initial Gaussian handoff"
            )
        else:
            recovered.phase = CampaignPhase.INITIAL_FEREBUS
            recovered.iteration = 0
            decision = "INITIAL_FEREBUS: valid initial AIMAll handoff exists without committed models"
            notes.append(
                "re-entry at INITIAL_FEREBUS to commit initial reference-data/model version 0"
            )
        recovered.reference_data_version = -1
        recovered.validation_set_version = -1
        recovered.models_version = -1
    elif not tv and not mv and initial_handoff_indicated:
        recovered.phase = CampaignPhase.HALTED
        recovered.reference_data_version = -1
        recovered.validation_set_version = -1
        recovered.models_version = -1
        if initial_handoff_valid:
            decision = "HALTED: valid initial AIMAll handoff exists but unsafe artefacts need review"
            notes.append(
                "re-entry HALTED because unsafe artefacts block INITIAL_FEREBUS recovery"
            )
        else:
            decision = "HALTED: INITIAL_AIMALL completed but initial FEREBUS handoff is missing"
            notes.append(
                "re-entry HALTED because no committed reference-data/model versions or valid initial AIMAll handoff exist"
            )
        if initial_handoff_error:
            unsafe_reasons.append(
                "initial AIMAll handoff invalid or missing: "
                + initial_handoff_error
            )
            blocking_artifacts.append(".DATA/STAGING/initial")
    elif (
        not tv
        and not mv
        and not existing_loaded
        and phase_a_handoff_valid
        and not unsafe_reasons
    ):
        recovered.phase = CampaignPhase.INITIAL_GAUSSIAN
        recovered.reference_data_version = -1
        recovered.validation_set_version = -1
        recovered.models_version = -1
        decision = "INITIAL_GAUSSIAN: valid Phase A sample exists without committed models"
        notes.append("re-entry at INITIAL_GAUSSIAN to process the Phase A sample")
    elif not tv and not mv and not existing_loaded:
        recovered.phase = CampaignPhase.INIT
        decision = "INIT: no existing state or committed iterations"
        notes.append("no committed iterations; re-entry at INIT")
    elif not tv and not mv:
        recovered.phase = CampaignPhase.HALTED
        recovered.reference_data_version = -1
        recovered.validation_set_version = -1
        recovered.models_version = -1
        decision = "HALTED: existing state has no committed versions or recoverable initial handoff"
        notes.append(
            "re-entry HALTED because existing state has no committed reference-data/model versions"
        )
    elif no_coherent_pair:
        recovered.phase = CampaignPhase.HALTED
        decision = "HALTED: no coherent committed reference-data/model pair"
        notes.append("re-entry HALTED because no coherent reference-data/model pair exists")
    elif valid_reference_data_versions and mv and not valid_model_versions:
        recovered.phase = CampaignPhase.HALTED
        decision = "HALTED: committed model artefacts are present but invalid"
        notes.append(
            "re-entry HALTED because committed model artefacts failed validation"
        )
    elif valid_reference_data_versions and not valid_model_versions:
        recovered.phase = (
            CampaignPhase.INITIAL_FEREBUS
            if recovered.reference_data_version == 0
            else CampaignPhase.FEREBUS
        )
        if recovered.phase is CampaignPhase.INITIAL_FEREBUS:
            # Bootstrap phases are always iteration zero, even when the stale
            # state was halted after an operator increased max_iterations.
            recovered.iteration = 0
        decision = (
            "INITIAL_FEREBUS: exact point allocation is complete and valid "
            "bootstrap training exists without a committed model"
            if recovered.phase is CampaignPhase.INITIAL_FEREBUS
            else "FEREBUS: valid training exists without a committed model"
        )
        notes.append(
            "re-entry at "
            + recovered.phase.value
            + " to produce model version "
            + str(recovered.reference_data_version)
        )
    elif coherent_pairs and unsafe_reasons:
        recovered.phase = CampaignPhase.HALTED
        decision = "HALTED: coherent committed versions exist but unsafe artefacts need review"
        notes.append(
            "re-entry HALTED because committed artefacts need operator review"
        )
    elif reference_data_model_skew_reentry:
        recovered.phase = CampaignPhase.FEREBUS
        decision = "FEREBUS: committed training is one version ahead of committed models"
        notes.append(
            "re-entry at FEREBUS to produce model version "
            + str(recovered.reference_data_version)
        )
    elif unsafe_reasons:
        recovered.phase = CampaignPhase.HALTED
        decision = "HALTED: unsafe artefacts need operator review"
        notes.append(
            "re-entry HALTED because unsafe committed artefacts need operator review"
        )
    else:
        recovered.phase = CampaignPhase.STOP_CHECK
        decision = "STOP_CHECK: latest coherent committed reference-data/model pair is trusted"
        notes.append("re-entry at STOP_CHECK (next tick decides loop/terminate)")
    recovered.pending_jobs = {}
    recovered.shutdown_requested = bool(preserve_stopped_state)
    if recovered.phase is not CampaignPhase.HALTED and not active_intents:
        try:
            _validate_recovered_state_contract(
                campaign,
                recovered,
                bootstrap_handoff=bootstrap_handoff,
                phase_a_handoff=phase_a_handoff,
            )
        except Exception as exc:
            recovered.phase = CampaignPhase.HALTED
            reason = (
                "proposed recovery failed final contract validation: "
                + type(exc).__name__
                + ": "
                + str(exc)[:180]
            )
            unsafe_reasons.append(reason)
            blocking_artifacts.append("proposed state")
            decision = "HALTED: proposed recovery failed final contract validation"
            notes.append("re-entry HALTED because proposed state failed final contract validation")
    if recovered.phase is CampaignPhase.HALTED and not recommended_actions:
        recommended_actions.append("Inspect unsafe recovery reasons before applying.")
    if recovered.phase is CampaignPhase.HALTED:
        origin_phase = (
            existing.phase
            if existing is not None
            else CampaignPhase.HALTED
        )
        recovered.lifecycle_context = make_lifecycle_context(
            disposition="halted",
            reason_code="reconcile_unsafe",
            message=(
                str(decision)
                + (
                    ": " + "; ".join(str(reason) for reason in unsafe_reasons[:3])
                    if unsafe_reasons
                    else ""
                )
            )[:500],
            from_phase=origin_phase,
            iteration=int(recovered.iteration),
            source="reconcile",
            recovery_action=(
                str(recommended_actions[0])
                if recommended_actions
                else "inspect unsafe recovery reasons"
            ),
            details={"unsafe_reason_count": len(unsafe_reasons)},
        )
    elif preserve_completed_state:
        if not (
            isinstance(recovered.lifecycle_context, dict)
            and recovered.lifecycle_context.get("disposition") == "completed"
        ):
            recovered.lifecycle_context = make_lifecycle_context(
                disposition="completed",
                reason_code="recovered_completed_state",
                message="completed campaign lifecycle preserved by reconcile",
                from_phase=CampaignPhase.DONE,
                iteration=int(recovered.iteration),
                source="reconcile",
                recovery_action=(
                    "use resume --reopen-converged only after deliberately "
                    "increasing max_iterations"
                ),
            )
    elif preserve_stopped_state:
        if not (
            isinstance(recovered.lifecycle_context, dict)
            and recovered.lifecycle_context.get("disposition") == "stopped"
        ):
            recovered.lifecycle_context = make_lifecycle_context(
                disposition="stopped",
                reason_code="recovered_stop_request",
                message="existing campaign stop request preserved by reconcile",
                from_phase=recovered.phase,
                iteration=int(recovered.iteration),
                source="reconcile",
                recovery_action="use resume to continue from the recorded phase",
            )
    else:
        recovered.lifecycle_context = None

    return ReconciliationReport(
        proposed_state=recovered,
        committed_reference_data_versions=tv,
        committed_model_versions=mv,
        valid_reference_data_versions=valid_reference_data_versions,
        valid_model_versions=valid_model_versions,
        last_phase_in_journal=last_phase,
        last_iteration_in_journal=last_iter,
        last_phase_event_in_journal=last_phase_event,
        last_phase_retryable=last_phase_retryable,
        last_halt_event=last_halt_event,
        script_inventory=script_inventory,
        scratch_inventory=scratch_inventory,
        reconcile_transactions=reconcile_transactions,
        notes=notes,
        existing_state_loaded=existing_loaded,
        unsafe_reasons=unsafe_reasons,
        active_submission_intents=active_intents,
        decision=decision,
        trusted_artifacts=trusted_artifacts,
        blocking_artifacts=blocking_artifacts,
        recommended_actions=recommended_actions,
        recovery_candidates=recovery_candidates,
        partial_array_recovery=(
            compact_array_recovery_summary(partial_array_recovery)
            if isinstance(partial_array_recovery, dict)
            else None
        ),
        bootstrap_handoff=bootstrap_handoff,
        phase_a_handoff=phase_a_handoff,
    )



def write_proposed_state(
    campaign_dir: Union[str, Path],
    report: ReconciliationReport,
    *,
    data_subdir: Union[str, Path] = Path(".DATA") / "ACTIVE_LEARNING",
) -> Path:
    """Write the proposed state to <state_path>.proposed and return the
    path. The operator promotes it manually via mv state.json.proposed
    state.json after reviewing."""
    target = (
        Path(campaign_dir) / data_subdir / (DEFAULT_STATE_FILENAME + RECONCILE_SUFFIX)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    write_state(target, report.proposed_state)
    return target



