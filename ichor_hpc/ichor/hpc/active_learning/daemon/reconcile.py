"""Recovery for a corrupt or missing state.json with user in the loop.

The daemon's design treats 'state.json' as the only authoritative checkpoint.
If it goes missing or fails schema validation, the daemon refuses to auto-
recover; a silent reconstruction is exactly the bug class we want to avoid.

"reconcile" inspects the on-disk artefacts that DO exist (QM_REFERENCE_DATA/
committed iterations, TRAINED_MODELS/, journal entries) and proposes a
"CampaignState" it believes is consistent with them. The proposal is
written to "<state_path>.proposed" and the user must explicitly
promote it ("mv state.json.proposed state.json") before restarting the
daemon.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from ..strict_json import strict_json as json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..acquisition.trajectory_pool import (
    POOL_MANIFEST_FILENAME,
    POOL_SUBDIR,
    POOL_XYZ_FILENAME,
    TrajectoryPoolManifest,
)
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
    verify_state_referenced_artifacts,
)
from .artifact_snapshot import (
    CommittedArtifactSnapshot,
    build_committed_artifact_snapshot,
)
from .journal import tail_events
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
from .ariadne_publication import classify_ariadne_publication
from .state import (
    CampaignPhase,
    CampaignState,
    DEFAULT_STATE_FILENAME,
    StateSchemaError,
    _fsync_parent_dir,
    fresh_campaign_state,
    make_lifecycle_context,
    read_state,
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
        CampaignPhase.REFERENCE_COMMIT,
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
        for path in [staging] + children:
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
        "user_stop_requested",
        "user_stop_boundary_reached",
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
    add_matches(".DATA/ACTIVE_LEARNING/reference_commit_transactions/*.json")
    add_matches(".DATA/ACTIVE_LEARNING/execution_identity.json")
    add_matches(".DATA/ACTIVE_LEARNING/environment_current.json")
    add_matches(".DATA/ACTIVE_LEARNING/environment_generations/*.json")
    config_lock = campaign / ".DATA" / "ACTIVE_LEARNING" / "config_lock.json"
    pool_manifest = campaign / ".DATA" / "TRAJECTORY" / "pool.manifest.json"
    if config_lock.is_file() and not pool_manifest.is_file():
        findings.append(".DATA/ACTIVE_LEARNING/config_lock.json")
    journal_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    if journal_path.is_file():
        try:
            for event in tail_events(journal_path):
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
    if scripts.is_dir() and any(scripts.iterdir()):
        findings.append(".DATA/SCRIPTS")
    scratch = campaign / ".DATA" / "SCRATCH"
    if scratch.is_dir() and any(scratch.iterdir()):
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
    reference_commit_transactions: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    existing_state_loaded: bool = False
    unsafe_reasons: List[str] = field(default_factory=list)
    active_submission_intents: List[Dict[str, Any]] = field(default_factory=list)
    receipt_backed_intent_repairs: List[Dict[str, Any]] = field(default_factory=list)
    decision: str = ""
    trusted_artifacts: List[str] = field(default_factory=list)
    blocking_artifacts: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    recovery_candidates: List[Dict[str, Any]] = field(default_factory=list)
    partial_array_recovery: Optional[Dict[str, Any]] = None
    ariadne_publication_recovery: Optional[Dict[str, Any]] = None
    ferebus_candidate_recovery: Optional[Dict[str, Any]] = None
    bootstrap_handoff: Optional[Dict[str, Any]] = None
    phase_a_handoff: Optional[Dict[str, Any]] = None
    artifact_snapshot: Optional[CommittedArtifactSnapshot] = field(
        default=None,
        repr=False,
    )
    deep_verification_required: bool = False


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
    if "SHA mismatch" in msg or (
        isinstance(exc, RuntimeError) and "pool drift detected" in msg
    ):
        return "trajectory pool SHA mismatch"
    if isinstance(exc, ValueError):
        if "natoms" in msg or "atom_types" in msg or "masses" in msg:
            return "trajectory pool atom count invalid"
        return "trajectory pool unreadable"
    return "trajectory pool unreadable"


def _scripts_inventory(
    campaign: Path,
    *,
    intent_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
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
        if scripts.is_symlink():
            payload["has_symlink"] = True
            payload["symlink_entries"] = [str(scripts.relative_to(campaign))]
            return payload
        legacy_scripts = [path for path in scripts.glob("*.sh") if path.is_file()]
        payload["legacy_flat_script_count"] = len(legacy_scripts)
        payload["count"] = len(legacy_scripts)
        payload["sample"] = [
            str(path.relative_to(campaign)) for path in legacy_scripts[:5]
        ]
        bundles = []
        records: Sequence[Mapping[str, Any]]
        if intent_records is None:
            intent_inventory = _submission_intent.inventory_intents(campaign)
            records = tuple(intent_inventory.get("records", []))
        else:
            records = tuple(intent_records)
        for intent_payload in records:
            if not isinstance(intent_payload, Mapping):
                continue
            identity = str(intent_payload.get("submission_identity") or "")
            script_text = str(intent_payload.get("submitted_script_path") or "")
            if not identity or not script_text:
                continue
            try:
                script_path = Path(script_text)
                if not script_path.is_absolute():
                    script_path = campaign / script_path
                script_path = campaign_owned_path(campaign, script_path)
                script_relative = script_path.relative_to(campaign.absolute())
            except (OSError, ValueError):
                payload["invalid_bundle_entries"].append(script_text)
                continue
            parts = script_relative.parts
            if (
                len(parts) != 8
                or tuple(parts[:3]) != (".DATA", "SCRIPTS", "JOBS")
                or parts[-1] != "job.sh"
            ):
                payload["invalid_bundle_entries"].append(script_relative.as_posix())
                continue
            backend, phase, iteration_token, bundle_identity = parts[3:7]
            if bundle_identity != identity:
                payload["invalid_bundle_entries"].append(script_relative.as_posix())
                continue
            bundles.append(
                {
                    "backend": backend,
                    "phase": phase,
                    "iteration": iteration_token,
                    "submission_identity": bundle_identity,
                    "path": script_relative.parent.as_posix(),
                    "intent_status": str(intent_payload.get("status") or ""),
                    "output_log_count": None,
                    "error_log_count": None,
                    "has_array_task_map": None,
                }
            )
            payload["count"] = int(payload["count"]) + 1
            if len(payload["sample"]) < 5:
                payload["sample"].append(script_relative.as_posix())
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
    verification_level: str = "authority",
) -> Optional[Dict[str, Any]]:
    from .input_staging import (
        QUANTUM_ACCEPTANCE_SCHEMA_VERSION,
        _validate_pointdir_basename,
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
    if verification_level == "authority":
        phase_path = quantum_acceptance_manifest_path(
            initial,
            phase_name=phase,
        )
        try:
            manifest = _json.loads(
                phase_path.read_text(encoding="utf-8"),
                source=phase_path,
            )
            if not isinstance(manifest, dict):
                raise ValueError("quantum acceptance manifest must be an object")
            if manifest.get("schema_version") != QUANTUM_ACCEPTANCE_SCHEMA_VERSION:
                raise ValueError("unsupported quantum acceptance manifest schema")
            if manifest.get("phase") != phase:
                raise ValueError("quantum acceptance manifest phase mismatch")
            iteration = manifest.get("iteration")
            if (
                isinstance(iteration, bool)
                or not isinstance(iteration, int)
                or iteration != int(expected_iteration)
            ):
                raise ValueError("quantum acceptance manifest iteration mismatch")
            accepted = manifest.get("accepted_pointdirs")
            rejected = manifest.get("rejected", [])
            n_total = manifest.get("n_total")
            if not isinstance(accepted, list) or not accepted:
                raise ValueError("quantum acceptance accepted list is invalid")
            if not isinstance(rejected, list):
                raise ValueError("quantum acceptance rejected list is invalid")
            if (
                isinstance(n_total, bool)
                or not isinstance(n_total, int)
                or n_total != len(accepted) + len(rejected)
            ):
                raise ValueError("quantum acceptance total is invalid")
            accepted_names = [
                _validate_pointdir_basename(value) for value in accepted
            ]
            rejected_names = []
            for record in rejected:
                if not isinstance(record, dict) or set(record) != {
                    "pointdir",
                    "reason",
                }:
                    raise ValueError("quantum rejection record is invalid")
                rejected_names.append(
                    _validate_pointdir_basename(record.get("pointdir"))
                )
                if not isinstance(record.get("reason"), str) or not str(
                    record.get("reason")
                ).strip():
                    raise ValueError("quantum rejection reason is invalid")
            dispositions = accepted_names + rejected_names
            if len(dispositions) != len(set(dispositions)):
                raise ValueError("quantum acceptance dispositions are duplicated")
        except Exception:
            return None
        return {
            "path": str(initial),
            "manifest_path": str(phase_path),
            "phase": phase,
            "iteration": int(iteration),
            "n_total": int(n_total),
            "accepted_count": len(accepted_names),
            "archived": bool(archived),
            "campaign_uid": str(manifest.get("campaign_uid") or ""),
        }
    if verification_level not in {"metadata", "deep"}:
        raise ValueError("bootstrap handoff verification level is invalid")
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
    verification_level: str = "authority",
) -> Optional[Dict[str, Any]]:
    campaign = Path(campaign_dir)
    live = _read_bootstrap_handoff_at(
        campaign / ".DATA" / "STAGING" / "initial",
        expected_iteration=int(iteration),
        archived=False,
        verification_level=verification_level,
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
            verification_level=verification_level,
        )
        if handoff is not None:
            return handoff
    return None


def _find_phase_a_handoff(
    campaign_dir: Union[str, Path],
    *,
    verification_level: str = "authority",
) -> Optional[Dict[str, Any]]:
    from ..handoff_manifests import (
        PHASE_A_SAMPLE_SCHEMA_VERSION,
        phase_a_sample_manifest_path,
        read_phase_a_sample_manifest,
    )
    from ..layout import bootstrap_selection_dir

    initial = bootstrap_selection_dir(Path(campaign_dir))
    manifest_path = phase_a_sample_manifest_path(initial)
    if verification_level == "authority":
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                source=manifest_path,
            )
            if not isinstance(manifest, dict):
                raise ValueError("Phase A sample manifest must be an object")
            if manifest.get("schema_version") != PHASE_A_SAMPLE_SCHEMA_VERSION:
                raise ValueError("unsupported Phase A sample manifest schema")
            if manifest.get("phase") != CampaignPhase.PHASE_A_DIVERSITY.value:
                raise ValueError("Phase A sample manifest phase mismatch")
            iteration = manifest.get("iteration")
            n_select = manifest.get("n_select")
            selected = manifest.get("selected_indices")
            if iteration != 0:
                raise ValueError("Phase A iteration must be zero")
            if (
                isinstance(n_select, bool)
                or not isinstance(n_select, int)
                or n_select <= 0
            ):
                raise ValueError("Phase A n_select is invalid")
            if not isinstance(selected, list) or len(selected) != n_select:
                raise ValueError("Phase A selected index count is invalid")
            from ..sampling.diversity_contract import selector_contract_matches

            if not selector_contract_matches(manifest.get("selector")):
                raise ValueError("Phase A selector contract is invalid")
        except Exception:
            return None
    elif verification_level in {"metadata", "deep"}:
        try:
            manifest = read_phase_a_sample_manifest(initial, require_nonempty=True)
        except Exception:
            return None
    else:
        raise ValueError("Phase A handoff verification level is invalid")
    return {
        "path": str(initial),
        "manifest_path": str(manifest_path),
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
    source_evidence = _read_bootstrap_handoff_at(
        source,
        expected_iteration=iteration,
        archived=True,
        verification_level="authority",
    )
    if source_evidence is None or str(source_evidence.get("phase")) != phase:
        raise ValueError("archived bootstrap authority evidence is invalid")
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
    try:
        os.replace(source, target)
        _fsync_parent_dir(source)
        _fsync_parent_dir(target)
        restored = _read_bootstrap_handoff_at(
            target,
            expected_iteration=iteration,
            archived=False,
            verification_level="authority",
        )
        if restored is None or str(restored.get("phase")) != phase:
            raise ValueError("restored bootstrap authority evidence is invalid")
    except Exception:
        if target.exists() and not source.exists():
            os.replace(target, source)
            _fsync_parent_dir(target)
            _fsync_parent_dir(source)
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
    artifact_snapshot: Optional[CommittedArtifactSnapshot] = None,
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
            view = (
                artifact_snapshot.reference_view(int(version))
                if artifact_snapshot is not None
                else reference_versions.resolve(int(version), verification="authority")
            )
        except Exception:
            continue
        add("reference-data version " + str(int(version)), view.campaign_uid)
    model_versions = TrainedModelVersioning(campaign / models_dir_name)
    for version in valid_model_versions:
        try:
            model_set = (
                artifact_snapshot.model_set(int(version))
                if artifact_snapshot is not None
                else model_versions.resolve(int(version), verification="authority")
            )
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
    verification: str = "authority",
    artifact_snapshot: Optional[CommittedArtifactSnapshot] = None,
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
    validate_phase_recovery_contract(
        campaign_dir,
        state,
        verification=verification,
        artifact_snapshot=artifact_snapshot,
    )
    verify_state_referenced_artifacts(
        campaign_dir,
        state,
        verification=verification,
        snapshot=artifact_snapshot,
    )


def propose_recovery(
    campaign_dir: Union[str, Path],
    *,
    reference_data_dir_name: str = QM_REFERENCE_DATA_DIRNAME,
    models_dir_name: str = TRAINED_MODELS_DIRNAME,
    data_subdir: Union[str, Path] = Path(".DATA") / "ACTIVE_LEARNING",
    iteration_prefix: str = "iteration",
    allow_fresh_init_on_nonempty: bool = False,
    _active_reconcile_transaction_id: Optional[str] = None,
    artifact_snapshot: Optional[CommittedArtifactSnapshot] = None,
    verification_level: str = "authority",
    progress_stream: Optional[Any] = None,
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

    if verification_level not in {"authority", "metadata", "deep"}:
        raise ValueError(
            "reconcile verification must be authority, metadata or deep"
        )
    if artifact_snapshot is not None:
        if artifact_snapshot.verification_level != verification_level:
            raise ValueError(
                "supplied artefact snapshot verification level does not match reconcile"
            )
        if list(artifact_snapshot.committed_reference_data_versions) != list(tv):
            raise ValueError(
                "supplied artefact snapshot reference-data inventory is stale"
            )
        if list(artifact_snapshot.committed_model_versions) != list(mv):
            raise ValueError("supplied artefact snapshot model inventory is stale")
    else:
        try:
            artifact_snapshot = build_committed_artifact_snapshot(
                campaign,
                verification_level=verification_level,
                progress_stream=progress_stream,
            )
        except Exception as exc:
            unsafe_reasons.append(
                "committed artefact snapshot failed: "
                + type(exc).__name__
                + ": "
                + str(exc)[:200]
            )
            blocking_artifacts.append("committed artefact snapshot")

    deep_verification_required = bool(not existing_loaded and (tv or mv))

    #last phase observed in journal (informational)
    last_phase = None
    last_iter = None
    last_phase_event = None
    last_phase_retryable = False
    last_halt_event = None
    journal_path = data / "journal.ndjson"
    if journal_path.exists():
        try:
            for event in tail_events(journal_path):
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
    bootstrap_handoff = (
        None
        if tv or mv
        else _find_bootstrap_handoff(
            campaign,
            iteration=bootstrap_iteration,
            verification_level=verification_level,
        )
    )
    phase_a_handoff = (
        None
        if tv or mv
        else _find_phase_a_handoff(
            campaign,
            verification_level=verification_level,
        )
    )
    initial_handoff_indicated = _initial_aimall_handoff_indicated(
        existing=existing,
        last_phase=last_phase,
    )
    if bootstrap_handoff is not None:
        initial_handoff_indicated = True

    active_intents: List[Dict[str, Any]] = []
    receipt_backed_intent_repairs: List[Dict[str, Any]] = []
    if artifact_snapshot is not None:
        intent_records = getattr(artifact_snapshot, "submission_intents", ())
        intent_errors = getattr(
            artifact_snapshot,
            "submission_intent_errors",
            (),
        )
        completion_receipt_records = getattr(
            artifact_snapshot,
            "completion_receipts",
            (),
        )
        completion_receipt_errors = getattr(
            artifact_snapshot,
            "completion_receipt_errors",
            (),
        )
    else:
        intent_inventory = _submission_intent.inventory_intents(campaign)
        intent_records = tuple(intent_inventory.get("records", []))
        intent_errors = tuple(intent_inventory.get("errors", []))
        completion_receipt_records = None
        completion_receipt_errors = ()
    for error in intent_errors:
        unsafe_reasons.append(
            "malformed submission intent: "
            + str(error.get("path") or "unknown")
            + ": "
            + str(error.get("error") or "invalid intent")[:180]
        )
        blocking_artifacts.append("malformed submission intent")
    for error in completion_receipt_errors:
        unsafe_reasons.append(
            "malformed phase-completion receipt: "
            + str(error.get("path") or "unknown")
            + ": "
            + str(error.get("error") or "invalid receipt")[:180]
        )
        blocking_artifacts.append("phase completion receipts")
    if existing is not None:
        classification = _submission_intent.classify_completed_unsubmitted_intents(
            campaign,
            existing,
            intents=tuple(intent_records),
            completion_receipts=completion_receipt_records,
            valid_reference_data_versions=(
                artifact_snapshot.valid_reference_data_versions
                if artifact_snapshot is not None
                else None
            ),
            valid_model_versions=(
                artifact_snapshot.valid_model_versions
                if artifact_snapshot is not None
                else None
            ),
        )
        receipt_backed_intent_repairs = [
            dict(item) for item in classification.get("repairs", [])
        ]
        for error in classification.get("errors", []):
            unsafe_reasons.append(
                "phase-completion intent classification failed: "
                + str(error.get("path") or "unknown")
                + ": "
                + str(error.get("error") or "invalid evidence")[:180]
            )
            blocking_artifacts.append("phase completion receipts")
    repair_keys = {
        (
            str(item.get("phase") or ""),
            int(item.get("iteration", 0)),
            str(item.get("submission_identity") or ""),
        )
        for item in receipt_backed_intent_repairs
    }
    for payload in intent_records:
        key = (
            str(payload.get("phase") or ""),
            int(payload.get("iteration", 0)),
            str(payload.get("submission_identity") or ""),
        )
        if (
            str(payload.get("status")) in _submission_intent.ACTIVE_STATUSES
            and key not in repair_keys
        ):
            active_intents.append(dict(payload))
    if receipt_backed_intent_repairs:
        notes.append(
            str(len(receipt_backed_intent_repairs))
            + " receipt-backed scheduler-free submission intent(s) will be retired on apply"
        )
        trusted_artifacts.extend(
            "completion receipt for "
            + str(item.get("phase"))
            + "@"
            + str(item.get("iteration"))
            for item in receipt_backed_intent_repairs
        )
    active_intents.sort(
        key=lambda item: (
            str(item.get("updated_at_iso") or item.get("updated_iso") or ""),
            str(item.get("phase") or ""),
            int(item.get("iteration") or 0),
        )
    )
    scratch_inventory = [
        {
            "status": "prepared",
            "path": str(intent.get("scratch_path_template") or ""),
            "phase": str(intent.get("phase") or ""),
            "iteration": int(intent.get("iteration") or 0),
            "job_id": str(intent.get("job_id") or "PRE_SUBMIT"),
            "submission_identity": str(
                intent.get("submission_identity") or ""
            ),
            "source": "submission_intent",
        }
        for intent in active_intents
    ]
    if artifact_snapshot is not None:
        reconcile_transactions = [
            dict(record)
            for record in getattr(
                artifact_snapshot,
                "reconcile_transactions",
                (),
            )
        ]
        reference_commit_transactions = [
            dict(record)
            for record in getattr(
                artifact_snapshot,
                "reference_commit_transactions",
                (),
            )
        ]
    else:
        from .reconcile_transaction import inventory_reconcile_transactions
        from .reference_commit import inventory_reference_commits

        reconcile_transactions = inventory_reconcile_transactions(campaign)
        reference_commit_transactions = inventory_reference_commits(
            campaign,
            verification=(
                "authority" if verification_level == "authority" else "metadata"
            ),
        )
    invalid_reference_commits = [
        record
        for record in reference_commit_transactions
        if str(record.get("state") or "") == "invalid"
    ]
    active_reference_commits = [
        record
        for record in reference_commit_transactions
        if str(record.get("state") or "") not in {"complete", "invalid"}
    ]
    reference_commit_decisions: List[RecoveryDecision] = []
    if invalid_reference_commits:
        unsafe_reasons.append(
            "invalid reference-commit transaction evidence: "
            + "; ".join(
                str(record.get("path"))
                + " ("
                + str(record.get("reason") or "invalid")
                + ")"
                for record in invalid_reference_commits[:5]
            )
        )
        blocking_artifacts.append("reference-commit transaction evidence")
    if len(active_reference_commits) > 1:
        unsafe_reasons.append(
            "multiple incomplete reference-commit transactions require user review"
        )
        blocking_artifacts.append("reference-commit transaction evidence")
    elif active_reference_commits:
        transaction = active_reference_commits[0]
        ledger = transaction.get("ledger") or {}
        decision = RecoveryDecision(
            phase=CampaignPhase.REFERENCE_COMMIT,
            iteration=int(ledger["iteration"]),
            reason=(
                "resume reference commit from "
                + str(transaction.get("state"))
                + " transaction state"
            ),
            trusted_artifact=str(transaction.get("path") or ""),
        )
        reference_commit_decisions.append(decision)
        _append_recovery_candidate(recovery_candidates, decision)
        trusted_artifacts.append(
            "reference-commit transaction "
            + str(transaction.get("state"))
            + " at "
            + str(transaction.get("path"))
        )
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
    protected_reference_source_paths = set()
    for transaction in active_reference_commits:
        ledger = transaction.get("ledger") or {}
        bucket_name = (
            "initial"
            if str(ledger.get("context")) == "bootstrap"
            else "iter_" + str(int(ledger.get("iteration", 0)))
        )
        protected_reference_source_paths.add(
            str((staging_root / bucket_name).resolve(strict=False))
        )
    if protected_reference_source_paths:
        unexpected_staging_children = [
            path
            for path in unexpected_staging_children
            if str(path.resolve(strict=False)) not in protected_reference_source_paths
        ]
    scripts_root = campaign / ".DATA" / "SCRIPTS"
    script_inventory = _scripts_inventory(
        campaign,
        intent_records=intent_records,
    )
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
            from .ferebus_candidate_recovery import read_recovery_request

            recovery_request = read_recovery_request(
                campaign,
                expected_campaign_uid=(
                    str(existing.campaign_uid) if existing is not None else None
                ),
            )
            recovery_source = (
                campaign_owned_path(
                    campaign,
                    str(recovery_request.get("source_path") or ""),
                )
                if isinstance(recovery_request, dict)
                else None
            )
            recovery_staging = (
                campaign_owned_path(
                    campaign,
                    str(recovery_request.get("staging_path") or ""),
                )
                if isinstance(recovery_request, dict)
                and recovery_request.get("staging_path")
                else None
            )
            if (
                isinstance(recovery_request, dict)
                and str(recovery_request.get("status")) in {
                    "prepared",
                    "measurement_incomplete",
                    "materialised",
                }
                and (
                    recovery_source == model_iteration_staging.absolute()
                    or recovery_staging == model_iteration_staging.absolute()
                )
            ):
                recoverable_ferebus_staging = True
                recoverable_ferebus_reason = (
                    "FEREBUS iteration-staging is protected by recovery request "
                    + str(recovery_request.get("request_sha256") or "")
                )
        except Exception as exc:
            recoverable_ferebus_reason = type(exc).__name__ + ": " + str(exc)[:180]
    if (
        not recoverable_ferebus_staging
        and verification_level != "authority"
        and has_model_iteration_staging
        and not model_iteration_staging.is_symlink()
    ):
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
    elif has_model_iteration_staging:
        recoverable_ferebus_reason = (
            "unpublished FEREBUS staging is non-authoritative and will be "
            "archived before retry"
        )

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
    protected_reference_staging_paths = {
        str(
            ReferenceDataVersioning(training_dir).staging_path(
                int((record.get("ledger") or {}).get("iteration", -1))
            ).resolve(strict=False)
        )
        for record in active_reference_commits
        if isinstance(record.get("ledger"), dict)
    }
    unprotected_dangling_reference_data = [
        path
        for path in dangling_reference_data
        if str(path.resolve(strict=False)) not in protected_reference_staging_paths
    ]
    if unprotected_dangling_reference_data:
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

    valid_reference_data_versions = (
        list(artifact_snapshot.valid_reference_data_versions)
        if artifact_snapshot is not None
        else []
    )
    valid_model_versions = (
        list(artifact_snapshot.valid_model_versions)
        if artifact_snapshot is not None
        else []
    )
    trusted_artifacts.extend(
        "reference-data version " + str(version)
        for version in valid_reference_data_versions
    )
    trusted_artifacts.extend(
        "model version " + str(version)
        for version in valid_model_versions
    )
    if artifact_snapshot is not None:
        for version, error in artifact_snapshot.reference_errors.items():
            unsafe_reasons.append(
                "committed reference-data version "
                + str(int(version))
                + " manifest invalid: "
                + str(error)[:180]
            )
            blocking_artifacts.append("reference-data version " + str(int(version)))
        for version, error in artifact_snapshot.model_errors.items():
            unsafe_reasons.append(
                "committed model version "
                + str(int(version))
                + " manifest invalid: "
                + str(error)[:180]
            )
            blocking_artifacts.append("model version " + str(int(version)))
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
            artifact_snapshot=artifact_snapshot,
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
            manifest_path = campaign / POOL_SUBDIR / POOL_MANIFEST_FILENAME
            manifest_payload = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                source=manifest_path,
            )
            manifest = TrajectoryPoolManifest.from_dict(manifest_payload)
            expected_pool = (campaign / POOL_XYZ_FILENAME).absolute()
            if Path(manifest.canonical_path).absolute() != expected_pool:
                raise ValueError("pool manifest canonical path does not match campaign")
            if manifest.natoms <= 0:
                raise ValueError("pool natoms must be positive")
            if verification_level == "deep":
                from ..versioning.manifest import sha256_file

                if sha256_file(expected_pool) != str(manifest.sha256):
                    raise ValueError("trajectory pool SHA mismatch")
            trusted_artifacts.append(
                "trajectory pool authority SHA " + str(manifest.sha256)[:12]
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
            reference_scales_models_version=(
                existing.reference_scales_models_version
            ),
            reference_scales_model_manifest_sha256=(
                existing.reference_scales_model_manifest_sha256
            ),
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
        protected_staging_handoffs = staging_handoff_decisions(
            campaign,
            recovered,
            verification=verification_level,
            artifact_snapshot=artifact_snapshot,
        )
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
            "multiple valid staging handoffs need user review: "
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
        partial_iteration_handoffs = active_iteration_handoff_decisions(
            campaign,
            recovered,
            verification=verification_level,
            artifact_snapshot=artifact_snapshot,
        )
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
            "multiple valid active-iteration handoffs need user review: "
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
    combined_handoffs = (
        list(protected_staging_handoffs)
        + list(partial_iteration_handoffs)
        + list(reference_commit_decisions)
    )
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
    ariadne_publication_recovery: Optional[Dict[str, Any]] = None
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
    if (
        isinstance(partial_array_recovery, dict)
        and str(partial_array_recovery.get("phase") or "")
        == CampaignPhase.ARIADNE_ARRAY.value
    ):
        try:
            ariadne_publication_recovery = classify_ariadne_publication(
                campaign,
                int(partial_array_recovery["iteration"]),
                expected_campaign_uid=str(recovered.campaign_uid),
            )
        except Exception as exc:
            ariadne_publication_recovery = {
                "state": "invalid",
                "archive_required": False,
                "reason": type(exc).__name__ + ": " + str(exc),
            }
        publication_state = str(
            ariadne_publication_recovery.get("state") or "invalid"
        )
        if publication_state == "invalid":
            reason = (
                "invalid ARIADNE publication: "
                + str(ariadne_publication_recovery.get("reason") or "unknown error")
            )
            unsafe_reasons.append(reason)
            blocking_artifacts.append("invalid ARIADNE publication")
        elif publication_state == "complete" and bool(
            ariadne_publication_recovery.get("accepted", False)
        ):
            ariadne_publication_recovery["archive_for_replay"] = True
            unsafe_reasons.append("stale ARIADNE publication")
            blocking_artifacts.append("stale ARIADNE publication")
            notes.append(
                "accepted ARIADNE publication was not committed by state and will "
                "be archived before local postprocessing replay"
            )
        elif publication_state == "complete":
            unsafe_reasons.append(
                "complete rejected ARIADNE batch decision requires user review"
            )
            blocking_artifacts.append("rejected ARIADNE batch decision")
        elif bool(ariadne_publication_recovery.get("archive_required", False)):
            unsafe_reasons.append("stale ARIADNE publication")
            blocking_artifacts.append("stale ARIADNE publication")
            notes.append(
                "ARIADNE derived publication will be archived before postprocessing: "
                + publication_state
            )
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
            verify_state_referenced_artifacts(
                campaign,
                existing,
                verification=verification_level,
                snapshot=artifact_snapshot,
            )
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
                verification=verification_level,
                artifact_snapshot=artifact_snapshot,
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
        decision = "STOPPED: existing user stop request remains authoritative"
        notes.append(
            "preserved user stop request; only resume may clear it"
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
                    + ": selected from active submission intent for user review"
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
            recovered.phase = CampaignPhase.REFERENCE_COMMIT
            recovered.iteration = 0
            decision = "REFERENCE_COMMIT: valid initial AIMAll handoff exists"
            notes.append(
                "re-entry at REFERENCE_COMMIT to publish reference-data version 0"
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
                "re-entry HALTED because unsafe artefacts block REFERENCE_COMMIT recovery"
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
            # state was halted after a user increased max_iterations.
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
            "re-entry HALTED because committed artefacts need user review"
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
        decision = "HALTED: unsafe artefacts need user review"
        notes.append(
            "re-entry HALTED because unsafe committed artefacts need user review"
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
                verification=verification_level,
                artifact_snapshot=artifact_snapshot,
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
    ferebus_candidate_recovery: Optional[Dict[str, Any]] = None
    if recovered.phase in {
        CampaignPhase.INITIAL_FEREBUS,
        CampaignPhase.FEREBUS,
    }:
        try:
            from .ferebus_candidate_recovery import discover_recovery_candidate

            ferebus_candidate_recovery = discover_recovery_candidate(
                campaign,
                expected_campaign_uid=str(recovered.campaign_uid),
                reference_data_version=int(recovered.reference_data_version),
            )
        except Exception as exc:
            reason = (
                "FEREBUS candidate recovery is ambiguous or invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)[:300]
            )
            unsafe_reasons.append(reason)
            blocking_artifacts.append("FEREBUS candidate recovery")
            recovered.phase = CampaignPhase.HALTED
            decision = "HALTED: FEREBUS candidate recovery requires user review"
        else:
            if ferebus_candidate_recovery is not None:
                trusted_artifacts.append(
                    "recoverable FEREBUS raw candidate at "
                    + str(ferebus_candidate_recovery.get("source_path") or "")
                )
                notes.append(
                    "existing authenticated FEREBUS output can be reprocessed "
                    "without another scheduler submission"
                )

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
        reference_commit_transactions=reference_commit_transactions,
        notes=notes,
        existing_state_loaded=existing_loaded,
        unsafe_reasons=unsafe_reasons,
        active_submission_intents=active_intents,
        receipt_backed_intent_repairs=receipt_backed_intent_repairs,
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
        ariadne_publication_recovery=(
            dict(ariadne_publication_recovery)
            if isinstance(ariadne_publication_recovery, dict)
            else None
        ),
        ferebus_candidate_recovery=ferebus_candidate_recovery,
        bootstrap_handoff=bootstrap_handoff,
        phase_a_handoff=phase_a_handoff,
        artifact_snapshot=artifact_snapshot,
        deep_verification_required=deep_verification_required,
    )



def write_proposed_state(
    campaign_dir: Union[str, Path],
    report: ReconciliationReport,
    *,
    data_subdir: Union[str, Path] = Path(".DATA") / "ACTIVE_LEARNING",
) -> Path:
    """Write the proposed state to <state_path>.proposed and return the
    path. The user promotes it manually via mv state.json.proposed
    state.json after reviewing."""
    target = (
        Path(campaign_dir) / data_subdir / (DEFAULT_STATE_FILENAME + RECONCILE_SUFFIX)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    write_state(target, report.proposed_state)
    return target



