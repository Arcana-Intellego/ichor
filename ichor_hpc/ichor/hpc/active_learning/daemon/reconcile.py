"""Recovery for a corrupt or missing state.json with user in the loop.

The daemon's design treats 'state.json' as the only authoritative checkpoint.
If it goes missing or fails schema validation, the daemon refuses to auto-
recover; a silent reconstruction is exactly the bug class we want to avoid.

"reconcile" inspects the on-disk artefacts that DO exist (5_TRAINING/
committed iterations, 6_TRAINED_MODELS/, journal entries) and proposes a
"CampaignState" it believes is consistent with them. The proposal is
written to "<state_path>.proposed" and the operator must explicitly
promote it ("mv state.json.proposed state.json") before restarting the
daemon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..acquisition.trajectory_pool import TrajectoryPool
from ..versioning.training_set import TrainingSetVersioning
from .artifact_contracts import (
    verify_committed_model_version,
    verify_committed_training_version,
)
from .journal import iter_events
from . import submission_intent as _submission_intent
from .state import (
    CampaignPhase,
    CampaignState,
    DEFAULT_STATE_FILENAME,
    StateSchemaError,
    fresh_campaign_state,
    read_state,
    write_state,
)


__all__ = [
    "ReconciliationReport",
    "data_staging_inventory",
    "propose_recovery",
    "stateful_campaign_artifacts",
    "write_proposed_state",
    "RECONCILE_SUFFIX",
]


RECONCILE_SUFFIX = ".proposed"


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

    def add_matches(pattern: str) -> None:
        for path in sorted(campaign.glob(pattern)):
            findings.append(str(path.relative_to(campaign)))

    add_matches("5_TRAINING/iteration-*")
    add_matches("6_TRAINED_MODELS/iteration-*")
    add_matches("7_ACTIVE_LEARNING/iteration-*")
    if (campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson").is_file():
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
    return findings


@dataclass
class ReconciliationReport:
    """Diagnostic bundle returned by :func: propose_recovery."""

    proposed_state: CampaignState
    committed_training_versions: List[int] = field(default_factory=list)
    committed_model_versions: List[int] = field(default_factory=list)
    last_phase_in_journal: Optional[str] = None
    last_iteration_in_journal: Optional[int] = None
    notes: List[str] = field(default_factory=list)
    existing_state_loaded: bool = False
    unsafe_reasons: List[str] = field(default_factory=list)
    active_submission_intents: List[Dict[str, Any]] = field(default_factory=list)
    decision: str = ""
    trusted_artifacts: List[str] = field(default_factory=list)
    blocking_artifacts: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


def _needs_trajectory_pool_check(
    *,
    existing_loaded: bool,
    training_versions: List[int],
    model_versions: List[int],
    active_intents: List[Dict[str, Any]],
    last_phase: Optional[str],
    staging_children: List[Path],
    script_files: List[Path],
    dangling_training: List[Path],
    dangling_models: List[Path],
    has_model_iteration_staging: bool,
) -> bool:
    return any(
        (
            existing_loaded,
            bool(training_versions),
            bool(model_versions),
            bool(active_intents),
            bool(last_phase),
            bool(staging_children),
            bool(script_files),
            bool(dangling_training),
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


def propose_recovery(
    campaign_dir: Union[str, Path],
    *,
    training_dir_name: str = "5_TRAINING",
    models_dir_name: str = "6_TRAINED_MODELS",
    data_subdir: Union[str, Path] = Path(".DATA") / "ACTIVE_LEARNING",
    iteration_prefix: str = "iteration",
    allow_fresh_init_on_nonempty: bool = False,
) -> ReconciliationReport:
    """Inspect the campaign tree and propose a recovered CampaignState.

    The recovered state is conservative: it always positions the daemon at
    a stable "safe" entry point (STOP_CHECK for completed iterations or
    INIT when nothing is committed yet) and clears any pending_jobs so the
    next run re-submits rather than blindly polls unknown JobIDs.

    Heuristics:
        - committed training versions in "5_TRAINING/" define the maximum
          completed iteration; the next iteration to plan from is one past
          that.
        - committed model versions in "6_TRAINED_MODELS/" likewise.
        - journal entries inform "last_phase_in_journal" for the report only.
        - if a prior state.json exists and parses, its "max_iterations",
          "campaign_uid", and "campaign_started_iso" are preserved so the
          recovery does not destroy provenance.
    """
    campaign = Path(campaign_dir)
    data = campaign / data_subdir
    state_path = data / DEFAULT_STATE_FILENAME

    notes: List[str] = []
    unsafe_reasons: List[str] = []
    trusted_artifacts: List[str] = []
    blocking_artifacts: List[str] = []
    recommended_actions: List[str] = []

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
                import json as _json
                raw = _json.loads(state_path.read_text(encoding="utf-8"))
                salvaged_uid = raw.get("campaign_uid")
                salvaged_started = raw.get("campaign_started_iso")
                if salvaged_uid:
                    notes.append("salvaged campaign_uid from the unparsable state.json")
                else:
                    notes.append("WARNING: no campaign_uid to salvage; a fresh one will be minted")
            except Exception:
                notes.append("WARNING: state.json not even json; a fresh campaign_uid will be minted")
        except Exception as exc:
            notes.append("existing state.json unreadable: " + type(exc).__name__)
            notes.append("WARNING: a fresh campaign_uid will be minted; restore a good state.json to keep provenance")
    else:
        notes.append("no existing state.json")

    #committed training versions
    training_dir = campaign / training_dir_name
    if training_dir.is_dir():
        tv = TrainingSetVersioning(training_dir, prefix=iteration_prefix).list_committed_versions()
    else:
        tv = []
        notes.append("training dir " + training_dir_name + " missing")
    #committed model versions
    models_dir = campaign / models_dir_name
    if models_dir.is_dir():
        mv = TrainingSetVersioning(models_dir, prefix=iteration_prefix).list_committed_versions()
    else:
        mv = []
        notes.append("models dir " + models_dir_name + " missing")

    #last phase observed in journal (informational)
    last_phase = None
    last_iter = None
    journal_path = data / "journal.ndjson"
    if journal_path.exists():
        for event in iter_events(journal_path):
            if event.get("event") == "phase_transition":
                #to_phase tells us what was advanced to most recently
                last_phase = event.get("to_phase") or last_phase
                last_iter = event.get("iteration", last_iter)

    active_intents: List[Dict[str, Any]] = []
    intent_root = _submission_intent.intent_dir(campaign)
    if intent_root.is_dir():
        for p in sorted(intent_root.glob("*.json")):
            try:
                import json as _json
                payload = _json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                unsafe_reasons.append("unreadable submission intent: " + str(p))
                continue
            if isinstance(payload, dict) and str(payload.get("status")) in _submission_intent.ACTIVE_STATUSES:
                active_intents.append(payload)
    active_intents.sort(
        key=lambda item: (
            str(item.get("updated_at_iso") or item.get("updated_iso") or ""),
            str(item.get("phase") or ""),
            int(item.get("iteration") or 0),
        )
    )

    staging_root = campaign / ".DATA" / "STAGING"
    staging_children = [
        p for p in (staging_root.iterdir() if staging_root.is_dir() else [])
        if p.name not in (".", "..")
    ]
    scripts_root = campaign / ".DATA" / "SCRIPTS"
    script_files = [
        p for p in (scripts_root.glob("*.sh") if scripts_root.is_dir() else [])
        if p.is_file()
    ]
    dangling_training = (
        TrainingSetVersioning(training_dir, prefix=iteration_prefix).list_dangling_staging()
        if training_dir.is_dir() else []
    )
    dangling_models = (
        TrainingSetVersioning(models_dir, prefix=iteration_prefix).list_dangling_staging()
        if models_dir.is_dir() else []
    )
    model_iteration_staging = models_dir / "iteration-staging"
    has_model_iteration_staging = model_iteration_staging.is_dir()

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
    if staging_children:
        unsafe_reasons.append(".DATA/STAGING is non-empty")
        blocking_artifacts.append(".DATA/STAGING")
    if script_files:
        unsafe_reasons.append(".DATA/SCRIPTS contains sbatch scripts")
        trusted_artifacts.append(".DATA/SCRIPTS can be archived by reconcile --apply")
    if dangling_training:
        unsafe_reasons.append("dangling training staging directories exist")
        blocking_artifacts.append("dangling training staging")
    if dangling_models or has_model_iteration_staging:
        unsafe_reasons.append("dangling model staging directories exist")
        blocking_artifacts.append("dangling model staging")

    valid_training_versions: List[int] = []
    for version in tv:
        try:
            verify_committed_training_version(
                campaign,
                int(version),
                training_dir_name=training_dir_name,
            )
            valid_training_versions.append(int(version))
            trusted_artifacts.append("training version " + str(version))
        except Exception as exc:
            unsafe_reasons.append(
                "committed training version "
                + str(version)
                + " manifest invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)[:160]
            )
            blocking_artifacts.append("training version " + str(version))
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

    if _needs_trajectory_pool_check(
        existing_loaded=existing_loaded,
        training_versions=tv,
        model_versions=mv,
        active_intents=active_intents,
        last_phase=last_phase,
        staging_children=staging_children,
        script_files=script_files,
        dangling_training=dangling_training,
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
            training_set_version=existing.training_set_version,
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

    coherent_pairs = sorted(set(valid_training_versions).intersection(valid_model_versions))
    latest_training_only = max(valid_training_versions) if valid_training_versions else None
    latest_model_only = max(valid_model_versions) if valid_model_versions else None
    no_coherent_pair = False

    if coherent_pairs:
        coherent = int(coherent_pairs[-1])
        if recovered.training_set_version != coherent:
            notes.append(
                "training_set_version set to coherent committed version "
                + str(coherent)
            )
        if recovered.models_version != coherent:
            notes.append(
                "models_version set to coherent committed version "
                + str(coherent)
            )
        recovered.training_set_version = coherent
        recovered.models_version = coherent
        if latest_training_only is not None and latest_training_only > coherent:
            unsafe_reasons.append(
                "newer committed training version has no matching model: "
                + str(latest_training_only)
            )
        if latest_model_only is not None and latest_model_only > coherent:
            unsafe_reasons.append(
                "newer committed model version has no matching training set: "
                + str(latest_model_only)
            )
    elif valid_training_versions and not valid_model_versions:
        recovered.training_set_version = int(latest_training_only)
        recovered.models_version = -1
        notes.append(
            "valid training exists without any valid model; re-entry must train FEREBUS"
        )
    elif valid_training_versions or valid_model_versions:
        no_coherent_pair = True
        unsafe_reasons.append(
            "no coherent committed training/model version pair exists "
            + "(training="
            + repr(valid_training_versions)
            + ", models="
            + repr(valid_model_versions)
            + ")"
        )

    # choose a safe re-entry phase. If we have NOTHING committed, start at
    #  INIT; otherwise rewind to STOP_CHECK so the next tick decides whether
    # to loop or terminate.
    if not tv and not mv and not existing_loaded and unsafe_reasons and not allow_fresh_init_on_nonempty:
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
    elif not tv and not mv and not existing_loaded:
        recovered.phase = CampaignPhase.INIT
        decision = "INIT: no existing state or committed iterations"
        notes.append("no committed iterations; re-entry at INIT")
    elif no_coherent_pair:
        recovered.phase = CampaignPhase.HALTED
        decision = "HALTED: no coherent committed training/model pair"
        notes.append("re-entry HALTED because no coherent training/model pair exists")
    elif valid_training_versions and not valid_model_versions:
        recovered.phase = CampaignPhase.INITIAL_FEREBUS if recovered.training_set_version == 0 else CampaignPhase.FEREBUS
        decision = recovered.phase.value + ": valid training exists without a committed model"
        notes.append(
            "re-entry at "
            + recovered.phase.value
            + " to produce model version "
            + str(recovered.training_set_version)
        )
    elif coherent_pairs and unsafe_reasons:
        recovered.phase = CampaignPhase.HALTED
        decision = "HALTED: coherent committed versions exist but unsafe artefacts need review"
        notes.append(
            "re-entry HALTED because committed artefacts need operator review"
        )
    elif unsafe_reasons:
        recovered.phase = CampaignPhase.HALTED
        decision = "HALTED: unsafe artefacts need operator review"
        notes.append(
            "re-entry HALTED because unsafe committed artefacts need operator review"
        )
    else:
        recovered.phase = CampaignPhase.STOP_CHECK
        decision = "STOP_CHECK: latest coherent committed training/model pair is trusted"
        notes.append("re-entry at STOP_CHECK (next tick decides loop/terminate)")
    recovered.pending_jobs = {}
    recovered.shutdown_requested = False
    if recovered.phase is CampaignPhase.HALTED and not recommended_actions:
        recommended_actions.append("Inspect unsafe recovery reasons before applying.")

    return ReconciliationReport(
        proposed_state=recovered,
        committed_training_versions=tv,
        committed_model_versions=mv,
        last_phase_in_journal=last_phase,
        last_iteration_in_journal=last_iter,
        notes=notes,
        existing_state_loaded=existing_loaded,
        unsafe_reasons=unsafe_reasons,
        active_submission_intents=active_intents,
        decision=decision,
        trusted_artifacts=trusted_artifacts,
        blocking_artifacts=blocking_artifacts,
        recommended_actions=recommended_actions,
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



