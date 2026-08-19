"""CLI for the ICHOR active-learning daemon.

Console entry point 'ichor-al-daemon' registered in
'ichor_cli/setup.cfg'. Subcommands available:

    start      Start the live daemon detached by default; use --foreground to block.
    stop       Write a durable immediate or boundary stop request.
    status     Print the current state snapshot.
    resume     Continue a stopped or safely recovered campaign.
    reconcile  Inspect on-disk artefacts and propose a recovered state.
    journal    Tail or filter the campaign journal.
    init       Bootstrap campaign.yaml, daemon state, config lock, and pool.
    export-batch-geometries
               Export accepted active-iteration seed/final geometry pairs.

Commands that operate on a campaign accept '--campaign-dir DIR'. When it is
omitted, the CLI uses the current working directory if it contains
'campaign.yaml'.

The CLI binds each campaign to exactly one immutable execution mode on first
start: ``live`` or ``dry_run``.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import re
import shlex
import signal
from .strict_json import strict_json as json
import secrets
import subprocess
import sys
import time
import numpy as np
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import CampaignConfig
from .daemon.daemon import (
    DAEMON_HEARTBEAT_FILENAME,
    DAEMON_LEASE_DIRNAME,
    DAEMON_LOCK_FILENAME,
    DEFAULT_DATA_SUBDIR,
    Daemon,
)
from .daemon.background_startup import (
    BACKGROUND_CHILD_ENV,
    BACKGROUND_LAUNCH_ID_ENV,
    BACKGROUND_READINESS_ENV,
    BACKGROUND_STARTUP_ACKNOWLEDGED_STATES,
    BACKGROUND_STARTUP_ACTIVE_STATES,
    BACKGROUND_STARTUP_FILENAME,
    BACKGROUND_STARTUP_PATH_ENV,
    BACKGROUND_STARTUP_SCHEMA_VERSION,
    initialise_background_startup,
    launch_id_from_environment,
    read_background_startup,
    startup_path_from_environment,
    update_background_startup,
)
from .daemon.journal import (
    JOURNAL_EVENT_CONTEXTS,
    JOURNAL_PHASE_FIRST_EVENTS,
    KNOWN_EVENT_TYPES,
    JournalCorruptionError,
    read_events,
)
from .daemon.lease import evaluate_lease_liveness, validate_lease_heartbeat
from .submit.slurm_contracts import (
    parse_squeue_job_id,
    run_scheduler_command,
    validate_parent_job_id,
)
from .submit.scheduler_backend import (
    current_scheduler_user,
    get_scheduler_backend,
)
from .daemon.cluster_profile import profile_value, require_cluster_profile
from .daemon.config_lock import (
    archive_ferebus_iteration_staging_for_retrain,
    archive_scripts_for_reconcile,
    archive_data_staging_for_ferebus_reentry,
    archive_data_staging_for_operator_reconcile,
    archive_reference_data_staging_for_reconcile,
    assert_config_unchanged_for_start,
    clean_model_iteration_staging_for_reconcile,
    clean_reentry_staging,
    canonical_config,
    config_fingerprint,
    config_lock_path,
    ensure_config_lock,
    read_config_lock,
    ferebus_reentry_can_archive_data_staging,
    format_config_review,
    review_config_changes,
    restore_config_from_lock_proposal,
    restore_config_lock_from_history,
    reference_data_staging_can_archive_for_reconcile,
)
from .daemon.dry_run_executor import DryRunPhaseExecutor
from .daemon.dry_run_sacct import DryRunSacctPoller
from .daemon.job_names import live_job_name
from .daemon.scheduler_recovery import (
    classify_terminal_scheduler_evidence,
    classify_unaccepted_scheduler_intent,
    load_scheduler_terminal_receipt,
    scheduler_terminal_receipt_records_cancellation,
    scheduler_terminal_receipt_path,
    write_scheduler_terminal_receipt,
)
from .daemon.live_executor import (
    LiveBackendNotAvailableError,
    LiveBackendsPhaseExecutor,
    make_live_job_accounting_finder,
    make_live_job_finder,
    make_live_job_liveness_checker,
    make_live_queue_diagnostics_collector,
)
from .daemon.preflight import check_backends, missing_backend_message
from .daemon.presentation_assessment import (
    assess_campaign_presentation,
    classify_operator_failure,
    config_review_evidence,
    invalid_config_review_evidence,
)
from .daemon.submitted_environment_smoke import run_submitted_environment_smoke
from .daemon.recovery_contracts import (
    recovery_contract_status,
    staging_handoff_decisions,
    validate_phase_recovery_contract,
)
from .daemon.reconcile import (
    archive_incomplete_scalar_diversity_publication,
    data_staging_inventory,
    propose_recovery,
    restore_archived_bootstrap_handoff,
    stateful_campaign_artifacts,
    write_proposed_state,
)
from .daemon.ferebus_staging_recovery import (
    classify_ferebus_staging_recovery,
    is_redundant_committed_parent_staging,
    restore_archived_ferebus_producer_staging,
)
from .daemon.artifact_snapshot import (
    ArtefactSnapshotError,
    build_committed_artifact_snapshot,
)
from .daemon.aimall_quality_revalidation import (
    apply_aimall_quality_revalidation,
    inspect_aimall_quality_revalidation,
)
from .daemon.array_recovery import (
    array_ledger_path,
    archive_existing_array_task_outputs,
    compact_array_recovery_summary,
    prepare_aimall_upstream_gaussian_recovery,
    read_array_ledger,
    refresh_array_ledger,
    supports_partial_array_recovery,
)
from .daemon.ariadne_publication import (
    archive_ariadne_publication,
    classify_ariadne_publication,
)
from .daemon.reconcile_transaction import (
    ReconcileTransaction,
    apply_reconcile_transaction_recovery,
    begin_reconcile_transaction,
    build_reconcile_commit_plan,
    inspect_reconcile_transaction_recovery,
    publish_reconcile_config_target,
    publish_reconcile_intent_target,
    publish_reconcile_state_backup,
    restore_version_pointer,
    snapshot_version_pointer,
)
from .daemon import submission_intent as _submission_intent
from .daemon import scratch as _scratch
from .daemon.state import (
    CampaignState,
    CampaignPhase,
    DEFAULT_STATE_FILENAME,
    StateSchemaError,
    fresh_campaign_state,
    read_state,
    write_state,
    make_lifecycle_context,
)
from .daemon.filesystem import campaign_owned_path, operational_data_dir, operational_path
from .daemon.status_recommendations import (
    build_status_recommendations,
    recommendation_dicts,
)
from .versioning.trained_models import TrainedModelVersioning
from .layout import active_learning_dir, trained_models_dir


__all__ = [
    "build_parser",
    "expand_boolean_short_flag_clusters",
    "main",
    "resolve_campaign_dir",
    "ShortFlagClusterError",
]


BACKGROUND_LOG_FILENAME = "daemon.out"
BACKGROUND_PID_FILENAME = "daemon.pid"
BACKGROUND_PID_SCHEMA_VERSION = 1


class CampaignDirResolutionError(ValueError):
    """Raised when a command cannot infer a valid campaign directory."""


class CampaignBootstrapError(ValueError):
    """Raised when init cannot safely create or validate daemon state."""


class ShortFlagClusterError(ValueError):
    """Raised when a compact short-flag cluster would be ambiguous."""


def resolve_campaign_dir(
    value: Optional[str],
    *,
    require_campaign_yaml: bool = True,
) -> Path:
    """Resolve a campaign directory from an explicit value or cwd.

    Explicit ``--campaign-dir`` values always win. When omitted, the current
    directory is accepted only if it looks like a campaign directory, i.e. it
    contains ``campaign.yaml``.
    """
    if value:
        campaign = Path(value).expanduser().resolve()
        if not campaign.exists():
            raise CampaignDirResolutionError(
                "campaign directory does not exist: " + str(campaign)
            )
        if not campaign.is_dir():
            raise CampaignDirResolutionError(
                "campaign path is not a directory: " + str(campaign)
            )
        if require_campaign_yaml and not (campaign / "campaign.yaml").is_file():
            raise CampaignDirResolutionError(
                "campaign directory does not contain campaign.yaml: "
                + str(campaign)
            )
        return campaign

    campaign = Path.cwd().resolve()
    if require_campaign_yaml and not (campaign / "campaign.yaml").is_file():
        raise CampaignDirResolutionError(
            "No campaign directory supplied and the current directory does not "
            "contain campaign.yaml. Use --campaign-dir DIR or cd into a "
            "campaign directory."
        )
    return campaign


_BOOLEAN_SHORT_CLUSTERS = {
    "start": frozenset({"b"}),
    "resume": frozenset({"b"}),
    "stop": frozenset({"x"}),
    "status": frozenset({"j", "v"}),
    "reconcile": frozenset({"a", "F"}),
    "journal": frozenset({"j", "r", "v"}),
    "init": frozenset({"f", "y"}),
    "import-pool": frozenset({"f", "y"}),
}


_VALUE_SHORT_FLAGS = {
    "start": frozenset({"c", "g", "p", "t", "P", "o", "i"}),
    "resume": frozenset({"c", "g", "p", "t", "P", "o", "i"}),
    "stop": frozenset({"c"}),
    "status": frozenset({"c"}),
    "reconcile": frozenset({"c"}),
    "journal": frozenset({"c", "s", "e", "n"}),
    "init": frozenset({"c", "s"}),
    "import-pool": frozenset({"c", "s"}),
    "preflight": frozenset({"c"}),
}


def expand_boolean_short_flag_clusters(argv: Optional[Sequence[str]]) -> List[str]:
    """Expand safe boolean short-flag clusters such as ``-lb``.

    Only clusters made entirely from known no-value boolean flags for the
    selected subcommand are expanded. Options that take values are left for
    argparse to handle normally.
    """
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        return args
    command = args[0]
    allowed = _BOOLEAN_SHORT_CLUSTERS.get(command)
    value_flags = _VALUE_SHORT_FLAGS.get(command, frozenset())
    if not allowed:
        return args
    out = [command]
    for token in args[1:]:
        if (
            token.startswith("-")
            and not token.startswith("--")
            and len(token) > 2
            and all(ch in allowed for ch in token[1:])
        ):
            out.extend("-" + ch for ch in token[1:])
        elif (
            token.startswith("-")
            and not token.startswith("--")
            and len(token) > 2
            and token[1] in (allowed | value_flags)
        ):
            raise ShortFlagClusterError(
                "unsupported short flag cluster "
                + repr(token)
                + "; use separate flags such as '-b', and pass values as "
                "separate arguments such as '-t 10'."
            )
        else:
            out.append(token)
    return out


RETRYABLE_CLEANED_REENTRY_PHASES = {
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
    CampaignPhase.REFERENCE_COMMIT,
    CampaignPhase.FEREBUS,
}


def _campaign_paths(campaign_dir: Path):
    data = operational_data_dir(campaign_dir)
    return {
        "data": data,
        "state": operational_path(campaign_dir, DEFAULT_STATE_FILENAME),
        "lock": operational_path(campaign_dir, DAEMON_LOCK_FILENAME),
        "lease": operational_path(campaign_dir, DAEMON_LEASE_DIRNAME),
        "journal": operational_path(campaign_dir, "journal.ndjson"),
        "background_log": operational_path(campaign_dir, BACKGROUND_LOG_FILENAME),
        "background_pid": operational_path(campaign_dir, BACKGROUND_PID_FILENAME),
        "background_startup": operational_path(
            campaign_dir, BACKGROUND_STARTUP_FILENAME
        ),
        "stop_request": operational_path(campaign_dir, "stop_request.json"),
    }


def _campaign_command(
    campaign: Path,
    command: str,
    suffix: str = "",
) -> str:
    """Build one copyable daemon command for a campaign."""
    return (
        "ichor-al-daemon "
        + str(command)
        + " --campaign-dir "
        + shlex.quote(str(campaign))
        + str(suffix)
    )


def _stop_control_status(
    campaign: Path,
    *,
    expected_campaign_uid: Optional[str] = None,
    state: Optional[CampaignState] = None,
) -> Dict[str, Any]:
    from .daemon.stop_control import (
        read_stop_request,
        stop_request_disposition,
        stop_request_summary,
        validate_stop_request_for_recovery,
    )

    try:
        request = read_stop_request(
            campaign,
            expected_campaign_uid=expected_campaign_uid,
        )
    except Exception as exc:
        return {
            "stop_request": None,
            "stop_control_error": type(exc).__name__ + ": " + str(exc),
        }
    result = {
        "stop_request": stop_request_summary(request),
        "stop_control_error": None,
    }
    if state is not None:
        try:
            disposition = stop_request_disposition(request, state)
            if str(disposition.get("kind") or "") == "unreachable":
                try:
                    disposition = validate_stop_request_for_recovery(
                        campaign,
                        request,
                        state,
                    )
                except Exception as exc:
                    disposition = dict(disposition)
                    disposition["reason"] = str(
                        disposition.get("reason") or exc
                    )
            result["_presentation_stop_disposition"] = disposition
        except Exception as exc:
            result["stop_control_error"] = type(exc).__name__ + ": " + str(exc)
    return result


def _missing_state_context(campaign: Path) -> Dict[str, Any]:
    artefacts = stateful_campaign_artifacts(campaign)
    return {
        "campaign_yaml_exists": bool((campaign / "campaign.yaml").is_file()),
        "stateful_artifacts": [str(item) for item in artefacts],
        "stateful_artifacts_count": int(len(artefacts)),
        "fresh_init_safe": bool(not artefacts),
    }


def _probe_daemon_lock(lock_path: Path) -> dict:
    status = {
        "lock_file_exists": lock_path.exists(),
        "lock_held": False,
    }
    if not lock_path.exists():
        return status
    try:
        import portalocker
    except Exception as exc:
        status["lock_held"] = None
        status["lock_probe_error"] = type(exc).__name__ + ": " + str(exc)
        return status

    try:
        with portalocker.Lock(
            str(lock_path),
            mode="a",
            timeout=0,
            fail_when_locked=True,
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        ):
            status["lock_held"] = False
    except (portalocker.AlreadyLocked, portalocker.LockException):
        status["lock_held"] = True
    except Exception as exc:
        status["lock_held"] = None
        status["lock_probe_error"] = type(exc).__name__ + ": " + str(exc)
    return status


@contextmanager
def _exclusive_operator_lock(campaign: Path):
    """Hold the daemon lock across an authoritative user mutation."""
    import portalocker

    lock_path = operational_path(campaign, DAEMON_LOCK_FILENAME)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = portalocker.Lock(
        str(lock_path),
        mode="a+",
        timeout=0,
        fail_when_locked=True,
        flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
    )
    try:
        lock.acquire()
    except (portalocker.AlreadyLocked, portalocker.LockException) as exc:
        raise RuntimeError(
            "daemon lock is held; refusing concurrent user mutation"
        ) from exc
    try:
        yield
    finally:
        try:
            lock.release()
        except Exception:
            pass


def _runtime_liveness_policy(campaign: Path) -> Tuple[int, int]:
    try:
        config = CampaignConfig.from_yaml(Path(campaign) / "campaign.yaml")
        return (
            int(config.runtime.lease_stale_seconds),
            int(config.runtime.clock_skew_tolerance_seconds),
        )
    except Exception:
        defaults = CampaignConfig()
        return (
            int(defaults.runtime.lease_stale_seconds),
            int(defaults.runtime.clock_skew_tolerance_seconds),
        )


def _runtime_scheduler_policy(campaign: Path) -> Tuple[int, int]:
    """Return bounded scheduler command and cancellation timeouts."""
    try:
        config = CampaignConfig.from_yaml(Path(campaign) / "campaign.yaml")
    except Exception:
        config = CampaignConfig()
    return (
        int(config.runtime.scheduler_command_timeout_seconds),
        int(config.runtime.cancellation_confirmation_timeout_seconds),
    )


def _call_timeout_aware(
    function: Any,
    *args: Any,
    timeout_seconds: int,
    **kwargs: Any,
) -> Any:
    """Preserve simple injected test adapters while timing real commands."""
    try:
        signature = inspect.signature(function)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        accepts_timeout = (
            "timeout_seconds" in signature.parameters or accepts_kwargs
        )
    except (TypeError, ValueError):
        signature = None
        accepts_kwargs = True
        accepts_timeout = True
    if accepts_timeout:
        kwargs["timeout_seconds"] = int(timeout_seconds)
    if signature is not None and not accepts_kwargs:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
    return function(*args, **kwargs)


def _probe_daemon_lease(
    lease_path: Path,
    *,
    stale_seconds: int = 900,
    clock_skew_tolerance_seconds: int = 60,
) -> dict:
    status = {
        "lease_dir_exists": lease_path.is_dir(),
        "lease_path": str(lease_path),
        "lease_heartbeat": None,
        "lease_stale_seconds": int(stale_seconds),
        "clock_skew_tolerance_seconds": int(clock_skew_tolerance_seconds),
    }
    heartbeat_path = lease_path / DAEMON_HEARTBEAT_FILENAME
    if not heartbeat_path.is_file():
        return status
    try:
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        status["lease_heartbeat"] = validate_lease_heartbeat(heartbeat)
        liveness = evaluate_lease_liveness(
            heartbeat,
            stale_seconds=int(stale_seconds),
            clock_skew_tolerance_seconds=int(clock_skew_tolerance_seconds),
        )
        status["lease_liveness"] = liveness.disposition
        status["lease_fresh"] = bool(liveness.fresh)
        status["lease_stale"] = bool(liveness.stale)
        status["lease_age_seconds"] = liveness.age_seconds
        if liveness.error:
            status["lease_probe_error"] = liveness.error
    except Exception as exc:
        status["lease_probe_error"] = type(exc).__name__ + ": " + str(exc)
    return status


def _daemon_control_ownership(
    state: CampaignState,
    lock_status: Mapping[str, Any],
    lease_status: Mapping[str, Any],
) -> Tuple[str, str]:
    """Classify whether a live daemon safely owns the control plane."""
    lock_held = lock_status.get("lock_held")
    heartbeat = lease_status.get("lease_heartbeat")
    lease_fresh = lease_status.get("lease_fresh") is True
    lease_uid = (
        str(heartbeat.get("campaign_uid") or "")
        if isinstance(heartbeat, Mapping)
        else ""
    )
    matching_lease = lease_fresh and lease_uid == str(state.campaign_uid)
    if lock_held is True and matching_lease:
        return "active", "daemon lock held with a fresh matching lease"
    if lock_held is False and not lease_fresh:
        return "inactive", "daemon lock is free and no fresh lease exists"
    if lock_held is True and not lease_fresh:
        return (
            "inconclusive",
            "daemon lock is held but no fresh authenticated lease is available",
        )
    if lock_held is False and lease_fresh:
        return (
            "inconclusive",
            "a fresh daemon lease exists without matching lock ownership",
        )
    return (
        "inconclusive",
        str(lock_status.get("lock_probe_error") or "daemon lock is inconclusive"),
    )


def _cancel_running_boundary_stop_request(
    campaign: Path,
    paths: Mapping[str, Path],
    state: CampaignState,
) -> int:
    from .daemon.stop_control import (
        StopControlError,
        cancel_pending_boundary_stop_request,
        describe_stop_request,
        read_stop_request,
    )

    try:
        request = read_stop_request(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        if request is None:
            print(
                "no pending boundary stop request is active; the daemon "
                "remains running"
            )
            return 0
        expected_request_id = str(request.get("request_id"))
        cancelled, archive_path, changed = (
            cancel_pending_boundary_stop_request(
                campaign,
                expected_campaign_uid=str(state.campaign_uid),
                expected_request_id=expected_request_id,
            )
        )
    except (OSError, StopControlError) as exc:
        print("stop request was not cancelled: " + str(exc), file=sys.stderr)
        return 7
    if cancelled is None:
        print("stop request was not cancelled because it no longer exists")
        return 0
    try:
        journal_state = read_state(paths["state"])
    except Exception:
        journal_state = state
    if changed:
        try:
            from .daemon.journal import append_event

            append_event(
                paths["journal"],
                "user_stop_request_cancelled",
                phase=journal_state.phase.value,
                iteration=int(journal_state.iteration),
                request_id=str(cancelled.get("request_id")),
                mode=str(cancelled.get("mode")),
                target_phase=cancelled.get("target_phase"),
                target_iteration=cancelled.get("target_iteration"),
                target_replacement_round=cancelled.get(
                    "target_replacement_round"
                ),
                daemon_was_running=True,
                archive_path=(
                    None if archive_path is None else str(archive_path)
                ),
            )
        except Exception:
            pass
    print("cancelled: " + describe_stop_request(cancelled))
    print("the existing daemon remains running; no campaign work was restarted")
    return 0


def _pid_is_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_background_pid_payload(pid_path: Path) -> Dict[str, Any]:
    if not pid_path.is_file():
        return {}
    text = pid_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            return {"pid": int(text)}
        except Exception:
            return {"pid_parse_error": "not_json_or_int"}
    return payload if isinstance(payload, dict) else {}


def _probe_background_daemon(
    pid_path: Path,
    log_path: Path,
    startup_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = _read_background_pid_payload(pid_path)
    startup_path = (
        Path(startup_path)
        if startup_path is not None
        else pid_path.with_name(BACKGROUND_STARTUP_FILENAME)
    )
    startup = read_background_startup(startup_path)
    pid = payload.get("pid")
    if pid is None:
        pid = startup.get("pid")
    alive = _pid_is_alive(pid)
    startup_state = str(startup.get("state") or "") or None
    observed_state = startup_state
    startup_failure = startup.get("failure")
    if (
        startup_state in BACKGROUND_STARTUP_ACTIVE_STATES
        and pid is not None
        and not alive
    ):
        observed_state = "failed"
        startup_failure = startup_failure or (
            "background child is no longer alive before startup completed"
        )
    out = {
        "background_pid_path": str(pid_path),
        "background_log_path": str(
            payload.get("log_path") or startup.get("log_path") or log_path
        ),
        "background_pid": pid,
        "background_pid_alive": alive,
        "background_startup_path": str(startup_path),
        "background_startup_state": observed_state,
        "background_startup_recorded_state": startup_state,
        "background_startup_stage": startup.get("stage"),
        "background_startup_failure": startup_failure,
        "background_startup_started_at": startup.get("started_at_iso"),
        "background_startup_ownership_acquired_at": startup.get(
            "ownership_acquired_at_iso"
        ),
        "background_startup_ready_at": startup.get("ready_at_iso"),
    }
    if payload:
        out["background_pid_payload"] = payload
    if startup:
        out["background_startup_payload"] = startup
    return out


def _background_probe_is_current_startup_child(
    campaign: Path,
    probe: Mapping[str, Any],
) -> bool:
    """Recognise only this process's authenticated launcher handshake."""
    if os.environ.get(BACKGROUND_CHILD_ENV) != "1":
        return False
    launch_id = launch_id_from_environment()
    if (
        len(launch_id) != 32
        or any(character not in "0123456789abcdef" for character in launch_id)
    ):
        return False
    startup_path = startup_path_from_environment()
    expected_path = _campaign_paths(Path(campaign).resolve())["background_startup"]
    if startup_path is None:
        return False
    try:
        if Path(startup_path).resolve() != expected_path.resolve():
            return False
    except OSError:
        return False
    startup = probe.get("background_startup_payload")
    if not isinstance(startup, Mapping):
        return False
    try:
        recorded_pid = int(startup.get("pid"))
        observed_pid = int(probe.get("background_pid"))
    except (TypeError, ValueError):
        return False
    recorded_campaign_raw = startup.get("campaign_dir")
    if (
        not isinstance(recorded_campaign_raw, str)
        or not recorded_campaign_raw.strip()
        or not Path(recorded_campaign_raw).is_absolute()
    ):
        return False
    try:
        recorded_campaign = Path(recorded_campaign_raw).resolve()
    except (OSError, ValueError):
        return False
    return bool(
        isinstance(startup.get("schema_version"), int)
        and not isinstance(startup.get("schema_version"), bool)
        and startup.get("schema_version") == BACKGROUND_STARTUP_SCHEMA_VERSION
        and str(startup.get("launch_id") or "") == launch_id
        and str(startup.get("state") or "") in BACKGROUND_STARTUP_ACTIVE_STATES
        and recorded_pid == int(os.getpid())
        and observed_pid == int(os.getpid())
        and probe.get("background_pid_alive") is True
        and recorded_campaign == Path(campaign).resolve()
    )


def _lease_is_fresh(
    heartbeat: Any,
    *,
    stale_seconds: int = 900,
    clock_skew_tolerance_seconds: int = 60,
) -> bool:
    return evaluate_lease_liveness(
        heartbeat,
        stale_seconds=int(stale_seconds),
        clock_skew_tolerance_seconds=int(clock_skew_tolerance_seconds),
    ).fresh


def _reconcile_runtime_status(
    campaign: Path,
    *,
    operator_lock_owned: bool = False,
) -> Dict[str, Any]:
    paths = _campaign_paths(campaign)
    stale_seconds, clock_skew = _runtime_liveness_policy(campaign)
    status: Dict[str, Any] = {}
    if operator_lock_owned:
        status.update(
            {
                "lock_file_exists": paths["lock"].exists(),
                "lock_held": False,
                "user_lock_owned": True,
            }
        )
    else:
        status.update(_probe_daemon_lock(paths["lock"]))
    status.update(
        _probe_daemon_lease(
            paths["lease"],
            stale_seconds=stale_seconds,
            clock_skew_tolerance_seconds=clock_skew,
        )
    )
    status.update(
        _probe_background_daemon(
            paths["background_pid"],
            paths["background_log"],
            paths["background_startup"],
        )
    )
    blockers: List[str] = []
    if status.get("lock_held") is True:
        blockers.append("daemon lock is held")
    elif status.get("lock_held") is None:
        blockers.append(
            "daemon lock state is unknown: "
            + str(status.get("lock_probe_error", "unknown error"))
        )
    if status.get("lease_probe_error"):
        blockers.append(
            "daemon lease state is unknown: "
            + str(status.get("lease_probe_error"))
        )
    elif _lease_is_fresh(
        status.get("lease_heartbeat"),
        stale_seconds=stale_seconds,
        clock_skew_tolerance_seconds=clock_skew,
    ):
        blockers.append("daemon lease heartbeat is fresh")
    if status.get("background_pid_alive"):
        blockers.append(
            "background daemon pid "
            + str(status.get("background_pid"))
            + " appears alive"
        )
    expected_uid = None
    try:
        expected_uid = str(read_state(paths["state"]).campaign_uid)
    except Exception:
        pass
    status.update(
        _stop_control_status(
            campaign,
            expected_campaign_uid=expected_uid,
        )
    )
    if status.get("stop_control_error"):
        blockers.append(
            "stop-control metadata is invalid: "
            + str(status.get("stop_control_error"))
        )
    status["reconcile_apply_blockers"] = blockers
    return status


def _deep_reconcile_quiescence_blockers(campaign: Path) -> List[str]:
    blockers: List[str] = []
    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    try:
        state = read_state(state_path)
    except (FileNotFoundError, StateSchemaError, ValueError):
        state = None
        if state_path.is_file():
            try:
                raw_state = json.loads(state_path.read_text(encoding="utf-8"))
                raw_pending = raw_state.get("pending_jobs")
            except Exception:
                raw_pending = None
            if isinstance(raw_pending, dict) and any(
                value is not None and str(value)
                for value in raw_pending.values()
            ):
                blockers.append(
                    "malformed state still records scheduler ownership"
                )
    if state is not None:
        pending = [
            str(phase) + "=" + str(job_id)
            for phase, job_id in state.pending_jobs.items()
            if job_id is not None and str(job_id)
        ]
        if pending:
            blockers.append(
                "state records scheduler ownership: " + ", ".join(pending[:8])
            )
    intent_errors: List[Dict[str, str]] = []
    active_intents = _load_active_submission_intents(
        campaign,
        errors=intent_errors,
        state=state,
    )
    if active_intents:
        blockers.append(
            "active or scheduler-inconclusive submission intents are present"
        )
    if intent_errors:
        blockers.append("submission-intent ownership is malformed or inconclusive")
    try:
        from .daemon.reconcile_transaction import inventory_reconcile_transactions

        transactions = inventory_reconcile_transactions(campaign)
    except Exception as exc:
        blockers.append(
            "reconcile transaction inventory is inconclusive: "
            + type(exc).__name__
            + ": "
            + str(exc)[:120]
        )
    else:
        if any(
            str(record.get("status") or "") not in {"COMMITTED", "FAILED"}
            for record in transactions
        ):
            blockers.append("an incomplete reconcile transaction is present")
    return blockers


def _print_reconcile_runtime_warning(status: Dict[str, Any], campaign: Path) -> None:
    blockers = list(status.get("reconcile_apply_blockers") or [])
    if not blockers:
        return
    daemon_blockers = [
        blocker
        for blocker in blockers
        if not str(blocker).startswith("stop-control metadata is invalid:")
    ]
    if daemon_blockers:
        print("WARNING: a daemon may still be running for this campaign.", file=sys.stderr)
        for blocker in daemon_blockers:
            print("  - " + str(blocker), file=sys.stderr)
        print("Lock: " + _lock_summary(status.get("lock_held")), file=sys.stderr)
        print("Lease: " + _heartbeat_summary(status.get("lease_heartbeat")), file=sys.stderr)
        print(
            "Background pid: "
            + str(status.get("background_pid"))
            + " alive="
            + str(status.get("background_pid_alive")),
            file=sys.stderr,
        )
        print(
            "Do not apply recovery while the daemon is active. If this daemon should stop, run:",
            file=sys.stderr,
        )
        print(
            "  ichor-al-daemon stop --campaign-dir "
            + str(campaign)
            + " --cancel-jobs",
            file=sys.stderr,
        )
    if status.get("stop_control_error"):
        print(
            "WARNING: user stop control is invalid: "
            + str(status.get("stop_control_error")),
            file=sys.stderr,
        )


def _read_last_exception_summary(campaign: Path) -> str:
    path = campaign / DEFAULT_DATA_SUBDIR / "LAST_EXCEPTION.json"
    if not path.is_file():
        return "none"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "unreadable " + type(exc).__name__ + ": " + str(exc)[:100]
    if not isinstance(payload, dict):
        return "unreadable: payload is not an object"
    parts = [
        str(payload.get("timestamp") or "?"),
        str(payload.get("exception_type") or "?"),
        str(payload.get("message") or "")[:120],
    ]
    phase = payload.get("phase")
    iteration = payload.get("iteration")
    if phase is not None:
        parts.append("phase=" + str(phase))
    if iteration is not None:
        parts.append("iteration=" + str(iteration))
    return " | ".join(parts)


def _format_staging_inventory_summary(inventory: Dict[str, Any]) -> str:
    if not inventory.get("exists"):
        return "missing"
    if inventory.get("is_symlink"):
        return "unsafe symlink"
    if not inventory.get("is_dir"):
        return "not a directory"
    top_level_count = int(inventory.get("top_level_count") or 0)
    if top_level_count <= 0:
        return "empty"
    sample = ", ".join(str(x) for x in list(inventory.get("top_level_entries") or [])[:5])
    suffix = ""
    if sample:
        suffix = " sample=[" + sample + "]"
    if inventory.get("has_symlink"):
        suffix += " has_symlink=true"
    return (
        "non-empty top_level="
        + str(top_level_count)
        + " total_entries="
        + str(inventory.get("total_entries"))
        + " total_bytes="
        + str(inventory.get("total_bytes"))
        + suffix
    )


def _operator_staging_archive_blockers(
    campaign: Path,
    report: ReconciliationReport,
    runtime_status: Dict[str, Any],
) -> List[str]:
    blockers = list(runtime_status.get("reconcile_apply_blockers") or [])
    if report.active_submission_intents:
        blockers.append("active submission intent(s) are present")
    try:
        state = read_state(_campaign_paths(campaign)["state"])
        pending = {
            str(phase): str(job_id)
            for phase, job_id in (state.pending_jobs or {}).items()
            if job_id
        }
        if pending:
            blockers.append("state.json still has pending job(s): " + repr(pending))
    except FileNotFoundError:
        pass
    except StateSchemaError:
        # Reconcile has already proposed a conservative recovery state; do not
        # make a corrupt old state an absolute blocker once the user has
        # explicitly requested an archive through --apply.
        pass
    inventory = data_staging_inventory(campaign)
    if inventory.get("is_symlink"):
        blockers.append(".DATA/STAGING is a symlink")
    if inventory.get("has_symlink"):
        blockers.append(".DATA/STAGING contains symlink entries")
    try:
        protected_handoffs = staging_handoff_decisions(
            campaign,
            report.proposed_state,
        )
    except Exception:
        protected_handoffs = []
    for protected in protected_handoffs:
        blockers.append(
            ".DATA/STAGING contains a valid protected handoff for "
            + protected.phase.value
            + "@"
            + str(int(protected.iteration))
            + ": "
            + str(protected.trusted_artifact)
        )
    if inventory.get("error"):
        blockers.append(".DATA/STAGING inventory error: " + str(inventory.get("error")))
    return blockers


def _timestamped_sibling(path: Path, marker: str) -> Path:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = path.with_name(path.name + marker + stamp)
    suffix = 1
    while target.exists():
        target = path.with_name(path.name + marker + stamp + "." + str(suffix))
        suffix += 1
    return target


def _copy_existing_timestamped(path: Path, marker: str) -> Optional[Path]:
    if not path.exists():
        return None
    import shutil

    target = _timestamped_sibling(path, marker)
    shutil.copy2(path, target)
    return target


def _restore_state_from_backup_atomic(
    target: Path,
    backup_path: Optional[Path],
) -> None:
    if backup_path is None:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        return
    restored_state = read_state(backup_path)
    write_state(target, restored_state)


def _rename_existing_timestamped(path: Path, marker: str) -> Optional[Path]:
    if not path.exists():
        return None
    target = _timestamped_sibling(path, marker)
    path.rename(target)
    return target


def _print_cleanup_already_happened(paths: Sequence[str]) -> None:
    if not paths:
        return
    print("Apply result: partially completed", file=sys.stderr)
    print("", file=sys.stderr)
    print("Completed before the interruption", file=sys.stderr)
    for path in paths:
        print("  - " + str(path), file=sys.stderr)
    print("  campaign state: previous authoritative state retained", file=sys.stderr)
    print("  config lock: unchanged", file=sys.stderr)


def _atomic_write_background_pid(pid_path: Path, payload: Dict[str, Any]) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pid_path.with_name(pid_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(pid_path)


def _report_background_startup(
    state: str,
    stage: str,
    *,
    failure: Optional[str] = None,
    **details: Any,
) -> Optional[Dict[str, Any]]:
    """Best-effort child update of the durable launch-status record."""
    startup_path = startup_path_from_environment()
    if startup_path is None:
        return None
    launch_id = launch_id_from_environment()
    if not launch_id:
        launch_id = str(read_background_startup(startup_path).get("launch_id") or "")
    if not launch_id:
        return None
    updates: Dict[str, Any] = {
        "state": str(state),
        "stage": str(stage),
        "pid": int(os.getpid()),
    }
    if failure is not None:
        updates["failure"] = str(failure)
    updates.update(details)
    try:
        return update_background_startup(startup_path, launch_id, **updates)
    except Exception:
        # Startup telemetry must never become a new daemon-start blocker.
        return None


def _finish_background_child_status(
    return_code: int,
    *,
    failure: Optional[str] = None,
) -> None:
    """Record command-level startup failure when the daemon never took over."""
    startup_path = startup_path_from_environment()
    if startup_path is None:
        return
    payload = read_background_startup(startup_path)
    state = str(payload.get("state") or "")
    if state in {"ready", "failed", "stopped"}:
        return
    if int(return_code) != 0 or failure is not None:
        _report_background_startup(
            "failed",
            "command_exit",
            failure=(
                failure
                or "background daemon command exited before startup acknowledgement "
                + "with code "
                + str(return_code)
            ),
            exit_code=int(return_code),
        )
    elif state not in BACKGROUND_STARTUP_ACKNOWLEDGED_STATES:
        _report_background_startup(
            "failed",
            "command_exit",
            failure="background daemon command exited before startup acknowledgement",
            exit_code=int(return_code),
        )


def _default_background_path(raw: Optional[str], default_path: Path) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    return default_path.resolve()


def _background_child_argv(args: argparse.Namespace, campaign: Path) -> List[str]:
    command = str(getattr(args, "command", "start"))
    argv = [
        sys.executable,
        "-m",
        "ichor.hpc.active_learning.cli",
        command,
        "--campaign-dir",
        str(campaign),
    ]
    config = getattr(args, "config", None)
    if config:
        argv.extend(["--config", str(Path(config).expanduser().resolve())])
    if getattr(args, "mode", None):
        argv.extend(["--mode", str(args.mode)])
    if getattr(args, "poll_interval", None) is not None:
        argv.extend(["--poll-interval", str(int(args.poll_interval))])
    if getattr(args, "max_ticks", None) is not None:
        argv.extend(["--max-ticks", str(int(args.max_ticks))])
    return argv


def _tail_text(path: Path, *, max_lines: int = 40) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return "<could not read log: " + type(exc).__name__ + ": " + str(exc) + ">"
    return "\n".join(lines[-max_lines:])


def _launch_background_daemon(args: argparse.Namespace, campaign: Path) -> int:
    if os.environ.get(BACKGROUND_CHILD_ENV) == "1":
        print("--background is not allowed inside a background child process", file=sys.stderr)
        return 2
    paths = _campaign_paths(campaign)
    paths["data"].mkdir(parents=True, exist_ok=True)
    lock_status = _probe_daemon_lock(paths["lock"])
    if lock_status.get("lock_held") is True:
        print("refusing background launch: daemon lock is already held", file=sys.stderr)
        return 8
    if lock_status.get("lock_held") is None:
        print(
            "refusing background launch: daemon lock state is unknown: "
            + str(lock_status.get("lock_probe_error", "")),
            file=sys.stderr,
        )
        return 8

    log_path = _default_background_path(
        getattr(args, "background_log", None),
        paths["background_log"],
    )
    pid_path = _default_background_path(
        getattr(args, "background_pid", None),
        paths["background_pid"],
    )
    startup_path = paths["background_startup"]
    existing = _probe_background_daemon(pid_path, log_path, startup_path)
    if existing.get("background_pid_alive"):
        print(
            "refusing background launch: daemon pid "
            + str(existing.get("background_pid"))
            + " from "
            + str(pid_path)
            + " is still alive",
            file=sys.stderr,
        )
        return 8
    if pid_path.exists() and existing.get("background_pid") is not None:
        print("stale daemon pid file found; replacing " + str(pid_path))

    argv = _background_child_argv(args, campaign)
    env = os.environ.copy()
    env[BACKGROUND_CHILD_ENV] = "1"
    launch_id = secrets.token_hex(16)
    env[BACKGROUND_LAUNCH_ID_ENV] = launch_id
    env[BACKGROUND_STARTUP_PATH_ENV] = str(startup_path)
    # Retain the old environment variable for one release so an independently
    # installed CLI frontend can still locate the durable startup record.
    env[BACKGROUND_READINESS_ENV] = str(startup_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    header = (
        "\n===== ichor-al-daemon background launch "
        + timestamp
        + " command="
        + " ".join(argv)
        + " =====\n"
    )
    initialise_background_startup(
        startup_path,
        {
            "schema_version": BACKGROUND_STARTUP_SCHEMA_VERSION,
            "launch_id": launch_id,
            "state": "prepared",
            "stage": "launcher",
            "campaign_dir": str(campaign),
            "command": argv,
            "log_path": str(log_path),
            "host": os.uname().nodename if hasattr(os, "uname") else "",
            "python_executable": sys.executable,
            "launcher_pid": int(os.getpid()),
        },
    )
    try:
        with open(os.devnull, "rb") as stdin_fh, open(log_path, "a", encoding="utf-8") as log_fh:
            log_fh.write(header)
            log_fh.flush()
            child = subprocess.Popen(
                argv,
                stdin=stdin_fh,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
    except Exception as exc:
        try:
            update_background_startup(
                startup_path,
                launch_id,
                state="failed",
                stage="process_spawn",
                failure=type(exc).__name__ + ": " + str(exc),
            )
        except Exception:
            pass
        print(
            "could not launch background daemon: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=sys.stderr,
        )
        return 9

    update_background_startup(
        startup_path,
        launch_id,
        state="spawned",
        stage="child_process",
        pid=int(child.pid),
    )
    if pid_path.resolve() != paths["background_pid"].resolve():
        _atomic_write_background_pid(
            pid_path,
            {
                "schema_version": BACKGROUND_PID_SCHEMA_VERSION,
                "pid": int(child.pid),
                "launch_id": launch_id,
                "campaign_dir": str(campaign),
                "log_path": str(log_path),
                "startup_path": str(startup_path),
            },
        )

    timeout_seconds = 60
    try:
        config_path = (
            Path(args.config).expanduser().resolve()
            if getattr(args, "config", None)
            else campaign / "campaign.yaml"
        )
        timeout_seconds = int(
            CampaignConfig.from_yaml(config_path).runtime.background_readiness_timeout_seconds
        )
    except Exception:
        pass
    deadline = time.monotonic() + float(timeout_seconds)
    startup_payload: Dict[str, Any] = {}
    acknowledged = False
    while time.monotonic() < deadline:
        startup_payload = read_background_startup(startup_path)
        state_name = str(startup_payload.get("state") or "")
        if str(startup_payload.get("launch_id") or "") == launch_id:
            if state_name in BACKGROUND_STARTUP_ACKNOWLEDGED_STATES:
                acknowledged = True
                break
            if state_name == "stopped" and startup_payload.get("ready_at_iso"):
                acknowledged = True
                break
            if state_name == "failed":
                print(
                    "background daemon startup failed; log: " + str(log_path),
                    file=sys.stderr,
                )
                failure = startup_payload.get("failure")
                if failure:
                    print(str(failure), file=sys.stderr)
                tail = _tail_text(log_path)
                if tail:
                    print(tail, file=sys.stderr)
                return 9
        rc = child.poll()
        if rc is not None:
            startup_payload = read_background_startup(startup_path)
            if startup_payload.get("ready_at_iso"):
                acknowledged = True
                break
            try:
                update_background_startup(
                    startup_path,
                    launch_id,
                    state="failed",
                    stage="process_exit",
                    exit_code=int(rc),
                    failure=(
                        "background daemon exited before startup acknowledgement "
                        + "with code "
                        + str(rc)
                    ),
                )
            except Exception:
                pass
            print(
                "background daemon exited during startup with code "
                + str(rc)
                + "; log: "
                + str(log_path),
                file=sys.stderr,
            )
            tail = _tail_text(log_path)
            if tail:
                print(tail, file=sys.stderr)
            return int(rc) if int(rc) != 0 else 9
        time.sleep(0.1)

    if not acknowledged:
        rc = child.poll()
        if rc is not None:
            try:
                update_background_startup(
                    startup_path,
                    launch_id,
                    state="failed",
                    stage="process_exit",
                    exit_code=int(rc),
                    failure=(
                        "background daemon exited before startup acknowledgement "
                        + "with code "
                        + str(rc)
                    ),
                )
            except Exception:
                pass
            print(
                "background daemon exited during startup with code "
                + str(rc)
                + "; log: "
                + str(log_path),
                file=sys.stderr,
            )
            return int(rc) if int(rc) != 0 else 9
        startup_payload = read_background_startup(startup_path)
        print("daemon launch continues in background")
        print(
            "  startup: acknowledgement still pending after "
            + str(timeout_seconds)
            + " seconds; the live child was not terminated"
        )
    else:
        startup_state = str(startup_payload.get("state") or "unknown")
        if startup_state == "ready" or startup_payload.get("ready_at_iso"):
            print("daemon started successfully in the background")
        else:
            print("background process owns the campaign and is still starting")
        print("  startup state: " + startup_state)
    print("  pid: " + str(child.pid))
    print("  status: " + _campaign_command(campaign, "status"))
    print("  journal: " + _campaign_command(campaign, "journal", " --last-n 40"))
    return 0


def _format_value(value: Any) -> str:
    if value is None:
        return "none"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value) if value else "none"
    return str(value)


def _section(title: str, rows: Iterable[tuple[str, Any]]) -> List[str]:
    lines = [title]
    for key, value in rows:
        lines.append("  " + key + ": " + _format_value(value))
    return lines


def _heartbeat_summary(heartbeat: Any) -> str:
    if not isinstance(heartbeat, dict):
        return "none"
    parts = []
    host = heartbeat.get("host")
    pid = heartbeat.get("pid")
    phase = heartbeat.get("phase")
    iteration = heartbeat.get("iteration")
    if host:
        parts.append(str(host))
    if pid is not None:
        parts.append("pid " + str(pid))
    if phase:
        parts.append("phase " + str(phase))
    if iteration is not None:
        parts.append("iter " + str(iteration))
    try:
        age = max(0.0, time.time() - float(heartbeat.get("time")))
        parts.append("age " + str(int(round(age))) + "s")
    except Exception:
        pass
    return ", ".join(parts) if parts else "present"


def _lock_summary(lock_held: Any) -> str:
    if lock_held is True:
        return "held"
    if lock_held is False:
        return "free"
    return "unknown"


def _format_contract_status(contract: Any) -> Optional[str]:
    if not isinstance(contract, dict):
        return None
    ok = contract.get("ok")
    if ok is True:
        return "ok"
    if ok is False:
        error = str(contract.get("error") or "")
        if len(error) > 180:
            error = error[:177] + "..."
        return "problem" + (" - " + error if error else "")
    return "not checked"


_PHASE_TITLES: Dict[str, str] = {
    CampaignPhase.INIT.value: "Campaign setup",
    CampaignPhase.PHASE_A_DIVERSITY.value: "Bootstrap diversity selection",
    CampaignPhase.INITIAL_GAUSSIAN.value: "Bootstrap Gaussian calculations",
    CampaignPhase.INITIAL_AIMALL.value: "Bootstrap AIMAll calculations",
    CampaignPhase.INITIAL_ALLOCATION_CHECK.value: "Bootstrap allocation check",
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value: "Bootstrap replacement Gaussian calculations",
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value: "Bootstrap replacement AIMAll calculations",
    CampaignPhase.REFERENCE_COMMIT.value: "QM reference-data publication",
    CampaignPhase.INITIAL_FEREBUS.value: "Initial FEREBUS training",
    CampaignPhase.SEED_SELECT.value: "ARIADNE seed selection",
    CampaignPhase.ARIADNE_ARRAY.value: "ARIADNE landing",
    CampaignPhase.PHASE_B_DIVERSITY.value: "Phase B diversity selection",
    CampaignPhase.SPLIT.value: "QM batch allocation",
    CampaignPhase.GAUSSIAN.value: "Gaussian calculations",
    CampaignPhase.AIMALL.value: "AIMAll calculations",
    CampaignPhase.ALLOCATION_CHECK.value: "Point-allocation check",
    CampaignPhase.REPLACEMENT_GAUSSIAN.value: "Replacement Gaussian calculations",
    CampaignPhase.REPLACEMENT_AIMALL.value: "Replacement AIMAll calculations",
    CampaignPhase.FEREBUS.value: "FEREBUS retraining",
    CampaignPhase.STOP_CHECK.value: "Iteration completion check",
    CampaignPhase.DONE.value: "Campaign complete",
    CampaignPhase.HALTED.value: "Campaign halted",
}


_PHASE_MEANINGS: Dict[str, str] = {
    CampaignPhase.INIT.value: "prepare the campaign before its first run",
    CampaignPhase.PHASE_A_DIVERSITY.value: "choose a diverse bootstrap set from the trajectory pool",
    CampaignPhase.INITIAL_GAUSSIAN.value: "calculate Gaussian wavefunctions for the bootstrap set",
    CampaignPhase.INITIAL_AIMALL.value: "analyse bootstrap wavefunctions with AIMAll",
    CampaignPhase.INITIAL_ALLOCATION_CHECK.value: "check whether every bootstrap data slot has a valid result",
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value: "calculate Gaussian replacements for missing bootstrap results",
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value: "analyse replacement bootstrap wavefunctions with AIMAll",
    CampaignPhase.REFERENCE_COMMIT.value: "publish accepted QM data for model training",
    CampaignPhase.INITIAL_FEREBUS.value: "train the first FEREBUS model set",
    CampaignPhase.SEED_SELECT.value: "choose trajectory frames for adversarial sampling",
    CampaignPhase.ARIADNE_ARRAY.value: "generate adversarial geometries from selected seeds",
    CampaignPhase.PHASE_B_DIVERSITY.value: "select a safe, diverse set of ARIADNE results",
    CampaignPhase.SPLIT.value: "assign selected geometries to the next QM batch",
    CampaignPhase.GAUSSIAN.value: "calculate Gaussian wavefunctions for the active-learning batch",
    CampaignPhase.AIMALL.value: "analyse active-learning wavefunctions with AIMAll",
    CampaignPhase.ALLOCATION_CHECK.value: "check whether every active-learning data slot has a valid result",
    CampaignPhase.REPLACEMENT_GAUSSIAN.value: "calculate Gaussian replacements for missing active-learning results",
    CampaignPhase.REPLACEMENT_AIMALL.value: "analyse replacement active-learning wavefunctions with AIMAll",
    CampaignPhase.FEREBUS.value: "train and assess the next FEREBUS model set",
    CampaignPhase.STOP_CHECK.value: "decide whether another active-learning iteration is required",
    CampaignPhase.DONE.value: "campaign is complete",
    CampaignPhase.HALTED.value: "campaign is halted and needs user review",
}


_PHASE_AUTOMATIC_OUTCOMES: Dict[str, str] = {
    CampaignPhase.INIT.value: (
        "the daemon will select a diverse bootstrap set from the trajectory pool"
    ),
    CampaignPhase.PHASE_A_DIVERSITY.value: (
        "after bootstrap diversity selection, the daemon will submit the initial "
        "Gaussian calculations"
    ),
    CampaignPhase.INITIAL_GAUSSIAN.value: (
        "after Gaussian finishes, the daemon will submit AIMAll analysis for the "
        "bootstrap points"
    ),
    CampaignPhase.INITIAL_AIMALL.value: (
        "after AIMAll finishes, the daemon will validate the bootstrap results and "
        "check whether replacements are needed"
    ),
    CampaignPhase.INITIAL_ALLOCATION_CHECK.value: (
        "the daemon will publish the bootstrap QM data when every slot is complete, "
        "or allocate replacements for missing slots"
    ),
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value: (
        "after replacement Gaussian calculations finish, the daemon will submit "
        "replacement AIMAll analysis"
    ),
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value: (
        "after replacement AIMAll analysis, the daemon will check the bootstrap "
        "allocation again"
    ),
    CampaignPhase.REFERENCE_COMMIT.value: (
        "after publishing the accepted QM data, the daemon will train the next "
        "FEREBUS model"
    ),
    CampaignPhase.INITIAL_FEREBUS.value: (
        "after the initial model passes its quality checks, the daemon will begin "
        "iteration 1 seed selection"
    ),
    CampaignPhase.SEED_SELECT.value: (
        "after seed selection, the daemon will submit ARIADNE landing calculations"
    ),
    CampaignPhase.ARIADNE_ARRAY.value: (
        "after ARIADNE finishes, the daemon will validate the results and begin "
        "Phase B diversity selection"
    ),
    CampaignPhase.PHASE_B_DIVERSITY.value: (
        "after Phase B selection, the daemon will allocate the next QM batch"
    ),
    CampaignPhase.SPLIT.value: (
        "after allocating the QM batch, the daemon will submit Gaussian calculations"
    ),
    CampaignPhase.GAUSSIAN.value: (
        "after Gaussian finishes, the daemon will submit AIMAll analysis for the "
        "active-learning batch"
    ),
    CampaignPhase.AIMALL.value: (
        "after AIMAll finishes, the daemon will validate the batch and check whether "
        "replacements are needed"
    ),
    CampaignPhase.ALLOCATION_CHECK.value: (
        "the daemon will publish the iteration's QM data when every slot is complete, "
        "or allocate replacements for missing slots"
    ),
    CampaignPhase.REPLACEMENT_GAUSSIAN.value: (
        "after replacement Gaussian calculations finish, the daemon will submit "
        "replacement AIMAll analysis"
    ),
    CampaignPhase.REPLACEMENT_AIMALL.value: (
        "after replacement AIMAll analysis, the daemon will check the iteration's "
        "allocation again"
    ),
    CampaignPhase.FEREBUS.value: (
        "after the updated model passes its quality checks, the daemon will finalise "
        "the iteration"
    ),
    CampaignPhase.STOP_CHECK.value: (
        "the daemon will finalise the iteration, then either begin the next iteration "
        "or complete the campaign"
    ),
    CampaignPhase.DONE.value: "no further daemon work is scheduled",
    CampaignPhase.HALTED.value: "no further work will run until the halt is reviewed",
}


_PHASE_WORK_NAMES: Dict[str, str] = {
    CampaignPhase.PHASE_A_DIVERSITY.value: "bootstrap diversity",
    CampaignPhase.INITIAL_GAUSSIAN.value: "Gaussian",
    CampaignPhase.INITIAL_AIMALL.value: "AIMAll",
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value: "replacement Gaussian",
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value: "replacement AIMAll",
    CampaignPhase.INITIAL_FEREBUS.value: "FEREBUS",
    CampaignPhase.ARIADNE_ARRAY.value: "ARIADNE",
    CampaignPhase.PHASE_B_DIVERSITY.value: "Phase B diversity",
    CampaignPhase.GAUSSIAN.value: "Gaussian",
    CampaignPhase.AIMALL.value: "AIMAll",
    CampaignPhase.REPLACEMENT_GAUSSIAN.value: "replacement Gaussian",
    CampaignPhase.REPLACEMENT_AIMALL.value: "replacement AIMAll",
    CampaignPhase.FEREBUS.value: "FEREBUS",
}


_BOOTSTRAP_COLLECTION_PHASES = {
    CampaignPhase.INIT.value,
    CampaignPhase.PHASE_A_DIVERSITY.value,
    CampaignPhase.INITIAL_GAUSSIAN.value,
    CampaignPhase.INITIAL_AIMALL.value,
    CampaignPhase.INITIAL_ALLOCATION_CHECK.value,
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value,
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value,
}

_BOOTSTRAP_NOT_READY_PHASES = {
    CampaignPhase.INIT.value,
    CampaignPhase.PHASE_A_DIVERSITY.value,
    CampaignPhase.INITIAL_GAUSSIAN.value,
    CampaignPhase.INITIAL_AIMALL.value,
    CampaignPhase.INITIAL_ALLOCATION_CHECK.value,
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value,
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value,
    CampaignPhase.REFERENCE_COMMIT.value,
}

_TRAINING_REQUIRED_PHASES = {
    CampaignPhase.SEED_SELECT.value,
    CampaignPhase.ARIADNE_ARRAY.value,
    CampaignPhase.PHASE_B_DIVERSITY.value,
    CampaignPhase.SPLIT.value,
    CampaignPhase.GAUSSIAN.value,
    CampaignPhase.AIMALL.value,
    CampaignPhase.ALLOCATION_CHECK.value,
    CampaignPhase.REPLACEMENT_GAUSSIAN.value,
    CampaignPhase.REPLACEMENT_AIMALL.value,
    CampaignPhase.FEREBUS.value,
    CampaignPhase.STOP_CHECK.value,
}

_MODELS_REQUIRED_PHASES = {
    CampaignPhase.SEED_SELECT.value,
    CampaignPhase.ARIADNE_ARRAY.value,
    CampaignPhase.PHASE_B_DIVERSITY.value,
    CampaignPhase.SPLIT.value,
    CampaignPhase.GAUSSIAN.value,
    CampaignPhase.AIMALL.value,
    CampaignPhase.ALLOCATION_CHECK.value,
    CampaignPhase.REPLACEMENT_GAUSSIAN.value,
    CampaignPhase.REPLACEMENT_AIMALL.value,
    CampaignPhase.STOP_CHECK.value,
}


def _phase_meaning(phase: Any) -> str:
    return _PHASE_MEANINGS.get(str(phase or ""), "phase is not recognised")


def _phase_title(phase: Any) -> str:
    return _PHASE_TITLES.get(str(phase or ""), str(phase or "Unknown phase"))


def _sentence_fragment(value: str) -> str:
    if not value or (len(value) > 1 and value[:2].isupper()):
        return value
    return value[:1].lower() + value[1:]


def _short_status_error(errors: Iterable[Any], *, limit: int = 180) -> str:
    for error in errors:
        text = str(error)
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text
    return ""


def _product_ok_text(item: Dict[str, Any], *, label: str) -> str:
    ok = item.get("ok")
    if ok is True:
        version_int = _status_version(item)
        if version_int >= 0:
            return "ready (v" + str(version_int) + ")"
        return "not produced yet"
    if ok is False:
        error = _short_status_error(item.get("errors") or [])
        return "problem" + (" - " + error if error else "")
    return label + " status not checked"


def _status_version(item: Dict[str, Any]) -> int:
    try:
        return int(item.get("version", -1))
    except Exception:
        return -1


def _reference_data_product_status(phase: str, item: Dict[str, Any]) -> str:
    del phase
    return _product_ok_text(item, label="QM reference data")


def _models_product_status(phase: str, item: Dict[str, Any]) -> str:
    del phase
    return _product_ok_text(item, label="models")


def _format_data_products_summary(
    status: Any,
    *,
    state_contract: Any = None,
    phase: Any = None,
    verbose: bool,
) -> List[str]:
    phase_name = str(phase or "")
    rows: List[tuple[str, Any]] = []
    if not isinstance(status, dict):
        rows.append(("QM reference data", "not checked"))
        rows.append(("FEREBUS models", "not checked"))
    else:
        reference_data = status.get("reference_data")
        if isinstance(reference_data, dict):
            rows.append(
                (
                    "QM reference data",
                    _reference_data_product_status(phase_name, reference_data),
                )
            )
            if (
                phase_name in _BOOTSTRAP_NOT_READY_PHASES
                and reference_data.get("ok") is not True
            ):
                rows.append(("reference data expected after", "REFERENCE_COMMIT"))
            if verbose:
                rows.append(("reference-data version", reference_data.get("version")))
                for error in reference_data.get("errors") or []:
                    rows.append(("reference-data detail", error))
        else:
            rows.append(("QM reference data", "not checked"))

        models = status.get("models")
        if isinstance(models, dict):
            rows.append(("FEREBUS models", _models_product_status(phase_name, models)))
            if (
                phase_name in _BOOTSTRAP_NOT_READY_PHASES
                and models.get("ok") is not True
            ):
                rows.append(("models expected after", "INITIAL_FEREBUS"))
            if verbose:
                rows.append(("models version", models.get("version")))
                for error in models.get("errors") or []:
                    rows.append(("models detail", error))
        else:
            rows.append(("FEREBUS models", "not checked"))
        if status.get("error"):
            rows.append(("artefact check", "problem - " + str(status.get("error"))))

    contract_text = _format_contract_status(state_contract)
    if contract_text == "ok" and verbose and phase_name not in {
        CampaignPhase.HALTED.value,
        CampaignPhase.DONE.value,
    }:
        rows.append(("current phase check", "verified"))
    elif contract_text is not None:
        rows.append(("current phase contract", contract_text))
    else:
        rows.append(("current phase contract", "not checked"))
    if verbose and isinstance(state_contract, dict):
        for error in state_contract.get("errors") or []:
            rows.append(("phase contract detail", error))
    return _section("Data Products", rows)


def _format_artifact_summary(
    status: Any,
    *,
    state_contract: Any = None,
    verbose: bool,
) -> List[str]:
    if not isinstance(status, dict):
        return _section("Artifacts", [("status", "unavailable")])
    rows = []
    contract_text = _format_contract_status(state_contract)
    if contract_text is not None:
        rows.append(("state contract", contract_text))
        if verbose and isinstance(state_contract, dict):
            for error in state_contract.get("errors") or []:
                rows.append(("state contract detail", error))
    for label in ("reference_data", "models"):
        item = status.get(label)
        if isinstance(item, dict):
            version = item.get("version")
            errors = [str(error) for error in (item.get("errors") or [])]
            ok = item.get("ok")
            if ok is True and isinstance(version, int) and version < 0:
                status_text = "not required yet"
            elif ok is True:
                status_text = "ok"
            elif ok is False:
                status_text = "problem"
                if errors:
                    first = errors[0]
                    if len(first) > 180:
                        first = first[:177] + "..."
                    status_text += " - " + first
            else:
                status_text = "not checked"
            rows.append((label + " version", version))
            rows.append((label + " status", status_text))
            if verbose:
                for error in errors:
                    rows.append((label + " detail", error))
        elif item is not None:
            rows.append((label, item))
    if "error" in status:
        rows.append(("error", status.get("error")))
    return _section("Artifacts", rows or [("status", "not checked")])


def _format_active_submission_intents(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    samples: List[str] = []
    for intent in value[:3]:
        if not isinstance(intent, dict):
            continue
        phase = str(intent.get("phase", "?"))
        iteration = str(intent.get("iteration", "?"))
        status = str(intent.get("status", "?"))
        job_id = intent.get("job_id")
        lifecycle = intent.get("queue_lifecycle")
        lifecycle_bits: List[str] = []
        if isinstance(lifecycle, dict):
            def advisory_duration(label: str) -> None:
                raw = lifecycle.get(label)
                if raw is None:
                    return
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    lifecycle_bits.append(label + "=invalid")
                    return
                if not np.isfinite(value) or value < 0.0:
                    lifecycle_bits.append(label + "=invalid")
                    return
                lifecycle_bits.append(
                    label.replace("_seconds", "")
                    + "="
                    + str(round(value, 1))
                    + "s"
                )

            if lifecycle.get("first_squeue_at_iso"):
                lifecycle_bits.append("squeue_seen")
            if lifecycle.get("first_sacct_at_iso"):
                lifecycle_bits.append("sacct_seen")
            if lifecycle.get("terminal_at_iso"):
                lifecycle_bits.append("terminal")
            advisory_duration("queue_wait_seconds")
            advisory_duration("postprocess_seconds")
            expected = lifecycle.get("n_expected", intent.get("expected_tasks"))
            observed = lifecycle.get("n_observed")
            missing = lifecycle.get("n_missing")
            if expected is not None:
                lifecycle_bits.append("expected=" + str(expected))
            if observed is not None:
                lifecycle_bits.append("observed=" + str(observed))
            if missing is not None:
                lifecycle_bits.append("missing=" + str(missing))
        replacement_round = intent.get("replacement_round")
        if replacement_round is not None:
            lifecycle_bits.append("round=" + str(replacement_round))
        if job_id:
            text = phase + "@" + iteration + " " + status + " job_id=" + str(job_id)
        else:
            text = phase + "@" + iteration + " " + status
        if lifecycle_bits:
            text += " [" + ", ".join(lifecycle_bits) + "]"
        samples.append(text)
    summary = str(len(value))
    if samples:
        summary += " (" + "; ".join(samples) + ")"
        if len(value) > len(samples):
            summary += " ..."
    return summary


def _format_background_daemon(
    pid: Any,
    alive: Any,
    startup_state: Any = None,
    startup_stage: Any = None,
) -> str:
    if pid is None:
        if startup_state == "failed":
            return "not running (last startup failed)"
        return "not running"
    if alive is True and startup_state in BACKGROUND_STARTUP_ACTIVE_STATES:
        detail = str(startup_state)
        if startup_stage:
            detail += ": " + str(startup_stage)
        return "pid " + str(pid) + " (starting; " + detail + ")"
    if alive is True and startup_state == "ready":
        return "pid " + str(pid) + " (alive; ready)"
    status = "alive" if alive is True else "not running"
    return "pid " + str(pid) + " (" + status + ")"


def _first_recommendation(payload: Dict[str, Any]) -> Dict[str, Any]:
    recommendations = payload.get("recommendations")
    if isinstance(recommendations, list) and recommendations:
        first = recommendations[0]
        if isinstance(first, dict):
            return first
    return {
        "code": "recommendation_unavailable",
        "severity": "watch",
        "primary": "run status again with --verbose or inspect the journal",
        "why": "status recommendation payload was not available",
    }


_STATUS_BLOCKED_RECOMMENDATIONS = frozenset(
    {
        "runtime_probe_failed",
        "state_unreadable",
        "pool_feasibility_failed",
        "halted_mandatory_custom_bootstrap_failed",
        "halted_replacement_reserve_exhausted",
        "halted_scheduler_hard_failure",
        "halted_seed_pool_exhausted",
        "reconcile_transaction_manual_review",
        "allocation_check_environment_blocked",
        "config_change_blocked",
        "config_lock_review_failed",
        "scheduler_recovery_invalid",
    }
)


_STATUS_REVIEW_RECOMMENDATIONS = frozenset(
    {
        "campaign_completed_scientific_convergence",
        "campaign_completed_max_iterations",
        "campaign_done",
    }
)


_STATUS_PRESENTATION_RECOMMENDATIONS = frozenset(
    {
        "recommendation_unavailable",
        "stale_background_pid",
        "runtime_probe_failed",
        "daemon_running",
        "pending_state_job",
        "active_submission_intent",
        "local_submission_intent",
        "halted_scheduler_uncertain",
        "halted_mandatory_custom_bootstrap_failed",
        "halted_replacement_reserve_exhausted",
        "halted_ferebus_quality_failed",
        "halted_seed_pool_exhausted",
        "halted_backend_submission_failed",
        "halted_scheduler_transient",
        "halted_scheduler_hard_failure",
        "halted_contract_failure",
        "halted_config_changed",
        "halted_unknown",
        "initial_ferebus_bootstrap_contract_problem",
        "stop_check_no_committed_pair",
        "version_skew",
        "reference_data_missing",
        "models_missing",
        "state_artifact_contract_invalid",
        "committed_artifact_invalid",
        "campaign_completed_scientific_convergence",
        "campaign_completed_max_iterations",
        "campaign_done",
        "execution_identity_unavailable",
        "phase_unknown",
        "campaign_config_invalid",
        "partial_array_recovery_invalid",
        "journal_corrupt",
        "submission_intent_invalid",
        "stop_control_invalid",
        "pool_feasibility_failed",
        "state_missing_fresh_init",
        "campaign_missing",
        "state_missing",
        "state_schema_invalid",
        "state_unreadable",
        "user_stop_cancellation_incomplete",
        "user_stop_draining",
        "shutdown_requested",
        "reconcile_transaction_recoverable",
        "reconcile_transaction_manual_review",
        "allocation_check_environment_blocked",
        "allocation_check_environment_retry",
        "config_change_pending_running",
        "config_change_reconcile_required",
        "config_change_blocked",
        "config_lock_review_failed",
        "scheduler_recovery_invalid",
        "scheduler_terminal_recovery_resume",
        "background_startup_failed",
    }
    | {
        "phase_" + phase.value.lower() + "_ready"
        for phase in CampaignPhase
        if phase.value in _PHASE_AUTOMATIC_OUTCOMES
        and phase not in {CampaignPhase.DONE, CampaignPhase.HALTED}
    }
)


@dataclass(frozen=True)
class _StatusPresentation:
    campaign: Tuple[Tuple[str, Any], ...]
    current_status: Tuple[Tuple[str, Any], ...]
    progress_so_far: Tuple[Tuple[str, Any], ...]
    what_happens_next: Tuple[Tuple[str, Any], ...]


def _format_pool_feasibility_status(feasibility: Any) -> List[str]:
    if not isinstance(feasibility, dict):
        return []
    rows: List[tuple[str, Any]] = []
    if feasibility.get("error"):
        rows.append(("status", "unavailable"))
        rows.append(("error", feasibility.get("error")))
    else:
        rows.extend(
            [
                ("status", "ready" if feasibility.get("ok") else "blocked"),
                ("frames available", feasibility.get("pool_n_frames")),
                ("frames required", feasibility.get("required_pool_frames")),
                ("bootstrap custom geometries", feasibility.get("bootstrap_custom_count")),
                ("bootstrap model training rows", feasibility.get("bootstrap_model_training_count")),
                (
                    "bootstrap pool frames",
                    feasibility.get("bootstrap_pool_frame_count"),
                ),
                ("requirement", feasibility.get("expression")),
            ]
        )
    return _section("Trajectory Pool", rows)


def _active_pending_jobs_summary(pending: Any) -> str:
    if not isinstance(pending, dict) or not pending:
        return "none recorded in state"
    active: List[str] = []
    completed_markers = 0
    for phase, job_id in sorted(pending.items()):
        if job_id:
            active.append(str(phase) + "=" + str(job_id))
        else:
            completed_markers += 1
    if active:
        text = ", ".join(active[:4])
        if len(active) > 4:
            text += " ..."
        return text
    return "none active (" + str(completed_markers) + " completed markers)"


def _daemon_activity_status(payload: Dict[str, Any]) -> str:
    startup_state = payload.get("background_startup_state")
    if (
        payload.get("background_pid_alive") is True
        and startup_state in BACKGROUND_STARTUP_ACTIVE_STATES
    ):
        stage = payload.get("background_startup_stage")
        suffix = ": " + str(stage) if stage else ""
        return (
            "starting (background pid "
            + str(payload.get("background_pid"))
            + suffix
            + ")"
        )
    active: List[str] = []
    if payload.get("lock_held") is True:
        active.append("foreground lock held")
    if payload.get("lease_dir_exists") and _lease_is_fresh(
        payload.get("lease_heartbeat"),
        stale_seconds=int(payload.get("lease_stale_seconds", 900)),
        clock_skew_tolerance_seconds=int(
            payload.get("clock_skew_tolerance_seconds", 60)
        ),
    ):
        active.append("fresh lease")
    if payload.get("background_pid_alive") is True:
        active.append("background pid " + str(payload.get("background_pid")))
    if active:
        return "running (" + ", ".join(active) + ")"
    if payload.get("lock_held") is None:
        return "unknown (lock probe failed)"
    return "not running"


def _status_daemon_active(payload: Dict[str, Any]) -> bool:
    return _daemon_activity_status(payload).startswith(("running", "starting"))


def _status_active_job_count(payload: Dict[str, Any]) -> int:
    return assess_campaign_presentation(payload).scheduler.pending_job_count


def _status_intent_counts(payload: Dict[str, Any]) -> Tuple[int, int]:
    scheduler = assess_campaign_presentation(payload).scheduler
    return scheduler.scheduler_intent_count, scheduler.local_intent_count


def _scheduler_display_name(
    kind: Optional[str] = None,
    *,
    intents: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    recorded = {
        str(intent.get("scheduler_identity_kind") or "").strip().lower()
        for intent in (intents or [])
        if isinstance(intent, Mapping)
        and intent.get("job_id")
        and intent.get("scheduler_identity_kind")
    }
    if len(recorded) == 1:
        kind = next(iter(recorded))
    if not kind:
        kind = str(
            profile_value("hpc", "scheduler", default="slurm") or "slurm"
        ).strip().lower()
    try:
        return str(get_scheduler_backend(kind).display_name)
    except Exception:
        return "scheduler"


def _status_scheduler_display_name(payload: Mapping[str, Any]) -> str:
    raw_intents = payload.get("active_submission_intents")
    intents = (
        [item for item in raw_intents if isinstance(item, Mapping)]
        if isinstance(raw_intents, list)
        else []
    )
    return _scheduler_display_name(intents=intents)


def _seed_selection_progress_record(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    progress = payload.get("seed_selection_progress")
    if not isinstance(progress, dict) or progress.get("state") != "current":
        return None
    record = progress.get("record")
    return dict(record) if isinstance(record, dict) else None


def _runtime_progress_record(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    progress = payload.get("runtime_progress")
    if not isinstance(progress, dict) or progress.get("state") != "current":
        return None
    record = progress.get("record")
    return dict(record) if isinstance(record, dict) else None


def _format_elapsed_seconds(value: Any) -> Optional[str]:
    try:
        elapsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(elapsed) or elapsed < 0.0:
        return None
    seconds = int(elapsed)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return str(hours) + "h " + str(minutes).zfill(2) + "m"
    if minutes:
        return str(minutes) + "m " + str(seconds).zfill(2) + "s"
    return str(seconds) + "s"


def _format_generic_progress_activity(record: Mapping[str, Any]) -> str:
    from .daemon.phase_progress import format_progress_stage

    activity = format_progress_stage(record.get("stage"))
    status = str(record.get("status") or "running").lower()
    if status == "completed":
        return "Finished " + activity[:1].lower() + activity[1:] + "."
    if status == "failed":
        return "Failed while " + activity.lower() + "."
    return activity + "."


def _format_generic_progress_count(record: Mapping[str, Any]) -> Optional[str]:
    counters = record.get("counters")
    if not isinstance(counters, Mapping):
        return None
    completed = _event_int(dict(counters), "completed")
    total = _event_int(dict(counters), "total")
    unit = str(counters.get("unit") or "items")
    details = record.get("details")
    if record.get("producer_kind") == "scheduler" and isinstance(details, Mapping):
        parts: List[str] = []
        for key in ("completed", "running", "pending", "failed", "missing"):
            value = completed if key == "completed" else _event_int(dict(details), key)
            if value is not None and (value > 0 or key == "completed"):
                parts.append(str(value) + " " + key)
        if parts:
            return ", ".join(parts)
    detail_suffix = ""
    if isinstance(details, Mapping):
        detail_parts = []
        for key in ("accepted", "rejected"):
            value = _event_int(dict(details), key)
            if value is not None:
                detail_parts.append(str(value) + " " + key)
        if detail_parts:
            detail_suffix = " (" + ", ".join(detail_parts) + ")"
    if completed is not None and total is not None:
        return str(completed) + "/" + str(total) + " " + unit + detail_suffix
    if completed is not None:
        return str(completed) + " " + unit + detail_suffix
    return None


def _status_progress_rows(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    runtime = payload.get("runtime_progress")
    scheduler_recovery = payload.get(
        "_presentation_scheduler_recovery"
    )
    scheduler_assessment = assess_campaign_presentation(
        payload
    ).scheduler
    if (
        isinstance(scheduler_recovery, Mapping)
        and scheduler_assessment.recovery_state == "invalid"
    ):
        return []
    seed_record = (
        _seed_selection_progress_record(payload)
        if str(payload.get("phase") or "") == CampaignPhase.SEED_SELECT.value
        else None
    )
    if seed_record is not None:
        elapsed = _format_elapsed_seconds(
            seed_record.get(
                "stage_elapsed_seconds",
                seed_record.get("elapsed_seconds"),
            )
        )
        rows: List[Tuple[str, str]] = []
        completed = _event_int(seed_record, "completed")
        total = _event_int(seed_record, "total")
        if completed is not None and total is not None and total > 0:
            rows.append(("progress", str(completed) + "/" + str(total)))
        if elapsed:
            rows.append(("elapsed", elapsed))
        progress = payload.get("seed_selection_progress")
        if isinstance(progress, Mapping) and progress.get("age_seconds") is not None:
            age = _format_elapsed_seconds(progress.get("age_seconds"))
            if age:
                rows.append(("last update", age + " ago"))
        return rows
    scheduler_intents, _local_intents = _status_intent_counts(payload)
    if (
        isinstance(scheduler_recovery, Mapping)
        and scheduler_assessment.has_terminal_recovery
        and not _status_active_job_count(payload)
        and not scheduler_intents
    ):
        if scheduler_assessment.recovery_state == "legacy_unverified":
            return [
                ("reusable outputs", "none from legacy recovery evidence"),
                (
                    "retry work",
                    str(scheduler_assessment.retry_tasks)
                    + " task"
                    + (
                        ""
                        if scheduler_assessment.retry_tasks == 1
                        else "s"
                    ),
                ),
            ]
        if scheduler_assessment.recovery_state == "validated":
            return [
                (
                    "reusable outputs",
                    str(scheduler_assessment.reusable_outputs)
                    + " validated",
                ),
                (
                    "retry work",
                    str(scheduler_assessment.retry_tasks)
                    + " task"
                    + (
                        ""
                        if scheduler_assessment.retry_tasks == 1
                        else "s"
                    ),
                ),
            ]
        return [
            (
                "awaiting validation",
                str(scheduler_assessment.completed_candidates)
                + " scheduler-completed output"
                + (
                    ""
                    if scheduler_assessment.completed_candidates == 1
                    else "s"
                ),
            ),
            (
                "recorded retry work",
                str(scheduler_assessment.retry_tasks)
                + " task"
                + (
                    ""
                    if scheduler_assessment.retry_tasks == 1
                    else "s"
                ),
            ),
        ]
    record = _runtime_progress_record(payload)
    if record is None:
        return []
    rows = []
    historical_scheduler_progress = bool(
        str(record.get("producer_kind") or "") == "scheduler"
        and not _status_daemon_active(payload)
    )
    count = _format_generic_progress_count(record)
    if count:
        progress_label = "progress"
        if historical_scheduler_progress:
            progress_label = (
                "last recorded "
                + _status_scheduler_display_name(payload)
                + " progress"
            )
        rows.append((progress_label, count))
    details = record.get("details")
    if (
        str(record.get("producer_kind") or "") == "scheduler"
        and isinstance(details, Mapping)
    ):
        diagnostic_prefix = "recorded " if historical_scheduler_progress else ""
        for key, label in (
            ("pending_reason", "pending reason"),
            ("pending_queue", "scheduler queue"),
            ("scheduler_native_state", "native scheduler state"),
        ):
            value = " ".join(str(details.get(key) or "").split())
            if value:
                rows.append((diagnostic_prefix + label, value[:160]))
    elapsed = _format_elapsed_seconds(record.get("elapsed_seconds"))
    if elapsed:
        rows.append(
            (
                "last recorded elapsed"
                if historical_scheduler_progress
                else "elapsed",
                elapsed,
            )
        )
    try:
        throughput = float(record.get("throughput"))
    except (TypeError, ValueError):
        throughput = 0.0
    if (
        not historical_scheduler_progress
        and np.isfinite(throughput)
        and throughput > 0.0
    ):
        counters = record.get("counters")
        unit = (
            str(counters.get("unit") or "items")
            if isinstance(counters, Mapping)
            else "items"
        )
        rows.append(
            (
                "throughput",
                format(throughput, ".2f") + " " + unit + "/s",
            )
        )
    if isinstance(runtime, Mapping) and runtime.get("age_seconds") is not None:
        age = _format_elapsed_seconds(runtime.get("age_seconds"))
        if age:
            rows.append(("last update", age + " ago"))
    return rows


def _round_journal_seconds(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(seconds):
        return None
    if seconds >= 0.0:
        return int(np.floor(seconds + 0.5))
    return int(np.ceil(seconds - 0.5))


def _format_journal_detail_value(key: str, value: Any) -> str:
    if str(key) in {"reason", "error"}:
        failure = classify_operator_failure(value)
        if failure.family != "unknown":
            return failure.summary
    if str(key).endswith("_seconds"):
        rounded = _round_journal_seconds(value)
        if rounded is not None:
            return str(rounded)
    return _format_value(value)


_SEED_SELECTION_PROGRESS_FORMATTER_STAGES = frozenset(
    {
        "trajectory_authority",
        "trajectory_coordinates",
        "model_authority",
        "models",
        "model_factors",
        "features",
        "seed_exclusions",
        "sampling_protocol",
        "reference_neighbours",
        "reference_scales",
        "filtering",
        "random",
        "variance",
        "shortlist",
        "d_optimal",
        "publishing",
        "failed",
        "complete",
    }
)


def _format_seed_selection_progress(record: Dict[str, Any]) -> str:
    stage = str(record.get("stage") or "loading")
    completed = _event_int(record, "completed")
    total = _event_int(record, "total")
    count = (
        " " + str(completed) + "/" + str(total)
        if completed is not None and total is not None and total > 0
        else ""
    )
    details: List[str] = []
    elapsed_seconds = _round_journal_seconds(
        record.get("stage_elapsed_seconds", record.get("elapsed_seconds"))
    )
    if elapsed_seconds is not None and elapsed_seconds >= 1:
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        details.append(
            (
                str(hours) + "h " + str(minutes).zfill(2) + "m"
                if hours
                else str(minutes) + "m " + str(seconds).zfill(2) + "s"
            )
            + " elapsed"
        )
    progress_suffix = " (" + ", ".join(details) + ")" if details else ""

    def _finish(message: str) -> str:
        return message + progress_suffix + "."

    if stage == "loading":
        return _finish("Loading and validating the trajectory pool and models" + count)
    if stage == "trajectory_authority":
        return _finish("Validating trajectory-pool authority" + count)
    if stage == "trajectory_coordinates":
        cache_status = str(record.get("cache_status") or "")
        verb = "Restoring cached" if cache_status == "hit" else "Preparing"
        return _finish(verb + " trajectory coordinates" + count)
    if stage == "model_authority":
        return _finish("Validating current-model authority" + count)
    if stage == "models":
        return _finish("Loading current models" + count)
    if stage == "model_factors":
        return _finish("Restoring or computing model factors" + count)
    if stage == "features":
        cache_status = str(record.get("cache_status") or "")
        verb = "Restoring cached" if cache_status == "hit" else "Preparing cached"
        return _finish(verb + " trajectory features" + count)
    if stage == "seed_exclusions":
        return _finish("Resolving committed and recent seed exclusions" + count)
    if stage == "sampling_protocol":
        return _finish("Resolving sampling history and protocol" + count)
    if stage == "reference_neighbours":
        return _finish("Finding reference-scale neighbours" + count)
    if stage == "reference_scales":
        sample = _event_int(record, "sample")
        samples = _event_int(record, "samples")
        mode = _event_int(record, "mode")
        modes = _event_int(record, "modes")
        reference_pass = str(record.get("reference_pass") or "").strip()
        if sample is not None and samples is not None and reference_pass:
            return _finish(
                "Computing reference scales: sample "
                + str(sample)
                + "/"
                + str(samples)
                + ", "
                + reference_pass.replace("_", " ")
                + " stencils"
            )
        if None not in {sample, samples, mode, modes}:
            return _finish(
                "Computing reference scales: sample "
                + str(sample)
                + "/"
                + str(samples)
                + ", mode "
                + str(mode)
                + "/"
                + str(modes)
            )
        return _finish("Computing reference scales" + count)
    if stage == "filtering":
        return _finish("Filtering eligible trajectory frames" + count)
    if stage == "random":
        return _finish("Selecting the deterministic random subset" + count)
    if stage == "variance":
        return _finish("Scoring posterior variance" + count)
    if stage == "shortlist":
        return _finish("Preparing the D-optimal shortlist" + count)
    if stage == "d_optimal":
        shortlist = _event_int(record, "shortlist_size")
        shortlist_suffix = (
            " from " + str(shortlist) + " shortlisted frames"
            if shortlist is not None
            else ""
        )
        return _finish("Selecting D-optimal seeds" + count + shortlist_suffix)
    if stage == "publishing":
        return _finish("Publishing the seed selection" + count)
    if stage == "failed":
        return "Seed selection stopped after an error."
    if stage == "complete":
        return "Seed selection has been published."
    return "Seed selection is running (" + stage.replace("_", " ") + ")."


def _status_current_activity(payload: Dict[str, Any]) -> str:
    phase = str(payload.get("phase") or "")
    title = _sentence_fragment(_phase_title(phase))
    daemon_active = _status_daemon_active(payload)
    active_jobs = _status_active_job_count(payload)
    scheduler_intents, local_intents = _status_intent_counts(payload)
    runtime_progress = _runtime_progress_record(payload)
    progress = _seed_selection_progress_record(payload)
    scheduler_name = _status_scheduler_display_name(payload)
    aimall_recovery = payload.get(
        "_presentation_aimall_postprocess_recovery"
    )
    ariadne_terminal = payload.get(
        "_presentation_ariadne_terminal_postprocess"
    )
    partial_recovery = payload.get("partial_array_recovery")
    scheduler_recovery = payload.get(
        "_presentation_scheduler_recovery"
    )
    scheduler_assessment = assess_campaign_presentation(
        payload
    ).scheduler
    if str(payload.get("background_startup_state") or "") in {
        "prepared",
        "spawned",
        "ownership_acquired",
    } and payload.get("background_pid_alive") is True:
        stage = str(payload.get("background_startup_stage") or "initial checks")
        return "The background process is still starting (" + stage + ")."
    if (
        str(payload.get("background_startup_state") or "") == "failed"
        and not daemon_active
    ):
        diversity_transition = payload.get(
            "_presentation_diversity_transition"
        )
        if (
            str(payload.get("background_startup_stage") or "")
            == "environment_transition"
            and isinstance(diversity_transition, Mapping)
            and bool(diversity_transition.get("safe", False))
        ):
            job_id = str(
                diversity_transition.get("producer_job_id") or ""
            )
            transition_kind = str(
                diversity_transition.get("transition_kind") or ""
            )
            if transition_kind.endswith("complete_publication_adoption"):
                selected = int(
                    diversity_transition.get("selected_count") or 0
                )
                return (
                    "The previous diversity job"
                    + (" " + job_id if job_id else "")
                    + " is terminal; "
                    + str(selected)
                    + " existing selected geometr"
                    + ("y" if selected == 1 else "ies")
                    + " await local validation, and no diversity job will be "
                    "submitted."
                )
            if phase == CampaignPhase.PHASE_B_DIVERSITY.value:
                return (
                    "The previous Phase B job"
                    + (" " + job_id if job_id else "")
                    + " is terminal; "
                    + str(
                        int(
                            diversity_transition.get(
                                "ariadne_accepted_tasks"
                            )
                            or 0
                        )
                    )
                    + " accepted ARIADNE results are ready for one Phase B retry; "
                    + str(
                        int(
                            diversity_transition.get(
                                "ariadne_rejected_tasks"
                            )
                            or 0
                        )
                    )
                    + " rejected results remain excluded."
                )
            return (
                "The previous bootstrap diversity job"
                + (" " + job_id if job_id else "")
                + " is terminal; bootstrap diversity is ready to retry."
            )
        stage = str(payload.get("background_startup_stage") or "startup")
        return (
            "The last daemon startup failed during "
            + stage.replace("_", " ")
            + "; no campaign work is running."
        )
    if phase == CampaignPhase.SEED_SELECT.value and progress is not None:
        return _format_seed_selection_progress(progress)
    if (
        isinstance(scheduler_recovery, Mapping)
        and scheduler_assessment.recovery_state == "invalid"
    ):
        return (
            "Scheduler recovery evidence could not be validated; no retry "
            "work is being submitted."
        )
    if (
        isinstance(scheduler_recovery, Mapping)
        and scheduler_assessment.has_terminal_recovery
        and not active_jobs
        and not scheduler_intents
    ):
        disposition = str(
            scheduler_recovery.get("publication_disposition") or ""
        )
        if disposition == "adopted":
            return (
                "A complete diversity publication is ready to be adopted "
                "without rerunning its scheduler job."
            )
        if disposition == "rerun":
            return (
                "The interrupted diversity publication was incomplete; its "
                "single scheduler job is ready to be rerun."
            )
        if scheduler_assessment.recovery_state == "legacy_unverified":
            retry = scheduler_assessment.retry_tasks
            return (
                "Older scheduler recovery evidence cannot safely authorise "
                "output reuse; "
                + str(retry)
                + " affected task"
                + ("" if retry == 1 else "s")
                + " will be retried."
            )
        if scheduler_assessment.recovery_state == "validated":
            reusable = scheduler_assessment.reusable_outputs
            retry = scheduler_assessment.retry_tasks
            return (
                str(reusable)
                + " completed task"
                + ("" if reusable == 1 else "s")
                + " passed output validation; "
                + str(retry)
                + " task"
                + ("" if retry == 1 else "s")
                + " will be retried."
            )
        completed = scheduler_assessment.completed_candidates
        retry = scheduler_assessment.retry_tasks
        return (
            str(completed)
            + " scheduler-completed output"
            + ("" if completed == 1 else "s")
            + " await local validation; "
            + str(retry)
            + " unfinished task"
            + ("" if retry == 1 else "s")
            + (" is" if retry == 1 else " are")
            + " recorded for retry."
        )
    if isinstance(aimall_recovery, Mapping):
        total = int(aimall_recovery.get("logical_total") or 0)
        if daemon_active:
            return (
                "The daemon is validating "
                + str(total)
                + " scheduler-completed AIMAll output candidate"
                + ("" if total == 1 else "s")
                + " locally; only structurally invalid or unfinished tasks "
                "will be submitted."
            )
        return (
            str(total)
            + " scheduler-completed AIMAll output candidate"
            + ("" if total == 1 else "s")
            + (" is" if total == 1 else " are")
            + " ready for local validation; resume will submit only "
            "structurally invalid or unfinished tasks."
        )
    if isinstance(ariadne_terminal, Mapping):
        completed = int(
            ariadne_terminal.get("n_scheduler_completed") or 0
        )
        failed = int(ariadne_terminal.get("n_scheduler_failed") or 0)
        if daemon_active:
            return (
                "The daemon is validating all terminal ARIADNE output slots "
                "locally; no ARIADNE scheduler job is being submitted."
            )
        return (
            str(completed)
            + " scheduler-completed ARIADNE output candidate"
            + ("" if completed == 1 else "s")
            + " and "
            + str(failed)
            + " scheduler-failed slot"
            + ("" if failed == 1 else "s")
            + " are ready for local validation; resume will submit no "
            "ARIADNE job."
        )
    if (
        isinstance(partial_recovery, Mapping)
        and phase
        in {
            CampaignPhase.PHASE_A_DIVERSITY.value,
            CampaignPhase.PHASE_B_DIVERSITY.value,
        }
        and partial_recovery.get("selected_count") is not None
        and int(partial_recovery.get("n_retry") or 0) == 0
    ):
        selected = int(partial_recovery.get("selected_count") or 0)
        if daemon_active:
            return (
                "The daemon is validating "
                + str(selected)
                + " existing diversity geometr"
                + ("y" if selected == 1 else "ies")
                + " locally; no scheduler job is being submitted."
            )
        return (
            str(selected)
            + " existing diversity geometr"
            + ("y is" if selected == 1 else "ies are")
            + " ready for local validation; no scheduler job will be submitted."
        )
    if runtime_progress is not None:
        if str(runtime_progress.get("producer_kind") or "") == "scheduler":
            count = max(active_jobs, scheduler_intents, 1)
            work_name = _PHASE_WORK_NAMES.get(phase, title)
            kind = _status_slurm_work_kind(payload)
            if daemon_active:
                return (
                    "waiting for "
                    + str(count)
                    + " "
                    + work_name
                    + " "
                    + scheduler_name
                    + " "
                    + kind
                    + ("" if count == 1 else "s")
                )
            return (
                "The daemon is stopped; "
                + str(count)
                + " "
                + work_name
                + " "
                + scheduler_name
                + " "
                + kind
                + (" remains recorded." if count == 1 else "s remain recorded.")
            )
        return _format_generic_progress_activity(runtime_progress)
    if active_jobs or scheduler_intents:
        count = max(active_jobs, scheduler_intents)
        work_name = _PHASE_WORK_NAMES.get(phase, title)
        kind = _status_slurm_work_kind(payload)
        if daemon_active:
            return (
                "waiting for "
                + str(count)
                + " "
                + work_name
                + " "
                + scheduler_name
                + " "
                + kind
                + ("" if count == 1 else "s")
            )
        return (
            "The daemon is stopped; "
            + str(count)
            + " recorded "
            + scheduler_name
            + " job"
            + (" still needs monitoring or postprocessing." if count == 1 else "s still need monitoring or postprocessing.")
        )
    if local_intents:
        intents = payload.get("active_submission_intents")
        local_phase = phase
        if isinstance(intents, list):
            for intent in intents:
                if isinstance(intent, dict) and not intent.get("job_id"):
                    local_phase = str(intent.get("phase") or phase)
                    break
        local_title = _sentence_fragment(_phase_title(local_phase))
        return (
            "The daemon is completing local work for " + local_title + "."
            if daemon_active
            else "The daemon is stopped; local work for " + local_title + " is prepared but not running."
        )
    if phase == CampaignPhase.HALTED.value:
        return "The campaign is halted; no work is running."
    if phase == CampaignPhase.DONE.value:
        return "The campaign is complete; no work is running."
    if daemon_active:
        return "The daemon is working on " + title + "."
    return "The daemon is stopped; " + title + " is the next campaign step."


def _status_slurm_work_kind(payload: Mapping[str, Any]) -> str:
    intents = payload.get("active_submission_intents")
    if isinstance(intents, list):
        for intent in intents:
            if not isinstance(intent, Mapping) or not intent.get("job_id"):
                continue
            try:
                if int(intent.get("expected_tasks") or 0) > 1:
                    return "array"
            except (TypeError, ValueError):
                continue
    return "job"


def _status_ownership_uncertain(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("lock_probe_error")
        or payload.get("lease_probe_error")
        or ("lock_held" in payload and payload.get("lock_held") is None)
    )


def _status_artifact_problem(payload: Mapping[str, Any]) -> bool:
    contract = payload.get("state_artifact_contract_status")
    if isinstance(contract, Mapping) and contract.get("ok") is False:
        return True
    products = payload.get("artifact_manifest_status")
    if not isinstance(products, Mapping):
        return False
    if products.get("error"):
        return True
    return any(
        isinstance(products.get(label), Mapping)
        and products[label].get("ok") is False
        for label in ("reference_data", "models")
    )


def _status_control_problem(payload: Mapping[str, Any]) -> bool:
    config = payload.get("campaign_config_status")
    feasibility = payload.get("pool_feasibility")
    progress = _runtime_progress_record(dict(payload))
    assessment = assess_campaign_presentation(payload)
    return bool(
        (isinstance(config, Mapping) and config.get("ok") is False)
        or assessment.config_blocks_progress
        or payload.get("partial_array_recovery_error")
        or payload.get("journal_error")
        or payload.get("submission_intent_errors")
        or payload.get("stop_control_error")
        or assessment.scheduler.recovery_state == "invalid"
        or (
            isinstance(feasibility, Mapping)
            and feasibility.get("ok") is False
            and "FileNotFoundError" not in str(feasibility.get("error") or "")
        )
        or (
            isinstance(progress, Mapping)
            and str(progress.get("status") or "").lower() == "failed"
        )
    )


def _status_overall(payload: Dict[str, Any]) -> str:
    phase = str(payload.get("phase") or "")
    code = str(_first_recommendation(payload).get("code") or "")
    stop_disposition = payload.get("_presentation_stop_disposition")
    if _status_ownership_uncertain(payload):
        return "cannot determine safely"
    if (
        isinstance(stop_disposition, Mapping)
        and str(stop_disposition.get("kind") or "") == "unreachable"
    ):
        return "blocked"
    if code == "reconcile_transaction_manual_review":
        return "blocked"
    if code == "reconcile_transaction_recoverable":
        return "needs attention"
    if code == "allocation_check_environment_blocked":
        return "needs attention"
    if code == "allocation_check_environment_retry":
        return "stopped and ready to continue"
    if code == "config_change_pending_running":
        return "running normally"
    if code == "config_change_reconcile_required":
        return "stopped; reconcile required"
    if code in {
        "config_change_blocked",
        "config_lock_review_failed",
        "scheduler_recovery_invalid",
    }:
        return "blocked"
    if code == "background_startup_failed":
        return "startup failed"
    if _status_control_problem(payload) or _status_artifact_problem(payload):
        return "blocked" if code in _STATUS_BLOCKED_RECOMMENDATIONS else "needs attention"
    if phase == CampaignPhase.HALTED.value:
        return "halted"
    if phase == CampaignPhase.DONE.value:
        return "complete"
    stop_request = payload.get("stop_request")
    if isinstance(stop_request, Mapping):
        if str(stop_request.get("status") or "") == "completed":
            return "stopped by request"
        return (
            "stopping as requested"
            if _status_daemon_active(payload)
            else "stopped with a pending stop request"
        )
    if payload.get("shutdown_requested"):
        context = payload.get("lifecycle_context")
        if isinstance(context, Mapping) and str(
            context.get("disposition") or ""
        ) == "stopped":
            return "stopped by request"
        return "needs attention"
    if (
        payload.get("background_pid_alive") is True
        and str(payload.get("background_startup_state") or "")
        in BACKGROUND_STARTUP_ACTIVE_STATES
    ):
        return "starting"
    if _status_daemon_active(payload):
        return "running normally"
    scheduler_intents, local_intents = _status_intent_counts(payload)
    if _status_active_job_count(payload) or scheduler_intents or local_intents:
        return "stopped with unfinished work"
    return "stopped and ready to continue"


def _status_daemon_label(payload: Dict[str, Any]) -> str:
    if _status_ownership_uncertain(payload):
        return "ownership uncertain"
    if (
        payload.get("background_pid_alive") is True
        and str(payload.get("background_startup_state") or "")
        in BACKGROUND_STARTUP_ACTIVE_STATES
    ):
        return "starting"
    return "running" if _status_daemon_active(payload) else "not running"


def _format_runtime_status(payload: Dict[str, Any], *, verbose: bool) -> List[str]:
    from .daemon.stop_control import describe_stop_request

    stop_request = payload.get("stop_request")
    if isinstance(stop_request, dict):
        stop_summary = describe_stop_request(stop_request)
    elif payload.get("stop_control_error"):
        stop_summary = "invalid: " + str(payload.get("stop_control_error"))
    else:
        stop_summary = "none"
    rows: List[tuple[str, Any]] = [
        ("daemon", _daemon_activity_status(payload)),
        (
            "recorded " + _status_scheduler_display_name(payload) + " jobs",
            _active_pending_jobs_summary(payload.get("pending_jobs")),
        ),
        (
            "submission intents",
            _format_active_submission_intents(payload.get("active_submission_intents")),
        ),
        ("user stop control", stop_summary),
        ("shutdown requested", "yes" if payload.get("shutdown_requested") else "no"),
    ]
    if payload.get("background_startup_state"):
        rows.extend(
            [
                ("background startup", payload.get("background_startup_state")),
                ("background startup stage", payload.get("background_startup_stage")),
                ("background startup began", payload.get("background_startup_started_at")),
                ("background startup ready", payload.get("background_startup_ready_at")),
                ("background startup failure", payload.get("background_startup_failure")),
            ]
        )
    environment = payload.get("active_environment_generation")
    if isinstance(environment, dict) and environment.get("generation") is not None:
        rows.extend(
            [
                ("environment generation", environment.get("generation")),
                (
                    "environment transition",
                    environment.get("last_transition")
                    or environment.get("created_at_iso"),
                ),
            ]
        )
    elif verbose and isinstance(environment, dict) and environment.get("error"):
        rows.append(("environment generation error", environment.get("error")))
    ferebus_recovery = payload.get("ferebus_candidate_recovery")
    if isinstance(ferebus_recovery, dict):
        rows.extend(
            [
                ("FEREBUS candidate recovery", ferebus_recovery.get("status")),
                (
                    "FEREBUS recovery source",
                    ferebus_recovery.get("source_path"),
                ),
            ]
        )
    allocation = payload.get("point_allocation_summary")
    if isinstance(allocation, dict):
        rows.extend(
            [
                ("allocation accepted", allocation.get("accepted_total")),
                ("allocation deficit", allocation.get("deficit_total")),
                ("allocation reserve available", allocation.get("reserve_available")),
            ]
        )
    transactions = payload.get("reference_commit_transactions")
    if isinstance(transactions, list):
        active_transactions = [
            record
            for record in transactions
            if str(record.get("state") or "") != "complete"
        ]
        for record in active_transactions[:1]:
            ledger = record.get("ledger") or {}
            rows.extend(
                [
                    ("reference commit", record.get("state")),
                    (
                        "reference move progress",
                        str(ledger.get("moved_points", 0))
                        + "/"
                        + str(len(ledger.get("point_bindings") or [])),
                    ),
                    ("reference moved bytes", ledger.get("moved_bytes", 0)),
                    ("row shards repaired", ledger.get("shards_repaired", 0)),
                ]
            )
    if verbose:
        lease = "active: " + _heartbeat_summary(payload.get("lease_heartbeat"))
        if not payload.get("lease_dir_exists"):
            lease = "none"
        rows.extend(
            [
                ("foreground lock", _lock_summary(payload.get("lock_held"))),
                ("daemon lease", lease),
                (
                    "background daemon",
                    _format_background_daemon(
                        payload.get("background_pid"),
                        payload.get("background_pid_alive"),
                        payload.get("background_startup_state"),
                        payload.get("background_startup_stage"),
                    ),
                ),
                (
                    "stop request id",
                    (
                        stop_request.get("request_id")
                        if isinstance(stop_request, dict)
                        else None
                    ),
                ),
            ]
        )
        progress = payload.get("seed_selection_progress")
        if isinstance(progress, dict):
            rows.append(("seed-selection progress", progress.get("state")))
            if progress.get("age_seconds") is not None:
                rows.append(
                    (
                        "seed-selection progress age",
                        str(round(float(progress["age_seconds"]), 1)) + "s",
                    )
                )
            if progress.get("reason"):
                rows.append(("seed-selection progress detail", progress.get("reason")))
        runtime_progress = payload.get("runtime_progress")
        if isinstance(runtime_progress, dict):
            rows.append(("phase progress", runtime_progress.get("state")))
            if runtime_progress.get("age_seconds") is not None:
                rows.append(
                    (
                        "phase progress age",
                        str(round(float(runtime_progress["age_seconds"]), 1)) + "s",
                    )
                )
            if runtime_progress.get("reason"):
                rows.append(("phase progress detail", runtime_progress.get("reason")))
            ignored_errors = runtime_progress.get("ignored_errors")
            if isinstance(ignored_errors, list) and ignored_errors:
                rows.append(
                    (
                        "ignored progress records",
                        "; ".join(str(item) for item in ignored_errors[:4]),
                    )
                )
        return _section("Runtime", rows)


def _format_lifecycle_status(payload: Dict[str, Any]) -> List[str]:
    context = payload.get("lifecycle_context")
    if not isinstance(context, dict):
        return []
    rows: List[tuple[str, Any]] = [
        ("disposition", context.get("disposition")),
        ("reason code", context.get("reason_code")),
        ("message", context.get("message")),
        ("recorded from", str(context.get("from_phase")) + " iteration " + str(context.get("iteration"))),
        ("recorded at", context.get("timestamp_iso")),
    ]
    if context.get("job_id"):
        rows.append(("job id", context.get("job_id")))
    if context.get("scheduler_uncertain"):
        rows.append(("scheduler state", "uncertain; pending job identity preserved"))
    if context.get("recovery_action"):
        rows.append(("recovery action", context.get("recovery_action")))
    return _section("Lifecycle", rows)


def _iteration_summary(payload: Dict[str, Any]) -> str:
    if int(payload.get("iteration") or 0) == 0:
        return (
            "bootstrap ("
            + str(payload.get("max_iterations"))
            + " active iterations planned)"
        )
    return (
        str(payload.get("iteration"))
        + " of "
        + str(payload.get("max_iterations"))
    )


def _status_event_age(event: Mapping[str, Any]) -> Optional[str]:
    from datetime import datetime, timezone

    try:
        updated = datetime.fromisoformat(
            str(event.get("ts") or "").replace("Z", "+00:00")
        )
        if updated.tzinfo is None or updated.utcoffset() is None:
            return None
        seconds = max(
            0.0,
            (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds(),
        )
    except (TypeError, ValueError):
        return None
    elapsed = _format_elapsed_seconds(seconds)
    return elapsed + " ago" if elapsed else None


def _latest_status_journal_activity(
    payload: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[str, Optional[str]]]:
    phase = str(payload.get("phase") or "")
    try:
        iteration = int(payload.get("iteration"))
    except (TypeError, ValueError):
        return None
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for index, raw in enumerate(events):
        event = dict(raw)
        if _event_int(event, "iteration") != iteration:
            continue
        phase_values = {
            str(event.get(key) or "")
            for key in ("phase", "to_phase", "from_phase")
            if event.get(key)
        }
        if phase_values and phase not in phase_values:
            continue
        if (
            not phase_values
            and str(event.get("event") or "") not in JOURNAL_PHASE_FIRST_EVENTS
        ):
            continue
        candidates.append((index, event))
    if not candidates:
        return None
    index, event = candidates[-1]
    previous = dict(events[index - 1]) if index > 0 else None
    summary = _journal_operator_summary(
        event,
        previous_event=previous,
    ) or _journal_event_label(event)
    return summary, _status_event_age(event)


def _status_version_scope(version: int) -> str:
    return "bootstrap" if version == 0 else "iteration " + str(version)


def _status_product_description(
    *,
    version: int,
    ok: Optional[bool],
    product: str,
) -> str:
    if version < 0:
        return "not produced yet"
    scope = _status_version_scope(version)
    if ok is False:
        return "recorded through " + scope + ", but validation failed"
    if product == "model":
        return "FEREBUS model trained through " + scope
    return "committed through " + scope


def _status_progress_so_far_rows(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    phase = str(payload.get("phase") or "")
    iteration = int(payload.get("iteration") or 0)
    reference_version = int(payload.get("reference_data_version", -1))
    model_version = int(payload.get("models_version", -1))
    products = payload.get("artifact_manifest_status")
    reference_ok: Optional[bool] = None
    model_ok: Optional[bool] = None
    if isinstance(products, Mapping):
        reference = products.get("reference_data")
        models = products.get("models")
        if isinstance(reference, Mapping):
            reference_ok = reference.get("ok")
        if isinstance(models, Mapping):
            model_ok = models.get("ok")
    rows = [
        (
            "QM data",
            _status_product_description(
                version=reference_version,
                ok=reference_ok,
                product="reference",
            ),
        ),
        (
            "model",
            _status_product_description(
                version=model_version,
                ok=model_ok,
                product="model",
            ),
        ),
    ]
    if _status_artifact_problem(payload):
        readiness = "the committed data or model needs attention before processing can continue"
    elif phase == CampaignPhase.HALTED.value:
        readiness = (
            "the last committed data and model remain available, but the campaign is halted"
            if reference_version >= 0 or model_version >= 0
            else "no committed QM data or model is currently available"
        )
    elif phase == CampaignPhase.DONE.value:
        readiness = (
            "final data and model are committed through "
            + _status_version_scope(min(reference_version, model_version))
            if min(reference_version, model_version) >= 0
            else "the campaign is complete, but no committed data/model pair is available"
        )
    elif phase in _BOOTSTRAP_COLLECTION_PHASES:
        readiness = (
            "bootstrap data collection is still in progress"
            if _status_daemon_active(payload)
            else "bootstrap data collection has not finished"
        )
    elif phase == CampaignPhase.REFERENCE_COMMIT.value:
        activity = "being published" if _status_daemon_active(payload) else "ready to be published"
        readiness = (
            "bootstrap QM data is " + activity + "; initial model training follows"
            if iteration == 0
            else "QM data for iteration "
            + str(iteration)
            + " is "
            + activity
            + "; the model update follows"
        )
    elif phase == CampaignPhase.INITIAL_FEREBUS.value:
        readiness = (
            "bootstrap QM data is ready; initial model training is "
            + ("in progress" if _status_daemon_active(payload) else "pending")
        )
    elif phase == CampaignPhase.FEREBUS.value:
        readiness = (
            "QM data includes iteration "
            + str(iteration)
            + "; the model update for iteration "
            + str(iteration)
            + " is "
            + ("in progress" if _status_daemon_active(payload) else "pending")
        )
    elif phase == CampaignPhase.STOP_CHECK.value:
        readiness = (
            "data and model include completed iteration " + str(iteration)
            if reference_version == model_version == iteration
            else "the completed iteration's data/model pair needs attention"
        )
    elif iteration >= 1 and reference_version == model_version == iteration - 1:
        readiness = "data and model are up to date for iteration " + str(iteration)
    elif reference_version == model_version and reference_version >= 0:
        readiness = (
            "data and model are aligned through "
            + _status_version_scope(reference_version)
        )
    else:
        readiness = "data and model do not match the expected position for this phase"
    rows.append(("readiness", readiness))
    return rows


def _status_phase_outcome(payload: Dict[str, Any]) -> str:
    phase = str(payload.get("phase") or "")
    iteration = int(payload.get("iteration") or 0)
    scheduler_recovery = payload.get(
        "_presentation_scheduler_recovery"
    )
    if isinstance(
        payload.get("_presentation_ariadne_terminal_postprocess"),
        Mapping,
    ):
        return (
            "the daemon will validate all terminal ARIADNE output slots "
            "locally, submit no ARIADNE job, and continue to Phase B if the "
            "frozen failure policy and allocation count pass"
        )
    if isinstance(scheduler_recovery, Mapping):
        disposition = str(
            scheduler_recovery.get("publication_disposition") or ""
        )
        if disposition == "adopted":
            return (
                "the daemon will adopt the completed diversity publication "
                "and continue to the next phase"
            )
        if disposition == "rerun":
            return (
                "the daemon will rerun the interrupted diversity job, then "
                "continue normally"
            )
        retry = int(scheduler_recovery.get("n_retry") or 0)
        reusable = int(scheduler_recovery.get("n_reusable") or 0)
        if scheduler_recovery.get("state") == "legacy_unverified":
            return (
                "the daemon will retry "
                + str(retry)
                + " task"
                + ("" if retry == 1 else "s")
                + " because older recovery evidence cannot safely authorise "
                "scientific output reuse"
            )
        if scheduler_recovery.get("state") == "validated":
            return (
                "the daemon will retain "
                + str(reusable)
                + " validated task"
                + ("" if reusable == 1 else "s")
                + " and submit only "
                + str(retry)
                + " retry task"
                + ("" if retry == 1 else "s")
            )
        completed = int(
            scheduler_recovery.get("n_scheduler_completed") or 0
        )
        return (
            "the daemon will validate "
            + str(completed)
            + " scheduler-completed output"
            + ("" if completed == 1 else "s")
            + " locally and retry only unfinished or invalid tasks"
        )
    if isinstance(
        payload.get("_presentation_aimall_postprocess_recovery"),
        Mapping,
    ):
        return (
            "the daemon will validate the scheduler-completed AIMAll output "
            "candidates locally, submit only structurally invalid or "
            "unfinished AIMAll tasks, then update point allocation"
        )
    if phase == CampaignPhase.REFERENCE_COMMIT.value and iteration == 0:
        return (
            "after publishing the bootstrap QM data, the daemon will train the "
            "initial FEREBUS model"
        )
    if phase == CampaignPhase.STOP_CHECK.value:
        if iteration >= int(payload.get("max_iterations") or 0):
            return "after final checks, the daemon will mark the campaign complete"
        return (
            "the daemon will finalise iteration "
            + str(iteration)
            + " and begin iteration "
            + str(iteration + 1)
            + " unless a stopping criterion has been reached"
        )
    return _PHASE_AUTOMATIC_OUTCOMES.get(
        phase,
        "the daemon cannot determine the next phase safely",
    )


def _status_plain_reason(payload: Dict[str, Any], code: str) -> Optional[str]:
    def _lifecycle_detail() -> Optional[str]:
        context = payload.get("lifecycle_context")
        if not isinstance(context, Mapping):
            return None
        text = str(context.get("message") or "").strip().replace("\n", " ")
        for _ in range(3):
            simplified = re.sub(r"^[a-z][a-z0-9_]*:\s*", "", text)
            if simplified == text:
                break
            text = simplified
        text = re.sub(
            r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):\s*",
            "",
            text,
        ).strip()
        if not text:
            return None
        return text if len(text) <= 240 else text[:237] + "..."

    if code in {"campaign_config_invalid", "pool_feasibility_failed"}:
        return "the campaign configuration or trajectory-pool plan is not currently valid"
    if code == "config_change_reconcile_required":
        return "campaign.yaml differs from the locked configuration used by the stopped daemon"
    if code in {"config_change_blocked", "config_lock_review_failed"}:
        return "the requested configuration cannot be approved safely at the current campaign position"
    if code == "scheduler_recovery_invalid":
        return "the recorded terminal scheduler evidence does not satisfy the recovery contract"
    if code == "scheduler_terminal_recovery_resume":
        return "the previous scheduler job is conclusively terminal and only local validation or retry preparation remains"
    if code == "background_startup_failed":
        failure = str(payload.get("background_startup_failure") or "").strip()
        failure = re.sub(
            r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):\s*",
            "",
            failure,
        )
        failure = " ".join(failure.split())
        if failure:
            return (
                failure
                if len(failure) <= 240
                else failure[:237] + "..."
            )
        return "the previous background process stopped before daemon startup completed"
    if code in {
        "runtime_probe_failed",
        "state_missing",
        "state_schema_invalid",
        "state_unreadable",
    }:
        return "status cannot establish a safe campaign state from the control files"
    if code in {
        "reference_data_missing",
        "models_missing",
        "version_skew",
        "state_artifact_contract_invalid",
        "committed_artifact_invalid",
        "initial_ferebus_bootstrap_contract_problem",
        "stop_check_no_committed_pair",
    }:
        return "the recorded state and committed data/model evidence do not agree"
    if code.startswith("halted_") or str(payload.get("phase") or "") == CampaignPhase.HALTED.value:
        context = payload.get("lifecycle_context")
        detail = _lifecycle_detail()
        if isinstance(context, Mapping) and context.get("from_phase"):
            reason = "the campaign halted during " + _phase_title(
                context.get("from_phase")
            )
        else:
            reason = "the campaign is halted and requires review"
        if detail:
            reason += ": " + detail
        if code == "journal_corrupt":
            reason += "; the journal also needs attention"
        return reason
    if code in {
        "submission_intent_invalid",
        "partial_array_recovery_invalid",
        "stop_control_invalid",
        "journal_corrupt",
    }:
        return "one or more campaign control records could not be validated"
    if code == "reconcile_transaction_recoverable":
        return "a previous reconcile was interrupted, but its recorded evidence is recoverable"
    if code == "reconcile_transaction_manual_review":
        return "a previous reconcile was interrupted and its recorded evidence is ambiguous"
    return None


def _status_next_rows(
    payload: Dict[str, Any],
    *,
    campaign: Path,
) -> List[Tuple[str, str]]:
    phase = str(payload.get("phase") or "")
    overall = _status_overall(payload)
    first = _first_recommendation(payload)
    code = str(first.get("code") or "recommendation_unavailable")
    daemon_active = _status_daemon_active(payload)
    stop_request = payload.get("stop_request")
    review = code in _STATUS_REVIEW_RECOMMENDATIONS or (
        phase == CampaignPhase.DONE.value and overall == "complete"
    )
    no_action = bool(
        review
        or overall in {"running normally", "starting"}
        or (overall == "stopping as requested" and daemon_active)
    )
    if review:
        automatic = "no further daemon work is scheduled"
    elif overall == "stopped; reconcile required":
        automatic = (
            "nothing will run until reconcile applies the reviewed campaign "
            "changes"
        )
    elif overall == "startup failed":
        automatic = "the daemon did not become ready and no new work will start"
    elif overall in {"cannot determine safely", "needs attention", "blocked", "halted"}:
        automatic = "no further work can be relied upon until this condition is reviewed"
        if (
            isinstance(stop_request, Mapping)
            and str(stop_request.get("status") or "") != "completed"
        ):
            automatic += (
                "; the recorded stop request remains active and will be "
                "honoured after recovery reaches its boundary"
            )
    elif isinstance(stop_request, Mapping):
        if daemon_active and str(stop_request.get("status") or "") != "completed":
            from .daemon.stop_control import describe_stop_request

            automatic = "the daemon will honour this request: " + describe_stop_request(
                dict(stop_request)
            )
        else:
            automatic = "nothing further will run until the stopped campaign is resumed"
    elif overall == "stopped with unfinished work":
        scheduler_intents, local_intents = _status_intent_counts(payload)
        if _status_active_job_count(payload) or scheduler_intents:
            automatic = (
                "recorded "
                + _status_scheduler_display_name(payload)
                + " work may continue, but the daemon will not monitor or "
                "postprocess it until resumed"
            )
        elif local_intents:
            automatic = "prepared local work will remain paused until the daemon is resumed"
        else:
            automatic = "unfinished campaign work will remain paused until the daemon is resumed"
    elif overall == "starting":
        automatic = (
            "after startup checks finish, " + _status_phase_outcome(payload)
        )
    elif overall == "stopped and ready to continue":
        verb = "started" if phase == CampaignPhase.INIT.value else "resumed"
        automatic = (
            "nothing will run until the daemon is "
            + verb
            + "; after that, "
            + _status_phase_outcome(payload)
        )
    else:
        automatic = _status_phase_outcome(payload)

    if code == "config_change_pending_running":
        config_state = assess_campaign_presentation(payload).config_state
        user_action = (
            "nothing now; repair campaign.yaml before the next start"
            if config_state == "invalid"
            else (
                "nothing now; after the daemon stops, review or revert the "
                "blocked change before restarting"
                if config_state == "blocked"
                else (
                    "nothing now; after the daemon stops, preview reconcile "
                    "before restarting"
                )
            )
        )
        command_label = "follow progress"
    elif no_action:
        user_action = "nothing"
        command_label = "review" if review else "follow progress"
    else:
        user_action = str(first.get("primary") or "inspect the campaign status")
        command_label = "run"
    command = str(first.get("command") or "").strip()
    if not command:
        command = (
            "ichor-al-daemon journal --campaign-dir "
            + shlex.quote(str(campaign))
            + " --last-n 40"
            if no_action
            else "ichor-al-daemon status --campaign-dir "
            + shlex.quote(str(campaign))
            + " --verbose"
        )
    rows: List[Tuple[str, str]] = [
        ("automatic", automatic),
        ("you need to do", user_action),
    ]
    reason = _status_plain_reason(payload, code)
    if reason and not no_action:
        rows.append(("because", reason))
    if (
        daemon_active
        and isinstance(stop_request, Mapping)
        and str(stop_request.get("status") or "") == "requested"
        and str(stop_request.get("mode") or "")
        in {"after_phase", "after_iteration"}
    ):
        rows.append(
            (
                "cancel stop request",
                _campaign_command(campaign, "resume")
                + " --cancel-stop-request",
            )
        )
    rows.append((command_label, command))
    return rows


def _build_status_presentation(
    payload: Dict[str, Any],
    *,
    campaign: Path,
    journal_events: Sequence[Mapping[str, Any]],
) -> _StatusPresentation:
    campaign_rows: List[Tuple[str, Any]] = [
        ("phase", _phase_title(payload.get("phase"))),
        ("purpose", _phase_meaning(payload.get("phase"))),
        ("iteration", _iteration_summary(payload)),
    ]
    replacement_round = int(payload.get("replacement_round") or 0)
    if replacement_round > 0:
        campaign_rows.append(("replacement round", replacement_round))
    current_rows: List[Tuple[str, Any]] = [
        ("overall", _status_overall(payload)),
        ("daemon", _status_daemon_label(payload)),
        ("current work", _status_current_activity(payload)),
    ]
    assessment = assess_campaign_presentation(payload)
    config_count = (
        assessment.config_allowed_count + assessment.config_blocked_count
    )
    if assessment.config_state == "allowed":
        if _status_daemon_active(payload):
            config_text = (
                str(config_count)
                + " pending change"
                + ("" if config_count == 1 else "s")
                + "; this process still uses the locked configuration"
            )
        else:
            config_text = (
                str(config_count)
                + " pending change"
                + ("" if config_count == 1 else "s")
                + "; reconcile is required before resume"
            )
        current_rows.append(("configuration", config_text))
    elif assessment.config_state == "blocked":
        config_text = (
            str(assessment.config_blocked_count)
            + " blocked change"
            + ("" if assessment.config_blocked_count == 1 else "s")
        )
        if _status_daemon_active(payload):
            config_text += (
                "; this process still uses the locked configuration"
            )
        current_rows.append(("configuration", config_text))
    elif assessment.config_state == "invalid":
        config_text = "could not be compared with the campaign lock"
        if _status_daemon_active(payload):
            config_text += (
                "; this process still uses the locked configuration"
            )
        current_rows.append(
            ("configuration", config_text)
        )
    stop_request = payload.get("stop_request")
    if isinstance(stop_request, Mapping):
        from .daemon.stop_control import describe_stop_request

        current_rows.append(
            ("stop request", describe_stop_request(dict(stop_request)))
        )
    elif payload.get("stop_control_error"):
        current_rows.append(("stop request", "could not be read safely"))
    elif payload.get("shutdown_requested"):
        current_rows.append(("stop request", "shutdown is recorded in campaign state"))
    progress_rows = _status_progress_rows(payload)
    current_rows.extend(progress_rows)
    if not progress_rows:
        latest = _latest_status_journal_activity(payload, journal_events)
        if latest is not None:
            summary, age = latest
            current_rows.append(("last recorded activity", summary))
            if age:
                current_rows.append(("last update", age))
    return _StatusPresentation(
        campaign=tuple(campaign_rows),
        current_status=tuple(current_rows),
        progress_so_far=tuple(_status_progress_so_far_rows(payload)),
        what_happens_next=tuple(
            _status_next_rows(payload, campaign=campaign)
        ),
    )


def _format_status(
    payload: Dict[str, Any],
    *,
    verbose: bool,
    campaign: Path,
    journal_events: Sequence[Mapping[str, Any]],
) -> str:
    presentation = _build_status_presentation(
        payload,
        campaign=campaign,
        journal_events=journal_events,
    )
    lines: List[str] = []
    lines.extend(_section("Campaign", presentation.campaign))
    lines.append("")
    lines.extend(_section("Current status", presentation.current_status))
    lines.append("")
    lines.extend(_section("Progress so far", presentation.progress_so_far))
    lines.append("")
    lines.extend(_section("What happens next", presentation.what_happens_next))
    if not verbose:
        return "\n".join(lines) + "\n"

    lines.append("")
    lines.extend(
        _section(
            "Campaign diagnostics",
            [
                ("internal phase", payload.get("phase")),
                ("uid", payload.get("campaign_uid")),
                ("initialised", payload.get("campaign_started_iso")),
            ],
        )
    )

    pool_lines = _format_pool_feasibility_status(payload.get("pool_feasibility"))
    if pool_lines:
        lines.append("")
        lines.extend(pool_lines)
    lines.append("")
    lines.extend(
        _format_data_products_summary(
            payload.get("artifact_manifest_status"),
            state_contract=payload.get("state_artifact_contract_status"),
            phase=payload.get("phase"),
            verbose=True,
        )
    )
    lines.append("")
    lines.extend(_format_runtime_status(payload, verbose=True))
    sampling_protocol = payload.get("_presentation_sampling_protocol")
    if isinstance(sampling_protocol, Mapping):
        policy_version = int(sampling_protocol.get("policy_version") or 0)
        baseline_source = str(
            sampling_protocol.get("baseline_source") or "unavailable"
        )
        baseline_source = {
            "normalised_ariadne_landing_history": (
                "accepted ARIADNE movement history, normalised by producer preset"
            ),
            "ariadne_landing_history": "accepted ARIADNE movement history",
            "fallback": "configured fallback because usable history is unavailable",
        }.get(baseline_source, baseline_source.replace("_", " "))
        protocol_rows: List[Tuple[str, Any]] = [
            (
                "sampling aggressiveness",
                sampling_protocol.get("sampling_aggressiveness"),
            ),
            ("preset policy", "v" + str(policy_version)),
        ]
        if policy_version >= 2:
            protocol_rows.extend(
                [
                    (
                        "target movement",
                        str(sampling_protocol.get("target_motion_ratio"))
                        + "x historical baseline",
                    ),
                    (
                        "initial trust radius",
                        str(sampling_protocol.get("initial_trust_multiplier"))
                        + "x nominal",
                    ),
                ]
            )
            if policy_version >= 3:
                protocol_rows.extend(
                    [
                        (
                            "preferred movement band",
                            str(sampling_protocol.get("movement_target_low_ratio"))
                            + "-"
                            + str(sampling_protocol.get("movement_target_high_ratio"))
                            + "x target",
                        ),
                        (
                            "movement utility",
                            "lambda="
                            + str(sampling_protocol.get("lambda_move"))
                            + ", band/progress="
                            + str(sampling_protocol.get("movement_band_fraction"))
                            + "/"
                            + str(sampling_protocol.get("movement_progress_fraction")),
                        ),
                        (
                            "movement progress",
                            sampling_protocol.get("movement_progress_normalisation"),
                        ),
                        (
                            "under-movement retry",
                            "limit="
                            + str(sampling_protocol.get("under_move_retry_limit"))
                            + ", factor="
                            + str(sampling_protocol.get("under_move_feedback_min_factor"))
                            + "-"
                            + str(sampling_protocol.get("under_move_feedback_max_factor")),
                        ),
                    ]
                )
        else:
            protocol_rows.append(
                (
                    "legacy movement/trust multiplier",
                    sampling_protocol.get("movement_trust_multiplier"),
                )
            )
        protocol_rows.append(
            (
                "movement baseline",
                baseline_source,
            )
        )
        lines.append("")
        lines.extend(_section("Sampling protocol", protocol_rows))
    lifecycle = _format_lifecycle_status(payload)
    if lifecycle:
        lines.append("")
        lines.extend(lifecycle)
    partial_array = payload.get("partial_array_recovery")
    if isinstance(partial_array, dict):
        publication = payload.get("ariadne_publication_recovery")
        publication_state = (
            str(publication.get("state"))
            if isinstance(publication, Mapping)
            else None
        )
        lines.append("")
        lines.extend(
            _section(
                "Partial Array Recovery",
                [
                    (
                        "phase",
                        str(partial_array.get("phase"))
                        + "@"
                        + str(partial_array.get("iteration")),
                    ),
                    (
                        "tasks",
                        "logical="
                        + str(partial_array.get("logical_total"))
                        + " reusable="
                        + str(partial_array.get("n_reuse"))
                        + " retry="
                        + str(partial_array.get("n_retry")),
                    ),
                    ("force resubmit", partial_array.get("force_resubmit")),
                    ("ledger", partial_array.get("ledger")),
                    ("batch publication", publication_state),
                ],
            )
        )
    quarantine = payload.get("ariadne_retry_quarantine")
    if isinstance(quarantine, Mapping) and (
        quarantine.get("attempts") or quarantine.get("errors")
    ):
        lines.append("")
        lines.extend(
            _section(
                "ARIADNE Retry Quarantine",
                [
                    ("retained attempts", len(quarantine.get("attempts") or [])),
                    ("retained bytes", quarantine.get("total_bytes", 0)),
                    ("invalid entries", len(quarantine.get("errors") or [])),
                ],
            )
        )
    try:
        from .daemon.staging_retirement import retained_staging_inventory

        retained_staging = retained_staging_inventory(campaign)
    except Exception:
        retained_staging = []
    if retained_staging:
        lines.append("")
        lines.extend(
            _section(
                "Retained Staging Diagnostics",
                [("bucket", path) for path in retained_staging],
            )
        )
    lines.append("")
    lines.extend(
        _section(
            "Paths",
            [
                ("state", payload.get("state_path")),
                ("lock", payload.get("lock_path")),
                ("lease", payload.get("lease_path")),
                ("background_log", payload.get("background_log_path")),
                ("background_pid_file", payload.get("background_pid_path")),
                ("stop_request", payload.get("stop_request_path")),
            ],
        )
    )
    if payload.get("lock_probe_error"):
        lines.append("")
        lines.extend(_section("Diagnostics", [("lock_probe_error", payload["lock_probe_error"])]))
    return "\n".join(lines) + "\n"


def _format_status_unavailable(
    payload: Dict[str, Any],
    *,
    verbose: bool,
) -> str:
    first = _first_recommendation(payload)
    code = str(first.get("code") or "recommendation_unavailable")
    fresh = code in {"state_missing_fresh_init", "campaign_missing"}
    state_label = {
        "state_missing": "campaign state is missing",
        "state_schema_invalid": "campaign state is invalid",
        "state_unreadable": "campaign state cannot be read",
    }.get(str(payload.get("status_error") or ""), "campaign state is unavailable")
    if fresh:
        state_label = "campaign setup has not been completed"
    command = str(first.get("command") or "").strip()
    lines: List[str] = []
    campaign_rows: List[Tuple[str, Any]] = [
        ("phase", "unavailable"),
        ("purpose", "establish a valid campaign state before processing"),
        ("iteration", "unavailable"),
    ]
    if "campaign_yaml_exists" in payload:
        campaign_rows.append(
            (
                "campaign file",
                "campaign.yaml is present"
                if payload.get("campaign_yaml_exists")
                else "campaign.yaml is missing",
            )
        )
    lines.extend(_section("Campaign", campaign_rows))
    lines.append("")
    lines.extend(
        _section(
            "Current status",
            [
                ("overall", "setup required" if fresh else "cannot determine safely"),
                ("daemon", "not checked"),
                ("current work", state_label),
            ],
        )
    )
    lines.append("")
    lines.extend(
        _section(
            "Progress so far",
            [
                ("QM data", "not checked"),
                ("model", "not checked"),
                (
                    "readiness",
                    "committed campaign progress cannot be established without valid state",
                ),
            ],
        )
    )
    lines.append("")
    next_rows: List[Tuple[str, Any]] = [
        ("automatic", "no daemon work will start until campaign state is available"),
        ("you need to do", first.get("primary")),
    ]
    reason = _status_plain_reason(payload, code)
    if reason:
        next_rows.append(("because", reason))
    if command:
        next_rows.append(("run", command))
    lines.extend(_section("What happens next", next_rows))
    if verbose:
        lines.append("")
        lines.extend(
            _section(
                "State diagnostics",
                [
                    ("status", payload.get("status_error")),
                    ("state path", payload.get("state_path")),
                    ("error", payload.get("state_error")),
                    ("campaign.yaml present", payload.get("campaign_yaml_exists")),
                    ("stateful artefacts", payload.get("stateful_artifacts_count")),
                    ("fresh init safe", payload.get("fresh_init_safe")),
                    ("stop-control error", payload.get("stop_control_error")),
                ],
            )
        )
    return "\n".join(lines) + "\n"


def _event_time(event: Dict[str, Any]) -> str:
    ts = str(event.get("ts", "")).strip()
    if not ts:
        return "--:--:--"
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            from datetime import timezone

            parsed = parsed.replace(tzinfo=timezone.utc)
        from datetime import timezone

        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        if "T" in ts:
            date_part, tail = ts.split("T", 1)
            time_part = (
                tail.split(".", 1)[0]
                .replace("+00:00", "")
                .replace("Z", "")
            )
            if date_part and time_part:
                return date_part[:10] + " " + time_part[:8] + " UTC"
        return ts


def _event_iteration(event: Dict[str, Any]) -> str:
    value = event.get("iteration")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return "iter=n/a"
    return "iter=" + str(value)


_CAMPAIGN_PHASE_VALUES = frozenset(phase.value for phase in CampaignPhase)

_JOURNAL_CONTEXT_WIDTH = len(CampaignPhase.PHASE_A_DIVERSITY.value)
_JOURNAL_PHASE_CONTEXT_ALIASES = {
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value: "INIT_REPL_GAUSS",
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value: "INIT_REPL_AIMALL",
    CampaignPhase.INITIAL_ALLOCATION_CHECK.value: "INIT_ALLOC_CHECK",
    CampaignPhase.REPLACEMENT_GAUSSIAN.value: "REPL_GAUSSIAN",
    CampaignPhase.REPLACEMENT_AIMALL.value: "REPL_AIMALL",
}
_JOURNAL_OVERWIDTH_PHASES = frozenset(
    value
    for value in _CAMPAIGN_PHASE_VALUES
    if len(value) > _JOURNAL_CONTEXT_WIDTH
)
if set(_JOURNAL_PHASE_CONTEXT_ALIASES) != set(_JOURNAL_OVERWIDTH_PHASES):
    raise RuntimeError("journal context aliases do not cover every long phase")
_JOURNAL_RENDERED_CONTEXTS = frozenset(
    _JOURNAL_PHASE_CONTEXT_ALIASES.get(value, value)
    for value in _CAMPAIGN_PHASE_VALUES
) | frozenset(JOURNAL_EVENT_CONTEXTS.values()) | {"UNCLASSIFIED"}
if any(len(value) > _JOURNAL_CONTEXT_WIDTH for value in _JOURNAL_RENDERED_CONTEXTS):
    raise RuntimeError("journal context exceeds the presentation width")


def _event_context(event: Dict[str, Any]) -> str:
    raw = str(event.get("event") or "")
    if raw in JOURNAL_PHASE_FIRST_EVENTS:
        for key in ("phase", "to_phase", "from_phase"):
            value = event.get(key)
            if isinstance(value, str) and value in _CAMPAIGN_PHASE_VALUES:
                return _JOURNAL_PHASE_CONTEXT_ALIASES.get(value, value)
    return JOURNAL_EVENT_CONTEXTS.get(raw, "UNCLASSIFIED")


JOURNAL_EVENT_LABELS: Dict[str, str] = {
    "campaign_started": "campaign started",
    "campaign_completed": "campaign completed",
    "campaign_reopened": "completed campaign reopened",
    "scientific_convergence_reached": "scientific convergence reached",
    "phase_transition": "phase transition",
    "sbatch": "job submitted",
    "phase_succeeded": "phase completed",
    "phase_succeeded_live": "live phase accepted outputs",
    "sacct_error": "Slurm accounting error",
    "sacct_error_timeout": "Slurm accounting error timeout",
    "shutdown_requested": "shutdown requested",
    "daemon_started": "daemon started",
    "daemon_stopped": "daemon stopped",
    "daemon_interrupted": "daemon interrupted",
    "state_corrupt": "state file corrupt",
    "tick_error": "daemon tick error",
    "tick_exception_halted": "daemon halted after exception",
    "subspace_built": "subspace built",
    "reference_data_committed": "QM reference data committed",
    "reference_commit_started": "QM reference commit started",
    "reference_commit_move_progress": "QM reference move progress",
    "reference_commit_shard_progress": "FEREBUS row-shard progress",
    "reference_commit_shards_resolved": "FEREBUS row shards resolved",
    "reference_commit_cache_complete": "FEREBUS row cache complete",
    "reference_commit_published": "QM reference version published",
    "models_committed": "models committed",
    "seed_selection_started": "seed selection started",
    "seed_selection_progress": "seed selection progress",
    "seed_selection_cache": "seed selection cache",
    "seed_selected": "seeds selected",
    "anti_overlap_flagged": "anti-overlap flagged",
    "reference_scales_computed": "reference scales computed",
    "failure_action": "phase failure decision",
    "halt": "daemon halted",
    "live_postprocess_refused": "live postprocess refused",
    "effective_config_diff": "effective configuration recorded",
    "autotune_applied": "autotune applied",
    "trajectory_pool_imported": "trajectory pool imported",
    "bootstrap_inputs_confirmed": "bootstrap inputs confirmed",
    "model_bootstrap_staged": "imported models staged",
    "model_bootstrap_committed": "imported models committed",
    "quantum_output_rejected": "QM output rejected",
    "quantum_quality_summary": "QM quality summarised",
    "quantum_quality_rejected": "QM quality rejected",
    "ferebus_quality_summary": "FEREBUS quality summarised",
    "ferebus_quality_measurement_incomplete": (
        "FEREBUS quality measurement incomplete"
    ),
    "ferebus_candidate_recovery_prepared": (
        "FEREBUS candidate recovery prepared"
    ),
    "ferebus_candidate_recovery_materialised": (
        "FEREBUS candidate recovery materialised"
    ),
    "ferebus_candidate_reprocessed": "FEREBUS candidate reprocessed",
    "ferebus_candidate_rejected": "FEREBUS candidate rejected",
    "ariadne_landing_rejected": "ARIADNE landing rejected",
    "ariadne_landing_summary": "ARIADNE landing summary",
    "ariadne_optional_diagnostics_warning": "ARIADNE diagnostics warning",
    "ariadne_legacy_missing_trajectory_sha256": "ARIADNE legacy provenance",
    "ariadne_provenance_reconstructed": "ARIADNE provenance rebuilt",
    "ariadne_seed_provenance_repaired": "ARIADNE seed provenance repaired",
    "ariadne_seed_provenance_staged": "ARIADNE seed provenance staged",
    "ariadne_stale_outputs_quarantined": "ARIADNE stale outputs quarantined",
    "ariadne_publication_archived": "ARIADNE publication archived",
    "ariadne_task_rejected_missing_result": "ARIADNE result missing",
    "ariadne_task_rejected_malformed_result": "ARIADNE malformed result",
    "ariadne_task_rejected_unusable_result": "ARIADNE result unusable",
    "ariadne_task_rejected_unsafe_landing": "ARIADNE landing rejected",
    "ariadne_task_rejected_invalid_output": "ARIADNE output invalid",
    "ariadne_task_salvaged_from_nonzero_exit": "ARIADNE task salvaged",
    "error_calibration_summary": "error calibration summarised",
    "error_calibration_failed": "error calibration failed",
    "phase_output_contract_invalid": "phase output contract invalid",
    "required_phase_output_missing_after_failure": "failure output missing",
    "reconcile_applied": "reconcile applied",
    "reconcile_transaction_recovered": "interrupted reconcile recovered",
    "reconcile_resolved_terminal_intent": "terminal intent resolved",
    "partial_array_recovery_prepared": "partial array recovery prepared",
    "partial_array_recovery_postprocess_only": "partial array postprocess ready",
    "committed_artifact_settle_retry": "waiting for committed artefacts",
    "resolved_phase_resources": "resources resolved",
    "scheduler_usage_recorded": "Slurm usage recorded",
    "scheduler_usage_warning": "Slurm usage unavailable",
    "checkpoint_failed": "checkpoint failed",
    "checkpoint_verified": "checkpoint verified",
    "user_cancelled_jobs": "user cancelled jobs",
    "user_stop_requested": "user stop requested",
    "user_stop_boundary_reached": "user stop boundary reached",
    "user_stop_control_invalid": "user stop control invalid",
    "user_stop_request_cancelled": "user stop request cancelled",
    "user_stop_resumed": "stop cleared for resume",
    "stop_request_completion_deferred": "stop completion deferred",
    "sacct_empty_timeout": "Slurm accounting empty timeout",
    "sacct_missing_timeout": "Slurm accounting timeout",
    "sacct_unknown_timeout": "Slurm UNKNOWN timeout",
    "sacct_empty_but_squeue_active": "waiting for Slurm accounting",
    "sacct_rows_missing_but_squeue_active": "waiting for array accounting",
    "squeue_liveness_inconclusive": "scheduler liveness unclear",
    "scheduler_uncertain_resumed": "scheduler-uncertain job resumed",
    "transient_phase_retry": "transient phase retry",
    "transient_retry_ledger_invalid": "retry ledger invalid",
    "job_adopt_check_failed": "job adoption check failed",
    "adopted_inflight_job": "adopted running job",
    "adopted_accounted_job": "adopted completed job",
    "expected_tasks_inference_failed": "task-count inference failed",
    "submission_intent_expected_tasks_invalid": "expected task count invalid",
    "submission_intent_update_failed": "submission intent update failed",
    "submission_intent_read_failed": "submission intent read failed",
    "submission_intent_completion_deferred": "intent completion persistence deferred",
    "submission_intent_retired_without_submission": (
        "unused submission intent retired"
    ),
    "phase_completion_replayed": "phase completion replayed",
    "provenance_index_repaired": "provenance index repaired",
    "provenance_index_repair_failed": "provenance index repair failed",
    "postprocess_settle_retry": "waiting for filesystem visibility",
    "phase_pre_submit_intent": "pre-submit recorded",
    "phase_submitted": "submission recorded",
    "queue_lifecycle_update": "queue state updated",
    "staging_archived": "staging archived",
    "staging_restored_from_archive": "staging restored",
    "daemon_lease_conflict": "daemon lease conflict",
    "daemon_lease_stale_recovered": "stale daemon lease recovered",
    "daemon_lease_cleanup_failed": "daemon lease cleanup failed",
    "daemon_lease_heartbeat_failed": "daemon heartbeat write failed",
    "daemon_lease_heartbeat_recovered": "daemon heartbeat recovered",
    "environment_drift_halted": "environment change halted startup",
    "environment_rebound": "environment generation changed",
    "environment_generation_advanced": "environment generation advanced",
    "geometry_novelty_scale_precomputed": "novelty scale computed",
    "sampling_protocol_resolved": "sampling protocol resolved",
    "phase_b_novelty_threshold_relaxed": "Phase B novelty relaxed",
    "phase_b_geometry_novelty_relaxed": "Phase B geometry novelty relaxed",
    "pool_feasibility_checked": "pool feasibility checked",
    "seed_posterior_fallback": "seed posterior fallback",
    "initial_training_existing_without_bootstrap_handoff": "bootstrap handoff missing",
    "active_iteration_finalised": "active-learning iteration finalised",
    "aimall_skipped_no_gaussian_acceptances": "AIMAll skipped because Gaussian produced no accepted results",
    "aimall_quality_revalidated": "AIMAll quality revalidated",
    "ariadne_sampling_protocol_replay_failed": "ARIADNE sampling protocol replay failed",
    "legacy_sampling_protocol_repreview": "sampling protocol preview regenerated",
    "point_allocation_complete": "point allocation complete",
    "point_allocation_quantum_recorded": "QM allocation result recorded",
    "point_allocation_replacement_prepared": "replacement allocation prepared",
    "dry_run_trajectory_pool_created": "dry-run trajectory pool created",
    "daemon_startup_progress": "daemon startup progress",
    "phase_activity_started": "phase activity started",
    "phase_activity_progress": "phase activity progress",
    "phase_activity_completed": "phase activity completed",
    "phase_activity_failed": "phase activity failed",
    "scheduler_progress": "Slurm progress",
    "checkpoint_progress": "checkpoint progress",
}


def _journal_event_label(event: Dict[str, Any]) -> str:
    raw = str(event.get("event", "<missing>"))
    if str(event.get("scheduler_identity_kind") or "").lower() == "sge":
        sge_labels = {
            "sacct_error": "Sun Grid Engine accounting error",
            "sacct_error_timeout": "Sun Grid Engine accounting error timeout",
            "scheduler_usage_recorded": "Sun Grid Engine usage recorded",
            "scheduler_usage_warning": "Sun Grid Engine usage unavailable",
            "sacct_empty_timeout": "Sun Grid Engine accounting empty timeout",
            "sacct_missing_timeout": "Sun Grid Engine accounting timeout",
            "sacct_unknown_timeout": "Sun Grid Engine unknown-state timeout",
            "sacct_empty_but_squeue_active": (
                "waiting for Sun Grid Engine accounting"
            ),
            "scheduler_progress": "Sun Grid Engine progress",
        }
        if raw in sge_labels:
            return sge_labels[raw]
    if raw in JOURNAL_EVENT_LABELS:
        return JOURNAL_EVENT_LABELS[raw]
    return raw.replace("_", " ")


def _event_int(event: Dict[str, Any], key: str) -> Optional[int]:
    try:
        value = event.get(key)
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


_JOURNAL_OK_EVENTS = {
    "active_iteration_finalised",
    "aimall_quality_revalidated",
    "adopted_accounted_job",
    "campaign_completed",
    "checkpoint_verified",
    "daemon_stopped",
    "dry_run_trajectory_pool_created",
    "environment_generation_advanced",
    "environment_rebound",
    "scientific_convergence_reached",
    "phase_succeeded",
    "phase_succeeded_live",
    "reference_data_committed",
    "reference_commit_published",
    "models_committed",
    "trajectory_pool_imported",
    "bootstrap_inputs_confirmed",
    "model_bootstrap_committed",
    "phase_transition",
    "point_allocation_complete",
    "point_allocation_quantum_recorded",
    "point_allocation_replacement_prepared",
    "provenance_index_repaired",
    "reconcile_applied",
    "reconcile_transaction_recovered",
    "reconcile_resolved_terminal_intent",
    "reference_commit_shards_resolved",
    "reference_scales_computed",
    "scheduler_usage_recorded",
    "seed_selected",
    "staging_archived",
    "staging_restored_from_archive",
    "subspace_built",
    "error_calibration_summary",
    "submission_intent_retired_without_submission",
    "user_stop_boundary_reached",
    "user_stop_request_cancelled",
    "user_stop_resumed",
}

_JOURNAL_RUN_EVENTS = {
    "campaign_started",
    "daemon_started",
    "phase_pre_submit_intent",
    "phase_submitted",
    "sbatch",
    "adopted_inflight_job",
    "partial_array_recovery_prepared",
    "partial_array_recovery_postprocess_only",
    "scheduler_uncertain_resumed",
    "reference_commit_started",
    "reference_commit_move_progress",
    "reference_commit_cache_complete",
    "ferebus_candidate_recovery_prepared",
    "ferebus_candidate_recovery_materialised",
    "model_bootstrap_staged",
    "reference_commit_shard_progress",
    "seed_selection_started",
    "seed_selection_progress",
}

_JOURNAL_WAIT_EVENTS = {
    "sacct_empty_but_squeue_active",
    "sacct_rows_missing_but_squeue_active",
    "postprocess_settle_retry",
    "committed_artifact_settle_retry",
    "user_stop_requested",
    "shutdown_requested",
    "stop_request_completion_deferred",
}

_JOURNAL_WARN_EVENTS = {
    "aimall_skipped_no_gaussian_acceptances",
    "anti_overlap_flagged",
    "ariadne_legacy_missing_trajectory_sha256",
    "ariadne_provenance_reconstructed",
    "ariadne_publication_archived",
    "ariadne_seed_provenance_repaired",
    "ariadne_seed_provenance_staged",
    "ariadne_stale_outputs_quarantined",
    "ariadne_task_rejected_invalid_output",
    "ariadne_task_rejected_malformed_result",
    "ariadne_task_rejected_missing_result",
    "ariadne_task_rejected_unsafe_landing",
    "ariadne_task_rejected_unusable_result",
    "ariadne_task_salvaged_from_nonzero_exit",
    "campaign_reopened",
    "phase_completion_replayed",
    "submission_intent_completion_deferred",
    "sacct_error",
    "quantum_output_rejected",
    "ariadne_landing_rejected",
    "ariadne_optional_diagnostics_warning",
    "phase_b_novelty_threshold_relaxed",
    "phase_b_geometry_novelty_relaxed",
    "quantum_quality_rejected",
    "daemon_lease_stale_recovered",
    "daemon_interrupted",
    "daemon_lease_cleanup_failed",
    "daemon_lease_heartbeat_failed",
    "daemon_lease_heartbeat_recovered",
    "ferebus_quality_measurement_incomplete",
    "initial_training_existing_without_bootstrap_handoff",
    "legacy_sampling_protocol_repreview",
    "provenance_index_repair_failed",
    "scheduler_usage_warning",
    "seed_posterior_fallback",
    "squeue_liveness_inconclusive",
    "transient_phase_retry",
}

_JOURNAL_FAIL_EVENTS = {
    "ariadne_sampling_protocol_replay_failed",
    "checkpoint_failed",
    "daemon_lease_conflict",
    "environment_drift_halted",
    "expected_tasks_inference_failed",
    "ferebus_candidate_rejected",
    "halt",
    "tick_error",
    "tick_exception_halted",
    "state_corrupt",
    "phase_output_contract_invalid",
    "required_phase_output_missing_after_failure",
    "sacct_empty_timeout",
    "sacct_missing_timeout",
    "sacct_unknown_timeout",
    "sacct_error_timeout",
    "live_postprocess_refused",
    "error_calibration_failed",
    "job_adopt_check_failed",
    "submission_intent_expected_tasks_invalid",
    "submission_intent_read_failed",
    "submission_intent_update_failed",
    "transient_retry_ledger_invalid",
    "user_stop_control_invalid",
}

_JOURNAL_INFO_EVENTS = {
    "autotune_applied",
    "effective_config_diff",
    "geometry_novelty_scale_precomputed",
    "pool_feasibility_checked",
    "resolved_phase_resources",
    "sampling_protocol_resolved",
    "seed_selection_cache",
}

_JOURNAL_DYNAMIC_EVENTS = {
    "ariadne_landing_summary",
    "failure_action",
    "ferebus_candidate_reprocessed",
    "ferebus_quality_summary",
    "quantum_quality_summary",
    "queue_lifecycle_update",
    "user_cancelled_jobs",
    "daemon_startup_progress",
    "phase_activity_started",
    "phase_activity_progress",
    "phase_activity_completed",
    "phase_activity_failed",
    "scheduler_progress",
    "checkpoint_progress",
}

_SQUEUE_RUNNING_STATES = {"R", "RUNNING", "CG", "COMPLETING"}
_SQUEUE_PENDING_STATES = {"PD", "PENDING", "CF", "CONFIGURING"}


def _journal_event_severity(event: Dict[str, Any]) -> str:
    raw = str(event.get("event", ""))
    if raw in {
        "daemon_startup_progress",
        "phase_activity_started",
        "phase_activity_progress",
        "phase_activity_completed",
        "phase_activity_failed",
        "scheduler_progress",
        "checkpoint_progress",
    }:
        status = str(event.get("status") or "running").lower()
        if status in {"failed", "error", "blocked"}:
            return "FAIL"
        if status in {"completed", "verified", "published"}:
            return "OK"
        if status in {"waiting", "pending"}:
            return "WAIT"
        return "RUN"
    if raw == "ferebus_candidate_reprocessed":
        outcome = str(event.get("outcome") or "")
        if outcome == "accepted":
            return (
                "WARN"
                if (_event_int(event, "n_warnings") or 0) > 0
                else "OK"
            )
        if outcome == "measurement_incomplete":
            return "WARN"
        if outcome in {"rejected", "failed"}:
            return "FAIL"
        return "INFO"
    if raw == "ariadne_landing_summary":
        rejected = _event_int(event, "rejected")
        return "WARN" if rejected is not None and rejected > 0 else "OK"
    if raw == "queue_lifecycle_update":
        queue_event = str(event.get("queue_event") or "")
        status = str(event.get("status") or "").upper()
        user_cancelled = bool(
            event.get("user_requested_cancellation") is True
            or str(event.get("terminal_cause") or "") == "user_stop"
        )
        if queue_event == "postprocess_started":
            return "RUN"
        if queue_event == "postprocess_finished":
            return (
                "FAIL"
                if status in {"FAILED", "FAILURE", "ERROR"}
                else "OK"
            )
        if queue_event == "first_sacct":
            return "OK"
        if status == "CANCELLED" and user_cancelled:
            return "OK"
        if status in {
            "FAILED",
            "FAILURE",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
        }:
            return "FAIL"
        if status in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
            return "OK"
        if status in _SQUEUE_PENDING_STATES:
            return "WAIT"
        if status in _SQUEUE_RUNNING_STATES:
            return "RUN"
        return "INFO"
    if raw == "user_cancelled_jobs":
        failed = _event_int(event, "n_failed") or 0
        skipped = _event_int(event, "n_skipped") or 0
        if failed > 0:
            return "FAIL"
        if skipped > 0:
            return "WARN"
        return "OK"
    if raw == "failure_action":
        action = str(event.get("action") or "").upper()
        if action == "HALT":
            return "FAIL"
        if action in {"SCRUB", "RETRY", "CONTINUE"}:
            return "WARN"
        return "INFO"
    if raw in {"quantum_quality_summary", "ferebus_quality_summary"}:
        accepted = event.get("accepted")
        rejected = _event_int(event, "n_rejected")
        total = _event_int(event, "n_total")
        if accepted is False or (
            total is not None and total > 0 and rejected is not None and rejected >= total
        ):
            return "FAIL"
        if rejected is not None and rejected > 0:
            return "WARN"
        if raw == "ferebus_quality_summary" and (
            (_event_int(event, "n_warned") or 0) > 0
            or (_event_int(event, "n_warnings") or 0) > 0
        ):
            return "WARN"
        return "OK"
    if raw in _JOURNAL_FAIL_EVENTS:
        return "FAIL"
    if raw in _JOURNAL_WARN_EVENTS:
        return "WARN"
    if raw in _JOURNAL_WAIT_EVENTS:
        return "WAIT"
    if raw in _JOURNAL_RUN_EVENTS:
        return "RUN"
    if raw in _JOURNAL_OK_EVENTS:
        return "OK"
    if raw in _JOURNAL_INFO_EVENTS:
        return "INFO"
    return "INFO"


def _squeue_counts(event: Dict[str, Any]) -> Dict[str, int]:
    raw_counts = event.get("squeue_state_counts")
    out: Dict[str, int] = {}
    if isinstance(raw_counts, dict):
        for key, value in raw_counts.items():
            try:
                out[str(key).upper()] = int(value)
            except (TypeError, ValueError):
                continue
    return out


def _squeue_running_pending(event: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    counts = _squeue_counts(event)
    if not counts:
        return None, None
    running = sum(count for state, count in counts.items() if state in _SQUEUE_RUNNING_STATES)
    pending = sum(count for state, count in counts.items() if state in _SQUEUE_PENDING_STATES)
    return int(running), int(pending)


def _format_progress_value(value: Optional[int]) -> str:
    return "-" if value is None else str(int(value))


def _journal_array_progress(event: Dict[str, Any]) -> str:
    total = (
        _event_int(event, "n_expected")
        or _event_int(event, "expected_tasks")
        or _event_int(event, "n_tasks")
        or _event_int(event, "logical_total")
    )
    completed = _event_int(event, "n_completed")
    if completed is None and str(event.get("event", "")).startswith("partial_array_"):
        completed = _event_int(event, "n_reuse")
    failed = _event_int(event, "n_failed")
    missing = _event_int(event, "n_missing")
    raw = str(event.get("event", ""))
    if completed is None and raw in {
        "phase_pre_submit_intent",
        "phase_submitted",
        "sbatch",
        "adopted_inflight_job",
    }:
        completed = 0
    running, pending = _squeue_running_pending(event)
    if total is None:
        return ""
    if total <= 1 and failed in (None, 0) and missing in (None, 0):
        return ""
    parts = [
        "total="
        + str(int(total))
        + " completed="
        + _format_progress_value(completed)
        + " running="
        + _format_progress_value(running)
        + " pending="
        + _format_progress_value(pending)
    ]
    if failed is not None and failed > 0:
        parts.append("fail=" + str(int(failed)))
    if missing is not None and missing > 0:
        parts.append("missing=" + str(int(missing)))
    recovery = event.get("array_recovery")
    if isinstance(recovery, dict):
        reuse = _event_int(recovery, "n_reuse")
        retry = _event_int(recovery, "n_retry")
        logical = _event_int(recovery, "logical_total")
        if reuse is not None:
            parts.append("reuse=" + str(int(reuse)))
        if retry is not None:
            parts.append("retry=" + str(int(retry)))
        if logical is not None and logical != total:
            parts.append("logical=" + str(int(logical)))
    return " ".join(parts)


_JOURNAL_ARRAY_PROGRESS_EVENTS = frozenset(
    {
        "sbatch",
        "phase_pre_submit_intent",
        "phase_submitted",
        "adopted_accounted_job",
        "adopted_inflight_job",
        "queue_lifecycle_update",
        "phase_succeeded",
        "phase_succeeded_live",
        "scheduler_progress",
        "user_cancelled_jobs",
        "reconcile_resolved_terminal_intent",
        "partial_array_recovery_postprocess_only",
        "partial_array_recovery_prepared",
        "sacct_empty_timeout",
        "sacct_unknown_timeout",
        "sacct_missing_timeout",
        "sacct_empty_but_squeue_active",
        "sacct_rows_missing_but_squeue_active",
        "squeue_liveness_inconclusive",
    }
)


def _journal_scheduler_name(event: Dict[str, Any]) -> str:
    return (
        "Sun Grid Engine"
        if str(event.get("scheduler_identity_kind") or "").lower() == "sge"
        else "Slurm"
    )


def _sacct_row_summary(event: Dict[str, Any]) -> str:
    expected = _event_int(event, "n_expected")
    observed = _event_int(event, "n_observed")
    missing = _event_int(event, "n_missing")
    parts: List[str] = []
    if observed is not None:
        parts.append("observed=" + str(observed))
    if expected is not None:
        parts.append("expected=" + str(expected))
    if missing is not None:
        parts.append("missing=" + str(missing))
    return " ".join(parts)


def _journal_operator_summary(
    event: Dict[str, Any],
    *,
    previous_event: Optional[Dict[str, Any]] = None,
) -> str:
    raw = str(event.get("event", ""))
    if raw in {
        "daemon_startup_progress",
        "phase_activity_started",
        "phase_activity_progress",
        "phase_activity_completed",
        "phase_activity_failed",
        "scheduler_progress",
        "checkpoint_progress",
    }:
        from .daemon.phase_progress import format_progress_stage

        message = format_progress_stage(event.get("stage"))
        completed = _event_int(event, "completed")
        total = _event_int(event, "total")
        unit = str(event.get("unit") or "items")
        if raw == "scheduler_progress":
            counts: List[str] = []
            if completed is not None:
                counts.append(
                    str(completed)
                    + ("/" + str(total) if total is not None else "")
                    + " completed"
                )
            for key in ("running", "pending", "failed", "missing"):
                value = _event_int(event, key)
                if value is not None and (value > 0 or key in {"running", "pending"}):
                    counts.append(str(value) + " " + key)
            if counts:
                message += ": " + ", ".join(counts) + " " + unit
            diagnostics = []
            for key, label in (
                ("pending_reason", "pending reason"),
                ("pending_queue", "queue"),
                ("scheduler_native_state", "native state"),
            ):
                value = " ".join(str(event.get(key) or "").split())
                if value:
                    diagnostics.append(label + "=" + value[:160])
            if diagnostics:
                message += "; " + ", ".join(diagnostics)
        elif completed is not None and total is not None:
            message += ": " + str(completed) + "/" + str(total) + " " + unit
        elif completed is not None:
            message += ": " + str(completed) + " " + unit
        progress_status = str(event.get("status") or "running").lower()
        if progress_status == "completed":
            message = "Finished " + message[:1].lower() + message[1:]
        elif progress_status == "failed":
            message = "Failed while " + message.lower()
        return message
    if raw == "seed_selection_started":
        return "seed selection started"
    if raw == "seed_selection_progress":
        return _format_seed_selection_progress(event).rstrip(".")
    if raw == "seed_selection_cache":
        return (
            str(event.get("cache_kind") or "derived data").replace("_", " ")
            + " cache "
            + str(event.get("cache_status") or "updated").replace("_", " ")
        )
    if raw == "resolved_phase_resources":
        n_tasks = (
            _event_int(event, "n_tasks")
            or _event_int(event, "expected_tasks")
        )
        task_text = (
            ""
            if n_tasks is None
            else " for " + str(n_tasks) + " task" + ("" if n_tasks == 1 else "s")
        )
        evidence_mode = str(event.get("resource_evidence_mode") or "")
        dimension_source = str(event.get("gradient_dimension_source") or "")
        if evidence_mode == "reused":
            source = str(
                event.get("resource_evidence_source_attempt_id")
                or event.get("resource_evidence_source_submission_identity")
                or "an authenticated prior attempt"
            )
            return (
                "resources resolved"
                + task_text.replace(" task", " retry task")
                + " using validated ARIADNE evidence from "
                + source
            )
        if dimension_source == "configured_safe_upper_bound":
            dimension = _event_int(event, "gradient_dimension")
            return (
                "resources resolved"
                + task_text
                + " using configured ARIADNE dimension bound "
                + ("unknown" if dimension is None else str(dimension))
            )
        if dimension_source:
            return "resources resolved" + task_text + " using exact ARIADNE dimensions"
        return "resources resolved" + task_text
    if raw == "user_cancelled_jobs":
        completed = _event_int(event, "n_scheduler_completed_tasks")
        retry = _event_int(event, "n_retry_tasks")
        failed = _event_int(event, "n_failed") or 0
        skipped = _event_int(event, "n_skipped") or 0
        if completed is not None and retry is not None:
            description = (
                "scheduler cancellation recorded: "
                + str(completed)
                + " completed task"
                + ("" if completed == 1 else "s")
                + " preserved for validation, "
                + str(retry)
                + " task"
                + ("" if retry == 1 else "s")
                + " to retry"
            )
            if failed:
                description += ", " + str(failed) + " job cancellation failure"
                if failed != 1:
                    description += "s"
            if skipped:
                description += ", " + str(skipped) + " inconclusive job"
                if skipped != 1:
                    description += "s"
            return description
    if (
        raw == "reconcile_resolved_terminal_intent"
        and event.get("ariadne_terminal_postprocess") is True
    ):
        completed = _event_int(
            event,
            "scheduler_completed_task_candidates",
        ) or 0
        failed = _event_int(
            event,
            "scheduler_failed_task_candidates",
        ) or 0
        return (
            "terminal ARIADNE intent resolved: "
            + str(completed)
            + " completed candidate"
            + ("" if completed == 1 else "s")
            + ", "
            + str(failed)
            + " failed slot"
            + ("" if failed == 1 else "s")
            + " retained for local validation, no tasks resubmitted"
        )
    if raw == "reconcile_applied":
        config_changes = _event_int(event, "n_allowed_config_changes") or 0

        def _with_config_changes(description: str) -> str:
            if config_changes <= 0:
                return description
            return (
                description
                + "; "
                + str(config_changes)
                + " configuration change"
                + ("" if config_changes == 1 else "s")
                + " applied"
            )

        terminal_ariadne_completed = _event_int(
            event,
            "ariadne_scheduler_completed_candidates",
        )
        terminal_ariadne_failed = _event_int(
            event,
            "ariadne_scheduler_failed_candidates",
        )
        terminal_ariadne_submitted = _event_int(
            event,
            "ariadne_tasks_resubmitted",
        )
        if (
            event.get("ariadne_terminal_postprocess") is True
            and terminal_ariadne_completed is not None
            and terminal_ariadne_failed is not None
            and terminal_ariadne_submitted == 0
        ):
            return _with_config_changes(
                "reconcile applied; preserving "
                + str(terminal_ariadne_completed)
                + " scheduler-completed ARIADNE output candidate"
                + ("" if terminal_ariadne_completed == 1 else "s")
                + " and "
                + str(terminal_ariadne_failed)
                + " scheduler-failed slot"
                + ("" if terminal_ariadne_failed == 1 else "s")
                + " for local validation, no ARIADNE tasks resubmitted"
            )

        diversity_selected = _event_int(
            event,
            "diversity_selected_geometries",
        )
        diversity_retry = _event_int(event, "diversity_retry_tasks")
        diversity_submitted = _event_int(
            event,
            "diversity_tasks_resubmitted",
        )
        if (
            diversity_selected is not None
            and diversity_retry is not None
            and diversity_submitted is not None
        ):
            if diversity_retry:
                return _with_config_changes(
                    "reconcile applied; archived an incomplete diversity "
                    "publication, prepared "
                    + str(diversity_retry)
                    + " scalar task"
                    + ("" if diversity_retry == 1 else "s")
                    + " for retry after resume, no scheduler job submitted"
                )
            return _with_config_changes(
                "reconcile applied; preserving "
                + str(diversity_selected)
                + " selected diversity geometr"
                + ("y" if diversity_selected == 1 else "ies")
                + " for local validation, no diversity job resubmitted"
            )
        scheduler_completed = _event_int(
            event,
            "scheduler_completed_task_candidates",
        )
        scheduler_retry = _event_int(event, "scheduler_retry_tasks")
        scheduler_submitted = _event_int(
            event,
            "scheduler_tasks_resubmitted",
        )
        if (
            scheduler_completed is not None
            and scheduler_retry is not None
            and scheduler_submitted is not None
        ):
            submission_text = (
                "no jobs submitted"
                if scheduler_submitted == 0
                else str(scheduler_submitted) + " retry tasks submitted"
            )
            return _with_config_changes(
                "reconcile applied; preserving "
                + str(scheduler_completed)
                + " scheduler-completed task candidate"
                + ("" if scheduler_completed == 1 else "s")
                + " for local validation, "
                + str(scheduler_retry)
                + " task"
                + ("" if scheduler_retry == 1 else "s")
                + " prepared for retry, "
                + submission_text
            )
        accepted = _event_int(event, "ariadne_accepted_tasks")
        rejected = _event_int(event, "ariadne_rejected_tasks")
        resubmitted = _event_int(event, "ariadne_tasks_resubmitted")
        if accepted is not None and rejected is not None and resubmitted is not None:
            rejected_text = (
                "no rejected tasks"
                if rejected == 0
                else str(rejected) + " rejected tasks excluded"
            )
            resubmitted_text = (
                "no ARIADNE jobs resubmitted"
                if resubmitted == 0
                else str(resubmitted) + " ARIADNE tasks resubmitted"
            )
            return _with_config_changes(
                "reconcile applied; reusing "
                + str(accepted)
                + " accepted ARIADNE results, "
                + rejected_text
                + ", "
                + resubmitted_text
            )
        aimall_completed = _event_int(event, "aimall_completed_outputs")
        aimall_resubmitted = _event_int(event, "aimall_tasks_resubmitted")
        if aimall_completed is not None and aimall_resubmitted is not None:
            return _with_config_changes(
                "reconcile applied; preserving "
                + str(aimall_completed)
                + " scheduler-completed AIMAll output candidate"
                + ("" if aimall_completed == 1 else "s")
                + " for local validation; reconcile submitted no work"
            )
        if config_changes:
            return _with_config_changes("reconcile applied")
    if raw in {"user_stop_requested", "user_stop_boundary_reached"}:
        from .daemon.stop_control import describe_stop_request

        description = describe_stop_request(
            event,
            completed=raw == "user_stop_boundary_reached",
        )
        resulting_phase = event.get("resulting_phase")
        resulting_iteration = _event_int(event, "resulting_iteration")
        if (
            raw == "user_stop_boundary_reached"
            and isinstance(resulting_phase, str)
            and resulting_phase
            and resulting_iteration is not None
        ):
            if str(event.get("mode") or "") == "after_iteration":
                target = _event_int(event, "target_iteration")
                if target is not None:
                    description = "iteration " + str(target) + " completed"
            return (
                description
                + "; campaign paused before "
                + resulting_phase
                + " iteration "
                + str(resulting_iteration)
            )
        return description
    if raw in {"phase_pre_submit_intent", "phase_submitted", "sbatch"}:
        return "array submitted" if _journal_array_progress(event) else "job submitted"
    if raw == "phase_transition":
        if (
            isinstance(previous_event, dict)
            and str(previous_event.get("event") or "")
            == "user_stop_boundary_reached"
            and str(previous_event.get("resulting_phase") or "")
            == str(event.get("to_phase") or "")
            and _event_int(previous_event, "resulting_iteration")
            == _event_int(event, "iteration")
        ):
            return (
                "next phase recorded for resume: "
                + str(event.get("to_phase") or "unknown")
            )
        return (
            "phase changed from "
            + str(event.get("from_phase") or "unknown")
            + " -> "
            + str(event.get("to_phase") or "unknown")
        )
    if raw == "queue_lifecycle_update":
        queue_event = str(event.get("queue_event") or "")
        status = str(event.get("status") or "").upper()
        scheduler_name = _journal_scheduler_name(event)
        user_cancelled = bool(
            event.get("user_requested_cancellation") is True
            or str(event.get("terminal_cause") or "") == "user_stop"
        )
        if queue_event == "postprocess_started":
            return "local postprocessing started"
        if queue_event == "postprocess_finished":
            return (
                "local postprocessing failed"
                if status in {"FAILED", "FAILURE", "ERROR"}
                else "local postprocessing completed"
            )
        if queue_event == "terminal":
            if status == "CANCELLED" and user_cancelled:
                return scheduler_name + " work cancelled as requested"
            return (
                scheduler_name + " tasks failed"
                if status in {
                    "FAILED",
                    "FAILURE",
                    "CANCELLED",
                    "TIMEOUT",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                }
                else scheduler_name + " tasks completed"
            )
        if queue_event == "first_sacct":
            return scheduler_name + " accounting available"
        if status in _SQUEUE_PENDING_STATES:
            return "array pending" if _journal_array_progress(event) else "job pending"
        if status in _SQUEUE_RUNNING_STATES:
            return "array active" if _journal_array_progress(event) else "job active"
        return "queue state updated"
    if raw in {"phase_succeeded", "phase_succeeded_live"}:
        return "array complete" if _journal_array_progress(event) else "phase succeeded"
    if raw == "halt":
        failure = classify_operator_failure(
            event.get("reason") or event.get("error")
        )
        return "campaign halted; " + failure.summary
    if raw == "sacct_rows_missing_but_squeue_active":
        return "waiting for accounting"
    if raw == "sacct_empty_but_squeue_active":
        return "waiting for accounting"
    if raw in {"sacct_missing_timeout", "sacct_empty_timeout", "sacct_unknown_timeout"}:
        return "accounting timeout"
    if raw == "squeue_liveness_inconclusive":
        return "scheduler liveness unclear"
    if raw == "pool_feasibility_checked":
        frames = event.get("pool_n_frames")
        required = event.get("required_pool_frames")
        return "pool frames=" + str(frames) + " required=" + str(required)
    if raw == "failure_action":
        failed = event.get("n_failed")
        tasks = event.get("n_tasks")
        action = event.get("action")
        return "action=" + str(action) + " failed=" + str(failed) + "/" + str(tasks)
    if raw.startswith("reference_commit_"):
        moved = event.get("moved_points")
        total = event.get("total_points")
        if moved is not None and total is not None:
            return "reference points=" + str(moved) + "/" + str(total)
        repaired = event.get("shards_repaired")
        reused = event.get("shards_reused")
        if repaired is not None or reused is not None:
            return (
                "row shards reused="
                + str(reused or 0)
                + " repaired="
                + str(repaired or 0)
            )
    return ""


_JOURNAL_THROUGHPUT_FIELDS = frozenset(
    {"throughput", "throughput_per_second", "throughput_per_s"}
)


def _compact_event_details(event: Dict[str, Any]) -> str:
    detail_keys = [
        ("job_id", "job"),
        ("campaign_uid", "uid"),
        ("accepted", "accepted"),
        ("rejected", "rejected"),
        ("salvaged", "salvaged"),
        ("backtracked", "backtracked"),
        ("n_tasks", "tasks"),
        ("n_kept", "kept"),
        ("n_rejected", "rejected"),
        ("n_warned", "warned"),
        ("n_warnings", "warnings"),
        ("n_frames", "frames"),
        ("moved_points", "moved"),
        ("total_points", "total"),
        ("moved_bytes", "bytes"),
        ("shards_reused", "shards_reused"),
        ("shards_repaired", "shards_repaired"),
        ("elapsed_seconds", "elapsed_s"),
        ("pool_n_frames", "pool"),
        ("required_pool_frames", "required"),
        ("action", "action"),
        ("reason", "reason"),
        ("error", "error"),
    ]
    parts: List[str] = []
    seen_labels: set[str] = set()
    raw = str(event.get("event", ""))
    for key, label in detail_keys:
        if raw == "seed_selection_progress" and key == "elapsed_seconds":
            continue
        if raw == "resolved_phase_resources" and key == "n_tasks":
            continue
        if raw == "halt" and key in {"reason", "error"}:
            continue
        if key in event and event.get(key) is not None:
            if label in seen_labels:
                continue
            value = _format_journal_detail_value(key, event.get(key))
            if key == "campaign_uid" and len(value) > 8:
                value = value[:8]
            if key in {"reason", "error"} and len(value) > 90:
                value = value[:87] + "..."
            parts.append(label + "=" + value)
            seen_labels.add(label)
    if _event_int(event, "_aggregated_count") not in {None, 1}:
        parts.insert(0, "events=" + str(_event_int(event, "_aggregated_count")))
    progress = (
        _journal_array_progress(event)
        if raw in _JOURNAL_ARRAY_PROGRESS_EVENTS
        else ""
    )
    if progress:
        insert_at = 1 if parts and parts[0].startswith("job=") else 0
        parts.insert(insert_at, progress)
    if raw in {"sacct_missing_timeout", "sacct_empty_timeout", "sacct_unknown_timeout"}:
        streak = _event_int(event, "streak")
        max_ticks = _event_int(event, "max_ticks")
        if streak is not None and max_ticks is not None:
            parts.append("ticks=" + str(streak) + "/" + str(max_ticks))
    return " ".join(parts)


def _verbose_event_details(event: Dict[str, Any]) -> str:
    raw = str(event.get("event", "<missing>"))
    skip = {
        "ts",
        "event",
        "phase",
        "to_phase",
        "from_phase",
        "iteration",
        "job_id",
        "campaign_uid",
        "accepted",
        "rejected",
        "salvaged",
        "backtracked",
        "n_tasks",
        "n_kept",
        "n_rejected",
        "n_warned",
        "n_warnings",
        "n_frames",
        "pool_n_frames",
        "required_pool_frames",
        "action",
        "reason",
        "error",
        "expected_tasks",
        "n_expected",
        "n_completed",
        "n_failed",
        "n_missing",
        "squeue_state_counts",
    }
    parts = ["raw=" + raw]
    if raw == "halt":
        for key in ("reason", "error"):
            if event.get(key) is not None:
                parts.append(
                    "raw_" + key + "=" + _format_value(event.get(key))
                )
    for key in sorted(event):
        if key in skip or key in _JOURNAL_THROUGHPUT_FIELDS:
            continue
        value = _format_journal_detail_value(key, event[key])
        if len(value) > 60:
            value = value[:57] + "..."
        parts.append(str(key) + "=" + value)
    return " ".join(parts)


_AGGREGATED_JOURNAL_EVENTS = {
    "ariadne_provenance_reconstructed",
    "ariadne_seed_provenance_repaired",
    "ariadne_seed_provenance_staged",
    "ariadne_task_rejected_invalid_output",
    "ariadne_task_rejected_malformed_result",
    "ariadne_task_rejected_missing_result",
    "ariadne_task_rejected_unsafe_landing",
    "ariadne_task_rejected_unusable_result",
    "ariadne_task_salvaged_from_nonzero_exit",
    "point_allocation_quantum_recorded",
    "reference_commit_shard_progress",
}


def _aggregate_journal_events(
    events: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse repetitive task records while preserving first-seen order."""
    aggregated: List[Dict[str, Any]] = []
    positions: Dict[Tuple[str, str, str, str], int] = {}
    for event in events:
        raw = str(event.get("event") or "")
        if raw not in _AGGREGATED_JOURNAL_EVENTS:
            aggregated.append(event)
            continue
        key = (
            raw,
            str(event.get("phase") or ""),
            str(event.get("iteration") if event.get("iteration") is not None else ""),
            str(event.get("reason") or event.get("error") or ""),
        )
        if key not in positions:
            compact = dict(event)
            compact["_aggregated_count"] = 1
            compact.pop("job_id", None)
            positions[key] = len(aggregated)
            aggregated.append(compact)
            continue
        index = positions[key]
        aggregated[index]["_aggregated_count"] = int(
            aggregated[index].get("_aggregated_count", 1)
        ) + 1
    return aggregated


def _format_journal_events(events: Sequence[Dict[str, Any]], *, verbose: bool) -> str:
    if not events:
        return "Timeline\n  no matching events\n"
    rows: List[Tuple[Dict[str, Any], str, str, str, str, str]] = []
    previous_event: Optional[Dict[str, Any]] = None
    for event in events:
        summary = _journal_operator_summary(
            event,
            previous_event=previous_event,
        ) or _journal_event_label(event)
        context = _event_context(event)
        rows.append(
            (
                event,
                _event_time(event),
                _event_iteration(event),
                context,
                _journal_event_severity(event),
                summary,
            )
        )
        previous_event = event
    time_width = max(19, max(len(row[1]) for row in rows))
    iteration_width = max(6, max(len(row[2]) for row in rows))
    context_width = _JOURNAL_CONTEXT_WIDTH

    lines: List[str] = ["Timeline"]
    for event, event_time, iteration, context, severity, summary in rows:
        line = (
            "  "
            + event_time.ljust(time_width)
            + "  "
            + ("[" + severity + "]").ljust(7)
            + "  "
            + context.ljust(context_width)
            + "   "
            + iteration.ljust(iteration_width)
            + "      "
            + summary
        )
        details = _compact_event_details(event)
        if details:
            line += "  " + details
        if verbose:
            verbose_details = _verbose_event_details(event)
            if verbose_details:
                line += " " + verbose_details
        lines.append(line)
    return "\n".join(lines) + "\n"


def cmd_start(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(args.campaign_dir)
    _report_background_startup(
        "starting",
        "campaign_validation",
        campaign_dir=str(campaign),
    )
    if not campaign.exists():
        print("campaign-dir does not exist: " + str(campaign), file=sys.stderr)
        return 2

    config_path = Path(args.config).resolve() if args.config else campaign / "campaign.yaml"
    if not config_path.exists():
        print("campaign config not found: " + str(config_path), file=sys.stderr)
        return 2
    if bool(getattr(args, "foreground", False)) and bool(
        getattr(args, "background", False)
    ):
        print("--foreground and --background are mutually exclusive", file=sys.stderr)
        return 2
    if bool(getattr(args, "foreground", False)) and (
        getattr(args, "background_log", None)
        or getattr(args, "background_pid", None)
    ):
        print(
            "--background-log and --background-pid cannot be used with --foreground",
            file=sys.stderr,
        )
        return 2
    if os.environ.get(BACKGROUND_CHILD_ENV) == "1" and bool(
        getattr(args, "background", False)
    ):
        print(
            "--background is not allowed inside a background child process",
            file=sys.stderr,
        )
        return 2
    if os.environ.get(BACKGROUND_CHILD_ENV) == "1":
        args.background = False
    elif not bool(getattr(args, "foreground", False)) and not bool(
        getattr(args, "background", False)
    ):
        args.background = True

    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    state_for_lock = None
    if not state_path.exists():
        missing_context = _missing_state_context(campaign)
        artefacts = list(missing_context.get("stateful_artifacts") or [])
        if bool(missing_context.get("fresh_init_safe", False)):
            print(
                "state.json is missing for a fresh campaign.",
                file=sys.stderr,
            )
            print("Run:", file=sys.stderr)
            print(
                "  " + _campaign_command(campaign, "init"),
                file=sys.stderr,
            )
            print("Then start the live daemon with:", file=sys.stderr)
            print(
                "  ichor-al-daemon start --campaign-dir "
                + str(campaign),
                file=sys.stderr,
            )
            return 8
        if artefacts:
            print(
                "state.json is missing but this campaign is not empty.",
                file=sys.stderr,
            )
            print(
                "Refusing to initialise a fresh state because that could "
                "overwrite provenance.",
                file=sys.stderr,
            )
            print("Stateful artefacts:", file=sys.stderr)
            for artefact in artefacts[:12]:
                print("  - " + artefact, file=sys.stderr)
            if len(artefacts) > 12:
                print("  ... " + str(len(artefacts) - 12) + " more", file=sys.stderr)
            print("Run:", file=sys.stderr)
            print(
                "  " + _campaign_command(campaign, "reconcile"),
                file=sys.stderr,
            )
            return 8
    else:
        try:
            state_for_lock = read_state(state_path)
        except (StateSchemaError, json.JSONDecodeError) as exc:
            print(
                "state.json is invalid; run `ichor-al-daemon reconcile --campaign-dir "
                + str(campaign)
                + "` before starting: "
                + str(exc),
                file=sys.stderr,
            )
            return 5
        if state_for_lock.phase is CampaignPhase.HALTED:
            context = state_for_lock.lifecycle_context or {}
            print(
                "campaign is HALTED"
                + (
                    ": " + str(context.get("message"))
                    if context.get("message")
                    else ""
                ),
                file=sys.stderr,
            )
            print(
                "Run `ichor-al-daemon reconcile --campaign-dir "
                + str(campaign)
                + "` and apply only a validated recovery proposal.",
                file=sys.stderr,
            )
            return 20
        if state_for_lock.phase is CampaignPhase.DONE:
            print(
                "campaign is DONE; ordinary start cannot reopen a completed campaign. "
                "Use resume --reopen-converged only after increasing "
                "campaign.max_iterations.",
                file=sys.stderr,
            )
            return 6
        if state_for_lock.shutdown_requested:
            print(
                "campaign has a user stop request; use `ichor-al-daemon resume` "
                "rather than start so the stop is cleared explicitly.",
                file=sys.stderr,
            )
            return 6
    try:
        config = CampaignConfig.from_yaml(config_path)
    except Exception as exc:
        print(
            "campaign config could not be loaded: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=sys.stderr,
        )
        return 2

    try:
        lock_review = assert_config_unchanged_for_start(
            campaign,
            config,
            state_for_lock,
        )
    except Exception as exc:
        print("config lock check failed: " + str(exc), file=sys.stderr)
        return 7
    if lock_review.changed:
        print(
            "campaign.yaml changed since the config lock was written; "
            "run `ichor-al-daemon reconcile --campaign-dir "
            + str(campaign)
            + " --apply` if the edits are safe.",
            file=sys.stderr,
        )
        formatted = format_config_review(lock_review)
        if formatted:
            print(formatted, file=sys.stderr)
        return 7

    from .execution_identity import (
        ExecutionIdentityError,
        ensure_execution_identity,
        execution_identity_path,
    )

    placeholder_system = config.campaign.system_name in {
        "SYSTEM",
        "CHANGE_ME_SYSTEM",
    }
    requested_mode = getattr(args, "mode", None)
    identity_path = execution_identity_path(campaign)
    if requested_mode is None and not (
        identity_path.is_file() or identity_path.is_symlink()
    ):
        requested_mode = "live"
    if requested_mode == "live" and placeholder_system:
        print(
            "live mode requires campaign.system_name to be set to the real "
            "molecular system, not " + repr(config.campaign.system_name),
            file=sys.stderr,
        )
        return 2
    try:
        effective_mode, execution_identity = ensure_execution_identity(
            campaign,
            campaign_uid=str(state_for_lock.campaign_uid),
            config=config,
            requested_mode=requested_mode,
        )
    except ExecutionIdentityError as exc:
        print("execution identity refused start: " + str(exc), file=sys.stderr)
        return 13
    if effective_mode == "live" and placeholder_system:
        print(
            "live mode requires campaign.system_name to be set to the real "
            "molecular system, not " + repr(config.campaign.system_name),
            file=sys.stderr,
        )
        return 2

    if bool(getattr(args, "background", False)):
        return _launch_background_daemon(args, campaign)

    # Journal the validated effective configuration diff for user review.
    try:
        from .config import diff_against_defaults
        from .daemon.journal import append_event
        effective_diff = diff_against_defaults(config)
        #the diff can be very nested; produce a compact summary suitable
        #for the 4000-byte PIPE_BUF cap.
        def _flatten_keys(d, prefix=""):
            out = []
            for k, v in d.items():
                if k == "schema_version":
                    continue
                path = (prefix + "." + k) if prefix else k
                if isinstance(v, dict):
                    out.extend(_flatten_keys(v, path))
                else:
                    out.append(path)
            return out
        non_default_paths = sorted(_flatten_keys(effective_diff))
        journal_path = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        append_event(
            journal_path, "effective_config_diff",
            phase=(
                state_for_lock.phase.value
                if state_for_lock is not None
                else CampaignPhase.INIT.value
            ),
            iteration=(
                int(state_for_lock.iteration)
                if state_for_lock is not None
                else 0
            ),
            n_non_default=len(non_default_paths),
            # Truncate the paths list to stay well below PIPE_BUF.
            non_default_paths=non_default_paths[:50],
            paths_truncated=(len(non_default_paths) > 50),
        )
    except Exception:
        #Observability compliance; never block the daemon start on a
        #journal failure.
        pass

    # Only the live branch adopts scheduler jobs; dry-run has no scheduler.
    job_finder = None
    job_name_accounting_finder = None
    job_liveness_checker = None
    queue_diagnostics_collector = None
    resource_usage_collector = None
    if effective_mode == "live":
        _report_background_startup(
            "starting",
            "backend_preflight",
            campaign_dir=str(campaign),
            mode=str(effective_mode),
        )
        avail = check_backends()
        preflight = evaluate_campaign_preflight(
            campaign,
            config=config,
            avail=avail,
        )
        if not bool(preflight.get("ready", False)):
            print(_format_preflight(preflight, verbose=True), file=sys.stderr, end="")
            return 12
        try:
            executor = LiveBackendsPhaseExecutor(
                campaign_dir=campaign,
                config=config,
            )
        except LiveBackendNotAvailableError as exc:
            print(str(exc), file=sys.stderr)
            return 12
        scheduler_kind = str(
            getattr(executor, "scheduler_identity_kind", "slurm")
        )
        scheduler_backend = get_scheduler_backend(scheduler_kind)
        scheduler_timeout = int(
            config.runtime.scheduler_command_timeout_seconds
        )

        def _live_scheduler_poller(
            job_id,
            *,
            expected_job_name=None,
            expected_owner=None,
        ):
            return scheduler_backend.poll_job(
                job_id,
                timeout_seconds=scheduler_timeout,
                expected_job_name=expected_job_name,
                expected_owner=expected_owner,
            )

        sacct_poller = _live_scheduler_poller
        # live mode: let the daemon spot + adopt an orphaned in-flight job on (re)entry rather than
        # double-submitting after a crash or reconcile (A24/A25).
        job_finder = _call_timeout_aware(
            make_live_job_finder,
            campaign_dir=campaign,
            timeout_seconds=scheduler_timeout,
            scheduler_kind=scheduler_kind,
        )
        job_name_accounting_finder = _call_timeout_aware(
            make_live_job_accounting_finder,
            timeout_seconds=scheduler_timeout,
            scheduler_kind=scheduler_kind,
        )
        job_liveness_checker = _call_timeout_aware(
            make_live_job_liveness_checker,
            timeout_seconds=scheduler_timeout,
            scheduler_kind=scheduler_kind,
        )
        queue_diagnostics_collector = _call_timeout_aware(
            make_live_queue_diagnostics_collector,
            timeout_seconds=scheduler_timeout,
            scheduler_kind=scheduler_kind,
        )
        from .daemon.resource_usage import collect_usage

        resource_usage_collector = collect_usage
    elif effective_mode == "dry_run":
        executor = DryRunPhaseExecutor(
            campaign_dir=campaign,
            config=config,
        )
        sacct_poller = DryRunSacctPoller(elapsed_seconds=0)
    else:  # pragma: no cover - identity validation guarantees this
        raise AssertionError("validated execution mode was not handled")

    daemon_kwargs = {
        "campaign_dir": campaign,
        "config": config,
        "executor": executor,
        "scheduler_identity_kind": (
            str(getattr(executor, "scheduler_identity_kind", "slurm"))
            if effective_mode == "live"
            else "synthetic"
        ),
        "environment_preflight_ok": True,
        "poll_interval_override_seconds": (
            None
            if args.poll_interval is None
            else int(args.poll_interval)
        ),
    }
    if sacct_poller is not None:
        daemon_kwargs["sacct_poller"] = sacct_poller
    if job_finder is not None:
        daemon_kwargs["job_finder"] = job_finder
    if job_name_accounting_finder is not None:
        daemon_kwargs["job_name_accounting_finder"] = job_name_accounting_finder
    if job_liveness_checker is not None:
        daemon_kwargs["job_liveness_checker"] = job_liveness_checker
    if queue_diagnostics_collector is not None:
        daemon_kwargs["queue_diagnostics_collector"] = queue_diagnostics_collector
    if resource_usage_collector is not None:
        daemon_kwargs["resource_usage_collector"] = resource_usage_collector
    d = Daemon(**daemon_kwargs)
    startup_callback = None
    if startup_path_from_environment() is not None:

        def _record_background_startup(
            state_name: str,
            stage: str,
            failure: Optional[str] = None,
        ) -> None:
            _report_background_startup(
                state_name,
                stage,
                failure=failure,
                campaign_dir=str(campaign),
                mode=str(effective_mode),
                execution_identity_digest_sha256=str(
                    execution_identity.get("digest_sha256") or ""
                ),
            )

        startup_callback = _record_background_startup
    _report_background_startup(
        "starting",
        "daemon_initialisation",
        campaign_dir=str(campaign),
        mode=str(effective_mode),
    )
    return d.run(
        max_ticks=args.max_ticks,
        catch_keyboard_interrupt=True,
        startup_callback=startup_callback,
    )


_FEREBUS_JOB_NAME_EXTERNAL_PHASES = frozenset()


def _load_active_submission_intents(
    campaign: Path,
    *,
    errors: Optional[List[Dict[str, str]]] = None,
    fail_on_error: bool = False,
    expected_campaign_uid: Optional[str] = None,
    state: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    inventory = _submission_intent.inventory_intents(
        campaign,
        expected_campaign_uid=expected_campaign_uid,
    )
    invalid = [dict(item) for item in inventory.get("errors", [])]
    if errors is not None:
        errors.extend(invalid)
    if invalid and fail_on_error:
        raise ValueError(
            "submission-intent inventory is incomplete: "
            + "; ".join(
                str(item.get("path")) + ": " + str(item.get("error"))
                for item in invalid[:8]
            )
        )
    active = [
        dict(payload)
        for payload in inventory.get("records", [])
        if str(payload.get("status")) in _submission_intent.ACTIVE_STATUSES
    ]
    if state is None:
        return active
    classification = _submission_intent.classify_completed_unsubmitted_intents(
        campaign,
        state,
        intents=tuple(inventory.get("records", [])),
    )
    classification_errors = [
        dict(item) for item in classification.get("errors", [])
    ]
    if errors is not None:
        errors.extend(classification_errors)
    if classification_errors and fail_on_error:
        raise ValueError(
            "completion-receipt intent classification is inconclusive: "
            + "; ".join(
                str(item.get("path")) + ": " + str(item.get("error"))
                for item in classification_errors[:8]
            )
        )
    repair_keys = {
        (
            str(item.get("phase") or ""),
            int(item.get("iteration", 0)),
            str(item.get("submission_identity") or ""),
        )
        for item in classification.get("repairs", [])
    }
    return [
        item
        for item in active
        if (
            str(item.get("phase") or ""),
            int(item.get("iteration", 0)),
            str(item.get("submission_identity") or ""),
        )
        not in repair_keys
    ]


def _lookup_active_slurm_job_for_cancel(
    job_id: str,
    *,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    cmd = [
        "squeue",
        "--user",
        current_scheduler_user(),
        "-j",
        str(job_id),
        "--noheader",
        "--format=%i|%T|%j|%u",
    ]
    try:
        requested_job_id = validate_parent_job_id(job_id)
        completed = run_scheduler_command(
            subprocess.run,
            cmd,
            timeout_seconds=int(timeout_seconds),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return {
            "active": False,
            "inconclusive": True,
            "rows": [],
            "error": type(exc).__name__ + ": " + str(exc),
        }
    if int(getattr(completed, "returncode", 1)) != 0:
        stderr = getattr(completed, "stderr", "") or ""
        from .submit.sacct_poll import _squeue_invalid_job_id

        if _squeue_invalid_job_id(stderr):
            return {
                "active": False,
                "inconclusive": False,
                "rows": [],
                "error": None,
            }
        return {
            "active": False,
            "inconclusive": True,
            "rows": [],
            "error": "squeue exited with code "
            + str(int(getattr(completed, "returncode", 1)))
            + ": "
            + repr(stderr),
        }
    rows = []
    stdout = getattr(completed, "stdout", "") or ""
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            return {
                "active": False,
                "inconclusive": True,
                "rows": rows,
                "error": "malformed squeue cancellation row",
            }
        row_job_id = parts[0].strip()
        try:
            row_parent = parse_squeue_job_id(row_job_id)
        except ValueError as exc:
            return {"active": False, "inconclusive": True, "rows": rows, "error": str(exc)}
        if row_parent != requested_job_id:
            return {
                "active": False,
                "inconclusive": True,
                "rows": rows,
                "error": "squeue returned a foreign JobID during cancellation",
            }
        rows.append({
            "job_id": row_job_id,
            "state": parts[1].strip() if len(parts) > 1 else "",
            "job_name": parts[2].strip() if len(parts) > 2 else "",
            "owner": parts[3].strip() if len(parts) > 3 else "",
        })
    return {
        "active": bool(rows),
        "inconclusive": False,
        "rows": rows,
        "error": None,
    }


def _run_scancel(job_id: str, *, timeout_seconds: int = 60) -> Tuple[bool, str]:
    try:
        canonical_job_id = validate_parent_job_id(job_id)
        completed = run_scheduler_command(
            subprocess.run,
            ["scancel", canonical_job_id],
            timeout_seconds=int(timeout_seconds),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)
    if int(getattr(completed, "returncode", 1)) == 0:
        return True, ""
    stderr = getattr(completed, "stderr", "") or ""
    stdout = getattr(completed, "stdout", "") or ""
    message = stderr.strip() or stdout.strip() or "scancel failed"
    return False, message


def _confirm_cancelled_slurm_job(
    job_id: str,
    *,
    expected_tasks: Optional[int],
    submission_kind: str,
    confirmation_timeout_seconds: int,
    command_timeout_seconds: int,
    campaign_dir: Optional[Path] = None,
    intent: Optional[Mapping[str, Any]] = None,
    classification_sink: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    from .submit.sacct_poll import aggregate_states, poll_job

    deadline = time.monotonic() + float(confirmation_timeout_seconds)
    last_reason = "scheduler has not confirmed cancellation"
    expected_job_name = (
        str(intent.get("expected_job_name") or "")
        if isinstance(intent, Mapping)
        else ""
    )
    expected_owner = current_scheduler_user() if expected_job_name else None
    while True:
        observations: Sequence[Any] = ()
        summary = None
        try:
            observations = poll_job(
                job_id,
                timeout_seconds=int(command_timeout_seconds),
                expected_job_name=(expected_job_name or None),
                expected_owner=expected_owner,
            )
            summary = aggregate_states(
                job_id,
                observations,
                expected_task_count=expected_tasks,
                submission_kind=submission_kind,
            )
            if summary.n_unknown:
                last_reason = "accounting returned unknown cancellation state"
            elif summary.n_missing:
                last_reason = "accounting is missing expected cancellation rows"
        except Exception as exc:
            last_reason = type(exc).__name__ + ": " + str(exc)
        lookup = _call_timeout_aware(
            _lookup_active_slurm_job_for_cancel,
            job_id,
            timeout_seconds=int(command_timeout_seconds),
        )
        if lookup.get("inconclusive"):
            last_reason = "squeue cancellation lookup is inconclusive: " + str(
                lookup.get("error") or "unknown error"
            )
        elif lookup.get("active"):
            rows = list(lookup.get("rows") or [])
            if expected_job_name and not _job_name_matches_expected(
                rows,
                [expected_job_name],
            ):
                return False, "scheduler identity changed during cancellation"
            if expected_owner and any(
                str(row.get("owner") or "") != expected_owner for row in rows
            ):
                return False, "scheduler owner changed during cancellation"
            last_reason = "job remains active or completing in squeue"
        elif summary is not None and summary.is_terminal and not summary.n_missing:
            if campaign_dir is not None and intent is not None:
                try:
                    classification = classify_terminal_scheduler_evidence(
                        campaign_dir,
                        intent,
                        observations,
                        queue_active=False,
                    )
                except Exception as exc:
                    return False, (
                        "terminal scheduler evidence is invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    )
                if classification_sink is not None:
                    classification_sink.clear()
                    classification_sink.update(classification)
            return True, ""
        if time.monotonic() >= deadline:
            return False, (
                "cancellation confirmation timed out after "
                + str(int(confirmation_timeout_seconds))
                + " seconds: "
                + last_reason
            )
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _job_name_matches_expected(rows: Sequence[Dict[str, Any]], expected_names: Sequence[str]) -> bool:
    expected = {str(name) for name in expected_names if str(name)}
    if not expected or not rows:
        return False
    actual = {str(row.get("job_name") or "") for row in rows}
    return (
        bool(actual)
        and "" not in actual
        and len(actual) == 1
        and actual.issubset(expected)
    )


def _collect_stop_cancel_jobs(campaign: Path, state: Any) -> Dict[str, Dict[str, Any]]:
    jobs: Dict[str, Dict[str, Any]] = {}
    profile_scheduler_kind = str(
        profile_value("hpc", "scheduler", default="slurm") or "slurm"
    ).strip().lower()

    def ensure(job_id: str) -> Dict[str, Any]:
        return jobs.setdefault(
            str(job_id),
            {
                "job_id": str(job_id),
                "phases": set(),
                "expected_job_names": set(),
                "intent_keys": set(),
                "expected_tasks": None,
                "submission_kind": None,
                "scheduler_identity_kinds": set(),
                "scheduler_identity_from_intent": False,
                "intents": [],
            },
        )

    for phase, job_id in sorted((state.pending_jobs or {}).items()):
        if not job_id:
            continue
        item = ensure(str(job_id))
        phase_name = str(phase)
        item["phases"].add(phase_name)
        item["submission_kind"] = _submission_intent.submission_kind_for_phase(
            phase_name
        )
        item["scheduler_identity_kinds"].add(profile_scheduler_kind)
        if phase_name not in _FEREBUS_JOB_NAME_EXTERNAL_PHASES:
            item["expected_job_names"].add(
                _submission_intent.expected_job_name(
                    state.campaign_uid,
                    phase_name,
                    int(state.iteration),
                    scheduler_identity_kind=profile_scheduler_kind,
                )
            )

    campaign_uid = str(getattr(state, "campaign_uid", "") or "")
    for intent in _load_active_submission_intents(
        campaign,
        fail_on_error=True,
        expected_campaign_uid=(campaign_uid or None),
    ):
        job_id = str(intent.get("job_id") or "")
        if not job_id:
            continue
        phase_name, iteration = _intent_phase_iteration(intent)
        item = ensure(job_id)
        if phase_name:
            item["phases"].add(phase_name)
            item["intent_keys"].add((phase_name, int(iteration)))
        expected = str(intent.get("expected_job_name") or "")
        if expected and phase_name not in _FEREBUS_JOB_NAME_EXTERNAL_PHASES:
            item["expected_job_names"].add(expected)
        item["expected_tasks"] = intent.get("expected_tasks")
        item["submission_kind"] = str(intent.get("submission_kind"))
        item["intents"].append(dict(intent))
        recorded_scheduler = str(
            intent.get("scheduler_identity_kind") or "slurm"
        )
        if item["scheduler_identity_from_intent"] is False:
            item["scheduler_identity_kinds"].clear()
            item["scheduler_identity_from_intent"] = True
        item["scheduler_identity_kinds"].add(recorded_scheduler)

    return jobs


def _lookup_active_scheduler_job_for_cancel(
    job_id: str,
    *,
    scheduler_kind: str,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    if str(scheduler_kind).strip().lower() == "slurm":
        return _call_timeout_aware(
            _lookup_active_slurm_job_for_cancel,
            job_id,
            timeout_seconds=int(timeout_seconds),
        )
    backend = get_scheduler_backend(scheduler_kind)
    return backend.cancellation_lookup(
        job_id,
        timeout_seconds=int(timeout_seconds),
    )


def _run_scheduler_cancel(
    job_id: str,
    *,
    scheduler_kind: str,
    timeout_seconds: int = 60,
) -> Tuple[bool, str]:
    if str(scheduler_kind).strip().lower() == "slurm":
        return _call_timeout_aware(
            _run_scancel,
            job_id,
            timeout_seconds=int(timeout_seconds),
        )
    return get_scheduler_backend(scheduler_kind).cancel(
        job_id,
        timeout_seconds=int(timeout_seconds),
    )


def _sge_rows_were_never_started(rows: Sequence[Mapping[str, Any]]) -> bool:
    pending_states = {
        "qw",
        "hqw",
        "hRqw",
        "s",
        "S",
        "T",
        "ts",
        "tsS",
        "tT",
    }
    return bool(rows) and all(
        str(row.get("state") or "") in pending_states for row in rows
    )


def _record_inactive_scheduler_terminal_evidence(
    campaign: Path,
    *,
    job_id: str,
    item: Mapping[str, Any],
    scheduler_kind: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Authenticate a job that became terminal before qdel/scancel ran."""
    expected_tasks = item.get("expected_tasks")
    if (
        isinstance(expected_tasks, bool)
        or not isinstance(expected_tasks, int)
        or expected_tasks <= 0
    ):
        raise ValueError("submission intent has no exact expected task count")
    submission_kind = str(item.get("submission_kind") or "")
    if submission_kind not in {"scalar", "array"}:
        raise ValueError("submission kind is unavailable")
    intent_records = [
        dict(record)
        for record in item.get("intents", [])
        if str(record.get("job_id") or "") == str(job_id)
    ]
    if len(intent_records) != 1:
        raise ValueError(
            "terminal scheduler job is not owned by exactly one submission intent"
        )
    intent = intent_records[0]
    backend = get_scheduler_backend(scheduler_kind)
    expected_name = str(intent.get("expected_job_name") or "")
    accounted = backend.find_accounted_job_by_name(
        expected_name,
        expected_task_count=int(expected_tasks),
        submission_kind=submission_kind,
        cancellation_requested=True,
        timeout_seconds=int(timeout_seconds),
    )
    if accounted.inconclusive:
        raise ValueError(
            "terminal scheduler job-name lookup is inconclusive: "
            + str(accounted.error or "unknown error")
        )
    if (
        str(accounted.job_id or "") != str(job_id)
        or not bool(accounted.terminal)
    ):
        raise ValueError(
            "terminal scheduler accounting does not match the expected "
            "campaign job name"
        )
    observations = backend.poll_job(
        job_id,
        timeout_seconds=int(timeout_seconds),
        cancellation_requested=True,
        expected_job_name=expected_name,
        expected_owner=current_scheduler_user(),
    )
    classification = classify_terminal_scheduler_evidence(
        campaign,
        intent,
        observations,
        queue_active=False,
    )
    receipt = write_scheduler_terminal_receipt(
        campaign,
        intent,
        classification,
    )
    receipt_path = scheduler_terminal_receipt_path(
        campaign,
        phase=receipt["phase"],
        iteration=int(receipt["iteration"]),
        replacement_round=int(receipt["replacement_round"]),
        submission_identity=str(receipt["submission_identity"]),
    )
    return {
        "job_id": str(job_id),
        "scheduler_identity_kind": str(scheduler_kind),
        "phases": sorted(
            str(phase) for phase in item.get("phases", set())
        ),
        "intent_keys": [
            {"phase": str(phase_name), "iteration": int(iteration)}
            for phase_name, iteration in sorted(
                item.get("intent_keys", set())
            )
        ],
        "terminal_receipt": str(receipt_path),
        "terminal_receipt_sha256": str(receipt["receipt_sha256"]),
        "n_completed": int(receipt["n_completed"]),
        "n_retry": int(receipt["n_retry"]),
        "reason": "job was already terminal when cancellation was checked",
    }


def _confirm_cancelled_scheduler_job(
    job_id: str,
    *,
    scheduler_kind: str,
    expected_tasks: Optional[int],
    submission_kind: str,
    pre_cancel_rows: Sequence[Mapping[str, Any]],
    confirmation_timeout_seconds: int,
    command_timeout_seconds: int,
    campaign_dir: Optional[Path] = None,
    intent: Optional[Mapping[str, Any]] = None,
    classification_sink: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    from .submit.sacct_poll import aggregate_states

    if str(scheduler_kind).strip().lower() == "slurm":
        return _confirm_cancelled_slurm_job(
            job_id,
            expected_tasks=expected_tasks,
            submission_kind=submission_kind,
            confirmation_timeout_seconds=int(
                confirmation_timeout_seconds
            ),
            command_timeout_seconds=int(command_timeout_seconds),
            campaign_dir=campaign_dir,
            intent=intent,
            classification_sink=classification_sink,
        )
    backend = get_scheduler_backend(scheduler_kind)
    deadline = time.monotonic() + float(confirmation_timeout_seconds)
    last_reason = backend.display_name + " has not confirmed cancellation"
    while True:
        observations: Sequence[Any] = ()
        summary = None
        try:
            observations = backend.poll_job(
                job_id,
                timeout_seconds=int(command_timeout_seconds),
                cancellation_requested=True,
                expected_job_name=(
                    str(intent.get("expected_job_name") or "")
                    if isinstance(intent, Mapping)
                    else None
                ),
                expected_owner=(
                    current_scheduler_user()
                    if isinstance(intent, Mapping)
                    else None
                ),
            )
            summary = aggregate_states(
                job_id,
                observations,
                expected_task_count=expected_tasks,
                submission_kind=submission_kind,
                strict_parent_job_id=(scheduler_kind == "slurm"),
            )
            if summary.n_unknown:
                last_reason = "accounting returned an unknown cancellation state"
            elif summary.n_missing:
                last_reason = "accounting is missing expected cancellation rows"
        except Exception as exc:
            last_reason = type(exc).__name__ + ": " + str(exc)
        lookup = _lookup_active_scheduler_job_for_cancel(
            job_id,
            scheduler_kind=scheduler_kind,
            timeout_seconds=int(command_timeout_seconds),
        )
        if lookup.get("inconclusive"):
            last_reason = (
                backend.display_name
                + " cancellation lookup is inconclusive: "
                + str(lookup.get("error") or "unknown error")
            )
        elif lookup.get("active"):
            last_reason = "job remains active or is still leaving the scheduler"
        elif (
            summary is not None
            and summary.is_terminal
            and int(summary.n_missing) == 0
        ) or (
            scheduler_kind == "sge"
            and not observations
            and _sge_rows_were_never_started(pre_cancel_rows)
        ):
            if campaign_dir is not None and intent is not None:
                try:
                    classification = classify_terminal_scheduler_evidence(
                        campaign_dir,
                        intent,
                        observations,
                        queue_active=False,
                        pre_cancel_rows=pre_cancel_rows,
                    )
                except Exception as exc:
                    return False, (
                        "terminal scheduler evidence is invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    )
                if classification_sink is not None:
                    classification_sink.clear()
                    classification_sink.update(classification)
            return True, ""
        if time.monotonic() >= deadline:
            return False, (
                "cancellation confirmation timed out after "
                + str(int(confirmation_timeout_seconds))
                + " seconds: "
                + last_reason
            )
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _cancel_recorded_scheduler_jobs(
    campaign: Path,
    state: Any,
    *,
    command_timeout_seconds: int = 60,
    confirmation_timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Cancel campaign-owned work through its recorded scheduler contract."""
    cancelled: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    campaign_uid = str(getattr(state, "campaign_uid", "") or "")
    pre_submit_intents = [
        intent
        for intent in _load_active_submission_intents(
            campaign,
            fail_on_error=True,
            expected_campaign_uid=(campaign_uid or None),
        )
        if not str(intent.get("job_id") or "")
        and str(intent.get("status") or "") == "PRE_SUBMIT"
    ]
    for original_intent in pre_submit_intents:
        phase_name, iteration = _intent_phase_iteration(original_intent)
        scheduler_kind = str(
            original_intent.get("scheduler_identity_kind") or "slurm"
        ).strip().lower()
        try:
            backend = get_scheduler_backend(scheduler_kind)
        except ValueError as exc:
            failed.append(
                {
                    "job_id": "",
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": str(exc),
                }
            )
            continue
        expected_name = str(original_intent.get("expected_job_name") or "")
        expected_tasks = original_intent.get("expected_tasks")
        if not expected_name:
            failed.append(
                {
                    "job_id": "",
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": "PRE_SUBMIT cancellation lacks an exact job name",
                }
            )
            continue
        if expected_tasks is not None and (
            isinstance(expected_tasks, bool)
            or not isinstance(expected_tasks, int)
            or expected_tasks <= 0
        ):
            failed.append(
                {
                    "job_id": "",
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": "PRE_SUBMIT task count is invalid",
                }
            )
            continue
        submission_kind = str(
            original_intent.get("submission_kind") or ""
        )
        if submission_kind not in {"scalar", "array"}:
            failed.append(
                {
                    "job_id": "",
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": "PRE_SUBMIT submission kind is unavailable",
                }
            )
            continue
        deadline = time.monotonic() + float(confirmation_timeout_seconds)
        adopted_job_id = ""
        while True:
            try:
                current_intent = _submission_intent.load_intent(
                    campaign,
                    phase_name,
                    int(iteration),
                    expected_campaign_uid=(campaign_uid or None),
                )
            except Exception as exc:
                failed.append(
                    {
                        "job_id": "",
                        "scheduler_identity_kind": scheduler_kind,
                        "reason": (
                            "PRE_SUBMIT intent became unreadable during "
                            "cancellation: "
                            + type(exc).__name__
                            + ": "
                            + str(exc)
                        ),
                    }
                )
                break
            if not isinstance(current_intent, Mapping):
                failed.append(
                    {
                        "job_id": "",
                        "scheduler_identity_kind": scheduler_kind,
                        "reason": "PRE_SUBMIT intent disappeared during cancellation",
                    }
                )
                break
            current_expected_tasks = current_intent.get("expected_tasks")
            if current_expected_tasks is not None:
                if (
                    isinstance(current_expected_tasks, bool)
                    or not isinstance(current_expected_tasks, int)
                    or current_expected_tasks <= 0
                ):
                    failed.append(
                        {
                            "job_id": "",
                            "scheduler_identity_kind": scheduler_kind,
                            "reason": (
                                "PRE_SUBMIT intent acquired an invalid task "
                                "count during cancellation"
                            ),
                        }
                    )
                    break
                expected_tasks = int(current_expected_tasks)
            recorded_job_id = str(current_intent.get("job_id") or "")
            if recorded_job_id:
                adopted_job_id = recorded_job_id
                break
            if str(current_intent.get("status") or "") in {
                "FAILED",
                "SUPERSEDED",
            }:
                if str(current_intent.get("reason") or "") not in {
                    "user_cancelled_before_scheduler_acceptance",
                    "user_cancelled_via_stop",
                }:
                    failed.append(
                        {
                            "job_id": "",
                            "scheduler_identity_kind": scheduler_kind,
                            "reason": (
                                "PRE_SUBMIT intent became terminal for a "
                                "reason unrelated to the stop request"
                            ),
                        }
                    )
                    break
                lookup = backend.find_accounted_job_by_name(
                    expected_name,
                    expected_task_count=(
                        None
                        if expected_tasks is None
                        else int(expected_tasks)
                    ),
                    submission_kind=submission_kind,
                    timeout_seconds=int(command_timeout_seconds),
                )
                if lookup.inconclusive:
                    failed.append(
                        {
                            "job_id": "",
                            "scheduler_identity_kind": scheduler_kind,
                            "reason": (
                                "PRE_SUBMIT no-acceptance proof is "
                                "inconclusive: "
                                + str(lookup.error or "unknown error")
                            ),
                        }
                    )
                    break
                if lookup.job_id:
                    failed.append(
                        {
                            "job_id": str(lookup.job_id),
                            "scheduler_identity_kind": scheduler_kind,
                            "reason": (
                                "scheduler accepted a job after the intent "
                                "was marked as not submitted"
                            ),
                        }
                    )
                    break
                if expected_tasks is None:
                    cancelled.append(
                        {
                            "job_id": "",
                            "scheduler_identity_kind": scheduler_kind,
                            "phases": [phase_name],
                            "intent_keys": [
                                {
                                    "phase": phase_name,
                                    "iteration": int(iteration),
                                }
                            ],
                            "n_completed": 0,
                            "task_count_unknown": True,
                            "reason": (
                                "scheduler submission stopped before task "
                                "staging completed"
                            ),
                        }
                    )
                    break
                try:
                    classification = classify_unaccepted_scheduler_intent(
                        campaign,
                        current_intent,
                    )
                    receipt = write_scheduler_terminal_receipt(
                        campaign,
                        current_intent,
                        classification,
                    )
                    receipt_path = scheduler_terminal_receipt_path(
                        campaign,
                        phase=receipt["phase"],
                        iteration=int(receipt["iteration"]),
                        replacement_round=int(
                            receipt["replacement_round"]
                        ),
                        submission_identity=str(
                            receipt["submission_identity"]
                        ),
                    )
                except Exception as exc:
                    failed.append(
                        {
                            "job_id": "",
                            "scheduler_identity_kind": scheduler_kind,
                            "reason": (
                                "PRE_SUBMIT cancellation evidence could not "
                                "be recorded: "
                                + type(exc).__name__
                                + ": "
                                + str(exc)
                            ),
                        }
                    )
                    break
                cancelled.append(
                    {
                        "job_id": "",
                        "scheduler_identity_kind": scheduler_kind,
                        "phases": [phase_name],
                        "intent_keys": [
                            {
                                "phase": phase_name,
                                "iteration": int(iteration),
                            }
                        ],
                        "n_completed": 0,
                        "n_retry": int(expected_tasks),
                        "terminal_receipt": str(receipt_path),
                        "terminal_receipt_sha256": str(
                            receipt["receipt_sha256"]
                        ),
                        "reason": "scheduler submission stopped before acceptance",
                    }
                )
                break
            lookup = backend.find_accounted_job_by_name(
                expected_name,
                expected_task_count=(
                    None
                    if expected_tasks is None
                    else int(expected_tasks)
                ),
                submission_kind=submission_kind,
                timeout_seconds=int(command_timeout_seconds),
            )
            if lookup.inconclusive:
                failed.append(
                    {
                        "job_id": "",
                        "scheduler_identity_kind": scheduler_kind,
                        "reason": "PRE_SUBMIT job-name lookup is inconclusive: "
                        + str(lookup.error or "unknown error"),
                    }
                )
                break
            if lookup.job_id:
                if expected_tasks is None:
                    failed.append(
                        {
                            "job_id": str(lookup.job_id),
                            "scheduler_identity_kind": scheduler_kind,
                            "reason": (
                                "scheduler accepted a PRE_SUBMIT job before "
                                "its immutable task count was recorded"
                            ),
                        }
                    )
                    break
                adopted_job_id = str(lookup.job_id)
                try:
                    _submission_intent.mark_submitted(
                        campaign,
                        phase_name,
                        int(iteration),
                        adopted_job_id,
                        expected_tasks=int(expected_tasks),
                    )
                except Exception as exc:
                    failed.append(
                        {
                            "job_id": adopted_job_id,
                            "scheduler_identity_kind": scheduler_kind,
                            "reason": (
                                "accepted scheduler job could not be bound to "
                                "its PRE_SUBMIT intent: "
                                + type(exc).__name__
                                + ": "
                                + str(exc)
                            ),
                        }
                    )
                    adopted_job_id = ""
                break
            if time.monotonic() >= deadline:
                failed.append(
                    {
                        "job_id": "",
                        "scheduler_identity_kind": scheduler_kind,
                        "reason": (
                            "PRE_SUBMIT cancellation timed out before the "
                            "daemon acknowledged the stop gate or a scheduler "
                            "job became visible"
                        ),
                    }
                )
                break
            time.sleep(
                min(0.25, max(0.0, deadline - time.monotonic()))
            )

    jobs = _collect_stop_cancel_jobs(campaign, state)
    for intent in _load_active_submission_intents(
        campaign,
        fail_on_error=True,
        expected_campaign_uid=(campaign_uid or None),
    ):
        if str(intent.get("job_id") or ""):
            continue
        phase_name, iteration = _intent_phase_iteration(intent)
        failed.append(
            {
                "job_id": "",
                "scheduler_identity_kind": str(
                    intent.get("scheduler_identity_kind") or "slurm"
                ),
                "reason": "active submission intent still has no job_id: "
                + str(phase_name)
                + "@"
                + str(iteration),
            }
        )

    expected_owner = current_scheduler_user()
    for job_id, item in sorted(jobs.items()):
        scheduler_kinds = {
            str(kind)
            for kind in item.get("scheduler_identity_kinds", set())
            if str(kind)
        }
        if len(scheduler_kinds) != 1:
            failed.append(
                {
                    "job_id": job_id,
                    "reason": "recorded scheduler identity is missing or contradictory",
                }
            )
            continue
        scheduler_kind = next(iter(scheduler_kinds))
        try:
            backend = get_scheduler_backend(scheduler_kind)
        except ValueError as exc:
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": str(exc),
                }
            )
            continue
        lookup = _lookup_active_scheduler_job_for_cancel(
            job_id,
            scheduler_kind=scheduler_kind,
            timeout_seconds=int(command_timeout_seconds),
        )
        if bool(lookup.get("inconclusive")):
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": (
                        "squeue lookup inconclusive: "
                        if scheduler_kind == "slurm"
                        else backend.display_name + " lookup inconclusive: "
                    )
                    + str(lookup.get("error") or "unknown error"),
                }
            )
            continue
        if not bool(lookup.get("active")):
            matching_intents = [
                record
                for record in item.get("intents", [])
                if str(record.get("job_id") or "") == str(job_id)
            ]
            if not matching_intents:
                # Legacy state-only ownership has no immutable task map from
                # which an exact terminal receipt can be built.  Preserve the
                # established inactive-job behaviour without claiming partial
                # recovery evidence.
                skipped.append(
                    {
                        "job_id": job_id,
                        "scheduler_identity_kind": scheduler_kind,
                        "reason": "not active in " + backend.queue_command,
                    }
                )
                continue
            try:
                terminal_record = _record_inactive_scheduler_terminal_evidence(
                    campaign,
                    job_id=job_id,
                    item=item,
                    scheduler_kind=scheduler_kind,
                    timeout_seconds=int(command_timeout_seconds),
                )
            except Exception as exc:
                failed.append(
                    {
                        "job_id": job_id,
                        "scheduler_identity_kind": scheduler_kind,
                        "reason": (
                            "job left "
                            + backend.display_name
                            + " but exact terminal accounting is unavailable: "
                            + type(exc).__name__
                            + ": "
                            + str(exc)
                        ),
                    }
                )
            else:
                cancelled.append(terminal_record)
            continue
        rows = list(lookup.get("rows") or [])
        foreign_owners = sorted(
            {
                str(row.get("owner") or "")
                for row in rows
                if str(row.get("owner") or "")
                and str(row.get("owner") or "") != expected_owner
            }
        )
        missing_owner = any(
            not str(row.get("owner") or "") for row in rows
        )
        if missing_owner:
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": "scheduler ownership is unavailable",
                }
            )
            continue
        if foreign_owners:
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": "scheduler ownership mismatch",
                    "expected_owner": expected_owner,
                    "actual_owners": foreign_owners,
                }
            )
            continue
        expected_names = sorted(
            str(name)
            for name in item.get("expected_job_names", set())
            if str(name)
        )
        if not _job_name_matches_expected(rows, expected_names):
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": "scheduler job name mismatch",
                    "expected_job_names": expected_names,
                    "actual_job_names": sorted(
                        {str(row.get("job_name") or "") for row in rows}
                    ),
                }
            )
            continue
        expected_tasks = item.get("expected_tasks")
        if expected_tasks is not None and (
            isinstance(expected_tasks, bool)
            or not isinstance(expected_tasks, int)
            or expected_tasks <= 0
        ):
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": "submission intent has an invalid expected task count",
                }
            )
            continue
        submission_kind = str(item.get("submission_kind") or "")
        if submission_kind not in {"scalar", "array"}:
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": "submission kind is unavailable for cancellation confirmation",
                }
            )
            continue
        if submission_kind == "array" and expected_tasks is None:
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": (
                        "array task cardinality is unavailable; cancellation was not "
                        "issued because complete terminal confirmation would be impossible"
                    ),
                }
            )
            continue
        intent_records = [
            dict(record)
            for record in item.get("intents", [])
            if str(record.get("job_id") or "") == str(job_id)
        ]
        if len(intent_records) != 1:
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": (
                        "scheduler cancellation requires exactly one "
                        "submission intent for the job"
                    ),
                }
            )
            continue
        producer_intent = intent_records[0]
        ok, message = _run_scheduler_cancel(
            job_id,
            scheduler_kind=scheduler_kind,
            timeout_seconds=int(command_timeout_seconds),
        )
        if not ok:
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": backend.cancel_command + " failed: " + message,
                }
            )
            continue
        classification: Dict[str, Any] = {}
        confirmation_result = _confirm_cancelled_scheduler_job(
            job_id,
            scheduler_kind=scheduler_kind,
            expected_tasks=expected_tasks,
            submission_kind=submission_kind,
            pre_cancel_rows=rows,
            confirmation_timeout_seconds=int(confirmation_timeout_seconds),
            command_timeout_seconds=int(command_timeout_seconds),
            campaign_dir=(campaign if producer_intent is not None else None),
            intent=producer_intent,
            classification_sink=classification,
        )
        confirmed, confirmation_reason = confirmation_result
        if not confirmed:
            failed.append(
                {
                    "job_id": job_id,
                    "scheduler_identity_kind": scheduler_kind,
                    "reason": confirmation_reason,
                }
            )
            continue
        terminal_evidence: Dict[str, Any] = {}
        if producer_intent is not None and classification:
            try:
                receipt = write_scheduler_terminal_receipt(
                    campaign,
                    producer_intent,
                    classification,
                )
                receipt_path = scheduler_terminal_receipt_path(
                    campaign,
                    phase=receipt["phase"],
                    iteration=int(receipt["iteration"]),
                    replacement_round=int(receipt["replacement_round"]),
                    submission_identity=str(receipt["submission_identity"]),
                )
            except Exception as exc:
                failed.append(
                    {
                        "job_id": job_id,
                        "scheduler_identity_kind": scheduler_kind,
                        "reason": (
                            "scheduler cancellation completed but terminal evidence "
                            "could not be recorded: "
                            + type(exc).__name__
                            + ": "
                            + str(exc)
                        ),
                    }
                )
                continue
            terminal_evidence = {
                "terminal_receipt": str(receipt_path),
                "terminal_receipt_sha256": str(receipt["receipt_sha256"]),
                "n_completed": int(receipt["n_completed"]),
                "n_retry": int(receipt["n_retry"]),
            }
        cancelled.append(
            {
                "job_id": job_id,
                "scheduler_identity_kind": scheduler_kind,
                "phases": sorted(
                    str(phase) for phase in item.get("phases", set())
                ),
                "intent_keys": [
                    {"phase": str(phase_name), "iteration": int(iteration)}
                    for phase_name, iteration in sorted(
                        item.get("intent_keys", set())
                    )
                ],
                **terminal_evidence,
            }
        )
    return {"cancelled": cancelled, "skipped": skipped, "failed": failed}


def _mark_cancelled_intents_without_state(
    campaign: Path,
    summary: Mapping[str, Any],
) -> None:
    for item in (summary.get("cancelled") or []):
        if not isinstance(item, Mapping):
            continue
        for key in (item.get("intent_keys") or []):
            if not isinstance(key, Mapping):
                continue
            try:
                _submission_intent.mark_failed(
                    campaign,
                    str(key["phase"]),
                    int(key["iteration"]),
                    "user_cancelled_via_stop",
                )
            except Exception:
                pass


def _journal_cancel_jobs_summary(
    journal_path: Path,
    summary: Dict[str, Any],
    *,
    state: Any = None,
) -> None:
    try:
        from .daemon.journal import append_event

        context: Dict[str, Any] = {}
        if state is not None and getattr(state, "phase", None) is not None:
            phase = getattr(state, "phase")
            context["phase"] = phase.value if hasattr(phase, "value") else str(phase)
            context["iteration"] = int(getattr(state, "iteration", 0))
        recorded_kinds = {
            str(item.get("scheduler_identity_kind") or "").strip().lower()
            for bucket in ("cancelled", "skipped", "failed")
            for item in (summary.get(bucket) or [])
            if isinstance(item, Mapping)
            and item.get("scheduler_identity_kind")
        }
        if len(recorded_kinds) == 1:
            context["scheduler_identity_kind"] = next(iter(recorded_kinds))
        classified = [
            item
            for item in (summary.get("cancelled") or [])
            if isinstance(item, Mapping)
            and isinstance(item.get("n_completed"), int)
            and isinstance(item.get("n_retry"), int)
        ]
        append_event(
            journal_path,
            "user_cancelled_jobs",
            **context,
            n_cancelled=len(summary.get("cancelled") or []),
            n_skipped=len(summary.get("skipped") or []),
            n_failed=len(summary.get("failed") or []),
            cancelled_job_ids=[
                str(item.get("job_id"))
                for item in (summary.get("cancelled") or [])
            ],
            skipped_job_ids=[
                str(item.get("job_id"))
                for item in (summary.get("skipped") or [])
            ],
            failed_job_ids=[
                str(item.get("job_id"))
                for item in (summary.get("failed") or [])
            ],
            n_scheduler_completed_tasks=sum(
                int(item["n_completed"]) for item in classified
            ),
            n_retry_tasks=sum(int(item["n_retry"]) for item in classified),
        )
    except Exception:
        pass


def _print_cancel_jobs_summary(
    summary: Dict[str, Any],
    *,
    verbose: bool = False,
) -> None:
    cancelled = list(summary.get("cancelled") or [])
    skipped = list(summary.get("skipped") or [])
    failed = list(summary.get("failed") or [])
    recorded_kinds = {
        str(item.get("scheduler_identity_kind") or "").strip().lower()
        for item in cancelled + skipped + failed
        if isinstance(item, Mapping)
        and item.get("scheduler_identity_kind")
    }
    scheduler_name = _scheduler_display_name(
        next(iter(recorded_kinds)) if len(recorded_kinds) == 1 else None
    )
    if not verbose:
        print(
            scheduler_name
            + " cancellation: "
            + str(len(cancelled))
            + " cancelled, "
            + str(len(skipped))
            + " already inactive or not cancellable, "
            + str(len(failed))
            + " unresolved."
        )
        if failed:
            print(
                "Cancellation needs attention: "
                + str(failed[0].get("reason") or "scheduler confirmation failed"),
                file=sys.stderr,
            )
        completed_tasks = sum(
            int(item.get("n_completed", 0))
            for item in cancelled
            if isinstance(item, Mapping)
        )
        retry_tasks = sum(
            int(item.get("n_retry", 0))
            for item in cancelled
            if isinstance(item, Mapping)
        )
        if completed_tasks or retry_tasks:
            print(
                "Recorded task outcomes: "
                + str(completed_tasks)
                + " scheduler-completed, "
                + str(retry_tasks)
                + " to validate or retry."
            )
        return
    if cancelled:
        print("Cancelled " + scheduler_name + " jobs:")
        for item in cancelled:
            phases = ", ".join(str(phase) for phase in item.get("phases", []))
            suffix = " (" + phases + ")" if phases else ""
            print("  - " + str(item.get("job_id")) + suffix)
    if skipped:
        print("Skipped " + scheduler_name + " jobs:")
        for item in skipped:
            print("  - " + str(item.get("job_id")) + ": " + str(item.get("reason")))
    if failed:
        print("Jobs not cancelled:", file=sys.stderr)
        for item in failed:
            print("  - " + str(item.get("job_id")) + ": " + str(item.get("reason")), file=sys.stderr)
    if not cancelled and not skipped and not failed:
        print("No recorded active " + scheduler_name + " jobs found.")


def _signal_recorded_background_daemon(
    campaign: Path,
    paths: Mapping[str, Path],
) -> Dict[str, Any]:
    """Send SIGTERM only to a launch record bound to this campaign and PID."""
    background = _probe_background_daemon(
        paths["background_pid"],
        paths["background_log"],
        paths["background_startup"],
    )
    result: Dict[str, Any] = {
        "attempted": False,
        "signalled": False,
        "reason": "no live background child",
    }
    if background.get("background_pid_alive") is not True:
        return result
    startup = background.get("background_startup_payload")
    if not isinstance(startup, dict):
        result["reason"] = "live PID has no authenticated startup record"
        return result
    startup_state = str(startup.get("state") or "")
    if (
        startup_state not in BACKGROUND_STARTUP_ACTIVE_STATES
        and startup_state != "ready"
    ):
        result["reason"] = (
            "startup record is not active: " + repr(startup_state or "missing")
        )
        return result
    pid = background.get("background_pid")
    try:
        startup_pid = int(startup.get("pid"))
        pid_int = int(pid)
    except (TypeError, ValueError):
        result["reason"] = "startup record has an invalid PID"
        return result
    if startup_pid != pid_int:
        result["reason"] = "startup-record PID does not match the live PID"
        return result
    recorded_campaign = str(startup.get("campaign_dir") or "")
    if not recorded_campaign:
        result["reason"] = "startup record has no campaign identity"
        return result
    try:
        campaign_matches = Path(recorded_campaign).resolve() == campaign.resolve()
    except OSError:
        campaign_matches = False
    if not campaign_matches:
        result["reason"] = "startup record belongs to a different campaign"
        return result
    launch_id = str(startup.get("launch_id") or "")
    if not launch_id:
        result["reason"] = "startup record has no launch identity"
        return result

    result["attempted"] = True
    result["pid"] = pid_int
    try:
        os.kill(pid_int, signal.SIGTERM)
    except OSError as exc:
        result["reason"] = type(exc).__name__ + ": " + str(exc)
        return result
    result["signalled"] = True
    result["reason"] = "SIGTERM sent after durable immediate-stop request"
    try:
        update_background_startup(
            paths["background_startup"],
            launch_id,
            stop_signal_requested_at_unix=float(time.time()),
            stop_signal="SIGTERM",
        )
    except Exception:
        pass
    return result


def cmd_stop(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(
        args.campaign_dir,
        require_campaign_yaml=False,
    )
    paths = _campaign_paths(campaign)
    command_timeout, confirmation_timeout = _runtime_scheduler_policy(campaign)
    if not paths["state"].exists():
        if bool(getattr(args, "cancel_jobs", False)):
            print(
                "state.json is missing, so pending_jobs could not be read; "
                "falling back to active submission intents.",
                file=sys.stderr,
            )
            fallback_state = SimpleNamespace(
                pending_jobs={},
                campaign_uid="",
                iteration=0,
            )
            try:
                cancel_summary = _cancel_recorded_scheduler_jobs(
                    campaign,
                    fallback_state,
                    command_timeout_seconds=command_timeout,
                    confirmation_timeout_seconds=confirmation_timeout,
                )
            except Exception as exc:
                print(
                    "could not establish complete scheduler ownership: " + str(exc),
                    file=sys.stderr,
                )
                return 10
            _mark_cancelled_intents_without_state(campaign, cancel_summary)
            _journal_cancel_jobs_summary(paths["journal"], cancel_summary)
            _print_cancel_jobs_summary(cancel_summary)
            return 10 if cancel_summary.get("failed") else 0
        print("no state.json at " + str(paths["state"]) + "; daemon not running?", file=sys.stderr)
        return 4
    try:
        state = read_state(paths["state"])
    except (StateSchemaError, ValueError) as exc:
        if bool(getattr(args, "cancel_jobs", False)):
            print("state.json invalid: " + str(exc), file=sys.stderr)
            print(
                "pending_jobs could not be read; falling back to active "
                "submission intents.",
                file=sys.stderr,
            )
            fallback_state = SimpleNamespace(
                pending_jobs={},
                campaign_uid="",
                iteration=0,
            )
            try:
                cancel_summary = _cancel_recorded_scheduler_jobs(
                    campaign,
                    fallback_state,
                    command_timeout_seconds=command_timeout,
                    confirmation_timeout_seconds=confirmation_timeout,
                )
            except Exception as exc:
                print(
                    "could not establish complete scheduler ownership: " + str(exc),
                    file=sys.stderr,
                )
                return 10
            _mark_cancelled_intents_without_state(campaign, cancel_summary)
            _journal_cancel_jobs_summary(paths["journal"], cancel_summary)
            _print_cancel_jobs_summary(cancel_summary)
            return 10 if cancel_summary.get("failed") else 0
        print("state.json invalid: " + str(exc), file=sys.stderr)
        return 5
    if state.is_terminal:
        print(
            "campaign is already "
            + state.phase.value
            + "; no daemon stop request was recorded",
            file=sys.stderr,
        )
        return 6
    from .daemon.stop_control import (
        StopControlError,
        build_stop_request,
        describe_stop_request,
        install_stop_request,
        update_stop_request,
    )

    after_iteration = getattr(args, "after_iteration", None)
    mode = (
        "after_iteration"
        if after_iteration is not None
        else str(getattr(args, "stop_mode", "immediate") or "immediate")
    )
    cancel_jobs = bool(getattr(args, "cancel_jobs", False))
    if cancel_jobs and mode != "immediate":
        print("--cancel-jobs is valid only with --immediate", file=sys.stderr)
        return 2
    phase_started = bool(state.pending_jobs.get(state.phase.value))
    try:
        intent = _submission_intent.load_intent(
            campaign,
            state.phase.value,
            int(state.iteration),
            expected_campaign_uid=str(state.campaign_uid),
        )
    except Exception:
        intent = None
    if isinstance(intent, dict):
        try:
            phase_started = phase_started or int(
                intent.get("replacement_round", 0)
            ) == int(state.replacement_round)
        except (TypeError, ValueError):
            phase_started = True
    target_iteration = None
    if mode == "after_iteration" and after_iteration not in (None, -1):
        target_iteration = int(after_iteration)
    try:
        requested = build_stop_request(
            state,
            mode=mode,
            target_iteration=target_iteration,
            phase_started=phase_started,
            cancel_jobs=cancel_jobs,
        )
        request, disposition = install_stop_request(campaign, requested)
    except StopControlError as exc:
        print("stop request rejected: " + str(exc), file=sys.stderr)
        return 2
    try:
        from .daemon.journal import append_event

        append_event(
            paths["journal"],
            "user_stop_requested",
            request_id=str(request.get("request_id")),
            mode=str(request.get("mode")),
            status=str(request.get("status")),
            phase=state.phase.value,
            iteration=int(state.iteration),
            target_phase=request.get("target_phase"),
            target_iteration=request.get("target_iteration"),
            target_replacement_round=request.get("target_replacement_round"),
            phase_started=bool(request.get("phase_started_at_request")),
            disposition=str(disposition),
        )
    except Exception:
        pass
    background_signal = None
    cancel_summary = None
    if mode == "immediate" and cancel_jobs:
        # The durable cancelling request is already installed.  Signalling now
        # lets a daemon in PRE_SUBMIT reach its final pre-acceptance gate while
        # this command proves whether a scheduler job exists.
        background_signal = _signal_recorded_background_daemon(
            campaign,
            paths,
        )
    if cancel_jobs:
        try:
            cancel_summary = _cancel_recorded_scheduler_jobs(
                campaign,
                state,
                command_timeout_seconds=command_timeout,
                confirmation_timeout_seconds=confirmation_timeout,
            )
        except Exception as exc:
            cancel_summary = {
                "cancelled": [],
                "skipped": [],
                "failed": [{"job_id": "unresolved", "reason": str(exc)}],
            }
        next_status = "cancelling" if cancel_summary.get("failed") else "requested"
        updated = update_stop_request(
            campaign,
            str(request.get("request_id")),
            status=next_status,
            cancellation_summary=cancel_summary,
        )
        if updated is not None:
            request = updated
    if cancel_summary is not None:
        _journal_cancel_jobs_summary(
            paths["journal"], cancel_summary, state=state
        )
    if mode == "immediate" and background_signal is None:
        background_signal = _signal_recorded_background_daemon(
            campaign,
            paths,
        )
    print(describe_stop_request(request))
    stop_verbose = bool(getattr(args, "verbose", False))
    if stop_verbose:
        print("request id: " + str(request.get("request_id")))
        print("stop request: " + str(paths["stop_request"]))
    background = _probe_background_daemon(
        paths["background_pid"],
        paths["background_log"],
        paths["background_startup"],
    )
    if stop_verbose and background.get("background_pid") is not None:
        suffix = " alive" if background.get("background_pid_alive") else " not running"
        print("background pid: " + str(background.get("background_pid")) + " (" + suffix.strip() + ")")
        print("background log: " + str(background.get("background_log_path")))
    if isinstance(background_signal, dict) and background_signal.get("attempted"):
        if stop_verbose:
            print("background signal: " + str(background_signal.get("reason")))
    elif isinstance(background_signal, dict) and background.get("background_pid_alive") is True:
        print(
            "background signal not sent: " + str(background_signal.get("reason")),
            file=sys.stderr,
        )
    cancellation_failed = False
    if cancel_summary is not None:
        _print_cancel_jobs_summary(cancel_summary, verbose=stop_verbose)
        cancellation_failed = bool(cancel_summary.get("failed"))
    elif cancel_jobs:
        print(
            "No recorded "
            + _scheduler_display_name()
            + " jobs required cancellation."
        )
    else:
        print(
            "Recorded "
            + _scheduler_display_name()
            + " jobs, if any, were left running."
        )
    print("Monitor the stop:")
    print("  " + _campaign_command(campaign, "status"))
    print("  " + _campaign_command(campaign, "journal", " --last-n 20"))
    if cancellation_failed:
        return 10
    if (
        isinstance(background_signal, dict)
        and background_signal.get("attempted")
        and not background_signal.get("signalled")
    ):
        return 10
    return 0


def format_recovery_dashboard(campaign_dir: Path) -> str:
    """Return a read-only recovery dashboard for a campaign directory."""
    from .daemon.stop_control import describe_stop_request

    campaign = Path(campaign_dir).expanduser().resolve()
    paths = _campaign_paths(campaign)
    lines: List[str] = [
        "Recovery dashboard",
        "Campaign: " + str(campaign),
        "",
    ]

    state = None
    state_status = "missing"
    state_invalid = False
    if paths["state"].is_file():
        try:
            state = read_state(paths["state"])
            state_status = (
                "present phase="
                + state.phase.value
                + " iteration="
                + str(state.iteration)
            )
        except Exception as exc:
            state_invalid = True
            state_status = (
                "invalid "
                + type(exc).__name__
                + ": "
                + str(exc)[:120]
            )
    lines.extend(_section("State", [("state.json", state_status)]))
    stop_status = _stop_control_status(
        campaign,
        expected_campaign_uid=(
            str(state.campaign_uid) if state is not None else None
        ),
    )
    stop_request = stop_status.get("stop_request")
    lines.extend(
        _section(
            "User stop control",
            [
                (
                    "request",
                    (
                        describe_stop_request(stop_request)
                        if isinstance(stop_request, dict)
                        else "none"
                    ),
                ),
                ("error", stop_status.get("stop_control_error")),
            ],
        )
    )
    lines.extend(
        _section(
            "Last exception",
            [("LAST_EXCEPTION.json", _read_last_exception_summary(campaign))],
        )
    )

    lock_status = _probe_daemon_lock(paths["lock"])
    stale_seconds, clock_skew = _runtime_liveness_policy(campaign)
    lease_status = _probe_daemon_lease(
        paths["lease"],
        stale_seconds=stale_seconds,
        clock_skew_tolerance_seconds=clock_skew,
    )
    background = _probe_background_daemon(
        paths["background_pid"],
        paths["background_log"],
        paths["background_startup"],
    )
    lines.extend(
        _section(
            "Runtime",
            [
                ("daemon lock", _lock_summary(lock_status.get("lock_held"))),
                ("lease", _heartbeat_summary(lease_status.get("lease_heartbeat"))),
                (
                    "background pid",
                    str(background.get("background_pid"))
                    + " alive="
                    + str(background.get("background_pid_alive")),
                ),
            ],
        )
    )

    intent_errors: List[Dict[str, str]] = []
    intents = _load_active_submission_intents(
        campaign,
        errors=intent_errors,
        state=state,
    )
    intent_summary = str(len(intents))
    if intents:
        sample = [
            str(item.get("phase"))
            + "@"
            + str(item.get("iteration"))
            + " "
            + str(item.get("status"))
            + " job_id="
            + str(item.get("job_id"))
            for item in intents[:5]
        ]
        intent_summary += " (" + "; ".join(sample) + ")"
    lines.extend(_section("Submission intents", [("active", intent_summary)]))

    try:
        from .acquisition.trajectory_pool import (
            POOL_MANIFEST_FILENAME,
            POOL_SUBDIR,
            TrajectoryPoolManifest,
        )
        from .strict_json import strict_json as _json

        manifest_path = campaign / POOL_SUBDIR / POOL_MANIFEST_FILENAME
        manifest = TrajectoryPoolManifest.from_dict(
            _json.loads(
                manifest_path.read_text(encoding="utf-8"),
                source=manifest_path,
            )
        )
        pool_status = "authority sha=" + str(manifest.sha256)[:12]
    except Exception as exc:
        pool_status = type(exc).__name__ + ": " + str(exc)[:120]
    lines.extend(_section("Trajectory pool", [("status", pool_status)]))

    staging_inventory = data_staging_inventory(campaign)
    staging_archive_hint = "not needed"
    if int(staging_inventory.get("top_level_count") or 0) > 0:
        staging_archive_hint = (
            "preview reconcile first; archive only after the preview reports it is safe"
        )
    if staging_inventory.get("has_symlink") or staging_inventory.get("is_symlink"):
        staging_archive_hint = "unsafe: symlink present; inspect manually"
    lines.extend(
        _section(
            "Staging",
            [
                ("inventory", _format_staging_inventory_summary(staging_inventory)),
                ("archive", staging_archive_hint),
            ],
        )
    )
    if intent_errors:
        lines.extend(
            _section(
                "Invalid submission intents",
                [
                    (
                        str(item.get("path")),
                        str(item.get("error")),
                    )
                    for item in intent_errors[:8]
                ],
            )
        )

    cfg_path = campaign / "campaign.yaml"
    if not cfg_path.is_file():
        if config_lock_path(campaign).is_file():
            config_status = "campaign.yaml missing; config lock present"
        else:
            config_status = "campaign.yaml missing; config lock missing"
    else:
        try:
            cfg = CampaignConfig.from_yaml(cfg_path)
            if state is None:
                config_status = "loaded; state unavailable for lock review"
            else:
                review = review_config_changes(
                    campaign,
                    cfg,
                    state,
                    initialise_missing=False,
                )
                if review.blocked_changes:
                    config_status = (
                        "blocked changes: "
                        + str(len(review.blocked_changes))
                    )
                elif review.allowed_changes:
                    config_status = (
                        "allowed changes: "
                        + str(len(review.allowed_changes))
                    )
                elif review.changed:
                    config_status = "changed"
                else:
                    config_status = "clean"
        except Exception as exc:
            config_status = type(exc).__name__ + ": " + str(exc)[:120]
    lines.extend(_section("Config lock", [("status", config_status)]))

    status_command = _campaign_command(campaign, "status")
    reconcile_command = _campaign_command(campaign, "reconcile")
    recommendation = "no recovery action is needed; run " + status_command
    try:
        report = propose_recovery(campaign)
        partial = getattr(report, "partial_array_recovery", None)
        if isinstance(partial, dict):
            lines.extend(
                _section(
                    "Partial Array Recovery",
                    [
                        (
                            "phase",
                            str(partial.get("phase"))
                            + "@"
                            + str(partial.get("iteration")),
                        ),
                        (
                            "tasks",
                            "logical="
                            + str(partial.get("logical_total"))
                            + " reusable="
                            + str(partial.get("n_reuse"))
                            + " retry="
                            + str(partial.get("n_retry")),
                        ),
                        ("ledger", partial.get("ledger")),
                    ],
                )
            )
        if report.unsafe_reasons:
            recommendation = "inspect the recovery blockers: " + reconcile_command
        elif state_invalid:
            recommendation = "preview recovery: " + reconcile_command
        elif state is None:
            artefacts = stateful_campaign_artifacts(campaign)
            recommendation = (
                "preview recovery: " + reconcile_command
                if artefacts
                else (
                    "initialise the campaign: ichor-al-daemon init --campaign-dir "
                    + shlex.quote(str(campaign))
                )
            )
        elif state.phase is CampaignPhase.HALTED:
            recommendation = "preview recovery: " + reconcile_command
    except Exception as exc:
        recommendation = (
            "preview recovery because the recovery check failed: "
            + reconcile_command
            + " ("
            + str(exc)[:120]
            + ")"
        )
    ownership_active = bool(
        lock_status.get("lock_held")
        or _lease_is_fresh(
            lease_status.get("lease_heartbeat"),
            stale_seconds=stale_seconds,
            clock_skew_tolerance_seconds=clock_skew,
        )
        or background.get("background_pid_alive")
    )
    has_slurm_job = any(
        isinstance(intent, dict) and bool(intent.get("job_id"))
        for intent in intents
    )
    if ownership_active or has_slurm_job:
        recommendation = (
            "cancel the recorded "
            + _scheduler_display_name(intents=intents)
            + " work first: ichor-al-daemon stop --campaign-dir "
            + shlex.quote(str(campaign))
            + " --cancel-jobs"
            if has_slurm_job
            else "monitor the active daemon: " + status_command
        )
    if not cfg_path.is_file() and config_lock_path(campaign).is_file():
        recommendation = "restore campaign.yaml from config lock"
    lines.extend(_section("Recommendation", [("next action", recommendation)]))
    return "\n".join(lines) + "\n"


def _load_seed_selection_progress_status(
    campaign: Path,
    state: CampaignState,
    runtime_payload: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    from datetime import datetime, timezone

    path = (
        Path(campaign)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "runtime_progress"
        / "SEED_SELECT.json"
    )
    if not path.exists() and not path.is_symlink():
        return None
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("progress path is not a regular file")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ValueError("unsupported seed-selection progress schema")
        if str(record.get("campaign_uid") or "") != str(state.campaign_uid):
            raise ValueError("progress campaign identity mismatch")
        if str(record.get("phase") or "") != CampaignPhase.SEED_SELECT.value:
            raise ValueError("progress phase mismatch")
        if int(record.get("iteration")) != int(state.iteration):
            return {"state": "stale", "reason": "iteration_mismatch"}
        updated = datetime.fromisoformat(str(record.get("updated_iso") or ""))
        if updated.tzinfo is None:
            raise ValueError("progress update time has no timezone")
        age = max(
            0.0,
            (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds(),
        )
        daemon_active = _daemon_activity_status(dict(runtime_payload)).startswith(
            ("running", "starting")
        )
        recorded_pid = int(record.get("pid"))
        active_pid = None
        if runtime_payload.get("background_pid_alive") is True:
            active_pid = runtime_payload.get("background_pid")
        heartbeat = runtime_payload.get("lease_heartbeat")
        if active_pid is None and isinstance(heartbeat, dict):
            try:
                if _lease_is_fresh(
                    heartbeat,
                    stale_seconds=int(
                        runtime_payload.get("lease_stale_seconds") or 900
                    ),
                    clock_skew_tolerance_seconds=int(
                        runtime_payload.get("clock_skew_tolerance_seconds") or 60
                    ),
                ):
                    active_pid = heartbeat.get("pid")
            except Exception:
                active_pid = None
        if (
            active_pid is None
            and runtime_payload.get("lock_held") is True
            and recorded_pid == int(os.getpid())
        ):
            active_pid = recorded_pid
        pid_matches = active_pid is not None and int(active_pid) == recorded_pid
        startup = runtime_payload.get("background_startup_payload")
        launch_id = (
            str(startup.get("launch_id") or "").strip()
            if isinstance(startup, dict)
            else ""
        )
        start_matches = not launch_id or str(
            record.get("daemon_start_identity") or ""
        ) == launch_id
        current = (
            state.phase is CampaignPhase.SEED_SELECT
            and daemon_active
            and pid_matches
            and start_matches
            and str(record.get("status") or "") in {"running", "complete"}
        )
        return {
            "state": "current" if current else "stale",
            "age_seconds": float(age),
            "record": record if current else None,
            "reason": None if current else "inactive_or_old",
        }
    except Exception as exc:
        return {
            "state": "malformed",
            "reason": type(exc).__name__ + ": " + str(exc),
        }


def _load_runtime_progress_status(
    campaign: Path,
    state: CampaignState,
    runtime_payload: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    from datetime import datetime, timezone
    from .daemon.phase_progress import (
        newest_matching_progress,
        read_phase_progress_records,
    )

    progress_errors: List[str] = []
    records = read_phase_progress_records(
        campaign,
        phase=state.phase.value,
        expected_campaign_uid=str(state.campaign_uid),
        errors=progress_errors,
    )
    if not records:
        if progress_errors:
            return {
                "state": "malformed",
                "record": None,
                "reason": progress_errors[0],
                "ignored_errors": progress_errors,
            }
        return None
    active_pid: Optional[int] = None
    if runtime_payload.get("background_pid_alive") is True:
        try:
            active_pid = int(runtime_payload.get("background_pid"))
        except (TypeError, ValueError):
            active_pid = None
    heartbeat = runtime_payload.get("lease_heartbeat")
    if active_pid is None and isinstance(heartbeat, Mapping):
        try:
            if _lease_is_fresh(
                heartbeat,
                stale_seconds=int(runtime_payload.get("lease_stale_seconds") or 900),
                clock_skew_tolerance_seconds=int(
                    runtime_payload.get("clock_skew_tolerance_seconds") or 60
                ),
            ):
                active_pid = int(heartbeat.get("pid"))
        except (TypeError, ValueError):
            active_pid = None
    active_start_id: Optional[str] = None
    startup_payload = runtime_payload.get("background_startup_payload")
    if isinstance(startup_payload, Mapping):
        launch_id = str(startup_payload.get("launch_id") or "").strip()
        if launch_id:
            active_start_id = launch_id
    job_ids = {
        str(job_id)
        for job_id in dict(state.pending_jobs).values()
        if job_id
    }
    intents = runtime_payload.get("active_submission_intents")
    if isinstance(intents, list):
        job_ids.update(
            str(intent.get("job_id"))
            for intent in intents
            if isinstance(intent, Mapping) and intent.get("job_id")
        )
    if job_ids:
        job_bound_records = [
            record
            for record in records
            if str(record.get("job_id") or "") in job_ids
            and str(record.get("producer_kind") or "")
            in {"local", "scheduler", "worker"}
        ]
        if job_bound_records:
            priority_groups = (
                [
                    record
                    for record in job_bound_records
                    if str(record.get("producer_kind") or "") == "local"
                    and str(record.get("status") or "") == "running"
                ],
                [
                    record
                    for record in job_bound_records
                    if str(record.get("producer_kind") or "") == "worker"
                    and str(record.get("status") or "") == "running"
                ],
                [
                    record
                    for record in job_bound_records
                    if str(record.get("producer_kind") or "") == "scheduler"
                ],
                job_bound_records,
            )
            records = next(group for group in priority_groups if group)
        else:
            return {
                "state": "stale",
                "record": None,
                "reason": "awaiting_job_progress",
                "ignored_errors": progress_errors,
            }
    record = newest_matching_progress(
        records,
        campaign_uid=str(state.campaign_uid),
        phase=state.phase.value,
        iteration=int(state.iteration),
        replacement_round=int(state.replacement_round),
        daemon_pid=active_pid,
        daemon_start_id=active_start_id,
        job_ids=job_ids,
    )
    if record is None:
        return {
            "state": "stale",
            "record": None,
            "reason": "identity_or_state_mismatch",
        }
    producer_kind = str(record.get("producer_kind") or "")
    ownership_active = _status_daemon_active(dict(runtime_payload))
    if producer_kind == "checkpoint":
        current = _pid_is_alive(record.get("daemon_pid"))
    elif producer_kind == "local" and not ownership_active:
        current = False
    elif producer_kind in {"scheduler", "worker"} and record.get("job_id"):
        current = str(record.get("job_id")) in job_ids
    else:
        current = ownership_active or bool(job_ids)
    try:
        updated = datetime.fromisoformat(
            str(record.get("updated_at_iso") or "").replace("Z", "+00:00")
        )
        if updated.tzinfo is None or updated.utcoffset() is None:
            raise ValueError("progress update time has no timezone")
        age = max(
            0.0,
            (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds(),
        )
    except (TypeError, ValueError):
        return {
            "state": "malformed",
            "record": None,
            "reason": "updated_at_invalid",
        }
    return {
        "state": "current" if current else "stale",
        "record": record if current else None,
        "age_seconds": float(age),
        "reason": None if current else "producer_not_active",
        "ignored_errors": progress_errors,
    }


def _reference_commit_runtime_progress(
    payload: Mapping[str, Any],
    state: CampaignState,
) -> Optional[Dict[str, Any]]:
    from datetime import datetime, timezone

    if state.phase is not CampaignPhase.REFERENCE_COMMIT:
        return None
    transactions = payload.get("reference_commit_transactions")
    if not isinstance(transactions, list):
        return None
    matching: List[Mapping[str, Any]] = []
    for record in transactions:
        if not isinstance(record, Mapping):
            continue
        ledger = record.get("ledger")
        if isinstance(ledger, Mapping) and ledger.get("iteration") == int(
            state.iteration
        ):
            matching.append(record)
    if not matching:
        return None
    transaction = matching[-1]
    ledger = dict(transaction.get("ledger") or {})
    transaction_state = str(transaction.get("state") or "invalid")
    stage_by_state = {
        "prepared": "reference_commit_move",
        "partially_moved": "reference_commit_move",
        "cache_incomplete": "reference_commit_cache",
        "publication_incomplete": "reference_commit_publication",
        "published": "reference_commit_pointer",
        "pointer_incomplete": "reference_commit_pointer",
        "complete": "reference_commit_complete",
        "invalid": "reference_commit_validation",
    }
    total = int(len(ledger.get("point_bindings") or []))
    moved = int(ledger.get("moved_points") or 0)
    now = datetime.now(timezone.utc)
    updated_text = str(ledger.get("updated_at_iso") or "")
    created_text = str(ledger.get("created_at_iso") or updated_text)
    try:
        updated = datetime.fromisoformat(updated_text.replace("Z", "+00:00"))
        created = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
        if (
            updated.tzinfo is None
            or updated.utcoffset() is None
            or created.tzinfo is None
            or created.utcoffset() is None
        ):
            raise ValueError("reference transaction timestamp has no timezone")
        age = max(0.0, (now - updated.astimezone(timezone.utc)).total_seconds())
        elapsed = max(
            0.0,
            (now - created.astimezone(timezone.utc)).total_seconds(),
        )
    except (TypeError, ValueError):
        age = 0.0
        elapsed = 0.0
    record: Dict[str, Any] = {
        "schema_version": 1,
        "campaign_uid": str(state.campaign_uid),
        "phase": state.phase.value,
        "iteration": int(state.iteration),
        "replacement_round": int(state.replacement_round),
        "producer_kind": "local",
        "stage": stage_by_state.get(
            transaction_state, "reference_commit_validation"
        ),
        "status": (
            "failed"
            if transaction_state == "invalid"
            else "completed"
            if transaction_state == "complete"
            else "running"
        ),
        "counters": {
            "completed": moved,
            "total": total,
            "unit": "point directories",
        },
        "elapsed_seconds": float(elapsed),
        "throughput": None,
        "started_at_iso": created_text,
        "updated_at_iso": updated_text,
        "details": {
            "moved_bytes": int(ledger.get("moved_bytes") or 0),
            "shards_reused": int(ledger.get("shards_reused") or 0),
            "shards_repaired": int(ledger.get("shards_repaired") or 0),
        },
    }
    daemon_active = _status_daemon_active(dict(payload))
    return {
        "state": "current" if daemon_active else "stale",
        "record": record if daemon_active else None,
        "age_seconds": float(age),
        "reason": None if daemon_active else "producer_not_active",
    }


def _load_scheduler_recovery_status(
    campaign: Path,
    state: CampaignState,
) -> Optional[Dict[str, Any]]:
    """Read bounded cancellation recovery evidence for human status output."""
    from .daemon.phase_executor import SBATCH_PHASES
    from .daemon.scheduler_recovery import (
        PHASE_RECOVERY_LEDGER_SCHEMA_VERSION,
        SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION,
        phase_recovery_ledger_path,
        read_phase_recovery_ledger,
        scheduler_terminal_recoveries,
    )

    if state.phase.value not in SBATCH_PHASES:
        return None
    try:
        recoveries = scheduler_terminal_recoveries(
            campaign,
            campaign_uid=str(state.campaign_uid),
            phase=state.phase.value,
            iteration=int(state.iteration),
            replacement_round=int(state.replacement_round),
        )
        if not recoveries:
            return None
        latest_by_task: Dict[int, Mapping[str, Any]] = {}
        latest_receipt_schema_by_task: Dict[int, int] = {}
        for recovery in recoveries:
            for outcome in recovery["receipt"].get("outcomes", []):
                logical_task_id = int(outcome["logical_task_id"])
                latest_by_task[logical_task_id] = outcome
                latest_receipt_schema_by_task[logical_task_id] = int(
                    recovery["receipt"].get("schema_version") or 0
                )
        scheduler_completed = sum(
            1
            for outcome in latest_by_task.values()
            if str(outcome.get("status") or "") == "COMPLETED"
            and outcome.get("exit_code") == [0, 0]
        )
        scheduler_retry = len(latest_by_task) - scheduler_completed
        legacy_retry_tasks = sum(
            1
            for task_id, outcome in latest_by_task.items()
            if latest_receipt_schema_by_task.get(task_id)
            != SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION
            or not (
                str(outcome.get("status") or "") == "COMPLETED"
                and outcome.get("exit_code") == [0, 0]
            )
        )
        receipt_digests = {
            str(recovery["receipt"]["receipt_sha256"])
            for recovery in recoveries
        }
        status: Dict[str, Any] = {
            "state": "awaiting_validation",
            "phase": state.phase.value,
            "iteration": int(state.iteration),
            "replacement_round": int(state.replacement_round),
            "n_terminal_receipts": len(recoveries),
            "n_scheduler_completed": int(scheduler_completed),
            "n_scheduler_retry": int(scheduler_retry),
            "original_job_ids": [
                str(recovery["receipt"]["job_id"])
                for recovery in recoveries
                if recovery["receipt"].get("job_id")
            ],
        }
        ledger_path = phase_recovery_ledger_path(
            campaign,
            phase=state.phase.value,
            iteration=int(state.iteration),
            replacement_round=int(state.replacement_round),
        )
        if not ledger_path.exists() and not ledger_path.is_symlink():
            if any(
                schema != SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION
                for schema in latest_receipt_schema_by_task.values()
            ):
                status.update(
                    {
                        "state": "legacy_unverified",
                        "n_reusable": 0,
                        "n_retry": int(legacy_retry_tasks),
                        "reason": "legacy_scheduler_identity_unproven",
                    }
                )
            return status
        ledger = read_phase_recovery_ledger(ledger_path)
        identity = (
            str(ledger.get("campaign_uid") or ""),
            str(ledger.get("phase") or ""),
            int(ledger.get("iteration", -1)),
            int(ledger.get("replacement_round", -1)),
        )
        expected = (
            str(state.campaign_uid),
            state.phase.value,
            int(state.iteration),
            int(state.replacement_round),
        )
        if identity != expected:
            raise ValueError("phase recovery ledger identity mismatch")
        ledger_receipts = {
            str(record.get("receipt_sha256") or "")
            for record in ledger.get("source_terminal_receipts", [])
            if isinstance(record, Mapping)
        }
        if ledger_receipts != receipt_digests:
            return status
        if (
            int(ledger.get("schema_version") or 0)
            != PHASE_RECOVERY_LEDGER_SCHEMA_VERSION
        ):
            status.update(
                {
                    "state": "legacy_unverified",
                    "n_reusable": 0,
                    "n_retry": len(latest_by_task),
                    "reason": "legacy_recovery_authority_unproven",
                }
            )
            return status
        status.update(
            {
                "state": "validated",
                "n_reusable": int(ledger.get("n_reusable") or 0),
                "n_retry": int(ledger.get("n_retry") or 0),
                "publication_disposition": ledger.get(
                    "publication_disposition"
                ),
                "ledger": str(ledger_path),
                "ledger_sha256": str(ledger.get("ledger_sha256") or ""),
            }
        )
        return status
    except Exception as exc:
        return {
            "state": "invalid",
            "phase": state.phase.value,
            "iteration": int(state.iteration),
            "replacement_round": int(state.replacement_round),
            "error": type(exc).__name__ + ": " + str(exc),
        }


def _ferebus_staging_presentation_evidence(
    campaign: Path,
    state: CampaignState,
    *,
    artifact_snapshot: Optional[Any],
    config: Optional[CampaignConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Return control-only FEREBUS staging recovery evidence for operators."""
    if state.phase not in {
        CampaignPhase.INITIAL_FEREBUS,
        CampaignPhase.FEREBUS,
    }:
        return None
    if (
        state.phase == CampaignPhase.FEREBUS
        and is_redundant_committed_parent_staging(
            campaign,
            campaign_uid=str(state.campaign_uid),
            target_reference_data_version=int(state.reference_data_version),
            parent_model_version=int(state.models_version),
            replacement_round=int(state.replacement_round),
            artifact_snapshot=artifact_snapshot,
        )
    ):
        return None
    try:
        reference_view = (
            None
            if artifact_snapshot is None
            else artifact_snapshot.reference_view(
                int(state.reference_data_version)
            )
        )
        context = classify_ferebus_staging_recovery(
            campaign,
            campaign_uid=str(state.campaign_uid),
            phase=state.phase.value,
            iteration=int(state.iteration),
            replacement_round=int(state.replacement_round),
            reference_data_version=int(state.reference_data_version),
            reference_head_manifest_sha256=(
                None
                if reference_view is None
                else str(reference_view.head_manifest_sha256)
            ),
            reference_view_sha256=(
                None
                if reference_view is None
                else str(reference_view.cumulative_view_sha256)
            ),
            config=config,
        )
        return context.summary()
    except Exception as exc:
        return {
            "disposition": "contradictory",
            "reason": type(exc).__name__ + ": " + str(exc),
        }


def _ferebus_recovery_ledger_has_missing_map_failure(
    campaign: Path,
    state: CampaignState,
) -> bool:
    """Identify ledgers produced by the historical per-task map-missing bug."""
    if state.phase not in {
        CampaignPhase.INITIAL_FEREBUS,
        CampaignPhase.FEREBUS,
    }:
        return False
    try:
        from .daemon.scheduler_recovery import (
            phase_recovery_ledger_path,
            read_phase_recovery_ledger,
        )

        path = phase_recovery_ledger_path(
            campaign,
            phase=state.phase.value,
            iteration=int(state.iteration),
            replacement_round=int(state.replacement_round),
        )
        if not path.is_file() or path.is_symlink():
            return False
        ledger = read_phase_recovery_ledger(path)
        invalid = ledger.get("invalid_completed_tasks")
        if not isinstance(invalid, list) or not invalid:
            return False
        reasons = [
            str(item.get("reason") or "").upper()
            for item in invalid
            if isinstance(item, Mapping)
        ]
        return bool(reasons) and all(
            "TASK MAP" in reason
            or "TASK_MAP" in reason
            or "FEREBUS_TASK_MAP.JSON" in reason
            for reason in reasons
        )
    except Exception:
        return False


def _ariadne_terminal_postprocess_presentation_evidence(
    campaign: Path,
    state: CampaignState,
) -> Optional[Dict[str, Any]]:
    """Resolve authenticated scheduler-free ARIADNE recovery evidence."""
    if state.phase is not CampaignPhase.ARIADNE_ARRAY:
        return None
    from .daemon.submission_intent import (
        ARIADNE_TERMINAL_POSTPROCESS_REASON,
        AriadneTerminalPostprocessNotApplicable,
        load_intent,
        resolve_ariadne_terminal_postprocess_source,
    )

    current = load_intent(
        campaign,
        state.phase.value,
        int(state.iteration),
        expected_campaign_uid=str(state.campaign_uid),
    )
    if not isinstance(current, Mapping):
        return None
    if not (
        str(current.get("reason") or "")
        == ARIADNE_TERMINAL_POSTPROCESS_REASON
        or isinstance(current.get("postprocess_source"), Mapping)
    ):
        return None
    try:
        return resolve_ariadne_terminal_postprocess_source(
            campaign,
            campaign_uid=str(state.campaign_uid),
            iteration=int(state.iteration),
            intent=current,
        )
    except AriadneTerminalPostprocessNotApplicable:
        return None


def cmd_status(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(
        args.campaign_dir,
        require_campaign_yaml=False,
    )
    paths = _campaign_paths(campaign)
    try:
        from .daemon.ariadne_quarantine import inventory_quarantine_authority

        quarantine_status = inventory_quarantine_authority(campaign)
    except Exception as exc:
        quarantine_status = {
            "attempts": [],
            "errors": [{"path": "", "error": type(exc).__name__ + ": " + str(exc)}],
            "total_bytes": 0,
        }
    if not paths["state"].exists():
        payload: Dict[str, Any] = {
            "status_error": "state_missing",
            "state_path": str(paths["state"]),
            "campaign_dir": str(campaign),
            "stop_request_path": str(paths["stop_request"]),
            "runtime_progress": {
                "state": "unavailable",
                "record": None,
                "reason": "state_missing",
            },
        }
        payload.update(_stop_control_status(campaign))
        payload["ariadne_retry_quarantine"] = quarantine_status
        payload.update(_missing_state_context(campaign))
        try:
            cfg = CampaignConfig.from_yaml(campaign / "campaign.yaml")
            payload["campaign_config_status"] = {"ok": True}
            payload["pool_feasibility"] = _pool_feasibility_summary(campaign, cfg)
        except Exception as exc:
            payload["campaign_config_status"] = {
                "ok": False,
                "error": type(exc).__name__ + ": " + str(exc),
            }
        payload["recommendations"] = recommendation_dicts(
            build_status_recommendations(campaign, payload, paths["journal"])
        )
        payload["next_action"] = payload["recommendations"][0]["primary"]
        if bool(getattr(args, "json", False)):
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        else:
            print(
                _format_status_unavailable(
                    payload,
                    verbose=bool(getattr(args, "verbose", False)),
                ),
                end="",
            )
            print("no state.json at " + str(paths["state"]), file=sys.stderr)
        return 4
    try:
        state = read_state(paths["state"])
    except (StateSchemaError, json.JSONDecodeError, UnicodeError, OSError) as exc:
        payload = {
            "status_error": (
                "state_schema_invalid"
                if isinstance(exc, (StateSchemaError, json.JSONDecodeError))
                else "state_unreadable"
            ),
            "state_error": type(exc).__name__ + ": " + str(exc),
            "state_path": str(paths["state"]),
            "campaign_dir": str(campaign),
            "stop_request_path": str(paths["stop_request"]),
            "runtime_progress": {
                "state": "unavailable",
                "record": None,
                "reason": "state_unreadable",
            },
        }
        payload.update(_stop_control_status(campaign))
        payload["ariadne_retry_quarantine"] = quarantine_status
        payload["recommendations"] = recommendation_dicts(
            build_status_recommendations(campaign, payload, paths["journal"])
        )
        payload["next_action"] = payload["recommendations"][0]["primary"]
        if bool(getattr(args, "json", False)):
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        else:
            print(
                _format_status_unavailable(
                    payload,
                    verbose=bool(getattr(args, "verbose", False)),
                ),
                end="",
            )
            print("state.json invalid: " + str(exc), file=sys.stderr)
        return 5
    payload = state.to_dict()
    payload["ariadne_retry_quarantine"] = quarantine_status
    payload["state_path"] = str(paths["state"])
    payload["lock_path"] = str(paths["lock"])
    payload["stop_request_path"] = str(paths["stop_request"])
    payload.update(
        _stop_control_status(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
            state=state,
        )
    )
    payload.update(_probe_daemon_lock(paths["lock"]))
    stale_seconds, clock_skew = _runtime_liveness_policy(campaign)
    payload.update(
        _probe_daemon_lease(
            paths["lease"],
            stale_seconds=stale_seconds,
            clock_skew_tolerance_seconds=clock_skew,
        )
    )
    payload.update(
        _probe_background_daemon(
            paths["background_pid"],
            paths["background_log"],
            paths["background_startup"],
        )
    )
    progress_status = _load_seed_selection_progress_status(
        campaign,
        state,
        payload,
    )
    if progress_status is not None:
        payload["seed_selection_progress"] = progress_status
    status_journal_events: List[Dict[str, Any]] = []
    try:
        from .daemon.journal import tail_events

        status_journal_events = tail_events(paths["journal"], max_records=4096)
    except Exception as exc:
        payload["journal_error"] = type(exc).__name__ + ": " + str(exc)
    try:
        from .execution_identity import read_active_environment_generation

        active_environment = read_active_environment_generation(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )["generation"]
        transition = None
        for event in reversed(status_journal_events):
            if str(event.get("event") or "") in {
                "environment_generation_advanced",
                "environment_rebound",
            }:
                transition = str(event.get("ts") or "")
                break
        payload["active_environment_generation"] = {
            "generation": int(active_environment["generation"]),
            "digest_sha256": str(active_environment["digest_sha256"]),
            "created_at_iso": str(active_environment["created_at_iso"]),
            "last_transition": transition,
        }
    except Exception as exc:
        payload["active_environment_generation"] = {
            "error": type(exc).__name__ + ": " + str(exc)
        }
    intent_errors: List[Dict[str, str]] = []
    payload["active_submission_intents"] = _load_active_submission_intents(
        campaign,
        errors=intent_errors,
        expected_campaign_uid=str(state.campaign_uid),
        state=state,
    )
    if intent_errors:
        payload["submission_intent_errors"] = intent_errors
    runtime_progress_status = _load_runtime_progress_status(
        campaign,
        state,
        payload,
    )
    if runtime_progress_status is not None:
        payload["runtime_progress"] = runtime_progress_status
    elif progress_status is not None:
        # Preserve the established SEED_SELECT sidecar while exposing it
        # through the additive generic status contract.
        payload["runtime_progress"] = dict(progress_status)
    try:
        from .point_allocation import point_allocation_path, read_point_allocation

        allocation_context = "bootstrap" if int(state.iteration) == 0 else "active"
        allocation_path = point_allocation_path(
            campaign,
            context=allocation_context,
            iteration=0 if allocation_context == "bootstrap" else int(state.iteration),
        )
        if allocation_path.is_file():
            allocation = read_point_allocation(
                allocation_path,
                expected_campaign_uid=str(state.campaign_uid),
                expected_context=allocation_context,
                expected_iteration=(
                    0
                    if allocation_context == "bootstrap"
                    else int(state.iteration)
                ),
            )
            payload["point_allocation_summary"] = dict(allocation.get("summary") or {})
    except Exception as exc:
        payload["point_allocation_summary"] = {
            "error": type(exc).__name__ + ": " + str(exc)
        }
    try:
        from .daemon.reference_commit import inventory_reference_commits

        payload["reference_commit_transactions"] = inventory_reference_commits(
            campaign
        )
    except Exception as exc:
        payload["reference_commit_transactions"] = [
            {
                "state": "invalid",
                "error": type(exc).__name__ + ": " + str(exc),
            }
        ]
    existing_runtime_progress = payload.get("runtime_progress")
    if not (
        isinstance(existing_runtime_progress, Mapping)
        and existing_runtime_progress.get("state") == "current"
    ):
        reference_progress = _reference_commit_runtime_progress(payload, state)
        if reference_progress is not None:
            payload["runtime_progress"] = reference_progress
    if "runtime_progress" not in payload:
        payload["runtime_progress"] = {
            "state": "absent",
            "record": None,
            "reason": "no_progress_record",
        }
    try:
        from .daemon.ferebus_candidate_recovery import read_recovery_request

        payload["ferebus_candidate_recovery"] = read_recovery_request(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
    except Exception as exc:
        payload["ferebus_candidate_recovery"] = {
            "status": "invalid",
            "error": type(exc).__name__ + ": " + str(exc),
        }
    payload["latest_halt_event"] = next(
        (
            dict(event)
            for event in reversed(status_journal_events)
            if str(event.get("event") or "") == "halt"
        ),
        None,
    )
    aimall_postprocess_recovery_status = None
    ariadne_terminal_postprocess_status = None
    scalar_postprocess_recovery_status = None
    try:
        if supports_partial_array_recovery(state.phase):
            ledger = read_array_ledger(campaign, state.phase, int(state.iteration))
            if isinstance(ledger, dict):
                payload["partial_array_recovery"] = compact_array_recovery_summary(ledger)
                if state.phase is CampaignPhase.ARIADNE_ARRAY:
                    payload["ariadne_publication_recovery"] = (
                        classify_ariadne_publication(
                            campaign,
                            int(state.iteration),
                            expected_campaign_uid=str(state.campaign_uid),
                        )
                    )
    except Exception as exc:
        payload["partial_array_recovery_error"] = (
            type(exc).__name__ + ": " + str(exc)
        )
    try:
        from .daemon.reconcile import inspect_aimall_postprocess_recovery

        aimall_postprocess_recovery_status = (
            inspect_aimall_postprocess_recovery(campaign, state)
        )
    except Exception:
        aimall_postprocess_recovery_status = None
    if isinstance(aimall_postprocess_recovery_status, Mapping):
        payload["partial_array_recovery"] = compact_array_recovery_summary(
            dict(aimall_postprocess_recovery_status)
        )
        payload.pop("partial_array_recovery_error", None)
    if state.phase is CampaignPhase.ARIADNE_ARRAY:
        try:
            ariadne_terminal_postprocess_status = (
                _ariadne_terminal_postprocess_presentation_evidence(
                    campaign,
                    state,
                )
            )
        except Exception:
            ariadne_terminal_postprocess_status = None
    try:
        from .daemon.reconcile import (
            inspect_scalar_diversity_postprocess_recovery,
        )

        scalar_postprocess_recovery_status = (
            inspect_scalar_diversity_postprocess_recovery(campaign, state)
        )
    except Exception as exc:
        scalar_postprocess_recovery_status = None
        payload["partial_array_recovery_error"] = (
            "scalar publication recovery evidence is invalid: "
            + type(exc).__name__
            + ": "
            + str(exc)
        )
    if isinstance(scalar_postprocess_recovery_status, Mapping):
        payload["partial_array_recovery"] = compact_array_recovery_summary(
            dict(scalar_postprocess_recovery_status)
        )
        payload.pop("partial_array_recovery_error", None)
    config_review_status: Dict[str, Any]
    cfg: Optional[CampaignConfig] = None
    try:
        cfg = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        payload["campaign_config_status"] = {"ok": True}
        payload["pool_feasibility"] = _pool_feasibility_summary(campaign, cfg)
        try:
            config_review_status = config_review_evidence(
                review_config_changes(
                    campaign,
                    cfg,
                    state,
                    initialise_missing=False,
                ),
                allow_unbound=(
                    state.phase is CampaignPhase.INIT
                    and not any(
                        job_id for job_id in state.pending_jobs.values()
                    )
                    and not payload.get("active_submission_intents")
                ),
            )
        except Exception as exc:
            config_review_status = invalid_config_review_evidence(
                type(exc).__name__ + ": " + str(exc)
            )
    except Exception as exc:
        payload["campaign_config_status"] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }
        config_review_status = invalid_config_review_evidence(
            type(exc).__name__ + ": " + str(exc)
        )
    status_snapshot: Optional[Any] = None
    try:
        from .daemon.artifact_contracts import (
            artifact_manifest_status,
            state_artifact_contract_status,
        )
        status_snapshot = build_committed_artifact_snapshot(
            campaign,
            verification_level="authority",
        )
        payload["artifact_manifest_status"] = artifact_manifest_status(
            campaign,
            state,
            verification="authority",
            snapshot=status_snapshot,
        )
        payload["state_artifact_contract_status"] = state_artifact_contract_status(
            campaign,
            state,
            verification="authority",
            snapshot=status_snapshot,
        )
        payload["artifact_verification"] = status_snapshot.verification_payload()
        status_transaction_recovery = inspect_reconcile_transaction_recovery(
            campaign,
            artifact_snapshot=status_snapshot,
        )
    except Exception as exc:
        try:
            status_transaction_recovery = inspect_reconcile_transaction_recovery(
                campaign,
                artifact_snapshot=None,
            )
        except Exception:
            status_transaction_recovery = None
        payload["artifact_manifest_status"] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }
        payload["state_artifact_contract_status"] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
            "errors": [type(exc).__name__ + ": " + str(exc)],
        }
    presentation_payload = dict(payload)
    presentation_payload["_presentation_config_review"] = dict(
        config_review_status
    )
    scheduler_recovery_status = _load_scheduler_recovery_status(
        campaign,
        state,
    )
    ferebus_staging_evidence = _ferebus_staging_presentation_evidence(
        campaign,
        state,
        artifact_snapshot=status_snapshot,
        config=cfg,
    )
    if isinstance(ferebus_staging_evidence, Mapping):
        presentation_payload[
            "_presentation_ferebus_staging_recovery"
        ] = dict(ferebus_staging_evidence)
        disposition = str(
            ferebus_staging_evidence.get("disposition") or ""
        )
        if disposition == "contradictory":
            scheduler_recovery_status = {
                "state": "invalid",
                "phase": state.phase.value,
                "iteration": int(state.iteration),
                "replacement_round": int(state.replacement_round),
                "error": str(
                    ferebus_staging_evidence.get("reason")
                    or "FEREBUS staging recovery evidence is contradictory"
                ),
            }
        elif (
            ferebus_staging_evidence.get("source_submission_identities")
            and (
                scheduler_recovery_status is None
                or disposition == "archived_terminal_producer"
                or _ferebus_recovery_ledger_has_missing_map_failure(
                    campaign,
                    state,
                )
            )
        ):
            scheduler_recovery_status = {
                "state": "awaiting_validation",
                "phase": state.phase.value,
                "iteration": int(state.iteration),
                "replacement_round": int(state.replacement_round),
                "n_terminal_receipts": len(
                    ferebus_staging_evidence.get(
                        "terminal_receipt_paths", []
                    )
                ),
                "n_scheduler_completed": int(
                    ferebus_staging_evidence.get(
                        "scheduler_completed_candidates", 0
                    )
                ),
                "n_scheduler_retry": int(
                    ferebus_staging_evidence.get(
                        "known_retry_candidates", 0
                    )
                ),
                "original_job_ids": list(
                    ferebus_staging_evidence.get("source_job_ids", [])
                ),
                "staging_disposition": disposition,
            }
    if scheduler_recovery_status is not None:
        presentation_payload[
            "_presentation_scheduler_recovery"
        ] = scheduler_recovery_status
    if state.phase in {
        CampaignPhase.INITIAL_ALLOCATION_CHECK,
        CampaignPhase.ALLOCATION_CHECK,
    }:
        try:
            from .execution_identity import (
                inspect_allocation_check_transition_boundary,
            )

            presentation_payload[
                "_presentation_allocation_check_transition"
            ] = inspect_allocation_check_transition_boundary(campaign, state)
        except Exception as exc:
            presentation_payload[
                "_presentation_allocation_check_transition"
            ] = {
                "safe": False,
                "reason": type(exc).__name__ + ": " + str(exc),
            }
    if state.phase in {
        CampaignPhase.PHASE_A_DIVERSITY,
        CampaignPhase.PHASE_B_DIVERSITY,
    }:
        try:
            from .execution_identity import (
                inspect_scalar_diversity_transition_boundary,
            )

            presentation_payload[
                "_presentation_diversity_transition"
            ] = inspect_scalar_diversity_transition_boundary(
                campaign,
                state,
            )
        except Exception as exc:
            presentation_payload[
                "_presentation_diversity_transition"
            ] = {
                "safe": False,
                "reason": type(exc).__name__ + ": " + str(exc),
            }
    if isinstance(aimall_postprocess_recovery_status, Mapping):
        presentation_payload[
            "_presentation_aimall_postprocess_recovery"
        ] = dict(aimall_postprocess_recovery_status)
    if isinstance(ariadne_terminal_postprocess_status, Mapping):
        presentation_payload[
            "_presentation_ariadne_terminal_postprocess"
        ] = dict(ariadne_terminal_postprocess_status)
    try:
        profile = require_cluster_profile()
        presentation_payload["_presentation_scheduler_kind"] = str(
            profile.config[profile.machine]["hpc"]["scheduler"]
        ).strip().lower()
    except Exception:
        pass
    if isinstance(status_transaction_recovery, Mapping) and str(
        status_transaction_recovery.get("state") or ""
    ) != "none":
        presentation_payload["_presentation_reconcile_transaction_recovery"] = (
            dict(status_transaction_recovery)
        )
    scheduler_assessment = assess_campaign_presentation(
        presentation_payload
    ).scheduler
    environment_evidence = _preflight_environment_generation_evidence(
        campaign,
        state,
        cfg,
        scheduler_ownership_clear=(
            not _status_daemon_active(presentation_payload)
            and not scheduler_assessment.has_unresolved_scheduler_work
            and not payload.get("submission_intent_errors")
        ),
    )
    if isinstance(environment_evidence, Mapping):
        presentation_payload["_presentation_environment_generation"] = dict(
            environment_evidence
        )
    presentation_payload["_presentation_execution_identity_checked"] = True
    try:
        from .execution_identity import (
            execution_identity_path,
            read_execution_identity,
        )

        if execution_identity_path(campaign).is_file():
            identity = read_execution_identity(
                campaign,
                expected_campaign_uid=str(state.campaign_uid),
            )
            presentation_payload["_presentation_execution_mode"] = str(
                identity["mode"]
            )
    except Exception as exc:
        presentation_payload["_presentation_execution_identity_error"] = (
            type(exc).__name__ + ": " + str(exc)
        )
    payload["recommendations"] = recommendation_dicts(
        build_status_recommendations(
            campaign,
            presentation_payload,
            paths["journal"],
        )
    )
    payload["next_action"] = payload["recommendations"][0]["primary"]
    if not bool(getattr(args, "json", False)):
        payload["_presentation_config_review"] = dict(config_review_status)
    if (
        not bool(getattr(args, "json", False))
        and isinstance(aimall_postprocess_recovery_status, Mapping)
    ):
        payload["_presentation_aimall_postprocess_recovery"] = dict(
            aimall_postprocess_recovery_status
        )
    if (
        not bool(getattr(args, "json", False))
        and isinstance(ariadne_terminal_postprocess_status, Mapping)
    ):
        payload["_presentation_ariadne_terminal_postprocess"] = dict(
            ariadne_terminal_postprocess_status
        )
    if (
        not bool(getattr(args, "json", False))
        and scheduler_recovery_status is not None
    ):
        payload["_presentation_scheduler_recovery"] = dict(
            scheduler_recovery_status
        )
    if (
        not bool(getattr(args, "json", False))
        and isinstance(
            presentation_payload.get(
                "_presentation_diversity_transition"
            ),
            Mapping,
        )
    ):
        payload["_presentation_diversity_transition"] = dict(
            presentation_payload["_presentation_diversity_transition"]
        )
    if (
        not bool(getattr(args, "json", False))
        and bool(getattr(args, "verbose", False))
    ):
        try:
            from .layout import active_iteration_dir
            from .sampling_protocol import (
                read_sampling_protocol_resolved,
                sampling_protocol_resolved_path,
            )

            iteration_dir = active_iteration_dir(campaign, int(state.iteration))
            protocol_path = sampling_protocol_resolved_path(iteration_dir)
            if protocol_path.is_file() and not protocol_path.is_symlink():
                protocol = read_sampling_protocol_resolved(
                    iteration_dir,
                    expected_iteration=int(state.iteration),
                )
                policy = dict(protocol.get("sampling_policy") or {})
                scale_model = dict(protocol.get("sampling_scale_model") or {})
                payload["_presentation_sampling_protocol"] = {
                    "sampling_aggressiveness": int(
                        protocol["sampling_aggressiveness"]
                    ),
                    "policy_version": int(protocol["sampling_policy_version"]),
                    "target_motion_ratio": policy.get("target_motion_ratio"),
                    "initial_trust_multiplier": policy.get(
                        "initial_trust_multiplier"
                    ),
                    "movement_trust_multiplier": policy.get(
                        "movement_trust_multiplier"
                    ),
                    "movement_target_low_ratio": policy.get(
                        "movement_target_low_to_target"
                    ),
                    "movement_target_high_ratio": policy.get(
                        "movement_target_high_to_target"
                    ),
                    "lambda_move": policy.get("lambda_move"),
                    "movement_band_fraction": policy.get(
                        "movement_band_fraction"
                    ),
                    "movement_progress_fraction": policy.get(
                        "movement_progress_fraction"
                    ),
                    "movement_progress_normalisation": policy.get(
                        "movement_progress_normalisation"
                    ),
                    "under_move_retry_limit": policy.get(
                        "under_move_retry_limit"
                    ),
                    "under_move_feedback_min_factor": policy.get(
                        "under_move_feedback_min_factor"
                    ),
                    "under_move_feedback_max_factor": policy.get(
                        "under_move_feedback_max_factor"
                    ),
                    "baseline_source": dict(
                        scale_model.get("geometry_motion_scale") or {}
                    ).get("source"),
                }
        except Exception:
            pass
    if bool(getattr(args, "json", False)):
        machine_payload = dict(payload)
        machine_payload.pop("_presentation_stop_disposition", None)
        print(
            json.dumps(
                machine_payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    else:
        print(
            _format_status(
                payload,
                verbose=bool(getattr(args, "verbose", False)),
                campaign=campaign,
                journal_events=status_journal_events,
            ),
            end="",
        )
    return 0


def _state_payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finish_resume_transaction(
    campaign: Path,
    state_path: Path,
    *,
    before_state: Optional[CampaignState] = None,
    after_state: Optional[CampaignState] = None,
    request_id: Optional[str] = None,
    operation: Optional[str] = None,
    archive_status: str = "resumed",
) -> Tuple[CampaignState, Path]:
    """Finish a receipt-backed resume state/control transaction idempotently."""
    from .daemon.stop_control import (
        archive_and_clear_stop_request,
        archive_resume_transaction,
        prepare_resume_transaction,
        read_resume_transaction,
        stop_request_history_dir,
        update_resume_transaction,
    )

    transaction = read_resume_transaction(campaign)
    if transaction is None:
        if before_state is None or after_state is None or operation is None:
            raise RuntimeError("no resume transaction is available to finish")
        transaction = prepare_resume_transaction(
            campaign,
            before_state=before_state,
            after_state=after_state,
            request_id=request_id,
            operation=operation,
        )
    transaction_id = str(transaction["transaction_id"])
    status = str(transaction["status"])
    archived_request_path = transaction.get("stop_request_archive_path")
    transaction_request_id = transaction.get("request_id")
    if status == "prepared":
        if transaction_request_id is not None:
            archived = archive_and_clear_stop_request(
                campaign,
                status=str(archive_status),
                expected_request_id=str(transaction_request_id),
            )
            if archived is None:
                expected = stop_request_history_dir(campaign) / (
                    str(transaction_request_id) + ".json"
                )
                if not expected.is_file() or expected.is_symlink():
                    raise RuntimeError(
                        "resume transaction lost its stop-request archive evidence"
                    )
                archived = expected
            archived_request_path = str(archived)
        transaction = update_resume_transaction(
            campaign,
            transaction_id,
            status="control_archived",
            stop_request_archive_path=archived_request_path,
        )
        status = "control_archived"
    if status == "control_archived":
        current_state = read_state(state_path)
        current_digest = _state_payload_digest(current_state.to_dict())
        before_digest = str(transaction["before_state_sha256"])
        after_digest = str(transaction["after_state_sha256"])
        if current_digest not in {before_digest, after_digest}:
            raise RuntimeError(
                "state changed outside the active resume transaction"
            )
        target_state = CampaignState.from_dict(dict(transaction["after_state"]))
        if current_digest != after_digest:
            write_state(state_path, target_state)
        transaction = update_resume_transaction(
            campaign,
            transaction_id,
            status="state_written",
        )
        status = "state_written"
    if status != "state_written":
        raise RuntimeError("resume transaction did not reach state_written")
    final_state = read_state(state_path)
    if _state_payload_digest(final_state.to_dict()) != str(
        transaction["after_state_sha256"]
    ):
        raise RuntimeError("resume transaction target state was not persisted")
    history_path = archive_resume_transaction(campaign, transaction_id)
    return final_state, history_path


def _scheduler_uncertain_resume_target(
    campaign: Path,
    state: CampaignState,
    *,
    stop_request: Optional[Mapping[str, Any]],
    cancel_stop_request: bool,
) -> Tuple[CampaignState, Dict[str, Any]]:
    """Validate and prepare re-polling of one preserved scheduler job."""
    context = state.lifecycle_context
    if not isinstance(context, dict):
        raise ValueError("HALTED state has no lifecycle context")
    if str(context.get("disposition") or "") != "halted":
        raise ValueError("lifecycle disposition is not halted")
    if context.get("scheduler_uncertain") is not True:
        raise ValueError("halt is not marked scheduler-uncertain")
    if str(context.get("source") or "") != "daemon":
        raise ValueError("halt was not produced by the daemon scheduler guard")
    if bool(state.shutdown_requested):
        raise ValueError("scheduler-uncertain state also has shutdown_requested=true")

    try:
        from_phase = CampaignPhase(str(context.get("from_phase") or ""))
    except ValueError as exc:
        raise ValueError("lifecycle from_phase is invalid") from exc
    if from_phase in {CampaignPhase.HALTED, CampaignPhase.DONE}:
        raise ValueError("lifecycle from_phase is not resumable")

    pending = [
        (str(phase_name), str(job_id))
        for phase_name, job_id in state.pending_jobs.items()
        if job_id not in (None, "")
    ]
    if len(pending) != 1:
        raise ValueError(
            "scheduler-uncertain recovery requires exactly one pending job"
        )
    pending_phase, pending_job_id = pending[0]
    if pending_phase != from_phase.value:
        raise ValueError("pending-job phase does not match lifecycle from_phase")
    if str(context.get("job_id") or "") != pending_job_id:
        raise ValueError("pending JobID does not match lifecycle JobID")

    intent = _submission_intent.load_active_intent(
        campaign,
        from_phase.value,
        int(state.iteration),
        expected_campaign_uid=str(state.campaign_uid),
    )
    if intent is None:
        raise ValueError("matching active submission intent is missing")
    if str(intent.get("status") or "") not in {"SUBMITTED", "ADOPTED"}:
        raise ValueError("matching submission intent is not submitted or adopted")
    if str(intent.get("job_id") or "") != pending_job_id:
        raise ValueError("submission-intent JobID does not match pending JobID")

    target = CampaignState.from_dict(state.to_dict())
    target.phase = from_phase
    target.lifecycle_context = None
    target.shutdown_requested = False
    if stop_request is not None and not cancel_stop_request:
        from .daemon.stop_control import validate_stop_request_for_recovery

        disposition = validate_stop_request_for_recovery(
            campaign,
            stop_request,
            target,
        )
        if str(disposition.get("kind") or "") not in {
            "completed",
            "pending_boundary",
        }:
            raise ValueError(
                "the active stop request cannot be preserved while "
                "scheduler-uncertain work is resumed: "
                + str(
                    disposition.get("reason")
                    or disposition.get("kind")
                    or "unknown disposition"
                )
            )
    target.sacct_empty_streak = {
        key: value
        for key, value in target.sacct_empty_streak.items()
        if key != pending_job_id
        and not str(key).startswith(pending_job_id + ":")
    }
    return target, intent


def _verify_unaccepted_pre_submit_cancellation(
    campaign: Path,
    intent: Mapping[str, Any],
    stop_request: Mapping[str, Any],
    *,
    command_timeout_seconds: int,
) -> bool:
    """Re-prove that a cancelled PRE_SUBMIT attempt never reached a scheduler."""
    if (
        str(intent.get("status") or "") != "FAILED"
        or str(intent.get("reason") or "")
        != "user_cancelled_before_scheduler_acceptance"
        or intent.get("job_id") not in (None, "")
        or intent.get("expected_tasks") is not None
    ):
        return False
    phase = str(intent.get("phase") or "")
    iteration = int(intent.get("iteration", -1))
    replacement_round = int(intent.get("replacement_round", 0))
    if (
        str(stop_request.get("mode") or "") != "immediate"
        or stop_request.get("cancel_jobs_requested") is not True
        or str(stop_request.get("observed_phase") or "") != phase
        or int(stop_request.get("observed_iteration", -1)) != iteration
        or int(stop_request.get("observed_replacement_round", -1))
        != replacement_round
    ):
        raise ValueError(
            "unaccepted PRE_SUBMIT evidence does not match the stop request"
        )
    summary = stop_request.get("cancellation_summary")
    if not isinstance(summary, Mapping):
        raise ValueError(
            "unaccepted PRE_SUBMIT cancellation summary is missing"
        )
    matching = []
    for item in summary.get("cancelled", []):
        if not isinstance(item, Mapping):
            continue
        keys = item.get("intent_keys")
        if not isinstance(keys, list):
            continue
        if any(
            isinstance(key, Mapping)
            and str(key.get("phase") or "") == phase
            and int(key.get("iteration", -1)) == iteration
            for key in keys
        ):
            matching.append(item)
    if len(matching) != 1:
        raise ValueError(
            "stop request does not contain one exact unaccepted PRE_SUBMIT "
            "cancellation record"
        )
    record = matching[0]
    scheduler_kind = str(
        intent.get("scheduler_identity_kind") or "slurm"
    ).strip().lower()
    if (
        record.get("job_id") != ""
        or record.get("task_count_unknown") is not True
        or record.get("n_completed") != 0
        or record.get("terminal_receipt") not in (None, "")
        or str(record.get("scheduler_identity_kind") or "").strip().lower()
        != scheduler_kind
        or record.get("phases") != [phase]
        or record.get("intent_keys")
        != [{"phase": phase, "iteration": iteration}]
    ):
        raise ValueError(
            "unaccepted PRE_SUBMIT cancellation record is contradictory"
        )
    expected_job_name = str(intent.get("expected_job_name") or "")
    submission_kind = str(intent.get("submission_kind") or "")
    if not expected_job_name or submission_kind not in {"scalar", "array"}:
        raise ValueError(
            "unaccepted PRE_SUBMIT intent lacks its scheduler identity"
        )
    backend = get_scheduler_backend(scheduler_kind)
    lookup = backend.find_accounted_job_by_name(
        expected_job_name,
        expected_task_count=None,
        submission_kind=submission_kind,
        timeout_seconds=int(command_timeout_seconds),
    )
    if bool(getattr(lookup, "inconclusive", False)):
        raise ValueError(
            "scheduler no-acceptance proof is inconclusive: "
            + str(getattr(lookup, "error", None) or "unknown error")
        )
    if str(getattr(lookup, "job_id", None) or ""):
        raise ValueError(
            "the scheduler contains a job matching the cancelled PRE_SUBMIT "
            "attempt"
        )
    return True


def _cancelled_scheduler_resume_target(
    state: CampaignState,
    *,
    phase: str,
    intent: Mapping[str, Any],
) -> CampaignState:
    """Clear only scheduler ownership retired by an exact stop receipt."""
    target = CampaignState.from_dict(state.to_dict())
    target.shutdown_requested = False
    context = target.lifecycle_context
    if isinstance(context, Mapping) and str(
        context.get("disposition") or ""
    ) == "stopped":
        target.lifecycle_context = None
    target.pending_jobs[str(phase)] = None
    job_id = str(intent.get("job_id") or "")
    if job_id:
        target.sacct_empty_streak = {
            key: value
            for key, value in target.sacct_empty_streak.items()
            if key != job_id and not str(key).startswith(job_id + ":")
        }
    return target


def cmd_resume(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(args.campaign_dir)
    paths = _campaign_paths(campaign)
    state_path = paths["state"]
    if state_path.exists():
        try:
            state = read_state(state_path)
        except (StateSchemaError, json.JSONDecodeError) as exc:
            print("state.json invalid: " + str(exc), file=sys.stderr)
            return 5
        lock_status = _probe_daemon_lock(paths["lock"])
        stale_seconds, skew_seconds = _runtime_liveness_policy(campaign)
        lease_status = _probe_daemon_lease(
            paths["lease"],
            stale_seconds=stale_seconds,
            clock_skew_tolerance_seconds=skew_seconds,
        )
        if bool(getattr(args, "cancel_stop_request", False)):
            ownership, ownership_reason = _daemon_control_ownership(
                state,
                lock_status,
                lease_status,
            )
            if ownership == "active":
                return _cancel_running_boundary_stop_request(
                    campaign,
                    paths,
                    state,
                )
            if ownership == "inconclusive":
                print(
                    "cannot safely cancel the stop request while daemon "
                    "ownership is inconclusive: "
                    + ownership_reason,
                    file=sys.stderr,
                )
                return 7
        if lock_status.get("lock_held") is not False:
            print(
                "cannot resume while daemon lock ownership is active or inconclusive",
                file=sys.stderr,
            )
            return 7
        if lease_status.get("lease_fresh") is True:
            print(
                "cannot resume while a fresh daemon lease exists",
                file=sys.stderr,
            )
            return 7
        lifecycle = state.lifecycle_context or {}
        scheduler_uncertain_halt = bool(
            state.phase is CampaignPhase.HALTED
            and isinstance(lifecycle, Mapping)
            and lifecycle.get("scheduler_uncertain") is True
        )
        if (
            state.phase is CampaignPhase.HALTED
            and not scheduler_uncertain_halt
        ):
            print(
                "campaign is HALTED; preview reconcile and apply only after "
                "it reports a safe recovery",
                file=sys.stderr,
            )
            return 6
        if (
            state.phase is CampaignPhase.DONE
            and not bool(getattr(args, "reopen_converged", False))
        ):
            print(
                "campaign is DONE; rerun resume with --reopen-converged "
                "only after deliberately increasing campaign.max_iterations",
                file=sys.stderr,
            )
            return 6
        config_path = (
            Path(args.config).expanduser().resolve()
            if getattr(args, "config", None)
            else campaign / "campaign.yaml"
        )
        try:
            resume_config = CampaignConfig.from_yaml(config_path)
            resume_config_review = assert_config_unchanged_for_start(
                campaign,
                resume_config,
                state,
            )
        except Exception as exc:
            print(
                "resume precheck could not validate the campaign "
                "configuration: "
                + str(exc),
                file=sys.stderr,
            )
            print(
                "preview reconcile with `"
                + _campaign_command(campaign, "reconcile")
                + "` before resuming.",
                file=sys.stderr,
            )
            return 7
        if resume_config_review.changed:
            print(
                "campaign.yaml differs from the campaign configuration lock; "
                "the stop request and campaign state were left unchanged.",
                file=sys.stderr,
            )
            formatted = format_config_review(resume_config_review)
            if formatted:
                print(formatted, file=sys.stderr)
            print(
                "preview reconcile with `"
                + _campaign_command(campaign, "reconcile")
                + "` and apply only after it reports the change is safe.",
                file=sys.stderr,
            )
            return 7
        try:
            from .daemon.stop_control import (
                archive_and_clear_stop_request,
                read_resume_transaction,
                read_stop_request,
                validate_stop_request_for_recovery,
            )

            pending_resume = read_resume_transaction(campaign)
            if pending_resume is not None:
                operation = str(pending_resume.get("operation") or "")
                archive_status = "cancelled" if "cancel" in operation else "resumed"
                state, history_path = _finish_resume_transaction(
                    campaign,
                    state_path,
                    archive_status=archive_status,
                )
                print(
                    "completed interrupted resume transaction: " + str(history_path)
                )
            stop_request = read_stop_request(
                campaign,
                expected_campaign_uid=str(state.campaign_uid),
            )
        except Exception as exc:
            print(
                "stop-control metadata is invalid; run status and reconcile "
                "before resuming: " + str(exc),
                file=sys.stderr,
            )
            return 7
        cancelled_resume_phase: Optional[str] = None
        cancelled_resume_intent: Optional[Mapping[str, Any]] = None
        if (
            isinstance(stop_request, Mapping)
            and str(stop_request.get("mode") or "") == "immediate"
            and stop_request.get("cancel_jobs_requested") is True
        ):
            stop_phase = str(stop_request.get("observed_phase") or "")
            stop_iteration = int(
                stop_request.get("observed_iteration", state.iteration)
            )
            try:
                stopped_intent = _submission_intent.load_intent(
                    campaign,
                    stop_phase,
                    stop_iteration,
                    expected_campaign_uid=str(state.campaign_uid),
                )
            except Exception as exc:
                print(
                    "cancelled scheduler ownership could not be read; preview "
                    "reconcile before resuming: "
                    + str(exc),
                    file=sys.stderr,
                )
                return 7
            if isinstance(stopped_intent, Mapping):
                try:
                    terminal_receipt = load_scheduler_terminal_receipt(
                        campaign,
                        stopped_intent,
                    )
                except Exception as exc:
                    print(
                        "cancelled scheduler evidence is invalid; preview "
                        "reconcile before resuming: "
                        + str(exc),
                        file=sys.stderr,
                    )
                    return 7
                if (
                    terminal_receipt is None
                    and str(stopped_intent.get("job_id") or "")
                    and str(stopped_intent.get("status") or "")
                    in _submission_intent.ACTIVE_STATUSES
                ):
                    resolved, blockers = (
                        _resolve_terminal_submission_intents_for_apply(
                            campaign,
                            [dict(stopped_intent)],
                            persist_terminal_receipts=True,
                        )
                    )
                    if blockers or not resolved:
                        detail = (
                            str(blockers[0].get("reason"))
                            if blockers
                            else "terminal evidence is unavailable"
                        )
                        print(
                            "cancelled scheduler work cannot yet be recovered "
                            "safely; run reconcile to review it: "
                            + detail,
                            file=sys.stderr,
                        )
                        return 7
                    terminal_receipt = load_scheduler_terminal_receipt(
                        campaign,
                        stopped_intent,
                    )
                if terminal_receipt is not None:
                    if str(stopped_intent.get("status") or "") in (
                        _submission_intent.ACTIVE_STATUSES
                    ):
                        _submission_intent.mark_failed(
                            campaign,
                            stop_phase,
                            stop_iteration,
                            "user_cancelled_via_stop",
                        )
                    cancelled_resume_phase = stop_phase
                    cancelled_resume_intent = stopped_intent
                    if str(stop_request.get("status") or "") != "completed":
                        from .daemon.stop_control import complete_stop_request

                        completed_stop = complete_stop_request(
                            campaign,
                            str(stop_request["request_id"]),
                            reason="scheduler_cancellation_recovered",
                        )
                        if completed_stop is not None:
                            stop_request = completed_stop
                elif terminal_receipt is None:
                    try:
                        command_timeout, _confirmation_timeout = (
                            _runtime_scheduler_policy(campaign)
                        )
                        unaccepted_pre_submit = (
                            _verify_unaccepted_pre_submit_cancellation(
                                campaign,
                                stopped_intent,
                                stop_request,
                                command_timeout_seconds=command_timeout,
                            )
                        )
                    except Exception as exc:
                        print(
                            "cancelled PRE_SUBMIT work cannot be recovered "
                            "safely; run reconcile to review it: "
                            + str(exc),
                            file=sys.stderr,
                        )
                        return 7
                    if unaccepted_pre_submit:
                        cancelled_resume_phase = stop_phase
                        cancelled_resume_intent = stopped_intent
                        if str(stop_request.get("status") or "") != "completed":
                            from .daemon.stop_control import complete_stop_request

                            completed_stop = complete_stop_request(
                                campaign,
                                str(stop_request["request_id"]),
                                reason=(
                                    "scheduler_submission_stopped_before_staging"
                                ),
                            )
                            if completed_stop is None:
                                print(
                                    "cancelled PRE_SUBMIT stop request changed "
                                    "before it could be completed",
                                    file=sys.stderr,
                                )
                                return 7
                            stop_request = completed_stop
        try:
            stop_disposition = validate_stop_request_for_recovery(
                campaign,
                stop_request,
                state,
            )
        except Exception as exc:
            print(
                "the recorded stop request cannot be honoured from the "
                "current campaign position; preview reconcile before "
                "resuming: "
                + str(exc),
                file=sys.stderr,
            )
            return 7
        stop_kind = str(stop_disposition.get("kind") or "none")
        if (
            state.shutdown_requested
            and stop_request is not None
            and stop_kind in {"pending_boundary", "pending_immediate"}
        ):
            from .daemon.stop_control import complete_stop_request

            completed_stop = complete_stop_request(
                campaign,
                str(stop_request["request_id"]),
                reason="shutdown_state_recovered",
                completion_receipt=state.last_completion_receipt,
            )
            if completed_stop is None:
                print(
                    "the stopped campaign's user stop request changed before "
                    "it could be completed",
                    file=sys.stderr,
                )
                return 7
            stop_request = completed_stop
            stop_kind = "completed"
        if state.phase is CampaignPhase.HALTED:
            try:
                target_state, preserved_intent = _scheduler_uncertain_resume_target(
                    campaign,
                    state,
                    stop_request=stop_request,
                    cancel_stop_request=bool(
                        getattr(args, "cancel_stop_request", False)
                    ),
                )
            except (OSError, TypeError, ValueError) as exc:
                print(
                    "campaign is HALTED; run `ichor-al-daemon reconcile --campaign-dir "
                    + str(campaign)
                    + " --apply` if the recovery proposal is safe before resuming; "
                    + "scheduler-uncertain resume is unavailable: "
                    + str(exc),
                    file=sys.stderr,
                )
                return 6
            cancelling_stop = bool(
                getattr(args, "cancel_stop_request", False)
            )
            archive_recovered_stop = bool(
                cancelling_stop
                or stop_kind in {"campaign_terminal", "completed"}
            )
            try:
                state, resume_history = _finish_resume_transaction(
                    campaign,
                    state_path,
                    before_state=state,
                    after_state=target_state,
                    request_id=(
                        None
                        if stop_request is None or not archive_recovered_stop
                        else str(stop_request.get("request_id"))
                    ),
                    operation=(
                        "resume_scheduler_uncertain_cancel_stop"
                        if cancelling_stop
                        else (
                            "resume_scheduler_uncertain_completed_stop"
                            if archive_recovered_stop
                            else "resume_scheduler_uncertain_with_pending_stop"
                        )
                    ),
                    archive_status="cancelled" if cancelling_stop else "resumed",
                )
                if archive_recovered_stop:
                    stop_request = None
            except Exception as exc:
                print(
                    "could not complete scheduler-uncertain resume transaction: "
                    + str(exc),
                    file=sys.stderr,
                )
                return 7
            try:
                from .daemon.journal import append_event

                append_event(
                    paths["journal"],
                    "scheduler_uncertain_resumed",
                    phase=state.phase.value,
                    iteration=int(state.iteration),
                    job_id=str(preserved_intent["job_id"]),
                    intent_status=str(preserved_intent["status"]),
                    resume_transaction_history=str(resume_history),
                )
            except Exception:
                pass
            print(
                "restored "
                + state.phase.value
                + " to re-poll preserved "
                + _scheduler_display_name(
                    str(
                        preserved_intent.get("scheduler_identity_kind")
                        or "slurm"
                    )
                )
                + " job "
                + str(preserved_intent["job_id"])
                + "; no job was resubmitted"
            )
        def archive_stop_request(status: str, event_type: str) -> bool:
            nonlocal stop_request
            if stop_request is None:
                return True
            archived_request = dict(stop_request)
            try:
                archived = archive_and_clear_stop_request(
                    campaign,
                    status=status,
                    expected_request_id=str(archived_request.get("request_id")),
                )
            except Exception as exc:
                print(
                    "could not archive user stop request: " + str(exc),
                    file=sys.stderr,
                )
                return False
            stop_request = None
            try:
                from .daemon.journal import append_event

                append_event(
                    paths["journal"],
                    event_type,
                    phase=state.phase.value,
                    iteration=int(state.iteration),
                    request_id=str(archived_request.get("request_id")),
                    mode=str(archived_request.get("mode")),
                    prior_status=str(archived_request.get("status")),
                    archive_path=(None if archived is None else str(archived)),
                )
            except Exception:
                pass
            return True

        if state.phase is CampaignPhase.DONE:
            if not bool(getattr(args, "reopen_converged", False)):
                print(
                    "campaign is DONE; rerun resume with --reopen-converged "
                    "only after deliberately increasing campaign.max_iterations",
                    file=sys.stderr,
                )
                return 6
            config_path = (
                Path(args.config).expanduser().resolve()
                if getattr(args, "config", None)
                else campaign / "campaign.yaml"
            )
            try:
                config = CampaignConfig.from_yaml(config_path)
            except Exception as exc:
                print("campaign config could not be loaded: " + str(exc), file=sys.stderr)
                return 2
            try:
                lock_review = assert_config_unchanged_for_start(
                    campaign,
                    config,
                    state,
                )
            except Exception as exc:
                print("config lock check failed: " + str(exc), file=sys.stderr)
                return 7
            if lock_review.changed:
                print(
                    "campaign.yaml changed since the config lock was written; "
                    "preview reconcile and apply only after it reports the change is safe",
                    file=sys.stderr,
                )
                formatted = format_config_review(lock_review)
                if formatted:
                    print(formatted, file=sys.stderr)
                return 7
            configured_max = int(config.campaign.max_iterations)
            if configured_max <= int(state.iteration):
                print(
                    "--reopen-converged requires campaign.max_iterations greater "
                    "than the completed iteration ("
                    + str(int(state.iteration))
                    + ")",
                    file=sys.stderr,
                )
                return 6
            completed_iteration = int(state.iteration)
            target_state = CampaignState.from_dict(state.to_dict())
            target_state.phase = CampaignPhase.SEED_SELECT
            target_state.iteration = completed_iteration + 1
            target_state.max_iterations = configured_max
            target_state.shutdown_requested = False
            target_state.lifecycle_context = None
            target_state.last_completion_receipt = None
            cancelling_stop = bool(getattr(args, "cancel_stop_request", False))
            try:
                state, resume_history = _finish_resume_transaction(
                    campaign,
                    state_path,
                    before_state=state,
                    after_state=target_state,
                    request_id=(
                        None
                        if stop_request is None
                        else str(stop_request.get("request_id"))
                    ),
                    operation=(
                        "reopen_completed_cancel_stop"
                        if cancelling_stop
                        else "reopen_completed"
                    ),
                    archive_status="cancelled" if cancelling_stop else "resumed",
                )
                stop_request = None
            except Exception as exc:
                print(
                    "could not complete the resume transaction: " + str(exc),
                    file=sys.stderr,
                )
                return 7
            try:
                from .daemon.journal import append_event

                append_event(
                    _campaign_paths(campaign)["journal"],
                    "campaign_reopened",
                    phase=state.phase.value,
                    iteration=int(state.iteration),
                    completed_iteration=completed_iteration,
                    next_iteration=int(state.iteration),
                    max_iterations=configured_max,
                    explicit_reopen=True,
                    resume_transaction_history=str(resume_history),
                )
            except Exception:
                pass
            print(
                "reopened completed campaign at SEED_SELECT iteration "
                + str(int(state.iteration))
            )
        if state.shutdown_requested:
            context = state.lifecycle_context or {}
            if context and str(context.get("disposition") or "") != "stopped":
                print(
                    "shutdown_requested is not recorded as a user stop; "
                    "run reconcile before resuming",
                    file=sys.stderr,
                )
                return 6
            if (
                cancelled_resume_phase is not None
                and cancelled_resume_intent is not None
            ):
                target_state = _cancelled_scheduler_resume_target(
                    state,
                    phase=cancelled_resume_phase,
                    intent=cancelled_resume_intent,
                )
            else:
                target_state = CampaignState.from_dict(state.to_dict())
                target_state.shutdown_requested = False
                target_state.lifecycle_context = None
            cancelling_stop = bool(getattr(args, "cancel_stop_request", False))
            try:
                state, resume_history = _finish_resume_transaction(
                    campaign,
                    state_path,
                    before_state=state,
                    after_state=target_state,
                    request_id=(
                        None
                        if stop_request is None
                        else str(stop_request.get("request_id"))
                    ),
                    operation=(
                        "resume_stopped_cancel_stop"
                        if cancelling_stop
                        else "resume_stopped"
                    ),
                    archive_status="cancelled" if cancelling_stop else "resumed",
                )
                stop_request = None
            except Exception as exc:
                print(
                    "could not complete the resume transaction: " + str(exc),
                    file=sys.stderr,
                )
                return 7
            print("shutdown_requested=false set in " + str(state_path))
            try:
                from .daemon.journal import append_event

                append_event(
                    paths["journal"],
                    (
                        "user_stop_request_cancelled"
                        if cancelling_stop
                        else "user_stop_resumed"
                    ),
                    phase=state.phase.value,
                    iteration=int(state.iteration),
                    resume_transaction_history=str(resume_history),
                )
            except Exception:
                pass
        elif (
            cancelled_resume_phase is not None
            and cancelled_resume_intent is not None
        ):
            target_state = _cancelled_scheduler_resume_target(
                state,
                phase=cancelled_resume_phase,
                intent=cancelled_resume_intent,
            )
            cancelling_stop = bool(
                getattr(args, "cancel_stop_request", False)
            )
            try:
                state, resume_history = _finish_resume_transaction(
                    campaign,
                    state_path,
                    before_state=state,
                    after_state=target_state,
                    request_id=(
                        None
                        if stop_request is None
                        else str(stop_request.get("request_id"))
                    ),
                    operation=(
                        "resume_scheduler_cancellation_cancel_stop"
                        if cancelling_stop
                        else "resume_scheduler_cancellation"
                    ),
                    archive_status=(
                        "cancelled" if cancelling_stop else "resumed"
                    ),
                )
                stop_request = None
            except Exception as exc:
                print(
                    "could not complete the scheduler-cancellation resume "
                    "transaction: "
                    + str(exc),
                    file=sys.stderr,
                )
                return 7
            try:
                from .daemon.journal import append_event

                append_event(
                    paths["journal"],
                    (
                        "user_stop_request_cancelled"
                        if cancelling_stop
                        else "user_stop_resumed"
                    ),
                    phase=state.phase.value,
                    iteration=int(state.iteration),
                    resumed_scheduler_phase=cancelled_resume_phase,
                    resume_transaction_history=str(resume_history),
                )
            except Exception:
                pass
        elif (
            stop_request is not None
            and stop_kind in {"campaign_terminal", "completed"}
        ):
            target_state = CampaignState.from_dict(state.to_dict())
            try:
                state, resume_history = _finish_resume_transaction(
                    campaign,
                    state_path,
                    before_state=state,
                    after_state=target_state,
                    request_id=str(stop_request.get("request_id")),
                    operation="resume_completed_stop",
                    archive_status="resumed",
                )
                stop_request = None
            except Exception as exc:
                print(
                    "could not archive the completed user stop request: "
                    + str(exc),
                    file=sys.stderr,
                )
                return 7
            try:
                from .daemon.journal import append_event

                append_event(
                    paths["journal"],
                    "user_stop_resumed",
                    phase=state.phase.value,
                    iteration=int(state.iteration),
                    resume_transaction_history=str(resume_history),
                )
            except Exception:
                pass
        elif bool(getattr(args, "cancel_stop_request", False)):
            if not archive_stop_request(
                "cancelled",
                "user_stop_request_cancelled",
            ):
                return 7
        elif stop_request is not None:
            print(
                "active stop request retained; the resumed daemon will honour "
                + str(stop_request.get("mode"))
                + " request "
                + str(stop_request.get("request_id"))
            )
    return cmd_start(args)


def _intent_phase_iteration(intent: Dict[str, Any]) -> Tuple[str, int]:
    phase = str(intent.get("phase") or "")
    try:
        iteration = int(intent.get("iteration"))
    except Exception:
        iteration = -1
    return phase, iteration


def _matching_sacct_observations(job_id: str, observations: Sequence[Any]) -> List[Any]:
    prefix = str(job_id) + "_"
    return [
        observation
        for observation in observations
        if str(getattr(observation, "job_id", "")) == str(job_id)
        or str(getattr(observation, "job_id", "")).startswith(prefix)
    ]


def _terminal_sacct_state_for_intent(
    job_id: str,
    observations: Sequence[Any],
    *,
    scheduler_kind: str = "slurm",
    expected_task_count: Optional[int] = None,
    submission_kind: Optional[str] = None,
) -> Tuple[Optional[str], str, int]:
    from .submit import sacct_poll

    scheduler_name = _scheduler_display_name(scheduler_kind)
    accounting_name = (
        "qacct"
        if str(scheduler_kind).strip().lower() == "sge"
        else "sacct"
    )
    matching = _matching_sacct_observations(job_id, observations)
    if not matching:
        return (
            None,
            accounting_name + " returned no rows for job " + str(job_id),
            0,
        )
    if str(scheduler_kind).strip().lower() == "sge":
        summary = sacct_poll.aggregate_states(
            str(job_id),
            matching,
            expected_task_count=expected_task_count,
            submission_kind=submission_kind,
            strict_parent_job_id=False,
        )
        if summary.conflicting_task_indices:
            return (
                None,
                "Sun Grid Engine accounting returned conflicting task rows",
                len(matching),
            )
        if summary.out_of_range_task_indices:
            return (
                None,
                "Sun Grid Engine accounting returned out-of-range task rows",
                len(matching),
            )
        if int(summary.n_missing) > 0:
            return (
                None,
                "Sun Grid Engine accounting is still missing "
                + str(int(summary.n_missing))
                + " of "
                + str(int(summary.n_expected))
                + " expected task rows",
                len(matching),
            )
        matching = list(summary.observations)
    states = [observation.status for observation in matching]
    if any(state in sacct_poll.NON_TERMINAL_STATES for state in states):
        return (
            None,
            scheduler_name + " still has non-terminal rows for job " + str(job_id),
            len(matching),
        )
    if any(state == sacct_poll.JobStatus.UNKNOWN for state in states):
        return (
            None,
            scheduler_name + " returned unknown rows for job " + str(job_id),
            len(matching),
        )
    if not all(state in sacct_poll.TERMINAL_STATES for state in states):
        return (
            None,
            scheduler_name
            + " accounting rows are not conclusively terminal for job "
            + str(job_id),
            len(matching),
        )
    failures = [state for state in states if state in sacct_poll.FAILURE_STATES]
    if failures:
        return failures[0].value, "", len(matching)
    return (
        None,
        scheduler_name + " shows successful completion; reconcile will not clear "
        "the intent without postprocess verification",
        len(matching),
    )


def _intent_age_seconds(intent: Dict[str, Any]) -> Optional[float]:
    from datetime import datetime, timezone

    raw = (
        intent.get("updated_at_iso")
        or intent.get("updated_iso")
        or intent.get("created_iso")
    )
    if not isinstance(raw, str) or not raw:
        return None
    try:
        created = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - created).total_seconds())


def _intent_matches_cancel_stop_request(
    campaign: Path,
    intent: Mapping[str, Any],
) -> bool:
    """Return whether an immediate stop owns this exact scheduler attempt."""
    from .daemon.stop_control import read_stop_request

    try:
        request = read_stop_request(
            campaign,
            expected_campaign_uid=str(intent.get("campaign_uid") or ""),
        )
    except Exception:
        return False
    if (
        not isinstance(request, Mapping)
        or str(request.get("mode") or "") != "immediate"
        or request.get("cancel_jobs_requested") is not True
        or str(request.get("observed_phase") or "")
        != str(intent.get("phase") or "")
        or int(request.get("observed_iteration", -1))
        != int(intent.get("iteration", -2))
        or int(request.get("observed_replacement_round", -1))
        != int(intent.get("replacement_round", 0))
    ):
        return False
    job_id = str(intent.get("job_id") or "")
    summary = request.get("cancellation_summary")
    if not isinstance(summary, Mapping):
        return str(request.get("status") or "") == "cancelling"
    recorded_ids = {
        str(item.get("job_id") or "")
        for category in ("cancelled", "failed", "skipped")
        for item in (summary.get(category) or [])
        if isinstance(item, Mapping)
    }
    return job_id in recorded_ids or str(request.get("status") or "") == "cancelling"


def _resolve_terminal_submission_intents_for_apply(
    campaign: Path,
    active_intents: Sequence[Dict[str, Any]],
    *,
    pre_submit_stale_seconds: int = 900,
    persist_terminal_receipts: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Classify conclusively terminal active intents before safe apply.

    This is intentionally fail-closed.  ``reconcile --apply`` may clear a stale
    active intent only when the scheduler queue no longer sees the job and
    accounting reports terminal failure for the recorded JobID, or for the
    expected job name in the PRE_SUBMIT crash window where the JobID was never
    persisted. Missing scheduler data keeps the intent blocking so a user
    cannot accidentally duplicate a live job.
    """
    from .submit import sacct_poll

    terminal_candidates: List[Dict[str, Any]] = []
    stale_pre_submit_no_job: List[Dict[str, Any]] = []
    blocking: List[Dict[str, Any]] = []
    for intent in active_intents:
        phase, iteration = _intent_phase_iteration(intent)
        scheduler_kind = str(
            intent.get("scheduler_identity_kind") or "slurm"
        ).strip().lower()
        intent_identity = {
            "submission_identity": str(
                intent.get("submission_identity") or ""
            ),
            "attempt_id": str(intent.get("attempt_id") or ""),
            "replacement_round": int(
                intent.get("replacement_round") or 0
            ),
            "scheduler_identity_kind": scheduler_kind,
        }
        try:
            scheduler_backend = get_scheduler_backend(scheduler_kind)
        except ValueError as exc:
            blocking.append(
                {
                    "phase": phase,
                    "iteration": iteration,
                    "job_id": str(intent.get("job_id") or ""),
                    "expected_job_name": str(
                        intent.get("expected_job_name") or ""
                    ),
                    "reason": str(exc),
                }
            )
            continue
        job_id = str(intent.get("job_id") or "")
        cancellation_owned = _intent_matches_cancel_stop_request(
            campaign,
            intent,
        )
        intent_identity["intent_job_id"] = job_id
        expected_job_name = str(intent.get("expected_job_name") or "")
        status = str(intent.get("status") or "")
        if not phase or iteration < 0:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "expected_job_name": expected_job_name,
                "reason": "submission intent has malformed phase or iteration",
            })
            continue
        if not job_id:
            if (
                status == "PRE_SUBMIT"
                and phase == CampaignPhase.ARIADNE_ARRAY.value
                and isinstance(intent.get("postprocess_source"), Mapping)
            ):
                try:
                    terminal = (
                        _submission_intent.resolve_ariadne_terminal_postprocess_source(
                            campaign,
                            campaign_uid=str(intent.get("campaign_uid") or ""),
                            iteration=int(iteration),
                            intent=intent,
                        )
                    )
                except _submission_intent.AriadneTerminalPostprocessNotApplicable:
                    terminal = None
                except Exception as exc:
                    blocking.append(
                        {
                            "phase": phase,
                            "iteration": iteration,
                            "job_id": job_id,
                            "expected_job_name": expected_job_name,
                            "reason": (
                                "terminal ARIADNE postprocess wrapper is invalid: "
                                + type(exc).__name__
                                + ": "
                                + str(exc)
                            ),
                        }
                    )
                    continue
                if terminal is not None:
                    source = dict(terminal["postprocess_source"])
                    receipt = dict(terminal["terminal_receipt"])
                    terminal_candidates.append(
                        {
                            **intent_identity,
                            "phase": phase,
                            "iteration": iteration,
                            "job_id": str(source["job_id"]),
                            "producer_job_id": str(source["job_id"]),
                            "producer_submission_identity": str(
                                source["submission_identity"]
                            ),
                            "terminal_state": "LOCAL_POSTPROCESS_INTERRUPTED",
                            "n_sacct_rows": len(receipt.get("outcomes", [])),
                            "scheduler_recovery": False,
                            "ariadne_terminal_postprocess": True,
                            "n_completed": int(
                                terminal["n_scheduler_completed"]
                            ),
                            "n_retry": int(terminal["n_scheduler_failed"]),
                            "n_failed": int(terminal["n_scheduler_failed"]),
                            "scheduler_observation_sha256": str(
                                receipt["scheduler_observation_sha256"]
                            ),
                            "terminal_receipt": str(
                                scheduler_terminal_receipt_path(
                                    campaign,
                                    phase=receipt["phase"],
                                    iteration=int(receipt["iteration"]),
                                    replacement_round=int(
                                        receipt["replacement_round"]
                                    ),
                                    submission_identity=str(
                                        receipt["submission_identity"]
                                    ),
                                )
                            ),
                            "terminal_receipt_sha256": str(
                                receipt["receipt_sha256"]
                            ),
                        }
                    )
                    continue
            if status == "PRE_SUBMIT":
                if not expected_job_name:
                    blocking.append({
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": job_id,
                        "expected_job_name": expected_job_name,
                        "reason": "PRE_SUBMIT intent has no expected job name",
                    })
                    continue
                raw_expected_tasks = intent.get("expected_tasks")
                try:
                    expected_tasks = (
                        int(raw_expected_tasks)
                        if raw_expected_tasks is not None
                        else None
                    )
                except (TypeError, ValueError):
                    expected_tasks = None
                if expected_tasks is None:
                    from .daemon.scheduler_contracts import (
                        infer_expected_tasks_from_artifacts,
                    )

                    expected_tasks = infer_expected_tasks_from_artifacts(
                        campaign,
                        phase=phase,
                        iteration=int(iteration),
                        replacement_round=int(intent.get("replacement_round", 0) or 0),
                    )
                lookup = scheduler_backend.find_accounted_job_by_name(
                    expected_job_name,
                    expected_task_count=expected_tasks,
                    submission_kind=str(intent.get("submission_kind") or ""),
                    cancellation_requested=bool(cancellation_owned),
                )
                if lookup.inconclusive:
                    blocking.append({
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": job_id,
                        "expected_job_name": expected_job_name,
                        "reason": "job-name lookup inconclusive: "
                        + str(lookup.error or "unknown error"),
                    })
                    continue
                if lookup.job_id and not bool(lookup.terminal):
                    blocking.append({
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": str(lookup.job_id),
                        "expected_job_name": expected_job_name,
                        "reason": "matching scheduler job exists but no job_id "
                        "was persisted in the intent",
                    })
                    continue
                if lookup.job_id and bool(lookup.successful):
                    blocking.append({
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": str(lookup.job_id),
                        "expected_job_name": expected_job_name,
                        "reason": "matching scheduler job completed successfully "
                        "but no job_id was persisted in the intent; "
                        "postprocess/user review is required",
                    })
                    continue
                if lookup.job_id and bool(lookup.failed):
                    terminal_state = "FAILED"
                    for _row_job_id, row_state in list(lookup.rows):
                        if row_state and row_state != "COMPLETED":
                            terminal_state = str(row_state)
                            break
                    terminal_candidates.append({
                        **intent_identity,
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": str(lookup.job_id),
                        "expected_job_name": expected_job_name,
                        "terminal_state": terminal_state,
                        "n_sacct_rows": len(lookup.rows),
                    })
                    continue
                if lookup.job_id and bool(lookup.terminal):
                    blocking.append({
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": str(lookup.job_id),
                        "expected_job_name": expected_job_name,
                        "reason": "matching scheduler job is terminal but not "
                        "conclusively failed or successful; user review is required",
                    })
                    continue
                age = _intent_age_seconds(intent)
                if age is None:
                    blocking.append({
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": job_id,
                        "expected_job_name": expected_job_name,
                        "reason": "PRE_SUBMIT intent has no parseable timestamp",
                    })
                    continue
                if age < int(pre_submit_stale_seconds):
                    blocking.append({
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": job_id,
                        "expected_job_name": expected_job_name,
                        "reason": "PRE_SUBMIT intent is too recent to supersede",
                    })
                    continue
                stale_pre_submit_no_job.append({
                    **intent_identity,
                    "phase": phase,
                    "iteration": iteration,
                    "job_id": job_id,
                    "expected_job_name": expected_job_name,
                    "terminal_state": "PRE_SUBMIT_NO_JOB_ID",
                    "n_sacct_rows": len(lookup.rows),
                })
                continue
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "expected_job_name": expected_job_name,
                "reason": "submission intent has no job_id",
            })
            continue
        try:
            existing_terminal_receipt = load_scheduler_terminal_receipt(
                campaign,
                intent,
            )
        except Exception as exc:
            blocking.append(
                {
                    "phase": phase,
                    "iteration": iteration,
                    "job_id": job_id,
                    "expected_job_name": expected_job_name,
                    "reason": (
                        "stored scheduler terminal evidence is invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    ),
                }
            )
            continue
        if existing_terminal_receipt is not None:
            ordinary_ariadne = bool(
                not cancellation_owned
                and phase == CampaignPhase.ARIADNE_ARRAY.value
                and not scheduler_terminal_receipt_records_cancellation(
                    existing_terminal_receipt
                )
            )
            terminal_candidates.append(
                {
                    **intent_identity,
                    "phase": phase,
                    "iteration": iteration,
                    "job_id": job_id,
                    "terminal_state": (
                        "COMPLETED"
                        if ordinary_ariadne
                        and int(existing_terminal_receipt["n_retry"]) == 0
                        else "FAILED"
                        if ordinary_ariadne
                        else "USER_CANCELLED_TERMINAL"
                    ),
                    "n_sacct_rows": len(
                        existing_terminal_receipt.get("outcomes", [])
                    ),
                    "scheduler_recovery": not ordinary_ariadne,
                    "ariadne_terminal_postprocess": ordinary_ariadne,
                    "n_completed": int(
                        existing_terminal_receipt["n_completed"]
                    ),
                    "n_retry": int(existing_terminal_receipt["n_retry"]),
                    "n_failed": int(existing_terminal_receipt["n_retry"]),
                    "scheduler_observation_sha256": str(
                        existing_terminal_receipt.get(
                            "scheduler_observation_sha256"
                        )
                        or ""
                    ),
                    "terminal_receipt": str(
                        scheduler_terminal_receipt_path(
                            campaign,
                            phase=existing_terminal_receipt["phase"],
                            iteration=int(
                                existing_terminal_receipt["iteration"]
                            ),
                            replacement_round=int(
                                existing_terminal_receipt[
                                    "replacement_round"
                                ]
                            ),
                            submission_identity=str(
                                existing_terminal_receipt[
                                    "submission_identity"
                                ]
                            ),
                        )
                    ),
                    "terminal_receipt_sha256": str(
                        existing_terminal_receipt["receipt_sha256"]
                    ),
                }
            )
            continue
        queue_lookup = scheduler_backend.find_active_job_by_id(
            job_id,
            expected_job_name=expected_job_name,
            expected_owner=current_scheduler_user(),
        )
        if queue_lookup.inconclusive:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "expected_job_name": expected_job_name,
                "reason": scheduler_backend.display_name
                + " queue lookup inconclusive: "
                + str(queue_lookup.error or "unknown error"),
            })
            continue
        if queue_lookup.active:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "expected_job_name": expected_job_name,
                "reason": "job is still active in "
                + scheduler_backend.queue_command,
            })
            continue
        try:
            observations = scheduler_backend.poll_job(
                job_id,
                cancellation_requested=bool(cancellation_owned),
                expected_job_name=expected_job_name,
                expected_owner=current_scheduler_user(),
            )
        except Exception as exc:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "expected_job_name": expected_job_name,
                "reason": scheduler_backend.display_name
                + " accounting lookup failed: "
                + type(exc).__name__
                + ": "
                + str(exc),
            })
            continue
        if cancellation_owned:
            try:
                classification = classify_terminal_scheduler_evidence(
                    campaign,
                    intent,
                    observations,
                    queue_active=False,
                )
                receipt = (
                    write_scheduler_terminal_receipt(
                        campaign,
                        intent,
                        classification,
                    )
                    if persist_terminal_receipts
                    else None
                )
            except Exception as exc:
                blocking.append(
                    {
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": job_id,
                        "expected_job_name": expected_job_name,
                        "reason": (
                            "cancelled scheduler work is not yet recoverable: "
                            + type(exc).__name__
                            + ": "
                            + str(exc)
                        ),
                    }
                )
                continue
            terminal_candidates.append(
                {
                    **intent_identity,
                    "phase": phase,
                    "iteration": iteration,
                    "job_id": job_id,
                    "terminal_state": "USER_CANCELLED_TERMINAL",
                    "n_sacct_rows": len(observations),
                    "scheduler_recovery": True,
                    "n_completed": int(classification["n_completed"]),
                    "n_retry": int(classification["n_retry"]),
                    "terminal_receipt": (
                        None
                        if receipt is None
                        else str(
                            scheduler_terminal_receipt_path(
                                campaign,
                                phase=receipt["phase"],
                                iteration=int(receipt["iteration"]),
                                replacement_round=int(
                                    receipt["replacement_round"]
                                ),
                                submission_identity=str(
                                    receipt["submission_identity"]
                                ),
                            )
                        )
                    ),
                    "terminal_receipt_sha256": (
                        None
                        if receipt is None
                        else str(receipt["receipt_sha256"])
                    ),
                }
            )
            continue
        if phase == CampaignPhase.ARIADNE_ARRAY.value:
            try:
                classification = classify_terminal_scheduler_evidence(
                    campaign,
                    intent,
                    observations,
                    queue_active=False,
                )
                receipt = (
                    write_scheduler_terminal_receipt(
                        campaign,
                        intent,
                        classification,
                    )
                    if persist_terminal_receipts
                    else None
                )
            except Exception as exc:
                blocking.append(
                    {
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": job_id,
                        "expected_job_name": expected_job_name,
                        "reason": (
                            "terminal ARIADNE accounting is not recoverable: "
                            + type(exc).__name__
                            + ": "
                            + str(exc)
                        ),
                    }
                )
                continue
            n_failed = int(classification["n_retry"])
            terminal_candidates.append(
                {
                    **intent_identity,
                    "phase": phase,
                    "iteration": iteration,
                    "job_id": job_id,
                    "terminal_state": (
                        "COMPLETED" if n_failed == 0 else "FAILED"
                    ),
                    "n_sacct_rows": len(observations),
                    "scheduler_recovery": False,
                    "ariadne_terminal_postprocess": True,
                    "n_completed": int(classification["n_completed"]),
                    "n_retry": n_failed,
                    "n_failed": n_failed,
                    "scheduler_observation_sha256": str(
                        classification["scheduler_observation_sha256"]
                    ),
                    "terminal_receipt": (
                        None
                        if receipt is None
                        else str(
                            scheduler_terminal_receipt_path(
                                campaign,
                                phase=receipt["phase"],
                                iteration=int(receipt["iteration"]),
                                replacement_round=int(
                                    receipt["replacement_round"]
                                ),
                                submission_identity=str(
                                    receipt["submission_identity"]
                                ),
                            )
                        )
                    ),
                    "terminal_receipt_sha256": (
                        None
                        if receipt is None
                        else str(receipt["receipt_sha256"])
                    ),
                }
            )
            continue
        terminal_state, reason, n_rows = _terminal_sacct_state_for_intent(
            job_id,
            observations,
            scheduler_kind=scheduler_kind,
            expected_task_count=(
                int(intent["expected_tasks"])
                if isinstance(intent.get("expected_tasks"), int)
                and not isinstance(intent.get("expected_tasks"), bool)
                else None
            ),
            submission_kind=str(intent.get("submission_kind") or ""),
        )
        if terminal_state is None:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "expected_job_name": expected_job_name,
                "reason": reason,
            })
            continue
        terminal_candidates.append({
            **intent_identity,
            "phase": phase,
            "iteration": iteration,
            "job_id": job_id,
            "terminal_state": terminal_state,
            "n_sacct_rows": n_rows,
        })
    if blocking:
        return [], blocking
    resolved: List[Dict[str, Any]] = []
    for candidate in stale_pre_submit_no_job:
        reason = "reconcile_apply_pre_submit_no_job_id"
        payload = dict(candidate)
        payload["reason"] = reason
        payload["target_status"] = "SUPERSEDED"
        resolved.append(payload)
    for candidate in terminal_candidates:
        terminal_state = str(candidate["terminal_state"])
        failure_reason = (
            _submission_intent.ARIADNE_TERMINAL_POSTPROCESS_REASON
            if bool(candidate.get("ariadne_terminal_postprocess", False))
            else "user_cancelled_via_stop"
            if bool(candidate.get("scheduler_recovery", False))
            else "reconcile_apply_terminal_job:" + terminal_state
        )
        payload = dict(candidate)
        payload["reason"] = failure_reason
        payload["target_status"] = "FAILED"
        resolved.append(payload)
    return resolved, blocking


def _apply_terminal_intent_recovery_to_report(
    report: Any,
    resolved: Sequence[Mapping[str, Any]],
    *,
    persist_for_apply: bool,
) -> None:
    """Remove only proven terminal cancellation ownership from a report."""
    if not resolved:
        return
    keys = {
        (
            str(item.get("phase") or ""),
            int(item.get("iteration") or 0),
            str(item.get("job_id") or ""),
        )
        for item in resolved
    }
    intent_keys = {
        (phase, iteration)
        for phase, iteration, _job_id in keys
    }
    report.active_submission_intents = [
        intent
        for intent in list(
            getattr(report, "active_submission_intents", []) or []
        )
        if (
            str(intent.get("phase") or ""),
            int(intent.get("iteration") or 0),
        )
        not in intent_keys
    ]
    report.unsafe_reasons = [
        reason
        for reason in list(getattr(report, "unsafe_reasons", []) or [])
        if not str(reason).startswith("active submission intent(s) present:")
        and not str(reason).startswith(
            "scheduler-inconclusive prepared scratch task(s)"
        )
    ]
    report.blocking_artifacts = [
        value
        for value in list(getattr(report, "blocking_artifacts", []) or [])
        if str(value)
        not in {
            "prepared scratch ownership",
            "active submission intent(s)",
        }
    ]
    for scratch_record in list(
        getattr(report, "scratch_inventory", []) or []
    ):
        key = (
            str(scratch_record.get("phase") or ""),
            int(scratch_record.get("iteration") or 0),
            str(scratch_record.get("job_id") or ""),
        )
        if key in keys:
            scratch_record["status"] = "terminal_intent"
    for item in resolved:
        phase = str(item.get("phase") or "")
        job_id = str(item.get("job_id") or "")
        if (
            phase
            and getattr(report, "proposed_state", None) is not None
            and str(report.proposed_state.pending_jobs.get(phase) or "")
            == job_id
        ):
            report.proposed_state.pending_jobs[phase] = None
    report.scheduler_cancellation_recovery = [
        dict(item)
        for item in resolved
        if bool(item.get("scheduler_recovery", False))
    ]
    if persist_for_apply:
        existing = list(
            getattr(report, "receipt_backed_intent_repairs", []) or []
        )
        existing_keys = {
            (
                str(item.get("phase") or ""),
                int(item.get("iteration") or 0),
                str(item.get("job_id") or ""),
            )
            for item in existing
        }
        existing.extend(
            dict(item)
            for item in resolved
            if (
                str(item.get("phase") or ""),
                int(item.get("iteration") or 0),
                str(item.get("job_id") or ""),
            )
            not in existing_keys
        )
        report.receipt_backed_intent_repairs = existing


def _aimall_upstream_gaussian_recovery_evidence(
    report: Any,
) -> Optional[Dict[str, Any]]:
    """Return the exact AIMAll rewind selected by recovery planning."""
    evidence = getattr(
        report,
        "aimall_upstream_gaussian_recovery",
        None,
    )
    if not isinstance(evidence, Mapping):
        return None
    if str(evidence.get("upstream_rewind") or "") != "missing_aimall_pointdir":
        raise ValueError("invalid AIMAll-to-Gaussian recovery marker")
    source_by_target = {
        CampaignPhase.INITIAL_GAUSSIAN.value:
            CampaignPhase.INITIAL_AIMALL.value,
        CampaignPhase.GAUSSIAN.value: CampaignPhase.AIMALL.value,
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value:
            CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value,
        CampaignPhase.REPLACEMENT_GAUSSIAN.value:
            CampaignPhase.REPLACEMENT_AIMALL.value,
    }
    target_phase = str(report.proposed_state.phase.value)
    expected_source = source_by_target.get(target_phase)
    if (
        expected_source is None
        or str(evidence.get("phase") or "") != target_phase
        or str(evidence.get("source_phase") or "") != expected_source
        or int(evidence.get("iteration", -1))
        != int(report.proposed_state.iteration)
        or int(evidence.get("replacement_round", -1))
        != int(getattr(report.proposed_state, "replacement_round", 0))
    ):
        raise ValueError(
            "AIMAll-to-Gaussian recovery evidence does not match the "
            "selected recovery state"
        )
    retry_ids = [int(value) for value in list(evidence.get("retry_task_ids") or [])]
    if (
        not retry_ids
        or len(retry_ids) != len(set(retry_ids))
        or any(value < 0 for value in retry_ids)
        or int(evidence.get("n_retry") or 0) != len(retry_ids)
    ):
        raise ValueError("AIMAll-to-Gaussian recovery task set is invalid")
    partial = getattr(report, "partial_array_recovery", None)
    if not isinstance(partial, Mapping):
        raise ValueError("AIMAll-to-Gaussian partial recovery summary is missing")
    for key in ("phase", "iteration", "logical_total", "n_complete", "n_retry"):
        if str(partial.get(key)) != str(evidence.get(key)):
            raise ValueError(
                "AIMAll-to-Gaussian recovery summary does not match its "
                "authoritative evidence"
            )
    return dict(evidence)


def _perform_reconcile_apply_mutations(
    campaign: Path,
    report: Any,
    *,
    transaction: ReconcileTransaction,
    retrain_ferebus: bool,
    force_resubmit_array: bool,
    partial_array: Optional[Dict[str, Any]],
    force_array_phase: CampaignPhase,
    force_array_iteration: int,
    archive_existing_array_outputs: bool,
    data_staging_archive_mode: Optional[str],
    verification: str = "authority",
    artifact_snapshot: Optional[Any] = None,
    campaign_config: Optional[CampaignConfig] = None,
) -> Dict[str, Any]:
    """Perform only lossless, transaction-recorded reconcile mutations."""
    transaction_payload = getattr(transaction, "payload", None)
    archive_identity = (
        str(transaction_payload.get("transaction_id"))
        if isinstance(transaction_payload, Mapping)
        and transaction_payload.get("transaction_id")
        else None
    )
    result: Dict[str, Any] = {
        "ferebus_retrain_archive": [],
        "archived_array_outputs": [],
        "refreshed_array_ledger": None,
        "archived_scripts": [],
        "archived_data_staging": [],
        "archived_model_staging": [],
        "ferebus_staging_restore": None,
        "archived_reference_data_staging": [],
        "archived_reentry_staging": [],
        "archived_ariadne_publication": [],
        "archived_scalar_diversity_publication": [],
        "aimall_upstream_gaussian_recovery": None,
        "retired_completed_staging": {
            "retired": [],
            "deleted": [],
            "preserved": [],
            "warnings": [],
            "n_retired": 0,
            "n_deleted": 0,
            "n_preserved": 0,
        },
    }
    if (
        isinstance(partial_array, Mapping)
        and str(partial_array.get("phase") or "")
        in {
            CampaignPhase.PHASE_A_DIVERSITY.value,
            CampaignPhase.PHASE_B_DIVERSITY.value,
        }
        and str(partial_array.get("publication_disposition") or "")
        == "archive_and_retry"
    ):
        if archive_identity is None:
            raise ValueError(
                "scalar diversity publication archival requires a transaction identity"
            )
        scalar_archive = archive_incomplete_scalar_diversity_publication(
            campaign,
            partial_array,
            archive_identity=archive_identity,
        )
        archive_paths = [str(scalar_archive["archive_dir"])]
        result["archived_scalar_diversity_publication"] = archive_paths
        transaction.record_paths(
            "archive_scalar_diversity_publication",
            archive_paths,
        )
    upstream_evidence = _aimall_upstream_gaussian_recovery_evidence(report)
    if upstream_evidence is not None:
        upstream_recovery = prepare_aimall_upstream_gaussian_recovery(
            campaign,
            str(upstream_evidence["source_phase"]),
            int(report.proposed_state.iteration),
        )
        result["aimall_upstream_gaussian_recovery"] = upstream_recovery
        if bool(upstream_recovery.get("changed", False)):
            transaction.record_paths(
                "restore_gaussian_task_membership",
                [str(upstream_recovery["points_file"])],
            )
    publication = getattr(report, "ariadne_publication_recovery", None)
    archive_for_replay = bool(
        isinstance(publication, dict)
        and publication.get("archive_for_replay", False)
    )
    if isinstance(publication, dict) and (
        bool(publication.get("archive_required", False)) or archive_for_replay
    ):
        iteration = int(report.proposed_state.iteration)
        intent = _submission_intent.load_intent(
            campaign,
            CampaignPhase.ARIADNE_ARRAY.value,
            iteration,
            expected_campaign_uid=str(report.proposed_state.campaign_uid),
        )
        archived_publication = archive_ariadne_publication(
            campaign,
            iteration,
            reason="reconcile_postprocess_recovery",
            campaign_uid=str(report.proposed_state.campaign_uid),
            submission_identity=(
                str(intent.get("submission_identity"))
                if isinstance(intent, dict) and intent.get("submission_identity")
                else None
            ),
            classification=publication,
            force=archive_for_replay,
            archive_identity=archive_identity,
        )
        if bool(archived_publication.get("changed", False)):
            paths = [str(archived_publication["archive_dir"])]
            result["archived_ariadne_publication"] = paths
            transaction.record_paths("archive_ariadne_publication", paths)
    if retrain_ferebus:
        archived = archive_ferebus_iteration_staging_for_retrain(
            campaign,
            report.proposed_state,
            archive_identity=archive_identity,
        )
        if archived is not None:
            result["ferebus_retrain_archive"] = [str(archived)]
            transaction.record_paths("archive_ferebus_retrain", [str(archived)])

    ferebus_staging = getattr(report, "ferebus_staging_recovery", None)
    preserve_ferebus_iteration_staging = (
        isinstance(ferebus_staging, Mapping)
        and str(ferebus_staging.get("disposition") or "")
        in {"terminal_producer", "archived_terminal_producer"}
    )
    if (
        isinstance(ferebus_staging, Mapping)
        and str(ferebus_staging.get("disposition") or "")
        == "archived_terminal_producer"
    ):
        reference_version = int(
            ferebus_staging.get("iteration")
            if ferebus_staging.get("iteration") is not None
            else report.proposed_state.reference_data_version
        )
        reference_view = (
            artifact_snapshot.reference_view(reference_version)
            if artifact_snapshot is not None
            else None
        )
        current_context = classify_ferebus_staging_recovery(
            campaign,
            campaign_uid=str(report.proposed_state.campaign_uid),
            phase=str(ferebus_staging.get("phase") or ""),
            iteration=int(ferebus_staging.get("iteration") or 0),
            replacement_round=int(
                ferebus_staging.get("replacement_round") or 0
            ),
            reference_data_version=reference_version,
            reference_head_manifest_sha256=(
                None
                if reference_view is None
                else str(reference_view.head_manifest_sha256)
            ),
            reference_view_sha256=(
                None
                if reference_view is None
                else str(reference_view.cumulative_view_sha256)
            ),
            config=campaign_config,
        )
        if (
            current_context.disposition
            != "archived_terminal_producer"
            or str(current_context.source_transaction_id or "")
            != str(ferebus_staging.get("source_transaction_id") or "")
            or str(current_context.task_map_sha256 or "")
            != str(ferebus_staging.get("task_map_sha256") or "")
            or tuple(current_context.completed_logical_task_ids)
            != tuple(
                int(value)
                for value in ferebus_staging.get(
                    "completed_logical_task_ids", []
                )
            )
        ):
            raise ValueError(
                "FEREBUS staging recovery evidence changed after preview"
            )
        if archive_identity is None:
            raise ValueError(
                "FEREBUS producer staging restoration requires a transaction identity"
            )
        restoration = restore_archived_ferebus_producer_staging(
            campaign,
            current_context,
            transaction_id=archive_identity,
        )
        result["ferebus_staging_restore"] = restoration
        restored_paths = [str(restoration["restored_path"])]
        if restoration.get("archived_input_path"):
            restored_paths.append(str(restoration["archived_input_path"]))
        transaction.record_paths(
            "restore_ferebus_producer_staging",
            restored_paths,
        )

    if force_resubmit_array and isinstance(partial_array, dict):
        if archive_existing_array_outputs:
            archived_outputs = archive_existing_array_task_outputs(
                campaign,
                force_array_phase,
                int(force_array_iteration),
                archive_identity=archive_identity,
            )
            result["archived_array_outputs"] = list(archived_outputs)
            transaction.record_paths(
                "archive_array_outputs",
                archived_outputs,
            )
        refreshed = refresh_array_ledger(
            campaign,
            force_array_phase,
            int(force_array_iteration),
            force_resubmit=True,
        )
        result["refreshed_array_ledger"] = refreshed
        transaction.record_paths(
            "refresh_array_ledger",
            [str(array_ledger_path(campaign, force_array_phase, force_array_iteration))],
        )

    if ".DATA/SCRIPTS contains sbatch scripts" in report.unsafe_reasons:
        paths = archive_scripts_for_reconcile(
            campaign,
            archive_identity=archive_identity,
        )
        result["archived_scripts"] = paths
        transaction.record_paths("archive_scripts", paths)

    retirement_plan = getattr(report, "completed_staging_retirement", None)
    retirement_eligible = (
        list(retirement_plan.get("eligible") or [])
        if isinstance(retirement_plan, Mapping)
        else []
    )
    retirement_tombstones = (
        list(retirement_plan.get("pending_tombstones") or [])
        if isinstance(retirement_plan, Mapping)
        else []
    )
    if retirement_eligible or retirement_tombstones:
        from .daemon.staging_retirement import retire_completed_staging_buckets

        retirement = retire_completed_staging_buckets(
            campaign,
            classification={
                "eligible": retirement_eligible,
                "ambiguous": [],
                "noncanonical": [],
                "pending_tombstones": retirement_tombstones,
            },
        )
        result["retired_completed_staging"] = retirement
        transaction.record_paths(
            "retire_completed_staging",
            list(retirement["retired"]) + list(retirement["preserved"]),
        )

    if ".DATA/STAGING is non-empty" in report.unsafe_reasons:
        if data_staging_archive_mode == "ferebus_reentry":
            paths = archive_data_staging_for_ferebus_reentry(
                campaign,
                report.proposed_state,
                verification=verification,
                artifact_snapshot=artifact_snapshot,
                archive_identity=archive_identity,
            )
        elif data_staging_archive_mode == "user":
            paths = archive_data_staging_for_operator_reconcile(
                campaign,
                archive_identity=archive_identity,
            )
        else:
            paths = []
        result["archived_data_staging"] = paths
        transaction.record_paths("archive_data_staging", paths)

    if "dangling model staging directories exist" in report.unsafe_reasons:
        paths = clean_model_iteration_staging_for_reconcile(
            campaign,
            report.proposed_state,
            verification=verification,
            artifact_snapshot=artifact_snapshot,
            archive_identity=archive_identity,
            preserve_ferebus_iteration_staging=(
                preserve_ferebus_iteration_staging
            ),
        )
        result["archived_model_staging"] = paths
        transaction.record_paths("archive_model_staging", paths)

    if "dangling reference-data staging directories exist" in report.unsafe_reasons:
        paths = archive_reference_data_staging_for_reconcile(
            campaign,
            report.proposed_state,
            verification=verification,
            artifact_snapshot=artifact_snapshot,
            archive_identity=archive_identity,
        )
        result["archived_reference_data_staging"] = paths
        transaction.record_paths("archive_reference_data_staging", paths)

    paths = clean_reentry_staging(
        campaign,
        report.proposed_state.phase,
        archive_identity=archive_identity,
        preserve_ferebus_iteration_staging=preserve_ferebus_iteration_staging,
    )
    result["archived_reentry_staging"] = paths
    transaction.record_paths("archive_reentry_staging", paths)
    return result


def _fail_reconcile_transaction(
    transaction: Optional[ReconcileTransaction],
    reason: str,
) -> None:
    if transaction is None:
        return
    try:
        if (
            int(transaction.payload.get("schema_version", 1)) >= 2
            and str(transaction.payload.get("status") or "")
            in {"MUTATING", "COMMITTING"}
        ):
            transaction.record_interruption(reason)
        else:
            transaction.set_status("FAILED", reason=reason)
    except Exception:
        pass


def _restore_reconcile_pointer_snapshots(
    campaign: Path,
    snapshots: Sequence[Mapping[str, Any]],
) -> List[str]:
    errors: List[str] = []
    for snapshot in reversed(list(snapshots)):
        try:
            restore_version_pointer(campaign, snapshot)
        except Exception as exc:
            errors.append(
                str(snapshot.get("label") or "pointer")
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            )
    return errors


def _publish_reconcile_intent_transitions(
    campaign: Path,
    resolved_intents: Sequence[Mapping[str, Any]],
    recovered_state: Any,
    *,
    commit_plan: Optional[Mapping[str, Any]] = None,
) -> None:
    """Publish scheduler-backed intent transitions after the core commit."""
    from .daemon.journal import append_event

    planned = (
        {
            (str(item["phase"]), int(item["iteration"])): item
            for item in commit_plan.get("intent_transitions") or []
        }
        if isinstance(commit_plan, Mapping)
        else {}
    )
    published: set[Tuple[str, int]] = set()
    for item in resolved_intents:
        phase = str(item["phase"])
        iteration = int(item["iteration"])
        reason = str(item["reason"])
        target_status = str(item["target_status"])
        completion_receipt = item.get("completion_receipt")
        if isinstance(commit_plan, Mapping):
            plan_record = planned.get((phase, iteration))
            if not isinstance(plan_record, Mapping):
                raise ValueError("reconcile intent commit plan is incomplete")
            publish_reconcile_intent_target(campaign, plan_record)
        elif target_status == "SUPERSEDED":
            _submission_intent.mark_superseded(
                campaign,
                phase,
                iteration,
                reason,
                completion_receipt=(
                    dict(completion_receipt)
                    if isinstance(completion_receipt, Mapping)
                    else None
                ),
            )
        elif target_status == "FAILED":
            _submission_intent.mark_failed(campaign, phase, iteration, reason)
        else:
            raise ValueError(
                "unsupported reconcile intent transition: " + target_status
            )
        published.add((phase, iteration))
        receipt_backed = isinstance(completion_receipt, Mapping)
        append_event(
            campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
            (
                "submission_intent_retired_without_submission"
                if receipt_backed
                else "reconcile_resolved_terminal_intent"
            ),
            phase=phase,
            iteration=iteration,
            submission_identity=str(item.get("submission_identity") or ""),
            job_id=str(item.get("job_id") or ""),
            terminal_state=str(item.get("terminal_state") or ""),
            n_sacct_rows=int(item.get("n_sacct_rows") or 0),
            reason=reason,
            target_status=target_status,
            completion_receipt=(
                dict(completion_receipt) if receipt_backed else None
            ),
            ariadne_terminal_postprocess=bool(
                item.get("ariadne_terminal_postprocess", False)
            ),
            scheduler_completed_task_candidates=(
                int(item.get("n_completed") or 0)
                if bool(item.get("ariadne_terminal_postprocess", False))
                else None
            ),
            scheduler_failed_task_candidates=(
                int(item.get("n_failed") or item.get("n_retry") or 0)
                if bool(item.get("ariadne_terminal_postprocess", False))
                else None
            ),
            scheduler_tasks_resubmitted=(
                0
                if bool(item.get("ariadne_terminal_postprocess", False))
                else None
            ),
            terminal_receipt=item.get("terminal_receipt"),
            terminal_receipt_sha256=item.get("terminal_receipt_sha256"),
        )

    for key, plan_record in planned.items():
        if key not in published:
            publish_reconcile_intent_target(campaign, plan_record)
    if not isinstance(commit_plan, Mapping):
        phase = recovered_state.phase.value
        iteration = int(recovered_state.iteration)
        current = _submission_intent.load_intent(
            campaign,
            phase,
            iteration,
            expected_campaign_uid=str(recovered_state.campaign_uid),
        )
        if current is not None and str(current.get("status")) == "FAILED":
            _submission_intent.mark_superseded(
                campaign,
                phase,
                iteration,
                "reconcile_apply_retry",
            )


def _complete_reconcile_scheduler_cancellation_stop(
    campaign: Path,
    recovered_state: Any,
    recoveries: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Complete only the immediate stop that owns the recovered attempts."""
    if not recoveries:
        return None
    from .daemon.stop_control import (
        complete_stop_request,
        read_stop_request,
    )

    request = read_stop_request(
        campaign,
        expected_campaign_uid=str(recovered_state.campaign_uid),
    )
    if not isinstance(request, Mapping):
        raise ValueError(
            "scheduler cancellation recovery has no matching stop request"
        )
    expected_identity = (
        str(recovered_state.phase.value),
        int(recovered_state.iteration),
        int(getattr(recovered_state, "replacement_round", 0)),
    )
    observed_identity = (
        str(request.get("observed_phase") or ""),
        int(request.get("observed_iteration", -1)),
        int(request.get("observed_replacement_round", -1)),
    )
    if (
        str(request.get("mode") or "") != "immediate"
        or request.get("cancel_jobs_requested") is not True
        or observed_identity != expected_identity
    ):
        raise ValueError(
            "scheduler cancellation recovery does not match the active "
            "immediate stop request"
        )
    recovery_job_ids = {
        str(item.get("job_id") or "")
        for item in recoveries
        if str(item.get("job_id") or "")
    }
    summary = request.get("cancellation_summary")
    recorded_job_ids = {
        str(item.get("job_id") or "")
        for category in ("cancelled", "failed", "skipped")
        for item in (
            summary.get(category, [])
            if isinstance(summary, Mapping)
            else []
        )
        if isinstance(item, Mapping) and str(item.get("job_id") or "")
    }
    if recovery_job_ids and not recovery_job_ids.issubset(recorded_job_ids):
        raise ValueError(
            "scheduler cancellation recovery JobIDs do not match the stop "
            "request"
        )
    if str(request.get("status") or "") == "completed":
        return dict(request)
    completed = complete_stop_request(
        campaign,
        str(request["request_id"]),
        reason="scheduler_cancellation_recovered",
    )
    if not isinstance(completed, Mapping):
        raise ValueError("scheduler cancellation stop request was not completed")
    return dict(completed)


def _prepare_reconcile_stop_guard(
    campaign: Path,
    proposed_state: CampaignState,
) -> Dict[str, Any]:
    """Bind reconcile to one validated stop-control snapshot."""
    from .daemon.stop_control import (
        describe_stop_request,
        stop_request_path,
        validate_stop_request,
        validate_stop_request_for_recovery,
    )

    path = stop_request_path(campaign)
    if not path.exists():
        return {
            "path": path,
            "raw": None,
            "request": None,
            "mutable_during_reconcile": False,
            "description": None,
        }
    if path.is_symlink() or not path.is_file():
        raise ValueError("stop request is not a regular file: " + str(path))
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), source=path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("stop request is unreadable: " + str(path)) from exc
    request = validate_stop_request(
        payload,
        expected_campaign_uid=str(proposed_state.campaign_uid),
    )
    if path.read_bytes() != raw:
        raise ValueError("stop request changed during reconcile inspection")
    mutable_during_reconcile = bool(
        str(request.get("mode") or "") == "immediate"
        and request.get("cancel_jobs_requested") is True
    )
    if mutable_during_reconcile:
        disposition = {
            "kind": (
                "completed"
                if str(request.get("status") or "") == "completed"
                else "cancelling"
            ),
            "launchable": False,
            "preserve": True,
        }
    else:
        disposition = validate_stop_request_for_recovery(
            campaign,
            request,
            proposed_state,
        )
    return {
        "path": path,
        "raw": raw,
        "request": request,
        "disposition": disposition,
        "mutable_during_reconcile": mutable_during_reconcile,
        "description": describe_stop_request(request),
    }


def _recheck_reconcile_stop_guard(
    campaign: Path,
    guard: Mapping[str, Any],
    proposed_state: CampaignState,
) -> None:
    """Fail before state publication if stop control changed or became unsafe."""
    from .daemon.stop_control import (
        stop_request_path,
        validate_stop_request,
        validate_stop_request_for_recovery,
    )

    path = stop_request_path(campaign)
    expected_raw = guard.get("raw")
    if bool(guard.get("mutable_during_reconcile", False)):
        current = _prepare_reconcile_stop_guard(campaign, proposed_state)
        if current.get("request") is None:
            raise ValueError(
                "stop request disappeared while reconcile was applying recovery"
            )
        return
    if expected_raw is None:
        if path.exists() or path.is_symlink():
            raise ValueError(
                "a stop request appeared while reconcile was applying recovery"
            )
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "stop request changed type while reconcile was applying recovery"
        )
    observed_raw = path.read_bytes()
    if observed_raw != expected_raw:
        raise ValueError(
            "stop request changed while reconcile was applying recovery"
        )
    request = validate_stop_request(
        json.loads(observed_raw.decode("utf-8"), source=path),
        expected_campaign_uid=str(proposed_state.campaign_uid),
    )
    validate_stop_request_for_recovery(campaign, request, proposed_state)


def _retry_phase_from_cleaned_report(report) -> Optional[CampaignPhase]:
    if not bool(getattr(report, "last_phase_retryable", False)):
        return None
    try:
        phase = CampaignPhase(str(report.last_phase_in_journal))
    except Exception:
        return None
    if phase in RETRYABLE_CLEANED_REENTRY_PHASES:
        return phase
    return None


def _apply_retry_phase_after_cleaned_halt(report, original_report) -> bool:
    if report.proposed_state.phase is not CampaignPhase.STOP_CHECK:
        return False
    retry_phase = _retry_phase_from_cleaned_report(original_report)
    if retry_phase is None:
        return False
    report.proposed_state.phase = retry_phase
    if original_report.last_iteration_in_journal is not None:
        try:
            report.proposed_state.iteration = int(original_report.last_iteration_in_journal)
        except (TypeError, ValueError):
            pass
    report.proposed_state.pending_jobs = {}
    report.proposed_state.shutdown_requested = False
    report.notes.append(
        "re-entry at "
        + retry_phase.value
        + " after cleaning transient run artefacts"
    )
    return True


def _propose_recovery_after_cleanup(
    campaign: Path,
    *,
    allow_fresh_init_on_nonempty: bool,
    active_reconcile_transaction_id: Optional[str],
    artifact_snapshot: Any,
    verification_level: str,
    terminal_recoveries: Sequence[Mapping[str, Any]],
) -> Any:
    """Reinspect cleanup without discarding proven terminal ownership."""
    preserved_terminal_recoveries = [
        dict(item) for item in terminal_recoveries
    ]
    report = propose_recovery(
        campaign,
        allow_fresh_init_on_nonempty=allow_fresh_init_on_nonempty,
        _active_reconcile_transaction_id=active_reconcile_transaction_id,
        _terminal_submission_intents=preserved_terminal_recoveries,
        artifact_snapshot=artifact_snapshot,
        verification_level=verification_level,
    )
    _apply_terminal_intent_recovery_to_report(
        report,
        preserved_terminal_recoveries,
        persist_for_apply=True,
    )
    return report


def _apply_runtime_config_to_recovered_state(report, config: Optional[CampaignConfig]) -> None:
    if config is None:
        return
    try:
        configured_max = int(config.campaign.max_iterations)
    except Exception:
        return
    if int(report.proposed_state.max_iterations) != configured_max:
        report.notes.append(
            "max_iterations set from campaign.yaml runtime config: "
            + str(configured_max)
        )
        report.proposed_state.max_iterations = configured_max


def _reconcile_apply_contract_error(
    campaign: Path,
    state: Any,
    *,
    verification: str = "authority",
    artifact_snapshot: Optional[Any] = None,
) -> Optional[str]:
    from .daemon.artifact_contracts import state_artifact_contract_status

    try:
        validate_phase_recovery_contract(
            campaign,
            state,
            verification=verification,
            artifact_snapshot=artifact_snapshot,
        )
    except Exception as exc:
        return (
            "phase="
            + str(getattr(state, "phase", "UNKNOWN"))
            + " iteration="
            + str(getattr(state, "iteration", "UNKNOWN"))
            + ": "
            + type(exc).__name__
            + ": "
            + str(exc)
        )
    status = state_artifact_contract_status(
        campaign,
        state,
        verification=verification,
        snapshot=artifact_snapshot,
    )
    if bool(status.get("ok")):
        return None
    detail = str(status.get("error") or "state contract invalid")
    return (
        "phase="
        + str(status.get("phase"))
        + " reference_data_version="
        + str(status.get("reference_data_version"))
        + " models_version="
        + str(status.get("models_version"))
        + ": "
        + detail
    )


def _reconcile_relative_path(campaign: Path, value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        return text
    try:
        return str(path.relative_to(campaign))
    except ValueError:
        return text


def _format_reconcile_contract_item(campaign: Path, item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    phase = str(item.get("phase") or "UNKNOWN")
    iteration = str(item.get("iteration") if item.get("iteration") is not None else "?")
    path = _reconcile_relative_path(campaign, item.get("path"))
    label = phase + "@" + iteration
    if path:
        label += " -> " + path
    return label


def _print_reconcile_bullets(title: str, items: Sequence[Any]) -> None:
    print(title + ":")
    if not items:
        print("  - none")
        return
    for item in items:
        print("  - " + str(item))


def _reconcile_cleanable_reasons(report: Any) -> List[str]:
    cleanable: List[str] = []
    reasons = list(getattr(report, "unsafe_reasons", []))
    revalidation = getattr(report, "aimall_quality_revalidation", None)
    revalidation_ready = bool(
        isinstance(revalidation, Mapping)
        and revalidation.get("state") in {"eligible", "resumable"}
        and revalidation.get("eligible") is True
    )
    if ".DATA/SCRIPTS contains sbatch scripts" in reasons:
        cleanable.append(".DATA/SCRIPTS contains sbatch scripts")
    if "dangling model staging directories exist" in reasons:
        cleanable.append("dangling model staging directories exist")
    if "dangling reference-data staging directories exist" in reasons:
        cleanable.append("dangling reference-data staging directories exist")
    if ".DATA/STAGING is non-empty" in reasons and not revalidation_ready:
        cleanable.append(
            ".DATA/STAGING is non-empty (requires --archive-staging when safe)"
        )
    if "stale ARIADNE publication" in reasons:
        cleanable.append("stale ARIADNE publication")
    retirement = getattr(report, "completed_staging_retirement", None)
    if isinstance(retirement, Mapping):
        for record in retirement.get("eligible") or []:
            action = str(record.get("action") or "")
            cleanable.append(
                "completed staging for iteration "
                + str(int(record.get("iteration", 0)))
                + " will be "
                + ("deleted" if action == "delete" else "preserved for diagnostics")
            )
        if retirement.get("pending_tombstones"):
            cleanable.append("incomplete completed-staging deletion will be retried")
    return cleanable


def _reconcile_campaign_assessment(
    state: Optional[CampaignState],
    report: Any,
) -> Any:
    intents = list(
        getattr(report, "active_submission_intents", []) or []
    )
    recoveries = [
        dict(item)
        for item in (
            getattr(report, "scheduler_cancellation_recovery", []) or []
        )
        if isinstance(item, Mapping)
    ]
    payload: Dict[str, Any] = (
        state.to_dict() if state is not None else {}
    )
    payload["active_submission_intents"] = intents
    if recoveries:
        payload["_presentation_scheduler_recovery"] = {
            "state": "awaiting_validation",
            "phase": (
                state.phase.value
                if state is not None
                else str(recoveries[0].get("phase") or "")
            ),
            "iteration": (
                int(state.iteration)
                if state is not None
                else int(recoveries[0].get("iteration") or 0)
            ),
            "replacement_round": (
                int(state.replacement_round)
                if state is not None
                else int(recoveries[0].get("replacement_round") or 0)
            ),
            "original_job_ids": [
                str(item.get("job_id"))
                for item in recoveries
                if item.get("job_id")
            ],
        }
    return assess_campaign_presentation(payload)


def _reconcile_hard_blockers(
    report: Any,
    contract_status: Optional[Dict[str, Any]] = None,
) -> List[str]:
    revalidation = getattr(report, "aimall_quality_revalidation", None)
    revalidation_ready = bool(
        isinstance(revalidation, Mapping)
        and revalidation.get("state") in {"eligible", "resumable"}
        and revalidation.get("eligible") is True
    )
    cleanable_exact = {
        ".DATA/SCRIPTS contains sbatch scripts",
        "dangling model staging directories exist",
        "dangling reference-data staging directories exist",
        "stale ARIADNE publication",
    }
    cleanable_blocking_artifacts = set()
    reasons = set(str(reason) for reason in getattr(report, "unsafe_reasons", []))
    if "dangling model staging directories exist" in reasons:
        cleanable_blocking_artifacts.add("dangling model staging")
    if "dangling reference-data staging directories exist" in reasons:
        cleanable_blocking_artifacts.add("dangling reference-data staging")
    if "stale ARIADNE publication" in reasons:
        cleanable_blocking_artifacts.add("stale ARIADNE publication")
    blockers: List[str] = []
    for reason in getattr(report, "unsafe_reasons", []):
        if reason in cleanable_exact:
            continue
        if reason == ".DATA/STAGING is non-empty":
            if revalidation_ready:
                continue
            blockers.append(
                ".DATA/STAGING is non-empty unless --archive-staging is explicitly requested"
            )
            continue
        blockers.append(str(reason))
    for item in getattr(report, "blocking_artifacts", []):
        text = str(item)
        if revalidation_ready and text == ".DATA/STAGING":
            continue
        if text in cleanable_blocking_artifacts:
            continue
        blockers.append(text)
    scheduler = _reconcile_campaign_assessment(
        getattr(report, "proposed_state", None),
        report,
    ).scheduler
    if scheduler.scheduler_intents or scheduler.local_intents:
        blockers.append("active submission intent(s) present")
    if contract_status is not None:
        phase = str(contract_status.get("selected_phase") or "")
        for item in contract_status.get("missing_or_invalid_inputs", []):
            text = str(item)
            if phase == CampaignPhase.HALTED.value and text.startswith("phase HALTED"):
                continue
            blockers.append("missing/invalid input: " + text)
    seen = set()
    out: List[str] = []
    for item in blockers:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _reconcile_valid_candidates(
    campaign: Path,
    contract_status: Dict[str, Any],
    report: Any,
) -> List[str]:
    report_candidates = getattr(report, "recovery_candidates", None)
    if isinstance(report_candidates, list) and report_candidates:
        return [
            _format_reconcile_contract_item(campaign, item)
            for item in report_candidates
        ]
    return [
        _format_reconcile_contract_item(campaign, item)
        for item in contract_status.get("trusted_handoffs", [])
    ]


def _reconcile_decision_payload(
    campaign: Path,
    report: Any,
    contract_status: Dict[str, Any],
    *,
    proposed_state_path: Optional[Path] = None,
    runtime_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cleanable = _reconcile_cleanable_reasons(report)
    blockers = _reconcile_hard_blockers(report, contract_status)
    candidates = _reconcile_valid_candidates(campaign, contract_status, report)
    selected_state = report.proposed_state
    snapshot = getattr(report, "artifact_snapshot", None)
    deep_pending = bool(
        getattr(report, "deep_verification_required", False)
        and (
            snapshot is None
            or str(getattr(snapshot, "verification_level", "authority")) != "deep"
        )
    )
    revalidation = getattr(report, "aimall_quality_revalidation", None)
    revalidation_ready = bool(
        isinstance(revalidation, Mapping)
        and revalidation.get("state") in {"eligible", "resumable"}
        and revalidation.get("eligible") is True
    )
    intentionally_stopped = _is_intentionally_stopped_state(selected_state)
    runnable = (
        bool(contract_status.get("contract_ok"))
        and not blockers
        and not deep_pending
        and not intentionally_stopped
        and selected_state.phase not in {
            CampaignPhase.HALTED,
            CampaignPhase.DONE,
        }
    )
    why_not_runnable: List[str] = []
    if not bool(contract_status.get("contract_ok")):
        why_not_runnable.extend(
            "missing/invalid input: " + str(item)
            for item in contract_status.get("missing_or_invalid_inputs", [])
        )
    why_not_runnable.extend(blockers)
    if deep_pending:
        why_not_runnable.append(
            "deep verification is required before campaign authority can be reconstructed"
        )
    if selected_state.phase in (CampaignPhase.HALTED, CampaignPhase.DONE):
        why_not_runnable.append(
            "selected phase is terminal: " + selected_state.phase.value
        )
    if intentionally_stopped:
        why_not_runnable.append(
            "campaign is intentionally "
            + _stopped_boundary_description(selected_state)
        )
    if deep_pending:
        next_command = _reconcile_apply_command(campaign, report)
    elif intentionally_stopped and (
        cleanable
        or _reconcile_has_only_explicit_cleanup_blockers(report, blockers)
    ):
        next_command = _reconcile_apply_command(campaign, report)
    elif intentionally_stopped:
        next_command = _campaign_command(campaign, "resume")
    elif runnable:
        next_command = _campaign_command(campaign, "resume")
    elif revalidation_ready or (cleanable and not blockers):
        next_command = _reconcile_apply_command(campaign, report)
    else:
        next_command = "inspect blockers before restarting"
    if selected_state.phase is CampaignPhase.DONE:
        next_command = "campaign is DONE; inspect outputs or initialise a new campaign"
    verification = (
        snapshot.verification_payload(
            deep_required=bool(
                getattr(report, "deep_verification_required", False)
                and snapshot.verification_level != "deep"
            )
        )
        if snapshot is not None
        else {
            "level": "authority",
            "deep_required": bool(
                getattr(report, "deep_verification_required", False)
            ),
            "recursive_scan": False,
            "payload_hashing": False,
            "control_files_checked": 0,
            "files_inspected": 0,
            "payload_files_hashed": 0,
            "payload_bytes_hashed": 0,
            "elapsed_seconds": 0.0,
            "anchor_sha256": None,
            "first_invalid_version": None,
        }
    )
    return {
        "schema_version": 5,
        "campaign_dir": str(campaign),
        "proposed_state_path": (
            str(proposed_state_path) if proposed_state_path is not None else None
        ),
        "selected_phase": selected_state.phase.value,
        "selected_iteration": int(selected_state.iteration),
        "runnable": bool(runnable),
        "decision": str(getattr(report, "decision", "") or ""),
        "why_selected": list(getattr(report, "notes", []) or []),
        "why_not_runnable": why_not_runnable,
        "trusted_artifacts": [str(item) for item in getattr(report, "trusted_artifacts", [])],
        "trusted_inputs": [str(item) for item in contract_status.get("trusted_inputs", [])],
        "trusted_handoffs": [
            _format_reconcile_contract_item(campaign, item)
            for item in contract_status.get("trusted_handoffs", [])
        ],
        "protected_artifacts": [
            _format_reconcile_contract_item(campaign, item)
            for item in contract_status.get("protected_artifacts", [])
        ],
        "hard_blockers": blockers,
        "cleanable_artifacts": cleanable,
        "safe_cleanup_actions": cleanable,
        "valid_recovery_candidates": candidates,
        "partial_array_recovery": (
            compact_array_recovery_summary(getattr(report, "partial_array_recovery", None))
            if isinstance(getattr(report, "partial_array_recovery", None), dict)
            else None
        ),
        "ariadne_publication_recovery": (
            dict(getattr(report, "ariadne_publication_recovery", {}) or {})
            or None
        ),
        "ferebus_candidate_recovery": (
            dict(getattr(report, "ferebus_candidate_recovery", {}) or {})
            or None
        ),
        "aimall_quality_revalidation": (
            dict(revalidation) if isinstance(revalidation, Mapping) else None
        ),
        "active_submission_intents": list(getattr(report, "active_submission_intents", []) or []),
        "receipt_backed_intent_repairs": list(
            getattr(report, "receipt_backed_intent_repairs", []) or []
        ),
        "recommended_actions": list(getattr(report, "recommended_actions", []) or []),
        "next_command": next_command,
        "contract": contract_status,
        "runtime_status": runtime_status or {},
        "verification": verification,
    }


def _reconcile_apply_command(campaign: Path, report: Any) -> str:
    command = _campaign_command(campaign, "reconcile")
    if bool(getattr(report, "deep_verification_required", False)):
        command += " --deep-verify"
    revalidation = getattr(report, "aimall_quality_revalidation", None)
    revalidation_ready = bool(
        isinstance(revalidation, Mapping)
        and revalidation.get("state") in {"eligible", "resumable"}
        and revalidation.get("eligible") is True
    )
    if (
        ".DATA/STAGING is non-empty"
        in list(getattr(report, "unsafe_reasons", []))
        and not revalidation_ready
    ):
        command += " --archive-staging"
    return command + " --apply"


def _reconcile_human_reason(reason: Any) -> str:
    text = str(reason)
    mapping = {
        ".DATA/SCRIPTS contains sbatch scripts": (
            "stale scheduler submission scripts"
        ),
        "dangling model staging directories exist": "dangling model staging",
        "dangling reference-data staging directories exist": "dangling reference-data staging",
        ".DATA/STAGING is non-empty": ".DATA/STAGING is non-empty",
        ".DATA/STAGING is non-empty (requires --archive-staging when safe)": (
            ".DATA/STAGING is non-empty; requires --archive-staging when safe"
        ),
        "active submission intent(s) present": "active submission intent(s)",
        "stale ARIADNE publication": "stale ARIADNE batch publication",
    }
    return mapping.get(text, text)


def _reconcile_has_only_explicit_cleanup_blockers(
    report: Any,
    blockers: Sequence[str],
) -> bool:
    if not blockers:
        return False
    reasons = set(str(reason) for reason in getattr(report, "unsafe_reasons", []))
    if ".DATA/STAGING is non-empty" not in reasons:
        return False
    staging_blockers = {
        ".DATA/STAGING",
        ".DATA/STAGING is non-empty unless --archive-staging is explicitly requested",
    }
    return all(str(blocker) in staging_blockers for blocker in blockers)


def _reconcile_contract_status_label(
    state: Any,
    contract_status: Mapping[str, Any],
) -> str:
    phase = getattr(getattr(state, "phase", None), "value", getattr(state, "phase", "UNKNOWN"))
    if str(phase) == CampaignPhase.DONE.value:
        return "terminal"
    if str(phase) == CampaignPhase.HALTED.value:
        return "not runnable"
    return "ok" if bool(contract_status.get("contract_ok")) else "invalid"


def _reconcile_result_label(
    report: Any,
    contract_status: Dict[str, Any],
    *,
    apply_mode: bool = False,
) -> str:
    if apply_mode:
        return "applied"
    state = report.proposed_state
    cleanable = _reconcile_cleanable_reasons(report)
    blockers = _reconcile_hard_blockers(report, contract_status)
    phase = state.phase
    if blockers and not _reconcile_has_only_explicit_cleanup_blockers(
        report,
        blockers,
    ):
        return "blocked"
    revalidation = getattr(report, "aimall_quality_revalidation", None)
    if (
        isinstance(revalidation, Mapping)
        and revalidation.get("state") in {"eligible", "resumable"}
        and revalidation.get("eligible") is True
    ):
        return "safe to apply"
    if cleanable:
        return "safe to apply"
    if phase is CampaignPhase.HALTED:
        return "blocked"
    if phase is CampaignPhase.DONE:
        return "no recovery needed"
    if not bool(contract_status.get("contract_ok")):
        return "blocked"
    return "safe to apply"


def _is_intentionally_stopped_state(state: Any) -> bool:
    """Return whether *state* records a deliberate, resumable user stop."""
    if not bool(getattr(state, "shutdown_requested", False)):
        return False
    phase = getattr(state, "phase", None)
    if phase in {CampaignPhase.HALTED, CampaignPhase.DONE}:
        return False
    context = getattr(state, "lifecycle_context", None)
    return bool(
        isinstance(context, Mapping)
        and str(context.get("disposition") or "") == "stopped"
    )


def _stopped_boundary_description(state: Any) -> str:
    phase = getattr(getattr(state, "phase", None), "value", "unknown phase")
    iteration = int(getattr(state, "iteration", 0))
    context = getattr(state, "lifecycle_context", None)
    details = context.get("details") if isinstance(context, Mapping) else None
    if isinstance(details, Mapping):
        mode = str(details.get("mode") or "")
        target_iteration = details.get("target_iteration")
        valid_iteration = bool(
            isinstance(target_iteration, int)
            and not isinstance(target_iteration, bool)
            and target_iteration >= 0
        )
        if mode == "after_iteration" and valid_iteration:
            return (
                "paused after iteration "
                + str(target_iteration)
                + "; next phase is "
                + str(phase)
                + " iteration "
                + str(iteration)
            )
        target_phase = details.get("target_phase")
        if (
            mode == "after_phase"
            and isinstance(target_phase, str)
            and target_phase
            and valid_iteration
        ):
            return (
                "paused after phase "
                + target_phase
                + " in iteration "
                + str(target_iteration)
                + "; next phase is "
                + str(phase)
                + " iteration "
                + str(iteration)
            )
    return "paused at " + str(phase) + " iteration " + str(iteration)


def _reconcile_apply_status(
    report: Any,
    contract_status: Dict[str, Any],
) -> str:
    result = _reconcile_result_label(report, contract_status)
    if result == "safe to apply":
        return "safe"
    if result == "no recovery needed":
        return "not applicable"
    return "blocked"


def _reconcile_latest_human_reason(report: Any) -> str:
    decision = str(getattr(report, "decision", "") or "").strip()
    if not decision:
        return "-"
    if ": " in decision:
        return decision.split(": ", 1)[1]
    return decision


def _reconcile_format_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return text


def _reconcile_wrap_line(prefix: str, text: str, *, width: int = 100) -> List[str]:
    import textwrap

    if len(prefix) + len(text) <= width:
        return [prefix + text]
    return textwrap.wrap(
        text,
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    ) or [prefix]


@dataclass(frozen=True)
class _ReconcilePresentation:
    """One coherent human view of an existing reconciliation decision."""

    result: str
    campaign_state: Tuple[Tuple[str, str], ...]
    planned_changes: Tuple[Tuple[str, str], ...]
    data_safety: Tuple[Tuple[str, str], ...]
    after_apply: Tuple[Tuple[str, str], ...]
    blockers: Tuple[str, ...]
    warnings: Tuple[str, ...]
    preserved_diagnostics: Tuple[str, ...]
    next_label: str
    next_command: str
    next_effect: str


def _reconcile_plain_text(value: Any) -> str:
    """Remove implementation exception names from concise human output."""

    text = str(value or "").strip()
    failure = classify_operator_failure(text)
    if failure.family != "unknown":
        return failure.summary
    text = re.sub(
        r"(^|[;:])\s*[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):\s*",
        r"\1 ",
        text,
    ).strip()
    return " ".join(text.split()) or "the campaign evidence could not be validated"


def _reconcile_join_phrases(items: Sequence[str]) -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        return "no reconcile changes are required"
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values[0] + " and " + values[1]
    return ", ".join(values[:-1]) + ", and " + values[-1]


def _reconcile_phase_display(
    phase: Any,
    iteration: Any,
    *,
    replacement_round: int = 0,
) -> str:
    raw = getattr(phase, "value", phase)
    title = _PHASE_TITLES.get(str(raw), str(raw).replace("_", " ").title())
    if len(title) < 2 or not title[:2].isupper():
        title = title[:1].lower() + title[1:]
    try:
        number = int(iteration)
    except (TypeError, ValueError):
        return title
    text = title + ", iteration " + str(number)
    if int(replacement_round or 0) > 0:
        text += ", replacement round " + str(int(replacement_round))
    return text


def _reconcile_config_label(path: Any) -> str:
    leaf = str(path or "configuration").rsplit(".", 1)[-1]
    return leaf.replace("_", " ")


def _reconcile_config_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _reconcile_load_current_state(
    campaign: Path,
) -> Tuple[Optional[CampaignState], Optional[str]]:
    path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    if not path.is_file():
        return None, "campaign state is missing"
    try:
        return read_state(path), None
    except Exception as exc:
        return None, "campaign state is unreadable: " + _reconcile_plain_text(exc)


def _reconcile_state_summary(
    state: Optional[CampaignState],
    error: Optional[str],
) -> str:
    if state is None:
        return str(error or "campaign state is unavailable")
    if _is_intentionally_stopped_state(state):
        context = state.lifecycle_context if isinstance(state.lifecycle_context, Mapping) else {}
        details = context.get("details") if isinstance(context, Mapping) else None
        if isinstance(details, Mapping):
            target_iteration = details.get("target_iteration")
            if str(details.get("mode") or "") == "after_iteration" and isinstance(
                target_iteration, int
            ):
                return "paused after iteration " + str(target_iteration)
            target_phase = details.get("target_phase")
            if str(details.get("mode") or "") == "after_phase" and target_phase:
                return (
                    "paused after "
                    + _reconcile_phase_display(target_phase, target_iteration)
                )
        return "paused at " + _reconcile_phase_display(
            state.phase,
            state.iteration,
            replacement_round=int(state.replacement_round),
        )
    if state.phase is CampaignPhase.DONE:
        return "complete after iteration " + str(int(state.iteration))
    if state.phase is CampaignPhase.HALTED:
        return "halted at iteration " + str(int(state.iteration))
    return _reconcile_phase_display(
        state.phase,
        state.iteration,
        replacement_round=int(state.replacement_round),
    )


def _reconcile_state_differs(
    current: Optional[CampaignState],
    proposed: CampaignState,
) -> bool:
    if current is None:
        return True
    return asdict(current) != asdict(proposed)


def _reconcile_display_target(report: Any) -> Tuple[Any, int, int]:
    proposed = report.proposed_state
    if proposed.phase is CampaignPhase.HALTED:
        candidates = list(getattr(report, "recovery_candidates", []) or [])
        if candidates and isinstance(candidates[0], Mapping):
            candidate = candidates[0]
            try:
                return (
                    CampaignPhase(str(candidate.get("phase"))),
                    int(candidate.get("iteration")),
                    int(candidate.get("replacement_round") or 0),
                )
            except (TypeError, ValueError):
                pass
    return proposed.phase, int(proposed.iteration), int(proposed.replacement_round)


def _reconcile_active_work(
    current: Optional[CampaignState],
    report: Any,
    runtime_status: Optional[Mapping[str, Any]],
) -> str:
    runtime_blockers = list(
        (runtime_status or {}).get("reconcile_apply_blockers") or []
    )
    if runtime_blockers:
        return "campaign ownership is active or cannot be confirmed safely"
    intents = list(getattr(report, "active_submission_intents", []) or [])
    scheduler = _reconcile_campaign_assessment(current, report).scheduler
    scheduler_jobs = {
        job_id for _phase, job_id in scheduler.pending_jobs
    }
    scheduler_jobs.update(
        str(item.get("job_id"))
        for item in scheduler.scheduler_intents
        if item.get("job_id")
    )
    if scheduler_jobs:
        count = len(scheduler_jobs)
        return (
            str(count)
            + " recorded "
            + _scheduler_display_name(intents=intents)
            + " job"
            + ("" if count == 1 else "s")
        )
    local = [
        item
        for item in intents
        if isinstance(item, Mapping) and not item.get("job_id")
    ]
    if local:
        return "local recovery work is prepared"
    return "none"


def _reconcile_cleanup_rows(
    report: Any,
) -> Tuple[List[Tuple[str, str]], List[str], List[str], List[str]]:
    rows: List[Tuple[str, str]] = []
    reasons: List[str] = []
    warnings: List[str] = []
    preserved: List[str] = []
    raw_reasons = set(str(item) for item in getattr(report, "unsafe_reasons", []) or [])
    retirement = getattr(report, "completed_staging_retirement", None)
    if isinstance(retirement, Mapping):
        for record in retirement.get("eligible") or []:
            iteration = int(record.get("iteration") or 0)
            context = "bootstrap" if str(record.get("context")) == "bootstrap" else "iteration-" + str(iteration)
            if str(record.get("action")) == "delete":
                rows.append(
                    (
                        "staging",
                        "delete temporary "
                        + context
                        + " data already committed to QM reference data",
                    )
                )
            else:
                rows.append(
                    (
                        "staging",
                        "move " + context + " diagnostic data outside active staging",
                    )
                )
                preserved.append(
                    context + " rejected or unrecognised diagnostic data will be retained"
                )
        if retirement.get("eligible"):
            reasons.append("completed temporary staging remains")
        if retirement.get("pending_tombstones"):
            rows.append(("staging", "finish an interrupted temporary-data deletion"))
            reasons.append("an earlier staging deletion is incomplete")
        warnings.extend(str(item) for item in retirement.get("warnings") or [])
    if "dangling model staging directories exist" in raw_reasons:
        rows.append(("model staging", "remove unfinished temporary model output"))
        reasons.append("unfinished temporary model output remains")
    if "dangling reference-data staging directories exist" in raw_reasons:
        rows.append(("QM staging", "archive unfinished temporary QM publication data"))
        reasons.append("unfinished QM publication data remains")
    if ".DATA/SCRIPTS contains sbatch scripts" in raw_reasons:
        rows.append(
            (
                "submission scripts",
                "archive stale temporary "
                + _scheduler_display_name()
                + " scripts",
            )
        )
        reasons.append("stale submission scripts remain")
    if ".DATA/STAGING is non-empty" in raw_reasons:
        rows.append(("staging", "archive unclassified temporary staging for review"))
        reasons.append("unclassified temporary staging remains")
    if "stale ARIADNE publication" in raw_reasons:
        rows.append(("ARIADNE publication", "archive stale derived batch files"))
        reasons.append("a stale ARIADNE publication remains")
    return rows, reasons, warnings, preserved


def _reconcile_environment_launch_assessment(
    campaign: Path,
    state: Optional[CampaignState],
    *,
    scheduler_ownership_clear: bool,
) -> Optional[Dict[str, Any]]:
    """Use the shared launch classifier for a proposed reconcile state."""
    if state is None:
        return None
    if not (campaign / "campaign.yaml").is_file():
        return None
    try:
        from .execution_identity import inspect_environment_generation_launch

        config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        return inspect_environment_generation_launch(
            campaign,
            state=state,
            config=config,
            scheduler_ownership_clear=scheduler_ownership_clear,
        )
    except Exception as exc:
        return {
            "disposition": "invalid",
            "launchable": False,
            "generation": -1,
            "reason": "execution environment evidence is invalid: " + str(exc),
            "error": type(exc).__name__ + ": " + str(exc),
        }


def _reconcile_environment_config_binding_repair(
    campaign: Path,
    state: Optional[CampaignState],
    *,
    scheduler_ownership_clear: bool = True,
) -> Optional[Dict[str, Any]]:
    """Compatibility hook for the former private presentation helper."""
    return _reconcile_environment_launch_assessment(
        campaign,
        state,
        scheduler_ownership_clear=scheduler_ownership_clear,
    )


def _reconcile_presentation(
    campaign: Path,
    report: Any,
    contract_status: Dict[str, Any],
    *,
    config_review: Any = None,
    runtime_status: Optional[Dict[str, Any]] = None,
) -> _ReconcilePresentation:
    current, current_error = _reconcile_load_current_state(campaign)
    active_intents = list(
        getattr(report, "active_submission_intents", []) or []
    )
    scheduler_name = _scheduler_display_name(intents=active_intents)
    proposed = report.proposed_state
    allocation_transition: Optional[Dict[str, Any]] = None
    diversity_transition: Optional[Dict[str, Any]] = None
    if proposed.phase in {
        CampaignPhase.INITIAL_ALLOCATION_CHECK,
        CampaignPhase.ALLOCATION_CHECK,
    }:
        try:
            from .execution_identity import (
                inspect_allocation_check_transition_boundary,
            )

            allocation_transition = inspect_allocation_check_transition_boundary(
                campaign,
                proposed,
            )
        except Exception as exc:
            allocation_transition = {
                "safe": False,
                "reason": type(exc).__name__ + ": " + str(exc),
            }
    if proposed.phase in {
        CampaignPhase.PHASE_A_DIVERSITY,
        CampaignPhase.PHASE_B_DIVERSITY,
    }:
        try:
            from .execution_identity import (
                inspect_scalar_diversity_transition_boundary,
            )

            diversity_transition = inspect_scalar_diversity_transition_boundary(
                campaign,
                proposed,
            )
        except Exception as exc:
            diversity_transition = {
                "safe": False,
                "reason": type(exc).__name__ + ": " + str(exc),
            }
    target_phase, target_iteration, target_round = _reconcile_display_target(report)
    raw_blockers = _reconcile_hard_blockers(report, contract_status)
    explicit_staging_cleanup = _reconcile_has_only_explicit_cleanup_blockers(
        report,
        raw_blockers,
    )
    blockers = [] if explicit_staging_cleanup else list(raw_blockers)
    blockers.extend(
        str(item)
        for item in (runtime_status or {}).get("reconcile_apply_blockers", []) or []
    )
    if (
        isinstance(allocation_transition, Mapping)
        and not bool(allocation_transition.get("safe", False))
    ):
        blockers.append(
            "allocation-check recovery is unsafe: "
            + str(allocation_transition.get("reason") or "unknown allocation evidence")
        )
    failed_diversity_transition = bool(
        str((runtime_status or {}).get("background_startup_state") or "")
        == "failed"
        and str((runtime_status or {}).get("background_startup_stage") or "")
        == "environment_transition"
        and isinstance(diversity_transition, Mapping)
        and not bool(diversity_transition.get("safe", False))
    )
    if failed_diversity_transition:
        blockers.append(
            "scalar diversity environment transition is unsafe: "
            + str(
                diversity_transition.get("reason")
                or "terminal retry evidence is incomplete"
            )
        )
    blocked_changes = list(getattr(config_review, "blocked_changes", []) or [])
    blockers.extend(
        "configuration change is locked: " + str(change.path)
        for change in blocked_changes
    )
    snapshot = getattr(report, "artifact_snapshot", None)
    deep_pending = bool(
        getattr(report, "deep_verification_required", False)
        and (
            snapshot is None
            or str(getattr(snapshot, "verification_level", "authority")) != "deep"
        )
    )
    if deep_pending:
        blockers.append("deep verification is required before authority can be recovered")
    blockers = list(dict.fromkeys(_reconcile_plain_text(item) for item in blockers))

    cleanup_rows, reason_parts, warnings, preserved = _reconcile_cleanup_rows(report)
    planned: List[Tuple[str, str]] = list(cleanup_rows)
    transaction_recovery = getattr(
        report,
        "reconcile_transaction_recovery",
        None,
    )
    if isinstance(transaction_recovery, Mapping):
        if bool(transaction_recovery.get("recoverable", False)):
            disposition = str(transaction_recovery.get("disposition") or "")
            if disposition == "abandoned_before_mutation":
                description = "close a previous reconcile that stopped before changing campaign data"
            elif disposition == "completed_safe_cleanup":
                description = "finish bookkeeping for interrupted temporary-data cleanup"
            elif disposition == "adopted_committed_state":
                description = "confirm campaign changes that were already published"
            elif disposition == "rolled_forward":
                description = "complete partially published campaign bookkeeping"
            else:
                description = _reconcile_plain_text(
                    transaction_recovery.get("reason")
                )
            planned.insert(0, ("interrupted reconcile", description))
            backup_status = str(
                transaction_recovery.get("state_backup_status") or ""
            )
            action = str(transaction_recovery.get("action") or "")
            if backup_status == "repairable_partial":
                planned.insert(
                    1,
                    (
                        "state backup",
                        "replace the interrupted partial backup from the "
                        "transaction's authenticated before-state",
                    ),
                )
            elif backup_status == "absent" and action in {
                "adopt",
                "roll_forward",
            }:
                planned.insert(
                    1,
                    (
                        "state backup",
                        "create the missing authenticated backup from the "
                        "transaction's recorded before-state",
                    ),
                )
            reason_parts.insert(0, "a previous reconcile was interrupted")
        elif str(transaction_recovery.get("state") or "") == "blocked":
            blocker = _reconcile_plain_text(transaction_recovery.get("reason"))
            if blocker not in blockers:
                blockers.insert(0, blocker)
        elif str(transaction_recovery.get("state") or "") == "recovered":
            disposition = str(transaction_recovery.get("disposition") or "")
            description = {
                "abandoned_before_mutation": "closed an interrupted reconcile that made no campaign changes",
                "completed_safe_cleanup": "completed bookkeeping for interrupted temporary-data cleanup",
                "adopted_committed_state": "adopted campaign changes that were already fully published",
                "rolled_back_exactly": "restored the exact pre-reconcile authority",
                "rolled_forward": "completed partially published campaign bookkeeping",
            }.get(
                disposition,
                _reconcile_plain_text(transaction_recovery.get("reason")),
            )
            planned.insert(0, ("interrupted reconcile", description))
            backup_status = str(
                transaction_recovery.get("state_backup_status") or ""
            )
            if backup_status == "repairable_partial":
                planned.insert(
                    1,
                    (
                        "state backup",
                        "replaced the interrupted partial backup from "
                        "authenticated transaction evidence",
                    ),
                )
            elif backup_status == "absent":
                planned.insert(
                    1,
                    (
                        "state backup",
                        "created the missing authenticated state backup",
                    ),
                )
            reason_parts.insert(0, "an interrupted reconcile was recovered")
    allowed_changes = list(getattr(config_review, "allowed_changes", []) or [])
    grouped_config_changes: Dict[str, List[str]] = {}
    for change in allowed_changes:
        description = (
            _reconcile_config_label(change.path)
            + " "
            + _reconcile_config_value(change.old)
            + " -> "
            + _reconcile_config_value(change.new)
        )
        grouped_config_changes.setdefault(description, []).append(
            str(change.path)
        )
    for description, paths in grouped_config_changes.items():
        if len(paths) > 1:
            description += (
                " ("
                + str(len(paths))
                + " affected settings)"
            )
        planned.append(("configuration", description))
    if allowed_changes:
        reason_parts.insert(0, "configuration changed")
        planned.append(
            (
                "config takes effect",
                _reconcile_phase_display(
                    target_phase,
                    target_iteration,
                    replacement_round=target_round,
                ),
            )
        )
        if any(
            str(change.path).startswith("resources.")
            for change in allowed_changes
        ):
            planned.append(
                (
                    "retry resources",
                    "approved resource changes apply only to the retry "
                    "attempt; reused outputs keep their original producer "
                    "record",
                )
            )

    scheduler_assessment = _reconcile_campaign_assessment(
        current,
        report,
    ).scheduler
    environment_assessment = _reconcile_environment_config_binding_repair(
        campaign,
        proposed,
        scheduler_ownership_clear=(
            not (runtime_status or {}).get("reconcile_apply_blockers")
            and not scheduler_assessment.has_unresolved_scheduler_work
        ),
    )
    environment_disposition = str(
        (environment_assessment or {}).get("disposition")
        or (
            "reconcile_required"
            if (environment_assessment or {}).get("kind")
            else "current"
        )
    )
    environment_reason = _reconcile_plain_text(
        (environment_assessment or {}).get("reason") or ""
    )
    if environment_disposition in {"invalid", "ownership_blocked"}:
        if environment_reason not in blockers:
            blockers.append(environment_reason)
    elif environment_disposition == "reconcile_required":
        if planned or _reconcile_state_differs(current, proposed):
            planned.append(
                (
                    "environment configuration",
                    "advance the environment generation so it is bound to the "
                    "current campaign configuration after the listed recovery "
                    "makes the transition boundary safe",
                )
            )
            reason_parts.append("environment recovery is required before launch")
        else:
            blockers.append(environment_reason)

    scheduler_cancellation = [
        dict(item)
        for item in (
            getattr(report, "scheduler_cancellation_recovery", []) or []
        )
        if isinstance(item, Mapping)
    ]
    if scheduler_cancellation:
        legacy_scheduler_recovery = any(
            item.get("terminal_receipt")
            and not Path(str(item["terminal_receipt"])).name.endswith(
                "-v2.json"
            )
            for item in scheduler_cancellation
        )
        completed = sum(
            int(item.get("n_completed") or 0)
            for item in scheduler_cancellation
        )
        retry = sum(
            int(item.get("n_retry") or 0)
            for item in scheduler_cancellation
        )
        if legacy_scheduler_recovery:
            retry += completed
            planned.append(
                (
                    "scheduler recovery",
                    "retain the older terminal records for task accounting, "
                    "but retry all "
                    + str(retry)
                    + " affected task"
                    + ("" if retry == 1 else "s")
                    + "; their output cannot be trusted for reuse",
                )
            )
            reason_parts.append(
                "older scheduler recovery evidence requires conservative retry"
            )
        else:
            planned.append(
                (
                    "scheduler recovery",
                    "preserve "
                    + str(completed)
                    + " scheduler-completed task"
                    + ("" if completed == 1 else "s")
                    + " as candidate"
                    + ("" if completed == 1 else "s")
                    + " for local output validation after resume; "
                    + str(retry)
                    + " unfinished task"
                    + ("" if retry == 1 else "s")
                    + " will be retried",
                )
            )
            reason_parts.append(
                "cancelled scheduler work has exact recoverable task outcomes"
            )

    revalidation = getattr(report, "aimall_quality_revalidation", None)
    if isinstance(revalidation, Mapping) and revalidation.get("state") in {
        "eligible",
        "resumable",
    } and revalidation.get("eligible") is True:
        count = int(revalidation.get("candidate_count") or 0)
        planned.append(
            ("AIMAll results", "revalidate " + str(count) + " existing point" + ("" if count == 1 else "s"))
        )
        reason_parts.append("existing AIMAll results can be corrected without rerunning them")

    aimall_postprocess = getattr(
        report,
        "aimall_postprocess_recovery",
        None,
    )
    if isinstance(aimall_postprocess, Mapping) and aimall_postprocess:
        completed = int(aimall_postprocess.get("logical_total") or 0)
        planned.append(
            (
                "AIMAll results",
                "preserve "
                + str(completed)
                + " scheduler-completed output candidate"
                + ("" if completed == 1 else "s")
                + " for local validation after resume; only structurally "
                "invalid or unfinished tasks may be retried",
            )
        )
        reason_parts.append(
            "scheduler-completed AIMAll output candidates are awaiting local "
            "validation"
        )

    ariadne_terminal = getattr(
        report,
        "ariadne_terminal_postprocess_recovery",
        None,
    )
    if isinstance(ariadne_terminal, Mapping) and ariadne_terminal:
        completed = int(
            ariadne_terminal.get("n_scheduler_completed") or 0
        )
        failed = int(ariadne_terminal.get("n_scheduler_failed") or 0)
        planned.append(
            (
                "ARIADNE recovery",
                "preserve "
                + str(completed)
                + " scheduler-completed output candidate"
                + ("" if completed == 1 else "s")
                + " and "
                + str(failed)
                + " scheduler-failed slot"
                + ("" if failed == 1 else "s")
                + " for local validation; submit zero ARIADNE tasks",
            )
        )
        reason_parts.append(
            "terminal ARIADNE outputs are awaiting scheduler-free local "
            "validation"
        )

    partial = getattr(report, "partial_array_recovery", None)
    if (
        isinstance(partial, Mapping)
        and partial
        and not isinstance(aimall_postprocess, Mapping)
        and not scheduler_cancellation
    ):
        reuse = int(partial.get("n_reuse") or partial.get("n_complete") or 0)
        retry = int(partial.get("n_retry") or 0)
        if str(partial.get("phase") or "") in {
            CampaignPhase.PHASE_A_DIVERSITY.value,
            CampaignPhase.PHASE_B_DIVERSITY.value,
        }:
            if (
                str(partial.get("publication_disposition") or "")
                == "archive_and_retry"
            ):
                planned.append(
                    (
                        "diversity recovery",
                        "archive the incomplete publication and prepare one "
                        "scalar diversity task for retry after resume",
                    )
                )
                reason_parts.append(
                    "an incomplete diversity publication requires a guarded retry"
                )
            else:
                selected = int(partial.get("selected_count") or 0)
                planned.append(
                    (
                        "diversity recovery",
                        "validate "
                        + str(selected)
                        + " existing selected geometr"
                        + ("y" if selected == 1 else "ies")
                        + " locally; submit zero diversity tasks",
                    )
                )
                reason_parts.append(
                    "a completed diversity publication awaits local validation"
                )
        else:
            planned.append(
                (
                    "array recovery",
                    "reuse " + str(reuse) + " completed task" + ("" if reuse == 1 else "s")
                    + " and prepare " + str(retry) + " for retry",
                )
            )
            reason_parts.append("an interrupted array has recoverable task results")

    ariadne = getattr(report, "ariadne_results_recovery", None)
    if isinstance(ariadne, Mapping) and ariadne:
        accepted = int(ariadne.get("accepted_tasks") or 0)
        rejected = int(ariadne.get("rejected_tasks") or 0)
        planned.append(
            (
                "ARIADNE results",
                "reuse " + str(accepted) + " accepted result" + ("" if accepted == 1 else "s")
                + "; exclude " + str(rejected) + " rejected task" + ("" if rejected == 1 else "s")
                + (
                    "; resubmit no ARIADNE tasks"
                    if int(ariadne.get("tasks_resubmitted") or 0) == 0
                    else "; prepare "
                    + str(int(ariadne.get("tasks_resubmitted") or 0))
                    + " ARIADNE tasks for retry"
                ),
            )
        )

    repairs = list(getattr(report, "receipt_backed_intent_repairs", []) or [])
    scheduler_recovery_keys = {
        (
            str(item.get("phase") or ""),
            int(item.get("iteration") or 0),
            str(item.get("job_id") or ""),
        )
        for item in scheduler_cancellation
    }
    ariadne_terminal_keys = (
        {
            (
                CampaignPhase.ARIADNE_ARRAY.value,
                int(ariadne_terminal.get("iteration") or 0),
                str(ariadne_terminal.get("producer_job_id") or ""),
            )
        }
        if isinstance(ariadne_terminal, Mapping)
        else set()
    )
    ordinary_repairs = [
        item
        for item in repairs
        if (
            str(item.get("phase") or ""),
            int(item.get("iteration") or 0),
            str(item.get("job_id") or ""),
        )
        not in scheduler_recovery_keys
        and (
            str(item.get("phase") or ""),
            int(item.get("iteration") or 0),
            str(item.get("job_id") or ""),
        )
        not in ariadne_terminal_keys
    ]
    if ordinary_repairs:
        planned.append(
            (
                "submission records",
                "retire "
                + str(len(ordinary_repairs))
                + " completed local record"
                + ("" if len(ordinary_repairs) == 1 else "s"),
            )
        )
        reason_parts.append("completed submission bookkeeping needs to be retired")

    candidate = getattr(report, "ferebus_candidate_recovery", None)
    if isinstance(candidate, Mapping) and candidate:
        if (
            candidate.get("candidate_kind")
            == "quality_rejected_relative_regression"
        ):
            planned.append(
                (
                    "FEREBUS",
                    "reuse the fully measured model candidate; record its "
                    "incumbent-relative RMSE regression as an advisory warning",
                )
            )
            reason_parts.append(
                "a fully measured FEREBUS candidate was rejected only by the old relative-regression policy"
            )
        else:
            planned.append(
                (
                    "FEREBUS",
                    "prepare the validated existing model candidate for recovery",
                )
            )
            reason_parts.append("a completed FEREBUS candidate can be recovered")

    ferebus_staging = getattr(report, "ferebus_staging_recovery", None)
    if isinstance(ferebus_staging, Mapping) and ferebus_staging.get(
        "source_submission_identities"
    ):
        disposition = str(ferebus_staging.get("disposition") or "")
        completed = int(
            ferebus_staging.get("scheduler_completed_candidates") or 0
        )
        retry = int(ferebus_staging.get("known_retry_candidates") or 0)
        if disposition == "archived_terminal_producer":
            planned.append(
                (
                    "FEREBUS staging",
                    "restore the authenticated terminal producer from reconcile "
                    "transaction "
                    + str(ferebus_staging.get("source_transaction_id") or ""),
                )
            )
            reason_parts.append(
                "interrupted FEREBUS producer staging requires restoration"
            )
        planned.append(
            (
                "FEREBUS recovery",
                "preserve "
                + str(completed)
                + " scheduler-completed task candidate"
                + ("" if completed == 1 else "s")
                + " for local validation; "
                + str(retry)
                + " known unfinished task"
                + ("" if retry == 1 else "s")
                + " will be retried",
            )
        )

    state_differs = _reconcile_state_differs(current, proposed)
    if (
        state_differs
        and proposed.phase not in {CampaignPhase.HALTED, CampaignPhase.DONE}
    ) or (
        proposed.phase is CampaignPhase.HALTED
        and target_phase is not CampaignPhase.HALTED
    ):
        planned.insert(
            0,
            (
                "campaign state",
                "recover to "
                + _reconcile_phase_display(
                    target_phase,
                    target_iteration,
                    replacement_round=target_round,
                ),
            ),
        )
        reason_parts.append("campaign state needs recovery")

    scheduler_resume_only = bool(
        scheduler_cancellation
        and not blockers
        and current is not None
        and current.phase is proposed.phase
        and int(current.iteration) == int(proposed.iteration)
        and int(current.replacement_round) == int(proposed.replacement_round)
        and not cleanup_rows
        and not allowed_changes
        and not blocked_changes
        and not isinstance(transaction_recovery, Mapping)
        and not (
            isinstance(revalidation, Mapping)
            and revalidation.get("eligible") is True
        )
        and not isinstance(aimall_postprocess, Mapping)
        and not isinstance(ariadne, Mapping)
        and not ordinary_repairs
        and not isinstance(candidate, Mapping)
        and environment_disposition in {
            "current",
            "unbound_first_start",
            "rebindable_on_resume",
        }
    )
    has_changes = bool(planned)
    if blockers:
        result = "blocked"
        reason = blockers[0]
    elif proposed.phase is CampaignPhase.DONE:
        result = "no recovery needed"
        reason = "the completed campaign and its committed data are coherent"
    elif scheduler_resume_only:
        result = "no reconcile changes needed"
        reason = (
            "terminal scheduler evidence is coherent and resume can validate "
            "completed outputs and prepare retry work directly"
        )
    elif has_changes:
        result = "ready to apply"
        reason = _reconcile_join_phrases(reason_parts)
    elif proposed.phase is CampaignPhase.HALTED:
        result = "blocked"
        reason = _reconcile_plain_text(_reconcile_latest_human_reason(report))
    else:
        result = "no reconcile changes needed"
        if (
            isinstance(allocation_transition, Mapping)
            and str(allocation_transition.get("replacement_sample_state") or "")
            in {"missing_rebuildable", "partial_rebuildable"}
        ):
            reason = (
                "campaign authority is coherent; the daemon can rebuild the "
                "missing replacement sample when it resumes"
            )
        else:
            reason = (
                "the deliberate pause remains valid"
                if _is_intentionally_stopped_state(proposed)
                else "campaign state, configuration and authority records already agree"
            )

    state_rows: List[Tuple[str, str]] = [
        ("status", _reconcile_state_summary(current, current_error)),
    ]
    if result == "ready to apply" or (
        result == "blocked" and proposed.phase is not CampaignPhase.HALTED
    ):
        state_rows.append(
            (
                "continue from" if _is_intentionally_stopped_state(proposed) else "recovery target",
                _reconcile_phase_display(
                    target_phase,
                    target_iteration,
                    replacement_round=target_round,
                ),
            )
        )
    state_rows.extend(
        [
            ("reason", reason),
            ("active work", _reconcile_active_work(current, report, runtime_status)),
        ]
    )
    retained_stop = str(
        (runtime_status or {}).get("_retained_stop_description") or ""
    )
    if retained_stop:
        state_rows.append(
            ("stop request", retained_stop + "; remains active")
        )

    if result == "ready to apply":
        planned.append(
            (
                scheduler_name + " work",
                "none; reconcile will not start jobs",
            )
        )

    protected = list(contract_status.get("protected_artifacts", []) or [])
    safety = [
        ("committed QM data", "unchanged"),
        ("committed models", "unchanged"),
        (
            "protected evidence affected",
            "none" if not protected else "none; listed evidence remains protected",
        ),
    ]
    after: List[Tuple[str, str]] = []
    if result == "ready to apply":
        after = [
            (
                "campaign state",
                _reconcile_phase_display(
                    target_phase,
                    target_iteration,
                    replacement_round=target_round,
                ),
            ),
            ("daemon", "remains stopped"),
            ("automatic campaign work", "none"),
        ]
        if retained_stop:
            after.append(
                ("stop request", retained_stop + "; remains active")
            )
        if environment_disposition in {
            "rebindable_on_resume",
            "reconcile_required",
        }:
            after.append(
                (
                    "software environment",
                    "resume will record a correctly bound generation before "
                    "scientific work begins",
                )
            )

    if result == "ready to apply":
        next_label = "run"
        next_command = _reconcile_apply_command(campaign, report)
        next_effect = (
            "apply the listed changes; no daemon or "
            + scheduler_name
            + " job will start"
        )
    elif result == "no reconcile changes needed":
        next_label = "run"
        next_command = _campaign_command(campaign, "resume")
        if scheduler_resume_only:
            completed = sum(
                int(item.get("n_completed") or 0)
                for item in scheduler_cancellation
            )
            retry = sum(
                int(item.get("n_retry") or 0)
                for item in scheduler_cancellation
            )
            next_effect = (
                "validate "
                + str(completed)
                + " scheduler-completed output"
                + ("" if completed == 1 else "s")
                + " locally and retry "
                + str(retry)
                + " unfinished task"
                + ("" if retry == 1 else "s")
            )
        elif (
            isinstance(allocation_transition, Mapping)
            and str(allocation_transition.get("replacement_sample_state") or "")
            in {"missing_rebuildable", "partial_rebuildable"}
        ):
            pending = int(allocation_transition.get("pending_tasks") or 0)
            next_effect = (
                "advance the software environment, rebuild the missing replacement "
                "sample and continue with "
                + str(pending)
                + " pending replacement task"
                + ("" if pending == 1 else "s")
            )
        else:
            next_effect = "continue the campaign from " + _reconcile_phase_display(
                target_phase,
                target_iteration,
                replacement_round=target_round,
            )
            if environment_disposition == "rebindable_on_resume":
                next_effect += (
                    "; startup first records a correctly bound environment "
                    "generation"
                )
    elif result == "no recovery needed":
        next_label = "review"
        next_command = _campaign_command(campaign, "status")
        next_effect = "review the completed campaign and its final data products"
    elif deep_pending:
        next_label = "run"
        next_command = _campaign_command(campaign, "reconcile", " --deep-verify")
        next_effect = "perform the required scientific-payload verification before reviewing apply"
    elif blocked_changes:
        next_label = "run"
        next_command = _campaign_command(campaign, "config-check", " --human")
        next_effect = "review the configuration changes that cannot be applied here"
    elif failed_diversity_transition:
        next_label = "run"
        next_command = _campaign_command(campaign, "reconcile", " --verbose")
        next_effect = (
            "review the scalar diversity transition evidence; resume would "
            "repeat the same refusal"
        )
    elif (runtime_status or {}).get("reconcile_apply_blockers"):
        next_label = "run"
        next_command = _campaign_command(campaign, "status")
        next_effect = "review active ownership before attempting recovery"
    else:
        next_label = "run"
        next_command = _campaign_command(campaign, "reconcile", " --verbose")
        next_effect = "review the technical evidence behind the blocker"

    return _ReconcilePresentation(
        result=result,
        campaign_state=tuple(state_rows),
        planned_changes=tuple(planned),
        data_safety=tuple(safety),
        after_apply=tuple(after),
        blockers=tuple(blockers),
        warnings=tuple(dict.fromkeys(_reconcile_plain_text(item) for item in warnings)),
        preserved_diagnostics=tuple(dict.fromkeys(preserved)),
        next_label=next_label,
        next_command=next_command,
        next_effect=next_effect,
    )


def _print_reconcile_key_values(items: Sequence[Tuple[str, Any]]) -> None:
    width = max((len(label) for label, _value in items), default=0)
    for label, value in items:
        print("  " + label.ljust(width) + ": " + str(value))


def _print_reconcile_list(label: str, items: Sequence[Any], *, indent: str = "  ") -> None:
    print(indent + label + ":")
    if not items:
        print(indent + "  - none")
        return
    for item in items:
        print(indent + "  - " + str(item))


def _reconcile_read_current_state_summary(
    campaign: Path,
    report: Any,
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    if not state_path.is_file():
        return {"state": "missing"}
    try:
        state = read_state(state_path)
    except Exception as exc:
        uid_status = "salvaged" if any(
            "salvaged campaign_uid" in str(note)
            for note in getattr(report, "notes", [])
        ) else "unknown"
        return {
            "state": "invalid",
            "reason": type(exc).__name__ + ": " + str(exc),
            "campaign_uid": uid_status,
        }
    active_intents = list(getattr(report, "active_submission_intents", []) or [])
    scheduler_intents = [
        intent
        for intent in active_intents
        if isinstance(intent, dict) and intent.get("job_id")
    ]
    local_intents = [
        intent
        for intent in active_intents
        if isinstance(intent, dict) and not intent.get("job_id")
    ]
    pending_jobs = getattr(state, "pending_jobs", {}) or {}
    pending_count = sum(1 for job_id in pending_jobs.values() if job_id)
    completed_markers = sum(1 for job_id in pending_jobs.values() if not job_id)
    scheduler_name = _scheduler_display_name(intents=active_intents)
    if scheduler_intents:
        job_text = (
            str(len(scheduler_intents))
            + " recorded "
            + scheduler_name
            + " job(s)"
        )
    elif pending_count:
        job_text = (
            str(pending_count) + " recorded " + scheduler_name + " job(s)"
        )
    elif local_intents:
        job_text = "none; local recovery work is prepared"
    else:
        job_text = "none"
    summary: Dict[str, Any] = {
        "state": state.phase.value + " at iteration " + str(int(state.iteration)),
        "recorded jobs": job_text,
    }
    if verbose:
        summary["versions"] = (
            "reference_data="
            + str(int(state.reference_data_version))
            + (
                " valid"
                if int(state.reference_data_version) in set(getattr(report, "valid_reference_data_versions", []))
                else ""
            )
            + ", models="
            + (
                "none"
                if int(state.models_version) < 0
                else str(int(state.models_version))
                + (
                    " valid"
                    if int(state.models_version) in set(getattr(report, "valid_model_versions", []))
                    else ""
                )
            )
        )
        summary["shutdown requested"] = "yes" if bool(state.shutdown_requested) else "no"
        summary["campaign uid"] = str(state.campaign_uid)
        if completed_markers:
            summary["completed job markers"] = completed_markers
    return summary


def _reconcile_bootstrap_summary(campaign: Path, report: Any) -> str:
    handoff = getattr(report, "bootstrap_handoff", None)
    if not isinstance(handoff, dict):
        return "none"
    phase = str(handoff.get("phase") or "UNKNOWN")
    accepted = handoff.get("accepted_count")
    total = handoff.get("n_total")
    prefix = "archived " if bool(handoff.get("archived")) else ""
    suffix = ""
    if accepted is not None and total is not None:
        suffix = " accepted " + str(accepted) + "/" + str(total)
    path = _reconcile_relative_path(campaign, handoff.get("path"))
    if path:
        suffix += " at " + path
    return prefix + phase + suffix


def _reconcile_phase_a_summary(campaign: Path, report: Any) -> str:
    handoff = getattr(report, "phase_a_handoff", None)
    if not isinstance(handoff, dict):
        return "none"
    path = _reconcile_relative_path(campaign, handoff.get("path"))
    if path:
        return "available at " + path
    return "available"


def _reconcile_staging_summary(
    campaign: Path,
    contract_status: Dict[str, Any],
    report: Any,
) -> str:
    protected = [
        _format_reconcile_contract_item(campaign, item)
        for item in contract_status.get("protected_artifacts", [])
    ]
    if protected:
        return "protected: " + "; ".join(protected)
    if ".DATA/STAGING is non-empty" in list(getattr(report, "unsafe_reasons", [])):
        return "non-empty, user review required"
    retirement = getattr(report, "completed_staging_retirement", None)
    if isinstance(retirement, Mapping) and retirement.get("eligible"):
        return (
            str(len(retirement["eligible"]))
            + " completed bucket(s), routine retirement on apply"
        )
    if isinstance(retirement, Mapping) and retirement.get("pending_tombstones"):
        return "interrupted completed-staging deletion, retry on apply"
    return "none"


def _reconcile_scripts_summary(report: Any) -> str:
    inv = getattr(report, "script_inventory", {}) or {}
    if not isinstance(inv, dict) or not inv.get("exists"):
        return "none"
    count = int(inv.get("count") or 0)
    if ".DATA/SCRIPTS contains sbatch scripts" in list(getattr(report, "unsafe_reasons", [])):
        return str(count) + " stale file" + ("" if count == 1 else "s") + ", cleanable"
    return str(count) + " file" + ("" if count == 1 else "s")


def _reconcile_model_staging_summary(report: Any) -> str:
    if "dangling model staging directories exist" in list(getattr(report, "unsafe_reasons", [])):
        return "dangling, cleanable"
    return "none"


def _reconcile_reference_data_staging_summary(report: Any) -> str:
    if "dangling reference-data staging directories exist" in list(getattr(report, "unsafe_reasons", [])):
        return "dangling, cleanable"
    return "none"


def _reconcile_reference_commit_summary(report: Any) -> str:
    records = getattr(report, "reference_commit_transactions", None)
    if not isinstance(records, list) or not records:
        return "none"
    active = [
        record
        for record in records
        if str(record.get("state") or "") != "complete"
    ]
    selected = active[0] if active else records[-1]
    state = str(selected.get("state") or "unknown")
    ledger = selected.get("ledger")
    if not isinstance(ledger, dict):
        reason = str(selected.get("reason") or "")
        return state + ((": " + reason) if reason else "")
    return (
        state
        + ", moved "
        + str(ledger.get("moved_points", 0))
        + "/"
        + str(len(ledger.get("point_bindings") or []))
        + ", bytes "
        + str(ledger.get("moved_bytes", 0))
        + ", shard repairs "
        + str(ledger.get("shards_repaired", 0))
    )


def _print_reconcile_header(campaign: Path, *, mode: str, result: str) -> None:
    print("ICHOR Reconcile")
    print("Campaign: " + str(campaign))
    if mode.endswith("-apply") or mode == "apply":
        print("Apply result: " + str(result).lower())
    elif mode == "restore-config":
        print("Proposal result: " + str(result).lower())
    elif mode == "restore-lock":
        print("Restore result: " + str(result).lower())
    else:
        print("Preview result: " + str(result).lower())
    print("")


def _print_reconcile_recovery_target(
    report: Any,
    contract_status: Dict[str, Any],
) -> None:
    state = report.proposed_state
    if _is_intentionally_stopped_state(state):
        print("Paused Campaign")
        _print_reconcile_key_values(
            [
                ("status", _stopped_boundary_description(state)),
                (
                    "next phase",
                    state.phase.value + " iteration " + str(int(state.iteration)),
                ),
                ("apply", _reconcile_apply_status(report, contract_status)),
            ]
        )
        print("")
        return
    if state.phase is CampaignPhase.HALTED:
        decision = "stay HALTED"
    elif state.phase is CampaignPhase.DONE:
        decision = "campaign is DONE"
    else:
        decision = (
            "recover to "
            + state.phase.value
            + " iteration "
            + str(int(state.iteration))
        )
    print("Recovery Target")
    _print_reconcile_key_values(
        [
            ("decision", decision),
            ("reason", _reconcile_latest_human_reason(report)),
            ("apply", _reconcile_apply_status(report, contract_status)),
        ]
    )
    print("")


def _print_reconcile_aimall_quality_revalidation(report: Any) -> None:
    revalidation = getattr(report, "aimall_quality_revalidation", None)
    if not isinstance(revalidation, Mapping) or revalidation.get("state") not in {
        "eligible",
        "resumable",
        "complete",
    }:
        return
    print("AIMAll Quality Revalidation")
    validation = (
        "all candidates pass the locked quality gates"
        if bool(revalidation.get("all_accepted_on_revalidation"))
        else str(revalidation.get("reason") or "not available")
    )
    _print_reconcile_key_values(
        [
            ("status", str(revalidation.get("state"))),
            ("candidates", int(revalidation.get("candidate_count") or 0)),
            ("previous reason", str(revalidation.get("old_reason") or "-")),
            ("validation", validation),
            (
                "allocation generation",
                revalidation.get("allocation_generation")
                if revalidation.get("allocation_generation") is not None
                else "pending",
            ),
            (
                "scheduler work",
                "none; no "
                + _scheduler_display_name()
                + " jobs will be submitted",
            ),
        ]
    )
    print("")


def _print_reconcile_completed_staging(report: Any) -> None:
    retirement = getattr(report, "completed_staging_retirement", None)
    if not isinstance(retirement, Mapping):
        return
    records = list(retirement.get("eligible") or [])
    tombstones = list(retirement.get("pending_tombstones") or [])
    if not records and not tombstones:
        return
    print("Completed Staging")
    if records:
        _print_reconcile_list(
            "retire on apply",
            [
                ("bootstrap" if str(record.get("context")) == "bootstrap" else "iteration " + str(int(record.get("iteration", 0))))
                + ": "
                + (
                    "delete duplicate-only residue"
                    if str(record.get("action")) == "delete"
                    else "preserve diagnostic residue outside active staging"
                )
                for record in records
            ],
        )
    if tombstones:
        print("  interrupted deletions to retry: " + str(len(tombstones)))
    print("  " + _scheduler_display_name() + " work submitted: none")
    print("")


def _print_reconcile_current_position(
    campaign: Path,
    report: Any,
    *,
    verbose: bool = False,
) -> None:
    print("Current Position")
    summary = _reconcile_read_current_state_summary(
        campaign,
        report,
        verbose=verbose,
    )
    _print_reconcile_key_values(list(summary.items()))
    print("")


def _print_reconcile_last_failure_compact(
    report: Any,
    *,
    verbose: bool = False,
) -> None:
    event = getattr(report, "last_halt_event", None)
    if not isinstance(event, dict):
        return
    source_phase = str(getattr(report, "source_state_phase", "") or "")
    historical = bool(
        source_phase
        and source_phase != CampaignPhase.HALTED.value
    )
    if historical and not verbose:
        return
    print("Resolved Historical Failure" if historical else "Last Failure")
    phase = str(event.get("from_phase") or event.get("phase") or "-")
    iteration = str(event.get("iteration", "-"))
    _print_reconcile_key_values(
        [
            ("phase", phase + " iteration " + iteration),
            ("time", _reconcile_format_time(event.get("ts"))),
        ]
    )
    reason = str(event.get("reason") or "-")
    for line in _reconcile_wrap_line("  reason: ", reason):
        print(line)
    print("")


def _print_reconcile_safety(
    campaign: Path,
    report: Any,
    contract_status: Dict[str, Any],
) -> None:
    cleanable = [_reconcile_human_reason(item) for item in _reconcile_cleanable_reasons(report)]
    blockers = [_reconcile_human_reason(item) for item in _reconcile_hard_blockers(report, contract_status)]
    protected = [
        _format_reconcile_contract_item(campaign, item)
        for item in contract_status.get("protected_artifacts", [])
    ]
    if blockers:
        status = "blocked"
    elif cleanable:
        status = "cleanup required, no hard blockers"
    else:
        status = "safe"
    print("Recovery Safety")
    _print_reconcile_key_values([("status", status)])
    _print_reconcile_list("cleanable", cleanable)
    _print_reconcile_list("protected", protected)
    _print_reconcile_list("blockers", blockers)
    print("")


def _print_reconcile_artefacts(
    campaign: Path,
    report: Any,
    contract_status: Dict[str, Any],
    *,
    verbose: bool = False,
) -> None:
    print("Artefacts")
    _print_reconcile_key_values(
        [
            (
                "reference-data versions",
                "committed "
                + repr(getattr(report, "committed_reference_data_versions", []))
                + ", valid "
                + repr(getattr(report, "valid_reference_data_versions", [])),
            ),
            (
                "model versions",
                "committed "
                + repr(getattr(report, "committed_model_versions", []))
                + ", valid "
                + repr(getattr(report, "valid_model_versions", [])),
            ),
            ("bootstrap handoff", _reconcile_bootstrap_summary(campaign, report)),
            ("Phase A handoff", _reconcile_phase_a_summary(campaign, report)),
            ("scripts", _reconcile_scripts_summary(report)),
            ("staging", _reconcile_staging_summary(campaign, contract_status, report)),
            ("model staging", _reconcile_model_staging_summary(report)),
            ("reference-data staging", _reconcile_reference_data_staging_summary(report)),
            ("reference commit", _reconcile_reference_commit_summary(report)),
        ]
    )
    candidate_recovery = getattr(report, "ferebus_candidate_recovery", None)
    if isinstance(candidate_recovery, dict) and candidate_recovery:
        print(
            "  FEREBUS candidate recovery: "
            + str(candidate_recovery.get("status") or "unknown")
            + " at "
            + _reconcile_relative_path(
                campaign,
                candidate_recovery.get("source_path"),
            )
        )
    inv = getattr(report, "script_inventory", {}) or {}
    sample = inv.get("sample") if isinstance(inv, dict) else None
    if verbose and isinstance(sample, list) and sample:
        display = [str(item) for item in sample[:8]]
        count = int(inv.get("count") or len(display))
        if count > len(display):
            display.append("... " + str(count - len(display)) + " more")
        _print_reconcile_list("script sample", display)
    if verbose and getattr(report, "notes", []):
        _print_reconcile_list("diagnostic notes", [str(item) for item in report.notes])
    if verbose and getattr(report, "unsafe_reasons", []):
        _print_reconcile_list(
            "raw blockers",
            [str(item) for item in report.unsafe_reasons],
        )
    if verbose and getattr(report, "trusted_artifacts", []):
        _print_reconcile_list("trusted artefacts", [str(item) for item in report.trusted_artifacts])
    if verbose and getattr(report, "blocking_artifacts", []):
        _print_reconcile_list("artefact inventory", [str(item) for item in report.blocking_artifacts])
    print("")


def _print_reconcile_partial_array(campaign: Path, report: Any) -> None:
    partial = getattr(report, "partial_array_recovery", None)
    if not isinstance(partial, dict) or not partial:
        return
    phase = str(partial.get("phase") or "UNKNOWN")
    scalar_publication = phase in {
        CampaignPhase.PHASE_A_DIVERSITY.value,
        CampaignPhase.PHASE_B_DIVERSITY.value,
    }
    print(
        "Scalar Publication Recovery"
        if scalar_publication
        else "Partial Array Recovery"
    )
    iteration = str(partial.get("iteration") if partial.get("iteration") is not None else "?")
    total = int(partial.get("logical_total") or 0)
    reuse = int(partial.get("n_reuse") or partial.get("n_complete") or 0)
    retry = int(partial.get("n_retry") or 0)
    scalar_retry = (
        scalar_publication
        and str(partial.get("publication_disposition") or "")
        == "archive_and_retry"
    )
    mode = (
        "archive incomplete publication and retry scalar task"
        if scalar_retry
        else (
            "full resubmission requested"
            if bool(partial.get("force_resubmit"))
            else "reuse completed outputs"
        )
    )
    publication = getattr(report, "ariadne_publication_recovery", None)
    publication_state = (
        str(publication.get("state"))
        if isinstance(publication, Mapping)
        else None
    )
    rows = [
        ("phase", phase + " iteration " + iteration),
        ("tasks", "total=" + str(total) + ", reusable=" + str(reuse) + ", retry=" + str(retry)),
        ("ledger", _reconcile_relative_path(campaign, partial.get("ledger"))),
        ("retry task file", _reconcile_relative_path(campaign, partial.get("retry_task_file")) or "none"),
        ("mode", mode),
        ("batch publication", publication_state),
    ]
    if scalar_publication:
        if scalar_retry:
            rows.extend(
                [
                    ("publication", "incomplete; archive during apply"),
                    ("scheduler tasks after resume", 1),
                    ("reason", str(partial.get("reason") or "incomplete output")),
                ]
            )
        else:
            rows.extend(
                [
                    ("selected geometries", int(partial.get("selected_count") or 0)),
                    ("scheduler tasks to submit", 0),
                    (
                        "ordering",
                        str(partial.get("ordering_classification") or "canonical"),
                    ),
                ]
            )
    _print_reconcile_key_values(rows)
    print("")


def _print_reconcile_ariadne_reuse(report: Any) -> None:
    summary = getattr(report, "ariadne_results_recovery", None)
    if not isinstance(summary, Mapping) or not summary:
        return
    print("ARIADNE Handoff")
    _print_reconcile_key_values(
        [
            ("accepted results", int(summary.get("accepted_tasks") or 0)),
            (
                "rejected tasks",
                str(int(summary.get("rejected_tasks") or 0))
                + " excluded from Phase B",
            ),
            (
                "missing rejected outputs",
                int(summary.get("missing_rejected_outputs") or 0),
            ),
            ("ARIADNE tasks to resubmit", 0),
        ]
    )
    print("")


def _print_reconcile_intent_repairs(report: Any) -> None:
    repairs = list(getattr(report, "receipt_backed_intent_repairs", []) or [])
    if not repairs:
        return
    print("Receipt-Backed Intent Repairs")
    _print_reconcile_list(
        "retire on apply",
        [
            str(item.get("phase"))
            + "@"
            + str(item.get("iteration"))
            + " submission_identity="
            + str(item.get("submission_identity"))
            for item in repairs
        ],
    )
    print("")


def _print_reconcile_config_changes(
    config_review: Any,
    *,
    verbose: bool = False,
) -> None:
    if config_review is None:
        return
    allowed = list(getattr(config_review, "allowed_changes", []) or [])
    blocked = list(getattr(config_review, "blocked_changes", []) or [])
    if not allowed and not blocked:
        if not verbose:
            return
        print("Config Changes")
        _print_reconcile_key_values([("status", "no campaign.yaml changes against config lock")])
        print("")
        return
    print("Config Changes")
    allowed_lines = [
        str(change.path)
        + ": "
        + repr(change.old)
        + " -> "
        + repr(change.new)
        + (" (" + str(change.reason) + ")" if str(change.reason or "") else "")
        for change in allowed
    ]
    blocked_lines = [
        str(change.path)
        + ": "
        + repr(change.old)
        + " -> "
        + repr(change.new)
        + (" (" + str(change.reason) + ")" if str(change.reason or "") else "")
        for change in blocked
    ]
    _print_reconcile_list("allowed", allowed_lines)
    _print_reconcile_list("blocked", blocked_lines)
    print("")


def _print_reconcile_contract_compact(
    campaign: Path,
    report: Any,
    contract_status: Dict[str, Any],
) -> None:
    state = report.proposed_state
    print("Recovery Contract")
    _print_reconcile_key_values(
        [
            (
                "selected phase",
                str(contract_status.get("selected_phase"))
                + " iteration "
                + str(contract_status.get("iteration")),
            ),
            ("status", _reconcile_contract_status_label(state, contract_status)),
        ]
    )
    _print_reconcile_list(
        "required",
        [str(item) for item in contract_status.get("required_inputs", [])],
    )
    trusted = [str(item) for item in contract_status.get("trusted_inputs", [])]
    trusted.extend(
        _format_reconcile_contract_item(campaign, item)
        for item in contract_status.get("trusted_handoffs", [])
    )
    protected = [
        _format_reconcile_contract_item(campaign, item)
        for item in contract_status.get("protected_artifacts", [])
    ]
    if protected:
        trusted.extend("protected " + item for item in protected)
    _print_reconcile_list("trusted", trusted)
    _print_reconcile_list(
        "missing",
        [str(item) for item in contract_status.get("missing_or_invalid_inputs", [])],
    )
    print("")


def _print_reconcile_apply_plan_compact(
    campaign: Path,
    report: Any,
    contract_status: Dict[str, Any],
    *,
    proposed_state_path: Optional[Path],
) -> None:
    cleanable = [_reconcile_human_reason(item) for item in _reconcile_cleanable_reasons(report)]
    blockers = _reconcile_hard_blockers(report, contract_status)
    revalidation = getattr(report, "aimall_quality_revalidation", None)
    revalidation_ready = bool(
        isinstance(revalidation, Mapping)
        and revalidation.get("state") in {"eligible", "resumable"}
        and revalidation.get("eligible") is True
    )
    explicit_cleanup_available = _reconcile_has_only_explicit_cleanup_blockers(
        report,
        blockers,
    )
    print("Apply Plan")
    if proposed_state_path is not None:
        print("  proposed state: " + _reconcile_relative_path(campaign, proposed_state_path))
    if blockers and not explicit_cleanup_available:
        print("  apply: blocked")
        _print_reconcile_list("next action", ["inspect blockers before restarting"])
        print("")
        return
    if revalidation_ready:
        _print_reconcile_list(
            "if applied",
            [
                "revalidate the parser-rejected AIMAll point(s)",
                "complete the existing point allocation",
                "prepare QM reference publication without submitting scheduler work",
                "recover to REFERENCE_COMMIT",
            ],
        )
        print("  command:")
        print("    " + _reconcile_apply_command(campaign, report))
        print("")
        return
    if cleanable or explicit_cleanup_available:
        _print_reconcile_list(
            "if applied",
            cleanable
            + [
                "recompute recovery",
                "write recovered state only if the final contract is ok",
            ],
        )
        candidates = _reconcile_valid_candidates(campaign, contract_status, report)
        if candidates:
            _print_reconcile_list(
                "expected recovery after cleanup",
                [str(item).split(" -> ", 1)[0] for item in candidates],
            )
        print("  command:")
        print("    " + _reconcile_apply_command(campaign, report))
        print("")
        return
    state = report.proposed_state
    if state.phase in (CampaignPhase.HALTED, CampaignPhase.DONE):
        print("  apply: not applicable")
        print("")
        return
    print("  reviewed recovery command:")
    print("    " + _reconcile_apply_command(campaign, report))
    print("")


def _print_reconcile_inspect_compact(
    campaign: Path,
    *,
    include_staging: bool = False,
) -> None:
    print("Inspect")
    commands: List[Tuple[str, Any]] = [
        ("status", _campaign_command(campaign, "status")),
        ("journal", _campaign_command(campaign, "journal", " --last-n 20")),
    ]
    if include_staging:
        commands.append(
            (
                "staging",
                "find "
                + str(campaign / ".DATA" / "STAGING")
                + " -maxdepth 3 -type f | sort",
            )
        )
    _print_reconcile_key_values(commands)
    print("")


def _print_reconcile_presentation(
    campaign: Path,
    presentation: _ReconcilePresentation,
    *,
    result_label: str = "Preview result",
) -> None:
    print("ICHOR Reconcile")
    print("Campaign: " + str(campaign))
    print(result_label + ": " + presentation.result)
    print("")

    print("Campaign state")
    _print_reconcile_key_values(presentation.campaign_state)
    print("")

    if presentation.result == "blocked":
        print("Manual review required")
        problem = presentation.blockers[0] if presentation.blockers else dict(
            presentation.campaign_state
        ).get("reason", "campaign recovery is blocked")
        _print_reconcile_key_values(
            [
                ("problem", problem),
                ("effect", "reconcile cannot safely change campaign state"),
                ("changes made", "none"),
            ]
        )
        if len(presentation.blockers) > 1:
            _print_reconcile_list("other blockers", presentation.blockers[1:])
        print("")
    elif presentation.result == "ready to apply" and presentation.planned_changes:
        print("Planned changes")
        _print_reconcile_key_values(presentation.planned_changes)
        print("")

    if presentation.preserved_diagnostics:
        print("Preserved diagnostics")
        _print_reconcile_list("evidence retained", presentation.preserved_diagnostics)
        print("")
    if presentation.warnings:
        print("Warnings")
        _print_reconcile_list("review", presentation.warnings)
        print("")

    print("Data safety")
    _print_reconcile_key_values(presentation.data_safety)
    print("")

    if presentation.after_apply:
        print("After apply")
        _print_reconcile_key_values(presentation.after_apply)
        print("")

    print("Next step")
    _print_reconcile_key_values(
        [
            (presentation.next_label, presentation.next_command),
            ("effect", presentation.next_effect),
        ]
    )
    print("")


def _print_reconcile_technical_details(
    campaign: Path,
    report: Any,
    contract_status: Dict[str, Any],
    *,
    proposed_state_path: Optional[Path],
    config_review: Any,
) -> None:
    print("Technical details")
    snapshot = getattr(report, "artifact_snapshot", None)
    verification_rows: List[Tuple[str, Any]] = []
    if snapshot is None:
        verification_rows.append(("verification", "incomplete"))
    else:
        deep_required = bool(getattr(report, "deep_verification_required", False))
        verification_rows.extend(
            [
                (
                    "verification",
                    "deep verified"
                    if snapshot.verification_level == "deep"
                    else "deep verification required"
                    if deep_required
                    else "authority verified",
                ),
                ("files inspected", int(snapshot.files_inspected)),
                (
                    "payload hashed",
                    str(int(snapshot.payload_files_hashed))
                    + " files / "
                    + str(int(snapshot.payload_bytes_hashed))
                    + " bytes",
                ),
                ("authority anchor", str(snapshot.anchor_sha256)),
            ]
        )
    if proposed_state_path is not None:
        verification_rows.append(
            ("proposal", _reconcile_relative_path(campaign, proposed_state_path))
        )
    verification_rows.append(("raw decision", str(getattr(report, "decision", "") or "-")))
    _print_reconcile_key_values(verification_rows)
    print("")
    scheduler_terminal_blockers = list(
        getattr(report, "scheduler_terminal_blockers", []) or []
    )
    if scheduler_terminal_blockers:
        print("Scheduler terminal evidence")
        _print_reconcile_list(
            "blockers",
            [
                str(item.get("phase") or "unknown phase")
                + "@"
                + str(item.get("iteration") or 0)
                + (" job=" + str(item.get("job_id")) if item.get("job_id") else "")
                + ": "
                + str(item.get("reason") or "unknown accounting problem")
                for item in scheduler_terminal_blockers
            ],
        )
        print("")
    _print_reconcile_last_failure_compact(report, verbose=True)
    _print_reconcile_artefacts(campaign, report, contract_status, verbose=True)
    _print_reconcile_aimall_quality_revalidation(report)
    _print_reconcile_ariadne_reuse(report)
    _print_reconcile_partial_array(campaign, report)
    _print_reconcile_intent_repairs(report)
    _print_reconcile_config_changes(config_review, verbose=True)
    _print_reconcile_contract_compact(campaign, report, contract_status)


def _print_reconcile_operator_report(
    campaign: Path,
    report: Any,
    contract_status: Dict[str, Any],
    *,
    mode: str,
    proposed_state_path: Optional[Path],
    config_review: Any = None,
    runtime_status: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> None:
    presentation = _reconcile_presentation(
        campaign,
        report,
        contract_status,
        config_review=config_review,
        runtime_status=runtime_status,
    )
    _print_reconcile_presentation(campaign, presentation)
    if verbose:
        _print_reconcile_technical_details(
            campaign,
            report,
            contract_status,
            proposed_state_path=proposed_state_path,
            config_review=config_review,
        )


def _print_reconcile_applied_operator_report(
    campaign: Path,
    report: Any,
    contract_status: Dict[str, Any],
    *,
    backup_path: Optional[Path],
    applied_proposal_path: Optional[Path],
    removed: Sequence[str],
    removed_model_staging: Sequence[str],
    archived_scripts: Sequence[str],
    archived: Sequence[str],
    archived_reference_data_staging: Sequence[str],
    restored_bootstrap_handoff: Sequence[str],
    retired_completed_staging: Optional[Mapping[str, Any]] = None,
    archived_ariadne_publication: Sequence[str] = (),
    archived_scalar_diversity_publication: Sequence[str] = (),
    ferebus_retrain_archive: Sequence[str] = (),
    ferebus_staging_restore: Optional[Mapping[str, Any]] = None,
    archived_array_outputs: Sequence[str] = (),
    refreshed_array_ledger: Optional[Mapping[str, Any]] = None,
    config_review: Any = None,
    resolved_intents: Sequence[Mapping[str, Any]] = (),
    environment_transition: Optional[Mapping[str, Any]] = None,
    environment_transition_deferred: bool = False,
    verbose: bool = False,
) -> None:
    print("ICHOR Reconcile")
    print("Campaign: " + str(campaign))
    print("Apply result: completed")
    print("")
    print("Applied changes")
    applied_rows: List[Tuple[str, str]] = [("campaign state", "written and verified")]
    transaction_recovery = getattr(
        report,
        "reconcile_transaction_recovery",
        None,
    )
    if (
        isinstance(transaction_recovery, Mapping)
        and str(transaction_recovery.get("state") or "") == "recovered"
    ):
        backup_status = str(
            transaction_recovery.get("state_backup_status") or ""
        )
        if backup_status == "repairable_partial":
            applied_rows.append(
                (
                    "state backup",
                    "replaced the interrupted partial backup from authenticated "
                    "transaction evidence",
                )
            )
        elif backup_status == "absent":
            applied_rows.append(
                (
                    "state backup",
                    "created the missing authenticated state backup",
                )
            )
    applied_rows.extend(("stale artefact", "removed " + _reconcile_relative_path(campaign, item)) for item in removed)
    applied_rows.extend(("model staging", "removed " + _reconcile_relative_path(campaign, item)) for item in removed_model_staging)
    applied_rows.extend(("submission scripts", "archived " + _reconcile_relative_path(campaign, item)) for item in archived_scripts)
    applied_rows.extend(("staging", "archived " + _reconcile_relative_path(campaign, item)) for item in archived)
    applied_rows.extend(("QM staging", "archived " + _reconcile_relative_path(campaign, item)) for item in archived_reference_data_staging)
    retirement = retired_completed_staging or {}
    applied_rows.extend(
        ("completed staging", "retired " + _reconcile_relative_path(campaign, item))
        for item in retirement.get("retired") or []
    )
    applied_rows.extend(
        ("diagnostic staging", "preserved " + _reconcile_relative_path(campaign, item))
        for item in retirement.get("preserved") or []
    )
    applied_rows.extend(
        ("bootstrap staging", "restored " + _reconcile_relative_path(campaign, item))
        for item in restored_bootstrap_handoff
    )
    applied_rows.extend(
        ("ARIADNE publication", "archived " + _reconcile_relative_path(campaign, item))
        for item in archived_ariadne_publication
    )
    applied_rows.extend(
        (
            "diversity publication",
            "archived " + _reconcile_relative_path(campaign, item),
        )
        for item in archived_scalar_diversity_publication
    )
    applied_rows.extend(
        ("FEREBUS output", "archived " + _reconcile_relative_path(campaign, item))
        for item in ferebus_retrain_archive
    )
    if isinstance(ferebus_staging_restore, Mapping):
        applied_rows.append(
            (
                "FEREBUS staging",
                "restored terminal producer at "
                + _reconcile_relative_path(
                    campaign,
                    ferebus_staging_restore.get("restored_path"),
                ),
            )
        )
    if refreshed_array_ledger is not None:
        applied_rows.append(
            (
                "array recovery",
                "prepared "
                + str(int(refreshed_array_ledger.get("n_retry") or 0))
                + " task(s) for resubmission after resume",
            )
        )
    aimall_postprocess = getattr(
        report,
        "aimall_postprocess_recovery",
        None,
    )
    if isinstance(aimall_postprocess, Mapping):
        completed = int(aimall_postprocess.get("logical_total") or 0)
        applied_rows.append(
            (
                "AIMAll results",
                "retained "
                + str(completed)
                + " scheduler-completed output candidate"
                + ("" if completed == 1 else "s")
                + " for local validation; reconcile submitted no work",
            )
        )
    ariadne_terminal = getattr(
        report,
        "ariadne_terminal_postprocess_recovery",
        None,
    )
    if isinstance(ariadne_terminal, Mapping):
        completed = int(
            ariadne_terminal.get("n_scheduler_completed") or 0
        )
        failed = int(ariadne_terminal.get("n_scheduler_failed") or 0)
        applied_rows.append(
            (
                "ARIADNE recovery",
                "retained "
                + str(completed)
                + " scheduler-completed candidate"
                + ("" if completed == 1 else "s")
                + " and "
                + str(failed)
                + " scheduler-failed slot"
                + ("" if failed == 1 else "s")
                + " for local validation; reconcile submitted no work",
            )
        )
    applied_rows.extend(("array output", "archived " + _reconcile_relative_path(campaign, item)) for item in archived_array_outputs)
    for change in list(getattr(config_review, "allowed_changes", []) or []):
        applied_rows.append(
            (
                "configuration",
                _reconcile_config_label(change.path)
                + " "
                + _reconcile_config_value(change.old)
                + " -> "
                + _reconcile_config_value(change.new),
            )
        )
    if resolved_intents:
        applied_rows.append(
            (
                "submission records",
                "retired " + str(len(resolved_intents)) + " completed record(s)",
            )
        )
    if isinstance(environment_transition, Mapping) and environment_transition.get("changed"):
        binding_repair = str(
            environment_transition.get("environment_binding_repair_kind") or ""
        )
        applied_rows.append(
            (
                (
                    "environment configuration"
                    if binding_repair
                    else "software environment"
                ),
                "recorded generation "
                + str(environment_transition.get("generation"))
                + (
                    " bound to the current campaign configuration"
                    if binding_repair
                    else ""
                ),
            )
        )
    _print_reconcile_key_values(applied_rows)
    print("")

    warnings = [str(item) for item in retirement.get("warnings") or []]
    if warnings:
        print("Warnings")
        _print_reconcile_list("review", warnings)
        print("")

    final_phase = report.proposed_state.phase
    intentionally_stopped = _is_intentionally_stopped_state(
        report.proposed_state
    )
    if environment_transition_deferred:
        applied_rows = [("software environment", "will be checked when resume clears the completed stop")]
        print("Deferred check")
        _print_reconcile_key_values(applied_rows)
        print("")

    print("Data safety")
    _print_reconcile_key_values(
        [
            ("committed QM data", "unchanged"),
            ("committed models", "unchanged"),
            ("final authority check", "passed" if contract_status.get("contract_ok") else "failed"),
        ]
    )
    print("")

    print("Campaign state")
    _print_reconcile_key_values(
        [
            ("now", _reconcile_state_summary(report.proposed_state, None)),
            ("daemon", "not running"),
            (_scheduler_display_name() + " jobs submitted", "none"),
        ]
    )
    print("")

    print("Next step")
    if bool(contract_status.get("contract_ok")) and final_phase not in {
        CampaignPhase.HALTED,
        CampaignPhase.DONE,
    }:
        effect = (
            (
                "validate "
                + str(int(aimall_postprocess.get("logical_total") or 0))
                + " scheduler-completed AIMAll output candidates locally, "
                "submit only invalid or unfinished tasks if required, then "
                "continue from "
            )
            if isinstance(aimall_postprocess, Mapping)
            else (
                "clear the completed pause and continue from "
                if intentionally_stopped
                else "continue the campaign from "
            )
        )
        _print_reconcile_key_values(
            [
                ("run", _campaign_command(campaign, "resume")),
                (
                    "effect",
                    effect
                    + _reconcile_phase_display(
                        report.proposed_state.phase,
                        report.proposed_state.iteration,
                        replacement_round=int(report.proposed_state.replacement_round),
                    ),
                ),
            ]
        )
    else:
        _print_reconcile_key_values(
            [
                ("run", _campaign_command(campaign, "reconcile", " --verbose")),
                ("effect", "review why the recovered campaign cannot continue yet"),
            ]
        )
    print("")
    if verbose:
        print("Technical details")
        _print_reconcile_key_values(
            [
                (
                    "state written",
                    _reconcile_relative_path(
                        campaign,
                        campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME,
                    ),
                ),
                ("previous state backup", _reconcile_relative_path(campaign, backup_path) if backup_path is not None else "none"),
                ("applied proposal archive", _reconcile_relative_path(campaign, applied_proposal_path) if applied_proposal_path is not None else "none"),
            ]
        )
        print("")
        _print_reconcile_contract_compact(campaign, report, contract_status)


def _print_reconcile_apply_blocked(
    campaign: Path,
    *,
    title: str,
    reasons: Sequence[Any],
    next_actions: Sequence[Any] = (),
) -> None:
    print("ICHOR Reconcile", file=sys.stderr)
    print("Campaign: " + str(campaign), file=sys.stderr)
    print("Apply result: blocked", file=sys.stderr)
    print("", file=sys.stderr)
    print("Manual review required", file=sys.stderr)
    problem = _reconcile_plain_text(reasons[0]) if reasons else "the apply safety check failed"
    rows = [
        ("area", title),
        ("problem", problem),
        ("effect", "reconcile cannot safely change campaign state"),
        ("changes made", "none"),
    ]
    width = max(len(label) for label, _value in rows)
    for label, value in rows:
        print("  " + label.ljust(width) + ": " + str(value), file=sys.stderr)
    if len(reasons) > 1:
        print("  other blockers:", file=sys.stderr)
        for reason in reasons[1:]:
            print("    - " + _reconcile_plain_text(reason), file=sys.stderr)
    print("", file=sys.stderr)
    if next_actions:
        print("Next step", file=sys.stderr)
        action = str(next_actions[0])
        label = "run" if action.startswith("ichor-al-daemon ") else "review"
        print("  " + label + ": " + action, file=sys.stderr)
        print("  effect: resolve or inspect the blocker before retrying apply", file=sys.stderr)
        print("", file=sys.stderr)


def _print_reconcile_follow_up_required(
    campaign: Path,
    state: CampaignState,
    *,
    problem: Any,
    next_command: str,
    next_effect: str,
    verbose_detail: Optional[str] = None,
) -> None:
    print("ICHOR Reconcile", file=sys.stderr)
    print("Campaign: " + str(campaign), file=sys.stderr)
    print("Apply result: recovery applied; follow-up required", file=sys.stderr)
    print("", file=sys.stderr)
    print("Campaign state", file=sys.stderr)
    print("  now: " + _reconcile_state_summary(state, None), file=sys.stderr)
    print("  daemon: not running", file=sys.stderr)
    print(
        "  " + _scheduler_display_name() + " jobs submitted: none",
        file=sys.stderr,
    )
    print("", file=sys.stderr)
    print("Follow-up required", file=sys.stderr)
    print("  problem: " + _reconcile_plain_text(problem), file=sys.stderr)
    print("  recovery state: already committed", file=sys.stderr)
    print("", file=sys.stderr)
    print("Next step", file=sys.stderr)
    print("  run: " + next_command, file=sys.stderr)
    print("  effect: " + next_effect, file=sys.stderr)
    if verbose_detail:
        print("", file=sys.stderr)
        print("Technical detail", file=sys.stderr)
        print("  " + str(verbose_detail), file=sys.stderr)
    print("", file=sys.stderr)


def _advance_environment_after_reconcile(
    campaign: Path,
    state: CampaignState,
    config: CampaignConfig,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Advance environment identity, or defer it for an intentional stop."""
    if _is_intentionally_stopped_state(state):
        return None, True

    from .execution_identity import (
        advance_environment_generation,
        execution_identity_path,
        read_execution_identity,
    )

    if not execution_identity_path(campaign).is_file():
        return None, False
    identity = read_execution_identity(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    live_preflight_ok = True
    if str(identity["mode"]) == "live":
        availability = check_backends()
        preflight = evaluate_campaign_preflight(
            campaign,
            config=config,
            avail=availability,
        )
        live_preflight_ok = bool(
            preflight.get("_presentation_base_environment_ready", False)
        )
        if not live_preflight_ok:
            details = _preflight_failure_details(preflight)
            raise ValueError(
                "live backend preflight failed: " + "; ".join(details[:8])
            )
    transition = advance_environment_generation(
        campaign,
        config=config,
        live_preflight_ok=live_preflight_ok,
        scheduler_ownership_clear=True,
    )
    return transition, False


def _reconcile_environment_failure_guidance(
    campaign: Path,
    state: CampaignState,
    detail: str,
) -> Tuple[str, str, str]:
    """Choose a truthful follow-up after a committed reconcile transition fails."""
    next_command = _campaign_command(campaign, "preflight")
    next_effect = "verify the reported startup blocker before attempting to resume"
    if state.phase in {
        CampaignPhase.PHASE_A_DIVERSITY,
        CampaignPhase.PHASE_B_DIVERSITY,
    }:
        try:
            from .execution_identity import (
                inspect_scalar_diversity_transition_boundary,
            )

            transition_evidence = inspect_scalar_diversity_transition_boundary(
                campaign,
                state,
            )
        except Exception:
            transition_evidence = {"safe": False}
        if not bool(transition_evidence.get("safe", False)):
            next_command = _campaign_command(
                campaign,
                "reconcile",
                " --verbose",
            )
            next_effect = (
                "review the scalar diversity transition evidence; resume "
                "would repeat the same refusal"
            )
    problem = (
        "the software-environment transition failed: "
        + _reconcile_plain_text(detail)
    )
    return problem, next_command, next_effect


def _scratch_intent_index(campaign: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    root = _submission_intent.intent_dir(campaign)
    if not root.is_dir():
        return index
    paths = list(root.glob("*.json"))
    history = root / _submission_intent.INTENT_HISTORY_DIR_NAME
    if history.is_dir() and not history.is_symlink():
        paths.extend(history.glob("*.json"))
    for path in sorted(paths):
        if path.is_symlink():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("attempt_id", "submission_identity"):
            value = str(payload.get(key) or "")
            if value:
                index[value] = payload
    return index


def _scratch_scheduler_state(
    job_id: str,
    expected_task_count: Optional[int] = None,
    scheduler_kind: str = "slurm",
    expected_job_name: Optional[str] = None,
) -> Tuple[str, str]:
    """Return active, inactive, or inconclusive for one recorded scheduler job."""
    from .submit import sacct_poll

    backend = get_scheduler_backend(scheduler_kind)
    identity_kwargs = (
        {
            "expected_job_name": expected_job_name,
            "expected_owner": current_scheduler_user(),
        }
        if expected_job_name
        else {}
    )
    try:
        if scheduler_kind == "slurm":
            queue = sacct_poll.find_active_job_by_id_detailed(
                str(job_id),
                **identity_kwargs,
            )
        else:
            queue = backend.find_active_job_by_id(
                str(job_id),
                **identity_kwargs,
            )
    except Exception as exc:
        return "inconclusive", type(exc).__name__ + ": " + str(exc)
    if bool(getattr(queue, "inconclusive", False)):
        return (
            "inconclusive",
            str(
                getattr(queue, "error", None)
                or backend.display_name + " queue lookup failed"
            ),
        )
    if bool(getattr(queue, "active", False)):
        return "active", backend.display_name + " reports active rows"
    try:
        if scheduler_kind == "slurm":
            observations = sacct_poll.poll_job(
                str(job_id),
                **identity_kwargs,
            )
        else:
            observations = backend.poll_job(
                str(job_id),
                **identity_kwargs,
            )
    except Exception as exc:
        return (
            "inconclusive",
            backend.display_name
            + " accounting lookup failed: "
            + type(exc).__name__
            + ": "
            + str(exc),
        )
    if not observations:
        return "inconclusive", backend.display_name + " returned no accounting rows"
    matching = _matching_sacct_observations(str(job_id), observations)
    if not matching:
        return (
            "inconclusive",
            backend.accounting_command + " returned no matching task rows",
        )
    summary = sacct_poll.aggregate_states(
        str(job_id),
        matching,
        expected_task_count=expected_task_count,
        strict_parent_job_id=(scheduler_kind == "slurm"),
    )
    if summary.conflicting_task_indices:
        return (
            "inconclusive",
            backend.accounting_command + " returned conflicting task rows",
        )
    if summary.out_of_range_task_indices:
        return (
            "inconclusive",
            backend.accounting_command + " returned out-of-range task rows",
        )
    if summary.n_missing:
        return (
            "inconclusive",
            backend.accounting_command
            + " is missing "
            + str(summary.n_missing)
            + " of "
            + str(summary.n_expected)
            + " expected task rows",
        )
    states = [observation.status for observation in summary.observations]
    if any(state in sacct_poll.NON_TERMINAL_STATES for state in states):
        return (
            "active",
            backend.accounting_command + " reports non-terminal rows",
        )
    if any(state == sacct_poll.JobStatus.UNKNOWN for state in states):
        return (
            "inconclusive",
            backend.accounting_command + " reports unknown rows",
        )
    if not all(state in sacct_poll.TERMINAL_STATES for state in states):
        return (
            "inconclusive",
            backend.accounting_command + " rows are not conclusively terminal",
        )
    return "inactive", backend.display_name + " rows are terminal"


def _scratch_attempt_report(campaign: Path) -> List[Dict[str, Any]]:
    records = _scratch.inventory(campaign)
    invalid = [record for record in records if record.get("status") == "invalid"]
    grouped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if record.get("status") == "invalid":
            continue
        key = str(record.get("attempt_path") or record.get("submission_identity") or "")
        group = grouped.setdefault(
            key,
            {
                "attempt_id": str(record.get("attempt_id") or ""),
                "submission_identity": str(record.get("submission_identity") or ""),
                "phase": str(record.get("phase") or ""),
                "iteration": int(record.get("iteration", 0)),
                "path": str(record.get("attempt_path") or ""),
                "task_count": 0,
                "task_statuses": set(),
                "job_ids": set(),
            },
        )
        group["task_count"] += 1
        group["task_statuses"].add(str(record.get("status") or "prepared"))
        group["job_ids"].add(str(record.get("job_id") or ""))
    intent_index = _scratch_intent_index(campaign)
    scheduler_cache: Dict[
        Tuple[str, Optional[int], str, str], Tuple[str, str]
    ] = {}
    output: List[Dict[str, Any]] = [dict(item) for item in invalid]
    for group in grouped.values():
        attempt_id = str(group["attempt_id"])
        identity = str(group["submission_identity"])
        intent = intent_index.get(attempt_id) or intent_index.get(identity) or {}
        scheduler_kind = str(
            intent.get("scheduler_identity_kind") or "slurm"
        ).strip().lower()
        expected_job_name = str(intent.get("expected_job_name") or "")
        expected_tasks: Optional[int] = None
        expected_tasks_error: Optional[str] = None
        if intent.get("expected_tasks") is not None:
            try:
                if isinstance(intent["expected_tasks"], bool):
                    raise ValueError
                expected_tasks = int(intent["expected_tasks"])
            except (TypeError, ValueError):
                expected_tasks_error = (
                    "submission intent expected_tasks is malformed"
                )
            if expected_tasks is not None and expected_tasks <= 0:
                expected_tasks_error = (
                    "submission intent expected_tasks must be > 0"
                )
                expected_tasks = None
        job_states: Dict[str, str] = {}
        scheduler_reasons: Dict[str, str] = {}
        for job_id in sorted(group["job_ids"]):
            if expected_tasks_error is not None:
                job_states[job_id] = "inconclusive"
                scheduler_reasons[job_id] = expected_tasks_error
                continue
            cache_key = (
                job_id,
                expected_tasks,
                scheduler_kind,
                expected_job_name,
            )
            if cache_key not in scheduler_cache:
                scratch_kwargs: Dict[str, Any] = {
                    "expected_task_count": expected_tasks,
                }
                if scheduler_kind != "slurm":
                    scratch_kwargs["scheduler_kind"] = scheduler_kind
                if expected_job_name:
                    scratch_kwargs["expected_job_name"] = expected_job_name
                scheduler_cache[cache_key] = _scratch_scheduler_state(
                    job_id,
                    **scratch_kwargs,
                )
            scheduler_state, reason = scheduler_cache[cache_key]
            job_states[job_id] = scheduler_state
            scheduler_reasons[job_id] = reason
        intent_status = str(intent.get("status") or "")
        states = set(job_states.values())
        task_statuses = set(group["task_statuses"])
        if "active" in states:
            classification = "active"
            cleanable = False
        elif "inconclusive" in states:
            classification = "scheduler_inconclusive"
            cleanable = False
        elif "failed_retained" in task_statuses:
            classification = "retained_failure"
            cleanable = True
        elif task_statuses == {"completed"} or intent_status == "COMPLETED":
            classification = "completed_leftover"
            cleanable = True
        elif intent_status == "PRE_SUBMIT":
            classification = "stale_pre_submit"
            cleanable = True
        elif intent:
            classification = "interrupted"
            cleanable = True
        else:
            classification = "orphaned"
            cleanable = True
        output.append(
            {
                "status": classification,
                "cleanable": bool(cleanable),
                "attempt_id": attempt_id,
                "submission_identity": identity,
                "phase": group["phase"],
                "iteration": group["iteration"],
                "path": group["path"],
                "task_count": int(group["task_count"]),
                "task_statuses": sorted(task_statuses),
                "job_states": job_states,
                "scheduler_reasons": scheduler_reasons,
                "intent_status": intent_status or None,
            }
        )
    return output


def _cmd_reconcile_scratch(
    campaign: Path,
    *,
    apply: bool,
    json_output: bool,
    selected_attempts: Sequence[str],
) -> int:
    records = _scratch_attempt_report(campaign)
    invalid = [item for item in records if item.get("status") == "invalid"]
    attempts = [item for item in records if item.get("status") != "invalid"]
    selected = {str(value) for value in selected_attempts}
    known = {
        str(item.get("submission_identity") or "")
        for item in attempts
    } | {str(item.get("attempt_id") or "") for item in attempts}
    missing = sorted(selected - known)
    if missing:
        print(
            "unknown scratch attempt selector(s): " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    considered = [
        item
        for item in attempts
        if not selected
        or str(item.get("submission_identity")) in selected
        or str(item.get("attempt_id")) in selected
    ]
    cleanable = [item for item in considered if bool(item.get("cleanable"))]
    blocked = [item for item in considered if not bool(item.get("cleanable"))]
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "campaign_dir": str(campaign),
        "mode": "apply" if apply else "proposal",
        "invalid": invalid,
        "cleanable_attempts": cleanable,
        "blocked_attempts": blocked,
        "removed": [],
    }
    if invalid:
        if json_output:
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        else:
            print("Scratch clean-up is blocked by invalid ownership evidence:", file=sys.stderr)
            for item in invalid:
                print("  - " + str(item.get("path")) + ": " + str(item.get("reason")), file=sys.stderr)
        return 9
    if apply and selected and blocked:
        print("refusing to clean selected active or scheduler-inconclusive attempt(s):", file=sys.stderr)
        for item in blocked:
            print("  - " + str(item.get("submission_identity")) + " (" + str(item.get("status")) + ")", file=sys.stderr)
        return 9
    if apply and cleanable:
        allowed = [str(item["submission_identity"]) for item in cleanable]
        protected_jobs = {
            job_id
            for item in blocked
            for job_id in dict(item.get("job_states") or {})
        }
        payload["removed"] = _scratch.clean_inactive_attempts(
            campaign,
            active_job_ids=protected_jobs,
            allowed_attempts=allowed,
        )
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    _print_reconcile_header(
        campaign,
        mode="scratch-apply" if apply else "scratch-proposal",
        result="CLEANED" if apply else "INSPECT",
    )
    if cleanable:
        print("Conclusively inactive scratch attempts:")
        for item in cleanable:
            print(
                "  - "
                + str(item.get("submission_identity"))
                + " "
                + str(item.get("status"))
                + " tasks="
                + str(item.get("task_count"))
            )
    else:
        print("No conclusively inactive scratch attempts were found.")
    if blocked:
        print("Protected scratch attempts:")
        for item in blocked:
            print("  - " + str(item.get("submission_identity")) + " " + str(item.get("status")))
    if apply:
        print("Removed scratch attempt directories: " + str(len(payload["removed"])))
    else:
        print("Rerun with --clean-scratch --apply to remove the listed inactive attempts.")
    return 0


def _cmd_reconcile_ariadne_quarantine(
    campaign: Path,
    *,
    apply: bool,
    json_output: bool,
    selected_attempts: Sequence[str],
) -> int:
    from .daemon.ariadne_quarantine import clean_quarantine, inventory_quarantine

    inventory = inventory_quarantine(campaign)
    attempts = list(inventory.get("attempts") or [])
    errors = list(inventory.get("errors") or [])
    known = {str(item.get("attempt_id")) for item in attempts}
    requested = {str(value) for value in selected_attempts}
    missing = sorted(requested - known)
    if missing:
        print(
            "unknown ARIADNE quarantine attempt selector(s): " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    selected = [
        item
        for item in attempts
        if not requested or str(item.get("attempt_id")) in requested
    ]
    payload = {
        "schema_version": 1,
        "campaign_dir": str(campaign),
        "attempts": selected,
        "errors": errors,
        "total_bytes": int(sum(int(item.get("verified_bytes", 0)) for item in selected)),
        "removed": [],
    }
    if errors:
        if json_output:
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        else:
            print(
                "ARIADNE quarantine clean-up is blocked by invalid evidence:",
                file=sys.stderr,
            )
            for item in errors:
                print(
                    "  - " + str(item.get("path")) + ": " + str(item.get("error")),
                    file=sys.stderr,
                )
        return 9
    if apply:
        payload["removed"] = clean_quarantine(
            campaign,
            attempt_ids=(None if not requested else sorted(requested)),
        )
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    _print_reconcile_header(
        campaign,
        mode="ariadne-quarantine-apply" if apply else "ariadne-quarantine-proposal",
        result="CLEANED" if apply else "INSPECT",
    )
    if selected:
        print("Retained ARIADNE retry attempts:")
        for item in selected:
            print(
                "  - "
                + str(item.get("attempt_id"))
                + " iteration="
                + str(item.get("iteration"))
                + " bytes="
                + str(item.get("verified_bytes"))
            )
    else:
        print("No retained ARIADNE retry attempts were found.")
    if apply:
        print("Removed ARIADNE quarantine attempts: " + str(len(payload["removed"])))
    else:
        print(
            "Rerun with --clean-ariadne-quarantine --apply to remove the listed attempts."
        )
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    clean_scratch = bool(getattr(args, "clean_scratch", False))
    clean_quarantine = bool(getattr(args, "clean_ariadne_quarantine", False))
    scratch_attempts = [
        str(value)
        for value in (getattr(args, "scratch_attempt", None) or [])
        if str(value)
    ]
    quarantine_attempts = [
        str(value)
        for value in (getattr(args, "ariadne_quarantine_attempt", None) or [])
        if str(value)
    ]
    restore_config = bool(getattr(args, "restore_config_from_lock", False))
    restore_lock = bool(getattr(args, "restore_config_lock_history", False))
    archive_staging_requested = bool(getattr(args, "archive_staging", False))
    force_resubmit_array = bool(getattr(args, "force_resubmit_array_tasks", False))
    archive_existing_array_outputs = bool(
        getattr(args, "archive_existing_array_task_outputs", False)
    )
    retrain_ferebus = bool(getattr(args, "retrain_ferebus", False))
    deep_verify = bool(getattr(args, "deep_verify", False))
    verification_level = "deep" if deep_verify else "authority"
    campaign = resolve_campaign_dir(
        args.campaign_dir,
        require_campaign_yaml=False,
    )
    authoritative_mutation = bool(getattr(args, "apply", False)) or restore_lock
    if authoritative_mutation and not bool(
        getattr(args, "_operator_lock_owned", False)
    ):
        try:
            with _exclusive_operator_lock(campaign):
                args._operator_lock_owned = True
                return cmd_reconcile(args)
        except (OSError, RuntimeError) as exc:
            print("reconcile mutation blocked: " + str(exc), file=sys.stderr)
            return 3
        finally:
            args._operator_lock_owned = False
    operator_lock_owned = bool(getattr(args, "_operator_lock_owned", False))
    try:
        accepts_lock_ownership = (
            "operator_lock_owned"
            in inspect.signature(_reconcile_runtime_status).parameters
        )
    except (TypeError, ValueError):
        accepts_lock_ownership = False
    if operator_lock_owned and accepts_lock_ownership:
        runtime_status = _reconcile_runtime_status(
            campaign,
            operator_lock_owned=True,
        )
    else:
        runtime_status = _reconcile_runtime_status(campaign)
    if scratch_attempts and not clean_scratch:
        print("refusing --scratch-attempt without --clean-scratch", file=sys.stderr)
        return 2
    if quarantine_attempts and not clean_quarantine:
        print(
            "refusing --ariadne-quarantine-attempt without "
            "--clean-ariadne-quarantine",
            file=sys.stderr,
        )
        return 2
    if clean_scratch and clean_quarantine:
        print(
            "refusing to combine --clean-scratch and --clean-ariadne-quarantine",
            file=sys.stderr,
        )
        return 2
    if clean_scratch:
        incompatible = [
            name
            for name, enabled in (
                ("--restore-config-from-lock", restore_config),
                ("--restore-config-lock-history", restore_lock),
                ("--archive-staging", archive_staging_requested),
                ("--force-resubmit-array-tasks", force_resubmit_array),
                ("--retrain-ferebus", retrain_ferebus),
            )
            if enabled
        ]
        if incompatible:
            print(
                "refusing --clean-scratch with " + ", ".join(incompatible),
                file=sys.stderr,
            )
            return 2
    if clean_quarantine:
        incompatible = [
            name
            for name, enabled in (
                ("--restore-config-from-lock", restore_config),
                ("--restore-config-lock-history", restore_lock),
                ("--archive-staging", archive_staging_requested),
                ("--force-resubmit-array-tasks", force_resubmit_array),
                ("--retrain-ferebus", retrain_ferebus),
            )
            if enabled
        ]
        if incompatible:
            print(
                "refusing --clean-ariadne-quarantine with " + ", ".join(incompatible),
                file=sys.stderr,
            )
            return 2
    if restore_lock:
        if restore_config or bool(getattr(args, "apply", False)):
            print(
                "refusing --restore-config-lock-history with another reconcile mutation",
                file=sys.stderr,
            )
            return 2
        if runtime_status.get("reconcile_apply_blockers"):
            _print_reconcile_runtime_warning(runtime_status, campaign)
            return 9
        try:
            state = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
            restored = restore_config_lock_from_history(
                campaign,
                expected_campaign_uid=str(state.campaign_uid),
            )
        except Exception as exc:
            print("could not restore config lock from history: " + str(exc), file=sys.stderr)
            return 8
        _print_reconcile_header(campaign, mode="restore-lock", result="RESTORED")
        print("  config lock: " + _reconcile_relative_path(campaign, restored))
        print("  campaign UID: " + str(state.campaign_uid))
        return 0
    if bool(getattr(args, "json", False)) and bool(getattr(args, "apply", False)):
        print(
            json.dumps(
                {
                    "schema_version": 5,
                    "error": "json_apply_not_supported",
                    "message": "reconcile --json is proposal-only; rerun without --json to apply",
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    if archive_staging_requested and not bool(getattr(args, "apply", False)):
        print(
            "refusing --archive-staging without --apply; this option mutates "
            ".DATA/STAGING and must be explicit",
            file=sys.stderr,
        )
        return 2
    if archive_existing_array_outputs and not bool(getattr(args, "apply", False)):
        print(
            "refusing --archive-existing-array-task-outputs without --apply; "
            "this option mutates existing task outputs",
            file=sys.stderr,
        )
        return 2
    if archive_existing_array_outputs and not force_resubmit_array:
        print(
            "refusing --archive-existing-array-task-outputs without "
            "--force-resubmit-array-tasks",
            file=sys.stderr,
        )
        return 2
    if retrain_ferebus and not bool(getattr(args, "apply", False)):
        print(
            "refusing --retrain-ferebus without --apply; complete model "
            "output is archived before a new fit is submitted",
            file=sys.stderr,
        )
        return 2
    if restore_config:
        if archive_staging_requested:
            print(
                "refusing --restore-config-from-lock together with --archive-staging",
                file=sys.stderr,
            )
            return 2
        if bool(getattr(args, "apply", False)):
            print(
                "refusing --restore-config-from-lock together with --apply; "
                "restore the config proposal first, then preview reconcile",
                file=sys.stderr,
            )
            return 2
        if runtime_status.get("reconcile_apply_blockers"):
            _print_reconcile_runtime_warning(runtime_status, campaign)
        try:
            target_config = restore_config_from_lock_proposal(campaign)
        except Exception as exc:
            print(
                "could not restore campaign.yaml from config lock: "
                + str(exc),
                file=sys.stderr,
            )
            return 8
        _print_reconcile_header(
            campaign,
            mode="restore-config",
            result="proposal written",
        )
        print("Config Proposal")
        _print_reconcile_key_values(
            [
                ("proposal", _reconcile_relative_path(campaign, target_config)),
                ("source", ".DATA/ACTIVE_LEARNING/config_lock.json"),
                ("status", "dense full config snapshot written"),
            ]
        )
        print("")
        print("Next")
        print("  review the proposal and restore the locked values through the campaign config editor")
        print("  do not move the proposal over campaign.yaml by hand")
        print("  then preview recovery:")
        print("    " + _campaign_command(campaign, "reconcile"))
        print("")
        return 0
    if bool(getattr(args, "apply", False)) and runtime_status.get("reconcile_apply_blockers"):
        _print_reconcile_apply_blocked(
            campaign,
            title="Runtime Safety",
            reasons=list(runtime_status.get("reconcile_apply_blockers") or []),
            next_actions=[
                "stop the daemon or wait for the lock/lease to clear",
                "rerun reconcile without --apply to review the new recovery preview",
            ],
        )
        _print_reconcile_runtime_warning(runtime_status, campaign)
        return 9
    if clean_scratch:
        return _cmd_reconcile_scratch(
            campaign,
            apply=bool(getattr(args, "apply", False)),
            json_output=bool(getattr(args, "json", False)),
            selected_attempts=scratch_attempts,
        )
    if clean_quarantine:
        return _cmd_reconcile_ariadne_quarantine(
            campaign,
            apply=bool(getattr(args, "apply", False)),
            json_output=bool(getattr(args, "json", False)),
            selected_attempts=quarantine_attempts,
        )
    if (
        not bool(getattr(args, "apply", False))
        and not bool(getattr(args, "json", False))
        and runtime_status.get("reconcile_apply_blockers")
    ):
        _print_reconcile_runtime_warning(runtime_status, campaign)
    deep_blockers = (
        list(runtime_status.get("reconcile_apply_blockers") or [])
        + _deep_reconcile_quiescence_blockers(campaign)
        if deep_verify
        else []
    )
    if deep_blockers:
        _print_reconcile_apply_blocked(
            campaign,
            title="Deep Verification Safety",
            reasons=deep_blockers,
            next_actions=[
                "stop the daemon and wait for scheduler ownership to become conclusive",
                "rerun reconcile --deep-verify",
            ],
        )
        return 9

    inspection_stream = sys.stderr
    print("Checking campaign recovery...", file=inspection_stream, flush=True)
    if bool(getattr(args, "verbose", False)) or deep_verify or bool(
        getattr(args, "json", False)
    ):
        print(
            "Verification: " + verification_level,
            file=inspection_stream,
            flush=True,
        )
        print(
            "Scientific payload hashing: "
            + ("enabled" if deep_verify else "disabled"),
            file=inspection_stream,
            flush=True,
        )
        print(
            "Inspecting committed artefact chains...",
            file=inspection_stream,
            flush=True,
        )
    try:
        artifact_snapshot = build_committed_artifact_snapshot(
            campaign,
            verification_level=verification_level,
            progress_stream=(sys.stderr if deep_verify else None),
        )
    except Exception as exc:
        print(
            "committed artefact verification failed: " + _reconcile_plain_text(exc),
            file=sys.stderr,
        )
        return 9
    transaction_recovery = inspect_reconcile_transaction_recovery(
        campaign,
        artifact_snapshot=artifact_snapshot,
    )
    transaction_recovery_results: List[Dict[str, Any]] = []
    if bool(getattr(args, "apply", False)) and bool(
        transaction_recovery.get("recoverable", False)
    ):
        for _ in range(4):
            try:
                result = apply_reconcile_transaction_recovery(
                    campaign,
                    transaction_recovery,
                    artifact_snapshot=artifact_snapshot,
                )
            except Exception as exc:
                print(
                    "interrupted reconcile transaction could not be recovered: "
                    + _reconcile_plain_text(exc),
                    file=sys.stderr,
                )
                return 9
            transaction_recovery_results.append(dict(result))
            if result.get("transaction_id") and result.get("disposition"):
                try:
                    from .daemon.journal import append_event

                    append_event(
                        _campaign_paths(campaign)["journal"],
                        "reconcile_transaction_recovered",
                        phase=(str(result.get("phase") or "") or None),
                        iteration=int(result.get("iteration") or 0),
                        transaction_id=str(result["transaction_id"]),
                        prior_status=str(
                            result.get("transaction_status") or ""
                        ),
                        disposition=str(result["disposition"]),
                        reason=str(result.get("reason") or ""),
                        scheduler_jobs_submitted=0,
                    )
                except Exception:
                    pass
            try:
                artifact_snapshot = build_committed_artifact_snapshot(
                    campaign,
                    verification_level=verification_level,
                    progress_stream=(sys.stderr if deep_verify else None),
                )
            except Exception as exc:
                print(
                    "campaign authority could not be rechecked after transaction recovery: "
                    + _reconcile_plain_text(exc),
                    file=sys.stderr,
                )
                return 9
            transaction_recovery = inspect_reconcile_transaction_recovery(
                campaign,
                artifact_snapshot=artifact_snapshot,
            )
            if not bool(transaction_recovery.get("recoverable", False)):
                break
        if bool(transaction_recovery.get("recoverable", False)):
            print(
                "interrupted reconcile transaction recovery did not converge",
                file=sys.stderr,
            )
            return 9
    recovery_record = transaction_recovery.get("record")
    active_recovery_id = None
    if bool(transaction_recovery.get("recoverable", False)):
        active_recovery_id = str(
            transaction_recovery.get("transaction_id")
            or (
                recovery_record.get("transaction_id")
                if isinstance(recovery_record, Mapping)
                else ""
            )
            or ""
        ) or None
    terminal_recoveries: List[Dict[str, Any]] = []
    try:
        report = propose_recovery(
            campaign,
            allow_fresh_init_on_nonempty=bool(
                getattr(args, "allow_fresh_init", False)
            ),
            _active_reconcile_transaction_id=active_recovery_id,
            artifact_snapshot=artifact_snapshot,
            verification_level=verification_level,
        )
    except (ArtefactSnapshotError, ValueError) as exc:
        print(
            "reconcile inspection became stale or invalid: " + str(exc),
            file=sys.stderr,
        )
        return 9
    if report.active_submission_intents:
        terminal_recoveries, terminal_blockers = (
            _resolve_terminal_submission_intents_for_apply(
                campaign,
                report.active_submission_intents,
                persist_terminal_receipts=bool(
                    getattr(args, "apply", False)
                ),
            )
        )
        report.scheduler_terminal_blockers = [
            dict(item) for item in terminal_blockers
        ]
        if terminal_recoveries and not terminal_blockers:
            report = propose_recovery(
                campaign,
                allow_fresh_init_on_nonempty=bool(
                    getattr(args, "allow_fresh_init", False)
                ),
                _active_reconcile_transaction_id=active_recovery_id,
                _terminal_submission_intents=terminal_recoveries,
                artifact_snapshot=artifact_snapshot,
                verification_level=verification_level,
            )
            _apply_terminal_intent_recovery_to_report(
                report,
                terminal_recoveries,
                persist_for_apply=bool(getattr(args, "apply", False)),
            )
    if transaction_recovery_results:
        report.reconcile_transaction_recovery = {
            "state": "recovered",
            "recoverable": False,
            "results": [dict(item) for item in transaction_recovery_results],
            "disposition": str(
                transaction_recovery_results[-1].get("disposition") or ""
            ),
            "reason": str(
                transaction_recovery_results[-1].get("reason") or ""
            ),
            "state_backup_status": str(
                transaction_recovery_results[-1].get("state_backup_status")
                or ""
            ),
        }
    else:
        report.reconcile_transaction_recovery = (
            dict(transaction_recovery)
            if str(transaction_recovery.get("state") or "") != "none"
            else None
        )
    config_path = campaign / "campaign.yaml"
    config = None
    config_review = None
    if config_path.is_file():
        try:
            config = CampaignConfig.from_yaml(config_path)
            force_phase = None
            if force_resubmit_array:
                partial_for_review = getattr(report, "partial_array_recovery", None)
                if isinstance(partial_for_review, dict):
                    try:
                        candidate_phase = CampaignPhase(str(partial_for_review.get("phase")))
                    except Exception:
                        candidate_phase = None
                    if candidate_phase is not None and supports_partial_array_recovery(candidate_phase):
                        force_phase = candidate_phase
                if force_phase is None and supports_partial_array_recovery(
                    report.proposed_state.phase
                ):
                    force_phase = report.proposed_state.phase
            config_review = review_config_changes(
                campaign,
                config,
                report.proposed_state,
                initialise_missing=False,
                force_resubmit_array_phase=force_phase,
                force_retrain_ferebus=retrain_ferebus,
            )
        except Exception as exc:
            if bool(getattr(args, "apply", False)):
                print("campaign config could not be loaded: " + str(exc), file=sys.stderr)
                return 8
            print("campaign config could not be reviewed: " + str(exc), file=sys.stderr)
    _apply_runtime_config_to_recovered_state(report, config)
    if config is not None:
        revalidation_inspection = inspect_aimall_quality_revalidation(
            campaign,
            config=config,
        )
        if revalidation_inspection.get("state") in {
            "eligible",
            "resumable",
            "complete",
        }:
            report.aimall_quality_revalidation = revalidation_inspection
    if (
        bool(getattr(args, "apply", False))
        and report.deep_verification_required
        and verification_level != "deep"
    ):
        _print_reconcile_apply_blocked(
            campaign,
            title="Deep Verification Safety",
            reasons=[
                "deep verification is required before campaign authority can be "
                "reconstructed from committed artefacts"
            ],
            next_actions=[
                _campaign_command(campaign, "reconcile", " --deep-verify")
            ],
        )
        return 10
    revalidation = getattr(report, "aimall_quality_revalidation", None)
    revalidation_ready = bool(
        isinstance(revalidation, Mapping)
        and revalidation.get("state") in {"eligible", "resumable"}
        and revalidation.get("eligible") is True
    )
    if bool(getattr(args, "apply", False)) and revalidation_ready:
        if config is None:
            print(
                "refusing AIMAll quality revalidation because campaign.yaml is missing",
                file=sys.stderr,
            )
            return 8
        if config_review is not None and config_review.blocked_changes:
            print(
                "refusing AIMAll quality revalidation because campaign.yaml has locked changes",
                file=sys.stderr,
            )
            print(format_config_review(config_review), file=sys.stderr)
            return 8
        preliminary_contract = recovery_contract_status(
            campaign,
            report.proposed_state,
            verification=verification_level,
            artifact_snapshot=artifact_snapshot,
        )
        preliminary_blockers = _reconcile_hard_blockers(
            report,
            preliminary_contract,
        )
        if preliminary_blockers:
            _print_reconcile_apply_blocked(
                campaign,
                title="AIMAll Quality Revalidation",
                reasons=preliminary_blockers,
                next_actions=["inspect blockers before applying revalidation"],
            )
            return 9
        try:
            artifact_snapshot.assert_anchors_unchanged(campaign)
            revalidation_result = apply_aimall_quality_revalidation(
                campaign,
                config=config,
            )
            from .daemon.journal import append_event

            try:
                append_event(
                    _campaign_paths(campaign)["journal"],
                    "aimall_quality_revalidated",
                    phase=CampaignPhase.AIMALL.value,
                    iteration=int(revalidation_result["iteration"]),
                    candidate_count=int(revalidation_result["candidate_count"]),
                    prior_reason=str(revalidation_result["old_reason"]),
                    allocation_generation=int(
                        revalidation_result["allocation_generation"]
                    ),
                    ledger_path=str(revalidation_result["ledger_path"]),
                    scheduler_jobs_submitted=0,
                )
            except Exception as journal_exc:
                print(
                    "AIMAll quality revalidation completed, but its journal event "
                    "could not be written: " + _reconcile_plain_text(journal_exc),
                    file=sys.stderr,
                )
            artifact_snapshot = build_committed_artifact_snapshot(
                campaign,
                verification_level=verification_level,
                progress_stream=(sys.stderr if deep_verify else None),
            )
            report = propose_recovery(
                campaign,
                allow_fresh_init_on_nonempty=bool(
                    getattr(args, "allow_fresh_init", False)
                ),
                artifact_snapshot=artifact_snapshot,
                verification_level=verification_level,
            )
            report.aimall_quality_revalidation = revalidation_result
            _apply_runtime_config_to_recovered_state(report, config)
            config_review = review_config_changes(
                campaign,
                config,
                report.proposed_state,
                initialise_missing=False,
                force_retrain_ferebus=retrain_ferebus,
            )
        except Exception as exc:
            print(
                "AIMAll quality revalidation did not complete: "
                + _reconcile_plain_text(exc),
                file=sys.stderr,
            )
            print(
                "No backend job was submitted. Rerun reconcile to inspect or "
                "resume the recorded correction transaction.",
                file=sys.stderr,
            )
            return 9
    stop_guard: Optional[Dict[str, Any]] = None
    try:
        stop_guard = _prepare_reconcile_stop_guard(
            campaign,
            report.proposed_state,
        )
    except Exception as exc:
        stop_problem = (
            "the user stop request is incompatible with the proposed "
            "recovery: "
            + _reconcile_plain_text(exc)
        )
        runtime_status = dict(runtime_status)
        runtime_status["reconcile_apply_blockers"] = list(
            runtime_status.get("reconcile_apply_blockers") or []
        ) + [stop_problem]
        if bool(getattr(args, "apply", False)):
            _print_reconcile_apply_blocked(
                campaign,
                title="Stop Control",
                reasons=[stop_problem],
                next_actions=[
                    _campaign_command(campaign, "reconcile", " --verbose")
                ],
            )
            return 9
    if stop_guard is not None and stop_guard.get("description"):
        runtime_status = dict(runtime_status)
        runtime_status["_retained_stop_description"] = str(
            stop_guard["description"]
        )
    if bool(getattr(args, "apply", False)):
        try:
            artifact_snapshot.assert_anchors_unchanged(campaign)
        except ArtefactSnapshotError as exc:
            print("reconcile snapshot became stale: " + str(exc), file=sys.stderr)
            return 9
    target = write_proposed_state(campaign, report)
    if bool(getattr(args, "json", False)):
        contract_status = recovery_contract_status(
            campaign,
            report.proposed_state,
            verification=verification_level,
            artifact_snapshot=artifact_snapshot,
        )
        print(
            json.dumps(
                _reconcile_decision_payload(
                    campaign,
                    report,
                    contract_status,
                    proposed_state_path=target,
                    runtime_status=runtime_status,
                ),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    contract_status = recovery_contract_status(
        campaign,
        report.proposed_state,
        verification=verification_level,
        artifact_snapshot=artifact_snapshot,
    )
    target_canonical = target.with_name(DEFAULT_STATE_FILENAME)
    if not bool(getattr(args, "apply", False)):
        _print_reconcile_operator_report(
            campaign,
            report,
            contract_status,
            mode="dry_run",
            proposed_state_path=target,
            config_review=config_review,
            runtime_status=runtime_status,
            verbose=bool(getattr(args, "verbose", False)),
        )
        return 0

    candidate_labels = _reconcile_valid_candidates(campaign, contract_status, report)
    hard_blockers = _reconcile_hard_blockers(report, contract_status)
    cleanable = _reconcile_cleanable_reasons(report)
    if (
        report.proposed_state.phase is CampaignPhase.HALTED
        and candidate_labels
        and not hard_blockers
        and not cleanable
    ):
        print(
            "refusing --apply because reconcile found runnable recovery "
            "candidate(s) but selected HALTED:",
            file=sys.stderr,
        )
        for item in candidate_labels:
            print("  - " + str(item), file=sys.stderr)
        print(
            "This is an internal recovery-selection inconsistency; inspect "
            "the candidate contracts before restarting.",
            file=sys.stderr,
        )
        return 9

    if report.proposed_state.phase is CampaignPhase.DONE:
        print(
            "refusing --apply because proposed state is "
            + report.proposed_state.phase.value,
            file=sys.stderr,
        )
        return 9
    resolved_intents: List[Dict[str, Any]] = [
        dict(item)
        for item in (
            getattr(report, "receipt_backed_intent_repairs", []) or []
        )
    ]
    if report.active_submission_intents:
        scheduler_resolved, blocking_intents = _resolve_terminal_submission_intents_for_apply(
            campaign,
            report.active_submission_intents,
            pre_submit_stale_seconds=(
                int(config.runtime.lease_stale_seconds)
                if config is not None
                else 900
            ),
        )
        resolved_intents.extend(scheduler_resolved)
        if blocking_intents:
            _print_reconcile_apply_blocked(
                campaign,
                title="Submission Intents",
                reasons=[
                    str(item.get("phase"))
                    + "@"
                    + str(item.get("iteration"))
                    + " job_id="
                    + str(item.get("job_id"))
                    + " expected_job_name="
                    + str(item.get("expected_job_name"))
                    + ": "
                    + str(item.get("reason"))
                    for item in blocking_intents
                ],
                next_actions=[
                    _campaign_command(campaign, "stop", " --cancel-jobs"),
                    _campaign_command(campaign, "reconcile"),
                ],
            )
            return 9
        if resolved_intents:
            print(
                "Applying reconcile plan: retiring completed submission records...",
                file=sys.stderr,
                flush=True,
            )
            report.active_submission_intents = []
            report.unsafe_reasons = [
                reason
                for reason in report.unsafe_reasons
                if not str(reason).startswith("active submission intent(s) present:")
                and not str(reason).startswith(
                    "scheduler-inconclusive prepared scratch task(s)"
                )
            ]
            report.blocking_artifacts = [
                value
                for value in report.blocking_artifacts
                if str(value) != "prepared scratch ownership"
            ]
            resolved_keys = {
                (
                    str(item.get("phase") or ""),
                    int(item.get("iteration") or 0),
                    str(item.get("job_id") or ""),
                )
                for item in resolved_intents
            }
            for scratch_record in report.scratch_inventory:
                key = (
                    str(scratch_record.get("phase") or ""),
                    int(scratch_record.get("iteration") or 0),
                    str(scratch_record.get("job_id") or ""),
                )
                if key in resolved_keys:
                    scratch_record["status"] = "terminal_intent"
    if report.active_submission_intents:
        print(
            "refusing --apply while active submission intents are present",
            file=sys.stderr,
        )
        return 9
    if config is None:
        print("refusing --apply because campaign.yaml is missing", file=sys.stderr)
        return 8
    if config_review is not None and config_review.blocked_changes:
        print("refusing --apply because campaign.yaml has locked changes", file=sys.stderr)
        print(format_config_review(config_review), file=sys.stderr)
        return 8
    if retrain_ferebus and report.proposed_state.phase not in (
        CampaignPhase.INITIAL_FEREBUS,
        CampaignPhase.FEREBUS,
    ):
        print(
            "refusing --retrain-ferebus because reconcile did not select a "
            "FEREBUS recovery phase",
            file=sys.stderr,
        )
        return 9
    partial_array = getattr(report, "partial_array_recovery", None)
    force_array_phase = report.proposed_state.phase
    force_array_iteration = int(report.proposed_state.iteration)
    if isinstance(partial_array, dict):
        try:
            force_array_phase = CampaignPhase(str(partial_array.get("phase")))
            force_array_iteration = int(partial_array.get("iteration"))
        except Exception:
            force_array_phase = report.proposed_state.phase
            force_array_iteration = int(report.proposed_state.iteration)
    scalar_archive_required = bool(
        isinstance(partial_array, Mapping)
        and str(partial_array.get("phase") or "")
        in {
            CampaignPhase.PHASE_A_DIVERSITY.value,
            CampaignPhase.PHASE_B_DIVERSITY.value,
        }
        and str(partial_array.get("publication_disposition") or "")
        == "archive_and_retry"
    )
    if force_resubmit_array:
        if not (
            isinstance(partial_array, dict)
            and supports_partial_array_recovery(force_array_phase)
        ):
            print(
                "refusing --force-resubmit-array-tasks because reconcile did "
                "not identify a supported current array phase",
                file=sys.stderr,
            )
            return 9
        if int(partial_array.get("n_complete") or 0) > 0 and not archive_existing_array_outputs:
            print(
                "refusing --force-resubmit-array-tasks because completed task "
                "outputs already exist; rerun with "
                "--archive-existing-array-task-outputs to preserve them before "
                "the full array is resubmitted",
                file=sys.stderr,
            )
            return 9

    cleanable_reasons = {
        "dangling model staging directories exist",
        ".DATA/SCRIPTS contains sbatch scripts",
        "stale ARIADNE publication",
    }
    data_staging_archive_mode = None
    archive_staging_refusal_reasons: List[str] = []
    if ".DATA/STAGING is non-empty" in report.unsafe_reasons:
        ok_to_archive_staging, staging_reason = ferebus_reentry_can_archive_data_staging(
            campaign,
            report.proposed_state,
            verification=verification_level,
            artifact_snapshot=artifact_snapshot,
        )
        if ok_to_archive_staging:
            cleanable_reasons.add(".DATA/STAGING is non-empty")
            data_staging_archive_mode = "ferebus_reentry"
        elif archive_staging_requested:
            archive_blockers = _operator_staging_archive_blockers(
                campaign,
                report,
                runtime_status,
            )
            if not archive_blockers:
                cleanable_reasons.add(".DATA/STAGING is non-empty")
                data_staging_archive_mode = "user"
            else:
                archive_staging_refusal_reasons = list(archive_blockers)
                report.notes.append(
                    ".DATA/STAGING cannot be archived automatically: "
                    + "; ".join(str(item) for item in archive_blockers)
                )
        else:
            report.notes.append(
                ".DATA/STAGING cannot be archived automatically: " + staging_reason
            )
    if "dangling reference-data staging directories exist" in report.unsafe_reasons:
        ok_to_archive_reference_data, reference_data_reason = reference_data_staging_can_archive_for_reconcile(
            campaign,
            report.proposed_state,
            verification=verification_level,
            artifact_snapshot=artifact_snapshot,
        )
        if ok_to_archive_reference_data:
            cleanable_reasons.add("dangling reference-data staging directories exist")
        else:
            report.notes.append(
                "dangling reference-data staging cannot be archived automatically: "
                + reference_data_reason
            )
    uncleanable = [
        reason
        for reason in report.unsafe_reasons
        if reason not in cleanable_reasons
    ]
    if uncleanable:
        reasons = list(uncleanable)
        if archive_staging_refusal_reasons:
            reasons.extend(
                "archive staging blocker: " + str(reason)
                for reason in archive_staging_refusal_reasons
            )
        _print_reconcile_apply_blocked(
            campaign,
            title="Unsafe Artefacts",
            reasons=reasons,
            next_actions=["inspect blockers before restarting"],
        )
        return 9

    cleanable_now = [
        _reconcile_human_reason(item)
        for item in _reconcile_cleanable_reasons(report)
    ]
    if data_staging_archive_mode is not None:
        cleanable_now = [
            item for item in cleanable_now if not item.startswith(".DATA/STAGING")
        ]
        cleanable_now.append("archive .DATA/STAGING")
    if scalar_archive_required:
        cleanable_now.append("archive incomplete scalar diversity publication")
    if cleanable_now:
        print(
            "Applying reconcile plan: cleaning reviewed temporary data...",
            file=sys.stderr,
            flush=True,
        )

    original_report = report
    transaction: Optional[ReconcileTransaction] = None
    planned_operations = [
        "validate_recovery_contract",
        "prepare_ferebus_candidate_recovery",
        "repair_current_pointers",
        "write_recovered_state",
        "update_config_lock",
        "publish_intent_transitions",
    ]
    ferebus_staging_recovery = getattr(
        report,
        "ferebus_staging_recovery",
        None,
    )
    if (
        isinstance(ferebus_staging_recovery, Mapping)
        and str(ferebus_staging_recovery.get("disposition") or "")
        == "archived_terminal_producer"
    ):
        planned_operations.insert(0, "restore_ferebus_producer_staging")
    if _aimall_upstream_gaussian_recovery_evidence(report) is not None:
        planned_operations.insert(0, "restore_gaussian_task_membership")
    completed_staging_records = list(
        (getattr(report, "completed_staging_retirement", {}) or {}).get(
            "eligible", []
        )
    )
    completed_staging_tombstones = list(
        (getattr(report, "completed_staging_retirement", {}) or {}).get(
            "pending_tombstones", []
        )
    )
    if completed_staging_records or completed_staging_tombstones:
        planned_operations.insert(0, "retire_completed_staging")
    if scalar_archive_required:
        planned_operations.insert(0, "archive_scalar_diversity_publication")
    if cleanable_now or retrain_ferebus or force_resubmit_array:
        planned_operations.insert(0, "archive_reconcile_evidence")
    transaction_id = secrets.token_hex(16)
    mutation_plan = [
        {
            "operation_id": "operation-" + str(index).zfill(4),
            "kind": "reconcile_operation",
            "name": str(name),
            "archive_identity": transaction_id,
        }
        for index, name in enumerate(planned_operations)
    ]
    try:
        artifact_snapshot.assert_anchors_unchanged(campaign)
        transaction = begin_reconcile_transaction(
            campaign,
            proposed_phase=report.proposed_state.phase.value,
            proposed_iteration=int(report.proposed_state.iteration),
            planned_operations=planned_operations,
            intent_transitions=resolved_intents,
            campaign_uid=str(report.proposed_state.campaign_uid),
            source_authority_anchor_sha256=str(
                artifact_snapshot.anchor_sha256
            ),
            mutation_plan=mutation_plan,
            transaction_id=transaction_id,
        )
        transaction.set_status("MUTATING")
        mutation_result = _perform_reconcile_apply_mutations(
            campaign,
            report,
            transaction=transaction,
            retrain_ferebus=retrain_ferebus,
            force_resubmit_array=force_resubmit_array,
            partial_array=partial_array,
            force_array_phase=force_array_phase,
            force_array_iteration=int(force_array_iteration),
            archive_existing_array_outputs=archive_existing_array_outputs,
            data_staging_archive_mode=data_staging_archive_mode,
            verification=verification_level,
            artifact_snapshot=artifact_snapshot,
            campaign_config=config,
        )
    except Exception as exc:
        _fail_reconcile_transaction(
            transaction,
            "reconcile evidence archival failed: " + type(exc).__name__ + ": " + str(exc),
        )
        print(
            "could not archive reconcile evidence losslessly: "
            + _reconcile_plain_text(exc),
            file=sys.stderr,
        )
        return 9

    ferebus_retrain_archive = list(mutation_result["ferebus_retrain_archive"])
    archived_array_outputs = list(mutation_result["archived_array_outputs"])
    refreshed = mutation_result["refreshed_array_ledger"]
    archived_scripts = list(mutation_result["archived_scripts"])
    archived = list(mutation_result["archived_data_staging"])
    removed_model_staging = list(mutation_result["archived_model_staging"])
    ferebus_staging_restore = mutation_result["ferebus_staging_restore"]
    archived_reference_data_staging = list(
        mutation_result["archived_reference_data_staging"]
    )
    removed = list(mutation_result["archived_reentry_staging"])
    archived_ariadne_publication = list(
        mutation_result["archived_ariadne_publication"]
    )
    archived_scalar_diversity_publication = list(
        mutation_result["archived_scalar_diversity_publication"]
    )
    retired_completed_staging = dict(
        mutation_result["retired_completed_staging"]
    )
    if any(
        (
            retired_completed_staging.get("retired"),
            archived_ariadne_publication,
            archived_scalar_diversity_publication,
            ferebus_retrain_archive,
            refreshed,
        )
    ):
        print(
            "Applying reconcile plan: reviewed temporary data updated...",
            file=sys.stderr,
            flush=True,
        )
    cleanup_paths_already_done = (
        list(archived_scripts)
        + list(archived_array_outputs)
        + list(archived)
        + list(removed_model_staging)
        + (
            []
            if not isinstance(ferebus_staging_restore, Mapping)
            else [
                str(path)
                for path in (
                    ferebus_staging_restore.get("restored_path"),
                    ferebus_staging_restore.get("archived_input_path"),
                )
                if path
            ]
        )
        + list(archived_reference_data_staging)
        + list(ferebus_retrain_archive)
        + list(removed)
        + list(archived_ariadne_publication)
        + list(archived_scalar_diversity_publication)
        + list(retired_completed_staging.get("retired") or [])
        + list(retired_completed_staging.get("preserved") or [])
    )

    if original_report.proposed_state.phase is CampaignPhase.HALTED:
        try:
            report = _propose_recovery_after_cleanup(
                campaign,
                allow_fresh_init_on_nonempty=bool(
                    getattr(args, "allow_fresh_init", False)
                ),
                active_reconcile_transaction_id=(
                    str(transaction.payload["transaction_id"])
                    if transaction is not None
                    else None
                ),
                artifact_snapshot=artifact_snapshot,
                verification_level=verification_level,
                terminal_recoveries=terminal_recoveries,
            )
        except Exception as exc:
            _fail_reconcile_transaction(
                transaction,
                "post-archive recovery inspection failed: "
                + type(exc).__name__
                + ": "
                + str(exc),
            )
            print(
                "post-archive recovery inspection failed: "
                + _reconcile_plain_text(exc),
                file=sys.stderr,
            )
            _print_cleanup_already_happened(cleanup_paths_already_done)
            return 9
        _apply_retry_phase_after_cleaned_halt(report, original_report)
        _apply_runtime_config_to_recovered_state(report, config)
        target = write_proposed_state(campaign, report)
        if config is not None:
            try:
                config_review = review_config_changes(
                    campaign,
                    config,
                    report.proposed_state,
                    initialise_missing=False,
                    force_retrain_ferebus=retrain_ferebus,
                )
            except Exception as exc:
                _fail_reconcile_transaction(
                    transaction,
                    "post-archive config review failed: " + str(exc),
                )
                print("campaign config could not be reviewed: " + str(exc), file=sys.stderr)
                _print_cleanup_already_happened(cleanup_paths_already_done)
                return 8
        if report.unsafe_reasons:
            _fail_reconcile_transaction(
                transaction,
                "post-archive recovery remained unsafe: " + "; ".join(report.unsafe_reasons),
            )
            print(
                "refusing --apply because cleanup did not produce a safe recovery proposal:",
                file=sys.stderr,
            )
            for reason in report.unsafe_reasons:
                print("  - " + reason, file=sys.stderr)
            _print_cleanup_already_happened(cleanup_paths_already_done)
            return 9
        if report.proposed_state.phase in (CampaignPhase.HALTED, CampaignPhase.DONE):
            _fail_reconcile_transaction(
                transaction,
                "post-archive recovery selected " + report.proposed_state.phase.value,
            )
            print(
                "refusing --apply because proposed state is "
                + report.proposed_state.phase.value
                + " after cleanup",
                file=sys.stderr,
            )
            _print_cleanup_already_happened(cleanup_paths_already_done)
            return 9
        if config_review is not None and config_review.blocked_changes:
            _fail_reconcile_transaction(
                transaction,
                "post-archive config review found locked changes",
            )
            print("refusing --apply because campaign.yaml has locked changes", file=sys.stderr)
            print(format_config_review(config_review), file=sys.stderr)
            _print_cleanup_already_happened(cleanup_paths_already_done)
            return 8
        recomputed_status = recovery_contract_status(
            campaign,
            report.proposed_state,
            verification=verification_level,
            artifact_snapshot=artifact_snapshot,
        )
        print(
            "Applying reconcile plan: recovery recomputed and validated...",
            file=sys.stderr,
            flush=True,
        )

    restored_bootstrap_handoff: List[str] = []
    try:
        restored_bootstrap_handoff = restore_archived_bootstrap_handoff(
            campaign,
            report,
        )
        cleanup_paths_already_done.extend(restored_bootstrap_handoff)
        if transaction is not None:
            transaction.record_paths(
                "restore_bootstrap_handoff",
                restored_bootstrap_handoff,
            )
    except Exception as exc:
        _fail_reconcile_transaction(
            transaction,
            "bootstrap handoff restoration failed: " + type(exc).__name__ + ": " + str(exc),
        )
        print(
            "refusing --apply because archived bootstrap staging could not be restored:",
            file=sys.stderr,
        )
        print("  - " + _reconcile_plain_text(exc), file=sys.stderr)
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 9

    contract_error = _reconcile_apply_contract_error(
        campaign,
        report.proposed_state,
        verification=verification_level,
        artifact_snapshot=artifact_snapshot,
    )
    if contract_error is not None:
        _fail_reconcile_transaction(
            transaction,
            "final state/artefact contract failed: " + str(contract_error),
        )
        if cleanup_paths_already_done:
            print(
                "Final campaign-state validation failed: "
                + _reconcile_plain_text(contract_error),
                file=sys.stderr,
            )
            _print_cleanup_already_happened(cleanup_paths_already_done)
        else:
            _print_reconcile_apply_blocked(
                campaign,
                title="Final State/Artefact Contract",
                reasons=[contract_error],
                next_actions=["inspect recovery contract before restarting"],
            )
        return 9

    ferebus_recovery_request = None
    candidate_recovery = getattr(report, "ferebus_candidate_recovery", None)
    if (
        isinstance(candidate_recovery, dict)
        and candidate_recovery
        and not retrain_ferebus
    ):
        try:
            from .daemon.ferebus_candidate_recovery import (
                prepare_recovery_request,
            )

            ferebus_recovery_request = prepare_recovery_request(
                campaign,
                candidate=candidate_recovery,
                campaign_uid=str(report.proposed_state.campaign_uid),
                phase=report.proposed_state.phase.value,
                iteration=int(report.proposed_state.iteration),
                reference_data_version=int(
                    report.proposed_state.reference_data_version
                ),
            )
        except Exception as exc:
            _fail_reconcile_transaction(
                transaction,
                "FEREBUS candidate recovery request failed: "
                + type(exc).__name__
                + ": "
                + str(exc),
            )
            print(
                "refusing --apply because the existing FEREBUS candidate "
                "could not be bound to a recovery request: "
                + _reconcile_plain_text(exc),
                file=sys.stderr,
            )
            return 9

    try:
        if transaction is None:
            raise RuntimeError("reconcile transaction was not created")
        commit_plan = build_reconcile_commit_plan(
            campaign,
            transaction_id=str(transaction.payload["transaction_id"]),
            proposed_state=report.proposed_state,
            config=config,
            intent_transitions=resolved_intents,
            artifact_snapshot=artifact_snapshot,
        )
        transaction.prepare_commit(commit_plan)
    except Exception as exc:
        reason = (
            "reconcile commit plan preparation failed: "
            + type(exc).__name__
            + ": "
            + str(exc)
        )
        _fail_reconcile_transaction(transaction, reason)
        print(
            "refusing --apply because the recovery commit plan could not be "
            "prepared:",
            file=sys.stderr,
        )
        print("  - " + _reconcile_plain_text(exc), file=sys.stderr)
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 9

    pointer_snapshots: List[Dict[str, Any]] = []
    try:
        state_train_version = int(report.proposed_state.reference_data_version)
        if state_train_version >= 0:
            from .versioning.reference_data import ReferenceDataVersioning

            reference_versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
            pointer_snapshots.append(
                snapshot_version_pointer(
                    campaign,
                    reference_versioning,
                    label="reference_data",
                    requested_version=state_train_version,
                )
            )
            transaction.record_pointer_snapshots(pointer_snapshots)
            reference_versioning.ensure_current(state_train_version)
        state_model_version = int(report.proposed_state.models_version)
        if state_model_version >= 0:
            model_versioning = TrainedModelVersioning(trained_models_dir(campaign))
            pointer_snapshots.append(
                snapshot_version_pointer(
                    campaign,
                    model_versioning,
                    label="trained_models",
                    requested_version=state_model_version,
                )
            )
            transaction.record_pointer_snapshots(pointer_snapshots)
            model_versioning.ensure_current(state_model_version)
    except Exception as exc:
        pointer_errors = _restore_reconcile_pointer_snapshots(campaign, pointer_snapshots)
        reason = (
            "current pointer repair failed: "
            + type(exc).__name__
            + ": "
            + str(exc)
        )
        if pointer_errors:
            reason += "; pointer rollback failed: " + "; ".join(pointer_errors)
        _fail_reconcile_transaction(transaction, reason)
        print(
            "refusing --apply because committed current pointers could not be "
            "repaired:",
            file=sys.stderr,
        )
        print("  - " + _reconcile_plain_text(exc), file=sys.stderr)
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 9

    try:
        if transaction is None:
            raise RuntimeError("reconcile transaction was not created")
        backup_path = publish_reconcile_state_backup(campaign, transaction)
    except Exception as exc:
        pointer_errors = _restore_reconcile_pointer_snapshots(campaign, pointer_snapshots)
        reason = "state backup failed: " + type(exc).__name__ + ": " + str(exc)
        if pointer_errors:
            reason += "; pointer rollback failed: " + "; ".join(pointer_errors)
        _fail_reconcile_transaction(transaction, reason)
        print(_reconcile_plain_text(reason), file=sys.stderr)
        return 9
    try:
        if stop_guard is not None:
            _recheck_reconcile_stop_guard(
                campaign,
                stop_guard,
                report.proposed_state,
            )
        write_state(target_canonical, report.proposed_state)
    except Exception as exc:
        pointer_errors = _restore_reconcile_pointer_snapshots(campaign, pointer_snapshots)
        reason = "state write failed: " + type(exc).__name__ + ": " + str(exc)
        if pointer_errors:
            reason += "; pointer rollback failed: " + "; ".join(pointer_errors)
        _fail_reconcile_transaction(transaction, reason)
        print(
            "failed to write recovered state.json: "
            + _reconcile_plain_text(exc),
            file=sys.stderr,
        )
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 9
    try:
        config_commit = commit_plan.get("config_lock")
        if not isinstance(config_commit, Mapping):
            raise ValueError("reconcile config-lock commit plan is missing")
        publish_reconcile_config_target(campaign, config_commit)
    except Exception as exc:
        try:
            _restore_state_from_backup_atomic(target_canonical, backup_path)
            restore_message = "restored the previous state.json"
        except Exception as restore_exc:
            restore_message = (
                "could not restore the previous state.json atomically from "
                + str(backup_path)
                + ": "
                + type(restore_exc).__name__
                + ": "
                + str(restore_exc)
            )
        pointer_errors = _restore_reconcile_pointer_snapshots(campaign, pointer_snapshots)
        if pointer_errors:
            restore_message += "; pointer rollback failed: " + "; ".join(pointer_errors)
        _fail_reconcile_transaction(
            transaction,
            "config lock update failed: " + type(exc).__name__ + ": " + str(exc),
        )
        print(
            "failed to update config lock after writing state.json; "
            + _reconcile_plain_text(restore_message)
            + ": "
            + _reconcile_plain_text(exc),
            file=sys.stderr,
        )
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 8
    try:
        _publish_reconcile_intent_transitions(
            campaign,
            resolved_intents,
            report.proposed_state,
            commit_plan=commit_plan,
        )
    except Exception as exc:
        reason = (
            "recovered state committed, but intent publication remains incomplete: "
            + type(exc).__name__
            + ": "
            + str(exc)
        )
        _fail_reconcile_transaction(transaction, reason)
        _print_reconcile_follow_up_required(
            campaign,
            report.proposed_state,
            problem="completed submission bookkeeping could not be finalised",
            next_command=_campaign_command(campaign, "reconcile"),
            next_effect="preview the remaining bookkeeping repair before applying it",
            verbose_detail=reason if bool(getattr(args, "verbose", False)) else None,
        )
        return 9
    try:
        _complete_reconcile_scheduler_cancellation_stop(
            campaign,
            report.proposed_state,
            [
                dict(item)
                for item in (
                    getattr(
                        report,
                        "scheduler_cancellation_recovery",
                        [],
                    )
                    or []
                )
                if isinstance(item, Mapping)
            ],
        )
    except Exception as exc:
        reason = (
            "recovered scheduler work was committed, but the matching stop "
            "request could not be completed: "
            + type(exc).__name__
            + ": "
            + str(exc)
        )
        _fail_reconcile_transaction(transaction, reason)
        _print_reconcile_follow_up_required(
            campaign,
            report.proposed_state,
            problem=(
                "scheduler recovery was applied, but stop-control "
                "bookkeeping remains incomplete"
            ),
            next_command=_campaign_command(campaign, "reconcile"),
            next_effect="finish the recorded stop-control repair before resuming",
            verbose_detail=(
                reason if bool(getattr(args, "verbose", False)) else None
            ),
        )
        return 9
    try:
        applied_proposal_path = _rename_existing_timestamped(target, ".applied-")
    except Exception as exc:
        applied_proposal_path = None
        if transaction is not None:
            transaction.add_warning(
                "could not archive applied proposal: "
                + type(exc).__name__
                + ": "
                + str(exc)
            )
    try:
        from .daemon.journal import append_event

        if archived:
            append_event(
                campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
                "staging_archived",
                phase=report.proposed_state.phase.value,
                iteration=int(report.proposed_state.iteration),
                mode=str(data_staging_archive_mode),
                archived_staging_paths=archived,
            )
        if restored_bootstrap_handoff:
            append_event(
                campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
                "staging_restored_from_archive",
                phase=report.proposed_state.phase.value,
                iteration=int(report.proposed_state.iteration),
                restored_paths=restored_bootstrap_handoff,
                source_path=(
                    report.bootstrap_handoff.get("path")
                    if report.bootstrap_handoff
                    else None
                ),
                source_phase=(
                    report.bootstrap_handoff.get("phase")
                    if report.bootstrap_handoff
                    else None
                ),
                n_total=(
                    report.bootstrap_handoff.get("n_total")
                    if report.bootstrap_handoff
                    else None
                ),
            )
        if isinstance(ferebus_recovery_request, dict):
            recovery_candidate = getattr(
                original_report,
                "ferebus_candidate_recovery",
                None,
            )
            append_event(
                campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
                "ferebus_candidate_recovery_prepared",
                phase=report.proposed_state.phase.value,
                iteration=int(report.proposed_state.iteration),
                reference_data_version=int(
                    report.proposed_state.reference_data_version
                ),
                candidate_id=str(
                    ferebus_recovery_request.get("candidate_id") or ""
                ),
                source_path=str(
                    ferebus_recovery_request.get("source_path") or ""
                ),
                request_sha256=str(
                    ferebus_recovery_request.get("request_sha256") or ""
                ),
                candidate_kind=(
                    str(recovery_candidate.get("candidate_kind") or "")
                    if isinstance(recovery_candidate, Mapping)
                    else ""
                ),
                n_warned=(
                    int(recovery_candidate.get("n_warned") or 0)
                    if isinstance(recovery_candidate, Mapping)
                    else 0
                ),
                n_warnings=(
                    int(recovery_candidate.get("n_warnings") or 0)
                    if isinstance(recovery_candidate, Mapping)
                    else 0
                ),
            )
        if archived_ariadne_publication:
            append_event(
                campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
                "ariadne_publication_archived",
                phase=report.proposed_state.phase.value,
                iteration=int(report.proposed_state.iteration),
                reason="reconcile_postprocess_recovery",
                archive_path=str(archived_ariadne_publication[0]),
                n_files=int(
                    len(
                        list(
                            (getattr(original_report, "ariadne_publication_recovery", {}) or {}).get(
                                "files", []
                            )
                        )
                    )
                ),
            )
        ariadne_recovery = getattr(report, "ariadne_results_recovery", None)
        ariadne_event_fields = (
            {
                "ariadne_expected_tasks": int(
                    ariadne_recovery.get("expected_tasks") or 0
                ),
                "ariadne_accepted_tasks": int(
                    ariadne_recovery.get("accepted_tasks") or 0
                ),
                "ariadne_rejected_tasks": int(
                    ariadne_recovery.get("rejected_tasks") or 0
                ),
                "ariadne_missing_rejected_outputs": int(
                    ariadne_recovery.get("missing_rejected_outputs") or 0
                ),
                "ariadne_tasks_resubmitted": int(
                    ariadne_recovery.get("tasks_resubmitted") or 0
                ),
            }
            if isinstance(ariadne_recovery, Mapping)
            else {}
        )
        aimall_recovery = getattr(
            report,
            "aimall_postprocess_recovery",
            None,
        )
        aimall_event_fields = (
            {
                "aimall_completed_outputs": int(
                    aimall_recovery.get("logical_total") or 0
                ),
                "aimall_tasks_resubmitted": 0,
                "aimall_validation_disposition": str(
                    aimall_recovery.get("validation") or ""
                ),
            }
            if isinstance(aimall_recovery, Mapping)
            else {}
        )
        partial_recovery = getattr(report, "partial_array_recovery", None)
        scalar_event_fields = (
            {
                "diversity_selected_geometries": int(
                    partial_recovery.get("selected_count") or 0
                ),
                "diversity_tasks_resubmitted": 0,
                "diversity_retry_tasks": int(
                    partial_recovery.get("n_retry") or 0
                ),
                "diversity_publication_disposition": str(
                    partial_recovery.get("publication_disposition") or "adopt"
                ),
                "diversity_ordering_classification": str(
                    partial_recovery.get("ordering_classification")
                    or "canonical"
                ),
                "diversity_producer_job_id": str(
                    partial_recovery.get("producer_job_id") or ""
                ),
                "diversity_publication_archive_path": (
                    archived_scalar_diversity_publication[0]
                    if archived_scalar_diversity_publication
                    else None
                ),
            }
            if isinstance(partial_recovery, Mapping)
            and str(partial_recovery.get("phase") or "")
            in {
                CampaignPhase.PHASE_A_DIVERSITY.value,
                CampaignPhase.PHASE_B_DIVERSITY.value,
            }
            else {}
        )
        scheduler_recoveries = [
            dict(item)
            for item in (
                getattr(report, "scheduler_cancellation_recovery", []) or []
            )
            if isinstance(item, Mapping)
        ]
        scheduler_event_fields = (
            {
                "scheduler_completed_task_candidates": sum(
                    int(item.get("n_completed") or 0)
                    for item in scheduler_recoveries
                ),
                "scheduler_retry_tasks": sum(
                    int(item.get("n_retry") or 0)
                    for item in scheduler_recoveries
                ),
                "scheduler_original_job_ids": sorted(
                    {
                        str(item.get("job_id") or "")
                        for item in scheduler_recoveries
                        if item.get("job_id")
                    }
                ),
                "scheduler_tasks_resubmitted": 0,
            }
            if scheduler_recoveries
            else {}
        )
        terminal_ariadne = getattr(
            report,
            "ariadne_terminal_postprocess_recovery",
            None,
        )
        terminal_ariadne_event_fields = (
            {
                "ariadne_terminal_postprocess": True,
                "ariadne_scheduler_completed_candidates": int(
                    terminal_ariadne.get("n_scheduler_completed") or 0
                ),
                "ariadne_scheduler_failed_candidates": int(
                    terminal_ariadne.get("n_scheduler_failed") or 0
                ),
                "ariadne_original_job_id": str(
                    terminal_ariadne.get("producer_job_id") or ""
                ),
                "ariadne_tasks_resubmitted": 0,
                "ariadne_validation_disposition": str(
                    terminal_ariadne.get("validation") or ""
                ),
            }
            if isinstance(terminal_ariadne, Mapping)
            else {}
        )
        environment_binding_repair = (
            _reconcile_environment_config_binding_repair(
                campaign,
                report.proposed_state,
                scheduler_ownership_clear=True,
            )
        )
        append_event(
            campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
            "reconcile_applied",
            phase=report.proposed_state.phase.value,
            iteration=int(report.proposed_state.iteration),
            n_allowed_config_changes=(
                len(config_review.allowed_changes) if config_review is not None else 0
            ),
            n_removed_stale_paths=len(removed) + len(removed_model_staging),
            n_archived_staging_paths=len(archived) + len(archived_reference_data_staging),
            n_retired_completed_staging=int(
                retired_completed_staging.get("n_retired") or 0
            ),
            n_deleted_completed_staging=int(
                retired_completed_staging.get("n_deleted") or 0
            ),
            n_preserved_completed_staging=int(
                retired_completed_staging.get("n_preserved") or 0
            ),
            retired_completed_staging_paths=list(
                retired_completed_staging.get("retired") or []
            ),
            archived_staging_path=(archived[0] if archived else None),
            archived_reference_data_staging_paths=archived_reference_data_staging,
            ferebus_staging_disposition=(
                str(
                    (getattr(report, "ferebus_staging_recovery", {}) or {}).get(
                        "disposition"
                    )
                    or ""
                )
                or None
            ),
            ferebus_staging_source_transaction=(
                (getattr(report, "ferebus_staging_recovery", {}) or {}).get(
                    "source_transaction_id"
                )
            ),
            ferebus_original_job_ids=(
                list(
                    (getattr(report, "ferebus_staging_recovery", {}) or {}).get(
                        "source_job_ids", []
                    )
                )
            ),
            ferebus_scheduler_completed_candidates=int(
                (getattr(report, "ferebus_staging_recovery", {}) or {}).get(
                    "scheduler_completed_candidates", 0
                )
            ),
            ferebus_known_retry_candidates=int(
                (getattr(report, "ferebus_staging_recovery", {}) or {}).get(
                    "known_retry_candidates", 0
                )
            ),
            n_archived_scripts_paths=len(archived_scripts),
            archived_scripts_path=(archived_scripts[0] if archived_scripts else None),
            archived_ariadne_publication_path=(
                archived_ariadne_publication[0]
                if archived_ariadne_publication
                else None
            ),
            recomputed_after_transient_cleanup=(
                original_report.proposed_state.phase is CampaignPhase.HALTED
            ),
            recovery_source_phase=(
                original_report.last_phase_in_journal
                or original_report.proposed_state.phase.value
            ),
            recovery_selected_phase=report.proposed_state.phase.value,
            recovery_reason=str(report.decision or ""),
            environment_binding_repair_kind=(
                None
                if not isinstance(environment_binding_repair, Mapping)
                or str(environment_binding_repair.get("disposition") or "")
                in {"current", "unbound_first_start"}
                else "campaign_config_generation_binding_advanced"
            ),
            **ariadne_event_fields,
            **aimall_event_fields,
            **scalar_event_fields,
            **scheduler_event_fields,
            **terminal_ariadne_event_fields,
        )
    except Exception:
        pass
    if transaction is not None:
        try:
            transaction.resolve(
                status="COMMITTED",
                disposition="committed",
                reason="reconcile apply completed",
            )
        except Exception as exc:
            detail = type(exc).__name__ + ": " + str(exc)
            _print_reconcile_follow_up_required(
                campaign,
                report.proposed_state,
                problem=(
                    "the recovery transaction record could not be finalised; "
                    "the software-environment update was not attempted"
                ),
                next_command=_campaign_command(campaign, "reconcile"),
                next_effect="finish the recorded recovery transaction before resuming",
                verbose_detail=(detail if bool(getattr(args, "verbose", False)) else None),
            )
            return 9
    try:
        (
            environment_transition,
            environment_transition_deferred,
        ) = _advance_environment_after_reconcile(
            campaign,
            report.proposed_state,
            config,
        )
    except Exception as exc:
        detail = type(exc).__name__ + ": " + str(exc)
        problem, next_command, next_effect = (
            _reconcile_environment_failure_guidance(
                campaign,
                report.proposed_state,
                detail,
            )
        )
        _print_reconcile_follow_up_required(
            campaign,
            report.proposed_state,
            problem=problem,
            next_command=next_command,
            next_effect=next_effect,
            verbose_detail=(detail if bool(getattr(args, "verbose", False)) else None),
        )
        return 13
    final_contract_status = recovery_contract_status(
        campaign,
        report.proposed_state,
        verification=verification_level,
        artifact_snapshot=artifact_snapshot,
    )
    _print_reconcile_applied_operator_report(
        campaign,
        report,
        final_contract_status,
        backup_path=backup_path,
        applied_proposal_path=applied_proposal_path,
        removed=removed,
        removed_model_staging=removed_model_staging,
        archived_scripts=archived_scripts,
        archived=archived,
        archived_reference_data_staging=archived_reference_data_staging,
        retired_completed_staging=retired_completed_staging,
        restored_bootstrap_handoff=restored_bootstrap_handoff,
        archived_ariadne_publication=archived_ariadne_publication,
        archived_scalar_diversity_publication=(
            archived_scalar_diversity_publication
        ),
        ferebus_retrain_archive=ferebus_retrain_archive,
        ferebus_staging_restore=ferebus_staging_restore,
        archived_array_outputs=archived_array_outputs,
        refreshed_array_ledger=refreshed,
        config_review=config_review,
        resolved_intents=resolved_intents,
        environment_transition=environment_transition,
        environment_transition_deferred=environment_transition_deferred,
        verbose=bool(getattr(args, "verbose", False)),
    )
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(
        args.campaign_dir,
        require_campaign_yaml=False,
    )
    if bool(getattr(args, "list_event_types", False)):
        for event_type in sorted(KNOWN_EVENT_TYPES):
            print(event_type)
        return 0
    journal_path = _campaign_paths(campaign)["journal"]
    if not journal_path.exists():
        print("Timeline")
        print("  no journal yet at " + str(journal_path))
        return 4
    since = getattr(args, "since", None)
    if since is not None:
        try:
            from datetime import datetime

            datetime.fromisoformat(str(since).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            print(
                "invalid --since timestamp; use ISO-8601, for example "
                "2026-07-20T12:30:00+00:00",
                file=sys.stderr,
            )
            return 2
    last_n = getattr(args, "last_n", None)
    if last_n is not None and int(last_n) <= 0:
        print("journal: error: --last-n must be a positive integer", file=sys.stderr)
        return 2
    try:
        iterator = read_events(
            journal_path,
            since=since,
            event_type=getattr(args, "event_type", None) or None,
        )
        if last_n is None:
            events = list(iterator)
        else:
            events = list(deque(iterator, maxlen=int(last_n)))
    except (JournalCorruptionError, OSError) as exc:
        print("journal is corrupt or unreadable: " + str(exc), file=sys.stderr)
        return 5
    except ValueError as exc:
        print("journal contains an invalid timestamp: " + str(exc), file=sys.stderr)
        return 5
    if bool(getattr(args, "json", False)) or bool(getattr(args, "raw", False)):
        for event in events:
            print(json.dumps(event, sort_keys=True, allow_nan=False))
    else:
        if (
            not bool(getattr(args, "verbose", False))
            and not (getattr(args, "event_type", None) or [])
        ):
            events = _aggregate_journal_events(events)
        print(
            _format_journal_events(
                events,
                verbose=bool(getattr(args, "verbose", False)),
            ),
            end="",
        )
    return 0


def _resolve_init_campaign_dir(raw_campaign_dir: Optional[str]) -> Path:
    if raw_campaign_dir:
        campaign = Path(raw_campaign_dir).expanduser().resolve()
    else:
        campaign = Path.cwd().resolve()
    if campaign.exists() and not campaign.is_dir():
        raise CampaignDirResolutionError(
            "campaign path is not a directory: " + str(campaign)
        )
    campaign.mkdir(parents=True, exist_ok=True)
    return campaign


def _resolve_init_source(
    campaign: Path,
    raw_source: Optional[str],
) -> Path:
    from .operator_paths import (
        reject_operator_input_symlinks,
        resolve_campaign_input_path,
    )

    source = resolve_campaign_input_path(
        campaign,
        raw_source if raw_source else "pool.xyz",
    )
    reject_operator_input_symlinks(source)
    if not source.exists():
        raise FileNotFoundError(
            "source trajectory does not exist: "
            + str(source)
            + ". Place pool.xyz in the campaign directory or pass --source PATH."
        )
    if not source.is_file():
        raise FileNotFoundError("source trajectory is not a file: " + str(source))
    return source


def _read_pool_candidate(source: Path) -> tuple[List[Any], str]:
    from ichor.core.files.xyz import Trajectory
    from .versioning.manifest import sha256_file

    from .operator_paths import reject_operator_input_symlinks

    reject_operator_input_symlinks(source)
    try:
        trajectory = Trajectory(source)
        trajectory.read()
        frames = [frame.copy() for frame in trajectory]
    except Exception as exc:
        raise ValueError("pool source is unreadable: " + str(source)) from exc
    if not frames:
        raise ValueError("pool source contains no geometries: " + str(source))
    expected = tuple(str(value) for value in frames[0].types_extended)
    for index, frame in enumerate(frames):
        observed = tuple(str(value) for value in frame.types_extended)
        coordinates = np.asarray(frame.coordinates, dtype=float)
        if observed != expected:
            raise ValueError(
                "pool frame " + str(index) + " atom order/type mismatch"
            )
        if coordinates.shape != (len(expected), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("pool frame " + str(index) + " has invalid coordinates")
    return frames, sha256_file(source)


def _prompt_bootstrap_alf(
    split: str,
    path: Path,
    atom_names: Sequence[str],
) -> Sequence[int]:
    print("")
    print("Detected bootstrap CSV: " + str(path))
    print("Molecule atom order:")
    for index, atom_name in enumerate(atom_names, start=1):
        print("  " + str(index).rjust(3) + "  " + str(atom_name))
    required = 2 if len(atom_names) == 2 else 3
    print(
        "Enter " + str(required) + " 1-based ALF atom numbers for " + split
        + " (central, x-axis" + (", xy-plane" if required == 3 else "") + "):"
    )
    try:
        raw = input("> ").strip()
    except EOFError as exc:
        raise ValueError(
            "CSV bootstrap requires an ALF; provide bootstrap/alf.yaml for "
            "non-interactive initialisation"
        ) from exc
    values = raw.split()
    if len(values) != required:
        raise ValueError(
            "CSV bootstrap ALF must contain exactly " + str(required) + " integers"
        )
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise ValueError("CSV bootstrap ALF entries must be integers") from exc


def _print_bootstrap_plan(plan: Any, *, verbose: bool = False) -> None:
    labels = {
        "train": "Training",
        "int_val": "Internal validation",
        "ext_val": "External validation",
    }
    if not verbose:
        print("Bootstrap inputs")
        print("  custom inputs: " + ("enabled" if plan.custom_enabled else "not supplied"))
        print("  training geometries: " + str(plan.effective_training_count))
        print(
            "  internal validation geometries: "
            + str(plan.configured_targets["int_val"])
        )
        print(
            "  external validation geometries: "
            + str(plan.configured_targets["ext_val"])
        )
        return
    print("Bootstrap discovery complete")
    print("")
    print("  custom bootstrap: " + ("enabled" if plan.custom_enabled else "disabled"))
    print("  pool SHA-256: " + str(plan.pool_sha256))
    for split in ("train", "int_val", "ext_val"):
        source = plan.sources.get(split)
        print("")
        print(labels[split] + ":")
        if split == "train" and plan.model is not None:
            print("  source: bootstrap/model_krig")
            print("  model files: " + str(len(plan.model.files)))
            print("  model rows: " + str(plan.model.training_count))
            print("  atoms: " + ", ".join(str(value) for value in plan.model.atoms))
            print(
                "  properties: "
                + ", ".join(str(value) for value in plan.model.properties)
            )
            print("  configured target: ignored")
            print("  diversity top-up: 0")
            continue
        print("  source: " + ("ICHOR diversity" if source is None else str(source.path)))
        print("  supplied: " + str(0 if source is None else source.count))
        print("  configured target: " + str(plan.configured_targets[split]))
        print("  Diversity top-up: " + str(plan.diversity_deficits[split]))
        if source is not None and source.alf_zero_indexed is not None:
            print(
                "  ALF (1-based): "
                + repr([int(value) + 1 for value in source.alf_zero_indexed])
            )
    print("")
    print(
        "Final planned allocation: training="
        + str(plan.effective_training_count)
        + ", internal_validation="
        + str(plan.configured_targets["int_val"])
        + ", external_validation="
        + str(plan.configured_targets["ext_val"])
    )


def _bootstrap_split_count_summary(counts: Mapping[str, Any]) -> str:
    return (
        "training="
        + str(int(counts.get("train", 0)))
        + ", internal_validation="
        + str(int(counts.get("int_val", 0)))
        + ", external_validation="
        + str(int(counts.get("ext_val", 0)))
    )


def _confirm_bootstrap_plan(*, assume_yes: bool) -> bool:
    if assume_yes:
        print("Bootstrap plan accepted by --yes.")
        return True
    try:
        response = input("Proceed? [y/N] ").strip().lower()
    except EOFError:
        return False
    return response in {"y", "yes"}


def _import_pool_impl(args: argparse.Namespace, campaign: Path, source: Path) -> int:
    """Pull the user's MD trajectory into the campaign's canonical
    pool location and write a SHA-pinned manifest next to it.

    Refuses to overwrite an existing pool unless --force is passed --
    overwriting wipes the SHA the previously-committed iterations were
    pinned to, so we make the user say it out loud.
    """
    from .acquisition.trajectory_pool import TrajectoryPool
    from .versioning.manifest import sha256_file

    existing_manifest = campaign / ".DATA" / "TRAJECTORY" / "pool.manifest.json"
    canonical_pool = campaign / "pool.xyz"
    if (
        existing_manifest.is_file()
        and canonical_pool.is_file()
        and source.is_file()
        and sha256_file(source) != sha256_file(canonical_pool)
        and (
            (campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME).is_file()
            or stateful_campaign_artifacts(campaign)
        )
    ):
        print(
            "refusing to replace campaign pool.xyz after stateful daemon "
            "artefacts exist; create a new campaign directory",
            file=sys.stderr,
        )
        return 14

    if bool(getattr(args, "force", False)):
        state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
        if existing_manifest.exists():
            blockers: List[str] = []
            if state_path.is_file():
                blockers.append(str(state_path.relative_to(campaign)))
            blockers.extend(stateful_campaign_artifacts(campaign))
            if blockers:
                print(
                    "refusing to overwrite the trajectory pool after the campaign "
                    "has stateful daemon artefacts.",
                    file=sys.stderr,
                )
                print(
                    "Changing the pool would invalidate recorded frame IDs and "
                    "trajectory SHA handoffs. Create a new campaign directory "
                    "instead, or reconcile/remove state explicitly.",
                    file=sys.stderr,
                )
                print("blocking artefacts:", file=sys.stderr)
                for item in blockers[:20]:
                    print("  " + str(item), file=sys.stderr)
                if len(blockers) > 20:
                    print("  ...", file=sys.stderr)
                return 16

    try:
        pool = TrajectoryPool.import_from(
            source, campaign, overwrite=bool(args.force),
        )
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        print("Re-run with --force to overwrite the existing pool.", file=sys.stderr)
        return 13
    except (FileNotFoundError, ValueError) as exc:
        print("trajectory import failed: " + str(exc), file=sys.stderr)
        return 14
    # Journal the import. This is intentionally verbatim: the daemon no
    # longer filters the initial trajectory pool during import.
    try:
        from .daemon.journal import append_event
        journal_path = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        append_event(
            journal_path, "trajectory_pool_imported",
            phase=CampaignPhase.INIT.value,
            iteration=0,
            n_imported=int(pool.n_frames()),
        )
    except Exception:
        # Robust journal write; never block the import on this.
        pass
    print(
        "Imported pool: "
        + str(pool.canonical_path)
        + " (" + str(pool.n_frames()) + " frames, "
        + str(pool.manifest.natoms) + " atoms, SHA " + pool.sha256[:12] + "...)"
    )
    return 0


def _trajectory_pool_summary(campaign: Path) -> Dict[str, Any]:
    from .acquisition.trajectory_pool import TrajectoryPool

    pool = TrajectoryPool.load(campaign)
    return {
        "status": "ok",
        "frames": int(pool.n_frames()),
        "atoms": int(pool.manifest.natoms),
        "sha256": str(pool.sha256),
        "path": str(pool.canonical_path),
    }


def _pool_feasibility_summary(campaign: Path, config: CampaignConfig) -> Dict[str, Any]:
    try:
        from .daemon.pool_feasibility import evaluate_pool_feasibility

        result = evaluate_pool_feasibility(
            campaign,
            config,
            verify_pool_payload=False,
        )
        return result.to_dict()
    except Exception as exc:
        return {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }


def _print_pool_summary(summary: Dict[str, Any], *, verbose: bool = False) -> None:
    if str(summary.get("status")) == "ok":
        print(
            "  trajectory pool: ok, frames="
            + str(summary.get("frames"))
            + ", atoms="
            + str(summary.get("atoms"))
            + (
                ", sha=" + str(summary.get("sha256", ""))[:12]
                if verbose
                else ""
            )
        )
    elif str(summary.get("status")) == "missing":
        print("  trajectory pool: missing")
    else:
        print("  trajectory pool: invalid - " + str(summary.get("error", "unknown")))


def _print_pool_feasibility(
    summary: Dict[str, Any],
    *,
    file=None,
    verbose: bool = False,
) -> None:
    stream = sys.stdout if file is None else file
    if summary.get("error"):
        print(
            "  pool feasibility: unavailable - " + str(summary.get("error")),
            file=stream,
        )
        return
    print(
        "  pool feasibility: "
        + ("ok" if bool(summary.get("ok")) else "failed")
        + ", frames="
        + str(summary.get("pool_n_frames"))
        + ", required="
        + str(summary.get("required_pool_frames")),
        file=stream,
    )
    if verbose and summary.get("expression"):
        print("    " + str(summary.get("expression")), file=stream)


def _bootstrap_fresh_campaign_state(
    campaign: Path,
    config: CampaignConfig,
) -> Dict[str, Any]:
    from .layout import reject_legacy_campaign_layout

    try:
        reject_legacy_campaign_layout(campaign)
    except RuntimeError as exc:
        raise CampaignBootstrapError(
            "campaign uses a rejected legacy or mixed sampling layout; "
            "start a new campaign or inspect it with `ichor-al-daemon reconcile "
            "--campaign-dir "
            + str(campaign)
            + "`: "
            + str(exc)
        ) from exc
    paths = _campaign_paths(campaign)
    paths["data"].mkdir(parents=True, exist_ok=True)
    (campaign / ".DATA" / "STAGING").mkdir(parents=True, exist_ok=True)
    active_learning_dir(campaign).mkdir(parents=True, exist_ok=True)

    state_path = paths["state"]
    if state_path.is_file():
        try:
            state = read_state(state_path)
        except (StateSchemaError, json.JSONDecodeError) as exc:
            raise CampaignBootstrapError(
                "state.json is invalid; run `ichor-al-daemon reconcile --campaign-dir "
                + str(campaign)
                + "` before reinitialising: "
                + str(exc)
            ) from exc
        review = review_config_changes(
            campaign,
            config,
            state,
            initialise_missing=True,
        )
        if review.changed:
            formatted = format_config_review(review)
            raise CampaignBootstrapError(
                "campaign.yaml differs from config_lock.json; use reconcile "
                "to approve safe edits before continuing."
                + (("\n" + formatted) if formatted else "")
            )
        return {
            "state_status": "already_initialised",
            "state": state,
            "config_lock_status": "ok",
            "state_path": str(state_path),
            "config_lock_path": str(config_lock_path(campaign)),
        }

    artefacts = stateful_campaign_artifacts(campaign)
    if artefacts:
        lines = [
            "state.json is missing but this campaign has stateful run artefacts.",
            "Refusing to create a fresh state because that could overwrite provenance.",
            "Run:",
            "  " + _campaign_command(campaign, "reconcile"),
            "Stateful artefacts:",
        ]
        lines.extend("  - " + str(item) for item in artefacts[:12])
        if len(artefacts) > 12:
            lines.append("  ... " + str(len(artefacts) - 12) + " more")
        raise CampaignBootstrapError("\n".join(lines))

    lock_path = config_lock_path(campaign)
    if lock_path.is_file():
        lock_payload = read_config_lock(campaign)
        state = fresh_campaign_state(max_iterations=int(config.campaign.max_iterations))
        lock_uid = str(lock_payload.get("campaign_uid") or "")
        if lock_uid:
            state.campaign_uid = lock_uid
        review = review_config_changes(
            campaign,
            config,
            state,
            initialise_missing=False,
        )
        if review.changed:
            formatted = format_config_review(review)
            raise CampaignBootstrapError(
                "config_lock.json already exists and does not match campaign.yaml; "
                "refusing to initialise a fresh state without user review."
                + (("\n" + formatted) if formatted else "")
            )
        lock_status = "ok"
    else:
        # The first lock and state share one campaign identity. The lock is
        # published first so a failed write cannot leave an unlocked state.
        state = fresh_campaign_state(max_iterations=int(config.campaign.max_iterations))
        ensure_config_lock(
            campaign,
            config,
            campaign_uid=str(state.campaign_uid),
        )
        lock_status = "created"
    write_state(state_path, state)
    return {
        "state_status": "created",
        "state": state,
        "config_lock_status": lock_status,
        "state_path": str(state_path),
        "config_lock_path": str(config_lock_path(campaign)),
    }


def cmd_init(args: argparse.Namespace) -> int:
    """Inspect, confirm, and atomically admit campaign bootstrap inputs."""
    try:
        campaign = _resolve_init_campaign_dir(getattr(args, "campaign_dir", None))
    except CampaignDirResolutionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        from .campaign_yaml import CampaignYamlError, prepare_campaign_yaml

        _campaign, config, prepared_campaign_yaml = prepare_campaign_yaml(campaign)
    except CampaignYamlError as exc:
        print("campaign.yaml initialisation failed: " + str(exc), file=sys.stderr)
        return 15
    except Exception as exc:
        print("campaign.yaml initialisation failed: " + str(exc), file=sys.stderr)
        return 15

    from .custom_bootstrap import (
        BootstrapInputError,
        commit_bootstrap_plan,
        inspect_bootstrap_inputs,
    )

    pool_summary: Dict[str, Any]
    raw_source = getattr(args, "source", None)
    source: Optional[Path] = None
    existing_pool = False
    if raw_source is None and not bool(getattr(args, "force", False)):
        try:
            existing_pool = _trajectory_pool_summary(campaign).get("status") == "ok"
        except Exception:
            existing_pool = False
    if not existing_pool:
        try:
            source = _resolve_init_source(
                campaign,
                raw_source,
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    try:
        if source is None:
            from .acquisition.trajectory_pool import TrajectoryPool

            existing = TrajectoryPool.load(campaign)
            pool_frames = existing.to_atoms_list()
            pool_sha = str(existing.sha256)
        else:
            pool_frames, pool_sha = _read_pool_candidate(source)
        plan = inspect_bootstrap_inputs(
            campaign,
            config,
            pool_frames,
            pool_sha256=pool_sha,
            alf_prompt=(
                None if bool(getattr(args, "yes", False))
                else _prompt_bootstrap_alf
            ),
        )
    except (BootstrapInputError, ValueError, FileNotFoundError) as exc:
        print("campaign bootstrap inspection failed: " + str(exc), file=sys.stderr)
        return 18

    # Initialisation is an explicit review boundary. Always show the complete
    # bootstrap allocation before the user confirms it.
    _print_bootstrap_plan(plan, verbose=True)
    if not _confirm_bootstrap_plan(assume_yes=bool(getattr(args, "yes", False))):
        print("Campaign initialisation cancelled; no campaign files were changed.")
        return 19

    from .daemon.state import atomic_write_text

    atomic_write_text(campaign / "campaign.yaml", prepared_campaign_yaml)

    if source is not None:
        state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
        if not state_path.is_file() and stateful_campaign_artifacts(campaign):
            try:
                _bootstrap_fresh_campaign_state(campaign, config)
            except CampaignBootstrapError as exc:
                print("campaign bootstrap failed: " + str(exc), file=sys.stderr)
                return 16
        import_rc = _import_pool_impl(args, campaign, source)
        if import_rc != 0:
            return import_rc

    try:
        pool_summary = _trajectory_pool_summary(campaign)
    except FileNotFoundError:
        pool_summary = {"status": "missing"}
    except Exception as exc:
        pool_summary = {
            "status": "invalid",
            "error": type(exc).__name__ + ": " + str(exc),
        }
    feasibility_summary: Dict[str, Any] = {}
    if str(pool_summary.get("status")) == "ok":
        if str(pool_summary.get("sha256") or "") != str(plan.pool_sha256):
            print(
                "campaign bootstrap failed: imported pool SHA does not match the "
                "confirmed bootstrap plan",
                file=sys.stderr,
            )
            return 14
        from .daemon.pool_feasibility import evaluate_pool_feasibility_manifest

        feasibility_summary = evaluate_pool_feasibility_manifest(
            int(pool_summary["frames"]),
            config,
            {
                "effective_qm_targets": plan.effective_qm_targets,
                "diversity_deficits": dict(plan.diversity_deficits),
                "supplied_counts": plan.supplied_counts,
                "excluded_pool_frame_ids": list(plan.excluded_pool_frame_ids),
                "model": (
                    None if plan.model is None else {
                        "training_count": int(plan.model.training_count)
                    }
                ),
            },
        ).to_dict()
        if not bool(feasibility_summary.get("ok", False)):
            print("campaign bootstrap failed: trajectory pool is infeasible", file=sys.stderr)
            _print_pool_feasibility(
                feasibility_summary,
                file=sys.stderr,
                verbose=True,
            )
            return 17

    try:
        bootstrap_manifest = commit_bootstrap_plan(plan)
    except Exception as exc:
        print(
            "campaign bootstrap evidence commit failed: "
            + type(exc).__name__ + ": " + str(exc),
            file=sys.stderr,
        )
        return 18

    try:
        bootstrap = _bootstrap_fresh_campaign_state(campaign, config)
    except CampaignBootstrapError as exc:
        print("campaign bootstrap failed: " + str(exc), file=sys.stderr)
        return 16

    state = bootstrap["state"]
    try:
        from .daemon.journal import append_event

        append_event(
            campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
            "bootstrap_inputs_confirmed",
            phase=state.phase.value,
            iteration=int(state.iteration),
            custom_bootstrap=bool(config.campaign.custom_bootstrap),
            plan_identity_sha256=str(
                bootstrap_manifest.get("plan_identity_sha256") or ""
            ),
            supplied_counts=dict(bootstrap_manifest.get("supplied_counts") or {}),
            diversity_deficits=dict(
                bootstrap_manifest.get("diversity_deficits") or {}
            ),
            model_training_rows=int(
                (bootstrap_manifest.get("model") or {}).get("training_count", 0)
            ),
        )
    except Exception:
        pass
    print("Campaign initialised")
    print("  campaign: " + str(campaign))
    print("  campaign.yaml: valid")
    _print_pool_summary(pool_summary, verbose=True)
    print("  bootstrap inputs: confirmed")
    print(
        "  custom bootstrap: "
        + ("enabled" if bool(bootstrap_manifest.get("custom_bootstrap")) else "disabled")
    )
    print(
        "  supplied bootstrap geometries: "
        + _bootstrap_split_count_summary(
            dict(bootstrap_manifest.get("supplied_counts") or {})
        )
    )
    print(
        "  diversity top-up: "
        + _bootstrap_split_count_summary(
            dict(bootstrap_manifest.get("diversity_deficits") or {})
        )
    )
    print("  campaign schema: " + str(config.schema_version))
    print(
        "  bootstrap identity: "
        + str(bootstrap_manifest.get("plan_identity_sha256", ""))
    )
    if feasibility_summary:
        _print_pool_feasibility(feasibility_summary, verbose=True)
    print(
        "  state.json: "
        + str(bootstrap["state_status"])
        + ", phase="
        + str(state.phase.value)
    )
    print("  config_lock.json: " + str(bootstrap["config_lock_status"]))
    print("")
    if str(pool_summary.get("status")) == "ok":
        print("Next:")
        print("  " + _campaign_command(campaign, "preflight"))
        print("  Live:    " + _campaign_command(campaign, "start"))
        print(
            "  Dry run: "
            + _campaign_command(campaign, "start", " --mode dry_run")
        )
    else:
        print("Next:")
        print(
            "  "
            + _campaign_command(campaign, "init", " --source /path/to/pool.xyz")
        )
    return 0


def cmd_import_pool(args: argparse.Namespace) -> int:
    print(
        "warning: import-pool is deprecated; use 'ichor-al-daemon init' instead.",
        file=sys.stderr,
    )
    return cmd_init(args)


def cmd_config_check(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(getattr(args, "campaign_dir", None))
    try:
        config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    except Exception as exc:
        if bool(getattr(args, "human", False)):
            print("Config check")
            print("Config")
            print("  result: invalid")
            print("  reason: " + type(exc).__name__ + ": " + str(exc))
            print("")
            print("Pool")
            print("  result: not checked because campaign.yaml is invalid")
            print("")
            print("Action")
            print("  correct campaign.yaml, then run:")
            print(
                "  ichor-al-daemon config-check --campaign-dir "
                + shlex.quote(str(campaign))
                + " --human"
            )
            return 2
        print(
            "campaign.yaml invalid: " + type(exc).__name__ + ": " + str(exc),
            file=sys.stderr,
        )
        return 2
    from .ferebus_prior import resolve_ferebus_prior_contract

    prior = resolve_ferebus_prior_contract(config)
    summary = {
        "schema_version": int(config.schema_version),
        "campaign": {
            "system_name": str(config.campaign.system_name),
            "max_iterations": int(config.campaign.max_iterations),
            "sampling_aggressiveness": int(
                config.campaign.sampling_aggressiveness
            ),
            "custom_bootstrap": bool(config.campaign.custom_bootstrap),
        },
        "point_allocation": {
            "bootstrap_training_size": int(
                config.point_allocation.bootstrap_training_size
            ),
            "bootstrap_internal_validation_size": int(
                config.point_allocation.bootstrap_internal_validation_size
            ),
            "bootstrap_external_validation_size": int(
                config.point_allocation.bootstrap_external_validation_size
            ),
            "bootstrap_total_size": int(
                config.point_allocation.bootstrap_total_size
            ),
            "batch_training_size": int(
                config.point_allocation.batch_training_size
            ),
            "batch_internal_validation_size": int(
                config.point_allocation.batch_internal_validation_size
            ),
            "batch_total_size": int(config.point_allocation.batch_total_size),
        },
        "ferebus_physical_prior": {
            "mean_type": int(prior.mean_type),
            "configured_level_of_theory": str(
                config.ferebus.prior_mean_level_of_theory
            ),
            "resolved_level_of_theory": prior.level_of_theory,
            "iqa_deviation_factor": float(prior.iqa_deviation_factor),
            "units": "ha",
            "feature_scaling": bool(prior.feature_scaling),
            "property_scaling": bool(prior.property_scaling),
            "contract_sha256": prior.contract_sha256,
        },
    }
    try:
        summary["pool_feasibility"] = _pool_feasibility_summary(campaign, config)
        from .acquisition.trajectory_pool import TrajectoryPool

        pool_manifest = campaign / ".DATA" / "TRAJECTORY" / "pool.manifest.json"
        if pool_manifest.is_file():
            pool = TrajectoryPool.load(campaign)
            resolve_ferebus_prior_contract(
                config,
                atom_labels=pool.manifest.atom_types,
            )
    except Exception as exc:
        summary["pool_feasibility"] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }
    if bool(getattr(args, "human", False)):
        pool = summary["pool_feasibility"]
        print("Config check")
        print("Config")
        print("  result: valid")
        print("  system: " + str(summary["campaign"]["system_name"]))
        print(
            "  planned active-learning iterations: "
            + str(summary["campaign"]["max_iterations"])
        )
        print("")
        print("Pool")
        if bool(pool.get("ok", False)):
            print("  result: sufficient for the configured campaign")
            print("  frames available: " + str(pool.get("pool_n_frames")))
            print("  frames required: " + str(pool.get("required_pool_frames")))
            print("")
            print("Action")
            print("  no configuration or pool correction is required")
        else:
            print("  result: insufficient or unavailable")
            print(
                "  reason: "
                + str(pool.get("error") or pool.get("expression") or "unknown")
            )
            print("")
            print("Action")
            print("  correct the campaign sizing or import a larger trajectory pool")
    else:
        # No flag intentionally retains the historical machine-readable output.
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if bool(summary["pool_feasibility"].get("ok", False)) else 10


def _preflight_mark(ok: Any, *, warn: bool = False) -> str:
    if warn:
        return "[WARN]"
    return "[OK]" if bool(ok) else "[FAIL]"


def _preflight_check_line(
    label: str,
    ok: Any,
    detail: Any,
    *,
    warn: bool = False,
) -> str:
    return "  " + _preflight_mark(ok, warn=warn) + " " + label + ": " + _format_value(detail)


def _preflight_environment_generation_evidence(
    campaign: Path,
    state: Optional[CampaignState],
    config: Optional[CampaignConfig],
    *,
    scheduler_ownership_clear: bool,
) -> Optional[Dict[str, Any]]:
    """Return the shared read-only environment launch disposition."""
    if state is None or config is None:
        return None
    try:
        from .execution_identity import inspect_environment_generation_launch

        return inspect_environment_generation_launch(
            campaign,
            state=state,
            config=config,
            scheduler_ownership_clear=scheduler_ownership_clear,
        )
    except Exception as exc:
        return {
            "disposition": "invalid",
            "launchable": False,
            "generation": -1,
            "reason": "execution environment evidence is invalid: " + str(exc),
            "error": type(exc).__name__ + ": " + str(exc),
        }


def _preflight_payload(
    campaign: Path,
    avail: Any,
    config_summary: Dict[str, Any],
    feasibility_summary: Dict[str, Any],
    state_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    backend_payload = asdict(avail)
    config_ok = bool(config_summary.get("ok", False))
    feasibility_ok = bool(feasibility_summary.get("ok", False))
    state_payload = dict(state_summary or {"ok": True})
    state_ok = bool(state_payload.get("ok", False))
    base_environment_ready = bool(
        avail.all_present and config_ok and feasibility_ok
    )
    ready = bool(base_environment_ready and state_ok)
    payload: Dict[str, Any] = dict(backend_payload)
    payload.update(
        {
            "schema_version": 1,
            "campaign_dir": str(campaign),
            "backend_availability": backend_payload,
            "campaign_config": dict(config_summary),
            "pool_feasibility": dict(feasibility_summary),
            "campaign_state": state_payload,
            "all_backends_present": bool(avail.all_present),
            "ready": ready,
            "missing_backends": list(avail.missing),
            "missing_backend_message": (
                "" if avail.all_present else missing_backend_message(avail)
            ),
            "next_action": (
                "start live campaign" if ready else "fix failed checks before live start"
            ),
            "_presentation_base_environment_ready": base_environment_ready,
        }
    )
    return payload


def _preflight_reports_intentional_pause(payload: Mapping[str, Any]) -> bool:
    state = payload.get("campaign_state")
    if not isinstance(state, Mapping):
        return False
    return bool(
        str(state.get("condition") or "") == "paused"
        or "campaign is intentionally paused" in str(state.get("error") or "")
    )


def _preflight_failure_details(payload: Dict[str, Any]) -> List[str]:
    details: List[str] = []
    missing = payload.get("missing_backends")
    if isinstance(missing, list):
        for name in missing:
            if name == "profile":
                profile_error = str(payload.get("profile_error") or "").strip()
                details.append(
                    "fix active ICHOR profile"
                    + (": " + profile_error if profile_error else "")
                )
            elif name == "sbatch":
                details.append("make sbatch available on the Slurm login node")
            elif name == "sacct":
                details.append("make sacct available; the daemon needs it for polling")
            elif name == "squeue":
                details.append("make squeue available; the daemon uses it to confirm live Slurm ownership")
            elif name == "qsub":
                details.append(
                    "make qsub available on the Sun Grid Engine submission host"
                )
            elif name == "qacct":
                details.append(
                    "make qacct available; the daemon needs it for accounting"
                )
            elif name == "qstat":
                details.append(
                    "make qstat available; the daemon uses it to confirm live Sun Grid Engine ownership"
                )
            elif name == "qdel":
                details.append(
                    "make qdel available; immediate stop uses it for cancellation"
                )
            elif name == "batch_python":
                details.append(
                    "configure an absolute submitted Python path and install the ICHOR runtime in that environment"
                )
            elif name == "bc":
                details.append("make bc available; pyferebus scripts use it")
            elif name == "gaussian":
                details.append(
                    "configure the machine profile's Gaussian module and executable"
                )
            elif name == "aimall":
                details.append(
                    "configure the machine profile's AIMAll module and executable"
                )
            elif name == "ferebus":
                details.append("configure or install the FEREBUS executable")
            elif name == "ariadne":
                details.append(
                    "install/build ariadne in the configured submitted Python venv"
                )
            elif name == "pyferebus":
                details.append(
                    "install pyferebus in the configured submitted Python venv"
                )
    config = payload.get("campaign_config")
    if isinstance(config, dict) and not bool(config.get("ok", False)):
        details.append(
            "fix campaign.yaml: " + str(config.get("error") or "configuration is not readable")
        )
    pool = payload.get("pool_feasibility")
    if isinstance(pool, dict) and not bool(pool.get("ok", False)):
        details.append(
            "fix trajectory pool feasibility: "
            + str(pool.get("error") or pool.get("expression") or "pool is not usable")
        )
    state = payload.get("campaign_state")
    if isinstance(state, dict) and not bool(state.get("ok", False)):
        condition = str(state.get("condition") or "")
        issues = [
            str(item)
            for item in (state.get("issues") or [])
            if str(item).strip()
        ]
        if (
            not condition
            and _preflight_reports_intentional_pause(payload)
        ):
            details.append(
                str(
                    payload.get("next_action")
                    or "resume the campaign to clear the completed stop"
                )
            )
        elif condition in {"paused", "reconcile_required", "complete"}:
            details.extend(
                (
                    classify_operator_failure(issue).summary
                    if classify_operator_failure(issue).family != "unknown"
                    else issue
                )
                for issue in issues
            )
        else:
            state_error = str(
                state.get("error")
                or state.get("phase")
                or "state is not runnable"
            )
            failure = classify_operator_failure(state_error)
            details.append(
                "campaign launch state: "
                + (
                    failure.summary
                    if failure.family != "unknown"
                    else state_error
                )
            )
    submitted_smoke = payload.get("submitted_environment_smoke")
    if isinstance(submitted_smoke, dict) and not bool(submitted_smoke.get("ok", False)):
        details.append(
            "fix submitted environment smoke: "
            + str(submitted_smoke.get("error") or "submitted job did not verify its runtime")
        )
    return details


def _format_preflight(payload: Dict[str, Any], *, verbose: bool = False) -> str:
    avail = payload.get("backend_availability")
    if not isinstance(avail, dict):
        avail = {}
    config = payload.get("campaign_config")
    if not isinstance(config, dict):
        config = {}
    pool = payload.get("pool_feasibility")
    if not isinstance(pool, dict):
        pool = {}
    state = payload.get("campaign_state")
    if not isinstance(state, dict):
        state = {}

    environment_ready = bool(
        payload.get("all_backends_present", False)
        and config.get("ok") is True
        and pool.get("ok") is True
    )
    submitted_smoke = payload.get("submitted_environment_smoke")
    if isinstance(submitted_smoke, Mapping):
        environment_ready = environment_ready and bool(
            submitted_smoke.get("ok", False)
        )
    condition = str(state.get("condition") or "")
    launch_labels = {
        "ready": "ready",
        "stop_scheduled": "ready; stop boundary retained",
        "paused": "paused",
        "reconcile_required": "reconcile required",
        "halted": "halted",
        "complete": "campaign complete",
        "missing": "not initialised",
        "unreadable": "state unreadable",
        "blocked": "blocked",
    }
    lines: List[str] = []
    lines.extend(
        _section(
            "Preflight",
            [
                ("result", "ready" if payload.get("ready") else "blocked"),
                (
                    "environment readiness",
                    "ready" if environment_ready else "blocked",
                ),
                (
                    "campaign launch",
                    launch_labels.get(
                        condition,
                        "ready" if state.get("ok") else "blocked",
                    ),
                ),
                ("active profile", avail.get("active_profile") or "<unresolved>"),
                ("campaign", payload.get("campaign_dir")),
            ],
        )
    )
    if verbose and avail.get("profile_error"):
        lines.append("  profile error: " + str(avail.get("profile_error")))

    lines.append("")
    lines.append("Scheduler")
    if str(avail.get("scheduler_kind") or "slurm") == "sge":
        for command in ("qsub", "qacct", "qstat", "qdel"):
            lines.append(
                _preflight_check_line(
                    command,
                    avail.get(command),
                    avail.get(command + "_path") or "not found",
                )
            )
    else:
        lines.append(_preflight_check_line("sbatch", avail.get("sbatch"), avail.get("sbatch_path") or "not found"))
        lines.append(_preflight_check_line("sacct", avail.get("sacct"), avail.get("sacct_path") or "not found"))
        lines.append(_preflight_check_line("squeue", avail.get("squeue"), avail.get("squeue_path") or "not found"))
    lines.append(_preflight_check_line("bc", avail.get("bc"), avail.get("bc_path") or "not found"))

    lines.append("")
    lines.append("Python Environment")
    python_path = avail.get("python_executable") or "not configured in profile"
    lines.append(
        _preflight_check_line(
            "configured python",
            bool(avail.get("batch_python")),
            (
                python_path
                + (" (" + str(avail.get("batch_python_version")) + ")" if avail.get("batch_python_version") else "")
                + (": " + str(avail.get("batch_python_error")) if avail.get("batch_python_error") else "")
            ),
        )
    )
    runtime_modules = avail.get("batch_runtime_modules")
    if verbose and isinstance(runtime_modules, (list, tuple)):
        lines.append(
            "  submitted module stack: "
            + (", ".join(str(value) for value in runtime_modules) or "<none>")
        )
    lines.append(
        _preflight_check_line(
            "ariadne",
            avail.get("ariadne"),
            "importable in submitted environment"
            if avail.get("ariadne")
            else avail.get("ariadne_probe_error") or "not importable",
        )
    )
    lines.append(
        _preflight_check_line(
            "pyferebus",
            avail.get("pyferebus"),
            "importable in submitted environment"
            if avail.get("pyferebus")
            else avail.get("pyferebus_probe_error") or "not importable",
        )
    )

    lines.append("")
    lines.append("Quantum Backends")
    gaussian_available = bool(avail.get("gaussian"))
    gaussian_verified = bool(avail.get("gaussian_verified"))
    gaussian_detail = (
        avail.get("gaussian_binary")
        or avail.get("gaussian_probe_error")
        or "not configured"
    )
    if gaussian_available and not gaussian_verified:
        gaussian_detail = (
            str(gaussian_detail)
            + " (jobscript-only; run submitted-environment smoke to verify)"
        )
    lines.append(
        _preflight_check_line(
            "Gaussian submitted environment",
            gaussian_available and gaussian_verified,
            gaussian_detail,
            warn=gaussian_available and not gaussian_verified,
        )
    )
    if (
        verbose
        and gaussian_available
        and not gaussian_verified
        and avail.get("gaussian_probe_error")
    ):
        lines.append(
            "  login-node probe: "
            + str(avail.get("gaussian_probe_error"))
        )
    lines.append(
        _preflight_check_line(
            "AIMAll submitted environment",
            avail.get("aimall_verified", avail.get("aimall")),
            avail.get("aimall_path")
            or avail.get("aimall_probe_error")
            or "not found",
        )
    )

    lines.append("")
    lines.append("FEREBUS")
    lines.append(_preflight_check_line("executable", avail.get("ferebus"), avail.get("ferebus_path") or "not found"))

    submitted_smoke = payload.get("submitted_environment_smoke")
    if isinstance(submitted_smoke, dict):
        lines.append("")
        lines.append("Submitted Environment Smoke")
        smoke_detail = (
            "job " + str(submitted_smoke.get("job_id"))
            if submitted_smoke.get("job_id")
            else str(submitted_smoke.get("error") or "not submitted")
        )
        lines.append(
            _preflight_check_line(
                "compute-node runtime",
                submitted_smoke.get("ok"),
                smoke_detail,
            )
        )
        if submitted_smoke.get("output_path"):
            lines.append("  output: " + str(submitted_smoke.get("output_path")))

    lines.append("")
    lines.append("Campaign Config")
    if config.get("ok"):
        lines.append(_preflight_check_line("campaign.yaml", True, "valid"))
        if verbose and config.get("schema_version") is not None:
            lines.append("  schema version: " + str(config.get("schema_version")))
        if config.get("system_name"):
            lines.append("  system: " + str(config.get("system_name")))
        if verbose and config.get("prior_mean_level_of_theory"):
            lines.append(
                "  FEREBUS prior: type "
                + str(config.get("prior_mean_type"))
                + ", level "
                + str(config.get("prior_mean_level_of_theory"))
                + ", units "
                + str(config.get("prior_mean_units"))
            )
            lines.append(
                "  FEREBUS scaling: features="
                + str(config.get("feature_scaling"))
                + ", properties="
                + str(config.get("property_scaling"))
            )
            lines.append(
                "  FEREBUS prior contract: "
                + str(config.get("prior_mean_contract_sha256"))
            )
    else:
        lines.append(
            _preflight_check_line(
                "campaign.yaml",
                False,
                config.get("error") or "not readable",
            )
        )

    lines.append("")
    lines.append("Campaign State")
    intentionally_paused = _preflight_reports_intentional_pause(payload)
    state_warn = bool(
        condition in {
            "stop_scheduled",
            "paused",
            "reconcile_required",
            "complete",
        }
        or intentionally_paused
    )
    state_detail = state.get("error")
    if not state_detail and state.get("phase") is not None:
        state_detail = (
            str(state.get("phase"))
            + " iteration "
            + str(state.get("iteration"))
        )
    lines.append(
        _preflight_check_line(
            (
                "campaign pause"
                if intentionally_paused
                else (
                    "stop boundary"
                    if condition == "stop_scheduled"
                    else "runnable state"
                )
            ),
            state.get("ok"),
            state_detail or "state is not runnable",
            warn=state_warn,
        )
    )
    scheduler_repoll = payload.get("_presentation_scheduler_repoll")
    if isinstance(scheduler_repoll, Mapping):
        if scheduler_repoll.get("job_id"):
            lines.append(
                _preflight_check_line(
                    "preserved scheduler re-poll",
                    True,
                    str(scheduler_repoll.get("phase"))
                    + " job "
                    + str(scheduler_repoll.get("job_id"))
                    + "; resume will poll existing work and submit no job",
                    warn=True,
                )
            )
        elif verbose and scheduler_repoll.get("error"):
            lines.append(
                "  preserved re-poll error: "
                + str(scheduler_repoll.get("error"))
            )
    ariadne_terminal = payload.get(
        "_presentation_ariadne_terminal_postprocess"
    )
    if isinstance(ariadne_terminal, Mapping):
        if ariadne_terminal.get("error"):
            lines.append(
                _preflight_check_line(
                    "ARIADNE terminal postprocessing",
                    False,
                    "scheduler evidence is invalid"
                    + (
                        ": " + str(ariadne_terminal.get("error"))
                        if verbose
                        else ""
                    ),
                )
            )
        else:
            completed = int(
                ariadne_terminal.get("n_scheduler_completed") or 0
            )
            failed = int(
                ariadne_terminal.get("n_scheduler_failed") or 0
            )
            lines.append(
                _preflight_check_line(
                    "ARIADNE terminal postprocessing",
                    True,
                    "resume will validate "
                    + str(completed)
                    + " scheduler-completed candidate"
                    + ("" if completed == 1 else "s")
                    + " and "
                    + str(failed)
                    + " scheduler-failed slot"
                    + ("" if failed == 1 else "s")
                    + " locally and submit no ARIADNE job",
                    warn=True,
                )
            )
    ferebus_staging = payload.get(
        "_presentation_ferebus_staging_recovery"
    )
    if (
        isinstance(ferebus_staging, Mapping)
        and ferebus_staging.get("source_submission_identities")
    ):
        staging_disposition = str(
            ferebus_staging.get("disposition") or ""
        )
        completed = int(
            ferebus_staging.get("scheduler_completed_candidates") or 0
        )
        retry = int(
            ferebus_staging.get("known_retry_candidates") or 0
        )
        if staging_disposition == "archived_terminal_producer":
            staging_detail = (
                "reconcile must restore the authenticated producer before "
                "resume"
            )
            staging_ready = False
        elif staging_disposition == "terminal_producer":
            staging_detail = (
                "resume will validate "
                + str(completed)
                + " historical output"
                + ("" if completed == 1 else "s")
                + " and submit only "
                + str(retry)
                + " unresolved task"
                + ("" if retry == 1 else "s")
            )
            staging_ready = True
        else:
            staging_detail = (
                "producer outputs are unavailable; resume will safely prepare "
                + str(int(ferebus_staging.get("n_tasks") or 0))
                + " retry tasks"
            )
            staging_ready = staging_disposition in {
                "absent",
                "input_only",
                "partial_preparation",
                "prepared",
            }
        lines.append(
            _preflight_check_line(
                "FEREBUS staging recovery",
                staging_ready,
                staging_detail,
                warn=staging_ready,
            )
        )
    environment_generation = payload.get(
        "_presentation_environment_generation"
    )
    if isinstance(environment_generation, Mapping):
        disposition = str(environment_generation.get("disposition") or "")
        if not disposition:
            disposition = (
                "current"
                if environment_generation.get("config_matches") is True
                else "reconcile_required"
                if environment_generation.get("config_matches") is False
                else "invalid"
            )
        launchable = disposition in {
            "current",
            "unbound_first_start",
            "rebindable_on_resume",
        }
        generation = int(environment_generation.get("generation", -1))
        detail = str(environment_generation.get("reason") or disposition)
        if disposition == "current":
            detail = (
                str(generation)
                + "; bound to the current campaign configuration"
            )
        lines.append(
            _preflight_check_line(
                "environment generation",
                launchable,
                detail,
                warn=disposition in {
                    "unbound_first_start",
                    "rebindable_on_resume",
                },
            )
        )
    if (
        verbose
        and isinstance(environment_generation, Mapping)
        and environment_generation.get("error")
    ):
        lines.append(
            "  technical environment error: "
            + str(environment_generation["error"])
        )
    issues = [
        str(item)
        for item in (state.get("issues") or [])
        if str(item).strip()
    ]
    for issue in issues[1:] if state.get("error") else issues:
        lines.append("  - " + issue)
    if condition == "stop_scheduled":
        stop_evidence = payload.get("_presentation_stop")
        stop_request = (
            stop_evidence.get("stop_request")
            if isinstance(stop_evidence, Mapping)
            else None
        )
        if isinstance(stop_request, Mapping):
            from .daemon.stop_control import describe_stop_request

            lines.append(
                "  - "
                + describe_stop_request(stop_request)
                + "; the daemon will retain and honour this request"
            )
    if state.get("contract_error"):
        lines.append("  contract error: " + str(state.get("contract_error")))
    diversity_transition = payload.get(
        "_presentation_diversity_transition"
    )
    if (
        isinstance(diversity_transition, Mapping)
        and bool(diversity_transition.get("safe", False))
    ):
        job_id = str(diversity_transition.get("producer_job_id") or "")
        transition_kind = str(
            diversity_transition.get("transition_kind") or ""
        )
        if transition_kind.endswith("complete_publication_adoption"):
            selected = int(diversity_transition.get("selected_count") or 0)
            detail = (
                "ready; terminal scheduler evidence"
                + (" for job " + job_id if job_id else "")
                + " permits local validation of "
                + str(selected)
                + " existing geometr"
                + ("y" if selected == 1 else "ies")
                + " with zero scheduler submissions"
            )
        else:
            detail = (
                "ready; terminal scheduler evidence"
                + (" for job " + job_id if job_id else "")
                + " permits a clean scalar retry"
            )
        lines.append(
            _preflight_check_line(
                "diversity retry boundary",
                True,
                detail,
            )
        )
    elif (
        verbose
        and isinstance(diversity_transition, Mapping)
        and diversity_transition.get("reason")
    ):
        lines.append(
            "  diversity transition: "
            + _reconcile_plain_text(diversity_transition.get("reason"))
        )

    lines.append("")
    lines.append("Trajectory Pool")
    if pool.get("error"):
        lines.append(_preflight_check_line("status", False, pool.get("error")))
    else:
        pool_ok = bool(pool.get("ok", False))
        pool_n = pool.get("pool_n_frames")
        required = pool.get("required_pool_frames")
        lines.append(_preflight_check_line("frames available", pool_ok, pool_n))
        lines.append(_preflight_check_line("frames required", pool_ok, required))
        if verbose and pool.get("bootstrap_custom_count") is not None:
            lines.append(
                "  bootstrap custom geometries: "
                + str(pool.get("bootstrap_custom_count"))
            )
        if verbose and pool.get("bootstrap_model_training_count") is not None:
            lines.append(
                "  bootstrap model training rows: "
                + str(pool.get("bootstrap_model_training_count"))
            )
        if verbose and pool.get("bootstrap_pool_frame_count") is not None:
            lines.append(
                "  bootstrap pool frames: "
                + str(pool.get("bootstrap_pool_frame_count"))
            )
        if verbose and pool.get("expression"):
            lines.append("  requirement: " + str(pool.get("expression")))
        if verbose and pool.get("reserve_after_bootstrap") is not None:
            lines.append("  reserve after bootstrap: " + str(pool.get("reserve_after_bootstrap")))
        try:
            surplus = int(pool_n) - int(required)
        except Exception:
            surplus = None
        if surplus is not None:
            lines.append(
                _preflight_check_line(
                    "surplus frames after planned campaign",
                    surplus >= 0,
                    surplus,
                    warn=surplus == 0,
                )
            )

    lines.append("")
    lines.append("Next Action")
    if payload.get("ready"):
        lines.append("  " + str(payload.get("next_action")))
        command = payload.get("_presentation_next_command")
        if command:
            lines.append("    " + str(command))
    else:
        lines.append("  " + str(payload.get("next_action")))
        command = payload.get("_presentation_next_command")
        if command:
            lines.append("    " + str(command))
        for detail in _preflight_failure_details(payload)[:12]:
            if detail != str(payload.get("next_action")):
                lines.append("  - " + detail)
    if verbose and not payload.get("ready") and not payload.get("all_backends_present"):
        lines.append("")
        lines.append("Backend Details")
        for line in str(payload.get("missing_backend_message") or "").splitlines():
            lines.append("  " + line)
    return "\n".join(lines) + "\n"


def _preflight_launch_advice(
    campaign: Path,
    payload: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    has_environment_evidence = any(
        key in payload
        for key in (
            "all_backends_present",
            "campaign_config",
            "pool_feasibility",
            "submitted_environment_smoke",
        )
    )
    if (
        not has_environment_evidence
        and _preflight_reports_intentional_pause(payload)
    ):
        return (
            "resume the campaign to clear the completed stop",
            _campaign_command(campaign, "resume"),
        )
    environment_ready = bool(
        payload.get("all_backends_present", False)
        and isinstance(payload.get("campaign_config"), Mapping)
        and payload["campaign_config"].get("ok") is True
        and isinstance(payload.get("pool_feasibility"), Mapping)
        and payload["pool_feasibility"].get("ok") is True
    )
    submitted_smoke = payload.get("submitted_environment_smoke")
    if isinstance(submitted_smoke, Mapping):
        environment_ready = environment_ready and bool(
            submitted_smoke.get("ok", False)
        )

    state = payload.get("_presentation_state")
    state_summary = payload.get("campaign_state")
    if not isinstance(state, CampaignState):
        if not environment_ready:
            return "fix failed checks before live start", None
        condition = (
            str(state_summary.get("condition") or "")
            if isinstance(state_summary, Mapping)
            else ""
        )
        if condition == "missing":
            missing = _missing_state_context(campaign)
            if bool(missing.get("fresh_init_safe", False)):
                return (
                    "initialise the campaign before live start",
                    _campaign_command(campaign, "init"),
                )
        return (
            "preview recovery before attempting a live start",
            _campaign_command(campaign, "reconcile"),
        )

    paths = _campaign_paths(campaign)
    status_payload = state.to_dict()
    status_payload["campaign_config_status"] = dict(
        payload.get("campaign_config") or {}
    )
    status_payload["pool_feasibility"] = dict(
        payload.get("pool_feasibility") or {}
    )
    status_payload["_presentation_config_review"] = dict(
        payload.get("_presentation_config_review") or {}
    )
    contract = payload.get("_presentation_artifact_contract")
    if isinstance(contract, Mapping):
        status_payload["state_artifact_contract_status"] = dict(contract)
    stop = payload.get("_presentation_stop")
    if isinstance(stop, Mapping):
        status_payload.update(dict(stop))
    diversity_transition = payload.get(
        "_presentation_diversity_transition"
    )
    if isinstance(diversity_transition, Mapping):
        status_payload["_presentation_diversity_transition"] = dict(
            diversity_transition
        )
    environment = payload.get("_presentation_environment_generation")
    if isinstance(environment, Mapping):
        status_payload["_presentation_environment_generation"] = dict(
            environment
        )

    lock = _probe_daemon_lock(paths["lock"])
    status_payload.update(lock)
    stale_seconds, clock_skew = _runtime_liveness_policy(campaign)
    lease = _probe_daemon_lease(
        paths["lease"],
        stale_seconds=stale_seconds,
        clock_skew_tolerance_seconds=clock_skew,
    )
    status_payload.update(lease)
    background = _probe_background_daemon(
        paths["background_pid"],
        paths["background_log"],
        paths["background_startup"],
    )
    status_payload.update(background)
    try:
        status_payload["active_submission_intents"] = (
            _load_active_submission_intents(
                campaign,
                expected_campaign_uid=str(state.campaign_uid),
                state=state,
            )
        )
    except Exception as exc:
        status_payload["submission_intent_errors"] = [
            {
                "path": "",
                "error": type(exc).__name__ + ": " + str(exc),
            }
        ]
    scheduler_recovery = _load_scheduler_recovery_status(campaign, state)
    if scheduler_recovery is not None:
        status_payload["_presentation_scheduler_recovery"] = (
            scheduler_recovery
        )
    try:
        transaction = inspect_reconcile_transaction_recovery(
            campaign,
            artifact_snapshot=None,
        )
        if isinstance(transaction, Mapping) and str(
            transaction.get("state") or ""
        ) != "none":
            status_payload[
                "_presentation_reconcile_transaction_recovery"
            ] = dict(transaction)
    except Exception as exc:
        status_payload[
            "_presentation_reconcile_transaction_recovery"
        ] = {
            "state": "blocked",
            "recoverable": False,
            "reason": type(exc).__name__ + ": " + str(exc),
        }
    status_payload["_presentation_execution_identity_checked"] = True
    try:
        from .execution_identity import (
            execution_identity_path,
            read_execution_identity,
        )

        if execution_identity_path(campaign).is_file():
            identity = read_execution_identity(
                campaign,
                expected_campaign_uid=str(state.campaign_uid),
            )
            status_payload["_presentation_execution_mode"] = str(
                identity["mode"]
            )
    except Exception as exc:
        status_payload["_presentation_execution_identity_error"] = (
            type(exc).__name__ + ": " + str(exc)
        )

    recommendations = build_status_recommendations(
        campaign,
        status_payload,
        paths["journal"],
    )
    if recommendations:
        first = recommendations[0]
        if _status_daemon_active(status_payload):
            return first.primary, first.command
        if first.code in {
            "runtime_probe_failed",
            "campaign_done",
            "campaign_config_invalid",
            "pool_feasibility_failed",
        } or "reconcile" in str(first.command or ""):
            return first.primary, first.command
        if not environment_ready:
            return "fix failed checks before live start", None
        return first.primary, first.command
    return (
        "inspect campaign status before live start",
        _campaign_command(campaign, "status"),
    )


def evaluate_campaign_preflight(
    campaign: Path,
    *,
    config: Optional[CampaignConfig] = None,
    avail: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return the one authoritative campaign-aware live readiness result."""
    campaign = Path(campaign).expanduser().resolve()
    availability = check_backends() if avail is None else avail
    config_summary: Dict[str, Any]
    feasibility_summary: Dict[str, Any]
    state_summary: Dict[str, Any]
    loaded_config = config
    try:
        from .layout import (
            reject_legacy_campaign_layout,
            qm_reference_data_dir,
            trained_models_dir,
        )
        from .versioning.reference_data import ReferenceDataVersioning
        from .versioning.trained_models import TrainedModelVersioning

        reject_legacy_campaign_layout(campaign)
        reference_versions = ReferenceDataVersioning(qm_reference_data_dir(campaign))
        model_versions = TrainedModelVersioning(trained_models_dir(campaign))
        reference_versions.list_committed_versions()
        reference_versions.current_version()
        model_versions.list_committed_versions()
        model_versions.current_version()
        if loaded_config is None:
            loaded_config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        from .ferebus_prior import resolve_ferebus_prior_contract

        prior = resolve_ferebus_prior_contract(loaded_config)
        checkpoint_summary: Dict[str, Any] = {
            "required": bool(loaded_config.retention.checkpoint_required),
            "destination": loaded_config.retention.checkpoint_destination,
            "ok": True,
        }
        if bool(loaded_config.retention.checkpoint_required):
            from .daemon.checkpoints import normalise_checkpoint_destination

            destination = normalise_checkpoint_destination(
                str(loaded_config.retention.checkpoint_destination)
            )
            if not destination.is_dir():
                raise ValueError(
                    "required checkpoint destination is absent or not a directory: "
                    + str(destination)
                )
            if not os.access(destination, os.W_OK | os.X_OK):
                raise ValueError(
                    "required checkpoint destination is not writable: "
                    + str(destination)
                )
            checkpoint_summary["resolved_destination"] = str(destination)
        resource_profile: Dict[str, Dict[str, Any]] = {}
        if bool(getattr(availability, "profile", False)):
            from .daemon.phase_executor import SBATCH_PHASES
            from .daemon.resource_solver import (
                validate_partition_supported,
                validate_partition_walltime,
            )

            for phase_name in sorted(SBATCH_PHASES):
                partition = str(
                    loaded_config.resources.partition_for(phase_name)
                )
                walltime = float(
                    loaded_config.resources.walltime_for(phase_name)
                )
                validate_partition_supported(partition)
                validate_partition_walltime(partition, walltime)
                resource_profile[phase_name] = {
                    "partition": partition,
                    "walltime_hours": walltime,
                }
        config_summary = {
            "ok": True,
            "schema_version": int(loaded_config.schema_version),
            "system_name": str(loaded_config.campaign.system_name),
            "prior_mean_type": int(prior.mean_type),
            "prior_mean_level_of_theory": prior.level_of_theory,
            "prior_mean_units": "ha",
            "prior_mean_contract_sha256": prior.contract_sha256,
            "feature_scaling": bool(prior.feature_scaling),
            "property_scaling": bool(prior.property_scaling),
            "checkpoint": checkpoint_summary,
            "resource_profile": resource_profile,
        }
    except Exception as exc:
        config_summary = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }
        feasibility_summary = {
            "ok": False,
            "error": "campaign_config_unavailable: "
            + type(exc).__name__
            + ": "
            + str(exc),
        }
    else:
        try:
            feasibility_summary = _pool_feasibility_summary(campaign, loaded_config)
            from .acquisition.trajectory_pool import (
                POOL_MANIFEST_FILENAME,
                POOL_SUBDIR,
                TrajectoryPoolManifest,
            )
            from .ferebus_prior import resolve_ferebus_prior_contract
            from .strict_json import strict_json as _json

            pool_manifest = campaign / POOL_SUBDIR / POOL_MANIFEST_FILENAME
            if pool_manifest.is_file():
                manifest = TrajectoryPoolManifest.from_dict(
                    _json.loads(
                        pool_manifest.read_text(encoding="utf-8"),
                        source=pool_manifest,
                    )
                )
                resolve_ferebus_prior_contract(
                    loaded_config,
                    atom_labels=manifest.atom_types,
                )
        except Exception as exc:
            feasibility_summary = {
                "ok": False,
                "error": type(exc).__name__ + ": " + str(exc),
            }

    presentation_state: Optional[CampaignState] = None
    presentation_contract: Optional[Dict[str, Any]] = None
    presentation_stop: Dict[str, Any] = {}
    presentation_diversity_transition: Optional[Dict[str, Any]] = None
    presentation_ferebus_staging: Optional[Dict[str, Any]] = None
    presentation_ariadne_terminal: Optional[Dict[str, Any]] = None
    presentation_environment: Optional[Dict[str, Any]] = None
    presentation_transaction: Optional[Dict[str, Any]] = None
    presentation_scheduler_repoll: Optional[Dict[str, Any]] = None
    preflight_snapshot: Optional[Any] = None
    presentation_config_review: Dict[str, Any] = {
        "state": "unavailable",
        "n_allowed": 0,
        "n_blocked": 0,
    }
    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    if not state_path.is_file():
        state_summary = {
            "ok": False,
            "condition": "missing",
            "error": "state.json is missing; initialise the campaign before live start",
            "issues": [
                "state.json is missing; initialise the campaign before live start"
            ],
        }
    else:
        try:
            state = read_state(state_path)
        except Exception as exc:
            state_summary = {
                "ok": False,
                "condition": "unreadable",
                "error": type(exc).__name__ + ": " + str(exc),
                "issues": [type(exc).__name__ + ": " + str(exc)],
            }
        else:
            presentation_state = state
            issues: List[str] = []
            condition = "ready"
            if state.phase is CampaignPhase.HALTED:
                condition = "halted"
                issues.append("campaign is HALTED and requires reconcile")
            elif state.phase is CampaignPhase.DONE:
                condition = "complete"
                issues.append("campaign is DONE; no live launch is required")

            try:
                from .daemon.stop_control import (
                    read_stop_request,
                    stop_request_summary,
                )

                validated_stop_request = read_stop_request(
                    campaign,
                    expected_campaign_uid=str(state.campaign_uid),
                )
                presentation_stop = {
                    "stop_request": stop_request_summary(
                        validated_stop_request
                    ),
                    "stop_control_error": None,
                }
            except Exception as exc:
                validated_stop_request = None
                presentation_stop = {
                    "stop_control_error": type(exc).__name__ + ": " + str(exc)
                }
            stop_request = presentation_stop.get("stop_request")
            stop_error = presentation_stop.get("stop_control_error")
            if stop_error:
                condition = "blocked"
                issues.append("user stop control could not be validated")
            elif isinstance(stop_request, Mapping):
                from .daemon.stop_control import (
                    validate_stop_request_for_recovery,
                )

                try:
                    stop_disposition = validate_stop_request_for_recovery(
                        campaign,
                        validated_stop_request,
                        state,
                    )
                except Exception as exc:
                    stop_disposition = {
                        "kind": "unreachable",
                        "launchable": False,
                        "reason": str(exc),
                    }
                stop_kind = str(stop_disposition.get("kind") or "")
                if stop_kind in {"pending_boundary", "pending_immediate"}:
                    if condition == "ready":
                        condition = "stop_scheduled"
                elif stop_kind == "completed":
                    if condition == "ready":
                        condition = "paused"
                    issues.append(
                        "campaign is intentionally paused; resume clears the "
                        "completed stop"
                    )
                elif stop_kind == "campaign_terminal":
                    if condition == "ready":
                        condition = "complete"
                else:
                    condition = "blocked"
                    issues.append(
                        str(
                            stop_disposition.get("reason")
                            or "user stop control is not ready to launch"
                        )
                    )
            elif state.shutdown_requested:
                if _is_intentionally_stopped_state(state):
                    if condition == "ready":
                        condition = "paused"
                    issues.append(
                        "campaign is intentionally paused; resume clears the "
                        "completed stop"
                    )
                else:
                    condition = "blocked"
                    issues.append("campaign has an invalid shutdown request")

            try:
                from .daemon.artifact_contracts import (
                    state_artifact_contract_status,
                )

                preflight_snapshot = build_committed_artifact_snapshot(
                    campaign,
                    verification_level="authority",
                )
                presentation_contract = state_artifact_contract_status(
                    campaign,
                    state,
                    verification="authority",
                    snapshot=preflight_snapshot,
                )
                if not bool(presentation_contract.get("ok", False)):
                    condition = "blocked"
                    issues.append(
                        str(
                            presentation_contract.get("error")
                            or "artefact contract invalid"
                        )
                    )
            except Exception as exc:
                presentation_contract = {
                    "ok": False,
                    "error": type(exc).__name__ + ": " + str(exc),
                }
                condition = "blocked"
                issues.append(
                    "committed artefact authority could not be validated"
                )

            if loaded_config is not None:
                try:
                    review = review_config_changes(
                        campaign,
                        loaded_config,
                        state,
                        initialise_missing=False,
                    )
                    presentation_config_review = config_review_evidence(
                        review,
                        allow_unbound=(
                            state.phase is CampaignPhase.INIT
                            and not any(
                                job_id
                                for job_id in state.pending_jobs.values()
                            )
                        ),
                    )
                    if review.blocked_changes:
                        condition = "blocked"
                        issues.append(
                            "campaign configuration contains blocked changes"
                        )
                    elif review.allowed_changes:
                        if condition in {"ready", "paused", "complete"}:
                            condition = "reconcile_required"
                        issues.append(
                            "campaign configuration differs from its lock"
                        )
                except Exception as exc:
                    presentation_config_review = (
                        invalid_config_review_evidence(
                            type(exc).__name__ + ": " + str(exc)
                        )
                    )
                    condition = "blocked"
                    issues.append(
                        "campaign configuration could not be compared with its lock"
                    )

                if bool(loaded_config.retention.checkpoint_required):
                    try:
                        from .daemon.checkpoints import checkpoint_store

                        store = checkpoint_store(
                            str(
                                loaded_config.retention.checkpoint_destination
                            ),
                            str(state.campaign_uid),
                        )
                        if (store / "current.json").exists():
                            from .daemon.checkpoints import (
                                checkpoint_authority_status,
                            )

                            checkpoint_authority_status(
                                campaign,
                                str(
                                    loaded_config.retention.checkpoint_destination
                                ),
                            )
                    except Exception as exc:
                        condition = "blocked"
                        issues.append(
                            "required checkpoint authority is invalid: "
                            + str(exc)
                        )

            ownership_payload = state.to_dict()
            paths = _campaign_paths(campaign)
            ownership_payload.update(_probe_daemon_lock(paths["lock"]))
            stale_seconds, clock_skew = _runtime_liveness_policy(campaign)
            ownership_payload.update(
                _probe_daemon_lease(
                    paths["lease"],
                    stale_seconds=stale_seconds,
                    clock_skew_tolerance_seconds=clock_skew,
                )
            )
            background_probe = _probe_background_daemon(
                paths["background_pid"],
                paths["background_log"],
                paths["background_startup"],
            )
            ownership_payload.update(background_probe)
            intent_errors: List[Dict[str, str]] = []
            ownership_payload["active_submission_intents"] = (
                _load_active_submission_intents(
                    campaign,
                    errors=intent_errors,
                    expected_campaign_uid=str(state.campaign_uid),
                    state=state,
                )
            )
            if intent_errors:
                ownership_payload["submission_intent_errors"] = intent_errors
            scheduler_recovery = _load_scheduler_recovery_status(campaign, state)
            if scheduler_recovery is not None:
                ownership_payload["_presentation_scheduler_recovery"] = (
                    scheduler_recovery
                )
            ownership = assess_campaign_presentation(
                ownership_payload
            ).scheduler
            try:
                from .daemon.preserved_scheduler_repoll import (
                    resolve_preserved_scheduler_repoll_authority,
                )

                scheduler_repoll_authority = (
                    resolve_preserved_scheduler_repoll_authority(
                        campaign,
                        state,
                    )
                )
                if scheduler_repoll_authority is not None:
                    presentation_scheduler_repoll = (
                        scheduler_repoll_authority.to_evidence()
                    )
            except Exception as exc:
                scheduler_repoll_authority = None
                presentation_scheduler_repoll = {
                    "error": type(exc).__name__ + ": " + str(exc),
                }
            launch_ownership_payload = dict(ownership_payload)
            if _background_probe_is_current_startup_child(
                campaign,
                background_probe,
            ):
                # The launcher publishes the child PID before the child can
                # acquire the daemon lock.  Ignore only that exact self-PID;
                # lock and lease probes remain authoritative for races with a
                # different daemon.
                launch_ownership_payload["background_pid_alive"] = False
            daemon_active = _status_daemon_active(launch_ownership_payload)
            scheduler_clear = bool(
                not daemon_active
                and (
                    not ownership.has_unresolved_scheduler_work
                    or scheduler_repoll_authority is not None
                )
                and not intent_errors
            )
            if daemon_active:
                condition = "blocked"
                issues.append("another daemon currently owns this campaign")
            elif (
                ownership.has_unresolved_scheduler_work
                and scheduler_repoll_authority is None
            ) or intent_errors:
                condition = "blocked"
                issues.append(
                    "scheduler ownership is active or cannot be established safely"
                )
            elif scheduler_repoll_authority is not None:
                condition = "ready"

            try:
                presentation_transaction = inspect_reconcile_transaction_recovery(
                    campaign,
                    artifact_snapshot=preflight_snapshot,
                )
            except Exception as exc:
                presentation_transaction = {
                    "state": "blocked",
                    "recoverable": False,
                    "reason": type(exc).__name__ + ": " + str(exc),
                }
            transaction_state = str(
                (presentation_transaction or {}).get("state") or "none"
            )
            if transaction_state == "recoverable" and bool(
                (presentation_transaction or {}).get("recoverable", False)
            ):
                if condition != "blocked":
                    condition = "reconcile_required"
                issues.append(
                    "an interrupted reconcile must be recovered before launch"
                )
            elif transaction_state not in {"none", "recovered"}:
                condition = "blocked"
                issues.append(
                    "interrupted reconcile evidence requires manual review"
                )

            presentation_environment = (
                _preflight_environment_generation_evidence(
                    campaign,
                    state,
                    loaded_config,
                    scheduler_ownership_clear=scheduler_clear,
                )
            )
            environment_disposition = str(
                (presentation_environment or {}).get("disposition") or "invalid"
            )
            if environment_disposition == "reconcile_required":
                if condition != "blocked":
                    condition = "reconcile_required"
                issues.append(
                    str(
                        (presentation_environment or {}).get("reason")
                        or "environment recovery is required before launch"
                    )
                )
            elif environment_disposition in {"invalid", "ownership_blocked"}:
                condition = "blocked"
                issues.append(
                    str(
                        (presentation_environment or {}).get("reason")
                        or "execution environment evidence is invalid"
                    )
                )

            if state.phase is CampaignPhase.ARIADNE_ARRAY:
                try:
                    presentation_ariadne_terminal = (
                        _ariadne_terminal_postprocess_presentation_evidence(
                            campaign,
                            state,
                        )
                    )
                except Exception as exc:
                    presentation_ariadne_terminal = {
                        "error": type(exc).__name__ + ": " + str(exc),
                    }
                    condition = "blocked"
                    issues.append(
                        "terminal ARIADNE postprocessing evidence is invalid"
                    )

            if state.phase in {
                CampaignPhase.PHASE_A_DIVERSITY,
                CampaignPhase.PHASE_B_DIVERSITY,
            }:
                try:
                    from .execution_identity import (
                        inspect_scalar_diversity_transition_boundary,
                    )

                    presentation_diversity_transition = (
                        inspect_scalar_diversity_transition_boundary(
                            campaign,
                            state,
                        )
                    )
                except Exception as exc:
                    presentation_diversity_transition = {
                        "safe": False,
                        "phase": state.phase.value,
                        "iteration": int(state.iteration),
                        "reason": type(exc).__name__ + ": " + str(exc),
                    }

            # A transaction-bound scheduler re-poll owns this launch.  Its
            # staging is intentionally preserved and must not be reclassified
            # as an idle recovery tree before terminal accounting is known.
            presentation_ferebus_staging = (
                None
                if scheduler_repoll_authority is not None
                else _ferebus_staging_presentation_evidence(
                    campaign,
                    state,
                    artifact_snapshot=preflight_snapshot,
                    config=loaded_config,
                )
            )
            if isinstance(presentation_ferebus_staging, Mapping):
                staging_disposition = str(
                    presentation_ferebus_staging.get("disposition") or ""
                )
                if staging_disposition == "contradictory":
                    condition = "blocked"
                    issues.append(
                        "FEREBUS staging recovery evidence is contradictory"
                    )
                elif staging_disposition == "archived_terminal_producer":
                    if condition != "blocked":
                        condition = "reconcile_required"
                    issues.append(
                        "FEREBUS producer staging must be restored by reconcile"
                    )

            state_summary = {
                "ok": not issues,
                "condition": condition,
                "phase": state.phase.value,
                "iteration": int(state.iteration),
                "issues": issues,
            }
            if issues:
                state_summary["error"] = issues[0]

    payload = _preflight_payload(
        campaign,
        availability,
        config_summary,
        feasibility_summary,
        state_summary,
    )
    payload["_presentation_state"] = presentation_state
    payload["_presentation_artifact_contract"] = presentation_contract
    payload["_presentation_stop"] = presentation_stop
    payload["_presentation_config_review"] = presentation_config_review
    payload["_presentation_diversity_transition"] = (
        presentation_diversity_transition
    )
    payload["_presentation_ferebus_staging_recovery"] = (
        presentation_ferebus_staging
    )
    if presentation_ariadne_terminal is not None:
        payload["_presentation_ariadne_terminal_postprocess"] = dict(
            presentation_ariadne_terminal
        )
    payload["_presentation_environment_generation"] = presentation_environment
    if presentation_scheduler_repoll is not None:
        payload["_presentation_scheduler_repoll"] = (
            presentation_scheduler_repoll
        )
    if isinstance(presentation_transaction, Mapping) and str(
        presentation_transaction.get("state") or ""
    ) != "none":
        payload["_presentation_reconcile_transaction_recovery"] = dict(
            presentation_transaction
        )
    return payload


def cmd_preflight(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(getattr(args, "campaign_dir", None))
    availability = check_backends()
    payload = evaluate_campaign_preflight(campaign, avail=availability)
    if bool(getattr(args, "submit_environment_smoke", False)):
        if bool(payload.get("ready", False)):
            try:
                config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
                smoke = run_submitted_environment_smoke(
                    campaign_dir=campaign,
                    config=config,
                    availability=availability,
                )
            except Exception as exc:
                smoke = {
                    "schema_version": 1,
                    "submitted": False,
                    "ok": False,
                    "job_id": "",
                    "output_path": "",
                    "error": type(exc).__name__ + ": " + str(exc),
                }
        else:
            smoke = {
                "schema_version": 1,
                "submitted": False,
                "ok": False,
                "job_id": "",
                "output_path": "",
                "error": "ordinary campaign preflight is blocked; no job was submitted",
            }
        payload["submitted_environment_smoke"] = smoke
        payload["ready"] = bool(payload.get("ready", False)) and bool(
            smoke.get("ok", False)
        )
    next_action, next_command = _preflight_launch_advice(campaign, payload)
    payload["next_action"] = next_action
    if bool(getattr(args, "json", False)):
        machine_payload = {
            key: value
            for key, value in payload.items()
            if not str(key).startswith("_presentation_")
        }
        print(
            json.dumps(
                machine_payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    else:
        presentation_payload = dict(payload)
        presentation_payload["_presentation_next_command"] = next_command
        print(
            _format_preflight(
                presentation_payload,
                verbose=bool(getattr(args, "verbose", False)),
            ),
            end="",
        )
    if payload["ready"]:
        return 0
    return 12


def cmd_resource_plan(args: argparse.Namespace) -> int:
    """Print a read-only production resource resolution preview."""
    from .daemon.resource_plan import build_resource_plan, format_resource_plan

    try:
        campaign = resolve_campaign_dir(args.campaign_dir)
        config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        state_path = (
            campaign
            / ".DATA"
            / "ACTIVE_LEARNING"
            / DEFAULT_STATE_FILENAME
        )
        if not state_path.is_file():
            print("resource-plan requires daemon state.json", file=sys.stderr)
            return 2
        state = read_state(state_path)
        phase_name = getattr(args, "phase", None)
        if phase_name is not None and str(phase_name) not in {
            phase.value for phase in CampaignPhase
        }:
            print("unknown campaign phase: " + str(phase_name), file=sys.stderr)
            return 2
        payload = build_resource_plan(
            campaign,
            config,
            current_phase=state.phase.value,
            current_iteration=int(state.iteration),
            phase_name=phase_name,
            iteration=getattr(args, "iteration", None),
            all_phases=bool(getattr(args, "all", False)),
            replacement_round=int(getattr(state, "replacement_round", 0)),
            current_models_version=int(getattr(state, "models_version", 0)),
            current_reference_data_version=int(
                getattr(state, "reference_data_version", 0)
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        print(
            "resource-plan could not verify its read-only inputs: " + str(exc),
            file=sys.stderr,
        )
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(
            format_resource_plan(
                payload,
                verbose=bool(getattr(args, "verbose", False)),
            ),
            end="",
        )
    if not bool(getattr(args, "all", False)):
        statuses = {str(plan.get("status")) for plan in payload["plans"]}
        if "evidence_invalid" in statuses:
            return 15
        if "evidence_not_yet_produced" in statuses:
            return 14
    return 0


def cmd_export_batch_geometries(args: argparse.Namespace) -> int:
    """Export accepted active-iteration seed and committed QM geometries."""
    from .batch_geometry_export import export_batch_geometries

    try:
        campaign = resolve_campaign_dir(getattr(args, "campaign_dir", None))
        summary = export_batch_geometries(
            campaign,
            getattr(args, "iteration"),
            output_dir=getattr(args, "output_dir", None),
        )
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print("Batch geometries were not exported: " + str(exc), file=sys.stderr)
        return 2

    if len(summary.iterations) == 1:
        iteration_text = str(summary.iterations[0])
    else:
        iteration_text = (
            str(summary.iterations[0])
            + "-"
            + str(summary.iterations[-1])
            + " ("
            + str(len(summary.iterations))
            + " iterations)"
        )
    print("Exported batch geometries")
    print("  output: " + str(summary.output_path))
    print("  iterations: " + iteration_text)
    print("  training files: " + str(summary.training_count))
    print(
        "  internal-validation files: "
        + str(summary.internal_validation_count)
    )
    print("  total XYZ files: " + str(summary.total_count))
    print(
        "  maximum coordinate discrepancy: "
        + format(summary.maximum_coordinate_discrepancy_angstrom, ".12g")
        + " A"
    )
    return 0


def _checkpoint_destination(
    config: CampaignConfig,
    override: Optional[str],
) -> str:
    value = override or config.retention.checkpoint_destination
    if value is None or not str(value).strip():
        raise ValueError(
            "no checkpoint destination is configured; set "
            "retention.checkpoint_destination or pass --destination"
        )
    return str(value)


def cmd_checkpoint(args: argparse.Namespace) -> int:
    """Create a verified checkpoint from the current idle boundary."""
    from .daemon.checkpoints import create_checkpoint

    progress_reporter = None
    try:
        campaign = resolve_campaign_dir(args.campaign_dir)
        config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        destination = _checkpoint_destination(
            config,
            getattr(args, "destination", None),
        )
        try:
            from .daemon.journal import append_event
            from .daemon.phase_progress import PhaseProgressReporter

            checkpoint_state = read_state(_campaign_paths(campaign)["state"])

            def journal_progress(event_type: str, payload: Mapping[str, Any]) -> None:
                append_event(
                    _campaign_paths(campaign)["journal"],
                    str(event_type),
                    max_bytes=int(config.runtime.journal_max_bytes),
                    retained_files=int(config.runtime.journal_retained_files),
                    lock_timeout_seconds=int(
                        config.runtime.ledger_lock_timeout_seconds
                    ),
                    **dict(payload),
                )

            progress_reporter = PhaseProgressReporter(
                campaign,
                campaign_uid=str(checkpoint_state.campaign_uid),
                phase=checkpoint_state.phase.value,
                iteration=int(checkpoint_state.iteration),
                replacement_round=int(checkpoint_state.replacement_round),
                producer_kind="checkpoint",
                identity={"daemon_pid": int(os.getpid())},
                journal_callback=journal_progress,
            )
            progress_reporter.start("checkpoint_copy")
        except Exception:
            progress_reporter = None
        with _exclusive_operator_lock(campaign):
            payload = create_checkpoint(
                campaign,
                destination,
                verify_after_write=True,
                allow_active_lease=False,
                progress_callback=(
                    progress_reporter.update
                    if progress_reporter is not None
                    else None
                ),
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if progress_reporter is not None:
            progress_reporter.fail(type(exc).__name__ + ": " + str(exc))
        print("Checkpoint was not created: " + str(exc), file=sys.stderr)
        print(
            "Check the destination and campaign status, then run checkpoint-status.",
            file=sys.stderr,
        )
        return 2
    finally:
        if progress_reporter is not None:
            progress_reporter.close()
    if progress_reporter is not None:
        progress_reporter.complete(stage="checkpoint_verification")
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("Checkpoint created and verified.")
        print("  checkpoint: " + str(payload["checkpoint"]))
        manifest = payload.get("manifest")
        if isinstance(manifest, dict) and manifest.get("iteration") is not None:
            print("  campaign iteration: " + str(manifest.get("iteration")))
        print("Action")
        print("  no further checkpoint action is required")
    return 0


def cmd_checkpoint_status(args: argparse.Namespace) -> int:
    """Report and verify the current checkpoint pointer."""
    from .daemon.checkpoints import checkpoint_status

    try:
        campaign = resolve_campaign_dir(args.campaign_dir)
        config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        destination = _checkpoint_destination(
            config,
            getattr(args, "destination", None),
        )
        payload = checkpoint_status(campaign, destination)
    except (OSError, TypeError, ValueError) as exc:
        print("Checkpoint status could not be verified: " + str(exc), file=sys.stderr)
        print("Do not restore from this checkpoint store until it is verified.", file=sys.stderr)
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        if payload["status"] == "verified":
            print("Current checkpoint is present and verified.")
            if isinstance(payload.get("current"), dict):
                print("  campaign iteration: " + str(payload["current"].get("iteration")))
            print("Action")
            print("  no checkpoint repair is required")
        else:
            print("No current checkpoint has been published for this campaign.")
            print("Action")
            print(
                "  ichor-al-daemon checkpoint --campaign-dir "
                + shlex.quote(str(campaign))
                + " --destination "
                + shlex.quote(str(destination))
            )
    return 0 if payload["status"] == "verified" else 1


def cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    """Deeply verify one published checkpoint."""
    from .daemon.checkpoints import verify_checkpoint

    try:
        payload = verify_checkpoint(args.checkpoint)
    except (OSError, TypeError, ValueError) as exc:
        print("Checkpoint verification failed: " + str(exc), file=sys.stderr)
        print("Do not restore this checkpoint.", file=sys.stderr)
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("Checkpoint is complete and all stored files were verified.")
        print("  checkpoint: " + str(payload["checkpoint"]))
        print("Action")
        print("  this checkpoint is safe to use for a restore preview")
    return 0


def cmd_restore_checkpoint(args: argparse.Namespace) -> int:
    """Restore a verified checkpoint into an empty target directory."""
    from .daemon.checkpoints import restore_checkpoint

    try:
        payload = restore_checkpoint(
            args.checkpoint,
            args.target_empty_dir,
            apply=bool(getattr(args, "apply", False)),
        )
    except (OSError, TypeError, ValueError) as exc:
        print("Checkpoint restore failed: " + str(exc), file=sys.stderr)
        print("No restored campaign was published.", file=sys.stderr)
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    elif payload["applied"]:
        print("Checkpoint restored and verified.")
        print("  restored campaign: " + str(payload["target"]))
        print("Action")
        print(
            "  ichor-al-daemon status --campaign-dir "
            + shlex.quote(str(payload["target"]))
        )
    else:
        print("Checkpoint restore verified; no files were written.")
        print("Target: " + str(payload["target"]))
        print("Action")
        print(
            "  ichor-al-daemon restore-checkpoint --checkpoint "
            + shlex.quote(str(args.checkpoint))
            + " --target-empty-dir "
            + shlex.quote(str(args.target_empty_dir))
            + " --apply"
        )
    return 0


def _positive_cli_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_cli_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _export_iteration_cli(value: str) -> Union[int, str]:
    text = str(value).strip().lower()
    if text == "all":
        return "all"
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a positive iteration number or 'all'"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "expected a positive iteration number or 'all'"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    examples = """\
Campaign directory:
  If -c/--campaign-dir is omitted, the current directory is used when it
  contains campaign.yaml.

Examples:
  cd ~/campaigns/water_001
  ichor-al-daemon init
  ichor-al-daemon status
  ichor-al-daemon start
  ichor-al-daemon start --foreground
  ichor-al-daemon journal -e phase_submitted
  ichor-al-daemon export-batch-geometries --iteration 8

  ichor-al-daemon start -c ~/campaigns/water_001
"""
    parser = argparse.ArgumentParser(
        prog="ichor-al-daemon",
        description="ICHOR active-learning campaign daemon.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_campaign(p):
        p.add_argument(
            "-c",
            "--campaign-dir",
            default=None,
            help=(
                "Campaign directory. If omitted, the current directory is used "
                "when it contains campaign.yaml."
            ),
        )
    def add_background_options(p):
        process_group = p.add_mutually_exclusive_group()
        process_group.add_argument(
            "-b",
            "--background",
            action="store_true",
            help=(
                "Explicitly run the daemon in the background (the default) and "
                "return after campaign ownership is confirmed. Use status to "
                "check whether startup has completed."
            ),
        )
        process_group.add_argument(
            "-f",
            "--foreground",
            action="store_true",
            help="Run in the invoking terminal instead of detaching.",
        )
        p.add_argument(
            "-o",
            "--background-log",
            default=None,
            help=(
                "Background daemon log path. Defaults to "
                "<campaign>/.DATA/ACTIVE_LEARNING/daemon.out."
            ),
        )
        p.add_argument(
            "-i",
            "--background-pid",
            default=None,
            help=(
                "Background daemon PID metadata path. Defaults to "
                "<campaign>/.DATA/ACTIVE_LEARNING/daemon.pid."
            ),
        )

    p_start = sub.add_parser(
        "start",
        help="Start the daemon.",
        description=(
            "Start a campaign daemon in live mode and in the background by "
            "default. The first effective mode is immutable thereafter. "
            "From inside a campaign directory, --campaign-dir can be omitted."
        ),
        epilog=(
            "Examples:\n"
            "  ichor-al-daemon start\n"
            "  ichor-al-daemon start --mode dry_run --max-ticks 10\n"
            "  ichor-al-daemon start --foreground"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_campaign(p_start)
    p_start.add_argument(
        "-g",
        "--config",
        default=None,
        help="Path to campaign.yaml (defaults to <campaign-dir>/campaign.yaml).",
    )
    p_start.add_argument(
        "--mode",
        choices=["live", "dry_run"],
        default=None,
        help=(
            "Execution mode. Defaults to live for the first start and then "
            "reuses the campaign's permanently bound mode."
        ),
    )
    p_start.add_argument(
        "-p",
        "--poll-interval",
        type=_positive_cli_int,
        default=None,
        help=(
            "Override the pending-work polling interval from the config; "
            "completed phase transitions continue immediately."
        ),
    )
    p_start.add_argument(
        "-t",
        "--max-ticks",
        type=_nonnegative_cli_int,
        default=None,
        help="Limit total tick count (testing / time-boxed runs).",
    )
    add_background_options(p_start)
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser(
        "stop",
        help="Request an immediate or receipt-backed boundary stop.",
        description=(
            "Request daemon shutdown for the resolved campaign. Plain stop is "
            "immediate at the next tick and does not cancel scheduler jobs."
        ),
    )
    add_campaign(p_stop)
    stop_modes = p_stop.add_mutually_exclusive_group()
    stop_modes.add_argument(
        "--immediate",
        dest="stop_mode",
        action="store_const",
        const="immediate",
        help="Stop at the next daemon tick (default).",
    )
    stop_modes.add_argument(
        "--after-phase",
        dest="stop_mode",
        action="store_const",
        const="after_phase",
        help="Finish the current started phase and stop before the next phase.",
    )
    stop_modes.add_argument(
        "--after-iteration",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        metavar="N",
        help=(
            "Finish the current iteration, or explicit iteration N, then stop "
            "before the next iteration."
        ),
    )
    p_stop.add_argument(
        "-x",
        "--cancel-jobs",
        action="store_true",
        help=(
            "Also cancel active scheduler jobs recorded by this campaign. "
            "Plain stop only requests daemon shutdown and leaves jobs alone."
        ),
    )
    p_stop.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show request identifiers, control paths and process-signal details.",
    )
    p_stop.set_defaults(func=cmd_stop, stop_mode="immediate")

    p_status = sub.add_parser(
        "status",
        help="Print the current daemon status.",
        description=(
            "Print the current daemon status for the resolved campaign. From "
            "inside a campaign directory, --campaign-dir can be omitted."
        ),
    )
    add_campaign(p_status)
    p_status.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Print the raw state/status payload as JSON.",
    )
    p_status.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include expanded artifact, lease, and path diagnostics.",
    )
    p_status.set_defaults(func=cmd_status)

    p_resume = sub.add_parser(
        "resume",
        help="Continue a stopped or safely recovered campaign.",
        description=(
            "Continue an existing campaign after a normal stop or reviewed "
            "recovery. Resume preserves the campaign's previously selected "
            "execution mode and runs in the background by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_campaign(p_resume)
    p_resume.add_argument("-g", "--config", default=None)
    p_resume.add_argument(
        "--mode",
        choices=["live", "dry_run"],
        default=None,
        help="Optional assertion of the campaign's already-bound execution mode.",
    )
    p_resume.add_argument("-p", "--poll-interval", type=_positive_cli_int, default=None)
    p_resume.add_argument("-t", "--max-ticks", type=_nonnegative_cli_int, default=None)
    p_resume.add_argument(
        "--reopen-converged",
        action="store_true",
        help=(
            "Explicitly reopen a DONE campaign at the next SEED_SELECT after "
            "campaign.max_iterations has been increased."
        ),
    )
    p_resume.add_argument(
        "--cancel-stop-request",
        action="store_true",
        help=(
            "Withdraw an unfinished after-phase or after-iteration stop "
            "request. If the daemon is already running, it remains running; "
            "a completed user stop is cleared by ordinary resume."
        ),
    )
    add_background_options(p_resume)
    p_resume.set_defaults(func=cmd_resume)

    p_recon = sub.add_parser(
        "reconcile",
        help="Inspect on-disk artefacts and propose a recovered state.",
        description=(
            "Inspect campaign evidence and show one recovery plan. With --apply, "
            "commit the reviewed bookkeeping and temporary-data changes. Reconcile "
            "does not start the daemon or submit scheduler jobs."
        ),
    )
    add_campaign(p_recon)
    p_recon.add_argument(
        "-F",
        "--allow-fresh-init",
        action="store_true",
        help=(
            "Allow reconcile to propose INIT over non-state campaign artefacts "
            "only when campaign identity remains trusted. This never permits "
            "minting a replacement UID for a non-empty campaign."
        ),
    )
    p_recon.add_argument(
        "-a",
        "--apply",
        action="store_true",
        help=(
            "After writing state.json.proposed, safely promote it to state.json, "
            "clean stale uncommitted re-entry staging, and update the campaign "
            "config lock. Refuses locked campaign.yaml changes."
        ),
    )
    p_recon.add_argument(
        "-j",
        "--json",
        action="store_true",
        help=(
            "Print a machine-readable recovery decision payload. Proposal-only; "
            "do not combine with --apply."
        ),
    )
    p_recon.add_argument(
        "--deep-verify",
        action="store_true",
        help=(
            "Hash every committed scientific payload once while reconciling. "
            "Required when campaign authority must be reconstructed without "
            "a valid state file; otherwise non-recursive authority verification "
            "is used."
        ),
    )
    p_recon.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Include detailed diagnostic notes, trusted artefacts, and inventory "
            "samples in the human-readable reconcile report."
        ),
    )
    p_recon.add_argument(
        "--restore-config-from-lock",
        action="store_true",
        help=(
            "When campaign.yaml is missing, write campaign.yaml.proposed from "
            "the locked canonical config. Proposal-only; never overwrites "
            "campaign.yaml."
        ),
    )
    p_recon.add_argument(
        "--restore-config-lock-history",
        action="store_true",
        help=(
            "Restore a missing config_lock.json from the unique latest verified "
            "history-chain entry matching state.json. Refuses an existing lock."
        ),
    )
    p_recon.add_argument(
        "--archive-staging",
        action="store_true",
        help=(
            "With --apply, archive non-empty .DATA/STAGING to a timestamped "
            "sibling when no active daemon work or recorded jobs remain. "
            "Never deletes staging contents."
        ),
    )
    p_recon.add_argument(
        "--clean-scratch",
        action="store_true",
        help=(
            "Inventory campaign-owned scratch and, with --apply, remove only "
            "conclusively inactive valid attempt directories. Active, "
            "scheduler-inconclusive, symlinked, or malformed scratch is never "
            "removed."
        ),
    )
    p_recon.add_argument(
        "--scratch-attempt",
        action="append",
        default=None,
        metavar="ID",
        help=(
            "Restrict --clean-scratch to a submission identity or attempt ID. "
            "Repeat this option to select more than one attempt."
        ),
    )
    p_recon.add_argument(
        "--clean-ariadne-quarantine",
        action="store_true",
        help=(
            "Inventory retained ARIADNE retry outputs and, with --apply, remove "
            "only attempts with complete, contained ownership manifests."
        ),
    )
    p_recon.add_argument(
        "--ariadne-quarantine-attempt",
        action="append",
        default=None,
        metavar="ID",
        help=(
            "Restrict --clean-ariadne-quarantine to one retained attempt ID. "
            "Repeat this option to select more than one attempt."
        ),
    )
    p_recon.add_argument(
        "--force-resubmit-array-tasks",
        action="store_true",
        help=(
            "With --apply, disable partial reuse for the current supported "
            "array phase and resubmit every logical task. Use only when a "
            "current-array science setting changed or the whole array must "
            "be rerun."
        ),
    )
    p_recon.add_argument(
        "--archive-existing-array-task-outputs",
        action="store_true",
        help=(
            "With --apply --force-resubmit-array-tasks, move existing "
            "daemon-owned task output files to an archive before the full "
            "array is resubmitted."
        ),
    )
    p_recon.add_argument(
        "--retrain-ferebus",
        action="store_true",
        help=(
            "With --apply in a FEREBUS recovery phase, losslessly archive "
            "complete uncommitted FEREBUS output and submit a new fit. "
            "Without this flag, threshold-only changes re-evaluate the "
            "existing immutable quality evidence."
        ),
    )
    p_recon.set_defaults(func=cmd_reconcile)

    p_jrn = sub.add_parser(
        "journal",
        help="Print campaign journal events.",
        description=(
            "Print or filter the campaign journal. From inside a campaign "
            "directory, --campaign-dir can be omitted."
        ),
        epilog=(
            "Examples:\n"
            "  ichor-al-daemon journal\n"
            "  ichor-al-daemon journal -e phase_submitted -n 20\n"
            "  ichor-al-daemon journal --last-n 40\n"
            "  ichor-al-daemon journal --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_campaign(p_jrn)
    p_jrn.add_argument(
        "-s",
        "--since",
        default=None,
        help="ISO timestamp lower bound (inclusive).",
    )
    p_jrn.add_argument(
        "-e",
        "--event-type",
        action="append",
        default=None,
        help="Filter to one or more event types (repeatable).",
    )
    p_jrn.add_argument(
        "-n",
        "--last-n",
        type=_positive_cli_int,
        default=None,
        help="Show only the last N events after filters are applied.",
    )
    p_jrn.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Print filtered events as NDJSON.",
    )
    p_jrn.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help="Alias for --json; keeps one JSON event per line.",
    )
    p_jrn.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print expanded key/value details for each event.",
    )
    p_jrn.add_argument(
        "--list-event-types",
        action="store_true",
        help="Print known daemon journal event names and exit.",
    )
    p_jrn.set_defaults(func=cmd_journal)

    def add_init_options(p, *, source_required: bool = False):
        add_campaign(p)
        p.add_argument(
            "-s",
            "--source",
            required=source_required,
            default=None,
            help=(
                "Copy this trajectory to <campaign>/pool.xyz before immutable "
                "pool import. Without it, <campaign>/pool.xyz is required."
            ),
        )
        p.add_argument(
            "-f",
            "--force", action="store_true",
            help=(
                "Replace an existing uncommitted pool only. Campaigns with "
                "committed run data are always refused."
            ),
        )
        p.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help=(
                "Accept the validated bootstrap summary non-interactively. "
                "CSV inputs still require bootstrap/alf.yaml."
            ),
        )
        p.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help=(
                "Compatibility flag; init now shows hashes, ALFs and detailed "
                "bootstrap evidence by default."
            ),
        )

    p_init = sub.add_parser(
        "init",
        help="Bootstrap campaign.yaml, daemon state, config lock, and campaign inputs.",
        description=(
            "Initialise or populate campaign.yaml from the packaged template, "
            "inspect fixed campaign-relative pool/bootstrap inputs, require "
            "user confirmation, and create SHA-pinned daemon state and "
            "manifests. From inside a campaign directory, --campaign-dir and "
            "--source can be omitted."
        ),
        epilog=(
            "Examples:\n"
            "  ichor-al-daemon init\n"
            "  ichor-al-daemon init -c ~/campaigns/water_001 -s pool.xyz\n"
            "  ichor-al-daemon start --campaign-dir ~/campaigns/water_001\n"
            "  ichor-al-daemon start --campaign-dir ~/campaigns/water_001 --mode dry_run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_init_options(p_init)
    p_init.set_defaults(func=cmd_init)

    p_imp = sub.add_parser(
        "import-pool",
        help=argparse.SUPPRESS,
        description=(
            "Deprecated compatibility alias for 'init'. Use "
            "'ichor-al-daemon init' for new workflows."
        ),
        epilog=(
            "Examples:\n"
            "  ichor-al-daemon init\n"
            "  ichor-al-daemon init -c ~/campaigns/water_001 -s pool.xyz"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_init_options(p_imp)
    p_imp.set_defaults(func=cmd_import_pool)

    p_pre = sub.add_parser(
        "preflight",
        help="Check configured live scheduler backends.",
        description=(
            "Check the configured scheduler/Gaussian/AIMAll/FEREBUS/ARIADNE "
            "backend profile, campaign.yaml, and trajectory-pool feasibility."
        ),
    )
    p_pre.add_argument(
        "-c",
        "--campaign-dir",
        default=".",
        help="Campaign directory containing campaign.yaml and the imported trajectory pool.",
    )
    p_pre.add_argument(
        "--json",
        action="store_true",
        help="Print one machine-readable JSON payload instead of the user summary.",
    )
    p_pre.add_argument(
        "--verbose",
        action="store_true",
        help="Include detailed backend guidance for failed checks.",
    )
    p_pre.add_argument(
        "--submit-environment-smoke",
        action="store_true",
        help=(
            "After ordinary preflight passes, submit one five-minute, one-core "
            "scheduler job that verifies the compute-node module, Python-import, "
            "and executable environment without running scientific work."
        ),
    )
    p_pre.set_defaults(func=cmd_preflight)

    p_resource = sub.add_parser(
        "resource-plan",
        help="Inspect submitted or prospective resources without changing state.",
        description=(
            "Use the production resource resolver in read-only mode for the "
            "current phase, one named phase, or every applicable phase."
        ),
    )
    add_campaign(p_resource)
    selection = p_resource.add_mutually_exclusive_group()
    selection.add_argument(
        "--phase",
        choices=[phase.value for phase in CampaignPhase],
        default=None,
        help="Inspect one explicit campaign phase.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Show every applicable bootstrap or active-cycle phase.",
    )
    p_resource.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Inspect this iteration instead of the current state iteration.",
    )
    p_resource.add_argument(
        "--json",
        action="store_true",
        help="Print schema-v2 machine-readable JSON.",
    )
    p_resource.add_argument(
        "--verbose",
        action="store_true",
        help="Show formulae, evidence hashes, profile limits and telemetry details.",
    )
    p_resource.set_defaults(func=cmd_resource_plan)

    p_export = sub.add_parser(
        "export-batch-geometries",
        help="Export accepted active-iteration seed/final geometry pairs.",
        description=(
            "Validate one completed active-learning model batch and export one "
            "two-frame XYZ file per accepted training or internal-validation "
            "slot. Use --iteration all to export every completed active iteration."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_campaign(p_export)
    p_export.add_argument(
        "--iteration",
        required=True,
        type=_export_iteration_cli,
        metavar="N|all",
        help="Completed active iteration number, or 'all'. Iteration 0 is not exportable.",
    )
    p_export.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Exact output directory. It must not already exist. Without this "
            "option, publish under EXPORTED_GEOMETRIES and atomically replace "
            "an earlier export of the same iteration selection."
        ),
    )
    p_export.set_defaults(func=cmd_export_batch_geometries)

    p_checkpoint = sub.add_parser(
        "checkpoint",
        help="Create and verify a durable campaign checkpoint.",
    )
    add_campaign(p_checkpoint)
    p_checkpoint.add_argument(
        "--destination",
        default=None,
        help="Override retention.checkpoint_destination for this checkpoint.",
    )
    p_checkpoint.add_argument("--json", action="store_true")
    p_checkpoint.set_defaults(func=cmd_checkpoint)

    p_checkpoint_status = sub.add_parser(
        "checkpoint-status",
        help="Verify and report the current checkpoint pointer.",
    )
    add_campaign(p_checkpoint_status)
    p_checkpoint_status.add_argument(
        "--destination",
        default=None,
        help="Override retention.checkpoint_destination for this query.",
    )
    p_checkpoint_status.add_argument("--json", action="store_true")
    p_checkpoint_status.set_defaults(func=cmd_checkpoint_status)

    p_verify_checkpoint = sub.add_parser(
        "verify-checkpoint",
        help="Deeply verify one published checkpoint.",
    )
    p_verify_checkpoint.add_argument("--checkpoint", required=True)
    p_verify_checkpoint.add_argument("--json", action="store_true")
    p_verify_checkpoint.set_defaults(func=cmd_verify_checkpoint)

    p_restore_checkpoint = sub.add_parser(
        "restore-checkpoint",
        help="Restore a checkpoint into an empty target directory.",
    )
    p_restore_checkpoint.add_argument("--checkpoint", required=True)
    p_restore_checkpoint.add_argument("--target-empty-dir", required=True)
    p_restore_checkpoint.add_argument(
        "--apply",
        action="store_true",
        help="Perform the restore after verification.",
    )
    p_restore_checkpoint.add_argument("--json", action="store_true")
    p_restore_checkpoint.set_defaults(func=cmd_restore_checkpoint)

    p_cfg = sub.add_parser(
        "config-check",
        help="Validate campaign.yaml and summarise effective split inputs.",
        description=(
            "Validate the current campaign.yaml schema and print the key "
            "campaign/bootstrap/FEREBUS split settings used by the daemon."
        ),
    )
    add_campaign(p_cfg)
    config_output = p_cfg.add_mutually_exclusive_group()
    config_output.add_argument(
        "--human",
        action="store_true",
        help="Print a concise Config and Pool summary for users.",
    )
    config_output.add_argument(
        "--json",
        action="store_true",
        help="Explicitly request the existing machine-readable JSON output.",
    )
    p_cfg.set_defaults(func=cmd_config_check)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        expanded_argv = expand_boolean_short_flag_clusters(argv)
    except ShortFlagClusterError as exc:
        parser.exit(2, parser.prog + ": error: " + str(exc) + "\n")
    args = parser.parse_args(expanded_argv)
    try:
        return_code = int(args.func(args))
    except CampaignDirResolutionError as exc:
        _finish_background_child_status(
            2,
            failure="CampaignDirResolutionError: " + str(exc),
        )
        parser.exit(2, parser.prog + ": error: " + str(exc) + "\n")
    except BaseException as exc:
        _finish_background_child_status(
            1,
            failure=type(exc).__name__ + ": " + str(exc),
        )
        raise
    _finish_background_child_status(return_code)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
