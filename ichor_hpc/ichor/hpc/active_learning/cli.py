"""CLI for the ICHOR active-learning daemon.

Console entry point 'ichor-al-daemon' registered in
'ichor_cli/setup.cfg'. Subcommands available:

    start      Start the live daemon detached by default; use --foreground to block.
    stop       Write a durable immediate or boundary stop request.
    status     Print the current state snapshot.
    resume     Equivalent to start when state.json already exists.
    reconcile  Inspect on-disk artefacts and propose a recovered state.
    journal    Tail or filter the campaign journal.
    init       Bootstrap campaign.yaml, daemon state, config lock, and pool.

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
from .strict_json import strict_json as json
import secrets
import subprocess
import sys
import time
import numpy as np
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
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
from .daemon.journal import (
    KNOWN_EVENT_TYPES,
    JournalCorruptionError,
    iter_events,
    read_events,
)
from .daemon.lease import evaluate_lease_liveness, validate_lease_heartbeat
from .submit.slurm_contracts import (
    parse_squeue_job_id,
    run_scheduler_command,
    validate_parent_job_id,
)
from .daemon.config_lock import (
    apply_config_lock_update,
    archive_ferebus_iteration_staging_for_retrain,
    archive_scripts_for_reconcile,
    archive_data_staging_for_ferebus_reentry,
    archive_data_staging_for_operator_reconcile,
    archive_reference_data_staging_for_reconcile,
    assert_config_unchanged_for_start,
    clean_model_iteration_staging_for_reconcile,
    clean_reentry_staging,
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
from .daemon.live_executor import (
    LiveBackendNotAvailableError,
    LiveBackendsPhaseExecutor,
    make_live_job_accounting_finder,
    make_live_job_finder,
    make_live_job_liveness_checker,
)
from .daemon.preflight import check_backends, missing_backend_message
from .daemon.submitted_environment_smoke import run_submitted_environment_smoke
from .daemon.recovery_contracts import (
    recovery_contract_status,
    staging_handoff_decisions,
    validate_phase_recovery_contract,
)
from .daemon.reconcile import (
    data_staging_inventory,
    propose_recovery,
    restore_archived_bootstrap_handoff,
    stateful_campaign_artifacts,
    write_proposed_state,
)
from .daemon.array_recovery import (
    array_ledger_path,
    archive_existing_array_task_outputs,
    compact_array_recovery_summary,
    read_array_ledger,
    refresh_array_ledger,
    supports_partial_array_recovery,
)
from .daemon.reconcile_transaction import (
    ReconcileTransaction,
    begin_reconcile_transaction,
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
from .daemon.filesystem import operational_data_dir, operational_path
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


BACKGROUND_CHILD_ENV = "ICHOR_DAEMON_BACKGROUND_CHILD"
BACKGROUND_READINESS_ENV = "ICHOR_DAEMON_READINESS_PATH"
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
    CampaignPhase.APPEND,
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
        "stop_request": operational_path(campaign_dir, "stop_request.json"),
    }


def _stop_control_status(
    campaign: Path,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    from .daemon.stop_control import read_stop_request, stop_request_summary

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
    return {
        "stop_request": stop_request_summary(request),
        "stop_control_error": None,
    }


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
        accepts_timeout = (
            "timeout_seconds" in signature.parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        )
    except (TypeError, ValueError):
        accepts_timeout = True
    if accepts_timeout:
        kwargs["timeout_seconds"] = int(timeout_seconds)
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
        status["lease_age_seconds"] = liveness.age_seconds
        if liveness.error:
            status["lease_probe_error"] = liveness.error
    except Exception as exc:
        status["lease_probe_error"] = type(exc).__name__ + ": " + str(exc)
    return status


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


def _probe_background_daemon(pid_path: Path, log_path: Path) -> Dict[str, Any]:
    payload = _read_background_pid_payload(pid_path)
    pid = payload.get("pid")
    alive = _pid_is_alive(pid)
    out = {
        "background_pid_path": str(pid_path),
        "background_log_path": str(payload.get("log_path") or log_path),
        "background_pid": pid,
        "background_pid_alive": alive,
    }
    if payload:
        out["background_pid_payload"] = payload
    return out


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
    status.update(_probe_background_daemon(paths["background_pid"], paths["background_log"]))
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
    print("Some cleanup/archive operations already happened:", file=sys.stderr)
    for path in paths:
        print("  - " + str(path), file=sys.stderr)
    print("State/config lock was not applied.", file=sys.stderr)


def _atomic_write_background_pid(pid_path: Path, payload: Dict[str, Any]) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pid_path.with_name(pid_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(pid_path)


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


def _terminate_unready_background_child(child: subprocess.Popen) -> None:
    """Stop a detached child that did not establish daemon ownership."""
    try:
        child.terminate()
    except OSError:
        return
    wait = getattr(child, "wait", None)
    if not callable(wait):
        return
    try:
        wait(timeout=5.0)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    kill = getattr(child, "kill", None)
    if callable(kill):
        try:
            kill()
            wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


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
    existing = _probe_background_daemon(pid_path, log_path)
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
    readiness_path = paths["data"] / (
        "daemon.readiness."
        + str(os.getpid())
        + "."
        + secrets.token_hex(8)
        + ".json"
    )
    env[BACKGROUND_READINESS_ENV] = str(readiness_path)
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
        print(
            "could not launch background daemon: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=sys.stderr,
        )
        return 9

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
    readiness_payload = None
    while time.monotonic() < deadline:
        rc = child.poll()
        if rc is not None:
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
            try:
                readiness_path.unlink(missing_ok=True)
            except OSError:
                pass
            return int(rc) if int(rc) != 0 else 9
        if readiness_path.is_file() and not readiness_path.is_symlink():
            try:
                readiness_payload = json.loads(
                    readiness_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                readiness_payload = None
            if isinstance(readiness_payload, dict) and readiness_payload.get("ready") is True:
                break
        time.sleep(0.1)
    if not isinstance(readiness_payload, dict) or readiness_payload.get("ready") is not True:
        _terminate_unready_background_child(child)
        print(
            "background daemon did not acknowledge preflight and lock readiness "
            "within " + str(timeout_seconds) + " seconds; log: " + str(log_path),
            file=sys.stderr,
        )
        try:
            readiness_path.unlink(missing_ok=True)
        except OSError:
            pass
        return 9

    payload = {
        "schema_version": BACKGROUND_PID_SCHEMA_VERSION,
        "pid": int(child.pid),
        "command": argv,
        "campaign_dir": str(campaign),
        "log_path": str(log_path),
        "started_at_utc": timestamp,
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "python_executable": sys.executable,
        "readiness_path": str(readiness_path),
        "readiness": readiness_payload,
    }
    _atomic_write_background_pid(pid_path, payload)
    try:
        readiness_path.unlink(missing_ok=True)
    except OSError:
        pass
    print("daemon started in background")
    print("  pid: " + str(child.pid))
    print("  log: " + str(log_path))
    print("  pid_file: " + str(pid_path))
    print("  status: ichor-al-daemon status --campaign-dir " + str(campaign))
    print("  journal: ichor-al-daemon journal --campaign-dir " + str(campaign) + " --json | tail -n 40")
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


def _latest_journal_event(journal_path: Path, event_type: str) -> Optional[Dict[str, Any]]:
    latest = None
    if not journal_path.exists():
        return None
    for event in iter_events(journal_path):
        if event.get("event") == event_type:
            latest = event
    return latest


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


_PHASE_MEANINGS: Dict[str, str] = {
    CampaignPhase.INIT.value: "campaign initialised; no sampling phase has run yet",
    CampaignPhase.PHASE_A_DIVERSITY.value: "initial ICHOR diversity sampling is next",
    CampaignPhase.INITIAL_GAUSSIAN.value: "initial Gaussian labelling is next",
    CampaignPhase.INITIAL_AIMALL.value: "initial AIMAll postprocessing is next",
    CampaignPhase.INITIAL_ALLOCATION_CHECK.value: "bootstrap point-allocation completeness is being checked",
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value: "replacement bootstrap Gaussian labelling is next",
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value: "replacement bootstrap AIMAll postprocessing is next",
    CampaignPhase.INITIAL_FEREBUS.value: "bootstrap FEREBUS model training is next",
    CampaignPhase.SEED_SELECT.value: "active-learning seed selection is next",
    CampaignPhase.ARIADNE_ARRAY.value: "ARIADNE adversarial landing is next",
    CampaignPhase.PHASE_B_DIVERSITY.value: "Phase B diversity selection is next",
    CampaignPhase.SPLIT.value: "the pre-QM exact slot allocation is being verified",
    CampaignPhase.GAUSSIAN.value: "active Gaussian labelling is next",
    CampaignPhase.AIMALL.value: "active AIMAll postprocessing is next",
    CampaignPhase.ALLOCATION_CHECK.value: "active point-allocation completeness is being checked",
    CampaignPhase.REPLACEMENT_GAUSSIAN.value: "replacement active Gaussian labelling is next",
    CampaignPhase.REPLACEMENT_AIMALL.value: "replacement active AIMAll postprocessing is next",
    CampaignPhase.APPEND.value: "accepted AIMAll outputs are being appended",
    CampaignPhase.FEREBUS.value: "FEREBUS model retraining is next",
    CampaignPhase.STOP_CHECK.value: "iteration stop/continue decision is next",
    CampaignPhase.DONE.value: "campaign is complete",
    CampaignPhase.HALTED.value: "campaign is halted and needs user review",
}

_BOOTSTRAP_NOT_READY_PHASES = {
    CampaignPhase.INIT.value,
    CampaignPhase.PHASE_A_DIVERSITY.value,
    CampaignPhase.INITIAL_GAUSSIAN.value,
    CampaignPhase.INITIAL_AIMALL.value,
    CampaignPhase.INITIAL_ALLOCATION_CHECK.value,
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN.value,
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL.value,
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
    CampaignPhase.APPEND.value,
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
    CampaignPhase.APPEND.value,
    CampaignPhase.STOP_CHECK.value,
}


def _phase_meaning(phase: Any) -> str:
    return _PHASE_MEANINGS.get(str(phase or ""), "phase is not recognised")


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
    if phase in _BOOTSTRAP_NOT_READY_PHASES:
        version = _status_version(item)
        if item.get("ok") is True and version >= 0:
            return "available early (v" + str(version) + ")"
        return "not produced yet"
    if phase == CampaignPhase.INITIAL_FEREBUS.value:
        version = _status_version(item)
        if item.get("ok") is True and version >= 0:
            return "ready for initial FEREBUS (v" + str(version) + ")"
        return "using initial AIMAll handoff"
    if phase in _TRAINING_REQUIRED_PHASES:
        return _product_ok_text(item, label="QM reference data")
    if phase in (CampaignPhase.DONE.value, CampaignPhase.HALTED.value):
        return _product_ok_text(item, label="QM reference data")
    return _product_ok_text(item, label="QM reference data")


def _models_product_status(phase: str, item: Dict[str, Any]) -> str:
    if phase in _BOOTSTRAP_NOT_READY_PHASES:
        version = _status_version(item)
        if item.get("ok") is True and version >= 0:
            return "available early (v" + str(version) + ")"
        return "not produced yet"
    if phase == CampaignPhase.INITIAL_FEREBUS.value:
        version = _status_version(item)
        if item.get("ok") is True and version >= 0:
            return "available (v" + str(version) + ")"
        return "being produced by INITIAL_FEREBUS"
    if phase == CampaignPhase.FEREBUS.value:
        version = _status_version(item)
        if item.get("ok") is True and version >= 0:
            return "current model ready (v" + str(version) + "); update in progress"
        return "being produced by FEREBUS"
    if phase in _MODELS_REQUIRED_PHASES:
        return _product_ok_text(item, label="models")
    if phase in (CampaignPhase.DONE.value, CampaignPhase.HALTED.value):
        return _product_ok_text(item, label="models")
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
        rows.append(("bootstrap QM reference data", "not checked"))
        rows.append(("FEREBUS models", "not checked"))
    else:
        reference_data = status.get("reference_data")
        if isinstance(reference_data, dict):
            rows.append(
                (
                    "bootstrap QM reference data",
                    _reference_data_product_status(phase_name, reference_data),
                )
            )
            if (
                phase_name in _BOOTSTRAP_NOT_READY_PHASES
                and reference_data.get("ok") is not True
            ):
                rows.append(("reference data expected after", "INITIAL_AIMALL / INITIAL_FEREBUS"))
            if verbose:
                rows.append(("reference-data version", reference_data.get("version")))
                for error in reference_data.get("errors") or []:
                    rows.append(("reference-data detail", error))
        else:
            rows.append(("bootstrap QM reference data", "not checked"))

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
    if contract_text == "ok":
        rows.append(("current phase contract", "ready for " + (phase_name or "current phase")))
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


def _format_background_daemon(pid: Any, alive: Any) -> str:
    if pid is None:
        return "not running"
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


def _format_recommendations(payload: Dict[str, Any]) -> List[str]:
    first = _first_recommendation(payload)
    rows: List[tuple[str, Any]] = [
        ("severity", first.get("severity")),
        ("primary", first.get("primary")),
        ("why", first.get("why")),
    ]
    if first.get("command"):
        rows.append(("command", first.get("command")))
    details = first.get("details")
    if isinstance(details, list):
        for detail in details[:5]:
            rows.append(("detail", detail))
    recommendations = payload.get("recommendations")
    if isinstance(recommendations, list) and len(recommendations) > 1:
        for recommendation in recommendations[1:4]:
            if isinstance(recommendation, dict):
                rows.append(
                    (
                        "secondary",
                        str(recommendation.get("code"))
                        + ": "
                        + str(recommendation.get("primary")),
                    )
                )
    return _section("Next Action", rows)


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
        ("recorded Slurm jobs", _active_pending_jobs_summary(payload.get("pending_jobs"))),
        (
            "submission intents",
            _format_active_submission_intents(payload.get("active_submission_intents")),
        ),
        ("user stop control", stop_summary),
        ("shutdown requested", "yes" if payload.get("shutdown_requested") else "no"),
    ]
    allocation = payload.get("point_allocation_summary")
    if isinstance(allocation, dict):
        rows.extend(
            [
                ("allocation accepted", allocation.get("accepted_total")),
                ("allocation deficit", allocation.get("deficit_total")),
                ("allocation reserve available", allocation.get("reserve_available")),
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
    return (
        str(payload.get("iteration"))
        + " of "
        + str(payload.get("max_iterations"))
        + " active iterations planned"
    )


def _format_status(payload: Dict[str, Any], *, verbose: bool, journal_path: Path) -> str:
    lines: List[str] = []
    lines.extend(
        _section(
            "Campaign",
            [
                ("phase", payload.get("phase")),
                ("meaning", _phase_meaning(payload.get("phase"))),
                ("iteration", _iteration_summary(payload)),
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
            verbose=verbose,
        )
    )
    lines.append("")
    lines.extend(_format_runtime_status(payload, verbose=verbose))
    lines.extend(_format_lifecycle_status(payload))
    partial_array = payload.get("partial_array_recovery")
    if isinstance(partial_array, dict):
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
    if payload.get("phase") == "HALTED":
        halt = _latest_journal_event(journal_path, "halt")
        lines.append("")
        lines.extend(
            _section(
                "Halt",
                [
                    ("reason", (halt or {}).get("reason", "unknown")),
                    ("from_phase", (halt or {}).get("from_phase")),
                ],
            )
        )
    lines.append("")
    lines.extend(_format_recommendations(payload))
    if verbose:
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


def _format_status_unavailable(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    campaign_rows = [
        ("phase", "unknown"),
        ("state", payload.get("status_error")),
    ]
    if "campaign_yaml_exists" in payload:
        campaign_rows.append(("campaign.yaml", "present" if payload.get("campaign_yaml_exists") else "missing"))
    if "stateful_artifacts_count" in payload:
        campaign_rows.append(("stateful artefacts", payload.get("stateful_artifacts_count")))
    if "fresh_init_safe" in payload:
        campaign_rows.append(("fresh init safe", bool(payload.get("fresh_init_safe"))))
    feasibility = payload.get("pool_feasibility")
    if isinstance(feasibility, dict):
        campaign_rows.append(
            (
                "pool feasibility",
                "ok" if feasibility.get("ok") else "failed",
            )
        )
    lines.extend(
        _section(
            "Campaign",
            campaign_rows,
        )
    )
    if payload.get("state_error"):
        lines.append("")
        lines.extend(_section("State", [("error", payload.get("state_error"))]))
    if payload.get("stop_request") or payload.get("stop_control_error"):
        from .daemon.stop_control import describe_stop_request

        lines.append("")
        stop_request = payload.get("stop_request")
        lines.extend(
            _section(
                "User Stop Control",
                [
                    (
                        "request",
                        (
                            describe_stop_request(stop_request)
                            if isinstance(stop_request, dict)
                            else "invalid"
                        ),
                    ),
                    ("error", payload.get("stop_control_error")),
                ],
            )
        )
    lines.append("")
    lines.extend(_format_recommendations(payload))
    return "\n".join(lines) + "\n"


def _event_time(event: Dict[str, Any]) -> str:
    ts = str(event.get("ts", "")).strip()
    if not ts:
        return "--:--:--"
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        if "T" in ts:
            date_part, tail = ts.split("T", 1)
            time_part = (
                tail.split(".", 1)[0]
                .replace("+00:00", "")
                .replace("Z", "")
            )
            if date_part and time_part:
                return date_part[:10] + " " + time_part[:8]
        return ts


def _event_iteration(event: Dict[str, Any]) -> str:
    value = event.get("iteration")
    if value is None:
        return "iter=-"
    return "iter=" + str(value)


def _event_phase(event: Dict[str, Any]) -> str:
    for key in ("phase", "to_phase", "from_phase"):
        value = event.get(key)
        if value is not None:
            return str(value)
    return "-"


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
    "models_committed": "models committed",
    "seed_selected": "seeds selected",
    "anti_overlap_flagged": "anti-overlap flagged",
    "reference_scales_computed": "reference scales computed",
    "failure_action": "phase failure decision",
    "halt": "daemon halted",
    "live_postprocess_refused": "live postprocess refused",
    "effective_config_diff": "config diff recorded",
    "autotune_applied": "autotune applied",
    "trajectory_pool_imported": "trajectory pool imported",
    "bootstrap_inputs_confirmed": "bootstrap inputs confirmed",
    "model_bootstrap_staged": "imported models staged",
    "model_bootstrap_committed": "imported models committed",
    "quantum_output_rejected": "QM output rejected",
    "quantum_quality_summary": "QM quality summarised",
    "ferebus_quality_summary": "FEREBUS quality summarised",
    "ariadne_landing_rejected": "ARIADNE landing rejected",
    "ariadne_landing_summary": "ARIADNE landing summary",
    "ariadne_optional_diagnostics_warning": "ARIADNE diagnostics warning",
    "ariadne_legacy_missing_trajectory_sha256": "ARIADNE legacy provenance",
    "ariadne_provenance_reconstructed": "ARIADNE provenance rebuilt",
    "ariadne_seed_provenance_repaired": "ARIADNE seed provenance repaired",
    "ariadne_seed_provenance_staged": "ARIADNE seed provenance staged",
    "ariadne_stale_outputs_quarantined": "ARIADNE stale outputs quarantined",
    "ariadne_task_rejected_missing_result": "ARIADNE result missing",
    "ariadne_task_rejected_malformed_result": "ARIADNE malformed result",
    "ariadne_task_rejected_unusable_result": "ARIADNE result unusable",
    "ariadne_task_rejected_unsafe_landing": "ARIADNE landing rejected",
    "ariadne_task_salvaged_from_nonzero_exit": "ARIADNE task salvaged",
    "error_calibration_summary": "error calibration summarised",
    "error_calibration_failed": "error calibration failed",
    "phase_output_contract_invalid": "phase output contract invalid",
    "required_phase_output_missing_after_failure": "failure output missing",
    "reconcile_applied": "reconcile applied",
    "reconcile_resolved_terminal_intent": "terminal intent resolved",
    "partial_array_recovery_prepared": "partial array recovery prepared",
    "partial_array_recovery_postprocess_only": "partial array postprocess ready",
    "committed_artifact_settle_retry": "waiting for committed artefacts",
    "resolved_phase_resources": "resources resolved",
    "user_cancelled_jobs": "user cancelled jobs",
    "user_stop_requested": "user stop requested",
    "user_stop_boundary_reached": "user stop boundary reached",
    "user_stop_control_invalid": "user stop control invalid",
    "user_stop_request_cancelled": "user stop request cancelled",
    "user_stop_resumed": "user stop resumed",
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
    "geometry_novelty_scale_precomputed": "novelty scale computed",
    "sampling_protocol_resolved": "sampling protocol resolved",
    "phase_b_novelty_threshold_relaxed": "Phase B novelty relaxed",
    "pool_feasibility_checked": "pool feasibility checked",
    "seed_posterior_fallback": "seed posterior fallback",
    "initial_training_existing_without_bootstrap_handoff": "bootstrap handoff missing",
}


def _journal_event_label(event: Dict[str, Any]) -> str:
    raw = str(event.get("event", "<missing>"))
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
    "campaign_completed",
    "scientific_convergence_reached",
    "phase_succeeded",
    "phase_succeeded_live",
    "reference_data_committed",
    "models_committed",
    "trajectory_pool_imported",
    "bootstrap_inputs_confirmed",
    "model_bootstrap_committed",
    "error_calibration_summary",
    "user_stop_boundary_reached",
    "user_stop_resumed",
}

_JOURNAL_RUN_EVENTS = {
    "daemon_started",
    "phase_pre_submit_intent",
    "phase_submitted",
    "sbatch",
    "adopted_inflight_job",
    "partial_array_recovery_prepared",
    "partial_array_recovery_postprocess_only",
    "scheduler_uncertain_resumed",
}

_JOURNAL_WAIT_EVENTS = {
    "sacct_empty_but_squeue_active",
    "sacct_rows_missing_but_squeue_active",
    "postprocess_settle_retry",
    "committed_artifact_settle_retry",
    "user_stop_requested",
}

_JOURNAL_WARN_EVENTS = {
    "campaign_reopened",
    "phase_completion_replayed",
    "submission_intent_completion_deferred",
    "sacct_error",
    "quantum_output_rejected",
    "ariadne_landing_rejected",
    "ariadne_optional_diagnostics_warning",
    "phase_b_novelty_threshold_relaxed",
    "daemon_lease_stale_recovered",
    "user_stop_request_cancelled",
}

_JOURNAL_FAIL_EVENTS = {
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
    "user_stop_control_invalid",
}

_SQUEUE_RUNNING_STATES = {"R", "RUNNING", "CG", "COMPLETING"}
_SQUEUE_PENDING_STATES = {"PD", "PENDING", "CF", "CONFIGURING"}


def _journal_event_severity(event: Dict[str, Any]) -> str:
    raw = str(event.get("event", ""))
    if raw == "ariadne_landing_summary":
        rejected = _event_int(event, "rejected")
        return "WARN" if rejected is not None and rejected > 0 else "OK"
    if raw == "queue_lifecycle_update":
        status = str(event.get("status") or "").upper()
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
        "T/C/R/P="
        + str(int(total))
        + "/"
        + _format_progress_value(completed)
        + "/"
        + _format_progress_value(running)
        + "/"
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


def _journal_operator_summary(event: Dict[str, Any]) -> str:
    raw = str(event.get("event", ""))
    if raw in {"user_stop_requested", "user_stop_boundary_reached"}:
        from .daemon.stop_control import describe_stop_request

        return describe_stop_request(
            event,
            completed=raw == "user_stop_boundary_reached",
        )
    if raw in {"phase_pre_submit_intent", "phase_submitted", "sbatch"}:
        return "array submitted" if _journal_array_progress(event) else "job submitted"
    if raw == "queue_lifecycle_update":
        status = str(event.get("status") or "").upper()
        if status in _SQUEUE_PENDING_STATES:
            return "array pending" if _journal_array_progress(event) else "job pending"
        if status in _SQUEUE_RUNNING_STATES:
            return "array active" if _journal_array_progress(event) else "job active"
        return "queue state updated"
    if raw in {"phase_succeeded", "phase_succeeded_live"}:
        return "array complete" if _journal_array_progress(event) else "phase succeeded"
    if raw == "halt":
        return "halt"
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
    return ""


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
        ("n_frames", "frames"),
        ("pool_n_frames", "pool"),
        ("required_pool_frames", "required"),
        ("action", "action"),
        ("reason", "reason"),
        ("error", "error"),
    ]
    parts: List[str] = []
    for key, label in detail_keys:
        if key in event and event.get(key) is not None:
            value = _format_value(event.get(key))
            if key == "campaign_uid" and len(value) > 8:
                value = value[:8]
            if key in {"reason", "error"} and len(value) > 90:
                value = value[:87] + "..."
            parts.append(label + "=" + value)
    progress = _journal_array_progress(event)
    if progress:
        insert_at = 1 if parts and parts[0].startswith("job=") else 0
        parts.insert(insert_at, progress)
    raw = str(event.get("event", ""))
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
    for key in sorted(event):
        if key in skip:
            continue
        value = _format_value(event[key])
        if len(value) > 60:
            value = value[:57] + "..."
        parts.append(str(key) + "=" + value)
    return " ".join(parts)


def _format_journal_events(events: Sequence[Dict[str, Any]], *, verbose: bool) -> str:
    if not events:
        return "Timeline\n  no matching events\n"
    rows: List[Tuple[Dict[str, Any], str, str, str, str, str]] = []
    for event in events:
        summary = _journal_operator_summary(event) or _journal_event_label(event)
        phase = _event_phase(event)
        rows.append(
            (
                event,
                _event_time(event),
                _event_iteration(event),
                phase,
                _journal_event_severity(event),
                summary,
            )
        )
    time_width = max(19, max(len(row[1]) for row in rows))
    iteration_width = max(6, max(len(row[2]) for row in rows))
    phase_width = max(18, max(len(row[3]) for row in rows))

    lines: List[str] = ["Timeline"]
    for event, event_time, iteration, phase, severity, summary in rows:
        line = (
            "  "
            + event_time.ljust(time_width)
            + "  "
            + ("[" + severity + "]").ljust(7)
            + "  "
            + phase.ljust(phase_width)
            + "  "
            + iteration.ljust(iteration_width)
            + "  "
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
                "  ichor-al-daemon init --campaign-dir " + str(campaign),
                file=sys.stderr,
            )
            print(
                "Then bind the new campaign to live mode with:",
                file=sys.stderr,
            )
            print(
                "  ichor-al-daemon start --campaign-dir "
                + str(campaign)
                + " --mode live",
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
                "  ichor-al-daemon reconcile --campaign-dir " + str(campaign),
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

    from .execution_identity import ExecutionIdentityError, ensure_execution_identity

    placeholder_system = config.campaign.system_name in {
        "SYSTEM",
        "CHANGE_ME_SYSTEM",
    }
    if getattr(args, "mode", None) == "live" and placeholder_system:
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
            requested_mode=getattr(args, "mode", None),
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
    resource_usage_collector = None
    if effective_mode == "live":
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
        sacct_poller = None
        # live mode: let the daemon spot + adopt an orphaned in-flight job on (re)entry rather than
        # double-submitting after a crash or reconcile (A24/A25).
        scheduler_timeout = int(
            config.runtime.scheduler_command_timeout_seconds
        )
        job_finder = _call_timeout_aware(
            make_live_job_finder,
            campaign_dir=campaign,
            timeout_seconds=scheduler_timeout,
        )
        job_name_accounting_finder = _call_timeout_aware(
            make_live_job_accounting_finder,
            timeout_seconds=scheduler_timeout,
        )
        job_liveness_checker = _call_timeout_aware(
            make_live_job_liveness_checker,
            timeout_seconds=scheduler_timeout,
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
            "slurm" if effective_mode == "live" else "synthetic"
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
    if resource_usage_collector is not None:
        daemon_kwargs["resource_usage_collector"] = resource_usage_collector
    d = Daemon(**daemon_kwargs)
    if args.poll_interval is not None:
        #override config-loaded poll interval per-invocation.
        d.config.runtime.poll_interval_seconds = int(args.poll_interval)
    readiness_callback = None
    readiness_value = os.environ.get(BACKGROUND_READINESS_ENV)
    if readiness_value:
        readiness_path = Path(readiness_value)

        def _acknowledge_background_readiness() -> None:
            from .daemon.state import atomic_write_json

            atomic_write_json(
                readiness_path,
                {
                    "schema_version": 1,
                    "ready": True,
                    "pid": int(os.getpid()),
                    "campaign_dir": str(campaign),
                    "mode": str(effective_mode),
                    "execution_identity_digest_sha256": str(
                        execution_identity.get("digest_sha256") or ""
                    ),
                    "acknowledged_at_unix": float(time.time()),
                },
            )

        readiness_callback = _acknowledge_background_readiness
    return d.run(
        max_ticks=args.max_ticks,
        catch_keyboard_interrupt=True,
        readiness_callback=readiness_callback,
    )


_FEREBUS_JOB_NAME_EXTERNAL_PHASES = frozenset()


def _load_active_submission_intents(
    campaign: Path,
    *,
    errors: Optional[List[Dict[str, str]]] = None,
    fail_on_error: bool = False,
    expected_campaign_uid: Optional[str] = None,
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
    return [
        dict(payload)
        for payload in inventory.get("records", [])
        if str(payload.get("status")) in _submission_intent.ACTIVE_STATUSES
    ]


def _lookup_active_slurm_job_for_cancel(
    job_id: str,
    *,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    cmd = [
        "squeue",
        "-j",
        str(job_id),
        "--noheader",
        "--format=%i|%T|%j",
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
        parts = line.split("|", 2)
        if len(parts) != 3:
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
) -> Tuple[bool, str]:
    from .submit.sacct_poll import JobStatus, aggregate_states, poll_job

    deadline = time.monotonic() + float(confirmation_timeout_seconds)
    last_reason = "scheduler has not confirmed cancellation"
    while True:
        try:
            observations = poll_job(
                job_id,
                timeout_seconds=int(command_timeout_seconds),
            )
            summary = aggregate_states(
                job_id,
                observations,
                expected_task_count=expected_tasks,
                submission_kind=submission_kind,
            )
            if summary.is_terminal:
                if summary.observations and all(
                    observation.status is JobStatus.CANCELLED
                    for observation in summary.observations
                ):
                    return True, ""
                states = sorted(
                    {
                        str(observation.status.value)
                        for observation in summary.observations
                    }
                )
                return False, (
                    "job became terminal without cancellation-derived states: "
                    + ", ".join(states)
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
            last_reason = "job remains active or completing in squeue"
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
    if not expected:
        return True
    actual = {str(row.get("job_name") or "") for row in rows}
    return bool(actual.intersection(expected))


def _collect_stop_cancel_jobs(campaign: Path, state: Any) -> Dict[str, Dict[str, Any]]:
    jobs: Dict[str, Dict[str, Any]] = {}

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
        if phase_name not in _FEREBUS_JOB_NAME_EXTERNAL_PHASES:
            item["expected_job_names"].add(
                live_job_name(state.campaign_uid, phase_name, int(state.iteration))
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

    return jobs


def _cancel_recorded_slurm_jobs(
    campaign: Path,
    state: Any,
    *,
    command_timeout_seconds: int = 60,
    confirmation_timeout_seconds: int = 120,
) -> Dict[str, Any]:
    jobs = _collect_stop_cancel_jobs(campaign, state)
    cancelled: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    campaign_uid = str(getattr(state, "campaign_uid", "") or "")
    for intent in _load_active_submission_intents(
        campaign,
        fail_on_error=True,
        expected_campaign_uid=(campaign_uid or None),
    ):
        if str(intent.get("job_id") or ""):
            continue
        phase_name, iteration = _intent_phase_iteration(intent)
        skipped.append({
            "job_id": "",
            "reason": "active submission intent has no job_id: "
            + str(phase_name)
            + "@"
            + str(iteration),
        })
    for job_id, item in sorted(jobs.items()):
        lookup = _call_timeout_aware(
            _lookup_active_slurm_job_for_cancel,
            job_id,
            timeout_seconds=int(command_timeout_seconds),
        )
        if bool(lookup.get("inconclusive")):
            failed.append({
                "job_id": job_id,
                "reason": "squeue lookup inconclusive: " + str(lookup.get("error") or "unknown error"),
            })
            continue
        if not bool(lookup.get("active")):
            skipped.append({
                "job_id": job_id,
                "reason": "not active in squeue",
            })
            continue
        expected_names = sorted(str(name) for name in item.get("expected_job_names", set()) if str(name))
        rows = list(lookup.get("rows") or [])
        if not _job_name_matches_expected(rows, expected_names):
            actual_names = sorted({str(row.get("job_name") or "") for row in rows})
            failed.append({
                "job_id": job_id,
                "reason": "scheduler job name mismatch",
                "expected_job_names": expected_names,
                "actual_job_names": actual_names,
            })
            continue
        expected_tasks = item.get("expected_tasks")
        if expected_tasks is not None and (
            isinstance(expected_tasks, bool)
            or not isinstance(expected_tasks, int)
            or expected_tasks <= 0
        ):
            failed.append({
                "job_id": job_id,
                "reason": "submission intent has an invalid expected task count",
            })
            continue
        submission_kind = str(item.get("submission_kind") or "")
        if submission_kind not in {"scalar", "array"}:
            failed.append({
                "job_id": job_id,
                "reason": "submission kind is unavailable for cancellation confirmation",
            })
            continue
        if submission_kind == "array" and expected_tasks is None:
            failed.append({
                "job_id": job_id,
                "reason": (
                    "array task cardinality is unavailable; cancellation was not "
                    "issued because complete terminal confirmation would be impossible"
                ),
            })
            continue
        ok, message = _call_timeout_aware(
            _run_scancel,
            job_id,
            timeout_seconds=int(command_timeout_seconds),
        )
        if not ok:
            failed.append({
                "job_id": job_id,
                "reason": "scancel failed: " + message,
            })
            continue
        confirmed, confirmation_reason = _confirm_cancelled_slurm_job(
            job_id,
            expected_tasks=expected_tasks,
            submission_kind=submission_kind,
            confirmation_timeout_seconds=int(confirmation_timeout_seconds),
            command_timeout_seconds=int(command_timeout_seconds),
        )
        if not confirmed:
            failed.append({
                "job_id": job_id,
                "reason": confirmation_reason,
            })
            continue
        phases = sorted(str(phase) for phase in item.get("phases", set()))
        cancelled.append({
            "job_id": job_id,
            "phases": phases,
            "intent_keys": [
                {"phase": str(phase_name), "iteration": int(iteration)}
                for phase_name, iteration in sorted(item.get("intent_keys", set()))
            ],
        })
    return {
        "cancelled": cancelled,
        "skipped": skipped,
        "failed": failed,
    }


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


def _journal_cancel_jobs_summary(journal_path: Path, summary: Dict[str, Any]) -> None:
    try:
        from .daemon.journal import append_event

        append_event(
            journal_path,
            "user_cancelled_jobs",
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
        )
    except Exception:
        pass


def _print_cancel_jobs_summary(summary: Dict[str, Any]) -> None:
    cancelled = list(summary.get("cancelled") or [])
    skipped = list(summary.get("skipped") or [])
    failed = list(summary.get("failed") or [])
    if cancelled:
        print("Cancelled Slurm jobs:")
        for item in cancelled:
            phases = ", ".join(str(phase) for phase in item.get("phases", []))
            suffix = " (" + phases + ")" if phases else ""
            print("  - " + str(item.get("job_id")) + suffix)
    if skipped:
        print("Skipped Slurm jobs:")
        for item in skipped:
            print("  - " + str(item.get("job_id")) + ": " + str(item.get("reason")))
    if failed:
        print("Jobs not cancelled:", file=sys.stderr)
        for item in failed:
            print("  - " + str(item.get("job_id")) + ": " + str(item.get("reason")), file=sys.stderr)
    if not cancelled and not skipped and not failed:
        print("No recorded active Slurm jobs found.")


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
                cancel_summary = _cancel_recorded_slurm_jobs(
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
                cancel_summary = _cancel_recorded_slurm_jobs(
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
    cancel_summary = None
    if cancel_jobs:
        try:
            cancel_summary = _cancel_recorded_slurm_jobs(
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
        _journal_cancel_jobs_summary(paths["journal"], cancel_summary)
    print(describe_stop_request(request))
    print("request id: " + str(request.get("request_id")))
    print("stop request: " + str(paths["stop_request"]))
    background = _probe_background_daemon(
        paths["background_pid"],
        paths["background_log"],
    )
    if background.get("background_pid") is not None:
        suffix = " alive" if background.get("background_pid_alive") else " not running"
        print("background pid: " + str(background.get("background_pid")) + " (" + suffix.strip() + ")")
        print("background log: " + str(background.get("background_log_path")))
    if cancel_summary is not None:
        _print_cancel_jobs_summary(cancel_summary)
        if cancel_summary.get("failed"):
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
    intents = _load_active_submission_intents(campaign, errors=intent_errors)
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
        from .acquisition.trajectory_pool import TrajectoryPool

        pool = TrajectoryPool.load(campaign)
        pool_status = "ok sha=" + str(pool.sha256)[:12]
    except Exception as exc:
        pool_status = type(exc).__name__ + ": " + str(exc)[:120]
    lines.extend(_section("Trajectory pool", [("status", pool_status)]))

    staging_inventory = data_staging_inventory(campaign)
    staging_archive_hint = "not needed"
    if int(staging_inventory.get("top_level_count") or 0) > 0:
        staging_archive_hint = (
            "inspect first; if stale and no active jobs remain, run "
            "reconcile --archive-staging --apply"
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

    recommendation = "start/resume is safe"
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
            recommendation = "run reconcile and inspect unsafe artefacts"
        elif state_invalid:
            recommendation = "run reconcile"
        elif state is None:
            artefacts = stateful_campaign_artifacts(campaign)
            recommendation = (
                "run reconcile"
                if artefacts
                else "start/resume is safe for clean first run"
            )
    except Exception as exc:
        recommendation = "run reconcile; recovery probe failed: " + str(exc)[:120]
    if (
        intents
        or lock_status.get("lock_held")
        or _lease_is_fresh(
            lease_status.get("lease_heartbeat"),
            stale_seconds=stale_seconds,
            clock_skew_tolerance_seconds=clock_skew,
        )
        or background.get("background_pid_alive")
    ):
        recommendation = "stop --cancel-jobs first, then rerun reconcile"
    if not cfg_path.is_file() and config_lock_path(campaign).is_file():
        recommendation = "restore campaign.yaml from config lock"
    lines.extend(_section("Recommendation", [("next action", recommendation)]))
    return "\n".join(lines) + "\n"


def cmd_status(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(
        args.campaign_dir,
        require_campaign_yaml=False,
    )
    paths = _campaign_paths(campaign)
    try:
        from .daemon.ariadne_quarantine import inventory_quarantine

        quarantine_status = inventory_quarantine(campaign)
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
            print(_format_status_unavailable(payload), end="")
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
            print(_format_status_unavailable(payload), end="")
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
    payload.update(_probe_background_daemon(paths["background_pid"], paths["background_log"]))
    intent_errors: List[Dict[str, str]] = []
    payload["active_submission_intents"] = _load_active_submission_intents(
        campaign,
        errors=intent_errors,
        expected_campaign_uid=str(state.campaign_uid),
    )
    if intent_errors:
        payload["submission_intent_errors"] = intent_errors
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
            )
            payload["point_allocation_summary"] = dict(allocation.get("summary") or {})
    except Exception as exc:
        payload["point_allocation_summary"] = {
            "error": type(exc).__name__ + ": " + str(exc)
        }
    try:
        payload["latest_halt_event"] = _latest_journal_event(
            paths["journal"], "halt"
        )
    except Exception as exc:
        payload["journal_error"] = type(exc).__name__ + ": " + str(exc)
    try:
        if supports_partial_array_recovery(state.phase):
            ledger = read_array_ledger(campaign, state.phase, int(state.iteration))
            if isinstance(ledger, dict):
                payload["partial_array_recovery"] = compact_array_recovery_summary(ledger)
    except Exception as exc:
        payload["partial_array_recovery_error"] = (
            type(exc).__name__ + ": " + str(exc)
        )
    try:
        cfg = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        payload["campaign_config_status"] = {"ok": True}
        payload["pool_feasibility"] = _pool_feasibility_summary(campaign, cfg)
    except Exception as exc:
        payload["campaign_config_status"] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }
    try:
        from .daemon.artifact_contracts import (
            artifact_manifest_status,
            state_artifact_contract_status,
        )
        payload["artifact_manifest_status"] = artifact_manifest_status(campaign, state)
        payload["state_artifact_contract_status"] = state_artifact_contract_status(
            campaign,
            state,
        )
    except Exception as exc:
        payload["artifact_manifest_status"] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }
        payload["state_artifact_contract_status"] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
            "errors": [type(exc).__name__ + ": " + str(exc)],
        }
    payload["recommendations"] = recommendation_dicts(
        build_status_recommendations(campaign, payload, paths["journal"])
    )
    payload["next_action"] = payload["recommendations"][0]["primary"]
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(
            _format_status(
                payload,
                verbose=bool(getattr(args, "verbose", False)),
                journal_path=paths["journal"],
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
    if stop_request is not None and not cancel_stop_request:
        raise ValueError(
            "an active stop request must be cancelled explicitly before recovery"
        )

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
    for suffix in (
        "",
        ":UNKNOWN",
        ":MISSING",
        ":SQUEUE_INCONCLUSIVE:empty",
        ":SQUEUE_INCONCLUSIVE:missing",
        ":ERROR",
    ):
        target.sacct_empty_streak.pop(pending_job_id + suffix, None)
    return target, intent


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
        if lock_status.get("lock_held") is not False:
            print(
                "cannot resume while daemon lock ownership is active or inconclusive",
                file=sys.stderr,
            )
            return 7
        stale_seconds, skew_seconds = _runtime_liveness_policy(campaign)
        lease_status = _probe_daemon_lease(
            paths["lease"],
            stale_seconds=stale_seconds,
            clock_skew_tolerance_seconds=skew_seconds,
        )
        if lease_status.get("lease_fresh") is True:
            print(
                "cannot resume while a fresh daemon lease exists",
                file=sys.stderr,
            )
            return 7
        try:
            from .daemon.stop_control import (
                archive_and_clear_stop_request,
                read_resume_transaction,
                read_stop_request,
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
                        "resume_scheduler_uncertain_cancel_stop"
                        if cancelling_stop
                        else "resume_scheduler_uncertain"
                    ),
                    archive_status="cancelled" if cancelling_stop else "resumed",
                )
                if cancelling_stop:
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
                + " to re-poll preserved Slurm job "
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
                    "run reconcile --apply before reopening a completed campaign",
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
) -> Tuple[Optional[str], str, int]:
    from .submit import sacct_poll

    matching = _matching_sacct_observations(job_id, observations)
    if not matching:
        return None, "sacct returned no rows for job " + str(job_id), 0
    states = [observation.status for observation in matching]
    if any(state in sacct_poll.NON_TERMINAL_STATES for state in states):
        return None, "sacct still has non-terminal rows for job " + str(job_id), len(matching)
    if any(state == sacct_poll.JobStatus.UNKNOWN for state in states):
        return None, "sacct returned UNKNOWN rows for job " + str(job_id), len(matching)
    if not all(state in sacct_poll.TERMINAL_STATES for state in states):
        return None, "sacct rows are not conclusively terminal for job " + str(job_id), len(matching)
    failures = [state for state in states if state in sacct_poll.FAILURE_STATES]
    if failures:
        return failures[0].value, "", len(matching)
    return (
        None,
        "sacct shows successful completion; reconcile will not clear "
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


def _resolve_terminal_submission_intents_for_apply(
    campaign: Path,
    active_intents: Sequence[Dict[str, Any]],
    *,
    pre_submit_stale_seconds: int = 900,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Classify conclusively terminal active intents before safe apply.

    This is intentionally fail-closed.  ``reconcile --apply`` may clear a stale
    active intent only when ``squeue`` no longer sees the job and ``sacct``
    reports terminal failure for the recorded JobID, or for the expected job
    name in the PRE_SUBMIT crash window where the JobID was never persisted.
    Missing scheduler data keeps the intent blocking so a user cannot
    accidentally duplicate a live job.
    """
    from .submit import sacct_poll

    terminal_candidates: List[Dict[str, Any]] = []
    stale_pre_submit_no_job: List[Dict[str, Any]] = []
    blocking: List[Dict[str, Any]] = []
    for intent in active_intents:
        phase, iteration = _intent_phase_iteration(intent)
        job_id = str(intent.get("job_id") or "")
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
                lookup = sacct_poll.find_accounted_job_by_name_detailed(
                    expected_job_name,
                    expected_task_count=expected_tasks,
                    use_squeue_fallback=True,
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
        queue_lookup = sacct_poll.find_active_job_by_id_detailed(job_id)
        if queue_lookup.inconclusive:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "expected_job_name": expected_job_name,
                "reason": "squeue lookup inconclusive: " + str(queue_lookup.error or "unknown error"),
            })
            continue
        if queue_lookup.active:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "expected_job_name": expected_job_name,
                "reason": "job is still active in squeue",
            })
            continue
        try:
            observations = sacct_poll.poll_job(job_id)
        except Exception as exc:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "expected_job_name": expected_job_name,
                "reason": "sacct lookup failed: " + type(exc).__name__ + ": " + str(exc),
            })
            continue
        terminal_state, reason, n_rows = _terminal_sacct_state_for_intent(
            job_id,
            observations,
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
        failure_reason = "reconcile_apply_terminal_job:" + terminal_state
        payload = dict(candidate)
        payload["reason"] = failure_reason
        payload["target_status"] = "FAILED"
        resolved.append(payload)
    return resolved, blocking


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
) -> Dict[str, Any]:
    """Perform only lossless, transaction-recorded reconcile mutations."""
    result: Dict[str, Any] = {
        "ferebus_retrain_archive": [],
        "archived_array_outputs": [],
        "refreshed_array_ledger": None,
        "archived_scripts": [],
        "archived_data_staging": [],
        "archived_model_staging": [],
        "archived_reference_data_staging": [],
        "archived_reentry_staging": [],
    }
    if retrain_ferebus:
        archived = archive_ferebus_iteration_staging_for_retrain(
            campaign,
            report.proposed_state,
        )
        if archived is not None:
            result["ferebus_retrain_archive"] = [str(archived)]
            transaction.record_paths("archive_ferebus_retrain", [str(archived)])

    if force_resubmit_array and isinstance(partial_array, dict):
        if archive_existing_array_outputs:
            archived_outputs = archive_existing_array_task_outputs(
                campaign,
                force_array_phase,
                int(force_array_iteration),
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
        paths = archive_scripts_for_reconcile(campaign)
        result["archived_scripts"] = paths
        transaction.record_paths("archive_scripts", paths)

    if ".DATA/STAGING is non-empty" in report.unsafe_reasons:
        if data_staging_archive_mode == "ferebus_reentry":
            paths = archive_data_staging_for_ferebus_reentry(
                campaign,
                report.proposed_state,
            )
        elif data_staging_archive_mode == "user":
            paths = archive_data_staging_for_operator_reconcile(campaign)
        else:
            paths = []
        result["archived_data_staging"] = paths
        transaction.record_paths("archive_data_staging", paths)

    if "dangling model staging directories exist" in report.unsafe_reasons:
        paths = clean_model_iteration_staging_for_reconcile(
            campaign,
            report.proposed_state,
        )
        result["archived_model_staging"] = paths
        transaction.record_paths("archive_model_staging", paths)

    if "dangling reference-data staging directories exist" in report.unsafe_reasons:
        paths = archive_reference_data_staging_for_reconcile(
            campaign,
            report.proposed_state,
        )
        result["archived_reference_data_staging"] = paths
        transaction.record_paths("archive_reference_data_staging", paths)

    paths = clean_reentry_staging(campaign, report.proposed_state.phase)
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
) -> None:
    """Publish scheduler-backed intent transitions after the core commit."""
    from .daemon.journal import append_event

    for item in resolved_intents:
        phase = str(item["phase"])
        iteration = int(item["iteration"])
        reason = str(item["reason"])
        target_status = str(item["target_status"])
        if target_status == "SUPERSEDED":
            _submission_intent.mark_superseded(
                campaign,
                phase,
                iteration,
                reason,
            )
        elif target_status == "FAILED":
            _submission_intent.mark_failed(
                campaign,
                phase,
                iteration,
                reason,
            )
        else:
            raise ValueError(
                "unsupported reconcile intent transition: " + target_status
            )
        append_event(
            campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
            "reconcile_resolved_terminal_intent",
            phase=phase,
            iteration=iteration,
            job_id=str(item.get("job_id") or ""),
            terminal_state=str(item.get("terminal_state") or ""),
            n_sacct_rows=int(item.get("n_sacct_rows") or 0),
            reason=reason,
            target_status=target_status,
        )

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


def _reconcile_apply_contract_error(campaign: Path, state: Any) -> Optional[str]:
    from .daemon.artifact_contracts import state_artifact_contract_status

    try:
        validate_phase_recovery_contract(campaign, state)
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
    status = state_artifact_contract_status(campaign, state)
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
    if ".DATA/SCRIPTS contains sbatch scripts" in reasons:
        cleanable.append(".DATA/SCRIPTS contains sbatch scripts")
    if "dangling model staging directories exist" in reasons:
        cleanable.append("dangling model staging directories exist")
    if "dangling reference-data staging directories exist" in reasons:
        cleanable.append("dangling reference-data staging directories exist")
    if ".DATA/STAGING is non-empty" in reasons:
        cleanable.append(
            ".DATA/STAGING is non-empty (requires --archive-staging when safe)"
        )
    return cleanable


def _reconcile_hard_blockers(
    report: Any,
    contract_status: Optional[Dict[str, Any]] = None,
) -> List[str]:
    cleanable_exact = {
        ".DATA/SCRIPTS contains sbatch scripts",
        "dangling model staging directories exist",
        "dangling reference-data staging directories exist",
    }
    cleanable_blocking_artifacts = set()
    reasons = set(str(reason) for reason in getattr(report, "unsafe_reasons", []))
    if "dangling model staging directories exist" in reasons:
        cleanable_blocking_artifacts.add("dangling model staging")
    if "dangling reference-data staging directories exist" in reasons:
        cleanable_blocking_artifacts.add("dangling reference-data staging")
    blockers: List[str] = []
    for reason in getattr(report, "unsafe_reasons", []):
        if reason in cleanable_exact:
            continue
        if reason == ".DATA/STAGING is non-empty":
            blockers.append(
                ".DATA/STAGING is non-empty unless --archive-staging is explicitly requested"
            )
            continue
        blockers.append(str(reason))
    for item in getattr(report, "blocking_artifacts", []):
        text = str(item)
        if text in cleanable_blocking_artifacts:
            continue
        blockers.append(text)
    if getattr(report, "active_submission_intents", []):
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
    runnable = bool(contract_status.get("contract_ok")) and not blockers
    why_not_runnable: List[str] = []
    if not bool(contract_status.get("contract_ok")):
        why_not_runnable.extend(
            "missing/invalid input: " + str(item)
            for item in contract_status.get("missing_or_invalid_inputs", [])
        )
    why_not_runnable.extend(blockers)
    if selected_state.phase in (CampaignPhase.HALTED, CampaignPhase.DONE):
        why_not_runnable.append(
            "selected phase is terminal: " + selected_state.phase.value
        )
    next_command = (
        "ichor-al-daemon start --campaign-dir " + str(campaign)
        if runnable and selected_state.phase is not CampaignPhase.DONE
        else "ichor-al-daemon reconcile --campaign-dir " + str(campaign) + " --apply"
        if cleanable and not blockers
        else "inspect blockers before restarting"
    )
    if selected_state.phase is CampaignPhase.DONE:
        next_command = "campaign is DONE; inspect outputs or initialise a new campaign"
    return {
        "schema_version": 1,
        "campaign_dir": str(campaign),
        "proposed_state_path": (
            str(proposed_state_path) if proposed_state_path is not None else None
        ),
        "selected_phase": selected_state.phase.value,
        "selected_iteration": int(selected_state.iteration),
        "runnable": bool(runnable and selected_state.phase is not CampaignPhase.HALTED),
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
        "active_submission_intents": list(getattr(report, "active_submission_intents", []) or []),
        "recommended_actions": list(getattr(report, "recommended_actions", []) or []),
        "next_command": next_command,
        "contract": contract_status,
        "runtime_status": runtime_status or {},
    }


def _reconcile_apply_command(campaign: Path, report: Any) -> str:
    command = "ichor-al-daemon reconcile --campaign-dir " + str(campaign)
    if ".DATA/STAGING is non-empty" in list(getattr(report, "unsafe_reasons", [])):
        command += " --archive-staging"
    return command + " --apply"


def _reconcile_human_reason(reason: Any) -> str:
    text = str(reason)
    mapping = {
        ".DATA/SCRIPTS contains sbatch scripts": "stale sbatch scripts",
        "dangling model staging directories exist": "dangling model staging",
        "dangling reference-data staging directories exist": "dangling reference-data staging",
        ".DATA/STAGING is non-empty": ".DATA/STAGING is non-empty",
        ".DATA/STAGING is non-empty (requires --archive-staging when safe)": (
            ".DATA/STAGING is non-empty; requires --archive-staging when safe"
        ),
        "active submission intent(s) present": "active submission intent(s)",
    }
    return mapping.get(text, text)


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
        return "APPLIED"
    state = report.proposed_state
    cleanable = _reconcile_cleanable_reasons(report)
    blockers = _reconcile_hard_blockers(report, contract_status)
    phase = state.phase
    if blockers:
        return "BLOCKED"
    if cleanable:
        return "CLEANUP REQUIRED"
    if phase is CampaignPhase.HALTED:
        return "BLOCKED"
    if phase is CampaignPhase.DONE:
        return "INSPECT ONLY"
    if not bool(contract_status.get("contract_ok")):
        return "BLOCKED"
    return "READY"


def _reconcile_apply_status(
    report: Any,
    contract_status: Dict[str, Any],
) -> str:
    result = _reconcile_result_label(report, contract_status)
    if result == "READY":
        return "safe"
    if result == "CLEANUP REQUIRED":
        return "cleanup required; no hard blockers"
    if result == "INSPECT ONLY":
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


def _reconcile_read_current_state_summary(campaign: Path, report: Any) -> Dict[str, Any]:
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
    pending_count = len(getattr(state, "pending_jobs", {}) or {})
    if active_intents:
        job_text = str(len(active_intents)) + " active or unresolved submission intent(s)"
    elif pending_count:
        job_text = str(pending_count) + " stale or unresolved state entr"
        job_text += "y" if pending_count == 1 else "ies"
    else:
        job_text = "none"
    return {
        "state": state.phase.value + " at iteration " + str(int(state.iteration)),
        "versions": (
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
        ),
        "recorded jobs": job_text,
        "shutdown requested": "yes" if bool(state.shutdown_requested) else "no",
        "campaign uid": str(state.campaign_uid),
    }


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


def _print_reconcile_header(campaign: Path, *, mode: str, result: str) -> None:
    print("ICHOR Reconcile")
    print("Campaign: " + str(campaign))
    print("Mode: " + mode)
    print("Result: " + result)
    print("")


def _print_reconcile_recovery_target(
    report: Any,
    contract_status: Dict[str, Any],
) -> None:
    state = report.proposed_state
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


def _print_reconcile_current_position(campaign: Path, report: Any) -> None:
    print("Current Position")
    summary = _reconcile_read_current_state_summary(campaign, report)
    _print_reconcile_key_values(list(summary.items()))
    print("")


def _print_reconcile_last_failure_compact(report: Any) -> None:
    event = getattr(report, "last_halt_event", None)
    if not isinstance(event, dict):
        return
    print("Last Failure")
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
        ]
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
    if verbose and getattr(report, "trusted_artifacts", []):
        _print_reconcile_list("trusted artefacts", [str(item) for item in report.trusted_artifacts])
    if verbose and getattr(report, "blocking_artifacts", []):
        _print_reconcile_list("artefact inventory", [str(item) for item in report.blocking_artifacts])
    print("")


def _print_reconcile_partial_array(campaign: Path, report: Any) -> None:
    partial = getattr(report, "partial_array_recovery", None)
    if not isinstance(partial, dict) or not partial:
        return
    print("Partial Array Recovery")
    phase = str(partial.get("phase") or "UNKNOWN")
    iteration = str(partial.get("iteration") if partial.get("iteration") is not None else "?")
    total = int(partial.get("logical_total") or 0)
    reuse = int(partial.get("n_reuse") or partial.get("n_complete") or 0)
    retry = int(partial.get("n_retry") or 0)
    mode = "full resubmission requested" if bool(partial.get("force_resubmit")) else "reuse completed outputs"
    _print_reconcile_key_values(
        [
            ("phase", phase + " iteration " + iteration),
            ("tasks", "total=" + str(total) + ", reusable=" + str(reuse) + ", retry=" + str(retry)),
            ("ledger", _reconcile_relative_path(campaign, partial.get("ledger"))),
            ("retry task file", _reconcile_relative_path(campaign, partial.get("retry_task_file")) or "none"),
            ("mode", mode),
        ]
    )
    print("")


def _print_reconcile_config_changes(config_review: Any) -> None:
    if config_review is None:
        return
    allowed = list(getattr(config_review, "allowed_changes", []) or [])
    blocked = list(getattr(config_review, "blocked_changes", []) or [])
    if not allowed and not blocked:
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
    print("Apply Plan")
    if proposed_state_path is not None:
        print("  proposed state: " + _reconcile_relative_path(campaign, proposed_state_path))
    if blockers:
        print("  apply: blocked")
        _print_reconcile_list("next action", ["inspect blockers before restarting"])
        print("")
        return
    if cleanable:
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
            _print_reconcile_list("expected recovery after cleanup", candidates)
        print("  command:")
        print("    " + _reconcile_apply_command(campaign, report))
        print("")
        return
    state = report.proposed_state
    if state.phase in (CampaignPhase.HALTED, CampaignPhase.DONE):
        print("  apply: not applicable")
        print("")
        return
    canonical = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    if proposed_state_path is not None:
        print("  manual promote:")
        print("    mv " + str(proposed_state_path) + " " + str(canonical))
    print("  safer command:")
    print("    ichor-al-daemon reconcile --campaign-dir " + str(campaign) + " --apply")
    print("")


def _print_reconcile_inspect_compact(campaign: Path) -> None:
    print("Inspect")
    _print_reconcile_key_values(
        [
            ("status", "ichor-al-daemon status --campaign-dir " + str(campaign)),
            ("journal", "ichor-al-daemon journal --campaign-dir " + str(campaign) + " --last-n 20"),
            ("staging", "find " + str(campaign / ".DATA" / "STAGING") + " -maxdepth 3 -type f | sort"),
        ]
    )
    print("")


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
    result = _reconcile_result_label(report, contract_status)
    _print_reconcile_header(campaign, mode=mode, result=result)
    _print_reconcile_recovery_target(report, contract_status)
    _print_reconcile_current_position(campaign, report)
    _print_reconcile_last_failure_compact(report)
    _print_reconcile_safety(campaign, report, contract_status)
    _print_reconcile_artefacts(campaign, report, contract_status, verbose=verbose)
    _print_reconcile_partial_array(campaign, report)
    _print_reconcile_config_changes(config_review)
    _print_reconcile_contract_compact(campaign, report, contract_status)
    if runtime_status and runtime_status.get("reconcile_apply_blockers"):
        print("Runtime")
        _print_reconcile_list(
            "apply blockers",
            [str(item) for item in runtime_status.get("reconcile_apply_blockers", [])],
        )
        print("")
    if verbose and getattr(report, "recommended_actions", []):
        _print_reconcile_list(
            "Recommended Actions",
            [str(item) for item in report.recommended_actions],
            indent="",
        )
        print("")
    _print_reconcile_apply_plan_compact(
        campaign,
        report,
        contract_status,
        proposed_state_path=proposed_state_path,
    )
    _print_reconcile_inspect_compact(campaign)


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
) -> None:
    _print_reconcile_header(campaign, mode="apply", result="APPLIED")
    _print_reconcile_recovery_target(report, contract_status)
    print("Applied Changes")
    cleanup_items: List[str] = []
    cleanup_items.extend("removed stale artefact: " + _reconcile_relative_path(campaign, item) for item in removed)
    cleanup_items.extend("removed model staging: " + _reconcile_relative_path(campaign, item) for item in removed_model_staging)
    cleanup_items.extend("archived stale scripts: " + _reconcile_relative_path(campaign, item) for item in archived_scripts)
    cleanup_items.extend("archived staging: " + _reconcile_relative_path(campaign, item) for item in archived)
    cleanup_items.extend("archived reference-data staging: " + _reconcile_relative_path(campaign, item) for item in archived_reference_data_staging)
    cleanup_items.extend("restored bootstrap staging: " + _reconcile_relative_path(campaign, item) for item in restored_bootstrap_handoff)
    _print_reconcile_key_values(
        [
            ("state written", _reconcile_relative_path(campaign, campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)),
            ("previous state backup", _reconcile_relative_path(campaign, backup_path) if backup_path is not None else "none"),
            ("applied proposal archive", _reconcile_relative_path(campaign, applied_proposal_path) if applied_proposal_path is not None else "none"),
            ("final contract", "ok" if contract_status.get("contract_ok") else "invalid"),
        ]
    )
    _print_reconcile_list("cleanup", cleanup_items)
    final_phase = report.proposed_state.phase
    if bool(contract_status.get("contract_ok")) and final_phase not in {CampaignPhase.HALTED, CampaignPhase.DONE}:
        _print_reconcile_list(
            "next",
            ["ichor-al-daemon start --campaign-dir " + str(campaign)],
        )
    else:
        _print_reconcile_list(
            "next",
            ["Campaign remains " + final_phase.value + "; do not start the daemon yet."],
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
    print("Mode: apply", file=sys.stderr)
    print("Result: BLOCKED", file=sys.stderr)
    print("", file=sys.stderr)
    print(title, file=sys.stderr)
    if reasons:
        print("  blockers:", file=sys.stderr)
        for reason in reasons:
            print("    - " + str(reason), file=sys.stderr)
    else:
        print("  blockers:", file=sys.stderr)
        print("    - unknown", file=sys.stderr)
    if next_actions:
        print("  next:", file=sys.stderr)
        for action in next_actions:
            print("    - " + str(action), file=sys.stderr)
    print("", file=sys.stderr)


def _scratch_intent_index(campaign: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    root = _submission_intent.intent_dir(campaign)
    if not root.is_dir():
        return index
    for path in sorted(root.rglob("*.json")):
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
) -> Tuple[str, str]:
    """Return active, inactive, or inconclusive for one recorded Slurm job."""
    from .submit import sacct_poll

    try:
        queue = sacct_poll.find_active_job_by_id_detailed(str(job_id))
    except Exception as exc:
        return "inconclusive", type(exc).__name__ + ": " + str(exc)
    if bool(getattr(queue, "inconclusive", False)):
        return "inconclusive", str(getattr(queue, "error", None) or "squeue lookup failed")
    if bool(getattr(queue, "active", False)):
        return "active", "squeue reports active rows"
    try:
        observations = sacct_poll.poll_job(str(job_id))
    except Exception as exc:
        return "inconclusive", "sacct lookup failed: " + type(exc).__name__ + ": " + str(exc)
    if not observations:
        return "inconclusive", "sacct returned no rows"
    matching = _matching_sacct_observations(str(job_id), observations)
    if not matching:
        return "inconclusive", "sacct returned no matching task rows"
    summary = sacct_poll.aggregate_states(
        str(job_id),
        matching,
        expected_task_count=expected_task_count,
    )
    if summary.conflicting_task_indices:
        return "inconclusive", "sacct returned conflicting task rows"
    if summary.out_of_range_task_indices:
        return "inconclusive", "sacct returned out-of-range task rows"
    if summary.n_missing:
        return (
            "inconclusive",
            "sacct is missing "
            + str(summary.n_missing)
            + " of "
            + str(summary.n_expected)
            + " expected task rows",
        )
    states = [observation.status for observation in summary.observations]
    if any(state in sacct_poll.NON_TERMINAL_STATES for state in states):
        return "active", "sacct reports non-terminal rows"
    if any(state == sacct_poll.JobStatus.UNKNOWN for state in states):
        return "inconclusive", "sacct reports UNKNOWN rows"
    if not all(state in sacct_poll.TERMINAL_STATES for state in states):
        return "inconclusive", "sacct rows are not conclusively terminal"
    return "inactive", "squeue absent and sacct rows are terminal"


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
    scheduler_cache: Dict[Tuple[str, Optional[int]], Tuple[str, str]] = {}
    output: List[Dict[str, Any]] = [dict(item) for item in invalid]
    for group in grouped.values():
        attempt_id = str(group["attempt_id"])
        identity = str(group["submission_identity"])
        intent = intent_index.get(attempt_id) or intent_index.get(identity) or {}
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
            cache_key = (job_id, expected_tasks)
            if cache_key not in scheduler_cache:
                scheduler_cache[cache_key] = _scratch_scheduler_state(
                    job_id,
                    expected_task_count=expected_tasks,
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
                    "schema_version": 1,
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
                "restore the config proposal first, then run reconcile --apply",
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
        _print_reconcile_header(campaign, mode="restore-config", result="CONFIG PROPOSAL")
        print("Config Proposal")
        _print_reconcile_key_values(
            [
                ("proposal", _reconcile_relative_path(campaign, target_config)),
                ("source", ".DATA/ACTIVE_LEARNING/config_lock.json"),
                ("status", "dense full config snapshot written"),
            ]
        )
        print("")
        print("Apply Plan")
        print("  manual promote:")
        print("    mv " + str(target_config) + " " + str(campaign / "campaign.yaml"))
        print("  then run:")
        print("    ichor-al-daemon reconcile --campaign-dir " + str(campaign) + " --apply")
        print("")
        return 0
    if bool(getattr(args, "apply", False)) and runtime_status.get("reconcile_apply_blockers"):
        _print_reconcile_apply_blocked(
            campaign,
            title="Runtime Safety",
            reasons=list(runtime_status.get("reconcile_apply_blockers") or []),
            next_actions=[
                "stop the daemon or wait for the lock/lease to clear",
                "rerun reconcile --apply",
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
    report = propose_recovery(
        campaign,
        allow_fresh_init_on_nonempty=bool(getattr(args, "allow_fresh_init", False)),
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
    target = write_proposed_state(campaign, report)
    if bool(getattr(args, "json", False)):
        contract_status = recovery_contract_status(campaign, report.proposed_state)
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
    contract_status = recovery_contract_status(campaign, report.proposed_state)
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
    resolved_intents: List[Dict[str, Any]] = []
    if report.active_submission_intents:
        resolved_intents, blocking_intents = _resolve_terminal_submission_intents_for_apply(
            campaign,
            report.active_submission_intents,
            pre_submit_stale_seconds=(
                int(config.runtime.lease_stale_seconds)
                if config is not None
                else 900
            ),
        )
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
                    "ichor-al-daemon stop --campaign-dir " + str(campaign) + " --cancel-jobs",
                    "ichor-al-daemon reconcile --campaign-dir " + str(campaign) + " --apply",
                ],
            )
            return 9
        if resolved_intents:
            print("Resolved terminal submission intents:")
            for item in resolved_intents:
                print(
                    "  - "
                    + str(item.get("phase"))
                    + "@"
                    + str(item.get("iteration"))
                    + " job_id="
                    + str(item.get("job_id"))
                    + " terminal_state="
                    + str(item.get("terminal_state"))
                )
            report.active_submission_intents = []
            report.unsafe_reasons = [
                reason
                for reason in report.unsafe_reasons
                if not str(reason).startswith("active submission intent(s) present:")
            ]
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
    }
    data_staging_archive_mode = None
    archive_staging_refusal_reasons: List[str] = []
    if ".DATA/STAGING is non-empty" in report.unsafe_reasons:
        ok_to_archive_staging, staging_reason = ferebus_reentry_can_archive_data_staging(
            campaign,
            report.proposed_state,
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

    cleanable_now = [_reconcile_human_reason(item) for item in _reconcile_cleanable_reasons(report)]
    if cleanable_now:
        print("Apply Plan")
        _print_reconcile_list(
            "cleanup before recovery",
            cleanable_now
            + [
                "recompute recovery",
                "validate the final state/artefact contract",
            ],
        )
        print("")

    original_report = report
    transaction: Optional[ReconcileTransaction] = None
    planned_operations = [
        "validate_recovery_contract",
        "repair_current_pointers",
        "write_recovered_state",
        "update_config_lock",
        "publish_intent_transitions",
    ]
    if cleanable_now or retrain_ferebus or force_resubmit_array:
        planned_operations.insert(0, "archive_reconcile_evidence")
    try:
        transaction = begin_reconcile_transaction(
            campaign,
            proposed_phase=report.proposed_state.phase.value,
            proposed_iteration=int(report.proposed_state.iteration),
            planned_operations=planned_operations,
            intent_transitions=resolved_intents,
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
        )
    except Exception as exc:
        _fail_reconcile_transaction(
            transaction,
            "reconcile evidence archival failed: " + type(exc).__name__ + ": " + str(exc),
        )
        print(
            "could not archive reconcile evidence losslessly: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=sys.stderr,
        )
        return 9

    ferebus_retrain_archive = list(mutation_result["ferebus_retrain_archive"])
    archived_array_outputs = list(mutation_result["archived_array_outputs"])
    refreshed = mutation_result["refreshed_array_ledger"]
    archived_scripts = list(mutation_result["archived_scripts"])
    archived = list(mutation_result["archived_data_staging"])
    removed_model_staging = list(mutation_result["archived_model_staging"])
    archived_reference_data_staging = list(
        mutation_result["archived_reference_data_staging"]
    )
    removed = list(mutation_result["archived_reentry_staging"])
    if ferebus_retrain_archive:
        print("Archived FEREBUS output for explicit retraining:")
        for path in ferebus_retrain_archive:
            print("  - " + str(path))
    if refreshed is not None:
        print(
            "Marked current array for full resubmission: "
            + str(force_array_phase.value)
            + "@"
            + str(int(force_array_iteration))
            + " retry_tasks="
            + str(int(refreshed.get("n_retry") or 0))
        )
        if archived_array_outputs:
            print("Archived existing array task outputs:")
            for path in archived_array_outputs[:8]:
                print("  - " + str(path))
            if len(archived_array_outputs) > 8:
                print("  - ... " + str(len(archived_array_outputs) - 8) + " more")
    cleanup_paths_already_done = (
        list(archived_scripts)
        + list(archived_array_outputs)
        + list(archived)
        + list(removed_model_staging)
        + list(archived_reference_data_staging)
        + list(ferebus_retrain_archive)
        + list(removed)
    )

    if original_report.proposed_state.phase is CampaignPhase.HALTED:
        report = propose_recovery(
            campaign,
            allow_fresh_init_on_nonempty=bool(getattr(args, "allow_fresh_init", False)),
            _active_reconcile_transaction_id=(
                str(transaction.payload["transaction_id"])
                if transaction is not None
                else None
            ),
        )
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
        recomputed_status = recovery_contract_status(campaign, report.proposed_state)
        print("Recovery After Cleanup")
        _print_reconcile_key_values(
            [
                (
                    "selected phase",
                    report.proposed_state.phase.value
                    + " iteration "
                    + str(int(report.proposed_state.iteration)),
                ),
                ("reason", str(report.decision or "-")),
                ("contract", "ok" if recomputed_status.get("contract_ok") else "invalid"),
            ]
        )
        _print_reconcile_list(
            "trusted handoffs",
            _reconcile_valid_candidates(campaign, recomputed_status, report),
        )
        print("")

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
        print("  - " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 9

    contract_error = _reconcile_apply_contract_error(campaign, report.proposed_state)
    if contract_error is not None:
        _fail_reconcile_transaction(
            transaction,
            "final state/artefact contract failed: " + str(contract_error),
        )
        _print_reconcile_apply_blocked(
            campaign,
            title="Final State/Artefact Contract",
            reasons=[contract_error],
            next_actions=["inspect recovery contract before restarting"],
        )
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 9

    pointer_snapshots: List[Dict[str, Any]] = []
    try:
        if transaction is None:
            raise RuntimeError("reconcile transaction was not created")
        transaction.set_status("COMMITTING")
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
        print("  - " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 9

    try:
        backup_path = _copy_existing_timestamped(
            target_canonical,
            ".before-reconcile-",
        )
    except Exception as exc:
        pointer_errors = _restore_reconcile_pointer_snapshots(campaign, pointer_snapshots)
        reason = "state backup failed: " + type(exc).__name__ + ": " + str(exc)
        if pointer_errors:
            reason += "; pointer rollback failed: " + "; ".join(pointer_errors)
        _fail_reconcile_transaction(transaction, reason)
        print(reason, file=sys.stderr)
        return 9
    try:
        write_state(target_canonical, report.proposed_state)
    except Exception as exc:
        pointer_errors = _restore_reconcile_pointer_snapshots(campaign, pointer_snapshots)
        reason = "state write failed: " + type(exc).__name__ + ": " + str(exc)
        if pointer_errors:
            reason += "; pointer rollback failed: " + "; ".join(pointer_errors)
        _fail_reconcile_transaction(transaction, reason)
        print(
            "failed to write recovered state.json: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=sys.stderr,
        )
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 9
    try:
        apply_config_lock_update(
            campaign,
            config,
            campaign_uid=str(report.proposed_state.campaign_uid),
        )
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
            + restore_message
            + ": "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=sys.stderr,
        )
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 8
    try:
        _publish_reconcile_intent_transitions(
            campaign,
            resolved_intents,
            report.proposed_state,
        )
    except Exception as exc:
        reason = (
            "recovered state committed, but intent publication remains incomplete: "
            + type(exc).__name__
            + ": "
            + str(exc)
        )
        _fail_reconcile_transaction(transaction, reason)
        print(reason, file=sys.stderr)
        print(
            "The committed state and current pointers are coherent. Rerun "
            "reconcile --apply to finish the recorded intent transaction.",
            file=sys.stderr,
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
            archived_staging_path=(archived[0] if archived else None),
            archived_reference_data_staging_paths=archived_reference_data_staging,
            n_archived_scripts_paths=len(archived_scripts),
            archived_scripts_path=(archived_scripts[0] if archived_scripts else None),
            recomputed_after_transient_cleanup=(
                original_report.proposed_state.phase is CampaignPhase.HALTED
            ),
            recovery_source_phase=(
                original_report.last_phase_in_journal
                or original_report.proposed_state.phase.value
            ),
            recovery_selected_phase=report.proposed_state.phase.value,
            recovery_reason=str(report.decision or ""),
        )
    except Exception:
        pass
    if transaction is not None:
        try:
            transaction.set_status("COMMITTED")
        except Exception as exc:
            print(
                "warning: reconcile committed, but its transaction receipt "
                "could not be sealed: "
                + type(exc).__name__
                + ": "
                + str(exc),
                file=sys.stderr,
            )
    final_contract_status = recovery_contract_status(campaign, report.proposed_state)
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
        restored_bootstrap_handoff=restored_bootstrap_handoff,
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
    try:
        iterator = read_events(
            journal_path,
            since=args.since,
            event_type=args.event_type or None,
        )
        last_n = getattr(args, "last_n", None)
        if last_n is None:
            events = list(iterator)
        elif int(last_n) > 0:
            events = list(deque(iterator, maxlen=int(last_n)))
        else:
            events = []
    except (JournalCorruptionError, ValueError, OSError) as exc:
        print("journal is corrupt or unreadable: " + str(exc), file=sys.stderr)
        return 5
    if bool(getattr(args, "json", False)) or bool(getattr(args, "raw", False)):
        for event in events:
            print(json.dumps(event, sort_keys=True, allow_nan=False))
    else:
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


def _print_bootstrap_plan(plan: Any) -> None:
    labels = {
        "train": "Training",
        "int_val": "Internal validation",
        "ext_val": "External validation",
    }
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

        result = evaluate_pool_feasibility(campaign, config)
        return result.to_dict()
    except Exception as exc:
        return {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }


def _print_pool_summary(summary: Dict[str, Any]) -> None:
    if str(summary.get("status")) == "ok":
        print(
            "  trajectory pool: ok, frames="
            + str(summary.get("frames"))
            + ", atoms="
            + str(summary.get("atoms"))
            + ", sha="
            + str(summary.get("sha256", ""))[:12]
        )
    elif str(summary.get("status")) == "missing":
        print("  trajectory pool: missing")
    else:
        print("  trajectory pool: invalid - " + str(summary.get("error", "unknown")))


def _print_pool_feasibility(summary: Dict[str, Any], *, file=None) -> None:
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
    if summary.get("expression"):
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
            "  ichor-al-daemon reconcile --campaign-dir " + str(campaign) + " --apply",
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

    _print_bootstrap_plan(plan)
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
            _print_pool_feasibility(feasibility_summary, file=sys.stderr)
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
    print("  campaign.yaml: ok, schema v" + str(config.schema_version))
    _print_pool_summary(pool_summary)
    print(
        "  bootstrap inputs: confirmed, identity="
        + str(bootstrap_manifest.get("plan_identity_sha256", ""))[:12]
        + "..."
    )
    if feasibility_summary:
        _print_pool_feasibility(feasibility_summary)
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
        print("  ichor-al-daemon preflight --campaign-dir " + str(campaign))
        print("  ichor-al-daemon start --campaign-dir " + str(campaign))
    else:
        print("Next:")
        print(
            "  ichor-al-daemon init --campaign-dir "
            + str(campaign)
            + " --source /path/to/pool.xyz"
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
    ready = bool(avail.all_present and config_ok and feasibility_ok and state_ok)
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
        }
    )
    return payload


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
            elif name == "bc":
                details.append("make bc available; pyferebus scripts use it")
            elif name == "gaussian":
                details.append("configure Gaussian module/executable or make g16 available")
            elif name == "aimall":
                details.append("configure an executable AIMAll aimqb.ish path")
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
        details.append(
            "fix campaign state readiness: "
            + str(state.get("error") or state.get("phase") or "state is not runnable")
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

    lines: List[str] = []
    lines.extend(
        _section(
            "Preflight",
            [
                ("result", "ready" if payload.get("ready") else "blocked"),
                ("active profile", avail.get("active_profile") or "<unresolved>"),
                ("campaign", payload.get("campaign_dir")),
            ],
        )
    )
    if verbose and avail.get("profile_error"):
        lines.append("  profile error: " + str(avail.get("profile_error")))

    lines.append("")
    lines.append("Scheduler")
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
            warn=not bool(avail.get("batch_python")),
        )
    )
    runtime_modules = avail.get("batch_runtime_modules")
    if isinstance(runtime_modules, (list, tuple)):
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
    lines.append(
        _preflight_check_line(
            "Gaussian submitted environment",
            avail.get("gaussian_verified"),
            avail.get("gaussian_binary")
            or avail.get("gaussian_probe_error")
            or "not configured",
        )
    )
    lines.append(_preflight_check_line("AIMAll", avail.get("aimall"), avail.get("aimall_path") or "not found"))

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
        if config.get("schema_version") is not None:
            lines.append("  schema version: " + str(config.get("schema_version")))
        if config.get("system_name"):
            lines.append("  system: " + str(config.get("system_name")))
        if config.get("prior_mean_level_of_theory"):
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
    lines.append(
        _preflight_check_line(
            "runnable state",
            state.get("ok"),
            state.get("error")
            or (
                str(state.get("phase"))
                + " iteration "
                + str(state.get("iteration"))
            ),
        )
    )
    if state.get("contract_error"):
        lines.append("  contract error: " + str(state.get("contract_error")))

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
        if pool.get("bootstrap_custom_count") is not None:
            lines.append(
                "  bootstrap custom geometries: "
                + str(pool.get("bootstrap_custom_count"))
            )
        if pool.get("bootstrap_model_training_count") is not None:
            lines.append(
                "  bootstrap model training rows: "
                + str(pool.get("bootstrap_model_training_count"))
            )
        if pool.get("bootstrap_pool_frame_count") is not None:
            lines.append(
                "  bootstrap pool frames: "
                + str(pool.get("bootstrap_pool_frame_count"))
            )
        if pool.get("expression"):
            lines.append("  requirement: " + str(pool.get("expression")))
        if pool.get("reserve_after_bootstrap") is not None:
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
        lines.append("  start live campaign:")
        lines.append(
            "    ichor-al-daemon start --campaign-dir "
            + str(payload.get("campaign_dir"))
            + " --mode live"
        )
    else:
        lines.append("  fix failed checks before live start")
        for detail in _preflight_failure_details(payload)[:12]:
            lines.append("  - " + detail)
    if verbose and not payload.get("ready") and not payload.get("all_backends_present"):
        lines.append("")
        lines.append("Backend Details")
        for line in str(payload.get("missing_backend_message") or "").splitlines():
            lines.append("  " + line)
    return "\n".join(lines) + "\n"


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
            from .acquisition.trajectory_pool import TrajectoryPool
            from .ferebus_prior import resolve_ferebus_prior_contract

            pool_manifest = (
                campaign / ".DATA" / "TRAJECTORY" / "pool.manifest.json"
            )
            if pool_manifest.is_file():
                pool = TrajectoryPool.load(campaign)
                resolve_ferebus_prior_contract(
                    loaded_config,
                    atom_labels=pool.manifest.atom_types,
                )
        except Exception as exc:
            feasibility_summary = {
                "ok": False,
                "error": type(exc).__name__ + ": " + str(exc),
            }

    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    if not state_path.is_file():
        state_summary = {
            "ok": False,
            "error": "state.json is missing; initialise the campaign before live start",
        }
    else:
        try:
            state = read_state(state_path)
            if state.phase is CampaignPhase.HALTED:
                raise ValueError("campaign is HALTED and requires reconcile")
            if state.phase is CampaignPhase.DONE:
                raise ValueError("campaign is DONE")
            if state.shutdown_requested:
                raise ValueError("campaign has a user stop request")
            from .daemon.artifact_contracts import state_artifact_contract_status

            contract = state_artifact_contract_status(campaign, state)
            if not bool(contract.get("ok", False)):
                raise ValueError(str(contract.get("error") or "artefact contract invalid"))
            if loaded_config is not None:
                review = review_config_changes(
                    campaign,
                    loaded_config,
                    state,
                    initialise_missing=False,
                )
                if review.changed:
                    raise ValueError("campaign config differs from its lock")
                if bool(loaded_config.retention.checkpoint_required):
                    from .daemon.checkpoints import checkpoint_store

                    store = checkpoint_store(
                        str(loaded_config.retention.checkpoint_destination),
                        str(state.campaign_uid),
                    )
                    if (store / "current.json").exists():
                        from .daemon.checkpoints import checkpoint_status

                        checkpoint_status(
                            campaign,
                            str(loaded_config.retention.checkpoint_destination),
                        )
            state_summary = {
                "ok": True,
                "phase": state.phase.value,
                "iteration": int(state.iteration),
            }
        except Exception as exc:
            state_summary = {
                "ok": False,
                "error": type(exc).__name__ + ": " + str(exc),
            }
    return _preflight_payload(
        campaign,
        availability,
        config_summary,
        feasibility_summary,
        state_summary,
    )


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
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(_format_preflight(payload, verbose=bool(getattr(args, "verbose", False))), end="")
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
        print(format_resource_plan(payload), end="")
    if not bool(getattr(args, "all", False)):
        statuses = {str(plan.get("status")) for plan in payload["plans"]}
        if "evidence_invalid" in statuses:
            return 15
        if "evidence_not_yet_produced" in statuses:
            return 14
    return 0


def cmd_environment_status(args: argparse.Namespace) -> int:
    """Compare the current process/native stack with the bound generation."""
    from .execution_identity import (
        environment_status,
        read_execution_identity,
    )

    try:
        campaign = resolve_campaign_dir(args.campaign_dir)
        config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        state = read_state(operational_path(campaign, DEFAULT_STATE_FILENAME))
        identity = read_execution_identity(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        payload = environment_status(
            campaign,
            campaign_uid=str(state.campaign_uid),
            config=config,
        )
        payload["mode"] = str(identity["mode"])
        payload["phase"] = state.phase.value
        payload["iteration"] = int(state.iteration)
    except (OSError, TypeError, ValueError) as exc:
        print("environment-status failed: " + str(exc), file=sys.stderr)
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(
            "Environment status: "
            + ("MATCH" if payload["matches"] else "DRIFTED")
        )
        print("Generation: " + str(payload["generation"]))
        print("Mode: " + str(payload["mode"]))
        if payload["changed_fields"]:
            print("Changed fields: " + ", ".join(payload["changed_fields"]))
    return 0 if bool(payload["matches"]) else 18


def cmd_rebind_environment(args: argparse.Namespace) -> int:
    """Create a new verified environment generation at an idle boundary."""
    from .execution_identity import (
        environment_status,
        read_execution_identity,
        rebind_environment,
    )

    try:
        campaign = resolve_campaign_dir(args.campaign_dir)
        config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        state = read_state(operational_path(campaign, DEFAULT_STATE_FILENAME))
        identity = read_execution_identity(
            campaign,
            expected_campaign_uid=str(state.campaign_uid),
        )
        if not bool(getattr(args, "apply", False)):
            payload = environment_status(
                campaign,
                campaign_uid=str(state.campaign_uid),
                config=config,
            )
            payload["applied"] = False
            payload["mode"] = str(identity["mode"])
        else:
            with _exclusive_operator_lock(campaign):
                runtime = _reconcile_runtime_status(
                    campaign,
                    operator_lock_owned=True,
                )
                blockers = list(runtime.get("reconcile_apply_blockers") or [])
                if blockers:
                    raise ValueError(
                        "environment rebind is blocked: " + "; ".join(blockers)
                    )
                live_preflight_ok = False
                if str(identity["mode"]) == "live":
                    availability = check_backends()
                    live_preflight_ok = bool(availability.all_present)
                    if not live_preflight_ok:
                        raise ValueError(missing_backend_message(availability))
                payload = rebind_environment(
                    campaign,
                    config=config,
                    live_preflight_ok=live_preflight_ok,
                    scheduler_ownership_clear=True,
                )
                payload["applied"] = True
                payload["mode"] = str(identity["mode"])
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print("rebind-environment failed: " + str(exc), file=sys.stderr)
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    elif not bool(getattr(args, "apply", False)):
        print(
            "Environment rebind preview: "
            + ("not required" if payload["matches"] else "required")
        )
        if payload["changed_fields"]:
            print("Changed fields: " + ", ".join(payload["changed_fields"]))
        print("Re-run with --apply after reviewing scheduler ownership.")
    elif bool(payload.get("changed", False)):
        print("Environment rebound to generation " + str(payload["generation"]))
        print("Generation SHA-256: " + str(payload["generation_digest_sha256"]))
    else:
        print(str(payload.get("message") or "Environment already matches."))
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

    try:
        campaign = resolve_campaign_dir(args.campaign_dir)
        config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        destination = _checkpoint_destination(
            config,
            getattr(args, "destination", None),
        )
        with _exclusive_operator_lock(campaign):
            payload = create_checkpoint(
                campaign,
                destination,
                verify_after_write=True,
                allow_active_lease=False,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print("checkpoint failed: " + str(exc), file=sys.stderr)
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("Checkpoint verified: " + str(payload["checkpoint"]))
        print("Manifest SHA-256: " + str(payload["manifest_sha256"]))
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
        print("checkpoint-status failed: " + str(exc), file=sys.stderr)
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("Checkpoint status: " + str(payload["status"]))
        print("Store: " + str(payload["store"]))
        if isinstance(payload.get("current"), dict):
            print("Iteration: " + str(payload["current"].get("iteration")))
    return 0 if payload["status"] == "verified" else 1


def cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    """Deeply verify one published checkpoint."""
    from .daemon.checkpoints import verify_checkpoint

    try:
        payload = verify_checkpoint(args.checkpoint)
    except (OSError, TypeError, ValueError) as exc:
        print("verify-checkpoint failed: " + str(exc), file=sys.stderr)
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("Checkpoint verified: " + str(payload["checkpoint"]))
        print("Manifest SHA-256: " + str(payload["manifest_sha256"]))
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
        print("restore-checkpoint failed: " + str(exc), file=sys.stderr)
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    elif payload["applied"]:
        print("Checkpoint restored and verified: " + str(payload["target"]))
    else:
        print("Checkpoint restore verified; no files were written.")
        print("Target: " + str(payload["target"]))
        print("Re-run with --apply to restore.")
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


def build_parser() -> argparse.ArgumentParser:
    examples = """\
Campaign directory:
  If -c/--campaign-dir is omitted, the current directory is used when it
  contains campaign.yaml.

Examples:
  cd ~/campaigns/water_001
  ichor-al-daemon init
  ichor-al-daemon status
  ichor-al-daemon start --mode live
  ichor-al-daemon start --mode live --background
  ichor-al-daemon journal -e phase_submitted

  ichor-al-daemon start -c ~/campaigns/water_001 --mode live
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
                "Launch the daemon as a detached background child. The foreground "
                "command validates the campaign, writes a PID file, and appends logs "
                "under .DATA/ACTIVE_LEARNING by default."
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
            "Start a campaign daemon. The first start requires an explicit "
            "--mode; the mode is immutable thereafter. "
            "From inside a campaign directory, --campaign-dir can be omitted."
        ),
        epilog=(
            "Examples:\n"
            "  ichor-al-daemon start --mode dry_run --max-ticks 10\n"
            "  ichor-al-daemon start --mode live\n"
            "  ichor-al-daemon start --mode live --foreground"
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
            "Execution mode. Required on the first start and permanently "
            "bound to the campaign execution identity."
        ),
    )
    p_start.add_argument(
        "-p",
        "--poll-interval",
        type=_positive_cli_int,
        default=None,
        help="Override poll_interval_seconds from the config.",
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
            "immediate at the next tick and does not cancel Slurm jobs."
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
            "Also scancel active Slurm jobs recorded by this campaign. "
            "Plain stop only requests daemon shutdown and leaves jobs alone."
        ),
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
        help="Equivalent to start (kept for symmetry; works only when state.json exists).",
        description=(
            "Resume a campaign daemon. This accepts the same execution-mode "
            "and background flags as start."
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
            "Archive and cancel an unfinished boundary stop request before "
            "resuming. A completed user stop is cleared by ordinary resume."
        ),
    )
    add_background_options(p_resume)
    p_resume.set_defaults(func=cmd_resume)

    p_recon = sub.add_parser(
        "reconcile",
        help="Inspect on-disk artefacts and propose a recovered state.",
        description=(
            "Inspect campaign artefacts and write state.json.proposed. With "
            "--apply, safely promote the proposal when the recovery is clean."
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
            "  ichor-al-daemon journal -j | tail -n 40"
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
        type=int,
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
            help="Overwrite an existing pool (DANGEROUS: invalidates every committed "
                 "iteration's frame-id provenance).",
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
            "  ichor-al-daemon start --campaign-dir ~/campaigns/water_001"
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
        help="Check configured live Slurm backends.",
        description=(
            "Check the configured Slurm/Gaussian/AIMAll/FEREBUS/ARIADNE "
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
            "Slurm job that verifies the compute-node module, Python-import, "
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
    p_resource.set_defaults(func=cmd_resource_plan)

    p_environment_status = sub.add_parser(
        "environment-status",
        help="Compare the current software/native stack with the active generation.",
    )
    add_campaign(p_environment_status)
    p_environment_status.add_argument("--json", action="store_true")
    p_environment_status.set_defaults(func=cmd_environment_status)

    p_rebind_environment = sub.add_parser(
        "rebind-environment",
        help="Bind an idle campaign to a newly verified environment generation.",
    )
    add_campaign(p_rebind_environment)
    p_rebind_environment.add_argument(
        "--apply",
        action="store_true",
        help="Create and publish the new generation after all safety checks pass.",
    )
    p_rebind_environment.add_argument("--json", action="store_true")
    p_rebind_environment.set_defaults(func=cmd_rebind_environment)

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
        return int(args.func(args))
    except CampaignDirResolutionError as exc:
        parser.exit(2, parser.prog + ": error: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
