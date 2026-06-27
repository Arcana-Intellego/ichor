"""CLI for the ICHOR active-learning daemon.

Console entry point 'ichor-al-daemon' registered in
'ichor_cli/setup.cfg'. Subcommands available:

    start      Start the daemon in the foreground, or use --background to detach.
    stop       Set shutdown_requested=true in state.json; running daemon picks
               it up on next tick.
    status     Print the current state snapshot.
    resume     Equivalent to start when state.json already exists.
    reconcile  Inspect on-disk artefacts and propose a recovered state.
    journal    Tail or filter the campaign journal.
    init       Initialise campaign.yaml and import the trajectory pool.

Commands that operate on a campaign accept '--campaign-dir DIR'. When it is
omitted, the CLI uses the current working directory if it contains
'campaign.yaml'.

The CLI builds the Daemon with default executor/sacct_poller for production;
'--mock-ariadne' swaps the executor for a deterministic Mock so dry-runs
exercise the state machine without invoking real backends.
"""
from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import CampaignConfig
from .daemon.daemon import (
    DAEMON_HEARTBEAT_FILENAME,
    DAEMON_LEASE_DIRNAME,
    DAEMON_LOCK_FILENAME,
    DEFAULT_DATA_SUBDIR,
    Daemon,
)
from .daemon.journal import KNOWN_EVENT_TYPES, iter_events, read_events
from .daemon.config_lock import (
    apply_config_lock_update,
    archive_scripts_for_reconcile,
    archive_data_staging_for_ferebus_reentry,
    archive_data_staging_for_operator_reconcile,
    archive_training_staging_for_reconcile,
    assert_config_unchanged_for_start,
    clean_model_iteration_staging_for_reconcile,
    clean_reentry_staging,
    config_lock_path,
    ferebus_reentry_can_archive_data_staging,
    format_config_review,
    review_config_changes,
    restore_config_from_lock_proposal,
    training_staging_can_archive_for_reconcile,
)
from .daemon.dry_run_executor import DryRunPhaseExecutor
from .daemon.dry_run_sacct import DryRunSacctPoller
from .daemon.job_names import live_job_name
from .daemon.live_executor import (
    LiveBackendNotAvailableError,
    LiveBackendsPhaseExecutor,
    make_live_job_finder,
    make_live_job_liveness_checker,
)
from .daemon.phase_executor import MockPhaseExecutor
from .daemon.preflight import check_backends, missing_backend_message
from .daemon.reconcile import (
    data_staging_inventory,
    propose_recovery,
    stateful_campaign_artifacts,
    write_proposed_state,
)
from .daemon import submission_intent as _submission_intent
from .daemon.state import (
    CampaignPhase,
    DEFAULT_STATE_FILENAME,
    StateSchemaError,
    read_state,
    write_state,
)
from .daemon.status_recommendations import (
    build_status_recommendations,
    recommendation_dicts,
)


__all__ = [
    "build_parser",
    "expand_boolean_short_flag_clusters",
    "main",
    "resolve_campaign_dir",
    "ShortFlagClusterError",
]


BACKGROUND_CHILD_ENV = "ICHOR_DAEMON_BACKGROUND_CHILD"
BACKGROUND_LOG_FILENAME = "daemon.out"
BACKGROUND_PID_FILENAME = "daemon.pid"
BACKGROUND_PID_SCHEMA_VERSION = 1


class CampaignDirResolutionError(ValueError):
    """Raised when a command cannot infer a valid campaign directory."""


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
    "start": frozenset({"l", "b", "d", "m"}),
    "resume": frozenset({"l", "b", "d", "m"}),
    "stop": frozenset({"x"}),
    "status": frozenset({"j", "v"}),
    "reconcile": frozenset({"a", "F"}),
    "journal": frozenset({"j", "r", "v"}),
    "init": frozenset({"f", "O"}),
    "import-pool": frozenset({"f", "O"}),
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
                + "; use separate flags such as '-l -b', and pass values as "
                "separate arguments such as '-t 10'."
            )
        else:
            out.append(token)
    return out


RETRYABLE_CLEANED_REENTRY_PHASES = {
    CampaignPhase.SEED_SELECT,
    CampaignPhase.ARIADNE_ARRAY,
    CampaignPhase.PHASE_B_POLUS,
    CampaignPhase.SPLIT,
    CampaignPhase.GAUSSIAN,
    CampaignPhase.AIMALL,
    CampaignPhase.APPEND,
    CampaignPhase.FEREBUS,
}


def _campaign_paths(campaign_dir: Path):
    data = campaign_dir / DEFAULT_DATA_SUBDIR
    return {
        "data": data,
        "state": data / DEFAULT_STATE_FILENAME,
        "lock": data / DAEMON_LOCK_FILENAME,
        "lease": data / DAEMON_LEASE_DIRNAME,
        "journal": data / "journal.ndjson",
        "background_log": data / BACKGROUND_LOG_FILENAME,
        "background_pid": data / BACKGROUND_PID_FILENAME,
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


def _probe_daemon_lease(lease_path: Path) -> dict:
    status = {
        "lease_dir_exists": lease_path.is_dir(),
        "lease_path": str(lease_path),
        "lease_heartbeat": None,
    }
    heartbeat_path = lease_path / DAEMON_HEARTBEAT_FILENAME
    if not heartbeat_path.is_file():
        return status
    try:
        status["lease_heartbeat"] = json.loads(heartbeat_path.read_text(encoding="utf-8"))
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


def _lease_is_fresh(heartbeat: Any, *, stale_seconds: int = 900) -> bool:
    if not isinstance(heartbeat, dict):
        return False
    try:
        age = time.time() - float(heartbeat.get("time"))
    except Exception:
        return False
    return age <= float(stale_seconds)


def _reconcile_runtime_status(campaign: Path) -> Dict[str, Any]:
    paths = _campaign_paths(campaign)
    status: Dict[str, Any] = {}
    status.update(_probe_daemon_lock(paths["lock"]))
    status.update(_probe_daemon_lease(paths["lease"]))
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
    elif _lease_is_fresh(status.get("lease_heartbeat")):
        blockers.append("daemon lease heartbeat is fresh")
    if status.get("background_pid_alive"):
        blockers.append(
            "background daemon pid "
            + str(status.get("background_pid"))
            + " appears alive"
        )
    status["reconcile_apply_blockers"] = blockers
    return status


def _print_reconcile_runtime_warning(status: Dict[str, Any], campaign: Path) -> None:
    blockers = list(status.get("reconcile_apply_blockers") or [])
    if not blockers:
        return
    print("WARNING: a daemon may still be running for this campaign.", file=sys.stderr)
    for blocker in blockers:
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
        # make a corrupt old state an absolute blocker once the operator has
        # explicitly requested an archive through --apply.
        pass
    inventory = data_staging_inventory(campaign)
    if inventory.get("is_symlink"):
        blockers.append(".DATA/STAGING is a symlink")
    if inventory.get("has_symlink"):
        blockers.append(".DATA/STAGING contains symlink entries")
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
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
    if bool(getattr(args, "mock_ariadne", False)):
        argv.append("--mock-ariadne")
    if bool(getattr(args, "dry_run", False)):
        argv.append("--dry-run")
    if bool(getattr(args, "live", False)):
        argv.append("--live")
    if getattr(args, "poll_interval", None) is not None:
        argv.extend(["--poll-interval", str(int(args.poll_interval))])
    if getattr(args, "max_ticks", None) is not None:
        argv.extend(["--max-ticks", str(int(args.max_ticks))])
    if getattr(args, "preset", None):
        argv.extend(["--preset", str(args.preset)])
    return argv


def _tail_text(path: Path, *, max_lines: int = 40) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return "<could not read log: " + type(exc).__name__ + ": " + str(exc) + ">"
    return "\n".join(lines[-max_lines:])


def _selected_mode_count(args: argparse.Namespace) -> int:
    return sum([
        bool(getattr(args, "live", False)),
        bool(getattr(args, "dry_run", False)),
        bool(getattr(args, "mock_ariadne", False)),
    ])


def _launch_background_daemon(args: argparse.Namespace, campaign: Path) -> int:
    if os.environ.get(BACKGROUND_CHILD_ENV) == "1":
        print("--background is not allowed inside a background child process", file=sys.stderr)
        return 2
    if _selected_mode_count(args) > 1:
        print(
            "--live, --dry-run, and --mock-ariadne are mutually exclusive; pick one.",
            file=sys.stderr,
        )
        return 2
    if _selected_mode_count(args) == 0:
        print("no execution mode selected. Pick --live, --dry-run, or --mock-ariadne.", file=sys.stderr)
        return 3

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

    time.sleep(1.5)
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
        return int(rc) if int(rc) != 0 else 0

    payload = {
        "schema_version": BACKGROUND_PID_SCHEMA_VERSION,
        "pid": int(child.pid),
        "command": argv,
        "campaign_dir": str(campaign),
        "log_path": str(log_path),
        "started_at_utc": timestamp,
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "python_executable": sys.executable,
    }
    _atomic_write_background_pid(pid_path, payload)
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
    for label in ("training", "models"):
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
        if job_id:
            samples.append(
                phase + "@" + iteration + " " + status + " job_id=" + str(job_id)
            )
        else:
            samples.append(phase + "@" + iteration + " " + status)
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
    return _section("Recommendation", rows)


def _format_status(payload: Dict[str, Any], *, verbose: bool, journal_path: Path) -> str:
    lines: List[str] = []
    lines.extend(
        _section(
            "Campaign",
            [
                ("phase", payload.get("phase")),
                (
                    "iteration",
                    str(payload.get("iteration")) + " / max " + str(payload.get("max_iterations")),
                ),
                ("uid", payload.get("campaign_uid")),
                ("started", payload.get("campaign_started_iso")),
            ],
        )
    )
    pending = payload.get("pending_jobs")
    job_rows = []
    if isinstance(pending, dict) and pending:
        for phase, job_id in sorted(pending.items()):
            job_rows.append(("pending Slurm job " + str(phase), job_id if job_id else "done"))
    else:
        job_rows.append(("pending Slurm jobs", "none recorded in state"))
    job_rows.append(
        (
            "active submission intents",
            _format_active_submission_intents(payload.get("active_submission_intents")),
        )
    )
    lines.append("")
    lines.extend(_section("Jobs", job_rows))
    lease = "active: " + _heartbeat_summary(payload.get("lease_heartbeat"))
    if not payload.get("lease_dir_exists"):
        lease = "none"
    lines.append("")
    lines.extend(
        _section(
            "Runtime",
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
                    "shutdown requested",
                    "yes" if payload.get("shutdown_requested") else "no",
                ),
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
    lines.extend(
        _format_artifact_summary(
            payload.get("artifact_manifest_status"),
            state_contract=payload.get("state_artifact_contract_status"),
            verbose=verbose,
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
                ],
            )
        )
        if payload.get("lock_probe_error"):
            lines.append("")
            lines.extend(_section("Diagnostics", [("lock_probe_error", payload["lock_probe_error"])]))
    return "\n".join(lines) + "\n"


def _format_status_unavailable(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.extend(
        _section(
            "Campaign",
            [
                ("phase", "unknown"),
                ("state", payload.get("status_error")),
            ],
        )
    )
    if payload.get("state_error"):
        lines.append("")
        lines.extend(_section("State", [("error", payload.get("state_error"))]))
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


def _compact_event_details(event: Dict[str, Any]) -> str:
    detail_keys = [
        ("job_id", "job"),
        ("expected_tasks", "tasks"),
        ("n_tasks", "tasks"),
        ("n_completed", "completed"),
        ("n_failed", "failed"),
        ("n_kept", "kept"),
        ("n_rejected", "rejected"),
        ("n_frames", "frames"),
        ("action", "action"),
        ("reason", "reason"),
    ]
    parts: List[str] = []
    for key, label in detail_keys:
        if key in event and event.get(key) is not None:
            value = _format_value(event.get(key))
            if key == "reason" and len(value) > 90:
                value = value[:87] + "..."
            parts.append(label + "=" + value)
    return "  ".join(parts)


def _format_journal_events(events: Sequence[Dict[str, Any]], *, verbose: bool) -> str:
    if not events:
        return ""
    rows: List[Tuple[Dict[str, Any], str, str, str, str, str]] = []
    for event in events:
        event_name = str(event.get("event", "<missing>"))
        phase = _event_phase(event)
        rows.append(
            (
                event,
                _event_time(event),
                _event_iteration(event),
                phase,
                event_name,
                _compact_event_details(event),
            )
        )
    time_width = max(19, max(len(row[1]) for row in rows))
    iteration_width = max(6, max(len(row[2]) for row in rows))
    phase_width = max(18, max(len(row[3]) for row in rows))
    event_width = max(24, max(len(row[4]) for row in rows))

    lines: List[str] = []
    for event, event_time, iteration, phase, event_name, details in rows:
        first_line = (
            event_time.ljust(time_width)
            + "  "
            + iteration.ljust(iteration_width)
            + "  "
            + phase.ljust(phase_width)
            + "  "
            + event_name.ljust(event_width)
        )
        lines.append(first_line + "  " + (details if details else "-"))
        if verbose:
            for key in sorted(event):
                if key in {"ts", "event", "phase", "to_phase", "from_phase", "iteration"}:
                    continue
                lines.append("  " + key + ": " + _format_value(event[key]))
    return "\n".join(lines) + "\n"


def cmd_start(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(args.campaign_dir)
    if not campaign.exists():
        print("campaign-dir does not exist: " + str(campaign), file=sys.stderr)
        return 2

    config_path = Path(args.config).resolve() if args.config else campaign / "campaign.yaml"
    if not config_path.exists() and not getattr(args, "preset", None):
        print("campaign config not found: " + str(config_path), file=sys.stderr)
        return 2

    mode_count = sum([
        bool(getattr(args, "live", False)),
        bool(getattr(args, "dry_run", False)),
        bool(getattr(args, "mock_ariadne", False)),
    ])
    if mode_count > 1:
        print(
            "--live, --dry-run, and --mock-ariadne are mutually exclusive; "
            "pick one.", file=sys.stderr,
        )
        return 2
    if mode_count == 0:
        print("no execution mode selected. Pick one of:", file=sys.stderr)
        print("  --live          run against configured Slurm backends (requires sbatch + Gaussian + AIMAll + FEREBUS + ariadne)", file=sys.stderr)
        print("  --dry-run       stub backends, real file-system flow", file=sys.stderr)
        print("  --mock-ariadne  pure state-machine progression test", file=sys.stderr)
        return 3

    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    state_for_lock = None
    if not state_path.exists():
        artefacts = stateful_campaign_artifacts(campaign)
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
        except StateSchemaError as exc:
            print(
                "state.json is invalid; run `ichor-al-daemon reconcile --campaign-dir "
                + str(campaign)
                + "` before starting: "
                + str(exc),
                file=sys.stderr,
            )
            return 5
    if bool(getattr(args, "background", False)):
        return _launch_background_daemon(args, campaign)
    #Fix: --preset overlays the operator-supplied campaign.yaml on top of the
    #named preset. The preset is the base; the YAML on disk is the overlay.
    import yaml as _yaml
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as _f:
            campaign_payload = _yaml.safe_load(_f) or {}
    else:
        campaign_payload = {"schema_version": 3}
    if getattr(args, "preset", None):
        from .preset_loader import apply_preset, PresetError
        try:
            effective, preset_payload = apply_preset(args.preset, campaign_payload)
        except PresetError as exc:
            print("preset load failed: " + str(exc), file=sys.stderr)
            return 2
        try:
            config = CampaignConfig.from_dict(effective)
        except Exception as exc:
            print("effective config validation failed: " + str(exc), file=sys.stderr)
            return 2
        #journal emits the preset name; the daemon emits on first tick.
        try:
            from .daemon.journal import append_event
            journal_path = (
                campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
            )
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            append_event(
                journal_path, "preset_loaded",
                preset_name=str(args.preset),
                n_overlaid_keys=int(len(preset_payload)),
            )
        except Exception:
            pass
    else:
        config = CampaignConfig.from_yaml(config_path)

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

    #NB: journal the effective-config diff against CampaignConfig()
    #defaults. Operators inspecting the journal can see EXACTLY what the
    #preset overlay + their campaign.yaml combined to produce, without
    #having to diff two files by hand.
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

    job_finder = None  # set in the live branch below; mock/dry leave it None (no adopt check)
    job_liveness_checker = None
    if getattr(args, "live", False):
        avail = check_backends()
        if not avail.all_present:
            print(missing_backend_message(avail), file=sys.stderr)
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
        job_finder = make_live_job_finder()
        job_liveness_checker = make_live_job_liveness_checker()
    elif getattr(args, "dry_run", False):
        executor = DryRunPhaseExecutor(
            campaign_dir=campaign,
            config=config,
        )
        sacct_poller = DryRunSacctPoller(elapsed_seconds=0)
    elif getattr(args, "mock_ariadne", False):
        executor = MockPhaseExecutor()
        sacct_poller = None
    else:  # pragma: no cover - mode_count is validated before side effects
        raise AssertionError("validated execution mode was not handled")

    daemon_kwargs = {
        "campaign_dir": campaign,
        "config": config,
        "executor": executor,
    }
    if sacct_poller is not None:
        daemon_kwargs["sacct_poller"] = sacct_poller
    if job_finder is not None:
        daemon_kwargs["job_finder"] = job_finder
    if job_liveness_checker is not None:
        daemon_kwargs["job_liveness_checker"] = job_liveness_checker
    d = Daemon(**daemon_kwargs)
    if args.poll_interval is not None:
        #override config-loaded poll interval per-invocation.
        d.config.poll_interval_seconds = int(args.poll_interval)
    return d.run(max_ticks=args.max_ticks, catch_keyboard_interrupt=True)


_FEREBUS_JOB_NAME_EXTERNAL_PHASES = frozenset({"INITIAL_FEREBUS", "FEREBUS"})


def _load_active_submission_intents(campaign: Path) -> List[Dict[str, Any]]:
    intents: List[Dict[str, Any]] = []
    root = _submission_intent.intent_dir(campaign)
    if not root.is_dir():
        return intents
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and str(payload.get("status")) in _submission_intent.ACTIVE_STATUSES:
            intents.append(payload)
    return intents


def _lookup_active_slurm_job_for_cancel(job_id: str) -> Dict[str, Any]:
    cmd = [
        "squeue",
        "-j",
        str(job_id),
        "--noheader",
        "--format=%i|%T|%j",
    ]
    try:
        completed = subprocess.run(
            cmd,
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
        rows.append({
            "job_id": parts[0].strip() if len(parts) > 0 else "",
            "state": parts[1].strip() if len(parts) > 1 else "",
            "job_name": parts[2].strip() if len(parts) > 2 else "",
        })
    return {
        "active": bool(rows),
        "inconclusive": False,
        "rows": rows,
        "error": None,
    }


def _run_scancel(job_id: str) -> Tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["scancel", str(job_id)],
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
            },
        )

    for phase, job_id in sorted((state.pending_jobs or {}).items()):
        if not job_id:
            continue
        item = ensure(str(job_id))
        phase_name = str(phase)
        item["phases"].add(phase_name)
        if phase_name not in _FEREBUS_JOB_NAME_EXTERNAL_PHASES:
            item["expected_job_names"].add(
                live_job_name(state.campaign_uid, phase_name, int(state.iteration))
            )

    for intent in _load_active_submission_intents(campaign):
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

    return jobs


def _cancel_recorded_slurm_jobs(campaign: Path, state: Any) -> Dict[str, Any]:
    jobs = _collect_stop_cancel_jobs(campaign, state)
    cancelled: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for intent in _load_active_submission_intents(campaign):
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
        lookup = _lookup_active_slurm_job_for_cancel(job_id)
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
        ok, message = _run_scancel(job_id)
        if not ok:
            failed.append({
                "job_id": job_id,
                "reason": "scancel failed: " + message,
            })
            continue
        phases = sorted(str(phase) for phase in item.get("phases", set()))
        cancelled.append({
            "job_id": job_id,
            "phases": phases,
        })
        for phase in phases:
            if state.pending_jobs.get(phase) == job_id:
                state.pending_jobs[phase] = None
        for phase_name, iteration in sorted(item.get("intent_keys", set())):
            try:
                _submission_intent.mark_failed(
                    campaign,
                    str(phase_name),
                    int(iteration),
                    "operator_cancelled_via_stop",
                )
            except Exception:
                pass
    return {
        "cancelled": cancelled,
        "skipped": skipped,
        "failed": failed,
    }


def _journal_cancel_jobs_summary(journal_path: Path, summary: Dict[str, Any]) -> None:
    try:
        from .daemon.journal import append_event

        append_event(
            journal_path,
            "operator_cancelled_jobs",
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
    campaign = resolve_campaign_dir(args.campaign_dir)
    paths = _campaign_paths(campaign)
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
            cancel_summary = _cancel_recorded_slurm_jobs(campaign, fallback_state)
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
            cancel_summary = _cancel_recorded_slurm_jobs(campaign, fallback_state)
            _journal_cancel_jobs_summary(paths["journal"], cancel_summary)
            _print_cancel_jobs_summary(cancel_summary)
            return 10 if cancel_summary.get("failed") else 0
        print("state.json invalid: " + str(exc), file=sys.stderr)
        return 5
    state.shutdown_requested = True
    cancel_summary = None
    if bool(getattr(args, "cancel_jobs", False)):
        cancel_summary = _cancel_recorded_slurm_jobs(campaign, state)
    write_state(paths["state"], state)
    if cancel_summary is not None:
        _journal_cancel_jobs_summary(paths["journal"], cancel_summary)
    print("shutdown_requested=true set in " + str(paths["state"]))
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
    lines.extend(
        _section(
            "Last exception",
            [("LAST_EXCEPTION.json", _read_last_exception_summary(campaign))],
        )
    )

    lock_status = _probe_daemon_lock(paths["lock"])
    lease_status = _probe_daemon_lease(paths["lease"])
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

    intents = _load_active_submission_intents(campaign)
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
        or _lease_is_fresh(lease_status.get("lease_heartbeat"))
        or background.get("background_pid_alive")
    ):
        recommendation = "stop --cancel-jobs first, then rerun reconcile"
    if not cfg_path.is_file() and config_lock_path(campaign).is_file():
        recommendation = "restore campaign.yaml from config lock"
    lines.extend(_section("Recommendation", [("next action", recommendation)]))
    return "\n".join(lines) + "\n"


def cmd_status(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(args.campaign_dir)
    paths = _campaign_paths(campaign)
    if not paths["state"].exists():
        payload: Dict[str, Any] = {
            "status_error": "state_missing",
            "state_path": str(paths["state"]),
            "campaign_dir": str(campaign),
        }
        payload["recommendations"] = recommendation_dicts(
            build_status_recommendations(campaign, payload, paths["journal"])
        )
        payload["next_action"] = payload["recommendations"][0]["primary"]
        if bool(getattr(args, "json", False)):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(_format_status_unavailable(payload), end="")
            print("no state.json at " + str(paths["state"]), file=sys.stderr)
        return 4
    try:
        state = read_state(paths["state"])
    except StateSchemaError as exc:
        payload = {
            "status_error": "state_schema_invalid",
            "state_error": "StateSchemaError: " + str(exc),
            "state_path": str(paths["state"]),
            "campaign_dir": str(campaign),
        }
        payload["recommendations"] = recommendation_dicts(
            build_status_recommendations(campaign, payload, paths["journal"])
        )
        payload["next_action"] = payload["recommendations"][0]["primary"]
        if bool(getattr(args, "json", False)):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(_format_status_unavailable(payload), end="")
            print("state.json invalid: " + str(exc), file=sys.stderr)
        return 5
    payload = state.to_dict()
    payload["state_path"] = str(paths["state"])
    payload["lock_path"] = str(paths["lock"])
    payload.update(_probe_daemon_lock(paths["lock"]))
    payload.update(_probe_daemon_lease(paths["lease"]))
    payload.update(_probe_background_daemon(paths["background_pid"], paths["background_log"]))
    payload["active_submission_intents"] = _load_active_submission_intents(campaign)
    payload["latest_halt_event"] = _latest_journal_event(paths["journal"], "halt")
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
        print(json.dumps(payload, indent=2, sort_keys=True))
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


def cmd_resume(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(args.campaign_dir)
    state_path = _campaign_paths(campaign)["state"]
    if state_path.exists():
        try:
            state = read_state(state_path)
        except StateSchemaError as exc:
            print("state.json invalid: " + str(exc), file=sys.stderr)
            return 5
        if state.phase is CampaignPhase.HALTED:
            print(
                "campaign is HALTED; run `ichor-al-daemon reconcile --campaign-dir "
                + str(campaign)
                + " --apply` if the recovery proposal is safe before resuming",
                file=sys.stderr,
            )
            return 6
        if state.shutdown_requested:
            state.shutdown_requested = False
            write_state(state_path, state)
            print("shutdown_requested=false set in " + str(state_path))
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
    """Make conclusively terminal active intents inactive before safe apply.

    This is intentionally fail-closed.  ``reconcile --apply`` may clear a stale
    active intent only when ``squeue`` no longer sees the job and ``sacct``
    reports terminal rows for the recorded JobID.  Missing scheduler data keeps
    the intent blocking so an operator cannot accidentally duplicate a live job.
    """
    from .daemon.journal import append_event
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
                if phase in _FEREBUS_JOB_NAME_EXTERNAL_PHASES:
                    blocking.append({
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": job_id,
                        "expected_job_name": expected_job_name,
                        "reason": "FEREBUS PRE_SUBMIT intent has no job_id; "
                        "job name is external, inspect scheduler manually",
                    })
                    continue
                if not expected_job_name:
                    blocking.append({
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": job_id,
                        "expected_job_name": expected_job_name,
                        "reason": "PRE_SUBMIT intent has no expected job name",
                    })
                    continue
                lookup = sacct_poll.find_running_job_by_name_detailed(
                    expected_job_name,
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
                if lookup.job_id:
                    blocking.append({
                        "phase": phase,
                        "iteration": iteration,
                        "job_id": str(lookup.job_id),
                        "expected_job_name": expected_job_name,
                        "reason": "matching scheduler job exists but no job_id "
                        "was persisted in the intent",
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
        phase = str(candidate["phase"])
        iteration = int(candidate["iteration"])
        reason = "reconcile_apply_pre_submit_no_job_id"
        _submission_intent.mark_superseded(
            campaign,
            phase,
            iteration,
            reason,
        )
        payload = dict(candidate)
        payload["reason"] = reason
        resolved.append(payload)
        try:
            append_event(
                campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
                "reconcile_resolved_terminal_intent",
                phase=phase,
                iteration=iteration,
                job_id="",
                terminal_state="PRE_SUBMIT_NO_JOB_ID",
                n_sacct_rows=int(candidate.get("n_sacct_rows") or 0),
                reason=reason,
            )
        except Exception:
            pass
    for candidate in terminal_candidates:
        phase = str(candidate["phase"])
        iteration = int(candidate["iteration"])
        terminal_state = str(candidate["terminal_state"])
        failure_reason = "reconcile_apply_terminal_job:" + terminal_state
        _submission_intent.mark_failed(
            campaign,
            phase,
            iteration,
            failure_reason,
        )
        payload = dict(candidate)
        payload["reason"] = failure_reason
        resolved.append(payload)
        try:
            append_event(
                campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
                "reconcile_resolved_terminal_intent",
                **payload,
            )
        except Exception:
            pass
    return resolved, blocking


def _retry_phase_from_cleaned_report(report) -> Optional[CampaignPhase]:
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


def _reconcile_apply_contract_error(campaign: Path, state: Any) -> Optional[str]:
    from .daemon.artifact_contracts import state_artifact_contract_status

    status = state_artifact_contract_status(campaign, state)
    if bool(status.get("ok")):
        return None
    detail = str(status.get("error") or "state contract invalid")
    return (
        "phase="
        + str(status.get("phase"))
        + " training_set_version="
        + str(status.get("training_set_version"))
        + " models_version="
        + str(status.get("models_version"))
        + ": "
        + detail
    )


def cmd_reconcile(args: argparse.Namespace) -> int:
    restore_config = bool(getattr(args, "restore_config_from_lock", False))
    archive_staging_requested = bool(getattr(args, "archive_staging", False))
    campaign = resolve_campaign_dir(
        args.campaign_dir,
        require_campaign_yaml=not restore_config,
    )
    runtime_status = _reconcile_runtime_status(campaign)
    if archive_staging_requested and not bool(getattr(args, "apply", False)):
        print(
            "refusing --archive-staging without --apply; this option mutates "
            ".DATA/STAGING and must be explicit",
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
        print("Config proposal written to: " + str(target_config))
        print(
            "The proposal is a dense full config snapshot restored from "
            "config_lock.json."
        )
        print("Review the proposal, then promote it manually:")
        print("    mv " + str(target_config) + " " + str(campaign / "campaign.yaml"))
        print("Then re-run reconcile:")
        print("    ichor-al-daemon reconcile --campaign-dir " + str(campaign) + " --apply")
        return 0
    if bool(getattr(args, "apply", False)) and runtime_status.get("reconcile_apply_blockers"):
        print("refusing --apply because a daemon may still be running", file=sys.stderr)
        _print_reconcile_runtime_warning(runtime_status, campaign)
        return 9
    report = propose_recovery(
        campaign,
        allow_fresh_init_on_nonempty=bool(getattr(args, "allow_fresh_init", False)),
    )
    target = write_proposed_state(campaign, report)
    config_path = campaign / "campaign.yaml"
    config = None
    config_review = None
    if config_path.is_file():
        try:
            config = CampaignConfig.from_yaml(config_path)
            config_review = review_config_changes(
                campaign,
                config,
                report.proposed_state,
                initialise_missing=False,
            )
        except Exception as exc:
            if bool(getattr(args, "apply", False)):
                print("campaign config could not be loaded: " + str(exc), file=sys.stderr)
                return 8
            print("campaign config could not be reviewed: " + str(exc), file=sys.stderr)
    print("Proposed state written to: " + str(target))
    print("")
    print("=== Diagnostic notes ===")
    for note in report.notes:
        print("  - " + note)
    print("")
    print("Committed training versions: " + repr(report.committed_training_versions))
    print("Valid training versions:     " + repr(report.valid_training_versions))
    print("Committed model versions:    " + repr(report.committed_model_versions))
    print("Valid model versions:        " + repr(report.valid_model_versions))
    print("Last phase in journal:       " + repr(report.last_phase_in_journal))
    print("Last iteration in journal:   " + repr(report.last_iteration_in_journal))
    if report.decision:
        print("Recovery decision:           " + str(report.decision))
    if report.trusted_artifacts:
        print("Trusted artefacts:           " + repr(report.trusted_artifacts))
    if report.blocking_artifacts:
        print("Blocking artefacts:          " + repr(report.blocking_artifacts))
    if report.unsafe_reasons:
        print("Unsafe recovery reasons:     " + repr(report.unsafe_reasons))
    if report.active_submission_intents:
        print("Active submission intents:   " + repr([
            {
                "phase": i.get("phase"),
                "iteration": i.get("iteration"),
                "status": i.get("status"),
                "job_id": i.get("job_id"),
                "expected_job_name": i.get("expected_job_name"),
            }
            for i in report.active_submission_intents
        ]))
    if report.recommended_actions:
        print("Recommended actions:")
        for action in report.recommended_actions:
            print("  - " + str(action))
    print("")
    if not bool(getattr(args, "apply", False)) and runtime_status.get("reconcile_apply_blockers"):
        _print_reconcile_runtime_warning(runtime_status, campaign)
    target_canonical = target.with_name(DEFAULT_STATE_FILENAME)
    if config_review is not None:
        print("=== Config lock review ===")
        formatted = format_config_review(config_review)
        if formatted:
            print(formatted)
        else:
            print("No campaign.yaml changes against the config lock.")
        print("")
    if not bool(getattr(args, "apply", False)):
        print("Review the proposal, then promote it manually:")
        print("    mv " + str(target) + " " + str(target_canonical))
        print("")
        print("Or apply the safe proposal automatically:")
        print("    ichor-al-daemon reconcile --campaign-dir " + str(campaign) + " --apply")
        return 0

    if report.proposed_state.phase is CampaignPhase.DONE:
        print(
            "refusing --apply because proposed state is "
            + report.proposed_state.phase.value,
            file=sys.stderr,
        )
        return 9
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
            print(
                "refusing --apply because active submission intents are still live "
                "or scheduler status is inconclusive:",
                file=sys.stderr,
            )
            for item in blocking_intents:
                print(
                    "  - "
                    + str(item.get("phase"))
                    + "@"
                    + str(item.get("iteration"))
                    + " job_id="
                    + str(item.get("job_id"))
                    + " expected_job_name="
                    + str(item.get("expected_job_name"))
                    + ": "
                    + str(item.get("reason")),
                    file=sys.stderr,
                )
            print("Next safe command if these jobs should be cancelled:", file=sys.stderr)
            print(
                "  ichor-al-daemon stop --campaign-dir "
                + str(campaign)
                + " --cancel-jobs",
                file=sys.stderr,
            )
            print(
                "  ichor-al-daemon reconcile --campaign-dir "
                + str(campaign)
                + " --apply",
                file=sys.stderr,
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
                data_staging_archive_mode = "operator"
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
    if "dangling training staging directories exist" in report.unsafe_reasons:
        ok_to_archive_training, training_reason = training_staging_can_archive_for_reconcile(
            campaign,
            report.proposed_state,
        )
        if ok_to_archive_training:
            cleanable_reasons.add("dangling training staging directories exist")
        else:
            report.notes.append(
                "dangling training staging cannot be archived automatically: "
                + training_reason
            )
    uncleanable = [
        reason
        for reason in report.unsafe_reasons
        if reason not in cleanable_reasons
    ]
    if uncleanable:
        print(
            "refusing --apply because reconcile reported unsafe artefacts "
            "that cannot be cleaned automatically:",
            file=sys.stderr,
        )
        for reason in uncleanable:
            print("  - " + reason, file=sys.stderr)
        if archive_staging_refusal_reasons:
            print("Archive staging blockers:", file=sys.stderr)
            for reason in archive_staging_refusal_reasons:
                print("  - " + str(reason), file=sys.stderr)
        return 9

    original_report = report
    archived_scripts = archive_scripts_for_reconcile(
        campaign
    ) if ".DATA/SCRIPTS contains sbatch scripts" in report.unsafe_reasons else []
    archived = []
    if ".DATA/STAGING is non-empty" in report.unsafe_reasons:
        if data_staging_archive_mode == "ferebus_reentry":
            archived = archive_data_staging_for_ferebus_reentry(
                campaign,
                report.proposed_state,
            )
        elif data_staging_archive_mode == "operator":
            archived = archive_data_staging_for_operator_reconcile(campaign)
    removed_model_staging = clean_model_iteration_staging_for_reconcile(
        campaign,
        report.proposed_state,
    ) if "dangling model staging directories exist" in report.unsafe_reasons else []
    archived_training_staging = archive_training_staging_for_reconcile(
        campaign,
        report.proposed_state,
    ) if "dangling training staging directories exist" in report.unsafe_reasons else []
    removed = clean_reentry_staging(campaign, report.proposed_state.phase)
    cleanup_paths_already_done = (
        list(archived_scripts)
        + list(archived)
        + list(removed_model_staging)
        + list(archived_training_staging)
        + list(removed)
    )

    if original_report.proposed_state.phase is CampaignPhase.HALTED:
        report = propose_recovery(
            campaign,
            allow_fresh_init_on_nonempty=bool(getattr(args, "allow_fresh_init", False)),
        )
        _apply_retry_phase_after_cleaned_halt(report, original_report)
        target = write_proposed_state(campaign, report)
        if config is not None:
            try:
                config_review = review_config_changes(
                    campaign,
                    config,
                    report.proposed_state,
                    initialise_missing=False,
                )
            except Exception as exc:
                print("campaign config could not be reviewed: " + str(exc), file=sys.stderr)
                _print_cleanup_already_happened(cleanup_paths_already_done)
                return 8
        if report.unsafe_reasons:
            print(
                "refusing --apply because cleanup did not produce a safe recovery proposal:",
                file=sys.stderr,
            )
            for reason in report.unsafe_reasons:
                print("  - " + reason, file=sys.stderr)
            _print_cleanup_already_happened(cleanup_paths_already_done)
            return 9
        if report.proposed_state.phase in (CampaignPhase.HALTED, CampaignPhase.DONE):
            print(
                "refusing --apply because proposed state is "
                + report.proposed_state.phase.value
                + " after cleanup",
                file=sys.stderr,
            )
            _print_cleanup_already_happened(cleanup_paths_already_done)
            return 9
        if config_review is not None and config_review.blocked_changes:
            print("refusing --apply because campaign.yaml has locked changes", file=sys.stderr)
            print(format_config_review(config_review), file=sys.stderr)
            _print_cleanup_already_happened(cleanup_paths_already_done)
            return 8

    contract_error = _reconcile_apply_contract_error(campaign, report.proposed_state)
    if contract_error is not None:
        print(
            "refusing --apply because the final proposed state fails the "
            "state/artefact contract:",
            file=sys.stderr,
        )
        print("  - " + contract_error, file=sys.stderr)
        _print_cleanup_already_happened(cleanup_paths_already_done)
        return 9

    backup_path = _copy_existing_timestamped(
        target_canonical,
        ".before-reconcile-",
    )
    try:
        write_state(target_canonical, report.proposed_state)
    except Exception as exc:
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
        apply_config_lock_update(campaign, config)
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
    for intent_path in sorted(
        (campaign / DEFAULT_DATA_SUBDIR / "submission_intents").glob("*.json")
        if (campaign / DEFAULT_DATA_SUBDIR / "submission_intents").is_dir()
        else []
    ):
        try:
            payload = json.loads(intent_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            str(payload.get("phase")) == report.proposed_state.phase.value
            and int(payload.get("iteration", -999999)) == int(report.proposed_state.iteration)
            and str(payload.get("status")) == "FAILED"
        ):
            _submission_intent.mark_superseded(
                campaign,
                report.proposed_state.phase.value,
                int(report.proposed_state.iteration),
                "reconcile_apply_retry",
            )
    applied_proposal_path = _rename_existing_timestamped(target, ".applied-")
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
        append_event(
            campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
            "reconcile_applied",
            phase=report.proposed_state.phase.value,
            iteration=int(report.proposed_state.iteration),
            n_allowed_config_changes=(
                len(config_review.allowed_changes) if config_review is not None else 0
            ),
            n_removed_stale_paths=len(removed) + len(removed_model_staging),
            n_archived_staging_paths=len(archived) + len(archived_training_staging),
            archived_staging_path=(archived[0] if archived else None),
            archived_training_staging_paths=archived_training_staging,
            n_archived_scripts_paths=len(archived_scripts),
            archived_scripts_path=(archived_scripts[0] if archived_scripts else None),
            recomputed_after_transient_cleanup=(
                original_report.proposed_state.phase is CampaignPhase.HALTED
            ),
        )
    except Exception:
        pass
    print("Applied proposed state: " + str(target_canonical))
    if backup_path is not None:
        print("Previous state backup: " + str(backup_path))
    if applied_proposal_path is not None:
        print("Applied proposal archive: " + str(applied_proposal_path))
    if removed:
        print("Removed stale uncommitted artefacts:")
        for path in removed:
            print("  - " + path)
    if removed_model_staging:
        print("Removed stale model staging:")
        for path in removed_model_staging:
            print("  - " + path)
    if archived_scripts:
        print("Archived stale .DATA/SCRIPTS:")
        for path in archived_scripts:
            print("  - " + path)
    if archived:
        print("Archived stale .DATA/STAGING:")
        for path in archived:
            print("  - " + path)
    if archived_training_staging:
        print("Archived stale training staging:")
        for path in archived_training_staging:
            print("  - " + path)
    print("")
    print("Start the daemon with:")
    print("    ichor-al-daemon start --campaign-dir " + str(campaign) + " --live")
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    campaign = resolve_campaign_dir(args.campaign_dir)
    if bool(getattr(args, "list_event_types", False)):
        for event_type in sorted(KNOWN_EVENT_TYPES):
            print(event_type)
        return 0
    journal_path = _campaign_paths(campaign)["journal"]
    if not journal_path.exists():
        print("no journal at " + str(journal_path), file=sys.stderr)
        return 4
    iterator = read_events(
        journal_path,
        since=args.since,
        event_type=args.event_type or None,
    )
    events = list(iterator)
    last_n = getattr(args, "last_n", None)
    if last_n is not None:
        events = events[-int(last_n):] if int(last_n) > 0 else []
    if bool(getattr(args, "json", False)) or bool(getattr(args, "raw", False)):
        for event in events:
            print(json.dumps(event, sort_keys=True))
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


def _resolve_init_source(campaign: Path, raw_source: Optional[str]) -> Path:
    if raw_source:
        source = Path(raw_source).expanduser().resolve()
    else:
        source = campaign / "pool.xyz"
    if not source.exists():
        raise FileNotFoundError(
            "source trajectory does not exist: "
            + str(source)
            + ". Put pool.xyz in the campaign directory or pass --source PATH."
        )
    if not source.is_file():
        raise FileNotFoundError("source trajectory is not a file: " + str(source))
    return source


def _import_pool_impl(args: argparse.Namespace, campaign: Path, source: Path) -> int:
    """Pull the operator's MD trajectory into the campaign's canonical
    pool location and write a SHA-pinned manifest next to it.

    Outlier-filter behaviour is layered:

      1. --no-outlier-filter on the command line wins regardless. handy
         when you want to import a pool verbatim for debugging.
      2. otherwise the outlier_filter block in campaign.yaml takes effect
         (enabled flag plus the two z-thresholds), if a campaign.yaml is
         present and parses cleanly.
      3. if neither applies (no config, no flag), the dataclass defaults
         kick in -- filter on, energy z-cap 3.0, per-atom-rmsd z-cap 4.0.

    Refuses to overwrite an existing pool unless --force is passed --
    overwriting wipes the SHA the previously-committed iterations were
    pinned to, so we make the operator say it out loud.
    """
    from .acquisition.trajectory_pool import TrajectoryPool
    from .config import CampaignConfig

    if bool(getattr(args, "force", False)):
        existing_manifest = campaign / ".DATA" / "TRAJECTORY" / "pool.manifest.json"
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

    # work out the outlier settings using the layered precedence above.
    # start from dataclass defaults; let campaign.yaml override; let the
    # CLI flag have the final word.
    cfg_path = campaign / "campaign.yaml"
    enabled = True
    energy_z = 3.0
    per_atom_z = 4.0
    if cfg_path.is_file():
        try:
            campaign_config = CampaignConfig.from_yaml(cfg_path)
            of = campaign_config.outlier_filter
            enabled = bool(of.enabled)
            energy_z = float(of.energy_z_threshold)
            per_atom_z = float(of.per_atom_rmsd_z_threshold)
        except Exception as _exc:
            # bad campaign.yaml shouldn't block import -- just fall back to
            # the defaults and let the daemon flag the config problem later.
            print(
                "warning: campaign.yaml not loaded for outlier-filter "
                "settings (" + str(_exc)[:80] + "); using defaults",
                file=sys.stderr,
            )
    if bool(getattr(args, "no_outlier_filter", False)):
        # CLI flag always wins. yes, this is the operator saying "off".
        enabled = False

    try:
        pool = TrajectoryPool.import_from(
            source, campaign, overwrite=bool(args.force),
            outlier_filter_enabled=enabled,
            energy_z_threshold=energy_z,
            per_atom_rmsd_z_threshold=per_atom_z,
        )
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        print("Re-run with --force to overwrite the existing pool.", file=sys.stderr)
        return 13
    except (FileNotFoundError, ValueError) as exc:
        print("trajectory import failed: " + str(exc), file=sys.stderr)
        return 14
    #journal the import + filter outcome so the operator has a
    #good record of how many frames were rejected and why.
    # defined before the try so the summary line below can rely on them even
    # if the journal bookkeeping blows up and gets swallowed.
    rejected_path = campaign / ".DATA" / "TRAJECTORY" / "rejected.json"
    n_rejected = 0
    try:
        import json as _json
        from .daemon.journal import append_event
        if rejected_path.is_file():
            with open(rejected_path, "r", encoding="utf-8") as _rf:
                rejected_payload = _json.load(_rf) or {}
            n_rejected = int(rejected_payload.get("rejected_count", 0))
        journal_path = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        append_event(
            journal_path, "trajectory_pool_filtered",
            n_imported=int(pool.n_frames()) + n_rejected,
            n_kept=int(pool.n_frames()),
            n_rejected=int(n_rejected),
            outlier_filter_enabled=bool(enabled),
            energy_z_threshold_used=float(energy_z),
            per_atom_rmsd_z_threshold_used=float(per_atom_z),
        )
    except Exception:
        #Robust journal write; never block the import on this.
        pass
    suffix = ""
    if enabled and rejected_path.is_file() and n_rejected > 0:
        suffix = " (" + str(n_rejected) + " frames rejected by outlier filter)"
    print(
        "Imported pool: "
        + str(pool.canonical_path)
        + " (" + str(pool.n_frames()) + " frames, "
        + str(pool.manifest.natoms) + " atoms, SHA " + pool.sha256[:12] + "...)"
        + suffix
    )
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Initialise campaign.yaml and import the trajectory pool."""
    try:
        campaign = _resolve_init_campaign_dir(getattr(args, "campaign_dir", None))
    except CampaignDirResolutionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        from .campaign_yaml import CampaignYamlError, initialise_campaign_yaml

        initialise_campaign_yaml(campaign)
    except CampaignYamlError as exc:
        print("campaign.yaml initialisation failed: " + str(exc), file=sys.stderr)
        return 15
    except Exception as exc:
        print("campaign.yaml initialisation failed: " + str(exc), file=sys.stderr)
        return 15
    try:
        source = _resolve_init_source(campaign, getattr(args, "source", None))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("Initialised campaign.yaml: " + str(campaign / "campaign.yaml"))
    return _import_pool_impl(args, campaign, source)


def cmd_import_pool(args: argparse.Namespace) -> int:
    print(
        "warning: import-pool is deprecated; use 'ichor-al-daemon init' instead.",
        file=sys.stderr,
    )
    return cmd_init(args)


def cmd_preflight(args: argparse.Namespace) -> int:
    avail = check_backends()
    print(json.dumps(asdict(avail), indent=2, sort_keys=True))
    if avail.all_present:
        return 0
    print("", file=sys.stderr)
    print(missing_backend_message(avail), file=sys.stderr)
    return 12


def build_parser() -> argparse.ArgumentParser:
    examples = """\
Campaign directory:
  If -c/--campaign-dir is omitted, the current directory is used when it
  contains campaign.yaml.

Examples:
  cd ~/campaigns/water_001
  ichor-al-daemon init
  ichor-al-daemon status
  ichor-al-daemon start -l
  ichor-al-daemon start -lb
  ichor-al-daemon journal -e phase_submitted

  ichor-al-daemon start -c ~/campaigns/water_001 --live
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
        p.add_argument(
            "-b",
            "--background",
            action="store_true",
            help=(
                "Launch the daemon as a detached background child. The foreground "
                "command validates the campaign, writes a PID file, and appends logs "
                "under .DATA/ACTIVE_LEARNING by default."
            ),
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
            "Start a campaign daemon. From inside a campaign directory, "
            "--campaign-dir can be omitted."
        ),
        epilog=(
            "Examples:\n"
            "  ichor-al-daemon start -d -t 10\n"
            "  ichor-al-daemon start -l\n"
            "  ichor-al-daemon start -lb\n"
            "  ichor-al-daemon start -c ~/campaigns/water_001 --live"
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
        "-m",
        "--mock-ariadne", action="store_true",
        help="Use MockPhaseExecutor for pure state-machine progression tests (no on-disk artefacts).",
    )
    p_start.add_argument(
        "-d",
        "--dry-run", action="store_true",
        help="Use DryRunPhaseExecutor: stub all backend calls but produce real on-disk artefacts (scripts, training-set versions, manifests).",
    )
    p_start.add_argument(
        "-l",
        "--live", action="store_true",
        help="Use LiveBackendsPhaseExecutor against configured Slurm backends (sbatch + Gaussian + AIMAll + FEREBUS + ARIADNE). Refuses with exit 12 if any are missing.",
    )
    p_start.add_argument(
        "-p",
        "--poll-interval",
        type=int,
        default=None,
        help="Override poll_interval_seconds from the config.",
    )
    p_start.add_argument(
        "-t",
        "--max-ticks",
        type=int,
        default=None,
        help="Limit total tick count (testing / time-boxed runs).",
    )
    p_start.add_argument(
        "-P",
        "--preset",
        default=None,
        help=(
            "Name of a YAML preset under ichor_hpc/.../presets/ to overlay "
            "campaign.yaml on top of. campaign.yaml wins on every "
            "explicitly-present key."
        ),
    )
    add_background_options(p_start)
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser(
        "stop",
        help="Request a graceful shutdown.",
        description=(
            "Request daemon shutdown for the resolved campaign. Plain stop does "
            "not cancel Slurm jobs unless --cancel-jobs is supplied."
        ),
    )
    add_campaign(p_stop)
    p_stop.add_argument(
        "-x",
        "--cancel-jobs",
        action="store_true",
        help=(
            "Also scancel active Slurm jobs recorded by this campaign. "
            "Plain stop only requests daemon shutdown and leaves jobs alone."
        ),
    )
    p_stop.set_defaults(func=cmd_stop)

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
    p_resume.add_argument("-m", "--mock-ariadne", action="store_true")
    p_resume.add_argument("-d", "--dry-run", action="store_true")
    p_resume.add_argument("-l", "--live", action="store_true")
    p_resume.add_argument("-p", "--poll-interval", type=int, default=None)
    p_resume.add_argument("-t", "--max-ticks", type=int, default=None)
    p_resume.add_argument("-P", "--preset", default=None)
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
            "Allow reconcile to propose INIT even when non-state campaign "
            "artifacts are present. Default is conservative HALTED/adoption-ready recovery."
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
        "--restore-config-from-lock",
        action="store_true",
        help=(
            "When campaign.yaml is missing, write campaign.yaml.proposed from "
            "the locked canonical config. Proposal-only; never overwrites "
            "campaign.yaml."
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
                "Path to the operator's MD trajectory (.xyz). Defaults to "
                "<campaign-dir>/pool.xyz when omitted."
            ),
        )
        p.add_argument(
            "-f",
            "--force", action="store_true",
            help="Overwrite an existing pool (DANGEROUS: invalidates every committed "
                 "iteration's frame-id provenance).",
        )
        p.add_argument(
            "-O",
            "--no-outlier-filter", action="store_true", dest="no_outlier_filter",
            help="Skip the pre-Phase-A outlier filter; import every frame verbatim. "
                 "Default behaviour (filter ON) rejects per-atom z > 4 frames and "
                 "writes rejected.json alongside pool.manifest.json.",
        )

    p_init = sub.add_parser(
        "init",
        help="Initialise campaign.yaml and import pool.xyz.",
        description=(
            "Initialise or populate campaign.yaml from the packaged template, "
            "then copy the operator trajectory into the campaign pool and "
            "write a SHA-pinned manifest. From inside a campaign directory, "
            "--campaign-dir and --source can be omitted when campaign.yaml "
            "and pool.xyz are present."
        ),
        epilog=(
            "Examples:\n"
            "  ichor-al-daemon init\n"
            "  ichor-al-daemon init -c ~/campaigns/water_001 -s pool.xyz"
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
            "backend profile. The campaign path is accepted for symmetry but "
            "is not used by the backend probe."
        ),
    )
    p_pre.add_argument(
        "-c",
        "--campaign-dir",
        default=".",
        help="Accepted for symmetry with other commands; not used by preflight.",
    )
    p_pre.set_defaults(func=cmd_preflight)

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
