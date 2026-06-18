"""CLI for the ICHOR active-learning daemon.

Console entry point 'ichor-al-daemon' registered in
'ichor_cli/setup.cfg'. Subcommands available:

    start      Start the daemon in the foreground (user backgrounds with nohup).
    stop       Set shutdown_requested=true in state.json; running daemon picks
               it up on next tick.
    status     Print the current state snapshot.
    resume     Equivalent to start when state.json already exists.
    reconcile  Inspect on-disk artefacts and propose a recovered state.
    journal    Tail or filter the campaign journal.
    import-pool Import and outlier-filter a trajectory into the campaign pool.

All commands take '--campaign-dir DIR' (required). 'start' takes
'--config FILE' (the campaign.yaml).

The CLI builds the Daemon with default executor/sacct_poller for production;
'--mock-ariadne' swaps the executor for a deterministic Mock so dry-runs
exercise the state machine without invoking real backends.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import CampaignConfig
from .daemon.daemon import (
    DAEMON_HEARTBEAT_FILENAME,
    DAEMON_LEASE_DIRNAME,
    DAEMON_LOCK_FILENAME,
    DEFAULT_DATA_SUBDIR,
    Daemon,
)
from .daemon.journal import iter_events, read_events
from .daemon.config_lock import (
    apply_config_lock_update,
    archive_data_staging_for_ferebus_reentry,
    assert_config_unchanged_for_start,
    clean_reentry_staging,
    ferebus_reentry_can_archive_data_staging,
    format_config_review,
    review_config_changes,
)
from .daemon.dry_run_executor import DryRunPhaseExecutor
from .daemon.dry_run_sacct import DryRunSacctPoller
from .daemon.live_executor import (
    LiveBackendNotAvailableError,
    LiveBackendsPhaseExecutor,
    make_live_job_finder,
    make_live_job_liveness_checker,
)
from .daemon.phase_executor import MockPhaseExecutor
from .daemon.preflight import check_backends, missing_backend_message
from .daemon.reconcile import propose_recovery, write_proposed_state
from .daemon import submission_intent as _submission_intent
from .daemon.state import (
    CampaignPhase,
    DEFAULT_STATE_FILENAME,
    StateSchemaError,
    read_state,
    write_state,
)


__all__ = ["build_parser", "main"]


def _campaign_paths(campaign_dir: Path):
    data = campaign_dir / DEFAULT_DATA_SUBDIR
    return {
        "data": data,
        "state": data / DEFAULT_STATE_FILENAME,
        "lock": data / DAEMON_LOCK_FILENAME,
        "lease": data / DAEMON_LEASE_DIRNAME,
        "journal": data / "journal.ndjson",
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


def _format_artifact_summary(status: Any, *, verbose: bool) -> List[str]:
    if not isinstance(status, dict):
        return _section("Artifacts", [("status", "unavailable")])
    rows = []
    for label in ("training", "models"):
        item = status.get(label)
        if isinstance(item, dict):
            ok = "ok" if item.get("ok") else "problem"
            version = item.get("version")
            rows.append((label + " v" + str(version), ok))
            if verbose:
                for error in item.get("errors") or []:
                    rows.append(("  " + label + " error", error))
        elif item is not None:
            rows.append((label, item))
    if "error" in status:
        rows.append(("error", status.get("error")))
    return _section("Artifacts", rows or [("status", "not checked")])


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
            job_rows.append((str(phase), job_id if job_id else "done"))
    else:
        job_rows.append(("pending", "none"))
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
                ("lock", _lock_summary(payload.get("lock_held"))),
                ("lease", lease),
                ("shutdown_requested", payload.get("shutdown_requested")),
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
            verbose=verbose,
        )
    )
    if verbose:
        lines.append("")
        lines.extend(
            _section(
                "Paths",
                [
                    ("state", payload.get("state_path")),
                    ("lock", payload.get("lock_path")),
                    ("lease", payload.get("lease_path")),
                ],
            )
        )
        if payload.get("lock_probe_error"):
            lines.append("")
            lines.extend(_section("Diagnostics", [("lock_probe_error", payload["lock_probe_error"])]))
    return "\n".join(lines) + "\n"


def _event_time(event: Dict[str, Any]) -> str:
    ts = str(event.get("ts", ""))
    if "T" in ts:
        tail = ts.split("T", 1)[1]
        return tail.split(".", 1)[0].replace("+00:00", "")
    return ts[:8] if ts else "--:--:--"


def _event_phase(event: Dict[str, Any]) -> str:
    for key in ("phase", "to_phase", "from_phase"):
        value = event.get(key)
        if value is not None:
            return str(value)
    return "-"


def _compact_event_details(event: Dict[str, Any]) -> str:
    detail_keys = [
        ("iteration", "iter"),
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
    rows: List[Tuple[Dict[str, Any], str, str, str, str]] = []
    for event in events:
        event_name = str(event.get("event", "<missing>"))
        phase = _event_phase(event)
        rows.append(
            (
                event,
                _event_time(event),
                event_name,
                phase,
                _compact_event_details(event),
            )
        )
    time_width = max(8, max(len(row[1]) for row in rows))
    event_width = max(24, max(len(row[2]) for row in rows))
    phase_width = max(18, max(len(row[3]) for row in rows))

    lines: List[str] = []
    for event, event_time, event_name, phase, details in rows:
        first_line = (
            event_time.ljust(time_width)
            + "  "
            + event_name.ljust(event_width)
            + "  "
            + phase.ljust(phase_width)
        )
        lines.append(first_line + (("  " + details) if details else ""))
        if verbose:
            for key in sorted(event):
                if key in {"ts", "event", "phase", "to_phase", "from_phase"}:
                    continue
                lines.append("  " + key + ": " + _format_value(event[key]))
    return "\n".join(lines) + "\n"


def cmd_start(args: argparse.Namespace) -> int:
    campaign = Path(args.campaign_dir).resolve()
    if not campaign.exists():
        print("campaign-dir does not exist: " + str(campaign), file=sys.stderr)
        return 2

    config_path = Path(args.config).resolve() if args.config else campaign / "campaign.yaml"
    if not config_path.exists() and not getattr(args, "preset", None):
        print("campaign config not found: " + str(config_path), file=sys.stderr)
        return 2
    #Fix: --preset overlays the operator-supplied campaign.yaml on top of the
    #named preset. The preset is the base; the YAML on disk is the overlay.
    import yaml as _yaml
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as _f:
            campaign_payload = _yaml.safe_load(_f) or {}
    else:
        campaign_payload = {"schema_version": 2}
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

    paths = _campaign_paths(campaign)
    state_for_lock = None
    if paths["state"].is_file():
        try:
            state_for_lock = read_state(paths["state"])
        except StateSchemaError:
            state_for_lock = None
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

    #mutually exclusive mode selection required.
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
    else:
        print("no execution mode selected. Pick one of:", file=sys.stderr)
        print("  --live          run against configured Slurm backends (requires sbatch + Gaussian + AIMAll + FEREBUS + ariadne)", file=sys.stderr)
        print("  --dry-run       stub backends, real file-system flow", file=sys.stderr)
        print("  --mock-ariadne  pure state-machine progression test", file=sys.stderr)
        return 3

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


def cmd_stop(args: argparse.Namespace) -> int:
    campaign = Path(args.campaign_dir).resolve()
    paths = _campaign_paths(campaign)
    if not paths["state"].exists():
        print("no state.json at " + str(paths["state"]) + "; daemon not running?", file=sys.stderr)
        return 4
    try:
        state = read_state(paths["state"])
    except StateSchemaError as exc:
        print("state.json invalid: " + str(exc), file=sys.stderr)
        return 5
    state.shutdown_requested = True
    write_state(paths["state"], state)
    print("shutdown_requested=true set in " + str(paths["state"]))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    campaign = Path(args.campaign_dir).resolve()
    paths = _campaign_paths(campaign)
    if not paths["state"].exists():
        print("no state.json at " + str(paths["state"]), file=sys.stderr)
        return 4
    try:
        state = read_state(paths["state"])
    except StateSchemaError as exc:
        print("state.json invalid: " + str(exc), file=sys.stderr)
        return 5
    payload = state.to_dict()
    payload["state_path"] = str(paths["state"])
    payload["lock_path"] = str(paths["lock"])
    payload.update(_probe_daemon_lock(paths["lock"]))
    payload.update(_probe_daemon_lease(paths["lease"]))
    try:
        from .daemon.artifact_contracts import artifact_manifest_status
        payload["artifact_manifest_status"] = artifact_manifest_status(campaign, state)
    except Exception as exc:
        payload["artifact_manifest_status"] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }
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
    campaign = Path(args.campaign_dir).resolve()
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


def _resolve_terminal_submission_intents_for_apply(
    campaign: Path,
    active_intents: Sequence[Dict[str, Any]],
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
    blocking: List[Dict[str, Any]] = []
    for intent in active_intents:
        phase, iteration = _intent_phase_iteration(intent)
        job_id = str(intent.get("job_id") or "")
        if not phase or iteration < 0:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "reason": "submission intent has malformed phase or iteration",
            })
            continue
        if not job_id:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "reason": "submission intent has no job_id",
            })
            continue
        queue_lookup = sacct_poll.find_active_job_by_id_detailed(job_id)
        if queue_lookup.inconclusive:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
                "reason": "squeue lookup inconclusive: " + str(queue_lookup.error or "unknown error"),
            })
            continue
        if queue_lookup.active:
            blocking.append({
                "phase": phase,
                "iteration": iteration,
                "job_id": job_id,
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


def cmd_reconcile(args: argparse.Namespace) -> int:
    campaign = Path(args.campaign_dir).resolve()
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
    print("Committed model versions:    " + repr(report.committed_model_versions))
    print("Last phase in journal:       " + repr(report.last_phase_in_journal))
    print("Last iteration in journal:   " + repr(report.last_iteration_in_journal))
    if report.unsafe_reasons:
        print("Unsafe recovery reasons:     " + repr(report.unsafe_reasons))
    if report.active_submission_intents:
        print("Active submission intents:   " + repr([
            {
                "phase": i.get("phase"),
                "iteration": i.get("iteration"),
                "status": i.get("status"),
                "job_id": i.get("job_id"),
            }
            for i in report.active_submission_intents
        ]))
    print("")
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

    if report.proposed_state.phase in (CampaignPhase.HALTED, CampaignPhase.DONE):
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
                    + ": "
                    + str(item.get("reason")),
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
    if ".DATA/STAGING is non-empty" in report.unsafe_reasons:
        ok_to_archive_staging, staging_reason = ferebus_reentry_can_archive_data_staging(
            campaign,
            report.proposed_state,
        )
        if ok_to_archive_staging:
            cleanable_reasons.add(".DATA/STAGING is non-empty")
        else:
            report.notes.append(
                ".DATA/STAGING cannot be archived automatically: " + staging_reason
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
        return 9

    backup_path = None
    if target_canonical.exists():
        from datetime import datetime, timezone
        import shutil

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_path = target_canonical.with_name(
            target_canonical.name + ".before-reconcile-" + stamp
        )
        shutil.copy2(target_canonical, backup_path)
    archived = archive_data_staging_for_ferebus_reentry(
        campaign,
        report.proposed_state,
    ) if ".DATA/STAGING is non-empty" in report.unsafe_reasons else []
    removed = clean_reentry_staging(campaign, report.proposed_state.phase)
    target.replace(target_canonical)
    write_state(target_canonical, report.proposed_state)
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
    apply_config_lock_update(campaign, config)
    try:
        from .daemon.journal import append_event

        append_event(
            campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
            "reconcile_applied",
            phase=report.proposed_state.phase.value,
            iteration=int(report.proposed_state.iteration),
            n_allowed_config_changes=(
                len(config_review.allowed_changes) if config_review is not None else 0
            ),
            n_removed_stale_paths=len(removed),
            n_archived_staging_paths=len(archived),
            archived_staging_path=(archived[0] if archived else None),
        )
    except Exception:
        pass
    print("Applied proposed state: " + str(target_canonical))
    if backup_path is not None:
        print("Previous state backup: " + str(backup_path))
    if removed:
        print("Removed stale uncommitted artefacts:")
        for path in removed:
            print("  - " + path)
    if archived:
        print("Archived stale .DATA/STAGING:")
        for path in archived:
            print("  - " + path)
    print("")
    print("Start the daemon with:")
    print("    ichor-al-daemon start --campaign-dir " + str(campaign) + " --live")
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    campaign = Path(args.campaign_dir).resolve()
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


def cmd_import_pool(args: argparse.Namespace) -> int:
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

    campaign = Path(args.campaign_dir).resolve()
    source = Path(args.source).resolve()
    if not campaign.exists():
        print("campaign-dir does not exist: " + str(campaign), file=sys.stderr)
        return 2
    if not source.exists():
        print("source trajectory does not exist: " + str(source), file=sys.stderr)
        return 2

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


def cmd_preflight(args: argparse.Namespace) -> int:
    avail = check_backends()
    print(json.dumps(asdict(avail), indent=2, sort_keys=True))
    if avail.all_present:
        return 0
    print("", file=sys.stderr)
    print(missing_backend_message(avail), file=sys.stderr)
    return 12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ichor-al-daemon",
        description="ICHOR active-learning campaign daemon.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_campaign(p):
        p.add_argument(
            "--campaign-dir", required=True,
            help="Root directory of the campaign.",
        )

    p_start = sub.add_parser("start", help="Start the daemon (foreground).")
    add_campaign(p_start)
    p_start.add_argument(
        "--config", default=None,
        help="Path to campaign.yaml (defaults to <campaign-dir>/campaign.yaml).",
    )
    p_start.add_argument(
        "--mock-ariadne", action="store_true",
        help="Use MockPhaseExecutor for pure state-machine progression tests (no on-disk artefacts).",
    )
    p_start.add_argument(
        "--dry-run", action="store_true",
        help="Use DryRunPhaseExecutor: stub all backend calls but produce real on-disk artefacts (scripts, training-set versions, manifests).",
    )
    p_start.add_argument(
        "--live", action="store_true",
        help="Use LiveBackendsPhaseExecutor against configured Slurm backends (sbatch + Gaussian + AIMAll + FEREBUS + ARIADNE). Refuses with exit 12 if any are missing.",
    )
    p_start.add_argument(
        "--poll-interval", type=int, default=None,
        help="Override poll_interval_seconds from the config.",
    )
    p_start.add_argument(
        "--max-ticks", type=int, default=None,
        help="Limit total tick count (testing / time-boxed runs).",
    )
    p_start.add_argument(
        "--preset", default=None,
        help=(
            "Name of a YAML preset under ichor_hpc/.../presets/ to overlay "
            "campaign.yaml on top of. campaign.yaml wins on every "
            "explicitly-present key."
        ),
    )
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Request a graceful shutdown.")
    add_campaign(p_stop)
    p_stop.set_defaults(func=cmd_stop)

    p_status = sub.add_parser("status", help="Print the current daemon status.")
    add_campaign(p_status)
    p_status.add_argument(
        "--json",
        action="store_true",
        help="Print the raw state/status payload as JSON.",
    )
    p_status.add_argument(
        "--verbose",
        action="store_true",
        help="Include expanded artifact, lease, and path diagnostics.",
    )
    p_status.set_defaults(func=cmd_status)

    p_resume = sub.add_parser(
        "resume",
        help="Equivalent to start (kept for symmetry; works only when state.json exists).",
    )
    add_campaign(p_resume)
    p_resume.add_argument("--config", default=None)
    p_resume.add_argument("--mock-ariadne", action="store_true")
    p_resume.add_argument("--dry-run", action="store_true")
    p_resume.add_argument("--live", action="store_true")
    p_resume.add_argument("--poll-interval", type=int, default=None)
    p_resume.add_argument("--max-ticks", type=int, default=None)
    p_resume.add_argument("--preset", default=None)
    p_resume.set_defaults(func=cmd_resume)

    p_recon = sub.add_parser(
        "reconcile",
        help="Inspect on-disk artefacts and propose a recovered state.",
    )
    add_campaign(p_recon)
    p_recon.add_argument(
        "--allow-fresh-init",
        action="store_true",
        help=(
            "Allow reconcile to propose INIT even when non-state campaign "
            "artifacts are present. Default is conservative HALTED/adoption-ready recovery."
        ),
    )
    p_recon.add_argument(
        "--apply",
        action="store_true",
        help=(
            "After writing state.json.proposed, safely promote it to state.json, "
            "clean stale uncommitted re-entry staging, and update the campaign "
            "config lock. Refuses locked campaign.yaml changes."
        ),
    )
    p_recon.set_defaults(func=cmd_reconcile)

    p_jrn = sub.add_parser("journal", help="Print campaign journal events.")
    add_campaign(p_jrn)
    p_jrn.add_argument(
        "--since", default=None,
        help="ISO timestamp lower bound (inclusive).",
    )
    p_jrn.add_argument(
        "--event-type", action="append", default=None,
        help="Filter to one or more event types (repeatable).",
    )
    p_jrn.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Show only the last N events after filters are applied.",
    )
    p_jrn.add_argument(
        "--json",
        action="store_true",
        help="Print filtered events as NDJSON.",
    )
    p_jrn.add_argument(
        "--raw",
        action="store_true",
        help="Alias for --json; keeps one JSON event per line.",
    )
    p_jrn.add_argument(
        "--verbose",
        action="store_true",
        help="Print expanded key/value details for each event.",
    )
    p_jrn.set_defaults(func=cmd_journal)

    p_imp = sub.add_parser(
        "import-pool",
        help=(
            "Import the MD / metadynamics trajectory into the canonical "
            "per-campaign pool location. Required once per campaign before "
            "the daemon can run; refuses to overwrite an existing pool."
        ),
    )
    add_campaign(p_imp)
    p_imp.add_argument(
        "--source", required=True,
        help="Path to the operator's MD trajectory (.xyz). Will be copied into "
             "<campaign-dir>/.DATA/TRAJECTORY/pool.xyz.",
    )
    p_imp.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing pool (DANGEROUS: invalidates every committed "
             "iteration's frame-id provenance).",
    )
    p_imp.add_argument(
        "--no-outlier-filter", action="store_true", dest="no_outlier_filter",
        help="Skip the pre-Phase-A outlier filter; import every frame verbatim. "
             "Default behaviour (filter ON) rejects per-atom z > 4 frames and "
             "writes rejected.json alongside pool.manifest.json.",
    )
    p_imp.set_defaults(func=cmd_import_pool)

    p_pre = sub.add_parser("preflight", help="Check configured live Slurm backends.")
    p_pre.add_argument(
        "--campaign-dir",
        default=".",
        help="Accepted for symmetry with other commands; not used by preflight.",
    )
    p_pre.set_defaults(func=cmd_preflight)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
