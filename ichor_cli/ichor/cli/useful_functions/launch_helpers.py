"""Cross-platform launchers used by the Active Learning Campaign menu.

The menu drives the active-learning daemon (``ichor-al-daemon``) in two
modes: foreground (which blocks the menu via a direct ``Daemon.run()``
call) and background (a detached child process so the menu stays usable).
This module owns the background-launch helper.

The detached child is wired up so it survives the menu process exiting:

* POSIX (Linux / macOS / CSF4 production target):
  ``subprocess.Popen(..., start_new_session=True, stdin=DEVNULL,
                     stdout=<log>, stderr=STDOUT, close_fds=True)``.
* Windows (developer testing only):
  ``creationflags = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS``. Windows
  detach semantics differ from POSIX; the child may still hold a console
  handle in odd terminal hosts. Production is Linux.

The PID is written to ``<campaign_dir>/.DATA/ACTIVE_LEARNING/menu_launched.pid``
so the menu can echo it back and the user can correlate the running
daemon with the menu invocation. The daemon's own ``daemon.lock`` flock
remains the source of truth for "is the daemon alive".
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ichor.hpc.active_learning.daemon.background_startup import (
    BACKGROUND_LAUNCH_ID_ENV,
    BACKGROUND_READINESS_ENV,
    BACKGROUND_STARTUP_ACKNOWLEDGED_STATES,
    BACKGROUND_STARTUP_FILENAME,
    BACKGROUND_STARTUP_PATH_ENV,
    initialise_background_startup,
    read_background_startup,
    update_background_startup,
)


__all__ = [
    "MENU_LAUNCHED_PID_FILENAME",
    "MENU_LAUNCHED_LOG_FILENAME",
    "DetachedLaunchResult",
    "build_daemon_argv",
    "launch_daemon_detached",
    "launch_daemon_detached_checked",
]


MENU_LAUNCHED_PID_FILENAME = "menu_launched.pid"
MENU_LAUNCHED_LOG_FILENAME = "daemon.menu_launched.out"


@dataclass(frozen=True)
class DetachedLaunchResult:
    pid: int
    argv: List[str]
    log_path: Path
    pid_path: Path
    returncode: Optional[int]
    exited_during_startup: bool
    ready: bool = False
    readiness_error: Optional[str] = None
    acknowledged: bool = False
    startup_pending: bool = False
    startup_state: Optional[str] = None
    control_completed: bool = False


def build_daemon_argv(
    campaign_dir: Path,
    *,
    mode: str,
    command: str = "start",
    poll_interval: Optional[int] = None,
    max_ticks: Optional[int] = None,
    config: Optional[Path] = None,
    reopen_converged: bool = False,
    cancel_stop_request: bool = False,
) -> List[str]:
    """Construct the daemon CLI argv for ``start`` or ``resume``."""
    if command not in ("start", "resume"):
        raise ValueError("command must be one of: start / resume; got " + repr(command))
    if mode not in ("dry_run", "live"):
        raise ValueError(
            "mode must be one of: dry_run / live; got " + repr(mode)
        )
    if reopen_converged and command != "resume":
        raise ValueError("reopen_converged is valid only with the resume command")
    if cancel_stop_request and command != "resume":
        raise ValueError("cancel_stop_request is valid only with the resume command")
    argv: List[str] = [
        sys.executable,
        "-m", "ichor.hpc.active_learning.cli",
        command,
        "--campaign-dir", str(campaign_dir),
        "--mode", mode,
    ]
    if config is not None:
        argv += ["--config", str(config)]
    if poll_interval is not None:
        argv += ["--poll-interval", str(int(poll_interval))]
    if max_ticks is not None:
        argv += ["--max-ticks", str(int(max_ticks))]
    if reopen_converged:
        argv.append("--reopen-converged")
    if cancel_stop_request:
        argv.append("--cancel-stop-request")
    return argv


def _popen_detached(
    argv: List[str],
    log_path: Path,
    *,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    log_fh = open(log_path, "ab")
    try:
        if os.name == "posix":
            popen_kwargs = {
                "stdin": subprocess.DEVNULL,
                "stdout": log_fh,
                "stderr": subprocess.STDOUT,
                "close_fds": True,
                "start_new_session": True,
            }
        elif os.name == "nt":
            popen_kwargs = {
                "stdin": subprocess.DEVNULL,
                "stdout": log_fh,
                "stderr": subprocess.STDOUT,
                "creationflags": (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | getattr(subprocess, "DETACHED_PROCESS", 0)
                ),
                "close_fds": False,
            }
        else:
            popen_kwargs = {
                "stdin": subprocess.DEVNULL,
                "stdout": log_fh,
                "stderr": subprocess.STDOUT,
                "close_fds": True,
            }
        return subprocess.Popen(argv, env=env, **popen_kwargs)
    finally:
        log_fh.close()


def launch_daemon_detached_checked(
    campaign_dir: Path,
    *,
    mode: str,
    command: str = "start",
    poll_interval: Optional[int] = None,
    max_ticks: Optional[int] = None,
    config: Optional[Path] = None,
    reopen_converged: bool = False,
    cancel_stop_request: bool = False,
    log_path: Optional[Path] = None,
    pid_path: Optional[Path] = None,
    startup_timeout_seconds: float = 60.0,
) -> DetachedLaunchResult:
    """Spawn the daemon and briefly wait for a durable startup acknowledgement.

    The child's stdout + stderr are appended to
    ``<campaign_dir>/.DATA/ACTIVE_LEARNING/daemon.menu_launched.out``; its
    PID is recorded in ``menu_launched.pid`` next to it.  Exhausting the wait
    budget returns a pending result and leaves a live child running.

    The menu must have already validated that ``campaign_dir`` exists and
    that a ``campaign.yaml`` is in place; this helper does the bare
    minimum and propagates any OS errors to the caller.
    """
    campaign_dir = Path(campaign_dir).resolve()
    data_dir = campaign_dir / ".DATA" / "ACTIVE_LEARNING"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path).expanduser().resolve() if log_path else (
        data_dir / MENU_LAUNCHED_LOG_FILENAME
    )
    pid_path = Path(pid_path).expanduser().resolve() if pid_path else (
        data_dir / MENU_LAUNCHED_PID_FILENAME
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    argv = build_daemon_argv(
        campaign_dir,
        mode=mode,
        command=command,
        poll_interval=poll_interval,
        max_ticks=max_ticks,
        config=config,
        reopen_converged=bool(reopen_converged),
        cancel_stop_request=bool(cancel_stop_request),
    )
    # This helper has already detached the process. Run the daemon itself in
    # foreground mode so it does not create an untracked grandchild.
    argv.append("--foreground")

    startup_path = data_dir / BACKGROUND_STARTUP_FILENAME
    launch_id = secrets.token_hex(16)
    initialise_background_startup(
        startup_path,
        {
            "launch_id": launch_id,
            "state": "prepared",
            "stage": "menu_launcher",
            "campaign_dir": str(campaign_dir),
            "command": argv,
            "log_path": str(log_path),
            "launcher_pid": int(os.getpid()),
            "python_executable": sys.executable,
        },
    )
    env = os.environ.copy()
    env[BACKGROUND_LAUNCH_ID_ENV] = launch_id
    env[BACKGROUND_STARTUP_PATH_ENV] = str(startup_path)
    env[BACKGROUND_READINESS_ENV] = str(startup_path)
    try:
        child = _popen_detached(argv, log_path, env=env)
    except Exception as exc:
        update_background_startup(
            startup_path,
            launch_id,
            state="failed",
            stage="process_spawn",
            failure=type(exc).__name__ + ": " + str(exc),
        )
        raise
    update_background_startup(
        startup_path,
        launch_id,
        state="spawned",
        stage="child_process",
        pid=int(child.pid),
    )

    temporary_pid_path = pid_path.with_name(
        pid_path.name + ".tmp." + str(os.getpid()) + "." + secrets.token_hex(4)
    )
    temporary_pid_path.write_text(str(child.pid) + "\n", encoding="utf-8")
    os.replace(temporary_pid_path, pid_path)

    deadline = time.monotonic() + max(0.0, float(startup_timeout_seconds))
    returncode = child.poll()
    startup_payload = read_background_startup(startup_path)
    acknowledged = False
    while returncode is None and time.monotonic() <= deadline:
        startup_payload = read_background_startup(startup_path)
        if str(startup_payload.get("launch_id") or "") == launch_id:
            state = str(startup_payload.get("state") or "")
            if state in BACKGROUND_STARTUP_ACKNOWLEDGED_STATES:
                acknowledged = True
                break
            if state == "stopped" and startup_payload.get("ready_at_iso"):
                acknowledged = True
                break
            if state == "failed":
                break
        time.sleep(0.1)
        returncode = child.poll()

    startup_payload = read_background_startup(startup_path)
    startup_state = str(startup_payload.get("state") or "") or None
    ready = bool(startup_payload.get("ready_at_iso")) or startup_state == "ready"
    control_completed = bool(
        command == "resume"
        and cancel_stop_request
        and returncode == 0
        and not acknowledged
    )
    if control_completed:
        update_background_startup(
            startup_path,
            launch_id,
            state="stopped",
            stage="control_completed",
            exit_code=0,
        )
        startup_state = "stopped"
        acknowledged = True
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass
    readiness_error = None
    if startup_state == "failed":
        readiness_error = str(
            startup_payload.get("failure") or "daemon startup failed"
        )
    elif returncode is not None and not acknowledged:
        readiness_error = "daemon exited before startup acknowledgement"
        try:
            update_background_startup(
                startup_path,
                launch_id,
                state="failed",
                stage="process_exit",
                exit_code=int(returncode),
                failure=readiness_error,
            )
        except Exception:
            pass
        startup_state = "failed"
    elif not acknowledged:
        readiness_error = (
            "startup acknowledgement wait budget elapsed; child continues running"
        )
    return DetachedLaunchResult(
        pid=child.pid,
        argv=argv,
        log_path=log_path,
        pid_path=pid_path,
        returncode=returncode,
        exited_during_startup=(
            (returncode is not None and not acknowledged)
            or startup_state == "failed"
        ),
        ready=ready,
        readiness_error=readiness_error,
        acknowledged=acknowledged,
        startup_pending=(
            returncode is None and not acknowledged and startup_state != "failed"
        ),
        startup_state=startup_state,
        control_completed=control_completed,
    )


def launch_daemon_detached(
    campaign_dir: Path,
    *,
    mode: str,
    poll_interval: Optional[int] = None,
    config: Optional[Path] = None,
) -> int:
    """Backward-compatible PID-only detached launch helper."""
    result = launch_daemon_detached_checked(
        campaign_dir,
        mode=mode,
        poll_interval=poll_interval,
        config=config,
        startup_timeout_seconds=60.0,
    )
    if result.exited_during_startup:
        raise RuntimeError(result.readiness_error or "daemon startup failed")
    return result.pid
