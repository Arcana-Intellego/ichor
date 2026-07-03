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
so the menu can echo it back and the operator can correlate the running
daemon with the menu invocation. The daemon's own ``daemon.lock`` flock
remains the source of truth for "is the daemon alive".
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


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


def build_daemon_argv(
    campaign_dir: Path,
    *,
    mode: str,
    command: str = "start",
    poll_interval: Optional[int] = None,
    max_ticks: Optional[int] = None,
    config: Optional[Path] = None,
    preset: Optional[str] = None,
) -> List[str]:
    """Construct the daemon CLI argv for ``start`` or ``resume``."""
    if command not in ("start", "resume"):
        raise ValueError("command must be one of: start / resume; got " + repr(command))
    if mode not in ("mock-ariadne", "dry-run", "live"):
        raise ValueError(
            "mode must be one of: mock-ariadne / dry-run / live; got " + repr(mode)
        )
    argv: List[str] = [
        sys.executable,
        "-m", "ichor.hpc.active_learning.cli",
        command,
        "--campaign-dir", str(campaign_dir),
        "--" + mode,
    ]
    if config is not None:
        argv += ["--config", str(config)]
    if preset is not None:
        argv += ["--preset", str(preset)]
    if poll_interval is not None:
        argv += ["--poll-interval", str(int(poll_interval))]
    if max_ticks is not None:
        argv += ["--max-ticks", str(int(max_ticks))]
    return argv


def _popen_detached(argv: List[str], log_path: Path) -> subprocess.Popen:
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
        return subprocess.Popen(argv, **popen_kwargs)
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
    preset: Optional[str] = None,
    log_path: Optional[Path] = None,
    pid_path: Optional[Path] = None,
    startup_grace_seconds: float = 0.25,
) -> DetachedLaunchResult:
    """Spawn the daemon as a detached child and report immediate startup exit.

    The child's stdout + stderr are appended to
    ``<campaign_dir>/.DATA/ACTIVE_LEARNING/daemon.menu_launched.out``; its
    PID is recorded in ``menu_launched.pid`` next to it.

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
        preset=preset,
    )

    child = _popen_detached(argv, log_path)

    pid_path.write_text(str(child.pid) + "\n", encoding="utf-8")
    if startup_grace_seconds > 0:
        time.sleep(float(startup_grace_seconds))
    returncode = child.poll()
    return DetachedLaunchResult(
        pid=child.pid,
        argv=argv,
        log_path=log_path,
        pid_path=pid_path,
        returncode=returncode,
        exited_during_startup=(returncode is not None),
    )


def launch_daemon_detached(
    campaign_dir: Path,
    *,
    mode: str,
    poll_interval: Optional[int] = None,
    config: Optional[Path] = None,
) -> int:
    """Backward-compatible PID-only detached launch helper."""
    return launch_daemon_detached_checked(
        campaign_dir,
        mode=mode,
        poll_interval=poll_interval,
        config=config,
        startup_grace_seconds=0.0,
    ).pid
