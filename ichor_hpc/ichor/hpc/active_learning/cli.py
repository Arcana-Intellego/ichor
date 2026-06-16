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
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

from .config import CampaignConfig
from .daemon.daemon import (
    DAEMON_HEARTBEAT_FILENAME,
    DAEMON_LEASE_DIRNAME,
    DAEMON_LOCK_FILENAME,
    DEFAULT_DATA_SUBDIR,
    Daemon,
)
from .daemon.journal import iter_events, read_events
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
from .daemon.state import (
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
    print(json.dumps(payload, indent=2, sort_keys=True))
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
        if state.shutdown_requested:
            state.shutdown_requested = False
            write_state(state_path, state)
            print("shutdown_requested=false set in " + str(state_path))
    return cmd_start(args)


def cmd_reconcile(args: argparse.Namespace) -> int:
    campaign = Path(args.campaign_dir).resolve()
    report = propose_recovery(
        campaign,
        allow_fresh_init_on_nonempty=bool(getattr(args, "allow_fresh_init", False)),
    )
    target = write_proposed_state(campaign, report)
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
    print("Review the proposal, then promote it manually:")
    target_canonical = target.with_name(DEFAULT_STATE_FILENAME)
    print("    mv " + str(target) + " " + str(target_canonical))
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
    for event in iterator:
        print(json.dumps(event, sort_keys=True))
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

    p_status = sub.add_parser("status", help="Print state.json as JSON.")
    add_campaign(p_status)
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
