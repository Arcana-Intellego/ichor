"""Tests for ichor.hpc.active_learning.cli."""
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning import cli as cli_mod
from ichor.hpc.active_learning.cli import build_parser, main
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import (
    DAEMON_LOCK_FILENAME,
    DEFAULT_DATA_SUBDIR,
)
from ichor.hpc.active_learning.daemon import submission_intent
from ichor.hpc.active_learning.daemon.config_lock import write_config_lock
from ichor.hpc.active_learning.daemon.job_names import live_job_name
from ichor.hpc.active_learning.daemon.journal import (
    KNOWN_EVENT_TYPES,
    append_event,
    iter_events,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    CampaignState,
    DEFAULT_STATE_FILENAME,
    fresh_campaign_state,
    make_lifecycle_context,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.daemon.status_recommendations import (
    build_status_recommendations,
    recommendation_dicts,
)
from ichor.hpc.active_learning.daemon.stop_control import (
    StopControlError,
    archive_and_clear_stop_request,
    build_stop_request,
    complete_stop_request,
    install_stop_request,
    read_stop_request,
    stop_request_history_dir,
    stop_request_path,
    update_stop_request,
)
from ichor.hpc.active_learning.handoff_manifests import (
    ARIADNE_RESULTS_SCHEMA_VERSION,
    write_ariadne_batch_decision,
    write_ariadne_results_manifest,
)
from ichor.hpc.active_learning.versioning.provenance import write_seed_provenance


def test_user_facing_cli_contracts_do_not_use_operator_terminology():
    repo_root = Path(__file__).resolve().parents[3]
    paths = [
        repo_root / "ichor_hpc" / "ichor" / "hpc" / "active_learning" / "cli.py",
        repo_root
        / "ichor_hpc"
        / "ichor"
        / "hpc"
        / "active_learning"
        / "daemon"
        / "journal.py",
        repo_root
        / "ichor_hpc"
        / "ichor"
        / "hpc"
        / "active_learning"
        / "daemon"
        / "reconcile.py",
        repo_root
        / "ichor_hpc"
        / "ichor"
        / "hpc"
        / "active_learning"
        / "daemon"
        / "status_recommendations.py",
        repo_root / "docs" / "source" / "active_learning_daemon.rst",
        repo_root / "scripts" / "install_ichor_csf.sh",
        repo_root / "scripts" / "upsert_ichor_config.py",
    ]
    for path in paths:
        assert (
            re.search(r"\boperator\b", path.read_text(encoding="utf-8"), re.I)
            is None
        ), str(path)
    assert not any(name.startswith("operator_") for name in KNOWN_EVENT_TYPES)


def _campaign_with_config(tmp_path) -> Path:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    CampaignConfig(max_iterations=2).to_yaml(campaign / "campaign.yaml")
    (campaign / "pool.xyz").write_text(
        "".join(
            "1\nframe " + str(index) + "\nH 0.0 0.0 0.0\n"
            for index in range(128)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return campaign


def _write_locked_state(campaign: Path, state) -> None:
    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(state_path, state)
    write_config_lock(
        campaign,
        CampaignConfig.from_yaml(campaign / "campaign.yaml"),
        campaign_uid=str(state.campaign_uid),
    )


def _commit_training_and_model_versions(campaign: Path, versions):
    if list(versions) != [0]:
        raise ValueError("CLI fixture helper currently supports version 0 only")
    from ichor.hpc.active_learning.daemon import input_staging as stg
    from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
    from ichor.hpc.active_learning.point_allocation import (
        create_point_allocation,
        pending_attempts,
        point_allocation_path,
        record_quantum_results,
    )
    from ichor.hpc.active_learning.handoff_manifests import (
        write_phase_a_sample_manifest,
    )
    from ichor.hpc.active_learning.sampling.diversity_contract import (
        diversity_selector_contract,
    )
    from ichor.hpc.active_learning.layout import bootstrap_selection_dir
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
    from ichor.hpc.active_learning.versioning.provenance import (
        enrich_with_point_allocation,
    )

    pool = TrajectoryPool.import_from(
        campaign / "pool.xyz",
        campaign,
        overwrite=True,
    )
    allocation_path = point_allocation_path(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    candidates = [
        {
            "candidate_id": "cli-bootstrap-" + str(index),
            "frame_id": index,
            "pointdir_name": "POINT_" + str(index).zfill(4) + ".pointdir",
        }
        for index in range(6)
    ]
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="cli-test",
        context="bootstrap",
        iteration=0,
        targets={"train": 2, "int_val": 2, "ext_val": 2, "total": 6},
        primary_candidates=candidates,
        reserve_candidates=[],
    )
    attempts = pending_attempts(allocation)
    selection_dir = bootstrap_selection_dir(campaign)
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "selected.xyz").write_text(
        "".join(
            "1\nbootstrap fixture "
            + str(index)
            + "\nH "
            + repr(0.01 * index)
            + " 0.0 0.0\n"
            for index in range(6)
        ),
        encoding="utf-8",
        newline="\n",
    )
    (selection_dir / "selected_indices.dat").write_text(
        "".join(str(index) + "\n" for index in range(6)),
        encoding="utf-8",
        newline="\n",
    )
    write_phase_a_sample_manifest(selection_dir, {
        "phase": "PHASE_A_DIVERSITY",
        "iteration": 0,
        "sample_xyz": "selection/selected.xyz",
        "index_path": "selection/selected_indices.dat",
        "n_select": 6,
        "selected_indices": list(range(6)),
        "selector": diversity_selector_contract(),
        "trajectory_sha256": str(pool.sha256),
        "source_pool_manifest": ".DATA/TRAJECTORY/pool.manifest.json",
        "point_allocation": {
            "manifest": "allocation/POINT_ALLOCATION.json",
            "primary": [
                {
                    "candidate_id": str(attempt["candidate_id"]),
                    "frame_id": int(attempt["frame_id"]),
                    "slot_id": int(attempt["slot_id"]),
                    "split": str(attempt["split"]),
                }
                for attempt in sorted(
                    attempts,
                    key=lambda record: int(record["frame_id"]),
                )
            ],
        },
    })
    pointdirs = []
    for attempt in attempts:
        pointdir = (
            campaign
            / ".DATA"
            / "STAGING"
            / "initial"
            / str(attempt["pointdir_name"])
        )
        pointdir.mkdir(parents=True, exist_ok=True)
        (pointdir / "fixture.txt").write_text("reference\n", encoding="utf-8")
        write_seed_provenance(
            pointdir,
            campaign_uid="cli-test",
            iteration=0,
            trajectory_sha256=str(pool.sha256),
            seed_frame_id=int(attempt["frame_id"]),
            seed_selection_origin="cli_fixture",
            seed_variance_at_selection=None,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
        )
        enrich_with_point_allocation(
            pointdir,
            candidate_id=str(attempt["candidate_id"]),
            context="bootstrap",
            slot_id=int(attempt["slot_id"]),
            split=str(attempt["split"]),
            allocation_slot_assignment_sha256=str(
                allocation["slot_assignment_sha256"]
            ),
        )
        pointdirs.append(pointdir)
    from ichor_hpc.tests.quantum_test_support import (
        attach_synthetic_quantum_batch,
    )

    results = [
        {
            "candidate_id": str(attempt["candidate_id"]),
            "accepted": True,
            "pointdir": str(pointdir.resolve()),
        }
        for attempt, pointdir in zip(attempts, pointdirs)
    ]
    attach_synthetic_quantum_batch(
        campaign,
        results,
        phase_name=CampaignPhase.INITIAL_AIMALL.value,
        iteration=0,
    )
    record_quantum_results(
        allocation_path,
        results,
    )
    stg.commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    DryRunPhaseExecutor(campaign, config)._commit_dry_model_snapshot(0)
    from ichor.hpc.active_learning.versioning.sampling_iterations import (
        finalise_bootstrap,
    )

    finalise_bootstrap(campaign, "cli-test")
    import shutil

    shutil.rmtree(campaign / "TRAINED_MODELS" / "iteration-staging")
    shutil.rmtree(campaign / ".DATA" / "SCRIPTS")
    shutil.rmtree(
        campaign / ".DATA" / "STAGING" / "initial",
        ignore_errors=True,
    )


def _write_valid_ariadne_results(campaign: Path, iteration: int = 1):
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
    from ichor.hpc.active_learning.ariadne_outputs import (
        write_optimisation_trajectory,
        write_seed_output_manifest,
    )
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.handoff_manifests import (
        build_seed_selection_manifest,
        seeds_picked_path,
    )
    from ichor.hpc.active_learning.layout import (
        active_ariadne_dir,
        active_iteration_dir,
        ariadne_seed_dir,
    )
    from ichor.hpc.active_learning.seed_identity import write_ariadne_task_map
    from ichor.hpc.active_learning.versioning.manifest import sha256_file

    try:
        pool = TrajectoryPool.load(campaign)
    except FileNotFoundError:
        source = campaign / "pool.xyz"
        if not source.is_file():
            source.write_text(
                "1\nreconcile fixture frame\nH 0.0 0.0 0.0\n",
                encoding="utf-8",
                newline="\n",
            )
        pool = TrajectoryPool.import_from(source, campaign)
    trajectory_sha = str(pool.sha256)
    iter_dir = active_iteration_dir(campaign, iteration)
    selection = build_seed_selection_manifest(
        campaign_uid="cli-test",
        campaign_random_seed=0,
        iteration=int(iteration),
        models_version=0,
        model_manifest_sha256="0" * 64,
        model_set_sha256="1" * 64,
        trajectory_sha256=trajectory_sha,
        selection_strategy="hybrid_variance",
        seed_records=[{
            "seed_id": 1,
            "frame_id": 0,
            "pool_row_index_zero_based": 0,
            "selection_origin": "bulk",
            "variance_at_selection": 0.0,
        }],
    )
    seed_uid = str(selection["seed_records"][0]["seed_uid"])
    selection_path = seeds_picked_path(iter_dir)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, selection)
    task_map_path = write_ariadne_task_map(iter_dir, selection)

    seed_dir = ariadne_seed_dir(iter_dir, 1)
    seed_dir.mkdir(parents=True, exist_ok=True)
    result_path = seed_dir / "result.json"
    landing_safety = {
        "accepted": True,
        "policy": "raw_final",
        "selected_origin": "raw_final",
        "selected_candidate_index": 0,
        "reasons": [],
        "record_only_reasons": [],
        "metrics": {"whitened_distance": 0.5},
        "raw_final": {},
        "n_candidates_evaluated": 1,
        "n_safe_candidates": 1,
    }
    atomic_write_json(result_path, {
        "iteration": int(iteration),
        "seed_id": 1,
        "seed_uid": seed_uid,
        "array_task_id": 0,
        "seed_frame_id": 0,
        "trajectory_sha256": trajectory_sha,
        "atom_types": ["H"],
        "final_coordinates": [[0.0, 0.0, 0.0]],
        "alpha_trajectory": [0.0, 1.0],
        "alpha_initial": 0.0,
        "alpha_final": 1.0,
        "n_evaluations": 2,
        "return_code": 0,
        "wall_seconds": 1.0,
        "fell_back_to_ds": False,
        "whitened_distance_final": 0.5,
        "landing_safety": landing_safety,
    })
    write_optimisation_trajectory(
        seed_dir,
        atom_types=["H"],
        coordinate_frames=[[[0.0, 0.0, 0.0]]],
        alpha_values=[1.0],
        gradient_norms=[0.0],
        origins=["raw_final"],
    )
    output_manifest = write_seed_output_manifest(
        seed_dir,
        campaign_uid="cli-test",
        iteration=iteration,
        seed_id=1,
        seed_uid=seed_uid,
        array_task_id=0,
        task_success=True,
        task_exit_code=0,
    )
    prov_path = write_seed_provenance(
        seed_dir,
        campaign_uid="cli-test",
        iteration=int(iteration),
        trajectory_sha256=trajectory_sha,
        seed_frame_id=0,
        seed_id=1,
        seed_uid=seed_uid,
        array_task_id_zero_based=0,
        seed_selection_origin="bulk",
        seed_variance_at_selection=0.0,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
        mode_weighting_policy="variance",
    )
    ariadne_root = active_ariadne_dir(iter_dir)
    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "campaign_uid": "cli-test",
        "iteration": int(iteration),
        "trajectory_sha256": trajectory_sha,
        "expected_n": 1,
        "n_accepted": 1,
        "n_rejected": 0,
        "task_map": {
            "path": task_map_path.relative_to(ariadne_root).as_posix(),
            "sha256": sha256_file(task_map_path),
        },
        "accepted": [{
            "seed_id": 1,
            "seed_uid": seed_uid,
            "array_task_id": 0,
            "seed_frame_id": 0,
            "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
            "result_json": result_path.relative_to(ariadne_root).as_posix(),
            "provenance_json": Path(prov_path).relative_to(ariadne_root).as_posix(),
            "output_manifest": output_manifest.relative_to(ariadne_root).as_posix(),
            "return_code": 0,
            "landing_safety": landing_safety,
            "result_sha256": sha256_file(result_path),
            "provenance_sha256": sha256_file(prov_path),
            "output_manifest_sha256": sha256_file(output_manifest),
        }],
        "rejected": [],
    })
    from ichor.hpc.active_learning.daemon.config_lock import (
        canonical_config,
        config_fingerprint,
    )

    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    write_ariadne_batch_decision(
        iter_dir,
        campaign_uid="cli-test",
        iteration=int(iteration),
        config_sha256=config_fingerprint(canonical_config(config)),
        failure_threshold_fraction=float(config.runtime.failure_threshold_fraction),
        expected_n=1,
        n_accepted=1,
        n_rejected=0,
        accepted=True,
        reasons=[],
    )
    return iter_dir


def _backend_availability(**overrides):
    from ichor.hpc.active_learning.daemon.preflight import BackendAvailability

    payload = {
        "profile": True,
        "sbatch": True,
        "sacct": True,
        "squeue": True,
        "batch_python": True,
        "batch_python_version": "3.11.15",
        "batch_python_error": "",
        "gaussian": True,
        "gaussian_verified": True,
        "gaussian_probe_error": "",
        "aimall": True,
        "ferebus": True,
        "ariadne": True,
        "pyferebus": True,
        "bc": True,
        "gaussian_binary": "jobscript:$g16root/g16/g16",
        "sbatch_path": "/usr/bin/sbatch",
        "sacct_path": "/usr/bin/sacct",
        "squeue_path": "/usr/bin/squeue",
        "bc_path": "/usr/bin/bc",
        "aimall_path": "/home/user/AIMAll/aimqb.ish",
        "ferebus_path": "/home/user/.local/bin/ferebus",
        "active_profile": "csf3",
        "profile_error": "",
        "python_executable": "/home/user/.venv/ichor-csf3/bin/python",
    }
    payload.update(overrides)
    return BackendAvailability(**payload)


def _pool_feasibility_payload(
    *,
    ok: bool = True,
    pool_n_frames: int = 1700,
    required_pool_frames: int = 90,
) -> dict:
    return {
        "pool_n_frames": int(pool_n_frames),
        "bootstrap_total_size": 70,
        "bootstrap_pool_frame_count": 70,
        "bootstrap_custom_count": 0,
        "bootstrap_model_training_count": 0,
        "excluded_pool_frame_count": 0,
        "max_iterations": 2,
        "n_seeds_per_iteration": 10,
        "batch_total_size": 10,
        "configured_seed_surplus": 0,
        "exclude_committed_seed_frames": True,
        "required_pool_frames": int(required_pool_frames),
        "reserve_after_bootstrap": int(pool_n_frames) - 70,
        "expression": (
            "bootstrap pool frames after custom inputs + "
            "max_iterations * seed_selection.n_seeds_per_iteration = 70 + 2 * 10 = 90"
        ),
        "ok": bool(ok),
    }


def test_build_parser_has_all_subcommands():
    p = build_parser()
    # Parse a known subcommand to confirm registration.
    args = p.parse_args(
        ["start", "--campaign-dir", "x", "--mode", "dry_run"]
    )
    assert args.command == "start"
    assert args.mode == "dry_run"


def test_parser_rejects_missing_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_config_check_default_and_explicit_json_are_compatible(
    tmp_path,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)

    default_rc = main(["config-check", "--campaign-dir", str(campaign)])
    default_output = capsys.readouterr().out
    explicit_rc = main(
        ["config-check", "--campaign-dir", str(campaign), "--json"]
    )
    explicit_output = capsys.readouterr().out

    assert explicit_rc == default_rc
    assert explicit_output == default_output
    assert isinstance(json.loads(default_output), dict)


def test_config_check_human_separates_config_and_pool_results(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)

    rc = main(["config-check", "--campaign-dir", str(campaign), "--human"])

    assert rc == 10
    out = capsys.readouterr().out
    assert out.startswith("Config check\nConfig\n  result: valid\n")
    assert "\nPool\n  result: insufficient or unavailable\n" in out
    assert "\nAction\n" in out
    assert not out.lstrip().startswith("{")


def test_config_check_human_reports_invalid_yaml_as_a_config_failure(
    tmp_path,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    (campaign / "campaign.yaml").write_text("campaign: [\n", encoding="utf-8")

    rc = main(["config-check", "--campaign-dir", str(campaign), "--human"])

    assert rc == 2
    out = capsys.readouterr().out
    assert "Config\n  result: invalid" in out
    assert "Pool\n  result: not checked" in out
    assert "config-check --campaign-dir" in out


def test_cli_preflight_prints_operator_dashboard_by_default(tmp_path, capsys, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state(max_iterations=2))
    monkeypatch.setattr(cli_mod, "check_backends", _backend_availability)
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(),
    )

    rc = main(["preflight", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Preflight\n" in out
    assert "  result: ready" in out
    assert "  active profile: csf3" in out
    assert "Scheduler\n" in out
    assert "  [OK] sbatch: /usr/bin/sbatch" in out
    assert "Python Environment\n" in out
    assert "  [OK] ariadne: importable" in out
    assert "Quantum Backends\n" in out
    assert "  [OK] Gaussian submitted environment: jobscript:$g16root/g16/g16" in out
    assert "Trajectory Pool\n" in out
    assert "  [OK] frames available: 1700" in out
    assert "  [OK] frames required: 90" in out
    assert "Next Action\n" in out
    assert "ichor-al-daemon start --campaign-dir " in out
    assert "submitted module stack" not in out
    assert "schema version" not in out
    assert "FEREBUS prior contract" not in out
    assert "requirement:" not in out
    assert not out.lstrip().startswith("{")


def test_campaign_preflight_allows_and_reports_pending_boundary_stop(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=2)
    _write_locked_state(campaign, state)
    install_stop_request(
        campaign,
        build_stop_request(state, mode="after_iteration"),
    )
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(),
    )

    payload = cli_mod.evaluate_campaign_preflight(
        campaign,
        avail=_backend_availability(),
    )

    assert payload["ready"] is True, json.dumps(payload, default=str, indent=2)
    assert payload["campaign_state"]["ok"] is True
    assert payload["campaign_state"]["condition"] == "stop_scheduled"
    assert payload["campaign_state"]["issues"] == []
    assert "warnings" not in payload["campaign_state"]
    output = cli_mod._format_preflight(payload)
    assert "[WARN] stop boundary" in output
    assert "stop requested after iteration 0" in output
    assert "retain and honour this request" in output


def test_campaign_preflight_validates_every_slurm_phase_resource_contract(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import resource_solver
    from ichor.hpc.active_learning.daemon.phase_executor import SBATCH_PHASES

    campaign = _campaign_with_config(tmp_path)
    _write_locked_state(campaign, fresh_campaign_state(max_iterations=2))
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(),
    )
    supported = []
    walltimes = []
    monkeypatch.setattr(
        resource_solver,
        "validate_partition_supported",
        lambda partition: supported.append(str(partition)),
    )
    monkeypatch.setattr(
        resource_solver,
        "validate_partition_walltime",
        lambda partition, hours: walltimes.append(
            (str(partition), float(hours))
        ),
    )

    payload = cli_mod.evaluate_campaign_preflight(
        campaign,
        avail=_backend_availability(),
    )

    assert payload["campaign_config"]["ok"] is True
    assert len(supported) == len(SBATCH_PHASES)
    assert len(walltimes) == len(SBATCH_PHASES)
    assert set(payload["campaign_config"]["resource_profile"]) == set(
        SBATCH_PHASES
    )


def test_campaign_preflight_uses_checkpoint_authority_not_payload_verification(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import checkpoints

    campaign = _campaign_with_config(tmp_path)
    destination = tmp_path / "checkpoint-store"
    destination.mkdir()
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    config.retention.checkpoint_destination = str(destination)
    config.retention.checkpoint_required = True
    config.to_yaml(campaign / "campaign.yaml")
    state = fresh_campaign_state(max_iterations=2)
    _write_locked_state(campaign, state)
    store = checkpoints.checkpoint_store(destination, str(state.campaign_uid))
    store.mkdir(parents=True)
    (store / "current.json").write_text("{}\n", encoding="utf-8")
    authority_calls = []
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(),
    )
    monkeypatch.setattr(
        checkpoints,
        "checkpoint_authority_status",
        lambda *args, **kwargs: authority_calls.append((args, kwargs))
        or {"status": "authority_verified"},
    )
    monkeypatch.setattr(
        checkpoints,
        "checkpoint_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight invoked deep checkpoint verification")
        ),
    )

    payload = cli_mod.evaluate_campaign_preflight(
        campaign,
        avail=_backend_availability(),
    )

    assert payload["ready"] is True
    assert len(authority_calls) == 1


def test_cli_preflight_json_prints_single_payload(tmp_path, capsys, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state(max_iterations=2))
    monkeypatch.setattr(cli_mod, "check_backends", _backend_availability)
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(),
    )

    rc = main(["preflight", "--campaign-dir", str(campaign), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["active_profile"] == "csf3"
    assert payload["python_executable"].endswith("ichor-csf3/bin/python")
    assert payload["backend_availability"]["ariadne"] is True
    assert payload["pool_feasibility"]["pool_n_frames"] == 1700


def test_cli_preflight_can_submit_explicit_environment_smoke(
    tmp_path, capsys, monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state(max_iterations=2))
    monkeypatch.setattr(cli_mod, "check_backends", _backend_availability)
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(),
    )
    calls = []

    def fake_smoke(**kwargs):
        calls.append(kwargs)
        return {
            "schema_version": 1,
            "submitted": True,
            "ok": True,
            "job_id": "12345",
            "output_path": str(campaign / "smoke.out"),
            "error": "",
        }

    monkeypatch.setattr(cli_mod, "run_submitted_environment_smoke", fake_smoke)

    rc = main(
        [
            "preflight",
            "--campaign-dir",
            str(campaign),
            "--submit-environment-smoke",
        ]
    )

    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["campaign_dir"] == campaign
    out = capsys.readouterr().out
    assert "Submitted Environment Smoke" in out
    assert "[OK] compute-node runtime: job 12345" in out


def test_cli_preflight_json_recomputes_next_action_after_smoke_failure(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    _write_locked_state(campaign, fresh_campaign_state(max_iterations=2))
    monkeypatch.setattr(cli_mod, "check_backends", _backend_availability)
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(),
    )
    monkeypatch.setattr(
        cli_mod,
        "run_submitted_environment_smoke",
        lambda **_kwargs: {
            "schema_version": 1,
            "submitted": True,
            "ok": False,
            "job_id": "12345",
            "output_path": str(campaign / "smoke.out"),
            "error": "import failed",
        },
    )

    rc = main(
        [
            "preflight",
            "--campaign-dir",
            str(campaign),
            "--submit-environment-smoke",
            "--json",
        ]
    )

    assert rc == 12
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["next_action"] == "fix failed checks before live start"


def test_cli_preflight_reports_all_failures_together(tmp_path, capsys, monkeypatch):
    campaign = _campaign_with_config(tmp_path)

    def bad_availability():
        return _backend_availability(sacct=False, ariadne=False, sacct_path="")

    monkeypatch.setattr(cli_mod, "check_backends", bad_availability)
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(
            ok=False,
            pool_n_frames=50,
            required_pool_frames=90,
        ),
    )

    rc = main(["preflight", "--campaign-dir", str(campaign)])

    assert rc == 12
    out = capsys.readouterr().out
    assert "  result: blocked" in out
    assert "  [FAIL] sacct: not found" in out
    assert "  [FAIL] ariadne: not importable" in out
    assert "  [FAIL] frames available: 50" in out
    assert "  [FAIL] frames required: 90" in out
    assert "fix failed checks before live start" in out
    assert "install/build ariadne in the configured submitted Python venv" in out
    assert "fix trajectory pool feasibility" in out


def test_cli_preflight_marks_missing_submitted_python_as_failed_and_actionable(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    _write_locked_state(campaign, fresh_campaign_state(max_iterations=2))
    monkeypatch.setattr(
        cli_mod,
        "check_backends",
        lambda: _backend_availability(
            squeue=False,
            squeue_path="",
            batch_python=False,
            batch_python_error="configured executable is missing",
        ),
    )
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(),
    )

    rc = main(["preflight", "--campaign-dir", str(campaign)])

    assert rc == 12
    out = capsys.readouterr().out
    assert "[FAIL] squeue: not found" in out
    assert "[FAIL] configured python:" in out
    assert "make squeue available" in out
    assert "configure an absolute submitted Python path" in out


def test_cli_preflight_ready_campaign_with_active_daemon_recommends_monitoring(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    _write_locked_state(campaign, fresh_campaign_state(max_iterations=2))
    monkeypatch.setattr(cli_mod, "check_backends", _backend_availability)
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(),
    )
    monkeypatch.setattr(
        cli_mod,
        "_probe_daemon_lock",
        lambda _path: {"lock_held": True},
    )

    rc = main(["preflight", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "the daemon is running; monitor it instead of starting another" in out
    assert "ichor-al-daemon journal --campaign-dir" in out
    assert "start --campaign-dir" not in out


def test_cli_preflight_warns_when_pool_has_no_surplus(tmp_path, capsys, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    state_path = campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state(max_iterations=2))
    monkeypatch.setattr(cli_mod, "check_backends", _backend_availability)
    monkeypatch.setattr(
        cli_mod,
        "_pool_feasibility_summary",
        lambda _campaign, _config: _pool_feasibility_payload(
            ok=True,
            pool_n_frames=90,
            required_pool_frames=90,
        ),
    )

    rc = main(["preflight", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "  [WARN] surplus frames after planned campaign: 0" in out


def test_recovery_dashboard_reports_missing_state(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    text = cli_mod.format_recovery_dashboard(campaign)

    assert "Recovery dashboard" in text
    assert "state.json: missing" in text
    assert "initialise the campaign: ichor-al-daemon init" in text


def test_recovery_dashboard_reports_invalid_state(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    (data / DEFAULT_STATE_FILENAME).write_text("{bad json", encoding="utf-8")

    text = cli_mod.format_recovery_dashboard(campaign)

    assert "state.json: invalid" in text
    assert "ichor-al-daemon reconcile" in text


def test_recovery_dashboard_reports_last_exception_and_staging(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    (data / "LAST_EXCEPTION.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "timestamp": "2026-01-02T03:04:05+00:00",
                "exception_type": "RuntimeError",
                "message": "boom",
                "phase": "GAUSSIAN",
                "iteration": 2,
            }
        ),
        encoding="utf-8",
    )
    staging = campaign / ".DATA" / "STAGING" / "GAUSSIAN"
    staging.mkdir(parents=True)
    (staging / "old.txt").write_text("old", encoding="utf-8")

    text = cli_mod.format_recovery_dashboard(campaign)

    assert "Last exception" in text
    assert "RuntimeError" in text
    assert "boom" in text
    assert "Staging" in text
    assert "non-empty top_level=1" in text
    assert "inspect the recovery blockers: ichor-al-daemon reconcile" in text
    assert "--apply" not in text


def test_recovery_dashboard_reports_active_intents(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid="uid123456789",
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
    )

    text = cli_mod.format_recovery_dashboard(campaign)

    assert "Submission intents" in text
    assert "INITIAL_FEREBUS@0 PRE_SUBMIT" in text
    assert "ichor-al-daemon reconcile" in text
    assert "--cancel-jobs" not in text


def test_live_job_name_rejects_control_characters_and_caps_length():
    phase = CampaignPhase.PHASE_A_DIVERSITY.value
    with pytest.raises(ValueError, match="unsafe Slurm job name"):
        live_job_name("bad\nuid", phase, 0)

    name = live_job_name("x" * 200, phase, 123456)
    assert len(name) <= 128
    assert "\n" not in name


def test_recovery_dashboard_uses_pool_authority_without_hashing_payload(tmp_path):
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool

    campaign = _campaign_with_config(tmp_path)
    source = tmp_path / "pool.xyz"
    source.write_text(
        "1\n"
        "frame 0\n"
        "H 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    TrajectoryPool.import_from(
        source,
        campaign,
        overwrite=True,
    )
    pool_xyz = campaign / "pool.xyz"
    with pool_xyz.open("a", encoding="utf-8") as f:
        f.write("# drift\n")

    text = cli_mod.format_recovery_dashboard(campaign)

    assert "Trajectory pool" in text
    assert "authority sha=" in text
    assert "pool drift detected" not in text


def test_journal_list_event_types_does_not_require_journal_file(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)

    rc = main(["journal", "--campaign-dir", str(campaign), "--list-event-types"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "reconcile_applied" in out
    assert "phase_submitted" in out


def test_cli_status_json_prints_state_payload(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    s = fresh_campaign_state(max_iterations=5)
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, s)
    rc = main(["status", "--campaign-dir", str(campaign), "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["iteration"] == 0
    assert payload["max_iterations"] == 5
    assert payload["state_path"].endswith(DEFAULT_STATE_FILENAME)
    assert payload["lock_file_exists"] is False
    assert payload["lock_held"] is False
    assert payload["active_submission_intents"] == []
    assert "artifact_manifest_status" in payload
    assert "state_artifact_contract_status" in payload
    assert payload["state_artifact_contract_status"]["ok"] is True
    assert payload["recommendations"][0]["code"] == "phase_init_ready"
    assert payload["next_action"] == payload["recommendations"][0]["primary"]


def test_cli_status_json_reports_aimall_outputs_as_reusable(
    tmp_path,
    capsys,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import reconcile as reconcile_module

    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=20)
    state.phase = CampaignPhase.AIMALL
    state.iteration = 14
    state.reference_data_version = 13
    state.models_version = 13
    write_state(
        campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME,
        state,
    )
    monkeypatch.setattr(
        reconcile_module,
        "inspect_aimall_postprocess_recovery",
        lambda *_args, **_kwargs: {
            "phase": CampaignPhase.AIMALL.value,
            "iteration": 14,
            "logical_total": 149,
            "n_complete": 149,
            "n_reuse": 149,
            "n_retry": 0,
            "retry_task_ids": [],
            "retry_task_file": None,
            "force_resubmit": False,
        },
    )

    rc = main(["status", "--campaign-dir", str(campaign), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["partial_array_recovery"]["n_reuse"] == 149
    assert payload["partial_array_recovery"]["n_retry"] == 0
    assert "_presentation_aimall_postprocess_recovery" not in payload


def test_cli_status_reads_recorded_environment_without_live_capture(
    tmp_path,
    capsys,
    monkeypatch,
):
    from ichor.hpc.active_learning import execution_identity

    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=2)
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, state)
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    execution_identity.ensure_execution_identity(
        campaign,
        campaign_uid=str(state.campaign_uid),
        config=config,
        requested_mode="dry_run",
    )
    transition_time = "2026-07-18T12:34:56+00:00"
    append_event(
        campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
        "environment_generation_advanced",
        ts=transition_time,
        generation=0,
    )

    def refuse_live_capture(*_args, **_kwargs):
        raise AssertionError("status attempted to capture the live environment")

    monkeypatch.setattr(
        execution_identity,
        "capture_environment_generation",
        refuse_live_capture,
    )

    rc = main(["status", "--campaign-dir", str(campaign), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    environment = payload["active_environment_generation"]
    assert environment["generation"] == 0
    assert environment["last_transition"] == transition_time


def test_cli_status_bound_dry_run_campaign_recommends_resume_without_mode_change(
    tmp_path,
    capsys,
):
    from ichor.hpc.active_learning import execution_identity

    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=2)
    _write_locked_state(campaign, state)
    execution_identity.ensure_execution_identity(
        campaign,
        campaign_uid=str(state.campaign_uid),
        config=CampaignConfig.from_yaml(campaign / "campaign.yaml"),
        requested_mode="dry_run",
    )

    assert main(["status", "--campaign-dir", str(campaign)]) == 0
    out = capsys.readouterr().out
    assert "ichor-al-daemon resume --campaign-dir" in out
    assert "--mode live" not in out


def test_cli_status_init_hides_not_due_committed_artifact_errors(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    s = fresh_campaign_state(max_iterations=2)
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, s)

    rc = main(["status", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Campaign\n" in out
    assert "  phase: Campaign setup" in out
    assert "  purpose: prepare the campaign before its first run" in out
    assert "Current status\n" in out
    assert "  overall: stopped and ready to continue" in out
    assert "  current work: The daemon is stopped; campaign setup is the next campaign step." in out
    assert "Progress so far\n" in out
    assert "  QM data: not produced yet" in out
    assert "  model: not produced yet" in out
    assert "What happens next\n" in out
    assert "  run: ichor-al-daemon start" in out
    assert " --mode live" not in out
    assert "CommittedArtifactError" not in out
    assert "training status: problem" not in out


def test_cli_status_default_prints_operator_friendly_summary(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    s = fresh_campaign_state(max_iterations=5)
    s.iteration = 3
    s.phase = CampaignPhase.STOP_CHECK
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, s)

    rc = main(["status", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Campaign\n" in out
    assert "  phase: Iteration completion check" in out
    assert "  iteration: 3 of 5" in out
    assert "Current status\n" in out
    assert "  overall: needs attention" in out
    assert "  daemon: not running" in out
    assert "Progress so far\n" in out
    assert "  QM data: not produced yet" in out
    assert "  model: not produced yet" in out
    assert "What happens next\n" in out
    assert "  you need to do: run reconcile; STOP_CHECK needs the latest coherent committed reference-data/model pair" in out
    assert "  because: the recorded state and committed data/model evidence do not agree" in out
    assert "  run: ichor-al-daemon reconcile --campaign-dir " in out
    assert "CommittedArtifactError" not in out
    assert "training v0: problem" not in out
    assert "background_pid" not in out
    assert "shutdown_requested" not in out
    assert "uid:" not in out
    assert "config lock" not in out.lower()
    assert "ledger" not in out.lower()
    assert not out.lstrip().startswith("{")


def test_cli_status_reports_initial_ferebus_bootstrap_contract_problem(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    s = fresh_campaign_state(max_iterations=5)
    s.phase = CampaignPhase.INITIAL_FEREBUS
    s.reference_data_version = -1
    s.models_version = -1
    write_state(data / DEFAULT_STATE_FILENAME, s)

    rc = main(["status", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Current status\n" in out
    assert "  overall: needs attention" in out
    assert "Progress so far\n" in out
    assert "  QM data: not produced yet" in out
    assert "  model: not produced yet" in out
    assert "  readiness: the committed data or model needs attention" in out
    assert "What happens next\n" in out
    assert "initial_ferebus_point_allocation_invalid" not in out
    assert "CommittedArtifactError" not in out
    assert "being produced" not in out


def test_cli_status_backend_submission_failure_recommends_reconcile_apply(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    write_state(data / DEFAULT_STATE_FILENAME, state)
    append_event(
        data / "journal.ndjson",
        "halt",
        from_phase=CampaignPhase.PHASE_A_DIVERSITY.value,
        iteration=0,
        reason="backend_submission_failed: partition 'multicore_small' is not present",
    )

    rc = main(["status", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "fix the configured backend or profile problem, then preview recovery" in out
    assert "ichor-al-daemon reconcile --campaign-dir " in out
    assert " --apply" not in out


def test_cli_status_halted_rendering_survives_a_malformed_journal(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.lifecycle_context = {
        "disposition": "halted",
        "message": "known lifecycle failure",
        "reason_code": "test_failure",
        "from_phase": CampaignPhase.INIT.value,
        "iteration": 0,
        "timestamp_iso": "2026-07-20T00:00:00+00:00",
    }
    write_state(data / DEFAULT_STATE_FILENAME, state)
    (data / "journal.ndjson").write_text("{bad json\n", encoding="utf-8")

    rc = main(["status", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Campaign halted" in out
    assert "known lifecycle failure" in out


def test_reconcile_apply_contract_guard_rejects_invalid_state(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=5)
    state.phase = CampaignPhase.STOP_CHECK
    state.reference_data_version = 0
    state.models_version = 0

    error = cli_mod._reconcile_apply_contract_error(campaign, state)

    assert error is not None
    assert "phase=STOP_CHECK" in error
    assert "reference_data_version=0" in error
    assert "models_version=0" in error
    assert "state references missing committed reference-data version" in error


def test_cli_status_default_summarises_active_submission_intents(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    s = fresh_campaign_state(max_iterations=5)
    _write_locked_state(campaign, s)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=s.campaign_uid,
        phase_name="GAUSSIAN",
        iteration=0,
    )
    submission_intent.mark_submitted(
        campaign,
        "GAUSSIAN",
        0,
        "12345",
        expected_tasks=20,
    )

    rc = main(["status", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "recorded Slurm job still needs monitoring or postprocessing" in out
    assert "ichor-al-daemon resume --campaign-dir" in out
    assert "GAUSSIAN@0" not in out
    assert "job_id=12345" not in out


def test_cli_status_describes_jobless_intent_as_prepared_local_work(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=5)
    _write_locked_state(campaign, state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
    )

    assert main(["status", "--campaign-dir", str(campaign)]) == 0
    out = capsys.readouterr().out
    assert "local work for initial FEREBUS training is prepared but not running" in out
    assert "active Slurm job" not in out
    assert "ichor-al-daemon resume --campaign-dir" in out


def test_cli_status_reports_stale_lock_file_as_not_held(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    (data / DAEMON_LOCK_FILENAME).write_text("stale\n", encoding="utf-8")

    rc = main(["status", "--campaign-dir", str(campaign), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lock_file_exists"] is True
    assert payload["lock_held"] is False


def test_cli_status_reports_actually_held_lock(tmp_path, capsys):
    pytest.importorskip("portalocker")
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())
    lock_path = data / DAEMON_LOCK_FILENAME

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import portalocker, sys, time\n"
                "lock = portalocker.Lock(sys.argv[1], mode='a', timeout=0, fail_when_locked=True)\n"
                "lock.acquire()\n"
                "print('ready', flush=True)\n"
                "time.sleep(10)\n"
                "lock.release()\n"
            ),
            str(lock_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        rc = main(["status", "--campaign-dir", str(campaign), "--json"])
    finally:
        holder.terminate()
        try:
            holder.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.communicate(timeout=5)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lock_file_exists"] is True
    assert payload["lock_held"] is True


def test_cli_status_surfaces_lock_probe_error(tmp_path, capsys, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())

    monkeypatch.setattr(
        cli_mod,
        "_probe_daemon_lock",
        lambda lock_path: {
            "lock_file_exists": True,
            "lock_held": None,
            "lock_probe_error": "RuntimeError: boom",
        },
    )
    rc = main(["status", "--campaign-dir", str(campaign), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lock_held"] is None
    assert payload["lock_probe_error"] == "RuntimeError: boom"


def test_cli_status_returns_4_when_state_missing(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["status", "--campaign-dir", str(campaign)])
    out = capsys.readouterr().out
    assert rc == 4
    assert "Campaign\n  phase: unavailable" in out
    assert "Current status\n  overall: setup required" in out
    assert "Progress so far\n" in out
    assert "What happens next\n" in out
    assert "bootstrap the fresh campaign" in out
    assert "fresh init safe" not in out


def test_cli_status_returns_json_recommendation_when_state_missing(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["status", "--campaign-dir", str(campaign), "--json"])
    assert rc == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["status_error"] == "state_missing"
    assert payload["fresh_init_safe"] is True
    assert payload["recommendations"][0]["code"] == "state_missing_fresh_init"
    assert payload["next_action"] == payload["recommendations"][0]["primary"]


def test_cli_status_unreadable_state_has_a_dedicated_recommendation(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    (data / DEFAULT_STATE_FILENAME).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        cli_mod,
        "read_state",
        lambda _path: (_ for _ in ()).throw(OSError("permission denied")),
    )

    rc = main(["status", "--campaign-dir", str(campaign), "--json"])

    assert rc != 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status_error"] == "state_unreadable"
    assert payload["recommendations"][0]["code"] == "state_unreadable"
    assert "reconcile" in payload["recommendations"][0]["command"]


def test_cli_status_recommends_reconcile_when_state_missing_with_artefacts(
    tmp_path,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    (campaign / "ACTIVE_LEARNING" / "iteration-000001").mkdir(parents=True)

    rc = main(["status", "--campaign-dir", str(campaign), "--json"])

    assert rc == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["fresh_init_safe"] is False
    assert payload["stateful_artifacts_count"] == 1
    assert payload["recommendations"][0]["code"] == "state_missing"
    assert "reconcile" in payload["recommendations"][0]["command"]


def test_cli_init_bootstraps_fresh_state_and_config_lock(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)

    rc = main(["init", "--campaign-dir", str(campaign), "--yes"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Campaign initialised" in out
    assert "Imported pool:" in out
    state = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert state.phase is CampaignPhase.INIT
    assert state.max_iterations == 2
    assert (campaign / DEFAULT_DATA_SUBDIR / "config_lock.json").is_file()
    assert "ichor-al-daemon start --campaign-dir " in out
    assert " --mode live" not in out
    assert " --mode dry_run" in out
    assert "pool SHA-256" not in out
    assert "ALF (1-based)" not in out
    assert " --source /path/to/pool.xyz" not in out


def test_cli_init_verbose_retains_bootstrap_diagnostics(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)

    rc = main(
        ["init", "--campaign-dir", str(campaign), "--yes", "--verbose"]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Bootstrap discovery complete" in out
    assert "pool SHA-256" in out
    assert "bootstrap identity" in out


def test_cli_init_rerun_preserves_existing_campaign_uid(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    assert main(["init", "--campaign-dir", str(campaign), "--yes"]) == 0
    first = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    capsys.readouterr()

    assert main(["init", "--campaign-dir", str(campaign), "--yes"]) == 0
    second = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)

    assert second.campaign_uid == first.campaign_uid
    assert "state.json: already_initialised" in capsys.readouterr().out


def test_cli_init_refuses_missing_state_with_stateful_artefacts(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / "ACTIVE_LEARNING" / "iteration-000001").mkdir(parents=True)

    rc = main(["init", "--campaign-dir", str(campaign), "--yes"])

    assert rc == 16
    err = capsys.readouterr().err
    assert "campaign bootstrap failed" in err
    assert "reconcile --campaign-dir" in err


def _recommendation_codes(campaign: Path, payload: dict) -> list[str]:
    complete_payload = {"lock_held": False, **payload}
    return [
        item.code
        for item in build_status_recommendations(campaign, complete_payload)
    ]


def test_status_recommendations_cover_runtime_and_job_blockers(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(campaign, {"lock_held": True}) == [
        "daemon_running"
    ]
    assert _recommendation_codes(
        campaign,
        {
            "active_submission_intents": [
                {"phase": "GAUSSIAN", "iteration": 0, "status": "SUBMITTED"}
            ]
        },
    ) == ["local_submission_intent"]
    assert _recommendation_codes(
        campaign,
        {"pending_jobs": {"INITIAL_GAUSSIAN": "12345"}},
    ) == ["pending_state_job"]
    assert _recommendation_codes(campaign, {"shutdown_requested": True}) == [
        "shutdown_requested"
    ]


def test_status_recommendations_cover_halted_reason_classes(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "latest_halt_event": {"reason": "NODE_FAIL during array"},
        },
    ) == ["halted_scheduler_transient"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "latest_halt_event": {"reason": "OUT_OF_MEMORY"},
        },
    ) == ["halted_scheduler_hard_failure"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "latest_halt_event": {"reason": "committed_model_contract_invalid"},
        },
    ) == ["halted_contract_failure"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "latest_halt_event": {"reason": "campaign.yaml changed"},
        },
    ) == ["halted_config_changed"]


def test_status_describes_phase_b_resource_handoff_failure_as_recovery(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    result = build_status_recommendations(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "latest_halt_event": {
                "reason": (
                    "backend_submission_failed: resource evidence not yet produced "
                    "for PHASE_B_DIVERSITY: ARIADNE rejected seed_dir directory missing"
                )
            },
        },
    )[0]

    assert result.code == "halted_backend_submission_failed"
    assert result.primary == "preview recovery of the phase input evidence"
    assert "backend or profile" not in result.primary
    assert result.command and "reconcile" in result.command


def test_status_recommends_repolling_preserved_scheduler_job(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    phase = CampaignPhase.INITIAL_GAUSSIAN.value
    job_id = "17615141"

    recommendations = build_status_recommendations(
        campaign,
        {
            "lock_held": False,
            "phase": CampaignPhase.HALTED.value,
            "iteration": 0,
            "pending_jobs": {phase: job_id},
            "active_submission_intents": [
                {
                    "phase": phase,
                    "iteration": 0,
                    "status": "SUBMITTED",
                    "job_id": job_id,
                }
            ],
            "lifecycle_context": {
                "disposition": "halted",
                "reason_code": "sacct_missing_timeout",
                "message": "expected Slurm array rows remained missing",
                "from_phase": phase,
                "iteration": 0,
                "source": "daemon",
                "job_id": job_id,
                "scheduler_uncertain": True,
            },
        },
    )

    assert [item.code for item in recommendations] == [
        "halted_scheduler_uncertain"
    ]
    assert "will not resubmit" in recommendations[0].primary
    assert recommendations[0].command.startswith("ichor-al-daemon resume ")
    assert "--mode" not in recommendations[0].command


@pytest.mark.parametrize(
    ("reason_code", "expected_code"),
    [
        (
            "mandatory_custom_bootstrap_failed",
            "halted_mandatory_custom_bootstrap_failed",
        ),
        (
            "replacement_reserve_exhausted",
            "halted_replacement_reserve_exhausted",
        ),
        ("ferebus_quality_failed", "halted_ferebus_quality_failed"),
    ],
)
def test_status_recommendations_use_authoritative_lifecycle_reason(
    tmp_path,
    reason_code,
    expected_code,
):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "lifecycle_context": {
                "disposition": "halted",
                "reason_code": reason_code,
                "message": "authoritative current failure",
                "from_phase": CampaignPhase.ALLOCATION_CHECK.value,
                "iteration": 1,
                "timestamp_iso": "2026-07-11T00:00:00+00:00",
            },
            "latest_halt_event": {"reason": "stale historical NODE_FAIL"},
        },
    ) == [expected_code]


def test_status_recommendations_report_scientific_completion_not_stop(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    codes = _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.DONE.value,
            "lifecycle_context": {
                "disposition": "completed",
                "reason_code": "scientific_convergence",
                "message": "scientific convergence criterion reached",
                "from_phase": CampaignPhase.STOP_CHECK.value,
                "iteration": 3,
                "timestamp_iso": "2026-07-11T00:00:00+00:00",
            },
        },
    )

    assert codes == ["campaign_completed_scientific_convergence"]


def test_status_recommendations_cover_contract_failure_classes(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.STOP_CHECK.value,
            "state_artifact_contract_status": {
                "ok": False,
                "error": "CommittedArtifactError: state references missing committed model version 0",
            },
        },
    ) == ["stop_check_no_committed_pair"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.GAUSSIAN.value,
            "state_artifact_contract_status": {
                "ok": False,
                "error": "CommittedArtifactError: state reference-data/model version skew",
            },
        },
    ) == ["version_skew"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.SEED_SELECT.value,
            "state_artifact_contract_status": {
                "ok": False,
                "error": "CommittedArtifactError: reference_data_version_invalid:0",
            },
        },
    ) == ["reference_data_missing"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.SEED_SELECT.value,
            "state_artifact_contract_status": {
                "ok": False,
                "error": "CommittedArtifactError: models_version_invalid:0",
            },
        },
    ) == ["models_missing"]


@pytest.mark.parametrize(
    ("phase", "code"),
    [
        (CampaignPhase.INIT, "phase_init_ready"),
        (CampaignPhase.PHASE_A_DIVERSITY, "phase_phase_a_diversity_ready"),
        (CampaignPhase.INITIAL_GAUSSIAN, "phase_initial_gaussian_ready"),
        (CampaignPhase.INITIAL_AIMALL, "phase_initial_aimall_ready"),
        (CampaignPhase.INITIAL_FEREBUS, "phase_initial_ferebus_ready"),
        (CampaignPhase.SEED_SELECT, "phase_seed_select_ready"),
        (CampaignPhase.ARIADNE_ARRAY, "phase_ariadne_array_ready"),
        (CampaignPhase.PHASE_B_DIVERSITY, "phase_phase_b_diversity_ready"),
        (CampaignPhase.SPLIT, "phase_split_ready"),
        (CampaignPhase.GAUSSIAN, "phase_gaussian_ready"),
        (CampaignPhase.AIMALL, "phase_aimall_ready"),
        (CampaignPhase.REFERENCE_COMMIT, "phase_reference_commit_ready"),
        (CampaignPhase.FEREBUS, "phase_ferebus_ready"),
        (CampaignPhase.STOP_CHECK, "phase_stop_check_ready"),
    ],
)
def test_status_recommendations_cover_all_idle_phases(tmp_path, phase, code):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(
        campaign,
        {
            "phase": phase.value,
            "state_artifact_contract_status": {"ok": True},
            "artifact_manifest_status": {},
        },
    ) == [code]


def test_status_recommendations_cover_done_and_unknown_phase(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(campaign, {"phase": CampaignPhase.DONE.value}) == [
        "campaign_done"
    ]
    assert _recommendation_codes(campaign, {"phase": "NOT_A_PHASE"}) == [
        "phase_unknown"
    ]


def test_status_recommendation_dicts_are_json_ready(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    payload = {
        "status_error": "state_schema_invalid",
        "state_error": "StateSchemaError: bad field",
    }

    data = recommendation_dicts(build_status_recommendations(campaign, payload))

    assert data[0]["code"] == "state_schema_invalid"
    assert data[0]["severity"] == "required"
    assert "primary" in data[0]


@pytest.mark.parametrize(
    ("recovery", "expected_code"),
    [
        (
            {
                "state": "recoverable",
                "recoverable": True,
                "reason": "the previous reconcile stopped before changing campaign data",
            },
            "reconcile_transaction_recoverable",
        ),
        (
            {
                "state": "blocked",
                "recoverable": False,
                "reason": "campaign authority is ambiguous",
            },
            "reconcile_transaction_manual_review",
        ),
    ],
)
def test_status_recommends_preview_for_interrupted_reconcile(
    tmp_path,
    recovery,
    expected_code,
):
    campaign = _campaign_with_config(tmp_path)
    payload = {
        "phase": CampaignPhase.SEED_SELECT.value,
        "_presentation_reconcile_transaction_recovery": recovery,
        "state_artifact_contract_status": {"ok": True},
        "artifact_manifest_status": {},
    }

    recommendations = build_status_recommendations(campaign, payload)

    assert recommendations[0].code == expected_code
    assert recommendations[0].command.startswith(
        "ichor-al-daemon reconcile --campaign-dir "
    )
    assert str(campaign) in recommendations[0].command


def test_status_recommends_repeating_incomplete_job_cancellation(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    recommendations = build_status_recommendations(
        campaign,
        {
            "shutdown_requested": False,
            "stop_request": {
                "request_id": "request-1",
                "mode": "immediate",
                "status": "cancelling",
            },
        },
    )

    assert recommendations[0].code == "user_stop_cancellation_incomplete"
    assert "--immediate --cancel-jobs" in str(recommendations[0].command)


def test_cli_stop_records_request_without_rewriting_state(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, fresh_campaign_state())
    rc = main(["stop", "--campaign-dir", str(campaign)])
    output_lines = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert output_lines[0] == (
        "immediate stop requested during INIT in iteration 0"
    )
    assert output_lines[1] == "Recorded Slurm jobs, if any, were left running."
    assert output_lines[2] == "Monitor the stop:"
    assert output_lines[3].strip().startswith("ichor-al-daemon status")
    assert "request id:" not in "\n".join(output_lines)
    s = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert s.shutdown_requested is False
    request = read_stop_request(campaign, expected_campaign_uid=s.campaign_uid)
    assert request is not None
    assert request["mode"] == "immediate"
    assert request["status"] == "requested"


def test_cli_stop_without_cancel_jobs_does_not_call_scancel(tmp_path, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] = "123"
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, state)
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: pytest.fail("plain stop must not call scancel"),
    )

    rc = main(["stop", "--campaign-dir", str(campaign)])

    assert rc == 0
    stopped = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert stopped.shutdown_requested is False
    assert stopped.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] == "123"


def test_cli_stop_records_phase_and_iteration_boundaries(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=2)
    state.phase = CampaignPhase.AIMALL
    state.iteration = 1
    state.replacement_round = 2
    _write_locked_state(campaign, state)

    assert main(["stop", "--campaign-dir", str(campaign), "--after-phase"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        "stop requested after phase AIMALL in iteration 1, replacement round 2"
    )
    phase_request = read_stop_request(campaign)
    assert phase_request["mode"] == "after_phase"
    assert phase_request["target_phase"] == CampaignPhase.AIMALL.value
    assert phase_request["target_iteration"] == 1

    assert main(["stop", "--campaign-dir", str(campaign), "--immediate"]) == 0
    capsys.readouterr()
    archive_and_clear_stop_request(campaign, status="test_reset")

    assert main(
        ["stop", "--campaign-dir", str(campaign), "--after-iteration"]
    ) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        "stop requested after iteration 1"
    )
    iteration_request = read_stop_request(campaign)
    assert iteration_request["mode"] == "after_iteration"
    assert iteration_request["target_iteration"] == 1


def test_cli_rejects_job_cancellation_for_boundary_stop(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())

    rc = main(
        [
            "stop",
            "--campaign-dir",
            str(campaign),
            "--after-phase",
            "--cancel-jobs",
        ]
    )

    assert rc == 2
    assert "valid only with --immediate" in capsys.readouterr().err


def test_cli_stop_rejects_already_terminal_campaign(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.phase = CampaignPhase.DONE
    write_state(data / DEFAULT_STATE_FILENAME, state)

    assert main(["stop", "--campaign-dir", str(campaign)]) == 6
    assert "already DONE" in capsys.readouterr().err
    assert not stop_request_path(campaign).exists()


def test_cli_stop_cancel_jobs_records_cancellation_without_rewriting_state(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.campaign_uid = "abc123def456-uid"
    state.iteration = 0
    phase = CampaignPhase.PHASE_A_DIVERSITY.value
    state.pending_jobs[phase] = "123"
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, state)
    expected_name = live_job_name(state.campaign_uid, phase, 0)
    cancelled = []
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [{"job_id": str(job_id), "state": "RUNNING", "job_name": expected_name}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: (cancelled.append(str(job_id)) or (True, "")),
    )
    monkeypatch.setattr(
        cli_mod,
        "_confirm_cancelled_slurm_job",
        lambda *args, **kwargs: (True, ""),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    out = capsys.readouterr().out

    assert rc == 0
    assert cancelled == ["123"]
    stopped = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert stopped.shutdown_requested is False
    assert stopped.pending_jobs[phase] == "123"
    request = read_stop_request(campaign, expected_campaign_uid=state.campaign_uid)
    assert request["status"] == "requested"
    assert request["cancellation_summary"]["cancelled"][0]["job_id"] == "123"
    assert "Slurm cancellation: 1 cancelled" in out


def test_cli_stop_cancel_jobs_records_mixed_terminal_task_evidence(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon.scheduler_recovery import (
        classify_terminal_scheduler_evidence,
        load_scheduler_terminal_receipt,
    )
    from ichor.hpc.active_learning.submit.sacct_poll import (
        JobObservation,
        JobStatus,
    )

    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 15
    phase = state.phase.value
    state.pending_jobs[phase] = "17923151"
    write_state(data / DEFAULT_STATE_FILENAME, state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=phase,
        iteration=15,
        expected_tasks=4,
    )
    submission_intent.mark_submitted(
        campaign,
        phase,
        15,
        "17923151",
        expected_tasks=4,
    )
    intent = submission_intent.load_intent(campaign, phase, 15)
    expected_name = str(intent["expected_job_name"])
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [
                {
                    "job_id": str(job_id),
                    "state": "RUNNING",
                    "job_name": expected_name,
                }
            ],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda _job_id: (True, ""),
    )

    observations = [
        JobObservation(
            job_id="17923151_" + str(task_id),
            status=(
                JobStatus.COMPLETED
                if task_id == 0
                else JobStatus.CANCELLED
            ),
            exit_code=((0, 0) if task_id == 0 else (0, 15)),
            elapsed_seconds=30,
        )
        for task_id in range(4)
    ]

    def confirm(*_args, **kwargs):
        classification = classify_terminal_scheduler_evidence(
            kwargs["campaign_dir"],
            kwargs["intent"],
            observations,
            queue_active=False,
        )
        kwargs["classification_sink"].update(classification)
        return True, ""

    monkeypatch.setattr(
        cli_mod,
        "_confirm_cancelled_slurm_job",
        confirm,
    )

    assert (
        main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
        == 0
    )
    receipt = load_scheduler_terminal_receipt(campaign, intent)
    assert receipt is not None
    assert receipt["completed_logical_task_ids"] == [0]
    assert receipt["retry_logical_task_ids"] == [1, 2, 3]
    resolved, blockers = (
        cli_mod._resolve_terminal_submission_intents_for_apply(
            campaign,
            [intent],
            persist_terminal_receipts=False,
        )
    )
    assert blockers == []
    assert len(resolved) == 1
    assert resolved[0]["submission_identity"] == intent[
        "submission_identity"
    ]
    assert resolved[0]["job_id"] == "17923151"
    request = read_stop_request(
        campaign,
        expected_campaign_uid=state.campaign_uid,
    )
    item = request["cancellation_summary"]["cancelled"][0]
    assert item["n_completed"] == 1
    assert item["n_retry"] == 3


def test_pre_submit_stop_records_no_scheduler_acceptance_receipt(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon.scheduler_recovery import (
        load_scheduler_terminal_receipt,
    )

    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 3
    write_state(data / DEFAULT_STATE_FILENAME, state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=state.phase.value,
        iteration=state.iteration,
        expected_tasks=3,
    )
    original_load = submission_intent.load_intent
    active_intent = original_load(
        campaign,
        state.phase.value,
        state.iteration,
    )
    submission_intent.mark_failed(
        campaign,
        state.phase.value,
        state.iteration,
        "user_cancelled_before_scheduler_acceptance",
    )
    inventory_calls = 0

    def active_then_terminal(*_args, **_kwargs):
        nonlocal inventory_calls
        inventory_calls += 1
        return [active_intent] if inventory_calls == 1 else []

    monkeypatch.setattr(
        cli_mod,
        "_load_active_submission_intents",
        active_then_terminal,
    )
    monkeypatch.setattr(
        cli_mod,
        "get_scheduler_backend",
        lambda _kind: SimpleNamespace(
            find_accounted_job_by_name=lambda *_args, **_kwargs: SimpleNamespace(
                job_id=None,
                inconclusive=False,
                error=None,
            )
        ),
    )

    summary = cli_mod._cancel_recorded_scheduler_jobs(
        campaign,
        state,
        confirmation_timeout_seconds=1,
    )

    assert summary["failed"] == []
    assert len(summary["cancelled"]) == 1
    assert summary["cancelled"][0]["job_id"] == ""
    assert summary["cancelled"][0]["n_completed"] == 0
    assert summary["cancelled"][0]["n_retry"] == 3
    intent = original_load(
        campaign,
        state.phase.value,
        state.iteration,
    )
    receipt = load_scheduler_terminal_receipt(campaign, intent)
    assert receipt is not None
    assert receipt["scheduler_acceptance"] == "not_accepted"
    assert receipt["job_id"] is None


def test_pre_submit_stop_before_task_staging_needs_no_fabricated_receipt(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.phase = CampaignPhase.INITIAL_FEREBUS
    write_state(data / DEFAULT_STATE_FILENAME, state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=state.phase.value,
        iteration=state.iteration,
    )
    original_load = submission_intent.load_intent
    active_intent = original_load(
        campaign,
        state.phase.value,
        state.iteration,
    )
    submission_intent.mark_failed(
        campaign,
        state.phase.value,
        state.iteration,
        "user_cancelled_before_scheduler_acceptance",
    )
    inventory_calls = 0

    def active_then_terminal(*_args, **_kwargs):
        nonlocal inventory_calls
        inventory_calls += 1
        return [active_intent] if inventory_calls == 1 else []

    class NoAcceptedJobBackend:
        def find_accounted_job_by_name(self, *_args, **_kwargs):
            return SimpleNamespace(
                job_id=None,
                inconclusive=False,
                error=None,
            )

    monkeypatch.setattr(
        cli_mod,
        "_load_active_submission_intents",
        active_then_terminal,
    )
    monkeypatch.setattr(
        cli_mod,
        "get_scheduler_backend",
        lambda _kind: NoAcceptedJobBackend(),
    )

    summary = cli_mod._cancel_recorded_scheduler_jobs(
        campaign,
        state,
        confirmation_timeout_seconds=1,
    )

    assert summary["failed"] == []
    assert summary["cancelled"] == [
        {
            "job_id": "",
            "scheduler_identity_kind": "slurm",
            "phases": ["INITIAL_FEREBUS"],
            "intent_keys": [
                {
                    "phase": "INITIAL_FEREBUS",
                    "iteration": 0,
                }
            ],
            "n_completed": 0,
            "task_count_unknown": True,
            "reason": (
                "scheduler submission stopped before task staging completed"
            ),
        }
    ]
    terminal_intent = original_load(
        campaign,
        state.phase.value,
        state.iteration,
    )
    assert terminal_intent["status"] == "FAILED"
    assert terminal_intent.get("expected_tasks") is None


def test_resume_reproves_unstaged_pre_submit_was_not_accepted(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state()
    state.phase = CampaignPhase.INITIAL_FEREBUS
    _write_locked_state(campaign, state)
    intent = submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=state.phase.value,
        iteration=state.iteration,
    )
    submission_intent.mark_failed(
        campaign,
        state.phase.value,
        state.iteration,
        "user_cancelled_before_scheduler_acceptance",
    )
    request, _ = install_stop_request(
        campaign,
        build_stop_request(
            state,
            mode="immediate",
            cancel_jobs=True,
        ),
    )
    request = update_stop_request(
        campaign,
        request["request_id"],
        status="requested",
        cancellation_summary={
            "cancelled": [
                {
                    "job_id": "",
                    "scheduler_identity_kind": "slurm",
                    "phases": [state.phase.value],
                    "intent_keys": [
                        {
                            "phase": state.phase.value,
                            "iteration": int(state.iteration),
                        }
                    ],
                    "n_completed": 0,
                    "task_count_unknown": True,
                    "reason": (
                        "scheduler submission stopped before task staging "
                        "completed"
                    ),
                }
            ],
            "skipped": [],
            "failed": [],
        },
    )
    assert request is not None
    lookup_calls = []

    class NoAcceptedJobBackend:
        def find_accounted_job_by_name(self, name, **kwargs):
            lookup_calls.append((name, kwargs))
            return SimpleNamespace(
                job_id=None,
                inconclusive=False,
                error=None,
            )

    started = []
    monkeypatch.setattr(
        cli_mod,
        "get_scheduler_backend",
        lambda _kind: NoAcceptedJobBackend(),
    )
    monkeypatch.setattr(
        cli_mod,
        "cmd_start",
        lambda args: (started.append(args) or 0),
    )

    assert (
        main(
            [
                "resume",
                "--campaign-dir",
                str(campaign),
                "--mode",
                "dry_run",
                "--foreground",
            ]
        )
        == 0
    )

    assert len(lookup_calls) == 1
    assert lookup_calls[0][0] == intent["expected_job_name"]
    assert lookup_calls[0][1]["expected_task_count"] is None
    assert started
    resumed = read_state(
        campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    )
    assert resumed.phase is CampaignPhase.INITIAL_FEREBUS
    assert resumed.shutdown_requested is False
    assert resumed.pending_jobs[state.phase.value] is None
    assert not stop_request_path(campaign).exists()
    assert (
        stop_request_history_dir(campaign)
        / (str(request["request_id"]) + ".json")
    ).is_file()


def test_resume_refuses_unstaged_pre_submit_when_scheduler_finds_job(
    tmp_path,
    monkeypatch,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state()
    state.phase = CampaignPhase.INITIAL_FEREBUS
    _write_locked_state(campaign, state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=state.phase.value,
        iteration=state.iteration,
    )
    submission_intent.mark_failed(
        campaign,
        state.phase.value,
        state.iteration,
        "user_cancelled_before_scheduler_acceptance",
    )
    request, _ = install_stop_request(
        campaign,
        build_stop_request(
            state,
            mode="immediate",
            cancel_jobs=True,
        ),
    )
    request = update_stop_request(
        campaign,
        request["request_id"],
        status="requested",
        cancellation_summary={
            "cancelled": [
                {
                    "job_id": "",
                    "scheduler_identity_kind": "slurm",
                    "phases": [state.phase.value],
                    "intent_keys": [
                        {
                            "phase": state.phase.value,
                            "iteration": int(state.iteration),
                        }
                    ],
                    "n_completed": 0,
                    "task_count_unknown": True,
                    "reason": (
                        "scheduler submission stopped before task staging "
                        "completed"
                    ),
                }
            ],
            "skipped": [],
            "failed": [],
        },
    )
    assert request is not None
    monkeypatch.setattr(
        cli_mod,
        "get_scheduler_backend",
        lambda _kind: SimpleNamespace(
            find_accounted_job_by_name=lambda *_args, **_kwargs: (
                SimpleNamespace(
                    job_id="12345",
                    inconclusive=False,
                    error=None,
                )
            )
        ),
    )
    monkeypatch.setattr(
        cli_mod,
        "cmd_start",
        lambda _args: pytest.fail("unsafe resume must not start the daemon"),
    )

    assert (
        main(
            [
                "resume",
                "--campaign-dir",
                str(campaign),
                "--mode",
                "dry_run",
                "--foreground",
            ]
        )
        == 7
    )
    assert "scheduler contains a job" in capsys.readouterr().err
    assert stop_request_path(campaign).is_file()


def test_cli_stop_cancel_jobs_records_ferebus_intent_for_daemon_cleanup(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    write_state(data / DEFAULT_STATE_FILENAME, state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
    )
    submission_intent.mark_submitted(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
        "456",
        expected_tasks=1,
    )
    expected_name = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )["expected_job_name"]
    cancelled = []
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [{"job_id": str(job_id), "state": "PENDING", "job_name": expected_name}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: (cancelled.append(str(job_id)) or (True, "")),
    )
    monkeypatch.setattr(
        cli_mod,
        "_confirm_cancelled_slurm_job",
        lambda *args, **kwargs: (True, ""),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])

    assert rc == 0
    assert cancelled == ["456"]
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUBMITTED"


def test_cli_stop_cancel_jobs_uses_intents_when_state_missing(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    phase = CampaignPhase.INITIAL_GAUSSIAN.value
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid="uid123456789",
        phase_name=phase,
        iteration=0,
    )
    submission_intent.mark_submitted(campaign, phase, 0, "999", expected_tasks=1)
    expected_name = submission_intent.load_intent(
        campaign,
        phase,
        0,
    )["expected_job_name"]
    cancelled = []
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [{"job_id": str(job_id), "state": "RUNNING", "job_name": expected_name}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: (cancelled.append(str(job_id)) or (True, "")),
    )
    monkeypatch.setattr(
        cli_mod,
        "_confirm_cancelled_slurm_job",
        lambda *args, **kwargs: (True, ""),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "state.json is missing" in captured.err
    assert cancelled == ["999"]
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "FAILED"
    assert intent["reason"] == "user_cancelled_via_stop"


def test_cli_stop_cancel_jobs_uses_intents_when_state_corrupt(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    (data / DEFAULT_STATE_FILENAME).write_text("{bad json", encoding="utf-8")
    phase = CampaignPhase.INITIAL_FEREBUS.value
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid="uid123456789",
        phase_name=phase,
        iteration=0,
    )
    submission_intent.mark_submitted(campaign, phase, 0, "1001", expected_tasks=1)
    expected_name = submission_intent.load_intent(
        campaign,
        phase,
        0,
    )["expected_job_name"]
    cancelled = []
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [{"job_id": str(job_id), "state": "RUNNING", "job_name": expected_name}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: (cancelled.append(str(job_id)) or (True, "")),
    )
    monkeypatch.setattr(
        cli_mod,
        "_confirm_cancelled_slurm_job",
        lambda *args, **kwargs: (True, ""),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "state.json invalid" in captured.err
    assert cancelled == ["1001"]
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "FAILED"
    assert intent["reason"] == "user_cancelled_via_stop"


def test_cli_stop_cancel_jobs_refuses_inconclusive_scheduler_lookup(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] = "789"
    write_state(data / DEFAULT_STATE_FILENAME, state)
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": False,
            "inconclusive": True,
            "rows": [],
            "error": "squeue unavailable",
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: pytest.fail("inconclusive lookup must not scancel"),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    err = capsys.readouterr().err

    assert rc == 10
    assert "squeue lookup inconclusive" in err
    stopped = read_state(data / DEFAULT_STATE_FILENAME)
    assert stopped.shutdown_requested is False
    assert stopped.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] == "789"


def test_cli_stop_cancel_jobs_skips_invalid_squeue_job_id(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] = "789"
    write_state(data / DEFAULT_STATE_FILENAME, state)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="slurm_load_jobs error: Invalid job id specified\n",
            )
        if cmd[0] == "scancel":
            pytest.fail("invalid squeue job id must not call scancel")
        raise AssertionError("unexpected command: " + repr(cmd))

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Slurm cancellation: 0 cancelled, 1 already inactive or not cancellable" in out
    stopped = read_state(data / DEFAULT_STATE_FILENAME)
    assert stopped.shutdown_requested is False
    assert stopped.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] == "789"


def test_cli_stop_cancel_jobs_refuses_campaign_job_name_mismatch(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    phase = CampaignPhase.INITIAL_GAUSSIAN.value
    state.pending_jobs[phase] = "321"
    write_state(data / DEFAULT_STATE_FILENAME, state)
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [{"job_id": str(job_id), "state": "RUNNING", "job_name": "other-campaign"}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: pytest.fail("job-name mismatch must not scancel"),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    err = capsys.readouterr().err

    assert rc == 10
    assert "scheduler job name mismatch" in err
    stopped = read_state(data / DEFAULT_STATE_FILENAME)
    assert stopped.pending_jobs[phase] == "321"


def test_cli_resume_explicitly_clears_shutdown_flag(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    request, _ = install_stop_request(
        campaign,
        build_stop_request(state, mode="immediate"),
    )
    complete_stop_request(campaign, request["request_id"], reason="immediate")
    state.shutdown_requested = True
    _write_locked_state(campaign, state)
    rc = main([
        "resume", "--campaign-dir", str(campaign),
        "--mode", "dry_run", "--max-ticks", "0", "--foreground",
    ])
    assert rc == 0
    s = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert s.shutdown_requested is False
    assert not stop_request_path(campaign).exists()
    assert (stop_request_history_dir(campaign) / (request["request_id"] + ".json")).is_file()


def test_cli_resume_archives_completed_stop_without_shutdown_flag(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=2)
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    _write_locked_state(campaign, state)
    request, _ = install_stop_request(
        campaign,
        build_stop_request(state, mode="immediate"),
    )
    complete_stop_request(campaign, request["request_id"], reason="immediate")
    started = []
    monkeypatch.setattr(
        cli_mod,
        "cmd_start",
        lambda args: started.append(args) or 0,
    )

    rc = main(
        [
            "resume",
            "--campaign-dir",
            str(campaign),
            "--mode",
            "dry_run",
        ]
    )

    assert rc == 0
    assert len(started) == 1
    assert not stop_request_path(campaign).exists()
    history = stop_request_history_dir(campaign) / (
        request["request_id"] + ".json"
    )
    assert history.is_file()
    assert json.loads(history.read_text())["archived_status"] == "resumed"


def test_cli_resume_retains_pending_iteration_stop_during_aimall_recovery(
    tmp_path,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=40)
    state.phase = CampaignPhase.AIMALL
    state.iteration = 14
    _write_locked_state(campaign, state)
    request, _ = install_stop_request(
        campaign,
        build_stop_request(state, mode="after_iteration"),
    )

    rc = main(
        [
            "resume",
            "--campaign-dir",
            str(campaign),
            "--mode",
            "dry_run",
            "--max-ticks",
            "0",
            "--foreground",
        ]
    )

    assert rc == 0
    retained = read_stop_request(
        campaign,
        expected_campaign_uid=state.campaign_uid,
    )
    assert retained is not None
    assert retained["request_id"] == request["request_id"]
    assert retained["status"] == "requested"
    assert retained["mode"] == "after_iteration"
    assert "active stop request retained" in capsys.readouterr().out


def test_cli_status_reports_pending_stop_request(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=2)
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    write_state(data / DEFAULT_STATE_FILENAME, state)
    install_stop_request(
        campaign,
        build_stop_request(state, mode="after_iteration"),
    )

    assert main(["status", "--campaign-dir", str(campaign), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stop_request"]["mode"] == "after_iteration"
    assert payload["stop_request"]["target_iteration"] == 1
    assert "_presentation_stop_disposition" not in payload
    assert payload["next_action"].startswith("run reconcile;")

    assert main(["status", "--campaign-dir", str(campaign)]) == 0
    out = capsys.readouterr().out
    assert "stop requested after iteration 1" in out
    assert "after_iteration" not in out
    assert "request_id" not in out
    assert "SEED_SELECT@" not in out


def test_reconcile_stop_guard_preserves_pending_boundary_bytes(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.AIMALL
    state.iteration = 2
    _write_locked_state(campaign, state)
    request, _ = install_stop_request(
        campaign,
        build_stop_request(state, mode="after_iteration"),
    )
    original = stop_request_path(campaign).read_bytes()

    guard = cli_mod._prepare_reconcile_stop_guard(campaign, state)

    assert guard["request"]["request_id"] == request["request_id"]
    assert guard["disposition"]["kind"] == "pending_boundary"
    assert stop_request_path(campaign).read_bytes() == original
    cli_mod._recheck_reconcile_stop_guard(campaign, guard, state)

    complete_stop_request(
        campaign,
        request["request_id"],
        reason="test_boundary",
    )
    with pytest.raises(ValueError, match="stop request changed"):
        cli_mod._recheck_reconcile_stop_guard(campaign, guard, state)


def test_reconcile_stop_guard_rejects_recovery_past_pending_boundary(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    observed = fresh_campaign_state(max_iterations=3)
    observed.phase = CampaignPhase.AIMALL
    observed.iteration = 2
    _write_locked_state(campaign, observed)
    install_stop_request(
        campaign,
        build_stop_request(observed, mode="after_iteration"),
    )
    proposed = CampaignState.from_dict(observed.to_dict())
    proposed.phase = CampaignPhase.SEED_SELECT
    proposed.iteration = 3

    with pytest.raises(StopControlError, match="precedes campaign iteration"):
        cli_mod._prepare_reconcile_stop_guard(campaign, proposed)


def test_reconcile_stop_guard_defers_cancelling_stop_to_scheduler_recovery(
    tmp_path,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 2
    _write_locked_state(campaign, state)
    request, _ = install_stop_request(
        campaign,
        build_stop_request(
            state,
            mode="immediate",
            cancel_jobs=True,
        ),
    )

    guard = cli_mod._prepare_reconcile_stop_guard(campaign, state)

    assert guard["request"]["request_id"] == request["request_id"]
    assert guard["disposition"]["kind"] == "cancelling"
    assert guard["mutable_during_reconcile"] is True


def test_reconcile_apply_refuses_malformed_stop_control(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    stop_request_path(campaign).write_text("{bad json", encoding="utf-8")

    rc = main(["reconcile", "--campaign-dir", str(campaign), "--apply"])

    assert rc == 9
    captured = capsys.readouterr()
    assert "stop-control metadata is invalid" in captured.err


def test_cli_start_background_spawns_child_without_shell(tmp_path, monkeypatch, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    calls = []
    real_popen = cli_mod.subprocess.Popen

    class FakePopen:
        pid = 4321

        def __new__(cls, argv, **kwargs):
            if cli_mod.BACKGROUND_STARTUP_PATH_ENV not in kwargs.get("env", {}):
                return real_popen(argv, **kwargs)
            return super().__new__(cls)

        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
            cli_mod.update_background_startup(
                Path(kwargs["env"][cli_mod.BACKGROUND_STARTUP_PATH_ENV]),
                kwargs["env"][cli_mod.BACKGROUND_LAUNCH_ID_ENV],
                state="ownership_acquired",
                stage="environment_transition",
                pid=self.pid,
            )

        def poll(self):
            return None

    monkeypatch.setattr(cli_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda seconds: None)

    rc = main([
        "start",
        "--campaign-dir",
        str(campaign),
        "--mode",
        "dry_run",
        "--max-ticks",
        "7",
        "--background",
    ])

    assert rc == 0
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:4] == [
        sys.executable,
        "-m",
        "ichor.hpc.active_learning.cli",
        "start",
    ]
    assert "--background" not in argv
    assert argv[argv.index("--mode") + 1] == "dry_run"
    assert "--max-ticks" in argv
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert "shell" not in kwargs
    assert kwargs["env"][cli_mod.BACKGROUND_CHILD_ENV] == "1"
    pid_path = campaign / DEFAULT_DATA_SUBDIR / cli_mod.BACKGROUND_PID_FILENAME
    assert not pid_path.exists(), "the parent must not publish the daemon-owned PID file"
    startup_path = campaign / DEFAULT_DATA_SUBDIR / cli_mod.BACKGROUND_STARTUP_FILENAME
    payload = json.loads(startup_path.read_text(encoding="utf-8"))
    assert payload["pid"] == 4321
    assert payload["schema_version"] == cli_mod.BACKGROUND_STARTUP_SCHEMA_VERSION
    assert payload["campaign_dir"] == str(campaign.resolve())
    assert payload["log_path"].endswith(cli_mod.BACKGROUND_LOG_FILENAME)
    assert payload["state"] == "ownership_acquired"
    assert "background process owns the campaign and is still starting" in capsys.readouterr().out


def test_cli_resume_background_clears_shutdown_and_spawns_resume(tmp_path, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.shutdown_requested = True
    _write_locked_state(campaign, state)
    calls = []
    real_popen = cli_mod.subprocess.Popen

    class FakePopen:
        pid = 4322

        def __new__(cls, argv, **kwargs):
            if cli_mod.BACKGROUND_STARTUP_PATH_ENV not in kwargs.get("env", {}):
                return real_popen(argv, **kwargs)
            return super().__new__(cls)

        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
            cli_mod.update_background_startup(
                Path(kwargs["env"][cli_mod.BACKGROUND_STARTUP_PATH_ENV]),
                kwargs["env"][cli_mod.BACKGROUND_LAUNCH_ID_ENV],
                state="ownership_acquired",
                stage="environment_transition",
                pid=self.pid,
            )

        def poll(self):
            return None

    monkeypatch.setattr(cli_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda seconds: None)

    rc = main([
        "resume",
        "--campaign-dir",
        str(campaign),
        "--mode",
        "dry_run",
        "--background",
    ])

    assert rc == 0
    assert read_state(data / DEFAULT_STATE_FILENAME).shutdown_requested is False
    argv, _kwargs = calls[0]
    assert argv[3] == "resume"
    assert "--background" not in argv


def test_cli_background_wait_timeout_leaves_live_child_running(
    tmp_path,
    monkeypatch,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    config.runtime.background_readiness_timeout_seconds = 1
    config.to_yaml(campaign / "campaign.yaml")
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    real_popen = cli_mod.subprocess.Popen

    class FakePopen:
        pid = 4330

        def __new__(cls, argv, **kwargs):
            if cli_mod.BACKGROUND_STARTUP_PATH_ENV not in kwargs.get("env", {}):
                return real_popen(argv, **kwargs)
            return super().__new__(cls)

        def __init__(self, argv, **kwargs):
            pass

        def poll(self):
            return None

        def terminate(self):
            raise AssertionError("a readiness timeout must not terminate the child")

        def kill(self):
            raise AssertionError("a readiness timeout must not kill the child")

    monkeypatch.setattr(cli_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda seconds: None)

    rc = main([
        "start",
        "--campaign-dir",
        str(campaign),
        "--mode",
        "dry_run",
        "--background",
    ])

    assert rc == 0
    payload = json.loads(
        (data / cli_mod.BACKGROUND_STARTUP_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["state"] == "spawned"
    assert payload["pid"] == 4330
    out = capsys.readouterr().out
    assert "continues in background" in out
    assert "was not terminated" in out


def test_cli_background_child_exit_before_acknowledgement_is_failure(
    tmp_path,
    monkeypatch,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    real_popen = cli_mod.subprocess.Popen

    class FakePopen:
        pid = 4331

        def __new__(cls, argv, **kwargs):
            if cli_mod.BACKGROUND_STARTUP_PATH_ENV not in kwargs.get("env", {}):
                return real_popen(argv, **kwargs)
            return super().__new__(cls)

        def __init__(self, argv, **kwargs):
            pass

        @staticmethod
        def poll():
            return 12

    monkeypatch.setattr(cli_mod.subprocess, "Popen", FakePopen)

    rc = main([
        "start",
        "--campaign-dir",
        str(campaign),
        "--mode",
        "dry_run",
        "--background",
    ])

    assert rc == 12
    payload = json.loads(
        (data / cli_mod.BACKGROUND_STARTUP_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["state"] == "failed"
    assert payload["exit_code"] == 12
    assert "exited during startup" in capsys.readouterr().err


def test_cli_background_refuses_recursive_child(tmp_path, monkeypatch, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    monkeypatch.setenv(cli_mod.BACKGROUND_CHILD_ENV, "1")

    rc = main([
        "start",
        "--campaign-dir",
        str(campaign),
        "--mode",
        "dry_run",
        "--background",
    ])

    assert rc == 2
    assert "background child" in capsys.readouterr().err


def test_cli_background_refuses_live_pid_file(tmp_path, monkeypatch, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    pid_path = data / cli_mod.BACKGROUND_PID_FILENAME
    pid_path.write_text(
        json.dumps({"pid": 99999, "schema_version": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_mod, "_pid_is_alive", lambda pid: True)

    rc = main([
        "start",
        "--campaign-dir",
        str(campaign),
        "--mode",
        "dry_run",
        "--background",
    ])

    assert rc == 8
    assert "still alive" in capsys.readouterr().err


def test_cli_background_refuses_live_startup_record_without_pid_file(
    tmp_path,
    monkeypatch,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    cli_mod.initialise_background_startup(
        data / cli_mod.BACKGROUND_STARTUP_FILENAME,
        {
            "launch_id": "still-starting",
            "state": "spawned",
            "stage": "backend_preflight",
            "pid": 99998,
            "campaign_dir": str(campaign.resolve()),
        },
    )
    monkeypatch.setattr(cli_mod, "_pid_is_alive", lambda pid: True)

    rc = main([
        "start",
        "--campaign-dir",
        str(campaign),
        "--mode",
        "dry_run",
        "--background",
    ])

    assert rc == 8
    assert "still alive" in capsys.readouterr().err


def test_cli_status_reports_background_pid_metadata(tmp_path, capsys, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())
    pid_path = data / cli_mod.BACKGROUND_PID_FILENAME
    log_path = data / cli_mod.BACKGROUND_LOG_FILENAME
    pid = 12345
    pid_path.write_text(
        json.dumps({
            "schema_version": 1,
            "pid": pid,
            "log_path": str(log_path),
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_mod, "_pid_is_alive", lambda value: True)

    rc = main(["status", "--campaign-dir", str(campaign), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["background_pid"] == pid
    assert payload["background_pid_alive"] is True
    assert payload["background_log_path"] == str(log_path)


def test_cli_immediate_stop_signals_authenticated_background_startup(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    pid = 12346
    (data / cli_mod.BACKGROUND_PID_FILENAME).write_text(
        str(pid) + "\n",
        encoding="utf-8",
    )
    cli_mod.initialise_background_startup(
        data / cli_mod.BACKGROUND_STARTUP_FILENAME,
        {
            "launch_id": "launch-stop-test",
            "state": "starting",
            "stage": "backend_preflight",
            "pid": pid,
            "campaign_dir": str(campaign.resolve()),
            "log_path": str(data / cli_mod.BACKGROUND_LOG_FILENAME),
        },
    )
    signals = []
    monkeypatch.setattr(cli_mod, "_pid_is_alive", lambda value: int(value) == pid)
    monkeypatch.setattr(
        cli_mod.os,
        "kill",
        lambda target, signum: signals.append((target, signum)),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--immediate", "--verbose"])

    assert rc == 0
    assert signals == [(pid, cli_mod.signal.SIGTERM)]
    assert "SIGTERM sent after durable immediate-stop request" in capsys.readouterr().out


def test_cli_stop_when_no_state_returns_4(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["stop", "--campaign-dir", str(campaign)])
    assert rc == 4


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"event": "queue_lifecycle_update", "status": "FAILED"}, "FAIL"),
        ({"event": "queue_lifecycle_update", "status": "COMPLETED"}, "OK"),
        ({"event": "queue_lifecycle_update", "status": "PENDING"}, "WAIT"),
        ({"event": "queue_lifecycle_update", "status": "RUNNING"}, "RUN"),
        (
            {
                "event": "queue_lifecycle_update",
                "queue_event": "first_sacct",
                "status": "FAILED",
            },
            "OK",
        ),
        ({"event": "failure_action", "action": "HALT"}, "FAIL"),
        ({"event": "failure_action", "action": "RETRY"}, "WARN"),
        ({"event": "quantum_quality_summary", "accepted": False}, "FAIL"),
        (
            {
                "event": "quantum_quality_summary",
                "accepted": True,
                "n_total": 3,
                "n_rejected": 1,
            },
            "WARN",
        ),
        (
            {
                "event": "ferebus_quality_summary",
                "accepted": True,
                "n_total": 4,
                "n_rejected": 0,
            },
            "OK",
        ),
        ({"event": "sacct_error"}, "WARN"),
        ({"event": "sacct_error_timeout"}, "FAIL"),
        ({"event": "campaign_completed"}, "OK"),
        ({"event": "phase_completion_replayed"}, "WARN"),
        ({"event": "user_stop_request_cancelled"}, "OK"),
        ({"event": "user_cancelled_jobs", "n_failed": 1}, "FAIL"),
        ({"event": "user_cancelled_jobs", "n_skipped": 1}, "WARN"),
        ({"event": "user_cancelled_jobs", "n_cancelled": 2}, "OK"),
        (
            {
                "event": "queue_lifecycle_update",
                "queue_event": "terminal",
                "status": "CANCELLED",
                "user_requested_cancellation": True,
            },
            "OK",
        ),
    ],
)
def test_journal_event_severity_uses_event_semantics(event, expected):
    assert cli_mod._journal_event_severity(event) == expected


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            {
                "event": "queue_lifecycle_update",
                "queue_event": "terminal",
                "status": "COMPLETED",
            },
            "Slurm tasks completed",
        ),
        (
            {
                "event": "queue_lifecycle_update",
                "queue_event": "postprocess_started",
                "status": "started",
            },
            "local postprocessing started",
        ),
        (
            {
                "event": "queue_lifecycle_update",
                "queue_event": "postprocess_finished",
                "status": "failed",
            },
            "local postprocessing failed",
        ),
        (
            {
                "event": "queue_lifecycle_update",
                "queue_event": "terminal",
                "status": "CANCELLED",
                "user_requested_cancellation": True,
            },
            "Slurm work cancelled as requested",
        ),
    ],
)
def test_journal_queue_lifecycle_names_scheduler_and_local_work_separately(
    event,
    expected,
):
    assert cli_mod._journal_operator_summary(event) == expected


def test_journal_uses_plain_config_and_stop_lifecycle_labels():
    assert (
        cli_mod.JOURNAL_EVENT_LABELS["effective_config_diff"]
        == "effective configuration recorded"
    )
    assert (
        cli_mod.JOURNAL_EVENT_LABELS["user_stop_resumed"]
        == "stop cleared for resume"
    )


def test_journal_reconcile_summary_includes_applied_config_changes():
    summary = cli_mod._journal_operator_summary(
        {
            "event": "reconcile_applied",
            "n_allowed_config_changes": 6,
        }
    )

    assert summary == "reconcile applied; 6 configuration changes applied"


def test_journal_reconcile_summary_reports_ariadne_reuse_without_resubmission():
    event = {
        "event": "reconcile_applied",
        "phase": CampaignPhase.PHASE_B_DIVERSITY.value,
        "iteration": 8,
        "ariadne_expected_tasks": 200,
        "ariadne_accepted_tasks": 193,
        "ariadne_rejected_tasks": 7,
        "ariadne_missing_rejected_outputs": 7,
        "ariadne_tasks_resubmitted": 0,
    }

    output = cli_mod._format_journal_events([event], verbose=False)

    assert "RECONCILE" in output
    assert "iter=8" in output
    assert (
        "reconcile applied; reusing 193 accepted ARIADNE results, "
        "7 rejected tasks excluded, no ARIADNE jobs resubmitted"
    ) in output


def test_journal_reconcile_summary_reports_aimall_local_reuse():
    event = {
        "event": "reconcile_applied",
        "phase": CampaignPhase.AIMALL.value,
        "iteration": 14,
        "aimall_completed_outputs": 149,
        "aimall_tasks_resubmitted": 0,
    }

    output = cli_mod._format_journal_events([event], verbose=False)

    assert "RECONCILE" in output
    assert "iter=14" in output
    assert (
        "reconcile applied; reusing 149 completed AIMAll outputs for local "
        "validation, no AIMAll tasks resubmitted"
    ) in output


def test_reconcile_human_summary_reports_phase_b_ariadne_reuse(capsys):
    report = SimpleNamespace(
        ariadne_results_recovery={
            "expected_tasks": 200,
            "accepted_tasks": 193,
            "rejected_tasks": 7,
            "missing_rejected_outputs": 7,
            "tasks_resubmitted": 0,
        }
    )

    cli_mod._print_reconcile_ariadne_reuse(report)

    output = capsys.readouterr().out
    assert "ARIADNE Handoff" in output
    assert re.search(r"accepted results\s+: 193", output)
    assert re.search(r"rejected tasks\s+: 7 excluded from Phase B", output)
    assert re.search(r"missing rejected outputs\s+: 7", output)
    assert re.search(r"ARIADNE tasks to resubmit\s*: 0", output)


def test_journal_completed_iteration_stop_describes_the_paused_boundary():
    stop_event = {
        "event": "user_stop_boundary_reached",
        "mode": "after_iteration",
        "target_iteration": 1,
        "resulting_phase": CampaignPhase.SEED_SELECT.value,
        "resulting_iteration": 2,
    }
    transition_event = {
        "event": "phase_transition",
        "from_phase": CampaignPhase.STOP_CHECK.value,
        "to_phase": CampaignPhase.SEED_SELECT.value,
        "iteration": 2,
    }

    output = cli_mod._format_journal_events(
        [stop_event, transition_event],
        verbose=False,
    )

    assert (
        "iteration 1 completed; campaign paused before SEED_SELECT iteration 2"
        in output
    )
    assert "next phase recorded for resume: SEED_SELECT" in output
    assert "phase changed from STOP_CHECK" not in output


def _intentionally_stopped_seed_select_state():
    state = fresh_campaign_state(max_iterations=40)
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 2
    state.shutdown_requested = True
    state.lifecycle_context = make_lifecycle_context(
        disposition="stopped",
        reason_code="user_stop_boundary_reached",
        message="stopped after iteration 1",
        from_phase=CampaignPhase.STOP_CHECK,
        iteration=2,
        source="daemon_stop_control",
        recovery_action="use resume to continue",
        details={"mode": "after_iteration", "target_iteration": 1},
    )
    return state


def test_reconcile_defers_environment_transition_for_completed_stop(
    tmp_path,
    monkeypatch,
):
    state = _intentionally_stopped_seed_select_state()

    def unexpected_backend_check():
        raise AssertionError("stopped reconcile must not run backend preflight")

    monkeypatch.setattr(cli_mod, "check_backends", unexpected_backend_check)

    transition, deferred = cli_mod._advance_environment_after_reconcile(
        tmp_path,
        state,
        CampaignConfig(),
    )

    assert transition is None
    assert deferred is True


def test_reconcile_stopped_state_is_not_runnable_and_hides_resolved_halt(
    tmp_path,
    capsys,
):
    state = _intentionally_stopped_seed_select_state()
    report = SimpleNamespace(
        proposed_state=state,
        source_state_phase=CampaignPhase.SEED_SELECT.value,
        last_halt_event={
            "event": "halt",
            "from_phase": CampaignPhase.ALLOCATION_CHECK.value,
            "iteration": 1,
            "reason": "resolved allocation failure",
        },
        unsafe_reasons=[],
        blocking_artifacts=[],
        active_submission_intents=[],
        decision="STOPPED: existing user stop request remains authoritative",
    )
    contract = {
        "contract_ok": True,
        "selected_phase": CampaignPhase.SEED_SELECT.value,
        "missing_or_invalid_inputs": [],
    }

    payload = cli_mod._reconcile_decision_payload(tmp_path, report, contract)
    cli_mod._print_reconcile_recovery_target(report, contract)
    cli_mod._print_reconcile_last_failure_compact(report)
    concise = capsys.readouterr().out

    assert payload["runnable"] is False
    assert payload["next_command"].startswith("ichor-al-daemon resume ")
    assert "intentionally paused after iteration 1" in payload["why_not_runnable"][-1]
    assert "Paused Campaign" in concise
    assert "paused after iteration 1" in concise
    assert "Last Failure" not in concise

    cli_mod._print_reconcile_last_failure_compact(report, verbose=True)
    verbose = capsys.readouterr().out
    assert "Resolved Historical Failure" in verbose
    assert "resolved allocation failure" in verbose


def test_reconcile_applied_report_explains_deferred_environment_check(
    tmp_path,
    capsys,
):
    state = _intentionally_stopped_seed_select_state()
    report = SimpleNamespace(
        proposed_state=state,
        unsafe_reasons=[],
        blocking_artifacts=[],
        active_submission_intents=[],
        decision="STOPPED: existing user stop request remains authoritative",
    )
    contract = {
        "contract_ok": True,
        "selected_phase": CampaignPhase.SEED_SELECT.value,
        "missing_or_invalid_inputs": [],
    }

    cli_mod._print_reconcile_applied_operator_report(
        tmp_path,
        report,
        contract,
        backup_path=None,
        applied_proposal_path=None,
        removed=[],
        removed_model_staging=[],
        archived_scripts=[],
        archived=[str(tmp_path / ".DATA" / "STAGING.archived-test")],
        archived_reference_data_staging=[],
        restored_bootstrap_handoff=[],
        environment_transition_deferred=True,
    )
    output = capsys.readouterr().out

    assert "Apply result: completed" in output
    assert "will be checked when resume clears the completed stop" in output
    assert "paused after iteration 1" in output
    assert "Slurm jobs submitted: none" in output
    assert "ichor-al-daemon resume" in output


def test_reconcile_preview_offers_explicit_staging_archive_command(
    tmp_path,
    capsys,
):
    report = SimpleNamespace(
        proposed_state=_intentionally_stopped_seed_select_state(),
        unsafe_reasons=[".DATA/STAGING is non-empty"],
        blocking_artifacts=[".DATA/STAGING"],
        active_submission_intents=[],
    )
    contract = {
        "contract_ok": True,
        "selected_phase": CampaignPhase.SEED_SELECT.value,
        "missing_or_invalid_inputs": [],
    }

    assert cli_mod._reconcile_result_label(report, contract) == "safe to apply"
    cli_mod._print_reconcile_apply_plan_compact(
        tmp_path,
        report,
        contract,
        proposed_state_path=tmp_path / ".DATA" / "state.json.proposed",
    )
    output = capsys.readouterr().out

    assert "--archive-staging --apply" in output
    assert "inspect blockers before restarting" not in output


def test_reconcile_preview_retires_completed_staging_without_archive_flag(
    tmp_path,
    capsys,
):
    report = SimpleNamespace(
        proposed_state=_intentionally_stopped_seed_select_state(),
        unsafe_reasons=[],
        blocking_artifacts=[],
        active_submission_intents=[],
        completed_staging_retirement={
            "eligible": [
                {
                    "context": "active",
                    "iteration": 8,
                    "action": "delete",
                }
            ]
        },
    )
    contract = {
        "contract_ok": True,
        "selected_phase": CampaignPhase.SEED_SELECT.value,
        "missing_or_invalid_inputs": [],
    }

    cli_mod._print_reconcile_completed_staging(report)
    cli_mod._print_reconcile_apply_plan_compact(
        tmp_path,
        report,
        contract,
        proposed_state_path=tmp_path / ".DATA" / "state.json.proposed",
    )
    output = capsys.readouterr().out

    assert "iteration 8: delete duplicate-only residue" in output
    assert "Slurm work submitted: none" in output
    assert "--archive-staging" not in output
    assert "--apply" in output


def test_reconcile_preview_uses_one_coherent_paused_campaign_report(
    tmp_path,
    capsys,
):
    state = fresh_campaign_state(max_iterations=40, campaign_uid="presentation-test")
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 9
    state.shutdown_requested = True
    state.lifecycle_context = make_lifecycle_context(
        disposition="stopped",
        reason_code="user_stop_boundary_reached",
        message="stopped after iteration 8",
        from_phase=CampaignPhase.STOP_CHECK,
        iteration=9,
        source="daemon_stop_control",
        recovery_action="use resume to continue",
        details={"mode": "after_iteration", "target_iteration": 8},
    )
    data = tmp_path / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True)
    write_state(data / DEFAULT_STATE_FILENAME, state)
    report = SimpleNamespace(
        proposed_state=state,
        unsafe_reasons=["dangling model staging directories exist"],
        blocking_artifacts=["dangling model staging"],
        active_submission_intents=[],
        decision="STOPPED: existing user stop request remains authoritative",
        completed_staging_retirement={
            "eligible": [
                {"context": "active", "iteration": 8, "action": "delete"}
            ],
            "pending_tombstones": [],
        },
    )
    contract = {
        "contract_ok": True,
        "selected_phase": CampaignPhase.SEED_SELECT.value,
        "missing_or_invalid_inputs": [],
        "protected_artifacts": [],
    }
    config_review = SimpleNamespace(
        allowed_changes=[
            SimpleNamespace(
                path="campaign.sampling_aggressiveness",
                old=5,
                new=6,
                reason="editable before seed selection",
            )
        ],
        blocked_changes=[],
    )

    cli_mod._print_reconcile_operator_report(
        tmp_path,
        report,
        contract,
        mode="dry_run",
        proposed_state_path=data / "state.json.proposed",
        config_review=config_review,
    )
    output = capsys.readouterr().out

    assert output.count("ICHOR Reconcile") == 1
    assert "Preview result: ready to apply" in output
    assert "status       : paused after iteration 8" in output
    assert "continue from: ARIADNE seed selection, iteration 9" in output
    assert "configuration changed" in output
    assert "completed temporary staging remains" in output
    assert "unfinished temporary model output remains" in output
    assert "delete temporary iteration-8 data already committed" in output
    assert "sampling aggressiveness 5 -> 6" in output
    assert "model staging" in output
    assert "Slurm work" in output and "reconcile will not start jobs" in output
    assert "committed QM data" in output and "unchanged" in output
    assert "daemon                 : remains stopped" in output
    assert "automatic campaign work: none" in output
    assert "--apply" in output
    assert "Recovery Safety" not in output
    assert "state.json.proposed" not in output


def test_reconcile_preview_clean_pause_recommends_resume_without_apply(
    tmp_path,
    capsys,
):
    state = _intentionally_stopped_seed_select_state()
    data = tmp_path / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True)
    write_state(data / DEFAULT_STATE_FILENAME, state)
    report = SimpleNamespace(
        proposed_state=state,
        unsafe_reasons=[],
        blocking_artifacts=[],
        active_submission_intents=[],
        decision="STOPPED: existing user stop request remains authoritative",
    )
    contract = {
        "contract_ok": True,
        "selected_phase": CampaignPhase.SEED_SELECT.value,
        "missing_or_invalid_inputs": [],
        "protected_artifacts": [],
    }

    cli_mod._print_reconcile_operator_report(
        tmp_path,
        report,
        contract,
        mode="dry_run",
        proposed_state_path=data / "state.json.proposed",
    )
    output = capsys.readouterr().out

    assert "Preview result: no reconcile changes needed" in output
    assert "Planned changes" not in output
    assert "ichor-al-daemon resume" in output
    assert "--apply" not in output


def test_reconcile_preview_describes_rebuildable_allocation_sample(
    tmp_path,
    capsys,
    monkeypatch,
):
    state = fresh_campaign_state(campaign_uid="allocation-recovery")
    state.phase = CampaignPhase.ALLOCATION_CHECK
    state.iteration = 14
    state.reference_data_version = 13
    state.validation_set_version = 13
    state.models_version = 13
    state.replacement_round = 1
    data = tmp_path / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True)
    write_state(data / DEFAULT_STATE_FILENAME, state)
    report = SimpleNamespace(
        proposed_state=state,
        unsafe_reasons=[],
        blocking_artifacts=[],
        active_submission_intents=[],
        decision="ALLOCATION_CHECK: pending replacement allocation needs sample repair",
    )
    contract = {
        "contract_ok": True,
        "selected_phase": CampaignPhase.ALLOCATION_CHECK.value,
        "missing_or_invalid_inputs": [],
        "protected_artifacts": [],
    }
    monkeypatch.setattr(
        "ichor.hpc.active_learning.execution_identity."
        "inspect_allocation_check_transition_boundary",
        lambda *_args, **_kwargs: {
            "safe": True,
            "pending_tasks": 1,
            "replacement_sample_state": "missing_rebuildable",
        },
    )

    cli_mod._print_reconcile_operator_report(
        tmp_path,
        report,
        contract,
        mode="dry_run",
        proposed_state_path=data / "state.json.proposed",
    )
    output = capsys.readouterr().out

    assert "Preview result: no reconcile changes needed" in output
    assert "can rebuild the missing replacement sample" in output
    assert "continue with 1 pending replacement task" in output
    assert "--apply" not in output


def test_preflight_describes_completed_stop_as_pause_not_backend_failure(tmp_path):
    payload = {
        "ready": False,
        "campaign_state": {
            "ok": False,
            "error": (
                "ValueError: campaign is intentionally paused; resume clears "
                "the completed stop"
            ),
        },
    }

    action, command = cli_mod._preflight_launch_advice(tmp_path, payload)
    payload["next_action"] = action
    payload["_presentation_next_command"] = command
    output = cli_mod._format_preflight(payload)

    assert action == "resume the campaign to clear the completed stop"
    assert command is not None and "ichor-al-daemon resume" in command
    assert cli_mod._preflight_failure_details(payload) == [action]
    assert "[WARN] campaign pause" in output
    assert "fix campaign state readiness" not in output


def test_cli_journal_json_prints_filtered_events(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(journal, "alpha", x=1, ts="2026-06-27T14:32:10.123456+00:00")
    append_event(journal, "beta", x=2)
    append_event(journal, "alpha", x=3)
    rc = main([
        "journal", "--campaign-dir", str(campaign),
        "--event-type", "alpha", "--json",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    assert [payload["event"] for payload in payloads] == ["alpha", "alpha"]
    assert payloads[0]["ts"] == "2026-06-27T14:32:10.123456+00:00"


def test_cli_journal_default_prints_readable_events(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(
        journal,
        "alpha",
        phase="INITIAL_GAUSSIAN",
        iteration=0,
        job_id="111",
        ts="2026-06-27T14:32:10.123456+00:00",
    )
    append_event(
        journal,
        "beta",
        phase="INITIAL_AIMALL",
        iteration=1,
        n_completed=2,
        ts="2026-06-27T14:42:55+00:00",
    )
    append_event(
        journal,
        "alpha",
        phase="FEREBUS",
        ts="2026-06-27T14:43:02Z",
    )

    rc = main(["journal", "--campaign-dir", str(campaign), "--last-n", "2"])

    assert rc == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines[0] == "Timeline"
    assert len(lines) == 3
    assert "alpha" in out
    assert "beta" in out
    assert "2026-06-27 14:42:55" in out
    assert "2026-06-27 14:43:02" in out
    assert "[INFO]" in lines[1]
    assert "[INFO]" in lines[2]
    assert "iter=1" in lines[1]
    assert "iter=n/a" in lines[2]
    assert lines[1].count("iter=1") == 1
    assert all(not line.startswith("  raw_event:") for line in lines)
    assert not out.lstrip().startswith("{")


def test_cli_journal_left_justifies_columns_for_long_events(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(
        journal,
        "sbatch",
        phase="PHASE_A_DIVERSITY",
        iteration=0,
        job_id="16175189",
        expected_tasks=1,
        ts="2026-06-27T14:32:10.123456+00:00",
    )
    append_event(
        journal,
        "sacct_rows_missing_but_squeue_active",
        phase="INITIAL_GAUSSIAN",
        iteration=12,
        job_id="16175294",
        n_expected=4,
        n_observed=2,
        n_missing=2,
        streak=1,
        squeue_rows_sample=[{"job_id": "16175294", "state": "PD"}],
        ts="2026-06-27T14:42:55+00:00",
    )

    rc = main(["journal", "--campaign-dir", str(campaign)])

    assert rc == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[0] == "Timeline"
    assert len(lines) == 3
    assert lines[1].startswith("  2026-06-27 14:32:10")
    assert lines[2].startswith("  2026-06-27 14:42:55")
    assert lines[1].index("iter=0") == lines[2].index("iter=12")
    assert lines[1].index("PHASE_A_DIVERSITY") == lines[2].index("INITIAL_GAUSSIAN")
    assert lines[1].index("job submitted") == lines[2].index("waiting for accounting")
    context_width = len(CampaignPhase.PHASE_A_DIVERSITY.value)
    assert cli_mod._JOURNAL_CONTEXT_WIDTH == context_width
    assert (
        lines[1].index("iter=0") - lines[1].index("PHASE_A_DIVERSITY")
        == context_width + 3
    )
    description_start = lines[2].index("waiting for accounting")
    iteration_end = lines[2].index("iter=12") + len("iter=12")
    assert lines[2][iteration_end:description_start] == " " * 6
    assert "[RUN]" in lines[1]
    assert "[WAIT]" in lines[2]
    assert "job=16175189" in lines[1]
    assert "job=16175294" in lines[2]
    assert "sacct_rows_missing_but_squeue_active" not in lines[2]
    assert "Slurm job is still active" not in lines[2]
    assert "total=4 completed=- running=- pending=-" in lines[2]
    assert "missing=2" in lines[2]


def test_journal_context_aliases_cover_every_overwidth_phase():
    expected = {
        "INITIAL_REPLACEMENT_GAUSSIAN": "INIT_REPL_GAUSS",
        "INITIAL_REPLACEMENT_AIMALL": "INIT_REPL_AIMALL",
        "INITIAL_ALLOCATION_CHECK": "INIT_ALLOC_CHECK",
        "REPLACEMENT_GAUSSIAN": "REPL_GAUSSIAN",
        "REPLACEMENT_AIMALL": "REPL_AIMALL",
    }
    assert cli_mod._JOURNAL_PHASE_CONTEXT_ALIASES == expected
    assert set(expected) == {
        phase.value
        for phase in CampaignPhase
        if len(phase.value) > cli_mod._JOURNAL_CONTEXT_WIDTH
    }
    assert max(len(value) for value in cli_mod._JOURNAL_RENDERED_CONTEXTS) == (
        cli_mod._JOURNAL_CONTEXT_WIDTH
    )


@pytest.mark.parametrize(
    ("phase", "alias"),
    [
        ("INITIAL_REPLACEMENT_GAUSSIAN", "INIT_REPL_GAUSS"),
        ("INITIAL_REPLACEMENT_AIMALL", "INIT_REPL_AIMALL"),
        ("INITIAL_ALLOCATION_CHECK", "INIT_ALLOC_CHECK"),
        ("REPLACEMENT_GAUSSIAN", "REPL_GAUSSIAN"),
        ("REPLACEMENT_AIMALL", "REPL_AIMALL"),
    ],
)
def test_journal_renders_short_phase_context_aliases(phase, alias):
    event = {
        "event": "phase_activity_progress",
        "phase": phase,
        "iteration": 12,
        "stage": "scheduler_wait",
    }

    assert cli_mod._event_context(event) == alias
    line = cli_mod._format_journal_events([event], verbose=False).splitlines()[1]
    assert alias in line
    assert phase not in line
    assert line.index("iter=12") - line.index(alias) == (
        cli_mod._JOURNAL_CONTEXT_WIDTH + 3
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12.49, 12),
        (12.5, 13),
        (12.99, 13),
        (59.6, 60),
        (3599.6, 3600),
    ],
)
def test_journal_rounds_finite_seconds_half_up(value, expected):
    assert cli_mod._round_journal_seconds(value) == expected


def test_journal_rounds_time_details_and_hides_all_throughput():
    event = {
        "event": "phase_activity_progress",
        "phase": "AIMALL",
        "iteration": 2,
        "stage": "scientific_quality",
        "status": "running",
        "elapsed_seconds": 59.6,
        "stage_elapsed_seconds": 12.5,
        "throughput": 0.5,
        "throughput_per_second": 0.75,
        "throughput_per_s": 0.25,
    }

    default = cli_mod._format_journal_events([event], verbose=False)
    verbose = cli_mod._format_journal_events([event], verbose=True)

    assert "elapsed_s=60" in default
    assert "stage_elapsed_seconds=13" in verbose
    assert "elapsed_seconds=60" in verbose
    for output in (default, verbose):
        assert "throughput" not in output
        assert "frames/s" not in output
        assert "modes/s" not in output
        assert "seeds/s" not in output
    assert event["elapsed_seconds"] == 59.6
    assert event["stage_elapsed_seconds"] == 12.5
    assert event["throughput"] == 0.5


def test_journal_malformed_seconds_remain_safely_renderable():
    assert (
        cli_mod._format_journal_detail_value("elapsed_seconds", "unknown")
        == "unknown"
    )


def test_journal_only_renders_exact_non_negative_iterations():
    assert cli_mod._event_iteration({"iteration": 0}) == "iter=0"
    assert cli_mod._event_iteration({"iteration": "2"}) == "iter=n/a"
    assert cli_mod._event_iteration({"iteration": True}) == "iter=n/a"
    assert cli_mod._event_iteration({"iteration": -1}) == "iter=n/a"


def test_cli_journal_formats_array_progress_tuple_with_squeue_counts(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(
        journal,
        "sacct_rows_missing_but_squeue_active",
        phase="INITIAL_GAUSSIAN",
        iteration=0,
        job_id="1840120",
        n_expected=70,
        n_completed=50,
        n_failed=0,
        n_missing=10,
        squeue_state_counts={"R": 10, "PD": 10},
        ts="2026-07-03T20:15:05+00:00",
    )

    rc = main(["journal", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Timeline\n" in out
    assert "[WAIT]" in out
    assert "waiting for accounting" in out
    assert "job=1840120" in out
    assert "total=70 completed=50 running=10 pending=10" in out
    assert "missing=10" in out


def test_cli_journal_filtered_empty_prints_timeline_message(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(journal, "alpha", ts="2026-06-27T14:32:10+00:00")

    rc = main([
        "journal",
        "--campaign-dir",
        str(campaign),
        "--event-type",
        "halt",
    ])

    assert rc == 0
    assert capsys.readouterr().out == "Timeline\n  no matching events\n"


def test_cli_journal_invalid_since_is_not_reported_as_corruption(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    append_event(data / "journal.ndjson", "daemon_started")

    rc = main(
        ["journal", "--campaign-dir", str(campaign), "--since", "not-a-time"]
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert "invalid --since timestamp" in captured.err
    assert "journal is malformed" not in captured.err


def test_cli_journal_rejects_non_positive_last_n_during_argument_parsing():
    with pytest.raises(SystemExit) as exc_info:
        main(["journal", "--last-n", "0"])
    assert exc_info.value.code == 2


def test_cli_journal_aggregates_repetitive_seed_events_by_default(
    tmp_path,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    journal = data / "journal.ndjson"
    for seed in (1, 2, 3):
        append_event(
            journal,
            "ariadne_task_salvaged_from_nonzero_exit",
            phase=CampaignPhase.ARIADNE_ARRAY.value,
            iteration=1,
            seed=seed,
            reason="usable output retained",
        )

    assert main(["journal", "--campaign-dir", str(campaign)]) == 0
    default_output = capsys.readouterr().out
    assert default_output.count("ARIADNE task salvaged") == 1
    assert "events=3" in default_output

    assert main(
        ["journal", "--campaign-dir", str(campaign), "--verbose"]
    ) == 0
    verbose_output = capsys.readouterr().out
    assert verbose_output.count("ARIADNE task salvaged") == 3


@pytest.mark.parametrize("machine_flag", ["--json", "--raw"])
def test_cli_journal_machine_output_keeps_raw_event_payload(
    tmp_path,
    capsys,
    machine_flag,
):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(
        journal,
        "sacct_rows_missing_but_squeue_active",
        phase="INITIAL_REPLACEMENT_GAUSSIAN",
        iteration=12,
        job_id="16175294",
        elapsed_seconds=12.5,
        throughput=0.75,
        ts="2026-06-27T14:42:55+00:00",
    )

    rc = main(["journal", "--campaign-dir", str(campaign), machine_flag])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["event"] == "sacct_rows_missing_but_squeue_active"
    assert payload["phase"] == "INITIAL_REPLACEMENT_GAUSSIAN"
    assert payload["elapsed_seconds"] == 12.5
    assert payload["throughput"] == 0.75


def test_cli_journal_unknown_event_gets_readable_fallback_label(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(
        journal,
        "custom_scheduler_note",
        phase="PHASE_A_DIVERSITY",
        iteration=0,
        ts="2026-06-27T14:42:55+00:00",
    )

    rc = main(["journal", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "custom scheduler note" in out
    assert "custom_scheduler_note" not in out


def test_cli_journal_verbose_prints_event_details(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(
        journal,
        "sbatch",
        phase="INITIAL_AIMALL",
        iteration=0,
        job_id="123",
        expected_tasks=10,
        ts="2026-06-27T14:42:55+00:00",
    )

    rc = main(["journal", "--campaign-dir", str(campaign), "--verbose"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "sbatch" in out
    assert "array submitted" in out
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines[0] == "Timeline"
    assert len(lines) == 2
    assert "raw=sbatch" in lines[1]
    assert "INITIAL_AIMALL" in out
    assert "2026-06-27 14:42:55" in out
    assert "iter=0" in out
    assert "job=123" in out
    assert "total=10 completed=0 running=- pending=-" in out
    assert "  expected_tasks: 10" not in out
    assert "  iteration: 0" not in out


def test_cli_journal_returns_4_when_no_journal(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["journal", "--campaign-dir", str(campaign)])
    assert rc == 4
    out = capsys.readouterr().out
    assert "Timeline" in out
    assert "no journal yet at" in out


def test_cli_reconcile_writes_proposed_state(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["reconcile", "--campaign-dir", str(campaign)])
    assert rc == 0
    proposed = campaign / DEFAULT_DATA_SUBDIR / (DEFAULT_STATE_FILENAME + ".proposed")
    assert proposed.exists()
    captured = capsys.readouterr()
    assert "Preview result: blocked" in captured.out
    assert "campaign state is missing" in captured.out
    assert "state.json.proposed" not in captured.out
    assert "reference-data versions: committed" not in captured.out
    assert "model versions" not in captured.out


def test_reconcile_current_position_does_not_count_empty_job_markers_as_jobs(
    tmp_path,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state()
    state.pending_jobs = {"INITIAL_GAUSSIAN": None, "INITIAL_AIMALL": None}
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    write_state(
        campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME,
        state,
    )
    report = SimpleNamespace(
        active_submission_intents=[],
        valid_reference_data_versions=[],
        valid_model_versions=[],
        notes=[],
    )

    concise = cli_mod._reconcile_read_current_state_summary(campaign, report)
    verbose = cli_mod._reconcile_read_current_state_summary(
        campaign,
        report,
        verbose=True,
    )

    assert concise["recorded jobs"] == "none"
    assert "completed job markers" not in concise
    assert verbose["completed job markers"] == 2


def test_cli_reconcile_json_outputs_machine_readable_decision(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)

    rc = main(["reconcile", "--campaign-dir", str(campaign), "--json"])

    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 5
    assert payload["campaign_dir"] == str(campaign)
    assert payload["proposed_state_path"].endswith("state.json.proposed")
    assert "selected_phase" in payload
    assert "hard_blockers" in payload
    assert "next_command" in payload
    assert payload["verification"]["level"] == "authority"
    assert payload["verification"]["recursive_scan"] is False
    assert payload["verification"]["payload_hashing"] is False
    assert payload["verification"]["payload_files_hashed"] == 0
    assert "Proposed state written" not in captured.out
    assert "Scientific payload hashing: disabled" in captured.err


@pytest.mark.parametrize("phase", [CampaignPhase.HALTED, CampaignPhase.DONE])
def test_reconcile_json_contract_never_calls_terminal_phases_runnable(
    tmp_path,
    phase,
):
    campaign = _campaign_with_config(tmp_path)
    report = cli_mod.propose_recovery(campaign)
    report.proposed_state.phase = phase

    payload = cli_mod._reconcile_decision_payload(
        campaign,
        report,
        {
            "contract_ok": True,
            "trusted_handoffs": [],
            "missing_or_invalid_inputs": [],
        },
    )

    assert payload["runnable"] is False
    assert "selected phase is terminal: " + phase.value in payload["why_not_runnable"]


def test_cli_reconcile_deep_verify_reports_deep_level_on_stderr(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)

    rc = main(
        [
            "reconcile",
            "--campaign-dir",
            str(campaign),
            "--json",
            "--deep-verify",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["verification"]["level"] == "deep"
    assert payload["verification"]["deep_required"] is False
    assert "Scientific payload hashing: enabled" in captured.err


def test_cli_reconcile_deep_verify_refuses_recorded_scheduler_ownership(
    tmp_path,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state()
    state.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] = "12345"
    _write_locked_state(campaign, state)

    rc = main(
        [
            "reconcile",
            "--campaign-dir",
            str(campaign),
            "--deep-verify",
        ]
    )

    assert rc == 9
    captured = capsys.readouterr()
    assert "Deep Verification Safety" in captured.err
    assert "state records scheduler ownership" in captured.err
    assert "Inspecting committed artefact chains" not in captured.err


def test_cli_reconcile_apply_requires_explicit_deep_verification(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    original = cli_mod.propose_recovery

    def require_deep(*args, **kwargs):
        report = original(*args, **kwargs)
        report.deep_verification_required = True
        return report

    monkeypatch.setattr(cli_mod, "propose_recovery", require_deep)
    rc = main(["reconcile", "--campaign-dir", str(campaign), "--apply"])

    assert rc == 10
    captured = capsys.readouterr()
    assert "--deep-verify" in captured.err
    assert "--deep-verify --apply" not in captured.err


def test_cli_reconcile_json_is_proposal_only(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)

    rc = main([
        "reconcile",
        "--campaign-dir",
        str(campaign),
        "--json",
        "--apply",
    ])

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "json_apply_not_supported"


def test_cli_reconcile_cleanable_scripts_reports_candidate_without_manual_mv(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    _commit_training_and_model_versions(campaign, [0])
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    state.campaign_uid = "cli-test"
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, state)
    write_config_lock(
        campaign,
        CampaignConfig.from_yaml(campaign / "campaign.yaml"),
        campaign_uid=state.campaign_uid,
    )
    append_event(
        campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
        "halt",
        from_phase="PHASE_B_DIVERSITY",
        iteration=1,
        reason="too_many_failures: 1/1",
    )
    _write_valid_ariadne_results(campaign, iteration=1)
    scripts = campaign / ".DATA" / "SCRIPTS"
    scripts.mkdir(parents=True)
    (scripts / "PHASE_B_DIVERSITY-1.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_reconcile_runtime_status", lambda campaign: {})

    rc = main(["reconcile", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "ICHOR Reconcile" in out
    assert "Preview result: ready to apply" in out
    assert "Campaign state" in out
    assert "Planned changes" in out
    assert "submission scripts" in out
    assert "archive stale temporary Slurm scripts" in out
    assert "Data safety" in out
    assert "After apply" in out
    assert "Slurm work" in out
    assert "RESULTS.json" not in out
    assert "Recovery Safety" not in out
    assert "Apply Plan" not in out
    assert "=== Recovery guidance ===" not in out
    assert "Operator-review artefacts:" not in out
    assert "mv " not in out


def test_reconcile_hard_blockers_filter_cleanable_staging_artifacts():
    report = SimpleNamespace(
        unsafe_reasons=[
            "dangling model staging directories exist",
            "dangling reference-data staging directories exist",
        ],
        blocking_artifacts=[
            "dangling model staging",
            "dangling reference-data staging",
            "reference-data version 3",
        ],
        active_submission_intents=[],
    )

    blockers = cli_mod._reconcile_hard_blockers(
        report,
        {
            "selected_phase": CampaignPhase.PHASE_B_DIVERSITY.value,
            "missing_or_invalid_inputs": [],
        },
    )

    assert "dangling model staging" not in blockers
    assert "dangling reference-data staging" not in blockers
    assert "reference-data version 3" in blockers


def test_reconcile_hard_blockers_keep_data_staging_without_archive_staging():
    report = SimpleNamespace(
        unsafe_reasons=[".DATA/STAGING is non-empty"],
        blocking_artifacts=[".DATA/STAGING"],
        active_submission_intents=[],
    )

    blockers = cli_mod._reconcile_hard_blockers(
        report,
        {
            "selected_phase": CampaignPhase.FEREBUS.value,
            "missing_or_invalid_inputs": [],
        },
    )

    assert (
        ".DATA/STAGING is non-empty unless --archive-staging is explicitly requested"
        in blockers
    )
    assert ".DATA/STAGING" in blockers


def test_reconcile_publishes_receipt_backed_intent_retirement_after_state_commit(
    tmp_path,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=2, campaign_uid="cli-repair")
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    intent = submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
        expected_tasks=6,
    )
    reference = {
        "path": ".DATA/ACTIVE_LEARNING/phase_completions/" + "a" * 64 + ".json",
        "receipt_id": "a" * 64,
        "sha256": "b" * 64,
    }

    cli_mod._publish_reconcile_intent_transitions(
        campaign,
        [
            {
                "phase": CampaignPhase.INITIAL_FEREBUS.value,
                "iteration": 0,
                "submission_identity": intent["submission_identity"],
                "target_status": "SUPERSEDED",
                "reason": "phase_completed_without_scheduler_submission",
                "completion_receipt": reference,
            }
        ],
        state,
    )

    retired = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert retired["status"] == "SUPERSEDED"
    assert retired["completion_receipt"] == reference


def test_cli_reconcile_apply_prints_final_recomputed_phase(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    _commit_training_and_model_versions(campaign, [0])
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    state.campaign_uid = "cli-test"
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, state)
    write_config_lock(
        campaign,
        CampaignConfig.from_yaml(campaign / "campaign.yaml"),
        campaign_uid=state.campaign_uid,
    )
    completed_staging = campaign / ".DATA" / "STAGING" / "initial"
    completed_staging.mkdir(parents=True)
    append_event(
        campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson",
        "halt",
        from_phase="PHASE_B_DIVERSITY",
        iteration=1,
        reason="too_many_failures: 1/1",
    )
    _write_valid_ariadne_results(campaign, iteration=1)
    scripts = campaign / ".DATA" / "SCRIPTS"
    scripts.mkdir(parents=True)
    (scripts / "PHASE_B_DIVERSITY-1.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_reconcile_runtime_status", lambda campaign: {})

    rc = main(["reconcile", "--campaign-dir", str(campaign), "--apply"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "ICHOR Reconcile" in out
    assert "Apply result: completed" in out
    assert "Applied changes" in out
    assert "phase B diversity selection, iteration 1" in out
    assert "campaign state" in out
    assert "written and verified" in out
    assert "final authority check" in out
    assert "completed staging" in out
    assert not completed_staging.exists()
    assert "Recovery Contract" not in out
    assert "RESULTS.json" not in out
    assert "ichor-al-daemon resume --campaign-dir " in out


def test_cli_resume_refuses_halted_state(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.phase = CampaignPhase.HALTED
    write_state(data / DEFAULT_STATE_FILENAME, state)

    rc = main(["resume", "--campaign-dir", str(campaign)])

    assert rc == 6
    assert "campaign is HALTED" in capsys.readouterr().err


def test_cli_resume_config_drift_preserves_stop_and_state(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=2)
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.shutdown_requested = True
    state.lifecycle_context = make_lifecycle_context(
        disposition="stopped",
        reason_code="user_stop_immediate",
        message="campaign stopped by user request",
        from_phase=CampaignPhase.SEED_SELECT,
        iteration=1,
        source="stop_control",
    )
    _write_locked_state(campaign, state)
    request, _status = install_stop_request(
        campaign,
        build_stop_request(state, mode="immediate"),
    )
    config_payload = CampaignConfig.from_yaml(
        campaign / "campaign.yaml"
    ).to_dict()
    config_payload["resources"]["defaults"]["partition"] = (
        "changed-partition"
    )
    CampaignConfig.from_dict(config_payload).to_yaml(
        campaign / "campaign.yaml"
    )
    state_path = (
        campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    )
    request_path = stop_request_path(campaign)
    before_state = state_path.read_bytes()
    before_request = request_path.read_bytes()

    rc = main(
        [
            "resume",
            "--campaign-dir",
            str(campaign),
            "--mode",
            "dry_run",
            "--foreground",
            "--max-ticks",
            "0",
        ]
    )

    assert rc == 7
    assert state_path.read_bytes() == before_state
    assert request_path.read_bytes() == before_request
    assert read_stop_request(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )["request_id"] == request["request_id"]
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    assert not journal.exists() or not any(
        event.get("event") == "user_stop_resumed"
        for event in iter_events(journal)
    )
    error = capsys.readouterr().err
    assert "stop request and campaign state were left unchanged" in error
    assert "preview reconcile" in error


def test_cli_resume_repolls_matching_scheduler_uncertain_job(
    tmp_path,
    monkeypatch,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=2)
    phase = CampaignPhase.INITIAL_GAUSSIAN
    job_id = "17615141"
    state.phase = CampaignPhase.HALTED
    state.pending_jobs[phase.value] = job_id
    state.sacct_empty_streak = {
        job_id + ":MISSING": 12,
        "unrelated": 3,
    }
    state.lifecycle_context = make_lifecycle_context(
        disposition="halted",
        reason_code="sacct_missing_timeout",
        message="expected Slurm array rows remained missing",
        from_phase=phase,
        iteration=0,
        source="daemon",
        job_id=job_id,
        scheduler_uncertain=True,
        recovery_action="inspect accounting, then resume",
    )
    _write_locked_state(campaign, state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase_name=phase.value,
        iteration=0,
        expected_tasks=1150,
    )
    submission_intent.mark_submitted(
        campaign,
        phase.value,
        0,
        job_id,
        expected_tasks=1150,
    )
    started = []
    monkeypatch.setattr(cli_mod, "cmd_start", lambda args: started.append(args) or 0)

    rc = main(
        [
            "resume",
            "--campaign-dir",
            str(campaign),
            "--mode",
            "live",
        ]
    )

    assert rc == 0
    assert len(started) == 1
    recovered = read_state(
        campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    )
    assert recovered.phase is phase
    assert recovered.pending_jobs == {phase.value: job_id}
    assert recovered.lifecycle_context is None
    assert recovered.sacct_empty_streak == {"unrelated": 3}
    intent = submission_intent.load_intent(
        campaign,
        phase.value,
        0,
        expected_campaign_uid=str(state.campaign_uid),
    )
    assert intent is not None
    assert intent["status"] == "SUBMITTED"
    assert intent["job_id"] == job_id
    assert any(
        event.get("event") == "scheduler_uncertain_resumed"
        for event in iter_events(
            campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
        )
    )
    assert "no job was resubmitted" in capsys.readouterr().out


def test_cli_resume_repolls_scheduler_uncertain_job_with_pending_boundary_stop(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=2)
    phase = CampaignPhase.INITIAL_GAUSSIAN
    job_id = "17615141"
    state.phase = CampaignPhase.HALTED
    state.pending_jobs[phase.value] = job_id
    state.lifecycle_context = make_lifecycle_context(
        disposition="halted",
        reason_code="sacct_missing_timeout",
        message="expected Slurm array rows remained missing",
        from_phase=phase,
        iteration=0,
        source="daemon",
        job_id=job_id,
        scheduler_uncertain=True,
        recovery_action="inspect accounting, then resume",
    )
    _write_locked_state(campaign, state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase_name=phase.value,
        iteration=0,
        expected_tasks=1150,
    )
    submission_intent.mark_submitted(
        campaign,
        phase.value,
        0,
        job_id,
        expected_tasks=1150,
    )
    request, _ = install_stop_request(
        campaign,
        build_stop_request(state, mode="after_iteration"),
    )
    started = []
    monkeypatch.setattr(
        cli_mod,
        "cmd_start",
        lambda args: started.append(args) or 0,
    )

    rc = main(
        [
            "resume",
            "--campaign-dir",
            str(campaign),
            "--mode",
            "live",
        ]
    )

    assert rc == 0
    assert len(started) == 1
    recovered = read_state(
        campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    )
    assert recovered.phase is phase
    retained = read_stop_request(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    assert retained is not None
    assert retained["request_id"] == request["request_id"]
    assert retained["status"] == "requested"


def test_cli_resume_refuses_scheduler_uncertain_job_with_mismatched_intent(
    tmp_path,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=2)
    phase = CampaignPhase.INITIAL_GAUSSIAN
    state.phase = CampaignPhase.HALTED
    state.pending_jobs[phase.value] = "17615141"
    state.lifecycle_context = make_lifecycle_context(
        disposition="halted",
        reason_code="sacct_missing_timeout",
        message="expected Slurm array rows remained missing",
        from_phase=phase,
        iteration=0,
        source="daemon",
        job_id="17615141",
        scheduler_uncertain=True,
    )
    _write_locked_state(campaign, state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=str(state.campaign_uid),
        phase_name=phase.value,
        iteration=0,
        expected_tasks=1150,
    )
    submission_intent.mark_submitted(
        campaign,
        phase.value,
        0,
        "17615142",
        expected_tasks=1150,
    )

    rc = main(["resume", "--campaign-dir", str(campaign), "--mode", "live"])

    assert rc == 6
    assert read_state(
        campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    ).phase is CampaignPhase.HALTED
    assert "submission-intent JobID does not match" in capsys.readouterr().err


def test_cli_resume_refuses_done_without_explicit_reopen(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=2)
    state.phase = CampaignPhase.DONE
    state.iteration = 2
    write_state(data / DEFAULT_STATE_FILENAME, state)

    rc = main(
        ["resume", "--campaign-dir", str(campaign), "--mode", "dry_run"]
    )

    assert rc == 6
    assert "campaign is DONE" in capsys.readouterr().err
    assert read_state(data / DEFAULT_STATE_FILENAME).phase is CampaignPhase.DONE


def test_cli_reopen_done_requires_config_lock_reconcile_first(tmp_path, capsys):
    from ichor.hpc.active_learning.daemon.config_lock import write_config_lock

    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    original = CampaignConfig(max_iterations=2)
    write_config_lock(campaign, original)
    CampaignConfig(max_iterations=3).to_yaml(campaign / "campaign.yaml")
    state = fresh_campaign_state(max_iterations=2)
    state.phase = CampaignPhase.DONE
    state.iteration = 2
    write_state(data / DEFAULT_STATE_FILENAME, state)

    rc = main([
        "resume",
        "--campaign-dir",
        str(campaign),
        "--mode",
        "dry_run",
        "--reopen-converged",
        "--max-ticks",
        "0",
        "--foreground",
    ])

    assert rc == 7
    error = capsys.readouterr().err
    assert "preview reconcile" in error
    assert "apply only after it reports" in error
    unchanged = read_state(data / DEFAULT_STATE_FILENAME)
    assert unchanged.phase is CampaignPhase.DONE
    assert unchanged.iteration == 2


def test_cli_reopen_done_is_explicit_and_advances_one_iteration(tmp_path):
    from ichor.hpc.active_learning.daemon.config_lock import write_config_lock

    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    config = CampaignConfig(max_iterations=3)
    config.to_yaml(campaign / "campaign.yaml")
    write_config_lock(campaign, config)
    state = fresh_campaign_state(max_iterations=2)
    state.phase = CampaignPhase.DONE
    state.iteration = 2
    write_state(data / DEFAULT_STATE_FILENAME, state)

    rc = main([
        "resume",
        "--campaign-dir",
        str(campaign),
        "--mode",
        "dry_run",
        "--reopen-converged",
        "--max-ticks",
        "0",
        "--foreground",
    ])

    assert rc == 0
    reopened = read_state(data / DEFAULT_STATE_FILENAME)
    assert reopened.phase is CampaignPhase.SEED_SELECT
    assert reopened.iteration == 3
    assert reopened.max_iterations == 3
    assert reopened.lifecycle_context is None
    events = list(iter_events(data / "journal.ndjson"))
    assert any(event.get("event") == "campaign_reopened" for event in events)


def test_cli_start_in_dry_run_mode_drives_state_machine(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    assert main(["init", "--campaign-dir", str(campaign), "--yes"]) == 0
    rc = main([
        "start", "--campaign-dir", str(campaign),
        "--mode", "dry_run", "--max-ticks", "5",
        "--poll-interval", "1",
        "--foreground",
    ])
    assert rc == 0
    # state.json should now exist and be parseable.
    state = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    # Within 5 ticks we should at least have advanced past INIT.
    assert state.phase.value != "INIT"


def test_cli_first_start_defaults_to_live_background(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.execution_identity import (
        read_execution_identity,
    )

    campaign = _campaign_with_config(tmp_path)
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    config.campaign.system_name = "WATER"
    config.to_yaml(campaign / "campaign.yaml")
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    launched = {}

    def _launch(args, selected_campaign):
        launched["background"] = bool(args.background)
        launched["campaign"] = selected_campaign
        return 0

    monkeypatch.setattr(cli_mod, "_launch_background_daemon", _launch)
    rc = main(["start", "--campaign-dir", str(campaign)])

    assert rc == 0
    assert launched == {
        "background": True,
        "campaign": campaign.resolve(),
    }
    identity = read_execution_identity(
        campaign,
        expected_campaign_uid=read_state(
            data / DEFAULT_STATE_FILENAME
        ).campaign_uid,
    )
    assert identity["mode"] == "live"


def test_plain_start_reuses_existing_dry_run_mode(tmp_path, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state()
    _write_locked_state(campaign, state)
    assert (
        main(
            [
                "start",
                "--campaign-dir",
                str(campaign),
                "--mode",
                "dry_run",
                "--max-ticks",
                "0",
                "--foreground",
            ]
        )
        == 0
    )
    launched = {}
    monkeypatch.setattr(
        cli_mod,
        "_launch_background_daemon",
        lambda args, selected: launched.update(
            mode=args.mode,
            background=bool(args.background),
            campaign=selected,
        )
        or 0,
    )

    assert main(["start", "--campaign-dir", str(campaign)]) == 0
    assert launched["mode"] is None
    assert launched["background"] is True


def test_cli_start_missing_state_for_clean_campaign_recommends_init(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)

    rc = main(["start", "--campaign-dir", str(campaign), "--mode", "live"])

    assert rc == 8
    err = capsys.readouterr().err
    assert "state.json is missing for a fresh campaign" in err
    assert "ichor-al-daemon init --campaign-dir" in err
    assert str(campaign) in err
    assert "reconcile --campaign-dir" not in err


def test_cli_start_missing_state_with_artefacts_recommends_reconcile(
    tmp_path,
    capsys,
):
    campaign = _campaign_with_config(tmp_path)
    (campaign / "ACTIVE_LEARNING" / "iteration-000001").mkdir(parents=True)

    rc = main(["start", "--campaign-dir", str(campaign), "--mode", "live"])

    assert rc == 8
    err = capsys.readouterr().err
    assert "state.json is missing but this campaign is not empty" in err
    assert "ichor-al-daemon reconcile --campaign-dir" in err
    assert str(campaign) in err


def test_cli_start_rejects_removed_execution_flags(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    with pytest.raises(SystemExit):
        main(["start", "--campaign-dir", str(campaign), "--live"])
    with pytest.raises(SystemExit):
        main(["start", "--campaign-dir", str(campaign), "--dry-run"])


def test_live_placeholder_is_rejected_before_execution_identity_creation(
    tmp_path,
    capsys,
):
    from ichor.hpc.active_learning.execution_identity import execution_identity_path

    campaign = _campaign_with_config(tmp_path)
    assert main(["init", "--campaign-dir", str(campaign), "--yes"]) == 0
    capsys.readouterr()

    rc = main(["start", "--campaign-dir", str(campaign), "--foreground"])

    assert rc == 2
    assert "real molecular system" in capsys.readouterr().err
    assert not execution_identity_path(campaign).exists()


def test_cli_start_live_on_windows_refuses_with_exit_12(tmp_path, capsys):
    """When live mode is requested but backends are absent (the off-cluster
    case), the CLI must refuse with exit 12 and a message naming the missing
    binaries -- not silently spin a daemon."""
    campaign = _campaign_with_config(tmp_path)
    live_config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    live_config.campaign.system_name = "TEST_SYSTEM"
    live_config.to_yaml(campaign / "campaign.yaml")
    assert main(["init", "--campaign-dir", str(campaign), "--yes"]) == 0
    capsys.readouterr()
    rc = main(
        ["start", "--campaign-dir", str(campaign), "--mode", "live", "--foreground"]
    )
    # On a CSF4 host with all binaries present this test would skip; in our
    # CI / Windows environment, the backends are absent and exit 12 is the
    # expected refusal code.
    from ichor.hpc.active_learning.daemon.preflight import check_backends
    if check_backends().all_present:
        import pytest
        pytest.skip("all live backends present; refusal path not exercised here")
    assert rc == 12
    captured = capsys.readouterr()
    assert "backends are not available" in captured.err


def test_cli_start_live_reaches_daemon_with_all_backends_present(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool

    campaign = _campaign_with_config(tmp_path)
    live_config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    live_config.campaign.system_name = "TEST_SYSTEM"
    live_config.to_yaml(campaign / "campaign.yaml")
    TrajectoryPool.import_from(campaign / "pool.xyz", campaign)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    _write_locked_state(campaign, fresh_campaign_state())
    captured = {}

    class FakeExecutor:
        strict_committed_artifact_verification = True

        def __init__(self, *, campaign_dir, config):
            captured["executor_campaign"] = campaign_dir

    class FakeDaemon:
        def __init__(self, **kwargs):
            captured["daemon_kwargs"] = kwargs

        def run(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return 0

    monkeypatch.setattr(cli_mod, "check_backends", _backend_availability)
    monkeypatch.setattr(cli_mod, "LiveBackendsPhaseExecutor", FakeExecutor)
    monkeypatch.setattr(cli_mod, "Daemon", FakeDaemon)
    monkeypatch.setattr(
        cli_mod,
        "make_live_job_finder",
        lambda *, campaign_dir: ("finder", campaign_dir),
    )
    monkeypatch.setattr(cli_mod, "make_live_job_accounting_finder", lambda: "accounting")
    monkeypatch.setattr(cli_mod, "make_live_job_liveness_checker", lambda: "liveness")

    rc = main([
        "start", "--campaign-dir", str(campaign), "--mode", "live",
        "--max-ticks", "0", "--foreground",
    ])

    assert rc == 0
    assert captured["executor_campaign"] == campaign.resolve()
    assert captured["daemon_kwargs"]["job_finder"] == ("finder", campaign.resolve())
    assert captured["daemon_kwargs"]["job_name_accounting_finder"] == "accounting"
    assert captured["daemon_kwargs"]["job_liveness_checker"] == "liveness"


def test_checkpoint_status_absence_prints_the_safe_creation_command(
    tmp_path,
    capsys,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import checkpoints

    campaign = _campaign_with_config(tmp_path)
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    destination = tmp_path / "checkpoint-store"
    config.retention.checkpoint_destination = str(destination)
    config.to_yaml(campaign / "campaign.yaml")
    monkeypatch.setattr(
        checkpoints,
        "checkpoint_status",
        lambda _campaign, _destination: {"status": "absent"},
    )

    rc = main(["checkpoint-status", "--campaign-dir", str(campaign)])

    assert rc == 1
    out = capsys.readouterr().out
    assert "No current checkpoint has been published" in out
    assert "ichor-al-daemon checkpoint --campaign-dir" in out


def test_restore_checkpoint_preview_prints_the_exact_apply_command(
    tmp_path,
    capsys,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import checkpoints

    checkpoint = tmp_path / "checkpoint"
    target = tmp_path / "restored"
    monkeypatch.setattr(
        checkpoints,
        "restore_checkpoint",
        lambda _checkpoint, _target, *, apply: {
            "applied": bool(apply),
            "target": str(target),
        },
    )

    rc = main(
        [
            "restore-checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--target-empty-dir",
            str(target),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "verified; no files were written" in out
    assert "restore-checkpoint --checkpoint" in out
    assert "--apply" in out


def test_aimall_upstream_recovery_requires_explicit_selected_evidence():
    state = fresh_campaign_state(max_iterations=20)
    state.phase = CampaignPhase.GAUSSIAN
    state.iteration = 14

    report = SimpleNamespace(
        proposed_state=state,
        source_state_phase=CampaignPhase.HALTED.value,
        last_phase_in_journal=CampaignPhase.AIMALL.value,
        partial_array_recovery={
            "phase": CampaignPhase.GAUSSIAN.value,
            "iteration": 14,
            "logical_total": 149,
            "n_complete": 148,
            "n_retry": 1,
        },
        aimall_upstream_gaussian_recovery=None,
    )

    assert cli_mod._aimall_upstream_gaussian_recovery_evidence(report) is None


def test_aimall_upstream_recovery_validates_selected_evidence():
    state = fresh_campaign_state(max_iterations=20)
    state.phase = CampaignPhase.GAUSSIAN
    state.iteration = 14
    evidence = {
        "phase": CampaignPhase.GAUSSIAN.value,
        "source_phase": CampaignPhase.AIMALL.value,
        "iteration": 14,
        "replacement_round": 0,
        "upstream_rewind": "missing_aimall_pointdir",
        "logical_total": 149,
        "n_complete": 148,
        "n_retry": 1,
        "retry_task_ids": [8],
    }
    report = SimpleNamespace(
        proposed_state=state,
        partial_array_recovery={
            "phase": CampaignPhase.GAUSSIAN.value,
            "iteration": 14,
            "logical_total": 149,
            "n_complete": 148,
            "n_retry": 1,
        },
        aimall_upstream_gaussian_recovery=evidence,
    )

    assert cli_mod._aimall_upstream_gaussian_recovery_evidence(report) == evidence


def test_cli_reconcile_reselects_after_terminal_intent_before_proposal(
    tmp_path,
    monkeypatch,
    capsys,
):
    from ichor.hpc.active_learning.daemon.reconcile import (
        ReconciliationReport,
    )

    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(
        max_iterations=2,
        campaign_uid="cli-test",
    )
    state.phase = CampaignPhase.HALTED
    state.iteration = 1
    state.reference_data_version = 0
    state.validation_set_version = 0
    state.models_version = 0
    state.pending_jobs[CampaignPhase.ARIADNE_ARRAY.value] = "17923151"
    state_path = (
        campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(state_path, state)
    intent = {
        "phase": CampaignPhase.ARIADNE_ARRAY.value,
        "iteration": 1,
        "replacement_round": 0,
        "job_id": "17923151",
        "submission_identity": "r0000-a0001-fixture",
        "attempt_id": "fixture-attempt",
        "scheduler_identity_kind": "slurm",
    }
    partial = {
        "phase": CampaignPhase.ARIADNE_ARRAY.value,
        "iteration": 1,
        "logical_total": 200,
        "n_complete": 4,
        "n_reuse": 4,
        "n_retry": 196,
        "path": str(
            campaign
            / DEFAULT_DATA_SUBDIR
            / "array_task_ledgers"
            / "ARIADNE_ARRAY-000001.json"
        ),
    }
    blocked_state = CampaignState.from_dict(state.to_dict())
    recovered_state = CampaignState.from_dict(state.to_dict())
    recovered_state.phase = CampaignPhase.ARIADNE_ARRAY
    recovered_state.pending_jobs = {}
    blocked = ReconciliationReport(
        proposed_state=blocked_state,
        valid_reference_data_versions=[0],
        valid_model_versions=[0],
        source_state_phase=CampaignPhase.HALTED.value,
        existing_state_loaded=True,
        unsafe_reasons=[
            "active submission intent(s) present: "
            "ARIADNE_ARRAY@1 job_id=17923151",
            "scheduler-inconclusive prepared scratch task(s) preserve "
            "job ownership: 17923151@ARIADNE_ARRAY",
        ],
        active_submission_intents=[dict(intent)],
        blocking_artifacts=[
            "active submission intent(s)",
            "prepared scratch ownership",
        ],
        scratch_inventory=[
            {
                "status": "prepared",
                "phase": CampaignPhase.ARIADNE_ARRAY.value,
                "iteration": 1,
                "job_id": "17923151",
            }
        ],
        decision="HALTED: unsafe artefacts need user review",
        recovery_candidates=[
            {
                "phase": CampaignPhase.ARIADNE_ARRAY.value,
                "iteration": 1,
                "path": str(partial["path"]),
                "reason": "partial array recovery available",
            }
        ],
        partial_array_recovery=dict(partial),
    )
    recovered = ReconciliationReport(
        proposed_state=recovered_state,
        valid_reference_data_versions=[0],
        valid_model_versions=[0],
        source_state_phase=CampaignPhase.HALTED.value,
        existing_state_loaded=True,
        decision="partial array recovery available",
        recovery_candidates=list(blocked.recovery_candidates),
        partial_array_recovery=dict(partial),
    )
    calls = []

    def propose(*_args, **kwargs):
        calls.append(dict(kwargs))
        return blocked if len(calls) == 1 else recovered

    terminal = {
        **intent,
        "scheduler_recovery": True,
        "n_completed": 4,
        "n_retry": 196,
        "target_status": "FAILED",
        "reason": "user_cancelled_via_stop",
    }
    monkeypatch.setattr(cli_mod, "_reconcile_runtime_status", lambda *_a: {})
    monkeypatch.setattr(
        cli_mod,
        "build_committed_artifact_snapshot",
        lambda *_a, **_k: SimpleNamespace(),
    )
    monkeypatch.setattr(
        cli_mod,
        "inspect_reconcile_transaction_recovery",
        lambda *_a, **_k: {"state": "none", "recoverable": False},
    )
    monkeypatch.setattr(cli_mod, "propose_recovery", propose)
    monkeypatch.setattr(
        cli_mod,
        "_resolve_terminal_submission_intents_for_apply",
        lambda *_a, **_k: ([dict(terminal)], []),
    )
    monkeypatch.setattr(
        cli_mod,
        "review_config_changes",
        lambda *_a, **_k: SimpleNamespace(
            allowed_changes=[
                SimpleNamespace(
                    path="resources." + name + ".partition",
                    old="multicore_small",
                    new="multicore",
                )
                for name in (
                    "ariadne",
                    "gaussian",
                    "aimall",
                    "ferebus",
                    "diversity",
                    "defaults",
                )
            ],
            blocked_changes=[],
        ),
    )
    monkeypatch.setattr(
        cli_mod,
        "inspect_aimall_quality_revalidation",
        lambda *_a, **_k: {"state": "ineligible"},
    )
    monkeypatch.setattr(
        cli_mod,
        "recovery_contract_status",
        lambda *_a, **_k: {
            "contract_ok": True,
            "selected_phase": CampaignPhase.ARIADNE_ARRAY.value,
            "missing_or_invalid_inputs": [],
            "trusted_handoffs": [],
            "protected_artifacts": [],
            "trusted_inputs": [],
        },
    )

    rc = main(["reconcile", "--campaign-dir", str(campaign)])

    assert rc == 0
    assert len(calls) == 2
    assert calls[1]["_terminal_submission_intents"] == [terminal]
    proposed = read_state(
        state_path.with_name(DEFAULT_STATE_FILENAME + ".proposed")
    )
    assert proposed.phase is CampaignPhase.ARIADNE_ARRAY
    output = capsys.readouterr().out
    assert output.count("partition multicore_small -> multicore") == 1
    assert "(6 affected settings)" in output
    assert "array recovery" not in output
    assert re.search(r"active work\s*: none", output)


def test_reconcile_terminal_scheduler_recovery_without_changes_uses_resume(
    tmp_path,
):
    from ichor.hpc.active_learning.daemon.reconcile import (
        ReconciliationReport,
    )

    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(
        max_iterations=40,
        campaign_uid="cli-test",
    )
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 15
    state.reference_data_version = 14
    state.models_version = 14
    state.pending_jobs = {
        CampaignPhase.ARIADNE_ARRAY.value: "17923151"
    }
    state.shutdown_requested = True
    state.lifecycle_context = make_lifecycle_context(
        disposition="stopped",
        reason_code="user_stop_immediate",
        message="campaign stopped after scheduler cancellation",
        from_phase=CampaignPhase.ARIADNE_ARRAY,
        iteration=15,
        source="stop_control",
    )
    state_path = (
        campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(state_path, state)
    intent = {
        "phase": CampaignPhase.ARIADNE_ARRAY.value,
        "iteration": 15,
        "replacement_round": 0,
        "job_id": "17923151",
    }
    report = ReconciliationReport(
        proposed_state=CampaignState.from_dict(state.to_dict()),
        active_submission_intents=[intent],
        partial_array_recovery={
            "phase": CampaignPhase.ARIADNE_ARRAY.value,
            "iteration": 15,
            "logical_total": 200,
            "n_complete": 4,
            "n_reuse": 4,
            "n_retry": 196,
        },
    )
    report.scheduler_cancellation_recovery = [
        {
            **intent,
            "n_completed": 4,
            "n_retry": 196,
        }
    ]
    presentation = cli_mod._reconcile_presentation(
        campaign,
        report,
        {
            "contract_ok": True,
            "missing_or_invalid_inputs": [],
            "protected_artifacts": [],
        },
        config_review=SimpleNamespace(
            allowed_changes=[],
            blocked_changes=[],
        ),
        runtime_status={},
    )

    assert presentation.result == "no reconcile changes needed"
    assert dict(presentation.campaign_state)["active work"] == "none"
    labels = [label for label, _value in presentation.planned_changes]
    assert labels == ["scheduler recovery"]
    assert presentation.next_command.startswith("ichor-al-daemon resume")
    assert "validate 4 scheduler-completed outputs" in (
        presentation.next_effect
    )
    assert "retry 196 unfinished tasks" in presentation.next_effect
