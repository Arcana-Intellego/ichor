"""Wave 14 adversarial verification and release-closure gates."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.cli import (
    _exclusive_operator_lock,
    cmd_checkpoint,
    cmd_reconcile,
)
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import (
    Daemon,
    PHASE_ORDER,
    TickStatus,
    next_phase,
)
from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
from ichor.hpc.active_learning.daemon.dry_run_sacct import DryRunSacctPoller
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.handoff_manifests import (
    build_seed_selection_manifest,
)
from ichor.hpc.active_learning.versioning.trained_models import (
    build_trained_model_set_payload,
)


_EXPECTED_DEFAULT_TRANSITIONS = {
    CampaignPhase.INIT: CampaignPhase.PHASE_A_DIVERSITY,
    CampaignPhase.PHASE_A_DIVERSITY: CampaignPhase.INITIAL_GAUSSIAN,
    CampaignPhase.INITIAL_GAUSSIAN: CampaignPhase.INITIAL_AIMALL,
    CampaignPhase.INITIAL_AIMALL: CampaignPhase.INITIAL_ALLOCATION_CHECK,
    CampaignPhase.INITIAL_ALLOCATION_CHECK: CampaignPhase.INITIAL_FEREBUS,
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN: CampaignPhase.INITIAL_REPLACEMENT_AIMALL,
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL: CampaignPhase.INITIAL_ALLOCATION_CHECK,
    CampaignPhase.INITIAL_FEREBUS: CampaignPhase.SEED_SELECT,
    CampaignPhase.SEED_SELECT: CampaignPhase.ARIADNE_ARRAY,
    CampaignPhase.ARIADNE_ARRAY: CampaignPhase.PHASE_B_DIVERSITY,
    CampaignPhase.PHASE_B_DIVERSITY: CampaignPhase.SPLIT,
    CampaignPhase.SPLIT: CampaignPhase.GAUSSIAN,
    CampaignPhase.GAUSSIAN: CampaignPhase.AIMALL,
    CampaignPhase.AIMALL: CampaignPhase.ALLOCATION_CHECK,
    CampaignPhase.ALLOCATION_CHECK: CampaignPhase.APPEND,
    CampaignPhase.REPLACEMENT_GAUSSIAN: CampaignPhase.REPLACEMENT_AIMALL,
    CampaignPhase.REPLACEMENT_AIMALL: CampaignPhase.ALLOCATION_CHECK,
    CampaignPhase.APPEND: CampaignPhase.FEREBUS,
    CampaignPhase.FEREBUS: CampaignPhase.STOP_CHECK,
}


def test_independent_fsm_enumeration_matches_every_nonterminal_phase():
    assert set(PHASE_ORDER) == set(_EXPECTED_DEFAULT_TRANSITIONS) | {
        CampaignPhase.STOP_CHECK
    }
    for phase, expected in _EXPECTED_DEFAULT_TRANSITIONS.items():
        iteration = 0 if phase.value.startswith("INITIAL_") or phase in {
            CampaignPhase.INIT,
            CampaignPhase.PHASE_A_DIVERSITY,
        } else 3
        observed, _next_iteration = next_phase(phase, iteration, 25)
        assert observed is expected

    assert next_phase(CampaignPhase.STOP_CHECK, 3, 25) == (
        CampaignPhase.SEED_SELECT,
        4,
    )
    assert next_phase(CampaignPhase.STOP_CHECK, 25, 25) == (
        CampaignPhase.DONE,
        25,
    )
    with pytest.raises(ValueError, match="bootstrap iteration 0"):
        next_phase(CampaignPhase.INITIAL_FEREBUS, 1, 25)
    with pytest.raises(ValueError, match="active iteration"):
        next_phase(CampaignPhase.STOP_CHECK, 0, 25)
    for terminal in (CampaignPhase.DONE, CampaignPhase.HALTED):
        with pytest.raises(ValueError, match="unexpected current phase"):
            next_phase(terminal, 1, 25)


def test_authoritative_operator_mutations_are_mutually_exclusive(
    tmp_path,
    capsys,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = CampaignConfig()
    config.to_yaml(campaign / "campaign.yaml")
    checkpoint_store = tmp_path / "checkpoints"
    checkpoint_store.mkdir()

    with _exclusive_operator_lock(campaign):
        with pytest.raises(RuntimeError, match="daemon lock is held"):
            with _exclusive_operator_lock(campaign):
                pass
        checkpoint_rc = cmd_checkpoint(
            SimpleNamespace(
                campaign_dir=str(campaign),
                destination=str(checkpoint_store),
                json=False,
            )
        )
        reconcile_rc = cmd_reconcile(
            SimpleNamespace(campaign_dir=str(campaign), apply=True)
        )

    assert checkpoint_rc == 2
    assert reconcile_rc == 3
    errors = capsys.readouterr().err
    assert "daemon lock is held" in errors


def _scientific_model_task(*, model_sha256: str = "1" * 64) -> dict:
    def file_record(path: str, digest: str) -> dict:
        return {
            "path": path,
            "size": 64,
            "sha256": digest,
        }

    return {
        "task_index": 0,
        "property": "iqa",
        "atom": "O1",
        "alf_1_indexed": [1, 2, 3],
        "directory": "iqa/O1",
        "model": file_record("iqa/O1/WATER_iqa_O1.model", model_sha256),
        "config": file_record("iqa/O1/ferebus.config", "2" * 64),
        "execution_receipt": file_record(
            "iqa/O1/FEREBUS_TASK_QUALITY.json",
            "3" * 64,
        ),
        "datasets": {
            "train": file_record("iqa/O1/train.csv", "4" * 64),
            "int_val": file_record("iqa/O1/int_val.csv", "5" * 64),
            "ext_val": file_record("iqa/O1/ext_val.csv", "6" * 64),
        },
        "auxiliary": {
            "opt": None,
            "perf": None,
            "pred": None,
            "scurve": None,
            "sol": None,
        },
    }


def _trained_model_payload(
    *,
    attempt_id: str,
    model_sha256: str = "1" * 64,
) -> dict:
    task = _scientific_model_task(model_sha256=model_sha256)
    return build_trained_model_set_payload(
        campaign_uid="wave14-scientific-identity",
        version=1,
        system="WATER",
        reference_data_head_manifest_sha256="7" * 64,
        reference_data_view_sha256="8" * 64,
        parent=None,
        source_task_manifest={
            "path": "FEREBUS_TASKS.json",
            "size": 64,
            "sha256": "9" * 64,
            "attempt_id": attempt_id,
        },
        quality_manifest={
            "path": "FEREBUS_QUALITY.json",
            "size": 64,
            "sha256": "a" * 64,
        },
        quality_decision_manifest={
            "path": "FEREBUS_QUALITY_DECISION.json",
            "size": 64,
            "sha256": "b" * 64,
        },
        properties=["iqa"],
        atoms=["O1"],
        tasks=[task],
        root_files=[{
            "path": "FEREBUS_TASKS.json",
            "size": 64,
            "sha256": "c" * 64,
        }],
    )


def test_scientific_model_identity_excludes_operational_evidence():
    first = _trained_model_payload(attempt_id="attempt-one")
    second = _trained_model_payload(attempt_id="attempt-two")

    assert first["model_set_sha256"] == second["model_set_sha256"]
    assert first["evidence_set_sha256"] != second["evidence_set_sha256"]

    changed_model = _trained_model_payload(
        attempt_id="attempt-two",
        model_sha256="d" * 64,
    )
    assert changed_model["model_set_sha256"] != first["model_set_sha256"]


def test_seed_identity_ignores_full_manifest_receipt_churn():
    common = {
        "campaign_uid": "wave14-seed-identity",
        "campaign_random_seed": 4,
        "iteration": 1,
        "models_version": 0,
        "model_set_sha256": "e" * 64,
        "trajectory_sha256": "f" * 64,
        "selection_strategy": "hybrid_variance",
        "seed_records": [{
            "seed_id": 1,
            "frame_id": 7,
            "pool_row_index_zero_based": 7,
            "selection_origin": "bulk",
            "variance_at_selection": 0.25,
        }],
    }
    first = build_seed_selection_manifest(
        **common,
        model_manifest_sha256="1" * 64,
    )
    second = build_seed_selection_manifest(
        **common,
        model_manifest_sha256="2" * 64,
    )

    assert first["model_manifest_sha256"] != second["model_manifest_sha256"]
    assert first["selection_fingerprint_sha256"] == second[
        "selection_fingerprint_sha256"
    ]
    assert first["seed_records"][0]["seed_uid"] == second["seed_records"][0][
        "seed_uid"
    ]


def _long_run_config() -> CampaignConfig:
    config = CampaignConfig(max_iterations=25, poll_interval_seconds=1)
    config.campaign.random_seed = 9173
    config.point_allocation.bootstrap_training_size = 2
    config.point_allocation.bootstrap_internal_validation_size = 2
    config.point_allocation.bootstrap_external_validation_size = 2
    config.point_allocation.batch_training_size = 1
    config.point_allocation.batch_internal_validation_size = 0
    config.seed_selection.n_seeds_per_iteration = 1
    config.seed_selection.bulk_fraction = 1.0
    config.stop.min_iterations_before_stop = 100
    config._validate()
    return config


def _new_dry_daemon(campaign: Path, config: CampaignConfig) -> Daemon:
    return Daemon(
        campaign_dir=campaign,
        config=config,
        executor=DryRunPhaseExecutor(campaign_dir=campaign, config=config),
        sacct_poller=DryRunSacctPoller(elapsed_seconds=0),
        sleep_fn=lambda _seconds: None,
    )


def _drive_25_iterations(
    root: Path,
    *,
    restart_every_tick: bool,
) -> Path:
    campaign = root / ("restarted" if restart_every_tick else "uninterrupted")
    campaign.mkdir()
    config = _long_run_config()
    config.to_yaml(campaign / "campaign.yaml")
    state = fresh_campaign_state(
        max_iterations=25,
        campaign_uid="wave14-deterministic-campaign",
    )
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    initial = _new_dry_daemon(campaign, config)
    initial.executor._ensure_dry_run_trajectory_pool()
    daemon = initial
    for _tick in range(2500):
        if restart_every_tick:
            daemon = _new_dry_daemon(campaign, config)
        status = daemon.tick()
        current = read_state(daemon.state_path())
        if current.phase is CampaignPhase.HALTED:
            raise AssertionError(
                "dry campaign halted: " + str(current.lifecycle_context)
            )
        if current.phase is CampaignPhase.DONE:
            assert status in {TickStatus.ADVANCED, TickStatus.TERMINAL}
            break
    else:
        raise AssertionError("25-iteration dry campaign exceeded 2500 ticks")
    final_state = read_state(daemon.state_path())
    assert final_state.phase is CampaignPhase.DONE
    assert final_state.iteration == 25
    assert final_state.reference_data_version == 25
    assert final_state.models_version == 25
    return campaign


def _decision_snapshot(campaign: Path) -> list[dict]:
    snapshot = []
    for iteration in range(1, 26):
        root = campaign / "ACTIVE_LEARNING" / (
            "iteration-" + str(iteration).zfill(6)
        )
        seeds = json.loads(
            (root / "seed_selection" / "SELECTION.json").read_text(
                encoding="utf-8"
            )
        )
        ariadne = json.loads(
            (root / "ariadne" / "RESULTS.json").read_text(encoding="utf-8")
        )
        phase_b = json.loads(
            (root / "phase_b" / "SELECTION.json").read_text(encoding="utf-8")
        )
        seed_results = []
        for record in sorted(ariadne["accepted"], key=lambda row: row["seed_id"]):
            result = json.loads(
                (root / "ariadne" / str(record["result_json"])).read_text(
                    encoding="utf-8"
                )
            )
            seed_results.append(
                {
                    "seed_id": record["seed_id"],
                    "seed_uid": record["seed_uid"],
                    "seed_frame_id": record["seed_frame_id"],
                    "selection_origin": record["selection_origin"],
                    "alpha_initial": result["alpha_initial"],
                    "alpha_final": result["alpha_final"],
                    "final_coordinates": result["final_coordinates"],
                    "raw_final_coordinates": result.get("raw_final_coordinates"),
                    "landing_safety": result["landing_safety"],
                    "landing_candidates": result.get("landing_candidates"),
                    "randomness": result["randomness"],
                }
            )
        snapshot.append(
            {
                "iteration": iteration,
                "seed_records": seeds["seed_records"],
                "seed_randomness": seeds["diagnostics"]["randomness"],
                "ariadne": seed_results,
                "phase_b_final": [
                    {
                        "seed_id": row["seed_id"],
                        "seed_uid": row["seed_uid"],
                        "seed_frame_id": row["seed_frame_id"],
                        "slot_id": row["slot_id"],
                        "split": row["split"],
                    }
                    for row in phase_b["final"]
                ],
            }
        )
    return snapshot


@pytest.mark.slow
@pytest.mark.integration
def test_25_iteration_dry_campaign_is_restart_equivalent(tmp_path):
    uninterrupted = _drive_25_iterations(tmp_path, restart_every_tick=False)
    restarted = _drive_25_iterations(tmp_path, restart_every_tick=True)

    assert _decision_snapshot(restarted) == _decision_snapshot(uninterrupted)
