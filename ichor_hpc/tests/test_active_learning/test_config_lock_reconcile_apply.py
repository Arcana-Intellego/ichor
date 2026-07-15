import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import ichor.hpc.active_learning.cli as cli_mod
import ichor.hpc.active_learning.daemon.reconcile as reconcile_mod
import pytest
from ichor.hpc.active_learning.cli import cmd_reconcile, cmd_start
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.daemon import config_lock as config_lock_mod
from ichor.hpc.active_learning.daemon.config_lock import (
    config_fingerprint,
    config_lock_path,
    review_config_changes,
    restore_config_lock_from_history,
    restore_config_from_lock_proposal,
    write_config_lock,
)
from ichor.hpc.active_learning.daemon.journal import iter_events
from ichor.hpc.active_learning.daemon.reconcile import ReconciliationReport
from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.daemon import submission_intent
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state as _fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.point_allocation import (
    create_point_allocation,
    pending_attempts,
    point_allocation_path,
    read_point_allocation,
    record_quantum_results,
)
from ichor.hpc.active_learning.layout import bootstrap_selection_dir
from ichor.hpc.active_learning.submit import sacct_poll
from ichor.hpc.active_learning.versioning.provenance import (
    enrich_with_point_allocation,
    write_seed_provenance,
)
from ichor.hpc.active_learning.versioning.versioned_directory import VersionedDirectory


_FIXTURE_CAMPAIGN_UID = "config-lock-test"


class _FerebusFixturePointsDirectory(list):
    """Emit finite three-atom CSV data without parsing fixture quantum files."""

    def __init__(self, path, needs_parsing=True):
        super().__init__()
        self.path = path

    def alf_dict(self, _calculator):
        return {"O1": (0, 1, 2), "H2": (1, 0, 2), "H3": (2, 0, 1)}

    def features_with_properties_to_csv(
        self,
        _system_alf,
        str_to_append_to_fname="_train.csv",
        property_types=None,
    ):
        properties = [str(value) for value in (property_types or ["iqa"])]
        for atom, base in (("O1", -75.0), ("H2", -0.5), ("H3", -0.5)):
            path = Path(atom + str_to_append_to_fname)
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("f1,f2,f3," + ",".join(properties) + "\n")
                for row_index in range(len(self)):
                    values = {"iqa": base - row_index * 0.01}
                    handle.write(
                        f"{row_index + 0.1},{row_index + 0.2},"
                        f"{row_index + 0.3},"
                        + ",".join(str(values[prop]) for prop in properties)
                        + "\n"
                    )


def fresh_campaign_state(**kwargs):
    """Create fixture state under the same identity as committed artefacts."""
    kwargs.setdefault("campaign_uid", _FIXTURE_CAMPAIGN_UID)
    return _fresh_campaign_state(**kwargs)


def _campaign(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    (campaign / "QM_REFERENCE_DATA").mkdir()
    (campaign / "TRAINED_MODELS").mkdir()
    return campaign


def _write_pool(campaign):
    src = campaign / "pool_source.xyz"
    src.write_text(
        "1\n"
        "frame 0\n"
        "H 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    TrajectoryPool.import_from(
        src,
        campaign,
        overwrite=True,
    )


def _commit_reference_data_version(campaign, version=0):
    from ichor.hpc.active_learning.daemon.input_staging import (
        commit_initial_reference_data,
    )

    if int(version) != 0:
        raise ValueError("fixture helper only supports bootstrap version 0")
    _write_pool(campaign)
    allocation_path = point_allocation_path(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid=_FIXTURE_CAMPAIGN_UID,
        context="bootstrap",
        iteration=0,
        targets={"train": 1, "int_val": 0, "ext_val": 0, "total": 1},
        primary_candidates=[
            {
                "candidate_id": "bootstrap-candidate-0",
                "frame_id": 0,
                "pointdir_name": "POINT_0000.pointdir",
            }
        ],
        reserve_candidates=[],
    )
    attempt = pending_attempts(allocation)[0]
    pointdir = campaign / ".DATA" / "STAGING" / "initial" / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    (pointdir / "input.gjf").write_text("# fixture\n", encoding="utf-8")
    write_seed_provenance(
        pointdir,
        campaign_uid=_FIXTURE_CAMPAIGN_UID,
        iteration=0,
        trajectory_sha256="0" * 64,
        seed_frame_id=0,
        seed_selection_origin="bootstrap_fixture",
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
    from ichor_hpc.tests.quantum_test_support import (
        attach_synthetic_quantum_batch,
    )

    results = [
        {
            "candidate_id": str(attempt["candidate_id"]),
            "accepted": True,
            "pointdir": str(pointdir.resolve()),
        }
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
    assert commit_initial_reference_data(campaign) is True


def _write_config(campaign, config):
    config.to_yaml(campaign / "campaign.yaml")


def _write_phase_a_sample(campaign):
    from ichor.hpc.active_learning.handoff_manifests import write_phase_a_sample_manifest

    _write_pool(campaign)
    initial = bootstrap_selection_dir(campaign)
    initial.mkdir(parents=True, exist_ok=True)
    sample = initial / "selected.xyz"
    index = initial / "selected_indices.dat"
    sample.write_text("1\nframe 0\nH 0.0 0.0 0.0\n", encoding="utf-8")
    index.write_text("0\n", encoding="utf-8")
    allocation_path = point_allocation_path(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    if allocation_path.is_file():
        allocation = read_point_allocation(allocation_path)
    else:
        allocation = create_point_allocation(
            allocation_path,
            campaign_uid="config-lock-test",
            context="bootstrap",
            iteration=0,
            targets={"train": 1, "int_val": 0, "ext_val": 0, "total": 1},
            primary_candidates=[
                {"candidate_id": "bootstrap-candidate-0", "frame_id": 0}
            ],
            reserve_candidates=[],
        )
    slot = allocation["slots"][0]
    primary = [{
        **slot["attempts"][0],
        "slot_id": int(slot["slot_id"]),
        "split": str(slot["split"]),
    }]
    write_phase_a_sample_manifest(initial, {
        "phase": "PHASE_A_DIVERSITY",
        "iteration": 0,
        "sample_xyz": "selection/selected.xyz",
        "index_path": "selection/selected_indices.dat",
        "n_select": 1,
        "n_frames": 1,
        "selected_indices": [0],
        "descriptor": "mass_weighted_rmsd",
        "n_pool_frames": 1,
        "bootstrap_total_size": 1,
        "source_pool_manifest": ".DATA/TRAJECTORY/pool.manifest.json",
        "point_allocation": {
            "manifest": "allocation/POINT_ALLOCATION.json",
            "targets": dict(allocation["targets"]),
            "primary": primary,
            "reserve_frame_ids": [],
            "reserve_count": 0,
        },
    })


def _write_live_initial_gaussian_handoff(campaign):
    initial = campaign / ".DATA" / "STAGING" / "initial"
    pointdir = initial / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    (pointdir / "input.wfn").write_text("wfn\n", encoding="utf-8")
    stg.write_points_file(initial, [pointdir])
    stg.write_quantum_acceptance_manifest(
        initial,
        phase_name=CampaignPhase.INITIAL_GAUSSIAN.value,
        iteration=0,
        accepted=[pointdir],
        rejected=[],
    )


def _write_halted_pre_ferebus_state(campaign):
    _write_pool(campaign)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.reference_data_version = 0
    state.models_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    return state


def _write_submitted_initial_ferebus_intent(campaign, job_id="16218598"):
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
    )
    return submission_intent.mark_submitted(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
        job_id,
        expected_tasks=12,
    )


def _write_archived_bootstrap_handoff(campaign, *, phase, suffix="20260627-201927"):
    initial = campaign / ".DATA" / ("STAGING.archived-" + suffix) / "initial"
    pointdir = initial / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    (pointdir / "input.wfn").write_text("wfn\n", encoding="utf-8")
    stg.write_points_file(initial, [pointdir])
    stg.write_quantum_acceptance_manifest(
        initial,
        phase_name=phase,
        iteration=0,
        accepted=[pointdir],
        rejected=[],
    )
    return initial


def _write_stale_pre_submit_intent(campaign, phase):
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    payload = submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=phase,
        iteration=0,
    )
    path = submission_intent.intent_path(campaign, phase, 0)
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    payload["created_iso"] = old
    payload["updated_iso"] = old
    payload["updated_at_iso"] = old
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_ferebus_physical_prior_scale_change_allowed_before_initial_ferebus(tmp_path):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    state = _write_halted_pre_ferebus_state(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ferebus.physical_prior_scale = 0.95
    _write_config(campaign, changed)

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_FEREBUS
    proposed.reference_data_version = 0
    proposed.models_version = -1
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert [c.path for c in review.allowed_changes] == [
        "ferebus.physical_prior_scale"
    ]
    assert not review.blocked_changes
    assert state.phase is CampaignPhase.HALTED


def test_physical_prior_contract_is_locked_after_first_ferebus_model(tmp_path):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    consumed_model = (
        campaign / "TRAINED_MODELS" / "iteration-000000" / "O1_iqa.model"
    )
    consumed_model.parent.mkdir(parents=True, exist_ok=True)
    consumed_model.write_text("consumed\n", encoding="utf-8")
    changed = CampaignConfig()
    changed.ferebus.physical_prior_scale = 0.99

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.SEED_SELECT
    proposed.models_version = 0
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "ferebus.physical_prior_scale"
    ]
    assert review.blocked_changes[0].category == "postprocess_locked"
    assert "FEREBUS" in review.blocked_changes[0].reason


def test_reconcile_refuses_nonempty_campaign_without_trusted_campaign_identity(
    tmp_path,
    capsys,
):
    campaign = _campaign(tmp_path)
    _write_pool(campaign)
    selection = bootstrap_selection_dir(campaign)
    selection.mkdir(parents=True, exist_ok=True)
    (selection / "selected.xyz").write_text(
        "1\nframe 0\nH 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    config = CampaignConfig()
    _write_config(campaign, config)
    write_config_lock(campaign, config)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=False,
            archive_staging=False,
            restore_config_from_lock=False,
        )
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "ICHOR Reconcile" in out
    assert "Result: BLOCKED" in out
    assert "Recovery Contract" in out
    assert "no recoverable trusted campaign_uid" in out
    assert "selected phase: HALTED iteration 0" in out
    assert "status        : not runnable" in out
    assert "=== Recovery guidance ===" not in out
    assert "Recovery Target" in out
    assert "Apply Plan" in out
    assert "apply: blocked" in out
    assert "Inspect" in out


def test_reconcile_prints_protected_staging_handoff(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _write_live_initial_gaussian_handoff(campaign)
    config = CampaignConfig()
    _write_config(campaign, config)
    write_config_lock(campaign, config)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=False,
            archive_staging=False,
            restore_config_from_lock=False,
        )
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Recovery Safety" in out
    assert "protected:" in out
    assert "INITIAL_AIMALL@0 -> .DATA\\STAGING\\initial" in out or (
        "INITIAL_AIMALL@0 -> .DATA/STAGING/initial" in out
    )
    assert "Operator-review artefacts:" not in out
    assert "blockers:" in out


def test_phase_walltime_changes_are_allowed_runtime_changes(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.resources.ferebus.walltime_hours = 2
    changed.resources.gaussian.walltime_hours = 3
    _write_config(campaign, changed)

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_FEREBUS
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert sorted(c.path for c in review.allowed_changes) == [
        "resources.ferebus.walltime_hours",
        "resources.gaussian.walltime_hours",
    ]
    assert not review.blocked_changes


def test_legacy_config_lock_is_rejected_without_compatibility_migration(tmp_path):
    campaign = _campaign(tmp_path)
    current = CampaignConfig()
    old_v3 = {
        "schema_version": 3,
        "system_name": current.system_name,
        "resources": {
            "partition": "multicore",
            "default_walltime_hours": 24,
            "polus_walltime_hours": 2,
            "gaussian_walltime_hours": 24,
            "diversity_cpus_per_task": "auto",
            "gaussian_cpus_per_task": "auto",
            "aimall_cpus_per_task": "auto",
            "ariadne_cpus_per_task": "auto",
            "ferebus_cpus_per_task": "auto",
            "polus_mem_per_cpu": "auto",
            "gaussian_mem_per_cpu": "auto",
            "aimall_mem_per_cpu": "auto",
            "ariadne_mem_per_cpu": "auto",
            "ferebus_mem_per_cpu": "auto",
            "gaussian_memory_mode": "slurm_env",
            "gaussian_link0_mem": "8GB",
            "gaussian_memory_fraction_of_slurm": 0.85,
            "array_concurrency_limit": None,
            "fail_on_memory_estimate_exceeds_request": True,
            "gradient_parallel_backend": "process",
        },
    }
    path = config_lock_path(campaign)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_schema_version": 3,
                "created_at_iso": "2026-01-01T00:00:00+00:00",
                "last_checked_at_iso": "2026-01-01T00:00:00+00:00",
                "canonical_config": old_v3,
                "fingerprint_sha256": config_fingerprint(old_v3),
                "field_policy_version": 1,
            }
        ),
        encoding="utf-8",
    )
    review = review_config_changes(campaign, current, fresh_campaign_state())
    assert not review.allowed
    assert [change.path for change in review.blocked_changes] == ["config_lock"]
    assert "unsupported config lock schema version" in review.blocked_changes[0].reason


def test_retry_phase_requires_retryable_journal_event():
    state = fresh_campaign_state()
    success_report = ReconciliationReport(
        proposed_state=state,
        last_phase_in_journal=CampaignPhase.PHASE_B_DIVERSITY.value,
        last_iteration_in_journal=0,
        last_phase_event_in_journal="phase_succeeded",
        last_phase_retryable=False,
    )
    halted_report = ReconciliationReport(
        proposed_state=state,
        last_phase_in_journal=CampaignPhase.PHASE_B_DIVERSITY.value,
        last_iteration_in_journal=0,
        last_phase_event_in_journal="halt",
        last_phase_retryable=True,
    )

    assert cli_mod._retry_phase_from_cleaned_report(success_report) is None
    assert cli_mod._retry_phase_from_cleaned_report(halted_report) is CampaignPhase.PHASE_B_DIVERSITY


def test_reconcile_apply_uses_safe_max_iterations_change(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    original = CampaignConfig(max_iterations=1)
    write_config_lock(campaign, original)
    changed = CampaignConfig(max_iterations=7)
    _write_config(campaign, changed)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )

    assert rc == 0
    capsys.readouterr()
    recovered = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert recovered.max_iterations == 7


def test_removed_memory_guard_lock_is_rejected_without_migration(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    path = config_lock_path(campaign)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["canonical_config"]["resources"][
        "fail_on_memory_estimate_exceeds_request"
    ] = True
    payload["fingerprint_sha256"] = config_fingerprint(payload["canonical_config"])
    payload.pop("campaign_uid", None)
    payload.pop("history_sequence", None)
    payload.pop("history_entry_sha256", None)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    changed = CampaignConfig()
    _write_config(campaign, changed)

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_AIMALL
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [change.path for change in review.blocked_changes] == ["config_lock"]


def test_missing_current_lock_restores_only_from_verified_history(tmp_path):
    campaign = _campaign(tmp_path)
    state = fresh_campaign_state()
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    config_lock_path(campaign).unlink()

    restored = restore_config_lock_from_history(
        campaign,
        expected_campaign_uid=state.campaign_uid,
    )

    payload = json.loads(restored.read_text(encoding="utf-8"))
    assert payload["campaign_uid"] == state.campaign_uid
    assert payload["fingerprint_sha256"] == config_fingerprint(config.to_dict())


def test_config_lock_history_restore_rejects_fork(tmp_path):
    campaign = _campaign(tmp_path)
    state = fresh_campaign_state()
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    write_config_lock(campaign, CampaignConfig())
    root = config_lock_path(campaign).parent / "config_lock_history"
    original = next(root.glob("*.json"))
    fork = json.loads(original.read_text(encoding="utf-8"))
    fork["reason"] = "fork"
    fork["entry_sha256"] = config_lock_mod._history_entry_sha256(fork)
    (root / ("00000000-" + fork["entry_sha256"] + ".json")).write_text(
        json.dumps(fork),
        encoding="utf-8",
    )
    config_lock_path(campaign).unlink()

    with pytest.raises(ValueError, match="one root|forked"):
        restore_config_lock_from_history(
            campaign,
            expected_campaign_uid=state.campaign_uid,
        )


def test_legacy_acquisition_driver_lock_is_rejected_without_migration(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    path = config_lock_path(campaign)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["canonical_config"]["acquisition"]["driver"]
    payload["schema_version"] = 1
    payload["fingerprint_sha256"] = config_fingerprint(payload["canonical_config"])
    payload.pop("campaign_uid", None)
    payload.pop("history_sequence", None)
    payload.pop("history_entry_sha256", None)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    changed = CampaignConfig()
    _write_config(campaign, changed)

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [change.path for change in review.blocked_changes] == ["config_lock"]


def test_default_campaign_config_has_no_unclassified_lock_paths():
    config = CampaignConfig()
    proposed = fresh_campaign_state()
    campaign = Path("/tmp/campaign-placeholder")
    unexpected = []
    for path, value in sorted(config_lock_mod._flatten(config.to_dict()).items()):
        change = config_lock_mod._classify_change(
            campaign,
            proposed,
            path,
            None,
            value,
        )
        if change.category == "unclassified_locked":
            unexpected.append(path)

    assert unexpected == []


def test_ariadne_backtransform_changes_are_future_safe(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ariadne.trqn_backtransform_mode = "newton"
    changed.ariadne.trqn_geodesic_bt_mode = "matrix_free"
    changed.ariadne.trqn_geodesic_dt = 0.02
    changed.ariadne.trqn_geodesic_tol = 2.0e-8
    changed.ariadne.trqn_bt_ic_tol = 2.0e-6
    changed.ariadne.trqn_max_backtransform_iter = 75
    changed.ariadne.trqn_trust_min = 2.0e-4

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert sorted(c.path for c in review.allowed_changes) == [
        "ariadne.trqn_backtransform_mode",
        "ariadne.trqn_bt_ic_tol",
        "ariadne.trqn_geodesic_bt_mode",
        "ariadne.trqn_geodesic_dt",
        "ariadne.trqn_geodesic_tol",
        "ariadne.trqn_max_backtransform_iter",
        "ariadne.trqn_trust_min",
    ]
    assert {c.category for c in review.allowed_changes} == {"pre_ariadne"}
    assert not review.blocked_changes


def test_ariadne_config_change_blocks_after_result_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ariadne.trqn_backtransform_mode = "newton"
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001" / "ariadne" / "seeds" / "seed-000001"
    iter_dir.mkdir(parents=True)
    (iter_dir / "result.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "ariadne.trqn_backtransform_mode"
    ]
    assert review.blocked_changes[0].category == "postprocess_locked"


def test_ariadne_config_change_blocks_halted_uncommitted_result(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ariadne.trqn_backtransform_mode = "newton"
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001" / "ariadne" / "seeds" / "seed-000001"
    iter_dir.mkdir(parents=True)
    (iter_dir / "result.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.HALTED
    proposed.iteration = 1
    proposed.reference_data_version = 0
    proposed.models_version = 0
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "ariadne.trqn_backtransform_mode"
    ]
    assert "uncommitted ARIADNE" in review.blocked_changes[0].reason


def test_ariadne_config_change_allows_committed_historical_result(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ariadne.trqn_backtransform_mode = "newton"
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001" / "ariadne" / "seeds" / "seed-000001"
    iter_dir.mkdir(parents=True)
    (iter_dir / "result.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.STOP_CHECK
    proposed.iteration = 1
    proposed.reference_data_version = 1
    proposed.models_version = 1
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert [c.path for c in review.allowed_changes] == [
        "ariadne.trqn_backtransform_mode"
    ]
    assert not review.blocked_changes


def test_phase_b_config_change_blocks_after_selection_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.phase_b.beta = 0.40
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    (iter_dir / "phase_b").mkdir(parents=True)
    (iter_dir / "phase_b" / "SELECTION.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.PHASE_B_DIVERSITY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == ["phase_b.beta"]
    assert review.blocked_changes[0].category == "postprocess_locked"


def test_batch_allocation_change_blocks_after_phase_b_selection_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.point_allocation.batch_training_size = 5
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    (iter_dir / "phase_b").mkdir(parents=True)
    (iter_dir / "phase_b" / "SELECTION.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.PHASE_B_DIVERSITY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "point_allocation.batch_training_size"
    ]
    assert "Phase B" in review.blocked_changes[0].reason


def test_seed_selection_change_blocks_after_ariadne_result_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.seed_selection.n_seeds_per_iteration = 49
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001" / "ariadne" / "seeds" / "seed-000001"
    iter_dir.mkdir(parents=True)
    (iter_dir / "result.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "seed_selection.n_seeds_per_iteration"
    ]
    assert "ARIADNE" in review.blocked_changes[0].reason


def test_geometry_novelty_change_allowed_before_phase_b_outputs(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.geometry_novelty.fallback_scale_angstrom = 0.02

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert sorted(c.path for c in review.allowed_changes) == [
        "geometry_novelty.fallback_scale_angstrom",
    ]
    assert not review.blocked_changes


def test_geometry_novelty_change_blocks_after_ariadne_result_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.geometry_novelty.fallback_scale_angstrom = 0.02
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001" / "ariadne" / "seeds" / "seed-000001"
    iter_dir.mkdir(parents=True)
    (iter_dir / "result.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "geometry_novelty.fallback_scale_angstrom"
    ]
    assert review.blocked_changes[0].category == "postprocess_locked"
    assert "ARIADNE" in review.blocked_changes[0].reason


def test_geometry_novelty_change_blocks_after_phase_b_selection_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.geometry_novelty.fallback_scale_angstrom = 0.02
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    (iter_dir / "phase_b").mkdir(parents=True)
    (iter_dir / "phase_b" / "SELECTION.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.PHASE_B_DIVERSITY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "geometry_novelty.fallback_scale_angstrom"
    ]
    assert review.blocked_changes[0].category == "postprocess_locked"


def test_sampling_aggressiveness_change_allowed_before_ariadne_outputs(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.campaign.sampling_aggressiveness = 6

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert [c.path for c in review.allowed_changes] == [
        "campaign.sampling_aggressiveness"
    ]
    assert not review.blocked_changes


def test_sampling_aggressiveness_change_blocks_after_protocol_manifest_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.campaign.sampling_aggressiveness = 6
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    (iter_dir / "protocol").mkdir(parents=True)
    (iter_dir / "protocol" / "SAMPLING_PROTOCOL_RESOLVED.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "campaign.sampling_aggressiveness"
    ]
    assert review.blocked_changes[0].category == "postprocess_locked"
    assert "ARIADNE" in review.blocked_changes[0].reason


def test_sampling_aggressiveness_change_blocks_after_scale_model_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.campaign.sampling_aggressiveness = 6
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    (iter_dir / "protocol").mkdir(parents=True)
    (iter_dir / "protocol" / "SAMPLING_SCALE_MODEL.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "campaign.sampling_aggressiveness"
    ]
    assert review.blocked_changes[0].category == "postprocess_locked"


def test_sampling_aggressiveness_change_blocks_after_phase_b_selection_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.campaign.sampling_aggressiveness = 6
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    (iter_dir / "phase_b").mkdir(parents=True)
    (iter_dir / "phase_b" / "SELECTION.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.PHASE_B_DIVERSITY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "campaign.sampling_aggressiveness"
    ]
    assert review.blocked_changes[0].category == "postprocess_locked"
    assert "Phase B" in review.blocked_changes[0].reason


def test_sampling_aggressiveness_same_value_is_noop_after_outputs(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    (iter_dir / "ariadne").mkdir(parents=True)
    (iter_dir / "ariadne" / "RESULTS.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.iteration = 1
    review = review_config_changes(campaign, changed, proposed)

    assert not review.changed
    assert not review.allowed_changes
    assert not review.blocked_changes


def test_phase_b_config_change_blocks_halted_uncommitted_selection(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.phase_b.beta = 0.40
    iter_dir = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    (iter_dir / "phase_b").mkdir(parents=True)
    (iter_dir / "phase_b" / "SELECTION.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.HALTED
    proposed.iteration = 1
    proposed.reference_data_version = 0
    proposed.models_version = 0
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == ["phase_b.beta"]
    assert "uncommitted Phase B" in review.blocked_changes[0].reason


def test_gaussian_method_change_allowed_before_first_gaussian(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.gaussian.method = "b3lyp"

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.PHASE_A_DIVERSITY
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert [c.path for c in review.allowed_changes] == ["gaussian.method"]
    assert not review.blocked_changes


def test_gaussian_basis_change_allowed_after_phase_a_before_gaussian(tmp_path):
    campaign = _campaign(tmp_path)
    _write_phase_a_sample(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.gaussian.basis_set = "def2-TZVP"

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_GAUSSIAN
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert [c.path for c in review.allowed_changes] == ["gaussian.basis_set"]
    assert not review.blocked_changes


def test_gaussian_basis_change_blocks_after_gaussian_staging_exists(tmp_path):
    campaign = _campaign(tmp_path)
    staging = campaign / ".DATA" / "STAGING" / "initial" / "POINT_0000.pointdir"
    staging.mkdir(parents=True)
    (staging / "input.gjf").write_text("%chk=input.chk\n", encoding="utf-8")
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.gaussian.basis_set = "def2-TZVP"

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_GAUSSIAN
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == ["gaussian.basis_set"]
    assert "Gaussian staging" in review.blocked_changes[0].reason


def test_bootstrap_size_blocks_after_phase_a_output(tmp_path):
    campaign = _campaign(tmp_path)
    _write_phase_a_sample(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.point_allocation.bootstrap_training_size = 24

    review = review_config_changes(campaign, changed, fresh_campaign_state())

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "point_allocation.bootstrap_training_size"
    ]
    assert "Phase A" in review.blocked_changes[0].reason


def test_custom_bootstrap_flag_blocks_after_phase_a_output(tmp_path):
    campaign = _campaign(tmp_path)
    _write_phase_a_sample(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.campaign.custom_bootstrap = True

    review = review_config_changes(campaign, changed, fresh_campaign_state())

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == ["campaign.custom_bootstrap"]
    assert "Phase A" in review.blocked_changes[0].reason


def test_system_name_is_immutable_once_config_lock_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.system_name = "SYSTEM2"

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_FEREBUS
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == ["campaign.system_name"]
    assert "immutable" in review.blocked_changes[0].reason


def test_system_name_blocks_after_ferebus_staging_exists(tmp_path):
    campaign = _campaign(tmp_path)
    staging = campaign / "TRAINED_MODELS" / "iteration-staging"
    staging.mkdir(parents=True)
    (staging / "FEREBUS_TASKS.json").write_text("{}", encoding="utf-8")
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.system_name = "SYSTEM2"

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_FEREBUS
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == ["campaign.system_name"]
    assert "immutable" in review.blocked_changes[0].reason


def test_removed_fractional_split_controls_are_absent_from_schema(tmp_path):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    assert not hasattr(config.ferebus, "train_fraction")
    assert not hasattr(config.ferebus, "internal_validation_fraction")
    assert not hasattr(config, "active_batch")
    assert not hasattr(config, "bootstrap")


def test_removed_post_qm_split_block_is_absent_from_schema(tmp_path):
    assert not hasattr(CampaignConfig(), "split")


def test_reconcile_apply_promotes_state_and_cleans_ferebus_staging(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    _write_config(campaign, original)
    stale = campaign / "TRAINED_MODELS" / "iteration-staging"
    stale.mkdir(parents=True)
    (stale / "runFerebus.sh").write_text("# stale\n", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is CampaignPhase.INITIAL_FEREBUS
    assert state.reference_data_version == 0
    assert state.models_version == -1
    assert not stale.exists()
    assert "Result: APPLIED" in out
    assert "Applied Changes" in out
    assert "state written" in out
    assert not (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").exists()
    assert sorted(
        (campaign / ".DATA" / "ACTIVE_LEARNING").glob(
            "state.json.proposed.applied-*"
        )
    )
    assert sorted(
        (campaign / ".DATA" / "ACTIVE_LEARNING").glob(
            "state.json.before-reconcile-*"
        )
    )
    lock = json.loads(config_lock_path(campaign).read_text(encoding="utf-8"))
    assert lock["canonical_config"]["ferebus"]["kernel"] == "periodic_rbf"


def test_completed_ferebus_staging_is_archived_when_committed_models_match(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    import ichor.core.files as core_files
    from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
    from ichor_hpc.tests.test_active_learning.test_ferebus_staging import (
        _prepare_bootstrap_training,
    )

    config = CampaignConfig()
    _prepare_bootstrap_training(
        campaign,
        config,
        n_train=2,
        n_int_val=2,
        n_ext_val=2,
    )
    monkeypatch.setattr(
        core_files,
        "PointsDirectory",
        _FerebusFixturePointsDirectory,
    )
    DryRunPhaseExecutor(campaign, config)._commit_dry_model_snapshot(0)
    staging = campaign / "TRAINED_MODELS" / "iteration-staging"
    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.models_version = 0

    archived = config_lock_mod.clean_model_iteration_staging_for_reconcile(
        campaign,
        proposed,
    )

    assert len(archived) == 1
    archived_path = Path(archived[0])
    assert archived_path.name.startswith("iteration-staging.before-reconcile-")
    assert not staging.exists()
    assert (
        archived_path / "iqa" / "O1" / "SYSTEM_iqa_O1.model"
    ).is_file()
    assert (
        campaign
        / "TRAINED_MODELS"
        / "iteration-000000"
        / "iqa"
        / "O1"
        / "SYSTEM_iqa_O1.model"
    ).is_file()


def test_tampered_ferebus_staging_is_archived_without_changing_committed_model(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    import ichor.core.files as core_files
    from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
    from ichor_hpc.tests.test_active_learning.test_ferebus_staging import (
        _prepare_bootstrap_training,
    )

    config = CampaignConfig()
    _prepare_bootstrap_training(
        campaign,
        config,
        n_train=2,
        n_int_val=2,
        n_ext_val=2,
    )
    monkeypatch.setattr(
        core_files,
        "PointsDirectory",
        _FerebusFixturePointsDirectory,
    )
    DryRunPhaseExecutor(campaign, config)._commit_dry_model_snapshot(0)
    staging = campaign / "TRAINED_MODELS" / "iteration-staging"
    staged_model = staging / "iqa" / "O1" / "SYSTEM_iqa_O1.model"
    staged_model.write_text(
        staged_model.read_text(encoding="utf-8") + "\n# hash drift\n",
        encoding="utf-8",
    )
    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.models_version = 0

    archived = config_lock_mod.clean_model_iteration_staging_for_reconcile(
        campaign,
        proposed,
    )

    assert len(archived) == 1
    archived_model = (
        Path(archived[0]) / "iqa" / "O1" / "SYSTEM_iqa_O1.model"
    )
    assert "# hash drift" in archived_model.read_text(encoding="utf-8")
    committed_model = (
        campaign
        / "TRAINED_MODELS"
        / "iteration-000000"
        / "iqa"
        / "O1"
        / "SYSTEM_iqa_O1.model"
    )
    assert "# hash drift" not in committed_model.read_text(encoding="utf-8")


def test_reconcile_archives_dangling_trained_model_version_staging(tmp_path):
    campaign = _campaign(tmp_path)
    dangling = campaign / "TRAINED_MODELS" / "iteration-000001.staging"
    dangling.mkdir(parents=True)
    (dangling / "partial.txt").write_text("partial\n", encoding="utf-8")

    archived = config_lock_mod.clean_model_iteration_staging_for_reconcile(
        campaign,
        fresh_campaign_state(),
    )

    assert len(archived) == 1
    archived_path = Path(archived[0])
    assert archived_path.name.startswith(
        "iteration-000001.staging.before-reconcile-"
    )
    assert (archived_path / "partial.txt").is_file()
    assert not dangling.exists()


def test_reconcile_apply_write_state_failure_keeps_old_state(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    old_state = _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)

    def fail_write_state(path, state):
        raise OSError("write failed")

    monkeypatch.setattr(cli_mod, "write_state", fail_write_state)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "failed to write recovered state.json" in err
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is old_state.phase
    assert state.reference_data_version == old_state.reference_data_version
    assert (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").is_file()


def test_reconcile_apply_config_lock_failure_restores_old_state(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    old_state = _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    reference_versions = VersionedDirectory(campaign / "QM_REFERENCE_DATA")
    current_link = reference_versions.current_link_path()
    current_pointer = reference_versions._pointer_path()
    if current_link.is_symlink():
        current_link.unlink()
    if current_pointer.is_file():
        current_pointer.unlink()
    assert reference_versions.current_version() is None

    def fail_config_lock(path, config):
        raise OSError("lock failed")

    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    real_write_state = cli_mod.write_state
    write_state_calls = []

    def spy_write_state(path, state):
        write_state_calls.append(Path(path))
        return real_write_state(path, state)

    monkeypatch.setattr(cli_mod, "write_state", spy_write_state)
    monkeypatch.setattr(cli_mod, "apply_config_lock_update", fail_config_lock)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
        )
    )
    err = capsys.readouterr().err

    assert rc == 8
    assert "failed to update config lock" in err
    state = read_state(state_path)
    assert state.phase is old_state.phase
    assert state.reference_data_version == old_state.reference_data_version
    assert reference_versions.current_version() is None
    assert (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").is_file()
    assert write_state_calls.count(state_path) >= 2


def test_reconcile_config_lock_failure_reports_prior_staging_archive(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    old_state = _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    data_staging = campaign / ".DATA" / "STAGING"
    stale_file = data_staging / "INITIAL_AIMALL" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old scratch", encoding="utf-8")

    def fail_config_lock(path, config):
        raise OSError("lock failed")

    monkeypatch.setattr(cli_mod, "apply_config_lock_update", fail_config_lock)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
        )
    )
    err = capsys.readouterr().err

    assert rc == 8
    assert "failed to update config lock" in err
    assert "Some cleanup/archive operations already happened" in err
    assert "STAGING.before-reconcile-" in err
    assert "State/config lock was not applied." in err
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is old_state.phase


def test_reconcile_apply_refuses_when_daemon_lock_may_be_active(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)

    monkeypatch.setattr(
        cli_mod,
        "_reconcile_runtime_status",
        lambda campaign: {
            "lock_held": True,
            "lease_heartbeat": None,
            "background_pid": None,
            "background_pid_alive": False,
            "reconcile_apply_blockers": ["daemon lock is held"],
        },
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "daemon may still be running" in err
    assert not (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").exists()


def test_reconcile_non_apply_warns_when_daemon_lock_may_be_active(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    monkeypatch.setattr(
        cli_mod,
        "_reconcile_runtime_status",
        lambda campaign: {
            "lock_held": True,
            "lease_heartbeat": None,
            "background_pid": None,
            "background_pid_alive": False,
            "reconcile_apply_blockers": ["daemon lock is held"],
        },
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=False,
            restore_config_from_lock=False,
        )
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "daemon may still be running" in captured.err
    assert (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").exists()


def test_reconcile_restore_config_from_lock_writes_proposal(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    config.system_name = "RESTORED"
    write_config_lock(campaign, config)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=False,
            restore_config_from_lock=True,
        )
    )
    out = capsys.readouterr().out

    proposed = campaign / "campaign.yaml.proposed"
    assert rc == 0
    assert proposed.is_file()
    assert "Result: CONFIG PROPOSAL" in out
    assert "Config Proposal" in out
    restored = CampaignConfig.from_yaml(proposed)
    assert restored.system_name == "RESTORED"


def test_reconcile_restore_config_from_lock_uses_current_directory(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    config.system_name = "RESTORED_CWD"
    write_config_lock(campaign, config)
    monkeypatch.chdir(campaign)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=None,
            allow_fresh_init=False,
            apply=False,
            restore_config_from_lock=True,
        )
    )
    capsys.readouterr()

    proposed = campaign / "campaign.yaml.proposed"
    assert rc == 0
    assert proposed.is_file()
    restored = CampaignConfig.from_yaml(proposed)
    assert restored.system_name == "RESTORED_CWD"


def test_restore_config_from_lock_archives_existing_proposal(tmp_path):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    config.system_name = "RESTORED_ARCHIVE"
    write_config_lock(campaign, config)
    old_proposal = campaign / "campaign.yaml.proposed"
    old_proposal.write_text("old proposal marker\n", encoding="utf-8")

    target = restore_config_from_lock_proposal(campaign)

    archived = sorted(campaign.glob("campaign.yaml.proposed.before-*"))
    assert target == old_proposal
    assert target.is_file()
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == "old proposal marker\n"
    restored = CampaignConfig.from_yaml(target)
    assert restored.system_name == "RESTORED_ARCHIVE"


def test_reconcile_restore_config_from_lock_refuses_existing_campaign_yaml(
    tmp_path,
    capsys,
):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=False,
            restore_config_from_lock=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 8
    assert "campaign.yaml already exists" in err


def test_reconcile_apply_archives_data_staging_for_ferebus_reentry(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    write_config_lock(campaign, CampaignConfig())
    changed = CampaignConfig()
    changed.ferebus.kernel = "rbf"
    _write_config(campaign, changed)
    data_staging = campaign / ".DATA" / "STAGING"
    stale_file = data_staging / "INITIAL_AIMALL" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old scratch", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    captured = capsys.readouterr()
    out = captured.out

    assert rc == 0, captured.err
    archived = sorted((campaign / ".DATA").glob("STAGING.before-reconcile-*"))
    assert len(archived) == 1
    assert (archived[0] / "INITIAL_AIMALL" / "old.txt").read_text(
        encoding="utf-8"
    ) == "old scratch"
    assert data_staging.is_dir()
    assert list(data_staging.iterdir()) == []
    assert "archived staging:" in out


def test_reconcile_apply_archives_safe_dangling_training_staging(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    tv = VersionedDirectory(campaign / "QM_REFERENCE_DATA")
    dangling = tv.staging_path(1)
    dangling.mkdir(parents=True)
    (dangling / "partial.txt").write_text("partial\n", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
        )
    )
    captured = capsys.readouterr()
    out = captured.out

    assert rc == 0, captured.err
    archived = sorted((campaign / "QM_REFERENCE_DATA").glob("iteration-000001.staging.before-reconcile-*"))
    assert len(archived) == 1
    assert (archived[0] / "partial.txt").read_text(encoding="utf-8") == "partial\n"
    assert "archived reference-data staging:" in out


def test_reconcile_apply_cleans_transient_halted_ariadne_reentry(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.reference_data_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    scripts = campaign / ".DATA" / "SCRIPTS"
    (scripts / "OUTPUTS").mkdir(parents=True)
    (scripts / "ERRORS").mkdir()
    (scripts / "ARIADNE_ARRAY-1.sh").write_text("# stale\n", encoding="utf-8")
    (scripts / "ERRORS" / "ARIADNE_ARRAY-1.e").write_text(
        "old error\n",
        encoding="utf-8",
    )
    model_staging = campaign / "TRAINED_MODELS" / "iteration-staging"
    model_staging.mkdir(parents=True)
    (model_staging / "stale.model").write_text("stale\n", encoding="utf-8")
    committed_model = campaign / "TRAINED_MODELS" / "iteration-000000"
    committed_model.mkdir()
    (committed_model / "marker.txt").write_text("committed\n", encoding="utf-8")
    committed_training = campaign / "QM_REFERENCE_DATA" / "iteration-000000"
    committed_training.mkdir()
    (committed_training / "marker.txt").write_text("committed\n", encoding="utf-8")

    first_state = fresh_campaign_state(max_iterations=1)
    first_state.phase = CampaignPhase.HALTED
    first_state.iteration = 1
    first_state.reference_data_version = 0
    first_state.models_version = 0
    second_state = fresh_campaign_state(max_iterations=1)
    second_state.phase = CampaignPhase.STOP_CHECK
    second_state.iteration = 1
    second_state.reference_data_version = 0
    second_state.models_version = 0
    reports = iter([
        ReconciliationReport(
            proposed_state=first_state,
            committed_reference_data_versions=[0],
            committed_model_versions=[0],
            last_phase_in_journal=CampaignPhase.ARIADNE_ARRAY.value,
            last_iteration_in_journal=1,
            last_phase_event_in_journal="halt",
            last_phase_retryable=True,
            notes=["re-entry HALTED because committed artefacts need operator review"],
            unsafe_reasons=[
                ".DATA/SCRIPTS contains sbatch scripts",
                "dangling model staging directories exist",
            ],
        ),
        ReconciliationReport(
            proposed_state=second_state,
            committed_reference_data_versions=[0],
            committed_model_versions=[0],
            last_phase_in_journal=CampaignPhase.ARIADNE_ARRAY.value,
            last_iteration_in_journal=1,
            notes=["re-entry at STOP_CHECK (next tick decides loop/terminate)"],
            unsafe_reasons=[],
        ),
    ])

    monkeypatch.setattr(cli_mod, "propose_recovery", lambda *args, **kwargs: next(reports))
    monkeypatch.setattr(
        config_lock_mod,
        "verify_committed_model_version",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        cli_mod,
        "_reconcile_apply_contract_error",
        lambda *args, **kwargs: None,
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    captured = capsys.readouterr()
    out = captured.out

    assert rc == 0, captured.err
    recovered = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert recovered.phase is CampaignPhase.ARIADNE_ARRAY
    assert recovered.iteration == 1
    assert not model_staging.exists()
    assert (committed_model / "marker.txt").read_text(encoding="utf-8") == "committed\n"
    archived_scripts = sorted(
        (scripts / "LEGACY_BEFORE_RECONCILE").glob("*")
    )
    assert len(archived_scripts) == 1
    assert (archived_scripts[0] / "ARIADNE_ARRAY-1.sh").is_file()
    assert (archived_scripts[0] / "ERRORS" / "ARIADNE_ARRAY-1.e").is_file()
    assert not (scripts / "OUTPUTS").exists()
    assert not (scripts / "ERRORS").exists()
    assert "archived stale scripts:" in out
    assert "removed model staging:" in out


def test_reconcile_apply_recovers_prebootstrap_phase_a_submission_failure(
    tmp_path,
    capsys,
):
    campaign = _campaign(tmp_path)
    _write_pool(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.iteration = 0
    state.reference_data_version = 0
    state.validation_set_version = 0
    state.models_version = 0
    state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] = None
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.PHASE_A_DIVERSITY.value,
        iteration=0,
    )
    submission_intent.mark_failed(
        campaign,
        CampaignPhase.PHASE_A_DIVERSITY.value,
        0,
        "backend_submission_failed: partition 'multicore_small' is not present",
    )
    from ichor.hpc.active_learning.daemon.journal import append_event

    append_event(
        campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson",
        "halt",
        from_phase=CampaignPhase.PHASE_A_DIVERSITY.value,
        iteration=0,
        reason="backend_submission_failed: partition 'multicore_small' is not present",
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
            archive_staging=False,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    recovered = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert recovered.phase is CampaignPhase.PHASE_A_DIVERSITY
    assert recovered.reference_data_version == -1
    assert recovered.validation_set_version == -1
    assert recovered.models_version == -1
    assert "ichor-al-daemon start --campaign-dir " + str(campaign) in out
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.PHASE_A_DIVERSITY.value,
        0,
    )
    assert intent["status"] == "SUPERSEDED"
    assert intent["reason"] == "reconcile_apply_retry"


def test_reconcile_apply_keeps_data_staging_blocked_for_non_ferebus_reentry(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.reference_data_version = 0
    state.models_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    cfg = CampaignConfig()
    write_config_lock(campaign, cfg)
    _write_config(campaign, cfg)
    data_staging = campaign / ".DATA" / "STAGING"
    stale_file = data_staging / "INITIAL_GAUSSIAN" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old scratch", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert ".DATA/STAGING is non-empty" in err
    assert stale_file.exists()


def test_reconcile_apply_archive_staging_explicitly_handles_non_ferebus_reentry(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _write_pool(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)

    tv = VersionedDirectory(campaign / "QM_REFERENCE_DATA")
    staging = tv.stage(None, 0)
    (staging / "marker.txt").write_text("training", encoding="utf-8")
    tv.commit(0)
    mv = VersionedDirectory(campaign / "TRAINED_MODELS")
    staging = mv.stage(None, 0)
    (staging / "marker.txt").write_text("model", encoding="utf-8")
    mv.commit(0)
    monkeypatch.setattr(
        reconcile_mod,
        "verify_committed_model_version",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        reconcile_mod,
        "verify_committed_reference_data_version",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        reconcile_mod,
        "_validate_recovered_state_contract",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        cli_mod,
        "_reconcile_apply_contract_error",
        lambda *args, **kwargs: None,
    )

    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    data_staging = campaign / ".DATA" / "STAGING"
    stale_file = data_staging / "GAUSSIAN" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old scratch", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
            archive_staging=True,
        )
    )
    captured = capsys.readouterr()
    out = captured.out

    assert rc == 0, captured.err
    archived = sorted((campaign / ".DATA").glob("STAGING.archived-*"))
    assert len(archived) == 1
    assert (archived[0] / "GAUSSIAN" / "old.txt").read_text(
        encoding="utf-8"
    ) == "old scratch"
    assert data_staging.is_dir()
    assert list(data_staging.iterdir()) == []
    assert "archived staging:" in out
    recovered = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert recovered.phase is CampaignPhase.SEED_SELECT
    events = list(iter_events(campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"))
    assert any(e.get("event") == "staging_archived" for e in events)


def test_operator_archive_staging_blocks_protected_active_handoff(tmp_path):
    campaign = _campaign(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    staging = campaign / ".DATA" / "STAGING" / "iter_1"
    pointdir = staging / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    stg.write_points_file(staging, [pointdir])
    stg.write_quantum_acceptance_manifest(
        staging,
        phase_name=CampaignPhase.GAUSSIAN.value,
        iteration=1,
        accepted=[pointdir],
        rejected=[],
    )
    report = ReconciliationReport(proposed_state=state)

    blockers = cli_mod._operator_staging_archive_blockers(campaign, report, {})

    assert any("protected handoff for AIMALL@1" in item for item in blockers)


def test_reconcile_apply_restores_archived_initial_gaussian_handoff(
    tmp_path,
    capsys,
):
    campaign = _campaign(tmp_path)
    _write_pool(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.reference_data_version = -1
    state.models_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    archived = _write_archived_bootstrap_handoff(
        campaign,
        phase=CampaignPhase.INITIAL_GAUSSIAN.value,
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
            archive_staging=False,
        )
    )
    captured = capsys.readouterr()
    out = captured.out

    assert rc == 0, captured.err
    recovered = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert recovered.phase is CampaignPhase.INITIAL_AIMALL
    restored = campaign / ".DATA" / "STAGING" / "initial"
    assert (restored / "accepted_pointdirs.json").is_file()
    assert (restored / "POINT_0000.pointdir" / "input.wfn").read_text(
        encoding="utf-8"
    ) == "wfn\n"
    assert "restored bootstrap staging:" in out
    events = list(iter_events(campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"))
    restored_events = [
        e for e in events if e.get("event") == "staging_restored_from_archive"
    ]
    assert len(restored_events) == 1
    assert restored_events[0]["source_path"] == str(archived)
    assert restored_events[0]["source_phase"] == CampaignPhase.INITIAL_GAUSSIAN.value


def test_reconcile_apply_archive_staging_refuses_pending_job(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.pending_jobs[CampaignPhase.GAUSSIAN.value] = "12345"
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    data_staging = campaign / ".DATA" / "STAGING"
    stale_file = data_staging / "GAUSSIAN" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old scratch", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
            archive_staging=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "DATA/STAGING is non-empty" in err
    assert "pending job" in err
    assert stale_file.exists()


def test_reconcile_apply_refuses_locked_config_change(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.campaign.custom_bootstrap = True
    _write_phase_a_sample(campaign)
    _write_config(campaign, changed)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 8
    assert "locked changes" in err
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is CampaignPhase.HALTED


def test_reconcile_apply_resolves_terminal_submission_intent(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    _write_submitted_initial_ferebus_intent(campaign)

    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda job_id: sacct_poll.JobQueueLookup(active=False, rows=[]),
    )
    monkeypatch.setattr(
        sacct_poll,
        "poll_job",
        lambda job_id: [
            sacct_poll.JobObservation(
                job_id=str(job_id) + "_[1-12]",
                status=sacct_poll.JobStatus.CANCELLED,
                exit_code=(0, 0),
                elapsed_seconds=0,
            )
        ],
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Resolved terminal submission intents" in out
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is CampaignPhase.INITIAL_FEREBUS
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUPERSEDED"
    assert intent["reason"] == "reconcile_apply_retry"


def test_reconcile_apply_resolves_cancelled_intent_when_squeue_invalid_job_id(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    _write_submitted_initial_ferebus_intent(campaign)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "squeue":
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="slurm_load_jobs error: Invalid job id specified\n",
            )
        if cmd[0] == "sacct":
            return SimpleNamespace(
                returncode=0,
                stdout="".join(
                    "16218598_" + str(index) + "|CANCELLED by 494098|0:0|00:00:00\n"
                    for index in range(12)
                ),
                stderr="",
            )
        raise AssertionError("unexpected command: " + repr(cmd))

    monkeypatch.setattr(sacct_poll.subprocess, "run", fake_run)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Resolved terminal submission intents" in out
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is CampaignPhase.INITIAL_FEREBUS
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUPERSEDED"
    assert intent["reason"] == "reconcile_apply_retry"


def test_reconcile_apply_supersedes_stale_pre_submit_without_job_id(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    config.runtime.lease_stale_seconds = 10
    config.runtime.lease_heartbeat_seconds = 1
    config.runtime.lease_heartbeat_failure_max = 3
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    phase = CampaignPhase.GAUSSIAN.value
    _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            None,
            inconclusive=False,
            rows=[],
        ),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Resolved terminal submission intents" in out
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "SUPERSEDED"
    assert intent["reason"] == "reconcile_apply_pre_submit_no_job_id"


def test_reconcile_apply_uses_job_name_accounting_for_ferebus_pre_submit_without_job_id(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    config.runtime.lease_stale_seconds = 10
    config.runtime.lease_heartbeat_seconds = 1
    config.runtime.lease_heartbeat_failure_max = 3
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    phase = CampaignPhase.INITIAL_FEREBUS.value
    _write_stale_pre_submit_intent(campaign, phase)

    def fail_running_lookup(name, **kwargs):
        raise AssertionError("PRE_SUBMIT recovery must use accounting lookup")

    monkeypatch.setattr(sacct_poll, "find_running_job_by_name_detailed", fail_running_lookup)
    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            None,
            inconclusive=False,
            rows=[],
        ),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Resolved terminal submission intents" in out
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "SUPERSEDED"
    assert intent["reason"] == "reconcile_apply_pre_submit_no_job_id"


def test_reconcile_apply_blocks_pre_submit_without_job_id_when_accounting_completed(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    config.runtime.lease_stale_seconds = 10
    config.runtime.lease_heartbeat_seconds = 1
    config.runtime.lease_heartbeat_failure_max = 3
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    phase = CampaignPhase.GAUSSIAN.value
    _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            "222",
            terminal=True,
            successful=True,
            failed=False,
            inconclusive=False,
            rows=[("222", "COMPLETED")],
        ),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "postprocess/operator review is required" in err
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "PRE_SUBMIT"


def test_reconcile_apply_marks_failed_pre_submit_without_job_id_from_accounting(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    config.runtime.lease_stale_seconds = 10
    config.runtime.lease_heartbeat_seconds = 1
    config.runtime.lease_heartbeat_failure_max = 3
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    phase = CampaignPhase.GAUSSIAN.value
    _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            "222",
            terminal=True,
            successful=False,
            failed=True,
            inconclusive=False,
            rows=[("222", "FAILED")],
        ),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Resolved terminal submission intents" in out
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "FAILED"
    assert intent["reason"] == "reconcile_apply_terminal_job:FAILED"


def test_reconcile_apply_blocks_pre_submit_without_job_id_when_job_exists(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    config.runtime.lease_stale_seconds = 10
    config.runtime.lease_heartbeat_seconds = 1
    config.runtime.lease_heartbeat_failure_max = 3
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    phase = CampaignPhase.GAUSSIAN.value
    payload = _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            "222",
            terminal=False,
            inconclusive=False,
            rows=[("222", "RUNNING")],
        ),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "matching scheduler job exists" in err
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "PRE_SUBMIT"
    assert intent["expected_job_name"] == payload["expected_job_name"]


def test_reconcile_apply_blocks_pre_submit_without_job_id_on_lookup_failure(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    config.runtime.lease_stale_seconds = 10
    config.runtime.lease_heartbeat_seconds = 1
    config.runtime.lease_heartbeat_failure_max = 3
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    phase = CampaignPhase.GAUSSIAN.value
    _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            None,
            inconclusive=True,
            rows=[],
            error="sacct unavailable",
        ),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "job-name lookup inconclusive" in err
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "PRE_SUBMIT"


def test_reconcile_apply_refuses_when_submission_intent_job_is_still_active(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    _write_submitted_initial_ferebus_intent(campaign)

    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda job_id: sacct_poll.JobQueueLookup(
            active=True,
            rows=[(str(job_id), "RUNNING")],
        ),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "job is still active in squeue" in err
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUBMITTED"


def test_reconcile_apply_refuses_when_terminal_intent_sacct_is_inconclusive(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    _write_submitted_initial_ferebus_intent(campaign)

    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda job_id: sacct_poll.JobQueueLookup(active=False, rows=[]),
    )
    monkeypatch.setattr(
        sacct_poll,
        "poll_job",
        lambda job_id: [],
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "sacct returned no rows" in err
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUBMITTED"


def test_reconcile_apply_refuses_completed_intent_without_postprocess_verification(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_reference_data_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    _write_submitted_initial_ferebus_intent(campaign)

    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda job_id: sacct_poll.JobQueueLookup(active=False, rows=[]),
    )
    monkeypatch.setattr(
        sacct_poll,
        "poll_job",
        lambda job_id: [
            sacct_poll.JobObservation(
                job_id=str(job_id) + "_[1-12]",
                status=sacct_poll.JobStatus.COMPLETED,
                exit_code=(0, 0),
                elapsed_seconds=60,
            )
        ],
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "without postprocess verification" in err
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUBMITTED"


def test_start_refuses_config_drift_without_reconcile_apply(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _write_halted_pre_ferebus_state(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ferebus.kernel = "rbf"
    _write_config(campaign, changed)

    rc = cmd_start(
        argparse.Namespace(
            campaign_dir=str(campaign),
            config=None,
            mode="dry_run",
            poll_interval=None,
            max_ticks=1,
            background=False,
            foreground=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 20
    assert "campaign is HALTED" in err
    assert "reconcile" in err


def test_start_allows_clean_first_run_with_config_and_pool(tmp_path):
    campaign = _campaign(tmp_path)
    config = CampaignConfig(max_iterations=1)
    _write_config(campaign, config)
    _write_pool(campaign)
    write_config_lock(campaign, config)
    write_state(
        campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json",
        fresh_campaign_state(max_iterations=1),
    )

    rc = cmd_start(
        argparse.Namespace(
            campaign_dir=str(campaign),
            config=None,
            mode="dry_run",
            poll_interval=None,
            max_ticks=1,
            background=False,
            foreground=True,
        )
    )

    assert rc == 0
    assert (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json").is_file()


def test_start_refuses_missing_state_when_config_lock_exists(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    config = CampaignConfig(max_iterations=1)
    _write_config(campaign, config)
    write_config_lock(campaign, config)

    rc = cmd_start(
        argparse.Namespace(
            campaign_dir=str(campaign),
            config=None,
            mode="dry_run",
            poll_interval=None,
            max_ticks=1,
            background=False,
        )
    )
    err = capsys.readouterr().err

    assert rc == 8
    assert "state.json is missing" in err
    assert ".DATA/ACTIVE_LEARNING/config_lock.json" in err
    assert not (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json").exists()


def test_start_refuses_missing_state_in_nonempty_campaign(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    _write_config(campaign, config)
    _commit_reference_data_version(campaign, 0)

    rc = cmd_start(
        argparse.Namespace(
            campaign_dir=str(campaign),
            config=None,
            mode="dry_run",
            poll_interval=None,
            max_ticks=1,
            background=False,
        )
    )
    err = capsys.readouterr().err

    assert rc == 8
    assert "state.json is missing but this campaign is not empty" in err
    assert "reconcile --campaign-dir" in err
    assert not (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json").exists()
