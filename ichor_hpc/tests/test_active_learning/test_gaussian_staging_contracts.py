"""Gaussian staging is exact, provenance-bound, and stale-output safe."""

from pathlib import Path

import pytest

from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.input_staging import stage_gaussian_inputs
from ichor.hpc.active_learning.daemon.state import (
    DEFAULT_STATE_FILENAME,
    fresh_campaign_state,
    write_state,
)
from ichor.hpc.active_learning.handoff_manifests import (
    write_phase_a_sample_manifest,
)
from ichor.hpc.active_learning.layout import bootstrap_selection_dir
from ichor.hpc.active_learning.point_allocation import (
    create_point_allocation,
    point_allocation_path,
)
from ichor.hpc.active_learning.sampling.diversity_contract import (
    diversity_selector_contract,
)


def _phase_a_campaign(tmp_path):
    campaign = tmp_path / "campaign"
    operational = campaign / ".DATA" / "ACTIVE_LEARNING"
    operational.mkdir(parents=True)
    uid = "gaussian-staging-test"
    write_state(
        operational / DEFAULT_STATE_FILENAME,
        fresh_campaign_state(campaign_uid=uid),
    )
    source = campaign / "pool-source.xyz"
    source.write_text(
        "2\nframe 0\nH 0.0 0.0 0.0\nH 0.0 0.0 0.75\n",
        encoding="utf-8",
        newline="\n",
    )
    TrajectoryPool.import_from(source, campaign, overwrite=True)
    selection = bootstrap_selection_dir(campaign)
    selection.mkdir(parents=True)
    sample = selection / "selected.xyz"
    index = selection / "selected_indices.dat"
    sample.write_bytes(source.read_bytes())
    index.write_text("0\n", encoding="utf-8", newline="\n")
    allocation_path = point_allocation_path(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid=uid,
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
    slot = allocation["slots"][0]
    primary = [
        {
            **slot["attempts"][0],
            "slot_id": int(slot["slot_id"]),
            "split": str(slot["split"]),
        }
    ]
    write_phase_a_sample_manifest(
        selection,
        {
            "phase": "PHASE_A_DIVERSITY",
            "iteration": 0,
            "selector": diversity_selector_contract(),
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
        },
    )
    return campaign, sample


def test_initial_gaussian_requires_phase_a_handoff(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    sample = campaign / "selected.xyz"
    sample.write_text(
        "1\nframe 0\nH 0.0 0.0 0.0\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="failed to load Phase A provenance context"):
        stage_gaussian_inputs(
            campaign,
            CampaignConfig(),
            "INITIAL_GAUSSIAN",
            0,
            sample,
        )


def test_changed_gjf_removes_stale_quantum_outputs(tmp_path):
    campaign, sample = _phase_a_campaign(tmp_path)
    config = CampaignConfig()
    staging, _ = stage_gaussian_inputs(
        campaign, config, "INITIAL_GAUSSIAN", 0, sample
    )
    pointdir = staging / "POINT_0000.pointdir"
    (pointdir / "input.gau").write_text("stale\n", encoding="utf-8")
    (pointdir / "input.wfn").write_text("stale\n", encoding="utf-8")

    config.gaussian.method = "PBE0"
    stage_gaussian_inputs(campaign, config, "INITIAL_GAUSSIAN", 0, sample)

    assert "PBE0" in (pointdir / "input.gjf").read_text(encoding="utf-8")
    assert not (pointdir / "input.gau").exists()
    assert not (pointdir / "input.wfn").exists()


def test_gaussian_staging_reports_bounded_progress_without_affecting_work(tmp_path):
    campaign, sample = _phase_a_campaign(tmp_path)
    progress = []

    staging, count = stage_gaussian_inputs(
        campaign,
        CampaignConfig(),
        "INITIAL_GAUSSIAN",
        0,
        sample,
        progress_callback=lambda **payload: progress.append(payload),
    )

    assert count == 1
    assert (staging / "POINT_0000.pointdir" / "input.gjf").is_file()
    assert [record["completed"] for record in progress] == [0, 1]
    assert {record["stage"] for record in progress} == {
        "gaussian_input_staging"
    }

    def reporting_failure(**_payload):
        raise OSError("reporting unavailable")

    _staging, repeated_count = stage_gaussian_inputs(
        campaign,
        CampaignConfig(),
        "INITIAL_GAUSSIAN",
        0,
        sample,
        progress_callback=reporting_failure,
    )
    assert repeated_count == 1


def test_extra_pointdirs_force_exact_staging_rebuild(tmp_path):
    campaign, sample = _phase_a_campaign(tmp_path)
    config = CampaignConfig()
    staging, _ = stage_gaussian_inputs(
        campaign, config, "INITIAL_GAUSSIAN", 0, sample
    )
    pointdir = staging / "POINT_0000.pointdir"
    (pointdir / "input.gau").write_text("stale\n", encoding="utf-8")
    extra = staging / "POINT_9999.pointdir"
    extra.mkdir()
    (extra / "input.gjf").write_text("extra\n", encoding="utf-8")

    stage_gaussian_inputs(campaign, config, "INITIAL_GAUSSIAN", 0, sample)

    assert not extra.exists()
    assert not (pointdir / "input.gau").exists()


def test_symlinked_staging_pointdir_is_rejected_before_cleanup(tmp_path):
    campaign, sample = _phase_a_campaign(tmp_path)
    staging = campaign / ".DATA" / "STAGING" / "initial"
    staging.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = staging / "POINT_0000.pointdir"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="symlink"):
        stage_gaussian_inputs(
            campaign,
            CampaignConfig(),
            "INITIAL_GAUSSIAN",
            0,
            sample,
        )
    assert outside.is_dir()
