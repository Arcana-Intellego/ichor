import hashlib
import json

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.geometry_protocol import PHASE_B_MIN_SEPARATION_SCALE
from ichor.hpc.active_learning.handoff_manifests import (
    ARIADNE_RESULTS_SCHEMA_VERSION,
    SEED_SELECTION_SCHEMA_VERSION,
    ariadne_landing_audit_path,
    ariadne_results_path,
    build_seed_selection_manifest,
    seeds_picked_path,
    write_ariadne_landing_audit,
    write_ariadne_results_manifest,
)
from ichor.hpc.active_learning.layout import (
    active_ariadne_dir,
    active_iteration_dir,
    ariadne_seed_dir,
)
from ichor.hpc.active_learning.ariadne_outputs import (
    SEED_OUTPUT_MANIFEST_FILENAME,
    write_optimisation_trajectory,
    write_seed_output_manifest,
)
from ichor.hpc.active_learning.daemon.state import atomic_write_json
from ichor.hpc.active_learning.daemon.error_calibration import (
    build_calibration_model,
    model_path as error_calibration_model_path,
    synthetic_dry_records,
    write_calibration_model,
)
from ichor.hpc.active_learning.seed_identity import write_ariadne_task_map
from ichor.hpc.active_learning.versioning.manifest import sha256_file
from ichor.hpc.active_learning.versioning.provenance import (
    PROVENANCE_FILENAME,
    write_seed_provenance,
)
from ichor.hpc.active_learning.sampling_protocol import (
    LEGACY_SAMPLING_AGGRESSIVENESS_POLICY_VERSION,
    SAMPLING_AGGRESSIVENESS_POLICY_VERSION,
    hidden_sampling_overrides,
    phase_b_min_separation_from_resolved,
    preview_sampling_protocol,
    read_sampling_protocol_audit,
    read_sampling_protocol_resolved,
    resolve_or_load_sampling_protocol,
    resolve_sampling_protocol,
    sampling_protocol_audit_path,
    sampling_protocol_resolved_path,
    sampling_policy_table_payload,
    sampling_policy_table_sha256,
)
from ichor.hpc.active_learning.sampling_scale_model import (
    LEGACY_SAMPLING_SCALE_MODEL_MODEL_VERSION,
    SAMPLING_SCALE_MODEL_MODEL_VERSION,
    read_sampling_scale_model,
    sampling_scale_model_path,
)


def _write_strict_history(
    campaign,
    *,
    iteration=1,
    movement=0.2,
    residual=0.3,
    min_pair=0.9,
    policy_version=1,
    sampling_aggressiveness=5,
    aligned_mass_movement=None,
):
    iter_dir = active_iteration_dir(campaign, iteration)
    ariadne_root = active_ariadne_dir(iter_dir)
    selection = build_seed_selection_manifest(
        campaign_uid="sampling-protocol-test",
        campaign_random_seed=0,
        iteration=int(iteration),
        models_version=0,
        model_manifest_sha256="a" * 64,
        model_set_sha256="c" * 64,
        trajectory_sha256="b" * 64,
        selection_strategy="hybrid_variance",
        seed_records=[{
            "seed_id": 1,
            "frame_id": 10,
            "pool_row_index_zero_based": 10,
            "selection_origin": "bulk",
            "variance_at_selection": 0.1,
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
    movement_for_geometry = (
        float(movement)
        if movement is not None
        else float(aligned_mass_movement or 0.1)
    )
    landing_metrics = {
        "aligned_rmsd_ang": movement_for_geometry,
        "fullspace_residual_distance": float(residual),
        "min_pair_distance_ang": float(min_pair),
    }
    if movement is not None:
        landing_metrics["movement_rmsd_ang"] = float(movement)
    if aligned_mass_movement is not None:
        landing_metrics["aligned_mass_weighted_rmsd_ang"] = float(
            aligned_mass_movement
        )
    landing_safety = {
        "accepted": True,
        "policy": "raw_final",
        "selected_origin": "raw_final",
        "selected_candidate_index": 0,
        "reasons": [],
        "record_only_reasons": [],
        "metrics": landing_metrics,
        "raw_final": {},
        "n_candidates_evaluated": 1,
        "n_safe_candidates": 1,
    }
    result = {
        "iteration": int(iteration),
        "seed_id": 1,
        "seed_uid": seed_uid,
        "array_task_id": 0,
        "seed_frame_id": 10,
        "trajectory_sha256": "b" * 64,
        "atom_types": ["H", "H"],
        "seed_coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "final_coordinates": [
            [movement_for_geometry / 2.0, 0.0, 0.0],
            [1.0 + movement_for_geometry, 0.0, 0.0],
        ],
        "alpha_trajectory": [0.0, 1.0],
        "alpha_initial": 0.0,
        "alpha_final": 1.0,
        "n_evaluations": 2,
        "return_code": 0,
        "wall_seconds": 1.0,
        "fell_back_to_ds": False,
        "whitened_distance_final": 0.5,
        "landing_safety": landing_safety,
    }
    atomic_write_json(result_path, result)
    write_optimisation_trajectory(
        seed_dir,
        atom_types=result["atom_types"],
        coordinate_frames=[result["final_coordinates"]],
        alpha_values=[1.0],
        gradient_norms=[0.0],
        origins=["raw_final"],
    )
    output_manifest = write_seed_output_manifest(
        seed_dir,
        campaign_uid=selection["campaign_uid"],
        iteration=int(iteration),
        seed_id=1,
        seed_uid=seed_uid,
        array_task_id=0,
        task_success=True,
        task_exit_code=0,
    )
    provenance_path = write_seed_provenance(
        seed_dir,
        campaign_uid=selection["campaign_uid"],
        iteration=int(iteration),
        trajectory_sha256="b" * 64,
        seed_frame_id=10,
        seed_id=1,
        seed_uid=seed_uid,
        array_task_id_zero_based=0,
        seed_selection_origin="bulk",
        seed_variance_at_selection=0.1,
        subspace_neighbour_frame_ids=[],
        subspace_dimension=0,
        subspace_eigenvalues=[],
        mode_weighting_policy="variance",
    )
    accepted_record = {
        "seed_id": 1,
        "seed_uid": seed_uid,
        "array_task_id": 0,
        "seed_frame_id": 10,
        "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
        "result_json": result_path.relative_to(ariadne_root).as_posix(),
        "provenance_json": provenance_path.relative_to(ariadne_root).as_posix(),
        "output_manifest": output_manifest.relative_to(ariadne_root).as_posix(),
        "return_code": 0,
        "landing_safety": landing_safety,
        "result_sha256": sha256_file(result_path),
        "provenance_sha256": sha256_file(provenance_path),
        "output_manifest_sha256": sha256_file(output_manifest),
    }
    write_ariadne_results_manifest(iter_dir, {
        "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
        "campaign_uid": selection["campaign_uid"],
        "iteration": int(iteration),
        "trajectory_sha256": "b" * 64,
        "task_map": {
            "path": task_map_path.relative_to(ariadne_root).as_posix(),
            "sha256": sha256_file(task_map_path),
        },
        "expected_n": 1,
        "n_accepted": 1,
        "n_rejected": 0,
        "accepted": [accepted_record],
        "rejected": [],
    })
    write_ariadne_landing_audit(iter_dir, {
        "iteration": int(iteration),
        "summary": {"accepted": 1, "rejected": 0},
        "seeds": [{
            "seed_id": 1,
            "seed_uid": seed_uid,
            "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
            "result_json": result_path.relative_to(ariadne_root).as_posix(),
            "landing_safety": landing_safety,
            "handoff_accepted": True,
        }],
    })
    policy_table = sampling_policy_table_payload(int(policy_version))
    protocol_path = sampling_protocol_resolved_path(iter_dir)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        protocol_path,
        {
            "schema_version": 2,
            "iteration": int(iteration),
            "sampling_aggressiveness": int(sampling_aggressiveness),
            "sampling_policy_version": int(policy_version),
            "sampling_policy_table_sha256": sampling_policy_table_sha256(
                int(policy_version)
            ),
            "sampling_policy": policy_table["levels"][
                str(int(sampling_aggressiveness))
            ],
        },
    )
    return iter_dir


def test_level_five_preview_matches_current_balanced_defaults():
    cfg = CampaignConfig()

    resolved = preview_sampling_protocol(cfg)

    assert resolved.sampling_aggressiveness == 5
    assert resolved.sampling_policy_version == 2
    assert resolved.resolved_geometry_scale_angstrom == pytest.approx(0.05)
    assert resolved.phase_b["min_separation_scaled"] == pytest.approx(
        PHASE_B_MIN_SEPARATION_SCALE
    )
    assert resolved.phase_b["effective_min_separation_angstrom"] == pytest.approx(
        PHASE_B_MIN_SEPARATION_SCALE * 0.05
    )
    assert resolved.scale_model_payload["geometry_motion_scale"]["value_angstrom"] == pytest.approx(
        0.05
    )
    assert resolved.scale_model_payload["aligned_rmsd_scale"]["value_angstrom"] == pytest.approx(
        0.05
    )
    assert resolved.scale_model_payload["residual_fullspace_scale"]["value_angstrom"] == pytest.approx(
        0.5
    )
    assert resolved.scale_model_payload["per_atom_mobility_scales"]["mode"] == "uniform"
    trust_policy = resolved.scale_model_payload["trust_radius_policy"]
    assert trust_policy["enabled"] is True
    assert trust_policy["normalisation"] == (
        "weighted_mobility_sqrt_effective_atoms"
    )
    assert "sqrt(n_effective_movement_atoms)" in trust_policy["formula"]
    assert trust_policy["target_motion_ratio"] == pytest.approx(1.0)
    assert resolved.adversarial_safety.max_whitened_distance == pytest.approx(10.0)
    assert resolved.adversarial_safety.backtrack_points == 16
    assert resolved.quality_gates.ariadne_max_displacement_angstrom == pytest.approx(1.25)
    assert resolved.quality_gates.ariadne_min_pair_distance_angstrom == pytest.approx(0.60)
    assert resolved.acquisition_config.weights.lambda_distance == pytest.approx(1.0)
    assert resolved.acquisition_config.fullspace_confinement.lambda_residual == pytest.approx(
        0.5
    )
    assert resolved.acquisition_config.movement_band.geometry_novelty_scale_angstrom == pytest.approx(0.05)
    assert resolved.acquisition_config.movement_band.target_peak_fraction == pytest.approx(
        1.0
    )
    assert resolved.acquisition_config.fullspace_confinement.rmsd_scale_ang == pytest.approx(0.05)
    assert resolved.acquisition_config.fullspace_confinement.fixed_residual_scale_ang == pytest.approx(0.5)
    assert resolved.ariadne_run_config.delta0 == pytest.approx(0.10)
    assert resolved.ariadne_run_config.delta_max == pytest.approx(0.40)
    dimensionless = resolved.scale_model_payload["dimensionless_policy"]
    assert dimensionless["max_scaled_whitened_distance"] == pytest.approx(10.0)
    assert dimensionless["max_scaled_atom_move"] == pytest.approx(36.0)
    assert dimensionless["max_scaled_rmsd"] == pytest.approx(4.2)
    assert dimensionless["normalised_chemistry_penalty_cap"] == pytest.approx(20.0)


def test_aggressiveness_policies_move_from_conservative_to_exploratory():
    conservative = CampaignConfig()
    conservative.campaign.sampling_aggressiveness = 1
    exploratory = CampaignConfig()
    exploratory.campaign.sampling_aggressiveness = 10

    low = preview_sampling_protocol(conservative)
    high = preview_sampling_protocol(exploratory)

    assert high.policy.fallback_scale_angstrom == low.policy.fallback_scale_angstrom
    assert high.policy.phase_b_min_separation_scale == low.policy.phase_b_min_separation_scale
    assert high.policy.max_whitened_distance == low.policy.max_whitened_distance
    assert high.policy.lambda_distance < low.policy.lambda_distance
    assert high.policy.max_atom_displacement_ang == low.policy.max_atom_displacement_ang
    assert high.policy.trust_max_to_initial_ratio == low.policy.trust_max_to_initial_ratio
    assert high.policy.target_motion_ratio > low.policy.target_motion_ratio
    assert high.policy.initial_trust_multiplier > low.policy.initial_trust_multiplier
    assert (
        high.scale_model_payload["trust_radius_policy"][
            "aggressiveness_multiplier"
        ]
        > low.scale_model_payload["trust_radius_policy"][
            "aggressiveness_multiplier"
        ]
    )
    assert high.policy.max_scaled_atom_move == low.policy.max_scaled_atom_move
    assert high.policy.max_scaled_rmsd == low.policy.max_scaled_rmsd
    assert (
        high.policy.normalised_chemistry_penalty_cap
        == low.policy.normalised_chemistry_penalty_cap
    )


def test_sampling_policy_table_is_versioned_complete_and_hashed():
    payload = sampling_policy_table_payload()

    assert payload["policy_version"] == SAMPLING_AGGRESSIVENESS_POLICY_VERSION
    assert set(payload["levels"]) == {str(level) for level in range(1, 11)}
    assert len(sampling_policy_table_sha256()) == 64
    assert payload["levels"]["5"]["target_motion_ratio"] == pytest.approx(1.0)
    assert payload["levels"]["5"]["initial_trust_multiplier"] == pytest.approx(1.0)
    assert payload["levels"]["5"]["trust_max_to_initial_ratio"] == pytest.approx(4.0)


def test_legacy_sampling_policy_table_identity_is_frozen():
    payload = sampling_policy_table_payload(
        LEGACY_SAMPLING_AGGRESSIVENESS_POLICY_VERSION
    )

    assert payload["policy_version"] == 1
    assert payload["levels"]["5"]["movement_trust_multiplier"] == pytest.approx(1.0)
    assert (
        sampling_policy_table_sha256(
            LEGACY_SAMPLING_AGGRESSIVENESS_POLICY_VERSION
        )
        == "ca7b11a3f041fddabb87f8b1b2e704e290f46c3817667b1f066b58423dde14e0"
    )


def test_v2_policy_table_has_exact_monotonic_exploration_controls():
    levels = sampling_policy_table_payload()["levels"]

    assert [levels[str(level)]["target_motion_ratio"] for level in range(1, 11)] == [
        0.35,
        0.45,
        0.58,
        0.75,
        1.0,
        1.25,
        1.55,
        1.9,
        2.3,
        2.75,
    ]
    assert [
        levels[str(level)]["initial_trust_multiplier"]
        for level in range(1, 11)
    ] == [0.6, 0.7, 0.82, 0.91, 1.0, 1.15, 1.32, 1.52, 1.75, 2.0]
    assert [levels[str(level)]["lambda_distance"] for level in range(1, 11)] == [
        2.0,
        1.68,
        1.41,
        1.19,
        1.0,
        0.84,
        0.71,
        0.6,
        0.5,
        0.42,
    ]
    safety_fields = (
        "fallback_scale_angstrom",
        "phase_b_min_separation_scale",
        "trust_max_to_initial_ratio",
        "max_whitened_distance",
        "max_atom_displacement_ang",
        "min_pair_distance_ang",
        "max_scaled_atom_move",
        "max_scaled_rmsd",
        "max_scaled_fullspace_residual",
        "normalised_chemistry_penalty_cap",
    )
    for field in safety_fields:
        assert len({levels[str(level)][field] for level in range(1, 11)}) == 1


def test_v2_resolved_controls_follow_target_and_trust_tables():
    target_ratios = []
    initial_trust = []
    maximum_trust = []
    safety_signatures = []
    for level in range(1, 11):
        cfg = CampaignConfig()
        cfg.campaign.sampling_aggressiveness = level
        resolved = preview_sampling_protocol(cfg)
        movement = resolved.acquisition_config.movement_band
        trust = resolved.scale_model_payload["trust_radius_policy"]
        target_ratios.append(float(movement.target_peak_fraction))
        initial_trust.append(float(trust["aggressiveness_multiplier"]))
        maximum_trust.append(
            float(trust["aggressiveness_multiplier"])
            * float(trust["max_to_initial_ratio"])
        )
        assert movement.hard_min_fraction == pytest.approx(
            0.25 * movement.target_peak_fraction
        )
        assert movement.target_low_fraction == pytest.approx(
            0.625 * movement.target_peak_fraction
        )
        assert movement.target_high_fraction == pytest.approx(
            1.875 * movement.target_peak_fraction
        )
        assert movement.hard_max_fraction == pytest.approx(
            3.125 * movement.target_peak_fraction
        )
        safety_signatures.append(
            (
                resolved.phase_b["min_separation_scaled"],
                resolved.adversarial_safety.max_whitened_distance,
                resolved.quality_gates.ariadne_max_displacement_angstrom,
                resolved.quality_gates.ariadne_min_pair_distance_angstrom,
            )
        )

    assert target_ratios == sorted(target_ratios)
    assert initial_trust == sorted(initial_trust)
    assert maximum_trust == sorted(maximum_trust)
    assert len(set(safety_signatures)) == 1


def test_v1_resolved_controls_retain_legacy_band_compilation(tmp_path):
    cfg = CampaignConfig()
    cfg.campaign.sampling_aggressiveness = 7

    resolved = resolve_sampling_protocol(
        tmp_path,
        cfg,
        iteration=1,
        write_manifest=False,
        _policy_version=1,
    )

    movement = resolved.acquisition_config.movement_band
    trust = resolved.scale_model_payload["trust_radius_policy"]
    assert movement.hard_min_fraction == pytest.approx(0.115)
    assert movement.target_low_fraction == pytest.approx(0.2875)
    assert movement.target_peak_fraction == pytest.approx(0.46)
    assert movement.target_high_fraction == pytest.approx(0.8625)
    assert movement.hard_max_fraction == pytest.approx(1.4375)
    assert trust["aggressiveness_multiplier"] == pytest.approx(1.15)
    assert trust["max_to_initial_ratio"] == pytest.approx(3.571)


def test_hidden_low_level_overrides_are_reported_not_applied():
    cfg = CampaignConfig()
    cfg.geometry_novelty.fallback_scale_angstrom = 9.0
    cfg.phase_b.beta = 0.9
    cfg.quality_gates.ariadne_min_pair_distance_angstrom = 0.2

    overrides = hidden_sampling_overrides(cfg)
    paths = {entry["path"] for entry in overrides}

    assert "geometry_novelty.fallback_scale_angstrom" in paths
    assert "phase_b.beta" not in paths
    assert "quality_gates.ariadne_min_pair_distance_angstrom" in paths

    resolved = preview_sampling_protocol(cfg)

    assert resolved.acquisition_config.weights.lambda_distance == pytest.approx(1.0)
    assert resolved.geometry_scale_payload["scale_angstrom"] == pytest.approx(0.05)
    assert resolved.phase_b["beta"] == pytest.approx(0.9)
    assert resolved.quality_gates.ariadne_min_pair_distance_angstrom == pytest.approx(0.60)


def test_resolver_writes_round_trippable_manifest(tmp_path):
    cfg = CampaignConfig()

    resolved = resolve_sampling_protocol(tmp_path, cfg, iteration=2)

    iter_dir = active_iteration_dir(tmp_path, 2)
    expected_path = sampling_protocol_resolved_path(iter_dir)
    audit_path = sampling_protocol_audit_path(iter_dir)
    scale_path = sampling_scale_model_path(iter_dir)
    assert resolved.manifest_path == expected_path
    assert resolved.audit_manifest_path == audit_path
    assert resolved.scale_model_path == scale_path
    assert expected_path.exists()
    assert audit_path.exists()
    assert scale_path.exists()

    payload = read_sampling_protocol_resolved(iter_dir, expected_iteration=2)
    audit_payload = read_sampling_protocol_audit(iter_dir, expected_iteration=2)
    scale_payload = read_sampling_scale_model(iter_dir, expected_iteration=2)
    assert payload["schema_version"] == 2
    assert payload["sampling_aggressiveness"] == 5
    assert payload["sampling_scale_model_manifest"] == str(scale_path)
    assert payload["sampling_protocol_audit_manifest"] == str(audit_path)
    assert payload["sampling_policy_version"] == 2
    assert payload["sampling_policy_table_sha256"] == sampling_policy_table_sha256()
    assert payload["dimensionless_policy"]["max_scaled_atom_move"] == pytest.approx(36.0)
    assert audit_payload["schema_version"] == 2
    assert audit_payload["dimensionless_policy"]["max_scaled_atom_move"] == pytest.approx(36.0)
    assert audit_payload["scheduler_impact"]["new_scheduler_jobs"] == 0
    assert scale_payload["schema_version"] == 2
    assert scale_payload["model_version"] == SAMPLING_SCALE_MODEL_MODEL_VERSION
    assert scale_payload["dimensionless_policy"]["max_scaled_rmsd"] == pytest.approx(4.2)
    assert scale_payload["geometry_motion_scale"]["value_angstrom"] == pytest.approx(0.05)
    assert payload["resolved_phase_b"]["effective_min_separation_angstrom"] == pytest.approx(
        PHASE_B_MIN_SEPARATION_SCALE * 0.05
    )
    assert payload["resolved_adversarial_safety"]["max_whitened_distance"] == pytest.approx(
        10.0
    )

    threshold, mode = phase_b_min_separation_from_resolved(resolved)
    assert threshold == pytest.approx(PHASE_B_MIN_SEPARATION_SCALE * 0.05)
    assert mode in {"absolute", "scaled", "scaled_geometry_novelty"}


def test_resolve_or_load_preserves_immutable_protocol_bytes(tmp_path):
    cfg = CampaignConfig()
    resolve_sampling_protocol(tmp_path, cfg, iteration=1)
    iter_dir = active_iteration_dir(tmp_path, 1)
    paths = (
        sampling_protocol_resolved_path(iter_dir),
        sampling_protocol_audit_path(iter_dir),
        sampling_scale_model_path(iter_dir),
    )
    before = {path: path.read_bytes() for path in paths}

    loaded = resolve_or_load_sampling_protocol(tmp_path, cfg, iteration=1)

    assert loaded.iteration == 1
    assert {path: path.read_bytes() for path in paths} == before


def test_resolve_or_load_preserves_legacy_v1_protocol_bytes(tmp_path):
    cfg = CampaignConfig()
    resolve_sampling_protocol(
        tmp_path,
        cfg,
        iteration=1,
        _policy_version=LEGACY_SAMPLING_AGGRESSIVENESS_POLICY_VERSION,
    )
    iter_dir = active_iteration_dir(tmp_path, 1)
    paths = (
        sampling_protocol_resolved_path(iter_dir),
        sampling_protocol_audit_path(iter_dir),
        sampling_scale_model_path(iter_dir),
    )
    before = {path: path.read_bytes() for path in paths}

    loaded = resolve_or_load_sampling_protocol(tmp_path, cfg, iteration=1)

    assert loaded.sampling_policy_version == 1
    assert (
        loaded.sampling_policy_table_sha256
        == sampling_policy_table_sha256(1)
    )
    assert (
        loaded.scale_model_payload["model_version"]
        == LEGACY_SAMPLING_SCALE_MODEL_MODEL_VERSION
    )
    assert {path: path.read_bytes() for path in paths} == before


def test_load_accepts_pre_v2_scale_payload_shape(tmp_path):
    cfg = CampaignConfig()
    resolve_sampling_protocol(
        tmp_path,
        cfg,
        iteration=1,
        _policy_version=LEGACY_SAMPLING_AGGRESSIVENESS_POLICY_VERSION,
    )
    iter_dir = active_iteration_dir(tmp_path, 1)
    protocol_path = sampling_protocol_resolved_path(iter_dir)
    scale_path = sampling_scale_model_path(iter_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    scale["trust_radius_policy"].pop("target_motion_ratio", None)
    scale["history"].pop("movement_metric", None)
    scale["history"].pop("policy_normalisation", None)
    scale["diagnostics"].pop("history_policy_normalisation", None)
    atomic_write_json(scale_path, scale)
    protocol["sampling_scale_model"] = scale
    canonical = json.dumps(scale, sort_keys=True, separators=(",", ":"))
    protocol["source_scale_hashes"]["sampling_scale_model_sha256"] = (
        hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    atomic_write_json(protocol_path, protocol)
    before = {
        protocol_path: protocol_path.read_bytes(),
        scale_path: scale_path.read_bytes(),
    }

    loaded = resolve_or_load_sampling_protocol(tmp_path, cfg, iteration=1)

    assert loaded.sampling_policy_version == 1
    assert "target_motion_ratio" not in loaded.scale_model_payload[
        "trust_radius_policy"
    ]
    assert {
        protocol_path: protocol_path.read_bytes(),
        scale_path: scale_path.read_bytes(),
    } == before


def test_resolve_or_load_uses_frozen_calibration_after_global_changes(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelVersioning,
    )

    cfg = CampaignConfig()
    cfg.error_calibration.mode = "apply_to_acquisition"
    cfg.error_calibration.apply_strength = 1.0
    cfg.error_calibration.min_records_to_apply = 1
    cfg.error_calibration.min_model_versions_to_apply = 1
    cfg.error_calibration.n_bins = 2
    cfg.error_calibration.min_bin_records = 1

    environment = {
        "generation": 2,
        "generation_digest_sha256": "e" * 64,
        "bound": True,
    }
    records = synthetic_dry_records(
        iteration=1,
        models_version=0,
        n_points=4,
        config=cfg,
        environment_binding=environment,
    )
    model = build_calibration_model(
        records,
        cfg,
        iteration=1,
        current_model_version=0,
        environment_binding=environment,
    )
    assert model["usable_for_acquisition"] is True
    write_calibration_model(tmp_path, model)
    versions = TrainedModelVersioning(tmp_path / "TRAINED_MODELS")
    versions.iteration_path(0).mkdir(parents=True)
    versions.update_current(0)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.error_calibration.active_environment_binding",
        lambda _campaign: dict(environment),
    )

    resolved = resolve_sampling_protocol(tmp_path, cfg, iteration=2)
    snapshot = resolved.error_calibration_snapshot
    frozen_model_sha = snapshot["model_sha256"]
    frozen_source_sha = snapshot["source_records_sha256"]
    assert snapshot["model_reason"] == "loaded"
    assert snapshot["model"] is not None

    error_calibration_model_path(tmp_path).unlink()
    cfg.error_calibration.max_model_age_iterations = 0

    loaded = resolve_or_load_sampling_protocol(tmp_path, cfg, iteration=2)

    assert loaded.error_calibration_snapshot["model_sha256"] == frozen_model_sha
    assert (
        loaded.error_calibration_snapshot["source_records_sha256"]
        == frozen_source_sha
    )
    assert loaded.error_calibration_snapshot["model"] == snapshot["model"]
    assert (
        loaded.effective_config.error_calibration.max_model_age_iterations
        == snapshot["settings"]["max_model_age_iterations"]
    )


def test_resolve_or_load_rejects_partial_protocol_snapshot(tmp_path):
    cfg = CampaignConfig()
    iter_dir = active_iteration_dir(tmp_path, 1)
    resolved_path = sampling_protocol_resolved_path(iter_dir)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text("{}\n", encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="snapshot is incomplete"):
        resolve_or_load_sampling_protocol(tmp_path, cfg, iteration=1)


def test_scale_model_uses_previous_result_json_motion_history(tmp_path):
    _write_strict_history(tmp_path, movement=0.2, residual=0.3, min_pair=0.9)

    cfg = CampaignConfig()
    resolved = resolve_sampling_protocol(tmp_path, cfg, iteration=2)

    scale = resolved.scale_model_payload
    assert scale["geometry_motion_scale"]["source"] == (
        "normalised_ariadne_landing_history"
    )
    assert scale["geometry_motion_scale"]["value_angstrom"] == pytest.approx(0.2)
    assert scale["aligned_rmsd_scale"]["value_angstrom"] == pytest.approx(0.2)
    assert scale["residual_fullspace_scale"]["value_angstrom"] == pytest.approx(0.3)
    assert resolved.acquisition_config.movement_band.geometry_novelty_scale_angstrom == pytest.approx(0.2)
    assert resolved.acquisition_config.fullspace_confinement.rmsd_scale_ang == pytest.approx(0.2)
    assert resolved.acquisition_config.fullspace_confinement.fixed_residual_scale_ang == pytest.approx(0.3)
    assert scale["per_atom_mobility_scales"]["mode"] == "per_atom_index"
    assert scale["per_atom_mobility_scales"]["values_angstrom"] == pytest.approx(
        [0.05, 0.05]
    )
    assert scale["model_version"] == SAMPLING_SCALE_MODEL_MODEL_VERSION
    assert scale["history"]["movement_metric"] == "movement_rmsd_ang"
    assert (
        scale["history"]["policy_normalisation"]["n_primary_movement_values"]
        == 1
    )


def test_sampling_history_cache_hit_does_not_reopen_result_payloads(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning import sampling_history

    _write_strict_history(tmp_path, movement=0.2)
    expected = sampling_history.load_or_build_sampling_history(
        tmp_path,
        iteration=1,
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("warm history cache reopened scientific payloads")

    monkeypatch.setattr(
        sampling_history,
        "read_historical_ariadne_records",
        _unexpected,
    )
    monkeypatch.setattr(sampling_history, "_load_result", _unexpected)

    observed = sampling_history.load_or_build_sampling_history(
        tmp_path,
        iteration=1,
    )
    assert observed == expected


def test_corrupt_sampling_history_cache_is_ignored_and_rebuilt(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning import sampling_history

    _write_strict_history(tmp_path, movement=0.2)
    sampling_history.load_or_build_sampling_history(tmp_path, iteration=1)
    records_path = next(
        (
            tmp_path
            / ".DATA"
            / "CACHE"
            / "SAMPLING_HISTORY"
            / "iteration-000001"
        ).glob("*/records.json")
    )
    records_path.write_text("{}\n", encoding="utf-8")
    original = sampling_history.read_historical_ariadne_records
    calls = {"count": 0}

    def _counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        sampling_history,
        "read_historical_ariadne_records",
        _counted,
    )
    rebuilt = sampling_history.load_or_build_sampling_history(
        tmp_path,
        iteration=1,
    )

    assert calls["count"] == 1
    assert rebuilt["records"][0]["metrics"]["movement_rmsd_ang"] == pytest.approx(
        0.2
    )


def test_sampling_history_cache_path_failure_uses_strict_history(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning import sampling_history

    _write_strict_history(tmp_path, movement=0.2)
    monkeypatch.setattr(
        sampling_history,
        "_cache_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("injected cache-path failure")
        ),
    )

    observed = sampling_history.load_or_build_sampling_history(
        tmp_path,
        iteration=1,
    )

    assert observed["records"][0]["metrics"]["movement_rmsd_ang"] == pytest.approx(
        0.2
    )


def test_scale_model_uses_one_primary_movement_value_per_result(tmp_path):
    _write_strict_history(
        tmp_path,
        movement=0.2,
        aligned_mass_movement=0.8,
    )

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=2)
    history = resolved.scale_model_payload["history"]

    assert history["movement_summary"]["n"] == 1
    assert history["movement_summary"]["median"] == pytest.approx(0.2)
    assert history["policy_normalisation"]["n_primary_movement_values"] == 1
    assert history["policy_normalisation"]["n_fallback_movement_values"] == 0


def test_scale_model_uses_global_fallback_only_without_primary_history(tmp_path):
    _write_strict_history(
        tmp_path,
        movement=None,
        aligned_mass_movement=0.4,
    )

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=2)
    history = resolved.scale_model_payload["history"]

    assert history["movement_metric"] == (
        "aligned_mass_weighted_rmsd_ang_fallback"
    )
    assert history["movement_summary"]["n"] == 1
    assert history["movement_summary"]["median"] == pytest.approx(0.4)
    assert history["policy_normalisation"]["n_primary_movement_values"] == 0
    assert history["policy_normalisation"]["n_fallback_movement_values"] == 1
    assert history["policy_normalisation"]["n_fallback_movement_values_used"] == 1


def test_scale_model_does_not_mix_fallback_with_available_primary_history(tmp_path):
    _write_strict_history(
        tmp_path,
        iteration=1,
        movement=None,
        aligned_mass_movement=0.8,
    )
    _write_strict_history(
        tmp_path,
        iteration=2,
        movement=0.2,
        aligned_mass_movement=0.9,
    )

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=3)
    history = resolved.scale_model_payload["history"]

    assert history["movement_metric"] == "movement_rmsd_ang"
    assert history["movement_summary"]["n"] == 1
    assert history["movement_summary"]["median"] == pytest.approx(0.2)
    assert history["policy_normalisation"]["n_fallback_movement_values"] == 1
    assert history["policy_normalisation"]["n_fallback_movement_values_used"] == 0


def test_scale_model_normalises_history_by_producer_policy(tmp_path):
    _write_strict_history(
        tmp_path,
        movement=0.31,
        policy_version=2,
        sampling_aggressiveness=7,
    )

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=2)
    scale = resolved.scale_model_payload

    assert scale["geometry_motion_scale"]["value_angstrom"] == pytest.approx(0.2)
    normalisation = scale["history"]["policy_normalisation"]
    assert normalisation["n_iterations_with_resolved_policy"] == 1
    assert normalisation["iterations"][0]["policy_version"] == 2
    assert normalisation["iterations"][0]["sampling_aggressiveness"] == 7
    assert normalisation["iterations"][0]["normalisation_factor"] == pytest.approx(
        1.55
    )


def test_scale_model_normalises_mixed_v1_and_v2_history(tmp_path):
    _write_strict_history(
        tmp_path,
        iteration=1,
        movement=0.23,
        policy_version=1,
        sampling_aggressiveness=7,
    )
    _write_strict_history(
        tmp_path,
        iteration=2,
        movement=0.31,
        policy_version=2,
        sampling_aggressiveness=7,
    )

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=3)
    scale = resolved.scale_model_payload

    assert scale["geometry_motion_scale"]["value_angstrom"] == pytest.approx(0.2)
    normalisation = scale["history"]["policy_normalisation"]
    assert [
        row["normalisation_factor"] for row in normalisation["iterations"]
    ] == pytest.approx([1.15, 1.55])
    assert [
        row["policy_version"] for row in normalisation["iterations"]
    ] == [1, 2]


def test_scale_model_keeps_pre_protocol_history_with_neutral_factor(tmp_path):
    iter_dir = _write_strict_history(tmp_path, movement=0.2)
    sampling_protocol_resolved_path(iter_dir).unlink()

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=2)
    normalisation = resolved.scale_model_payload["history"][
        "policy_normalisation"
    ]

    assert resolved.scale_model_payload["geometry_motion_scale"][
        "value_angstrom"
    ] == pytest.approx(0.2)
    assert normalisation["n_iterations_with_missing_policy"] == 1
    assert normalisation["iterations"][0]["normalisation_factor"] == pytest.approx(
        1.0
    )


def test_scale_model_skips_history_with_invalid_policy_evidence(tmp_path):
    iter_dir = _write_strict_history(tmp_path, movement=0.2)
    protocol_path = sampling_protocol_resolved_path(iter_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["sampling_policy_table_sha256"] = "0" * 64
    atomic_write_json(protocol_path, protocol)

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=2)
    normalisation = resolved.scale_model_payload["history"][
        "policy_normalisation"
    ]

    assert resolved.scale_model_payload["geometry_motion_scale"][
        "value_angstrom"
    ] == pytest.approx(0.05)
    assert normalisation["n_iterations_with_invalid_policy"] == 1
    assert normalisation["n_records_skipped_invalid_policy"] == 1


def test_scale_model_uses_only_strictly_accepted_history(tmp_path):
    _write_strict_history(tmp_path, movement=0.2, residual=0.3, min_pair=0.9)

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=2)
    scale = resolved.scale_model_payload

    assert scale["geometry_motion_scale"]["value_angstrom"] == pytest.approx(0.2)
    assert scale["residual_fullspace_scale"]["value_angstrom"] == pytest.approx(0.3)
    history_filter = scale["history"]["filter"]
    assert history_filter["n_seen"] == 1
    assert history_filter["n_used"] == 1
    assert history_filter["n_skipped_handoff_rejected"] == 0


def test_scale_model_rejects_results_only_legacy_history(tmp_path):
    iter1 = active_iteration_dir(tmp_path, 1)
    iter1.mkdir(parents=True)
    ariadne_landing_audit_path(iter1).parent.mkdir(parents=True, exist_ok=True)
    ariadne_landing_audit_path(iter1).write_text(
        json.dumps({"schema_version": 2, "iteration": 1, "seeds": []}),
        encoding="utf-8",
    )
    ariadne_results_path(iter1).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 1,
                "accepted": [
                    {
                        "seed_id": 1,
                        "landing_safety": {
                            "accepted": True,
                            "metrics": {
                                "movement_rmsd_ang": 0.25,
                                "aligned_rmsd_ang": 0.25,
                                "fullspace_residual_distance": 0.35,
                                "min_pair_distance_ang": 1.0,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=2)
    scale = resolved.scale_model_payload

    assert scale["geometry_motion_scale"]["value_angstrom"] == pytest.approx(0.05)
    assert scale["residual_fullspace_scale"]["value_angstrom"] == pytest.approx(0.5)
    assert scale["history"]["filter"]["n_skipped_malformed_record"] == 1
    assert scale["history"]["filter"]["n_fallback_results_records_used"] == 0


def test_scale_model_reports_malformed_and_duplicate_legacy_history(tmp_path):
    iter1 = active_iteration_dir(tmp_path, 1)
    iter1.mkdir(parents=True)
    ariadne_landing_audit_path(iter1).parent.mkdir(parents=True, exist_ok=True)
    ariadne_landing_audit_path(iter1).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 1,
                "seeds": [
                    "not-a-record",
                    {
                        "seed_id": 1,
                        "handoff_accepted": True,
                        "landing_safety": {"accepted": False, "metrics": {}},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    usable = {
        "seed_id": 2,
        "result_json": "seeds/seed-000002/result.json",
        "landing_safety": {
            "accepted": True,
            "metrics": {
                "movement_rmsd_ang": 0.25,
                "aligned_rmsd_ang": 0.25,
                "fullspace_residual_distance": 0.35,
                "min_pair_distance_ang": 1.0,
            },
        },
    }
    ariadne_results_path(iter1).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 1,
                "accepted": [
                    "not-a-record",
                    usable,
                    dict(usable),
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=2)
    history_filter = resolved.scale_model_payload["history"]["filter"]

    assert history_filter["n_skipped_malformed_record"] == 1
    assert history_filter["n_skipped_landing_rejected"] == 0
    assert history_filter["n_deduplicated_fallback_records"] == 0
    assert history_filter["n_results_records_used"] == 0
    assert history_filter["n_fallback_results_records_used"] == 0


def test_scale_model_populates_per_seed_records_from_seed_records(tmp_path):
    iter1 = active_iteration_dir(tmp_path, 1)
    iter1.mkdir(parents=True)
    seeds_picked_path(iter1).parent.mkdir(parents=True, exist_ok=True)
    selection = build_seed_selection_manifest(
        campaign_uid="sampling-protocol-test",
        campaign_random_seed=0,
        iteration=1,
        models_version=0,
        model_manifest_sha256="a" * 64,
        model_set_sha256="c" * 64,
        trajectory_sha256="b" * 64,
        selection_strategy="d_optimal",
        seed_records=[
                    {
                        "seed_id": 1,
                        "frame_id": 10,
                        "pool_row_index_zero_based": 10,
                        "selection_origin": "bulk",
                        "variance_at_selection": None,
                    },
                    {
                        "seed_id": 2,
                        "frame_id": 20,
                        "pool_row_index_zero_based": 20,
                        "selection_origin": "d_optimal",
                        "variance_at_selection": 0.4,
                    },
                ],
    )
    atomic_write_json(seeds_picked_path(iter1), selection)

    resolved = resolve_sampling_protocol(tmp_path, CampaignConfig(), iteration=1)
    per_seed = resolved.scale_model_payload["per_seed_scale_model"]

    assert [record["seed_id"] for record in per_seed] == [1, 2]
    assert [record["frame_id"] for record in per_seed] == [10, 20]


def test_preview_uses_campaign_history_without_writing_manifests(tmp_path):
    _write_strict_history(tmp_path, movement=0.22, residual=0.44, min_pair=0.9)

    resolved = preview_sampling_protocol(
        CampaignConfig(),
        campaign_dir=tmp_path,
        iteration=2,
    )

    assert resolved.scale_model_payload["geometry_motion_scale"]["value_angstrom"] == pytest.approx(0.22)
    assert resolved.scale_model_payload["residual_fullspace_scale"]["value_angstrom"] == pytest.approx(0.44)
    iter2 = active_iteration_dir(tmp_path, 2)
    assert not sampling_scale_model_path(iter2).exists()
    assert not sampling_protocol_resolved_path(iter2).exists()
    assert not sampling_protocol_audit_path(iter2).exists()
    assert not (
        tmp_path / ".DATA" / "CACHE" / "SAMPLING_HISTORY"
    ).exists()
