"""M12 schema-v2 specific tests: translator round-trip, dead-letter cleanup
proof, generic parser path-qualified error messages, subspace-dim guard.
"""
import pytest

from ichor.hpc.active_learning.config import (
    AcquisitionConfigBlock,
    CampaignConfig,
    ConfigValidationError,
)
from ichor.hpc.active_learning.config_dataclass import (
    DataclassParseError,
    parse_dataclass_block,
)


# --- generic parser ---------------------------------------------------


def test_parser_rejects_unknown_acquisition_key_with_full_path():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["barrier"]["clash_lamba"] = 1.0  # typo
    with pytest.raises(ConfigValidationError) as exc:
        CampaignConfig.from_dict(payload)
    assert "acquisition.barrier" in str(exc.value)


def test_parser_rejects_unknown_subspace_key_with_full_path():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["neighbor_count"] = 100  # US spelling typo
    with pytest.raises(ConfigValidationError) as exc:
        CampaignConfig.from_dict(payload)
    assert "acquisition.subspace" in str(exc.value)
    assert "neighbor_count" in str(exc.value)


def test_parser_int_type_mismatch_path_qualified():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["neighbour_count"] = "fifty"
    with pytest.raises(ConfigValidationError) as exc:
        CampaignConfig.from_dict(payload)
    assert "neighbour_count" in str(exc.value)


def test_parser_bool_type_mismatch_path_qualified():
    payload = CampaignConfig().to_dict()
    payload["seed_selection"]["exclude_committed_seed_frames"] = "yes"
    with pytest.raises(ConfigValidationError) as exc:
        CampaignConfig.from_dict(payload)
    assert "exclude_committed_seed_frames" in str(exc.value)


def test_ferebus_properties_default_and_multipole_roundtrip():
    c = CampaignConfig()
    assert c.ferebus.properties == ["iqa"]
    payload = c.to_dict()
    payload["ferebus"]["properties"] = ["iqa", "q00", "q10", "q44s"]
    loaded = CampaignConfig.from_dict(payload)
    assert loaded.ferebus.properties == ["iqa", "q00", "q10", "q44s"]


@pytest.mark.parametrize("properties", [
    [],
    ["iqa", "iqa"],
    ["not_a_property"],
    ["iqa", 7],
])
def test_ferebus_properties_validation(properties):
    payload = CampaignConfig().to_dict()
    payload["ferebus"]["properties"] = properties
    with pytest.raises(ConfigValidationError, match="ferebus.properties"):
        CampaignConfig.from_dict(payload)


def test_removed_uniform_posterior_fallback_is_rejected():
    cfg = CampaignConfig()
    assert not hasattr(cfg.acquisition, "allow_uniform_posterior_fallback")
    payload = cfg.to_dict()
    payload["acquisition"]["allow_uniform_posterior_fallback"] = True
    with pytest.raises(ConfigValidationError, match="unknown keys"):
        CampaignConfig.from_dict(payload)


def test_adversarial_safety_defaults_roundtrip():
    cfg = CampaignConfig()
    assert cfg.adversarial_safety.salvage_safe_iterate is True
    assert cfg.adversarial_safety.backtrack_to_safe_landing is True
    assert cfg.adversarial_safety.backtrack_points == 16
    assert cfg.adversarial_safety.under_move_retry is True
    payload = cfg.to_dict()
    payload["adversarial_safety"]["backtrack_points"] = 7
    loaded = CampaignConfig.from_dict(payload)
    assert loaded.adversarial_safety.backtrack_points == 7


def test_adversarial_safety_validation_rejects_bad_values():
    payload = CampaignConfig().to_dict()
    payload["adversarial_safety"]["salvage_safe_iterate"] = "yes"
    with pytest.raises(ConfigValidationError, match="salvage_safe_iterate"):
        CampaignConfig.from_dict(payload)
    payload = CampaignConfig().to_dict()
    payload["adversarial_safety"]["backtrack_points"] = 0
    with pytest.raises(ConfigValidationError, match="backtrack_points"):
        CampaignConfig.from_dict(payload)


# --- translator: to_acquisition_config --------------------------------


def test_to_acquisition_config_propagates_nested_values():
    c = CampaignConfig()
    c.acquisition.weights.lambda_force = 2.5
    c.acquisition.subspace.neighbour_count = 73
    c.acquisition.barrier.use_connectivity_barrier = False
    ac = c.to_acquisition_config()
    assert ac.weights.lambda_force == 2.5
    assert ac.subspace.neighbour_count == 73
    assert ac.barrier.use_connectivity_barrier is False


def test_subspace_dimension_is_capped_at_six():
    c = CampaignConfig()
    c.acquisition.subspace.max_subspace_dim = 10
    with pytest.raises(ConfigValidationError, match="max_subspace_dim"):
        CampaignConfig.from_dict(c.to_dict())


def test_subspace_dimension_six_uses_mandatory_active_gradient():
    c = CampaignConfig()
    c.acquisition.subspace.max_subspace_dim = 6
    ac = c.to_acquisition_config()
    assert ac.subspace.max_subspace_dim == 6
    assert not hasattr(ac.gradient, "mode")


def test_to_acquisition_config_propagates_active_fd_gradient_values():
    c = CampaignConfig()
    c.acquisition.gradient.active_step = 3.0e-3
    c.acquisition.gradient.regularization = 4.0e-9

    ac = c.to_acquisition_config()

    assert ac.gradient.active_step == pytest.approx(3.0e-3)
    assert ac.gradient.regularization == pytest.approx(4.0e-9)
    assert not hasattr(ac.gradient, "cartesian_step")


def test_removed_driver_block_is_rejected():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["driver"] = {"objective": "cheap_driver"}

    with pytest.raises(ConfigValidationError, match="unknown keys"):
        CampaignConfig.from_dict(payload)


# --- translator: to_ariadne_run_config --------------------------------


def test_to_ariadne_run_config_propagates_values():
    c = CampaignConfig()
    c.ariadne.max_iter = 350
    c.ariadne.convergence.gradient_max_tolerance_per_angstrom = 1.0e-5
    c.ariadne.delta0 = 0.07
    c.ariadne.trqn_scale_mode = "fixed"
    c.ariadne.trqn_fixed_objective_scale = 0.25
    c.ariadne.trqn_target_initial_grad_norm = 0.02
    c.ariadne.trqn_retry_target_initial_grad_norm = 0.004
    c.ariadne.trqn_backtransform_mode = "newton"
    c.ariadne.trqn_geodesic_bt_mode = "matrix_free"
    c.ariadne.trqn_geodesic_dt = 0.02
    c.ariadne.trqn_geodesic_tol = 2.0e-8
    c.ariadne.trqn_bt_ic_tol = 2.0e-6
    c.ariadne.trqn_max_backtransform_iter = 75
    c.ariadne.trqn_trust_min = 2.0e-4
    rc = c.to_ariadne_run_config()
    assert rc.max_iter == 350
    assert rc.convergence.gradient_max_tolerance_per_ang == pytest.approx(1.0e-5)
    assert rc.delta0 == pytest.approx(0.07)
    assert rc.trqn_scale_mode == "fixed"
    assert rc.trqn_fixed_objective_scale == pytest.approx(0.25)
    assert rc.trqn_target_initial_grad_norm == pytest.approx(0.02)
    assert rc.trqn_retry_target_initial_grad_norm == pytest.approx(0.004)
    assert rc.trqn_backtransform_mode == "newton"
    assert rc.trqn_geodesic_bt_mode == "matrix_free"
    assert rc.trqn_geodesic_dt == pytest.approx(0.02)
    assert rc.trqn_geodesic_tol == pytest.approx(2.0e-8)
    assert rc.trqn_bt_ic_tol == pytest.approx(2.0e-6)
    assert rc.trqn_max_backtransform_iter == 75
    assert rc.trqn_trust_min == pytest.approx(2.0e-4)


def test_ariadne_trqn_scale_validation_rejects_invalid_values():
    c = CampaignConfig()
    c.ariadne.trqn_scale_mode = "mystery"
    with pytest.raises(ConfigValidationError, match="trqn_scale_mode"):
        c._validate()

    c = CampaignConfig()
    c.ariadne.trqn_retry_target_initial_grad_norm = 0.02
    c.ariadne.trqn_target_initial_grad_norm = 0.01
    with pytest.raises(ConfigValidationError, match="retry_target"):
        c._validate()

    c = CampaignConfig()
    c.ariadne.trqn_min_objective_scale = 0.5
    c.ariadne.trqn_max_objective_scale = 0.1
    with pytest.raises(ConfigValidationError, match="min_objective_scale"):
        c._validate()

    c = CampaignConfig()
    c.ariadne.trqn_backtransform_mode = "mystery"
    with pytest.raises(ConfigValidationError, match="trqn_backtransform_mode"):
        c._validate()

    c = CampaignConfig()
    c.ariadne.trqn_geodesic_bt_mode = "mystery"
    with pytest.raises(ConfigValidationError, match="trqn_geodesic_bt_mode"):
        c._validate()

    for attr in (
        "trqn_geodesic_dt",
        "trqn_geodesic_tol",
        "trqn_bt_ic_tol",
        "trqn_trust_min",
    ):
        c = CampaignConfig()
        setattr(c.ariadne, attr, 0.0)
        with pytest.raises(ConfigValidationError, match=attr):
            c._validate()

    c = CampaignConfig()
    c.ariadne.trqn_max_backtransform_iter = 0
    with pytest.raises(ConfigValidationError, match="trqn_max_backtransform_iter"):
        c._validate()


# --- dead-letter cleanup proof: every previously-orphaned field is read --


def test_phase_b_beta_reaches_descriptor_factory():
    from ichor.hpc.active_learning.sampling.descriptors import (
        HybridAlfRmsdDescriptor,
        build_descriptor_from_config,
    )
    c = CampaignConfig()
    c.phase_b.beta = 0.42
    d = build_descriptor_from_config(c)
    assert isinstance(d, HybridAlfRmsdDescriptor)
    assert d.beta == pytest.approx(0.42)


def test_exact_point_allocation_consumed_by_split_executor(tmp_path):
    import json
    from types import SimpleNamespace
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import CampaignPhase
    from ichor.hpc.active_learning.point_allocation import (
        allocation_targets,
        create_point_allocation,
        point_allocation_path,
    )

    c = CampaignConfig()
    c.point_allocation.batch_training_size = 2
    c.point_allocation.batch_internal_validation_size = 1
    c.seed_selection.n_seeds_per_iteration = 3
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=c)
    targets = allocation_targets(c, "active")
    create_point_allocation(
        point_allocation_path(tmp_path, context="active", iteration=1),
        campaign_uid="config-test",
        context="active",
        iteration=1,
        targets=targets,
        primary_candidates=[
            {"candidate_id": "candidate-" + str(index)}
            for index in range(targets["total"])
        ],
        reserve_candidates=[],
    )
    ex.submit_or_run(
        SimpleNamespace(iteration=1, campaign_uid="config-test"),
        CampaignPhase.SPLIT,
    )
    split_json = (
        tmp_path
        / "ACTIVE_LEARNING"
        / "iteration-000001"
        / "allocation"
        / "SPLIT_RECEIPT.json"
    )
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    assert payload["strategy"] == "exact_pre_qm_point_allocation"
    assert payload["targets"] == targets


def test_seed_selection_bulk_fraction_consumed_by_executor(tmp_path):
    """bulk_fraction was a dead-letter in v1; in v2 it flows into
    select_seeds via _inline_seed_select."""
    import json
    from types import SimpleNamespace
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import CampaignPhase
    from ichor.hpc.active_learning.acquisition.trajectory_pool import (
        TrajectoryPool,
    )
    from ichor_hpc.tests.quantum_test_support import prepare_dry_submitted_phase
    from pathlib import Path

    FIXTURE = (
        Path(__file__).resolve().parent / "fixtures" / "water_tetramer.xyz"
    )
    c = CampaignConfig()
    c.max_iterations = 1
    c.seed_selection.n_seeds_per_iteration = 4
    c.seed_selection.bulk_fraction = 0.0  # pure variance pick
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=c)
    TrajectoryPool.import_from(FIXTURE, tmp_path)
    bootstrap_state = SimpleNamespace(
        iteration=0,
        campaign_uid="config-test",
        replacement_round=0,
        reference_data_version=-1,
        models_version=-1,
    )
    ex.postprocess(bootstrap_state, CampaignPhase.PHASE_A_DIVERSITY, observations=[])
    prepare_dry_submitted_phase(
        ex, bootstrap_state, CampaignPhase.INITIAL_GAUSSIAN
    )
    ex.postprocess(
        bootstrap_state,
        CampaignPhase.INITIAL_GAUSSIAN,
        observations=[],
    )
    prepare_dry_submitted_phase(
        ex, bootstrap_state, CampaignPhase.INITIAL_AIMALL
    )
    ex.postprocess(
        bootstrap_state,
        CampaignPhase.INITIAL_AIMALL,
        observations=[],
    )
    ex.submit_or_run(bootstrap_state, CampaignPhase.INITIAL_ALLOCATION_CHECK)
    committed = ex.submit_or_run(
        bootstrap_state,
        CampaignPhase.REFERENCE_COMMIT,
    )
    bootstrap_state.reference_data_version = int(
        committed.state_updates["reference_data_version"]
    )
    ex.postprocess(
        bootstrap_state,
        CampaignPhase.INITIAL_FEREBUS,
        observations=[],
    )
    ex.submit_or_run(
        SimpleNamespace(
            iteration=1,
            campaign_uid="config-test",
            reference_data_version=0,
            models_version=0,
        ),
        CampaignPhase.SEED_SELECT,
    )
    picked = json.loads(
        (
            tmp_path
            / "ACTIVE_LEARNING"
            / "iteration-000001"
            / "seed_selection"
            / "SELECTION.json"
        )
        .read_text(encoding="utf-8")
    )
    # bulk_fraction=0 means every pick is variance-based; bulk_indices empty.
    assert picked["bulk_indices"] == []
    assert len(picked["variance_indices"]) == 4


# --- M15 F4: subspace-dim cross-validation -----------------------------


def test_min_subspace_dim_exceeding_max_rejected():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["min_subspace_dim"] = 8
    payload["acquisition"]["subspace"]["max_subspace_dim"] = 6
    with pytest.raises(ConfigValidationError, match="min_subspace_dim"):
        CampaignConfig.from_dict(payload)


def test_neighbour_count_below_max_subspace_dim_rejected():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["neighbour_count"] = 4
    payload["acquisition"]["subspace"]["max_subspace_dim"] = 6
    with pytest.raises(ConfigValidationError, match="neighbour_count"):
        CampaignConfig.from_dict(payload)


def test_neighbour_count_zero_rejected():
    payload = CampaignConfig().to_dict()
    payload["acquisition"]["subspace"]["neighbour_count"] = 0
    with pytest.raises(ConfigValidationError, match="neighbour_count"):
        CampaignConfig.from_dict(payload)
