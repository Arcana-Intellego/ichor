from types import SimpleNamespace
import json

import numpy as np
import pytest

from ichor.hpc.active_learning.acquisition.ariadne_local_runner import (
    _ARIADNE_CARTESIAN_RECOVERY_NEWTON,
    _ARIADNE_GEO_BT_DENSE,
    _ARIADNE_ROT_PRIMITIVE_EXPMAP3,
    _ARIADNE_TRQN_CONTROLLER_NONE,
    _DS_INIT_PROFILE,
    _append_status_sample,
    _append_trace_event,
    _build_ds,
    _build_trqn,
    _compute_trqn_objective_scale,
    _ds_safe_init_kwargs,
    _finalise_optimiser_diagnostics,
    _new_optimiser_diagnostics,
    _validate_ds_init_kwargs,
    run_optimisation_against_calculator,
)
import ichor.hpc.active_learning.acquisition.ariadne_local_runner as local_runner
from ichor.hpc.active_learning.acquisition.ariadne_runner import AriadneRunConfig


def test_append_trace_event_writes_jsonl(tmp_path):
    trace = tmp_path / "seed_0000" / "ARIADNE_TRACE.jsonl"

    _append_trace_event(
        trace,
        event="accepted_step",
        step=3,
        optimiser="dissipative_symplectic",
        alpha=2.5,
        grad_norm=0.25,
        accepted=True,
        wall_seconds=1.5,
    )

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "schema_version": 1,
            "accepted": True,
            "alpha": 2.5,
            "event": "accepted_step",
            "grad_norm": 0.25,
            "optimiser": "dissipative_symplectic",
            "step": 3,
            "wall_seconds": 1.5,
        }
    ]


class _FakeOptimiser:
    def __init__(self):
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = dict(kwargs)


class _FakeTrqnFactory:
    def __init__(self):
        self.last = None

    def trust_region_qn(self):
        self.last = _FakeOptimiser()
        return self.last


class _FakeDsFactory:
    def __init__(self):
        self.last = None

    def dissipative_symplectic(self):
        self.last = _FakeOptimiser()
        return self.last


def _fake_ariadne():
    trqn = _FakeTrqnFactory()
    ds = _FakeDsFactory()
    return SimpleNamespace(
        Geometric_Trqn=trqn,
        Ds_Optimiser=ds,
        _trqn_factory=trqn,
        _ds_factory=ds,
    )


def _inputs():
    q0 = np.asfortranarray(np.zeros((2, 3), dtype=np.float64))
    g0 = np.asfortranarray(np.zeros((2, 3), dtype=np.float64))
    atom_list = np.asarray(["O ", "H "], dtype="S2")
    return q0, g0, atom_list


def test_trqn_init_forwards_ariadne_enum_defaults():
    ariadne = _fake_ariadne()
    q0, g0, atom_list = _inputs()

    _build_trqn(
        ariadne,
        q0,
        g0,
        atom_list,
        AriadneRunConfig(delta0=0.2, delta_max=0.5, hessian_model="schlegel"),
    )

    kwargs = ariadne._trqn_factory.last.init_kwargs
    assert kwargs["controller_mode"] == _ARIADNE_TRQN_CONTROLLER_NONE
    assert kwargs["cartesian_recovery_mode"] == _ARIADNE_CARTESIAN_RECOVERY_NEWTON
    assert kwargs["geo_bt_mode"] == _ARIADNE_GEO_BT_DENSE
    assert kwargs["rot_primitive_mode"] == _ARIADNE_ROT_PRIMITIVE_EXPMAP3
    assert kwargs["skip_bfgs_after_rot_reset"] is False
    assert kwargs["hessian_model"] == 3
    assert kwargs["trust0"] == 0.2
    assert kwargs["trust_max"] == 0.5


def test_ds_init_forwards_ariadne_geometry_enum_defaults():
    ariadne = _fake_ariadne()
    q0, g0, atom_list = _inputs()

    _build_ds(
        ariadne,
        q0,
        g0,
        atom_list,
        AriadneRunConfig(delta0=0.3, gamma=0.4, f_tol=1.0e-5, gradf_tol=2.0e-4),
    )

    kwargs = ariadne._ds_factory.last.init_kwargs
    assert kwargs["cartesian_recovery_mode"] == _ARIADNE_CARTESIAN_RECOVERY_NEWTON
    assert kwargs["geo_bt_mode"] == _ARIADNE_GEO_BT_DENSE
    assert kwargs["rot_primitive_mode"] == _ARIADNE_ROT_PRIMITIVE_EXPMAP3
    assert kwargs["h"] == 0.3
    assert kwargs["gamma"] == 0.4
    assert kwargs["f_tol"] == 1.0e-5
    assert kwargs["gradf_tol"] == 2.0e-4
    assert kwargs["hpos_estimator"] == "syev"
    assert kwargs["lanczos_k"] == 4
    assert kwargs["hessian_model"] == "almlof"
    assert kwargs["ds_controller"] == "safe"
    assert kwargs["block_scale_seed_mode"] == "uniform"
    assert kwargs["block_coupling_mode"] == "row_gram"
    assert kwargs["block_transfer_mode"] == "weighted"
    assert kwargs["delta0"] == 0.3
    assert kwargs["delta_min"] > 0.0
    assert kwargs["delta_max"] >= kwargs["delta0"]
    assert kwargs["delta_grow"] > 1.0
    assert 0.0 < kwargs["delta_shrink"] < 1.0
    assert kwargs["h_min"] > 0.0
    assert kwargs["h_max"] >= kwargs["h"]
    assert kwargs["gamma_min"] > 0.0
    assert kwargs["gamma_max"] >= kwargs["gamma"]
    assert kwargs["cart_step_inf_max"] > 0.0
    assert kwargs["cart_step_rms_max"] > 0.0
    assert kwargs["cart_atom_step_max"] > 0.0
    assert kwargs["cart_min_pair_dist"] > 0.0
    assert kwargs["htvi_p_bregman"] == 2.0
    assert kwargs["htvi_gamma_0"] > 0.0
    assert kwargs["htvi_max_inner"] >= 1
    assert kwargs["htvi_picard_max"] >= 1
    assert kwargs["htvi_picard_tol_m"] > 0.0
    assert kwargs["geo_sol_dt"] > 0.0
    assert kwargs["geo_sol_tol"] > 0.0
    assert kwargs["bt_ic_tol"] > 0.0
    assert kwargs["max_backtransform_iter"] >= 1
    assert kwargs["finish_after_uphill"] >= 1
    assert kwargs["finish_small_step_streak"] >= 1
    assert kwargs["finish_small_gmax_cap"] >= kwargs["finish_gmax_trigger"]
    assert kwargs["delta_regrow_step_frac"] > 0.0
    assert kwargs["controller_de_tol"] >= 0.0
    assert kwargs["finish_stall_window"] >= 1
    assert kwargs["finish_delta_grow"] > 1.0
    assert 0.0 < kwargs["finish_delta_shrink"] < 1.0
    assert kwargs["contact_max_edges_per_node"] >= 1
    assert kwargs["soft_capacity"] >= 1
    assert kwargs["soft_pulse_momentum_policy"] == 1
    assert kwargs["soft_negative_curvature_policy"] == 1
    assert kwargs["contact_participation_metric_mode"] == 2


def test_ds_safe_init_kwargs_validate_starter_pack_sensitive_defaults():
    kwargs = _ds_safe_init_kwargs(
        AriadneRunConfig(delta0=0.2, delta_max=0.5, gamma=0.4, hessian_model="schlegel")
    )

    assert kwargs["hessian_model"] == "schlegel"
    assert kwargs["hpos_estimator"] == "syev"
    assert kwargs["lanczos_k"] == 4
    assert kwargs["htvi_p_bregman"] == 2.0
    assert kwargs["htvi_gamma_0"] > 0.0
    assert kwargs["h_max"] >= kwargs["h"]
    assert kwargs["gamma_max"] >= kwargs["gamma"]
    assert kwargs["block_scale_seed_mode"] == "uniform"
    assert kwargs["block_coupling_mode"] == "row_gram"
    assert kwargs["block_transfer_mode"] == "weighted"


def test_ds_safe_init_validation_rejects_bad_profile_values():
    kwargs = _ds_safe_init_kwargs(AriadneRunConfig())
    kwargs["htvi_p_bregman"] = 0.0
    kwargs["lanczos_k"] = 0
    kwargs["block_coupling_mode"] = "none"

    with pytest.raises(ValueError, match="invalid DS init defaults"):
        _validate_ds_init_kwargs(kwargs)


def test_trqn_adaptive_objective_scale_targets_initial_gradient_norm():
    info = _compute_trqn_objective_scale(
        AriadneRunConfig(trqn_target_initial_grad_norm=0.01),
        64.0,
    )

    assert info["scale"] == pytest.approx(0.01 / 64.0)
    assert info["scaled_grad_norm"] == pytest.approx(0.01)
    assert info["reason"] == "adaptive_initial_gradient"


def test_trqn_adaptive_objective_scale_respects_bounds_and_zero_gradient():
    info = _compute_trqn_objective_scale(
        AriadneRunConfig(
            trqn_target_initial_grad_norm=0.01,
            trqn_min_objective_scale=1.0e-4,
        ),
        1.0e9,
    )
    assert info["scale"] == pytest.approx(1.0e-4)

    zero = _compute_trqn_objective_scale(AriadneRunConfig(), 0.0)
    assert zero["scale"] == pytest.approx(1.0)
    assert zero["reason"] == "zero_initial_gradient"


def test_optimiser_diagnostics_are_json_safe_for_no_trial_path():
    diagnostics = _new_optimiser_diagnostics("trust_region_qn")
    diagnostics["n_stage0_calls"] = 3
    diagnostics["n_no_proposal_pending"] = 3
    _append_status_sample(
        diagnostics,
        step_index=0,
        label="after_stage0",
        status=(np.float64(1.0), np.bool_(False), float("nan")),
    )

    _finalise_optimiser_diagnostics(
        diagnostics,
        return_code=1,
        converged=False,
    )

    assert diagnostics["last_return_code_reason"] == "max_iterations_no_trial_evaluations"
    assert diagnostics["status_samples_first"][0]["status"] == [1.0, False, "nan"]
    json.dumps(diagnostics)


def _trqn_status(
    *,
    proposal_pending=0,
    invalid_reason=2,
    step_state=4,
    skip_step_after_rebuild=0,
    run_idx=0,
    trust=1.0e-4,
    consecutive_bt_fail_count=1,
):
    status = [0] * 90
    status[0] = trust
    status[1] = 0.0
    status[2] = run_idx
    status[3] = proposal_pending
    status[23] = invalid_reason
    status[35] = step_state
    status[39] = 1
    status[54] = skip_step_after_rebuild
    status[65] = consecutive_bt_fail_count
    return tuple(status)


class _NoProposalTrqn:
    def __init__(self, *, skip_step_after_rebuild=False):
        self.q = np.zeros(6, dtype=np.float64)
        self.g = np.ones(6, dtype=np.float64)
        self.init_kwargs = None
        self.run_idx = 0
        self.skip_step_after_rebuild = bool(skip_step_after_rebuild)

    def init(self, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.q = np.asarray(kwargs["q0_xyz"], dtype=np.float64).reshape(-1).copy()
        self.g = np.asarray(kwargs["g0_xyz"], dtype=np.float64).reshape(-1).copy()

    def step_py(self, *, stage, f_old, f_new, g_xyz_new):
        if int(stage) == 1:
            self.run_idx += 1

    def get_status_py(self):
        return _trqn_status(
            run_idx=self.run_idx,
            skip_step_after_rebuild=int(self.skip_step_after_rebuild),
            consecutive_bt_fail_count=max(1, self.run_idx),
        )

    def get_state_flat_py(self, q_flat, g_flat):
        q_flat[:] = self.q
        g_flat[:] = self.g


class _AcceptingDs:
    def __init__(self):
        self.q = np.zeros(6, dtype=np.float64)
        self.g = np.ones(6, dtype=np.float64)
        self.init_kwargs = None
        self.pending = False
        self.accepted = False
        self.converged = False
        self.f_current = 0.0

    def init(self, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.q = np.asarray(kwargs["q0_xyz"], dtype=np.float64).reshape(-1).copy()
        self.g = np.asarray(kwargs["g0_xyz"], dtype=np.float64).reshape(-1).copy()

    def proposal_pending(self):
        return self.pending

    def step_py(self, *, stage, f_old, f_new, g_xyz_new):
        if int(stage) == 0:
            self.pending = True
            self.q = self.q + 0.1
            return
        self.pending = False
        self.accepted = True
        self.converged = True
        self.f_current = float(f_new)
        self.g = np.asarray(g_xyz_new, dtype=np.float64).reshape(-1).copy()

    def get_status_py(self):
        status = [0] * 12
        status[2] = self.f_current
        status[8] = int(self.converged)
        status[9] = int(self.accepted)
        return tuple(status)

    def get_state_flat_py(self, q_flat, g_flat):
        q_flat[:] = self.q
        g_flat[:] = self.g


class _NoProposalTrqnFactory:
    def __init__(self, *, skip_step_after_rebuild=False):
        self.skip_step_after_rebuild = bool(skip_step_after_rebuild)
        self.last = None
        self.instances = []

    def trust_region_qn(self):
        self.last = _NoProposalTrqn(
            skip_step_after_rebuild=self.skip_step_after_rebuild,
        )
        self.instances.append(self.last)
        return self.last


class _AcceptingDsFactory:
    def __init__(self):
        self.last = None

    def dissipative_symplectic(self):
        self.last = _AcceptingDs()
        return self.last


def _fake_runner_ariadne(*, skip_step_after_rebuild=False):
    trqn = _NoProposalTrqnFactory(
        skip_step_after_rebuild=skip_step_after_rebuild,
    )
    ds = _AcceptingDsFactory()
    return SimpleNamespace(
        Geometric_Trqn=trqn,
        Ds_Optimiser=ds,
        _trqn_factory=trqn,
        _ds_factory=ds,
    )


class _FakeAtom:
    def __init__(self, symbol):
        self.symbol = str(symbol)


class _FakeAtoms:
    def __init__(self, positions=None):
        self._symbols = ["O", "H"]
        self._positions = np.asarray(
            positions if positions is not None else np.zeros((2, 3)),
            dtype=np.float64,
        )
        self.calc = None

    def copy(self):
        copied = _FakeAtoms(self._positions.copy())
        copied.calc = self.calc
        return copied

    def __len__(self):
        return 2

    def __iter__(self):
        return iter([_FakeAtom(s) for s in self._symbols])

    def get_positions(self):
        return self._positions.copy()

    def set_positions(self, positions):
        self._positions = np.asarray(positions, dtype=np.float64).reshape(2, 3)

    def get_potential_energy(self):
        return 0.0

    def get_forces(self):
        return np.ones((2, 3), dtype=np.float64)


def test_repeated_trqn_no_proposal_backtransform_failure_falls_back_to_ds(monkeypatch):
    ariadne = _fake_runner_ariadne()
    monkeypatch.setattr(local_runner, "_import_ariadne", lambda: ariadne)

    result = run_optimisation_against_calculator(
        _FakeAtoms(),
        calculator=None,
        run_config=AriadneRunConfig(
            optimiser="trust_region_qn",
            max_iter=8,
            fallback_to_ds=True,
            trqn_retry_on_no_proposal=False,
        ),
    )

    assert result.return_code == 0
    assert result.fell_back_to_ds is True
    assert result.diagnostics["n_no_proposal_pending"] == 3
    assert result.diagnostics["n_no_proposal_backtransform_fail"] == 3
    assert result.diagnostics["n_no_proposal_recoveries"] == 1
    assert result.diagnostics["n_fallback_to_ds"] == 1
    assert result.diagnostics["ds_init_profile"] == _DS_INIT_PROFILE
    assert result.diagnostics["optimiser_final"] == "dissipative_symplectic"
    assert result.n_evaluations == 3
    assert len(result.candidate_positions_angstrom) == 2
    assert ariadne._ds_factory.last.init_kwargs["hpos_estimator"] == "syev"
    assert ariadne._ds_factory.last.init_kwargs["lanczos_k"] == 4
    assert ariadne._ds_factory.last.init_kwargs["htvi_p_bregman"] == 2.0
    assert ariadne._ds_factory.last.init_kwargs["htvi_gamma_0"] > 0.0


def test_trqn_no_proposal_retries_with_lower_objective_scale_before_ds(monkeypatch):
    ariadne = _fake_runner_ariadne()
    monkeypatch.setattr(local_runner, "_import_ariadne", lambda: ariadne)

    result = run_optimisation_against_calculator(
        _FakeAtoms(),
        calculator=None,
        run_config=AriadneRunConfig(
            optimiser="trust_region_qn",
            max_iter=9,
            fallback_to_ds=True,
            trqn_retry_on_no_proposal=True,
            trqn_target_initial_grad_norm=0.01,
            trqn_retry_target_initial_grad_norm=0.003,
        ),
    )

    first, retry = ariadne._trqn_factory.instances[:2]
    first_norm = np.linalg.norm(first.init_kwargs["g0_xyz"])
    retry_norm = np.linalg.norm(retry.init_kwargs["g0_xyz"])
    assert first_norm == pytest.approx(0.01)
    assert retry_norm == pytest.approx(0.003)
    assert result.return_code == 0
    assert result.fell_back_to_ds is True
    assert result.diagnostics["trqn_retry_attempted"] is True
    assert result.diagnostics["trqn_retry_objective_scale"] < (
        result.diagnostics["trqn_objective_scale"]
    )
    assert result.diagnostics["trqn_retry_succeeded"] is False
    assert result.diagnostics["trqn_failed_after_retry"] is True
    assert result.diagnostics["n_fallback_to_ds"] == 1


def test_repeated_trqn_no_proposal_backtransform_failure_fails_early(monkeypatch):
    ariadne = _fake_runner_ariadne()
    monkeypatch.setattr(local_runner, "_import_ariadne", lambda: ariadne)

    result = run_optimisation_against_calculator(
        _FakeAtoms(),
        calculator=None,
        run_config=AriadneRunConfig(
            optimiser="trust_region_qn",
            max_iter=8,
            fallback_to_ds=False,
            trqn_retry_on_no_proposal=False,
        ),
    )

    assert result.return_code == 2
    assert result.diagnostics["last_return_code_reason"] == (
        "trqn_no_proposal_backtransform_fail"
    )
    assert result.diagnostics["n_stage0_calls"] == 3
    assert result.diagnostics["n_no_proposal_backtransform_fail"] == 3
    assert result.diagnostics["n_fallback_to_ds"] == 0
    assert result.diagnostics["ds_init_profile"] is None


def test_rebuild_skip_no_proposal_path_is_not_classified_as_fatal(monkeypatch):
    ariadne = _fake_runner_ariadne(skip_step_after_rebuild=True)
    monkeypatch.setattr(local_runner, "_import_ariadne", lambda: ariadne)

    result = run_optimisation_against_calculator(
        _FakeAtoms(),
        calculator=None,
        run_config=AriadneRunConfig(
            optimiser="trust_region_qn",
            max_iter=2,
            fallback_to_ds=False,
            trqn_retry_on_no_proposal=False,
        ),
    )

    assert result.return_code == 1
    assert result.diagnostics["last_return_code_reason"] == (
        "max_iterations_no_trial_evaluations"
    )
    assert result.diagnostics["n_skip_step_after_rebuild"] == 2
    assert result.diagnostics["n_no_proposal_invalid"] == 0
