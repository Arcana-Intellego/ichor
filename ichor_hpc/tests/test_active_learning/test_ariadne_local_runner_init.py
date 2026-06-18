from types import SimpleNamespace
import json

import numpy as np

from ichor.hpc.active_learning.acquisition.ariadne_local_runner import (
    _ARIADNE_CARTESIAN_RECOVERY_NEWTON,
    _ARIADNE_GEO_BT_DENSE,
    _ARIADNE_ROT_PRIMITIVE_EXPMAP3,
    _ARIADNE_TRQN_CONTROLLER_NONE,
    _append_status_sample,
    _build_ds,
    _build_trqn,
    _finalise_optimiser_diagnostics,
    _new_optimiser_diagnostics,
)
from ichor.hpc.active_learning.acquisition.ariadne_runner import AriadneRunConfig


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
