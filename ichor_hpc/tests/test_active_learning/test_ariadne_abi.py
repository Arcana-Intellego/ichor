"""Strict ARIADNE wrapper-contract tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from ichor.hpc.active_learning.acquisition.ariadne_abi import (
    AriadneABIError,
    DS_STATUS_LENGTH,
    TRQN_STATUS_LENGTH,
    probe_ariadne_module,
)


class _Optimiser:
    def init(self, **_kwargs):
        return None

    def step_py(self, *, stage, f_old, f_new, g_xyz_new):
        return stage, f_old, f_new, g_xyz_new

    def get_state_flat_py(self, _q, _g):
        return None

    def get_status_py(self):
        return ()

    def set_invalid_trial_reason_py(self, _reason):
        return None


def _module(*, optimiser_type=_Optimiser, descriptor=None):
    module = SimpleNamespace(
        Geometric_Trqn=SimpleNamespace(trust_region_qn=optimiser_type),
        Ds_Optimiser=SimpleNamespace(dissipative_symplectic=optimiser_type),
    )
    if descriptor is not None:
        module.get_abi_info_py = descriptor
    return module


def test_probe_records_exact_status_contract_and_optional_descriptor():
    receipt = probe_ariadne_module(
        _module(
            descriptor=lambda: {
                "backend_abi": "test-v1",
                "status_lengths": (np.int64(89), np.int64(42)),
            }
        )
    )

    assert receipt["trqn"]["status_length"] == TRQN_STATUS_LENGTH == 89
    assert receipt["ds"]["status_length"] == DS_STATUS_LENGTH == 42
    assert receipt["backend_descriptor_available"] is True
    assert receipt["backend_descriptor"] == {
        "backend_abi": "test-v1",
        "status_lengths": [89, 42],
    }


def test_probe_rejects_missing_invalid_trial_reason_setter():
    class _Incomplete:
        init = _Optimiser.init
        step_py = _Optimiser.step_py
        get_state_flat_py = _Optimiser.get_state_flat_py
        get_status_py = _Optimiser.get_status_py

    with pytest.raises(AriadneABIError, match="set_invalid_trial_reason_py"):
        probe_ariadne_module(_module(optimiser_type=_Incomplete))


def test_probe_rejects_step_signature_drift():
    class _WrongStep(_Optimiser):
        def step_py(self, stage, objective):
            return stage, objective

    with pytest.raises(AriadneABIError, match="signature is missing parameters"):
        probe_ariadne_module(_module(optimiser_type=_WrongStep))


def test_probe_rejects_missing_wrapper_namespace():
    with pytest.raises(AriadneABIError, match="Geometric_Trqn and Ds_Optimiser"):
        probe_ariadne_module(SimpleNamespace())
