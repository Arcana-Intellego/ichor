"""Exact model-kernel protocol tests for active-learning FEREBUS artefacts."""

import numpy as np
import pytest

from ichor.core.models.kernels import ConstantKernel, PeriodicKernel, RBF, RBFCyclic
from ichor.hpc.active_learning.daemon.model_contract import (
    ModelContractError,
    _validate_kernel_family,
)


def test_rbf_contract_requires_one_exact_full_dimension_rbf():
    valid = RBF(
        "k1",
        np.ones(6),
        active_dims=np.arange(6),
    )
    _validate_kernel_family(valid, "rbf", 6)

    incomplete = RBF(
        "k1",
        np.ones(5),
        active_dims=np.arange(5),
    )
    with pytest.raises(ModelContractError, match="active_dims_mismatch:rbf"):
        _validate_kernel_family(incomplete, "rbf", 6)

    with pytest.raises(ModelContractError, match="family_mismatch:rbf"):
        _validate_kernel_family(ConstantKernel("k1", 1.0), "rbf", 6)


def test_periodic_rbf_contract_requires_exact_native_composition_and_dimensions():
    cyclic = RBFCyclic(
        "k1",
        np.ones(5),
        active_dims=np.array([0, 1, 2, 3, 4]),
    )
    periodic = PeriodicKernel(
        "k2",
        np.ones(1),
        np.array([2.0 * np.pi]),
        active_dims=np.array([5]),
    )
    _validate_kernel_family(cyclic * periodic, "periodic_rbf", 6)

    with pytest.raises(ModelContractError, match="composition_mismatch"):
        _validate_kernel_family(periodic * cyclic, "periodic_rbf", 6)

    wrong_periodic_dimensions = PeriodicKernel(
        "k2",
        np.ones(1),
        np.array([2.0 * np.pi]),
        active_dims=np.array([4]),
    )
    with pytest.raises(ModelContractError, match="active_dims_mismatch:periodic"):
        _validate_kernel_family(
            cyclic * wrong_periodic_dimensions,
            "periodic_rbf",
            6,
        )
