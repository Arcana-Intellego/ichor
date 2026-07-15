"""Clean-break tests for the sole mature acquisition objective."""

from __future__ import annotations

import pytest

from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
from ichor.core.adversarial.config import AcquisitionConfig


def test_driver_configuration_has_been_removed():
    config = AcquisitionConfig()
    assert not hasattr(config, "driver")


def test_components_rejects_legacy_cheap_driver_before_evaluation():
    acquisition = object.__new__(SeedLocalAdversarialAcquisition)
    with pytest.raises(ValueError, match="mature full acquisition"):
        acquisition.components(object(), objective="cheap_driver")


def test_gradient_rejects_non_full_objective_before_evaluation():
    acquisition = object.__new__(SeedLocalAdversarialAcquisition)
    with pytest.raises(ValueError, match="mature full acquisition"):
        acquisition.gradient(object(), objective="cheap_driver")


def test_gradient_rejects_cartesian_mode_before_evaluation():
    acquisition = object.__new__(SeedLocalAdversarialAcquisition)
    with pytest.raises(ValueError, match="active_fd only"):
        acquisition.gradient(object(), mode="cartesian_fd")


def test_gradient_defaults_to_active_subspace_path():
    acquisition = object.__new__(SeedLocalAdversarialAcquisition)
    sentinel = object()
    acquisition._active_finite_difference_gradient = (
        lambda atoms, objective="full": sentinel
    )
    assert acquisition.gradient(object()) is sentinel
