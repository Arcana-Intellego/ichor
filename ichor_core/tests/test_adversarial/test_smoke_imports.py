"""Smoke test verifying the adversarial sub-package is installed and importable.

This is intentionally minimal: it only confirms the public API surface resolves
and that AcquisitionConfig() instantiates with its defaults. Remeber - no numerical or
behavioural assertions belong here.
"""
import pytest


def test_public_symbols_importable():
    from ichor.core.adversarial import (
        AcquisitionBreakdown,
        AcquisitionConfig,
        AriadneAdversarialCalculator,
        BarrierConfig,
        GradientConfig,
        LocalSubspace,
        ModeEvaluation,
        ReferenceScaleConfig,
        SeedLocalAdversarialAcquisition,
        StencilConfig,
        SubspaceConfig,
        TotalEnergyPosterior,
        WeightConfig,
        build_local_subspace,
    )
    #Touch each symbol so flake8 / static analysers don't strip the import.
    assert AcquisitionBreakdown is not None
    assert AriadneAdversarialCalculator is not None
    assert LocalSubspace is not None
    assert ModeEvaluation is not None
    assert SeedLocalAdversarialAcquisition is not None
    assert TotalEnergyPosterior is not None
    assert build_local_subspace is not None
    for cfg_cls in (
        AcquisitionConfig,
        BarrierConfig,
        GradientConfig,
        ReferenceScaleConfig,
        StencilConfig,
        SubspaceConfig,
        WeightConfig,
    ):
        assert cfg_cls is not None


def test_default_acquisition_config_instantiates():
    from ichor.core.adversarial import AcquisitionConfig

    cfg = AcquisitionConfig()
    assert cfg.property_name == "iqa"
    assert cfg.subspace.variance_capture == pytest.approx(0.90)
    assert cfg.gradient.mode == "cartesian_fd"
    assert cfg.gradient.cartesian_step == pytest.approx(1.0e-4)


def test_adversarial_module_path():
    """Confirm the package lives where the rest of ichor.core does."""
    import ichor.core.adversarial as adv
    from pathlib import Path

    pkg_path = Path(adv.__file__).resolve().parent
    assert pkg_path.name == "adversarial"
    assert pkg_path.parent.name == "core"
