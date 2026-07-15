"""The mature gradient surface contains active-direction controls only."""

from ichor.core.adversarial.config import GradientConfig


def test_gradient_config_exposes_only_active_fd_numerics():
    config = GradientConfig(active_step=2.0e-3, regularization=3.0e-9)
    assert config.active_step == 2.0e-3
    assert config.regularization == 3.0e-9
    assert not hasattr(config, "mode")
    assert not hasattr(config, "cartesian_step")
    assert not hasattr(config, "cartesian_step_floor")
    assert not hasattr(config, "ghost_mass_threshold")
