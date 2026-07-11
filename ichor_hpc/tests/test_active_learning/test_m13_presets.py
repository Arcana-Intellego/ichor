"""M13 tests: preset_loader, mode_weighting_policy propagation,
reference-scale cache, --preset CLI flag.
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_available_presets_includes_three_canonical():
    from ichor.hpc.active_learning.preset_loader import available_presets
    names = available_presets()
    assert "balanced" in names
    assert "spectroscopy_focused" in names
    assert "thermodynamics_focused" in names


def test_load_preset_returns_dict():
    from ichor.hpc.active_learning.preset_loader import load_preset
    d = load_preset("spectroscopy_focused")
    assert isinstance(d, dict)
    assert d["acquisition"]["subspace"]["mode_weighting_policy"] == "inverse_frequency"


def test_load_preset_unknown_raises():
    from ichor.hpc.active_learning.preset_loader import load_preset, PresetError
    with pytest.raises(PresetError, match="unknown preset"):
        load_preset("nonexistent_preset")


def test_deep_merge_overlay_wins_on_explicit_keys():
    from ichor.hpc.active_learning.preset_loader import deep_merge
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    overlay = {"a": {"c": 99, "e": 4}, "d": 30}
    out = deep_merge(base, overlay)
    assert out == {"a": {"b": 1, "c": 99, "e": 4}, "d": 30}


def test_apply_preset_sparse_overlay_keeps_preset_defaults():
    from ichor.hpc.active_learning.preset_loader import apply_preset
    from ichor.hpc.active_learning.config import CampaignConfig
    sparse = {"schema_version": 9, "acquisition": {"weights": {"lambda_force": 7.5}}}
    eff, _preset = apply_preset("spectroscopy_focused", sparse)
    c = CampaignConfig.from_dict(eff)
    assert c.acquisition.weights.lambda_force == 7.5
    assert c.acquisition.subspace.mode_weighting_policy == "inverse_frequency"
    assert c.acquisition.gradient.mode == "active_fd"
    assert c.acquisition.subspace.max_subspace_dim == 10


def test_spectroscopy_preset_passes_subspace_dim_guard():
    from ichor.hpc.active_learning.preset_loader import apply_preset
    from ichor.hpc.active_learning.config import CampaignConfig
    eff, _ = apply_preset("spectroscopy_focused", {"schema_version": 9})
    c = CampaignConfig.from_dict(eff)
    ac = c.to_acquisition_config()
    assert ac.subspace.max_subspace_dim == 10
    assert ac.gradient.mode == "active_fd"


def test_thermodynamics_preset_emphasises_force():
    from ichor.hpc.active_learning.preset_loader import apply_preset
    from ichor.hpc.active_learning.config import CampaignConfig
    eff, _ = apply_preset("thermodynamics_focused", {"schema_version": 9})
    c = CampaignConfig.from_dict(eff)
    assert c.acquisition.weights.lambda_force == 1.5
    assert c.acquisition.weights.lambda_energy == 0.5
    assert c.acquisition.subspace.mode_weighting_policy == "variance"


def test_mode_weighting_policy_propagates_to_acquisition_config():
    from ichor.hpc.active_learning.config import CampaignConfig
    c = CampaignConfig()
    for policy in ("variance", "inverse_frequency", "uniform"):
        c.acquisition.subspace.mode_weighting_policy = policy
        ac = c.to_acquisition_config()
        assert ac.subspace.mode_weighting_policy == policy


def test_inverse_frequency_biases_toward_low_omega():
    from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    acq.subspace = SimpleNamespace(mode_weights=(0.5, 0.3, 0.2))
    cfg = SimpleNamespace()
    cfg.subspace = SimpleNamespace(mode_weighting_policy="inverse_frequency")
    cfg.stencils = SimpleNamespace(curvature_floor=1e-6)
    acq.config = cfg
    modes = [SimpleNamespace(omega=0.1), SimpleNamespace(omega=1.0), SimpleNamespace(omega=5.0)]
    w = acq._effective_mode_weights(modes)
    assert len(w) == 3
    assert sum(w) == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]
    assert w[0] > 0.5


def test_uniform_policy_returns_one_over_r():
    from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    acq.subspace = SimpleNamespace(mode_weights=(0.5, 0.5))
    cfg = SimpleNamespace()
    cfg.subspace = SimpleNamespace(mode_weighting_policy="uniform")
    cfg.stencils = SimpleNamespace(curvature_floor=1e-6)
    acq.config = cfg
    modes = [SimpleNamespace(omega=0.5), SimpleNamespace(omega=2.0), SimpleNamespace(omega=3.0)]
    w = acq._effective_mode_weights(modes)
    assert all(x == pytest.approx(1.0 / 3.0) for x in w)


def test_external_reference_scales_kwarg_in_init():
    from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
    import inspect
    src = inspect.getsource(SeedLocalAdversarialAcquisition.__init__)
    assert "external_reference_scales" in src
    assert "if external_reference_scales is not None" in src


def test_campaign_state_carries_reference_scales_fields():
    from ichor.hpc.active_learning.daemon.state import CampaignState
    st = CampaignState()
    assert st.reference_scales is None
    assert st.reference_scales_iteration == -1
    st.reference_scales = {"omega": 1.0}
    st.reference_scales_iteration = 5
    payload = st.to_dict()
    st2 = CampaignState.from_dict(payload)
    assert st2.reference_scales == {"omega": 1.0}
    assert st2.reference_scales_iteration == 5


def test_dry_run_refresh_policy_every_iteration_recomputes(tmp_path):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    cfg = CampaignConfig()
    cfg.acquisition.references.refresh_policy = "every_iteration"
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=cfg)
    state = SimpleNamespace(iteration=0, reference_scales=None, reference_scales_iteration=-1)
    assert ex._maybe_refresh_reference_scales(state) is True
    state.iteration = 1
    assert ex._maybe_refresh_reference_scales(state) is True


def test_dry_run_refresh_policy_never_only_once(tmp_path):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    cfg = CampaignConfig()
    cfg.acquisition.references.refresh_policy = "never"
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=cfg)
    state = SimpleNamespace(iteration=0, reference_scales=None, reference_scales_iteration=-1)
    assert ex._maybe_refresh_reference_scales(state) is True
    state.iteration = 1
    assert ex._maybe_refresh_reference_scales(state) is False
    state.iteration = 9
    assert ex._maybe_refresh_reference_scales(state) is False


def test_dry_run_refresh_policy_every_n_iterations(tmp_path):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    cfg = CampaignConfig()
    cfg.acquisition.references.refresh_policy = "every_n_iterations"
    cfg.acquisition.references.refresh_period = 3
    ex = DryRunPhaseExecutor(campaign_dir=tmp_path, config=cfg)
    state = SimpleNamespace(iteration=0, reference_scales=None, reference_scales_iteration=-1)
    assert ex._maybe_refresh_reference_scales(state) is True
    state.iteration = 1
    assert ex._maybe_refresh_reference_scales(state) is False
    state.iteration = 2
    assert ex._maybe_refresh_reference_scales(state) is False
    state.iteration = 3
    assert ex._maybe_refresh_reference_scales(state) is True
