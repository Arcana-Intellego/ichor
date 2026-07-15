import math

import pytest

from ichor.core.adversarial import acquisition as acquisition_mod
from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
from ichor.core.adversarial.config import (
    AcquisitionConfig,
    WeightConfig,
)
from ichor.core.atoms import Atom, Atoms


class _Posterior:
    def mean(self, atoms):
        return 0.0

    def variance(self, atoms):
        return 4.0

    def variance_components(self, atoms):
        return 4.0, {"C1": 2.0, "H2": 1.0}


def _acquisition(*, model=None, strength=0.0):
    cfg = AcquisitionConfig(
        weights=WeightConfig(
            lambda_force=0.0,
            lambda_frequency=0.0,
            lambda_anharmonic=0.0,
            lambda_energy=1.0,
            lambda_distance=0.0,
        ),
    )
    acq = SeedLocalAdversarialAcquisition.__new__(SeedLocalAdversarialAcquisition)
    acq.config = cfg
    acq.posterior = _Posterior()
    acq.reference_scales = {
        "energy": 2.0,
        "force": 1.0,
        "omega": 1.0,
        "anh": 1.0,
        "anh_std": 1.0,
    }
    acq.subspace = object()
    acq.barrier_state = object()
    acq.error_calibration_model = model
    acq.error_calibration_apply_strength = strength
    acq._fullspace_confinement_metrics = lambda atoms: {
        "residual_penalty": 0.0,
        "rmsd_penalty": 0.0,
        "fallback_reasons": [],
        "residual_distance": 0.0,
        "aligned_rmsd_ang": 0.0,
    }
    return acq


def test_record_only_strength_zero_leaves_energy_risk_unchanged(monkeypatch):
    monkeypatch.setattr(acquisition_mod, "whitened_distance_squared", lambda *args: 0.0)
    monkeypatch.setattr(
        acquisition_mod,
        "chemistry_barrier_value",
        lambda *args, **kwargs: 0.0,
    )
    monkeypatch.setattr(SeedLocalAdversarialAcquisition, "_mode_metrics", lambda self, atoms, mean_energy: ())

    atoms = Atoms([Atom("C", 0, 0, 0), Atom("H", 1, 0, 0)])
    acq = _acquisition(model={"tables": {}}, strength=0.0)

    breakdown = acq.components(atoms, include_movement=False)

    assert breakdown.calibration_applied is False
    assert breakdown.energy_risk == pytest.approx(
        math.log1p((4.0 / math.sqrt(len(atoms))) / 2.0)
    )
    assert breakdown.total == pytest.approx(breakdown.energy_risk)


def test_apply_to_acquisition_replaces_energy_risk_when_strength_one(monkeypatch):
    monkeypatch.setattr(acquisition_mod, "whitened_distance_squared", lambda *args: 0.0)
    monkeypatch.setattr(
        acquisition_mod,
        "chemistry_barrier_value",
        lambda *args, **kwargs: 0.0,
    )
    monkeypatch.setattr(SeedLocalAdversarialAcquisition, "_mode_metrics", lambda self, atoms, mean_energy: ())
    model = {
        "schema_version": 2,
        "output_units": "ha_per_sqrt_atom",
        "uncertainty_axis": "model_normalised",
        "reference_error_ha_per_sqrt_atom": 0.1,
        "tables": {
            "global_total": {
                "bins": [
                    {
                        "raw_uncertainty_min": 0.0,
                        "raw_uncertainty_max": 5.0,
                        "calibrated_abs_error_ha_per_sqrt_atom": 0.3,
                    }
                ]
            },
        },
    }

    atoms = Atoms([Atom("C", 0, 0, 0), Atom("H", 1, 0, 0)])
    acq = _acquisition(model=model, strength=1.0)

    breakdown = acq.components(atoms, include_movement=False)

    assert breakdown.calibration_applied is True
    assert breakdown.raw_energy_risk == pytest.approx(
        math.log1p((4.0 / math.sqrt(len(atoms))) / 2.0)
    )
    assert breakdown.calibrated_expected_iqa_error_ha_per_sqrt_atom == pytest.approx(0.3)
    assert breakdown.energy_risk == pytest.approx(
        math.log1p(0.3 / 0.1)
    )


def test_apply_to_acquisition_requires_global_total_table(monkeypatch):
    monkeypatch.setattr(acquisition_mod, "whitened_distance_squared", lambda *args: 0.0)
    monkeypatch.setattr(
        acquisition_mod,
        "chemistry_barrier_value",
        lambda *args, **kwargs: 0.0,
    )
    monkeypatch.setattr(SeedLocalAdversarialAcquisition, "_mode_metrics", lambda self, atoms, mean_energy: ())
    legacy_atom_only_model = {
        "schema_version": 2,
        "output_units": "ha_per_sqrt_atom",
        "uncertainty_axis": "model_normalised",
        "reference_error_ha_per_sqrt_atom": 0.1,
        "tables": {
            "global": {
                "bins": [
                    {
                        "raw_uncertainty_min": 0.0,
                        "raw_uncertainty_max": 5.0,
                        "calibrated_abs_error_ha_per_sqrt_atom": 9.0,
                    }
                ]
            },
        },
    }

    atoms = Atoms([Atom("C", 0, 0, 0), Atom("H", 1, 0, 0)])
    acq = _acquisition(model=legacy_atom_only_model, strength=1.0)

    breakdown = acq.components(atoms, include_movement=False)

    assert breakdown.calibration_applied is False
    assert breakdown.calibrated_expected_iqa_error_ha_per_sqrt_atom is None
    assert breakdown.energy_risk == pytest.approx(
        math.log1p((4.0 / math.sqrt(len(atoms))) / 2.0)
    )
