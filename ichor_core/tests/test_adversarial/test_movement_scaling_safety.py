from types import SimpleNamespace

import numpy as np

from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
from ichor.core.adversarial.config import (
    AcquisitionConfig,
    MovementBandConfig,
    MovementUtilityConfig,
    SizeNormalisationConfig,
    SubspaceConfig,
)
from ichor.core.atoms import Atom, Atoms


def _water_like_atoms(scale: float = 1.0) -> Atoms:
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96 * scale, 0.0, 0.0),
        Atom("H", -0.24 * scale, 0.93 * scale, 0.0),
    ])


def test_movement_band_falls_back_to_ordered_absolute_band_for_large_local_rmsd():
    acq = object.__new__(SeedLocalAdversarialAcquisition)
    acq._movement_band_cache = None
    seed = _water_like_atoms()
    neighbour = SimpleNamespace(atoms=_water_like_atoms(scale=100.0))
    acq.seed_atoms = seed
    acq.subspace = SimpleNamespace(
        seed_atoms=seed,
        neighbours=[neighbour],
        basis=np.eye(9, 1),
    )
    acq.config = AcquisitionConfig(
        movement_band=MovementBandConfig(),
    )

    band = acq.movement_band()

    assert band["min"] < band["low"] < band["peak"] < band["high"] < band["max"]
    assert band["high"] <= acq.config.movement_band.target_high_cap_ang
    assert band["max"] <= acq.config.movement_band.hard_max_cap_ang


def test_initial_projected_acquisition_gradient_is_projected_to_active_subspace():
    acq = object.__new__(SeedLocalAdversarialAcquisition)
    acq._movement_direction_cache = None
    seed = _water_like_atoms()
    active = np.zeros(9)
    active[0] = 1.0
    inactive = np.zeros(9)
    inactive[1] = 1.0
    acq.seed_atoms = seed
    acq.mode_directions = [active]
    acq.config = AcquisitionConfig(
        subspace=SubspaceConfig(covariance_regularization=1.0e-12),
        movement_utility=MovementUtilityConfig(
            direction="initial_projected_acquisition_gradient",
        ),
        size_normalisation=SizeNormalisationConfig(),
    )
    acq._base_cartesian_gradient = lambda atoms: (active + 10.0 * inactive).reshape(3, 3)

    direction, source = acq.movement_direction()

    assert source == "initial_projected_acquisition_gradient"
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert abs(direction[0]) > 0.999
    assert abs(direction[1]) < 1.0e-8
