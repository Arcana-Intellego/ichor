from types import SimpleNamespace

import numpy as np
import pytest

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


def test_geometry_novelty_scale_controls_movement_band_fractions():
    acq = object.__new__(SeedLocalAdversarialAcquisition)
    acq._movement_band_cache = None
    seed = _water_like_atoms()
    acq.seed_atoms = seed
    acq.subspace = SimpleNamespace(
        seed_atoms=seed,
        neighbours=[],
        basis=np.eye(9, 1),
    )
    acq.config = AcquisitionConfig(
        movement_band=MovementBandConfig(
            hard_min_fraction=0.10,
            target_low_fraction=0.25,
            target_peak_fraction=0.40,
            target_high_fraction=0.75,
            hard_max_fraction=1.25,
            geometry_novelty_scale_angstrom=0.04,
        ),
    )

    band = acq.movement_band()

    assert band["scale_source"] == "geometry_novelty"
    assert band["geometry_novelty_scale_angstrom"] == 0.04
    assert band["min"] == pytest.approx(0.004)
    assert band["low"] == pytest.approx(0.010)
    assert band["peak"] == pytest.approx(0.016)
    assert band["high"] == pytest.approx(0.030)
    assert band["max"] == pytest.approx(0.050)


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
    acq._base_active_gradient = lambda atoms: (
        active + 10.0 * inactive
    ).reshape(3, 3)

    direction, source = acq.movement_direction()

    assert source == "initial_projected_acquisition_gradient"
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert abs(direction[0]) > 0.999
    assert abs(direction[1]) < 1.0e-8


def _vector_movement_metrics(
    n_atoms: int,
    *,
    progress_normalisation: str,
    spectator_atoms: int = 0,
):
    total_atoms = n_atoms + spectator_atoms
    acq = object.__new__(SeedLocalAdversarialAcquisition)
    acq.seed_atoms = Atoms(
        [Atom("H", float(index), 0.0, 0.0) for index in range(total_atoms)]
    )
    basis = np.zeros((3 * total_atoms, 1), dtype=float)
    basis[0, 0] = 1.0
    acq.subspace = SimpleNamespace(seed_atoms=acq.seed_atoms, basis=basis)
    acq.config = AcquisitionConfig(
        movement_band=MovementBandConfig(metric="aligned_active_rmsd"),
        movement_utility=MovementUtilityConfig(
            progress_normalisation=progress_normalisation,
            low_softness_ang=0.01,
            high_softness_ang=0.01,
        ),
    )
    delta = np.zeros(3 * total_atoms, dtype=float)
    direction = np.zeros(3 * total_atoms, dtype=float)
    weights = np.zeros(3 * total_atoms, dtype=float)
    for index in range(n_atoms):
        delta[3 * index] = 0.1
        direction[3 * index] = 1.0
        weights[3 * index : 3 * index + 3] = 1.0
    for index in range(n_atoms, total_atoms):
        delta[3 * index] = 100.0
        direction[3 * index] = 1.0
    acq._movement_delta_and_weights = lambda atoms: (delta, weights)
    acq.movement_direction = lambda: (direction, "test_direction")
    acq.movement_distance = lambda atoms: (0.1, "aligned_active_rmsd")
    acq.movement_band = lambda: {
        "min": 0.01,
        "low": 0.05,
        "peak": 0.10,
        "high": 0.15,
        "max": 0.30,
        "scale_source": "test",
        "geometry_novelty_scale_angstrom": 0.10,
    }
    return acq.movement_metrics(acq.seed_atoms)


def test_active_weight_rmsd_progress_is_invariant_to_active_system_size():
    small = _vector_movement_metrics(
        1,
        progress_normalisation="active_weight_rmsd",
    )
    large = _vector_movement_metrics(
        30,
        progress_normalisation="active_weight_rmsd",
    )

    assert small["movement_progress_ang"] == pytest.approx(0.1)
    assert large["movement_progress_ang"] == pytest.approx(0.1)
    assert large["movement_progress_score"] == pytest.approx(
        small["movement_progress_score"]
    )
    assert large["movement_utility_score"] == pytest.approx(
        small["movement_utility_score"]
    )


def test_active_weight_rmsd_progress_ignores_inactive_spectators():
    base = _vector_movement_metrics(
        2,
        progress_normalisation="active_weight_rmsd",
    )
    spectators = _vector_movement_metrics(
        2,
        spectator_atoms=20,
        progress_normalisation="active_weight_rmsd",
    )

    assert spectators["movement_progress_ang"] == pytest.approx(
        base["movement_progress_ang"]
    )
    assert spectators["movement_utility_score"] == pytest.approx(
        base["movement_utility_score"]
    )


def test_legacy_projected_progress_retains_extensive_behaviour():
    small = _vector_movement_metrics(
        1,
        progress_normalisation="legacy_projected_displacement",
    )
    large = _vector_movement_metrics(
        25,
        progress_normalisation="legacy_projected_displacement",
    )

    assert small["movement_progress_ang"] == pytest.approx(0.1)
    assert large["movement_progress_ang"] == pytest.approx(0.5)


def test_global_v3_progress_uses_atomic_mass_weights():
    acq = object.__new__(SeedLocalAdversarialAcquisition)
    acq.seed_atoms = _water_like_atoms()
    moved = _water_like_atoms()
    moved[1].coordinates[0] += 0.1
    acq.config = AcquisitionConfig(
        movement_band=MovementBandConfig(metric="aligned_global_rmsd"),
        movement_utility=MovementUtilityConfig(
            progress_normalisation="active_weight_rmsd"
        ),
    )

    _, weights = acq._movement_delta_and_weights(moved)

    assert weights == pytest.approx(np.repeat(acq.seed_atoms.masses, 3))


def test_unknown_progress_normalisation_fails_closed():
    with pytest.raises(ValueError, match="unsupported movement progress"):
        _vector_movement_metrics(1, progress_normalisation="unknown")
