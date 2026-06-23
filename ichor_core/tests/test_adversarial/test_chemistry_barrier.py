import numpy as np

from ichor.core.adversarial.barrier import (
    ChemistryBarrierState,
    chemistry_barrier_components,
    chemistry_barrier_value,
)
from ichor.core.adversarial.config import BarrierConfig
from ichor.core.atoms import Atom, Atoms


def _water_like() -> Atoms:
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])


def _angle_radians(atoms: Atoms, i: int, j: int, k: int) -> float:
    coords = np.asarray(atoms.coordinates, dtype=float)
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    return float(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))


def _barrier_state() -> ChemistryBarrierState:
    atoms = _water_like()
    angle = (1, 0, 2)
    seed_angle = _angle_radians(atoms, *angle)
    return ChemistryBarrierState(
        seed_atoms=atoms,
        bonded_pairs=[(0, 1), (0, 2)],
        nonbonded_pairs=[(1, 2)],
        safe_nonbonded={(1, 2): 0.50},
        nonbonded_upper={(1, 2): 2.50},
        bond_lower={(0, 1): 0.10, (0, 2): 0.10},
        bond_upper={(0, 1): 4.00, (0, 2): 4.00},
        angle_lower={angle: seed_angle + 0.20},
        angle_upper={angle: min(np.pi, seed_angle + 0.40)},
        seed_energy=0.0,
        energy_cap=10.0,
        config=BarrierConfig(softplus_cap=None),
    )


def test_chemistry_barrier_components_sum_to_existing_value():
    atoms = _water_like()
    state = _barrier_state()

    components = chemistry_barrier_components(atoms, state, mean_energy=0.0)
    value = chemistry_barrier_value(atoms, state, mean_energy=0.0)

    assert value == sum(components.values())
    assert components["angle"] > 0.0


def test_chemistry_barrier_can_exclude_only_angle_terms():
    atoms = _water_like()
    state = _barrier_state()

    with_angles = chemistry_barrier_components(
        atoms,
        state,
        mean_energy=0.0,
        include_angles=True,
    )
    without_angles = chemistry_barrier_components(
        atoms,
        state,
        mean_energy=0.0,
        include_angles=False,
    )

    assert without_angles["angle"] == 0.0
    for key in ("clash", "nonbonded_expansion", "bond", "energy_cap"):
        assert without_angles[key] == with_angles[key]
    assert chemistry_barrier_value(
        atoms,
        state,
        mean_energy=0.0,
        include_angles=True,
    ) - chemistry_barrier_value(
        atoms,
        state,
        mean_energy=0.0,
        include_angles=False,
    ) == with_angles["angle"]
