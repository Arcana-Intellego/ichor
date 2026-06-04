from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from ichor.core.atoms import Atoms
from ichor.core.calculators.connectivity import default_connectivity_calculator

from .config import BarrierConfig
from .posterior import TotalEnergyPosterior


Pair = Tuple[int, int]


@dataclass(frozen=True)
class ChemistryBarrierState:
    seed_atoms: Atoms
    bonded_pairs: List[Pair]
    nonbonded_pairs: List[Pair]
    safe_nonbonded: Dict[Pair, float]
    bond_lower: Dict[Pair, float]
    bond_upper: Dict[Pair, float]
    seed_energy: float
    energy_cap: float
    config: BarrierConfig



def _softplus(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)



def _softplus_sq(u: float, delta: float, cap: float | None = None) -> float:
    """Squared softplus barrier.
    When 'cap' is provided, the argument 'u / delta' is clamped to at most
    'cap' before evaluating softplus. This bounds the per-term contribution
    near 'softplus(cap) ** 2' and prevents the chemistry barrier from
    emitting unbounded gradients when the optimiser briefly excursions into a
    deep clash region.
    """
    delta = max(float(delta), 1.0e-12)
    arg = u / delta
    if cap is not None:
        arg = min(arg, float(cap))
    value = _softplus(arg)
    return float(value * value)



def _pair_distance(atoms: Atoms, i: int, j: int) -> float:
    coords = np.asarray(atoms.coordinates, dtype=float)
    diff = coords[i] - coords[j]
    return float(np.linalg.norm(diff))



def build_chemistry_barrier_state(
    seed_atoms: Atoms,
    neighbours: Sequence[Atoms],
    posterior: TotalEnergyPosterior,
    config: BarrierConfig,
) -> ChemistryBarrierState:
    connectivity = default_connectivity_calculator(seed_atoms)
    bonded_pairs: List[Pair] = []
    nonbonded_pairs: List[Pair] = []
    natoms = len(seed_atoms)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            if connectivity[i, j]:
                bonded_pairs.append((i, j))
            else:
                nonbonded_pairs.append((i, j))

    safe_nonbonded: Dict[Pair, float] = {}
    for i, j in nonbonded_pairs:
        safe_nonbonded[(i, j)] = config.nonbonded_clash_scale * (seed_atoms[i].radius + seed_atoms[j].radius)

    bond_lower: Dict[Pair, float] = {}
    bond_upper: Dict[Pair, float] = {}
    for i, j in bonded_pairs:
        seed_dist = _pair_distance(seed_atoms, i, j)
        neighbour_dists = np.array([_pair_distance(atoms, i, j) for atoms in neighbours], dtype=float) if neighbours else np.array([seed_dist], dtype=float)
        q05 = float(np.quantile(neighbour_dists, 0.05))
        q95 = float(np.quantile(neighbour_dists, 0.95))
        bond_lower[(i, j)] = min(config.bond_lower_scale * seed_dist, q05)
        bond_upper[(i, j)] = max(config.bond_upper_scale * seed_dist, q95)

    seed_energy = posterior.mean(seed_atoms)
    if neighbours:
        neighbour_energies = np.array([posterior.mean(atoms) for atoms in neighbours], dtype=float)
        delta_e = neighbour_energies - seed_energy
        energy_cap = max(config.energy_cap_floor, float(np.quantile(delta_e, config.energy_cap_quantile)))
    else:
        energy_cap = config.energy_cap_floor

    return ChemistryBarrierState(
        seed_atoms=seed_atoms.copy(),
        bonded_pairs=bonded_pairs,
        nonbonded_pairs=nonbonded_pairs,
        safe_nonbonded=safe_nonbonded,
        bond_lower=bond_lower,
        bond_upper=bond_upper,
        seed_energy=seed_energy,
        energy_cap=energy_cap,
        config=config,
    )



def chemistry_barrier_value(
    atoms: Atoms,
    barrier_state: ChemistryBarrierState,
    mean_energy: float,
) -> float:
    cfg = barrier_state.config
    total = 0.0

    for pair, safe_distance in barrier_state.safe_nonbonded.items():
        dist = _pair_distance(atoms, *pair)
        total += cfg.clash_lambda * _softplus_sq(safe_distance - dist, cfg.clash_delta, cap=cfg.softplus_cap)

    if cfg.use_connectivity_barrier:
        for pair in barrier_state.bonded_pairs:
            dist = _pair_distance(atoms, *pair)
            total += cfg.bond_lambda * _softplus_sq(barrier_state.bond_lower[pair] - dist, cfg.bond_delta, cap=cfg.softplus_cap)
            total += cfg.bond_lambda * _softplus_sq(dist - barrier_state.bond_upper[pair], cfg.bond_delta, cap=cfg.softplus_cap)

    total += cfg.energy_cap_lambda * _softplus_sq(
        mean_energy - barrier_state.seed_energy - barrier_state.energy_cap,
        cfg.energy_cap_delta,
        cap=cfg.softplus_cap,
    )
    return float(total)
