from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from ichor.core.atoms import Atoms
from ichor.core.calculators.connectivity import default_connectivity_calculator

from .config import BarrierConfig
from .posterior import TotalEnergyPosterior


Pair = Tuple[int, int]
Angle = Tuple[int, int, int]


@dataclass(frozen=True)
class ChemistryBarrierState:
    seed_atoms: Atoms
    bonded_pairs: List[Pair]
    nonbonded_pairs: List[Pair]
    safe_nonbonded: Dict[Pair, float]
    nonbonded_upper: Dict[Pair, float]
    bond_lower: Dict[Pair, float]
    bond_upper: Dict[Pair, float]
    angle_lower: Dict[Angle, float]
    angle_upper: Dict[Angle, float]
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


def _angle_radians(atoms: Atoms, i: int, j: int, k: int) -> float:
    coords = np.asarray(atoms.coordinates, dtype=float)
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 0.0 or n2 <= 0.0:
        return float("nan")
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))



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
    nonbonded_upper: Dict[Pair, float] = {}
    for i, j in nonbonded_pairs:
        radius_sum = seed_atoms[i].radius + seed_atoms[j].radius
        safe_nonbonded[(i, j)] = config.nonbonded_clash_scale * radius_sum
        seed_dist = _pair_distance(seed_atoms, i, j)
        expansion_margin = max(float(config.nonbonded_expansion_delta), 0.0)
        if seed_dist <= (config.nonbonded_expansion_scale + expansion_margin) * radius_sum:
            neighbour_dists = (
                np.array([_pair_distance(atoms, i, j) for atoms in neighbours], dtype=float)
                if neighbours else np.array([seed_dist], dtype=float)
            )
            finite_dists = neighbour_dists[np.isfinite(neighbour_dists)]
            q95 = float(np.quantile(finite_dists, 0.95)) if finite_dists.size else seed_dist
            nonbonded_upper[(i, j)] = max(config.nonbonded_expansion_scale * seed_dist, q95)

    bond_lower: Dict[Pair, float] = {}
    bond_upper: Dict[Pair, float] = {}
    for i, j in bonded_pairs:
        seed_dist = _pair_distance(seed_atoms, i, j)
        neighbour_dists = np.array([_pair_distance(atoms, i, j) for atoms in neighbours], dtype=float) if neighbours else np.array([seed_dist], dtype=float)
        q05 = float(np.quantile(neighbour_dists, 0.05))
        q95 = float(np.quantile(neighbour_dists, 0.95))
        bond_lower[(i, j)] = min(config.bond_lower_scale * seed_dist, q05)
        bond_upper[(i, j)] = max(config.bond_upper_scale * seed_dist, q95)

    neighbours_by_centre: Dict[int, List[int]] = {i: [] for i in range(natoms)}
    for i, j in bonded_pairs:
        neighbours_by_centre[i].append(j)
        neighbours_by_centre[j].append(i)
    angle_lower: Dict[Angle, float] = {}
    angle_upper: Dict[Angle, float] = {}
    for centre, bonded_neighbours in neighbours_by_centre.items():
        ordered = sorted(bonded_neighbours)
        for a in range(len(ordered)):
            for b in range(a + 1, len(ordered)):
                i = ordered[a]
                k = ordered[b]
                seed_angle = _angle_radians(seed_atoms, i, centre, k)
                if not np.isfinite(seed_angle):
                    continue
                neighbour_angles = (
                    np.array(
                        [_angle_radians(atoms, i, centre, k) for atoms in neighbours],
                        dtype=float,
                    )
                    if neighbours else np.array([seed_angle], dtype=float)
                )
                finite_angles = neighbour_angles[np.isfinite(neighbour_angles)]
                if finite_angles.size:
                    q05 = float(np.quantile(finite_angles, 0.05))
                    q95 = float(np.quantile(finite_angles, 0.95))
                else:
                    q05 = q95 = seed_angle
                angle = (i, centre, k)
                angle_lower[angle] = max(0.0, min(config.angle_lower_scale * seed_angle, q05))
                angle_upper[angle] = min(np.pi, max(config.angle_upper_scale * seed_angle, q95))

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
        nonbonded_upper=nonbonded_upper,
        bond_lower=bond_lower,
        bond_upper=bond_upper,
        angle_lower=angle_lower,
        angle_upper=angle_upper,
        seed_energy=seed_energy,
        energy_cap=energy_cap,
        config=config,
    )



def chemistry_barrier_value(
    atoms: Atoms,
    barrier_state: ChemistryBarrierState,
    mean_energy: float,
    *,
    normalisation_mode: str = "raw_sum",
) -> float:
    cfg = barrier_state.config
    family_mean = str(normalisation_mode or "raw_sum").lower()
    family_mean = family_mean == "family_mean"
    clash_terms = []
    expansion_terms = []
    bond_terms = []
    angle_terms = []

    for pair, safe_distance in barrier_state.safe_nonbonded.items():
        dist = _pair_distance(atoms, *pair)
        clash_terms.append(
            cfg.clash_lambda
            * _softplus_sq(safe_distance - dist, cfg.clash_delta, cap=cfg.softplus_cap)
        )

    for pair, upper_distance in barrier_state.nonbonded_upper.items():
        dist = _pair_distance(atoms, *pair)
        expansion_terms.append(
            cfg.nonbonded_expansion_lambda
            * _softplus_sq(
                dist - upper_distance,
                cfg.nonbonded_expansion_delta,
                cap=cfg.softplus_cap,
            )
        )

    if cfg.use_connectivity_barrier:
        for pair in barrier_state.bonded_pairs:
            dist = _pair_distance(atoms, *pair)
            bond_terms.append(
                cfg.bond_lambda
                * _softplus_sq(
                    barrier_state.bond_lower[pair] - dist,
                    cfg.bond_delta,
                    cap=cfg.softplus_cap,
                )
            )
            bond_terms.append(
                cfg.bond_lambda
                * _softplus_sq(
                    dist - barrier_state.bond_upper[pair],
                    cfg.bond_delta,
                    cap=cfg.softplus_cap,
                )
            )
        for angle in barrier_state.angle_lower:
            value = _angle_radians(atoms, *angle)
            if not np.isfinite(value):
                angle_terms.append(
                    cfg.angle_lambda
                    * _softplus_sq(
                        np.pi,
                        cfg.angle_delta,
                        cap=cfg.softplus_cap,
                    )
                )
                continue
            angle_terms.append(
                cfg.angle_lambda
                * _softplus_sq(
                    barrier_state.angle_lower[angle] - value,
                    cfg.angle_delta,
                    cap=cfg.softplus_cap,
                )
            )
            angle_terms.append(
                cfg.angle_lambda
                * _softplus_sq(
                    value - barrier_state.angle_upper[angle],
                    cfg.angle_delta,
                    cap=cfg.softplus_cap,
                )
            )

    energy_term = cfg.energy_cap_lambda * _softplus_sq(
        mean_energy - barrier_state.seed_energy - barrier_state.energy_cap,
        cfg.energy_cap_delta,
        cap=cfg.softplus_cap,
    )
    if not family_mean:
        total = (
            float(np.sum(clash_terms))
            + float(np.sum(expansion_terms))
            + float(np.sum(bond_terms))
            + float(np.sum(angle_terms))
            + float(energy_term)
        )
        return float(total)

    def _mean(values) -> float:
        if not values:
            return 0.0
        return float(np.mean(np.asarray(values, dtype=float)))

    total = (
        _mean(clash_terms)
        + _mean(expansion_terms)
        + _mean(bond_terms)
        + _mean(angle_terms)
        + float(energy_term)
    )
    return float(total)
