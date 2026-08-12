from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from ichor.core.atoms import Atom, Atoms
from ichor.core.files.xyz import Trajectory, XYZ


ArrayLikePath = Union[str, Path]


@dataclass(frozen=True)
class Neighbour:
    index: int
    atoms: Atoms
    aligned_distance: float



def load_seed_atoms(seed: Union[ArrayLikePath, XYZ, Atoms]) -> Atoms:
    if isinstance(seed, Atoms):
        return seed.copy()
    if isinstance(seed, XYZ):
        return seed.atoms.copy()
    seed_path = Path(seed)
    xyz = XYZ(seed_path)
    xyz.read()
    return xyz.atoms.copy()



def load_trajectory(trajectory: Union[ArrayLikePath, Trajectory, Sequence[Atoms]]) -> List[Atoms]:
    if isinstance(trajectory, Trajectory):
        return [atoms.copy() for atoms in trajectory]
    if isinstance(trajectory, (list, tuple)) and trajectory and isinstance(trajectory[0], Atoms):
        return [atoms.copy() for atoms in trajectory]
    traj = Trajectory(Path(trajectory))
    traj.read()
    return [atoms.copy() for atoms in traj]



def atoms_to_coordinates(atoms: Atoms) -> np.ndarray:
    return np.asarray(atoms.coordinates, dtype=float)


def _ordered_atom_identity(atoms: Atoms) -> Tuple[Tuple[str, int, object], ...]:
    return tuple((str(atom.type), int(atom.index), atom.units) for atom in atoms)


def _validate_compatible_geometries(reference: Atoms, mobile: Atoms) -> None:
    if not reference or not mobile:
        raise ValueError("aligned geometry comparison requires non-empty molecules")
    if len(reference) != len(mobile):
        raise ValueError(
            "aligned geometry atom-count mismatch: "
            + str(len(reference))
            + " != "
            + str(len(mobile))
        )
    if _ordered_atom_identity(reference) != _ordered_atom_identity(mobile):
        raise ValueError(
            "aligned geometries must have identical ordered atom identities and units"
        )
    ref = atoms_to_coordinates(reference)
    mob = atoms_to_coordinates(mobile)
    if ref.shape != (len(reference), 3) or mob.shape != (len(mobile), 3):
        raise ValueError("aligned geometry coordinates must be shaped (n_atoms, 3)")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(mob)):
        raise ValueError("aligned geometry coordinates must be finite")



def coordinates_to_atoms(template: Atoms, coordinates: np.ndarray) -> Atoms:
    """Build a new Atoms with same atom identities as 'template' but new
    coordinates. Use Atom.from_atom() to preserve per-atom state
    (index, parent, units) that the legacy `Atom(type, x, y, z)` constructor
    silently dropped. Critical for any flow where the perturbed geometry
    feeds back into an ALF / index-aware downstream (the per-mode FD stencils
    in acquisition.py do exactly this on every call).
    """
    coordinates = np.asarray(coordinates, dtype=float)
    expected_shape = (len(template), 3)
    if not template:
        raise ValueError("coordinate reconstruction requires a non-empty template")
    if coordinates.shape != expected_shape:
        raise ValueError(
            "coordinates must have exact shape "
            + repr(expected_shape)
            + ", got "
            + repr(coordinates.shape)
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite")
    new_atoms = Atoms()
    for atom, coord in zip(template, coordinates):
        new_atom = Atom.from_atom(atom)
        # x/y/z are read-only properties; mutate the backing ndarray.
        new_atom.coordinates = np.array(
            [float(coord[0]), float(coord[1]), float(coord[2])], dtype=float,
        )
        new_atoms.add(new_atom)
    return new_atoms



def copy_atoms_with_flat_displacement(template: Atoms, displacement_flat: np.ndarray) -> Atoms:
    displacement = np.asarray(displacement_flat, dtype=float)
    expected_shape = (3 * len(template),)
    if displacement.shape != expected_shape:
        raise ValueError(
            "flat displacement must have exact shape "
            + repr(expected_shape)
            + ", got "
            + repr(displacement.shape)
        )
    if not np.all(np.isfinite(displacement)):
        raise ValueError("flat displacement must be finite")
    base = atoms_to_coordinates(template).reshape(-1)
    if base.shape != expected_shape or not np.all(np.isfinite(base)):
        raise ValueError("template coordinates must be finite and shaped (n_atoms, 3)")
    coords = base + displacement
    return coordinates_to_atoms(template, coords.reshape((-1, 3)))



def mass_vector(atoms: Atoms) -> np.ndarray:
    masses = np.asarray(atoms.masses, dtype=float)
    return np.repeat(masses, 3)



def _weighted_centroid(coords: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.mean(coords, axis=0)
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    return np.sum(w * coords, axis=0) / np.sum(w)



def kabsch_align(reference: np.ndarray, mobile: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return 'mobile' rigidly aligned to 'reference' using a weighted Kabsch fit."""
    ref = np.asarray(reference, dtype=float)
    mob = np.asarray(mobile, dtype=float)
    if ref.shape != mob.shape or ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError("reference and mobile must both be shaped (n_atoms, 3)")
    if ref.shape[0] == 0:
        raise ValueError("reference and mobile must contain at least one atom")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(mob)):
        raise ValueError("reference and mobile coordinates must be finite")
    if weights is not None:
        checked_weights = np.asarray(weights, dtype=float)
        if checked_weights.shape != (ref.shape[0],):
            raise ValueError("Kabsch weights must be shaped (n_atoms,)")
        if (
            not np.all(np.isfinite(checked_weights))
            or np.any(checked_weights <= 0.0)
        ):
            raise ValueError("Kabsch weights must be finite and positive")

    ref_centroid = _weighted_centroid(ref, weights)
    mob_centroid = _weighted_centroid(mob, weights)
    ref0 = ref - ref_centroid
    mob0 = mob - mob_centroid

    if weights is None:
        cov = mob0.T @ ref0
    else:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        cov = (w * mob0).T @ ref0

    u, _, vt = np.linalg.svd(cov, full_matrices=False)
    # Coordinates are row vectors and cov = mobile.T @ reference, so the
    # minimising map is U @ Vt.  V @ U.T is its inverse and makes a rigidly
    # rotated geometry look deformed.
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vt
    aligned = mob0 @ rot + ref_centroid
    return aligned


def _batch_kabsch_align(
    reference: np.ndarray,
    mobile: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Vectorised weighted Kabsch alignment over a geometry batch."""
    ref = np.asarray(reference, dtype=float)
    mobiles = np.asarray(mobile, dtype=float)
    mass = np.asarray(weights, dtype=float)
    if mobiles.ndim != 3 or mobiles.shape[1:] != ref.shape:
        raise ValueError("batched mobile coordinates have an invalid shape")
    if mass.shape != (ref.shape[0],) or np.any(mass <= 0.0):
        raise ValueError("batched Kabsch weights are invalid")
    weight_sum = float(np.sum(mass))
    ref_centroid = np.sum(mass[:, None] * ref, axis=0) / weight_sum
    mobile_centroids = np.sum(mass[None, :, None] * mobiles, axis=1) / weight_sum
    ref0 = ref - ref_centroid
    mobile0 = mobiles - mobile_centroids[:, None, :]
    covariance = np.einsum(
        "bni,nj,n->bij",
        mobile0,
        ref0,
        mass,
        optimize=True,
    )
    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    rotations = u @ vt
    reflected = np.linalg.det(rotations) < 0.0
    if np.any(reflected):
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        rotations = u @ vt
    return np.einsum(
        "bni,bij->bnj",
        mobile0,
        rotations,
        optimize=True,
    ) + ref_centroid


def _select_pool_neighbours_vectorised(
    seed_atoms: Atoms,
    trajectory,
    max_neighbours: int,
    deduplicate_rmsd: float,
    progress_callback: Optional[Callable[[int, int], None]],
) -> List[Neighbour]:
    coordinates = np.asarray(trajectory.coordinates_view(), dtype=np.float64)
    frame_ids = tuple(int(value) for value in trajectory.frame_ids())
    if coordinates.shape != (len(frame_ids), len(seed_atoms), 3):
        raise ValueError("trajectory coordinate view does not match the seed")
    if frame_ids != tuple(range(len(frame_ids))):
        raise ValueError("trajectory coordinate view requires canonical frame ordering")
    manifest = getattr(trajectory, "manifest", None)
    if manifest is not None:
        if tuple(str(atom.type) for atom in seed_atoms) != tuple(manifest.atom_types):
            raise ValueError("trajectory atom identities do not match the seed")
    reference = atoms_to_coordinates(seed_atoms)
    masses = np.asarray(seed_atoms.masses, dtype=float)
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("trajectory alignment masses must be finite and positive")

    aligned = np.empty_like(coordinates)
    distances = np.empty(len(frame_ids), dtype=float)
    chunk_size = 1024
    for start in range(0, len(frame_ids), chunk_size):
        stop = min(len(frame_ids), start + chunk_size)
        block = _batch_kabsch_align(reference, coordinates[start:stop], masses)
        aligned[start:stop] = block
        delta = block - reference[None, :, :]
        distances[start:stop] = np.sqrt(
            np.sum(masses[None, :, None] * delta * delta, axis=(1, 2))
        )
        if progress_callback is not None:
            progress_callback(int(stop), int(len(frame_ids)))
    order = np.argsort(distances, kind="stable")

    # Refine the selection boundary with the scalar implementation. This keeps
    # exact historical ordering for near ties while retaining batched work for
    # the full pool.
    refine_count = min(len(order), max(64, int(max_neighbours) * 4))
    refined = []
    for frame_id in order[:refine_count]:
        frame = trajectory.frame(int(frame_id))
        exact_aligned = kabsch_align(
            reference,
            atoms_to_coordinates(frame),
            weights=masses,
        )
        aligned[int(frame_id)] = exact_aligned
        exact_delta = exact_aligned - reference
        exact_distance = float(
            np.sqrt(np.sum(masses[:, None] * exact_delta * exact_delta))
        )
        distances[int(frame_id)] = exact_distance
        refined.append(
            (
                exact_distance,
                int(frame_id),
            )
        )
    refined.sort(key=lambda item: (item[0], item[1]))
    ordered_ids = [frame_id for _, frame_id in refined]
    ordered_ids.extend(int(value) for value in order[refine_count:])

    selected: List[Neighbour] = []
    selected_coordinates: List[np.ndarray] = []
    for frame_id in ordered_ids:
        candidate = aligned[frame_id]
        if any(
            float(np.sqrt(np.mean(np.sum((candidate - prior) ** 2, axis=1))))
            < deduplicate_rmsd
            for prior in selected_coordinates
        ):
            continue
        selected.append(
            Neighbour(
                index=int(frame_id),
                atoms=trajectory.frame(int(frame_id)),
                aligned_distance=float(distances[frame_id]),
            )
        )
        selected_coordinates.append(candidate)
        if len(selected) >= int(max_neighbours):
            break
    return selected



def aligned_mass_weighted_distance(reference: Atoms, mobile: Atoms) -> float:
    _validate_compatible_geometries(reference, mobile)
    ref = atoms_to_coordinates(reference)
    mob = atoms_to_coordinates(mobile)
    masses = np.asarray(reference.masses, dtype=float)
    aligned = kabsch_align(ref, mob, weights=masses)
    diff = aligned - ref
    m = mass_vector(reference)
    return float(np.sqrt(np.dot(m, diff.reshape(-1) ** 2)))


def _aligned_mass_weighted_rmsd_arrays(
    reference: np.ndarray,
    mobile: np.ndarray,
    masses: np.ndarray,
) -> float:
    """Exact scalar RMSD kernel shared by object and cached-coordinate paths."""
    ref = np.asarray(reference, dtype=float)
    mob = np.asarray(mobile, dtype=float)
    mass = np.asarray(masses, dtype=float)
    if ref.shape != mob.shape or ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError("aligned geometry coordinate arrays must match (n_atoms, 3)")
    if mass.shape != (ref.shape[0],):
        raise ValueError("aligned geometry masses must be shaped (n_atoms,)")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(mob)):
        raise ValueError("aligned geometry coordinates must be finite")
    aligned = kabsch_align(ref, mob, weights=mass)
    diff = aligned - ref
    denom = float(np.sum(mass))
    if denom <= 0.0 or not np.isfinite(denom):
        denom = float(ref.shape[0])
    if denom <= 0.0:
        return 0.0
    weighted_sq = float(np.sum(mass[:, None] * diff ** 2))
    return float(np.sqrt(max(0.0, weighted_sq / denom)))


def aligned_mass_weighted_rmsd(reference: Atoms, mobile: Atoms) -> float:
    """Return aligned mass-normalised RMSD in coordinate units.

    This is the RMSD-like version operators expect for duplicate filtering: the squared
    displacement is mass weighted, then normalised by the total molecular mass instead of growing
    with molecule size.
    """
    _validate_compatible_geometries(reference, mobile)
    ref = atoms_to_coordinates(reference)
    mob = atoms_to_coordinates(mobile)
    masses = np.asarray(reference.masses, dtype=float)
    return _aligned_mass_weighted_rmsd_arrays(ref, mob, masses)


def aligned_per_atom_displacements(reference: Atoms, mobile: Atoms) -> np.ndarray:
    """Return per-atom Euclidean displacements after mass-weighted alignment."""
    _validate_compatible_geometries(reference, mobile)
    ref = atoms_to_coordinates(reference)
    mob = atoms_to_coordinates(mobile)
    masses = np.asarray(reference.masses, dtype=float)
    aligned = kabsch_align(ref, mob, weights=masses)
    return np.linalg.norm(aligned - ref, axis=1)



def aligned_mass_weighted_displacement(reference: Atoms, mobile: Atoms) -> np.ndarray:
    _validate_compatible_geometries(reference, mobile)
    ref = atoms_to_coordinates(reference)
    mob = atoms_to_coordinates(mobile)
    masses = np.asarray(reference.masses, dtype=float)
    aligned = kabsch_align(ref, mob, weights=masses)
    diff = aligned - ref
    return np.sqrt(mass_vector(reference)) * diff.reshape(-1)



def select_local_neighbours(
    seed_atoms: Atoms,
    trajectory,
    max_neighbours: int,
    deduplicate_rmsd: float = 1.0e-3,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Neighbour]:
    """Select a local neighbourhood around the seed from a trajectory.

    Neighbours are ranked by aligned mass-weighted distance. Near-duplicate
    frames are thinned by an aligned RMSD threshold to avoid covariance
    estimates dominated by temporally adjacent copies of the same structure.

    'trajectory' may be either:

    * a plain 'Sequence[Atoms]' -- the historical contract. The
      'Neighbour.index' returned is simply the enumeration position
      within that sequence (transient, not stable across sessions).
    * any object that exposes 'frame(frame_id)' + 'frame_ids()'
      duck-typed methods (e.g. 'ichor.hpc.active_learning.acquisition.
      trajectory_pool.TrajectoryPool'). When given such an object, the
      returned 'Neighbour.index' is the **stable frame_id** from the
      pool, suitable for persistence in provenance ledgers / journal
      events.

    The duck-typing keeps 'ichor.core' free of an 'ichor.hpc' import.
    """
    if hasattr(trajectory, "coordinates_view"):
        return _select_pool_neighbours_vectorised(
            seed_atoms,
            trajectory,
            max_neighbours,
            deduplicate_rmsd,
            progress_callback,
        )
    #Build the (frame_id, atoms) pair generator from whichever form was passed.
    if hasattr(trajectory, "frame") and hasattr(trajectory, "frame_ids"):
        frame_ids = trajectory.frame_ids()
        pairs = ((int(fid), trajectory.frame(fid)) for fid in frame_ids)
        try:
            total_frames = int(len(frame_ids))
        except TypeError:
            total_frames = 0
    else:
        pairs = ((int(idx), atoms) for idx, atoms in enumerate(trajectory))
        total_frames = int(len(trajectory))

    ranked: List[Neighbour] = []
    for position, (frame_id, atoms) in enumerate(pairs, start=1):
        ranked.append(
            Neighbour(
                index=frame_id,
                atoms=atoms.copy(),
                aligned_distance=aligned_mass_weighted_distance(seed_atoms, atoms),
            )
        )
        if progress_callback is not None and (
            (total_frames > 0 and position == total_frames) or position % 256 == 0
        ):
            progress_callback(int(position), int(total_frames))
    ranked.sort(key=lambda item: item.aligned_distance)

    selected: List[Neighbour] = []
    selected_coords: List[np.ndarray] = []
    ref_coords = atoms_to_coordinates(seed_atoms)
    masses = np.asarray(seed_atoms.masses, dtype=float)
    for item in ranked:
        aligned = kabsch_align(ref_coords, atoms_to_coordinates(item.atoms), weights=masses)
        if selected_coords:
            is_duplicate = False
            for coords in selected_coords:
                rmsd = np.sqrt(np.mean(np.sum((aligned - coords) ** 2, axis=1)))
                if rmsd < deduplicate_rmsd:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
        selected.append(item)
        selected_coords.append(aligned)
        if len(selected) >= max_neighbours:
            break
    return selected
