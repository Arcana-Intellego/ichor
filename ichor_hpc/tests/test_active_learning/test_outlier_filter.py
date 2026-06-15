import json

import numpy as np
import pytest

from ichor.core.atoms import Atom, Atoms

from ichor.hpc.active_learning.sampling.outlier_filter import (
    OutlierFilterResult,
    filter_by_energy_zscore,
    filter_by_per_atom_rmsd_zscore,
    filter_initial_trajectory,
)


def _water_at(z=0.0, h_offset=0.0):
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0 + z),
        Atom("H", 0.96 + h_offset, 0.0, 0.0 + z),
        Atom("H", -0.24, 0.93, 0.0 + z),
    ])


def _atoms_from_coords(template, coords):
    return Atoms([
        Atom(template[i].type, *coords[i])
        for i in range(len(template))
    ])


def _rotated_translated(atoms, angle_rad, translation):
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    coords = np.asarray(atoms.coordinates, dtype=float) @ rot.T
    coords = coords + np.asarray(translation, dtype=float)[None, :]
    return _atoms_from_coords(atoms, coords)


def test_energy_zscore_keeps_well_behaved_samples():
    rng = np.random.default_rng(0)
    energies = rng.normal(0.0, 1.0, size=100).tolist()
    kept, rejected = filter_by_energy_zscore(energies, z_threshold=3.0)
    assert len(kept) >= 95
    assert len(rejected) <= 5


def test_energy_zscore_rejects_outlier():
    energies = list(np.zeros(50)) + [100.0]
    kept, rejected = filter_by_energy_zscore(energies, z_threshold=3.0)
    assert 50 in rejected
    assert len(rejected) == 1


def test_energy_zscore_handles_zero_variance():
    energies = [1.0, 1.0, 1.0, 1.0]
    kept, rejected = filter_by_energy_zscore(energies)
    assert kept == [0, 1, 2, 3]
    assert rejected == []


def test_energy_zscore_handles_empty():
    kept, rejected = filter_by_energy_zscore([])
    assert kept == []
    assert rejected == []


def test_per_atom_rmsd_zscore_keeps_jittered_frames():
    rng = np.random.default_rng(1)
    base = _water_at(0.0)
    frames = []
    for _ in range(30):
        jittered_coords = np.asarray(base.coordinates) + rng.normal(0, 0.01, size=(3, 3))
        frames.append(Atoms([
            Atom(base[i].type, *jittered_coords[i])
            for i in range(len(base))
        ]))
    kept, rejected = filter_by_per_atom_rmsd_zscore(frames, z_threshold=4.0)
    assert len(rejected) <= 2


def test_per_atom_rmsd_zscore_rejects_atom_flying_away():
    rng = np.random.default_rng(2)
    base = _water_at(0.0)
    frames = []
    for _ in range(20):
        jittered_coords = np.asarray(base.coordinates) + rng.normal(0, 0.01, size=(3, 3))
        frames.append(Atoms([
            Atom(base[i].type, *jittered_coords[i])
            for i in range(len(base))
        ]))
    # Append one frame with H1 flown far away.
    bad_coords = np.asarray(base.coordinates).copy()
    bad_coords[1, 0] += 5.0
    frames.append(Atoms([
        Atom(base[i].type, *bad_coords[i])
        for i in range(len(base))
    ]))
    kept, rejected = filter_by_per_atom_rmsd_zscore(frames, z_threshold=4.0)
    assert (len(frames) - 1) in rejected


def test_per_atom_rmsd_zscore_keeps_rigidly_rotated_frame():
    rng = np.random.default_rng(4)
    base = _water_at(0.0)
    frames = []
    for _ in range(20):
        jittered_coords = np.asarray(base.coordinates) + rng.normal(0, 0.005, size=(3, 3))
        frames.append(_atoms_from_coords(base, jittered_coords))
    frames.append(_rotated_translated(base, np.pi / 2.0, [5.0, -2.0, 1.0]))

    kept, rejected = filter_by_per_atom_rmsd_zscore(frames, z_threshold=4.0)

    assert (len(frames) - 1) in kept
    assert (len(frames) - 1) not in rejected


def test_per_atom_rmsd_zscore_rejects_rotated_frame_with_one_displaced_atom():
    rng = np.random.default_rng(5)
    base = _water_at(0.0)
    frames = []
    for _ in range(20):
        jittered_coords = np.asarray(base.coordinates) + rng.normal(0, 0.005, size=(3, 3))
        frames.append(_atoms_from_coords(base, jittered_coords))
    bad = _rotated_translated(base, np.pi / 2.0, [5.0, -2.0, 1.0])
    bad_coords = np.asarray(bad.coordinates, dtype=float)
    bad_coords[1, 0] += 5.0
    frames.append(_atoms_from_coords(base, bad_coords))

    kept, rejected = filter_by_per_atom_rmsd_zscore(frames, z_threshold=4.0)

    assert (len(frames) - 1) in rejected
    assert (len(frames) - 1) not in kept


def test_filter_initial_trajectory_combines_filters():
    rng = np.random.default_rng(3)
    base = _water_at(0.0)
    frames = []
    energies = []
    for _ in range(20):
        jittered_coords = np.asarray(base.coordinates) + rng.normal(0, 0.01, size=(3, 3))
        frames.append(Atoms([
            Atom(base[i].type, *jittered_coords[i])
            for i in range(len(base))
        ]))
        energies.append(float(rng.normal(0, 1)))
    # Append an energy outlier
    frames.append(_water_at(0.001))
    energies.append(50.0)
    # Append a per-atom RMSD outlier
    bad = np.asarray(_water_at(0.0).coordinates).copy()
    bad[2, 1] += 5.0
    frames.append(Atoms([
        Atom(base[i].type, *bad[i])
        for i in range(len(base))
    ]))
    energies.append(0.0)

    out = filter_initial_trajectory(frames, energies)
    assert out.n_kept + out.n_rejected == len(frames)
    assert 20 in out.rejected_indices
    assert 21 in out.rejected_indices
    reasons_by_index = dict(out.rejections_by_reason)
    assert reasons_by_index.get(20) == "energy_z"
    assert reasons_by_index.get(21) == "per_atom_rmsd_z"


def test_outlier_filter_result_as_json_serialisable():
    out = OutlierFilterResult(
        kept_indices=[0, 1, 2],
        rejected_indices=[3, 4],
        rejections_by_reason=[(3, "energy_z"), (4, "per_atom_rmsd_z")],
        energy_z_threshold=3.0,
        rmsd_z_threshold=4.0,
    )
    payload = out.as_json()
    s = json.dumps(payload)
    assert "rejected" in payload
    assert len(payload["rejected"]) == 2


def test_filter_handles_empty_trajectory():
    out = filter_initial_trajectory([])
    assert out.n_kept == 0
    assert out.n_rejected == 0
