import math

import numpy as np
import pytest

from ichor.core.atoms import Atom, Atoms

from ichor.hpc.active_learning.sampling.descriptors import (
    HybridAlfRmsdDescriptor,
    MassWeightedRMSDDescriptor,
    _normalise_hybrid_feature_matrix,
    kabsch_align,
    mass_weighted_rmsd,
)


def _water_at(z_offset=0.0):
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0 + z_offset),
        Atom("H", 0.96, 0.0, 0.0 + z_offset),
        Atom("H", -0.24, 0.93, 0.0 + z_offset),
    ])


def _rotated_water(angle=0.0):
    a = _water_at(0.0)
    c = np.cos(angle); s = np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotated = (np.asarray(a.coordinates) @ R.T) + np.array([3.0, -2.0, 1.5])
    return Atoms([
        Atom(a[i].type, float(rotated[i, 0]), float(rotated[i, 1]), float(rotated[i, 2]))
        for i in range(len(a))
    ])


def test_kabsch_recovers_identity_for_same_geometry():
    a = _water_at(0.0)
    coords = np.asarray(a.coordinates, dtype=float)
    aligned = kabsch_align(coords, coords)
    np.testing.assert_allclose(aligned, coords, atol=1.0e-10)


def test_mass_weighted_rmsd_zero_after_rotation_and_translation():
    a = _water_at(0.0)
    b = _rotated_water(angle=1.234)
    d = mass_weighted_rmsd(a, b)
    assert d < 1.0e-8, f"expected ~0 RMSD after pure rigid motion, got {d}"


def test_mass_weighted_rmsd_positive_for_distinct_geometries():
    a = _water_at(0.0)
    b = Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 1.10, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])
    d = mass_weighted_rmsd(a, b)
    assert d > 1.0e-3


def test_mass_weighted_descriptor_matrix_is_symmetric_zero_diag():
    frames = [_water_at(0.0), _water_at(0.1), _water_at(-0.2)]
    desc = MassWeightedRMSDDescriptor()
    D = desc.pairwise_distance_matrix(frames)
    assert D.shape == (3, 3)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(D, D.T, atol=1.0e-10)
    # Translation along z is rigid; RMSD should be ~ 0 for all pairs.
    assert np.max(D) < 1.0e-8


def test_hybrid_descriptor_reduces_to_rmsd_when_beta_one():
    frames = [_water_at(0.0), _water_at(0.05), _rotated_water(angle=0.0)]
    pure = MassWeightedRMSDDescriptor().pairwise_distance_matrix(frames)
    hybrid_only_rmsd = HybridAlfRmsdDescriptor(
        beta=1.0, feature_extractor=lambda a: np.zeros(1),
    ).pairwise_distance_matrix(frames)
    np.testing.assert_allclose(hybrid_only_rmsd, pure, atol=1.0e-10)


def test_hybrid_descriptor_default_alf_extractor_is_finite():
    frames = [
        _water_at(0.0),
        Atoms([
            Atom("O", 0.0, 0.0, 0.0),
            Atom("H", 1.05, 0.0, 0.0),
            Atom("H", -0.24, 0.93, 0.0),
        ]),
        Atoms([
            Atom("O", 0.0, 0.0, 0.0),
            Atom("H", 0.96, 0.0, 0.0),
            Atom("H", -0.30, 0.86, 0.0),
        ]),
    ]

    D = HybridAlfRmsdDescriptor(beta=0.3).pairwise_distance_matrix(frames)

    assert D.shape == (3, 3)
    assert np.all(np.isfinite(D))
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(D, D.T, atol=1.0e-10)
    assert np.all(D >= 0.0)
    assert np.max(D) > 0.0


def test_hybrid_descriptor_default_alf_only_mode_does_not_crash():
    frames = [
        _water_at(0.0),
        Atoms([
            Atom("O", 0.0, 0.0, 0.0),
            Atom("H", 1.08, 0.0, 0.0),
            Atom("H", -0.24, 0.90, 0.0),
        ]),
    ]

    D = HybridAlfRmsdDescriptor(beta=0.0).pairwise_distance_matrix(frames)

    assert D.shape == (2, 2)
    assert np.all(np.isfinite(D))
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(D, D.T, atol=1.0e-10)


def test_hybrid_descriptor_default_alf_handles_multi_atom_landing_frames():
    atom_types = ["C", "O", "N", "H", "H", "C", "H", "O", "H", "N", "H", "H"]
    coords = np.array([
        [0.00, 0.00, 0.00],
        [1.22, 0.18, 0.09],
        [-0.42, 1.15, 0.31],
        [0.18, -0.86, 0.72],
        [1.65, -0.42, -0.35],
        [-1.35, 0.22, -0.48],
        [-1.78, 1.04, 0.26],
        [0.74, 1.96, -0.58],
        [1.58, 2.34, 0.18],
        [-0.96, -1.18, -0.64],
        [-1.52, -1.94, 0.08],
        [0.48, -1.72, -1.12],
    ], dtype=float)
    frames = []
    for scale in (0.0, 0.01, -0.015, 0.025):
        shifted = coords.copy()
        shifted[:, 0] += scale * np.arange(len(atom_types), dtype=float)
        shifted[:, 1] -= scale * 0.5
        frames.append(Atoms([
            Atom(atom_type, float(x), float(y), float(z))
            for atom_type, (x, y, z) in zip(atom_types, shifted)
        ]))

    D = HybridAlfRmsdDescriptor(beta=0.3).pairwise_distance_matrix(frames)

    assert D.shape == (4, 4)
    assert np.all(np.isfinite(D))
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(D, D.T, atol=1.0e-10)
    assert np.max(D) > 0.0


def test_hybrid_descriptor_uses_feature_extractor_when_beta_zero():
    frames = [_water_at(0.0), _water_at(0.05), _water_at(0.10)]
    fake_feats = {id(frames[0]): np.array([1.0, 0.0]),
                  id(frames[1]): np.array([0.0, 1.0]),
                  id(frames[2]): np.array([1.0, 1.0])}

    def extractor(a):
        return fake_feats[id(a)]

    hybrid = HybridAlfRmsdDescriptor(beta=0.0, feature_extractor=extractor)
    D = hybrid.pairwise_distance_matrix(frames)
    assert D.shape == (3, 3)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(D, D.T, atol=1.0e-10)


def test_hybrid_descriptor_beta_out_of_range_raises():
    frames = [_water_at(0.0)]
    with pytest.raises(ValueError):
        HybridAlfRmsdDescriptor(beta=1.5).pairwise_distance_matrix(frames)
    with pytest.raises(ValueError):
        HybridAlfRmsdDescriptor(beta=-0.1).pairwise_distance_matrix(frames)


def test_hybrid_descriptor_wraps_bad_alf_geometry_with_phase_b_context():
    frames = [
        _water_at(0.0),
        Atoms([
            Atom("O", 0.0, 0.0, 0.0),
            Atom("H", 0.0, 0.0, 0.0),
            Atom("H", -0.24, 0.93, 0.0),
        ]),
    ]

    with pytest.raises(RuntimeError, match="hybrid_alf_rmsd feature extraction failed"):
        HybridAlfRmsdDescriptor(beta=0.3).pairwise_distance_matrix(frames)


def test_hybrid_descriptor_rejects_feature_length_mismatch():
    frames = [_water_at(0.0), _water_at(0.05)]

    def extractor(atoms):
        if atoms is frames[0]:
            return np.array([1.0, 2.0])
        return np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="feature length mismatch"):
        HybridAlfRmsdDescriptor(
            beta=0.3,
            feature_extractor=extractor,
        ).pairwise_distance_matrix(frames)


def test_cyclic_alf_azimuth_is_continuous_across_pi_boundary():
    epsilon = 1.0e-6
    values = np.asarray([
        [2.0, math.pi - epsilon],
        [2.0, -math.pi + epsilon],
    ])

    encoded = _normalise_hybrid_feature_matrix(
        values,
        np.asarray([False, True]),
        epsilon=1.0e-12,
    )

    assert np.linalg.norm(encoded[0] - encoded[1]) == pytest.approx(
        2.0 * epsilon,
        rel=1.0e-5,
    )
