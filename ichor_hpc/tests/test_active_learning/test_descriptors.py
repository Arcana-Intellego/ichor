import math

import numpy as np
import pytest

from ichor.core.atoms import Atom, Atoms

from ichor.hpc.active_learning.sampling.descriptors import (
    AcquisitionWeightedDescriptor,
    HybridAlfRmsdDescriptor,
    MassWeightedRMSDDescriptor,
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


def test_acquisition_weighted_descriptor_scales_distances_by_variance():
    frames = [_water_at(0.0), _water_at(0.10), _water_at(0.20)]

    class _StubPosterior:
        def __init__(self, vs):
            self._vs = vs

        def variance(self, atoms):
            return self._vs[id(atoms)]

    base_desc = MassWeightedRMSDDescriptor()
    base = base_desc.pairwise_distance_matrix(frames)

    high_low_low = _StubPosterior({id(frames[0]): 100.0, id(frames[1]): 1.0, id(frames[2]): 1.0})
    acq = AcquisitionWeightedDescriptor(
        posterior=high_low_low,
        base_descriptor=base_desc,
        sigma_ref=1.0,
    )
    D = acq.pairwise_distance_matrix(frames)
    np.testing.assert_allclose(D, D.T, atol=1.0e-12)
    np.testing.assert_allclose(np.diag(D), np.diag(base) * 10.0, atol=1.0e-12)
    assert D[0, 1] > base[0, 1]
    assert D[1, 2] == pytest.approx(base[1, 2], rel=1.0e-12)


def test_acquisition_weighted_requires_posterior():
    with pytest.raises(ValueError):
        AcquisitionWeightedDescriptor().pairwise_distance_matrix([_water_at(0.0)])
