from __future__ import annotations

import numpy as np
import pytest

from ichor.core.adversarial.geometry import (
    aligned_mass_weighted_displacement,
    aligned_mass_weighted_distance,
    aligned_mass_weighted_rmsd,
    coordinates_to_atoms,
    copy_atoms_with_flat_displacement,
)
from ichor.core.atoms import Atom, Atoms
from ichor.core.common.units import AtomicDistance


def _water(*, units=AtomicDistance.Angstroms) -> Atoms:
    return Atoms(
        [
            Atom("O", 0.0, 0.0, 0.0, units=units),
            Atom("H", 0.96, 0.0, 0.0, units=units),
            Atom("H", -0.24, 0.93, 0.0, units=units),
        ]
    )


def test_atoms_advanced_integer_and_boolean_indexing_is_numpy_safe():
    atoms = _water()
    integer_subset = atoms[np.asarray([0, 2], dtype=np.int64)]
    boolean_subset = atoms[np.asarray([True, False, True], dtype=np.bool_)]

    assert [atom.name for atom in integer_subset] == ["O1", "H3"]
    assert [atom.name for atom in boolean_subset] == ["O1", "H3"]
    integer_subset[0].coordinates[0] = 99.0
    assert atoms[0].x == 0.0
    assert atoms[0].parent is atoms


def test_atoms_boolean_mask_requires_exact_length():
    with pytest.raises(IndexError, match="mask length"):
        _water()[[True, False]]


def test_atoms_advanced_indexing_rejects_mixed_key_kinds():
    with pytest.raises(TypeError, match="subset indices"):
        _water()[[0, False]]


@pytest.mark.parametrize(
    "metric",
    [
        aligned_mass_weighted_distance,
        aligned_mass_weighted_rmsd,
        aligned_mass_weighted_displacement,
    ],
)
def test_aligned_metrics_reject_reordered_atom_identity(metric):
    reference = _water()
    reordered = Atoms(
        [
            Atom("H", 0.0, 0.0, 0.0),
            Atom("O", 0.96, 0.0, 0.0),
            Atom("H", -0.24, 0.93, 0.0),
        ]
    )
    with pytest.raises(ValueError, match="ordered atom identities"):
        metric(reference, reordered)


def test_aligned_metrics_reject_count_units_and_non_finite_coordinates():
    reference = _water()
    with pytest.raises(ValueError, match="atom-count mismatch"):
        aligned_mass_weighted_rmsd(
            reference,
            Atoms([Atom.from_atom(atom) for atom in reference[:2]]),
        )
    with pytest.raises(ValueError, match="identities and units"):
        aligned_mass_weighted_rmsd(
            reference, _water(units=AtomicDistance.Bohr)
        )
    invalid = _water()
    invalid[0].coordinates[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        aligned_mass_weighted_rmsd(reference, invalid)


@pytest.mark.parametrize(
    "coordinates",
    [
        np.zeros((2, 3)),
        np.zeros((4, 3)),
        np.zeros((3, 2)),
        np.asarray([[0.0, 0.0, np.inf]] * 3),
    ],
)
def test_coordinate_reconstruction_requires_exact_finite_cardinality(coordinates):
    with pytest.raises(ValueError):
        coordinates_to_atoms(_water(), coordinates)


def test_flat_displacement_requires_exact_finite_3n_shape():
    template = _water()
    with pytest.raises(ValueError, match="exact shape"):
        copy_atoms_with_flat_displacement(template, np.zeros(8))
    displacement = np.zeros(9)
    displacement[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        copy_atoms_with_flat_displacement(template, displacement)
