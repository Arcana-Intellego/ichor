"""Coordinates_to_atoms must preserve per-atom state beyond just
type+coords. The legacy implementation reconstructed Atom(type, x, y, z)
which dropped index, parent, units silently."""
import numpy as np
import pytest

from ichor.core.atoms import Atom, Atoms
from ichor.core.adversarial.geometry import (
    atoms_to_coordinates,
    coordinates_to_atoms,
)
from ichor.core.common.units import AtomicDistance


def test_index_preserved():
    template = Atoms([
        Atom("O", 0.0, 0.0, 0.0, index=1),
        Atom("H", 0.96, 0.0, 0.0, index=2),
        Atom("H", -0.24, 0.93, 0.0, index=3),
    ])
    new = coordinates_to_atoms(template, atoms_to_coordinates(template))
    for orig, copy in zip(template, new):
        assert copy._index == orig._index


def test_units_preserved():
    template = Atoms([
        Atom("O", 0.0, 0.0, 0.0, units=AtomicDistance.Bohr),
        Atom("H", 1.81, 0.0, 0.0, units=AtomicDistance.Bohr),
    ])
    new = coordinates_to_atoms(template, atoms_to_coordinates(template))
    for atom in new:
        assert atom.units == AtomicDistance.Bohr


def test_round_trip_identity_preserves_coordinates():
    template = Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
    ])
    new = coordinates_to_atoms(template, atoms_to_coordinates(template))
    np.testing.assert_allclose(
        np.asarray(new.coordinates, dtype=float),
        np.asarray(template.coordinates, dtype=float),
    )


def test_perturbed_coordinates_take_new_values():
    template = Atoms([
        Atom("H", 0.0, 0.0, 0.0),
        Atom("H", 1.0, 0.0, 0.0),
    ])
    new_coords = np.array([[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]])
    new = coordinates_to_atoms(template, new_coords)
    np.testing.assert_allclose(new.coordinates, new_coords)
    # And the type identity survives.
    assert [a.type for a in new] == ["H", "H"]


def test_atom_type_preserved_under_perturbation():
    """Critical for the FD stencil loop: type-dependent mass etc. must
    survive the perturbation -> Atoms -> back-to-coords cycle."""
    template = Atoms([
        Atom("C", 0.0, 0.0, 0.0),
        Atom("Zn", 1.0, 0.0, 0.0),
    ])
    eps = 1e-4
    coords = np.asarray(template.coordinates, dtype=float).reshape(-1)
    plus = coordinates_to_atoms(template, (coords + eps).reshape((-1, 3)))
    minus = coordinates_to_atoms(template, (coords - eps).reshape((-1, 3)))
    assert [a.type for a in plus] == ["C", "Zn"]
    assert [a.type for a in minus] == ["C", "Zn"]
    #Mass survives via type (mass is a derived property of type).
    assert plus[0].mass == template[0].mass
    assert plus[1].mass == template[1].mass


def test_independent_instances_returned():
    """The new atoms must NOT share storage with the template (mutating
    one must not mutate the other)."""
    template = Atoms([Atom("H", 1.0, 2.0, 3.0)])
    new = coordinates_to_atoms(template, np.array([[7.0, 8.0, 9.0]]))
    assert template[0].x == 1.0
    assert new[0].x == 7.0
    #Mutating the new should not touch the template.
    new[0].coordinates[0] = 99.0
    assert template[0].x == 1.0
    assert new[0].x == 99.0
