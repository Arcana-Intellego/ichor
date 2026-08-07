from typing import Dict, List, Union

import numpy as np
from ichor.core.calculators.alf import get_atom_alf
from ichor.core.calculators.c_matrix_calculator import calculate_c_matrix
from ichor.core.common.constants import ang2bohr
from ichor.core.common.units import AtomicDistance


default_distance_unit: AtomicDistance = AtomicDistance.Bohr


def calculate_alf_features_batch(
    coordinates_angstrom: np.ndarray,
    atom_names: List[str],
    alfs: Dict[str, "ichor.core.atoms.ALF"],  # noqa F821
    distance_unit: AtomicDistance = default_distance_unit,
) -> Dict[str, np.ndarray]:
    """Vectorised ALF features for homogeneous, ordered geometries."""
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise ValueError("batched ALF coordinates must have shape (n, natoms, 3)")
    if coordinates.shape[1] != len(atom_names):
        raise ValueError("batched ALF atom-name count does not match coordinates")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("batched ALF coordinates must be finite")
    natoms = int(coordinates.shape[1])
    if natoms < 2:
        raise ValueError(
            "batched ALF geometries need more than one atom to calculate features"
        )
    names = [str(name) for name in atom_names]
    if len(set(names)) != natoms:
        raise ValueError("batched ALF atom names must be unique")
    missing = [name for name in names if name not in alfs]
    if missing:
        raise KeyError("No model ALF found for atom(s) " + ", ".join(missing))

    name_to_index = {name: index for index, name in enumerate(names)}
    unit_conversion = (
        1.0 if distance_unit is AtomicDistance.Angstroms else ang2bohr
    )
    min_norm = 1.0e-12
    n_geometries = int(coordinates.shape[0])
    output: Dict[str, np.ndarray] = {}
    for atom_name in names:
        origin_index = name_to_index[atom_name]
        raw_alf = alfs[atom_name]
        if not hasattr(raw_alf, "origin_idx"):
            from ichor.core.atoms import ALF

            try:
                raw_alf = ALF(*[int(value) for value in raw_alf])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "batched ALF definition is invalid for atom " + atom_name
                ) from exc
        origin = int(raw_alf.origin_idx)
        x_index = int(raw_alf.x_axis_idx)
        xy_index = None if raw_alf.xy_plane_idx is None else int(
            raw_alf.xy_plane_idx
        )
        if origin != origin_index:
            raise ValueError("batched ALF origin does not match atom ordering")
        if x_index < 0 or x_index >= natoms or x_index == origin:
            raise ValueError("batched ALF x-axis index is invalid")

        x_raw = coordinates[:, x_index, :] - coordinates[:, origin, :]
        x_norm = np.linalg.norm(x_raw, axis=1)
        if np.any(~np.isfinite(x_norm)) or np.any(x_norm <= min_norm):
            raise ValueError(
                "ALF feature calculation failed: x-axis atom is coincident "
                "with the central atom"
            )
        feature_count = 1 if natoms == 2 else 3 * natoms - 6
        features = np.empty((n_geometries, feature_count), dtype=float)
        features[:, 0] = unit_conversion * x_norm
        if natoms == 2:
            output[atom_name] = features
            continue
        if (
            xy_index is None
            or xy_index < 0
            or xy_index >= natoms
            or xy_index in {origin, x_index}
        ):
            raise ValueError("batched ALF xy-plane index is invalid")

        xy_raw = coordinates[:, xy_index, :] - coordinates[:, origin, :]
        xy_norm = np.linalg.norm(xy_raw, axis=1)
        if np.any(~np.isfinite(xy_norm)) or np.any(xy_norm <= min_norm):
            raise ValueError(
                "ALF feature calculation failed: xy-plane atom is coincident "
                "with the central atom"
            )
        angle_ratio = np.sum(x_raw * xy_raw, axis=1) / (x_norm * xy_norm)
        features[:, 1] = unit_conversion * xy_norm
        features[:, 2] = np.arccos(np.clip(angle_ratio, -1.0, 1.0))

        row1 = x_raw / x_norm[:, None]
        sigma = -np.sum(x_raw * xy_raw, axis=1) / np.sum(x_raw * x_raw, axis=1)
        y_raw = sigma[:, None] * x_raw + xy_raw
        y_norm = np.linalg.norm(y_raw, axis=1)
        if np.any(~np.isfinite(y_norm)) or np.any(y_norm <= min_norm):
            raise ValueError("ALF feature calculation failed: frame atoms are collinear")
        row2 = y_raw / y_norm[:, None]
        row3 = np.cross(row1, row2)

        feature_index = 3
        for other_index, other_name in enumerate(names):
            if other_name in {atom_name, names[x_index], names[xy_index]}:
                continue
            raw = coordinates[:, other_index, :] - coordinates[:, origin, :]
            raw_norm = np.linalg.norm(raw, axis=1)
            if np.any(~np.isfinite(raw_norm)) or np.any(raw_norm <= min_norm):
                raise ValueError(
                    "ALF feature calculation failed: non-frame atom is "
                    "coincident with the central atom"
                )
            features[:, feature_index] = unit_conversion * raw_norm
            zeta_x = np.sum(row1 * raw, axis=1)
            zeta_y = np.sum(row2 * raw, axis=1)
            zeta_z = np.sum(row3 * raw, axis=1)
            features[:, feature_index + 1] = np.arccos(
                np.clip(zeta_z / raw_norm, -1.0, 1.0)
            )
            features[:, feature_index + 2] = np.arctan2(zeta_y, zeta_x)
            feature_index += 3
        if feature_index != feature_count:
            raise ValueError("batched ALF feature ordering is incomplete")
        output[atom_name] = features
    return output


def calculate_alf_features(
    atom: "ichor.core.atoms.Atom",  # noqa F821
    # need to be like this because importing classes leads to circular import issues
    alf: Union[
        "ichor.core.atoms.ALF",  # noqa F821
        List["ichor.core.atoms.ALF"],  # noqa F821
        List[List[int]],
        Dict[str, "ichor.core.atoms.ALF"],  # noqa F821
    ],  # noqa F821
    distance_unit: AtomicDistance = default_distance_unit,
) -> np.ndarray:
    """Calculates the features for the given central atom.

    Args:
        :param atom: an instance of the `Atom` class:
            This atom is the central atom for which we want to calculate the C rotation matrix.
        :param alf: A callable or instance of `ALF` that is used to
            calculate the atomic local frame for the atom. This atomic local frame then defines the
            features which are going to be calculated. If no ALF is passed by user,
            then the default way of calculating ALF is used.
        :param distance_unit: The distance units to use for the calculated distances
            which are part of the features. The default distance is Bohr.

    Returns:
        :type: `np.ndarray`
            A 1D numpy array of shape 3N-6, where N is the number of atoms
            in the system which `atom` is a part of. If there are only two atoms,
            then there is only 1 feature (the distance between the atoms).
    """

    alf = get_atom_alf(atom, alf)

    # if only 2 atoms are in parent, there are only 2 atoms
    # in the system so there is only 1 feature - distance.
    if len(atom.parent) == 2:
        feature_array = np.empty(1)
    elif len(atom.parent) > 2:
        feature_array = np.empty(
            3 * len(atom.parent) - 6
        )  # for systems with more than 2 atoms, we have 3N-6 features
    else:
        raise ValueError(
            "atom.parent needs to have more than 1 atom in order to calculate features."
        )

    # Convert to angstroms to make sure units are in angstroms to begin with
    # to_angstroms creates new instances which we use here to calculate features.
    # the atom outside of the function scope should remain the same as before
    atom = atom.to_angstroms()
    atom.parent = atom.parent.to_angstroms()

    unit_conversion = 1.0 if distance_unit is AtomicDistance.Angstroms else ang2bohr

    x_axis_atom_instance = atom.parent[alf.x_axis_idx]
    x_axis_vect = unit_conversion * (
        x_axis_atom_instance.coordinates - atom.coordinates
    )
    x_bond_norm = np.linalg.norm(x_axis_vect)
    min_norm = 1.0e-12
    if not np.isfinite(x_bond_norm) or x_bond_norm <= min_norm:
        raise ValueError(
            "ALF feature calculation failed: x-axis bond length is non-finite "
            "or the x-axis atom is coincident with the central atom."
        )

    # return array if only 2 atoms, i.e. only 1 feature needed
    if len(atom.parent) == 2:
        feature_array[0] = x_bond_norm
        return feature_array

    # this code is only needed if atom.parent is more than 2 atoms (so it has 3N-6 features)
    xy_plane_atom_instance = atom.parent[alf.xy_plane_idx]

    xy_plane_vect = unit_conversion * (
        xy_plane_atom_instance.coordinates - atom.coordinates
    )

    xy_bond_norm = np.linalg.norm(xy_plane_vect)
    if not np.isfinite(xy_bond_norm) or xy_bond_norm <= min_norm:
        raise ValueError(
            "ALF feature calculation failed: xy-plane bond length is non-finite "
            "or the xy-plane atom is coincident with the central atom."
        )

    angle_ratio = np.dot(x_axis_vect, xy_plane_vect.T) / (x_bond_norm * xy_bond_norm)
    angle = np.arccos(
        np.clip(angle_ratio, -1.0, 1.0)
    )

    feature_array[0] = x_bond_norm
    feature_array[1] = xy_bond_norm
    feature_array[2] = angle

    c_matrix = calculate_c_matrix(atom, alf)

    # the rest of the atoms are described as 3 features each:
    # distance(r), polar angle(theta), and azimuthal angle(phi) - physics convention
    # theta is between 0 and pi (not cyclic), phi is between -pi and pi (cyclic)

    if len(atom._parent) > 3:
        i_feat = 3
        for jatom in atom._parent:
            if jatom.name in [
                x_axis_atom_instance.name,
                xy_plane_atom_instance.name,
                atom.name,
            ]:
                continue

            r_vect = unit_conversion * (jatom.coordinates - atom.coordinates)
            r_vect_norm = np.linalg.norm(r_vect)
            if not np.isfinite(r_vect_norm) or r_vect_norm <= min_norm:
                raise ValueError(
                    "ALF feature calculation failed: non-frame bond length is "
                    "non-finite or the non-frame atom is coincident with the "
                    "central atom."
                )
            feature_array[i_feat] = r_vect_norm

            i_feat += 1

            zeta = np.dot(c_matrix, r_vect)
            # TODO: is few geometries zeta[2] / r_vect_norm can be evaluated as +-1.0000000001 or someting close
            # which is outside of the range of arccos
            # clipping zeta[2] / r_vect_norm between -1.0 and 1.0 solves the problem
            z2_rvect_clipped = np.clip(zeta[2] / r_vect_norm, -1.0, 1.0)
            feature_array[i_feat] = np.arccos(z2_rvect_clipped)

            i_feat += 1

            feature_array[i_feat] = np.arctan2(zeta[1], zeta[0])

            i_feat += 1

    return feature_array
