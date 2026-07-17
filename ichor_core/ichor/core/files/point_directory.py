from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np

from ichor.core.atoms import Atoms, AtomsNotFoundError
from ichor.core.files import OrcaInput, OrcaOutput
from ichor.core.files.aimall import Aim, IntDirectory
from ichor.core.files.ase.opt import XTB
from ichor.core.files.directory import AnnotatedDirectory
from ichor.core.files.file_data import HasAtoms, HasData
from ichor.core.files.gaussian import GaussianOutput, GJF, WFN
from ichor.core.files.xyz import XYZ


class PointDirectory(AnnotatedDirectory, HasAtoms, HasData):
    """
    A helper class that wraps around ONE directory which contains ONE point (one molecular geometry).

    :param path: Path to a directory which contains ONE point.
    """

    _suffix = ".pointdir"

    contents = {
        "xyz": XYZ,
        "xtb": XTB,
        "gjf": GJF,
        "gaussian_output": GaussianOutput,
        "orca_input": OrcaInput,
        "orca_output": OrcaOutput,
        "aim": Aim,
        "wfn": WFN,
        "ints": IntDirectory,
    }

    def __init__(self, path: Union[Path, str]):
        AnnotatedDirectory.__init__(self, path)

    @classmethod
    def check_path(cls, path: Path) -> bool:
        """Makes sure that path is PointDirectory-like"""
        return (path.suffix == cls._suffix) and path.is_dir()

    @property
    def raw_data(self) -> dict:

        all_data = {}

        for attr_name in self.contents.keys():
            # all contents which subclass from HasData should have raw_data attribute implemented
            attr = getattr(self, attr_name)
            # this also automatically checks for OptionalContent
            if isinstance(attr, HasData):
                d = attr.raw_data
                all_data[attr_name] = d

        return all_data

    @property
    def atoms(self) -> Atoms:
        """Returns the `Atoms` instance which the `PointDirectory` encapsulates."""

        # we should always have an xyz file, so just return the atoms from there
        # this is likely the best solution, so that we always know what is returned
        # and will error out if an xyz is not present
        if self.xyz:
            return self.xyz.atoms
        elif self.wfn:
            return self.wfn.atoms.to_angstroms()

        raise FileNotFoundError(
            f"There is no .xyz or .wfn file in the current {self.__class__.__name__} instance: {self.path.absolute()}"
        )

    def atoms_from_file(self, file_with_atoms: HasAtoms) -> Atoms:
        """Given a class (which is in the contents of the directory), obtain
        the Atoms instance from that specific file which is wrapped by the class.

        :param file_with_atoms: file class which subclasses from HasAtoms
            and has a ``.atoms`` attribute
        :raises ichor.core.atoms.AtomsNotFoundError: If file class does not contain atoms
        :return: _description_
        :rtype: ichor.core.atoms.Atoms
        """
        for f in self.files:
            # try to return atoms
            if isinstance(f, file_with_atoms):
                try:
                    return f.atoms
                # if for some reason the given file does not have atoms attribute
                except AttributeError:
                    raise AtomsNotFoundError(
                        f" {file_with_atoms.__class__.__name__} file does not contain atoms."
                    )

    def features(
        self,
        feature_calculator: Callable,
        *args,
        is_atomic=True,
        **kwargs,
    ):
        """Returns the features for this Atoms instance,
        corresponding to the features of each Atom instance held in this Atoms isinstance
        Features are calculated in the Atom class and concatenated to a 2d array here.

        The array shape is n_atoms x n_features (3*n_atoms - 6)

        :param is_atomic: whether the feature calculator calculates features
            for individual atoms or for the whole geometry.
        :param args: positional arguments to pass to feature calculator
        :param kwargs: key word arguments to pass to feature calculator

        Returns:
            :type: `np.ndarray` of shape n_atoms x n_features (3N-6)
                Return the feature matrix of this Atoms instance
        """
        return self.atoms.features(
            feature_calculator, *args, is_atomic=is_atomic, **kwargs
        )

    def properties(self, system_alf) -> Dict[str, Dict[str, float]]:
        """Return per-atom AIMAll properties for this point directory.

        PointsDirectory.features_with_properties_to_csv() calls this on each contained
        PointDirectory. The AIMAll parser needs the atom-local C matrices to rotate multipoles, so
        build them from the caller's system ALF and delegate to IntDirectory.properties().
        """
        if not self.ints:
            raise FileNotFoundError(
                "AIMAll .int directory missing for pointdir: " + str(self.path)
            )
        try:
            c_matrices = self.C_matrix_dict(system_alf)
        except Exception as exc:
            raise ValueError(
                "failed to build C-matrix dictionary for pointdir "
                + str(self.path)
            ) from exc
        try:
            props = self.ints.properties(c_matrices)
        except Exception as exc:
            raise ValueError(
                "failed to parse AIMAll .int properties for pointdir "
                + str(self.path)
            ) from exc
        if not props:
            raise ValueError("AIMAll .int properties are empty for pointdir: " + str(self.path))
        return props

    def feature_property_rows(
        self,
        system_alf,
        property_types: List[str],
        **kwargs,
    ) -> Dict[str, object]:
        """Return every atom's finite FEREBUS row from one parse of this point."""
        from ichor.core.calculators.features.alf_features_calculator import (
            calculate_alf_features,
        )

        properties = [str(value) for value in property_types]
        if not properties:
            raise ValueError("FEREBUS property_types must not be empty")
        atom_names = list(self.atoms.atom_names)
        features = np.asarray(
            self.features(calculate_alf_features, system_alf, **kwargs),
            dtype=np.float64,
        )
        if features.ndim != 2 or features.shape[0] != len(atom_names):
            raise ValueError(
                "FEREBUS feature matrix shape does not match point atom count: "
                + str(self.path)
            )
        point_properties = self.properties(system_alf)
        rows: Dict[str, np.ndarray] = {}
        for atom_index, atom_name in enumerate(atom_names):
            atom_properties = point_properties.get(atom_name)
            if not isinstance(atom_properties, dict):
                raise ValueError(
                    "FEREBUS properties are missing atom "
                    + atom_name
                    + " in "
                    + str(self.path)
                )
            try:
                targets = [float(atom_properties[property_name]) for property_name in properties]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "FEREBUS properties are incomplete for atom "
                    + atom_name
                    + " in "
                    + str(self.path)
                ) from exc
            row = np.concatenate(
                [features[atom_index], np.asarray(targets, dtype=np.float64)]
            )
            if not np.isfinite(row).all():
                raise ValueError(
                    "FEREBUS row contains a non-finite value for atom "
                    + atom_name
                    + " in "
                    + str(self.path)
                )
            rows[atom_name] = row
        return {
            "atom_names": atom_names,
            "feature_headers": ["f" + str(index + 1) for index in range(features.shape[1])],
            "property_headers": properties,
            "rows": rows,
        }

    @atoms.setter
    def atoms(self, atms: Atoms):
        """Overwrites the current .xyz file with the atoms info that is passed in.

        :param atms: An atoms instance containing geometry information
        """
        if atms:
            if not self.xyz.exists():
                self.xyz = XYZ(self.path / f"{self.path.name}{XYZ.get_filetype()}")
            self.xyz = XYZ(self.xyz.path, atms)

    def __repr__(self):
        """Returns string representation, including class name and path"""
        return self.__class__.__name__ + f'"{str(self.path)}"'
