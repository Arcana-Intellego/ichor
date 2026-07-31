import hashlib
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np
from ichor.core.atoms import ALF
from ichor.core.common.io import mkdir
from ichor.core.common.str import get_digits
from ichor.core.common.types import Version
from ichor.core.files.file import FileContents, ReadFile, WriteFile
from ichor.core.models.kernels import (
    ConstantKernel,
    Kernel,
    PeriodicKernel,
    RBF,
    RBFCyclic,
)
from ichor.core.models.kernels.interpreter import KernelInterpreter
from ichor.core.models.mean import (
    ConstantMean,
    LinearMean,
    Mean,
    QuadraticMean,
    ZeroMean,
)


def _get_default_input_units(nfeats: int) -> List[str]:
    units = [["bohr", "bohr", "radians"][i] for i in range(min(nfeats, 3))]
    for i in range(3, nfeats):
        units += [["bohr", "radians", "radians"][i % 3]]
    return units


def _get_default_output_unit(property: str) -> str:
    if property == "iqa":
        return "Ha"
    elif property == "q00":
        return "e"
    else:
        return "unknown"


class Model(ReadFile, WriteFile):
    """A model file that is returned back from our machine learning program FEREBUS.

    .. note::
        Another program can be used for the machine learning as
        long as it outputs files of the same format as the FEREBUS outputs.
    """

    _filetype = ".model"

    def __init__(
        self,
        path: Path,
        system_name: str = FileContents,
        atom_name: str = FileContents,
        prop: str = FileContents,
        alf: ALF = FileContents,
        natoms: int = FileContents,
        ntrain: int = FileContents,
        nfeats: int = FileContents,
        mean: Mean = FileContents,
        kernel: Kernel = FileContents,
        x: np.ndarray = FileContents,
        y: np.ndarray = FileContents,
        input_units: List[str] = FileContents,
        output_unit: str = FileContents,
        likelihood: float = FileContents,
        jitter: float = FileContents,
        weights: np.ndarray = FileContents,
        program: str = FileContents,
        program_version: Version = FileContents,
        notes: Dict[str, str] = FileContents,
        prefactor: float = FileContents,
    ):
        super(ReadFile, self).__init__(path)

        self.program = program
        self.system_name = system_name
        self.atom_name = atom_name
        self.prop = prop
        self.alf = alf
        self.natoms = natoms
        self.nfeats = nfeats
        self.ntrain = ntrain
        self.mean = mean
        self.kernel = kernel
        self.x = x
        self.y = y
        self.input_units = input_units
        self.output_unit = output_unit
        self.likelihood = likelihood
        self.jitter = jitter
        self.weights = weights
        self.program_version = program_version
        self.notes = notes
        self.prefactor = prefactor
        self._numeric_identity_cache = None
        self._lower_cholesky_cache = None

    @staticmethod
    def _update_numeric_digest(digest, label: str, value) -> None:
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\0")
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(tuple(array.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")

    @property
    def numeric_identity(self) -> str:
        """Return the immutable identity of this fitted numeric model.

        FEREBUS model objects are read-only after admission.  The digest gives
        posterior caches an identity that survives repeated property access and
        does not depend on temporary NumPy allocation addresses.
        """
        cached = getattr(self, "_numeric_identity_cache", None)
        if isinstance(cached, str) and len(cached) == 64:
            return cached
        digest = hashlib.sha256()
        self._update_numeric_digest(digest, "atom", str(self.atom_name))
        self._update_numeric_digest(digest, "property", str(self.prop))
        self._update_numeric_digest(digest, "ntrain", int(self.ntrain))
        self._update_numeric_digest(digest, "nfeats", int(self.nfeats))
        self._update_numeric_digest(digest, "jitter", float(self.jitter))
        self._update_numeric_digest(digest, "prefactor", float(self.kernel_prefactor))
        self._update_numeric_digest(digest, "kernel", self.kernel.write_str())
        self._update_numeric_digest(digest, "mean", self.mean.write_str())
        self._update_numeric_digest(digest, "x", np.asarray(self.x, dtype=float))
        self._update_numeric_digest(digest, "y", np.asarray(self.y, dtype=float))
        self._update_numeric_digest(
            digest,
            "weights",
            np.asarray(self.weights, dtype=float),
        )
        identity = digest.hexdigest()
        self._numeric_identity_cache = identity
        return identity

    def invalidate_numeric_cache(self) -> None:
        """Invalidate cached factors after an explicit in-memory model edit."""
        self._numeric_identity_cache = None
        self._lower_cholesky_cache = None

    def install_lower_cholesky(
        self,
        factor: np.ndarray,
        *,
        expected_numeric_identity: str,
    ) -> None:
        """Install a validated, read-only factor for this exact numeric model."""
        identity = self.numeric_identity
        if str(expected_numeric_identity) != identity:
            raise ValueError("model Cholesky identity does not match the model")
        values = np.asarray(factor)
        expected_shape = (int(self.ntrain), int(self.ntrain))
        if values.dtype != np.dtype(np.float64) or values.shape != expected_shape:
            raise ValueError("model Cholesky factor shape or dtype is invalid")
        if not np.all(np.isfinite(values)):
            raise ValueError("model Cholesky factor must be finite")
        scale = max(1.0, float(np.max(np.abs(values))))
        tolerance = np.finfo(np.float64).eps * max(1, int(self.ntrain)) * scale * 16.0
        if np.any(np.abs(np.triu(values, k=1)) > tolerance):
            raise ValueError("model Cholesky factor must be lower triangular")
        if np.any(np.diag(values) <= 0.0):
            raise ValueError("model Cholesky diagonal must be positive")
        values.setflags(write=False)
        self._lower_cholesky_cache = (identity, values)

    def _read_file(self, up_to: Optional[str] = None):
        """Read in a FEREBUS output file which contains the optimized
        hyperparameters, mean function, and other information that is needed to make predictions."""
        kernel_composition = ""
        kernel_dict = {}
        notes = {}
        declared_kernel_count = None
        prefactor_seen = False

        stop_reading = False

        with open(self.path, "r", encoding="utf-8", newline=None) as f:
            for line in f:
                if stop_reading:
                    break

                if up_to is not None and up_to in line:
                    stop_reading = True

                if "<TODO>" in line:
                    continue

                if "program" in line:
                    self.program = self.program or line.split()[-1]
                    continue

                if "version" in line:
                    self.program_version = self.program_version or Version(
                        line.split()[-1]
                    )
                    continue

                if "jitter" in line or "nugget" in line or "noise" in line:
                    # noise to add to the diagonal to help with numerical stability.
                    # Typically on the scale 1e-6 to 1e-10
                    self.jitter = self.jitter or float(line.split()[-1])
                    continue

                if "likelihood" in line:
                    self.likelihood = self.likelihood or float(line.split()[-1])
                    continue

                if "#" in line and "=" in line:
                    line = line.lstrip("#")
                    key, val = line.split("=")
                    notes[key.strip()] = val.strip()
                    continue

                if line.startswith("#"):
                    line = line.lstrip("#")
                    notes[line.strip()] = None

                if "name" in line:  # system name e.g. WATER
                    self.system_name = self.system_name or line.split()[1]
                    continue

                if line.startswith("atom"):  # atom for which a GP model was made eg. O1
                    self.atom_name = self.atom_name or line.split()[1].capitalize()
                    continue

                if (
                    "property" in line
                ):  # property (such as iqa or particular multipole moment) for which a GP model was made
                    self.prop = self.prop or line.split()[1]
                    continue

                if "ALF" in line:
                    tmp_line_split = line.split()[1:]
                    if tmp_line_split[-1] == "None":
                        self.alf = self.alf or ALF(
                            *[int(a) - 1 for a in line.split()[1:-1]], None
                        )
                    else:
                        self.alf = self.alf or ALF(
                            *[int(a) - 1 for a in line.split()[1:]]
                        )
                    continue

                if "number_of_atoms" in line:
                    self.natoms = self.natoms or int(line.split()[1])
                    continue

                if "number_of_features" in line:  # number of inputs to the GP
                    self.nfeats = self.nfeats or int(line.split()[1])
                    continue

                if (
                    "number_of_training_points" in line
                ):  # number of training points to make the GP model
                    self.ntrain = self.ntrain or int(line.split()[1])
                    continue

                # GP mean (mu) section
                if "[mean]" in line:
                    mean_type = next(f).split()[-1]  # type
                    if mean_type == "constant":
                        mean = ConstantMean(float(next(f).split()[1]))
                    elif mean_type == "zero":
                        mean = ZeroMean()
                    elif mean_type in ["linear", "quadratic"]:
                        beta = np.array([float(b) for b in next(f).split()[1:]])
                        xmin = np.array([float(x) for x in next(f).split()[1:]])
                        ymin = float(next(f).split()[-1])
                        if mean_type == "linear":
                            mean = LinearMean(beta, xmin, ymin)
                        elif mean_type == "quadratic":
                            mean = QuadraticMean(beta, xmin, ymin)
                    else:
                        raise ValueError(
                            "unsupported FEREBUS model mean type " + repr(mean_type)
                        )

                    self.mean = self.mean or mean
                    continue

                if line.strip().startswith("number_of_kernels "):
                    if declared_kernel_count is not None:
                        raise ValueError(
                            "duplicate FEREBUS number_of_kernels declaration"
                        )
                    declared_kernel_count = int(line.split()[-1])
                    if declared_kernel_count <= 0:
                        raise ValueError(
                            "FEREBUS number_of_kernels must be positive"
                        )
                    continue

                if line.strip().startswith("composition "):
                    # which kernels were used to make the GP model.
                    # Different kernels can be specified for different input dimensions
                    if kernel_composition:
                        raise ValueError(
                            "duplicate FEREBUS kernel composition declaration"
                        )
                    kernel_composition = line.split()[-1]
                    continue

                if line.strip().startswith("prefactor "):
                    if prefactor_seen:
                        raise ValueError("duplicate FEREBUS kernel prefactor")
                    self.prefactor = float(line.split()[-1])
                    prefactor_seen = True
                    continue

                # GP kernel section
                if "[kernel." in line:
                    kernel_name = line.split(".")[-1].rstrip().rstrip("]")
                    if kernel_name in kernel_dict:
                        raise ValueError(
                            "duplicate FEREBUS kernel section " + repr(kernel_name)
                        )
                    line = next(f)
                    kernel_type = line.split()[-1].strip()
                    ndims = int(next(f).split()[-1])  # number of dimensions
                    line = next(f)
                    if "TODO" not in line:
                        active_dims = np.asarray(
                            [int(ad) - 1 for ad in line.split()[1:]],
                            dtype=int,
                        )
                    else:
                        active_dims = np.arange(ndims)
                    if active_dims.size != ndims:
                        raise ValueError(
                            "FEREBUS kernel active-dimension count mismatch"
                        )

                    if kernel_type == "rbf":
                        thetas = np.asarray(
                            [float(hp) for hp in next(f).split()[1:]],
                            dtype=float,
                        )
                        if thetas.size != ndims:
                            raise ValueError("FEREBUS RBF theta count mismatch")
                        kernel_dict[kernel_name] = RBF(
                            kernel_name, thetas, active_dims=active_dims
                        )
                    elif kernel_type in [
                        "rbf-cyclic",
                        "rbf-cylic",
                    ]:  # Due to typo in FEREBUS 7.0
                        thetas = np.asarray(
                            [float(hp) for hp in next(f).split()[1:]],
                            dtype=float,
                        )
                        if thetas.size != ndims:
                            raise ValueError("FEREBUS cyclic-RBF theta count mismatch")
                        kernel_dict[kernel_name] = RBFCyclic(
                            kernel_name, thetas, active_dims=active_dims
                        )
                    elif kernel_type == "constant":
                        value = float(next(f).split()[-1])
                        kernel_dict[kernel_name] = ConstantKernel(
                            kernel_name, value, active_dims=active_dims
                        )
                    elif kernel_type == "periodic":
                        thetas = np.asarray(
                            [float(hp) for hp in next(f).split()[1:]],
                            dtype=float,
                        )
                        if thetas.size != ndims:
                            raise ValueError("FEREBUS periodic theta count mismatch")
                        kernel_dict[kernel_name] = PeriodicKernel(
                            kernel_name,
                            thetas,
                            np.full(thetas.shape, 2 * np.pi),
                            active_dims=active_dims,
                        )
                    else:
                        raise ValueError(
                            "unsupported FEREBUS model kernel type "
                            + repr(kernel_type)
                        )

                    continue

                if "units.x" in line:
                    self.input_units = self.input_units or line.split()[1:]

                if "units.y" in line:
                    self.output_unit = self.output_unit or line.split()[-1]

                # training inputs data
                if "[training_data.x]" in line:
                    x = np.empty((self.ntrain, self.nfeats))
                    for i in range(self.ntrain):
                        row = next(f)
                        if not row.strip():
                            raise ValueError("truncated FEREBUS training_data.x section")
                        values = np.array([float(num) for num in row.split()])
                        if values.size != self.nfeats:
                            raise ValueError(
                                "FEREBUS training_data.x feature count mismatch"
                            )
                        x[i, :] = values
                    separator = next(f)
                    if separator.strip():
                        raise ValueError(
                            "FEREBUS training_data.x contains too many rows"
                        )
                    self.x = x if self.x is FileContents else self.x
                    continue

                # training labels data
                if "[training_data.y]" in line:
                    y = np.empty((self.ntrain, 1))
                    for i in range(self.ntrain):
                        row = next(f)
                        if not row.strip():
                            raise ValueError("truncated FEREBUS training_data.y section")
                        y[i, 0] = float(row)
                    separator = next(f)
                    if separator.strip():
                        raise ValueError(
                            "FEREBUS training_data.y contains too many rows"
                        )
                    self.y = y if self.y is FileContents else self.y
                    continue

                if "[weights]" in line:
                    weights = np.empty((self.ntrain, 1))
                    for i in range(self.ntrain):
                        try:
                            row = next(f)
                        except StopIteration as exc:
                            raise ValueError("truncated FEREBUS weights section") from exc
                        if not row.strip():
                            raise ValueError("truncated FEREBUS weights section")
                        weights[i, 0] = float(row)
                    for trailing in f:
                        if trailing.strip():
                            raise ValueError(
                                "unexpected content after FEREBUS weights section"
                            )
                    self.weights = (
                        weights if self.weights is FileContents else self.weights
                    )
                    stop_reading = True

        if kernel_dict or kernel_composition:
            if not kernel_composition:
                raise ValueError("FEREBUS model is missing kernel composition")
            if declared_kernel_count is None:
                raise ValueError("FEREBUS model is missing number_of_kernels")
            if declared_kernel_count != len(kernel_dict):
                raise ValueError("FEREBUS kernel section count mismatch")
            referenced = set(
                re.findall(r"[A-Za-z_][A-Za-z0-9_]*", kernel_composition)
            )
            if referenced != set(kernel_dict):
                raise ValueError(
                    "FEREBUS kernel composition/section coverage mismatch"
                )
            if not prefactor_seen:
                raise ValueError("FEREBUS model is missing kernel prefactor")
            if not np.isfinite(float(self.prefactor)) or float(self.prefactor) <= 0.0:
                raise ValueError(
                    "FEREBUS kernel prefactor must be finite and positive"
                )
            self.kernel = (
                self.kernel
                if self.kernel
                else KernelInterpreter(kernel_composition, kernel_dict).interpret()
            )

    @property
    def ialf(self) -> np.ndarray:
        """Returns the atomic local frame, indices start at 0 (as in Python).

        :return: The 0-indexed np.ndarray corresponding to the alf of the atom.
        """
        return np.array(self.alf)

    @property
    def type(self) -> str:
        """alias for prop"""
        return self.prop

    @property
    def atom(self) -> str:
        """alias for atom_name"""
        return self.atom_name

    @property
    def atom_num(self) -> int:
        """Returns the integer that is in the atom name"""
        return get_digits(self.atom_name)

    @property
    def i(self) -> int:
        """Returns the integer that is one less than the one in the atom name.
        This is the index of the atom in Python objects such as lists (as indeces start at 0)."""
        return self.atom_num - 1

    def r(self, x_test: np.ndarray) -> np.ndarray:
        """Returns the n_train by n_test covariance matrix"""

        # make into a 2d array in case a 1d is passed in
        # add check here in case not called from predict method
        if x_test.ndim == 1:
            x_test = x_test[np.newaxis, ...]

        return self.prior_covariance(self.x, x_test)

    @property
    def kernel_prefactor(self) -> float:
        value = float(self.prefactor)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("model kernel prefactor must be finite and positive")
        return value

    def prior_covariance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self.kernel_prefactor * self.kernel.k(x1, x2)

    def prior_variance_diagonal(self, x: np.ndarray) -> np.ndarray:
        values = (
            self.kernel.k_diag(x)
            if hasattr(self.kernel, "k_diag")
            else np.diag(self.kernel.k(x, x))
        )
        return self.kernel_prefactor * np.asarray(values, dtype=float)

    @property
    def R(self) -> np.ndarray:
        """Returns the covariance matrix and adds a jitter
        to the diagonal for numerical stability. This jitter is a very
        small number on the order of 1e-6 to 1e-10."""
        return self.prior_covariance(self.x, self.x) + (
            self.jitter * np.identity(self.ntrain)
        )

    @property
    def invR(self) -> np.ndarray:
        """Returns the inverse of the covariance matrix R"""
        return np.linalg.inv(self.R)

    @property
    def lower_cholesky(self) -> np.ndarray:
        """Decomposes the covariance matrix into L and L^T. Returns the lower triangular matrix L."""
        identity = self.numeric_identity
        cached = getattr(self, "_lower_cholesky_cache", None)
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and cached[0] == identity
            and isinstance(cached[1], np.ndarray)
        ):
            return cached[1]
        factor = np.linalg.cholesky(self.R)
        factor.setflags(write=False)
        self._lower_cholesky_cache = (identity, factor)
        return factor

    @property
    def _y_minus_mean(self):
        return self.y - self.mean.value(self.x).reshape((-1, 1))

    @property
    def logdet(self):
        sign, logdet = np.linalg.slogdet(self.R)
        return sign * logdet

    def compute_weights(self) -> np.ndarray:
        """Computes the training weights from the data given"""
        lower_solution = np.linalg.solve(
            self.lower_cholesky,
            self._y_minus_mean,
        )
        return np.linalg.solve(self.lower_cholesky.T, lower_solution)

    def compute_log_marginal_likelihood(self) -> float:
        """Return the conventional Gaussian-process log marginal likelihood."""
        quadratic = float(
            np.dot(self._y_minus_mean.T, self.compute_weights()).item()
        )
        return (
            -0.5 * quadratic
            - float(np.sum(np.log(np.diag(self.lower_cholesky))))
            - 0.5 * self.ntrain * np.log(2 * np.pi)
        )

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Returns an array containing the test point predictions."""

        # make into a 2d array in case a 1d is passed in
        if x_test.ndim == 1:
            x_test = x_test[np.newaxis, ...]

        return (
            self.mean.value(x_test) + np.dot(self.r(x_test).T, self.weights)[:, -1]
        ).flatten()

    def variance(self, x_test: np.ndarray) -> np.ndarray:
        """Return the variance for the test data points."""
        train_test_covar = self.r(x_test)
        # temporary matrix, see Rasmussen Williams page 19 algo. 2.1
        v = np.linalg.solve(self.lower_cholesky, train_test_covar)

        # TODO: need to multiply by tau^2 in order to get "true" variance which can be used for error estimations.
        # here it can only be used to compare points to figure out which point has the largest variance.
        return self.prior_variance_diagonal(x_test) - np.diag(
            np.matmul(v.T, v)
        ).flatten()

    def _write_file(self, path: Path) -> None:
        if not path.parent.exists():
            mkdir(path.parent)
        if path.is_dir():
            path = (
                path
                / f"{self.system_name}_{self.prop}_{self.atom_name}{Model.get_filetype()}"
            )

        # these are so that the writing of models does not crash. They do not affect predictions
        if not self.jitter:
            self.jitter = 1e-6
        if not self.likelihood:
            self.likelihood = 1.0
        if not self.notes:
            self.notes = {}

        write_str = ""

        write_str += "# [metadata]\n"
        write_str += f"# program {self.program}\n"
        write_str += f"# version {self.program_version}\n"
        write_str += f"# jitter {self.jitter}\n"
        write_str += f"# likelihood {self.likelihood}\n"
        for key, val in self.notes.items():
            write_str += f"# {key} = {val}\n"
        write_str += "\n"
        write_str += "[system]\n"
        write_str += f"name {self.system_name}\n"
        write_str += f"atom {self.atom_name}\n"
        write_str += f"property {self.prop}\n"
        write_str += f"ALF {self.alf[0] + 1} {self.alf[1] + 1} {self.alf[2] + 1}\n"
        write_str += "\n"
        write_str += "[dimensions]\n"
        write_str += f"number_of_atoms {self.natoms}\n"
        write_str += f"number_of_features {self.nfeats}\n"
        write_str += f"number_of_training_points {self.ntrain}\n"
        write_str += "\n"
        write_str += self.mean.write_str()
        write_str += "\n"
        write_str += "[kernels]\n"
        write_str += f"number_of_kernels {self.kernel.nkernel}\n"
        write_str += f"composition {self.kernel.name}\n"
        write_str += f"prefactor {self.kernel_prefactor}\n"
        write_str += "\n"
        write_str += self.kernel.write_str()
        write_str += "\n"
        write_str += "[training_data]\n"
        write_str += f"units.x {' '.join(self.input_units)}\n"
        write_str += f"units.y {self.output_unit}\n"
        write_str += "scaling.x none\n"
        write_str += "scaling.y none\n"
        write_str += "\n"
        write_str += "[training_data.x]\n"
        for xi in self.x:
            write_str += f"{' '.join(map(str, xi))}\n"
        write_str += "\n"
        write_str += "[training_data.y]\n"
        write_str += "\n".join(map(str, self.y.flatten()))
        write_str += "\n\n"
        write_str += "[weights]\n"
        write_str += "\n".join(map(str, self.weights.flatten()))
        write_str += "\n"

        return write_str

    def __repr__(self):
        return f"{self.__class__.__name__}(system={self.system_name}, atom={self.atom_name}, type={self.prop})"
