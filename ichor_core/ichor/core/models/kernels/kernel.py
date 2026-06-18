from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


def _flat_kernel_params(value) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        parts = [_flat_kernel_params(part) for part in value]
        if not parts:
            return np.array([], dtype=float)
        return np.concatenate(parts)
    return np.asarray(value, dtype=float).reshape(-1)


class Kernel(ABC):
    """Base class for all kernels that implements dunder methods for addition or multiplication of separate kernels"""

    # TODO: figure out a good way to say if training data is standardized,
    # normalized, etc. because kernels can be affected (for example cyclic RBF is affected)

    def __init__(self, name: str, active_dims: Optional[np.ndarray] = None):
        self.name = name
        self._active_dims = active_dims

    @property
    def active_dims(self):
        if self._active_dims is not None:
            return self._active_dims
        return np.arange(len(self._thetas))

    @property
    def true_lengthscales(self):
        """These are the true lengthscale values. Typically the kernel
        equations are written with these values (l) instead of theta (see the
        kernel cookbook or Rasmussen and Williams for examples."""
        return np.sqrt(1.0 / (2 * self._thetas))

    @abstractmethod
    def params(self) -> np.ndarray:
        pass

    @property
    def nkernel(self) -> int:
        return 1

    @abstractmethod
    def k(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Calculates covariance matrix from two sets of points

        Args:
            :param: `x1` np.ndarray of shape n x ndimensions:
                First matrix of n points
            :param: `x2` np.ndarray of shape m x ndimensions:
                Second matrix of m points, can be identical to the first matrix `x1`

        Returns:
            :type: `np.ndarray`
                A covariance matrix of shape (n, m)
        """

    def r(self, x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        """helper method to return x_test, x_train covariance matrix K(X*, X)"""
        return self.k(x_train, x_test)

    def R(self, x_train: np.ndarray) -> np.ndarray:
        """helper method to return symmetric square matrix x_train, x_train covariance matrix K(X, X)"""
        return self.k(x_train, x_train)

    @abstractmethod
    def __repr__(self):
        pass

    def __add__(self, other):
        return KernelSum(self, other)

    def __mul__(self, other):
        return KernelProd(self, other)

    @abstractmethod
    def write_str(self):
        pass


class CompositeKernel(Kernel, ABC):
    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    @property
    def params(self) -> np.ndarray:
        return np.concatenate(
            (_flat_kernel_params(self.k1.params), _flat_kernel_params(self.k2.params))
        )

    @property
    def nkernel(self) -> int:
        return self.k1.nkernel + self.k2.nkernel

    def write_str(self) -> str:
        """Used to write out kernel info to model file"""
        str_to_write = ""

        str_to_write += self.k1.write_str()
        str_to_write += "\n"
        str_to_write += self.k2.write_str()

        return str_to_write


class KernelSum(CompositeKernel):
    """Kernel addition implementation"""

    def k(self, xi, xj):
        return self.k1.k(xi, xj) + self.k2.k(xi, xj)

    def k_diag(self, x):
        if hasattr(self.k1, "k_diag") and hasattr(self.k2, "k_diag"):
            return self.k1.k_diag(x) + self.k2.k_diag(x)
        return np.diag(self.k(x, x))

    def r(self, xi, x):
        return self.k1.r(xi, x) + self.k2.r(xi, x)

    def R(self, x):
        return self.k1.R(x) + self.k2.R(x)

    @property
    def name(self) -> str:
        return f"({self.k1.name}+{self.k2.name})"

    def __repr__(self):
        return f"({self.k1} + {self.k2})"


class KernelProd(CompositeKernel):
    """Kernel multiplication implementation"""

    @property
    def params(self):
        return np.concatenate(
            (_flat_kernel_params(self.k1.params), _flat_kernel_params(self.k2.params))
        )

    def k(self, xi, xj):
        return self.k1.k(xi, xj) * self.k2.k(xi, xj)

    def k_diag(self, x):
        if hasattr(self.k1, "k_diag") and hasattr(self.k2, "k_diag"):
            return self.k1.k_diag(x) * self.k2.k_diag(x)
        return np.diag(self.k(x, x))

    def r(self, xi, x):
        return self.k1.r(xi, x) * self.k2.r(xi, x)

    def R(self, x):
        return self.k1.R(x) * self.k2.R(x)

    @property
    def name(self) -> str:
        return f"({self.k1.name}*{self.k2.name})"

    def __repr__(self):
        return f"({self.k1} * {self.k2})"
