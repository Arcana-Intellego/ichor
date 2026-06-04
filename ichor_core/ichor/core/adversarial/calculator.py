from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from ichor.core.atoms import Atoms

from .acquisition import SeedLocalAdversarialAcquisition


@dataclass
class AriadneAdversarialCalculator:
    """A lightweight pseudo-calculator wrapper for ARIADNE-like optimisers.

    The wrapped acquisition is *maximised*, so the pseudo-energy returned here is
    the negative acquisition. The pseudo-forces are the Cartesian gradient of the
    acquisition which is the negative gradient of the pseudo-energy.
    """

    acquisition: SeedLocalAdversarialAcquisition
    gradient_mode: str | None = None

    def get_potential_energy(self, atoms: Atoms) -> float:
        return -float(self.acquisition.value(atoms))

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        return np.asarray(self.acquisition.gradient(atoms, mode=self.gradient_mode), dtype=float)

    def evaluate(self, atoms: Atoms) -> Tuple[float, np.ndarray]:
        energy = self.get_potential_energy(atoms)
        forces = self.get_forces(atoms)
        return energy, forces
