"""Interface for the ICHOR x ARIADNE adversarial-attack loop.

This is a flat module at 'ichor.hpc.active_learning.ariadne'.
It re-exports the most-used public names from the
'acquisition.*' submodules so callers can write::

    from ichor.hpc.active_learning.ariadne import (
        AdversarialASECalculator,
        AriadneRunConfig, AriadneRunResult,
        optimise_seed,
        project_out_rigid, rigid_basis,
        select_seeds, SeedSelection,
    )

Anything not yet wired (live ARIADNE driver, SLURM-array orchestration) is
deliberately not exported; tests target the submodule namespaces directly.
"""

from .acquisition.ase_calculator import (
    AdversarialASECalculator,
    ase_atoms_to_ichor_atoms,
)
from .acquisition.ariadne_runner import (
    AriadneRunConfig,
    AriadneRunResult,
    optimise_seed,
)
from .acquisition.rigid_projection import (
    mass_vector_for,
    project_out_rigid,
    rigid_basis,
)
from .acquisition.seed_selection import (
    SeedSelection,
    select_seeds,
)


__all__ = [
    "AdversarialASECalculator",
    "ase_atoms_to_ichor_atoms",
    "AriadneRunConfig",
    "AriadneRunResult",
    "optimise_seed",
    "mass_vector_for",
    "project_out_rigid",
    "rigid_basis",
    "SeedSelection",
    "select_seeds",
]
