"""ASE Calculator wrapping :class: SeedLocalAdversarialAcquisition.

Provides the bridge between the ICHOR adversarial acquisition (whose value
is to be MAXIMISED) and an ARIADNE-driven ASE optimiser (which MINIMISES
an energy). The translation is:

    E_pseudo (eV)         = -alpha(x) * Hartree
    F_pseudo (eV / A)     = +grad(alpha)(x) * Hartree

ASE convention is "F = -dE/dx". With "E = -alpha" we get
"F = -d(-alpha)/dx = +grad(alpha)", so the forces returned here are the
positive gradient of the acquisition. ARIADNE's "run_ariadne.py:613" reads
ASE forces and negates them to obtain the optimiser gradient. The Hartree
factor is only a reversible pseudo-energy scale used at the ASE boundary.

Optional safety net at the boundary (all opt-in via constructor args):

* "project_rigid": apply the mass-weighted rigid projector from
  :mod:".rigid_projection" to remove translation / rotation components from
  the gradient before the Hartree -> eV scaling. Defaults to "True".
* "max_acquisition_grad_per_ang": per-atom magnitude cap on the acquisition
  gradient before the pseudo-energy Hartree -> eV scaling. Defaults to 50.0;
  set <= 0 to disable. "max_force_per_atom_ha_per_ang" is accepted as a
  deprecated alias for existing configs and tests.
* "clamp_counter": optional "collections.Counter" updated whenever the
  per-atom cap clips a row; used by the daemon to report on saturation.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition

from .gradient_diagnostics import static_gradient_diagnostics
from .rigid_projection import project_out_rigid


__all__ = ["AdversarialASECalculator", "ase_atoms_to_ichor_atoms"]


def ase_atoms_to_ichor_atoms(ase_atoms):
    """Convert an "ase.Atoms" to an :class:"ichor.core.atoms.Atoms".

    Imported lazily so that callers using only ICHOR Atoms do not pull
    "ase" into their import graph.
    """
    from ichor.core.atoms import Atom, Atoms as IchorAtoms

    symbols = ase_atoms.get_chemical_symbols()
    positions = ase_atoms.get_positions()
    return IchorAtoms([
        Atom(sym, float(x), float(y), float(z))
        for sym, (x, y, z) in zip(symbols, positions)
    ])


class AdversarialASECalculator:
    """Lightweight ASE-style calculator.

    Implements only the subset of the ASE Calculator interface that ARIADNE
    needs: "get_potential_energy(atoms)" and "get_forces(atoms)". This
    avoids depending on the full "ase.calculators.calculator.Calculator"
    base class at unit-test time (which pulls in "ase.dft" and friends)
    while still being a drop-in replacement for the ARIADNE callback contract
    (see "run_ariadne.py:613" which only calls these two methods on an ASE
    "Atoms" instance).
    """

    implemented_properties = ("energy", "forces")

    def __init__(
        self,
        acquisition: SeedLocalAdversarialAcquisition,
        *,
        project_rigid: bool = True,
        max_acquisition_grad_per_ang: Optional[float] = None,
        max_force_per_atom_ha_per_ang: Optional[float] = 50.0,
        gradient_mode: Optional[str] = None,
        gradient_backend: Optional[str] = None,
        objective: str = "full",
        clamp_counter=None,
    ) -> None:
        self._acq = acquisition
        self._project_rigid = bool(project_rigid)
        legacy_limit = 50.0 if max_force_per_atom_ha_per_ang is None else float(
            max_force_per_atom_ha_per_ang
        )
        self._max_acquisition_grad = float(
            legacy_limit
            if max_acquisition_grad_per_ang is None
            else max_acquisition_grad_per_ang
        )
        self._gradient_mode = gradient_mode
        # None -> call acq.gradient as before. "process"/"thread"/"serial"
        # routes a cartesian_fd gradient through the parallel driver (process
        # forks a pool across the task's cores on linux; serial elsewhere).
        self._gradient_backend = gradient_backend
        self._objective = str(objective or "full")
        self._clamp_counter = clamp_counter
        #cache results so a paired get_potential_energy + get_forces call on
        #the same geometry does not recompute the acquisition gradient.
        self._cache_key = None
        self._cache_energy = None
        self._cache_forces = None
        self._last_atoms_for_diagnostics = None
        self._gradient_call_count = 0
        self._gradient_wall_seconds_total = 0.0
        self._gradient_wall_seconds_last = 0.0
        self._gradient_wall_seconds_max = 0.0
        self._last_gradient_norm = 0.0
        self._last_parallel_gradient_diagnostics = {}

    # --- public ASE-style API ----------------------------------------------

    def get_potential_energy(self, atoms) -> float:
        self._compute(atoms)
        return float(self._cache_energy)

    def get_forces(self, atoms) -> np.ndarray:
        self._compute(atoms)
        # hand back a copy: the cached array is reused across calls, so an
        # in-place mutation by a caller would silently poison it.
        return np.array(self._cache_forces, dtype=float)

    def evaluate(self, atoms):
        self._compute(atoms)
        return float(self._cache_energy), np.array(self._cache_forces, dtype=float)

    def gradient_diagnostics(self) -> dict:
        static = static_gradient_diagnostics(
            self._acq,
            self._last_atoms_for_diagnostics,
            gradient_mode=(
                self._gradient_mode
                or getattr(
                    getattr(getattr(self._acq, "config", None), "gradient", None),
                    "mode",
                    None,
                )
            ),
            gradient_backend=self._gradient_backend,
        )
        static["gradient_objective"] = self._objective
        driver_diag = getattr(self._acq, "_last_driver_gradient_diagnostics", None)
        if isinstance(driver_diag, dict):
            static.update(driver_diag)
        count = int(self._gradient_call_count)
        total = float(self._gradient_wall_seconds_total)
        static.update({
            "gradient_call_count": count,
            "gradient_wall_seconds_last": float(self._gradient_wall_seconds_last),
            "gradient_wall_seconds_total": total,
            "gradient_wall_seconds_mean": (total / count if count else 0.0),
            "gradient_wall_seconds_max": float(self._gradient_wall_seconds_max),
            "last_gradient_norm": float(self._last_gradient_norm),
        })
        if isinstance(self._last_parallel_gradient_diagnostics, dict):
            static.update(self._last_parallel_gradient_diagnostics)
        return static

    # --- internals ----------------------------------------------------------

    def _acquisition_value(self, atoms) -> float:
        try:
            return float(self._acq.value(atoms, objective=self._objective))
        except TypeError:
            if self._objective == "full":
                return float(self._acq.value(atoms))
            raise

    def _acquisition_gradient(self, atoms):
        try:
            return self._acq.gradient(
                atoms,
                mode=self._gradient_mode,
                objective=self._objective,
            )
        except TypeError:
            if self._objective == "full":
                return self._acq.gradient(atoms, mode=self._gradient_mode)
            raise

    def _hartree(self) -> float:
        """Hartree -> eV conversion. Lazy-import ase.units so unit tests
        that do not exercise the ASE path can run without ASE installed."""
        from ase.units import Hartree

        return float(Hartree)

    def _coords_key(self, atoms):
        if hasattr(atoms, "get_positions"):
            arr = np.asarray(atoms.get_positions(), dtype=float)
            species = tuple(getattr(atoms, "get_chemical_symbols", lambda: [])())
        else:
            arr = np.asarray(atoms.coordinates, dtype=float)
            species = tuple(a.type for a in atoms)
        #bytes -> hashable; rounded to suppress noise from ASE's internal
        #numerical operations on coordinates. species/order go in too so two
        #different systems sharing coordinates can't collide on a cache hit.
        return (species, arr.round(12).tobytes())

    def _compute(self, atoms) -> None:
        key = self._coords_key(atoms)
        if key == self._cache_key:
            return

        if hasattr(atoms, "get_chemical_symbols"):
            ichor_atoms = ase_atoms_to_ichor_atoms(atoms)
        else:
            ichor_atoms = atoms

        alpha = self._acquisition_value(ichor_atoms)
        hartree = self._hartree()
        energy_ev = -alpha * hartree

        gmode = self._gradient_mode or getattr(
            getattr(getattr(self._acq, "config", None), "gradient", None), "mode", None
        )
        parallel_diagnostics = {}
        if self._gradient_backend and gmode in {"cartesian_fd", "active_fd"}:
            from .parallel_gradient import (
                compute_active_gradient,
                compute_cartesian_gradient,
                last_gradient_parallel_diagnostics,
            )
            t_grad = time.perf_counter()
            try:
                if gmode == "active_fd":
                    raw_grad = compute_active_gradient(
                        self._acq,
                        ichor_atoms,
                        backend=self._gradient_backend,
                        objective=self._objective,
                    )
                else:
                    raw_grad = compute_cartesian_gradient(
                        self._acq,
                        ichor_atoms,
                        backend=self._gradient_backend,
                        objective=self._objective,
                    )
                grad = np.asarray(raw_grad, dtype=float)
                parallel_diagnostics = last_gradient_parallel_diagnostics()
            finally:
                elapsed = float(time.perf_counter() - t_grad)
                self._gradient_call_count += 1
                self._gradient_wall_seconds_last = elapsed
                self._gradient_wall_seconds_total += elapsed
                self._gradient_wall_seconds_max = max(
                    self._gradient_wall_seconds_max, elapsed
                )
        else:
            t_grad = time.perf_counter()
            try:
                grad = np.asarray(
                    self._acquisition_gradient(ichor_atoms),
                    dtype=float,
                )
            finally:
                elapsed = float(time.perf_counter() - t_grad)
                self._gradient_call_count += 1
                self._gradient_wall_seconds_last = elapsed
                self._gradient_wall_seconds_total += elapsed
                self._gradient_wall_seconds_max = max(
                    self._gradient_wall_seconds_max, elapsed
                )
        if grad.ndim == 1:
            grad = grad.reshape(-1, 3)
        self._last_atoms_for_diagnostics = ichor_atoms
        self._last_gradient_norm = float(np.linalg.norm(grad))
        self._last_parallel_gradient_diagnostics = dict(parallel_diagnostics)

        if self._project_rigid:
            grad = project_out_rigid(grad, ichor_atoms)

        if self._max_acquisition_grad > 0.0:
            # cap the largest per-atom acquisition gradient without bending the descent
            # direction: scale the whole gradient by a single factor. clamping
            # each atom on its own tilts the 3N vector, and scaling after the
            # projection would smuggle the rigid modes back in -- so project
            # first, then scale uniformly.
            norms = np.linalg.norm(grad, axis=1)
            max_norm = float(norms.max()) if norms.size else 0.0
            if max_norm > self._max_acquisition_grad:
                if self._clamp_counter is not None:
                    clipped = int(np.sum(norms > self._max_acquisition_grad))
                    self._clamp_counter["per_atom_acquisition_grad"] = (
                        int(self._clamp_counter.get("per_atom_acquisition_grad", 0))
                        + clipped
                    )
                grad = grad * (self._max_acquisition_grad / max_norm)

        forces_ev_per_ang = grad * hartree

        self._cache_key = key
        self._cache_energy = energy_ev
        self._cache_forces = forces_ev_per_ang
