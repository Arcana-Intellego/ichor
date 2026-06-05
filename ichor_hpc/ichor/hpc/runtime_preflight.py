import importlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np

import ichor.hpc.global_variables


class RuntimeBackendError(RuntimeError):
    """Raised when an optional runtime backend is not usable."""


def _expand_path(path: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(path))))


def _config_value(*keys, default=None):
    return ichor.hpc.global_variables.get_param_from_config(
        ichor.hpc.global_variables.ICHOR_CONFIG or {},
        ichor.hpc.global_variables.MACHINE,
        *keys,
        default=default,
    )


def configured_plumed_kernel() -> Optional[Path]:
    kernel = _config_value("software", "plumed", "kernel_path")
    if kernel is None:
        return None
    return _expand_path(kernel)


def configured_plumed_library_path() -> Optional[Path]:
    library_path = _config_value("software", "plumed", "library_path")
    if library_path is None:
        return None
    return _expand_path(library_path)


def resolve_plumed_kernel(kernel: Optional[Union[str, Path]] = None) -> Optional[Path]:
    if kernel is not None:
        return _expand_path(str(kernel))

    env_kernel = os.environ.get("PLUMED_KERNEL")
    if env_kernel:
        return _expand_path(env_kernel)

    return configured_plumed_kernel()


@contextmanager
def _temporary_plumed_kernel(kernel_path: Path):
    previous = os.environ.get("PLUMED_KERNEL")
    os.environ["PLUMED_KERNEL"] = str(kernel_path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PLUMED_KERNEL", None)
        else:
            os.environ["PLUMED_KERNEL"] = previous


def ensure_ase_available() -> None:
    try:
        importlib.import_module("ase")
    except Exception as exc:
        raise RuntimeBackendError(
            "ASE is not importable in the active Python environment. "
            "Install ichor_core dependencies into the same venv used for ICHOR."
        ) from exc


def ensure_xtb_ase_available(
    *, run_energy: bool = False, method: str = "GFN2-xTB"
) -> None:
    ensure_ase_available()
    try:
        xtb_module = importlib.import_module("xtb.ase.calculator")
        XTB = xtb_module.XTB
    except Exception as exc:
        raise RuntimeBackendError(
            "xtb.ase.calculator.XTB is not importable. Install the xtb Python "
            "package into the same venv used for ICHOR."
        ) from exc

    if not run_energy:
        return

    try:
        atoms_module = importlib.import_module("ase")
        atoms = atoms_module.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        atoms.calc = XTB(method=method)
        energy = atoms.get_potential_energy()
    except Exception as exc:
        raise RuntimeBackendError(
            "ASE+xTB imported but a tiny H2 single-point energy failed."
        ) from exc

    if not np.isfinite(energy):
        raise RuntimeBackendError("ASE+xTB H2 smoke returned a non-finite energy.")


def ensure_plumed_available(
    *, kernel: Optional[Union[str, Path]] = None, run_ase_smoke: bool = False
) -> None:
    try:
        plumed_module = importlib.import_module("plumed")
    except Exception as exc:
        raise RuntimeBackendError(
            "The plumed Python wrapper is not importable. Install the PyPI "
            "`plumed` package into the same venv used for ICHOR."
        ) from exc

    kernel_path = resolve_plumed_kernel(kernel)
    if kernel_path is None:
        raise RuntimeBackendError(
            "PLUMED_KERNEL is not set and <MACHINE>.software.plumed.kernel_path "
            "is missing from ~/ichor_config.yaml."
        )
    if not kernel_path.exists():
        raise RuntimeBackendError(f"PLUMED kernel does not exist: {kernel_path}")
    if not kernel_path.is_file() or not os.access(kernel_path, os.R_OK):
        raise RuntimeBackendError(f"PLUMED kernel is not readable: {kernel_path}")

    try:
        plumed_instance = plumed_module.Plumed(kernel=str(kernel_path))
        plumed_instance.finalize()
    except Exception as exc:
        raise RuntimeBackendError(
            f"plumed.Plumed(kernel={kernel_path!s}) failed to initialise."
        ) from exc

    if run_ase_smoke:
        smoke_ase_plumed_lennard_jones(kernel=kernel_path)


def smoke_ase_plumed_lennard_jones(
    *, kernel: Optional[Union[str, Path]] = None
) -> None:
    kernel_path = resolve_plumed_kernel(kernel)
    if kernel_path is None:
        raise RuntimeBackendError("Cannot run ASE+PLUMED smoke without PLUMED_KERNEL.")

    try:
        from ase import Atoms
        from ase.calculators.lj import LennardJones

        from ichor.core.files.mtd.plumed_calculator import Plumed
    except Exception as exc:
        raise RuntimeBackendError(
            "ASE, LennardJones, or the ICHOR PLUMED calculator is not importable."
        ) from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path.cwd()
        try:
            os.chdir(tmpdir)
            with _temporary_plumed_kernel(kernel_path):
                atoms = Atoms(
                    "Ar2",
                    positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
                    cell=[20.0, 20.0, 20.0],
                    pbc=False,
                )
                plumed_input = [
                    "UNITS LENGTH=A TIME=1 ENERGY=1",
                    "d: DISTANCE ATOMS=1,2",
                    "PRINT ARG=d STRIDE=1 FILE=COLVAR",
                ]
                calc = Plumed(
                    calc=LennardJones(),
                    input=plumed_input,
                    timestep=1.0,
                    atoms=atoms,
                    log="PLUMED.log",
                    kT=0.025,
                )
                atoms.calc = calc
                try:
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                finally:
                    if hasattr(calc, "plumed"):
                        calc.plumed.finalize()
        finally:
            os.chdir(cwd)

    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(forces)):
        raise RuntimeBackendError("ASE+PLUMED Lennard-Jones smoke returned non-finite values.")


def smoke_ase_xtb_plumed(
    *, kernel: Optional[Union[str, Path]] = None, method: str = "GFN2-xTB"
) -> None:
    ensure_xtb_ase_available(run_energy=False, method=method)
    kernel_path = resolve_plumed_kernel(kernel)
    if kernel_path is None:
        raise RuntimeBackendError("Cannot run ASE+xTB+PLUMED smoke without PLUMED_KERNEL.")

    try:
        from ase import Atoms
        from xtb.ase.calculator import XTB

        from ichor.core.files.mtd.plumed_calculator import Plumed
    except Exception as exc:
        raise RuntimeBackendError(
            "ASE, xTB, or the ICHOR PLUMED calculator is not importable."
        ) from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path.cwd()
        try:
            os.chdir(tmpdir)
            with _temporary_plumed_kernel(kernel_path):
                atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
                plumed_input = [
                    "UNITS LENGTH=A TIME=1 ENERGY=1",
                    "d: DISTANCE ATOMS=1,2",
                    "PRINT ARG=d STRIDE=1 FILE=COLVAR",
                ]
                calc = Plumed(
                    calc=XTB(method=method),
                    input=plumed_input,
                    timestep=1.0,
                    atoms=atoms,
                    log="PLUMED.log",
                    kT=0.025,
                )
                atoms.calc = calc
                try:
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                finally:
                    if hasattr(calc, "plumed"):
                        calc.plumed.finalize()
        finally:
            os.chdir(cwd)

    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(forces)):
        raise RuntimeBackendError("ASE+xTB+PLUMED smoke returned non-finite values.")


def ensure_metadynamics_available(
    *, run_xtb_smoke: bool = False, run_plumed_smoke: bool = False
) -> None:
    ensure_xtb_ase_available(run_energy=run_xtb_smoke)
    ensure_plumed_available(run_ase_smoke=run_plumed_smoke)
