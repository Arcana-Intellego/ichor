"""Per-seed ARIADNE driver.

Bridges the ICHOR daemon and the ARIADNE oneAPI optimiser. The daemon
submits a SLURM array job for the ARIADNE_ARRAY phase; each array task
invokes python -m ichor.hpc.active_learning.acquisition.ariadne_runner
which (after loading the campaign state) calls in here to actually drive
the adversarial descent for one seed.

Why this lives in its own file rather than alongside ariadne_runner.py:
the runner file holds the public contract -- the dataclasses and the
optimise_seed dispatcher -- and is imported by dry-run code that runs
off-cluster where ariadne is not on PYTHONPATH. Keeping the heavy
import ariadne in this side module lets the contract module stay
lightweight and importable everywhere.

The driver mirrors the pattern from ARIADNE/ariadne_starter_pack/
run_ariadne.py: a two-stage step_py loop (propose then accept/reject)
with energy + gradient supplied by an ASE-style calculator. For our
purposes the calculator is AdversarialASECalculator wrapping the
SeedLocalAdversarialAcquisition we built earlier.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# 1 Hartree expressed in eV. matches the value ASE uses; we redefine
# it here rather than depend on ase.units so this module is importable
# in environments where ase is not installed (e.g. very lean CI runs).
_HARTREE_EV = 27.211386245988

# ariadne tags hessian models by integer. mirror the starter pack
# HESSIAN_MODEL_MAP so a config naming a model in caps still resolves.
_HESSIAN_MODEL_MAP = {
    "constant": 0,
    "almlof":   1,
    "lindh":    2,
    "schlegel": 3,
}

# when TRQN rejects too many proposals in a row we give up and re-init
# under DS. three is the value the starter pack uses for
# recovery_accepts_to_exit which is the closest equivalent.
_REJECT_STREAK_TRIGGER = 3


@dataclass
class OptimisationResult:
    """Everything _live_optimise_seed needs to build an AriadneRunResult.

    Tracking the per-step trajectories lets the caller see how the descent
    progressed -- handy when reading per-seed result.json files later.
    """
    final_positions_angstrom: np.ndarray
    alpha_trajectory: List[float]
    grad_norm_trajectory: List[float]
    n_evaluations: int
    # return_code: 0 converged, 1 max_iter, 2 error during evaluation
    return_code: int
    wall_seconds: float
    fell_back_to_ds: bool
    rigid_force_clamps: int
    converged: bool
    candidate_positions_angstrom: List[np.ndarray]
    candidate_alphas: List[float]
    candidate_grad_norms: List[float]


def _import_ariadne():
    """Lazy import. Raises a friendly RuntimeError if the oneAPI .so is
    not on PYTHONPATH yet, which is the expected case off-cluster.
    """
    try:
        import ariadne  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "ARIADNE Python module not importable. Build the oneAPI .so "
            "and prepend build-oneapi/python to PYTHONPATH."
        ) from exc
    return ariadne


def _symbols_to_atom_list(symbols):
    """Build the S2 byte array ARIADNE expects for atom labels.

    2-char fixed-width, left-aligned, truncated to 2. matches the
    starter pack symbols_to_f90_labels.
    """
    cleaned = []
    for s in symbols:
        s = str(s).strip()
        if len(s) < 1 or len(s) > 2:
            raise ValueError(
                "atom symbols must be 1 or 2 characters; got " + repr(s)
            )
        cleaned.append(s)
    return np.asarray(["{:<2}".format(s)[:2] for s in cleaned], dtype="S2")


def _hessian_model_id(name) -> int:
    """Resolve a hessian-model name to the integer ARIADNE wants."""
    key = (str(name) if name is not None else "almlof").strip().lower()
    if key not in _HESSIAN_MODEL_MAP:
        raise ValueError(
            "unknown ariadne.hessian_model " + repr(name)
            + "; valid: " + repr(sorted(_HESSIAN_MODEL_MAP))
        )
    return _HESSIAN_MODEL_MAP[key]


def _eval_energy_gradient(atoms):
    """One calculator evaluation. Returns (e_hartree, g_xyz_hartree_ang).

    Mirrors evaluate_energy_and_gradient from the ARIADNE starter pack.
    The adversarial calculator returns energy + forces in ASE units
    (eV and eV/Angstrom). ARIADNE wants Hartree and Hartree/Angstrom,
    so we convert. Sign-flipping force -> gradient lives here too.
    """
    e_ev = float(atoms.get_potential_energy())
    forces_ev = np.asarray(atoms.get_forces(), dtype=np.float64)
    grad_ev = -forces_ev
    e_hartree = e_ev / _HARTREE_EV
    g_xyz_hartree = np.asfortranarray(grad_ev / _HARTREE_EV, dtype=np.float64)
    if not np.isfinite(e_hartree):
        raise RuntimeError(
            "calculator returned non-finite energy"
        )
    if not np.isfinite(g_xyz_hartree).all():
        raise RuntimeError(
            "calculator returned non-finite entries in the gradient"
        )
    return e_hartree, g_xyz_hartree


def _sync_state(opt, n_atoms):
    """Pull (q_xyz, g_xyz) out of the ariadne optimiser.

    Fortran side keeps positions and gradient in flat (3N,) arrays.
    We pre-allocate them and let get_state_flat_py fill them.
    """
    n3 = 3 * int(n_atoms)
    q_flat = np.empty(n3, dtype=np.float64, order="F")
    g_flat = np.empty(n3, dtype=np.float64, order="F")
    opt.get_state_flat_py(q_flat, g_flat)
    return q_flat.reshape(n_atoms, 3), g_flat.reshape(n_atoms, 3)


def _status_trqn(opt):
    """Pull (converged, last_step_accepted, f_current) off a TRQN opt.

    The TRQN get_status_py tuple from the starter pack has f_current at
    index 1, converged at index 4 and last_step_accepted at index 5.
    """
    status = opt.get_status_py()
    converged = bool(status[4])
    accepted = bool(status[5])
    f_current = float(status[1])
    return converged, accepted, f_current


def _status_ds(opt):
    """Pull (converged, last_step_accepted, f_current) off a DS opt.

    DS get_status_py layout has f_current at index 2, converged at
    index 8 and last_step_accepted at index 9.
    """
    status = opt.get_status_py()
    converged = bool(status[8])
    accepted = bool(status[9])
    f_current = float(status[2])
    return converged, accepted, f_current


def _push_positions(atoms, q_xyz):
    """Write a (N, 3) Angstrom array back into an ASE Atoms object."""
    atoms.set_positions(np.asarray(q_xyz, dtype=np.float64))


def _build_trqn(ariadne, q0_xyz, g0_xyz, atom_list, run_config):
    """Build and init a trust-region quasi-Newton optimiser.

    Minimal-required kwargs only -- everything else falls back to
    ariadne internal defaults. AriadneRunConfig does not expose
    trust_min so we pin a small floor here (1.0e-4 Angstrom), matching
    the starter pack default.
    """
    opt = ariadne.Geometric_Trqn.trust_region_qn()
    opt.init(
        q0_xyz=q0_xyz,
        g0_xyz=g0_xyz,
        atom_list=atom_list,
        trust0=float(run_config.delta0),
        trust_min=1.0e-4,
        trust_max=float(run_config.delta_max),
        hessian_model=_hessian_model_id(run_config.hessian_model),
    )
    return opt


def _build_ds(ariadne, q0_xyz, g0_xyz, atom_list, run_config):
    """Build and init a dissipative-symplectic optimiser.

    DS interprets h as its initial step size; we reuse delta0 from the
    config so the two optimisers start from the same trust scale.
    """
    opt = ariadne.Ds_Optimiser.dissipative_symplectic()
    opt.init(
        q0_xyz=q0_xyz,
        g0_xyz=g0_xyz,
        atom_list=atom_list,
        gamma=float(run_config.gamma),
        h=float(run_config.delta0),
        f_tol=float(run_config.f_tol),
        gradf_tol=float(run_config.gradf_tol),
    )
    return opt


def run_optimisation_against_calculator(
    seed_atoms,
    calculator,
    run_config,
):
    """Drive ARIADNE for a single seed against the supplied calculator.

    Parameters
    ----------
    seed_atoms
        ASE Atoms at the starting geometry. We work on a copy so the
        caller object is left alone.
    calculator
        ASE Calculator that returns energy + forces. For our case this
        is AdversarialASECalculator wrapping SeedLocalAdversarialAcquisition
        built for this seed.
    run_config
        AriadneRunConfig: which optimiser, max_iter, gradf_tol, f_tol,
        delta0, delta_max, gamma, fallback_to_ds, hessian_model.

    Returns
    -------
    OptimisationResult
    """
    ariadne = _import_ariadne()
    t0 = time.perf_counter()

    atoms = seed_atoms.copy()
    atoms.calc = calculator

    natoms = len(atoms)
    symbols = [a.symbol for a in atoms]
    atom_list = _symbols_to_atom_list(symbols)
    q0_xyz_angstrom = np.asfortranarray(
        atoms.get_positions(), dtype=np.float64,
    )

    # one calculator call before the loop -- ariadne needs an initial
    # energy + gradient to seed its internal hessian model.
    e0_hartree, g0_xyz = _eval_energy_gradient(atoms)
    n_evaluations = 1

    optimiser_name = (run_config.optimiser or "trust_region_qn").strip().lower()
    if optimiser_name in ("ds", "dissipative_symplectic"):
        opt = _build_ds(ariadne, q0_xyz_angstrom, g0_xyz, atom_list, run_config)
        is_trqn = False
    elif optimiser_name in ("trust_region_qn", "trqn"):
        opt = _build_trqn(ariadne, q0_xyz_angstrom, g0_xyz, atom_list, run_config)
        is_trqn = True
    else:
        raise ValueError(
            "unknown ariadne.optimiser " + repr(run_config.optimiser)
            + "; valid: trust_region_qn | dissipative_symplectic"
        )

    # alpha is what we want to MAXIMISE (the adversarial acquisition
    # value). the calculator already negated energy so ariadne can
    # minimise, so alpha = -energy.
    alpha_trajectory = [-e0_hartree]
    grad_norm_trajectory = [float(np.linalg.norm(g0_xyz))]
    candidate_positions = [np.asarray(q0_xyz_angstrom, dtype=np.float64).copy()]
    candidate_alphas = [float(-e0_hartree)]
    candidate_grad_norms = [float(np.linalg.norm(g0_xyz))]
    fell_back_to_ds = False
    rigid_force_clamps = 0
    consecutive_rejects = 0
    f_current = e0_hartree
    converged = False
    return_code = 1  # max_iter default until we converge or error

    zero_grad = np.zeros((natoms, 3), dtype=np.float64, order="F")

    for step_idx in range(int(run_config.max_iter)):
        f_old = f_current

        # stage 0 -- propose a step. ariadne does not need a trial
        # energy yet so we pass the previous f and a zero gradient.
        try:
            opt.step_py(
                stage=0, f_old=f_old, f_new=f_old, g_xyz_new=zero_grad,
            )
        except Exception:
            return_code = 2
            break

        # pull the proposed geometry out and push it into the ASE atoms
        # so the calculator can evaluate the trial point.
        q_proposed, _g_proposed = _sync_state(opt, natoms)
        _push_positions(atoms, q_proposed)

        try:
            f_trial, g_trial = _eval_energy_gradient(atoms)
            n_evaluations += 1
        except Exception:
            # calculator died on the trial point. break out and tag as
            # error -- the caller will write a small result.json and
            # the postprocess parser will flag the seed.
            return_code = 2
            break

        # stage 1 -- ariadne sees the trial energy + gradient and
        # decides whether to accept, reject, or declare convergence.
        try:
            opt.step_py(
                stage=1, f_old=f_old, f_new=float(f_trial),
                g_xyz_new=g_trial,
            )
        except Exception:
            return_code = 2
            break

        q_current, _g_current = _sync_state(opt, natoms)
        _push_positions(atoms, q_current)

        if is_trqn:
            opt_converged, accepted, f_current = _status_trqn(opt)
        else:
            opt_converged, accepted, f_current = _status_ds(opt)

        alpha_trajectory.append(-float(f_current))
        grad_norm_trajectory.append(float(np.linalg.norm(g_trial)))

        if accepted:
            consecutive_rejects = 0
            candidate_positions.append(
                np.asarray(atoms.get_positions(), dtype=np.float64).copy()
            )
            candidate_alphas.append(float(-float(f_current)))
            candidate_grad_norms.append(float(np.linalg.norm(g_trial)))
        else:
            consecutive_rejects += 1

        if opt_converged:
            converged = True
            return_code = 0
            break

        # also honour the daemon-side gradf_tol -- ariadne internal
        # convergence is conservative and the daemon may want tighter.
        if accepted and grad_norm_trajectory[-1] < float(run_config.gradf_tol):
            converged = True
            return_code = 0
            break

        # TRQN -> DS fallback. when too many consecutive proposals
        # get rejected and the config allows it, abandon TRQN and
        # re-init under DS at the current geometry. only one fallback
        # per run.
        if (is_trqn and bool(run_config.fallback_to_ds)
            and not fell_back_to_ds
            and consecutive_rejects >= _REJECT_STREAK_TRIGGER):
            try:
                # the trial gradient belongs to the rejected proposal, not the
                # geometry we are restarting DS from. re-evaluate here so the
                # (position, gradient) pair handed to DS actually match up.
                _f_here, g_here = _eval_energy_gradient(atoms)
                n_evaluations += 1
                opt = _build_ds(
                    ariadne,
                    np.asfortranarray(
                        atoms.get_positions(), dtype=np.float64,
                    ),
                    g_here, atom_list, run_config,
                )
                is_trqn = False
                fell_back_to_ds = True
                consecutive_rejects = 0
            except Exception:
                return_code = 2
                break

    # the adversarial calculator subscripts the clamp counter as a dict;
    # per_atom_force is the key it bumps when it had to clip a gradient.
    cc = getattr(calculator, "_clamp_counter", None)
    if isinstance(cc, dict):
        try:
            rigid_force_clamps = int(cc.get("per_atom_force", 0))
        except (TypeError, ValueError):
            rigid_force_clamps = 0

    return OptimisationResult(
        final_positions_angstrom=np.asarray(
            atoms.get_positions(), dtype=np.float64,
        ),
        alpha_trajectory=alpha_trajectory,
        grad_norm_trajectory=grad_norm_trajectory,
        n_evaluations=int(n_evaluations),
        return_code=int(return_code),
        wall_seconds=float(time.perf_counter() - t0),
        fell_back_to_ds=bool(fell_back_to_ds),
        rigid_force_clamps=int(rigid_force_clamps),
        converged=bool(converged),
        candidate_positions_angstrom=candidate_positions,
        candidate_alphas=candidate_alphas,
        candidate_grad_norms=candidate_grad_norms,
    )
