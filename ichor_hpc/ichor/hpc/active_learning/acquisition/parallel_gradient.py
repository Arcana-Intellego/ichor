"""Drive the acquisition's cartesian FD gradient across a process pool.

The core (SeedLocalAdversarialAcquisition) exposes _fd_indices_eps + _fd_single
so this can compute exactly the same per-coordinate central differences in
parallel. On CSF4 the ARIADNE array task forks one pool -- the trained model is
inherited copy-on-write, never pickled -- evaluates the 6N components across the
task's cpus, and gathers them by index, so the result is identical to the serial
loop, just faster. Off-cluster (no fork available) it falls back to serial.
"""
from __future__ import annotations

import os

import numpy as np

# the worker reads these from inherited (forked) memory rather than having the
# model pickled and shipped per task. set in the parent right before the pool
# is created, cleared in the finally.
_WORKER_ACQ = None
_WORKER_ATOMS = None
_WORKER_FLAT = None
_WORKER_EPS = None
_WORKER_ACTIVE_DIRECTIONS = None
_WORKER_OBJECTIVE = "full"
_INSIDE_GRADIENT_WORKER = False
_LAST_GRADIENT_PARALLEL_DIAGNOSTICS = {}


def _worker_component(i):
    return _WORKER_ACQ._fd_single(
        i,
        _WORKER_FLAT,
        _WORKER_EPS,
        _WORKER_ATOMS,
        objective=_WORKER_OBJECTIVE,
    )


def _worker_active_component(i):
    return _WORKER_ACQ._active_fd_single(
        _WORKER_ACTIVE_DIRECTIONS[int(i)],
        _WORKER_FLAT,
        _WORKER_EPS,
        _WORKER_ATOMS,
        objective=_WORKER_OBJECTIVE,
    )


def _worker_init():
    global _INSIDE_GRADIENT_WORKER
    _INSIDE_GRADIENT_WORKER = True
    os.environ["ICHOR_GRADIENT_WORKER"] = "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def inside_gradient_worker() -> bool:
    return bool(
        _INSIDE_GRADIENT_WORKER
        or os.environ.get("ICHOR_GRADIENT_WORKER") == "1"
    )


def _gradient_mp_disabled() -> bool:
    return os.environ.get("ICHOR_DISABLE_GRADIENT_MP") == "1"


def _slurm_cpu_cap() -> int:
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def _n_active_directions(acq) -> int:
    directions = getattr(acq, "mode_directions", None)
    if directions is None:
        return 0
    try:
        return len(directions)
    except TypeError:
        return 0


def _set_last_diagnostics(**payload) -> None:
    global _LAST_GRADIENT_PARALLEL_DIAGNOSTICS
    _LAST_GRADIENT_PARALLEL_DIAGNOSTICS = dict(payload)


def last_gradient_parallel_diagnostics() -> dict:
    return dict(_LAST_GRADIENT_PARALLEL_DIAGNOSTICS)


def parallel_cartesian_gradient(acq, atoms, *, workers, chunksize=2, objective="full"):
    """Cartesian FD gradient with the per-coordinate components spread across
    `workers` processes. Falls back to the acquisition's own serial gradient
    when workers <= 1 or the platform has no fork (Windows / spawn-only)."""
    if not workers or int(workers) <= 1:
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="workers_le_one",
        )
        return acq._cartesian_finite_difference_gradient(atoms, objective=objective)
    if _gradient_mp_disabled():
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="disabled_by_environment",
        )
        return acq._cartesian_finite_difference_gradient(atoms, objective=objective)
    if inside_gradient_worker():
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=True,
            parallel_fallback_reason="inside_gradient_worker",
        )
        return acq._cartesian_finite_difference_gradient(atoms, objective=objective)
    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
    except (ValueError, ImportError):
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="fork_unavailable",
        )
        return acq._cartesian_finite_difference_gradient(atoms, objective=objective)

    from concurrent.futures import ProcessPoolExecutor

    indices, flat, eps, shape = acq._fd_indices_eps(atoms)
    if len(indices) <= 1:
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="too_few_components",
        )
        return acq._cartesian_finite_difference_gradient(atoms, objective=objective)
    workers_used = resolve_workers(workers, n_tasks=len(indices))
    grad = np.zeros(flat.size, dtype=float)

    global _WORKER_ACQ, _WORKER_ATOMS, _WORKER_FLAT, _WORKER_EPS
    global _WORKER_OBJECTIVE
    _WORKER_ACQ, _WORKER_ATOMS, _WORKER_FLAT, _WORKER_EPS = acq, atoms, flat, eps
    _WORKER_OBJECTIVE = str(objective or "full")
    try:
        with ProcessPoolExecutor(
            max_workers=int(workers_used),
            mp_context=ctx,
            initializer=_worker_init,
        ) as pool:
            for i, g in zip(indices, pool.map(_worker_component, indices, chunksize=chunksize)):
                grad[i] = g
        _set_last_diagnostics(
            gradient_backend="process",
            workers_requested=workers,
            workers_used=int(workers_used),
            inside_gradient_worker=False,
            parallel_fallback_reason=None,
        )
    finally:
        _WORKER_ACQ = _WORKER_ATOMS = _WORKER_FLAT = _WORKER_EPS = None
        _WORKER_OBJECTIVE = "full"
    return grad.reshape(shape)


def parallel_active_gradient(acq, atoms, *, workers, chunksize=1, objective="full"):
    """Active-mode FD gradient with independent mode derivatives in a fork pool."""
    if not workers or int(workers) <= 1:
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="workers_le_one",
        )
        return acq._active_finite_difference_gradient(atoms, objective=objective)
    if _gradient_mp_disabled():
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="disabled_by_environment",
        )
        return acq._active_finite_difference_gradient(atoms, objective=objective)
    if inside_gradient_worker():
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=True,
            parallel_fallback_reason="inside_gradient_worker",
        )
        return acq._active_finite_difference_gradient(atoms, objective=objective)
    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
    except (ValueError, ImportError):
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="fork_unavailable",
        )
        return acq._active_finite_difference_gradient(atoms, objective=objective)

    from concurrent.futures import ProcessPoolExecutor

    directions, flat, eps, shape = acq._active_fd_setup(atoms)
    if len(directions) <= 1:
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="too_few_components",
        )
        return acq._active_finite_difference_gradient(atoms, objective=objective)
    workers_used = resolve_workers(workers, n_tasks=len(directions))

    global _WORKER_ACQ, _WORKER_ATOMS, _WORKER_FLAT, _WORKER_EPS
    global _WORKER_ACTIVE_DIRECTIONS, _WORKER_OBJECTIVE
    _WORKER_ACQ = acq
    _WORKER_ATOMS = atoms
    _WORKER_FLAT = flat
    _WORKER_EPS = eps
    _WORKER_ACTIVE_DIRECTIONS = directions
    _WORKER_OBJECTIVE = str(objective or "full")
    try:
        with ProcessPoolExecutor(
            max_workers=int(workers_used),
            mp_context=ctx,
            initializer=_worker_init,
        ) as pool:
            directional_derivs = list(
                pool.map(
                    _worker_active_component,
                    range(len(directions)),
                    chunksize=chunksize,
                )
            )
        _set_last_diagnostics(
            gradient_backend="process",
            workers_requested=workers,
            workers_used=int(workers_used),
            inside_gradient_worker=False,
            parallel_fallback_reason=None,
        )
    finally:
        _WORKER_ACQ = _WORKER_ATOMS = _WORKER_FLAT = _WORKER_EPS = None
        _WORKER_ACTIVE_DIRECTIONS = None
        _WORKER_OBJECTIVE = "full"
    return acq._active_fd_project(directions, directional_derivs, shape)


def resolve_workers(workers, *, n_tasks=None):
    """how many cores to spread the gradient over. an explicit number wins; otherwise we take
    what SLURM gave the task ($SLURM_CPUS_PER_TASK) and fall back to the machine cpu count, so a
    bare resolve_workers(None) on a compute node does the sensible thing. never returns < 1."""
    override = os.environ.get("ICHOR_GRADIENT_WORKERS")
    value = override if override else workers
    if value is not None:
        try:
            out = max(1, int(value))
        except (TypeError, ValueError):
            out = 1
    else:
        out = _slurm_cpu_cap()
    out = min(out, _slurm_cpu_cap())
    if n_tasks is not None:
        try:
            out = min(out, max(1, int(n_tasks)))
        except (TypeError, ValueError):
            pass
    return out


def _thread_cartesian_gradient(acq, atoms, workers, *, objective="full"):
    """thread-pool variant -- threads share memory so no fork needed, which is why this is the
    one parallel path that actually runs (not just falls back) off a linux node. same _fd_single
    components as serial, gathered by index, so it matches to the bit."""
    from concurrent.futures import ThreadPoolExecutor

    indices, flat, eps, shape = acq._fd_indices_eps(atoms)
    grad = np.zeros(flat.size, dtype=float)
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        for i, g in zip(
            indices,
            pool.map(
                lambda j: acq._fd_single(
                    j,
                    flat,
                    eps,
                    atoms,
                    objective=objective,
                ),
                indices,
            ),
        ):
            grad[i] = g
    return grad.reshape(shape)


def compute_cartesian_gradient(
    acq,
    atoms,
    *,
    backend="process",
    workers=None,
    objective="full",
):
    """single entry point the calculator + the runner call to get the cartesian FD gradient.

    backend picks how the per-coordinate components get spread:
      * "serial"  -- one core, just the acquisition's own loop.
      * "thread"  -- a thread pool (works everywhere, windows included).
      * "process" -- a fork pool on linux, transparently serial where fork is unavailable.
    every backend computes the SAME central differences via _fd_single, so the answer is identical
    to the serial loop -- only the speed differs. workers None means "use what SLURM gave us"; any
    backend with <= 1 worker is just the serial path.
    """
    backend = (backend or "serial").lower()
    w = resolve_workers(workers)
    if backend == "serial" or w <= 1:
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason=(
                "requested_serial" if backend == "serial" else "workers_le_one"
            ),
        )
        return acq._cartesian_finite_difference_gradient(atoms, objective=objective)
    if backend == "thread":
        if _gradient_mp_disabled() or inside_gradient_worker():
            _set_last_diagnostics(
                gradient_backend="serial",
                workers_requested=workers,
                workers_used=1,
                inside_gradient_worker=inside_gradient_worker(),
                parallel_fallback_reason=(
                    "inside_gradient_worker"
                    if inside_gradient_worker()
                    else "disabled_by_environment"
                ),
            )
            return acq._cartesian_finite_difference_gradient(atoms, objective=objective)
        _set_last_diagnostics(
            gradient_backend="thread",
            workers_requested=workers,
            workers_used=int(w),
            inside_gradient_worker=False,
            parallel_fallback_reason=None,
        )
        return _thread_cartesian_gradient(acq, atoms, w, objective=objective)
    # "process" (and anything unrecognised) -> the fork pool, which serial-falls-back off linux.
    try:
        return parallel_cartesian_gradient(
            acq,
            atoms,
            workers=w,
            objective=objective,
        )
    except Exception as exc:
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="process_exception:" + type(exc).__name__,
        )
        return acq._cartesian_finite_difference_gradient(atoms, objective=objective)


def compute_active_gradient(
    acq,
    atoms,
    *,
    backend="process",
    workers=None,
    objective="full",
):
    """Single entry point for active-mode FD gradient parallelism.

    Only the process backend does real parallel work. A thread request falls
    back to serial because the posterior/model caches are mutable shared
    Python state; process workers inherit copy-on-write state after fork.
    """
    backend = (backend or "serial").lower()
    w = resolve_workers(workers, n_tasks=_n_active_directions(acq))
    if backend == "serial" or w <= 1:
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason=(
                "requested_serial" if backend == "serial" else "workers_le_one"
            ),
        )
        return acq._active_finite_difference_gradient(atoms, objective=objective)
    if backend == "thread":
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="thread_backend_disabled_for_active_fd",
        )
        return acq._active_finite_difference_gradient(atoms, objective=objective)
    try:
        return parallel_active_gradient(
            acq,
            atoms,
            workers=w,
            objective=objective,
        )
    except Exception as exc:
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="process_exception:" + type(exc).__name__,
        )
        return acq._active_finite_difference_gradient(atoms, objective=objective)
