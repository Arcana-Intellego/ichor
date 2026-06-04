"""Drive the acquisition's cartesian FD gradient across a process pool.

The core (SeedLocalAdversarialAcquisition) exposes _fd_indices_eps + _fd_single
so this can compute exactly the same per-coordinate central differences in
parallel. On CSF4 the ARIADNE array task forks one pool -- the trained model is
inherited copy-on-write, never pickled -- evaluates the 6N components across the
task's cpus, and gathers them by index, so the result is identical to the serial
loop, just faster. Off-cluster (no fork available) it falls back to serial.
"""
from __future__ import annotations

import numpy as np

# the worker reads these from inherited (forked) memory rather than having the
# model pickled and shipped per task. set in the parent right before the pool
# is created, cleared in the finally.
_WORKER_ACQ = None
_WORKER_ATOMS = None
_WORKER_FLAT = None
_WORKER_EPS = None


def _worker_component(i):
    return _WORKER_ACQ._fd_single(i, _WORKER_FLAT, _WORKER_EPS, _WORKER_ATOMS)


def parallel_cartesian_gradient(acq, atoms, *, workers, chunksize=2):
    """Cartesian FD gradient with the per-coordinate components spread across
    `workers` processes. Falls back to the acquisition's own serial gradient
    when workers <= 1 or the platform has no fork (Windows / spawn-only)."""
    if not workers or int(workers) <= 1:
        return acq._cartesian_finite_difference_gradient(atoms)
    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
    except (ValueError, ImportError):
        return acq._cartesian_finite_difference_gradient(atoms)

    from concurrent.futures import ProcessPoolExecutor

    indices, flat, eps, shape = acq._fd_indices_eps(atoms)
    grad = np.zeros(flat.size, dtype=float)

    global _WORKER_ACQ, _WORKER_ATOMS, _WORKER_FLAT, _WORKER_EPS
    _WORKER_ACQ, _WORKER_ATOMS, _WORKER_FLAT, _WORKER_EPS = acq, atoms, flat, eps
    try:
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as pool:
            for i, g in zip(indices, pool.map(_worker_component, indices, chunksize=chunksize)):
                grad[i] = g
    finally:
        _WORKER_ACQ = _WORKER_ATOMS = _WORKER_FLAT = _WORKER_EPS = None
    return grad.reshape(shape)


def resolve_workers(workers):
    """how many cores to spread the gradient over. an explicit number wins; otherwise we take
    what SLURM gave the task ($SLURM_CPUS_PER_TASK) and fall back to the machine cpu count, so a
    bare resolve_workers(None) on a compute node does the sensible thing. never returns < 1."""
    if workers is not None:
        try:
            return max(1, int(workers))
        except (TypeError, ValueError):
            return 1
    import os
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def _thread_cartesian_gradient(acq, atoms, workers):
    """thread-pool variant -- threads share memory so no fork needed, which is why this is the
    one parallel path that actually runs (not just falls back) off a linux node. same _fd_single
    components as serial, gathered by index, so it matches to the bit."""
    from concurrent.futures import ThreadPoolExecutor

    indices, flat, eps, shape = acq._fd_indices_eps(atoms)
    grad = np.zeros(flat.size, dtype=float)
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        for i, g in zip(indices, pool.map(lambda j: acq._fd_single(j, flat, eps, atoms), indices)):
            grad[i] = g
    return grad.reshape(shape)


def compute_cartesian_gradient(acq, atoms, *, backend="process", workers=None):
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
        return acq._cartesian_finite_difference_gradient(atoms)
    if backend == "thread":
        return _thread_cartesian_gradient(acq, atoms, w)
    # "process" (and anything unrecognised) -> the fork pool, which serial-falls-back off linux.
    return parallel_cartesian_gradient(acq, atoms, workers=w)
