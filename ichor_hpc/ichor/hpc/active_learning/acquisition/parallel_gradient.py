"""Task-owned process workers for active-subspace acquisition gradients."""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np


_WORKER_ACQUISITION = None
_INSIDE_GRADIENT_WORKER = False
_LAST_GRADIENT_PARALLEL_DIAGNOSTICS = {}
_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _worker_active_payload(payload):
    direction, flat, step, atoms = payload
    posterior = getattr(_WORKER_ACQUISITION, "posterior", None)
    before_posterior = dict(
        getattr(posterior, "diagnostics", {})
    )
    before_acquisition = dict(
        getattr(_WORKER_ACQUISITION, "performance_diagnostics", {})
    )
    value = _WORKER_ACQUISITION._active_fd_pair(
        np.asarray(direction, dtype=float),
        np.asarray(flat, dtype=float),
        float(step),
        atoms,
    )
    counters = {}
    for prefix, before, after in (
        (
            "posterior_",
            before_posterior,
            getattr(posterior, "diagnostics", {}),
        ),
        (
            "acquisition_",
            before_acquisition,
            getattr(_WORKER_ACQUISITION, "performance_diagnostics", {}),
        ),
    ):
        for key, current in after.items():
            if isinstance(current, (int, np.integer)):
                counters[prefix + str(key)] = int(current) - int(before.get(key, 0))
    return float(value), counters


def _worker_ping(value):
    return int(value)


def _worker_initialise():
    global _INSIDE_GRADIENT_WORKER
    _INSIDE_GRADIENT_WORKER = True
    os.environ["ICHOR_GRADIENT_WORKER"] = "1"
    for name in _THREAD_ENV_NAMES:
        os.environ[name] = "1"


@contextmanager
def _capped_worker_thread_env():
    """Temporarily prevent nested BLAS/OpenMP parallelism before forking."""
    previous = {name: os.environ.get(name) for name in _THREAD_ENV_NAMES}
    try:
        for name in _THREAD_ENV_NAMES:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def inside_gradient_worker() -> bool:
    return bool(
        _INSIDE_GRADIENT_WORKER
        or os.environ.get("ICHOR_GRADIENT_WORKER") == "1"
    )


def _gradient_mp_disabled() -> bool:
    return os.environ.get("ICHOR_DISABLE_GRADIENT_MP") == "1"


def _scheduler_cpu_cap() -> int:
    # Resource resolution may allocate CPUs for memory only. Scientific
    # workers are capped separately through ICHOR_ACTIVE_WORKERS.
    raw = (
        os.environ.get("ICHOR_ACTIVE_WORKERS")
        or os.environ.get("ICHOR_SCHEDULER_CPUS")
        or os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("NSLOTS")
    )
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def _slurm_cpu_cap() -> int:
    """Compatibility alias for callers predating scheduler-neutral workers."""
    return _scheduler_cpu_cap()


def _n_active_directions(acquisition) -> int:
    directions = getattr(acquisition, "mode_directions", None)
    if directions is None:
        return 0
    try:
        return len(directions)
    except TypeError:
        return 0


def resolve_workers(workers, *, n_tasks=None) -> int:
    """Resolve active workers without exceeding Slurm or task cardinality."""
    override = os.environ.get("ICHOR_GRADIENT_WORKERS")
    value = override if override else workers
    if value is None:
        resolved = _scheduler_cpu_cap()
    else:
        try:
            resolved = max(1, int(value))
        except (TypeError, ValueError):
            resolved = 1
    resolved = min(resolved, _scheduler_cpu_cap())
    if n_tasks is not None:
        try:
            resolved = min(resolved, max(1, int(n_tasks)))
        except (TypeError, ValueError):
            pass
    return int(resolved)


def _set_last_diagnostics(**payload) -> None:
    global _LAST_GRADIENT_PARALLEL_DIAGNOSTICS
    _LAST_GRADIENT_PARALLEL_DIAGNOSTICS = dict(payload)


def last_gradient_parallel_diagnostics() -> dict:
    return dict(_LAST_GRADIENT_PARALLEL_DIAGNOSTICS)


class ActiveGradientWorkerPool:
    """One reusable fork pool owned by one ARIADNE seed task.

    The fitted acquisition is inherited copy-on-write once. Each gradient
    call sends only the current geometry and active directions. Platforms
    without ``fork`` retain the exact serial implementation.
    """

    def __init__(self, acquisition, *, workers=None, chunksize=1):
        self._acquisition = acquisition
        self._chunksize = max(1, int(chunksize))
        self._pool = None
        self._closed = False
        self._owns_worker_acquisition = False
        self._fallback_reason = None
        self._workers_requested = workers
        self._workers_used = resolve_workers(
            workers,
            n_tasks=_n_active_directions(acquisition),
        )
        if self._workers_used <= 1:
            self._fallback_reason = "workers_le_one"
            return
        if _gradient_mp_disabled() or inside_gradient_worker():
            self._fallback_reason = (
                "inside_gradient_worker"
                if inside_gradient_worker()
                else "disabled_by_environment"
            )
            self._workers_used = 1
            return
        try:
            import multiprocessing as mp

            context = mp.get_context("fork")
        except (ValueError, ImportError):
            self._fallback_reason = "fork_unavailable"
            self._workers_used = 1
            return

        from concurrent.futures import ProcessPoolExecutor

        global _WORKER_ACQUISITION
        if _WORKER_ACQUISITION is not None:
            raise RuntimeError("another acquisition gradient pool is already active")
        _WORKER_ACQUISITION = acquisition
        self._owns_worker_acquisition = True
        try:
            with _capped_worker_thread_env():
                self._pool = ProcessPoolExecutor(
                    max_workers=int(self._workers_used),
                    mp_context=context,
                    initializer=_worker_initialise,
                )
                # Force worker creation while the immutable acquisition is
                # installed in the fork image and native threads are capped.
                list(
                    self._pool.map(
                        _worker_ping,
                        range(int(self._workers_used)),
                        chunksize=1,
                    )
                )
        except Exception:
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
            if self._owns_worker_acquisition:
                _WORKER_ACQUISITION = None
                self._owns_worker_acquisition = False
            raise

    @property
    def diagnostics(self) -> dict:
        return {
            "gradient_backend": (
                "process_persistent" if self._pool is not None else "serial"
            ),
            "workers_requested": self._workers_requested,
            "workers_used": int(self._workers_used),
            "inside_gradient_worker": inside_gradient_worker(),
            "parallel_fallback_reason": self._fallback_reason,
            "gradient_pool_reused": bool(self._pool is not None),
        }

    def gradient(self, atoms):
        if self._closed:
            raise RuntimeError("active gradient worker pool is closed")
        directions, flat, step, shape = self._acquisition._active_fd_setup(atoms)
        if self._pool is None or len(directions) <= 1:
            reason = self._fallback_reason or "too_few_components"
            _set_last_diagnostics(
                **{**self.diagnostics, "parallel_fallback_reason": reason}
            )
            return self._acquisition._active_finite_difference_gradient(atoms)
        payloads = [
            (
                np.asarray(direction, dtype=float),
                np.asarray(flat, dtype=float),
                float(step),
                atoms,
            )
            for direction in directions
        ]
        try:
            worker_results = list(
                self._pool.map(
                    _worker_active_payload,
                    payloads,
                    chunksize=self._chunksize,
                )
            )
        except Exception as exc:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
            global _WORKER_ACQUISITION
            if self._owns_worker_acquisition:
                _WORKER_ACQUISITION = None
                self._owns_worker_acquisition = False
            self._fallback_reason = (
                "persistent_process_exception:" + type(exc).__name__
            )
            self._workers_used = 1
            _set_last_diagnostics(
                gradient_backend="serial",
                workers_requested=self._workers_requested,
                workers_used=1,
                inside_gradient_worker=False,
                parallel_fallback_reason=self._fallback_reason,
                gradient_pool_reused=False,
            )
            return self._acquisition._active_finite_difference_gradient(atoms)
        derivatives = [float(value) for value, _ in worker_results]
        aggregate = {}
        for _, counters in worker_results:
            for key, value in counters.items():
                aggregate[key] = int(aggregate.get(key, 0)) + int(value)
        _set_last_diagnostics(**self.diagnostics, **aggregate)
        return self._acquisition._active_fd_project(
            directions,
            derivatives,
            shape,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        global _WORKER_ACQUISITION
        if self._owns_worker_acquisition:
            if _WORKER_ACQUISITION is self._acquisition:
                _WORKER_ACQUISITION = None
            self._owns_worker_acquisition = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def compute_active_gradient(acquisition, atoms, *, backend="process", workers=None):
    """Evaluate the mature objective in active-subspace directions only."""
    selected_backend = str(backend or "serial").lower()
    if selected_backend not in {"serial", "process"}:
        raise ValueError("active gradient backend must be 'serial' or 'process'")
    if selected_backend == "serial":
        _set_last_diagnostics(
            gradient_backend="serial",
            workers_requested=workers,
            workers_used=1,
            inside_gradient_worker=inside_gradient_worker(),
            parallel_fallback_reason="requested_serial",
        )
        return acquisition._active_finite_difference_gradient(atoms)
    with ActiveGradientWorkerPool(acquisition, workers=workers) as pool:
        return pool.gradient(atoms)


__all__ = [
    "ActiveGradientWorkerPool",
    "compute_active_gradient",
    "inside_gradient_worker",
    "last_gradient_parallel_diagnostics",
    "resolve_workers",
]
