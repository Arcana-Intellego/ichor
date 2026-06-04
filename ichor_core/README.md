# ichor_core

The foundation package of the ICHOR namespace. Pure-Python file I/O and
data structures for computational-chemistry workflows; no cluster
dependencies, runs anywhere.

## What this package does

- **File classes**: reads / writes Gaussian (GJF, .log, .wfn), AIMAll
  (.int, .sum), xyz / .pointsdir / .pointdir, FEREBUS .model and config
  files. See `ichor.core.files` for the full inventory.
- **Atoms / Atom data structures**: lightweight mass / coordinate / type
  containers with feature-calculation helpers (ALF, C-matrix, internal
  coordinates).
- **Adversarial acquisition core**: `ichor.core.adversarial` implements
  the differential adversarial acquisition function used by the
  active-learning daemon (local subspace, chemistry barrier, posterior
  abstraction, ARIADNE-compatible gradient providers).
- **Database adapters**: SQLite + Parquet exporters for large-scale
  IQA datasets.

This is the only ICHOR package safe to install in environments that
never see a cluster (e.g. analysis workstations, CI runners).

## Install

```
python3 -m pip install -e ichor_core[dev]
```

The `[dev]` extra adds the test dependencies (`pytest`, `nbval`).

## Tests

The full unit test suite lives at `ichor_core/tests/`:

```
pytest ichor_core/tests
```

567 tests as of M16; runs in under 5 seconds.

## Documentation

- Full API: <https://ichor.readthedocs.io/en/latest/ichor_core/ichor.core.html>
- Architecture overview: `docs/source/index.rst` in this repo.

## Where to go from here

If you want job submission to a SLURM cluster, see `ichor_hpc`. If you
want the interactive menu system, see `ichor_cli`. If you want the
end-to-end active-learning daemon (which depends on all three),
see `docs/source/active_learning_daemon.rst`.
