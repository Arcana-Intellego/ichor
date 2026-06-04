# ichor_hpc

ICHOR's interface to high-performance compute clusters. Wraps SLURM
submission, sacct polling, the active-learning daemon orchestration,
and the per-backend bridges to Gaussian / AIMAll / FEREBUS / ARIADNE /
POLUS.

Depends on `ichor_core`.

## What this package does

- **Cluster submission**: `ichor.hpc.submit` builds and submits sbatch
  scripts, parses sbatch output, and polls sacct for terminal status.
  Auto-detects SGE (CSF3) vs SLURM (CSF4) where applicable; the
  active-learning daemon is SLURM-only by design.
- **Active-learning daemon**: `ichor.hpc.active_learning` is the
  end-to-end campaign orchestrator -- finite-state machine over campaign
  phases, atomic training-set versioning, journal-based recovery, three
  execution modes (`--dry-run`, `--mock-ariadne`, `--live`), and the
  trajectory-pool + per-pointdir provenance ledger. Console-script:
  `ichor-al-daemon`.
- **Backend wrappers**: `ichor.hpc.active_learning.acquisition`
  (ARIADNE + POLUS), `ichor.hpc.active_learning.submit.pyferebus_wrap`,
  and the bridge to `ichor.core.adversarial.acquisition`.
- **Active-learning sampling primitives**:
  `ichor.hpc.active_learning.sampling` (descriptors, outlier filter,
  stratified split, anti-overlap).

## Install

```
python3 -m pip install -e ichor_core[dev]
python3 -m pip install -e ichor_hpc
```

(Install order matters: `ichor_hpc` imports `ichor.core` and needs the
namespace package wired up first.)

## Tests

```
pytest ichor_hpc/tests -m "not live"
```

452 tests as of M16 (~60s); live-mode tests (10 of them) need real
cluster backends and are deselected by the `not live` filter for fast
local iteration. The marker is registered in the top-level
`pytest.ini`.

To run live tests on a real cluster:

```
pytest ichor_hpc/tests -m live
```

## Documentation

- **Daemon user guide**: `docs/source/active_learning_daemon.rst`
  (quickstart, mode reference, config schema, presets, cluster
  prerequisites, recovery, architecture sketch).
- **Bundled example**: `examples/dry_run_water_tetramer/` runs a
  complete 2-iteration dry-run campaign in ~30 seconds.
- Full API: <https://ichor.readthedocs.io/en/latest/ichor_hpc/ichor.hpc.html>.

## Console scripts

`ichor_hpc` ships one daemon CLI:

- `ichor-al-daemon` -- active-learning campaign driver.
  Subcommands: `start`, `stop`, `status`, `resume`, `reconcile`,
  `journal`, `import-pool`. Run `ichor-al-daemon --help` for full usage.

## Where to go from here

For an interactive menu wrapper around all of this, install `ichor_cli`
and run `ichor`.
