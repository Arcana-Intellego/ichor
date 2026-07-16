# First live iteration on CSF4 -- a walkthrough

This is the procedure for the first real `--mode live` iteration of the ICHOR
active-learning daemon on Manchester's CSF4 cluster. The goal is a single
water-tetramer iteration that finishes in under two hours and proves the
pipeline end-to-end with real backend output (real Gaussian SCFs, real
AIMAll IQA, real FEREBUS training, real ARIADNE descent, and exact ICHOR
diversity sampling). After this works once, scaling up to a real campaign is just
a matter of bumping the iteration count + the seed pool.

Everything below assumes you have already cloned the ICHOR repo and the
the FEREBUS_CPU and ARIADNE projects into sibling
directories under `~/projects/` (or wherever you keep code). Adjust the
paths as needed.

Recommended install path:

```bash
cd ~/projects/ichor-active-learning
bash scripts/install_ichor_csf.sh --machine csf4 --projects-dir ~/projects \
    --aimall-path ~/AIMAll/aimqb.ish
source scripts/env_ichor_csf.sh --smoke
```

The installer checks download access first, explains how to stage missing
Python/PLUMED/OpenBLAS sources, builds ARIADNE before PLUMED so Intel compiler
variables do not leak into the PLUMED build, installs the xTB/ASE stack, and
backs up `~/ichor_config.yaml` before upserting the CSF4 profile. The remaining
manual commands in this walkthrough are kept as a reference if you need to
debug one component by hand. For day-to-day use after installation, source the
runtime helper in each new CSF4 shell before launching `ichor-cli` or the
daemon.

## 1. ssh in and load the base module stack

```
ssh csf4
module purge
module load python/3.11.3-gcccore-12.3.0
module load python-bundle-pypi/2023.06-gcccore-12.3.0
module load gaussian/g16c01_em64t_detectcpu
```

The non-Anaconda Python module is the base for the daemon venv. Gaussian
g16 is the SCF backend for the INITIAL_GAUSSIAN + GAUSSIAN phases. AIMAll
and FEREBUS do not have modules; you install them yourself (see section 3).

Do not put `CC`, `CXX`, or `FC` exports in `.bashrc`, `.bash_profile`, or
`~/ichor_config.yaml`. They are build-time compiler selectors, not runtime
settings. ARIADNE and PLUMED can live in the same `ichor-csf4` venv, but they
should not be built under the same leaked compiler environment:

- ARIADNE build: Intel oneAPI (`CC=icx`, `CXX=icpx`, `FC=ifx`).
- PLUMED Python wrapper and local PLUMED source build: GCC (`CC=gcc`,
  `CXX=g++`).
- Runtime: unset compiler variables; use the venv plus `PLUMED_KERNEL`.

## 2. set up the Python venv

```
python -m venv ~/.venv/ichor-csf4
source ~/.venv/ichor-csf4/bin/activate
python -m pip install --upgrade pip

# the three ICHOR packages, in dependency order
pip install -e ~/projects/ichor-active-learning/ichor_core
pip install -e ~/projects/ichor-active-learning/ichor_hpc
pip install -e ~/projects/ichor-active-learning/ichor_cli

# the FEREBUS Python wrapper (submission staging only -- runtime is the
# Fortran binary you place under <MACHINE>.software.ferebus in step 4)
pip install -e ~/projects/FEREBUS_CPU/pyferebus --no-deps

```

Build ARIADNE with oneAPI, then immediately clear the compiler variables.
A single pip install puts the ARIADNE `.so` + Python wrapper in the venv
site-packages, so `import ariadne` works without any PYTHONPATH games on
login OR worker nodes.

```
module load compilers/oneapi/2024.2.0
module load compiler-rt tbb compiler
module load mkl/2024.2

cd ~/projects/ARIADNE
python -m pip install -r requirements-build.txt
export CC=icx
export CXX=icpx
export FC=ifx
python -m pip install . --no-build-isolation -v

unset CC CXX FC F77 F90
```

Build and install PLUMED without Conda. The Python package is only the
wrapper; the native kernel is the compiled `libplumedKernel.so`. Use a clean
GCC-side compiler environment here. This avoids the common failure where pip
tries to compile the PLUMED wrapper with a leaked `CC=icx` from the ARIADNE
build and exits with `error: command 'icx' failed: Permission denied`.

```
module purge
module load python/3.11.3-gcccore-12.3.0
module load python-bundle-pypi/2023.06-gcccore-12.3.0
module load gaussian/g16c01_em64t_detectcpu

source ~/.venv/ichor-csf4/bin/activate
unset CC CXX FC F77 F90
export CC=gcc
export CXX=g++

cd ~/projects
echo "5aaf718ac530a1c8df6e0644c22acc84ad4202778106a1d584477057775f2995  plumed-2.10.0.tgz" | sha256sum -c -
tar -xf plumed-2.10.0.tgz
printf '%s\n' "5aaf718ac530a1c8df6e0644c22acc84ad4202778106a1d584477057775f2995" > plumed-2.10.0/.ichor-source.sha256
cd plumed-2.10.0
./configure --prefix=$HOME/opt/plumed-2.10.0 \
    --disable-external-blas \
    --disable-external-lapack \
    --disable-mpi
make -j 4
make install

python -m pip install "plumed==2.10.0"

unset CC CXX FC F77 F90
export PLUMED_KERNEL=$HOME/opt/plumed-2.10.0/lib/libplumedKernel.so
export LD_LIBRARY_PATH=$HOME/opt/plumed-2.10.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

Before verification, load the ARIADNE runtime modules again. This is a
runtime library step, not an instruction to compile anything with Intel.
Keep the compiler variables unset.

```
module load compilers/oneapi/2024.2.0
module load compiler-rt tbb compiler
module load mkl/2024.2

unset CC CXX FC F77 F90
echo "CC=${CC:-<unset>} CXX=${CXX:-<unset>} FC=${FC:-<unset>}"
```

Confirm each backend imports cleanly:

```
python -c "import ichor.core, ichor.hpc, ichor.cli; print('ichor packages OK')"
python -c "import pyferebus.executors.trainer; print('pyferebus OK')"
python -c "import ariadne; print('ariadne OK')"
python -c "import ase; print('ASE OK')"
python -c "from xtb.ase.calculator import XTB; print('XTB ASE OK')"
python -c "import os, plumed; p=plumed.Plumed(kernel=os.environ['PLUMED_KERNEL']); p.finalize(); print('PLUMED OK')"
python -c "from ichor.hpc.runtime_preflight import ensure_xtb_ase_available, ensure_plumed_available; ensure_xtb_ase_available(run_energy=True); ensure_plumed_available(run_ase_smoke=True); print('ASE/xTB/PLUMED preflight OK')"
```

If any of these blow up, fix the import error before going further. The
daemon will refuse to start in `--mode live` if any backend is missing.

## 3. install the AIMAll script + the FEREBUS Fortran binary

Neither AIMAll nor FEREBUS ships as a CSF4 module so you have to put them
on the cluster yourself. The conventions ICHOR's defaults expect are:

* **AIMAll** -- copy the `aimqb.ish` driver script to `~/AIMAll/aimqb.ish`
  (preserve the executable bit). If you have a friend in the group with
  a working `~/AIMAll/`, an `scp -r` is the fastest path.
* **FEREBUS** -- build the Fortran binary from source and drop it into
  `$HOME/.local/bin/ferebus`. The pyferebus repo has the build scripts;
  if you have a colleague's working binary, you can just copy that file
  in (FEREBUS is statically linked).

You can override either path from `~/ichor_config.yaml` (see section 4)
if your install sits elsewhere; the daemon embeds the configured path
directly into every sbatch script it emits, so worker nodes pick it up
automatically.

## 4. declare the backend paths in ~/ichor_config.yaml

ICHOR reads backend executable paths from `~/ichor_config.yaml` at start.

The repository root [`ichor_config.yaml`](../../ichor_config.yaml) is the
single canonical CSF4 profile. Run `scripts/install_ichor_csf.sh --machine
csf4`; the installer copies that profile atomically into
`~/ichor_config.yaml` and overrides only the local Python, AIMAll, FEREBUS and
PLUMED paths. Do not maintain a second profile in this guide.

The canonical CSF4 contract uses login-shell Slurm scripts, 4 GB/core on both
`serial` and `multicore`, and a maximum of 40 shared-memory cores.
`multinode` is deliberately absent because the daemon's scientific
backends are node-local. Inspect the installed values with
`ichor-al-daemon preflight --verbose` before submitting work.

Pass non-default AIMAll or FEREBUS paths to the installer in step 3 so its
atomic profile upsert records them.

The active-learning daemon defaults all backend-specific
`resources.*_mem_per_cpu` fields to `auto`. On CSF4 that resolves to 4G/core
from the profile above. Gaussian live jobs use the Slurm allocation via
`GAUSS_PDEF` and `GAUSS_MDEF`, rather than hard-coding
`%NProcShared` or `%mem` inside every `.gjf`. Gaussian scratch is
daemon-owned under `.DATA/SCRATCH/GAUSSIAN/<phase>/`; successful Gaussian tasks
remove their own scratch directory, while failed tasks keep it for diagnosis.
The active-learning daemon deliberately ignores `software.gaussian.scratch_root`
for Gaussian phases so live campaign runtime files stay inside the campaign
tree. The first smoke uses `resources.array_concurrency_limit: null`, leaving
array concurrency to Slurm. Backend CPU fields default to `auto`; AIMAll combines that with
`aimall.naat: auto` to choose an atom-level parallelism appropriate to the
staged system size. The example AIMAll block uses `naat: auto`, `boaq:
auto_gs2`, and `iasmesh: medium` to keep the first integration pass cheap while
still emitting IQA terms with `encomp: 3`.
For larger shared-cluster campaigns
you can set `resources.array_concurrency_limit` to throttle arrays with Slurm's
`--array=...%N` syntax.

If a path is missing or points at a non-executable file, the daemon
refuses to start with a message naming the offending key. Confirm by
running:

```
python -c "from ichor.hpc.active_learning.daemon.preflight import check_backends; print(check_backends().missing)"
```

You should see `[]` (empty list -- all backends found). If anything is
missing, the message tells you which YAML key to set.

## 5. create the campaign directory and import the trajectory pool

```
mkdir -p ~/scratch/ichor_live_smoke
cd ~/scratch/ichor_live_smoke

# copy the canonical water-tetramer trajectory in. the test fixture has 20
# frames; for a real smoke you want ~200 -- enough to give Phase A diversity
# something to pick a diverse initial set from.
cp ~/projects/ichor-active-learning/ichor_hpc/tests/test_active_learning/fixtures/water_tetramer.xyz pool.xyz
cp ~/projects/ichor-active-learning/examples/csf4_first_live_iter/campaign.yaml campaign.yaml
# Edit campaign.system_name, then inspect the complete init summary.
ichor-al-daemon init
```

The init subcommand populates campaign.yaml when needed, validates the
campaign-local `pool.xyz`, and writes its SHA-pinned manifest under
`.DATA/TRAJECTORY/`. Once imported the SHA is the immutable identity every
iteration uses, so do not edit, delete, or replace the pool unless you really
mean to start a new campaign.

## 6. review the one-iteration campaign config

A minimal `campaign.yaml` (also shipped at
`examples/csf4_first_live_iter/campaign.yaml`):

```yaml
schema_version: 14

campaign:
  system_name: CHANGE_ME_SYSTEM
  max_iterations: 1
  sampling_aggressiveness: 5
  custom_bootstrap: false

runtime:
  poll_interval_seconds: 60

point_allocation:
  bootstrap_training_size: 8
  bootstrap_internal_validation_size: 2
  bootstrap_external_validation_size: 2
  batch_training_size: 3
  batch_internal_validation_size: 1

seed_selection:
  n_seeds_per_iteration: 8
  bulk_fraction: 0.2
  strategy: d_optimal
  d_optimal_degenerate_policy: score_backfill

resources:
  defaults:
    partition: multicore
    walltime_hours: 2
    cpus_per_task: auto
    mem_per_cpu: auto
  diversity:
    walltime_hours: 1
  gaussian:
    walltime_hours: 2
    memory_fraction_of_slurm: 0.85
  aimall:
    walltime_hours: 2
  ariadne:
    walltime_hours: 1
  ferebus:
    walltime_hours: 2
  array_concurrency_limit: null

gaussian:
  method: B3LYP
  basis_set: aug-cc-pVTZ

ariadne:
  optimiser: trust_region_qn
  hessian_model: SCHLEGEL
  max_iter: 50
  convergence:
    mode: fixed
    objective_change_tolerance: 1.0e-6
    gradient_rms_tolerance_per_angstrom: 1.0e-4
    gradient_max_tolerance_per_angstrom: 1.5e-4
    step_rms_tolerance_angstrom: 1.2e-3
    step_max_tolerance_angstrom: 1.8e-3
    consecutive_accepted_steps: 2
  trqn_backtransform_mode: geodesic
  trqn_geodesic_bt_mode: dense
  trqn_geodesic_dt: 1.0e-2
  trqn_geodesic_tol: 1.0e-8
  trqn_bt_ic_tol: 1.0e-6
  trqn_max_backtransform_iter: 50
  trqn_trust_min: 1.0e-4

ferebus:
  kernel: periodic_rbf
  prior_mean_strategy: physical_atomic_iqa
  prior_mean_level_of_theory: auto
  physical_prior_scale: 1.0

runtime:
  poll_sacct_missing_max_ticks: 30
  poll_sacct_unknown_max_ticks: 3
  postprocess_settle_attempts: 3
  postprocess_settle_seconds: 10
  transient_phase_retry_max: 1
```

These are deliberately tight numbers (small initial labelled set, few seeds,
modest ARIADNE iteration budget) so the whole run finishes inside two
hours. The whole point of the smoke is to prove the pipeline works at all;
production settings must be selected explicitly in `campaign.yaml`.

## 7. launch the daemon

```
ichor-al-daemon preflight --campaign-dir . --verbose
ichor-al-daemon preflight --campaign-dir . --verbose --submit-environment-smoke
ichor-al-daemon resource-plan --campaign-dir . --all
ichor-al-daemon start --campaign-dir . --mode live --max-ticks 2000
```

The first start creates immutable execution identity and environment-generation
records. Inspect them at any time without changing campaign state:

```bash
ichor-al-daemon environment-status --campaign-dir .
```

If Python packages, ICHOR source, ARIADNE, FEREBUS, modules, native-library
paths or the active machine profile change, the daemon halts before submission
or postprocess and preserves scheduler ownership. Restore the original
environment to finish already-submitted work. Rebind only at an idle
`SEED_SELECT` or `DONE` boundary, after reconcile reports no active or
inconclusive scheduler ownership and live preflight passes:

```bash
ichor-al-daemon rebind-environment --campaign-dir .
ichor-al-daemon rebind-environment --campaign-dir . --apply
```

Rebinding starts a new calibration eligibility window and invalidates derived
acquisition reference scales. Never use it to postprocess a job submitted under
an earlier environment generation.

The second preflight command is an explicit commissioning gate. It submits one
five-minute, one-core job which imports the configured daemon stack and resolves
Gaussian, AIMAll, FEREBUS and `bc` on a compute node. It does not run scientific
work. Inspect the reported output path and require a successful result before
the first live campaign; ordinary preflight and daemon start never submit this
smoke automatically.
The resource plan is read-only. Before the first submission it should resolve
Phase A from the manifest-verified root `pool.xyz`; later phases may report
`evidence_not_yet_produced` until their producer handoffs exist.

Leave it running. `--max-ticks 2000` is a safety net (the daemon will
not run forever even if something hangs). You can tail the journal in
another shell:

```
ichor-al-daemon journal --campaign-dir . | tail -n 20
```

The smoke config leaves Slurm arrays unthrottled with
`resources.array_concurrency_limit: null`. If you later set a manual `%N`
array throttle, keep `runtime.poll_sacct_missing_max_ticks` generous because
throttled pending rows may not appear in `sacct` immediately on every cluster.

Every state transition lands as a `phase_transition` event; every
successful sbatch postprocess lands as a `phase_succeeded_live` event.

## 8. expected outcomes

When the daemon stops normally (state.phase == DONE), check:

**state.json** should show:

- `phase: DONE`
- `iteration: 1` -- bootstrap is iteration 0; the first adversarial sampling
  loop is active iteration 1. A two-iteration run exits with `iteration: 2`.
- `reference_data_version: 1` (INITIAL + 1 iteration of APPEND)
- `models_version: 1` (INITIAL_FEREBUS + 1 iteration of FEREBUS)
- `reference_scales` populated with five real-float keys (energy, force,
  omega, anh, anh_std)

**journal.ndjson** should contain at least one `phase_succeeded_live`
event per SBATCH phase. count them:

```
grep phase_succeeded_live .DATA/ACTIVE_LEARNING/journal.ndjson | wc -l
```

expect at least 9 (one per SBATCH phase). more is also fine -- some
phases run twice.

**per-seed result.json** files should be at
`ACTIVE_LEARNING/iteration-000001/ariadne/seeds/seed-000001/result.json`
(and subsequent one-based seed directories). Each
should have:

- `wall_seconds` non-zero (real ARIADNE descent took time)
- `whitened_distance_final` is a positive finite float (real metric
  rather than the synthetic alpha-delta fallback)
- `alpha_trajectory` is non-empty (the descent made at least one step)

**sampling outputs** under the active iteration:

- `phase_b/selected.xyz` exists with the expected frame count
  (`batch_training_size + batch_internal_validation_size` = 4)
- `allocation/POINT_ALLOCATION.json` records three training slots and one internal
  validation slot; failed QM candidates consume the finite Phase-B reserve
  without changing those slot assignments
- `phase_b/SELECTION.json` binds both XYZ files, records the deduplication
  decision, and points at the exact allocation and sampling-protocol manifests
- every `ariadne/seeds/seed-NNNNNN/trajectory/` contains `trajectory.xyz`,
  `metrics.jsonl`, and `MANIFEST.json`; the optional `trace.jsonl` carries the
  detailed optimiser event stream
- `ITERATION_MANIFEST.json` seals the exact recursive inventory and links to
  the preceding active iteration manifest when one exists

## 9. tear-down and restart

If the run failed mid-iteration:

```
# inspect what went wrong first
ichor-al-daemon journal --campaign-dir . | tail -n 50
ichor-al-daemon status --campaign-dir .

# if state.json is inconsistent, corrupt, or HALTED:
ichor-al-daemon reconcile --campaign-dir .
# Review the recovery target, contract checks, protected artefacts, and config
# decision. Apply only when the dry-run report is safe:
ichor-al-daemon reconcile --campaign-dir . --apply
```

Do not move `state.json.proposed` over `state.json` manually. `--apply` holds the
daemon lock, rechecks scheduler intent evidence, archives safe stale staging,
updates the config lock, and verifies the final recovery contract.

`DONE` is a successful terminal lifecycle, not a stopped daemon. To extend a
completed campaign deliberately, first increase `campaign.max_iterations`, run
and review `reconcile --apply`, then use the explicit reopen command:

```
ichor-al-daemon resume --campaign-dir . --reopen-converged --max-ticks 2000
```

Never delete versioned campaign directories to force a rerun. For a genuinely
fresh run, create a new campaign directory and initialise it from the intended
pool/bootstrap inputs. A failed mandatory custom geometry or an exhausted
immutable replacement reserve cannot be repaired in place; start a new campaign
with valid bootstrap inputs, a larger reserve, or a smaller required batch.

## 10. common failure modes

**ARIADNE descent times out**. The smoke config caps `ariadne.max_iter`
at 50; if individual seed descents still take too long, lower it. Real
campaigns use 200; we use 50 here just for the smoke.

**FEREBUS training fails to converge**. usually means the initial training
set is too small or too clustered. Increase
`point_allocation.bootstrap_training_size` and try again, preserving suitable
internal/external validation counts. Also check that the Gaussian + AIMAll outputs in the pointdirs look
reasonable.

**Phase A diversity picks all-coincident frames**. happens when the input
trajectory is too short (under ~20 frames). use a longer trajectory; the
test fixture only has 20 frames which is OK for the smoke but production
wants 200+.

**`sacct_rows_missing_but_squeue_active` appears in the journal**. this means
CSF accounting has not emitted all expected task rows yet, but `squeue` still
shows active Slurm array jobs. The daemon keeps polling in this state.

**oneAPI import error at ARIADNE_ARRAY start**. usually means the
user did not activate the venv before invoking the daemon. SLURM
copies the submission shell's environment to worker nodes, so as long
as `~/.venv/ichor-csf4/bin/activate` was sourced before `ichor-al-daemon
start`, the worker will find ariadne in the venv site-packages.
Source the venv and re-launch.

**`phase_b/selected.xyz` is empty**. This probably means all the ARIADNE
descents collapsed to the same geometry. Inspect `ACTIVE_LEARNING/
iteration-000001/ariadne/seeds/seed-*/result.json` and the corresponding
`trajectory/trajectory.xyz` files to see if `whitened_distance_final` is suspiciously
small across the board. if so, the acquisition might be miscalibrated for
your system; try a different `phase_b.descriptor` or relax the
`anti_overlap.*` thresholds.

## 11. when this works end-to-end

Congratulations -- you have just run a real one-iteration ICHOR active-
learning campaign on CSF4. Real next steps from here:

- bump `max_iterations` to something like 20 or 50 for a real campaign.
- tune the explicit spectral and subspace fields only after reviewing the
  campaign's acquisition diagnostics.
- tune `geometry_novelty.fallback_scale_angstrom` if you find later
  iterations picking near-duplicates of earlier training points; Phase B
  derives its minimum separation from the geometry novelty protocol.
- pay attention to the `alpha_history` field in state.json -- the
  STOP_CHECK alpha-trend rules use it to decide when to terminate
  automatically.
