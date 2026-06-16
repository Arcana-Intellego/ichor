# First live iteration on CSF4 -- a walkthrough

This is the procedure for the first real `--live` iteration of the ICHOR
active-learning daemon on Manchester's CSF4 cluster. The goal is a single
water-tetramer iteration that finishes in under two hours and proves the
pipeline end-to-end with real backend output (real Gaussian SCFs, real
AIMAll IQA, real FEREBUS training, real ARIADNE descent, real POLUS
sub-sample). After this works once, scaling up to a real campaign is just
a matter of bumping the iteration count + the seed pool.

Everything below assumes you have already cloned the ICHOR repo and the
three external packages (POLUS, pyferebus, ARIADNE) into sibling
directories under `~/projects/` (or wherever you keep code). Adjust the
paths as needed.

Recommended install path:

```bash
cd ~/projects/ichor-active-learning
bash scripts/install_ichor_csf.sh --machine csf4 --projects-dir ~/projects \
    --aimall-path ~/AIMAll/aimqb.ish
source scripts/env_ichor_csf.sh csf4 --smoke
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

# the diversity sampler subtree used by the daemon
pip install -e ~/projects/POLUS/polus_core_subpackage --no-deps

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
tar -xf plumed-2.10.0.tgz
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
python -c "import polus.samplers.RS.randomSampling; print('polus RS OK')"
python -c "import pyferebus.executors.trainer; print('pyferebus OK')"
python -c "import ariadne; print('ariadne OK')"
python -c "import ase; print('ASE OK')"
python -c "from xtb.ase.calculator import XTB; print('XTB ASE OK')"
python -c "import os, plumed; p=plumed.Plumed(kernel=os.environ['PLUMED_KERNEL']); p.finalize(); print('PLUMED OK')"
python -c "from ichor.hpc.runtime_preflight import ensure_xtb_ase_available, ensure_plumed_available; ensure_xtb_ase_available(run_energy=True); ensure_plumed_available(run_ase_smoke=True); print('ASE/xTB/PLUMED preflight OK')"
```

If any of these blow up, fix the import error before going further. The
daemon will refuse to start in `--live` mode if any backend is missing.

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
A minimal CSF4 entry that the daemon will be happy with:

```yaml
# ~/ichor_config.yaml

csf4:

  hpc:
    scheduler: slurm
    max_array_task_id: 25000
    memory_per_core_gb: 4
    memory_per_core_gb_by_partition:
      serial: 4
      multicore: 4
      multinode: 4
    parallel_environments:
      serial: [1, 1]
      multicore: [2, 32]

  software:

    aimall:
      executable_path: "~/AIMAll/aimqb.ish"

    ferebus:
      executable_path: "$HOME/.local/bin/ferebus"
      pyferebus_platform: "CSF4"

    gaussian:
      executable_path: "$g16root/g16/g16"
      modules: ["gaussian/g16c01_em64t_detectcpu"]

    python:
      env_name: "ichor-csf4"
      python_path: "~/.venv/ichor-csf4/bin/python"
      modules: ["python/3.11.3-gcccore-12.3.0"]

    plumed:
      kernel_path: "$HOME/opt/plumed-2.10.0/lib/libplumedKernel.so"
      library_path: "$HOME/opt/plumed-2.10.0/lib"
      modules: []

    ariadne_runtime:
      modules:
        - "compilers/oneapi/2024.2.0"
        - "compiler-rt tbb compiler"
        - "mkl/2024.2"
```

Adjust the AIMAll + FEREBUS paths to match where you actually installed
them in step 3. The repo ships a fuller `ichor_config.yaml` at the repo
root that you can copy + edit.

The active-learning daemon defaults to `resources.mem_per_cpu: auto`. On CSF4
that resolves to 4G/core from the profile above. Gaussian live jobs use the
Slurm allocation via `GAUSS_PDEF` and `GAUSS_MDEF` by default, rather than
hard-coding `%NProcShared` or `%mem` inside every `.gjf`. Gaussian scratch is
daemon-owned under `.DATA/SCRATCH/GAUSSIAN/<phase>/`; successful Gaussian tasks
remove their own scratch directory, while failed tasks keep it for diagnosis.
For a cautious first smoke leaves `resources.array_concurrency_limit: null` so
Gaussian/AIMAll arrays can run with as much concurrency as Slurm policy and
cluster load allow. It also sets `resources.aimall_cpus_per_task: 8` to avoid
making AIMAll the first-smoke bottleneck. For larger shared-cluster campaigns
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
# frames; for a real smoke you want ~200 -- enough to give POLUS Phase-A
# something to pick a diverse initial set from.
cp ~/projects/ichor-active-learning/ichor_hpc/tests/test_active_learning/fixtures/water_tetramer.xyz pool.xyz

ichor-al-daemon import-pool --campaign-dir . --source pool.xyz
```

The import-pool subcommand copies the trajectory into
`.DATA/TRAJECTORY/pool.xyz` and writes a SHA-pinned manifest next to it.
Once imported the SHA is the anchor every iteration descends from, so do
not delete or re-import the pool unless you really mean to start a new
campaign.

## 6. drop in a one-iteration campaign config

A minimal `campaign.yaml` (also shipped at
`examples/csf4_first_live_iter/campaign.yaml`):

```yaml
schema_version: 2

max_iterations: 1
poll_interval_seconds: 60

initial_train_size: 8
initial_val_size: 2

batch_sizing:
  policy: linear
  floor: 4
  cap: 8

seed_selection:
  n_seeds_per_iteration: 4
  bulk_fraction: 0.5

resources:
  partition: multicore
  walltime_hours: 2
  mem_per_cpu: auto
  cpus_per_task: 2
  ntasks: 1
  aimall_cpus_per_task: 8
  ariadne_cpus_per_task: 8
  array_concurrency_limit: null

gaussian:
  nproc: 2
  memory_mode: slurm_env

ariadne:
  optimiser: trust_region_qn
  hessian_model: ALMLOF
  max_iter: 50
  gradf_tol: 1.0e-4

runtime:
  poll_sacct_missing_max_ticks: 30
  poll_sacct_unknown_max_ticks: 3
  postprocess_settle_attempts: 3
  postprocess_settle_seconds: 10
  transient_phase_retry_max: 1
```

These are deliberately tight numbers (small initial sample, few seeds,
modest ARIADNE iteration budget) so the whole run finishes inside two
hours. The whole point of the smoke is to prove the pipeline works at
all -- the production sizes for a real campaign live in the spectroscopy
or thermodynamics preset.

## 7. launch the daemon

```
ichor-al-daemon start --live --campaign-dir . --max-ticks 2000
```

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
- `iteration: 0` -- ICHOR iteration numbers are zero-indexed, so a
  `max_iterations: 1` campaign exits with `iteration: 0`. A two-
  iteration run would exit with `iteration: 1`, etc.
- `training_set_version: 1` (INITIAL + 1 iteration of APPEND)
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
`7_ACTIVE_LEARNING/iteration-0000/pool/seed_NNNN/result.json`. each
should have:

- `wall_seconds` non-zero (real ARIADNE descent took time)
- `whitened_distance_final` is a positive finite float (real metric
  rather than the synthetic alpha-delta fallback)
- `alpha_trajectory` is non-empty (the descent made at least one step)

**sampling outputs** at the iteration directory:

- `phase_b_SAMPLE.xyz` exists with the expected frame count
  (`batch_sizing.floor` = 4)
- `phase_b_dedup.json` exists; `n_dropped` is 0 by default (min_separation
  = 0.0 means the filter is off)

## 9. tear-down and restart

If the run failed mid-iteration:

```
# inspect what went wrong first
ichor-al-daemon journal --campaign-dir . | tail -n 50
ichor-al-daemon status --campaign-dir .

# if state.json is inconsistent / corrupt:
ichor-al-daemon reconcile --campaign-dir .
# review the proposed state, then promote manually:
# mv .DATA/ACTIVE_LEARNING/state.json.proposed .DATA/ACTIVE_LEARNING/state.json
```

For a clean re-run (wipes all on-disk state, keeps the trajectory pool):

```
rm -rf .DATA/ACTIVE_LEARNING
rm -rf 3_DIVERSITY_SAMPLING 5_TRAINING 6_TRAINED_MODELS 7_ACTIVE_LEARNING
# the trajectory pool at .DATA/TRAJECTORY/ is preserved
ichor-al-daemon start --live --campaign-dir . --max-ticks 2000
```

## 10. common failure modes

**ARIADNE descent times out**. The smoke config caps `ariadne.max_iter`
at 50; if individual seed descents still take too long, lower it. Real
campaigns use 200; we use 50 here just for the smoke.

**FEREBUS training fails to converge**. usually means the initial training
set is too small or too clustered. bump `initial_train_size` and try
again. or check that the Gaussian + AIMAll outputs in the pointdirs look
reasonable.

**Phase A POLUS picks all-coincident frames**. happens when the input
trajectory is too short (under ~20 frames). use a longer trajectory; the
test fixture only has 20 frames which is OK for the smoke but production
wants 200+.

**`sacct_rows_missing_but_squeue_active` appears in the journal**. this means
CSF accounting has not emitted all expected task rows yet, but `squeue` still
shows active Slurm array jobs. The daemon keeps polling in this state.

**oneAPI import error at ARIADNE_ARRAY start**. usually means the
operator did not activate the venv before invoking the daemon. SLURM
copies the submission shell's environment to worker nodes, so as long
as `~/.venv/ichor-csf4/bin/activate` was sourced before `ichor-al-daemon
start --live`, the worker will find ariadne in the venv site-packages.
Source the venv and re-launch.

**phase_b_SAMPLE.xyz is empty**. probably means all the ARIADNE descents
collapsed to the same geometry. inspect `7_ACTIVE_LEARNING/iteration-0000/
pool/seed_*/result.json` to see if `whitened_distance_final` is suspiciously
small across the board. if so, the acquisition might be miscalibrated for
your system; try a different `phase_b.descriptor` or relax the
`anti_overlap.*` thresholds.

## 11. when this works end-to-end

Congratulations -- you have just run a real one-iteration ICHOR active-
learning campaign on CSF4. Real next steps from here:

- bump `max_iterations` to something like 20 or 50 for a real campaign.
- swap in the `spectroscopy_focused` preset if your downstream target is
  vibrational spectra (the preset bumps the subspace dim and switches the
  mode-weighting policy to inverse-frequency).
- consider enabling `phase_b.min_separation` to a modest value (0.05 to
  0.1 Angstrom) if you find later iterations picking near-duplicates of
  earlier training points.
- pay attention to the `alpha_history` field in state.json -- the
  STOP_CHECK alpha-trend rules use it to decide when to terminate
  automatically.
