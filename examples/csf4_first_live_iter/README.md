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

## 1. ssh in and load the standard module stack

```
ssh csf4
module purge
module load python/3.11.3-gcccore-12.3.0
module load python-bundle-pypi/2023.06-gcccore-12.3.0
module load compilers/oneapi/2024.2.0
module load compiler-rt tbb compiler
module load mkl/2024.2
module load gaussian/g16c01_em64t_detectcpu
```

The non-Anaconda Python module is the base for the daemon venv. The Intel
oneAPI + MKL stack is needed to build ARIADNE. Gaussian g16 is the SCF
backend for the INITIAL_GAUSSIAN + GAUSSIAN phases. AIMAll and FEREBUS do
not have modules; you install them yourself (see section 3).

## 2. set up the Python venv

```
python -m venv ~/.venv/ichor-al
source ~/.venv/ichor-al/bin/activate
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

# ARIADNE -- a single pip install puts the oneAPI .so + Python wrapper
# in the venv site-packages, so `import ariadne` works without any
# PYTHONPATH games on login OR worker nodes.
cd ~/projects/ARIADNE
python -m pip install -r requirements-build.txt
export FC=ifx CC=icx CXX=icpx
python -m pip install . --no-build-isolation -v
```

Confirm each backend imports cleanly:

```
python -c "import ichor.core, ichor.hpc, ichor.cli; print('ichor packages OK')"
python -c "import polus.samplers.RS.randomSampling; print('polus RS OK')"
python -c "import pyferebus.executors.trainer; print('pyferebus OK')"
python -c "import ariadne; print('ariadne OK')"
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
    memory_per_core_gb: 4
    parallel_environments:
      serial: [1, 1]
      multicore: [2, 32]

  software:

    aimall:
      executable_path: "~/AIMAll/aimqb.ish"

    ferebus:
      executable_path: "$HOME/.local/bin/ferebus"

    gaussian:
      executable_path: "$g16root/g16/g16"
      modules: ["gaussian/g16c01_em64t_detectcpu"]

    python:
      env_name: "ichor-al"
      python_path: "~/.venv/ichor-al/bin/python"
      modules: ["python/3.11.3-gcccore-12.3.0"]

    ariadne_runtime:
      modules:
        - "compilers/oneapi/2024.2.0"
        - "compiler-rt tbb compiler"
        - "mkl/2024.2"
```

Adjust the AIMAll + FEREBUS paths to match where you actually installed
them in step 3. The repo ships a fuller `ichor_config.yaml` at the repo
root that you can copy + edit.

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

ariadne:
  optimiser: trust_region_qn
  hessian_model: ALMLOF
  max_iter: 50
  gradf_tol: 1.0e-4
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

**oneAPI import error at ARIADNE_ARRAY start**. usually means the
operator did not activate the venv before invoking the daemon. SLURM
copies the submission shell's environment to worker nodes, so as long
as `~/.venv/ichor/bin/activate` was sourced before `ichor-al-daemon
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
