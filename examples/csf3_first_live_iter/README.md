# First live iteration on CSF3 -- a walkthrough

This is the CSF3 equivalent of the CSF4 live smoke. It proves one real
active-learning iteration with Slurm, Gaussian, AIMAll, FEREBUS, ARIADNE, and
ICHOR-owned exact diversity sampling, while keeping Python, FEREBUS, ARIADNE,
and PLUMED outside the repo.

Recommended install path:

```bash
cd ~/projects/ichor-active-learning
bash scripts/install_ichor_csf.sh --machine csf3 --projects-dir ~/projects \
    --aimall-path ~/AIMAll/aimqb.ish
source scripts/env_ichor_csf.sh csf3 --smoke
```

The installer uses private CPython 3.11 by default, checks download access,
prints exact staging instructions when downloads are blocked, builds ARIADNE,
FEREBUS, and PLUMED with `--jobs 4` by default, verifies xTB/ASE, and backs up
`~/ichor_config.yaml` before upserting the CSF3 profile. The manual sections
below are kept as a fallback and for troubleshooting individual components.
For day-to-day use after installation, the sourced `env_ichor_csf.sh` helper is
the expected way to load oneAPI/MKL runtime modules and avoid ARIADNE `libmkl`
import errors.

## 1. Build private CPython 3.11

CSF3 currently exposes `python/3.13.1`, but the ICHOR scientific stack is safer
on Python 3.11 because RDKit, xTB, PLUMED, f90wrap, NumPy, and ARIADNE include
compiled components.

```bash
module purge
module load compilers/gcc/13.3.0
module load tools/gcc/cmake/3.31.6
module load libs/gcc/openssl/1.1.1w

mkdir -p ~/src ~/opt
cd ~/src
wget https://www.python.org/ftp/python/3.11.15/Python-3.11.15.tgz
tar -xzf Python-3.11.15.tgz
cd Python-3.11.15

OPENSSL_PREFIX="${EBROOTOPENSSL:-${OPENSSL_ROOT_DIR:-$(dirname "$(dirname "$(which openssl)")")}}"
./configure \
  --prefix=$HOME/opt/python-3.11.15 \
  --enable-shared \
  --with-ensurepip=install \
  --with-openssl="$OPENSSL_PREFIX" \
  --with-openssl-rpath=auto
make -j 8
make install

export LD_LIBRARY_PATH=$HOME/opt/python-3.11.15/lib:$LD_LIBRARY_PATH
$HOME/opt/python-3.11.15/bin/python3.11 -c "import ssl; print(ssl.OPENSSL_VERSION)"
$HOME/opt/python-3.11.15/bin/python3.11 -m venv ~/.venv/ichor-csf3
source ~/.venv/ichor-csf3/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

If direct download is blocked, download the tarball locally from `python.org`,
copy it to CSF3, and build it on CSF3.
The `import ssl` check is required: if it fails, `pip` cannot use PyPI over
HTTPS and the Python install must be rebuilt with the OpenSSL module loaded.

## 2. Install ICHOR and sibling Python packages

```bash
source ~/.venv/ichor-csf3/bin/activate
export LD_LIBRARY_PATH=$HOME/opt/python-3.11.15/lib:$LD_LIBRARY_PATH

pip install -e ~/projects/ichor-active-learning/ichor_core
pip install -e ~/projects/ichor-active-learning/ichor_hpc
pip install -e ~/projects/ichor-active-learning/ichor_cli

pip install -e ~/projects/FEREBUS_CPU/pyferebus --no-deps
```

## 3. Build ARIADNE with CSF3 oneAPI

```bash
module purge
module load compilers/intel/oneapi/2025.0.1
module load umf compiler-rt tbb compiler
module load mkl/2025.0
which icx icpx ifx

source ~/.venv/ichor-csf3/bin/activate
export LD_LIBRARY_PATH=$HOME/opt/python-3.11.15/lib:$LD_LIBRARY_PATH

cd ~/projects/ARIADNE
python -m pip install -r requirements-build.txt
export CC="$(command -v icx)"
export CXX="$(command -v icpx)"
export FC="$(command -v ifx)"
python -m pip install . --no-build-isolation -v \
    --config-settings=cmake.define.ARIADNE_SAFE_IFX_FLAGS=ON
unset CC CXX FC F77 F90
```

The runtime jobs must load the same oneAPI/MKL modules, but compiler variables
should stay unset after the build.

## 4. Install native backends outside the repo

Confirm CSF3 modules and scheduler tools:

```bash
module avail python
module search gaussian
module search aimall
which sbatch sacct squeue bc
```

The default CSF3 example profile uses Gaussian 09 because that module is
protected by the older `gaussian` group. Gaussian 16 on CSF3 is protected by
the separate `gaussian16` group; switch the module/path only if RI has granted
that access. AIMAll may be available centrally or via a group/user install;
record the actual `aimqb.ish` path in `~/ichor_config.yaml`.

Build or copy the FEREBUS Fortran executable outside the repo:

```bash
mkdir -p ~/.local/bin
# install/copy the working binary here:
ls -l ~/.local/bin/ferebus
```

PLUMED/xTB/ASE/RDKit are needed for the full CLI/metadynamics stack. They stay
in the venv and under `$HOME/opt`; no binaries are committed to this repo.

## 5. Configure `~/ichor_config.yaml`

```yaml
csf3:
  hpc:
    scheduler: slurm
    jobscript_shebang: "#!/bin/bash --login"
    max_array_task_id: 25000
    max_job_log_files_per_directory: 5000
    memory_per_core_gb: 8
    memory_per_core_gb_by_partition:
      multicore: 8
      interactive: 8
      serial: 5
      multicore_small: 5
      himem: 32
    parallel_environments:
      serial: [1, 1]
      multicore: [2, 168]
      interactive: [1, 168]
      multicore_small: [2, 32]
      himem: [1, 32]
    partitions:
      serial:
        min_cpus: 1
        max_cpus: 1
        memory_per_core_gb: 5
        max_walltime_hours: 168
        daemon_supported: true
      multicore:
        min_cpus: 2
        max_cpus: 168
        memory_per_core_gb: 8
        max_walltime_hours: 168
        daemon_supported: true
      interactive:
        min_cpus: 1
        max_cpus: 168
        memory_per_core_gb: 8
        max_walltime_hours: 24
        daemon_supported: true
      multicore_small:
        min_cpus: 2
        max_cpus: 32
        memory_per_core_gb: 5
        max_walltime_hours: 168
        daemon_supported: true
      himem:
        min_cpus: 1
        max_cpus: 32
        memory_per_core_gb: 32
        max_walltime_hours: 168
        daemon_supported: true

  software:
    python:
      env_name: "ichor-csf3"
      python_path: "$HOME/.venv/ichor-csf3/bin/python"
      modules: []

    gaussian:
      executable_path: "$g09root/g09/g09"
      modules: ["apps/binapps/gaussian/g09d01_em64t"]

    aimall:
      executable_path: "~/AIMAll/aimqb.ish"

    ferebus:
      executable_path: "$HOME/.local/bin/ferebus"
      pyferebus_platform: "CSF3"

    ariadne_runtime:
      modules:
        - "compilers/intel/oneapi/2025.0.1"
        - "umf compiler-rt tbb compiler"
        - "mkl/2025.0"

    plumed:
      kernel_path: "$HOME/opt/plumed-2.10.0/lib/libplumedKernel.so"
      library_path: "$HOME/opt/plumed-2.10.0/lib"
      modules: []
```

The active-learning daemon defaults all backend-specific `resources.<backend>`
overrides to `null`, which means "inherit from `resources.defaults`". The
canonical default partition is `multicore` for every backend. On CSF3 these
effective resources resolve from the active profile partition metadata: the
AMD `multicore` and `interactive` partitions use 8G/core, the lower-memory
Intel `serial`/`multicore_small` partitions use 5G/core, and `himem` is
available for larger memory jobs.
Gaussian live jobs use Slurm-provided memory through `GAUSS_PDEF` and
`GAUSS_MDEF` by default; only legacy `resources.gaussian.memory_mode: link0`
writes `%NProcShared` and `%mem` into `.gjf` files.

## 6. Preflight and launch

```bash
export ICHOR_MACHINE=csf3
source ~/projects/ichor-active-learning/scripts/env_ichor_csf.sh csf3 --smoke

ichor-al-daemon init
ichor-al-daemon preflight --campaign-dir . --verbose
ichor-al-daemon preflight --campaign-dir . --verbose --submit-environment-smoke
ichor-al-daemon resource-plan --campaign-dir . --all
ichor-al-daemon start --campaign-dir . --mode live --max-ticks 200
```

The submitted preflight is an explicit commissioning check: one five-minute,
one-core Slurm job imports the configured daemon Python stack and verifies
Gaussian, AIMAll, FEREBUS and `bc` from a compute node. It performs no
scientific work and is not submitted by ordinary preflight or daemon start.
The resource plan is read-only. Before the first submission it should resolve
Phase A from the manifest-verified root `pool.xyz`; later phases may report
`evidence_not_yet_produced` until their producer handoffs exist.

The smoke config throttles Slurm arrays with
`resources.array_concurrency_limit: 4` to be gentle on the scheduler. Backend
CPU fields default to `auto`; AIMAll combines that with `aimall.naat: auto`
to choose an atom-level parallelism appropriate to the staged system size.
The example AIMAll block uses `naat: auto`,
`boaq: auto_gs2`, and `iasmesh: medium` to keep the first integration pass
cheap while still emitting IQA terms with `encomp: 3`. If you set a manual
`%N` array throttle later, keep
`runtime.poll_sacct_missing_max_ticks` generous because throttled pending rows
may not appear in `sacct` immediately.

The generated Gaussian scripts should use `#!/bin/bash --login`, the CSF3
Gaussian module, `GAUSS_SCRDIR`, `GAUSS_PDEF`, and `GAUSS_MDEF`. Gaussian
scratch is daemon-owned under `.DATA/SCRATCH/GAUSSIAN/<phase>/`; successful
Gaussian tasks remove their own scratch directory, while failed tasks keep it
for diagnosis. The active-learning daemon deliberately ignores
`software.gaussian.scratch_root` for Gaussian phases so live campaign runtime
files stay inside the campaign tree. The generated FEREBUS script should come
from pyferebus with `platform="CSF3"` and the configured executable.

If the journal reports `sacct_rows_missing_but_squeue_active`, the daemon has
seen that `sacct` is lagging while `squeue` still shows active array tasks, so
it will keep polling rather than halting the campaign.

If the campaign becomes `HALTED`, inspect `status` and `journal`, run
`ichor-al-daemon reconcile --campaign-dir .` as a dry run, and use
`reconcile --apply` only after its recovery contract is safe. Do not overwrite
`state.json` with the proposal manually. A `DONE` campaign remains terminal;
after deliberately increasing `campaign.max_iterations` and applying that
config change through reconcile, extend it only with
`resume --reopen-converged`.
