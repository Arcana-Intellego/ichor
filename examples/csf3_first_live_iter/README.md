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
source scripts/env_ichor_csf.sh --smoke
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
echo "f4de1b10bd6c70cbb9fa1cd71fc5038b832747a74ee59d599c69ce4846defb50  Python-3.11.15.tgz" | sha256sum -c -
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
copy it to CSF3, verify the same SHA-256 on CSF3, and build it there.
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

The repository root [`ichor_config.yaml`](../../ichor_config.yaml) is the
single canonical CSF3 profile. Run `scripts/install_ichor_csf.sh --machine
csf3`; the installer copies that profile atomically into
`~/ichor_config.yaml` and overrides only the local Python, AIMAll, FEREBUS and
PLUMED paths. Do not maintain a second profile in this guide.

The canonical CSF3 contract uses a login-shell Slurm script, private CPython
3.11 plus `$HOME/opt/python-3.11.15/lib`, 8 GB/core on `multicore`, 4
GB/core on `serial`, 5 GB/core on `multicore_small`, and a six-hour
`interactive` ceiling. Inspect the installed values with
`ichor-al-daemon preflight --verbose` before submitting work.

The active-learning daemon defaults all backend-specific `resources.<backend>`
overrides to `null`, which means "inherit from `resources.defaults`". The
canonical default partition is `multicore` for every backend. On CSF3 these
effective resources resolve from the active profile partition metadata: the
AMD `multicore` and `interactive` partitions use 8G/core, the lower-memory
Intel `serial` partition uses 4G/core, `multicore_small` uses 5G/core, and
`himem` is available for larger memory jobs.
Gaussian live jobs always use the immutable Slurm allocation through
`GAUSS_PDEF` and `GAUSS_MDEF`. Per-input `%NProcShared` and `%Mem` directives
are rejected so a staged input cannot disagree with its submitted resources.

## 6. Preflight and launch

```bash
source ~/projects/ichor-active-learning/scripts/env_ichor_csf.sh --smoke

ichor-al-daemon init
ichor-al-daemon preflight --campaign-dir . --verbose
ichor-al-daemon preflight --campaign-dir . --verbose --submit-environment-smoke
ichor-al-daemon resource-plan --campaign-dir . --all
ichor-al-daemon start --campaign-dir . --mode live --max-ticks 200
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

The submitted preflight is an explicit commissioning check: one five-minute,
one-core Slurm job imports the configured daemon Python stack and verifies
Gaussian, AIMAll, FEREBUS and `bc` from a compute node. It performs no
scientific work and is not submitted by ordinary preflight or daemon start.
The resource plan is read-only. Before the first submission it should resolve
Phase A from the manifest-verified root `pool.xyz`; later phases may report
`evidence_not_yet_produced` until their producer handoffs exist.

The smoke config sets `resources.array_concurrency_limit: null`, which leaves
array concurrency to Slurm. Backend
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
