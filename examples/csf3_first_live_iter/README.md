# First live iteration on CSF3 -- a walkthrough

This is the CSF3 equivalent of the CSF4 live smoke. It proves one real
active-learning iteration with Slurm, Gaussian, AIMAll, FEREBUS, ARIADNE, and
POLUS, while keeping Python, FEREBUS, ARIADNE, and PLUMED outside the repo.

## 1. Build private CPython 3.11

CSF3 currently exposes `python/3.13.1`, but the ICHOR scientific stack is safer
on Python 3.11 because RDKit, xTB, PLUMED, f90wrap, NumPy, and ARIADNE include
compiled components.

```bash
mkdir -p ~/src ~/opt
cd ~/src
wget https://www.python.org/ftp/python/3.11.15/Python-3.11.15.tgz
tar -xzf Python-3.11.15.tgz
cd Python-3.11.15
./configure --prefix=$HOME/opt/python-3.11.15 --enable-shared --with-ensurepip=install
make -j 8
make install

export LD_LIBRARY_PATH=$HOME/opt/python-3.11.15/lib:$LD_LIBRARY_PATH
$HOME/opt/python-3.11.15/bin/python3.11 -m venv ~/.venv/ichor-al-csf3
source ~/.venv/ichor-al-csf3/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

If direct download is blocked, download the tarball locally from `python.org`,
copy it to CSF3, and build it on CSF3.

## 2. Install ICHOR and sibling Python packages

```bash
source ~/.venv/ichor-al-csf3/bin/activate
export LD_LIBRARY_PATH=$HOME/opt/python-3.11.15/lib:$LD_LIBRARY_PATH

pip install -e ~/projects/ichor-active-learning/ichor_core
pip install -e ~/projects/ichor-active-learning/ichor_hpc
pip install -e ~/projects/ichor-active-learning/ichor_cli

pip install -e ~/projects/POLUS/polus_core_subpackage --no-deps
pip install -e ~/projects/FEREBUS_CPU/pyferebus --no-deps
```

## 3. Build ARIADNE with CSF3 oneAPI

```bash
module purge
module load compilers/intel/oneapi/2025.0.1
module load umf compiler-rt tbb compiler
module load mkl/2025.0

source ~/.venv/ichor-al-csf3/bin/activate
export LD_LIBRARY_PATH=$HOME/opt/python-3.11.15/lib:$LD_LIBRARY_PATH

cd ~/projects/ARIADNE
python -m pip install -r requirements-build.txt
export CC=icx
export CXX=icpx
export FC=ifx
python -m pip install . --no-build-isolation -v
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

Gaussian 16 requires access to the Manchester Gaussian license group. AIMAll may
be available centrally or via a group/user install; record the actual `aimqb.ish`
path in `~/ichor_config.yaml`.

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
    jobscript_shebang: "#!/bin/bash --login"
    max_array_task_id: 25000
    memory_per_core_gb: 8
    parallel_environments:
      serial: [1, 1]
      multicore: [2, 168]

  software:
    python:
      env_name: "ichor-al-csf3"
      python_path: "$HOME/.venv/ichor-al-csf3/bin/python"
      modules: []

    gaussian:
      executable_path: "$g16root/g16/g16"
      modules: ["apps/binapps/gaussian/g16c01_em64t_detectcpu"]
      scratch_root: "/scratch/$USER"

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

## 6. Preflight and launch

```bash
export ICHOR_MACHINE=csf3
source ~/.venv/ichor-al-csf3/bin/activate
export LD_LIBRARY_PATH=$HOME/opt/python-3.11.15/lib:$LD_LIBRARY_PATH

ichor-al-daemon preflight --campaign-dir .
ichor-al-daemon import-pool --campaign-dir . --source pool.xyz
ichor-al-daemon start --live --campaign-dir . --max-ticks 200
```

The generated Gaussian scripts should use `#!/bin/bash --login`, the CSF3
Gaussian module, `GAUSS_SCRDIR`, and `GAUSS_PDEF`. The generated FEREBUS script
should come from pyferebus with `platform="CSF3"` and the configured executable.
