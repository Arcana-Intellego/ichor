# ichor
---

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Run Tests](https://github.com/popelier-group/ICHOR/actions/workflows/run_tests.yml/badge.svg)](https://github.com/popelier-group/ICHOR/actions/workflows/run_tests.yml)
![Release](https://img.shields.io/github/v/release/popelier-group/ICHOR?sort=semver)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/ichor/badge/?version=latest)](https://ichor.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11182102.svg)](https://doi.org/10.5281/zenodo.11182102)

`ichor` is a Python package used to simplify data management from computational chemistry programs and aid with machine learning force field development. If you would like to request missing features or run into a bug, don't hesitate to create an [issue](https://github.com/popelier-group/ICHOR/issues).

Here is a list of things that the package is intended to do:

1. provide interfaces to any computational chemistry software to allow for easy switching between similar software and results comparison
2. implement flexible data structures to allow for data management of hundreds of thousands of calculations from multiple programs
3. integrate common database formats for efficient data storage, sharing, and post-processing
4. provide interfaces to workload managers on compute clusters to automate job submission
5. collate tools for machine learning dataset and model analysis, as well as molecular dynamics simulation benchmarking
6. run a finite-state machine (daemon) over a fixed sequence of batch-sequential active learning campaign phases that train IQA atomic models

Realistically, the file management portion of `ichor` (as well as the workload manager integration) is very general, so it can be used for any type of data that might not even be computational chemistry related. However, the focus of the source code itself is on computational chemistry and machine learning force field development.


The namespace package `ichor` is divided into three parts, `ichor.core`, `ichor.hpc`, and `ichor.cli`.

### `ichor.core`
The `ichor.core` package contains classes and functions which make it easy to handle a very large number of files, perform many calculations with the outputs contained in the file, as well as aid with machine learning force field development.

### `ichor.hpc`
The `ichor.hpc` package is used to submit jobs on compute clusters (SGE/SLURM).

### `ichor.cli`
The `ichor.cli` package provides a simple to use command line interface (CLI), providing an easy access to access the most commonly used tools from ichor.


## Getting started


**You will need to have an `ichor_config.yaml` file in your home directory for configuration settings relating to HPC clusters, refer to the documentation for examples. An example `ichor_config.yaml` is provided in the repository.**


Install the three packages in dependency order:

```
python3 -m pip install -e ichor_core[dev]
python3 -m pip install -e ichor_hpc
python3 -m pip install -e ichor_cli
```

For a full Manchester CSF3/CSF4 active-learning install, use the cluster
installer instead. It checks the sibling POLUS/FEREBUS_CPU/ARIADNE trees,
creates the CSF-specific venv, builds ARIADNE/FEREBUS/PLUMED where needed,
and updates `~/ichor_config.yaml`:

```
bash scripts/install_ichor_csf.sh --machine csf4 --projects-dir ~/projects
# or
bash scripts/install_ichor_csf.sh --machine csf3 --projects-dir ~/projects
```

After installation, source the matching runtime helper in each new CSF shell
before running the CLI or daemon:

```
source scripts/env_ichor_csf.sh csf4 --smoke
# or
source scripts/env_ichor_csf.sh csf3 --smoke
```

## Active learning daemon
Run the bundled example to confirm everything works (no cluster
required, ~30 seconds on a laptop):

```
cd examples/dry_run_water_tetramer
ichor-al-daemon import-pool -s pool.xyz
ichor-al-daemon start -d -t 200
ichor-al-daemon status
```

After the third command you should see `"phase": "DONE"` and
`"models_version": 2`. See `examples/dry_run_water_tetramer/README.md`
for the full walkthrough and `docs/source/active_learning_daemon.rst`
for the daemon user guide.
When not already inside a campaign directory, pass `-c/--campaign-dir`.

### Backend availability

| Backend            | Install method                                        | Required for       | Verification              |
| ------------------ | ----------------------------------------------------- | ------------------ | ------------------------- |
| sbatch / sacct     | Cluster-side (SLURM)                                  | `--live`           | `which sbatch`            |
| Gaussian g16       | `module load gaussian/g16c01_em64t_detectcpu`         | `--live`           | `which g16`               |
| AIMAll             | Operator-installed at `~/AIMAll/aimqb.ish`            | `--live`           | `ls ~/AIMAll/aimqb.ish`   |
| FEREBUS            | Build `FEREBUS_CPU` binary + editable `pyferebus`      | `--live`           | `python -c "import pyferebus"` |
| ARIADNE            | Build/install sibling `ARIADNE` into the venv          | `--live` (skip with `--mock-ariadne`) | `python -c "import ariadne"`   |
| POLUS              | Editable sibling `POLUS/polus_core_subpackage`         | `--live`           | `python -c "import polus.samplers.RS.randomSampling"` |
| PLUMED             | Local kernel + PyPI wrapper                            | full CLI/metadynamics | `python -c "import plumed"` |
| xTB                | PyPI `xtb` package, used through ASE                   | full CLI/metadynamics | `python -c "from xtb.ase.calculator import XTB"` |

The daemon refuses to start in `--live` mode if any required backend is
missing (exit code 12). Use `--dry-run` off-cluster -- it writes real
artefacts without requiring any of the above.


## Papers

The published paper for `ichor` can be found [here](https://doi.org/10.1002/jcc.27477).

## Documentation

Documentation of all three packages, including examples, can be found [here](https://ichor.readthedocs.io/en/latest/).

## Contributing

Contributions are very welcome! More information on how to correctly contribute can be found in the [CONTRIBUTING.md](CONTRIBUTING.md) file.
