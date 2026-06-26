Active-Learning Daemon
======================

The :code:`ichor-al-daemon` command drives an end-to-end machine-learning-
force-field active-learning campaign on a SLURM cluster. It orchestrates
diversity sampling (POLUS), quantum chemistry (Gaussian + AIMAll),
Gaussian process training (FEREBUS), and differential adversarial seed acquisition
(ARIADNE) -- committing each iteration's artefacts and writing
an append-only journal that lets the campaign survive crashes, restarts,
and out-of-band recovery.

This page is the user guide. Read it once before launching the
first real campaign; bookmark the troubleshooting section for later.

.. contents::
   :depth: 2
   :local:


Quickstart
----------

Install the three packages in order::

    python3 -m pip install -e ichor_core[dev]
    python3 -m pip install -e ichor_hpc
    python3 -m pip install -e ichor_cli

For a full Manchester CSF3/CSF4 live active-learning install, use the cluster
installer. It creates the CSF-specific venv, checks sibling POLUS/FEREBUS_CPU/
ARIADNE trees, builds ARIADNE/FEREBUS/PLUMED where needed, verifies xTB/ASE,
and safely upserts ``~/ichor_config.yaml``::

    bash scripts/install_ichor_csf.sh --machine csf4 --projects-dir ~/projects
    bash scripts/install_ichor_csf.sh --machine csf3 --projects-dir ~/projects

After installation, source the matching runtime helper in every new CSF shell
before running :code:`ichor-cli` or :code:`ichor-al-daemon`::

    source scripts/env_ichor_csf.sh csf4 --smoke
    source scripts/env_ichor_csf.sh csf3 --smoke

Run the bundled example to confirm the install works (~30 seconds on a
laptop, no cluster required)::

    cd examples/dry_run_water_tetramer
    ichor-al-daemon init
    ichor-al-daemon start -d -t 200
    ichor-al-daemon status

After the third command you should see :code:`"phase": "DONE"` and
:code:`"models_version": 2`. The campaign has run two iterations of the
full pipeline -- using dry-run stub backends, so no Gaussian / AIMAll /
FEREBUS / ARIADNE binaries are required. See
:code:`examples/dry_run_water_tetramer/README.md` for the full walkthrough
and what to look at next.

Most daemon commands accept :code:`-c/--campaign-dir`. If it is omitted, the
current directory is used when it contains :code:`campaign.yaml`; otherwise the
command exits with a usage error. Common flags also have short aliases, so a
foreground live launch can be written as :code:`ichor-al-daemon start -l`, and a
background live launch can be written as :code:`ichor-al-daemon start -lb`.


Modes
-----

Three execution modes, selected by a single CLI flag:

.. list-table::
   :header-rows: 1
   :widths: 15 30 35

   * - Flag
     - What runs
     - Required backends
   * - :code:`--dry-run`
     - DryRunPhaseExecutor: stubs every external call but writes real
       on-disk artefacts (scripts, training-set versions, manifests,
       journal events, provenance sidecars). The dry-run sacct poller
       returns synthetic COMPLETED observations for every job.
     - None. Works fully off-cluster.
   * - :code:`--mock-ariadne`
     - DryRunPhaseExecutor + the mock ARIADNE optimiser (deterministic
       synthetic trajectories). Useful for testing daemon state-machine
       changes without the full ARIADNE compile.
     - None.
   * - :code:`--live`
     - LiveBackendsPhaseExecutor: real sbatch + sacct calls; per-phase
       postprocess parsers consume real Gaussian / AIMAll / FEREBUS
       output. Refuses with exit 12 if any backend is missing on PATH.
     - All cluster backends (see "Backend availability" below).

Always start a new campaign with :code:`--dry-run` to confirm the
file-system layout + config are correct before paying the cluster cost
of a live run.


Campaign config schema
----------------------

The :code:`campaign.yaml` file is a nested block layout. The full
:code:`CampaignConfig` dataclass tree lives at
:code:`ichor_hpc.ichor.hpc.active_learning.config.CampaignConfig`. A
minimal sparse config overrides only the keys you care about; every other
field falls back to its dataclass default::

    schema_version: 3

    max_iterations: 50
    poll_interval_seconds: 60

    initial_train_size: 250
    initial_val_size: 50

    batch_sizing:
      policy: linear
      floor: 5
      cap: 30

    seed_selection:
      n_seeds_per_iteration: 50
      bulk_fraction: 0.5

    anti_overlap:
      skip_training_seeds: true
      recent_seeds_cooldown: 3

    acquisition:
      property_name: iqa
      subspace:
        neighbour_count: 50
        max_subspace_dim: 6
        mode_weighting_policy: variance
      weights:
        lambda_force: 1.0
        lambda_frequency: 1.5
        lambda_anharmonic: 1.0
      gradient:
        mode: cartesian_fd

    ariadne:
      optimiser: trust_region_qn
      hessian_model: ALMLOF
      max_iter: 200

    stop:
      alpha0_streak_threshold: 1.0e-2
      alpha0_streak_length: 5
      min_iterations_before_stop: 8

Use the :code:`ichor` CLI menu (Active-learning campaign -> Edit campaign
config) to navigate the nested blocks interactively; every field has its
own menu item with input validation.

When you save from the menu, ICHOR writes a "diff-against-defaults" YAML
that contains ONLY the keys you explicitly touched. This is intentional --
it means combining your campaign.yaml with a preset overlay (e.g.
:code:`--preset spectroscopy_focused`) leaves preset values intact for
every key you did not override.


Presets
-------

Three YAML presets ship under
:code:`ichor_hpc/ichor/hpc/active_learning/presets/`. Apply one with
:code:`--preset NAME` at start time; :code:`campaign.yaml` wins for every
explicitly-set key, so the preset acts as a strong default::

    ichor-al-daemon start --live --campaign-dir . \
        --preset spectroscopy_focused

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - Preset
     - When to use
   * - :code:`balanced`
     - The dataclass defaults. Start here when you have no specific
       downstream priority.
   * - :code:`spectroscopy_focused`
     - Tunes the acquisition surface toward low-frequency modes via
       :code:`mode_weighting_policy: inverse_frequency` and bumps
       :code:`max_subspace_dim` to 10. Pins :code:`gradient.mode: active_fd`
       to avoid the cartesian_fd cost explosion.
   * - :code:`thermodynamics_focused`
     - Force-dominant for MD-trajectory stability:
       :code:`lambda_force: 1.5` and :code:`lambda_energy: 0.5`.

.. warning::

   None of the presets are yet calibrated against a real benchmark
   dataset. Treat the shipped values as best-effort starting points based
   on first-principles reasoning; recalibrate them once you have real
   water-tetramer / N-mer benchmark numbers from a successful live run.


Cluster prerequisites for :code:`--live`
----------------------------------------

A live campaign on a configured SLURM cluster needs the backend profile in
``~/ichor_config.yaml`` plus the Python modules/binaries for that profile.
Set ``ICHOR_MACHINE`` when the login hostname does not contain the intended
top-level profile key, for example ``ICHOR_MACHINE=csf3``. The daemon checks
the live backends at start and refuses with exit 12 if any are missing::

    module load python/3.11.3-gcccore-12.3.0
    module load compilers/oneapi/2024.2.0
    module load compiler-rt tbb compiler
    module load mkl/2024.2
    module load gaussian/g16c01_em64t_detectcpu
    # AIMAll lives in ~/AIMAll/ on most CSF nodes (operator-installed).
    # FEREBUS is invoked via the pyferebus Python wrapper; install it
    # in the same venv environment as ichor_hpc.
    # ARIADNE is the oneAPI .so + Python wrapper installed into that venv.

The exact module names and versions come from your local
:code:`~/ichor_config.yaml`; see :code:`ichor_config_setup.rst` for the
config-file format. CSF4 operators at Manchester can reuse the canonical
:code:`ichor_config.yaml` checked in at the repo root. CSF3 operators should
start from :code:`examples/csf3_first_live_iter/README.md`, which documents the
private CPython 3.11 non-Conda route and CSF3 oneAPI runtime modules.


Backend availability
--------------------

.. list-table::
   :header-rows: 1
   :widths: 18 25 25 32

   * - Backend
     - Install method
     - Required for
     - Verification command
   * - sbatch / sacct
     - Cluster-side (SLURM)
     - :code:`--live`
     - :code:`which sbatch && which sacct`
   * - Gaussian g16
     - Cluster module declared in :code:`~/ichor_config.yaml`
     - :code:`--live`
     - :code:`ichor-al-daemon preflight --campaign-dir .`
   * - AIMAll
     - Operator-installed at :code:`~/AIMAll/aimqb.ish`
     - :code:`--live`
     - :code:`ls ~/AIMAll/aimqb.ish`
   * - FEREBUS (pyferebus)
     - :code:`pip install -e FEREBUS_CPU/pyferebus --no-deps` (in same env as ichor_hpc)
     - :code:`--live`
     - :code:`python -c "import pyferebus"`
   * - ARIADNE (oneAPI .so)
     - Build from source with oneAPI/MKL and install into the active venv
     - :code:`--live` (skip with :code:`--mock-ariadne`)
     - :code:`python -c "import ariadne"`
   * - POLUS (DIVSampler)
     - :code:`pip install -e POLUS/polus_core_subpackage --no-deps` (in same env as ichor_hpc)
     - :code:`--live`
     - :code:`python -c "import polus.samplers.RS.randomSampling"`

The :code:`ichor-al-daemon` :code:`start --live` command exits cleanly
(no partial writes) if any of the above checks fail. You can also run
the backend preflight command directly::

    ichor-al-daemon preflight --campaign-dir .

which prints structured backend/profile diagnostics and exits with code 12 if
anything required for live mode is missing.


Recovery + troubleshooting
--------------------------

The daemon writes two artefacts you care about during recovery:

- :code:`<campaign>/.DATA/ACTIVE_LEARNING/state.json` -- the canonical
  state checkpoint. Authoritative. If this file is missing or corrupt
  the daemon refuses to start; auto-recovery would mask the bug class we
  most want to catch.
- :code:`<campaign>/.DATA/ACTIVE_LEARNING/journal.ndjson` -- append-only
  per-phase event log. Best-effort; you can drop the file and the
  daemon still works, you just lose post-mortem provenance.

If the daemon refuses to start because :code:`state.json` is missing or
fails schema validation, run::

    ichor-al-daemon reconcile --campaign-dir <DIR>

This inspects the on-disk artefacts (committed iterations in
:code:`5_TRAINING/` and :code:`6_TRAINED_MODELS/`, plus journal events)
and proposes a recovered state at :code:`state.json.proposed`. Plain
reconcile is diagnostic: it may write the proposal file and warn about a
live daemon, but it does not alter :code:`state.json`. Review the proposal,
then promote it manually::

    mv .DATA/ACTIVE_LEARNING/state.json.proposed .DATA/ACTIVE_LEARNING/state.json
    ichor-al-daemon resume --campaign-dir . --live

For routine crash recovery, use the guarded apply path instead::

    ichor-al-daemon reconcile --campaign-dir . --apply

The apply path refuses to run if the daemon lock is held, the lock state is
unknown, the lease heartbeat is fresh, or the background daemon PID appears
alive. Stop the daemon first when reconcile reports live jobs or an active
runtime::

    ichor-al-daemon stop --campaign-dir . --cancel-jobs
    ichor-al-daemon reconcile --campaign-dir . --apply

The reconcile path is conservative: it ALWAYS positions the daemon at a
"safe" re-entry point (STOP_CHECK for completed iterations or INIT when
nothing is committed) and clears any pending jobs so the next run
re-submits rather than blindly polls unknown JobIDs.

Reconcile also verifies the trajectory-pool contract for non-empty campaigns:
:code:`.DATA/TRAJECTORY/pool.xyz` and
:code:`.DATA/TRAJECTORY/pool.manifest.json` must exist and the SHA-256 in the
manifest must match the canonical pool. Pool corruption or deletion is reported
as unsafe; reconcile does not automatically repair the pool because it is
provenance-critical.

If :code:`campaign.yaml` was accidentally removed but the config lock is still
present, write a proposal from the lock::

    ichor-al-daemon reconcile --campaign-dir . --restore-config-from-lock

This creates :code:`campaign.yaml.proposed` only. The proposal is dense by
design: it is a full config snapshot restored from
:code:`config_lock.json`, not a sparse menu-style save. Inspect it, then
promote it manually before running :code:`reconcile --apply`.

What reconcile still does not do:

- it does not postprocess successful Slurm jobs whose daemon-side commit did
  not run;
- it does not harvest uncommitted Gaussian/AIMAll outputs into a training
  version;
- it does not rebuild committed manifests automatically;
- it does not roll back one-sided newer training/model versions without
  operator review.

Common failure modes:

- :code:`live_postprocess_refused` events in the journal mean a SBATCH
  phase ran but its parser is not yet registered in
  :code:`LIVE_POSTPROCESS_IMPLEMENTED`. After M16, every SBATCH phase
  has a parser; this event should never fire on a current build. If it
  does, file an issue.
- :code:`sacct_empty_timeout` after many ticks means a JobID disappeared
  from the SLURM accounting database (typically because the campaign
  ran for more than 24 hours and sacct aged out the record). The daemon
  routes such jobs to the failure handler after
  :code:`poll_sacct_empty_max_ticks` ticks (default 10).
- :code:`tick_error` is the catch-all. The journal payload includes
  the traceback; inspect it before the next restart.
- :code:`quantum_output_rejected` per-pointdir means a Gaussian SCF
  failed to converge or an AIMAll partial calculation; up to
  :code:`failure_threshold_fraction` of these per iteration is
  tolerated and the iteration still commits the surviving points.


Architecture sketch
-------------------

The daemon is a finite-state machine over campaign phases. Each phase
is either INLINE (runs synchronously in the daemon process; e.g.
SEED_SELECT, SPLIT, APPEND, STOP_CHECK) or SBATCH (submits via
:code:`sbatch --parsable`, polls sacct, then runs a postprocess parser
in the daemon process).

.. code-block:: text

   INIT
     |
     v
   PHASE_A_POLUS  (sbatch)        Initial diverse sub-sample.
     |
     v
   INITIAL_GAUSSIAN  (sbatch)     Single-point energies on the sample.
     |
     v
   INITIAL_AIMALL  (sbatch)       IQA decomposition.
     |
     v
   INITIAL_FEREBUS  (sbatch)      First GP fit. Commits iteration-0 to
     |                            5_TRAINING and 6_TRAINED_MODELS.
     v
   SEED_SELECT  (inline)  <----+  Pick seeds from the trajectory pool,
     |                         |  forbidding already-trained frames.
     v                         |
   ARIADNE_ARRAY  (sbatch)      |  Per-seed adversarial descent.
     |                         |
     v                         |
   PHASE_B_POLUS  (sbatch)      |  POLUS FPS over the adversarial pool.
     |                         |
     v                         |
   GAUSSIAN  (sbatch)           |  Per-point energies.
     |                         |
     v                         |
   AIMALL  (sbatch)             |  IQA decomposition.
     |                         |
     v                         |
   SPLIT  (inline)              |  Stratified train / val / holdout.
     |                         |
     v                         |
   APPEND  (inline)             |  Commit new training-set iteration.
     |                         |
     v                         |
   FEREBUS  (sbatch)            |  Re-fit GP. Commits new models iter.
     |                         |
     v                         |
   STOP_CHECK  (inline)  ------+  If alpha-trend triggered or max iter
     |                            reached, exit; else go to SEED_SELECT.
     v
   DONE

State transitions are persisted to :code:`state.json` BEFORE journaling,
so a crash between the two leaves the state authoritative and the
journal at most one event behind. The inline APPEND and FEREBUS commits
are idempotent: re-running after a partial crash is safe.

The full design rationale lives in
:code:`C:\Users\bukow\.claude\plans\you-are-a-phd-level-glistening-frost.md`
(the M9-M16 plan document). The most important reads are sections 1-7
(architecture + new infrastructure), section 11 (the three traps the
design avoids), and the M15 + M16 sections (hardening fixes + live
executor parsers).
