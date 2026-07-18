Active-Learning Daemon
======================

The :code:`ichor-al-daemon` command drives an end-to-end machine-learning-
force-field active-learning campaign on a SLURM cluster. It orchestrates
ICHOR-owned exact diversity sampling, quantum chemistry (Gaussian + AIMAll),
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
installer. It creates the CSF-specific venv, checks sibling FEREBUS_CPU/
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
    ichor-al-daemon init --yes
    ichor-al-daemon start --mode dry_run --foreground --max-ticks 200
    ichor-al-daemon status

After the third command you should see :code:`"phase": "DONE"` and
:code:`"models_version": 2`. The campaign has run two iterations of the
full pipeline -- using dry-run stub backends, so no Gaussian / AIMAll /
FEREBUS / ARIADNE binaries are required. See
:code:`examples/dry_run_water_tetramer/README.md` for the full walkthrough
and what to look at next.

Most daemon commands accept :code:`-c/--campaign-dir`. If it is omitted, the
current directory is used when it contains :code:`campaign.yaml`; otherwise the
command exits with a usage error. The first start must explicitly select
:code:`--mode live` or :code:`--mode dry_run`; that choice is recorded in the
campaign execution identity and cannot later be changed. Launches are detached
by default. Use :code:`--foreground` when the shell should remain attached.


Modes
-----

Two execution modes are available:

.. list-table::
   :header-rows: 1
   :widths: 15 30 35

   * - Mode
     - What runs
     - Required backends
   * - :code:`--mode dry_run`
     - DryRunPhaseExecutor: stubs every external call but writes real
       on-disk artefacts (scripts, reference-data delta versions, manifests,
       journal events, provenance sidecars). The dry-run sacct poller
       returns synthetic COMPLETED observations for every job.
     - None. Works fully off-cluster.
   * - :code:`--mode live`
     - LiveBackendsPhaseExecutor: real sbatch + sacct calls; per-phase
       postprocess parsers consume real Gaussian / AIMAll / FEREBUS
       output. Refuses with exit 12 if any backend is missing on PATH.
     - All cluster backends (see "Backend availability" below).

Use a separate campaign with :code:`--mode dry_run` to confirm the
file-system layout + config are correct before paying the cluster cost
of a live run. Execution mode is immutable, so a dry-run campaign cannot be
converted into a live campaign.


Campaign config schema
----------------------

The :code:`campaign.yaml` file is a nested block layout. The full
:code:`CampaignConfig` dataclass tree lives at
:code:`ichor_hpc.ichor.hpc.active_learning.config.CampaignConfig`. A
minimal sparse config overrides only the keys you care about; every other
field falls back to its dataclass default::

    schema_version: 14

    campaign:
      system_name: CHANGE_ME_SYSTEM
      max_iterations: 50
      sampling_aggressiveness: 5
      reproducibility_seed: 0
      custom_bootstrap: false

    runtime:
      poll_interval_seconds: 60

    point_allocation:
      bootstrap_training_size: 192
      bootstrap_internal_validation_size: 48
      bootstrap_external_validation_size: 60
      batch_training_size: 8
      batch_internal_validation_size: 2

    seed_selection:
      n_seeds_per_iteration: 20
      bulk_fraction: 0.2
      strategy: d_optimal
      d_optimal_degenerate_policy: score_backfill
      exclude_committed_seed_frames: true
      recent_seed_cooldown_iterations: 1

    ferebus:
      prior_mean_type: 21
      prior_mean_level_of_theory: auto
      prior_mean_iqa_deviation_factor: 1.0
      feature_scaling: true
      property_scaling: false

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

Active-learning FEREBUS models use mean type 21. For IQA models this is the
isolated-atom IQA energy in Hartree; auxiliary-property models retain a zero
prior mean. ``prior_mean_level_of_theory: auto`` resolves to the configured
Gaussian method and basis set. Property scaling must remain disabled so that
the physical prior and IQA targets have the same units. Feature scaling is an
independent setting.

Schema 13 is intentionally strict. Older campaign files are rejected rather
than migrated implicitly. The source pool has the fixed campaign path
``pool.xyz``. ``ichor-al-daemon init --source /path/to/source.xyz`` copies an
external source to that path before SHA-pinning it in daemon-owned metadata.
``sampling_aggressiveness`` is a dimensionless campaign control whose ARIADNE
trust radius is normalised for molecular size.

Custom bootstrap inputs
~~~~~~~~~~~~~~~~~~~~~~~

Set ``campaign.custom_bootstrap: true`` to enable discovery under the fixed
``bootstrap/`` directory. The recognised split files are
``training_set_bootstrap.xyz|csv``,
``internal_validation_set_bootstrap.xyz|csv``, and
``external_validation_set_bootstrap.xyz|csv``. Formats may be mixed between
splits, but supplying both formats for one split is an error. Every supplied
split may contain at most its matching ``point_allocation.bootstrap_*_size``;
ICHOR Phase A diversity fills any deficit. Supplied geometries are mandatory and cannot
be silently replaced if Gaussian or AIMAll rejects them.

CSV files must begin with the complete consecutive ``f1..f(3N-6)`` ALF feature
prefix. Unique finite numeric columns after that prefix, such as ``iqa`` or
``q00``, are accepted but ignored as scientific labels. ``init`` asks for one
1-based ALF per detected CSV. For unattended ``init --yes``, put those values
in ``bootstrap/alf.yaml`` under ``train``, ``int_val``, or ``ext_val``.
Gaussian/AIMAll always regenerate the authoritative labels.

Alternatively, ``bootstrap/model_krig/`` may provide the initial training
baseline. It is mutually exclusive with a training-set XYZ/CSV. The directory
must contain exactly one model for every molecule atom and every required
property: IQA plus all entries in ``ferebus.properties``. Model training-row
count replaces ``bootstrap_training_size``; only missing internal/external
validation slots are filled by ICHOR diversity sampling. Imported models become immutable model
version 0 without a FEREBUS optimisation job, and their original X/Y rows are
prepended and verified during every later retraining.

The lower-case ``bootstrap/`` directory is user-owned input. The
``.DATA/BOOTSTRAP/`` directory is daemon-owned Phase A output; do not put
user files there. ``bootstrap/model_krig/`` must contain regular
``.model`` files only, with no symlinks or unrelated sidecar files.

Every ``init`` prints the complete discovery and top-up summary and asks
``Proceed? [y/N]`` before committing evidence. ``--yes`` bypasses only that
final confirmation; it does not invent missing CSV ALFs.

Use the :code:`ichor` CLI menu (Active-learning campaign -> Edit campaign
config) to navigate the nested blocks interactively; every field has its
own menu item with input validation.

When you save from the menu, ICHOR writes a "diff-against-defaults" YAML
containing only the keys explicitly touched by the user. Named presets
are not supported; the complete effective configuration is determined by
the schema-14 defaults and the campaign file.


Cluster prerequisites for :code:`--mode live`
---------------------------------------------

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
    # AIMAll lives in ~/AIMAll/ on most CSF nodes (user-installed).
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
     - :code:`--mode live`
     - :code:`which sbatch && which sacct`
   * - Gaussian g16
     - Cluster module declared in :code:`~/ichor_config.yaml`
     - :code:`--mode live`
     - :code:`ichor-al-daemon preflight --campaign-dir .`
   * - AIMAll
     - User-installed at :code:`~/AIMAll/aimqb.ish`
     - :code:`--mode live`
     - :code:`ls ~/AIMAll/aimqb.ish`
   * - FEREBUS (pyferebus)
     - :code:`pip install -e FEREBUS_CPU/pyferebus --no-deps` (in same env as ichor_hpc)
     - :code:`--mode live`
     - :code:`python -c "import pyferebus"`
   * - ARIADNE (oneAPI .so)
     - Build from source with oneAPI/MKL and install into the active venv
     - :code:`--mode live`
     - :code:`python -c "import ariadne"`
The :code:`ichor-al-daemon` :code:`start --mode live` command exits cleanly
(no partial writes) if any of the above checks fail. You can also run
the backend preflight command directly::

    ichor-al-daemon preflight --campaign-dir .

which prints structured backend/profile diagnostics and exits with code 12 if
anything required for live mode is missing.


QM reference-data storage
-------------------------

Accepted Gaussian/AIMAll pointdirs are committed under
:code:`QM_REFERENCE_DATA/`. Each :code:`iteration-NNNNNN/` is a
content-verified **delta** containing only points first accepted in that
reference-data version. Older pointdirs are not copied or symlinked into later
versions.

Every delta contains :code:`REFERENCE_DATA_VERSION.json`, the exact
point-allocation snapshot and its generation history, and a generic
:code:`.manifest.json`. Version manifests form a SHA-256 parent chain and pin
the deterministic cumulative point order. Pointdir names use one global
six-digit ordinal, for example :code:`POINT_000137.pointdir`.

All cumulative consumers use the same reference-data resolver. In particular,
FEREBUS receives resolver-ordered pointdirs and records the reference-data
version, head-manifest hash, cumulative-view hash, row order, and row count in
:code:`FEREBUS_TASKS.json`. A committed model is rejected if that binding no
longer matches the authoritative reference view.

:code:`.DATA/ACTIVE_LEARNING/reference_data_view_cache.json` is a derived cache
only. It is never authoritative and can be deleted; the next resolution
rebuilds it from committed version manifests. The old :code:`5_TRAINING/`
layout is intentionally unsupported. Start a fresh campaign rather than
mixing the two storage contracts.


Trained-model storage
---------------------

Each successful FEREBUS phase commits one complete content-verified snapshot
under :code:`TRAINED_MODELS/iteration-NNNNNN/`. Task files are grouped by
property and atom; model and configuration files are never dumped at the
version root::

    TRAINED_MODELS/iteration-000001/
      FEREBUS_TASK_ARTEFACTS.json
      FEREBUS_TASKS.json
      FEREBUS_QUALITY.json
      iqa/O1/
        SYSTEM_iqa_O1.model
        ferebus_iqa_O1.config
        SYSTEM_iqa_O1.opt
        SYSTEM_iqa_O1.perf
        SYSTEM_iqa_O1.pred
        SYSTEM_iqa_O1.scurve
        SYSTEM_iqa_O1.sol
      q00/O1/
        SYSTEM_q00_O1.model
        ferebus_q00_O1.config

:code:`FEREBUS_TASK_ARTEFACTS.json` is the authoritative model-set manifest.
It records every expected task file and SHA-256, the exact property/atom
product, its parent model-manifest hash, and the QM reference-data version,
head hash, cumulative-view hash, row order, split rows, and exact input CSV
hashes used by FEREBUS.
Consumers load exactly the listed model files; recursive model discovery is
not used. Unknown files, missing files, symlinks, hash drift, task-order drift,
or a rejected quality manifest make the snapshot unusable.

The :code:`current` pointer is updated only after the new snapshot has been
committed and deeply verified. It may point only at the newest committed model
version. :code:`TRAINED_MODELS/iteration-staging/` remains a
rebuildable FEREBUS working directory and is not authoritative. The former
:code:`6_TRAINED_MODELS/` and nested :code:`task_artefacts/` layouts are
intentionally unsupported.

FEREBUS quality publication distinguishes a complete scientific measurement
from a parser or telemetry failure. Only complete measurements may become
:code:`FEREBUS_QUALITY.json` and a promotion decision. Incomplete measurements
are recorded under
:code:`.DATA/ACTIVE_LEARNING/ferebus_quality_attempts/`; raw model, performance
and task-receipt evidence remains available for recovery and is not classified
as a rejected model. The reader accepts the exact legacy FEREBUS performance
labels :code:`weights_l2_nor` and :code:`covariance_con`, canonicalising them to
their full names while rejecting ambiguous files containing both forms.

For the historical failure in which an incomplete measurement was moved under
:code:`TRAINED_MODELS/rejected-candidates/`, routine reconcile can identify one
unambiguous measurement-only candidate and record
:code:`ferebus_postprocess_recovery.json`. The next resume authenticates and
copies its raw files into clean staging, leaves the original quarantine
unchanged, and reruns postprocessing inline. It does not submit another
FEREBUS job. Genuine threshold or incumbent-regression failures remain rejected
and are never eligible for this path.


Sampling storage and identities
-------------------------------

Bootstrap selection is not an active-learning iteration. Its selected XYZ,
selected pool-row indices, Phase-A manifest, and allocation manifest live under
:code:`.DATA/BOOTSTRAP/selection/` and
:code:`.DATA/BOOTSTRAP/allocation/`. Bootstrap is the
only scientific stage identified by iteration and committed version zero.

Active iterations start at one and use six-digit canonical names. Active
iteration :code:`N` consumes committed reference/model version :code:`N-1` and,
after successful QM labelling and FEREBUS training, commits version :code:`N`::

    ACTIVE_LEARNING/iteration-000001/
      ITERATION_MANIFEST.json
      protocol/
        SAMPLING_PROTOCOL_RESOLVED.json
        SAMPLING_PROTOCOL_AUDIT.json
        SAMPLING_SCALE_MODEL.json
        reference_scales.json
      seed_selection/
        SELECTION.json
        seeds.xyz
      ariadne/
        TASK_MAP.json
        RESULTS.json
        AUDIT.json
        seeds/
          seed-000001/
            result.json
            provenance.json
            ARIADNE_OUTPUT_MANIFEST.json
            trajectory/
              trajectory.xyz
              metrics.jsonl
              trace.jsonl  (optional detailed optimiser trace)
              MANIFEST.json
      phase_b/
        considered_candidates.xyz
        selected.xyz
        SELECTION.json
      allocation/
        POINT_ALLOCATION.json
        SPLIT_RECEIPT.json
      calibration/
        ERROR_CALIBRATION_AUDIT.json  (when calibration is enabled)

Scientific seed IDs are one-based and contiguous. Scheduler array task IDs and
trajectory-pool row indices remain explicitly zero-based. :code:`TASK_MAP.json`
is the sole mapping between those domains; code must not infer one from a
directory name or array position.

Every completed active iteration is finalised by :code:`ITERATION_MANIFEST.json`.
The manifest records an exact recursive file inventory, SHA-256 bindings,
iteration/model/reference-data identities, and the parent iteration-manifest
hash. Extra files, missing files, hash drift, symlinks, partial output markers,
or a broken parent chain make the iteration invalid. Campaign files remain
owner-writable; hashes and exact inventories, rather than permission modes,
provide the integrity contract. The daemon may rebuild explicitly documented
derived caches, while authoritative manifests and task outputs remain
content-bound.

The former :code:`3_DIVERSITY_SAMPLING/`, :code:`7_ACTIVE_LEARNING/`, flat
iteration files, four-digit active iteration names, :code:`pool/seed_*/`, and
zero-based scientific seed directories are intentionally unsupported. Start a
fresh campaign rather than mixing storage contracts.


Resource evidence, scripts, and scratch
---------------------------------------

Live resource requests are resolved from producer-owned evidence after exact
staging. The daemon does not substitute guessed frame, atom, feature, or row
counts. Examples include the SHA-pinned root :code:`pool.xyz` for ICHOR Phase
A, :code:`ARIADNE_RESULTS.json` for Phase B, exact :code:`POINTS.txt` and
GJF/WFN dimensions for Gaussian and AIMAll, the ARIADNE task/model manifests,
and the FEREBUS task manifest plus split CSV hashes. Missing evidence halts an
actual submission with a precise error.

Every submission attempt receives an immutable schema-v1 resource record::

    .DATA/ACTIVE_LEARNING/resource_resolutions/
      PHASE_A_DIVERSITY/iteration-000000/r0000-a0001-1234abcd.json

The record is bound by SHA-256 to the submission intent and job script. It
contains formula inputs, active scientific workers, any CPUs allocated only
for memory, per-task and peak allocation, array concurrency, evidence paths
and hashes, and scratch requirements. A retry gets a new attempt identity and
does not overwrite the previous record.

Submitted scripts and logs are retained as operational evidence in
self-contained bundles::

    .DATA/SCRIPTS/JOBS/
      GAUSSIAN/GAUSSIAN/iteration-000001/<submission-identity>/
        job.sh
        array_task_map.json   # partial-array retries only
        OUTPUTS/
        ERRORS/

An array member writes
:code:`OUTPUTS/<parent-job-id>_<array-task-id>.o` and the matching
:code:`ERRORS/...e`. Therefore a 200-task array creates 200 files in each
directory. :code:`hpc.max_job_log_files_per_directory` in
:code:`~/ichor_config.yaml` caps each directory independently; submission is
refused before :code:`sbatch` when the cap would be exceeded. A partial retry
uses a dense Slurm index and preserves the dense-to-logical mapping in the
attempt bundle.

Scratch is campaign-owned and lazy::

    .DATA/SCRATCH/<BACKEND>/<PHASE>/iteration-NNNNNN/
      <submission-identity>/job-<job-id>/task-<array-id-or-0>/

Each task atomically writes :code:`TASK.json`, exports
:code:`ICHOR_JOB_SCRATCH`, :code:`TMPDIR`, :code:`TMP`, and :code:`TEMP`, and
removes its leaf on success. Failed or abruptly interrupted task scratch is
retained. AIMAll :code:`.int` files, ARIADNE results, FEREBUS models/CSVs, and
all other scientific handoffs remain in their canonical directories; scratch
is never a scientific input and is not needed to validate completed work.

Inspect retained scratch before deleting anything::

    ichor-al-daemon reconcile --campaign-dir . --clean-scratch
    ichor-al-daemon reconcile --campaign-dir . --clean-scratch \
        --scratch-attempt r0000-a0001-1234abcd --apply

Clean-up refuses active jobs, scheduler-inconclusive ownership, symlinks,
path escapes, and malformed task metadata. The CLI menu performs a preview
and requires an explicit confirmation before applying clean-up.

Resource planning and telemetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same production resolver without writing state or submitting work::

    ichor-al-daemon resource-plan --campaign-dir .
    ichor-al-daemon resource-plan --campaign-dir . --phase AIMALL
    ichor-al-daemon resource-plan --campaign-dir . --all
    ichor-al-daemon resource-plan --campaign-dir . --all --json

The default is the current phase and iteration. Submitted or completed work
shows its immutable resource record; ready work is previewed from validated
evidence; local phases report no Slurm resources. Under :code:`--all`, future
evidence that has not yet been produced is informational. Explicitly
requesting such a phase returns exit code 14.

When :code:`resources.scheduler_usage_telemetry` is enabled, terminal jobs
record bounded Slurm accounting summaries in
:code:`.DATA/ACTIVE_LEARNING/resource_usage_records.json`. The record includes
RSS/VM and elapsed-time quantiles, maxima, failure and outlier samples, plus
advisory :code:`p95 * 1.25` memory and :code:`p95 * 1.5` walltime values.
Telemetry never modifies later requests automatically and a telemetry failure
does not invalidate scientific output.

ICHOR diversity sampling stores exact float64 condensed pair distances. It uses blockwise process
workers with one BLAS thread each, so it never constructs full
:code:`N x N` or :code:`N x N x F` arrays. If the condensed store exceeds the
configured in-memory fraction, it is placed in the task scratch leaf and free
space is checked both before submission and at job start.

Manchester CSF scratch is unbacked and subject to retention policy. Keep the
campaign on an appropriate project/scratch filesystem, monitor free space,
and copy important committed results to backed-up storage periodically. The
daemon warns when a live campaign root is under :code:`$HOME`.


Recovery + troubleshooting
--------------------------

Stopping and resuming
~~~~~~~~~~~~~~~~~~~~~

User stops use a separate atomic control file rather than allowing the
CLI process to rewrite ``state.json`` while the daemon owns it. The default
stop is immediate at the next daemon tick; it does not cancel Slurm jobs::

    ichor-al-daemon stop --campaign-dir .

Two receipt-backed drain modes are available::

    ichor-al-daemon stop --campaign-dir . --after-phase
    ichor-al-daemon stop --campaign-dir . --after-iteration
    ichor-al-daemon stop --campaign-dir . --after-iteration 12

``--after-phase`` finishes the current phase only when durable evidence shows
that the phase has started. Otherwise it stops before entering that phase.
``--after-iteration`` stops after the current iteration by default, or after
the explicitly numbered future iteration. Phase and iteration drains are
completed only alongside the daemon's normal phase-completion receipt, so a
restart cannot guess that a boundary was crossed.

Job cancellation is deliberately restricted to immediate stops::

    ichor-al-daemon stop --campaign-dir . --immediate --cancel-jobs

The cancellation result is recorded in the stop control before the daemon
clears matching pending-job and submission-intent metadata. If the CLI is
interrupted while cancellation is in progress, rerun the same command.

Use ``status`` and ``journal`` to monitor a drain. Once the boundary has been
reached, ordinary ``resume`` archives the completed request and clears the
stopped lifecycle. An unfinished drain survives a daemon crash and is still
honoured by ordinary ``resume``. Withdraw it explicitly only when intended::

    ichor-al-daemon resume --campaign-dir . --cancel-stop-request

The control file lives at
``.DATA/ACTIVE_LEARNING/stop_request.json``. Archived requests are retained
under ``.DATA/ACTIVE_LEARNING/stop_request_history/`` for user provenance.

The daemon writes three artefacts you care about during recovery:

- :code:`<campaign>/.DATA/ACTIVE_LEARNING/state.json` -- the canonical
  state checkpoint. Authoritative. If this file is missing or corrupt
  the daemon refuses to start; auto-recovery would mask the bug class we
  most want to catch.
- :code:`<campaign>/.DATA/ACTIVE_LEARNING/journal.ndjson` -- append-only
  per-phase event log. Best-effort; you can drop the file and the
  daemon still works, you just lose post-mortem provenance.
- :code:`<campaign>/.DATA/ACTIVE_LEARNING/stop_request.json` -- current
  user stop control. A malformed file is fail-closed and must be reviewed
  before reconcile can be applied.

If the daemon refuses to start because :code:`state.json` is missing or
fails schema validation, run::

    ichor-al-daemon reconcile --campaign-dir <DIR>

Fresh :code:`state.json` initialisation is allowed only for a clean first-run
campaign. If daemon-owned artefacts such as committed reference-data/model
iterations, staging directories, submission intents, scripts, journal entries,
or :code:`ACTIVE_LEARNING/iteration-*` outputs already exist, start refuses
to create a new state because that could damage provenance. Use reconcile
instead.

This inspects the campaign's authority records (committed-version pointers,
manifests, receipts, submission intents and recovery ledgers) and proposes a
recovered state at :code:`state.json.proposed`. Routine reconcile uses
authority verification: schemas, identities, version continuity, parent
hashes and pointer coherence are checked without recursively walking
pointdirs, scripts, logs or scratch. WFN, INT, Gaussian and model payloads are
not opened or statted. Plain reconcile is diagnostic: it may write the
proposal file and warn about a live daemon, but it does not alter
:code:`state.json`. Review the proposal, then use the guarded apply path::

    ichor-al-daemon reconcile --campaign-dir . --apply

If the active Python, ICHOR, ARIADNE, FEREBUS or machine-profile identity has
changed, guarded reconcile or the next start/resume automatically records a
new environment generation when the campaign is at a safe transition
boundary. An active job, submission intent, publication transaction or unsafe
phase blocks the transition with the exact reason. No separate environment
maintenance command is required. Normal status displays the recorded active
generation without inspecting the current package installation.

When :code:`state.json` is missing or malformed and campaign authority must be
reconstructed solely from committed artefacts, reconcile reports that deep
verification is required. It never starts that potentially long scan
implicitly. Run the exact explicit command it prints::

    ichor-al-daemon reconcile --campaign-dir . --deep-verify --apply

Use :code:`--deep-verify` for suspected corruption as well. Deep mode requires
a quiescent campaign and hashes every committed scientific payload once;
progress is written to stderr. Checkpoint creation, verification and restore
retain their existing deep-verification contracts.

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
:code:`pool.xyz` and :code:`.DATA/TRAJECTORY/pool.manifest.json` must exist,
and the SHA-256 in the manifest must match the campaign-local pool. Pool
corruption or deletion is reported
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

- except for the authenticated FEREBUS measurement-recovery path described
  above, it does not postprocess successful Slurm jobs whose daemon-side commit
  did not run;
- it does not harvest uncommitted Gaussian/AIMAll outputs into a training
  version;
- it does not rebuild committed manifests automatically;
- it does not roll back one-sided newer reference-data/model versions without
  user review.

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
is either INLINE (runs synchronously in the daemon process) or SBATCH
(submits via :code:`sbatch --parsable`, polls sacct, then runs a
postprocess parser in the daemon process).

.. code-block:: text

   INIT
     |
     v
   PHASE_A_DIVERSITY  (sbatch)        Initial diverse sub-sample.
     |
     v
   INITIAL_GAUSSIAN  (sbatch)     Single-point energies on the sample.
     |
     v
   INITIAL_AIMALL  (sbatch)       IQA decomposition.
     |
     v
   INITIAL_ALLOCATION_CHECK       Verify exact bootstrap train/internal/
     |                            external slots. Failed non-custom slots
     |----> INITIAL_REPLACEMENT_GAUSSIAN -> INITIAL_REPLACEMENT_AIMALL
     |                                      | (bounded reserve only)
     |<-------------------------------------+
     v
   REFERENCE_COMMIT  (inline)     Atomically publish accepted pointdirs and cached
     |                            FEREBUS rows as reference-data version 0.
     v
   INITIAL_FEREBUS  (sbatch*)     First GP fit. Commits trained-model version 0.
     |                            With
     |                            model_krig, validated models are imported
     |                            directly and no scheduler job is submitted.
     v
   SEED_SELECT  (inline)  <----+  Pick seeds from the trajectory pool,
     |                         |  forbidding already-trained frames.
     v                         |
   ARIADNE_ARRAY  (sbatch)      |  Per-seed adversarial descent.
     |                         |
     v                         |
   PHASE_B_DIVERSITY  (sbatch)      |  Exact ICHOR maximin sampling over the adversarial pool.
     |                         |
     v                         |
   SPLIT  (inline)              |  Persist the pre-QM exact train/internal-
     |                         |  validation slot allocation.
     v                         |
   GAUSSIAN  (sbatch)           |  Per-point energies on selected candidates.
     |                         |
     v                         |
   AIMALL  (sbatch)             |  IQA decomposition.
     |                         |
     v                         |
   ALLOCATION_CHECK  (inline)   |  Verify every exact slot is accepted.
     |----> REPLACEMENT_GAUSSIAN -> REPLACEMENT_AIMALL
     |                              | (same inherited slot and split)
     |<-----------------------------+
     v                         |
   REFERENCE_COMMIT  (inline)   |  Atomically move accepted AIMAll pointdirs and
     |                         |  publish the new row-cache delta.
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
journal at most one event behind. The inline REFERENCE_COMMIT and FEREBUS
commits are idempotent: re-running after a partial crash is safe. Reference
publication uses same-filesystem atomic moves, so normal operation does not
copy or rehash the accepted scientific payload. Each successful AIMAll task also
writes one compact FEREBUS row shard, consisting of ``ROW_SHARD.json`` and
``ROWS.npy``. ``REFERENCE_COMMIT`` validates and assembles those shards before
publishing the reference version. Missing or invalid shards are repaired
serially and reported explicitly; ordinary FEREBUS staging never reparses
pointdirs. Derived row caches live below ``.DATA/CACHE/FEREBUS_ROWS``. An
environment transition preserves them; their feature-contract and row-encoding
identities are validated before reuse, and incompatible caches are rebuilt
lazily in a separate namespace. Checkpoint restoration may also require lazy
cache reconstruction.
