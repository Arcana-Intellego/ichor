# Dry-run water-tetramer example

A complete runnable example campaign for the ICHOR active learning daemon.
Drives the daemon through TWO iterations using `--dry-run` which writes on the disk
real artefacts, .i.e., scripts, reference-data delta versions, manifests,
journal events, provenance sidecars) without calling Gaussian / AIMAll /
FEREBUS / ARIADNE / POLUS. Finishes in ~30 seconds.

## Contents

- `pool.xyz` -- 20-frame synthetic water-tetramer trajectory (12 atoms).
- `campaign.yaml` -- minimal config tuned for fast iteration. Overrides
  the defaults only where needed; every other key comes from
  `CampaignConfig()` (see `ichor_hpc/.../config.py` for the full schema).
- `expected_journal_tail.txt` -- representative excerpt of what the
  final journal events look like at the end of a successful 2-iteration
  dry-run.

## Walkthrough

Run from this directory.

### 1. Import the MD pool

```
python -m ichor.hpc.active_learning.cli init \
    --campaign-dir . \
    --yes
```

The CLI validates the campaign-local `pool.xyz` and writes a SHA-pinned
manifest. Output (your SHA will differ):

```
Imported pool: ./pool.xyz (20 frames, 12 atoms, SHA e006ddc6...)
```

### 2. Start the daemon in dry-run mode

```
python -m ichor.hpc.active_learning.cli start \
    --dry-run \
    --foreground \
    --campaign-dir . \
    --max-ticks 200
```

(The console-script shortcut `ichor-al-daemon` accepts the same arguments
once `ichor_hpc` is installed in your environment.)

Runs silently for ~30 seconds and exits cleanly. The daemon advances
through every campaign phase (PHASE_A_POLUS -> INITIAL_GAUSSIAN -> ... ->
FEREBUS -> STOP_CHECK -> DONE), commits two training-set + models
iterations, and writes a journal entry per phase transition.

### 3. Inspect the result

```
python -m ichor.hpc.active_learning.cli status --campaign-dir .
python -m ichor.hpc.active_learning.cli journal --campaign-dir . | tail -20
```

`status` shows the canonical state.json content. You should see
`phase: DONE`, `iteration: 2`, `reference_data_version: 2`,
`models_version: 2`, and an `alpha_history` with two entries.

`journal` prints the full per-phase event stream; the last few lines look
like `expected_journal_tail.txt` (your timestamps + JobIDs will differ).

## What to look at next

- `QM_REFERENCE_DATA/iteration-NNNNNN/` -- immutable QM reference-data delta for
  that version. It contains only newly accepted pointdirs, a SHA-pinned
  `REFERENCE_DATA_VERSION.json`, its point-allocation history, and the generic
  directory manifest. The authoritative resolver reconstructs cumulative
  order from versions `000000..NNNNNN`; it never copies older pointdirs forward.
- `.DATA/ACTIVE_LEARNING/reference_data_view_cache.json` -- a derived view
  cache. It may be deleted at any time and is rebuilt from the version chain.
- `TRAINED_MODELS/iteration-NNNNNN/` -- immutable complete model snapshot.
  `FEREBUS_TASK_ARTEFACTS.json` binds the exact models to the reference-data
  view and split rows; model/config/auxiliary files are colocated under
  `<property>/<atom>/`.
- `.DATA/BOOTSTRAP/selection/` and `.DATA/BOOTSTRAP/allocation/` -- the one-off Phase-A
  selection and exact bootstrap allocation. Bootstrap alone uses iteration 0.
- `ACTIVE_LEARNING/iteration-NNNNNN/` -- one immutable, hash-chained sampling
  iteration. Active iterations start at 1. Protocol snapshots, seed selection,
  ARIADNE outputs, Phase-B selection, allocation, and calibration each have a
  dedicated subdirectory.
- `ACTIVE_LEARNING/iteration-NNNNNN/ariadne/seeds/seed-NNNNNN/` -- full
  per-seed provenance, the selected result, a hash-bound output manifest, and
  `trajectory/trajectory.xyz` plus per-frame optimisation metrics. Scientific
  seed IDs start at 1; `array_task_id` remains explicitly zero-based.
- `.DATA/ACTIVE_LEARNING/seed_frame_id_index.json` -- the O(1) lookup index
  used by SEED_SELECT to forbid frames already seeded into the QM reference data.
- `.DATA/ACTIVE_LEARNING/journal.ndjson` -- the full append-only event log.
- `.DATA/SCRIPTS/*.sh` -- the sbatch scripts the daemon WOULD have submitted
  in live mode (each contains the module-loads + backend invocation it would
  run on a real CSF4 worker).

## Re-running the example

To wipe the dry-run output and start over:

```
rm -rf .DATA QM_REFERENCE_DATA TRAINED_MODELS ACTIVE_LEARNING
```

Then repeat steps 1-3 above. The .gitignore already excludes these
directories so they never accidentally land in a commit.

## Stepping to a live campaign

Once you have access to CSF4 (or another SLURM cluster with the required
backends), swap `--dry-run` for `--live`. The daemon will refuse with exit
code 12 if Gaussian / AIMAll / FEREBUS / ARIADNE / POLUS are not on PATH.
See `docs/source/active_learning_daemon.rst` Section 5 for the cluster-
side prerequisites + the canonical `module load` lines.
