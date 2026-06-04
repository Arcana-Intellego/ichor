# Dry-run water-tetramer example

A complete runnable example campaign for the ICHOR active learning daemon.
Drives the daemon through TWO iterations using `--dry-run` which writes on the disk
real artefacts, .i.e., scripts, training-set versions, manifests,
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
python -m ichor.hpc.active_learning.cli import-pool \
    --campaign-dir . \
    --source pool.xyz
```

The CLI copies `pool.xyz` to `.DATA/TRAJECTORY/pool.xyz` and writes a
SHA-pinned manifest. Output (your SHA will differ):

```
Imported pool: ./.DATA/TRAJECTORY/pool.xyz (20 frames, 12 atoms, SHA e006ddc6...)
```

### 2. Start the daemon in dry-run mode

```
python -m ichor.hpc.active_learning.cli start \
    --dry-run \
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
`phase: DONE`, `iteration: 1`, `training_set_version: 2`,
`models_version: 2`, and an `alpha_history` with two entries.

`journal` prints the full per-phase event stream; the last few lines look
like `expected_journal_tail.txt` (your timestamps + JobIDs will differ).

## What to look at next

- `5_TRAINING/iteration-NNNN/` -- committed training set per iteration,
  each with a SHA-pinned manifest.json and one POINT_NNNN.pointdir per
  selected seed (the dry-run stubs are 1:1 with `7_ACTIVE_LEARNING/.../pool/seed_NNNN/`).
- `6_TRAINED_MODELS/iteration-NNNN/` -- per-iteration model commit.
- `7_ACTIVE_LEARNING/iteration-NNNN/pool/seed_*/.provenance.json` -- full
  per-seed provenance trail (which frame_id, which subspace, which ARIADNE
  result, anti-overlap flag, phase-B diversity rank).
- `.DATA/ACTIVE_LEARNING/seed_frame_id_index.json` -- the O(1) lookup index
  used by SEED_SELECT to forbid frames already seeded into the training set.
- `.DATA/ACTIVE_LEARNING/journal.ndjson` -- the full append-only event log.
- `.DATA/SCRIPTS/*.sh` -- the sbatch scripts the daemon WOULD have submitted
  in live mode (each contains the module-loads + backend invocation it would
  run on a real CSF4 worker).

## Re-running the example

To wipe the dry-run output and start over:

```
rm -rf .DATA 3_DIVERSITY_SAMPLING 5_TRAINING 6_TRAINED_MODELS 7_ACTIVE_LEARNING
```

Then repeat steps 1-3 above. The .gitignore already excludes these
directories so they never accidentally land in a commit.

## Stepping to a live campaign

Once you have access to CSF4 (or another SLURM cluster with the required
backends), swap `--dry-run` for `--live`. The daemon will refuse with exit
code 12 if Gaussian / AIMAll / FEREBUS / ARIADNE / POLUS are not on PATH.
See `docs/source/active_learning_daemon.rst` Section 5 for the cluster-
side prerequisites + the canonical `module load` lines.
