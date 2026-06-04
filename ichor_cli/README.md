# ichor_cli

ICHOR's interactive menu system. A `console_menu`-based TUI that
wraps the common workflows from `ichor_core` and `ichor_hpc` so
operators can drive a campaign without writing scripts.

Depends on `ichor_core` and `ichor_hpc`.

## What this package does

- **Main menu** (`ichor` command): top-level navigation into every
  ICHOR workflow.
- **Active-learning campaign menu**: import a trajectory pool, view /
  edit the campaign config (nested-block submenus for every
  `CampaignConfig` block including the full `acquisition` tree),
  start / stop / reconcile the daemon, browse the journal.
- **File-management menus**: PointDirectory / PointsDirectory walkers,
  manifest viewers, training-set inspectors.
- **Job-submission shortcuts**: one-shot Gaussian / AIMAll / FEREBUS
  array submissions for ad-hoc analysis (outside the daemon).

## Install

```
python3 -m pip install -e ichor_core[dev]
python3 -m pip install -e ichor_hpc
python3 -m pip install -e ichor_cli
```

(Order matters; `ichor_cli` imports from both other packages.)

## Run

```
ichor
```

The first screen prompts for a campaign directory if one is not yet
configured. Use the menu numbers (or arrow keys + Enter) to navigate.

## Tests

```
pytest ichor_cli/tests
```

14 smoke tests verify the menu modules import and have the expected
items registered. Heavier integration coverage lives in `ichor_hpc`.

## Documentation

- Menu walkthrough: <https://ichor.readthedocs.io/en/latest/ichor_cli/ichor.cli.html>.
- Daemon user guide (the most-used menu chain):
  `docs/source/active_learning_daemon.rst`.

## Where to go from here

For headless / scripted use, skip the menu and call
`ichor-al-daemon` directly (the same workflow without the prompts).
See `ichor_hpc/README.md`.
