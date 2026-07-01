"""Dry-run PhaseExecutor for end-to-end smoke testing of the daemon.

    > On CSF4 scratch, run "ichor-al-daemon start --dry-run"
    > against a small fixture (water tetramer, 100 frames). Verify all
    > directories, manifests, journal entries, sbatch invocations, sacct polls,
    > and atomic renames work end-to-end without real Gaussian / AIMAll /
    > FEREBUS.

What the dry-run executor DOES exercise (real, no mocking):

    * Directory layout setup (5_TRAINING/, 6_TRAINED_MODELS/, 7_ACTIVE_LEARNING/,
      .DATA/SCRIPTS/).
    * TrainingSetVersioning (stage, commit, update_current, manifest writes,
      atomic renames).
    * ".DATA/SCRIPTS/*.sh" stub script creation per SLURM-backed phase.
    * Journal events for every phase entry / postprocess / commit.

What it DOES NOT do (stubbed):

    * Gaussian, AIMAll, FEREBUS subprocesses are never spawned. Stub
      artefacts are produced in the canonical directory layout so a human
      operator can inspect the dry-run output.
    * ARIADNE is invoked through :func:"optimise_seed(mock=True)" so the
      adversarial pool is synthetic.

The executor is deterministic given a seeded RNG so the integration test
can assert exact counts of artefacts produced.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import CampaignConfig
from ..versioning.provenance import (
    DEFAULT_RECENT_SEEDS_COOLDOWN,
    PROVENANCE_FILENAME,
    append_recent_seeds,
    append_to_index,
    enrich_with_anti_overlap,
    enrich_with_ariadne,
    enrich_with_error_calibration_input,
    enrich_with_phase_b,
    ensure_index,
    load_recent_seed_frame_ids,
    load_recent_seeds_payload,
    load_training_seed_frame_ids,
    write_seed_provenance,
)
from ..versioning.training_set import TrainingSetVersioning
from .phase_executor import (
    BackendSubmissionError,
    FailureAction,
    INLINE_PHASES,
    PhaseResult,
    SBATCH_PHASES,
)
from .state import CampaignPhase, atomic_write_json


__all__ = [
    "DryRunPhaseExecutor",
    "DRYRUN_JOB_PREFIX",
]


DRYRUN_JOB_PREFIX = "DRYRUN-"


def anti_overlap_whitened_distance_bounds(config: CampaignConfig) -> tuple[float, float]:
    anti = getattr(config, "anti_overlap", None)
    return (
        float(getattr(anti, "min_post_ariadne_whitened_distance", 0.01)),
        float(getattr(anti, "max_post_ariadne_whitened_distance", 10.0)),
    )


@dataclass
class DryRunPhaseExecutor:
    """PhaseExecutor that drives the full file-system flow with stub artefacts."""

    campaign_dir: Path
    config: CampaignConfig
    rng_seed: int = 0
    training_dir_name: str = "5_TRAINING"
    models_dir_name: str = "6_TRAINED_MODELS"
    diversity_dir_name: str = "3_DIVERSITY_SAMPLING"
    al_dir_name: str = "7_ACTIVE_LEARNING"
    scripts_dir: Path = field(init=False)
    artefact_log: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.campaign_dir = Path(self.campaign_dir)
        self.scripts_dir = self.campaign_dir / ".DATA" / "SCRIPTS"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        (self.campaign_dir / self.training_dir_name).mkdir(parents=True, exist_ok=True)
        (self.campaign_dir / self.models_dir_name).mkdir(parents=True, exist_ok=True)
        (self.campaign_dir / self.diversity_dir_name).mkdir(parents=True, exist_ok=True)
        (self.campaign_dir / self.al_dir_name).mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(self.rng_seed)

    # --- PhaseExecutor protocol ----------------------------------------

    def submit_or_run(self, state, phase) -> PhaseResult:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        if phase_name in SBATCH_PHASES:
            self._write_stub_script(phase_name, state.iteration)
            job_id = (
                DRYRUN_JOB_PREFIX + phase_name + "-" + str(state.iteration)
            )
            return PhaseResult(is_complete=False, submitted_job_id=job_id)
        # inline phases produce their artefacts synchronously.
        updates = self._run_inline(state, phase_name)
        return PhaseResult(is_complete=True, state_updates=updates)

    def postprocess(self, state, phase, observations: Sequence[Any]) -> PhaseResult:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        updates = self._postprocess_phase(state, phase_name)
        return PhaseResult(is_complete=True, state_updates=updates)

    def handle_failure(self, state, phase, observations) -> FailureAction:
        # in dry-run we never produce failed jobs, but provide a sensible default
        # in case the caller wires us up with a hostile sacct stub.
        return FailureAction.SCRUB_AND_CONTINUE

    # --- internal: script stubbing -------------------------------------

    def _write_stub_script(self, phase_name: str, iteration: int) -> Path:
        path = self.scripts_dir / (phase_name + "-" + str(iteration) + ".sh")
        lines = [
            "#!/bin/sh",
            "# DRY-RUN stub for phase " + phase_name + ", iteration " + str(iteration),
            "# Generated by DryRunPhaseExecutor; no real backend invoked.",
            'echo "DRYRUN ' + phase_name + ' iteration=' + str(iteration) + '"',
            "exit 0",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        self.artefact_log.append(str(path))
        return path

    def _versioning(self, kind: str) -> TrainingSetVersioning:
        if kind == "training":
            return TrainingSetVersioning(self.campaign_dir / self.training_dir_name)
        if kind == "models":
            return TrainingSetVersioning(self.campaign_dir / self.models_dir_name)
        raise ValueError("unknown versioning kind: " + kind)

    # --- internal: inline phases ---------------------------------------

    def _run_inline(self, state, phase_name: str) -> Dict[str, Any]:
        handler = self._inline_handlers().get(phase_name)
        if handler is None:
            # INIT, DONE, HALTED, STOP_CHECK fall through with no work.
            return {}
        return handler(state)

    def _inline_handlers(self):
        return {
            "SEED_SELECT": self._inline_seed_select,
            "SPLIT": self._inline_split,
            "APPEND": self._inline_append,
            "STOP_CHECK": self._inline_stop_check,
        }

    def _seed_selection_posterior(self, state, training_atoms):
        """Posterior whose .variance(atoms) drives the exploit half of seed
        selection. The dry run has no trained GP, so every frame looks equally
        uncertain (a flat posterior); the live executor overrides this with the
        real GP variance."""
        class _UniformPosterior:
            def variance(self, _atoms):
                return 1.0
            def covariance(self, atoms_a, atoms_b):
                return 1.0 if atoms_a is atoms_b else 0.0
        return _UniformPosterior()

    def _seed_selection_score_transform(self):
        """Optional calibrated score transform for D-optimal seed ranking.

        This is deliberately cheap: it maps the already-computed total
        posterior variance through the global empirical calibration table.
        Missing, sparse, record-only, or malformed calibration falls back to
        raw posterior variance.
        """
        try:
            from .error_calibration import (
                load_calibration_model_for_acquisition,
                lookup_calibrated_abs_error,
            )

            model, reason = load_calibration_model_for_acquisition(
                self.campaign_dir,
                self.config,
            )
            if model is None:
                return None, reason

            def _transform(_selection_index: int, posterior_variance: float):
                return lookup_calibrated_abs_error(model, posterior_variance)

            return _transform, "loaded"
        except Exception as exc:
            return None, "error:" + type(exc).__name__

    def _inline_seed_select(self, state) -> Dict[str, Any]:
        """Current wiring: SEED_SELECT invokes :func:'select_seeds' against the
        canonical trajectory pool (if imported) with 'forbidden_frame_ids' =
        (committed-training seeds) | (recent-seeds cooldown cache).

        Behaviour:
          * If no trajectory pool is present (legacy dry-run without
            'ichor-al-daemon init'), writes a placeholder seeds.xyz
            and journals 'seed_selected' with 'pool_available=False'. This
            preserves the existing test_dry_run_executor.py contract.
          * If a pool IS present, treats every frame as an eligible seed
            row, uses a uniform-variance stub posterior (the real
            TotalEnergyPosterior only kicks in once the live executor
            has real FEREBUS models to lean on), and
            writes a seeds_picked.json artefact in the iteration dir so
            ARIADNE post-processing can stamp the real seed_frame_id into
            the per-pointdir provenance.
        """
        from ..acquisition.seed_selection import select_seeds
        from ..acquisition.trajectory_pool import TrajectoryPool

        iter_dir = self._iter_dir(state.iteration)
        seeds_path = iter_dir / "seeds.xyz"
        # re-entry guard: if this iteration's seeds were already picked (a
        # crash-retry after seeds_picked.json was written) don't pick again --
        # re-selecting would choose different frames and burn cooldown slots on
        # the abandoned ones. selection stays idempotent on re-run.
        if (iter_dir / "seeds_picked.json").is_file():
            from ..handoff_manifests import load_seeds_picked

            picked = load_seeds_picked(iter_dir, expected_iteration=int(state.iteration))
            history = load_recent_seeds_payload(self.campaign_dir).get("history", [])
            has_current = any(
                isinstance(entry, dict)
                and int(entry.get("iteration", -999999)) == int(state.iteration)
                for entry in history
            )
            if not has_current:
                append_recent_seeds(
                    self.campaign_dir,
                    iteration=int(state.iteration),
                    frame_ids=list(picked.get("frame_ids") or []),
                    cooldown=self._recent_seed_cooldown(),
                )
            self._journal_event(
                "seed_selected", iteration=int(state.iteration), reused_existing=True,
                repaired_recent_seeds=(not has_current),
            )
            return {}
        ensure_index(self.campaign_dir)
        # refresh reference scales per acquisition.references.refresh_policy.
        self._maybe_refresh_reference_scales(state)

        try:
            pool = TrajectoryPool.load(self.campaign_dir)
        except FileNotFoundError:
            seeds_path.write_text(
                "# DRYRUN seed selection for iteration "
                + str(state.iteration) + "\n"
                + "# n_seeds_per_iteration = "
                + str(self.config.seed_selection.n_seeds_per_iteration) + "\n"
                + "# pool_available=False (no .DATA/TRAJECTORY/pool.xyz imported)\n",
                encoding="utf-8",
            )
            self.artefact_log.append(str(seeds_path))
            self._journal_event(
                "seed_selected",
                iteration=int(state.iteration),
                pool_available=False,
                n_picked=0,
                forbidden_set_size=0,
            )
            return {}

        # anti-overlap case (a): forbid frame_ids already used to seed a committed training point.
        # honour the config switch -- it used to be hardcoded on, so an operator setting
        # skip_training_seeds: false (e.g. to allow re-seeding training frames on a tiny system) was
        # silently ignored (A35).
        if getattr(self.config.anti_overlap, "skip_training_seeds", True):
            training_forbidden = load_training_seed_frame_ids(
                self.campaign_dir,
                training_dir=self.campaign_dir / self.training_dir_name,
            )
        else:
            training_forbidden = set()
        recent_forbidden = load_recent_seed_frame_ids(self.campaign_dir)
        forbidden = frozenset(training_forbidden | recent_forbidden)

        training_atoms = pool.to_atoms_list()
        training_frame_ids = list(pool.frame_ids())

        posterior = self._seed_selection_posterior(state, training_atoms)
        if str(self.config.seed_selection.strategy) == "d_optimal":
            score_transform, score_transform_reason = self._seed_selection_score_transform()
        else:
            score_transform, score_transform_reason = None, "strategy_hybrid_variance"

        selection = select_seeds(
            training_atoms,
            posterior,
            n_seeds=int(self.config.seed_selection.n_seeds_per_iteration),
            bulk_fraction=float(self.config.seed_selection.bulk_fraction),
            rng_seed=int(self.rng_seed) + int(state.iteration),
            training_frame_ids=training_frame_ids,
            forbidden_frame_ids=forbidden,
            variance_chunk_size=int(self.config.seed_selection.variance_chunk_size),
            strategy=str(self.config.seed_selection.strategy),
            d_optimal_pool_multiplier=int(
                self.config.seed_selection.d_optimal_pool_multiplier
            ),
            d_optimal_jitter=float(self.config.seed_selection.d_optimal_jitter),
            d_optimal_novelty_floor=float(
                self.config.seed_selection.d_optimal_novelty_floor
            ),
            d_optimal_score_power=float(
                self.config.seed_selection.d_optimal_score_power
            ),
            score_transform=score_transform,
        )
        requested_n = int(self.config.seed_selection.n_seeds_per_iteration)
        if selection.n <= 0:
            raise BackendSubmissionError(
                "seed_pool_exhausted: no eligible trajectory frames remain "
                "after training/recent-seed exclusion; requested="
                + str(requested_n)
                + ", training_forbidden="
                + str(len(training_forbidden))
                + ", recent_forbidden="
                + str(len(recent_forbidden))
                + ", forbidden_union="
                + str(len(forbidden))
                + ", skip_training_seeds="
                + str(bool(getattr(self.config.anti_overlap, "skip_training_seeds", True)))
                + ", recent_seeds_cooldown="
                + str(int(getattr(self.config.anti_overlap, "recent_seeds_cooldown", 0)))
            )
        if selection.n < requested_n:
            raise BackendSubmissionError(
                "seed_pool_exhausted: only "
                + str(selection.n)
                + " eligible trajectory frames remain for requested batch "
                + str(requested_n)
                + "; training_forbidden="
                + str(len(training_forbidden))
                + ", recent_forbidden="
                + str(len(recent_forbidden))
                + ", forbidden_union="
                + str(len(forbidden))
                + ", skip_training_seeds="
                + str(bool(getattr(self.config.anti_overlap, "skip_training_seeds", True)))
                + ", recent_seeds_cooldown="
                + str(int(getattr(self.config.anti_overlap, "recent_seeds_cooldown", 0)))
            )

        bulk_set = {int(i) for i in selection.bulk_indices}
        variance_set = {int(i) for i in selection.variance_indices}
        seed_records = []
        diagnostics_by_index = {
            int(row.get("selection_index")): dict(row)
            for row in selection.selection_diagnostics
            if isinstance(row, dict) and row.get("selection_index") is not None
        }
        for seed_index, frame_id in enumerate(selection.frame_ids):
            selection_index = int(selection.indices[seed_index])
            if seed_index < len(selection.selection_origins):
                origin = str(selection.selection_origins[seed_index])
            elif selection_index in bulk_set:
                origin = "bulk"
            elif selection_index in variance_set:
                origin = "variance"
            else:
                origin = "unknown"
            variance_value = None
            if seed_index < len(selection.variances):
                variance_value = float(selection.variances[seed_index])
            record = {
                "seed_index": int(seed_index),
                "frame_id": frame_id if isinstance(frame_id, int) else None,
                "selection_index": selection_index,
                "selection_origin": origin,
                "variance_at_selection": variance_value,
            }
            diag = diagnostics_by_index.get(selection_index, {})
            for key in (
                "raw_variance",
                "raw_score",
                "d_optimal_conditional_variance",
                "d_optimal_gain",
                "d_optimal_prefilter_rank",
                "d_optimal_max_correlation_to_selected",
                "variance_rank",
            ):
                if key in diag:
                    record[key] = diag[key]
            seed_records.append(record)

        seeds_picked_path = iter_dir / "seeds_picked.json"
        seeds_picked_payload = {
            "schema_version": 1,
            "iteration": int(state.iteration),
            "selection_strategy": str(self.config.seed_selection.strategy),
            "n_picked": int(selection.n),
            "frame_ids": list(selection.frame_ids),
            "indices": list(selection.indices),
            "bulk_indices": list(selection.bulk_indices),
            "variance_indices": list(selection.variance_indices),
            "d_optimal_indices": [
                int(rec["selection_index"])
                for rec in seed_records
                if rec.get("selection_origin") == "d_optimal"
            ],
            "variances": [float(v) for v in selection.variances],
            "seed_records": seed_records,
            "forbidden_set_size": int(len(forbidden)),
            "skipped_unknown_provenance": int(selection.skipped_unknown_provenance),
            "score_source": (
                "error_calibration"
                if score_transform is not None
                else "posterior_variance"
            ),
            "score_source_reason": str(score_transform_reason),
            # pin the trajectory this selection was made against. frame ids are positional, so if
            # the pool ever got re-imported/swapped underneath the campaign, frame 7 would now be a
            # different geometry -- ARIADNE on the compute node cross-checks this and refuses rather
            # than silently attack the wrong point (A30).
            "trajectory_sha256": pool.sha256,
        }
        atomic_write_json(seeds_picked_path, seeds_picked_payload)
        from ..handoff_manifests import write_seed_selection_diagnostics

        diagnostics_payload = {
            "iteration": int(state.iteration),
            "strategy": str(self.config.seed_selection.strategy),
            "requested_n": int(requested_n),
            "n_picked": int(selection.n),
            "n_total": int(selection.diagnostics.get("n_total", len(training_atoms))),
            "n_eligible": int(selection.diagnostics.get("n_eligible", len(selection.indices))),
            "n_bulk": int(len(selection.bulk_indices)),
            "n_ranked": int(len(selection.variance_indices)),
            "prefilter_pool_size": int(selection.diagnostics.get("prefilter_pool_size", 0)),
            "forbidden_set_size": int(len(forbidden)),
            "skipped_unknown_provenance": int(selection.skipped_unknown_provenance),
            "score_source": (
                "error_calibration"
                if score_transform is not None
                else "posterior_variance"
            ),
            "score_source_reason": str(score_transform_reason),
            "trajectory_sha256": pool.sha256,
            "selected": seed_records,
            "summary": dict(selection.diagnostics),
        }
        seed_diag_path = write_seed_selection_diagnostics(iter_dir, diagnostics_payload)

        lines_out = [
            "# DRYRUN seed selection for iteration " + str(state.iteration),
            "# pool_available=True n_picked=" + str(selection.n)
            + " forbidden_set_size=" + str(len(forbidden))
            + " skipped_unknown_provenance="
            + str(selection.skipped_unknown_provenance),
        ]
        for k, fid in enumerate(selection.frame_ids):
            lines_out.append("# seed[" + str(k) + "].frame_id = " + str(fid))
        seeds_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        self.artefact_log.append(str(seeds_path))
        self.artefact_log.append(str(seeds_picked_path))
        self.artefact_log.append(str(seed_diag_path))

        append_recent_seeds(
            self.campaign_dir,
            iteration=int(state.iteration),
            frame_ids=selection.frame_ids,
            cooldown=self._recent_seed_cooldown(),
        )

        self._journal_event(
            "seed_selected",
            iteration=int(state.iteration),
            pool_available=True,
            selection_strategy=str(self.config.seed_selection.strategy),
            score_source=(
                "error_calibration"
                if score_transform is not None
                else "posterior_variance"
            ),
            score_source_reason=str(score_transform_reason),
            n_picked=int(selection.n),
            forbidden_set_size=int(len(forbidden)),
            skipped_unknown_provenance=int(selection.skipped_unknown_provenance),
        )
        #seeds_picked + forbidden_set_size live in the journal
        #("seed_selected" event above) and the seeds_picked.json sidecar;
        # they are not persisted state. The empty return keeps the
        # contract with daemon._apply_state_updates strict.
        return {}
    def _inline_split(self, state) -> Dict[str, Any]:
        """Sort this iteration's candidate set into train,
        validation and (optionally) holdout buckets using the strategy
        the operator picked in campaign.yaml split.strategy.

        The candidate set is whatever the ARIADNE descent + Phase-B FPS
        produced -- one row per seed in the iteration pool. Each row has
        an acquisition alpha attached, which the strategy sorts on.

        Strategies live in sampling/split.py:
          - stratified_with_holdout (default): top alphas go to train but
            a fraction of the top tier is held back as validation, so
            the validation set actually contains the points the model
            needs to learn from. mid-tier rows also go to validation.
          - random_80_20: plain uniform random split with train_fraction.
          - pure_top_k: top fraction to train, rest to validation, no
            holdout. simplest but validation never sees the hard cases.

        We write split.json with the schema that APPEND and reconcile
        downstream expect.
        """
        from ..sampling.split import get_split_strategy

        iter_dir = self._iter_dir(state.iteration)
        pool_dir = iter_dir / "pool"
        split_path = iter_dir / "split.json"

        # gather the per-candidate alpha values from each seed result.json.
        # the result.json files are written by ARIADNE_ARRAY postprocess
        # earlier in this iteration; if any are missing we fall back to
        # a default alpha of 0.0 for that row so the split still runs.
        seed_dirs = []
        if pool_dir.is_dir():
            seed_dirs = sorted(
                d for d in pool_dir.iterdir()
                if d.is_dir() and d.name.startswith("seed_")
            )
        alphas: List[float] = []
        for sd in seed_dirs:
            rj = sd / "result.json"
            if rj.is_file():
                try:
                    with open(rj, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    alpha_f = float(data.get("alpha_final", 0.0) or 0.0)
                except (OSError, ValueError):
                    alpha_f = 0.0
            else:
                alpha_f = 0.0
            alphas.append(alpha_f)

        # If we ended up with nothing (no seed dirs at all, eg SEED_SELECT
        # ran without a trajectory pool), fall back to the final active batch
        # size so APPEND still has something to do in synthetic dry runs.
        if not alphas:
            final_batch_size = int(self.config.active_batch.final_batch_size)
            payload = {
                "strategy": self.config.split.strategy,
                "train_fraction": self.config.split.train_fraction,
                "val_mid_fraction": self.config.split.val_mid_fraction,
                "high_holdout_fraction": self.config.split.high_holdout_fraction,
                "iteration": state.iteration,
                "train_indices": list(range(final_batch_size)),
                "val_indices": [final_batch_size],
                "holdout_indices": [],
            }
            atomic_write_json(split_path, payload)
            self.artefact_log.append(str(split_path))
            return {}

        # dispatch to the configured strategy. each strategy takes the
        # acquisition values plus the fractions it cares about.
        strategy_fn = get_split_strategy(self.config.split.strategy)
        rng_seed = int(self.rng_seed) + int(state.iteration)
        if self.config.split.strategy == "stratified_with_holdout":
            result = strategy_fn(
                alphas,
                train_fraction=float(self.config.split.train_fraction),
                val_mid_fraction=float(self.config.split.val_mid_fraction),
                high_holdout_fraction=float(self.config.split.high_holdout_fraction),
                rng_seed=rng_seed,
            )
        elif self.config.split.strategy == "random_80_20":
            result = strategy_fn(
                alphas,
                train_fraction=float(self.config.split.train_fraction),
                rng_seed=rng_seed,
            )
        elif self.config.split.strategy == "pure_top_k":
            result = strategy_fn(
                alphas,
                train_fraction=float(self.config.split.train_fraction),
            )
        else:
            # _validate caught unknown strategies earlier so we should
            # never land here -- but fall back to the default rather
            # than crash if somehow we do.
            result = strategy_fn(alphas)
        train_set = set(int(i) for i in result.train_indices)
        val_set = set(int(i) for i in result.val_indices)
        holdout_set = set(int(i) for i in result.holdout_indices)
        if train_set & val_set:
            raise BackendSubmissionError("split strategy returned overlapping train/val indices")
        if not holdout_set.issubset(val_set):
            raise BackendSubmissionError("split strategy returned holdout indices outside validation")
        if result.n_train <= 0 and len(alphas) > 0:
            raise BackendSubmissionError("split strategy returned no training rows for non-empty candidates")

        payload = {
            "strategy": result.strategy,
            "train_fraction": float(self.config.split.train_fraction),
            "val_mid_fraction": float(self.config.split.val_mid_fraction),
            "high_holdout_fraction": float(self.config.split.high_holdout_fraction),
            "iteration": int(state.iteration),
            "train_indices": list(result.train_indices),
            "val_indices": list(result.val_indices),
            "holdout_indices": list(result.holdout_indices),
        }
        atomic_write_json(split_path, payload)
        self.artefact_log.append(str(split_path))
        return {}

    def _inline_append(self, state) -> Dict[str, Any]:
        """Stage and commit the next training iteration via TrainingSetVersioning.

        The wiring: provenance sidecars from this iteration pool/seed_NNNN/
        are copied into the committed iteration directory as a synthetic 1:1
        mapping (POINT_k.pointdir <- seed_k); then the daemon
        seed_frame_id_index.json grows by one record per committed pointdir.

        Idempotency: if "state.training_set_version + 1" is already
        a committed version (we crashed between commit() and _persist on
        the previous run), skip the stage/commit cycle and return the
        existing version. Prevents orphan-iteration-directory bugs that
        the persist-before-journal swap would otherwise expose.
        """
        import shutil as _shutil

        v = self._versioning("training")
        committed = v.list_committed_versions()
        state_version = int(getattr(state, "training_set_version", 0))
        committed_max = max(committed) if committed else -1
        if committed_max > state_version:
            if int(committed_max) != int(state_version) + 1:
                raise BackendSubmissionError(
                    "training_set_version_gap: state="
                    + str(state_version)
                    + " committed_versions="
                    + repr(committed)
                )
            #already committed in a previous run; idempotent no-op.
            self._journal_event(
                "training_set_committed",
                iteration=int(state.iteration),
                training_set_version=int(committed_max),
                n_committed_points=0,
                idempotent_skip=True,
            )
            v.ensure_current(committed_max)
            return {"training_set_version": int(committed_max)}
        if committed_max < state_version:
            raise BackendSubmissionError(
                "training_set_version "
                + str(state_version)
                + " is ahead of committed training versions "
                + repr(committed)
            )
        if committed:
            v.ensure_current(committed_max)
        next_version = committed_max + 1
        v.recover_dangling_staging()
        source = committed_max if committed_max >= 0 else None
        staging = v.stage(source_version=source, target_version=next_version)
        marker = staging / ("DRYRUN_iter_" + str(state.iteration) + ".txt")
        marker.write_text(
            "DRYRUN training append at iteration " + str(state.iteration) + "\n"
            + "next_version=" + str(next_version) + "\n",
            encoding="utf-8",
        )
        pool_dir = self._iter_dir(state.iteration) / "pool"
        committed_pointdirs = []
        if pool_dir.is_dir():
            for seed_dir in sorted(d for d in pool_dir.iterdir() if d.is_dir()):
                k = seed_dir.name.split("_")[-1]
                pdir = staging / ("POINT_" + k + ".pointdir")
                pdir.mkdir(exist_ok=True)
                src_prov = seed_dir / PROVENANCE_FILENAME
                if src_prov.is_file():
                    _shutil.copyfile(src_prov, pdir / PROVENANCE_FILENAME)
                (pdir / ("DRYRUN_POINT_iter_" + str(state.iteration) + ".txt")).write_text(
                    "DRYRUN committed point for iteration " + str(state.iteration) + "\n",
                    encoding="utf-8",
                )
                committed_pointdirs.append(pdir.name)
        v.commit(next_version)
        v.update_current(next_version)
        ensure_index(self.campaign_dir)
        committed_iter_dir = v.iteration_path(next_version)
        for pdir_name in committed_pointdirs:
            pdir = committed_iter_dir / pdir_name
            seed_frame_id = self._read_seed_frame_id_from_pointdir(pdir)
            append_to_index(
                self.campaign_dir,
                iteration=int(next_version),
                pointdir_name=pdir_name,
                seed_frame_id=seed_frame_id,
            )
        self._journal_event(
            "training_set_committed",
            iteration=int(state.iteration),
            training_set_version=int(next_version),
            n_committed_points=int(len(committed_pointdirs)),
            expected_final_batch_size=int(self.config.active_batch.final_batch_size),
        )
        self.artefact_log.append(str(self._iter_dir(state.iteration) / "APPEND_marker"))
        return {"training_set_version": int(next_version)}

    def _inline_stop_check(self, state):
        """roll the last observed alpha0 into alpha_history and apply the
        config.stop convergence triggers. Two rules:

          - streak: alpha0 stayed below stop.alpha0_streak_threshold for
            the last stop.alpha0_streak_length iterations.
          - improvement: relative alpha improvement stayed below
            stop.rel_alpha_improvement_min for the last
            stop.rel_alpha_improvement_window iterations.

        Both gated by stop.min_iterations_before_stop.
        """
        stop = self.config.stop
        history = list(getattr(state, "alpha_history", None) or [])
        latest = getattr(state, "last_acquisition_alpha0", None)
        if latest is not None:
            try:
                history.append(float(latest))
            except (TypeError, ValueError):
                pass
        #keep at most max(streak_length, improvement_window) + 1 entries.
        cap = max(int(stop.alpha0_streak_length), int(stop.rel_alpha_improvement_window)) + 1
        if len(history) > cap:
            history = history[-cap:]

        shutdown = False
        reason = None
        if int(state.iteration) + 1 >= int(stop.min_iterations_before_stop):
            n_streak = int(stop.alpha0_streak_length)
            if n_streak > 0 and len(history) >= n_streak:
                tail = history[-n_streak:]
                if all(x < float(stop.alpha0_streak_threshold) for x in tail):
                    shutdown = True
                    reason = "alpha0_streak"
            n_window = int(stop.rel_alpha_improvement_window)
            if not shutdown and n_window > 0 and len(history) >= n_window + 1:
                tail = history[-(n_window + 1):]
                rels = []
                for prev, cur in zip(tail[:-1], tail[1:]):
                    denom = abs(prev)
                    # SIGNED, not clamped to >= 0. alpha is MAXIMISED, so cur > prev means it ROSE
                    # (the model got worse, or found a fresh hard region) -- that gives a negative
                    # rel and so reads as "not flat", which is exactly what we want: a rising alpha
                    # must never trigger a stop. the old max(0, ...) flattened a rise to 0, making it
                    # look identical to a converged plateau and nudging us toward shutdown (A50).
                    rels.append(0.0 if denom <= 0.0 else (prev - cur) / denom)
                # flat = alpha barely moved in EITHER direction across the whole window.
                flat = bool(rels) and all(abs(r) < float(stop.rel_alpha_improvement_min) for r in rels)
                # ...but a flat plateau only counts as CONVERGED when it is also LOW. a flat-but-high
                # alpha is the loop STUCK at a bad level, not converged -- it must keep going. reuse
                # the streak threshold as the "this alpha is low" bar (A50).
                low_now = history[-1] < float(stop.alpha0_streak_threshold)
                if flat and low_now:
                    shutdown = True
                    reason = "rel_alpha_improvement_plateau"

        updates = {"alpha_history": history}
        if shutdown:
            updates["shutdown_requested"] = True
            self._journal_event(
                "shutdown_requested",
                iteration=int(state.iteration),
                reason=str(reason),
                history_tail=[float(x) for x in history[-min(len(history), 8):]],
            )
        return updates

    def _iter_dir(self, iteration: int) -> Path:
        d = self.campaign_dir / self.al_dir_name / ("iteration-" + str(iteration).zfill(4))
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- internal: postprocess (SLURM phases) --------------------------

    def _postprocess_phase(self, state, phase_name: str) -> Dict[str, Any]:
        handler = self._postprocess_handlers().get(phase_name)
        if handler is None:
            return {}
        return handler(state)

    def _postprocess_handlers(self):
        return {
            "PHASE_A_POLUS": self._post_phase_a_polus,
            "INITIAL_GAUSSIAN": self._post_initial_gaussian,
            "INITIAL_AIMALL": self._post_initial_aimall,
            "INITIAL_FEREBUS": self._post_initial_ferebus,
            "ARIADNE_ARRAY": self._post_ariadne_array,
            "PHASE_B_POLUS": self._post_phase_b_polus,
            "GAUSSIAN": self._post_gaussian,
            "AIMALL": self._post_aimall,
            "FEREBUS": self._post_ferebus,
        }

    # --- per-phase postprocess handlers --------------------------------

    def _post_phase_a_polus(self, state) -> Dict[str, Any]:
        from ..handoff_manifests import write_phase_a_sample_manifest

        outdir = self.campaign_dir / self.diversity_dir_name / "initial"
        outdir.mkdir(parents=True, exist_ok=True)
        n = int(self.config.bootstrap.initial_labelled_size)
        sample = outdir / ("initial-SAMPLE-" + str(n) + ".xyz")
        sample.write_text(
            "# DRYRUN POLUS Phase-A sample (n=" + str(n) + ")\n",
            encoding="utf-8",
        )
        index = outdir / ("initial-INDEX-" + str(n) + ".dat")
        index.write_text("\n".join(str(i) for i in range(n)) + "\n", encoding="utf-8")
        manifest = write_phase_a_sample_manifest(outdir, {
            "phase": "PHASE_A_POLUS",
            "iteration": -1,
            "sample_xyz": str(sample.resolve()),
            "index_path": str(index.resolve()),
            "n_select": int(n),
            "n_frames": int(n),
            "selected_indices": [int(i) for i in range(n)],
            "descriptor": "rmsd_massweight",
            "n_pool_frames": int(n),
            "bootstrap_initial_labelled_size": int(n),
            "reserve_after_bootstrap": 0,
            "trajectory_sha256": "",
            "source_pool_manifest": "",
        })
        self.artefact_log.extend([str(sample), str(index), str(manifest)])
        return {}

    def _post_initial_gaussian(self, state) -> Dict[str, Any]:
        return self._stub_quantum_outputs(state, stage="GAUSSIAN", initial=True)

    def _post_initial_aimall(self, state) -> Dict[str, Any]:
        return self._stub_quantum_outputs(state, stage="AIMALL", initial=True)

    def _post_initial_ferebus(self, state) -> Dict[str, Any]:
        """Commit the initial training iteration and write a stub model file.

        This is the first time the training set versioning is exercised; the
        committed iteration-0000 holds the initial diverse sample's stub
        PointDirectories.
        """
        v_train = self._versioning("training")
        v_models = self._versioning("models")
        v_train.recover_dangling_staging()
        v_models.recover_dangling_staging()

        #commit training iteration 0 if it does not exist yet.
        if 0 not in v_train.list_committed_versions():
            staging = v_train.stage(source_version=None, target_version=0)
            (staging / "initial_marker.txt").write_text(
                "DRYRUN initial training iteration\n", encoding="utf-8",
            )
            v_train.commit(0)
            v_train.update_current(0)
        else:
            v_train.ensure_current(0)

        #commit models iteration 0.
        if 0 not in v_models.list_committed_versions():
            staging = v_models.stage(source_version=None, target_version=0)
            (staging / "model.iqa").write_text(
                "DRYRUN initial FEREBUS model\n", encoding="utf-8",
            )
            v_models.commit(0)
            v_models.update_current(0)
        else:
            v_models.ensure_current(0)
        return {"training_set_version": 0, "models_version": 0}

    def _post_ariadne_array(self, state) -> Dict[str, Any]:
        """Use the mock ARIADNE runner to produce per-seed result.json files
        AND write a per-seed .provenance.json that the APPEND phase later
        copies into the committed training pointdir."""
        #ocal imports to avoid circular dependencies at module load time.
        from ..acquisition.ariadne_runner import (
            AriadneRunConfig,
            optimise_seed,
        )
        from ..geometry_novelty import geometry_novelty_scale_path
        from ..sampling_protocol import resolve_sampling_protocol
        from ichor.core.atoms import Atom, Atoms as IchorAtoms
        from ..handoff_manifests import (
            ARIADNE_RESULTS_SCHEMA_VERSION,
            acquisition_maturity_audit_payload,
            load_seeds_picked,
            write_acquisition_maturity_audit,
            write_ariadne_landing_audit,
            write_ariadne_results_manifest,
        )

        n_seeds = max(1, min(self.config.seed_selection.n_seeds_per_iteration, 4))
        iter_dir = self._iter_dir(state.iteration)
        pool_dir = iter_dir / "pool"
        pool_dir.mkdir(parents=True, exist_ok=True)
        synthetic_seed = IchorAtoms([
            Atom("O", 0.0, 0.0, 0.0),
            Atom("H", 0.96, 0.0, 0.0),
            Atom("H", -0.24, 0.93, 0.0),
        ])
        last_alpha = 0.0
        traj_sha = self._trajectory_sha256_if_available()
        campaign_uid = str(getattr(state, "campaign_uid", "") or "")
        picked_frame_ids = self._read_picked_seed_frame_ids(state.iteration)
        try:
            picked_payload = load_seeds_picked(iter_dir, expected_iteration=int(state.iteration))
            seed_records = list(picked_payload["seed_records"])
            if seed_records:
                n_seeds = min(n_seeds, len(seed_records))
        except Exception:
            picked_payload = {}
            seed_records = [
                {
                    "seed_index": int(k),
                    "frame_id": (
                        picked_frame_ids[k]
                        if k < len(picked_frame_ids) and isinstance(picked_frame_ids[k], int)
                        else None
                    ),
                    "selection_index": int(k),
                    "selection_origin": "bulk" if k % 2 == 0 else "variance",
                    "variance_at_selection": None,
                }
                for k in range(n_seeds)
            ]
        accepted_records = []
        landing_audit_records = []
        flagged_count = 0
        resolved_protocol = resolve_sampling_protocol(
            self.campaign_dir,
            self.config,
            iteration=int(state.iteration),
        )
        geometry_scale_payload = dict(resolved_protocol.geometry_scale_payload)
        self.artefact_log.append(str(geometry_novelty_scale_path(iter_dir)))
        if resolved_protocol.manifest_path is not None:
            self.artefact_log.append(str(resolved_protocol.manifest_path))
        if resolved_protocol.scale_model_path is not None:
            self.artefact_log.append(str(resolved_protocol.scale_model_path))
        if resolved_protocol.audit_manifest_path is not None:
            self.artefact_log.append(str(resolved_protocol.audit_manifest_path))
        for k in range(n_seeds):
            seed_record = seed_records[k]
            seed_frame_id = (
                seed_record.get("frame_id")
                if isinstance(seed_record.get("frame_id"), int)
                else None
            )
            seed_dir = pool_dir / ("seed_" + str(k).zfill(4))
            seed_dir.mkdir(exist_ok=True)
            result = optimise_seed(
                models=None,
                seed=synthetic_seed,
                trajectory=[synthetic_seed],
                run_config=replace(
                    resolved_protocol.ariadne_run_config,
                    rng_seed=self.rng_seed + k,
                ),
                mock=True,
            )
            result_payload = result.to_dict()
            result_payload["seed_frame_id"] = seed_frame_id
            result_payload["seed_index"] = int(k)
            result_payload["iteration"] = int(state.iteration)
            result_payload["trajectory_sha256"] = str(picked_payload.get("trajectory_sha256", traj_sha))
            result_payload["geometry_novelty_scale"] = dict(geometry_scale_payload)
            result_payload["sampling_protocol"] = {
                "sampling_aggressiveness": int(
                    resolved_protocol.sampling_aggressiveness
                ),
                "resolved_manifest": (
                    None if resolved_protocol.manifest_path is None
                    else str(resolved_protocol.manifest_path.resolve())
                ),
                "scale_model_manifest": (
                    None if resolved_protocol.scale_model_path is None
                    else str(resolved_protocol.scale_model_path.resolve())
                ),
                "audit_manifest": (
                    None if resolved_protocol.audit_manifest_path is None
                    else str(resolved_protocol.audit_manifest_path.resolve())
                ),
                "hidden_overrides_detected": list(
                    resolved_protocol.hidden_overrides_detected
                ),
            }
            result_payload["sampling_scale_model"] = dict(
                resolved_protocol.scale_model_payload
            )
            if isinstance(result_payload.get("selection_diagnostics"), dict):
                result_payload["selection_diagnostics"]["model_version"] = int(
                    getattr(state, "models_version", -1)
                )
                result_payload["selection_diagnostics"]["seed_index"] = int(k)
                result_payload["selection_diagnostics"]["seed_frame_id"] = seed_frame_id
                result_payload["selection_diagnostics"]["result_json"] = str(
                    (seed_dir / "result.json").resolve()
                )
                result_payload["selection_diagnostics"]["geometry_novelty_scale"] = dict(
                    geometry_scale_payload
                )
            landing_safety = result_payload.get("landing_safety") or {
                "accepted": True,
                "policy": "mock_legacy_safe",
                "reasons": [],
                "record_only_reasons": ["synthetic_mock_safety_metrics"],
                "metrics": {},
            }
            audit_record = {
                "seed_index": int(k),
                "seed_dir": str(seed_dir.resolve()),
                "result_json": str((seed_dir / "result.json").resolve()),
                "landing_safety": dict(landing_safety),
                "landing_candidates": list(result_payload.get("landing_candidates") or []),
                "geometry_novelty_scale": dict(geometry_scale_payload),
            }
            if isinstance(result_payload.get("selection_diagnostics"), dict):
                audit_record["selection_diagnostics"] = dict(
                    result_payload["selection_diagnostics"]
                )
            landing_audit_records.append(audit_record)
            atomic_write_json(seed_dir / "result.json", result_payload)
            #initial provenance sidecar -- seed_frame_id is None in dry-run
            #(synthetic seed; no pool involvement). Current wiring replaces None
            #with the real frame_id chosen by select_seeds().
            write_seed_provenance(
                seed_dir,
                campaign_uid=campaign_uid,
                iteration=int(state.iteration),
                trajectory_sha256=traj_sha,
                seed_frame_id=seed_frame_id,
                seed_selection_origin=str(seed_record.get("selection_origin", "unknown")),
                seed_variance_at_selection=seed_record.get("variance_at_selection"),
                subspace_neighbour_frame_ids=[],
                subspace_dimension=0,
                subspace_eigenvalues=[],
                mode_weighting_policy=self._mode_weighting_policy_or_default(),
            )
            enrich_with_ariadne(
                seed_dir,
                alpha_initial=float(getattr(result, "alpha_initial", 0.0) or 0.0),
                alpha_final=float(getattr(result, "alpha_final", 0.0) or 0.0),
                n_evaluations=int(getattr(result, "n_evaluations", 0) or 0),
                fell_back_to_ds=bool(getattr(result, "fell_back_to_ds", False)),
                wall_seconds=float(getattr(result, "wall_seconds", 0.0) or 0.0),
                return_code=int(getattr(result, "return_code", 0) or 0),
            )
            if isinstance(result_payload.get("selection_diagnostics"), dict):
                enrich_with_error_calibration_input(
                    seed_dir,
                    dict(result_payload["selection_diagnostics"]),
                )
            #synthesise a placeholder whitened distance from the
            # alpha change during ARIADNE descent; threshold against the
            # trust-region bounds. Live executor swaps in the real metric.
            d_w = self._synthetic_whitened_distance(result)
            flag = None
            if d_w is not None:
                min_d, max_d = anti_overlap_whitened_distance_bounds(self.config)
                if d_w < min_d:
                    flag = "moved_too_little"
                elif d_w > max_d:
                    flag = "moved_too_far"
            enrich_with_anti_overlap(
                seed_dir,
                min_whitened_distance_to_training=d_w,
                passed=(flag is None),
                flag=flag,
            )
            if flag is not None:
                flagged_count += 1
                _picked = picked_frame_ids[k] if k < len(picked_frame_ids) else None
                self._journal_event(
                    "anti_overlap_flagged",
                    iteration=int(state.iteration),
                    seed_index=int(k),
                    seed_frame_id=(_picked if isinstance(_picked, int) else -1),
                    whitened_distance=float(d_w if d_w is not None else 0.0),
                    flag=str(flag),
                )
            if result.alpha_final is not None:
                last_alpha = max(last_alpha, float(result.alpha_final))
            self.artefact_log.append(str(seed_dir / "result.json"))
            self.artefact_log.append(str(seed_dir / PROVENANCE_FILENAME))
            accepted_records.append({
                "seed_index": int(k),
                "seed_dir": str(seed_dir.resolve()),
                "result_json": str((seed_dir / "result.json").resolve()),
                "provenance_json": str((seed_dir / PROVENANCE_FILENAME).resolve()),
                "seed_frame_id": seed_frame_id,
                "selection_index": int(seed_record.get("selection_index", k)),
                "selection_origin": str(seed_record.get("selection_origin", "unknown")),
                "variance_at_selection": seed_record.get("variance_at_selection"),
                "alpha_initial": float(getattr(result, "alpha_initial", 0.0) or 0.0),
                "alpha_final": float(getattr(result, "alpha_final", 0.0) or 0.0),
                "whitened_distance_final": d_w,
                "landing_safety": dict(landing_safety),
                "landing_policy": str(landing_safety.get("policy", "unknown")),
                "geometry_novelty_scale": dict(geometry_scale_payload),
                "selection_diagnostics": (
                    dict(result_payload["selection_diagnostics"])
                    if isinstance(result_payload.get("selection_diagnostics"), dict)
                    else None
                ),
                "return_code": int(getattr(result, "return_code", 0) or 0),
            })
        policies = {}
        for rec in landing_audit_records:
            safety = rec.get("landing_safety", {})
            policy = str(safety.get("policy", "unknown"))
            policies[policy] = int(policies.get(policy, 0)) + 1
        audit_path = write_ariadne_landing_audit(iter_dir, {
            "iteration": int(state.iteration),
            "summary": {
                "accepted": int(len(landing_audit_records)),
                "salvaged": 0,
                "backtracked": 0,
                "rejected": 0,
                "rejection_reasons": {},
                "policies": policies,
            },
            "seeds": landing_audit_records,
        })
        self.artefact_log.append(str(audit_path))
        maturity_path = write_acquisition_maturity_audit(
            iter_dir,
            acquisition_maturity_audit_payload(
                iteration=int(state.iteration),
                seed_records=landing_audit_records,
            ),
        )
        self.artefact_log.append(str(maturity_path))
        manifest_path = write_ariadne_results_manifest(iter_dir, {
            "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
            "iteration": int(state.iteration),
            "trajectory_sha256": str(picked_payload.get("trajectory_sha256", traj_sha)),
            "expected_n": int(n_seeds),
            "n_accepted": int(len(accepted_records)),
            "n_rejected": 0,
            "accepted": accepted_records,
            "rejected": [],
        })
        self.artefact_log.append(str(manifest_path))
        self._journal_event(
            "subspace_built",
            iteration=int(state.iteration),
            n_seeds=int(n_seeds),
        )
        return {
            "last_acquisition_alpha0": float(last_alpha),
            # the per-iteration flag count is now a persisted state
            # field (last_n_anti_overlap_flagged) rather than a transient
            # return; STOP_CHECK rules can consume it as a convergence signal.
            "last_n_anti_overlap_flagged": int(flagged_count),
        }

    def _post_phase_b_polus(self, state) -> Dict[str, Any]:
        from ..geometry_novelty import geometry_novelty_scale_path
        from ..sampling_protocol import (
            phase_b_min_separation_from_resolved,
            resolve_sampling_protocol,
        )
        from ..handoff_manifests import (
            PHASE_B_SELECTION_SCHEMA_VERSION,
            read_ariadne_results_manifest,
            write_phase_b_selection_manifest,
        )

        iter_dir = self._iter_dir(state.iteration)
        out = iter_dir / "phase_b_SAMPLE.xyz"
        out.write_text(
            "# DRYRUN POLUS Phase-B sample (iteration " + str(state.iteration) + ")\n"
            + "# descriptor=hybrid_alf_rmsd\n",
            encoding="utf-8",
        )
        self.artefact_log.append(str(out))
        pool_dir = iter_dir / "pool"
        accepted = []
        try:
            ariadne_manifest = read_ariadne_results_manifest(
                iter_dir,
                expected_iteration=int(state.iteration),
            )
            accepted = list(ariadne_manifest.get("accepted", []))
            ariadne_manifest_path = str((iter_dir / "ARIADNE_RESULTS.json").resolve())
        except Exception:
            ariadne_manifest_path = ""
            if pool_dir.is_dir():
                for seed_dir in sorted(pool_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    result_path = seed_dir / "result.json"
                    prov_path = seed_dir / PROVENANCE_FILENAME
                    if result_path.is_file() and prov_path.is_file():
                        try:
                            seed_index = int(seed_dir.name.split("_")[-1])
                        except (ValueError, IndexError):
                            seed_index = len(accepted)
                        accepted.append({
                            "seed_index": seed_index,
                            "seed_dir": str(seed_dir.resolve()),
                            "result_json": str(result_path.resolve()),
                            "provenance_json": str(prov_path.resolve()),
                            "seed_frame_id": None,
                            "selection_index": seed_index,
                            "selection_origin": "unknown",
                            "variance_at_selection": None,
                            "alpha_final": None,
                        })
        final_records = []
        final_batch_size = int(self.config.active_batch.final_batch_size)
        n_accepted_candidates = len(accepted)
        if n_accepted_candidates < final_batch_size:
            raise BackendSubmissionError(
                "active_batch_underfilled: wanted final_batch_size="
                + str(final_batch_size)
                + " but only "
                + str(n_accepted_candidates)
                + " safe ARIADNE candidates are available"
            )
        accepted = accepted[:final_batch_size]
        if pool_dir.is_dir():
            for seed_dir in sorted(pool_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                if not (seed_dir / PROVENANCE_FILENAME).is_file():
                    continue
                try:
                    rank = int(seed_dir.name.split("_")[-1])
                except (ValueError, IndexError):
                    rank = None
                enrich_with_phase_b(
                    seed_dir,
                    selected_after_fps=True,
                    diversity_rank=rank,
                    descriptor_used="hybrid_alf_rmsd",
                )
        for final_index, rec in enumerate(accepted):
            out_rec = dict(rec)
            out_rec["raw_index"] = int(final_index)
            out_rec["final_index"] = int(final_index)
            out_rec["kept_after_dedup"] = True
            out_rec["drop_reason"] = None
            final_records.append(out_rec)
        resolved_protocol = resolve_sampling_protocol(
            self.campaign_dir,
            self.config,
            iteration=int(state.iteration),
        )
        geometry_scale_payload = dict(resolved_protocol.geometry_scale_payload)
        self.artefact_log.append(
            str(geometry_novelty_scale_path(iter_dir))
        )
        if resolved_protocol.manifest_path is not None:
            self.artefact_log.append(str(resolved_protocol.manifest_path))
        if resolved_protocol.scale_model_path is not None:
            self.artefact_log.append(str(resolved_protocol.scale_model_path))
        if resolved_protocol.audit_manifest_path is not None:
            self.artefact_log.append(str(resolved_protocol.audit_manifest_path))
        effective_min_separation, threshold_mode = phase_b_min_separation_from_resolved(
            resolved_protocol
        )
        dedup_payload = {
            "n_candidates": int(len(final_records)),
            "n_kept": int(len(final_records)),
            "n_dropped": 0,
            "kept_indices": [int(i) for i in range(len(final_records))],
            "dropped_indices": [],
            "distances_to_nearest": [],
            "min_separation": float(effective_min_separation),
            "threshold_mode": str(threshold_mode),
            "effective_min_separation_angstrom": float(effective_min_separation),
            "scaled_distances_to_nearest": [],
            "novelty_scores": [],
            "geometry_novelty_scale": geometry_scale_payload,
            "sampling_protocol": {
                "sampling_aggressiveness": int(
                    resolved_protocol.sampling_aggressiveness
                ),
                "resolved_manifest": (
                    None if resolved_protocol.manifest_path is None
                    else str(resolved_protocol.manifest_path.resolve())
                ),
                "scale_model_manifest": (
                    None if resolved_protocol.scale_model_path is None
                    else str(resolved_protocol.scale_model_path.resolve())
                ),
                "audit_manifest": (
                    None if resolved_protocol.audit_manifest_path is None
                    else str(resolved_protocol.audit_manifest_path.resolve())
                ),
                "hidden_overrides_detected": list(
                    resolved_protocol.hidden_overrides_detected
                ),
            },
            "sampling_scale_model": dict(resolved_protocol.scale_model_payload),
            "relaxation": {"applied": False, "reason": None},
        }
        manifest_path = write_phase_b_selection_manifest(iter_dir, {
            "schema_version": PHASE_B_SELECTION_SCHEMA_VERSION,
            "iteration": int(state.iteration),
            "descriptor": str(self.config.phase_b.descriptor),
            "source_ariadne_manifest": ariadne_manifest_path,
            "expected_final_batch_size": int(final_batch_size),
            "n_candidates": int(n_accepted_candidates),
            "n_selected_raw": int(len(final_records)),
            "n_kept": int(len(final_records)),
            "raw": list(final_records),
            "final": list(final_records),
            "dedup": dedup_payload,
        })
        self.artefact_log.append(str(manifest_path))
        return {}

    def _post_gaussian(self, state) -> Dict[str, Any]:
        return self._stub_quantum_outputs(state, stage="GAUSSIAN", initial=False)

    def _post_aimall(self, state) -> Dict[str, Any]:
        return self._stub_quantum_outputs(state, stage="AIMALL", initial=False)

    def _post_ferebus(self, state) -> Dict[str, Any]:
        """Commit the next models iteration. Training set itself has already
        been committed by the inline APPEND phase one step earlier."""
        v_models = self._versioning("models")
        v_models.recover_dangling_staging()
        committed = v_models.list_committed_versions()
        expected_next = int(getattr(state, "training_set_version", -1))
        if expected_next < 0:
            raise ValueError(
                "FEREBUS requires a committed training_set_version, got "
                + repr(getattr(state, "training_set_version", None))
            )
        if expected_next in committed:
            repaired_version = int(expected_next)
            v_models.ensure_current(repaired_version)
            self._journal_event(
                "models_committed",
                phase="FEREBUS",
                iteration=int(state.iteration),
                models_version=int(repaired_version),
                idempotent_skip=True,
            )
            return {"models_version": int(repaired_version)}
        next_version = int(expected_next)
        staging = v_models.stage(source_version=max(committed) if committed else None,
                                  target_version=next_version)
        (staging / "model.iqa").write_text(
            "DRYRUN FEREBUS model for iteration " + str(state.iteration) + "\n",
            encoding="utf-8",
        )
        v_models.commit(next_version)
        v_models.update_current(next_version)
        return {"models_version": int(next_version)}

    # --- helpers --------------------------------------------------

    def _recent_seed_cooldown(self) -> int:
        return int(getattr(self.config.anti_overlap, "recent_seeds_cooldown", DEFAULT_RECENT_SEEDS_COOLDOWN))

    def _trajectory_sha256_if_available(self) -> str:
        """Return the SHA-256 of the imported trajectory pool manifest if
        present; empty string otherwise. The daemon does not require a pool
        to be imported in dry-run, so this is a soft lookup."""
        from ..acquisition.trajectory_pool import POOL_MANIFEST_FILENAME, POOL_SUBDIR
        import json as _json

        manifest_path = self.campaign_dir / POOL_SUBDIR / POOL_MANIFEST_FILENAME
        if not manifest_path.is_file():
            return ""
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            return ""
        sha = data.get("sha256", "")
        return str(sha) if isinstance(sha, str) else ""

    def _mode_weighting_policy_or_default(self) -> str:
        """Return the campaign-config mode-weighting policy if exposed; 
        otherwise the documented default 'variance'."""
        cfg_acq = getattr(self.config, "acquisition", None)
        if cfg_acq is not None:
            sub = getattr(cfg_acq, "subspace", None)
            if sub is not None and getattr(sub, "mode_weighting_policy", None):
                return str(sub.mode_weighting_policy)
        return "variance"

    def _read_picked_seed_frame_ids(self, iteration):
        """Return the list of seed_frame_ids written by SEED_SELECT.

        Loads seeds_picked.json from the iteration dir and returns its
        frame_ids field. Returns an empty list on missing file or any
        decode error."""
        path = self._iter_dir(iteration) / "seeds_picked.json"
        if not path.is_file():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
        fids = data.get("frame_ids", [])
        if not isinstance(fids, list):
            return []
        return [int(f) if isinstance(f, int) else None for f in fids]

    def _synthetic_whitened_distance(self, result):
        """Placeholder for the post-ARIADNE whitened distance.

        Proxies the real whitened distance with the magnitude of the
        alpha change during ARIADNE descent. The live executor will
        replace this with the true frozen-subspace metric value."""
        a0 = getattr(result, "alpha_initial", None)
        a1 = getattr(result, "alpha_final", None)
        if a0 is None or a1 is None:
            return None
        try:
            return float(abs(float(a1) - float(a0)))
        except (TypeError, ValueError):
            return None

    def _read_seed_frame_id_from_pointdir(self, pointdir: Path):
        """Read .provenance.json from a committed pointdir and return its
        seed.frame_id (or None if absent / malformed)."""
        import json as _json
        from ..versioning.provenance import PROVENANCE_FILENAME as _PFN

        prov_path = Path(pointdir) / _PFN
        if not prov_path.is_file():
            return None
        try:
            with open(prov_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            return None
        seed_block = data.get("seed")
        if not isinstance(seed_block, dict):
            return None
        fid = seed_block.get("frame_id")
        return int(fid) if isinstance(fid, int) else None

    def _journal_event(self, event_type: str, **payload) -> None:
        """Robust append to the daemon journal. The journal path is
        derived from the campaign_dir; failures are silently swallowed --
        provenance writes are the source of truth, journal is informational.
        """
        try:
            from .journal import append_event
        except Exception:
            return
        try:
            journal_path = self.campaign_dir / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
            append_event(journal_path, event_type, **payload)
        except Exception:
            pass


    def _maybe_refresh_reference_scales(self, state) -> bool:
        """thin wrapper so the live executor can override the policy
        without duplicating the dry-run synthetic-stub body.

        The dry-run path is the default; the live executor overrides this
        method to compute real GP-derived scales instead.
        """
        return self._maybe_refresh_reference_scales_dry(state)

    def _maybe_refresh_reference_scales_dry(self, state) -> bool:
        """Decide whether to recompute reference scales for this iteration
        based on acquisition.references.refresh_policy. Mutates state in place.

        Policies:
          every_iteration:      recompute on every call
          every_n_iterations:   recompute when iteration % refresh_period == 0
          never:                only on the very first call (iteration 0)

        In dry-run we cannot actually evaluate GP posteriors, so the scales
        we cache here are a stub payload. The wiring path is identical to
        what the live executor does with a real posterior.
        Returns True iff the cache was refreshed.
        """
        refs = self.config.acquisition.references
        policy = refs.refresh_policy
        prev_iter = int(getattr(state, "reference_scales_iteration", -1))
        prev_scales = getattr(state, "reference_scales", None)
        should_refresh = False
        if prev_scales is None:
            should_refresh = True
        elif policy == "every_iteration":
            should_refresh = True
        elif policy == "every_n_iterations":
            period = max(1, int(refs.refresh_period))
            should_refresh = (int(state.iteration) - prev_iter) >= period
        elif policy == "never":
            should_refresh = False
        if not should_refresh:
            return False
        floor = float(refs.floor)
        synthetic = {
            "energy": floor + 1.0e-3,
            "force": floor + 1.0e-2,
            "omega": floor + 1.0,
            "anh": floor + 1.0,
            "anh_std": floor + 1.0,
        }
        state.reference_scales = synthetic
        state.reference_scales_iteration = int(state.iteration)
        self._journal_event(
            "reference_scales_computed",
            iteration=int(state.iteration),
            policy=str(policy),
            n_keys=int(len(synthetic)),
        )
        return True

    def _stub_quantum_outputs(self, state, *, stage: str, initial: bool) -> Dict[str, Any]:
        """Create stub PointDirectories so APPEND has content to stage. We
        write into a transient staging area under .DATA/STAGING/quantum/ that
        the APPEND step copies into the next training iteration.
        """
        staging_root = (
            self.campaign_dir / ".DATA" / "STAGING"
            / ("initial" if initial else ("iter_" + str(state.iteration)))
        )
        staging_root.mkdir(parents=True, exist_ok=True)
        n_points = (
            int(self.config.bootstrap.initial_labelled_size)
            if bool(initial)
            else int(self.config.active_batch.final_batch_size)
        )
        for i in range(n_points):
            point_dir = staging_root / ("POINT_" + str(i).zfill(4) + ".pointdir")
            point_dir.mkdir(exist_ok=True)
            artefact_name = "stub_" + stage + ".txt"
            (point_dir / artefact_name).write_text(
                "DRYRUN " + stage + " output for point " + str(i) + "\n",
                encoding="utf-8",
            )
            self.artefact_log.append(str(point_dir / artefact_name))
        if stage == "AIMALL":
            from .quantum_quality import write_quantum_quality_manifest

            records = [
                {
                    "pointdir": "POINT_" + str(i).zfill(4) + ".pointdir",
                    "accepted": True,
                    "reasons": [],
                    "atom_count": 1,
                    "n_int": 1,
                    "sum_iqa_ha": -1.0,
                    "wfn_total_energy_ha": -1.0,
                    "iqa_energy_recovery_error_ha": 0.0,
                    "max_abs_integration_error": 0.0,
                    "per_atom": [
                        {
                            "atom": "X1",
                            "iqa_ha": -1.0,
                            "integration_error": 0.0,
                            "reasons": [],
                        }
                    ],
                }
                for i in range(n_points)
            ]
            manifest = write_quantum_quality_manifest(
                staging_root,
                phase_name="INITIAL_AIMALL" if initial else "AIMALL",
                iteration=int(state.iteration),
                records=records,
                gates=getattr(self.config, "quality_gates", None),
            )
            self.artefact_log.append(str(manifest))
            if not initial and bool(getattr(self.config.error_calibration, "enabled", True)):
                from .error_calibration import (
                    append_records,
                    build_calibration_model,
                    mark_calibration_model_stale,
                    synthetic_dry_records,
                    write_calibration_model,
                    write_iteration_audit,
                )

                try:
                    synthetic = synthetic_dry_records(
                        iteration=int(state.iteration),
                        models_version=int(getattr(state, "models_version", -1)),
                        n_points=n_points,
                    )
                    all_records, added, duplicate = append_records(
                        self.campaign_dir,
                        synthetic,
                    )
                    model = build_calibration_model(
                        all_records,
                        self.config,
                        iteration=int(state.iteration),
                        current_model_version=int(getattr(state, "models_version", -1)),
                    )
                    model_path = write_calibration_model(self.campaign_dir, model)
                    iter_dir = self._iter_dir(state.iteration)
                    audit_path = write_iteration_audit(
                        iter_dir,
                        {
                            "iteration": int(state.iteration),
                            "enabled": True,
                            "mode": str(self.config.error_calibration.mode),
                            "n_new_records": int(len(synthetic)),
                            "n_added_records": int(added),
                            "n_duplicate_records": int(duplicate),
                            "n_total_records": int(len(all_records)),
                            "n_usable_records": int(model.get("n_records", 0)),
                            "n_usable_total_records": int(model.get("n_total_error_records", 0)),
                            "usable_for_acquisition": bool(
                                model.get("usable_for_acquisition", False)
                            ),
                            "model": str(model_path.resolve()),
                            "skipped": {},
                        },
                    )
                    self.artefact_log.append(str(audit_path))
                    self.artefact_log.append(str(model_path))
                    self._journal_event(
                        "error_calibration_summary",
                        phase="AIMALL",
                        iteration=int(state.iteration),
                        n_added_records=int(added),
                        n_total_records=int(len(all_records)),
                        usable_for_acquisition=bool(
                            model.get("usable_for_acquisition", False)
                        ),
                    )
                except Exception as exc:
                    try:
                        mark_calibration_model_stale(
                            self.campaign_dir,
                            reason=type(exc).__name__ + ": " + str(exc)[:240],
                            iteration=int(state.iteration),
                        )
                    except Exception:
                        pass
                    self._journal_event(
                        "error_calibration_failed",
                        phase="AIMALL",
                        iteration=int(state.iteration),
                        reason=type(exc).__name__ + ": " + str(exc)[:240],
                    )
        return {}
