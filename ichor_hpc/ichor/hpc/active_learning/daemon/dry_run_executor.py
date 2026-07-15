"""Dry-run PhaseExecutor for end-to-end smoke testing of the daemon.

    > Run "ichor-al-daemon start --mode dry_run" in a dedicated campaign.
    > against a small fixture (water tetramer, 100 frames). Verify all
    > directories, manifests, journal entries, sbatch invocations, sacct polls,
    > and atomic renames work end-to-end without real Gaussian / AIMAll /
    > FEREBUS.

What the dry-run executor DOES exercise (real, no mocking):

    * Directory layout setup (QM_REFERENCE_DATA/, TRAINED_MODELS/, ACTIVE_LEARNING/,
      .DATA/SCRIPTS/).
    * VersionedDirectory (stage, commit, update_current, manifest writes,
      atomic renames).
    * Immutable ``.DATA/SCRIPTS/JOBS/.../job.sh`` attempt bundles per
      Slurm-backed phase.
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

import csv
from ..strict_json import strict_json as json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import CampaignConfig
from ..versioning.provenance import (
    PROVENANCE_FILENAME,
    append_recent_seeds,
    upsert_index_records,
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
from ..versioning.manifest import sha256_file
from ..versioning.reference_data import ReferenceDataVersioning
from ..versioning.trained_models import TrainedModelVersioning
from ..versioning.versioned_directory import VersionedDirectory
from ..layout import (
    QM_REFERENCE_DATA_DIRNAME,
    TRAINED_MODELS_DIRNAME,
    active_iteration_dir,
    active_learning_dir,
    active_seed_selection_dir,
    ariadne_seed_dir,
    ariadne_seeds_dir,
    bootstrap_dir,
    bootstrap_selection_dir,
    staging_pointdir_name,
)
from .phase_executor import (
    BackendSubmissionError,
    FailureAction,
    INLINE_PHASES,
    PhaseResult,
    SBATCH_PHASES,
)
from .state import CampaignPhase, atomic_write_json, atomic_write_text
from .script_bundles import prepare_attempt_bundle, write_attempt_script


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


def _frames_to_xyz(frames: Sequence[Any], comments: Sequence[str]) -> str:
    if len(frames) != len(comments):
        raise ValueError("XYZ frame/comment count mismatch")
    lines: List[str] = []
    for frame, comment in zip(frames, comments):
        lines.extend((str(len(frame)), str(comment)))
        for atom in frame:
            lines.append(
                "{atom} {x:.12f} {y:.12f} {z:.12f}".format(
                    atom=str(atom.type),
                    x=float(atom.x),
                    y=float(atom.y),
                    z=float(atom.z),
                )
            )
    return "\n".join(lines) + "\n"


@dataclass
class DryRunPhaseExecutor:
    """PhaseExecutor that drives the full file-system flow with stub artefacts."""

    campaign_dir: Path
    config: CampaignConfig
    rng_seed: int = 0
    reference_data_dir_name: str = QM_REFERENCE_DATA_DIRNAME
    models_dir_name: str = TRAINED_MODELS_DIRNAME
    strict_completion_receipt_evidence: bool = True
    scheduler_identity_kind: str = "synthetic"
    scripts_dir: Path = field(init=False)
    artefact_log: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        from ..layout import reject_legacy_campaign_layout
        from .filesystem import operational_data_dir

        self.campaign_dir = Path(self.campaign_dir)
        reject_legacy_campaign_layout(self.campaign_dir)
        operational_data_dir(self.campaign_dir).mkdir(parents=True, exist_ok=True)
        self.scripts_dir = self.campaign_dir / ".DATA" / "SCRIPTS"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        (self.campaign_dir / self.reference_data_dir_name).mkdir(parents=True, exist_ok=True)
        (self.campaign_dir / self.models_dir_name).mkdir(parents=True, exist_ok=True)
        bootstrap_dir(self.campaign_dir).mkdir(parents=True, exist_ok=True)
        active_learning_dir(self.campaign_dir).mkdir(parents=True, exist_ok=True)

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
        inline_result = self._run_inline(state, phase_name)
        if isinstance(inline_result, PhaseResult):
            return inline_result
        return PhaseResult(is_complete=True, state_updates=inline_result)

    def postprocess(self, state, phase, observations: Sequence[Any]) -> PhaseResult:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        postprocess_result = self._postprocess_phase(state, phase_name)
        if isinstance(postprocess_result, PhaseResult):
            return postprocess_result
        return PhaseResult(is_complete=True, state_updates=postprocess_result)

    def handle_failure(self, state, phase, observations) -> FailureAction:
        # in dry-run we never produce failed jobs, but provide a sensible default
        # in case the caller wires us up with a hostile sacct stub.
        return FailureAction.SCRUB_AND_CONTINUE

    # --- internal: script stubbing -------------------------------------

    def _write_stub_script(self, phase_name: str, iteration: int) -> Path:
        from .submission_intent import load_active_intent

        intent = load_active_intent(self.campaign_dir, phase_name, int(iteration))
        identity = (
            str(intent.get("submission_identity"))
            if isinstance(intent, dict) and intent.get("submission_identity")
            else "dry-run"
        )
        bundle = prepare_attempt_bundle(
            self.campaign_dir,
            phase_name,
            int(iteration),
            identity,
            array_size=None,
            max_log_files_per_directory=5000,
        )
        lines = [
            "#!/bin/sh",
            "# DRY-RUN stub for phase " + phase_name + ", iteration " + str(iteration),
            "# Generated by DryRunPhaseExecutor; no real backend invoked.",
            'echo "DRYRUN ' + phase_name + ' iteration=' + str(iteration) + '"',
            "exit 0",
            "",
        ]
        path = write_attempt_script(bundle, "\n".join(lines))
        self.artefact_log.append(str(path))
        return path

    def _versioning(self, kind: str) -> VersionedDirectory:
        if kind == "reference_data":
            return ReferenceDataVersioning(
                self.campaign_dir / self.reference_data_dir_name
            )
        if kind == "models":
            return TrainedModelVersioning(self.campaign_dir / self.models_dir_name)
        raise ValueError("unknown versioning kind: " + kind)

    def _models_staging_path(self) -> Path:
        return self.campaign_dir / self.models_dir_name / "iteration-staging"

    @staticmethod
    def _write_dry_ferebus_model(
        path: Path,
        *,
        system: str,
        atom: str,
        prop: str,
        alf_1_indexed: Sequence[int],
        ntrain: int,
        prior_mean_ha: float,
        kernel_family: str = "periodic_rbf",
        feature_rows: Optional[Sequence[Sequence[float]]] = None,
        target_rows: Optional[Sequence[float]] = None,
    ) -> None:
        """Write a small, fully parseable FEREBUS model for dry-run commits."""
        if int(ntrain) <= 0:
            raise BackendSubmissionError(
                "dry-run FEREBUS model requires at least one training row"
            )
        if feature_rows is None:
            nfeatures = 3
            resolved_feature_rows = [
                [0.1 + row * 0.1 + column * 0.01 for column in range(nfeatures)]
                for row in range(int(ntrain))
            ]
        else:
            resolved_feature_rows = [
                [float(value) for value in row] for row in feature_rows
            ]
            if len(resolved_feature_rows) != int(ntrain):
                raise BackendSubmissionError(
                    "dry-run FEREBUS feature-row count does not match ntrain"
                )
            nfeatures = len(resolved_feature_rows[0])
            if nfeatures <= 0 or any(
                len(row) != nfeatures for row in resolved_feature_rows
            ):
                raise BackendSubmissionError("dry-run FEREBUS feature rows are ragged")
        resolved_targets = (
            [float(-1.0 - row * 0.01) for row in range(int(ntrain))]
            if target_rows is None
            else [float(value) for value in target_rows]
        )
        if len(resolved_targets) != int(ntrain):
            raise BackendSubmissionError(
                "dry-run FEREBUS target-row count does not match ntrain"
            )
        lines = [
            "# jitter 1.0e-6",
            "# likelihood -1.0",
            "",
            "[system]",
            "name " + str(system),
            "atom " + str(atom),
            "property " + str(prop),
            "ALF " + " ".join(str(int(value)) for value in alf_1_indexed),
            "",
            "[dimensions]",
            "number_of_atoms 3",
            "number_of_features " + str(nfeatures),
            "number_of_training_points " + str(int(ntrain)),
            "",
            "[mean]",
            "type constant",
            "value " + repr(float(prior_mean_ha)),
            "",
            "[kernels]",
        ]
        if kernel_family == "rbf":
            lines.extend(
                [
                    "number_of_kernels 1",
                    "composition k1",
                    "prefactor 1.0",
                    "",
                    "[kernel.k1]",
                    "type rbf",
                    "number_of_dimensions " + str(nfeatures),
                    "active_dimensions "
                    + " ".join(str(index) for index in range(1, nfeatures + 1)),
                    "thetas " + " ".join("1.0" for _ in range(nfeatures)),
                    "",
                ]
            )
        elif kernel_family == "periodic_rbf":
            periodic = [
                index for index in range(4, nfeatures + 1) if index % 3 == 0
            ]
            cyclic = [
                index for index in range(1, nfeatures + 1) if index not in periodic
            ]
            lines.extend(
                [
                    "number_of_kernels 2",
                    "composition (k1*k2)",
                    "prefactor 1.0",
                    "",
                    "[kernel.k1]",
                    "type rbf-cyclic",
                    "number_of_dimensions " + str(len(cyclic)),
                    "active_dimensions " + " ".join(map(str, cyclic)),
                    "thetas " + " ".join("1.0" for _ in cyclic),
                    "",
                    "[kernel.k2]",
                    "type periodic",
                    "number_of_dimensions " + str(len(periodic)),
                    "active_dimensions " + " ".join(map(str, periodic)),
                    "thetas " + " ".join("1.0" for _ in periodic),
                    "",
                ]
            )
        else:
            raise BackendSubmissionError(
                "unsupported dry-run FEREBUS kernel family " + repr(kernel_family)
            )
        lines.extend(
            [
                "[training_data]",
                "units.x " + " ".join("unknown" for _ in range(nfeatures)),
                "units.y " + ("Ha" if str(prop) == "iqa" else "unknown"),
                "",
                "[training_data.x]",
            ]
        )
        lines.extend(" ".join(str(value) for value in row) for row in resolved_feature_rows)
        lines.extend(["", "[training_data.y]"])
        lines.extend(str(value) for value in resolved_targets)
        lines.extend(["", "[weights]"])
        lines.extend("0.0" for _ in range(int(ntrain)))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    @staticmethod
    def _write_dry_ferebus_csv(
        path: Path,
        *,
        prop: str,
        nrows: int,
    ) -> None:
        lines = ["f1,f2,f3," + str(prop)]
        for row in range(int(nrows)):
            lines.append(
                ",".join(
                    [
                        str(0.1 + row * 0.1),
                        str(0.2 + row * 0.1),
                        str(0.3 + row * 0.1),
                        str(-1.0 - row * 0.01),
                    ]
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    def _prepare_dry_staged_models(self, staging: Path) -> Dict[str, Any]:
        """Create deterministic dry models from the exact staged CSV rows."""
        import shutil

        from . import input_staging as _stg
        from ..ferebus_prior import (
            contract_from_payload,
            validate_ferebus_config_contract,
        )
        from ..versioning.manifest import sha256_file

        manifest = _stg.read_ferebus_manifest(staging)
        prior_contract = contract_from_payload(manifest.get("prior_mean_contract"))
        for task in manifest["tasks"]:
            prop = str(task["property"])
            atom = str(task["atom"])
            for field in (
                "training_csv",
                "int_validation_csv",
                "ext_validation_csv",
            ):
                destination = _stg.resolve_ferebus_task_path(staging, task[field], field)
                source = staging / prop / destination.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            training_csv = _stg.resolve_ferebus_task_path(
                staging,
                task["training_csv"],
                "training_csv",
            )
            with training_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            header = [str(value).strip() for value in rows[0]]
            feature_indexes = [
                index
                for index, value in enumerate(header)
                if value.startswith("f") and value[1:].isdigit()
            ]
            if prop not in header or not feature_indexes:
                raise BackendSubmissionError(
                    "dry-run staged FEREBUS CSV contract is invalid"
                )
            target_index = header.index(prop)
            features = [
                [float(row[index]) for index in feature_indexes]
                for row in rows[1:]
                if row and any(str(value).strip() for value in row)
            ]
            targets = [
                float(row[target_index])
                for row in rows[1:]
                if row and any(str(value).strip() for value in row)
            ]
            config_path = _stg.resolve_ferebus_task_path(
                staging,
                task["config_path"],
                "config_path",
            )
            config_path.write_text(
                "# Dry-run deterministic FEREBUS output.\n"
                + "mean_type = " + str(prior_contract.mean_type) + "\n"
                + 'level_of_theory = "'
                + str(prior_contract.level_of_theory or "not_applicable")
                + '"\n'
                + "iqaDeviationFactor = "
                + repr(prior_contract.iqa_deviation_factor)
                + "\n"
                + "scaling = "
                + ("1" if prior_contract.feature_scaling else "0")
                + "\n"
                + "scale_feats = "
                + ("1" if prior_contract.feature_scaling else "0")
                + "\nscale_prop = 0\n",
                encoding="utf-8",
                newline="\n",
            )
            parsed_contract = validate_ferebus_config_contract(
                config_path,
                prior_contract,
            )
            task["generated_config"] = {
                "path": str(task["config_path"]),
                "size": int(config_path.stat().st_size),
                "sha256": sha256_file(config_path),
                "parsed_contract": parsed_contract,
                "prior_mean_contract_sha256": prior_contract.contract_sha256,
            }
            model_path = _stg.resolve_ferebus_task_path(
                staging,
                task["expected_model_path"],
                "expected_model_path",
            )
            self._write_dry_ferebus_model(
                model_path,
                system=str(manifest["system"]),
                atom=atom,
                prop=prop,
                alf_1_indexed=task["alf_1_indexed"],
                ntrain=len(features),
                prior_mean_ha=float(task["prior_mean"]["expected_mean_ha"]),
                feature_rows=features,
                target_rows=targets,
                kernel_family=str(manifest["kernel_contract"]["family"]),
            )
        _stg._write_ferebus_manifest(staging, manifest)
        return _stg.read_ferebus_manifest(staging, verify_dataset_files=True)

    @staticmethod
    def _write_dry_ferebus_execution_evidence(staging: Path) -> None:
        """Publish deterministic task evidence through the live FEREBUS schema."""
        from .ferebus_task_runner import write_preexisting_model_receipts
        from ..strict_json import strict_json as json
        from ..submit.pyferebus_wrap import _write_structured_task_map

        root = Path(staging)
        manifest = _write_structured_task_map(
            root,
            executable="ferebus",
            execution_kind="synthetic_dry_run",
        )
        task_map = json.loads(manifest.read_text(encoding="utf-8"))
        for task in task_map["tasks"]:
            performance_path = root.joinpath(
                *str(task["expected_performance_path"]).split("/")
            )
            performance_path.write_text(
                "RMSE 0.0\n"
                "MAE 0.0\n"
                "covariance_condition_number 1.0\n",
                encoding="utf-8",
                newline="\n",
            )
        write_preexisting_model_receipts(
            root,
            execution_kind="synthetic_dry_run",
        )

    def _commit_prepared_dry_ferebus_snapshot(self, version: int) -> None:
        """Stage, measure, decide and commit one dry model with live contracts."""
        from . import input_staging as _stg
        from .config_lock import canonical_config, config_fingerprint
        from .ferebus_quality import (
            evaluate_ferebus_quality,
            write_ferebus_quality_decision,
            write_ferebus_quality_manifest,
        )
        from .live_executor import _write_ferebus_task_artefact_layout
        from .model_contract import validate_ferebus_model_contract
        from ..versioning.trained_models import (
            seal_trained_model_version,
            trained_models_commit_lock,
            validate_trained_model_snapshot,
        )

        target_version = int(version)
        staging, _ = _stg.stage_ferebus_inputs(
            self.campaign_dir,
            self.config,
            target_version,
            is_initial=target_version == 0,
        )
        manifest = self._prepare_dry_staged_models(staging)
        self._write_dry_ferebus_execution_evidence(staging)
        quality = evaluate_ferebus_quality(staging)
        write_ferebus_quality_manifest(staging, quality)
        config_sha = config_fingerprint(canonical_config(self.config))
        write_ferebus_quality_decision(
            staging,
            config_sha256=config_sha,
            gates=getattr(self.config, "quality_gates", None),
        )
        validate_ferebus_model_contract(staging)

        model_versioning = self._versioning("models")
        with trained_models_commit_lock(self.campaign_dir):
            committed = model_versioning.list_committed_versions()
            if target_version in committed:
                model_versioning.resolve(target_version, verification="deep")
                model_versioning.ensure_current(max(committed))
                return
            if committed != list(range(target_version)):
                raise BackendSubmissionError(
                    "dry-run trained-model versions are not contiguous before "
                    + str(target_version)
                )
            parent = (
                None
                if target_version == 0
                else model_versioning.resolve(target_version - 1, verification="deep")
            )
            staged = model_versioning.stage(
                source_version=None,
                target_version=target_version,
            )
            _write_ferebus_task_artefact_layout(
                staging,
                staged,
                manifest,
                models_version=target_version,
                parent_model_set=parent,
            )
            staged_set = validate_trained_model_snapshot(
                self.campaign_dir,
                staged,
                target_version,
                parent=parent,
                verification="deep",
            )
            validate_ferebus_model_contract(
                staged,
                committed=True,
                expected_version=target_version,
                trained_model_set=staged_set,
            )
            model_versioning.commit(target_version)
            committed_dir = model_versioning.iteration_path(target_version)
            seal_trained_model_version(committed_dir)
            model_versioning.resolve(target_version, verification="deep")
            model_versioning.update_current(target_version)

    def _commit_dry_model_snapshot(self, version: int) -> None:
        """Commit a dry model snapshot through the live storage contract."""
        self._commit_prepared_dry_ferebus_snapshot(int(version))

    # --- internal: inline phases ---------------------------------------

    def _run_inline(self, state, phase_name: str) -> Dict[str, Any]:
        handler = self._inline_handlers().get(phase_name)
        if handler is None:
            # INIT, DONE, HALTED, STOP_CHECK fall through with no work.
            return {}
        return handler(state)

    def _inline_handlers(self):
        return {
            "INITIAL_ALLOCATION_CHECK": self._inline_initial_allocation_check,
            "ALLOCATION_CHECK": self._inline_allocation_check,
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

            def _transform(
                _selection_index: int,
                posterior_variance: float,
                population_variance_scale: float,
            ):
                return lookup_calibrated_abs_error(
                    model,
                    posterior_variance,
                    application_uncertainty_scale=population_variance_scale,
                )

            return _transform, "loaded"
        except Exception as exc:
            return None, "error:" + type(exc).__name__

    def _ensure_dry_run_trajectory_pool(self):
        """Create a deterministic canonical pool when a dry campaign has none."""
        from ..acquisition.trajectory_pool import TrajectoryPool

        try:
            return TrajectoryPool.load(self.campaign_dir)
        except FileNotFoundError:
            pass

        bootstrap_total = int(self.config.point_allocation.bootstrap_total_size)
        active_total = (
            int(self.config.max_iterations)
            * int(self.config.seed_selection.n_seeds_per_iteration)
        )
        n_frames = max(32, bootstrap_total + active_total)
        source = (
            self.campaign_dir
            / ".DATA"
            / "STAGING"
            / "dry_run_trajectory_pool.xyz"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for frame_id in range(n_frames):
            bond_shift = 0.001 * float(frame_id % 7)
            bend_shift = 0.002 * float(frame_id % 11)
            out_of_plane = 0.003 * float((frame_id % 3) - 1)
            lines.extend(
                [
                    "3",
                    "dry-run synthetic water frame " + str(frame_id),
                    "O 0.0 0.0 0.0",
                    "H " + str(0.9572 + bond_shift) + " 0.0 0.0",
                    "H "
                    + str(-0.2399872 + bend_shift)
                    + " "
                    + str(0.927297 + bond_shift)
                    + " "
                    + str(out_of_plane),
                ]
            )
        atomic_write_text(source, "\n".join(lines) + "\n")
        pool = TrajectoryPool.import_from(source, self.campaign_dir)
        self._journal_event(
            "dry_run_trajectory_pool_created",
            n_frames=int(pool.n_frames()),
            trajectory_sha256=str(pool.sha256),
        )
        return pool

    def _inline_seed_select(self, state) -> Dict[str, Any]:
        """Current wiring: SEED_SELECT invokes :func:'select_seeds' against the
        canonical trajectory pool (if imported) with 'forbidden_frame_ids' =
        (bootstrap/model matches) | (committed-training seeds) |
        (recent-seeds cooldown cache).

        Behaviour:
          * If no trajectory pool is present, imports a deterministic synthetic
            water pool through the canonical immutable-pool contract.
          * If a pool IS present, treats every frame as an eligible seed
            row, uses a uniform-variance stub posterior (the real
            TotalEnergyPosterior only kicks in once the live executor
            has real FEREBUS models to lean on), and
            writes the seed_selection/SELECTION.json artefact so
            ARIADNE post-processing can stamp the real seed_frame_id into
            the per-pointdir provenance.
        """
        from ..acquisition.seed_selection import select_seeds
        from ..acquisition.trajectory_pool import TrajectoryPool
        from ..handoff_manifests import (
            build_seed_selection_manifest,
            seeds_picked_path,
        )

        iter_dir = self._iter_dir(state.iteration)
        selection_dir = active_seed_selection_dir(iter_dir)
        selection_dir.mkdir(parents=True, exist_ok=True)
        seeds_path = selection_dir / "seeds.xyz"
        selection_path = seeds_picked_path(iter_dir)
        # re-entry guard: if this iteration's seeds were already picked (a
        # crash retry after SELECTION.json was written) do not pick again --
        # re-selecting would choose different frames and burn cooldown slots on
        # the abandoned ones. selection stays idempotent on re-run.
        if selection_path.is_file():
            from ..handoff_manifests import load_seeds_picked
            from ..seed_identity import (
                read_ariadne_task_map,
                write_ariadne_task_map,
            )

            picked = load_seeds_picked(iter_dir, expected_iteration=int(state.iteration))
            if str(picked["campaign_uid"]) != str(state.campaign_uid):
                raise BackendSubmissionError(
                    "existing seed selection campaign UID does not match state"
                )
            if int(picked["models_version"]) != int(state.models_version):
                raise BackendSubmissionError(
                    "existing seed selection model version does not match state"
                )
            pool = self._ensure_dry_run_trajectory_pool()
            if str(picked["trajectory_sha256"]) != str(pool.sha256):
                raise BackendSubmissionError(
                    "existing seed selection trajectory SHA does not match the pool"
                )
            pool_frames = pool.to_atoms_list()
            picked_records = list(picked["seed_records"])
            try:
                selected_frames = [
                    pool_frames[int(record["pool_row_index_zero_based"])]
                    for record in picked_records
                ]
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                raise BackendSubmissionError(
                    "existing seed selection contains an invalid pool row"
                ) from exc
            if seeds_path.is_symlink():
                raise BackendSubmissionError(
                    "refusing to replace symlinked seed-selection XYZ"
                )
            comments = [
                "active iteration "
                + str(int(state.iteration))
                + " seed "
                + str(int(record["seed_id"]))
                + " frame_id="
                + str(record.get("frame_id"))
                for record in picked_records
            ]
            atomic_write_text(seeds_path, _frames_to_xyz(selected_frames, comments))
            from ..handoff_manifests import ariadne_task_map_path

            task_map_path = ariadne_task_map_path(iter_dir)
            if task_map_path.is_file() or task_map_path.is_symlink():
                read_ariadne_task_map(
                    iter_dir,
                    expected_iteration=int(state.iteration),
                )
            else:
                write_ariadne_task_map(iter_dir, picked)
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
                    trajectory_sha256=str(picked["trajectory_sha256"]),
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

        pool = self._ensure_dry_run_trajectory_pool()

        from ..custom_bootstrap import read_custom_bootstrap_manifest

        bootstrap_payload = read_custom_bootstrap_manifest(self.campaign_dir)
        bootstrap_forbidden = {
            int(value)
            for value in bootstrap_payload.get("excluded_pool_frame_ids", [])
        }
        invalid_bootstrap_ids = sorted(
            value
            for value in bootstrap_forbidden
            if value < 0 or value >= int(pool.n_frames())
        )
        if invalid_bootstrap_ids:
            raise BackendSubmissionError(
                "custom bootstrap excluded pool-frame IDs are outside the "
                "current trajectory pool: " + repr(invalid_bootstrap_ids[:20])
            )

        # Exclude stable frame identities that have already produced committed
        # reference data before any stochastic or model-based seed scoring.
        if self.config.seed_selection.exclude_committed_seed_frames:
            training_forbidden = load_training_seed_frame_ids(
                self.campaign_dir,
                reference_data_dir=self.campaign_dir / self.reference_data_dir_name,
                expected_trajectory_sha256=pool.sha256,
            )
        else:
            training_forbidden = set()
        recent_forbidden = load_recent_seed_frame_ids(
            self.campaign_dir,
            expected_trajectory_sha256=pool.sha256,
        )
        forbidden = frozenset(
            bootstrap_forbidden | training_forbidden | recent_forbidden
        )

        training_atoms = pool.to_atoms_list()
        training_frame_ids = list(pool.frame_ids())

        posterior = self._seed_selection_posterior(state, training_atoms)
        if str(self.config.seed_selection.strategy) == "d_optimal":
            score_transform, score_transform_reason = self._seed_selection_score_transform()
        else:
            score_transform, score_transform_reason = None, "strategy_hybrid_variance"

        from ..randomness import derive_rng_seed

        seed_selection_rng = derive_rng_seed(
            campaign_uid=str(state.campaign_uid),
            campaign_random_seed=int(self.config.campaign.random_seed),
            iteration=int(state.iteration),
            phase="SEED_SELECT",
            logical_task_id="seed-batch",
            random_purpose="bulk-seed-selection",
        )
        selection = select_seeds(
            training_atoms,
            posterior,
            n_seeds=int(self.config.seed_selection.n_seeds_per_iteration),
            bulk_fraction=float(self.config.seed_selection.bulk_fraction),
            rng_seed=int(seed_selection_rng.derived_seed_128),
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
            d_optimal_degenerate_policy=str(
                self.config.seed_selection.d_optimal_degenerate_policy
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
                + ", bootstrap_forbidden="
                + str(len(bootstrap_forbidden))
                + ", recent_forbidden="
                + str(len(recent_forbidden))
                + ", forbidden_union="
                + str(len(forbidden))
                + ", exclude_committed_seed_frames="
                + str(bool(self.config.seed_selection.exclude_committed_seed_frames))
                + ", recent_seed_cooldown_iterations="
                + str(int(self.config.seed_selection.recent_seed_cooldown_iterations))
            )
        if selection.n < requested_n:
            raise BackendSubmissionError(
                "seed_pool_exhausted: only "
                + str(selection.n)
                + " eligible trajectory frames remain for requested batch "
                + str(requested_n)
                + "; training_forbidden="
                + str(len(training_forbidden))
                + ", bootstrap_forbidden="
                + str(len(bootstrap_forbidden))
                + ", recent_forbidden="
                + str(len(recent_forbidden))
                + ", forbidden_union="
                + str(len(forbidden))
                + ", exclude_committed_seed_frames="
                + str(bool(self.config.seed_selection.exclude_committed_seed_frames))
                + ", recent_seed_cooldown_iterations="
                + str(int(self.config.seed_selection.recent_seed_cooldown_iterations))
            )

        bulk_set = {int(i) for i in selection.bulk_indices}
        variance_set = {int(i) for i in selection.variance_indices}
        seed_records = []
        diagnostics_by_index = {
            int(row.get("selection_index")): dict(row)
            for row in selection.selection_diagnostics
            if isinstance(row, dict) and row.get("selection_index") is not None
        }

        for seed_position, frame_id in enumerate(selection.frame_ids):
            seed_id = int(seed_position) + 1
            selection_index = int(selection.indices[seed_position])
            if seed_position < len(selection.selection_origins):
                origin = str(selection.selection_origins[seed_position])
            elif selection_index in bulk_set:
                origin = "bulk"
            elif selection_index in variance_set:
                origin = "variance"
            else:
                origin = "unknown"
            variance_value = None
            if seed_position < len(selection.variances):
                variance_value = float(selection.variances[seed_position])
            record = {
                "seed_id": int(seed_id),
                "frame_id": frame_id if isinstance(frame_id, int) else None,
                "pool_row_index_zero_based": selection_index,
                "selection_origin": origin,
                "variance_at_selection": variance_value,
            }
            diag = diagnostics_by_index.get(selection_index, {})
            for key in (
                "raw_variance",
                "raw_score",
                "d_optimal_conditional_variance",
                "d_optimal_raw_conditional_variance",
                "d_optimal_gain",
                "d_optimal_prefilter_rank",
                "d_optimal_max_correlation_to_selected",
                "variance_rank",
                "d_optimal_backfill_reason",
            ):
                if key in diag:
                    record[key] = diag[key]
            seed_records.append(record)

        model_version = int(getattr(state, "models_version", -1))
        if model_version < 0:
            raise BackendSubmissionError(
                "seed selection requires a committed model version"
            )
        model_set = TrainedModelVersioning(
            self.campaign_dir / self.models_dir_name
        ).resolve(model_version, verification="deep")
        from .input_staging import read_ferebus_manifest
        from ..ferebus_prior import contract_from_payload

        prior_contract_hash = contract_from_payload(
            read_ferebus_manifest(
                model_set.root,
                verify_dataset_files=False,
            ).get("prior_mean_contract")
        ).contract_sha256
        seeds_picked_payload = build_seed_selection_manifest(
            campaign_uid=str(state.campaign_uid),
            campaign_random_seed=int(self.config.campaign.random_seed),
            iteration=int(state.iteration),
            models_version=int(model_version),
            model_manifest_sha256=str(model_set.head_manifest_sha256),
            trajectory_sha256=pool.sha256,
            selection_strategy=str(self.config.seed_selection.strategy),
            seed_records=seed_records,
            prior_mean_contract_sha256=prior_contract_hash,
        )
        seed_records = list(seeds_picked_payload["seed_records"])
        seeds_picked_payload.update({
            "forbidden_set_size": int(len(forbidden)),
            "bootstrap_forbidden_set_size": int(len(bootstrap_forbidden)),
            "skipped_unknown_provenance": int(selection.skipped_unknown_provenance),
            "score_source": (
                "error_calibration"
                if score_transform is not None
                else "posterior_variance"
            ),
            "score_source_reason": str(score_transform_reason),
        })
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
            "bootstrap_forbidden_set_size": int(len(bootstrap_forbidden)),
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
            "randomness": seed_selection_rng.to_dict(),
        }
        seeds_picked_payload["diagnostics"] = diagnostics_payload
        from ..seed_identity import write_ariadne_task_map
        selected_frames = [training_atoms[int(index)] for index in selection.indices]
        comments = [
            "active iteration "
            + str(int(state.iteration))
            + " seed "
            + str(seed_id)
            + " frame_id="
            + str(selection.frame_ids[seed_id - 1])
            for seed_id in range(1, int(selection.n) + 1)
        ]
        atomic_write_text(seeds_path, _frames_to_xyz(selected_frames, comments))
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(selection_path, seeds_picked_payload)
        task_map_path = write_ariadne_task_map(iter_dir, seeds_picked_payload)
        self.artefact_log.append(str(seeds_path))
        self.artefact_log.append(str(selection_path))
        self.artefact_log.append(str(task_map_path))

        append_recent_seeds(
            self.campaign_dir,
            iteration=int(state.iteration),
            frame_ids=selection.frame_ids,
            trajectory_sha256=pool.sha256,
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
            bootstrap_forbidden_set_size=int(len(bootstrap_forbidden)),
            skipped_unknown_provenance=int(selection.skipped_unknown_provenance),
        )
        return {}

    def _allocation_check(self, state, *, context: str) -> PhaseResult:
        from ..point_allocation import pending_attempts, point_allocation_path, read_point_allocation
        from ..replacement_sampling import prepare_replacement_round

        iteration = 0 if str(context) == "bootstrap" else int(state.iteration)
        path = point_allocation_path(
            self.campaign_dir,
            context=str(context),
            iteration=int(iteration),
        )
        allocation = read_point_allocation(path)
        summary = dict(allocation.get("summary") or {})
        if bool(allocation.get("mandatory_custom_failed", False)):
            raise BackendSubmissionError(
                "mandatory_custom_bootstrap_failed: supplied slots cannot be replaced"
            )
        if bool(summary.get("complete", False)):
            next_phase = "INITIAL_FEREBUS" if context == "bootstrap" else "APPEND"
            self._journal_event(
                "point_allocation_complete",
                context=str(context),
                iteration=int(iteration),
                accepted=dict(summary.get("accepted") or {}),
                accepted_total=int(summary.get("accepted_total", 0)),
                replacement_round=int(getattr(state, "replacement_round", 0)),
            )
            return PhaseResult(
                is_complete=True,
                state_updates={"replacement_round": 0},
                next_phase_override=next_phase,
            )
        pending = pending_attempts(allocation)
        if pending:
            rounds = {int(record.get("round", -1)) for record in pending}
            if len(rounds) != 1 or min(rounds) <= 0:
                raise BackendSubmissionError(
                    "point_allocation_pending_state_invalid: " + repr(sorted(rounds))
                )
            replacement_round = next(iter(rounds))
        else:
            replacement_round = max(
                [
                    int(attempt.get("round", 0))
                    for slot in allocation["slots"]
                    for attempt in slot.get("attempts", [])
                ]
                + [0]
            ) + 1
        try:
            replacement = prepare_replacement_round(
                self.campaign_dir,
                context=str(context),
                iteration=int(iteration),
                replacement_round=int(replacement_round),
            )
        except Exception as exc:
            raise BackendSubmissionError(
                "point_allocation_replacement_unavailable: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        next_phase = (
            "INITIAL_REPLACEMENT_GAUSSIAN"
            if context == "bootstrap"
            else "REPLACEMENT_GAUSSIAN"
        )
        self._journal_event(
            "point_allocation_replacement_prepared",
            context=str(context),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            n_candidates=int(replacement.get("n_candidates", 0)),
            reserve_available=int(summary.get("reserve_available", 0)),
        )
        return PhaseResult(
            is_complete=True,
            state_updates={"replacement_round": int(replacement_round)},
            next_phase_override=next_phase,
        )

    def _inline_initial_allocation_check(self, state) -> PhaseResult:
        return self._allocation_check(state, context="bootstrap")

    def _inline_allocation_check(self, state) -> PhaseResult:
        return self._allocation_check(state, context="active")

    def _inline_split(self, state) -> Dict[str, Any]:
        """Persist a read-only view of the pre-QM exact slot allocation."""
        from ..layout import active_allocation_dir
        from ..point_allocation import point_allocation_path, read_point_allocation
        from ..versioning.manifest import sha256_file

        iter_dir = self._iter_dir(state.iteration)
        split_path = active_allocation_dir(iter_dir) / "SPLIT_RECEIPT.json"
        allocation_path = point_allocation_path(
            self.campaign_dir,
            context="active",
            iteration=int(state.iteration),
        )
        allocation = read_point_allocation(allocation_path)
        slots = [
            {
                "slot_id": int(slot["slot_id"]),
                "split": str(slot["split"]),
                "candidate_id": str(slot["attempts"][0]["candidate_id"]),
            }
            for slot in allocation["slots"]
        ]
        payload = {
            "schema_version": 3,
            "strategy": "exact_pre_qm_point_allocation",
            "iteration": int(state.iteration),
            "point_allocation_manifest": allocation_path.resolve().relative_to(
                iter_dir.resolve()
            ).as_posix(),
            "slot_assignment_sha256": str(allocation["slot_assignment_sha256"]),
            "targets": dict(allocation["targets"]),
            "slots": slots,
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(split_path, payload)
        self.artefact_log.append(str(split_path))
        return {}

    def _inline_append(self, state) -> Dict[str, Any]:
        """Commit this iteration's accepted QM points as one immutable delta."""
        from . import input_staging as _stg

        v = self._versioning("reference_data")
        committed = v.list_committed_versions()
        iteration = int(state.iteration)
        if iteration < 1:
            raise BackendSubmissionError("APPEND requires active iteration >= 1")
        state_version = int(getattr(state, "reference_data_version", -1))
        committed_max = max(committed) if committed else -1
        if committed != list(range(committed_max + 1)):
            raise BackendSubmissionError(
                "reference_data_version_gap: committed_versions=" + repr(committed)
            )
        if committed_max not in (iteration - 1, iteration):
            raise BackendSubmissionError(
                "APPEND reference-data head must be active iteration - 1 or iteration: "
                + repr(committed)
            )
        if state_version not in (iteration - 1, iteration):
            raise BackendSubmissionError(
                "APPEND state/reference-data version mismatch: state="
                + str(state_version)
                + " iteration="
                + str(iteration)
            )
        target_version = iteration
        try:
            view, committed_pointdirs, created = _stg.commit_reference_data_delta(
                self.campaign_dir,
                reference_data_version=target_version,
                context="active",
                iteration=int(state.iteration),
            )
        except Exception as exc:
            raise BackendSubmissionError(
                "reference-data APPEND transaction failed: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        ensure_index(self.campaign_dir)
        committed_iter_dir = v.iteration_path(target_version)
        upsert_index_records(
            self.campaign_dir,
            records=[
                self._seed_index_record_from_pointdir(
                    committed_iter_dir / pdir_name,
                    iteration=int(target_version),
                    pointdir_name=pdir_name,
                )
                for pdir_name in committed_pointdirs
            ],
        )
        self._journal_event(
            "reference_data_committed",
            iteration=int(state.iteration),
            reference_data_version=int(target_version),
            n_committed_points=int(len(committed_pointdirs)) if created else 0,
            cumulative_point_count=int(len(view.entries)),
            head_manifest_sha256=str(view.head_manifest_sha256),
            cumulative_view_sha256=str(view.cumulative_view_sha256),
            expected_batch_total=int(self.config.point_allocation.batch_total_size),
            idempotent_skip=not bool(created),
        )
        return {"reference_data_version": int(target_version)}

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
        iteration = int(state.iteration)
        if iteration < 1:
            raise BackendSubmissionError("STOP_CHECK requires active iteration >= 1")
        if (
            int(state.reference_data_version) != iteration
            or int(state.models_version) != iteration
        ):
            raise BackendSubmissionError(
                "STOP_CHECK requires reference-data/model version equal to active iteration"
            )
        from ..versioning.sampling_iterations import finalise_active_iteration

        iteration_manifest = finalise_active_iteration(
            self.campaign_dir,
            iteration,
            str(state.campaign_uid),
        )
        self.artefact_log.append(str(iteration_manifest))
        self._journal_event(
            "active_iteration_finalised",
            iteration=iteration,
            manifest=str(iteration_manifest),
        )
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
        if int(state.iteration) >= int(stop.min_iterations_before_stop):
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
            updates["campaign_completion_reason"] = str(reason)
            self._journal_event(
                "scientific_convergence_reached",
                iteration=int(state.iteration),
                reason=str(reason),
                history_tail=[float(x) for x in history[-min(len(history), 8):]],
            )
        return updates

    def _iter_dir(self, iteration: int) -> Path:
        d = active_iteration_dir(self.campaign_dir, int(iteration))
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
            "PHASE_A_DIVERSITY": self._post_phase_a_diversity,
            "INITIAL_GAUSSIAN": self._post_initial_gaussian,
            "INITIAL_AIMALL": self._post_initial_aimall,
            "INITIAL_REPLACEMENT_GAUSSIAN": self._post_initial_replacement_gaussian,
            "INITIAL_REPLACEMENT_AIMALL": self._post_initial_replacement_aimall,
            "INITIAL_FEREBUS": self._post_initial_ferebus,
            "ARIADNE_ARRAY": self._post_ariadne_array,
            "PHASE_B_DIVERSITY": self._post_phase_b_diversity,
            "GAUSSIAN": self._post_gaussian,
            "AIMALL": self._post_aimall,
            "REPLACEMENT_GAUSSIAN": self._post_replacement_gaussian,
            "REPLACEMENT_AIMALL": self._post_replacement_aimall,
            "FEREBUS": self._post_ferebus,
        }

    # --- per-phase postprocess handlers --------------------------------

    def _post_phase_a_diversity(self, state) -> Dict[str, Any]:
        from ..sampling.diversity import _run_phase_a
        from ..custom_bootstrap import (
            commit_bootstrap_plan,
            custom_bootstrap_manifest_path,
            inspect_bootstrap_inputs,
        )

        pool = self._ensure_dry_run_trajectory_pool()
        if not custom_bootstrap_manifest_path(self.campaign_dir).is_file():
            commit_bootstrap_plan(
                inspect_bootstrap_inputs(
                    self.campaign_dir,
                    self.config,
                    pool.to_atoms_list(),
                    pool_sha256=str(pool.sha256),
                )
            )
        return_code = int(
            _run_phase_a(
                self.campaign_dir,
                self.config,
                campaign_uid=str(getattr(state, "campaign_uid", "")),
            )
        )
        if return_code != 0:
            raise BackendSubmissionError(
                "dry-run Phase A bootstrap planning failed with code "
                + str(return_code)
            )
        outdir = bootstrap_selection_dir(self.campaign_dir)
        self.artefact_log.extend([
            str(outdir / "selected.xyz"),
            str(outdir / "selected_indices.dat"),
            str(outdir / "PHASE_A_SAMPLE.json"),
        ])
        return {}

    def _post_initial_gaussian(self, state) -> Dict[str, Any]:
        return self._stub_quantum_outputs(state, stage="GAUSSIAN", initial=True)

    def _post_initial_aimall(self, state) -> Dict[str, Any]:
        return self._stub_quantum_outputs(state, stage="AIMALL", initial=True)

    def _post_initial_replacement_gaussian(self, state) -> Dict[str, Any]:
        return self._stub_quantum_outputs(
            state, stage="GAUSSIAN", initial=True, replacement=True,
        )

    def _post_initial_replacement_aimall(self, state) -> PhaseResult:
        self._stub_quantum_outputs(
            state, stage="AIMALL", initial=True, replacement=True,
        )
        return PhaseResult(
            is_complete=True,
            next_phase_override="INITIAL_ALLOCATION_CHECK",
        )

    def _post_initial_ferebus(self, state) -> Dict[str, Any]:
        """Commit initial reference data and a complete dry model snapshot.

        This is the first time the QM reference-data versioning is exercised; the
        committed iteration-000000 holds the initial diverse sample's stub
        PointDirectories.
        """
        from . import input_staging as _stg

        if _stg._model_bootstrap_context(self.campaign_dir) is not None:
            staging, _n_tasks = _stg.stage_ferebus_inputs(
                self.campaign_dir,
                self.config,
                0,
                is_initial=True,
            )
            _stg.prepare_imported_model_bootstrap(staging)
            from .live_executor import LiveBackendsPhaseExecutor

            result = LiveBackendsPhaseExecutor._parse_ferebus_postprocess(
                self,
                state,
                "INITIAL_FEREBUS",
                [],
            )
            if result.failure_reason is not None:
                raise BackendSubmissionError(result.failure_reason)
            return dict(result.state_updates or {})

        v_train = self._versioning("reference_data")
        v_models = self._versioning("models")
        v_train.recover_dangling_staging()
        v_models.recover_dangling_staging()

        from .input_staging import commit_initial_reference_data

        commit_initial_reference_data(self.campaign_dir)
        v_train.ensure_current(0)

        self._commit_dry_model_snapshot(0)
        from ..versioning.sampling_iterations import finalise_bootstrap

        bootstrap_manifest = finalise_bootstrap(
            self.campaign_dir,
            str(state.campaign_uid),
        )
        self.artefact_log.append(str(bootstrap_manifest))
        return {"reference_data_version": 0, "models_version": 0}

    def _post_ariadne_array(self, state) -> Dict[str, Any]:
        """Publish mock seed outputs through the live ARIADNE contracts."""
        from ..strict_json import strict_json as json
        import os
        import shutil

        from ..acquisition.ariadne_runner import optimise_seed
        from ..acquisition.trajectory_pool import TrajectoryPool
        from ..ariadne_outputs import (
            SEED_OUTPUT_MANIFEST_FILENAME,
            SEED_RESULT_FILENAME,
            validate_seed_output,
            write_optimisation_trajectory,
            write_seed_output_manifest,
        )
        from ..handoff_manifests import (
            ARIADNE_RESULTS_SCHEMA_VERSION,
            acquisition_maturity_audit_payload,
            load_seeds_picked,
            write_acquisition_maturity_audit,
            write_ariadne_landing_audit,
            write_ariadne_batch_decision,
            write_ariadne_results_manifest,
        )
        from ..layout import active_ariadne_dir, ariadne_seed_dir
        from ..sampling_protocol import resolve_or_load_sampling_protocol
        from ..seed_identity import read_ariadne_task_map
        from ..versioning.manifest import sha256_file

        iter_dir = self._iter_dir(state.iteration)
        ariadne_root = active_ariadne_dir(iter_dir)
        ariadne_root.mkdir(parents=True, exist_ok=True)
        picked_payload = load_seeds_picked(
            iter_dir,
            expected_iteration=int(state.iteration),
        )
        task_map = read_ariadne_task_map(
            iter_dir,
            expected_iteration=int(state.iteration),
        )
        seed_records = list(picked_payload["seed_records"])
        tasks = list(task_map["tasks"])
        if len(seed_records) != len(tasks):
            raise BackendSubmissionError("ARIADNE task map/selection count mismatch")
        pool = TrajectoryPool.load(self.campaign_dir)
        if str(pool.sha256) != str(picked_payload["trajectory_sha256"]):
            raise BackendSubmissionError("ARIADNE dry-run trajectory SHA mismatch")

        last_alpha = 0.0
        campaign_uid = str(getattr(state, "campaign_uid", "") or "")
        accepted_records = []
        landing_audit_records = []
        flagged_count = 0
        resolved_protocol = resolve_or_load_sampling_protocol(
            self.campaign_dir,
            self.config,
            iteration=int(state.iteration),
        )
        if resolved_protocol.manifest_path is not None:
            self.artefact_log.append(str(resolved_protocol.manifest_path))
        if resolved_protocol.scale_model_path is not None:
            self.artefact_log.append(str(resolved_protocol.scale_model_path))
        if resolved_protocol.audit_manifest_path is not None:
            self.artefact_log.append(str(resolved_protocol.audit_manifest_path))

        def protocol_binding(path):
            if path is None:
                raise BackendSubmissionError(
                    "resolved sampling protocol artefact is missing"
                )
            resolved = path.resolve()
            return (
                resolved.relative_to(self.campaign_dir.resolve()).as_posix(),
                sha256_file(resolved),
            )

        resolved_manifest, resolved_manifest_sha256 = protocol_binding(
            resolved_protocol.manifest_path
        )
        scale_model_manifest, scale_model_manifest_sha256 = protocol_binding(
            resolved_protocol.scale_model_path
        )
        audit_manifest, audit_manifest_sha256 = protocol_binding(
            resolved_protocol.audit_manifest_path
        )
        for array_task_id, (task, seed_record) in enumerate(zip(tasks, seed_records)):
            seed_id = int(task["seed_id"])
            seed_uid = str(task["seed_uid"])
            if int(task["array_task_id"]) != array_task_id:
                raise BackendSubmissionError("ARIADNE task IDs are not contiguous")
            seed_frame_id = int(task["frame_id"])
            seed_atoms = pool.frame(seed_frame_id)
            seed_dir = ariadne_seed_dir(iter_dir, seed_id)
            result_path = seed_dir / SEED_RESULT_FILENAME
            if seed_dir.exists() or seed_dir.is_symlink():
                existing_output = validate_seed_output(
                    seed_dir,
                    expected_campaign_uid=campaign_uid,
                    expected_iteration=int(state.iteration),
                    expected_seed_id=seed_id,
                    expected_seed_uid=seed_uid,
                    expected_array_task_id=array_task_id,
                )
                if not bool(existing_output["task_success"]) or int(
                    existing_output["task_exit_code"]
                ) != 0:
                    raise BackendSubmissionError(
                        "existing dry-run ARIADNE seed output records task failure"
                    )
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
                result = None
            else:
                from ..randomness import derive_rng_seed

                ariadne_rng = derive_rng_seed(
                    campaign_uid=campaign_uid,
                    campaign_random_seed=int(self.config.campaign.random_seed),
                    iteration=int(state.iteration),
                    phase="ARIADNE_ARRAY",
                    logical_task_id=seed_uid,
                    random_purpose="ariadne-seed-optimisation",
                )
                result = optimise_seed(
                    models=None,
                    seed=seed_atoms,
                    trajectory=[seed_atoms],
                    run_config=replace(
                        resolved_protocol.ariadne_run_config,
                        rng_seed=int(ariadne_rng.derived_seed_128),
                    ),
                    mock=True,
                )
                result_payload = result.to_dict()
                result_payload.update({
                    "seed_frame_id": seed_frame_id,
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "array_task_id": array_task_id,
                    "iteration": int(state.iteration),
                    "trajectory_sha256": str(picked_payload["trajectory_sha256"]),
                    "sampling_protocol": {
                        "sampling_aggressiveness": int(
                            resolved_protocol.sampling_aggressiveness
                        ),
                        "resolved_manifest": resolved_manifest,
                        "resolved_manifest_sha256": resolved_manifest_sha256,
                        "scale_model_manifest": scale_model_manifest,
                        "scale_model_manifest_sha256": scale_model_manifest_sha256,
                        "audit_manifest": audit_manifest,
                        "audit_manifest_sha256": audit_manifest_sha256,
                        "hidden_overrides_detected": list(
                            resolved_protocol.hidden_overrides_detected
                        ),
                    },
                    "randomness": ariadne_rng.to_dict(),
                })
                if isinstance(result_payload.get("selection_diagnostics"), dict):
                    from ..ferebus_prior import resolve_ferebus_prior_contract

                    result_payload["selection_diagnostics"].update({
                        "model_version": int(state.models_version),
                        "prior_mean_contract_sha256": (
                            resolve_ferebus_prior_contract(self.config).contract_sha256
                        ),
                        "seed_id": seed_id,
                        "seed_uid": seed_uid,
                        "array_task_id": array_task_id,
                        "seed_frame_id": seed_frame_id,
                    })
                staging_dir = seed_dir.parent / ("." + seed_dir.name + ".partial-dry")
                if staging_dir.exists() and not staging_dir.is_symlink():
                    shutil.rmtree(staging_dir)
                staging_dir.mkdir(parents=True, exist_ok=False)
                atomic_write_json(staging_dir / SEED_RESULT_FILENAME, result_payload)
                trajectory_coordinates = list(result.optimisation_trajectory_coordinates)
                if not trajectory_coordinates:
                    trajectory_coordinates = [
                        np.asarray(seed_atoms.coordinates, dtype=float).tolist()
                    ]
                write_optimisation_trajectory(
                    staging_dir,
                    atom_types=[str(atom.type) for atom in seed_atoms],
                    coordinate_frames=trajectory_coordinates,
                    alpha_values=list(result.alpha_trajectory),
                    gradient_norms=list(result.grad_norm_trajectory),
                    origins=list(result.optimisation_trajectory_origins),
                )
                write_seed_output_manifest(
                    staging_dir,
                    campaign_uid=campaign_uid,
                    iteration=int(state.iteration),
                    seed_id=seed_id,
                    seed_uid=seed_uid,
                    array_task_id=array_task_id,
                    task_success=True,
                    task_exit_code=0,
                )
                os.replace(staging_dir, seed_dir)
            landing_safety = result_payload.get("landing_safety") or {
                "accepted": True,
                "policy": "mock_safe",
                "reasons": [],
                "record_only_reasons": ["synthetic_mock_safety_metrics"],
                "metrics": {},
            }
            audit_record = {
                "seed_id": seed_id,
                "seed_uid": seed_uid,
                "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
                "result_json": result_path.relative_to(ariadne_root).as_posix(),
                "landing_safety": dict(landing_safety),
                "landing_candidates": list(result_payload.get("landing_candidates") or []),
                "handoff_accepted": True,
            }
            if isinstance(result_payload.get("selection_diagnostics"), dict):
                audit_record["selection_diagnostics"] = dict(
                    result_payload["selection_diagnostics"]
                )
            landing_audit_records.append(audit_record)
            if not (seed_dir / PROVENANCE_FILENAME).is_file():
                write_seed_provenance(
                    seed_dir,
                    campaign_uid=campaign_uid,
                    iteration=int(state.iteration),
                    trajectory_sha256=str(picked_payload["trajectory_sha256"]),
                    seed_frame_id=seed_frame_id,
                    seed_id=seed_id,
                    seed_uid=seed_uid,
                    array_task_id_zero_based=array_task_id,
                    seed_selection_origin=str(seed_record["selection_origin"]),
                    seed_variance_at_selection=seed_record.get("variance_at_selection"),
                    subspace_neighbour_frame_ids=[],
                    subspace_dimension=0,
                    subspace_eigenvalues=[],
                    mode_weighting_policy=self._mode_weighting_policy_or_default(),
                )
            enrich_with_ariadne(
                seed_dir,
                alpha_initial=float(result_payload.get("alpha_initial") or 0.0),
                alpha_final=float(result_payload.get("alpha_final") or 0.0),
                n_evaluations=int(result_payload.get("n_evaluations") or 0),
                fell_back_to_ds=bool(result_payload.get("fell_back_to_ds", False)),
                wall_seconds=float(result_payload.get("wall_seconds") or 0.0),
                return_code=int(result_payload.get("return_code") or 0),
            )
            if isinstance(result_payload.get("selection_diagnostics"), dict):
                enrich_with_error_calibration_input(
                    seed_dir,
                    dict(result_payload["selection_diagnostics"]),
                )
            #synthesise a placeholder whitened distance from the
            # alpha change during ARIADNE descent; threshold against the
            # trust-region bounds. Live executor swaps in the real metric.
            d_w = result_payload.get("whitened_distance_final")
            if d_w is None and result is not None:
                d_w = self._synthetic_whitened_distance(result)
            flag = None
            if d_w is not None:
                min_d, max_d = anti_overlap_whitened_distance_bounds(
                    resolved_protocol.effective_config
                )
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
                self._journal_event(
                    "anti_overlap_flagged",
                    iteration=int(state.iteration),
                    seed_id=seed_id,
                    seed_frame_id=seed_frame_id,
                    whitened_distance=float(d_w if d_w is not None else 0.0),
                    flag=str(flag),
                )
            if result_payload.get("alpha_final") is not None:
                last_alpha = max(last_alpha, float(result_payload["alpha_final"]))
            self.artefact_log.append(str(result_path))
            self.artefact_log.append(str(seed_dir / PROVENANCE_FILENAME))
            accepted_records.append({
                "seed_id": seed_id,
                "seed_uid": seed_uid,
                "array_task_id": array_task_id,
                "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
                "result_json": result_path.relative_to(ariadne_root).as_posix(),
                "provenance_json": (seed_dir / PROVENANCE_FILENAME).relative_to(ariadne_root).as_posix(),
                "output_manifest": (seed_dir / SEED_OUTPUT_MANIFEST_FILENAME).relative_to(ariadne_root).as_posix(),
                "seed_frame_id": seed_frame_id,
                "pool_row_index_zero_based": int(seed_record["pool_row_index_zero_based"]),
                "selection_origin": str(seed_record["selection_origin"]),
                "variance_at_selection": seed_record.get("variance_at_selection"),
                "alpha_initial": float(result_payload.get("alpha_initial") or 0.0),
                "alpha_final": float(result_payload.get("alpha_final") or 0.0),
                "whitened_distance_final": d_w,
                "landing_safety": dict(landing_safety),
                "landing_policy": str(landing_safety.get("policy", "unknown")),
                "selection_diagnostics": (
                    dict(result_payload["selection_diagnostics"])
                    if isinstance(result_payload.get("selection_diagnostics"), dict)
                    else None
                ),
                "return_code": int(result_payload.get("return_code") or 0),
                "result_sha256": sha256_file(result_path),
                "output_manifest_sha256": sha256_file(
                    seed_dir / SEED_OUTPUT_MANIFEST_FILENAME
                ),
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
            "campaign_uid": campaign_uid,
            "iteration": int(state.iteration),
            "trajectory_sha256": str(picked_payload["trajectory_sha256"]),
            "task_map": {
                "path": "TASK_MAP.json",
                "sha256": sha256_file(ariadne_root / "TASK_MAP.json"),
            },
            "expected_n": int(len(tasks)),
            "n_accepted": int(len(accepted_records)),
            "n_rejected": 0,
            "accepted": accepted_records,
            "rejected": [],
        })
        self.artefact_log.append(str(manifest_path))
        decision_contract = self._decision_contract_for_submission(
            state,
            "ARIADNE_ARRAY",
        )

        decision_path = write_ariadne_batch_decision(
            iter_dir,
            campaign_uid=str(campaign_uid),
            iteration=int(state.iteration),
            config_sha256=str(decision_contract["config_sha256"]),
            failure_threshold_fraction=float(
                decision_contract["failure_threshold_fraction"]
            ),
            expected_n=int(len(tasks)),
            n_accepted=int(len(accepted_records)),
            n_rejected=0,
            accepted=True,
            reasons=[],
        )
        self.artefact_log.append(str(decision_path))
        self._journal_event(
            "subspace_built",
            iteration=int(state.iteration),
            n_seeds=int(len(tasks)),
        )
        return {
            "last_acquisition_alpha0": float(last_alpha),
            # the per-iteration flag count is now a persisted state
            # field (last_n_anti_overlap_flagged) rather than a transient
            # return; STOP_CHECK rules can consume it as a convergence signal.
            "last_n_anti_overlap_flagged": int(flagged_count),
        }

    def _post_phase_b_diversity(self, state) -> Dict[str, Any]:
        import hashlib

        from ..handoff_manifests import (
            PHASE_B_SELECTION_SCHEMA_VERSION,
            ariadne_candidate_frames,
            ariadne_results_path,
            write_phase_b_selection_manifest,
        )
        from ..layout import active_phase_b_dir
        from ..point_allocation import (
            allocation_targets,
            create_point_allocation,
            point_allocation_path,
            stable_candidate_id,
        )
        from ..sampling_protocol import (
            phase_b_min_separation_from_resolved,
            resolve_sampling_protocol,
        )
        from ..sampling.descriptors import (
            build_condensed_distance_store,
            build_descriptor_from_config,
            partition_descriptor_frames,
        )
        from ..sampling.diversity import fps_select, _selector_contract
        from ..versioning.provenance import (
            enrich_with_phase_b,
            enrich_with_point_allocation,
        )

        iter_dir = self._iter_dir(state.iteration)
        phase_b_dir = active_phase_b_dir(iter_dir)
        phase_b_dir.mkdir(parents=True, exist_ok=True)
        from .config_lock import canonical_config, config_fingerprint

        ariadne_manifest, candidate_frames, accepted = ariadne_candidate_frames(
            iter_dir,
            expected_iteration=int(state.iteration),
            expected_config_sha256=config_fingerprint(canonical_config(self.config)),
        )
        descriptor = build_descriptor_from_config(self.config)
        descriptor_indices, descriptor_rejections = partition_descriptor_frames(
            descriptor,
            candidate_frames,
        )
        if descriptor_rejections:
            candidate_frames = [candidate_frames[index] for index in descriptor_indices]
            accepted = [accepted[index] for index in descriptor_indices]
        matrix = build_condensed_distance_store(descriptor, candidate_frames, workers=1)
        selection = fps_select(
            matrix,
            len(candidate_frames),
            descriptor_name=descriptor.name,
        )
        candidate_frames = [candidate_frames[index] for index in selection.indices]
        accepted = [accepted[index] for index in selection.indices]
        batch_total = int(self.config.point_allocation.batch_total_size)
        n_accepted_candidates = len(accepted)
        if n_accepted_candidates < batch_total:
            raise BackendSubmissionError(
                "point_allocation_underfilled: wanted batch_total_size="
                + str(batch_total)
                + " but only "
                + str(n_accepted_candidates)
                + " safe ARIADNE candidates are available"
            )
        primary_source = accepted[:batch_total]
        reserve_source = accepted[batch_total:]

        def allocation_record(rec, *, primary_rank=None, reserve_rank=None):
            candidate_id = stable_candidate_id(
                campaign_uid=str(state.campaign_uid),
                context="active",
                iteration=int(state.iteration),
                source_identity={
                    "seed_id": int(rec["seed_id"]),
                    "seed_uid": str(rec["seed_uid"]),
                    "seed_frame_id": rec.get("seed_frame_id"),
                    "result_sha256": str(rec.get("result_sha256") or ""),
                },
            )
            out_rec = dict(rec)
            out_rec["candidate_id"] = candidate_id
            if reserve_rank is not None:
                out_rec["reserve_rank"] = int(reserve_rank)
            provenance_path = Path(str(out_rec.get("provenance_json") or ""))
            if not provenance_path.is_file():
                raise BackendSubmissionError(
                    "dry-run Phase B candidate provenance is missing: "
                    + str(provenance_path)
                )
            enrich_with_phase_b(
                provenance_path.parent,
                selected_after_fps=reserve_rank is None,
                diversity_rank=primary_rank,
                descriptor_used="hybrid_alf_rmsd",
                candidate_id=candidate_id,
                reserve_candidate=reserve_rank is not None,
            )
            return out_rec

        primary_records = [
            allocation_record(rec, primary_rank=rank)
            for rank, rec in enumerate(primary_source, start=1)
        ]
        reserve_records = [
            allocation_record(rec, reserve_rank=rank)
            for rank, rec in enumerate(reserve_source)
        ]
        allocation_path = point_allocation_path(
            self.campaign_dir,
            context="active",
            iteration=int(state.iteration),
        )
        allocation = create_point_allocation(
            allocation_path,
            campaign_uid=str(state.campaign_uid),
            context="active",
            iteration=int(state.iteration),
            targets=allocation_targets(self.config, "active"),
            primary_candidates=primary_records,
            reserve_candidates=reserve_records,
        )
        slots_by_candidate = {
            str(slot["attempts"][0]["candidate_id"]): {
                "slot_id": int(slot["slot_id"]),
                "split": str(slot["split"]),
            }
            for slot in allocation["slots"]
        }
        final_records = []
        for final_rank, rec in enumerate(primary_records, start=1):
            out_rec = dict(rec)
            out_rec["candidate_pool_index_zero_based"] = int(final_rank - 1)
            out_rec["considered_rank"] = int(final_rank)
            out_rec["final_rank"] = int(final_rank)
            out_rec["kept_after_dedup"] = True
            out_rec["drop_reason"] = None
            out_rec.update(slots_by_candidate[str(out_rec["candidate_id"])])
            enrich_with_point_allocation(
                Path(str(out_rec["seed_dir"])),
                candidate_id=str(out_rec["candidate_id"]),
                context="active",
                slot_id=int(out_rec["slot_id"]),
                split=str(out_rec["split"]),
                replacement_round=0,
                allocation_slot_assignment_sha256=str(
                    allocation["slot_assignment_sha256"]
                ),
            )
            out_rec["provenance_sha256"] = hashlib.sha256(
                Path(str(out_rec["provenance_json"])).read_bytes()
            ).hexdigest()
            final_records.append(out_rec)
        from ..sampling_protocol import load_sampling_protocol

        resolved_protocol = load_sampling_protocol(
            self.campaign_dir,
            self.config,
            iteration=int(state.iteration),
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
            "kept_considered_indexes_zero_based": [
                int(i) for i in range(len(final_records))
            ],
            "dropped_considered_indexes_zero_based": [],
            "distances_to_nearest": [],
            "min_separation": float(effective_min_separation),
            "threshold_mode": str(threshold_mode),
            "effective_min_separation_angstrom": float(effective_min_separation),
            "scaled_distances_to_nearest": [],
            "novelty_scores": [],
            "relaxation": {"applied": False, "reason": None},
        }
        considered_path = phase_b_dir / "considered_candidates.xyz"
        selected_path = phase_b_dir / "selected.xyz"
        selected_frames = candidate_frames[:batch_total]
        comments = [
            "active iteration " + str(int(state.iteration)) + " Phase B rank " + str(rank)
            for rank in range(1, len(selected_frames) + 1)
        ]
        atomic_write_text(considered_path, _frames_to_xyz(selected_frames, comments))
        atomic_write_text(selected_path, _frames_to_xyz(selected_frames, comments))

        def iteration_relative(record):
            out_rec = dict(record)
            for key in ("seed_dir", "result_json", "provenance_json", "output_manifest"):
                if out_rec.get(key):
                    out_rec[key] = Path(str(out_rec[key])).resolve().relative_to(
                        iter_dir.resolve()
                    ).as_posix()
            return out_rec

        final_records = [iteration_relative(record) for record in final_records]
        reserve_records = [
            iteration_relative(record)
            for record in reserve_records
        ]
        def protocol_binding(path):
            resolved = Path(path).resolve()
            return {
                "path": resolved.relative_to(iter_dir.resolve()).as_posix(),
                "size": int(resolved.stat().st_size),
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            }

        manifest_path = write_phase_b_selection_manifest(iter_dir, {
            "schema_version": PHASE_B_SELECTION_SCHEMA_VERSION,
            "status": "complete",
            "campaign_uid": str(state.campaign_uid),
            "iteration": int(state.iteration),
            "descriptor": str(self.config.phase_b.descriptor),
            "selector": _selector_contract(),
            "sampling_protocol": {
                "sampling_aggressiveness": int(
                    resolved_protocol.sampling_aggressiveness
                ),
                "resolved": protocol_binding(resolved_protocol.manifest_path),
                "audit": protocol_binding(resolved_protocol.audit_manifest_path),
                "scale_model": protocol_binding(resolved_protocol.scale_model_path),
            },
            "source_ariadne_manifest": "ariadne/RESULTS.json",
            "source_ariadne_manifest_sha256": hashlib.sha256(
                ariadne_results_path(iter_dir).read_bytes()
            ).hexdigest(),
            "considered_candidates_xyz": {
                "path": considered_path.relative_to(iter_dir).as_posix(),
                "size": int(considered_path.stat().st_size),
                "sha256": hashlib.sha256(considered_path.read_bytes()).hexdigest(),
            },
            "selected_xyz": {
                "path": selected_path.relative_to(iter_dir).as_posix(),
                "size": int(selected_path.stat().st_size),
                "sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
            },
            "point_allocation": {
                "manifest": allocation_path.resolve().relative_to(
                    iter_dir.resolve()
                ).as_posix(),
                "slot_assignment_sha256": str(allocation["slot_assignment_sha256"]),
                "targets": dict(allocation["targets"]),
                "reserve": reserve_records,
                "reserve_count": int(len(reserve_records)),
            },
            "n_candidates": int(n_accepted_candidates),
            "n_considered": int(len(final_records)),
            "n_kept": int(len(final_records)),
            "considered": list(final_records),
            "final": list(final_records),
            "dedup": dedup_payload,
            "safety_filter": {
                "enabled": True,
                "n_input": int(len(selection.indices) + len(descriptor_rejections)),
                "n_kept": int(len(selection.indices)),
                "n_dropped": int(len(descriptor_rejections)),
                "descriptor_rejections": descriptor_rejections,
            },
        })
        self.artefact_log.extend([str(considered_path), str(selected_path)])
        self.artefact_log.append(str(manifest_path))
        return {}

    def _post_gaussian(self, state) -> Dict[str, Any]:
        return self._stub_quantum_outputs(state, stage="GAUSSIAN", initial=False)

    def _post_aimall(self, state) -> Dict[str, Any]:
        return self._stub_quantum_outputs(state, stage="AIMALL", initial=False)

    def _post_replacement_gaussian(self, state) -> Dict[str, Any]:
        return self._stub_quantum_outputs(
            state, stage="GAUSSIAN", initial=False, replacement=True,
        )

    def _post_replacement_aimall(self, state) -> PhaseResult:
        self._stub_quantum_outputs(
            state, stage="AIMALL", initial=False, replacement=True,
        )
        return PhaseResult(
            is_complete=True,
            next_phase_override="ALLOCATION_CHECK",
        )

    def _post_ferebus(self, state) -> Dict[str, Any]:
        """Commit the next models iteration. QM reference data itself has already
        been committed by the inline APPEND phase one step earlier."""
        v_models = self._versioning("models")
        v_models.recover_dangling_staging()
        committed = v_models.list_committed_versions()
        expected_next = int(getattr(state, "reference_data_version", -1))
        if expected_next < 0:
            raise ValueError(
                "FEREBUS requires a committed reference_data_version, got "
                + repr(getattr(state, "reference_data_version", None))
            )
        if expected_next in committed:
            repaired_version = int(expected_next)
            v_models.resolve(repaired_version, verification="deep")
            v_models.ensure_current(max(committed))
            self._journal_event(
                "models_committed",
                phase="FEREBUS",
                iteration=int(state.iteration),
                models_version=int(repaired_version),
                idempotent_skip=True,
            )
            return {"models_version": int(repaired_version)}
        next_version = int(expected_next)
        from . import input_staging as _stg

        if _stg._model_bootstrap_context(self.campaign_dir) is not None:
            staging, _n_tasks = _stg.stage_ferebus_inputs(
                self.campaign_dir,
                self.config,
                next_version,
                is_initial=False,
            )
            self._prepare_dry_staged_models(staging)
            from .live_executor import LiveBackendsPhaseExecutor

            result = LiveBackendsPhaseExecutor._parse_ferebus_postprocess(
                self,
                state,
                "FEREBUS",
                [],
            )
            if result.failure_reason is not None:
                raise BackendSubmissionError(result.failure_reason)
            return dict(result.state_updates or {})
        self._commit_dry_model_snapshot(next_version)
        return {"models_version": int(next_version)}

    # --- helpers --------------------------------------------------

    def _recent_seed_cooldown(self) -> int:
        return int(self.config.seed_selection.recent_seed_cooldown_iterations)

    def _trajectory_sha256_if_available(self) -> str:
        """Return the SHA-256 of the imported trajectory pool manifest if
        present; empty string otherwise. The daemon does not require a pool
        to be imported in dry-run, so this is a soft lookup."""
        from ..acquisition.trajectory_pool import POOL_MANIFEST_FILENAME, POOL_SUBDIR
        from ..strict_json import strict_json as _json

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

        Loads seed_selection/SELECTION.json and returns its
        frame_ids field. A missing file returns an empty list; a malformed
        manifest is a hard contract error."""
        from ..handoff_manifests import load_seeds_picked, seeds_picked_path

        iter_dir = self._iter_dir(iteration)
        path = seeds_picked_path(iter_dir)
        if not path.is_file():
            return []
        data = load_seeds_picked(
            iter_dir,
            expected_iteration=int(iteration),
        )
        fids = data["frame_ids"]
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

    def _seed_index_record_from_pointdir(
        self,
        pointdir: Path,
        *,
        iteration: int,
        pointdir_name: str,
    ) -> Dict[str, Any]:
        """Build one exclusion-index record from authoritative provenance."""
        from ..versioning.provenance import validate_provenance

        data = validate_provenance(pointdir, iteration=int(iteration))
        seed = data["seed"]
        return {
            "iteration": int(iteration),
            "pointdir_name": str(pointdir_name),
            "seed_frame_id": seed.get("frame_id"),
            "trajectory_sha256": str(data["trajectory_sha256"]),
        }

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
            from .filesystem import operational_path

            journal_path = operational_path(self.campaign_dir, "journal.ndjson")
            append_event(
                journal_path,
                event_type,
                max_bytes=int(self.config.runtime.journal_max_bytes),
                retained_files=int(self.config.runtime.journal_retained_files),
                lock_timeout_seconds=int(
                    self.config.runtime.ledger_lock_timeout_seconds
                ),
                **payload,
            )
        except Exception:
            pass

    def _failure_threshold_for_submission(self, state, phase_name: str) -> float:
        """Read the immutable batch threshold captured before submission."""
        return float(
            self._decision_contract_for_submission(state, phase_name)[
                "failure_threshold_fraction"
            ]
        )

    def _decision_contract_for_submission(self, state, phase_name: str) -> Dict[str, Any]:
        from .submission_intent import load_intent

        intent = load_intent(
            self.campaign_dir,
            str(phase_name),
            int(state.iteration),
            expected_campaign_uid=str(state.campaign_uid),
        )
        if not isinstance(intent, dict):
            raise ValueError("submission intent is unavailable for batch decision")
        contract = intent.get("decision_contract")
        if not isinstance(contract, dict):
            raise ValueError("submission intent has no decision_contract snapshot")
        return dict(contract)


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
          never:                only on the first active-iteration call

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

    @staticmethod
    def _write_dry_quantum_wfn(
        path: Path,
        *,
        method: str,
        point_index: int,
        total_energy_ha: float,
    ) -> None:
        """Write a minimal parseable three-atom WFN for dry contract tests."""
        from ichor.core.atoms import Atom, Atoms
        from ichor.core.common.units import AtomicDistance
        from ichor.core.files.gaussian.wfn import MolecularOrbital, WFN

        displacement = 0.01 * int(point_index)
        atoms = Atoms(
            [
                Atom("O", 0.0, 0.0, 0.0, units=AtomicDistance.Bohr),
                Atom(
                    "H",
                    1.43 + displacement,
                    0.0,
                    0.0,
                    units=AtomicDistance.Bohr,
                ),
                Atom(
                    "H",
                    -0.36,
                    1.38 + displacement,
                    0.0,
                    units=AtomicDistance.Bohr,
                ),
            ]
        )
        wfn = WFN(path, method=str(method))
        wfn.atoms = atoms
        wfn.n_orbitals = 1
        wfn.n_primitives = 3
        wfn.n_nuclei = 3
        wfn.centre_assignments = [1, 2, 3]
        wfn.type_assignments = [1, 1, 1]
        wfn.primitive_exponents = [1.0, 1.0, 1.0]
        wfn.molecular_orbitals = [
            MolecularOrbital(1, 2.0, -0.5, [1.0, 0.0, 0.0])
        ]
        wfn.total_energy = float(total_energy_ha)
        wfn.virial_ratio = 2.0
        wfn.write()

    @staticmethod
    def _write_dry_aimall_int(
        path: Path,
        *,
        atom_name: str,
        method: str,
        iqa_ha: float,
        multipole_names: Sequence[str],
    ) -> None:
        """Write the smallest INT accepted by the production AIMAll parser."""
        lines = [
            "AIMAll Version 19.10.12,",
            "Current Directory .",
            "",
            "Inp File: input.inp",
            "Wfn File: input.wfn",
            "Out File: " + path.name,
            "",
            "Title: DRYRUN",
            "List of critical points",
            "Optional parameters",
            "",
            "DFT Model: " + str(method),
            "Integration is over atom " + str(atom_name),
            "Results of the basin integration",
            "N = 0.0 Charge = 0.0",
            "L = 0.0",
            "Atomic Traceless Quadrupole",
            "Real Spherical Harmonic Moments",
            "moment header 1",
            "moment header 2",
            "moment header 3",
        ]
        lines.extend(str(name) + " = 0.0" for name in multipole_names)
        lines.extend(
            [
                "IQA Energy Components",
                "Component Value",
                "E_IQA(A) = " + repr(float(iqa_ha)),
                "End IQA components",
                "Total time = 0",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    def _stub_quantum_outputs(
        self,
        state,
        *,
        stage: str,
        initial: bool,
        replacement: bool = False,
    ) -> Dict[str, Any]:
        """Create stub PointDirectories so APPEND has content to stage. We
        write into a transient staging area under .DATA/STAGING/quantum/ that
        the APPEND step copies into the next training iteration.
        """
        from . import input_staging as _stg
        from ..point_allocation import (
            pending_attempts,
            point_allocation_path,
            read_point_allocation,
        )
        from ..versioning.provenance import enrich_with_point_allocation

        context = "bootstrap" if initial else "active"
        allocation_iteration = 0 if initial else int(state.iteration)
        allocation_path = point_allocation_path(
            self.campaign_dir,
            context=context,
            iteration=allocation_iteration,
        )
        allocation = read_point_allocation(allocation_path)
        pending = pending_attempts(allocation)
        expected_round = int(getattr(state, "replacement_round", 0)) if replacement else 0
        attempts = [
            attempt for attempt in pending
            if int(attempt.get("round", -1)) == expected_round
        ]
        if len(attempts) != len(pending):
            raise BackendSubmissionError(
                "dry-run quantum phase does not match pending allocation round"
            )
        if replacement:
            from ..replacement_sampling import replacement_round_dir

            staging_root = replacement_round_dir(
                self.campaign_dir,
                context=context,
                iteration=allocation_iteration,
                replacement_round=expected_round,
            )
        else:
            from ..layout import staging_context_dir

            staging_root = staging_context_dir(
                self.campaign_dir,
                context=context,
                iteration=allocation_iteration,
            )
        staging_root.mkdir(parents=True, exist_ok=True)
        n_points = len(attempts)
        pointdirs = []
        for i, attempt in enumerate(attempts):
            point_index = (
                int(allocation["targets"]["total"])
                + int(attempt.get("reserve_rank", i))
                if replacement
                else i
            )
            point_dir = staging_root / staging_pointdir_name(point_index)
            point_dir.mkdir(exist_ok=True)
            provenance_source = Path(str(attempt.get("provenance_json") or ""))
            provenance_dest = point_dir / PROVENANCE_FILENAME
            if not provenance_dest.is_file() and provenance_source.is_file():
                import shutil

                shutil.copy2(provenance_source, provenance_dest)
            if not provenance_dest.is_file():
                write_seed_provenance(
                    point_dir,
                    campaign_uid=str(state.campaign_uid),
                    iteration=int(state.iteration),
                    trajectory_sha256=self._trajectory_sha256_if_available(),
                    seed_frame_id=attempt.get("frame_id"),
                    seed_selection_origin=str(attempt.get("source", "dry_run_quantum")),
                    seed_variance_at_selection=None,
                    subspace_neighbour_frame_ids=[],
                    subspace_dimension=0,
                    subspace_eigenvalues=[],
                    mode_weighting_policy="dry_run",
                )
            enrich_with_point_allocation(
                point_dir,
                candidate_id=str(attempt["candidate_id"]),
                context=context,
                slot_id=int(attempt["slot_id"]),
                split=str(attempt["split"]),
                replacement_round=int(attempt.get("round", 0)),
                allocation_slot_assignment_sha256=str(
                    allocation["slot_assignment_sha256"]
                ),
            )
            artefact_name = "stub_" + stage + ".txt"
            (point_dir / artefact_name).write_text(
                "DRYRUN " + stage + " output for point " + str(i) + "\n",
                encoding="utf-8",
            )
            self.artefact_log.append(str(point_dir / artefact_name))
            pointdirs.append(point_dir)
        _stg.write_points_file(staging_root, pointdirs)
        phase_name = (
            ("INITIAL_REPLACEMENT_" if initial else "REPLACEMENT_") + stage
            if replacement
            else (("INITIAL_" if initial else "") + stage)
        )
        _stg.write_quantum_acceptance_manifest(
            staging_root,
            phase_name=phase_name,
            iteration=int(state.iteration),
            accepted=pointdirs,
            rejected=[],
        )
        if stage == "AIMALL":
            from ichor.core.common.constants import multipole_names

            from .quantum_acceptance_receipts import (
                write_quantum_acceptance_receipt,
            )
            from .quantum_quality import (
                canonicalise_aimall_method,
                write_quantum_quality_manifest,
            )

            canonical_method = canonicalise_aimall_method(
                self.config.gaussian.method
            )

            records = []
            atom_names = ("O1", "H2", "H3")
            for point_index, point_dir in enumerate(pointdirs):
                total_energy = -1.0 - 0.01 * point_index
                atom_iqa = (0.6 * total_energy, 0.2 * total_energy, 0.2 * total_energy)
                records.append(
                    {
                        "pointdir": point_dir.name,
                        "accepted": True,
                        "reasons": [],
                        "atom_count": 3,
                        "expected_atom_names": list(atom_names),
                        "n_int": 3,
                        "sum_iqa_ha": total_energy,
                        "wfn_total_energy_ha": total_energy,
                        "wfn_virial_ratio": 2.0,
                        "iqa_energy_recovery_error_ha": 0.0,
                        "max_abs_integration_error": 0.0,
                        "per_atom": [
                            {
                                "atom": atom_name,
                                "int_file": atom_name.lower() + ".int",
                                "dft_model": canonical_method,
                                "canonical_dft_model": canonical_method,
                                "iqa_ha": value,
                                "integration_error": 0.0,
                                "multipoles": {
                                    name: 0.0 for name in multipole_names
                                },
                                "reasons": [],
                            }
                            for atom_name, value in zip(atom_names, atom_iqa)
                        ],
                    }
                )
            manifest = write_quantum_quality_manifest(
                staging_root,
                phase_name=phase_name,
                iteration=int(state.iteration),
                records=records,
                gates=getattr(self.config, "quality_gates", None),
            )
            self.artefact_log.append(str(manifest))
            for point_index, (point_dir, quality_record) in enumerate(
                zip(pointdirs, records)
            ):
                for filename in (
                    "input.gjf",
                    "input.gau",
                    "AIMALL_TASK.json",
                    "GAUSSIAN_TASK_RECEIPT.json",
                    "WFN_METHOD_RECEIPT.json",
                    "AIMALL_COMPLETION_RECEIPT.json",
                ):
                    path = point_dir / filename
                    if not path.exists():
                        path.write_text("DRYRUN synthetic evidence\n", encoding="utf-8")
                self._write_dry_quantum_wfn(
                    point_dir / "input.wfn",
                    method=canonical_method,
                    point_index=point_index,
                    total_energy_ha=float(quality_record["wfn_total_energy_ha"]),
                )
                atomic_dir = point_dir / "input_atomicfiles"
                atomic_dir.mkdir(exist_ok=True)
                for atom_record in quality_record["per_atom"]:
                    self._write_dry_aimall_int(
                        atomic_dir / str(atom_record["int_file"]),
                        atom_name=str(atom_record["atom"]),
                        method=canonical_method,
                        iqa_ha=float(atom_record["iqa_ha"]),
                        multipole_names=multipole_names,
                    )
                write_quantum_acceptance_receipt(
                    self.campaign_dir,
                    point_dir,
                    phase_name=phase_name,
                    iteration=int(state.iteration),
                    quality_manifest=manifest,
                    quality_record=quality_record,
                )
            gaussian_phase = (
                "INITIAL_REPLACEMENT_GAUSSIAN"
                if initial and replacement
                else "REPLACEMENT_GAUSSIAN"
                if replacement
                else "INITIAL_GAUSSIAN"
                if initial
                else "GAUSSIAN"
            )
            _stg.record_allocation_quantum_results(
                self.campaign_dir,
                context=context,
                iteration=allocation_iteration,
                staging_dir=staging_root,
                gaussian_phase=gaussian_phase,
                aimall_phase=phase_name,
                expected_method=str(self.config.gaussian.method),
            )
            if not initial and bool(getattr(self.config.error_calibration, "enabled", True)):
                from .error_calibration import (
                    append_records,
                    build_calibration_model,
                    mark_calibration_model_stale,
                    synthetic_dry_records,
                    write_calibration_model,
                    write_iteration_audit,
                )
                from ..ferebus_prior import resolve_ferebus_prior_contract

                try:
                    synthetic = synthetic_dry_records(
                        iteration=int(state.iteration),
                        models_version=int(getattr(state, "models_version", -1)),
                        n_points=n_points,
                        sampling_aggressiveness=int(
                            self.config.campaign.sampling_aggressiveness
                        ),
                        prior_mean_contract_sha256=(
                            resolve_ferebus_prior_contract(self.config).contract_sha256
                        ),
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
