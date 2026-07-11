"""POLUS wrapper providing a pluggable distance matrix.

Two layers:

* fps_select(distance_matrix, n_select, seed_index=None):
    Standalone greedy furthes-point sampling over an arbitrary distance
    matrix. Used in unit tests and as a small-data path that does not need
    the full POLUS install. Returns FPSResult (indices + diversities).


* make_pluggable_polus_sampler(precomputed_distance_matrix, **divsampler_kw):
    Lazily imports polus.trajectories.diversity.DIVSampler and returns a
    subclass that overrides SetRMSDMatrix to inject a caller-supplied
    distance matrix. Used in production for large trajectories where
    POLUS handles file I/O and the FPS step. The caller computes the
    distance matrix with whichever Descriptor they want.

Both layers produce the same selection sequence given the same input
(modulo any seed-geometry choice POLUS makes from ComputeCentroid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ichor.core.atoms import Atoms

from .descriptors import Descriptor, MassWeightedRMSDDescriptor


__all__ = [
    "fps_select",
    "FPSResult",
    "make_pluggable_polus_sampler",
    "DEFAULT_DESCRIPTORS",
]


DEFAULT_DESCRIPTORS = {
    "rmsd_massweight": MassWeightedRMSDDescriptor,
}
FPS_TIE_QUANTISATION = 1.0e-12






@dataclass(frozen=True)
class FPSResult:
    indices: List[int]
    diversities: List[float]
    distance_matrix_shape: Tuple[int, int]
    descriptor_name: str

    @property
    def n(self) -> int:
        return len(self.indices)


def fps_select(
    distance_matrix: np.ndarray,
    n_select: int,
    seed_index: Optional[int] = None,
    descriptor_name: str = "unknown",
) -> FPSResult:
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distance_matrix must be square 2D; got shape {D.shape}")
    if not np.allclose(D, D.T, atol=1.0e-9):
        raise ValueError("distance_matrix must be symmetric")
    n = D.shape[0]
    if n_select <= 0:
        return FPSResult(
            indices=[],
            diversities=[],
            distance_matrix_shape=(n, n),
            descriptor_name=descriptor_name,
        )
    if n_select > n:
        raise ValueError(f"n_select {n_select} > n {n}")

    if seed_index is None:
        row_sums = D.sum(axis=1)
        ranked = np.round(row_sums / FPS_TIE_QUANTISATION) * FPS_TIE_QUANTISATION
        seed = int(np.lexsort((np.arange(n, dtype=int), ranked))[0])
    else:
        if not 0 <= int(seed_index) < n:
            raise ValueError(f"seed_index {seed_index} out of range")
        seed = int(seed_index)

    selected = [seed]
    min_dists = D[seed].copy()
    min_dists[seed] = 0.0
    diversities: List[float] = [0.0]

    for _ in range(1, n_select):
        candidates = np.where(np.isin(np.arange(n), selected, invert=True))[0]
        if candidates.size == 0:
            break
        scores = min_dists[candidates]
        order = np.lexsort((candidates, -scores))
        best_local = int(order[0])
        nxt = int(candidates[best_local])
        diversities.append(float(scores[best_local]))
        selected.append(nxt)
        new_dists = D[nxt]
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[nxt] = 0.0

    return FPSResult(
        indices=selected,
        diversities=diversities,
        distance_matrix_shape=(n, n),
        descriptor_name=descriptor_name,
    )





def make_pluggable_polus_sampler(precomputed_distance_matrix, **divsampler_kwargs):
    from polus.trajectories.diversity import DIVSampler

    D = np.asarray(precomputed_distance_matrix, dtype=np.float32)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance matrix must be square 2D")

    class _PluggableMetricDIVSampler(DIVSampler):
        _PRECOMPUTED_MATRIX = D

        def SetRMSDMatrix(self):
            self.RotateTrajectory(self.refGeomFilename, self.rotateTraj, self.rotMethod)
            if not isinstance(self.rotTraj, list):
                raise RuntimeError("POLUS rotation produced no trajectory")
            self.samplePool = list(range(len(self.rotTraj)))
            if self.seedFilename is None:
                self.ComputeCentroid()
            else:
                self.GetSeedGeometry(self.seedFilename)
            expected = (self.ngeoms, self.ngeoms)
            if self._PRECOMPUTED_MATRIX.shape != expected:
                raise ValueError(
                    f"precomputed matrix shape {self._PRECOMPUTED_MATRIX.shape} "
                    f"!= POLUS ngeoms {expected}"
                )
            self.matrRMSD = self._PRECOMPUTED_MATRIX

    return _PluggableMetricDIVSampler(**divsampler_kwargs)

# --- helpers used by main(argv) below -------------------------------


def _write_xyz_file(frames, path):
    """Write a list of ICHOR Atoms objects to a plain xyz file.

    Format: natoms line, comment line, then one atom per line. POLUS
    parses this same shape so the same writer covers both Phase A
    (consumed by the daemon parser) and Phase B (consumed by APPEND).
    """
    out_lines = []
    for k, frame in enumerate(frames):
        out_lines.append(str(len(frame)))
        out_lines.append("frame " + str(k))
        for atom in frame:
            out_lines.append(
                "{symbol} {x:.6f} {y:.6f} {z:.6f}".format(
                    symbol=atom.type, x=float(atom.x),
                    y=float(atom.y), z=float(atom.z),
                )
            )
    Path(path).write_text(chr(10).join(out_lines) + chr(10), encoding="utf-8")


def _write_index_file(indices, path):
    """Write the per-line list of selected frame ids POLUS produces."""
    lines = []
    anchor_counter = 0
    for value in indices:
        if value is None:
            lines.append("anchor:" + str(anchor_counter))
            anchor_counter += 1
        else:
            lines.append(str(int(value)))
    Path(path).write_text(
        chr(10).join(lines) + chr(10),
        encoding="utf-8",
    )


def _phase_b_target_size(config, n_candidates, iteration=0):
    """Return the exact active point-allocation target for Phase-B FPS."""
    del iteration
    target = int(config.point_allocation.batch_total_size)
    if int(n_candidates) < target:
        raise ValueError(
            "point_allocation_underfilled: wanted batch_total_size="
            + str(target)
            + " but only "
            + str(int(n_candidates))
            + " safe ARIADNE candidates are available"
        )
    return target


def _load_committed_reference_data(campaign_dir):
    """Read the authoritative cumulative QM reference-data view.

    Returns an empty list when nothing has been committed yet, so the
    anti-overlap filter is a clean no-op against a fresh campaign.
    """
    from ..versioning.reference_data import ReferenceDataVersioning
    from ichor.core.files import PointDirectory
    v = ReferenceDataVersioning(Path(campaign_dir) / "QM_REFERENCE_DATA")
    cur = v.current_version()
    if cur is None:
        return []
    view = v.resolve(int(cur), verification="metadata")
    atoms_list = []
    for entry in view.entries:
        try:
            pd = PointDirectory(entry.pointdir_path)
            atoms_list.append(pd.atoms)
        except Exception as exc:
            raise ValueError(
                "committed QM reference point cannot be parsed: "
                + str(entry.pointdir_path)
            ) from exc
    return atoms_list


def _build_phase_b_posterior(campaign, config):
    """Load the current committed FEREBUS models for acquisition_weighted."""
    from ..daemon.state import DEFAULT_STATE_FILENAME, read_state
    from ichor.core.adversarial.posterior import TotalEnergyPosterior
    from ..versioning.trained_models import load_trained_models

    state_path = (
        Path(campaign)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / DEFAULT_STATE_FILENAME
    )
    state = read_state(state_path)
    models_version = int(getattr(state, "models_version", -1))
    if models_version < 0:
        raise FileNotFoundError("state has no committed models_version")
    property_name = str(config.acquisition.property_name)
    from ..daemon.artifact_contracts import verify_committed_model_version
    verify_committed_model_version(campaign, models_version)
    _, models = load_trained_models(
        campaign,
        models_version,
        verification="deep",
    )
    return TotalEnergyPosterior(
        models,
        property_name=property_name,
        scaled=bool(config.acquisition.use_scaled_posterior_covariance),
    )


def _phase_b_landing_safety_filter(
    candidate_frames,
    candidate_records,
    *,
    accept_legacy_missing_landing_safety: bool = False,
):
    missing = [
        i for i, rec in enumerate(candidate_records)
        if not isinstance(rec.get("landing_safety"), dict)
    ]
    if len(missing) == len(candidate_records):
        if not bool(accept_legacy_missing_landing_safety):
            raise ValueError(
                "all Phase B candidates are missing landing_safety metadata; "
                "set adversarial_safety.accept_legacy_missing_landing_safety "
                "to true only for deliberate legacy migration"
            )
        return (
            list(candidate_frames),
            list(candidate_records),
            {
                "enabled": True,
                "legacy_missing_safety": True,
                "n_input": int(len(candidate_records)),
                "n_kept": int(len(candidate_records)),
                "n_dropped": 0,
                "dropped": [],
            },
        )
    if missing:
        raise ValueError(
            "partial missing landing_safety metadata for candidate indices "
            + repr([int(i) for i in missing])
        )

    kept_frames = []
    kept_records = []
    dropped = []
    for i, (frame, rec) in enumerate(zip(candidate_frames, candidate_records)):
        safety = rec.get("landing_safety")
        if bool(safety.get("accepted", False)):
            kept_frames.append(frame)
            kept_records.append(rec)
        else:
            dropped.append({
                "candidate_pool_index_zero_based": int(i),
                "seed_id": int(rec.get("seed_id", i + 1)),
                "seed_uid": rec.get("seed_uid"),
                "policy": str(safety.get("policy", "unknown")),
                "reasons": [str(r) for r in safety.get("reasons", [])],
            })
    return (
        kept_frames,
        kept_records,
        {
            "enabled": True,
            "legacy_missing_safety": False,
            "n_input": int(len(candidate_records)),
            "n_kept": int(len(kept_records)),
            "n_dropped": int(len(dropped)),
            "dropped": dropped,
        },
    )


def _phase_b_json_safe(value):
    """Return a JSON-safe copy of Phase B diagnostics.

    Anti-overlap diagnostics can legitimately contain infinite nearest
    distances when no committed QM reference data exists. The manifest writer uses
    strict JSON, so non-finite diagnostic values are represented as null at the
    serialisation boundary.
    """
    if isinstance(value, dict):
        return {str(k): _phase_b_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_phase_b_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_phase_b_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if np.isfinite(val) else None
    return value


def _append_phase_b_journal_event(campaign, event_type, **payload) -> None:
    try:
        from ..daemon.journal import append_event

        journal_path = Path(campaign) / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
        append_event(journal_path, event_type, **payload)
    except Exception:
        return


def _phase_b_refill_after_anti_overlap(
    *,
    ordered_indices,
    candidate_frames,
    candidate_records,
    training,
    min_separation: float,
    target_size: int,
):
    from .anti_overlap import DedupReport, min_distance_to_training

    considered_indices = []
    considered_frames = []
    considered_records = []
    kept_indices = []
    dropped_indices = []
    distances = []
    kept_frames = []

    for candidate_index in ordered_indices:
        raw_index = len(considered_indices)
        cand = candidate_frames[int(candidate_index)]
        distance = float(
            min_distance_to_training(cand, list(training) + list(kept_frames))
        )
        considered_indices.append(int(candidate_index))
        considered_frames.append(cand)
        considered_records.append(candidate_records[int(candidate_index)])
        distances.append(distance)
        if distance < float(min_separation):
            dropped_indices.append(int(raw_index))
        else:
            kept_indices.append(int(raw_index))
            kept_frames.append(cand)
            if len(kept_indices) >= int(target_size):
                break

    rejected_distances = [
        float(distances[i])
        for i in dropped_indices
        if i < len(distances) and np.isfinite(float(distances[i]))
    ]
    rejected_distances.sort()
    refill = {
        "enabled": True,
        "target_batch_total_size": int(target_size),
        "reserve_order_size": int(len(ordered_indices)),
        "reserve_candidates_considered": int(len(considered_indices)),
        "refill_applied": int(len(considered_indices)) > int(target_size),
        "rejected_by_anti_overlap": int(len(dropped_indices)),
        "closest_rejected_distances_angstrom": rejected_distances[:8],
        "reserve_exhausted": int(len(kept_indices)) < int(target_size),
    }
    return (
        considered_indices,
        considered_frames,
        considered_records,
        DedupReport(
            kept_indices=tuple(kept_indices),
            dropped_indices=tuple(dropped_indices),
            distances_to_nearest=tuple(distances),
            min_separation=float(min_separation),
        ),
        refill,
    )


def _phase_b_build_reserve(
    *,
    ordered_indices,
    already_considered_indices,
    candidate_frames,
    candidate_records,
    training,
    selected_frames,
    min_separation: float,
):
    """Return the remaining safe FPS candidates in deterministic reserve order."""
    from .anti_overlap import min_distance_to_training

    considered = {int(value) for value in already_considered_indices}
    accepted_context = list(selected_frames)
    reserve_indices = []
    reserve_frames = []
    reserve_records = []
    reserve_distances = []
    rejected = []
    for candidate_index in ordered_indices:
        candidate_index = int(candidate_index)
        if candidate_index in considered:
            continue
        frame = candidate_frames[candidate_index]
        distance = float(
            min_distance_to_training(frame, list(training) + accepted_context)
        )
        if not np.isfinite(distance) or distance < float(min_separation):
            rejected.append({
                "candidate_pool_index_zero_based": candidate_index,
                "distance_to_nearest_angstrom": (
                    float(distance) if np.isfinite(distance) else None
                ),
                "reason": "min_separation",
            })
            continue
        reserve_indices.append(candidate_index)
        reserve_frames.append(frame)
        reserve_records.append(candidate_records[candidate_index])
        reserve_distances.append(distance)
        accepted_context.append(frame)
    return {
        "candidate_indices": reserve_indices,
        "frames": reserve_frames,
        "records": reserve_records,
        "distances_to_nearest_angstrom": reserve_distances,
        "rejected": rejected,
    }


def _run_phase_a(campaign, config):
    """POLUS Phase-A: pick a diverse subsample from the imported
    trajectory pool to seed the campaign with initial training points.

    Phase A always uses mass-weighted RMSD as the distance metric --
    no posterior exists yet, so any descriptor that needs one (e.g.
    acquisition-weighted) does not apply. The two outputs match what
    POLUS itself would write so the daemon postprocess parser does not
    care which path produced them.
    """
    from ..acquisition.trajectory_pool import TrajectoryPool
    from ..acquisition.trajectory_pool import POOL_MANIFEST_FILENAME, POOL_SUBDIR
    from ..handoff_manifests import write_phase_a_sample_manifest
    from .descriptors import MassWeightedRMSDDescriptor
    import sys as _sys

    try:
        pool = TrajectoryPool.load(campaign)
    except (FileNotFoundError, ValueError) as exc:
        print(
            "trajectory pool not loadable: " + str(exc),
            file=_sys.stderr,
        )
        return 3

    frames = pool.to_atoms_list()
    if not frames:
        print("trajectory pool is empty", file=_sys.stderr)
        return 3

    try:
        from ..bootstrap_anchor import (
            plan_bootstrap_anchors,
            write_bootstrap_anchor_manifest,
        )
        from ..daemon.pool_feasibility import require_pool_feasibility

        anchor_plan, anchor_frames = plan_bootstrap_anchors(
            campaign,
            config,
            pool_frames=frames,
        )
        feasibility = require_pool_feasibility(campaign, config)
    except Exception as exc:
        print(str(exc), file=_sys.stderr)
        return 3

    n_select = int(anchor_plan.bootstrap_total_size)
    pool_select = int(anchor_plan.pool_total_needed)
    excluded_pool_ids = set(int(i) for i in anchor_plan.excluded_pool_frame_ids)
    candidate_pool_ids = [
        int(i) for i in range(len(frames)) if int(i) not in excluded_pool_ids
    ]
    if pool_select > len(candidate_pool_ids):
        print(
            "point_allocation_bootstrap_total_exceeds_pool: "
            + "point_allocation.bootstrap_total_size="
            + str(n_select)
            + ", point_allocation.anchor_count="
            + str(int(anchor_plan.n_anchor))
            + ", pool_needed_after_anchors="
            + str(pool_select)
            + ", available_pool_frames="
            + str(len(candidate_pool_ids)),
            file=_sys.stderr,
        )
        return 3

    descriptor = MassWeightedRMSDDescriptor()
    selected_pool_indices: List[int] = []
    reserve_pool_indices: List[int] = []
    diversities: List[float] = []
    if pool_select > 0:
        candidate_frames = [frames[i] for i in candidate_pool_ids]
        matrix = descriptor.pairwise_distance_matrix(candidate_frames)
        sel = fps_select(matrix, len(candidate_frames), descriptor_name=descriptor.name)
        ordered_pool_indices = [int(candidate_pool_ids[i]) for i in sel.indices]
        selected_pool_indices = ordered_pool_indices[:pool_select]
        reserve_pool_indices = ordered_pool_indices[pool_select:]
        diversities = [float(v) for v in sel.diversities[:pool_select]]
    else:
        sel = FPSResult(
            indices=[],
            diversities=[],
            distance_matrix_shape=(len(candidate_pool_ids), len(candidate_pool_ids)),
            descriptor_name=descriptor.name,
        )
    selected_frames = list(anchor_frames) + [frames[i] for i in selected_pool_indices]
    selected_indices = [None] * int(anchor_plan.n_anchor) + [
        int(i) for i in selected_pool_indices
    ]

    from ..layout import bootstrap_selection_dir

    outdir = bootstrap_selection_dir(campaign)
    outdir.mkdir(parents=True, exist_ok=True)
    sample_path = outdir / "selected.xyz"
    index_path = outdir / "selected_indices.dat"
    _write_xyz_file(selected_frames, sample_path)
    _write_index_file(selected_indices, index_path)
    try:
        from ..daemon.state import DEFAULT_STATE_FILENAME, read_state
        from ..point_allocation import (
            allocation_targets,
            create_point_allocation,
            point_allocation_path,
            stable_candidate_id,
        )

        state = read_state(
            campaign / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
        )
        primary_candidates = []
        anchor_candidate_ids = []
        for anchor_index in range(int(anchor_plan.n_anchor)):
            candidate_id = stable_candidate_id(
                campaign_uid=str(state.campaign_uid),
                context="bootstrap",
                iteration=0,
                source_identity={"source": "anchor", "anchor_index": int(anchor_index)},
            )
            anchor_candidate_ids.append(candidate_id)
            primary_candidates.append({
                "candidate_id": candidate_id,
                "source": "anchor",
                "anchor_index": int(anchor_index),
                "frame_id": None,
            })
        for frame_id in selected_pool_indices:
            primary_candidates.append({
                "candidate_id": stable_candidate_id(
                    campaign_uid=str(state.campaign_uid),
                    context="bootstrap",
                    iteration=0,
                    source_identity={"source": "phase_a_polus", "frame_id": int(frame_id)},
                ),
                "source": "phase_a_polus",
                "frame_id": int(frame_id),
            })
        reserve_candidates = [
            {
                "candidate_id": stable_candidate_id(
                    campaign_uid=str(state.campaign_uid),
                    context="bootstrap",
                    iteration=0,
                    source_identity={"source": "phase_a_reserve", "frame_id": int(frame_id)},
                ),
                "source": "phase_a_reserve",
                "frame_id": int(frame_id),
                "reserve_rank": int(reserve_rank),
            }
            for reserve_rank, frame_id in enumerate(reserve_pool_indices)
        ]
        allocation_path = point_allocation_path(
            campaign,
            context="bootstrap",
            iteration=0,
        )
        allocation = create_point_allocation(
            allocation_path,
            campaign_uid=str(state.campaign_uid),
            context="bootstrap",
            iteration=0,
            targets=allocation_targets(config, "bootstrap"),
            primary_candidates=primary_candidates,
            reserve_candidates=reserve_candidates,
            anchor_candidate_ids=anchor_candidate_ids,
        )
        slot_by_candidate = {
            str(slot["attempts"][0]["candidate_id"]): {
                "slot_id": int(slot["slot_id"]),
                "split": str(slot["split"]),
            }
            for slot in allocation["slots"]
        }
        allocation_records = [
            {
                **record,
                **slot_by_candidate[str(record["candidate_id"])],
            }
            for record in primary_candidates
        ]
    except Exception as exc:
        print(
            "Phase A point allocation failed: "
            + type(exc).__name__ + ": " + str(exc),
            file=_sys.stderr,
        )
        return 3
    anchor_manifest_path = None
    if bool(anchor_plan.enabled):
        anchor_manifest_path = write_bootstrap_anchor_manifest(
            campaign,
            anchor_plan,
            selected_pool_frame_ids=selected_pool_indices,
            phase_a_sample_xyz=sample_path,
            phase_a_index_path=index_path,
        )
    write_phase_a_sample_manifest(outdir, {
        "phase": "PHASE_A_POLUS",
        "iteration": 0,
        "sample_xyz": sample_path.resolve().relative_to(outdir.parent.resolve()).as_posix(),
        "index_path": index_path.resolve().relative_to(outdir.parent.resolve()).as_posix(),
        "n_select": int(n_select),
        "n_frames": int(len(selected_frames)),
        "selected_indices": selected_indices,
        "selected_pool_indices": [int(i) for i in selected_pool_indices],
        "descriptor": str(descriptor.name),
        "fps_diversities": diversities,
        "n_pool_frames": int(len(frames)),
        "bootstrap_total_size": int(n_select),
        "point_allocation": {
            "manifest": allocation_path.resolve().relative_to(
                outdir.parent.resolve()
            ).as_posix(),
            "targets": dict(allocation["targets"]),
            "primary": allocation_records,
            "reserve_frame_ids": [int(value) for value in reserve_pool_indices],
            "reserve_count": int(len(reserve_pool_indices)),
        },
        "bootstrap_anchor_enabled": bool(anchor_plan.enabled),
        "bootstrap_anchor_count": int(anchor_plan.n_anchor),
        "bootstrap_pool_frame_count": int(pool_select),
        "bootstrap_anchor_path": str(anchor_plan.anchor_path),
        "bootstrap_anchor_manifest": (
            None
            if anchor_manifest_path is None
            else anchor_manifest_path.resolve().relative_to(
                outdir.parent.resolve()
            ).as_posix()
        ),
        "excluded_pool_frame_ids": [
            int(i) for i in anchor_plan.excluded_pool_frame_ids
        ],
        "reserve_after_bootstrap": int(feasibility.reserve_after_bootstrap),
        "pool_feasibility": feasibility.to_dict(),
        "trajectory_sha256": str(pool.sha256),
        "source_pool_manifest": (campaign / POOL_SUBDIR / POOL_MANIFEST_FILENAME).resolve().relative_to(
            campaign.resolve()
        ).as_posix(),
    })

    print(
        "Phase A: wrote " + str(n_select) + " frames to "
        + str(sample_path),
    )
    return 0


def _run_phase_b(args, campaign, config):
    """POLUS Phase-B: pick a diverse subsample from the adversarial
    pool ARIADNE just produced, then run the optional anti-overlap
    pass against the committed QM reference data.

    Two outputs are always written under ``phase_b/``:
      selected_raw.xyz -- the raw FPS selection.
      selected.xyz     -- the deduplicated final selection; identical to the
                          raw selection when the minimum separation is zero.
    ``SELECTION.json`` binds both files and records deduplication diagnostics.
    """
    from .descriptors import build_descriptor_from_config
    from ..daemon.state import atomic_write_json
    from ..geometry_novelty import (
        EXACT_DUPLICATE_EPSILON_ANGSTROM,
        novelty_score,
        scaled_distances,
    )
    from ..handoff_manifests import (
        PHASE_B_SELECTION_SCHEMA_VERSION,
        ariadne_candidate_frames,
        ariadne_results_path,
        write_phase_b_selection_manifest,
    )
    import sys as _sys

    from ..layout import active_iteration_dir, active_phase_b_dir

    iter_dir = active_iteration_dir(campaign, int(args.iteration))
    phase_b_dir = active_phase_b_dir(iter_dir)
    manifest_path = ariadne_results_path(iter_dir)
    if not manifest_path.is_file():
        print(
            "ARIADNE results manifest not found: " + str(manifest_path),
            file=_sys.stderr,
        )
        return 3
    try:
        from ..sampling_protocol import (
            load_sampling_protocol,
            phase_b_min_separation_from_resolved,
        )

        resolved_protocol = load_sampling_protocol(
            campaign,
            config,
            iteration=int(args.iteration),
        )
        effective_config = resolved_protocol.effective_config
    except Exception as exc:
        print(
            "Phase B sampling protocol resolution failed: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=_sys.stderr,
        )
        return 3

    accept_legacy_missing_landing_safety = bool(
        getattr(
            getattr(effective_config, "adversarial_safety", None),
            "accept_legacy_missing_landing_safety",
            False,
        )
    )

    try:
        from ..daemon.config_lock import canonical_config, config_fingerprint

        ariadne_manifest, candidate_frames, candidate_records = ariadne_candidate_frames(
            iter_dir,
            expected_iteration=int(args.iteration),
            accept_legacy_missing_landing_safety=accept_legacy_missing_landing_safety,
            expected_config_sha256=config_fingerprint(canonical_config(config)),
        )
    except Exception as exc:
        print(
            "ARIADNE results manifest invalid: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=_sys.stderr,
        )
        return 3

    if not candidate_frames:
        print("ARIADNE results manifest has no accepted candidates", file=_sys.stderr)
        return 3
    all_ariadne_candidate_records = [dict(record) for record in candidate_records]

    safety_filter = {
        "enabled": False,
        "n_input": int(len(candidate_records)),
        "n_kept": int(len(candidate_records)),
        "n_dropped": 0,
        "dropped": [],
    }
    if bool(getattr(effective_config.adversarial_safety, "phase_b_filter_enabled", True)):
        try:
            candidate_frames, candidate_records, safety_filter = (
                _phase_b_landing_safety_filter(
                    candidate_frames,
                    candidate_records,
                    accept_legacy_missing_landing_safety=accept_legacy_missing_landing_safety,
                )
            )
        except Exception as exc:
            print(
                "Phase B landing safety filter failed: "
                + type(exc).__name__
                + ": "
                + str(exc),
                file=_sys.stderr,
            )
            return 3
        if not candidate_frames:
            print(
                "Phase B landing safety filter removed every candidate",
                file=_sys.stderr,
            )
            return 3

    posterior = None
    if effective_config.phase_b.descriptor == "acquisition_weighted":
        try:
            posterior = _build_phase_b_posterior(campaign, effective_config)
        except Exception as exc:
            print(
                "acquisition_weighted descriptor could not load posterior for property "
                + repr(effective_config.acquisition.property_name)
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc),
                file=_sys.stderr,
            )
            return 3
    descriptor = build_descriptor_from_config(effective_config, posterior=posterior)
    try:
        matrix = descriptor.pairwise_distance_matrix(candidate_frames)
    except Exception as exc:
        print(
            "Phase B descriptor failed: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=_sys.stderr,
        )
        return 3
    try:
        n_select = _phase_b_target_size(
            effective_config,
            len(candidate_frames),
            int(args.iteration),
        )
    except Exception as exc:
        print(str(exc), file=_sys.stderr)
        return 3
    geometry_scale_payload = dict(resolved_protocol.geometry_scale_payload)
    min_sep, threshold_mode = phase_b_min_separation_from_resolved(
        resolved_protocol
    )

    # anti-overlap case (d): drop any selected candidate that lands too
    # close to an existing training point. In scaled mode the configured
    # threshold is dimensionless and is multiplied by the iteration scale.
    final_path = phase_b_dir / "selected.xyz"
    manifest_diagnostic_path = phase_b_dir / "SELECTION.json"
    try:
        training = _load_committed_reference_data(campaign) if min_sep > 0.0 else []
    except Exception as exc:
        print(
            "committed QM reference data invalid for Phase B anti-overlap: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=_sys.stderr,
        )
        return 3
    ordered_sel = fps_select(
        matrix,
        len(candidate_frames),
        descriptor_name=descriptor.name,
    )
    (
        selected_candidate_indices,
        selected_frames,
        selected_records,
        report,
        refill,
    ) = _phase_b_refill_after_anti_overlap(
        ordered_indices=ordered_sel.indices,
        candidate_frames=candidate_frames,
        candidate_records=candidate_records,
        training=training,
        min_separation=min_sep,
        target_size=n_select,
    )
    phase_b_dir.mkdir(parents=True, exist_ok=True)
    raw_path = phase_b_dir / "selected_raw.xyz"
    _write_xyz_file(selected_frames, raw_path)
    relaxation = {
        "applied": False,
        "reason": None,
    }
    if int(report.n_kept) <= 0 and threshold_mode == "scaled":
        finite_nonzero = []
        for raw_index, distance in enumerate(report.distances_to_nearest):
            try:
                d = float(distance)
            except (TypeError, ValueError):
                continue
            if np.isfinite(d) and d > EXACT_DUPLICATE_EPSILON_ANGSTROM:
                finite_nonzero.append((int(raw_index), float(d)))
        if finite_nonzero:
            keep_index, keep_distance = max(
                finite_nonzero,
                key=lambda item: (item[1], -item[0]),
            )
            dropped_indices = tuple(
                i for i in range(len(selected_frames)) if int(i) != int(keep_index)
            )
            report = type(report)(
                kept_indices=(int(keep_index),),
                dropped_indices=dropped_indices,
                distances_to_nearest=tuple(report.distances_to_nearest),
                min_separation=float(min_sep),
            )
            relaxation = {
                "applied": True,
                "reason": "all_candidates_below_scaled_threshold",
                "kept_raw_index": int(keep_index),
                "distance_to_nearest_angstrom": float(keep_distance),
                "effective_min_separation_angstrom": float(min_sep),
            }

    scale_angstrom = (
        geometry_scale_payload.get("scale_angstrom")
        if isinstance(geometry_scale_payload, dict)
        else None
    )
    scaled_nearest = (
        scaled_distances(report.distances_to_nearest, scale_angstrom)
        if threshold_mode == "scaled"
        else []
    )
    transform = (
        geometry_scale_payload.get("score_transform", "linear_cap")
        if isinstance(geometry_scale_payload, dict)
        else "linear_cap"
    )
    novelty_scores = (
        [
            novelty_score(distance, scale_angstrom, str(transform))
            for distance in report.distances_to_nearest
        ]
        if threshold_mode == "scaled"
        else []
    )
    dedup_payload = {
        "kept_raw_indexes_zero_based": list(report.kept_indices),
        "dropped_raw_indexes_zero_based": list(report.dropped_indices),
        "distances_to_nearest": list(report.distances_to_nearest),
        "min_separation": report.min_separation,
        "threshold_mode": threshold_mode,
        "effective_min_separation_angstrom": float(min_sep),
        "scaled_distances_to_nearest": list(scaled_nearest),
        "novelty_scores": list(novelty_scores),
        "relaxation": relaxation,
        "refill": refill,
        "n_kept": report.n_kept,
        "n_dropped": report.n_dropped,
        "n_candidates": len(selected_frames),
        "n_safe_candidates": len(candidate_frames),
        "fps_candidate_pool_indexes_zero_based": [int(i) for i in ordered_sel.indices],
        "considered_candidate_pool_indexes_zero_based": [
            int(i) for i in selected_candidate_indices
        ],
        "descriptor_used": descriptor.name,
    }
    def write_phase_b_failure(reason: str) -> Path:
        return write_phase_b_selection_manifest(
            iter_dir,
            _phase_b_json_safe({
                "schema_version": PHASE_B_SELECTION_SCHEMA_VERSION,
                "iteration": int(args.iteration),
                "status": "failed",
                "failure_reason": str(reason),
                "descriptor": str(descriptor.name),
                "source_ariadne_manifest": "ariadne/RESULTS.json",
                "n_candidates": int(len(candidate_frames)),
                "n_selected_raw": 0,
                "n_kept": 0,
                "raw": [],
                "final": [],
                "dedup": dedup_payload,
                "refill": refill,
                "safety_filter": safety_filter,
            }),
        )
    if bool(relaxation.get("applied", False)):
        _append_phase_b_journal_event(
            campaign,
            "phase_b_geometry_novelty_relaxed",
            iteration=int(args.iteration),
            reason=str(relaxation.get("reason", "")),
            kept_raw_index=int(relaxation.get("kept_raw_index", -1)),
            distance_to_nearest_angstrom=relaxation.get(
                "distance_to_nearest_angstrom"
            ),
            effective_min_separation_angstrom=float(min_sep),
            n_candidates=int(len(selected_frames)),
        )
    if int(report.n_kept) <= 0:
        write_phase_b_failure("no_non_duplicate_candidate")
        if threshold_mode == "scaled":
            print(
                "phase_b_geometry_novelty_no_non_duplicate_candidate: "
                + "kept 0/"
                + str(len(selected_frames))
                + " candidates; every candidate was an exact duplicate, "
                + "non-finite, or below the scaled novelty threshold; "
                + "diagnostics written to "
                + str(manifest_diagnostic_path),
                file=_sys.stderr,
            )
            return 3
        print(
            "phase_b_anti_overlap_removed_every_candidate: "
            + "kept 0/"
            + str(len(selected_frames))
            + " candidates after anti-overlap; diagnostics written to "
            + str(manifest_diagnostic_path),
            file=_sys.stderr,
        )
        return 3
    if int(report.n_kept) < int(config.point_allocation.batch_total_size):
        write_phase_b_failure("point_allocation_underfilled_after_anti_overlap")
        print(
            "phase_b_point_allocation_underfilled_after_anti_overlap: wanted "
            + str(int(config.point_allocation.batch_total_size))
            + ", kept "
            + str(int(report.n_kept))
            + " after considering "
            + str(len(selected_frames))
            + "/"
            + str(len(candidate_frames))
            + " safe reserve candidates"
            + "; diagnostics written to "
            + str(manifest_diagnostic_path),
            file=_sys.stderr,
        )
        return 3

    kept_frames = [selected_frames[i] for i in report.kept_indices]
    _write_xyz_file(kept_frames, final_path)

    kept_lookup = {int(raw_index): final_index for final_index, raw_index in enumerate(report.kept_indices)}
    raw_records = []
    final_records = []
    for raw_index, rec in enumerate(selected_records):
        out_rec = dict(rec)
        out_rec["candidate_pool_index_zero_based"] = int(
            selected_candidate_indices[raw_index]
        )
        out_rec["raw_rank"] = int(raw_index) + 1
        out_rec["kept_after_dedup"] = int(raw_index) in kept_lookup
        out_rec["final_rank"] = (
            int(kept_lookup[int(raw_index)]) + 1
            if int(raw_index) in kept_lookup else None
        )
        out_rec["drop_reason"] = None if out_rec["kept_after_dedup"] else "min_separation"
        out_rec["distance_to_nearest_angstrom"] = (
            report.distances_to_nearest[raw_index]
            if raw_index < len(report.distances_to_nearest)
            else None
        )
        out_rec["scaled_distance_to_nearest"] = (
            scaled_nearest[raw_index]
            if raw_index < len(scaled_nearest)
            else None
        )
        out_rec["novelty_score"] = (
            novelty_scores[raw_index]
            if raw_index < len(novelty_scores)
            else None
        )
        raw_records.append(dict(out_rec))
        if out_rec["kept_after_dedup"]:
            final_records.append(dict(out_rec))

    reserve = _phase_b_build_reserve(
        ordered_indices=ordered_sel.indices,
        already_considered_indices=selected_candidate_indices,
        candidate_frames=candidate_frames,
        candidate_records=candidate_records,
        training=training,
        selected_frames=kept_frames,
        min_separation=min_sep,
    )
    try:
        import hashlib

        from ..daemon.state import DEFAULT_STATE_FILENAME, read_state
        from ..point_allocation import (
            allocation_targets,
            create_point_allocation,
            point_allocation_path,
            stable_candidate_id,
        )
        from ..versioning.provenance import (
            enrich_with_phase_b,
            enrich_with_point_allocation,
        )

        state = read_state(
            campaign / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
        )

        def candidate_id_for(record):
            source_identity = {
                "seed_id": int(record["seed_id"]),
                "seed_uid": str(record["seed_uid"]),
                "seed_frame_id": record.get("seed_frame_id"),
                "result_sha256": str(record.get("result_sha256") or ""),
            }
            return stable_candidate_id(
                campaign_uid=str(state.campaign_uid),
                context="active",
                iteration=int(args.iteration),
                source_identity=source_identity,
            )

        primary_rank_by_uid = {
            str(record["seed_uid"]): rank
            for rank, record in enumerate(final_records, start=1)
        }
        reserve_rank_by_uid = {
            str(record["seed_uid"]): rank
            for rank, record in enumerate(reserve["records"], start=1)
        }
        candidate_id_by_uid = {
            str(record["seed_uid"]): candidate_id_for(record)
            for record in all_ariadne_candidate_records
        }
        for record in all_ariadne_candidate_records:
            uid = str(record["seed_uid"])
            seed_dir = Path(str(record.get("seed_dir") or ""))
            if not seed_dir.is_dir() or seed_dir.is_symlink():
                raise FileNotFoundError(
                    "Phase B candidate seed directory is invalid: " + str(seed_dir)
                )
            enrich_with_phase_b(
                seed_dir,
                selected_after_fps=(uid in primary_rank_by_uid),
                diversity_rank=primary_rank_by_uid.get(uid),
                descriptor_used=str(descriptor.name),
                candidate_id=candidate_id_by_uid[uid],
                reserve_candidate=(uid in reserve_rank_by_uid),
            )

        def allocation_candidate(record, *, reserve_rank=None, distance=None):
            provenance_path = Path(str(record.get("provenance_json") or ""))
            if not provenance_path.is_file():
                raise FileNotFoundError(
                    "Phase B candidate provenance is missing: " + str(provenance_path)
                )
            out = dict(record)
            out["candidate_id"] = candidate_id_by_uid[str(record["seed_uid"])]
            if reserve_rank is not None:
                out["reserve_rank"] = int(reserve_rank) + 1
            if distance is not None:
                out["distance_to_nearest_angstrom"] = float(distance)
            return _phase_b_json_safe(out)

        primary_allocation_records = [
            allocation_candidate(record)
            for record in final_records
        ]
        reserve_allocation_records = [
            allocation_candidate(
                record,
                reserve_rank=rank,
                distance=reserve["distances_to_nearest_angstrom"][rank],
            )
            for rank, record in enumerate(reserve["records"])
        ]
        allocation_path = point_allocation_path(
            campaign,
            context="active",
            iteration=int(args.iteration),
        )
        allocation = create_point_allocation(
            allocation_path,
            campaign_uid=str(state.campaign_uid),
            context="active",
            iteration=int(args.iteration),
            targets=allocation_targets(config, "active"),
            primary_candidates=primary_allocation_records,
            reserve_candidates=reserve_allocation_records,
        )
        slot_by_candidate = {
            str(slot["attempts"][0]["candidate_id"]): {
                "slot_id": int(slot["slot_id"]),
                "split": str(slot["split"]),
            }
            for slot in allocation["slots"]
        }
        final_records_with_slots = [
            {**record, **slot_by_candidate[str(record["candidate_id"])]}
            for record in primary_allocation_records
        ]
        assignment_sha = str(allocation["slot_assignment_sha256"])
        final_records = []
        for record in final_records_with_slots:
            seed_dir = Path(str(record["seed_dir"]))
            enrich_with_point_allocation(
                seed_dir,
                candidate_id=str(record["candidate_id"]),
                context="active",
                slot_id=int(record["slot_id"]),
                split=str(record["split"]),
                replacement_round=0,
                allocation_slot_assignment_sha256=assignment_sha,
            )
            provenance_path = Path(str(record["provenance_json"]))
            final_records.append({
                **record,
                "provenance_sha256": hashlib.sha256(
                    provenance_path.read_bytes()
                ).hexdigest(),
            })
        final_by_uid = {
            str(record["seed_uid"]): dict(record) for record in final_records
        }
        raw_records = [
            {
                **record,
                "candidate_id": candidate_id_by_uid[str(record["seed_uid"])],
                "provenance_sha256": hashlib.sha256(
                    Path(str(record["provenance_json"])).read_bytes()
                ).hexdigest(),
            }
            for record in raw_records
        ]

        def iteration_relative_record(record):
            out = dict(record)
            for key in ("seed_dir", "result_json", "provenance_json", "output_manifest"):
                if out.get(key):
                    out[key] = Path(str(out[key])).resolve().relative_to(
                        iter_dir.resolve()
                    ).as_posix()
            return out

        raw_records = [iteration_relative_record(record) for record in raw_records]
        final_records = [
            iteration_relative_record(final_by_uid[str(record["seed_uid"])])
            for record in final_records
        ]
        reserve_allocation_records = [
            iteration_relative_record(record)
            for record in reserve_allocation_records
        ]
    except Exception as exc:
        print(
            "Phase B point allocation failed: "
            + type(exc).__name__ + ": " + str(exc),
            file=_sys.stderr,
        )
        return 3
    def protocol_file_binding(path: Path) -> Dict[str, Any]:
        resolved = Path(path).resolve()
        return {
            "path": resolved.relative_to(iter_dir.resolve()).as_posix(),
            "size": int(resolved.stat().st_size),
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        }

    phase_b_manifest = {
        "schema_version": PHASE_B_SELECTION_SCHEMA_VERSION,
        "status": "complete",
        "campaign_uid": str(state.campaign_uid),
        "iteration": int(args.iteration),
        "descriptor": str(descriptor.name),
        "sampling_protocol": {
            "sampling_aggressiveness": int(
                resolved_protocol.sampling_aggressiveness
            ),
            "resolved": protocol_file_binding(resolved_protocol.manifest_path),
            "audit": protocol_file_binding(resolved_protocol.audit_manifest_path),
            "scale_model": protocol_file_binding(resolved_protocol.scale_model_path),
        },
        "source_ariadne_manifest": "ariadne/RESULTS.json",
        "source_ariadne_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "selected_raw_xyz": {
            "path": raw_path.resolve().relative_to(iter_dir.resolve()).as_posix(),
            "size": int(raw_path.stat().st_size),
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        },
        "selected_xyz": {
            "path": final_path.resolve().relative_to(iter_dir.resolve()).as_posix(),
            "size": int(final_path.stat().st_size),
            "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        },
        "point_allocation": {
            "manifest": allocation_path.resolve().relative_to(
                iter_dir.resolve()
            ).as_posix(),
            "slot_assignment_sha256": str(allocation["slot_assignment_sha256"]),
            "targets": dict(allocation["targets"]),
            "reserve": reserve_allocation_records,
            "reserve_count": int(len(reserve_allocation_records)),
        },
        "n_candidates": int(len(candidate_frames)),
        "n_considered_after_refill": int(len(selected_frames)),
        "n_selected_raw": int(len(raw_records)),
        "n_kept": int(len(final_records)),
        "raw": raw_records,
        "final": final_records,
        "dedup": dedup_payload,
        "refill": refill,
        "safety_filter": safety_filter,
        "source_expected_n": int(ariadne_manifest.get("expected_n", len(candidate_frames))),
    }
    manifest_path = write_phase_b_selection_manifest(
        iter_dir,
        _phase_b_json_safe(phase_b_manifest),
    )

    print(
        "Phase B: kept " + str(report.n_kept) + "/"
        + str(len(selected_frames)) + " candidates after dedup; "
        + "wrote " + str(final_path)
        + " and "
        + str(manifest_path),
    )
    return 0


def main(argv=None) -> int:
    """Command-line entrypoint for POLUS diversity sub-sampling.

    The daemon calls this from inside a sbatch script for both Phase A
    (initial pool sub-sample, run once at the start of a campaign) and
    Phase B (per-iteration sub-sample over the adversarial pool).

    --iteration 0 means Phase A on the trajectory pool.
    --iteration >= 1 means Phase B for that active iteration.

    Exit codes:
      0 -- success, sample xyz + index files written.
      2 -- bad command line (missing files, malformed args).
      3 -- per-iteration state was not in a runnable shape.
    """
    import argparse
    import sys as _sys
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(
        prog="python -m ichor.hpc.active_learning.sampling.polus_wrapper",
        description=(
            "Run POLUS diversity sub-sampling for either the initial "
            "pool (Phase A) or the per-iteration adversarial pool "
            "(Phase B). Iteration zero means Phase A."
        ),
    )
    parser.add_argument(
        "--descriptor", type=str, required=True,
        choices=["rmsd_massweight", "hybrid_alf_rmsd", "acquisition_weighted"],
        help="Distance metric used to build the pairwise matrix POLUS chews on.",
    )
    parser.add_argument(
        "--iteration", type=int, required=True,
        help="Campaign iteration. Zero means Phase A; positive values mean Phase B.",
    )
    parser.add_argument(
        "--campaign-dir", type=str, required=True,
        help="Path to the campaign root (where campaign.yaml lives).",
    )
    args = parser.parse_args(argv)

    campaign = _Path(args.campaign_dir).resolve()
    if not campaign.is_dir():
        print(
            "campaign-dir does not exist or is not a directory: "
            + str(campaign), file=_sys.stderr,
        )
        return 2
    cfg_path = campaign / "campaign.yaml"
    if not cfg_path.is_file():
        print(
            "campaign.yaml not found at " + str(cfg_path),
            file=_sys.stderr,
        )
        return 2

    from ..config import CampaignConfig
    config = CampaignConfig.from_yaml(cfg_path)

    # the descriptor actually used comes from campaign.yaml; --descriptor is
    # only the launching script stating its intent. surface a mismatch loudly
    # but let the config win rather than failing the whole job over it.
    if args.descriptor and args.descriptor != config.phase_b.descriptor:
        print(
            "note: --descriptor " + repr(args.descriptor) + " is overridden by "
            + "campaign.yaml phase_b.descriptor "
            + repr(config.phase_b.descriptor),
            file=_sys.stderr,
        )

    if int(args.iteration) == 0:
        return _run_phase_a(campaign, config)
    if int(args.iteration) < 0:
        print("iteration must be >= 0", file=_sys.stderr)
        return 2
    return _run_phase_b(args, campaign, config)


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
