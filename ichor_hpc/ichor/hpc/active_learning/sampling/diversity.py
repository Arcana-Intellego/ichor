"""Exact ICHOR-owned diversity selection for bootstrap and active batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from numbers import Integral
import os
import shutil

import numpy as np

from ichor.core.atoms import Atoms

from .descriptors import (
    Descriptor,
    MassWeightedRMSDDescriptor,
    build_condensed_distance_store,
)
from .diversity_contract import diversity_selector_contract


__all__ = [
    "fps_select",
    "FPSResult",
    "DEFAULT_DESCRIPTORS",
]


DEFAULT_DESCRIPTORS = {
    "rmsd_massweight": MassWeightedRMSDDescriptor,
}
FPS_TIE_QUANTISATION = 1.0e-12
FPS_METRIC_TOLERANCE = 1.0e-12


def _selector_contract() -> Dict[str, Any]:
    return diversity_selector_contract()


def _exact_non_negative_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(label + " must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(label + " must be non-negative")
    return result


def _validate_condensed_store(store: Any) -> int:
    n = _exact_non_negative_integer(store.n, "distance store n")
    expected = n * (n - 1) // 2
    values = np.asarray(store.values)
    if values.ndim != 1 or int(values.size) != expected:
        raise ValueError(
            "condensed distance store has "
            + str(int(values.size))
            + " values; expected "
            + str(expected)
        )
    for start in range(0, expected, 1_048_576):
        chunk = np.asarray(values[start:start + 1_048_576], dtype=float)
        if not np.all(np.isfinite(chunk)) or np.any(chunk < 0.0):
            raise ValueError(
                "condensed distance store must contain finite non-negative values"
            )
    return n


def _runtime_distance_store(
    descriptor: Descriptor,
    frames: Sequence[Atoms],
    *,
    workers: int,
    distance_store_path: Optional[Path],
):
    worker_count = _exact_non_negative_integer(workers, "workers")
    if worker_count == 0:
        raise ValueError("workers must be positive")
    mode = str(os.environ.get("ICHOR_DIVERSITY_DISTANCE_STORE_MODE", "memory")).strip().lower()
    if mode not in {"memory", "file"}:
        raise ValueError("ICHOR_DIVERSITY_DISTANCE_STORE_MODE must be memory or file")
    target = None
    if mode == "file":
        if distance_store_path is None:
            raise ValueError("file-backed diversity requires --distance-store")
        required_text = os.environ.get(
            "ICHOR_DIVERSITY_SCRATCH_REQUIRED_BYTES", "0"
        )
        if not str(required_text).isdigit():
            raise ValueError(
                "ICHOR_DIVERSITY_SCRATCH_REQUIRED_BYTES must be a non-negative integer"
            )
        required = int(required_text)
        free = int(shutil.disk_usage(distance_store_path.parent).free)
        if required > 0 and free < required:
            raise OSError(
                "file-backed diversity requires "
                + str(required)
                + " free bytes at job start; available="
                + str(free)
            )
        target = distance_store_path
    return build_condensed_distance_store(
        descriptor,
        frames,
        path=target,
        workers=worker_count,
    )






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
    distance_matrix: Any,
    n_select: int,
    seed_index: Optional[int] = None,
    descriptor_name: str = "unknown",
    progress_callback: Optional[Callable[..., None]] = None,
) -> FPSResult:
    from .descriptors import CondensedDistanceStore

    requested = _exact_non_negative_integer(n_select, "n_select")

    def report(stage: str, completed: int, total: int, unit: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                stage=str(stage),
                completed=int(completed),
                total=int(total),
                unit=str(unit),
            )
        except Exception:
            return

    if isinstance(distance_matrix, CondensedDistanceStore):
        D = distance_matrix
        n = _validate_condensed_store(D)
        row = D.row
        row_sums = D.row_sums
    else:
        dense = np.asarray(distance_matrix, dtype=float)
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
            raise ValueError(
                f"distance_matrix must be square 2D; got shape {dense.shape}"
            )
        if not np.all(np.isfinite(dense)):
            raise ValueError("distance_matrix must contain only finite values")
        if np.any(dense < 0.0):
            raise ValueError("distance_matrix must be non-negative")
        if np.any(np.abs(np.diag(dense)) > FPS_METRIC_TOLERANCE):
            raise ValueError("distance_matrix diagonal must be zero")
        if not np.allclose(
            dense,
            dense.T,
            atol=FPS_METRIC_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("distance_matrix must be symmetric")
        D = dense
        n = int(dense.shape[0])
        row = lambda index: dense[int(index)]
        row_sums = lambda: dense.sum(axis=1)
    if requested == 0:
        return FPSResult(
            indices=[],
            diversities=[],
            distance_matrix_shape=(n, n),
            descriptor_name=descriptor_name,
        )
    if requested > n:
        raise ValueError(f"n_select {requested} > n {n}")

    if seed_index is None:
        report("medoid_selection", 0, 1, "medoids")
        sums = row_sums()
        ranked = np.round(sums / FPS_TIE_QUANTISATION) * FPS_TIE_QUANTISATION
        seed = int(np.lexsort((np.arange(n, dtype=int), ranked))[0])
        report("medoid_selection", 1, 1, "medoids")
    else:
        seed = _exact_non_negative_integer(seed_index, "seed_index")
        if seed >= n:
            raise ValueError(f"seed_index {seed_index} out of range")

    selected = [seed]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[seed] = True
    min_dists = np.asarray(row(seed), dtype=float).copy()
    min_dists[seed] = 0.0
    diversities: List[float] = [0.0]
    report("farthest_point_sampling", 1, requested, "ranked frames")

    for selection_index in range(1, requested):
        candidates = np.flatnonzero(~selected_mask)
        if candidates.size == 0:
            break
        scores = min_dists[candidates]
        order = np.lexsort((candidates, -scores))
        best_local = int(order[0])
        nxt = int(candidates[best_local])
        diversities.append(float(scores[best_local]))
        selected.append(nxt)
        selected_mask[nxt] = True
        new_dists = np.asarray(row(nxt), dtype=float)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[nxt] = 0.0
        completed = int(selection_index + 1)
        if completed == requested or completed % 8 == 0:
            report(
                "farthest_point_sampling",
                completed,
                requested,
                "ranked frames",
            )

    return FPSResult(
        indices=selected,
        diversities=diversities,
        distance_matrix_shape=(n, n),
        descriptor_name=descriptor_name,
    )

# --- helpers used by main(argv) below -------------------------------


def _write_xyz_file(frames, path):
    """Write a list of ICHOR Atoms objects to a plain xyz file.

    Format: atom-count line, comment line, then one atom per line. The
    same writer covers both Phase A
    (consumed by the daemon parser) and Phase B (consumed by REFERENCE_COMMIT).
    """
    out_lines = []
    for k, frame in enumerate(frames):
        out_lines.append(str(len(frame)))
        out_lines.append("frame " + str(k))
        for atom in frame:
            out_lines.append(
                "{symbol} {x:.16g} {y:.16g} {z:.16g}".format(
                    symbol=atom.type, x=float(atom.x),
                    y=float(atom.y), z=float(atom.z),
                )
            )
    Path(path).write_text(chr(10).join(out_lines) + chr(10), encoding="utf-8")


def _write_index_file(indices, path):
    """Write the per-line list of selected stable frame identities."""
    lines = []
    custom_counter = 0
    for value in indices:
        if value is None:
            lines.append("custom:" + str(custom_counter))
            custom_counter += 1
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
    return _load_committed_reference_coordinates(campaign_dir).to_atoms_list()


def _load_committed_reference_coordinates(
    campaign_dir,
    *,
    workers: int = 8,
    progress_callback=None,
):
    from .phase_b_reference import load_phase_b_reference_coordinates

    return load_phase_b_reference_coordinates(
        Path(campaign_dir),
        workers=max(1, int(workers)),
        progress_callback=progress_callback,
    )


def _phase_b_landing_safety_filter(
    candidate_frames,
    candidate_records,
):
    missing = [
        i for i, rec in enumerate(candidate_records)
        if not isinstance(rec.get("landing_safety"), dict)
    ]
    if len(missing) == len(candidate_records):
        raise ValueError(
            "all Phase B candidates are missing mandatory landing_safety metadata"
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
        if safety.get("accepted") is True:
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
    distance_oracle=None,
):
    from .anti_overlap import DedupReport, min_distance_to_training

    considered_indices = []
    considered_frames = []
    considered_records = []
    kept_indices = []
    dropped_indices = []
    distances = []
    kept_frames = []
    kept_candidate_indices = []

    for candidate_index in ordered_indices:
        considered_index = len(considered_indices)
        cand = candidate_frames[int(candidate_index)]
        distance = float(
            distance_oracle.nearest_distance(
                int(candidate_index),
                kept_candidate_indices,
            )
            if distance_oracle is not None
            else min_distance_to_training(cand, list(training) + list(kept_frames))
        )
        considered_indices.append(int(candidate_index))
        considered_frames.append(cand)
        considered_records.append(candidate_records[int(candidate_index)])
        distances.append(distance)
        if distance < float(min_separation):
            dropped_indices.append(int(considered_index))
        else:
            kept_indices.append(int(considered_index))
            kept_frames.append(cand)
            kept_candidate_indices.append(int(candidate_index))
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
    selected_candidate_indices=None,
    distance_oracle=None,
):
    """Return the remaining safe FPS candidates in deterministic reserve order."""
    from .anti_overlap import min_distance_to_training

    considered = {int(value) for value in already_considered_indices}
    accepted_context = list(selected_frames)
    accepted_candidate_indices = [
        int(value) for value in (selected_candidate_indices or [])
    ]
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
            distance_oracle.nearest_distance(
                candidate_index,
                accepted_candidate_indices,
            )
            if distance_oracle is not None
            else min_distance_to_training(frame, list(training) + accepted_context)
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
        accepted_candidate_indices.append(candidate_index)
    return {
        "candidate_indices": reserve_indices,
        "frames": reserve_frames,
        "records": reserve_records,
        "distances_to_nearest_angstrom": reserve_distances,
        "rejected": rejected,
    }


def _relax_scaled_novelty(
    *,
    considered_frames,
    report,
    training,
    target_size: int,
    considered_candidate_indices=None,
    distance_oracle=None,
):
    """Admit the farthest non-duplicate candidates until the target is met."""
    from .anti_overlap import DedupReport, min_distance_to_training
    from ..geometry_novelty import EXACT_DUPLICATE_EPSILON_ANGSTROM

    kept = [int(value) for value in report.kept_indices]
    remaining = [int(value) for value in report.dropped_indices]
    accepted_context = [considered_frames[index] for index in kept]
    accepted_candidate_indices = [
        int(considered_candidate_indices[index]) for index in kept
    ] if considered_candidate_indices is not None else []
    admitted = []
    while len(kept) < int(target_size) and remaining:
        ranked = []
        context = list(training) + accepted_context
        for considered_index in remaining:
            distance = float(
                distance_oracle.nearest_distance(
                    int(considered_candidate_indices[considered_index]),
                    accepted_candidate_indices,
                )
                if distance_oracle is not None
                else min_distance_to_training(considered_frames[considered_index], context)
            )
            if np.isfinite(distance) and distance > EXACT_DUPLICATE_EPSILON_ANGSTROM:
                ranked.append((distance, considered_index))
        if not ranked:
            break
        distance, considered_index = max(ranked, key=lambda item: (item[0], -item[1]))
        kept.append(int(considered_index))
        remaining.remove(int(considered_index))
        accepted_context.append(considered_frames[int(considered_index)])
        if considered_candidate_indices is not None:
            accepted_candidate_indices.append(
                int(considered_candidate_indices[int(considered_index)])
            )
        admitted.append({
            "considered_index_zero_based": int(considered_index),
            "distance_to_nearest_angstrom": float(distance),
        })

    kept_set = set(kept)
    dropped = tuple(
        index for index in range(len(considered_frames)) if index not in kept_set
    )
    updated = DedupReport(
        kept_indices=tuple(kept),
        dropped_indices=dropped,
        distances_to_nearest=tuple(report.distances_to_nearest),
        min_separation=float(report.min_separation),
    )
    return updated, {
        "applied": bool(admitted),
        "reason": "scaled_novelty_underfill" if admitted else None,
        "admitted": admitted,
        "n_admitted": int(len(admitted)),
        "target_size": int(target_size),
        "target_satisfied": len(kept) >= int(target_size),
    }


def _run_phase_a(
    campaign,
    config,
    *,
    campaign_uid: Optional[str] = None,
    workers: int = 1,
    distance_store_path: Optional[Path] = None,
    progress_reporter: Any = None,
):
    """Select a diverse bootstrap subset from the imported trajectory pool.

    Phase A always uses mass-weighted RMSD because no trained posterior
    exists at bootstrap time.
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
    if progress_reporter is not None:
        progress_reporter.update(
            stage="handoff_validation",
            completed=int(len(frames)),
            total=int(len(frames)),
            unit="pool frames",
        )
    if not frames:
        print("trajectory pool is empty", file=_sys.stderr)
        return 3

    try:
        from ..custom_bootstrap import load_committed_bootstrap_frames

        custom_frames_by_split, bootstrap_manifest = load_committed_bootstrap_frames(
            campaign
        )
    except Exception as exc:
        print(str(exc), file=_sys.stderr)
        return 3

    effective_targets = {
        split: int(value)
        for split, value in dict(bootstrap_manifest["effective_qm_targets"]).items()
        if split in {"train", "int_val", "ext_val"}
    }
    effective_targets["total"] = sum(effective_targets.values())
    deficits = {
        split: int(value)
        for split, value in dict(bootstrap_manifest["diversity_deficits"]).items()
    }
    n_select = int(effective_targets["total"])
    pool_select = int(sum(deficits.values()))
    excluded_pool_ids = set(
        int(i) for i in bootstrap_manifest.get("excluded_pool_frame_ids", [])
    )
    candidate_pool_ids = [
        int(i) for i in range(len(frames)) if int(i) not in excluded_pool_ids
    ]
    if pool_select > len(candidate_pool_ids):
        print(
            "point_allocation_bootstrap_total_exceeds_pool: "
            + "effective bootstrap QM target="
            + str(n_select)
            + ", pool top-up needed="
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
        if progress_reporter is not None:
            progress_reporter.update(
                stage="descriptor_construction",
                completed=0,
                total=int(len(candidate_frames)),
                unit="frames",
            )
        matrix = _runtime_distance_store(
            descriptor,
            candidate_frames,
            workers=int(workers),
            distance_store_path=distance_store_path,
        )
        sel = fps_select(
            matrix,
            len(candidate_frames),
            descriptor_name=descriptor.name,
            progress_callback=(
                None
                if progress_reporter is None
                else lambda **payload: progress_reporter.update(**payload)
            ),
        )
        if progress_reporter is not None:
            progress_reporter.update(
                stage="farthest_point_sampling",
                completed=int(len(candidate_frames)),
                total=int(len(candidate_frames)),
                unit="ranked frames",
            )
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
    selected_pool_by_split: Dict[str, List[int]] = {
        split: [] for split in ("train", "int_val", "ext_val")
    }
    pool_cursor = 0
    for split in ("train", "int_val", "ext_val"):
        count = int(deficits[split])
        selected_pool_by_split[split] = selected_pool_indices[
            pool_cursor:pool_cursor + count
        ]
        pool_cursor += count
    selected_frames: List[Atoms] = []
    selected_indices: List[Optional[int]] = []
    selected_origins: List[Tuple[str, str, Optional[int]]] = []
    for split in ("train", "int_val", "ext_val"):
        for custom_index, frame in enumerate(custom_frames_by_split[split]):
            selected_frames.append(frame)
            selected_indices.append(None)
            selected_origins.append((split, "custom_bootstrap", int(custom_index)))
        for frame_id in selected_pool_by_split[split]:
            selected_frames.append(frames[frame_id])
            selected_indices.append(int(frame_id))
            selected_origins.append((split, "phase_a_diversity", int(frame_id)))

    from ..layout import bootstrap_selection_dir

    outdir = bootstrap_selection_dir(campaign)
    outdir.mkdir(parents=True, exist_ok=True)
    sample_path = outdir / "selected.xyz"
    index_path = outdir / "selected_indices.dat"
    _write_xyz_file(selected_frames, sample_path)
    _write_index_file(selected_indices, index_path)
    try:
        if progress_reporter is not None:
            progress_reporter.update(stage="allocation_join")
        from ..daemon.state import DEFAULT_STATE_FILENAME, read_state
        from ..point_allocation import (
            create_point_allocation,
            point_allocation_path,
            stable_candidate_id,
        )

        resolved_campaign_uid = str(campaign_uid or "").strip()
        if not resolved_campaign_uid:
            state = read_state(
                campaign / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
            )
            resolved_campaign_uid = str(state.campaign_uid)
        if not resolved_campaign_uid:
            raise ValueError("Phase A requires a non-empty campaign UID")
        primary_candidates = []
        forced_candidate_splits = {}
        mandatory_candidate_ids = []
        for split, source, source_index in selected_origins:
            identity = {
                "source": source,
                "split": split,
                "source_index": source_index,
                "bootstrap_identity": str(
                    bootstrap_manifest.get("plan_identity_sha256") or ""
                ),
            }
            candidate_id = stable_candidate_id(
                campaign_uid=resolved_campaign_uid,
                context="bootstrap",
                iteration=0,
                source_identity=identity,
            )
            record = {
                "candidate_id": candidate_id,
                "source": source,
                "bootstrap_split": split,
                "frame_id": (
                    int(source_index) if source == "phase_a_diversity" else None
                ),
                "custom_index": (
                    int(source_index) if source == "custom_bootstrap" else None
                ),
                "pool_sha256": (
                    str(pool.sha256) if source == "phase_a_diversity" else None
                ),
            }
            primary_candidates.append(record)
            forced_candidate_splits[candidate_id] = split
            if source == "custom_bootstrap":
                mandatory_candidate_ids.append(candidate_id)
        reserve_candidates = [
            {
                "candidate_id": stable_candidate_id(
                    campaign_uid=resolved_campaign_uid,
                    context="bootstrap",
                    iteration=0,
                    source_identity={"source": "phase_a_reserve", "frame_id": int(frame_id)},
                ),
                "source": "phase_a_reserve",
                "frame_id": int(frame_id),
                "reserve_rank": int(reserve_rank),
                "pool_sha256": str(pool.sha256),
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
            campaign_uid=resolved_campaign_uid,
            context="bootstrap",
            iteration=0,
            targets=effective_targets,
            primary_candidates=primary_candidates,
            reserve_candidates=reserve_candidates,
            forced_candidate_splits=forced_candidate_splits,
            mandatory_candidate_ids=mandatory_candidate_ids,
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
    final_split_dir = campaign / ".DATA" / "TRAJECTORY" / "bootstrap" / "final"
    final_split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "int_val", "ext_val"):
        final_frames = list(custom_frames_by_split[split]) + [
            frames[index] for index in selected_pool_by_split[split]
        ]
        _write_xyz_file(
            final_frames,
            final_split_dir / ({
                "train": "training_set_bootstrap.xyz",
                "int_val": "internal_validation_set_bootstrap.xyz",
                "ext_val": "external_validation_set_bootstrap.xyz",
            }[split]),
        )
    write_phase_a_sample_manifest(outdir, {
        "phase": "PHASE_A_DIVERSITY",
        "iteration": 0,
        "sample_xyz": sample_path.resolve().relative_to(outdir.parent.resolve()).as_posix(),
        "index_path": index_path.resolve().relative_to(outdir.parent.resolve()).as_posix(),
        "n_select": int(n_select),
        "n_frames": int(len(selected_frames)),
        "selected_indices": selected_indices,
        "selected_pool_indices": [int(i) for i in selected_pool_indices],
        "descriptor": str(descriptor.name),
        "selector": _selector_contract(),
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
        "custom_bootstrap": bool(bootstrap_manifest.get("custom_bootstrap", False)),
        "custom_bootstrap_manifest": (
            campaign / ".DATA" / "ACTIVE_LEARNING" / "CUSTOM_BOOTSTRAP.json"
        ).resolve().relative_to(campaign.resolve()).as_posix(),
        "custom_bootstrap_counts": dict(bootstrap_manifest.get("supplied_counts") or {}),
        "bootstrap_pool_frame_count": int(pool_select),
        "excluded_pool_frame_ids": [
            int(i) for i in bootstrap_manifest.get("excluded_pool_frame_ids", [])
        ],
        "reserve_after_bootstrap": int(len(candidate_pool_ids) - pool_select),
        "pool_feasibility": {
            "ok": True,
            "pool_frames": int(len(frames)),
            "bootstrap_pool_needed": int(pool_select),
            "excluded_pool_frame_count": int(len(excluded_pool_ids)),
            "reserve_after_bootstrap": int(len(candidate_pool_ids) - pool_select),
        },
        "trajectory_sha256": str(pool.sha256),
        "source_pool_manifest": (campaign / POOL_SUBDIR / POOL_MANIFEST_FILENAME).resolve().relative_to(
            campaign.resolve()
        ).as_posix(),
    })
    if progress_reporter is not None:
        progress_reporter.update(
            stage="split_publication",
            completed=int(n_select),
            total=int(n_select),
            unit="frames",
        )

    print(
        "Phase A: wrote " + str(n_select) + " frames to "
        + str(sample_path),
    )
    return 0


def _run_phase_b(args, campaign, config, *, progress_reporter: Any = None):
    """Select an exact diverse subset from safe ARIADNE landings.

    Two outputs are always written under ``phase_b/``:
      considered_candidates.xyz -- the FPS refill prefix inspected for novelty.
      selected.xyz              -- the final allocation batch.
    ``SELECTION.json`` binds both files and records deduplication diagnostics.
    """
    from .descriptors import build_descriptor_from_config, partition_descriptor_frames
    from ..geometry_novelty import (
        novelty_score,
        scaled_distances,
    )
    from ..handoff_manifests import (
        PHASE_B_SELECTION_SCHEMA_VERSION,
        ariadne_results_path,
        validate_phase_b_handoff,
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

    try:
        from ..handoff_manifests import (
            authoritative_ariadne_candidate_frames,
        )

        (
            ariadne_manifest,
            candidate_frames,
            candidate_records,
        ) = authoritative_ariadne_candidate_frames(
            campaign,
            int(args.iteration),
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

    if progress_reporter is not None:
        progress_reporter.update(
            stage="handoff_validation",
            completed=int(len(candidate_frames)),
            total=int(len(candidate_frames)),
            unit="ARIADNE candidates",
        )

    if not candidate_frames:
        print("ARIADNE results manifest has no accepted candidates", file=_sys.stderr)
        return 3
    all_ariadne_candidate_records = [dict(record) for record in candidate_records]
    safety_input_count = int(len(candidate_frames))

    try:
        if progress_reporter is not None:
            progress_reporter.update(
                stage="safety_filter",
                completed=0,
                total=int(len(candidate_frames)),
                unit="candidates",
            )
        candidate_frames, candidate_records, safety_filter = (
            _phase_b_landing_safety_filter(
                candidate_frames,
                candidate_records,
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
    if progress_reporter is not None:
        progress_reporter.update(
            stage="safety_filter",
            completed=safety_input_count,
            total=safety_input_count,
            unit="candidates",
            accepted=int(len(candidate_frames)),
            rejected=int(safety_filter.get("n_dropped", 0)),
        )
    if not candidate_frames:
        print(
            "Phase B landing safety filter removed every candidate",
            file=_sys.stderr,
        )
        return 3

    descriptor = build_descriptor_from_config(effective_config)
    descriptor_input_count = int(len(candidate_frames))
    if progress_reporter is not None:
        progress_reporter.update(
            stage="descriptor_construction",
            completed=0,
            total=int(len(candidate_frames)),
            unit="candidates",
        )
    descriptor_indices, descriptor_rejections = partition_descriptor_frames(
        descriptor,
        candidate_frames,
    )
    if descriptor_rejections:
        detailed_rejections = []
        for rejection in descriptor_rejections:
            index = int(rejection["candidate_index_zero_based"])
            record = candidate_records[index]
            detailed_rejections.append({
                **rejection,
                "seed_id": record.get("seed_id"),
                "seed_uid": record.get("seed_uid"),
            })
        safety_filter["descriptor_rejections"] = detailed_rejections
        safety_filter["n_descriptor_rejected"] = int(len(detailed_rejections))
        candidate_frames = [candidate_frames[index] for index in descriptor_indices]
        candidate_records = [candidate_records[index] for index in descriptor_indices]
        safety_filter["n_kept"] = int(len(candidate_frames))
        safety_filter["n_dropped"] = int(
            safety_filter.get("n_dropped", 0) + len(detailed_rejections)
        )
    else:
        safety_filter["descriptor_rejections"] = []
        safety_filter["n_descriptor_rejected"] = 0
    if not candidate_frames:
        print(
            "Phase B descriptor rejected every otherwise safe candidate",
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
        rejected_count = int(len(descriptor_rejections))
        detail = (
            "; descriptor-singular candidates rejected=" + str(rejected_count)
            if rejected_count
            else ""
        )
        print(str(exc) + detail, file=_sys.stderr)
        return 3
    try:
        matrix = _runtime_distance_store(
            descriptor,
            candidate_frames,
            workers=int(getattr(args, "workers", 1)),
            distance_store_path=(
                None
                if getattr(args, "distance_store", None) is None
                else Path(args.distance_store)
            ),
        )
    except Exception as exc:
        print(
            "Phase B descriptor failed: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=_sys.stderr,
        )
        return 3
    if progress_reporter is not None:
        progress_reporter.update(
            stage="descriptor_construction",
            completed=descriptor_input_count,
            total=descriptor_input_count,
            unit="candidates",
            accepted=int(len(candidate_frames)),
            rejected=int(len(descriptor_rejections)),
        )
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
        if progress_reporter is not None:
            progress_reporter.update(
                stage="reference_coordinate_authority",
                completed=0,
                total=1,
                unit="reference views",
            )
        reference_data = (
            _load_committed_reference_coordinates(
                campaign,
                workers=int(getattr(args, "workers", 1)),
                progress_callback=(
                    None
                    if progress_reporter is None
                    else lambda stage, payload: progress_reporter.update(
                        stage=str(stage), **dict(payload)
                    )
                ),
            )
            if min_sep > 0.0
            else []
        )
        if progress_reporter is not None:
            reference_count = (
                int(reference_data.n_references)
                if hasattr(reference_data, "n_references")
                else int(len(reference_data))
            )
            progress_reporter.update(
                stage="reference_coordinate_authority",
                completed=1,
                total=1,
                unit="reference views",
                reference_count=reference_count,
                cache_status=getattr(reference_data, "cache_status", "not_required"),
            )
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
        progress_callback=(
            None
            if progress_reporter is None
            else lambda **payload: progress_reporter.update(**payload)
        ),
    )
    if progress_reporter is not None:
        progress_reporter.update(
            stage="novelty_filter",
            completed=0,
            total=int(len(candidate_frames)),
            unit="ranked candidates",
        )
    distance_oracle = None
    if min_sep > 0.0:
        try:
            from .anti_overlap import PhaseBNoveltyDistanceOracle

            distance_oracle = PhaseBNoveltyDistanceOracle(
                candidate_frames,
                reference_data,
                workers=int(getattr(args, "workers", 1)),
                progress_callback=(
                    None
                    if progress_reporter is None
                    else lambda **payload: progress_reporter.update(**payload)
                ),
            )
        except Exception as exc:
            print(
                "Phase B exact novelty distance calculation failed: "
                + type(exc).__name__
                + ": "
                + str(exc),
                file=_sys.stderr,
            )
            return 3
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
        training=(),
        min_separation=min_sep,
        target_size=n_select,
        distance_oracle=distance_oracle,
    )
    phase_b_dir.mkdir(parents=True, exist_ok=True)
    considered_path = phase_b_dir / "considered_candidates.xyz"
    _write_xyz_file(selected_frames, considered_path)
    relaxation = {
        "applied": False,
        "reason": None,
    }
    if int(report.n_kept) < int(n_select) and threshold_mode == "scaled":
        report, relaxation = _relax_scaled_novelty(
            considered_frames=selected_frames,
            report=report,
            training=(),
            target_size=int(n_select),
            considered_candidate_indices=selected_candidate_indices,
            distance_oracle=distance_oracle,
        )
        relaxation["effective_min_separation_angstrom"] = float(min_sep)
    if progress_reporter is not None:
        progress_reporter.update(
            stage="novelty_filter",
            completed=int(len(selected_candidate_indices)),
            total=int(len(candidate_frames)),
            unit="ranked candidates",
            accepted=int(report.n_kept),
            rejected=int(report.n_dropped),
        )

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
        "kept_considered_indexes_zero_based": list(report.kept_indices),
        "dropped_considered_indexes_zero_based": list(report.dropped_indices),
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
                "n_considered": 0,
                "n_kept": 0,
                "considered": [],
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
            phase="PHASE_B_DIVERSITY",
            iteration=int(args.iteration),
            reason=str(relaxation.get("reason", "")),
            n_admitted=int(relaxation.get("n_admitted", 0)),
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

    kept_lookup = {int(considered_index): final_index for final_index, considered_index in enumerate(report.kept_indices)}
    considered_records = []
    for considered_index, rec in enumerate(selected_records):
        out_rec = dict(rec)
        out_rec["candidate_pool_index_zero_based"] = int(
            selected_candidate_indices[considered_index]
        )
        out_rec["considered_rank"] = int(considered_index) + 1
        out_rec["kept_after_dedup"] = int(considered_index) in kept_lookup
        out_rec["final_rank"] = (
            int(kept_lookup[int(considered_index)]) + 1
            if int(considered_index) in kept_lookup else None
        )
        out_rec["drop_reason"] = None if out_rec["kept_after_dedup"] else "min_separation"
        out_rec["distance_to_nearest_angstrom"] = (
            report.distances_to_nearest[considered_index]
            if considered_index < len(report.distances_to_nearest)
            else None
        )
        out_rec["scaled_distance_to_nearest"] = (
            scaled_nearest[considered_index]
            if considered_index < len(scaled_nearest)
            else None
        )
        out_rec["novelty_score"] = (
            novelty_scores[considered_index]
            if considered_index < len(novelty_scores)
            else None
        )
        considered_records.append(dict(out_rec))
    final_records = [
        dict(considered_records[int(considered_index)])
        for considered_index in report.kept_indices
    ]

    reserve = _phase_b_build_reserve(
        ordered_indices=ordered_sel.indices,
        already_considered_indices=selected_candidate_indices,
        candidate_frames=candidate_frames,
        candidate_records=candidate_records,
        training=(),
        selected_frames=kept_frames,
        min_separation=min_sep,
        selected_candidate_indices=[
            selected_candidate_indices[index] for index in report.kept_indices
        ],
        distance_oracle=distance_oracle,
    )
    try:
        if progress_reporter is not None:
            progress_reporter.update(stage="allocation_join")
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
                out["reserve_rank"] = int(reserve_rank)
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
        considered_records = [
            {
                **record,
                "candidate_id": candidate_id_by_uid[str(record["seed_uid"])],
                "provenance_sha256": hashlib.sha256(
                    Path(str(record["provenance_json"])).read_bytes()
                ).hexdigest(),
            }
            for record in considered_records
        ]

        def iteration_relative_record(record):
            out = dict(record)
            for key in ("seed_dir", "result_json", "provenance_json", "output_manifest"):
                if out.get(key):
                    out[key] = Path(str(out[key])).resolve().relative_to(
                        iter_dir.resolve()
                    ).as_posix()
            return out

        considered_records = [iteration_relative_record(record) for record in considered_records]
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
        "selector": _selector_contract(),
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
        "considered_candidates_xyz": {
            "path": considered_path.resolve().relative_to(iter_dir.resolve()).as_posix(),
            "size": int(considered_path.stat().st_size),
            "sha256": hashlib.sha256(considered_path.read_bytes()).hexdigest(),
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
        "n_considered": int(len(considered_records)),
        "n_kept": int(len(final_records)),
        "considered": considered_records,
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
    try:
        validate_phase_b_handoff(
            iter_dir,
            expected_iteration=int(args.iteration),
            expected_campaign_uid=str(state.campaign_uid),
        )
    except Exception as exc:
        print(
            "Phase B publication self-validation failed: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=_sys.stderr,
        )
        return 3
    if progress_reporter is not None:
        progress_reporter.update(
            stage="split_publication",
            completed=int(len(final_records)),
            total=int(len(final_records)),
            unit="frames",
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
    """Command-line entrypoint for ICHOR exact diversity selection.

    The daemon calls this from inside a scheduler script for both Phase A
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
        prog="python -m ichor.hpc.active_learning.sampling.diversity",
        description=(
            "Run ICHOR diversity selection for either the initial "
            "pool (Phase A) or the per-iteration adversarial pool "
            "(Phase B). Iteration zero means Phase A."
        ),
    )
    parser.add_argument(
        "--descriptor", type=str, required=True,
        choices=["rmsd_massweight", "hybrid_alf_rmsd"],
        help="Distance metric used to build the exact pairwise store.",
    )
    parser.add_argument(
        "--iteration", type=int, required=True,
        help="Campaign iteration. Zero means Phase A; positive values mean Phase B.",
    )
    parser.add_argument(
        "--campaign-dir", type=str, required=True,
        help="Path to the campaign root (where campaign.yaml lives).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Snapshotted number of scientific distance workers.",
    )
    parser.add_argument(
        "--distance-store",
        type=str,
        default=None,
        help="Campaign scratch path for a file-backed condensed distance store.",
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

    if int(args.iteration) == 0 and args.descriptor != "rmsd_massweight":
        print(
            "Phase A diversity requires descriptor 'rmsd_massweight'",
            file=_sys.stderr,
        )
        return 2
    if (
        int(args.iteration) >= 1
        and args.descriptor
        and args.descriptor != config.phase_b.descriptor
    ):
        print(
            "submitted descriptor "
            + repr(args.descriptor)
            + " does not match campaign.yaml phase_b.descriptor "
            + repr(config.phase_b.descriptor),
            file=_sys.stderr,
        )
        return 2
    if isinstance(args.workers, bool) or int(args.workers) <= 0:
        print("--workers must be a positive integer", file=_sys.stderr)
        return 2

    progress_reporter = None
    try:
        from ..daemon.journal import append_event
        from ..daemon.phase_progress import PhaseProgressReporter
        from ..daemon.state import DEFAULT_STATE_FILENAME, read_state

        state = read_state(
            campaign / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
        )
        phase_name = (
            "PHASE_A_DIVERSITY"
            if int(args.iteration) == 0
            else "PHASE_B_DIVERSITY"
        )
        journal_path = (
            campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
        )

        def journal_progress(event_type, payload):
            append_event(
                journal_path,
                str(event_type),
                max_bytes=int(config.runtime.journal_max_bytes),
                retained_files=int(config.runtime.journal_retained_files),
                lock_timeout_seconds=int(
                    config.runtime.ledger_lock_timeout_seconds
                ),
                **dict(payload),
            )

        progress_reporter = PhaseProgressReporter(
            campaign,
            campaign_uid=str(state.campaign_uid),
            phase=phase_name,
            iteration=int(args.iteration),
            replacement_round=int(getattr(state, "replacement_round", 0)),
            producer_kind="worker",
            identity={
                "job_id": str(
                    os.environ.get("ICHOR_SCHEDULER_JOB_ID")
                    or os.environ.get("SLURM_JOB_ID")
                    or os.environ.get("JOB_ID")
                    or ""
                ),
                "attempt_id": str(
                    os.environ.get("ICHOR_SUBMISSION_IDENTITY") or ""
                ),
            },
            journal_callback=journal_progress,
        )
        progress_reporter.start("handoff_validation")
    except Exception:
        progress_reporter = None

    try:
        if int(args.iteration) == 0:
            result = _run_phase_a(
                campaign,
                config,
                workers=int(args.workers),
                distance_store_path=(
                    None
                    if args.distance_store is None
                    else _Path(args.distance_store)
                ),
                progress_reporter=progress_reporter,
            )
        elif int(args.iteration) < 0:
            print("iteration must be >= 0", file=_sys.stderr)
            result = 2
        else:
            result = _run_phase_b(
                args,
                campaign,
                config,
                progress_reporter=progress_reporter,
            )
        if progress_reporter is not None:
            if int(result) == 0:
                progress_reporter.complete(stage="split_publication")
            else:
                progress_reporter.fail(
                    "diversity worker exited with status " + str(int(result))
                )
    except Exception as exc:
        if progress_reporter is not None:
            progress_reporter.fail(type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        if progress_reporter is not None:
            progress_reporter.close()
    return int(result)


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
